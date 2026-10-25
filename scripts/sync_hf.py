#!/usr/bin/env python3
"""Weekly Hugging Face sync + registry build. This is what keeps the page alive.

Per category (HF pipeline tag): pull top models by trendingScore and by downloads via the public
HF API, read each config.json for real arch numbers, run the FITS engine against the reference GPUs,
drop everything that can't at least `offload` on a T4 16 GB, merge with seeds + benchmark files +
compat report, score, and write data/registry.json. Models that vanish from HF are marked `stale`
(hidden after 60 days) — history is never deleted.

score = 0.30·fit + 0.20·capability + 0.35·popularity + 0.10·speed(measured, else 0.5×estimated) + 0.05·plugs
popularity = 0.75·likes + 0.25·downloads   (both log-scaled; likes are people's votes)
Published lists are then ordered by likes among models that fit, so the suggestion is the model
people ranked highest that still runs. HomePilot lists keep only models it can install (an Ollama tag).
ranking_version bumps whenever the weights or that order change, so every rank shift is explainable.
"""
from __future__ import annotations
import argparse, datetime, json, math, re, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request, parse

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from estimate_vram import estimate                                          # noqa: E402
import ollama_catalog                                                       # noqa: E402

HF = "https://huggingface.co/api"
RANKING_VERSION = "1.2"
# A model around this many Hugging Face likes saturates the popularity term.
LIKE_CAP = 20_000
POPULARITY_LIKES = 0.75
LLM_WEIGHTS = {"fit": 0.30, "capability": 0.20, "popularity": 0.35, "speed": 0.10, "plugs": 0.05}
TOP_N = 10
HOMEPILOT_LISTS = (
    ("homepilot-12gb", "HomePilot — most liked chat models on 12 GB", "text-generation", "rtx3060-12"),
    ("homepilot-16gb", "HomePilot — most liked chat models on 16 GB", "text-generation", "t4-16"),
    ("homepilot-vision", "HomePilot — most liked vision models on 16 GB", "image-text-to-text", "t4-16"),
)
CATEGORY_TAGS = ["text-generation", "image-text-to-text", "automatic-speech-recognition",
                 "text-to-speech", "sentence-similarity", "text-to-image"]
# Three orderings per tag: what is hot, what is used, and what just shipped. `lastModified`
# is what actually surfaces brand-new releases — trending and downloads both lag a launch.
DISCOVERY_SORTS = ["trendingScore", "downloads", "lastModified"]
PER_SORT = 60
ARCH_WORKERS = 12
# Re-upload mirrors: same weights, different container. They triple the registry with rows
# nobody can `ollama run`, so they never enter it.
MIRROR_AUTHORS = {"lmstudio-community", "mradermacher", "bartowski", "thebloke", "unsloth",
                  "second-state", "qwantz", "nold", "modelscope"}
# fp8/nvfp4/mxfp4 re-publishes are first-party but still the same weights in a different
# precision; they were entering as separate rows and then fighting the base model for its
# Ollama tag. gpt-oss-20b is MXFP4-native but its repo name carries no such suffix.
MIRROR_MARKERS = re.compile(
    r"-(mlx|gguf|awq|gptq|exl2|exl3|int4|int8|w4a16|w8a8|4bit|8bit|bnb|imatrix"
    r"|fp8|fp4|nvfp4|mxfp4)(-|$)|-(mlx|gguf)-", re.I)
FIT_SCORE = {"fits": 1.0, "tight": 0.7, "offload": 0.3, "no": 0.0}
TODAY = datetime.date.today()


def get(url: str):
    with request.urlopen(request.Request(url, headers={"User-Agent": "llm-fitlab/1.0"}), timeout=60) as r:
        return json.loads(r.read())


def hf_top(tag: str, sort: str, limit=PER_SORT) -> list[dict]:
    q = parse.urlencode({"pipeline_tag": tag, "sort": sort, "direction": -1,
                         "limit": limit, "full": "false"})
    try:
        return get(f"{HF}/models?{q}")
    except Exception as e:                                                  # HF down → sync degrades, never crashes
        print(f"⚠ HF API unavailable for {tag}/{sort}: {e}")
        return []


def hf_arch(hf_id: str) -> dict | None:
    """Read real architecture numbers from config.json (handles nested text_config for VLMs)."""
    try:
        cfg = get(f"https://huggingface.co/{hf_id}/raw/main/config.json")
    except Exception:
        return None
    t = cfg.get("text_config", cfg)
    heads = t.get("num_attention_heads"); hidden = t.get("hidden_size")
    kv = t.get("num_key_value_heads") or heads
    head_dim = t.get("head_dim") or (hidden // heads if heads and hidden else None)
    if not all([t.get("num_hidden_layers"), kv, head_dim]):
        return None
    return {"n_layers": t["num_hidden_layers"], "n_kv_heads": kv,
            "head_dim": head_dim, "hidden_size": hidden or 0,
            "max_context": t.get("max_position_embeddings", 0),
            "params_hint_b": _params_hint(cfg)}


def _params_hint(cfg) -> float | None:
    for k in ("num_parameters", "n_params"):
        if isinstance(cfg.get(k), (int, float)):
            return cfg[k] / 1e9
    return None


def params_from_name(hf_id: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", hf_id.replace("_", "-"))
    return float(m.group(1)) if m else None


# Capability signal for auto-discovered entries. Without it only the 22 hand-written seeds
# carried capabilities, so the leaderboard's Task and License filters saw an almost empty
# registry — "coding" matched 2 of 126 models. HF card tags are authoritative where they
# exist (license, conversational, tool-calling); the rest is repo naming, which publishers
# are consistent about for exactly these families.
_CAP_PATTERNS = {
    "coding":    re.compile(r"coder|codegemma|codestral|codellama|starcoder|deepcoder|devstral|"
                            r"granite-code|codegeex|(^|[-_])code([-_]|$)", re.I),
    "reasoning": re.compile(r"(^|[-_])r1([-_]|$)|reason|thinking|qwq|deepscaler|magistral|"
                            r"distill|(^|[-_])o1([-_]|$)", re.I),
    "vision":    re.compile(r"(^|[-_])vl([-_]|$)|vision|llava|moondream|pixtral|internvl|"
                            r"minicpm-v|smolvlm|paligemma", re.I),
    "math":      re.compile(r"(^|[-_])math([-_]|$)|prover", re.I),
    "moe":       re.compile(r"-a\d+(?:\.\d+)?b|(^|[-_])moe([-_]|$)", re.I),
}
_TAG_CAPS = {"tool-calling": "tools", "function-calling": "tools", "agent": "tools",
             "conversational": "chat", "chat": "chat", "long-context": "long-context",
             "code": "coding", "reasoning": "reasoning", "multilingual": "multilingual",
             "on-device": "small", "edge-ai": "small"}
_LANG = re.compile(r"^[a-z]{2}(-[A-Za-z]{2,4})?$")
_PIPELINE_CAPS = {"image-text-to-text": "vision", "sentence-similarity": "embeddings",
                  "feature-extraction": "embeddings", "automatic-speech-recognition": "asr",
                  "text-to-speech": "tts", "text-to-image": "image-gen"}


def card_license(card: dict) -> str | None:
    for t in card.get("tags", []):
        if t.startswith("license:"):
            return t.split(":", 1)[1]
    return None


def card_capabilities(hf_id: str, card: dict, pipeline: str, params_b: float) -> list[str]:
    tags = {t.lower() for t in card.get("tags", [])}
    caps = {_TAG_CAPS[t] for t in tags if t in _TAG_CAPS}
    if pipeline in _PIPELINE_CAPS:
        caps.add(_PIPELINE_CAPS[pipeline])
    name = hf_id.split("/")[-1]
    for cap, pat in _CAP_PATTERNS.items():
        if pat.search(name):
            caps.add(cap)
    if len({t for t in tags if _LANG.match(t)}) >= 2:
        caps.add("multilingual")
    if params_b <= 4:
        caps.add("small")
    caps.discard("chat") if caps & {"embeddings", "asr", "tts", "image-gen"} else None
    return sorted(caps)


def is_mirror(hf_id: str) -> bool:
    """A requantised re-upload of somebody else's weights, not a distinct model."""
    author, _, name = hf_id.partition("/")
    return author.lower() in MIRROR_AUTHORS or bool(MIRROR_MARKERS.search(name))


def slug(hf_id: str) -> str:
    return re.sub(r"[^a-z0-9._-]+", "-", hf_id.split("/")[-1].lower()).strip("-")


def norm_likes(likes: int) -> float:
    return min(math.log10(1 + max(0, int(likes or 0))) / math.log10(1 + LIKE_CAP), 1.0)


def norm_downloads(downloads: int) -> float:
    return min(math.log10(1 + max(0, int(downloads or 0))) / 7, 1.0)


def popularity(stats: dict | None) -> float:
    """People's ranking first. Downloads only corroborate a model nobody has liked yet."""
    stats = stats or {}
    return (POPULARITY_LIKES * norm_likes(stats.get("likes", 0))
            + (1 - POPULARITY_LIKES) * norm_downloads(stats.get("downloads_30d", 0)))


def model_likes(m: dict) -> int:
    return int((m.get("hf_stats") or {}).get("likes") or 0)


def reference_verdict(m: dict, ref: str) -> str | None:
    return ((m.get("fits") or {}).get(ref) or {}).get("Q4_K_M", {}).get("verdict")


def llm_score(m: dict, ref: str, bench_idx: dict, plugs_pass: float) -> float:
    fit = FIT_SCORE[reference_verdict(m, ref)]
    capability = min(math.log10(1 + m["params_b"]) / 1.35, 1.0)
    measured = bench_idx.get(f"{m['id']}|{ref}")
    if measured:
        speed = min(measured["metrics"]["gen_tps"] / 60, 1.0)
    else:
        est = m["fits"][ref]["Q4_K_M"].get("est_tps") or 0
        speed = min(est / 60, 1.0) * 0.5
    w = LLM_WEIGHTS
    return round(w["fit"] * fit + w["capability"] * capability + w["popularity"] * popularity(m.get("hf_stats"))
                 + w["speed"] * speed + w["plugs"] * plugs_pass, 4)


def suggestion_key(m: dict, ref: str):
    """Most likes first. The score only breaks a tie, then the id, so the order is stable."""
    return (-model_likes(m), -m["scores"][ref], m["id"])


def homepilot_installable(m: dict) -> bool:
    """HomePilot installs chat and vision through an Ollama tag. Embeddings are not chat."""
    if not m.get("ollama_tag") or m.get("status") == "stale":
        return False
    return "embeddings" not in (m.get("capabilities") or [])


def category_pool(registry: dict, cid: str, ref: str) -> list[dict]:
    pool = []
    for m in registry.values():
        if m.get("status") == "stale" or reference_verdict(m, ref) in (None, "no"):
            continue
        if "scores" not in m or ref not in m["scores"]:
            continue
        in_seed = cid in (m.get("seed_lists") or [])
        by_task = ((cid == "multimodal" and m.get("category") == "image-text-to-text")
                   or (cid in ("local-12gb", "colab-free") and m.get("category") == "text-generation"))
        if in_seed or by_task:
            pool.append(m)
    return pool


def build_lists(registry: dict, seed_categories: dict) -> dict:
    # HomePilot lists come first so `fitlab check` (category "all") suggests an installable
    # model before a fit-only row that happens to have a lot of likes.
    lists = {}
    for cid, title, task, ref in HOMEPILOT_LISTS:
        pool = [m for m in registry.values()
                if homepilot_installable(m) and m.get("category") == task
                and reference_verdict(m, ref) not in (None, "no") and ref in m.get("scores", {})]
        ranked = sorted(pool, key=lambda m: suggestion_key(m, ref))
        lists[cid] = {"title": title, "reference_gpu": ref, "order": "likes",
                      "install": "ollama",
                      "top": [{"id": m["id"], "score": m["scores"][ref], "likes": model_likes(m)}
                              for m in ranked[:TOP_N]]}
    for cid, cfg in seed_categories.items():
        ref = cfg["reference_gpu"]
        ranked = sorted(category_pool(registry, cid, ref), key=lambda m: suggestion_key(m, ref))
        lists[cid] = {"title": cfg["title"], "reference_gpu": ref,
                      "order": "likes",
                      "top": [{"id": m["id"], "score": m["scores"][ref], "likes": model_likes(m)}
                              for m in ranked[:TOP_N]]}
    return lists


def apply_scores(registry: dict, bench_idx: dict, plugs_pass: float, ref_ids: list[str]) -> None:
    for m in registry.values():
        if not m.get("params_b") or not m.get("fits"):
            continue
        m["scores"] = {gid: llm_score(m, gid, bench_idx, plugs_pass)
                       for gid in ref_ids if reference_verdict(m, gid)}


def write_registry(reg: dict) -> None:
    data = ROOT / "data/registry.json"
    site = ROOT / "site/registry.json"
    seed = ROOT / "src/fitlab/data/seed_registry.json"
    data.write_text(json.dumps(reg, indent=2, sort_keys=True) + "\n")
    site.write_text(json.dumps(reg, sort_keys=True))
    seed.write_text(data.read_text())


def refresh_hf_stats(registry: dict, today: str) -> int:
    """Fill likes and downloads for every model already in the registry. Missing calls keep the last value."""
    ids = [m["hf_id"] for m in registry.values() if m.get("hf_id")]

    def one(hf_id: str):
        url = f"{HF}/models/{parse.quote(hf_id, safe='/')}?expand[]=downloads&expand[]=likes"
        try:
            card = get(url)
        except Exception as e:
            print(f"::warning::{hf_id}: Hugging Face stats unavailable ({e})")
            return hf_id, None
        return hf_id, {"downloads_30d": int(card.get("downloads") or 0),
                       "likes": int(card.get("likes") or 0),
                       "trending_score": card.get("trendingScore", 0),
                       "fetched_at": today}

    updated = 0
    with ThreadPoolExecutor(max_workers=ARCH_WORKERS) as pool:
        for hf_id, stats in pool.map(one, ids):
            if not stats:
                continue
            for m in registry.values():
                if m.get("hf_id") == hf_id:
                    m["hf_stats"] = stats
                    updated += 1
    return updated


def rescore(offline: bool = False, today: str | None = None) -> int:
    """Recompute scores and the likes order from the registry on disk. No model configs are re-read."""
    today = today or datetime.date.today().isoformat()
    path = ROOT / "data/registry.json"
    reg = json.loads(path.read_text())
    registry = reg["models"]
    if not offline:
        print(f"✓ refreshed Hugging Face stats for {refresh_hf_stats(registry, today)} models")
    seeds = yaml.safe_load((ROOT / "data/models.seed.yaml").read_text())
    bench_idx = {k: {"metrics": {"gen_tps": v.get("gen_tps") or 0}}
                 for k, v in (reg.get("benchmarks_latest") or {}).items()}
    compat = reg.get("compat") or {}
    plugs_pass = (sum(1 for r in compat.values() if isinstance(r, dict) and r.get("ok")) / len(compat)) if compat else 0.5
    ref_ids = sorted({gid for m in registry.values() for gid in (m.get("fits") or {})})
    apply_scores(registry, bench_idx, plugs_pass, ref_ids)
    reg["categories"] = build_lists(registry, seeds["categories"])
    reg["ranking_version"] = RANKING_VERSION
    reg["generated_at"] = today
    reg["scoring"] = {"weights": LLM_WEIGHTS, "like_cap": LIKE_CAP,
                      "popularity": "0.75·log likes + 0.25·log downloads",
                      "order": "likes, then score, among models whose verdict is not no"}
    write_registry(reg)
    tops = {cid: [t["id"] for t in cat["top"][:3]] for cid, cat in reg["categories"].items()}
    print(f"✓ rescored {len(registry)} models at ranking {RANKING_VERSION}")
    for cid, names in tops.items():
        print(f"  {cid}: {', '.join(names)}")
    return 0


def main() -> int:
    hw = yaml.safe_load((ROOT / "data/hardware.yaml").read_text())
    seeds = yaml.safe_load((ROOT / "data/models.seed.yaml").read_text())
    defaults, ctx = hw["fit_defaults"], hw["fit_defaults"]["context_tokens"]
    # Shared hardware catalog; preserve the ranking's two reference lanes.
    sys.path.insert(0, str(ROOT / "src"))
    from fitlab import gpu_catalog
    hw["gpus"] = gpu_catalog.load()["gpus"]
    ref_gpus = {g["id"]: g for g in hw["gpus"] if g.get("reference")}
    t4 = ref_gpus["t4-16"]

    registry: dict[str, dict] = {m["id"]: {**m, "schema_version": "1.0"} for m in seeds["models"]}
    prev_path = ROOT / "data/registry.json"
    prev = json.loads(prev_path.read_text())["models"] if prev_path.exists() else {}
    for mid, m in prev.items():                                             # carry forward, mark stale
        registry.setdefault(mid, m)

    # Ollama library catalogue: the only source of truth for "can I actually run this?".
    try:
        catalog = ollama_catalog.load(refresh=True)
        print(f"✓ ollama catalogue: {catalog.get('count', 0)} library models")
    except Exception as e:                                                  # Ollama down → keep last-good
        print(f"⚠ Ollama catalogue refresh failed ({e}); using the copy on disk")
        catalog = ollama_catalog.load()

    seen_this_week = set()
    for tag in CATEGORY_TAGS:
        cards = {}
        for sort in DISCOVERY_SORTS:
            cards.update({c["id"]: c for c in hf_top(tag, sort)})
        cards = {i: c for i, c in cards.items()
                 if not (c.get("gated") or c.get("private") or is_mirror(i))}
        with ThreadPoolExecutor(max_workers=ARCH_WORKERS) as pool:
            arches = dict(zip(cards, pool.map(hf_arch, cards)))
        print(f"  {tag}: {len(cards)} candidates, {sum(a is not None for a in arches.values())} readable configs")
        for hf_id, card in cards.items():
            mid = slug(hf_id)
            arch = arches[hf_id]
            params = (arch or {}).get("params_hint_b") or params_from_name(hf_id) \
                     or registry.get(mid, {}).get("params_b")
            if not (arch and params):
                continue
            entry = registry.get(mid, {"id": mid, "status": "auto", "schema_version": "1.0",
                                       "capabilities": [], "ollama_tag": None})
            derived_caps = card_capabilities(hf_id, card, tag, params)
            if entry.get("status") == "seed":                                # curated wins
                derived_caps = sorted(set(entry.get("capabilities") or []) | set(derived_caps))
            entry.setdefault("license", "")
            if not entry["license"]:
                entry["license"] = card_license(card) or ""
            entry["capabilities"] = derived_caps
            entry.update({"hf_id": hf_id, "category": tag, "params_b": round(params, 2),
                          "arch": {k: arch[k] for k in ("n_layers", "n_kv_heads", "head_dim",
                                                        "hidden_size", "max_context")},
                          "hf_stats": {"downloads_30d": card.get("downloads", 0),
                                       "likes": card.get("likes", 0),
                                       "trending_score": card.get("trendingScore", 0),
                                       "fetched_at": TODAY.isoformat()}})
            # FITS filter: must at least offload on a T4 16 GB at Q4 to enter the registry
            if estimate(entry, "Q4_K_M", ctx, t4, defaults)["verdict"] == "no":
                continue
            registry[mid] = entry
            seen_this_week.add(mid)

    # Resolve every entry against the live Ollama library. Auto-discovered models used to
    # land with ollama_tag=None and were therefore invisible under the leaderboard's default
    # "runs on Ollama" filter; hand-written seed tags were never re-checked and rot silently.
    if catalog.get("models"):
        resolved = revoked = 0
        for mid, m in registry.items():
            hf_id = m.get("hf_id")
            if not hf_id:
                continue
            found = ollama_catalog.resolve(hf_id, catalog)
            current = m.get("ollama_tag")
            if current and not ollama_catalog.tag_exists(current, catalog):
                print(f"::warning::{mid}: ollama_tag '{current}' is no longer in the Ollama library")
                m["ollama_tag"], revoked = found, revoked + 1
            elif not current and found:
                m["ollama_tag"], resolved = found, resolved + 1
            m["ollama_verified_at"] = catalog.get("fetched_at")
        # One tag, one model. A resolver that is too loose hands a speech or vision variant
        # the text model's tag, which both misleads the user and hijacks its benchmarks.
        claims: dict[str, list[str]] = {}
        for mid, m in registry.items():
            if m.get("ollama_tag"):
                claims.setdefault(m["ollama_tag"], []).append(mid)
        for tag, mids in claims.items():
            if len(mids) > 1:
                keep = min(mids, key=len)                                # closest name to the tag
                for mid in mids:
                    if mid != keep:
                        print(f"::warning::{mid} also claimed '{tag}' (kept by {keep}); clearing")
                        registry[mid]["ollama_tag"] = None
        print(f"✓ ollama tags: {resolved} newly resolved, {revoked} revoked as dead, "
              f"{sum(1 for m in registry.values() if m.get('ollama_tag'))} runnable")

    # fits matrix + staleness
    for mid, m in registry.items():
        m["fits"] = {gid: {q: estimate(m, q, ctx, g, defaults) for q in ("Q4_K_M", "Q8_0")}
                     for gid, g in ref_gpus.items()}
        if m.get("status") != "seed" and mid not in seen_this_week:
            m.setdefault("stale_since", TODAY.isoformat())
            m["status"] = "stale"
        else:
            m.pop("stale_since", None)

    # merge measured benchmarks (latest per model×gpu wins) and compat report
    # A benchmark's model_id used to be derived by string-mangling the Ollama tag
    # ("qwen3:0.6b" -> "qwen3-06b"), which matches no registry slug at all — 9 of 17 tagged
    # models produced results that silently never joined. Resolve through ollama_tag too.
    by_tag = {m["ollama_tag"]: mid for mid, m in registry.items() if m.get("ollama_tag")}

    def bench_model_id(b: dict) -> str | None:
        if b.get("model_id") in registry:
            return b["model_id"]
        tag = b.get("ollama_tag")
        if tag in by_tag:
            return by_tag[tag]
        if tag:                                                             # qwen3:0.6b -> qwen3 family
            base = tag.split(":")[0]
            hits = [mid for t, mid in by_tag.items() if t.split(":")[0] == base]
            if len(hits) == 1:
                return hits[0]
        return None

    bench_idx: dict[str, dict] = {}
    orphans = []
    for f in sorted((ROOT / "data/benchmarks").glob("*.json")):
        b = json.loads(f.read_text())
        mid = bench_model_id(b)
        if not mid:
            orphans.append(f"{f.name} (model_id={b.get('model_id')}, tag={b.get('ollama_tag')})")
            continue
        bench_idx[f"{mid}|{b['gpu'].get('profile_id')}"] = b
    for o in orphans:
        print(f"::warning::benchmark does not match any registry model: {o}")
    compat_path = ROOT / "data/compat_report.json"
    compat = json.loads(compat_path.read_text())["results"] if compat_path.exists() else {}
    plugs_pass = (sum(1 for r in compat.values() if r["ok"]) / len(compat)) if compat else 0.5

    # Score every model against every reference GPU, not just the ones that land in a
    # category Top-10. With ~140 models and 3 categories, ranking only the top 10 of each
    # left most of the registry unscored and the leaderboard showed "unranked" for rows it
    # had all the numbers for. Lists are then ordered by likes among models that fit.
    apply_scores(registry, bench_idx, plugs_pass, list(ref_gpus))
    lists = build_lists(registry, seeds["categories"])

    # Report how the bandwidth model is doing against reality rather than quietly tuning
    # the constant to whatever few measurements exist. The first real GPU datapoint came
    # back 2.5x slower than the estimate, which the page should say out loud.
    all_gpus = {g["id"]: g for g in hw["gpus"]}
    ratios = []
    for key, b in bench_idx.items():
        mid, _, gid = key.partition("|")
        m, g = registry.get(mid), all_gpus.get(gid)
        meas = b["metrics"]["gen_tps"]
        # Benchmarks land on whatever hardware CI was given (a P100, a CPU runner), which is
        # rarely one of the two reference GPUs in `fits` — so estimate against the GPU that
        # actually ran, at the context that actually ran.
        if not (m and g and meas):
            continue
        est = estimate(m, b.get("quant", "Q4_K_M"), b.get("context", ctx), g, defaults).get("est_tps")
        if est:
            ratios.append(round(est / meas, 3))
    calibration = {"measured_pairs": len(ratios),
                   "estimate_over_measured": round(statistics.median(ratios), 2) if ratios else None,
                   "ratios": sorted(ratios)}

    out = {"schema_version": "1.0", "ranking_version": RANKING_VERSION, "calibration": calibration,
           "generated_at": TODAY.isoformat(), "categories": lists,
           "scoring": {"weights": LLM_WEIGHTS, "like_cap": LIKE_CAP,
                       "popularity": "0.75·log likes + 0.25·log downloads",
                       "order": "likes, then score, among models whose verdict is not no"},
           "compat": compat, "models": registry,
           "benchmarks_latest": {k: {"gen_tps": v["metrics"]["gen_tps"],
                                     "peak_vram_gb": v["metrics"].get("peak_vram_gb"),
                                     "date": v["run"]["date"], "source": v["provenance"]["source"],
                                     "submitter": v["provenance"].get("submitter", "")}
                                 for k, v in bench_idx.items()}}
    write_registry(out)
    print(f"✓ registry: {len(registry)} models, {len(bench_idx)} measured, "
          f"{sum(1 for m in registry.values() if m.get('status') == 'stale')} stale")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rescore", action="store_true",
                    help="recompute scores and the likes order from data/registry.json")
    ap.add_argument("--offline", action="store_true",
                    help="with --rescore, keep the Hugging Face stats already stored")
    args = ap.parse_args()
    raise SystemExit(rescore(offline=args.offline) if args.rescore else main())
