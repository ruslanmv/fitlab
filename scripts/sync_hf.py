#!/usr/bin/env python3
"""Weekly Hugging Face sync + registry build. This is what keeps the page alive.

Per category (HF pipeline tag): pull top models by trendingScore and by downloads via the public
HF API, read each config.json for real arch numbers, run the FITS engine against the reference GPUs,
drop everything that can't at least `offload` on a T4 16 GB, merge with seeds + benchmark files +
compat report, score, and write data/registry.json. Models that vanish from HF are marked `stale`
(hidden after 60 days) — history is never deleted.

score = 0.30·fit + 0.30·capability + 0.25·hf_momentum + 0.10·speed(measured, else 0.5×estimated) + 0.05·plugs
ranking_version bumps whenever the weights change, so every rank shift is explainable in the PR diff.
"""
from __future__ import annotations
import datetime, json, math, re, statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import request, parse

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from estimate_vram import estimate                                          # noqa: E402
import ollama_catalog                                                       # noqa: E402

HF = "https://huggingface.co/api"
RANKING_VERSION = "1.1"
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


def main() -> int:
    hw = yaml.safe_load((ROOT / "data/hardware.yaml").read_text())
    seeds = yaml.safe_load((ROOT / "data/models.seed.yaml").read_text())
    defaults, ctx = hw["fit_defaults"], hw["fit_defaults"]["context_tokens"]
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

    def score(m: dict, ref: str) -> float:
        fit = FIT_SCORE[m["fits"][ref]["Q4_K_M"]["verdict"]]
        stats = m.get("hf_stats", {})
        momentum = min(math.log10(1 + stats.get("downloads_30d", 0)) / 7, 1.0)
        # capability proxy: bigger models that still fit should outrank tiny-but-fast ones
        capability = min(math.log10(1 + m["params_b"]) / 1.35, 1.0)
        measured = bench_idx.get(f"{m['id']}|{ref}")                     # same GPU, or it is not measured
        if measured:
            speed = min(measured["metrics"]["gen_tps"] / 60, 1.0)
        else:
            speed = min(m["fits"][ref]["Q4_K_M"]["est_tps"] / 60, 1.0) * 0.5
        return round(0.30 * fit + 0.30 * capability + 0.25 * momentum
                     + 0.10 * speed + 0.05 * plugs_pass, 4)

    # Score every model against every reference GPU, not just the ones that land in a
    # category Top-10. With ~140 models and 3 categories, ranking only the top 10 of each
    # left most of the registry unscored and the leaderboard showed "unranked" for rows it
    # had all the numbers for.
    for m in registry.values():
        m["scores"] = {gid: score(m, gid) for gid in ref_gpus}

    cats = seeds["categories"]
    lists = {}
    for cid, cfg in cats.items():
        pool = [m for m in registry.values() if m.get("status") != "stale" and (
            cid in m.get("seed_lists", []) or
            (cid == "multimodal" and m["category"] == "image-text-to-text") or
            (cid in ("local-12gb", "colab-free") and m["category"] == "text-generation"))]
        ranked = sorted(pool, key=lambda m: score(m, cfg["reference_gpu"]), reverse=True)
        lists[cid] = {"title": cfg["title"], "reference_gpu": cfg["reference_gpu"],
                      "top": [{"id": m["id"], "score": score(m, cfg["reference_gpu"])} for m in ranked[:10]]}

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
           "compat": compat, "models": registry,
           "benchmarks_latest": {k: {"gen_tps": v["metrics"]["gen_tps"],
                                     "peak_vram_gb": v["metrics"].get("peak_vram_gb"),
                                     "date": v["run"]["date"], "source": v["provenance"]["source"],
                                     "submitter": v["provenance"].get("submitter", "")}
                                 for k, v in bench_idx.items()}}
    prev_path.write_text(json.dumps(out, indent=2, sort_keys=True))
    (ROOT / "site/registry.json").write_text(json.dumps(out, sort_keys=True))
    print(f"✓ registry: {len(registry)} models, {len(bench_idx)} measured, "
          f"{sum(1 for m in registry.values() if m.get('status') == 'stale')} stale")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
