#!/usr/bin/env python3
"""Weekly Hugging Face sync + registry build. This is what keeps the page alive.

Per category (HF pipeline tag): pull top models by trendingScore and by downloads via the public
HF API, read each config.json for real arch numbers, run the FITS engine against the reference GPUs,
drop everything that can't at least `offload` on a T4 16 GB, merge with seeds + benchmark files +
compat report, score, and write data/registry.json. Models that vanish from HF are marked `stale`
(hidden after 60 days) — history is never deleted.

score = 0.35·fit + 0.25·hf_momentum + 0.25·speed(measured, else 0.5×estimated) + 0.15·plugs
ranking_version bumps whenever the weights change, so every rank shift is explainable in the PR diff.
"""
from __future__ import annotations
import datetime, json, math, re
from pathlib import Path
from urllib import request, parse

import yaml

ROOT = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(ROOT / "scripts"))
from estimate_vram import estimate, BPW                                    # noqa: E402

HF = "https://huggingface.co/api"
RANKING_VERSION = "1.1"
CATEGORY_TAGS = ["text-generation", "image-text-to-text", "automatic-speech-recognition",
                 "text-to-speech", "sentence-similarity", "text-to-image"]
FIT_SCORE = {"fits": 1.0, "tight": 0.7, "offload": 0.3, "no": 0.0}
TODAY = datetime.date.today()


def get(url: str):
    with request.urlopen(request.Request(url, headers={"User-Agent": "llm-fitlab/1.0"}), timeout=60) as r:
        return json.loads(r.read())


def hf_top(tag: str, sort: str, limit=25) -> list[dict]:
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

    seen_this_week = set()
    for tag in CATEGORY_TAGS:
        cards = {c["id"]: c for c in hf_top(tag, "trendingScore") + hf_top(tag, "downloads")}
        for hf_id, card in cards.items():
            if card.get("gated") or card.get("private"):
                continue
            mid = slug(hf_id)
            arch = hf_arch(hf_id)
            params = (arch or {}).get("params_hint_b") or params_from_name(hf_id) \
                     or registry.get(mid, {}).get("params_b")
            if not (arch and params):
                continue
            entry = registry.get(mid, {"id": mid, "status": "auto", "schema_version": "1.0",
                                       "capabilities": [], "ollama_tag": None})
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
    bench_idx: dict[str, dict] = {}
    for f in sorted((ROOT / "data/benchmarks").glob("*.json")):
        b = json.loads(f.read_text())
        bench_idx[f"{b['model_id']}|{b['gpu'].get('profile_id')}"] = b
    compat_path = ROOT / "data/compat_report.json"
    compat = json.loads(compat_path.read_text())["results"] if compat_path.exists() else {}
    plugs_pass = (sum(1 for r in compat.values() if r["ok"]) / len(compat)) if compat else 0.5

    def score(m: dict, ref: str) -> float:
        fit = FIT_SCORE[m["fits"][ref]["Q4_K_M"]["verdict"]]
        stats = m.get("hf_stats", {})
        momentum = min(math.log10(1 + stats.get("downloads_30d", 0)) / 7, 1.0)
        # capability proxy: bigger models that still fit should outrank tiny-but-fast ones
        capability = min(math.log10(1 + m["params_b"]) / 1.35, 1.0)
        key = next((k for k in bench_idx if k.startswith(m["id"] + "|")), None)
        if key:
            speed = min(bench_idx[key]["metrics"]["gen_tps"] / 60, 1.0)
        else:
            speed = min(m["fits"][ref]["Q4_K_M"]["est_tps"] / 60, 1.0) * 0.5
        return round(0.30 * fit + 0.30 * capability + 0.25 * momentum
                     + 0.10 * speed + 0.05 * plugs_pass, 4)

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

    out = {"schema_version": "1.0", "ranking_version": RANKING_VERSION,
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
