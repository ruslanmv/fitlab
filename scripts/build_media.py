#!/usr/bin/env python3
"""MEDIA lane — rank image and video generation models for each reference GPU.

The LLM lane derives memory from a model's config.json. Diffusion pipelines have no such
single file, so data/media.seed.yaml declares each model's VRAM need (with the source of the
number) and the work its default workflow does. This script adds live Hugging Face momentum,
derives a verdict per GPU, scores with a versioned formula and writes data/media_registry.json.

Nothing is hand-ranked: to move a model, change a fact in the seed (with its source) or change
the formula here and bump MEDIA_RANKING_VERSION.

    score = 0.40·fit + 0.25·capability + 0.25·momentum + 0.10·speed

  fit         FIT_SCORE[verdict]; fits ≥ recommended VRAM, tight ≥ min, offload ≥ 0.7·min
              (ComfyUI low-VRAM mode: works, slowly), otherwise no
  capability  log10(1 + params_b) / log10(1 + 14) — larger models that still fit outrank tiny ones
  momentum    log10(1 + downloads_30d) / 7, the same scale as the LLM lane
  speed       1 − log10(1 + cost) / 3.5, cost = params_b · steps · megapixels · max(1, frames / 16)

The GPU-independent parts (capability, momentum, speed) are published per model as
`components`, with the weights under `scoring`, so a client can rank for any VRAM it detects
without re-implementing the formula — only the fit verdict depends on the user's GPU.

Usage:
  python scripts/build_media.py              # refresh HF stats, rebuild data/media_registry.json
  python scripts/build_media.py --offline    # keep the previous stats (no network)
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "data/media.seed.yaml"
OUT = ROOT / "data/media_registry.json"
GPU_CATALOG = ROOT / "data/gpu_catalog.json"

MEDIA_RANKING_VERSION = "1.0"
WEIGHTS = {"fit": 0.40, "capability": 0.25, "momentum": 0.25, "speed": 0.10}
FIT_SCORE = {"fits": 1.0, "tight": 0.7, "offload": 0.3, "no": 0.0}
OFFLOAD_FRACTION = 0.7
TOP_N = 10
HF_API = "https://huggingface.co/api/models/"


def verdict(vram_gb: float, need: dict) -> str:
    if vram_gb >= need["recommended"]:
        return "fits"
    if vram_gb >= need["min"]:
        return "tight"
    if vram_gb >= need["min"] * OFFLOAD_FRACTION:
        return "offload"
    return "no"


def components(m: dict) -> dict:
    capability = min(math.log10(1 + m["params_b"]) / math.log10(1 + 14), 1.0)
    downloads = (m.get("hf_stats") or {}).get("downloads_30d", 0) or 0
    momentum = min(math.log10(1 + downloads) / 7, 1.0)
    c = m.get("cost") or {}
    cost = m["params_b"] * c.get("steps", 25) * c.get("megapixels", 1.0) * max(1.0, c.get("frames", 1) / 16)
    speed = max(0.0, min(1.0, 1 - math.log10(1 + cost) / 3.5))
    return {"capability": round(capability, 4), "momentum": round(momentum, 4), "speed": round(speed, 4)}


def score(comp: dict, v: str) -> float:
    return round(WEIGHTS["fit"] * FIT_SCORE[v] + WEIGHTS["capability"] * comp["capability"]
                 + WEIGHTS["momentum"] * comp["momentum"] + WEIGHTS["speed"] * comp["speed"], 4)


def hf_stats(hf_id: str, timeout: float = 15.0) -> dict | None:
    url = HF_API + urllib.parse.quote(hf_id, safe="/") + "?expand[]=downloads&expand[]=likes"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            card = json.loads(r.read())
    except Exception as e:                     # network, 404, rate limit — keep the last-good value
        print(f"::warning::{hf_id}: Hugging Face stats unavailable ({e})")
        return None
    return {"downloads_30d": int(card.get("downloads") or 0), "likes": int(card.get("likes") or 0)}


def build(seed: dict, gpus: dict, previous: dict, offline: bool, today: str) -> dict:
    models = {}
    for raw in seed["models"]:
        m = {k: v for k, v in raw.items() if v is not None or k == "homepilot"}
        prev_stats = (previous.get(m["id"]) or {}).get("hf_stats")
        stats = None if offline else hf_stats(m["hf_id"])
        if stats:
            stats["fetched_at"] = today
        m["hf_stats"] = stats or prev_stats or {"downloads_30d": 0, "likes": 0, "fetched_at": None}
        m["components"] = components(m)
        models[m["id"]] = m

    refs = sorted({c["reference_gpu"] for c in seed["categories"].values()})
    for gid in refs:
        if gid not in gpus:
            raise SystemExit(f"reference GPU {gid} is not in data/gpu_catalog.json")
    for m in models.values():
        m["fits"] = {gid: {"verdict": verdict(gpus[gid]["vram_gb"], m["vram_gb"])} for gid in refs}
        m["scores"] = {gid: score(m["components"], m["fits"][gid]["verdict"]) for gid in refs}

    categories = {}
    for cid, cfg in seed["categories"].items():
        tasks = {"image": {"text-to-image"}, "video": {"text-to-video", "image-to-video"}}[cfg["task"]]
        gid = cfg["reference_gpu"]
        pool = [m for m in models.values() if m["task"] in tasks and m["fits"][gid]["verdict"] != "no"]
        ranked = sorted(pool, key=lambda m: (-m["scores"][gid], m["id"]))
        categories[cid] = {"title": cfg["title"], "task": cfg["task"], "reference_gpu": gid,
                           "top": [{"id": m["id"], "score": m["scores"][gid]} for m in ranked[:TOP_N]]}

    return {
        "schema_version": "1.0",
        "media_ranking_version": MEDIA_RANKING_VERSION,
        "generated_at": today,
        "scoring": {
            "weights": WEIGHTS,
            "fit_score": FIT_SCORE,
            "thresholds": {"fits": "vram >= vram_gb.recommended", "tight": "vram >= vram_gb.min",
                           "offload": f"vram >= {OFFLOAD_FRACTION} * vram_gb.min", "no": "otherwise"},
        },
        "categories": categories,
        "models": models,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--offline", action="store_true", help="do not call Hugging Face; keep previous stats")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    seed = yaml.safe_load(SEED.read_text(encoding="utf-8"))
    gpus = {g["id"]: g for g in json.loads(GPU_CATALOG.read_text(encoding="utf-8"))["gpus"]}
    out = Path(args.out)
    previous = json.loads(out.read_text(encoding="utf-8")).get("models", {}) if out.exists() else {}
    ids = [m["id"] for m in seed["models"]]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate media model id in data/media.seed.yaml")

    registry = build(seed, gpus, previous, args.offline, datetime.date.today().isoformat())
    out.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    counts = {c: len(v["top"]) for c, v in registry["categories"].items()}
    print(f"✓ media registry: {len(registry['models'])} models · {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
