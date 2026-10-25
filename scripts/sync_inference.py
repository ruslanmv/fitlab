#!/usr/bin/env python3
"""Inference lane: rank the hosted models HomePilot can run, by what people like.

For each HomePilot model type (chat, vision, image, video) this asks the Hugging Face Hub for
models served by Inference Providers, three ways — most liked, trending, newest — so a new model
with few likes yet is still a candidate. It keeps only models a provider serves *live* for the
task HomePilot calls, ranks them with ``fitlab.inference`` in every published order, and writes
``data/inference_registry.json`` (and the site copy). Nothing is hand-ranked.

    python scripts/sync_inference.py            # fetch and publish
    python scripts/sync_inference.py --top 20   # rows per list

A Hub call that fails leaves that query out; a type with no compatible model at all keeps the
previous published list, so one bad run never empties the feed.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path
from urllib import parse, request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fitlab import inference  # noqa: E402

HF = "https://huggingface.co/api/models"
EXPAND = ("likes", "downloads", "trendingScore", "lastModified", "createdAt", "tags",
          "library_name", "pipeline_tag", "inferenceProviderMapping")
QUERIES = (("likes", 100), ("trendingScore", 100), ("createdAt", 60))
OUT = ROOT / "data/inference_registry.json"
SITE = ROOT / "site/inference_registry.json"


def get(url: str):
    req = request.Request(url, headers={"User-Agent": "fitlab-sync-inference"})
    with request.urlopen(req, timeout=30) as r:  # noqa: S310 — fixed https host
        return json.loads(r.read().decode())


def candidates(kind: str) -> list[dict]:
    tag = inference.KINDS[kind]["pipeline_tag"]
    pool: dict[str, dict] = {}
    for sort, limit in QUERIES:
        q = [("pipeline_tag", tag), ("inference_provider", "all"), ("sort", sort), ("limit", str(limit))]
        q += [("expand[]", e) for e in EXPAND]
        try:
            for m in get(f"{HF}?{parse.urlencode(q)}"):
                pool.setdefault(m.get("id") or m.get("modelId"), m)
        except Exception as e:  # noqa: BLE001 — one query down is not the whole lane
            print(f"::warning::{kind}/{sort}: Hugging Face unavailable ({e})")
    return [m for k, m in pool.items() if k]


def build(models_by_kind: dict[str, list[dict]], now: dt.datetime, top: int, previous: dict) -> dict:
    kinds = {}
    for kind, models in models_by_kind.items():
        lists = {order: inference.rank(models, kind, order, now=now, limit=top) for order in inference.ORDERS}
        if not lists["popular"] and kind in (previous.get("kinds") or {}):
            print(f"::warning::{kind}: nothing compatible this run — keeping the previous list")
            kinds[kind] = previous["kinds"][kind]
            continue
        kinds[kind] = {
            "pipeline_tag": inference.KINDS[kind]["pipeline_tag"],
            "candidates": len(models),
            "compatible": len(inference.rank(models, kind, "popular", now=now)),
            "diversity": inference.diversity(lists["popular"]),
            "lists": lists,
        }
    return {
        "schema_version": "1.0",
        "kind": "homepilot.inference",
        "ranking_version": inference.RANKING_VERSION,
        "generated_at": now.date().isoformat(),
        "scoring": {
            "weights": inference.WEIGHTS,
            "caps": {"likes": inference.LIKE_CAP, "download_digits": inference.DOWNLOAD_DIGITS,
                     "trending": inference.TREND_CAP},
            "recency_half_life_days": inference.RECENCY_HALF_LIFE_DAYS,
            "rising_max_age_days": inference.RISING_MAX_AGE_DAYS,
            "compatible": "a provider serves it live for the task HomePilot calls; adapters and GGUF excluded",
            "note": "Popularity guides; it does not measure quality on your task.",
        },
        "orders": list(inference.ORDERS),
        "kinds": kinds,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--top", type=int, default=20, help="rows per published list")
    args = ap.parse_args(argv)
    now = dt.datetime.now(dt.timezone.utc)
    previous = json.loads(OUT.read_text()) if OUT.exists() else {}
    doc = build({kind: candidates(kind) for kind in inference.KINDS}, now, args.top, previous)
    OUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    SITE.write_text(json.dumps(doc, sort_keys=True))
    for kind, k in doc["kinds"].items():
        names = [r["id"] for r in k["lists"]["popular"][:3]]
        print(f"✓ {kind}: {k.get('compatible', '?')} compatible of {k.get('candidates', '?')} — {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
