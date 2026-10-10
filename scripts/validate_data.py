#!/usr/bin/env python3
"""Validate every document in data/ against data/schema/ — the gate `make validate` runs.

Checks the registry's model entries against model.schema.json and every appended
benchmark result against benchmark.schema.json, then the joins that no schema can
express: that every benchmark attaches to a real registry model, and that every
published Ollama tag is really in the Ollama library. Both of those failed silently
in production — 9 of 17 tagged models produced benchmarks that matched no model id,
and two seed tags pointed at library entries that no longer exist.

Exit code = number of invalid documents, so CI and `make test` fail honestly.

Usage:
  python scripts/validate_data.py [--data data]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def validate(data_dir: Path) -> int:
    schema_dir = data_dir / "schema"
    model_schema = _load(schema_dir / "model.schema.json")
    bench_schema = _load(schema_dir / "benchmark.schema.json")
    bad = 0

    registry_path = data_dir / "registry.json"
    if registry_path.exists():
        reg = _load(registry_path)
        models = reg.get("models", {})
        for mid, model in models.items():
            try:
                jsonschema.validate(model, model_schema)
            except jsonschema.ValidationError as e:
                bad += 1
                print(f"FAIL registry.json::{mid}: {e.message}")
        print(f"ok   registry.json — {len(models)} models, "
              f"schema_version {reg.get('schema_version')}, generated {reg.get('generated_at')}")
    else:
        print(f"WARN {registry_path} not found — nothing to validate")

    results = sorted((data_dir / "benchmarks").glob("*.json"))
    for path in results:
        try:
            jsonschema.validate(_load(path), bench_schema)
        except (jsonschema.ValidationError, json.JSONDecodeError) as e:
            bad += 1
            print(f"FAIL {path.relative_to(data_dir.parent)}: {e}")
    print(f"ok   benchmarks — {len(results)} result file(s)")
    sys.path.insert(0, str(ROOT / "src"))
    from fitlab.gpu_catalog import validate as validate_gpus
    try:
        catalog = _load(data_dir / "gpu_catalog.json")
        validate_gpus(catalog)
        print(f"ok   gpu_catalog.json — {len(catalog['gpus'])} profiles")
    except (OSError, ValueError, KeyError) as e:
        bad += 1
        print(f"FAIL gpu_catalog.json: {e}")
    bad += _check_joins(data_dir, results)
    bad += _check_media(data_dir)

    print("PASSED" if not bad else f"FAILED — {bad} invalid document(s)")
    return bad


def _check_joins(data_dir: Path, results: list[Path]) -> int:
    """Cross-document checks: benchmark → registry, and registry → Ollama library."""
    registry_path = data_dir / "registry.json"
    if not registry_path.exists():
        return 0
    models = _load(registry_path).get("models", {})
    by_tag = {m["ollama_tag"]: mid for mid, m in models.items() if m.get("ollama_tag")}
    bad = 0

    orphans = []
    for path in results:
        try:
            b = _load(path)
        except json.JSONDecodeError:
            continue                                     # already counted by the schema pass
        if b.get("model_id") in models or b.get("ollama_tag") in by_tag:
            continue
        orphans.append(f"{path.name} (model_id={b.get('model_id')}, tag={b.get('ollama_tag')})")
    for o in orphans:
        print(f"FAIL benchmark joins to no registry model: {o}")
    bad += len(orphans)
    print(f"ok   benchmark joins — {len(results) - len(orphans)}/{len(results)} attach to a model")

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        import ollama_catalog
        catalog = ollama_catalog.load()
    except Exception as e:
        print(f"WARN Ollama catalogue unavailable ({e}) — tag validation skipped")
        return bad
    if not catalog.get("models"):
        print("WARN no data/ollama_catalog.json — run scripts/ollama_catalog.py to enable "
              "tag validation")
        return bad
    dead = {tag: mid for tag, mid in by_tag.items()
            if not ollama_catalog.tag_exists(tag, catalog)}
    for tag, mid in sorted(dead.items()):
        print(f"FAIL {mid}: ollama_tag '{tag}' is not in the Ollama library")
    bad += len(dead)
    print(f"ok   ollama tags — {len(by_tag) - len(dead)}/{len(by_tag)} runnable "
          f"(library fetched {catalog.get('fetched_at')})")
    return bad


def _check_media(data_dir: Path) -> int:
    """The media lane (image + video): schema, then joins the schema cannot express."""
    path = data_dir / "media_registry.json"
    if not path.exists():
        print("WARN no data/media_registry.json — run scripts/build_media.py")
        return 0
    try:
        reg = _load(path)
        jsonschema.validate(reg, _load(data_dir / "schema" / "media.schema.json"))
    except (jsonschema.ValidationError, json.JSONDecodeError) as e:
        print(f"FAIL media_registry.json: {getattr(e, 'message', e)}")
        return 1
    models = reg["models"]
    gpus = {g["id"] for g in _load(data_dir / "gpu_catalog.json")["gpus"]}
    problems = []
    for cid, cat in reg["categories"].items():
        if cat["reference_gpu"] not in gpus:
            problems.append(f"category {cid}: unknown reference GPU {cat['reference_gpu']}")
        problems += [f"category {cid}: unknown model {t['id']}" for t in cat["top"] if t["id"] not in models]
    for mid, m in models.items():
        if mid != m["id"]:
            problems.append(f"{mid}: key does not match id {m['id']}")
        if m["vram_gb"]["min"] > m["vram_gb"]["recommended"]:
            problems.append(f"{mid}: vram_gb.min is above vram_gb.recommended")
    installs = [m["homepilot"]["model_id"] for m in models.values() if m.get("homepilot")]
    if len(installs) != len(set(installs)):
        problems.append("two media models share one HomePilot model_id")
    for p in problems:
        print(f"FAIL media_registry.json: {p}")
    print(f"ok   media_registry.json — {len(models)} models, {len(reg['categories'])} categories, "
          f"media_ranking_version {reg['media_ranking_version']}, generated {reg['generated_at']}")
    return len(problems)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "data"), help="path to the data/ directory")
    return validate(Path(ap.parse_args().data))


if __name__ == "__main__":
    sys.exit(main())
