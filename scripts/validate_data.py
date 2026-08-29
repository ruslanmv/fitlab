#!/usr/bin/env python3
"""Validate every document in data/ against data/schema/ — the gate `make validate` runs.

Checks the registry's model entries against model.schema.json and every appended
benchmark result against benchmark.schema.json. Exit code = number of invalid
documents, so CI and `make test` fail honestly.

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

    print("PASSED" if not bad else f"FAILED — {bad} invalid document(s)")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=str(ROOT / "data"), help="path to the data/ directory")
    return validate(Path(ap.parse_args().data))


if __name__ == "__main__":
    sys.exit(main())
