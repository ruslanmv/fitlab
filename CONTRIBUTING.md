# Contributing

**Benchmark results** (the most valuable contribution): run `colab/benchmark_colab.ipynb` or
`python scripts/bench_gguf.py --model <tag> --source community-local`, then open the
"Benchmark submission" issue with the JSON. CI does the rest.

**Model suggestions**: PRs edit `data/models.seed.yaml` only (seeds + editorial notes).
Never edit `data/registry.json` — it is generated. Ranks are computed, not negotiated;
to argue with a rank, argue with the formula in `scripts/sync_hf.py` (`ranking_version` bump required).

**Stack components**: add an entry to `data/stack.yaml` with a `probe` that proves the
integration end-to-end. "It imports" is a baseline probe; frameworks must make one real
call through OllaBridge.

Conventions: conventional commits, `data/benchmarks/` is append-only, all automation writes via PR.
