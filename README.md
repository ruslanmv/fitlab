<p align="center">
  <img src="site/logo.svg" width="96" alt="FitLab logo" />
</p>

<h1 align="center">LLM FitLab</h1>

<p align="center">
  <strong>An evidence-based compatibility radar for running open LLMs on the hardware you actually have.</strong>
</p>

<p align="center">
  <a href="https://github.com/ruslanmv/fitlab/actions/workflows/weekly-sync.yml"><img alt="Weekly sync" src="https://github.com/ruslanmv/fitlab/actions/workflows/weekly-sync.yml/badge.svg" /></a>
  <a href="https://github.com/ruslanmv/fitlab/actions/workflows/weekly-benchmark.yml"><img alt="Weekly benchmark" src="https://github.com/ruslanmv/fitlab/actions/workflows/weekly-benchmark.yml/badge.svg" /></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue" />
  <img alt="Code license" src="https://img.shields.io/badge/code-Apache--2.0-green" />
  <img alt="Data license" src="https://img.shields.io/badge/data-CC--BY--4.0-green" />
  <img alt="Status" src="https://img.shields.io/badge/status-beta-orange" />
</p>

---

FitLab answers the three questions every engineer asks before pulling a local model — and it
answers them with **regenerated data, not a hand-maintained list**. Every claim on the page is
produced by a script, validated against a JSON Schema, and merged through a reviewed pull request.

| Lane | Question it answers | Method | Confidence |
|---|---|---|---|
| **FITS** | Will this model fit my GPU? | Deterministic VRAM arithmetic from the model's own `config.json` — weights + KV cache + runtime overhead, per quantization and context length | Computed, reproducible |
| **RUNS** | How fast is it, really? | A real benchmark on a free T4 16 GB GPU every week, plus community-submitted runs on real consumer hardware | Measured, with raw data retained |
| **PLUGS** | Does it work with my stack? | Weekly end-to-end probes installing the **latest released** Ollama, [OllaBridge](https://github.com/ruslanmv/ollabridge), LangGraph, CrewAI, Langflow, Langfuse and DeepAgents, then issuing a genuine request through the whole chain | Verified, with exact versions recorded |

**Source of truth:** [`data/registry.json`](data/) · **Design rationale:** [`ARCHITECTURE.md`](ARCHITECTURE.md)

## Table of contents

- [Why FitLab exists](#why-fitlab-exists)
- [Installation](#installation)
- [Quick start](#quick-start)
- [CLI reference](#cli-reference)
- [Configuration](#configuration)
- [Running FitLab inside an organization](#running-fitlab-inside-an-organization)
- [Data contract](#data-contract)
- [How the numbers are produced](#how-the-numbers-are-produced)
- [Automation](#automation)
- [Development](#development)
- [Contributing](#contributing)
- [Security](#security)
- [Support and versioning](#support-and-versioning)
- [Related projects](#related-projects)
- [License](#license)

## Why FitLab exists

Curated "best model for X GB" lists go stale within a quarter. FitLab is built so that the
**data outlives any individual model**:

1. **Nothing is hand-ranked.** A weekly job pulls the leading models per Hugging Face pipeline tag
   through the public API, runs each one through the FITS engine, scores them with a versioned
   formula, and opens a review PR. Humans review; automation refreshes.
2. **Claims are tested, not copied.** One model per week receives a real benchmark on a free
   T4 16 GB — the same GPU class the Colab free tier allocates — so the numbers transfer directly.
3. **The community is the sensor network.** Anyone can run the
   [one-click Colab notebook](colab/benchmark_colab.ipynb) or the local benchmark script; the result
   is schema-validated and sanity-bounded by CI before it is merged, with attribution.
4. **Writes are gated.** Every JSON is validated against [`data/schema/`](data/schema/), and all
   automation writes through pull requests — never directly to the default branch.
5. **Compatibility is re-verified, not assumed.** A green PLUGS cell means the integration worked
   *this week*, at version pins recorded alongside the result.

## Installation

**Requirements:** Python 3.10 or newer. [Ollama](https://ollama.com) is required only for the
`bench` command and the wizard's benchmarking step — fit checks need no GPU and no runtime.

FitLab is published to PyPI from tagged GitHub releases via
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/):

```bash
pip install fitlab
```

Until the first release lands on PyPI, install from source. The Makefile bootstraps
[uv](https://docs.astral.sh/uv/), creates `./.venv` and installs everything:

```bash
git clone https://github.com/ruslanmv/fitlab && cd fitlab
make install
make run
```

Or without make, using any tool that reads `pyproject.toml`:

```bash
pip install .                     # or: uv sync
fitlab --version
```

The model list refreshes itself at runtime: the CLI fetches the weekly `registry.json`, caches it
for 24 hours, and falls back to a bundled seed when offline. Upgrading the package is not required
to see new models.

## Quick start

Run the wizard with no arguments — it detects your hardware, shows what fits, and offers to
benchmark your selection:

```bash
fitlab
```

```text
// llm fitlab — fit-check console · v0.1.0
your rig: NVIDIA GeForce RTX 3060 · 12.0 GB VRAM · ~360 GB/s
registry 2026-08-24 · 22 models · live (weekly)

  1 colab-free — Top 10 — Colab free / Kaggle (T4 16 GB)
  2 local-12gb — Top 10 — local inference, low-end GPU (12 GB)
  3 multimodal — Top multimodal (vision-language) for 12–16 GB
  4 all        — everything, deduplicated
? Category 1-4 (1)
```

Or check a specific GPU profile without running anything:

```bash
fitlab check --gpu rtx3060-12 --category local-12gb --limit 6
```

```text
                  fit check · RTX 3060 12GB · Q4_K_M · 8K ctx
┏━━━┳━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━┳━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ # ┃ model             ┃ params ┃ est GB ┃ verdict ┃ ~tok/s ┃ ollama tag      ┃
┡━━━╇━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━╇━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 1 │ Qwen/Qwen3-8B     │ 8.2B   │ 7.2    │ FITS    │ ~44.8  │ qwen3:8b        │
│ 2 │ ibm-granite/gran… │ 8.1B   │ 7.3    │ FITS    │ ~45.4  │ granite3.3:8b   │
│ 3 │ meta-llama/Llama… │ 8.0B   │ 7.0    │ FITS    │ ~45.9  │ llama3.1:8b     │
│ 4 │ mistralai/Minist… │ 8.0B   │ 7.1    │ FITS    │ ~45.9  │ ministral-8b    │
│ 5 │ Qwen/Qwen3-14B    │ 14.8B  │ 11.5   │ TIGHT   │ ~24.8  │ qwen3:14b       │
│ 6 │ deepseek-ai/Deep… │ 14.8B  │ 11.8   │ TIGHT   │ ~24.8  │ deepseek-r1:14b │
└───┴───────────────────┴────────┴────────┴─────────┴────────┴─────────────────┘
```

After a benchmark, the CLI prints a prefilled submission link so your result can join the public
dataset. Submitting is entirely optional; nothing is transmitted automatically.

## CLI reference

| Command | Purpose |
|---|---|
| `fitlab` | Interactive wizard: detect → fit table → select → benchmark |
| `fitlab detect` | Print detected hardware (NVIDIA via `nvidia-smi`, Apple Silicon via `sysctl`, CPU fallback) |
| `fitlab check` | Fit table for a GPU profile. No model download, no GPU required |
| `fitlab bench` | Benchmark one model non-interactively; writes a schema-valid JSON result |
| `fitlab update` | Force a registry refresh, bypassing the 24-hour cache |
| `fitlab --version` | Print the installed version |

**`fitlab check`**

| Flag | Default | Values |
|---|---|---|
| `--gpu` | `auto` | `auto`, `rtx3060-12`, `t4-16`, `rtx4070-12`, `rtx4060ti-16`, `apple-m-16`, `cpu-only` |
| `--quant` | `Q4_K_M` | `Q2_K`, `Q3_K_M`, `Q4_0`, `Q4_K_M`, `Q5_K_M`, `Q6_K`, `Q8_0`, `MXFP4`, `F16` |
| `--ctx` | `8192` | `4096`, `8192`, `16384`, `32768` |
| `--category` | `all` | A registry category id, or `all` |
| `--limit` | unset | Maximum rows to print |

**`fitlab bench`**

| Flag | Default | Notes |
|---|---|---|
| `--model` | *required* | Ollama tag (`qwen3:8b`) or registry slug (`qwen3-8b`) |
| `--reps` | `3` | Repetitions; `gen_tps` is reported as the median |
| `--context` | `4096` | Context length for the run |
| `--out` | `fitlab-results` | Output directory for the result JSON |
| `--source` | `community-local` | `ci-kaggle`, `ci-actions-cpu`, `community-colab`, `community-local`, `maintainer` |
| `--submitter` | empty | GitHub handle recorded for attribution |
| `--dry-run` | off | Print the fit verdict and execution plan without running |

Benchmarks are seeded and fixed-length (`fitlab-v1-256` prompt, `num_predict 256`, `seed 42`) so
runs are comparable across machines and across time.

## Configuration

All configuration is by environment variable — no config file, no implicit state beyond the cache.

| Variable | Default | Purpose |
|---|---|---|
| `FITLAB_REGISTRY_URL` | Public `registry.json` on the default branch | Point the CLI at a different registry — an internal mirror, a pinned commit, or a `file://` path |
| `FITLAB_CACHE` | `~/.cache/fitlab` | Cache directory for the fetched registry (24-hour TTL) |
| `FITLAB_REPO` | `OWNER/llm-fitlab` | Repository used to build benchmark submission links |
| `OLLABRIDGE_URL` | `http://127.0.0.1:11435/v1` | Gateway endpoint used by the PLUGS compatibility probes |
| `KAGGLE_USERNAME`, `KAGGLE_KEY` | unset | Required only by the weekly GPU benchmark workflow |

Network behaviour is deliberately narrow and predictable:

- `check`, `detect` and the wizard's fit stage perform **one** HTTPS GET for the registry.
- Nothing is uploaded. Benchmark results are written to local disk; sharing is a manual step.
- With no network, the CLI degrades in a defined order: fresh cache → stale cache → bundled seed,
  and the active source is printed on every run.

## Running FitLab inside an organization

**Internal registry mirror.** Fit verdicts and rankings are just data. To decouple from the public
repository — for change control, or to publish an approved-model list — host your own
`registry.json` and point clients at it:

```bash
export FITLAB_REGISTRY_URL="https://models.internal.example.com/fitlab/registry.json"
fitlab check --gpu rtx4060ti-16
```

**Pinned, reproducible data.** Because every registry carries `schema_version`, `ranking_version`
and `generated_at`, you can pin clients to a specific commit and get byte-identical results:

```bash
export FITLAB_REGISTRY_URL="https://raw.githubusercontent.com/ruslanmv/fitlab/<commit-sha>/data/registry.json"
```

**Air-gapped use.** Ship a registry file with your image and reference it directly; no egress is
required for fit checks:

```bash
export FITLAB_REGISTRY_URL="file:///opt/fitlab/registry.json"
```

**Custom hardware fleet.** Add your GPUs to [`data/hardware.yaml`](data/hardware.yaml) — including
capability flags such as `bf16`, `flash_attn2` and `marlin` — and regenerate the registry to get a
fit matrix for your own fleet rather than the public reference profiles.

**Extending the compatibility matrix.** Add a component to [`data/stack.yaml`](data/stack.yaml)
with a `probe` that exercises it end to end, and it will be verified on the same weekly cadence as
the rest of the stack.

## Data contract

Everything downstream depends on `data/`, and only on `data/`. The site is a renderer; the CLI is a
client. Both read the same documents.

```
data/
  registry.json              generated — the published source of truth (do not hand-edit)
  models.seed.yaml           human-curated seeds and editorial notes
  hardware.yaml              GPU profiles: VRAM, bandwidth, architecture capability flags
  stack.yaml                 ecosystem components and version-discovery rules
  benchmarks/                append-only measured results, one JSON per run
  schema/
    model.schema.json        registry model entries
    benchmark.schema.json    benchmark result submissions
```

`registry.json` top-level keys: `schema_version`, `generated_at`, `ranking_version`, `models`,
`categories`, `benchmarks_latest`, `compat`.

Stability guarantees:

- Every document carries `schema_version`, so historical data stays parseable indefinitely.
- `data/benchmarks/` is append-only; results are never rewritten or deleted.
- Models withdrawn from Hugging Face are marked `stale` and hidden after 60 days — never removed.
- Fit values are labeled `estimated` until a real benchmark upgrades them to `measured`.

## How the numbers are produced

**FITS** is arithmetic, not opinion:

```
est_vram = weights(params × bpw / 8 × 1.03)
         + kv_cache(2 × layers × kv_heads × head_dim × context × 2 bytes)
         + runtime_overhead
```

Verdicts are thresholds against the profile's VRAM: `fits` at ≤ 92%, `tight` at ≤ 100%,
`offload` at ≤ 160% (partial GPU offload viable), otherwise `no`. Mixture-of-experts models are
sized by *total* parameters for memory and by *active* parameters for speed; the registry stores
both.

**RUNS** measures load time, time-to-first-token, generation tokens/second and peak VRAM through
Ollama's `/api/generate` timing fields — chosen because Ollama is what the target audience actually
runs. The weekly rotation is `week_number % queue_length`, so every model is measured and
periodically re-measured.

**Ranking** is a transparent, versioned scored sort (currently `ranking_version` 1.1):

```
score = 0.35·fit + 0.25·hf_momentum + 0.25·speed + 0.15·plugs
```

where `speed` uses a measured value when one exists and a bandwidth-model estimate discounted by
0.5× otherwise. Changing the weights requires a version bump, making every rank shift visible in
the pull request diff.

## Automation

| Workflow | Schedule | What it does |
|---|---|---|
| [`weekly-sync`](.github/workflows/weekly-sync.yml) | Wed 04:41 UTC | Refreshes the registry from Hugging Face, re-runs the FITS engine, validates against the schema, opens a review PR |
| [`weekly-benchmark`](.github/workflows/weekly-benchmark.yml) | Mon 03:17 UTC | Benchmarks the rotation model on a free T4 via the Kaggle Kernels API, with a CPU lane as fallback, and runs the PLUGS probes |
| [`community-submission`](.github/workflows/community-submission.yml) | On labeled issues | Extracts, schema-validates and sanity-bounds a submitted result, then merges it via PR with attribution |
| [`publish-pypi`](.github/workflows/publish-pypi.yml) | On release | Builds, smoke-tests and publishes the package using PyPI Trusted Publishing (OIDC, no long-lived token) |

Operational properties: workflows use `concurrency` groups to prevent overlapping writes, every
script is idempotent and safe to re-run, and degradation is graceful — if Hugging Face is
unreachable the sync PR is simply not opened and the last-good registry stands; if Kaggle is
unavailable the CPU lane still lands a datapoint.

## Development

The repository is driven by a Makefile backed by [uv](https://docs.astral.sh/uv/). From a clean
clone, two commands give you a working environment and a running CLI — uv is bootstrapped
automatically if it is not already installed:

```bash
git clone https://github.com/ruslanmv/fitlab && cd fitlab
make install          # creates ./.venv and installs fitlab + all tooling
make run              # launches the fit-check wizard
```

`make run` depends on `make install`, so `make run` alone is enough on a fresh clone. Run
`make help` to list every target.

| Target | What it does |
|---|---|
| `make install` | Create `./.venv` and install the package plus the `dev` extra (`uv sync --extra dev`) |
| `make run` | Launch the interactive wizard |
| `make detect` / `make check` / `make bench` / `make update` | The corresponding `fitlab` subcommands |
| `make registry` | Rebuild `data/registry.json` from Hugging Face (FITS + sync) |
| `make estimate` | Full FITS verdict matrix across seed models × reference GPUs × quantizations |
| `make probe` | Run the PLUGS end-to-end stack probes |
| `make validate` | Schema-validate every document in `data/` |
| `make lint` / `make format` | Ruff check / safe auto-fix |
| `make smoke` | Exercise the CLI end to end without downloading a model |
| `make test` | `lint` + `validate` + `smoke` |
| `make build` | Build sdist and wheel into `dist/`, then `twine check` them |
| `make site` | Serve the static site on `http://127.0.0.1:8000` |
| `make clean` / `make distclean` | Remove artifacts / also remove `./.venv` |

Targets accept overrides on the command line:

```bash
make check GPU=t4-16 CATEGORY=colab-free LIMIT=5
make bench MODEL=qwen3:14b REPS=5
```

Available variables: `GPU`, `QUANT`, `CTX`, `CATEGORY`, `LIMIT`, `MODEL`, `REPS`, `OUT`.

In a checkout, the Makefile points `FITLAB_REGISTRY_URL` at the repository's own
`data/registry.json`, so every target reads the registry you are editing and works offline.
Override it to exercise a remote fetch:

```bash
make run FITLAB_REGISTRY_URL=https://example.com/fitlab/registry.json
```

Dependency groups live in `pyproject.toml`: the `maintainer` extra carries what the `scripts/`
engines need (`pyyaml`, `jsonschema`), and `dev` adds `ruff`, `build` and `twine` on top. `uv.lock`
is committed, so `make install` resolves to identical versions on every machine.

The engines can also be invoked directly, without make:

```bash
# FITS — verdict for one model on one GPU
python scripts/estimate_vram.py --model Qwen/Qwen3-14B --gpu rtx3060-12 --quant Q4_K_M

# Registry refresh from Hugging Face (writes data/registry.json locally)
python scripts/sync_hf.py

# RUNS — benchmark on whatever hardware you are on (local, Colab, or Kaggle)
python scripts/bench_gguf.py --model qwen3:8b --out data/benchmarks/

# PLUGS — end-to-end stack probes (CPU is sufficient)
python scripts/compat_probe.py --report data/compat_report.json

# Schema validation
python scripts/validate_data.py
```

`compat_probe.py` and `validate_data.py` both exit with the number of failures, so CI fails
honestly rather than reporting a green matrix built on skipped probes.

There is no unit-test suite yet; `make test` runs linting, schema validation and a CLI smoke test.

The published site is static and reads `data/*.json` directly; serve `site/` through GitHub Pages
or any static host.

## Contributing

Benchmark results are the most valuable contribution and take about three minutes on free
infrastructure:

1. Open [`colab/benchmark_colab.ipynb`](colab/benchmark_colab.ipynb) in Colab, select a T4 runtime,
   and run all cells — or run `python scripts/bench_gguf.py --model <tag> --source community-local`
   on your own machine.
2. The final cell prints your result and a link that opens a prefilled GitHub issue.
3. CI validates the JSON and merges it; your GPU and handle appear on the page.

Model suggestions belong in `data/models.seed.yaml` only — never in the generated `registry.json`.
Ranks are computed rather than negotiated: to change a rank, change the formula in
`scripts/sync_hf.py` and bump `ranking_version`. Full guidelines are in
[`CONTRIBUTING.md`](CONTRIBUTING.md).

## Security

- **Community submissions are data, never code.** Submitted JSON is parsed and validated; it is
  never executed.
- **Untrusted code paths are isolated.** The weekly stack probes install unpinned latest releases
  and run inside ephemeral CI runners, optionally under
  [MatrixLab](https://github.com/agent-matrix/matrixlab) Docker isolation for local maintainer runs
  (see [`docker-compose.matrixlab.yml`](docker-compose.matrixlab.yml)).
- **Secrets never reach fork-triggered workflows.** `pull_request_target` is deliberately avoided.
- **Publishing uses OIDC.** PyPI Trusted Publishing means no long-lived API token exists to leak.

Please report suspected vulnerabilities privately through
[GitHub Security Advisories](https://github.com/ruslanmv/fitlab/security/advisories/new) rather
than in a public issue.

## Support and versioning

This is a community open-source project maintained on a best-effort basis; no service level is
implied. Questions and bug reports belong in
[GitHub Issues](https://github.com/ruslanmv/fitlab/issues).

Three version numbers move independently, and only the first is a package concern:

- **Package version** (`fitlab`) follows semantic versioning.
- **`schema_version`** changes only for breaking data-format changes; consumers should pin it.
- **`ranking_version`** changes whenever scoring weights change, so ranking shifts are auditable.

Benchmark results are point-in-time measurements on specific hardware and driver versions. They are
a comparative signal, not a guarantee of what your machine will do.

## Related projects

- [OllaBridge](https://github.com/ruslanmv/ollabridge) — one OpenAI-compatible URL for all your local models
- [HomePilot](https://github.com/ruslanmv/HomePilot) — local-first chat, image and video studio
- [GitPilot](https://github.com/ruslanmv/gitpilot) — Git workflow automation
- [MatrixLab](https://github.com/agent-matrix/matrixlab) — Docker isolation sandbox used for untrusted probes
- [Best-of-the-Best](https://github.com/ruslanmv/Best-of-the-Best) — the auto-curation pattern this repository generalizes

## License

**Code** is licensed under [Apache-2.0](LICENSE). **Data** in `data/` and all published registries
are licensed under [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

Benchmark results are point-in-time measurements. Models referenced remain the property of their
respective authors and are subject to their own licenses.
