# GPU inventory and monthly maintenance

FitLab ships a versioned, offline GPU catalog in `data/gpu_catalog.json`. The first
snapshot has 85 profiles: NVIDIA's published RTX 50/40/30/20 desktop variants,
GTX 16 desktop variants, RTX 50/40/30 laptop variants, reviewed GTX 10 and 900
profiles, and the existing cloud/Apple/CPU reference lanes. This is broad coverage,
not a claim to enumerate every OEM card or every historical GeForce. Any unlisted
GPU can still receive VRAM fit estimates through the custom hardware fields.

## Usage

Both web pages offer GPU search, separate laptop entries, memory variants, and
custom GPU input. The selected GPU (including custom memory) survives a shared
URL. Hardware metadata shows catalog age and compatibility notes. Custom bandwidth
is optional; when it is unknown, speed is unavailable rather than guessed.

```sh
fitlab gpus --search 5060
fitlab check --gpu rtx5060ti-16 --limit 10
fitlab check --gpu 'RTX 4060 Ti 8GB'
fitlab check --gpu 'My unlisted GPU' --vram 20
fitlab check --gpu 'My unlisted GPU' --vram 20 --bandwidth 500
```

Ambiguous names require a profile ID or a VRAM variant. Desktop and laptop parts
are distinct even when the marketing numbers match. Device detection matches the
complete name and detected memory, so an 8 GB 4060 Ti never attaches to the 16 GB
benchmark profile. Unrecognized hardware uses measured VRAM and gets no invented
bandwidth or benchmark identity. Silicon codenames are not aliases: one die can
appear in multiple GPUs. No guessed PCI IDs are used to select physical devices.

## Sources and guarantees

- [NVIDIA desktop comparison](https://www.nvidia.com/en-us/geforce/graphics-cards/compare/)
- [NVIDIA laptop comparison](https://www.nvidia.com/en-us/geforce/laptops/compare/)
- [NVIDIA RTX 30 laptop comparison](https://www.nvidia.com/en-sg/geforce/laptops/compare/30-series/)
- [NVIDIA CUDA inventory](https://developer.nvidia.com/cuda/gpus)
- [NVIDIA legacy CUDA inventory](https://developer.nvidia.com/cuda-legacy-gpus)
- Legacy and bandwidth references are recorded per profile, including NVIDIA's
  900-series specs, PNY manufacturer specifications, and existing reviewed FitLab profiles.

Memory is split into separate capacity and technology variants. Raw source hashes,
source URLs, CUDA capability, architecture, memory type, CUDA cores when
unambiguous, and form factor are preserved. Missing specs stay null. NVIDIA's
Blackwell laptop comparison currently disagrees with the family CUDA inventory;
those laptop compute capabilities remain null with a warning in both the catalog
and refresh report. Do not infer a kernel/library compatibility guarantee from a
GPU's architecture or generic BF16/tensor-core flags. The original tested reference
profile capability flags are preserved.

Model memory estimates include weights, context-dependent KV cache, and runtime
buffers. A fits verdict is a capacity estimate, not a benchmark or a runtime
support guarantee. VRAM shown by the installed driver may differ from reference
capacity, and laptop TGP, driver, thermals, host memory, model architecture and
quantization affect speed. Only existing measured model × profile pairs are
labeled measured; no community token rates are copied without reproducible run
metadata. Reference ranking scores remain reference rankings. Bandwidth estimates
are optimistic ceilings, especially for CPU offload; host RAM availability is not
verified by the static page.

## Monthly refresh

`.github/workflows/monthly-gpu-catalog.yml` runs on the first day of each month at
05:15 UTC and supports manual dispatch. It fetches the four official sources,
normalizes tables, validates the result, builds runtime copies, runs offline
regressions, and opens/updates `automation/monthly-gpu-catalog` as a review PR.
There is no automatic merge. Enable **Allow GitHub Actions to create and approve
pull requests** in repository Actions settings for the default `GITHUB_TOKEN`
workflow; otherwise the PR creation step reports a permission error.

```sh
pip install -e '.[maintainer]'
python scripts/gpu_catalog.py                  # live refresh; report + runtime copies
python scripts/gpu_catalog.py --build          # regenerate copies after a reviewed edit
python scripts/gpu_catalog.py --check          # offline consistency check
python -m unittest discover -s tests
node --test tests/hardware-ui.test.cjs
```

HTTP fetches have bounded payloads, timeouts and retries. Every source must succeed
and parse correctly before files are written. Empty, malformed, unknown-generation
or substantially incomplete comparison tables fail the refresh. Models disappearing
from a source are retained and flagged in `data/gpu_refresh_report.json`; they are
never automatically deleted. The report also lists CUDA inventory models missing
from FitLab and conflicting specs for maintainer review. Known specs are preserved
when an upstream row omits them. Identical source hashes and normalized content
produce no catalog change. Git history supplies comparison and rollback; revert a
refresh commit and rerun `--build` if editing only the canonical catalog.

Runtime copies are generated into the Python package and both static pages. The
Hugging Face sync deploys both JS assets alongside the leaderboard. Consumers work
without runtime scraping or third-party catalog services. Releasing a new wheel is
required to update an installed CLI's bundled inventory; model-registry updates
remain independent.

## Launch behavior

[Ollama hardware support](https://docs.ollama.com/gpu) is the runtime authority.
Current docs require NVIDIA compute capability 5.0+ and driver 550+, or 570+ for
compute capability 5.0–6.2. Ollama also supports Apple Metal and supported AMD GPUs;
OllaBridge and HomePilot are integrations with a configured model runtime, not
replacement GPU drivers. No artificial 8 GB minimum is enforced.

```sh
ollama run qwen3:8b
ollama ps                       # inspect actual CPU/GPU placement
nvidia-smi -L                   # list physical GPU UUIDs
```

For multiple NVIDIA GPUs, set `CUDA_VISIBLE_DEVICES` **on the Ollama server** before
starting/restarting it, then run the client separately. A selector in FitLab only
changes fit estimates; a static webpage cannot execute a local process or select
a physical device. For a manually started Linux server, after stopping any existing
Ollama service:

```sh
CUDA_VISIBLE_DEVICES=GPU-your-uuid ollama serve
# In a second terminal:
ollama run qwen3:8b
```

On Windows PowerShell, set `$env:CUDA_VISIBLE_DEVICES='GPU-your-uuid'` before
`ollama serve`, or configure the environment and restart the Ollama desktop app.
OllaBridge uses the existing `ollabridge start` integration; HomePilot uses the
existing Ollama provider settings. Model-specific snippets in the leaderboard
remain the commands/configuration already validated by FitLab's stack probes.
