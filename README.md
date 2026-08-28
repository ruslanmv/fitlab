<p align="center">
  <img src="site/logo.svg" width="96" alt="FitLab" />
</p>

# LLM FitLab — Fits · Runs · Plugs

**The weekly-tested compatibility radar for local LLMs.**
One page that answers the three questions every local-AI engineer asks before pulling a model:

| Lane | Question | How we answer it |
|---|---|---|
| 🟩 **FITS** | Will it fit my GPU (12 GB local / Colab-free T4 16 GB)? | Deterministic VRAM math from the model config — weights + KV cache + CUDA overhead, per quantization |
| 🟦 **RUNS** | How fast, really? | A **real benchmark every week on a free T4 GPU** (Kaggle API, same GPU class as Colab free) + community-run Colab results |
| 🟪 **PLUGS** | Does it work with my stack? | Automated end-to-end probes: latest **Ollama → [OllaBridge](https://github.com/ruslanmv/ollabridge) → LangGraph / CrewAI / LangFlow / Langfuse / DeepAgents**, plus [HomePilot](https://github.com/ruslanmv/HomePilot) requirements and the top-10 PyPI libraries |

Live page: `https://<owner>.github.io/llm-fitlab/` · Data: [`data/registry.json`](data/) · Everything is CC-BY / Apache-2.0.

---


## Install the CLI

```bash
pip install fitlab
fitlab            # wizard: detects your GPU, shows this week's models, benchmarks your picks
```

```text
// llm fitlab — fit-check console · v0.1.0
your rig: NVIDIA GeForce RTX 3060 · 12.0 GB VRAM · ~360 GB/s
registry 2026-08-24 · 22 models · live (weekly)

  1 local-12gb — Top 10 — local rigs · 12 GB
  2 colab-free — Top 10 — Colab free / Kaggle · T4 16 GB
  3 multimodal — Top multimodal · vision-language
  4 all        — everything, deduplicated
? Category 1-4 (1)
```

The model list refreshes itself: the CLI pulls the weekly `registry.json` at runtime
(24 h cache, bundled seed as offline fallback) — no need to upgrade the package to see new models.

Non-interactive: `fitlab check --gpu rtx3060-12` · `fitlab bench --model qwen3:8b` ·
`fitlab detect` · `fitlab update`. After a benchmark the CLI prints a prefilled
submission link so your rig joins the public radar.

## Why this survives Hugging Face's evolution

Curated model lists die in ~3 months. FitLab is built so the **data outlives any particular model**:

1. **Nothing is hand-ranked.** A weekly workflow pulls the top models per Hugging Face pipeline tag (text-generation, image-text-to-text, ASR, TTS, embeddings…) via the public HF API, filters them through the FITS engine, and opens a review PR. Humans review, robots refresh.
2. **Claims are tested, not copied.** One model per week gets a real benchmark on a free T4 16 GB (Kaggle Kernels API — the only ToS-compliant way to automate a free GPU; Colab free has no headless API). T4 16 GB is the exact GPU Colab's free tier hands out, so results transfer 1:1.
3. **The community is the sensor network.** Anyone can run the [one-click Colab notebook](colab/benchmark_colab.ipynb) or the local script on their own GPU; it emits a signed JSON and opens a prefilled GitHub issue. CI validates and merges it. Contributors get credited on the page — that's the viral loop, and it's how the *local-GPU* matrix (3060, 4070, 4060 Ti…) fills itself with real numbers instead of estimates.
4. **Schema-versioned data, PR-gated writes.** Every JSON is validated against [`data/schema/`](data/schema/); automation writes via pull requests, never directly to `main`.
5. **Compat is re-verified, not assumed.** Every week CI installs the **latest released versions** of the ecosystem (Ollama, OllaBridge, LangGraph, CrewAI, LangFlow, Langfuse, DeepAgents + top-10 PyPI) and runs a real request through `framework → OllaBridge → Ollama → tiny model` on a plain CPU runner. Green means it worked *this week*, with the exact version pins recorded.

## Quickstart (maintainer)

```bash
git clone https://github.com/<owner>/llm-fitlab && cd llm-fitlab
pip install -r requirements.txt

# 1. Check what fits a 12 GB card at Q4_K_M
python scripts/estimate_vram.py --model Qwen/Qwen3-14B --gpu rtx3060-12 --quant Q4_K_M

# 2. Refresh the registry from Hugging Face (opens no PR locally, writes data/registry.json)
python scripts/sync_hf.py

# 3. Benchmark whatever GPU you're on (works locally, on Colab, on Kaggle)
python scripts/bench_gguf.py --model qwen3:8b --out data/benchmarks/

# 4. Probe the agent-framework stack end-to-end (CPU is fine)
python scripts/compat_probe.py --report data/compat_report.json
```

Secrets required for full automation: `KAGGLE_USERNAME`, `KAGGLE_KEY` (repo → Settings → Secrets). GitHub Pages: serve from `/site` (or fold into a Jekyll `_data/` setup like [Best-of-the-Best](https://github.com/ruslanmv/Best-of-the-Best) — the JSON contract is identical).

## Quickstart (contributor — 3 minutes, free)

1. Open [`colab/benchmark_colab.ipynb`](colab/benchmark_colab.ipynb) in Colab → Runtime → T4 GPU → Run all.
2. The last cell prints your result and a link that opens a prefilled GitHub issue.
3. CI validates it; when merged, your GPU + handle appear on the page.

## The ecosystem this page serves

[OllaBridge](https://github.com/ruslanmv/ollabridge) (one OpenAI-compatible URL for all your local models) · [HomePilot](https://github.com/ruslanmv/HomePilot) (local-first chat/image/video studio) · [GitPilot](https://github.com/ruslanmv/gitpilot) · [MatrixLab](https://github.com/agent-matrix/matrixlab) (the sandbox that runs our untrusted compat probes) · [Best-of-the-Best](https://github.com/ruslanmv/Best-of-the-Best) (the auto-curation pattern this repo generalizes).

## Repository layout

```
data/               single source of truth (JSON/YAML, schema-validated)
  registry.json     auto-generated model registry (do not hand-edit)
  models.seed.yaml  human-curated seeds + editorial notes
  hardware.yaml     GPU profiles (the rows of the compat matrix)
  stack.yaml        ecosystem components + version-discovery rules
  benchmarks/       append-only measured results (one JSON per run)
scripts/            fits / runs / plugs engines + HF sync + site build
.github/workflows/  weekly-sync · weekly-benchmark · community-submission · validate
colab/              one-click community benchmark notebook
site/               static page (reads data/*.json; GitHub Pages-ready)
```

## Prior art & honest positioning

VRAM calculators and "best model for X GB" pages exist (ModelFit, BenchLM, LocalAIMaster, r/LocalLLaMA wikis). None of them (a) re-benchmark weekly on free GPUs with public raw data, (b) test **agent-framework compatibility** against pinned latest versions, or (c) accept community results as validated PRs. That triple is the moat — and the reason an AI engineer bookmarks this page.

## License

Code Apache-2.0 · Data CC-BY-4.0. Benchmarks are point-in-time measurements; models belong to their authors.
