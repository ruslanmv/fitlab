# Architecture & Design Decisions

Goal: a reference page any AI engineer can trust for "what runs on my 12 GB GPU / free Colab, and does it plug into my agent stack" — built to stay correct as Hugging Face evolves, on a $0 infrastructure budget.

## 1. Core principle: data plane ≠ presentation plane

The site never contains facts. All facts live in `data/` as schema-versioned JSON/YAML; the page is a dumb renderer of `data/registry.json`. This is the same pattern as ml-tooling/best-of and ruslanmv/Best-of-the-Best, and it is the single biggest survival factor: when Hugging Face changes, only `scripts/sync_hf.py` changes — the page, the schema, and three years of benchmark history are untouched.

Rules that follow from it: automation writes through pull requests (review gate + full audit trail), `data/benchmarks/` is append-only (one JSON per run, named `YYYY-MM-DD_<model>_<gpu>.json`), `registry.json` is always regenerable from seeds + HF API + benchmark files, and every JSON carries `schema_version` so old data remains parseable forever.

## 2. The three lanes

**FITS (deterministic).** VRAM is arithmetic, not opinion: `weights(params × bits/8) + KV cache(2 × layers × kv_heads × head_dim × ctx × bytes) + runtime overhead`. `scripts/estimate_vram.py` computes a verdict per (model, quant, context, GPU): `fits / tight / offload / no`. This runs for *every* model in the registry on every sync — it is the filter that turns "top models on HF" into "top models for a 12 GB card". Estimates are labeled `estimated` until a real benchmark upgrades them to `measured`.

**RUNS (measured, weekly).** One model per week gets a real benchmark: load time, time-to-first-token, generation tokens/sec, peak VRAM (nvidia-smi polling). The rotation is `week_number % len(queue)` over the registry, so every model eventually gets measured and re-measured. Engine: Ollama's `/api/generate` timing fields (`eval_count/eval_duration`), because Ollama is the runtime the target audience actually uses — benchmark what people run, not what benchmarks best.

**PLUGS (verified, weekly).** A CPU-only CI job installs the *latest released* versions of the stack and runs a genuine request chain: `LangGraph/CrewAI/LangFlow/DeepAgents → OllaBridge (:11435/v1) → Ollama → qwen3:0.6b`. A 0.6 B model at Q4 needs ~0.6 GB RAM, so this real end-to-end test is free on a standard GitHub Actions runner. Each cell of the compat matrix records the exact version pair that passed and the run URL — "✅ langgraph 0.6.x ↔ ollabridge 1.x on 2026-08-24" is a claim with evidence, unlike every hand-maintained compat table on the internet.

## 3. The free-GPU decision (the honest part)

The requirement was "run a weekly benchmark on free Colab GPU from GitHub Actions". Free Colab **cannot** be driven headlessly: there is no execution API on the free tier, and browser automation against it violates Google's ToS and breaks constantly. Building the pipeline on that would make the repo unstable by design. So:

- **Automated weekly GPU lane → Kaggle Kernels API.** Kaggle (a Google product) offers ~30 GPU-hours/week on **T4 16 GB or P100 16 GB** with an official CLI (`kaggle kernels push/status/output`). The T4 16 GB is *the same GPU Colab's free tier allocates*, so every number is directly a "Colab free" number. The workflow pins `enable_gpu: true` and validates in-run that a T4 was actually assigned (falls back gracefully and labels P100 results as such).
- **Colab stays as the human channel.** `colab/benchmark_colab.ipynb` is the one-click community benchmark: it runs the identical `bench_gguf.py`, then emits a prefilled GitHub-issue URL. Issue-ops CI validates the JSON (schema + sanity bounds: tps within 5× of the bandwidth-model estimate, VRAM ≤ device VRAM) and merges via PR. This turns Colab's automation *weakness* into the project's *viral mechanic*: contributors see their handle and GPU on the page.
- **CPU fallback lane.** If Kaggle quota/API fails, the same benchmark runs on the Actions runner (CPU) with a small GGUF so the week never produces zero data. CPU tok/s is still a valid relative signal and is labeled `cpu`.
- **MatrixLab's role.** [agent-matrix/matrixlab](https://github.com/agent-matrix/matrixlab) is a Docker-isolation sandbox — ideal for the PLUGS lane (installing this week's untrusted latest-version stack inside `ruslanmv/matrix-lab-sandbox-python` via `matrixlab-sandbox run --workspace . --image python --cmd "python scripts/compat_probe.py"`), and for validating community-submitted scripts. It is *not* a GPU provider, so RUNS stays on Kaggle/Colab/local metal. `docker-compose.matrixlab.yml` wires it in for maintainers who want the extra isolation locally; in hosted CI the GitHub runner is already ephemeral, so the workflows call the probe directly and keep MatrixLab optional.

## 4. T4 truths the page must encode (this is the value-add)

"Fits in 16 GB" is not "works on Colab". The hardware profiles carry capability flags the FITS engine checks:

- T4 is Turing (SM 7.5): **no bfloat16** → HF models default-loading in bf16 need `torch_dtype=float16`; some newer checkpoints degrade. Flag: `bf16: false`.
- **No FlashAttention-2** (Ampere+ only) → vLLM/transformers fall back to xformers/SDPA. Flag: `flash_attn2: false`.
- Marlin/some AWQ-fused kernels need SM 8.0+ → GPTQ/AWQ still run, just slower. Flag: `marlin: false`.
- Kaggle's P100 (SM 6.0) has no tensor cores at all — results labeled separately.
- Practical ceilings at Q4_K_M with 8K context: **12 GB → ~14 B dense**; **T4 16 GB → ~20 B dense / gpt-oss-20 B MoE; 24 B is `tight`**. MoE models are ranked by *active* params for speed but *total* params for VRAM — the registry stores both.

## 5. Ranking (transparent, versioned, gameable-resistant)

Each category's Top-10 is a scored sort, never an LLM's opinion:
`score = 0.30·fit (verdict on the category's reference GPU) + 0.30·capability (log params — the strongest model that fits should win, not the fastest tiny one) + 0.25·hf_momentum (log downloads_30d, normalized) + 0.10·speed (measured tok/s, or bandwidth-model estimate discounted 0.5×) + 0.05·plugs (fraction of stack probes passing)`.
The formula lives in `scripts/build_registry.py` with a `ranking_version`; changing weights bumps the version and is visible in the PR diff. Categories mirror HF pipeline tags 1:1 (text-generation, image-text-to-text, automatic-speech-recognition, text-to-speech, sentence-similarity, text-to-image) so new HF categories are a config line, not a redesign.

## 6. Stability engineering checklist

- **Pin everything in CI** (actions by SHA, Python by minor, pip with a constraints file) — *except* the stack under test in PLUGS, which is deliberately unpinned "latest" because testing the latest is the product. The pins that passed are recorded in the output.
- **Budget guards:** Kaggle job hard-capped at 55 min; one GPU benchmark/week ≈ 4 of the 30 free GPU-hours/month of margin; Actions usage stays inside the free 2 000 min/month.
- **Graceful degradation:** HF API down → sync PR simply not opened, last-good registry stands. Kaggle down → CPU lane still commits a datapoint. A model gated/renamed on HF → marked `stale`, hidden after 60 days, never deleted (history is the asset).
- **Idempotent + retryable:** every script is safe to re-run; workflows use `concurrency` groups to prevent overlapping writes.
- **Security:** community submissions are data-only (JSON parsed, never executed); the only code paths that run untrusted-ish input are inside MatrixLab sandboxes or throwaway Kaggle kernels; secrets never reach fork-triggered workflows (`pull_request_target` avoided).
- **Naming policy:** brand names rendered exactly as upstream writes them (Ollama, OllaBridge, HomePilot, LangGraph, Langfuse, CrewAI, Langflow, Hugging Face).

## 7. Roadmap hooks (deliberately out of v1)

Quality-eval lane (small MMLU-Pro/IFEval slice on the weekly GPU run), per-model Ollama Modelfile presets, energy/watt column from nvidia-smi, HomePilot image/video-model FITS lane (ComfyUI workflows have their own VRAM math), and a `fitlab` pip CLI that answers `fitlab can-i-run qwen3:14b` from the published registry.

## 8. The `fitlab` PyPI package

`pip install fitlab` ships the same math and the same benchmark as CI, as a wizard:
detect the local GPU (NVIDIA via nvidia-smi, Apple Silicon via sysctl, CPU fallback),
fetch the **weekly registry at runtime** (24 h cache → bundled seed), let the user pick
models, then run the identical Ollama benchmark (`fitlab-v1-256` prompt, 3 seeded reps)
producing schema-valid JSON plus a prefilled community-submission URL. The bundled seed
is refreshed by the weekly-sync PR, so every release carries current data, and installed
copies never go stale because the registry is fetched live. Publishing is release-driven
via PyPI Trusted Publishing (`publish-pypi.yml`) — CLI and CI can never disagree about
what a benchmark means because they share the schema and the prompt.
