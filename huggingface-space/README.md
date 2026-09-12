---
title: LLM FitLab Leaderboard
emoji: 🧪
colorFrom: green
colorTo: gray
sdk: static
app_file: index.html
fullWidth: true
header: mini
pinned: false
license: apache-2.0
short_description: Ollama model fit, runtime and compatibility leaderboard
---

# LLM FitLab Leaderboard

Pick a GPU, see which Ollama models fit, how fast they run, and how to launch
them with **Ollama**, **OllaBridge** or **HomePilot**.

| Column | Meaning |
| --- | --- |
| Score | Deployment readiness — `30% fit + 30% capability + 25% HF momentum + 10% speed + 5% stack` |
| Fit | Live VRAM estimate for the selected rig, quant and context |
| Speed | Measured tok/s where a benchmark exists, otherwise a labelled estimate |
| Stack | `OL` Ollama · `OB` OllaBridge · `HP` HomePilot |

Task-quality scores (MMLU, pass@1, hallucination rate) are **not** collected —
the page says so rather than dressing runtime numbers up as model quality.

Filters, sort and rig selection are stored in the URL, so any view is shareable.

## Data

The page loads `./registry.json` and falls back to
[`site/registry.json`](https://github.com/ruslanmv/fitlab/blob/master/site/registry.json)
on GitHub. The registry is refreshed weekly by the `weekly-sync` workflow and
redeployed here by `sync-hf-space`.

## Deploy

Copy this directory plus `site/registry.json` (as `registry.json`) into a
Hugging Face **Static** Space. No server-side compute required.

> Deploys commit on top of the Space's existing history. Re-initialising the
> repo on each push makes the Hub treat the Space as newly created and re-show
> its "Get started with your new Static Space!" onboarding card to the owner.
