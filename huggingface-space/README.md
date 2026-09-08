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
short_description: Ollama LLM fit, runtime and software compatibility leaderboard
---

# LLM FitLab Leaderboard

Premium static leaderboard for the FitLab registry.

It compares Ollama-supported and Hugging Face models by:

- FitLab rank and ranking score
- estimated VRAM fit for a selected rig / quantization / context
- measured runtime speed when a matching benchmark exists
- clearly labeled estimated speed otherwise
- Ollama availability and one-click run commands
- software-oriented use profiles, strengths, and limitations
- agent-stack compatibility from the weekly PLUGS probe
- evaluation coverage, including an explicit warning that task-quality benchmarks are not yet part of the registry

The app tries `./registry.json` first and falls back to the public registry on GitHub. This makes it easy to deploy either as a standalone Hugging Face Space or together with a copied registry snapshot.

## Deploy

Create a Hugging Face **Static** Space and copy this directory to the Space repository root. No server-side compute is required.

For a fully self-contained snapshot, also copy `site/registry.json` from the FitLab repository to `registry.json` in the Space root.
