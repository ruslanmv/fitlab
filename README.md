# registry-latest

Published automatically by [`weekly-sync`](../../actions/workflows/weekly-sync.yml).
Data only — no code, no history worth reading. Consumers read:

- `data/registry.json` — the full registry (`fitlab` CLI)
- `site/registry.json`  — the same document, minified (leaderboard)
- `data/ollama_catalog.json` — the Ollama library index the tags were validated against
- `data/media_registry.json` — the media lane: image + video models ranked per GPU (HomePilot)
- `data/inference_registry.json` — the inference lane: hosted models HomePilot can run, by popularity

The reviewed copy lives on `master`.
