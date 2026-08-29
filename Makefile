# FitLab — developer entrypoints.
#
#   make install    create ./.venv and install everything (uv)
#   make run        launch the fit-check wizard
#
# Every target runs inside the uv-managed virtualenv at ./.venv; uv itself is
# bootstrapped automatically if it is not already on PATH. No target requires a
# pre-activated shell — `make run` works from a clean clone.

.DEFAULT_GOAL := help
.SUFFIXES:

ROOT   := $(CURDIR)
VENV   := $(ROOT)/.venv
UV     ?= uv
PY     := $(VENV)/bin/python
FITLAB := $(VENV)/bin/fitlab
RUFF   := $(VENV)/bin/ruff
STAMP  := $(VENV)/.install-stamp

# uv's default install location, so a freshly bootstrapped uv is found in the same run.
export PATH := $(HOME)/.local/bin:$(PATH)

# A dev checkout reads the registry it is editing rather than the published one, which
# keeps every target deterministic and offline-capable. Override for a real fetch:
#   make run FITLAB_REGISTRY_URL=https://example.com/registry.json
FITLAB_REGISTRY_URL ?= file://$(ROOT)/data/registry.json
export FITLAB_REGISTRY_URL

# Tunables — override on the command line, e.g. `make check GPU=t4-16 QUANT=Q5_K_M`
GPU      ?= auto
QUANT    ?= Q4_K_M
CTX      ?= 8192
CATEGORY ?= all
LIMIT    ?= 10
MODEL    ?= qwen3:8b
REPS     ?= 3
OUT      ?= fitlab-results

.PHONY: help uv install dev run wizard detect check bench update \
        registry estimate probe validate lint format smoke test build site clean distclean

help: ## Show this help
	@awk 'BEGIN { FS = ":.*##" } \
	     /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5); next } \
	     /^[a-zA-Z0-9_-]+:.*##/ { printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nvariables: GPU QUANT CTX CATEGORY LIMIT MODEL REPS OUT\n"
	@printf "example:   make check GPU=t4-16 CATEGORY=colab-free LIMIT=5\n\n"

##@ Setup

uv: ## Ensure the uv package manager is installed
	@command -v $(UV) >/dev/null 2>&1 || { \
	  echo "==> uv not found; installing from https://astral.sh/uv"; \
	  curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@command -v $(UV) >/dev/null 2>&1 || { \
	  echo "!! uv is still not on PATH — add \$$HOME/.local/bin to PATH and re-run"; exit 1; }

install: $(STAMP) ## Create ./.venv and install fitlab + all tooling

$(STAMP): pyproject.toml | uv
	@echo "==> installing fitlab and dev tooling into $(VENV)"
	$(UV) sync --extra dev
	@mkdir -p $(dir $@) && touch $@
	@echo "==> ready: $$($(PY) -V) at $(VENV)"
	@echo "    run it with:  make run"

dev: install ## Alias for install

##@ Use it

run: install ## Launch the interactive fit-check wizard
	@$(FITLAB)

wizard: run ## Alias for run

detect: install ## Show the detected GPU / hardware profile
	@$(FITLAB) detect

check: install ## Fit table for a GPU, no download and no GPU required
	@$(FITLAB) check --gpu $(GPU) --quant $(QUANT) --ctx $(CTX) --category $(CATEGORY) --limit $(LIMIT)

bench: install ## Benchmark one model (needs Ollama). Override with MODEL=<tag>
	@$(FITLAB) bench --model $(MODEL) --reps $(REPS) --out $(OUT)

update: install ## Force-refresh the model registry, bypassing the 24h cache
	@$(FITLAB) update

##@ Maintainer engines

registry: install ## FITS + HF sync: rebuild data/registry.json
	@$(PY) scripts/sync_hf.py

estimate: install ## FITS verdict matrix across seed models x reference GPUs x quants
	@$(PY) scripts/estimate_vram.py --all

probe: install ## PLUGS: end-to-end stack compatibility probes (CPU is enough)
	@$(PY) scripts/compat_probe.py --report data/compat_report.json

##@ Quality

validate: install ## Schema-validate everything in data/
	@$(PY) scripts/validate_data.py

lint: install ## Lint src/ and scripts/ with ruff
	@$(RUFF) check src scripts

format: install ## Auto-fix what ruff can fix safely
	@$(RUFF) check --fix src scripts

smoke: install ## Exercise the CLI end to end without downloading a model
	@$(FITLAB) --version
	@$(FITLAB) check --gpu rtx3060-12 --limit 5
	@$(FITLAB) bench --model qwen3:0.6b --dry-run
	@echo "==> smoke OK"

test: lint validate smoke ## Run every check (no unit-test suite yet)
	@echo "==> all checks passed"

##@ Build & housekeeping

build: install ## Build the sdist and wheel into dist/
	@cp data/registry.json src/fitlab/data/seed_registry.json
	@$(UV) build
	@$(VENV)/bin/twine check dist/*

site: ## Serve the static site at http://127.0.0.1:8000
	@echo "==> serving $(ROOT)/site on http://127.0.0.1:8000 (Ctrl-C to stop)"
	@cd site && python3 -m http.server 8000

clean: ## Remove build artifacts and caches, keep ./.venv
	@rm -rf dist build .ruff_cache $(OUT)
	@find . -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
	@echo "==> cleaned"

distclean: clean ## Also remove ./.venv
	@rm -rf $(VENV)
	@echo "==> removed $(VENV)"
