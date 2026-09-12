#!/usr/bin/env python3
"""Ollama library catalogue + Hugging Face → Ollama tag resolution.

Auto-discovered HF models used to enter the registry with `ollama_tag: null`, which made
them invisible on a leaderboard whose whole premise is "can I `ollama run` this?". This
module scrapes the public Ollama library index (https://ollama.com/library) and each
model's tag page, then resolves an HF repo id to a concrete `name:tag` — but only ever
to a tag that actually exists in the fetched catalogue. Nothing is guessed.

Usage:
  python scripts/ollama_catalog.py                 # refresh data/ollama_catalog.json
  python scripts/ollama_catalog.py --check Qwen/Qwen3-8B google/gemma-3-12b-it
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "data/ollama_catalog.json"
BASE = "https://ollama.com"
UA = {"User-Agent": "llm-fitlab/1.0 (+https://github.com/ruslanmv/fitlab)"}

# Publisher naming that no amount of normalisation will bridge.
# Left: normalised HF base name. Right: Ollama library name.
ALIASES = {
    "llama-3.2-vision": "llama3.2-vision",
    "llama-3.2-11b-vision": "llama3.2-vision",
    "llama-4-scout": "llama4",
    "llama-4-maverick": "llama4",
    "moondream2": "moondream",
    "minicpm-v-2-6": "minicpm-v",
    "minicpm-v-4": "minicpm-v",
    "deepseek-r1-distill-qwen": "deepseek-r1",
    "deepseek-r1-distill-llama": "deepseek-r1",
    "deepseek-r1-0528-qwen3": "deepseek-r1",
    "qwq": "qwq",
    "codellama": "codellama",
    "starcoder2": "starcoder2",
    "nomic-embed-text-v1.5": "nomic-embed-text",
    "nomic-embed-text-v2-moe": "nomic-embed-text",
    "bge-m3": "bge-m3",
    "multilingual-e5-large": "bge-m3",
}

# Repo-name noise that never appears in an Ollama library name.
_STRIP_SUFFIX = re.compile(
    r"-(instruct|it|chat|chat-hf|hf|base|preview|thinking|reasoner|distill|"
    r"fp8|fp16|bf16|nvfp4|mxfp4|awq|gptq|gguf|int4|int8|w4a16|w8a8|"
    r"unsloth-bnb-4bit|bnb-4bit|abliterated|uncensored)$"
)
_DATE_SUFFIX = re.compile(r"-(20\d{2}|\d{4})$")          # -2507, -2410, -0528
_SIZE = re.compile(r"(?<![a-z0-9.])(\d+(?:\.\d+)?)b(?![a-z0-9])")
_MOE = re.compile(r"-a\d+(?:\.\d+)?b")                   # -a3b, -a22b (active-param marker)


def _get(url: str, timeout: int = 45) -> str:
    with request.urlopen(request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read().decode("utf-8", "replace")


def library_names() -> list[str]:
    html = _get(f"{BASE}/library")
    return sorted(set(re.findall(r'href="/library/([a-z0-9][a-z0-9._-]*)"', html)))


def model_tags(name: str) -> list[str]:
    try:
        html = _get(f"{BASE}/library/{name}/tags")
    except (error.URLError, error.HTTPError, TimeoutError):
        return []
    tags = set(re.findall(rf'href="/library/{re.escape(name)}:([A-Za-z0-9._-]+)"', html))
    return sorted(tags)


def build(workers: int = 8) -> dict:
    names = library_names()
    if not names:
        raise RuntimeError("Ollama library index returned no models")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        tag_lists = list(pool.map(model_tags, names))
    models = {n: {"tags": t} for n, t in zip(names, tag_lists)}
    return {"schema_version": "1.0",
            "fetched_at": datetime.date.today().isoformat(),
            "source": f"{BASE}/library",
            "count": len(models),
            "models": models}


def load(refresh: bool = False) -> dict:
    """Catalogue from disk, refreshed on request. Never raises for a missing file."""
    if not refresh and CATALOG_PATH.exists():
        try:
            return json.loads(CATALOG_PATH.read_text())
        except json.JSONDecodeError:
            pass
    if refresh:
        cat = build()
        CATALOG_PATH.write_text(json.dumps(cat, indent=2, sort_keys=True))
        return cat
    return {"models": {}, "fetched_at": None}


# --------------------------------------------------------------------------- resolution
def _normalise(hf_id: str) -> tuple[str, str | None]:
    """(base name, size token) from an HF repo id, e.g. Qwen/Qwen3-8B -> ("qwen3", "8b")."""
    name = hf_id.split("/")[-1].lower().replace("_", "-")
    name = _MOE.sub("", name)
    for _ in range(4):                                   # Qwen3-4B-Instruct-2507 needs both passes
        new = _DATE_SUFFIX.sub("", _STRIP_SUFFIX.sub("", name))
        if new == name:
            break
        name = new
    m = _SIZE.search(name)
    size = f"{m.group(1)}b" if m else None
    base = _SIZE.sub("", name).strip("-").replace("--", "-") if size else name
    return base.strip("-"), size


def _candidates(base: str, size: str | None = None) -> list[str]:
    """Ollama library names worth trying, most specific first. Verified against the catalogue."""
    out = [base]
    if size:                                             # ministral-8b, granite4-350m: size is in the name
        out.append(f"{base}-{size}")
    if base in ALIASES:
        out.insert(0, ALIASES[base])
    # gemma-3 -> gemma3, llama-3.1 -> llama3.1, phi-4 -> phi4, granite-3.3 -> granite3.3
    out.append(re.sub(r"([a-z])-(\d)", r"\1\2", base))
    # qwen2.5-vl -> qwen2.5vl
    out.append(base.replace("-vl", "vl"))
    out.append(re.sub(r"([a-z])-(\d)", r"\1\2", base).replace("-vl", "vl"))
    out.append(base.replace(".", ""))                    # mistral-small-3.2 -> mistral-small-32
    seen, uniq = set(), []
    for c in out:
        c = c.strip("-")
        if c and c not in seen:
            seen.add(c)
            uniq.append(c)
    return uniq


def resolve(hf_id: str, catalog: dict) -> str | None:
    """Concrete `name:tag` that exists in the catalogue, or None. Never guesses."""
    models = catalog.get("models") or {}
    if not models:
        return None
    base, size = _normalise(hf_id)
    for cand in _candidates(base, size):
        entry = models.get(cand)
        if not entry:
            continue
        tags = entry.get("tags") or []
        if size and cand.endswith("-" + size):           # size lives in the name, not the tag
            return f"{cand}:latest" if "latest" not in tags and tags else cand
        if size:
            # Exact size tag wins; then a size-prefixed variant (8b-instruct-q4_K_M).
            if size in tags:
                return f"{cand}:{size}"
            prefixed = [t for t in tags if t.startswith(size + "-")]
            if prefixed:
                return f"{cand}:{min(prefixed, key=len)}"
            continue                                     # right family, wrong size → not this model
        if "latest" in tags:
            return cand
        if tags:
            return f"{cand}:{min(tags, key=len)}"
    return None


def tag_exists(tag: str, catalog: dict) -> bool:
    """True when `name` or `name:tag` is really in the fetched library."""
    models = catalog.get("models") or {}
    if not models:
        return True                                      # no catalogue → do not revoke on no evidence
    name, _, version = tag.partition(":")
    entry = models.get(name)
    if entry is None:
        return False
    return not version or version in (entry.get("tags") or [])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", nargs="*", metavar="HF_ID", help="resolve these ids and exit")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    if args.check:
        cat = load()
        if not cat.get("models"):
            print("no catalogue on disk — run without --check first")
            return 1
        for hf_id in args.check:
            print(f"{hf_id:<52} -> {resolve(hf_id, cat) or '(no Ollama tag)'}")
        return 0

    cat = build(workers=args.workers)
    CATALOG_PATH.write_text(json.dumps(cat, indent=2, sort_keys=True))
    tagged = sum(len(m["tags"]) for m in cat["models"].values())
    print(f"✓ ollama catalogue: {cat['count']} models, {tagged} tags → {CATALOG_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
