"""Weekly model registry: live fetch → 24h cache → bundled seed."""
import json
import os
import time
from importlib import resources
from pathlib import Path
from urllib import request

URL = os.environ.get(
    "FITLAB_REGISTRY_URL",
    "https://raw.githubusercontent.com/OWNER/llm-fitlab/main/data/registry.json",
)
CACHE_DIR = Path(os.environ.get("FITLAB_CACHE", Path.home() / ".cache" / "fitlab"))
TTL_S = 24 * 3600


def _bundled() -> dict:
    with resources.files("fitlab").joinpath("data/seed_registry.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load(refresh: bool = False):
    """Return (registry_dict, source) where source ∈ live|cache|stale-cache|bundled-seed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cached = CACHE_DIR / "registry.json"
    if not refresh and cached.exists() and time.time() - cached.stat().st_mtime < TTL_S:
        try:
            return json.loads(cached.read_text()), "cache"
        except Exception:
            pass
    try:
        with request.urlopen(URL, timeout=10) as r:
            data = json.loads(r.read())
        if data.get("models"):
            cached.write_text(json.dumps(data))
            return data, "live"
    except Exception:
        pass
    if cached.exists():
        try:
            return json.loads(cached.read_text()), "stale-cache"
        except Exception:
            pass
    return _bundled(), "bundled-seed"
