"""Real benchmark via Ollama — identical output schema to scripts/bench_gguf.py."""
import datetime
import json
import shutil
import statistics
import subprocess
import threading
import time
import os
from pathlib import Path
from urllib import request

OLLAMA = "http://127.0.0.1:11434"
PROMPT_ID = "fitlab-v1-256"
PROMPT = ("You are benchmarking. Write a precise 400-word technical explanation of how "
          "KV-cache memory grows with context length in transformer decoding, with one worked example.")
INSTALL_CMD = "curl -fsSL https://ollama.com/install.sh | sh"


def api(path: str, payload: dict | None = None, timeout=600):
    data = json.dumps(payload).encode() if payload else None
    req = request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ollama_installed() -> bool:
    return shutil.which("ollama") is not None


def install_ollama() -> bool:
    return subprocess.run(INSTALL_CMD, shell=True).returncode == 0


def ensure_server(wait_s: int = 60) -> str:
    """Start `ollama serve` if needed; return its version."""
    try:
        return api("/api/version", timeout=3)["version"]
    except Exception:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, start_new_session=True)
        for _ in range(wait_s):
            time.sleep(1)
            try:
                return api("/api/version", timeout=3)["version"]
            except Exception:
                pass
    raise RuntimeError("Ollama server did not start — try `ollama serve` in another terminal")


def pull(tag: str) -> None:
    subprocess.run(["ollama", "pull", tag], check=True)


class VramPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.peak = 0.0
        self.stop = threading.Event()

    def run(self):
        while not self.stop.is_set():
            q = subprocess.run("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
                               shell=True, text=True, capture_output=True)
            if q.returncode == 0 and q.stdout.strip():
                self.peak = max(self.peak, int(q.stdout.strip().splitlines()[0]) / 1024)
            time.sleep(0.5)


def bench_model(tag: str, gpu: dict, *, model_id: str | None = None, reps: int = 3,
                context: int = 4096, source: str = "community-local", submitter: str = "",
                out_dir: str | Path = "fitlab-results", log=print) -> tuple[dict, Path]:
    """Run the standard FitLab benchmark. `gpu` is hardware.schema_gpu(...) shaped."""
    version = ensure_server()
    t0 = time.time()
    log(f"  pulling {tag} …")
    pull(tag)
    mon = VramPeak()
    mon.start()

    load_t0 = time.time()
    api("/api/generate", {"model": tag, "prompt": "hi", "stream": False,
                          "options": {"num_predict": 1}})  # cold load
    load_s = time.time() - load_t0

    tps_runs, ttfts, prompt_tps, out_tokens = [], [], [], 0
    for i in range(reps):
        r = api("/api/generate", {"model": tag, "prompt": PROMPT, "stream": False,
                                  "options": {"num_predict": 256, "num_ctx": context,
                                              "temperature": 0.0, "seed": 42}})
        tps_runs.append(r["eval_count"] / (r["eval_duration"] / 1e9))
        ttfts.append(r["prompt_eval_duration"] / 1e6)
        prompt_tps.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
        out_tokens += r["eval_count"]
        log(f"  run {i + 1}/{reps}: {tps_runs[-1]:.1f} tok/s")
    mon.stop.set()
    mon.join(timeout=2)

    details = next((m for m in api("/api/tags")["models"]
                    if m["name"].startswith(tag.split(":")[0])), {})
    quant = (details.get("details") or {}).get("quantization_level", "Q4_K_M")

    result = {
        "schema_version": "1.0",
        "model_id": model_id or tag.replace(":", "-").replace(".", ""),
        "ollama_tag": tag,
        "runtime": {"engine": "ollama", "version": version},
        "gpu": gpu,
        "quant": quant,
        "context": context,
        "metrics": {"load_s": round(load_s, 2), "ttft_ms": round(statistics.median(ttfts), 1),
                    "prompt_tps": round(statistics.median(prompt_tps), 1),
                    "gen_tps": round(statistics.median(tps_runs), 2),
                    "gen_tps_runs": [round(t, 2) for t in tps_runs],
                    "peak_vram_gb": round(mon.peak, 2), "output_tokens": out_tokens},
        "run": {"date": datetime.date.today().isoformat(), "prompt_id": PROMPT_ID,
                "repetitions": reps, "duration_s": round(time.time() - t0, 1)},
        "provenance": {"source": source, "submitter": submitter,
                       "environment": os.environ.get("FITLAB_ENV", "cli"),
                       "run_url": os.environ.get("RUN_URL", "")},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fname = f"{result['run']['date']}_{result['model_id']}_{gpu.get('profile_id') or 'unknown'}.json"
    path = out / fname
    path.write_text(json.dumps(result, indent=2))
    return result, path
