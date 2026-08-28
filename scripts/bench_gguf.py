#!/usr/bin/env python3
"""RUNS engine: one benchmark script for local GPUs, Colab, and Kaggle.

Measures via Ollama's /api/generate timing fields (the runtime the audience actually uses):
load_s, ttft_ms (prompt eval), gen_tps (median of N), peak VRAM (nvidia-smi polling).
Installs/starts Ollama if needed. Emits a JSON matching data/schema/benchmark.schema.json.

Usage:
  python scripts/bench_gguf.py --model qwen3:8b --out data/benchmarks/ [--reps 3] [--source ci-kaggle]
"""
from __future__ import annotations
import argparse, datetime, json, os, shutil, statistics, subprocess, threading, time
from pathlib import Path
from urllib import request

OLLAMA = "http://127.0.0.1:11434"
PROMPT_ID = "fitlab-v1-256"
PROMPT = ("You are benchmarking. Write a precise 400-word technical explanation of how "
          "KV-cache memory grows with context length in transformer decoding, with one worked example.")


def sh(cmd: str, check=True, **kw):
    return subprocess.run(cmd, shell=True, check=check, text=True, capture_output=True, **kw)


def api(path: str, payload: dict | None = None, timeout=600):
    data = json.dumps(payload).encode() if payload else None
    req = request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def ensure_ollama() -> str:
    if not shutil.which("ollama"):
        print("→ installing Ollama"); sh("curl -fsSL https://ollama.com/install.sh | sh")
    try:
        return api("/api/version")["version"]
    except Exception:
        subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(60):
            time.sleep(1)
            try:
                return api("/api/version")["version"]
            except Exception:
                pass
        raise RuntimeError("Ollama did not start")


def gpu_info() -> dict:
    q = sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits", check=False)
    if q.returncode != 0 or not q.stdout.strip():
        return {"name": "cpu", "vram_gb": 0, "driver": "", "cuda": "", "profile_id": "cpu-only"}
    name, mem, drv = [s.strip() for s in q.stdout.strip().splitlines()[0].split(",")]
    cuda = sh("nvidia-smi --query-gpu=cuda_version --format=csv,noheader", check=False).stdout.strip()
    prof = {"Tesla T4": "t4-16", "Tesla P100-PCIE-16GB": "p100-16",
            "NVIDIA GeForce RTX 3060": "rtx3060-12", "NVIDIA GeForce RTX 4070": "rtx4070-12",
            "NVIDIA GeForce RTX 4060 Ti": "rtx4060ti-16", "NVIDIA L4": "l4-24"}.get(name)
    return {"name": name, "vram_gb": round(int(mem) / 1024, 1), "driver": drv, "cuda": cuda, "profile_id": prof}


class VramPeak(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.peak = 0.0; self.stop = threading.Event()
    def run(self):
        while not self.stop.is_set():
            q = sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits", check=False)
            if q.returncode == 0 and q.stdout.strip():
                self.peak = max(self.peak, int(q.stdout.strip().splitlines()[0]) / 1024)
            time.sleep(0.5)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Ollama tag, e.g. qwen3:8b")
    ap.add_argument("--out", default="data/benchmarks")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--context", type=int, default=4096)
    ap.add_argument("--source", default="maintainer",
                    choices=["ci-kaggle", "ci-actions-cpu", "community-colab", "community-local", "maintainer"])
    ap.add_argument("--submitter", default=os.environ.get("GITHUB_ACTOR", ""))
    args = ap.parse_args()

    version = ensure_ollama()
    gpu = gpu_info()
    print(f"→ Ollama {version} · GPU: {gpu['name']} {gpu['vram_gb']} GB")

    t0 = time.time(); sh(f"ollama pull {args.model}"); print(f"→ pulled in {time.time()-t0:.0f}s")
    mon = VramPeak(); mon.start()

    load_t0 = time.time()
    api("/api/generate", {"model": args.model, "prompt": "hi", "stream": False,
                          "options": {"num_predict": 1}})            # cold load
    load_s = time.time() - load_t0

    tps_runs, ttfts, out_tokens, prompt_tps = [], [], 0, []
    for i in range(args.reps):
        r = api("/api/generate", {"model": args.model, "prompt": PROMPT, "stream": False,
                                  "options": {"num_predict": 256, "num_ctx": args.context,
                                              "temperature": 0.0, "seed": 42}})
        tps_runs.append(r["eval_count"] / (r["eval_duration"] / 1e9))
        ttfts.append(r["prompt_eval_duration"] / 1e6)
        prompt_tps.append(r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9))
        out_tokens += r["eval_count"]
        print(f"  run {i+1}: {tps_runs[-1]:.1f} tok/s")
    mon.stop.set(); mon.join(timeout=2)

    details = next((m for m in api("/api/tags")["models"] if m["name"].startswith(args.model.split(":")[0])), {})
    quant = (details.get("details") or {}).get("quantization_level", "Q4_K_M")

    result = {
        "schema_version": "1.0",
        "model_id": args.model.replace(":", "-").replace(".", ""),
        "ollama_tag": args.model,
        "runtime": {"engine": "ollama", "version": version},
        "gpu": gpu,
        "quant": quant,
        "context": args.context,
        "metrics": {"load_s": round(load_s, 2), "ttft_ms": round(statistics.median(ttfts), 1),
                    "prompt_tps": round(statistics.median(prompt_tps), 1),
                    "gen_tps": round(statistics.median(tps_runs), 2),
                    "gen_tps_runs": [round(t, 2) for t in tps_runs],
                    "peak_vram_gb": round(mon.peak, 2), "output_tokens": out_tokens},
        "run": {"date": datetime.date.today().isoformat(), "prompt_id": PROMPT_ID,
                "repetitions": args.reps, "duration_s": round(time.time() - t0, 1)},
        "provenance": {"source": args.source, "submitter": args.submitter,
                       "environment": os.environ.get("FITLAB_ENV", "local"),
                       "run_url": os.environ.get("RUN_URL", "")},
    }
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    fname = f"{result['run']['date']}_{result['model_id']}_{gpu.get('profile_id') or 'unknown'}.json"
    (out_dir / fname).write_text(json.dumps(result, indent=2))
    print(f"\n✓ {out_dir / fname}\n" + json.dumps(result["metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
