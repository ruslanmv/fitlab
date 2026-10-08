"""Detect the user's GPU (NVIDIA / Apple Silicon / CPU) and map it to a FitLab profile."""
import platform
import subprocess
import re
import shlex

from . import gpu_catalog


def _sh(cmd):
    try:
        return subprocess.run(shlex.split(cmd), text=True, capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return subprocess.CompletedProcess(cmd, 1, "", "")


# Retain non-consumer accelerator fallbacks; GeForce matching is exact and VRAM-aware.
BW = [("h100", 3350), ("a100", 1555), ("l40", 864), ("rtx 6000", 960), ("a10", 600),
      ("v100", 900), ("p100", 732), ("l4", 300), ("t4", 320)]
PROFILES = [("t4", "t4-16"), ("p100", "p100-16"), ("l4", "l4-24")]
APPLE_BW = [
    ("m4 max", 546), ("m4 pro", 273), ("m4", 120),
    ("m3 max", 400), ("m3 pro", 150), ("m3", 100),
    ("m2 max", 400), ("m2 pro", 200), ("m2", 100),
    ("m1 max", 400), ("m1 pro", 200), ("m1", 68),
]


def _lookup(name, table, default):
    low = name.lower()
    for key, val in table:
        if key in low:
            return val
    return default


def detect() -> dict:
    """Return {kind, name, vram_gb, bw, profile_id, driver, cuda, note}."""
    q = _sh("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits")
    if q.returncode == 0 and q.stdout.strip():
        try:
            name, mem, drv = [s.strip() for s in q.stdout.strip().splitlines()[0].split(",")]
            vram = round(float(mem) / 1024, 1)
            if vram <= 0:
                raise ValueError("invalid GPU memory")
        except ValueError:
            name, drv, vram = "NVIDIA GPU (memory unavailable)", "", 0
        version = re.search(r"CUDA Version:\s*([\d.]+)", _sh("nvidia-smi").stdout)
        cuda = version[1] if version else ""
        profile = gpu_catalog.resolve(name, vram)
        note = "" if profile else "Unlisted GPU or memory variant; using detected VRAM. Compatibility unverified."
        if profile and profile.get('sm') and float(profile['sm']) <= 6.2:
            note = "Current Ollama requires NVIDIA driver 570+ for compute capability 5.0–6.2."
        return {"kind": "nvidia", "name": name, "vram_gb": vram,
                "bw": profile["bandwidth_gbs"] if profile else _lookup(name, BW, None),
                "profile_id": profile["id"] if profile else _lookup(name, PROFILES, None) or None,
                "driver": drv, "cuda": cuda, "note": note}
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        mem = _sh("sysctl -n hw.memsize").stdout.strip()
        chip = _sh("sysctl -n machdep.cpu.brand_string").stdout.strip() or "Apple Silicon"
        total = int(mem) / 1e9 if mem.isdigit() else 16
        vram = round(total * 0.7, 1)  # Metal usable ≈ 70% of unified memory
        return {"kind": "apple", "name": chip, "vram_gb": vram,
                "bw": _lookup(chip, APPLE_BW, 120), "profile_id": "apple-m-16",
                "driver": "", "cuda": "",
                "note": f"unified {total:.0f} GB → ~{vram} GB usable for Metal"}
    cores = _sh("nproc").stdout.strip() or "?"
    cpu = platform.processor() or platform.machine() or "cpu"
    return {"kind": "cpu", "name": f"{cpu} ({cores} threads)", "vram_gb": 0, "bw": 0,
            "profile_id": "cpu-only", "driver": "", "cuda": "",
            "note": "no GPU detected — benchmarks run on CPU (slow but valid)"}


def schema_gpu(hw: dict) -> dict:
    """Shape the detected hardware for benchmark.schema.json's gpu object."""
    return {"name": hw["name"] if hw["kind"] != "cpu" else "cpu",
            "vram_gb": hw["vram_gb"], "driver": hw.get("driver", ""),
            "cuda": hw.get("cuda", ""), "profile_id": hw.get("profile_id")}
