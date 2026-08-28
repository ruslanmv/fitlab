"""Detect the user's GPU (NVIDIA / Apple Silicon / CPU) and map it to a FitLab profile."""
import platform
import re
import subprocess


def _sh(cmd):
    return subprocess.run(cmd, shell=True, text=True, capture_output=True)


# memory bandwidth GB/s (substring match, order matters — first hit wins)
BW = [
    ("h100", 3350), ("a100", 1555), ("l40", 864), ("rtx 6000", 960), ("a10", 600),
    ("4090", 1008), ("4080", 717), ("4070 ti", 672), ("4070", 504),
    ("4060 ti", 288), ("4060", 272), ("3090", 936), ("3080", 760),
    ("3070", 448), ("3060", 360), ("2080", 448), ("2070", 448), ("2060", 336),
    ("v100", 900), ("p100", 732), ("l4", 300), ("t4", 320),
]
PROFILES = [
    ("t4", "t4-16"), ("p100", "p100-16"), ("3060", "rtx3060-12"),
    ("4070", "rtx4070-12"), ("4060 ti", "rtx4060ti-16"), ("l4", "l4-24"),
]
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
        name, mem, drv = [s.strip() for s in q.stdout.strip().splitlines()[0].split(",")]
        cuda = _sh("nvidia-smi --query-gpu=cuda_version --format=csv,noheader").stdout.strip()
        vram = round(int(mem) / 1024, 1)
        return {"kind": "nvidia", "name": name, "vram_gb": vram,
                "bw": _lookup(name, BW, 300),
                "profile_id": _lookup(name, PROFILES, None) or f"custom-{vram:.0f}gb",
                "driver": drv, "cuda": cuda, "note": ""}
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
