"""Deterministic VRAM fit math — mirrors scripts/estimate_vram.py and the site JS."""
BPW = {"Q2_K": 2.6, "Q3_K_M": 3.9, "Q4_0": 4.55, "Q4_K_M": 4.85, "Q5_K_M": 5.7,
       "Q6_K": 6.6, "Q8_0": 8.5, "MXFP4": 4.25, "F16": 16.0}
WEIGHTS_OVERHEAD = 1.03
KV_BYTES = 2
RUNTIME_GB = 0.9
TIGHT, OFFLOAD = 0.92, 1.6
EFFICIENCY = 0.62
CTX_CHOICES = (4096, 8192, 16384, 32768)


def fit(model: dict, quant: str, ctx: int, vram_gb: float, bw: float) -> dict:
    a = model["arch"]
    bpw = BPW[quant]
    w = model["params_b"] * bpw / 8 * WEIGHTS_OVERHEAD
    kv = 2 * a["n_layers"] * a["n_kv_heads"] * a["head_dim"] * ctx * KV_BYTES / 1e9
    est = w + kv + RUNTIME_GB
    if vram_gb and vram_gb > 0:
        if est <= vram_gb * TIGHT:
            verdict = "fits"
        elif est <= vram_gb:
            verdict = "tight"
        elif est <= vram_gb * OFFLOAD:
            verdict = "offload"
        else:
            verdict = "no"
    else:
        verdict = "-"
    active = model.get("active_params_b") or model["params_b"]
    tps = None
    if bw:
        tps = round(bw * 1e9 / (active * 1e9 * bpw / 8 + kv * 1e9 / ctx * 64) * EFFICIENCY, 1)
    return {"weights_gb": round(w, 2), "kv_gb": round(kv, 2), "est_gb": round(est, 2),
            "verdict": verdict, "est_tps": tps}
