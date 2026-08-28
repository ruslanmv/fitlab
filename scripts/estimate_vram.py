#!/usr/bin/env python3
"""FITS engine: deterministic VRAM estimation → verdict per (model, quant, context, GPU).

    est = weights(params × bpw/8 × overhead) + kv(2 × layers × kv_heads × head_dim × ctx × bytes) + runtime

Verdicts: fits (≤ vram×0.92) · tight (≤ vram) · offload (≤ vram×1.6, partial GPU offload viable) · no.
Also emits a bandwidth-model speed estimate (tok/s ≈ bandwidth / bytes-read-per-token) so unbenchmarked
models get a labeled `estimated` speed. MoE decode reads only active experts.

Usage:
  python scripts/estimate_vram.py --model Qwen/Qwen3-14B --gpu rtx3060-12 --quant Q4_K_M [--context 8192]
  python scripts/estimate_vram.py --all   # verdict matrix for every seed model × reference GPU × common quants
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

# GGUF bits-per-weight (effective, incl. scales) — extend as new quants appear.
BPW = {"Q2_K": 3.35, "Q3_K_M": 3.91, "Q4_K_M": 4.85, "Q5_K_M": 5.69, "Q6_K": 6.59,
       "Q8_0": 8.50, "MXFP4": 4.25, "F16": 16.0, "BF16": 16.0}


def load_data():
    hw = yaml.safe_load((ROOT / "data/hardware.yaml").read_text())
    seeds = yaml.safe_load((ROOT / "data/models.seed.yaml").read_text())
    return hw, seeds


def estimate(model: dict, quant: str, ctx: int, gpu: dict, defaults: dict) -> dict:
    bpw = BPW[quant]
    a = model["arch"]
    weights_gb = model["params_b"] * bpw / 8 * defaults["weights_overhead"]
    kv_gb = (2 * a["n_layers"] * a["n_kv_heads"] * a["head_dim"]
             * ctx * defaults["kv_cache_dtype_bytes"]) / 1e9
    est = weights_gb + kv_gb + defaults["runtime_overhead_gb"]

    vram = gpu["vram_gb"]
    if vram <= 0:
        verdict = "offload"                       # cpu profile: everything "runs", nothing "fits"
    elif est <= vram * defaults["tight_threshold"]:
        verdict = "fits"
    elif est <= vram:
        verdict = "tight"
    elif est <= vram * defaults["offload_threshold"]:
        verdict = "offload"
    else:
        verdict = "no"

    # bandwidth speed model: decode reads (active) weights + KV once per token
    active_b = model.get("active_params_b") or model["params_b"]
    bytes_per_tok = active_b * 1e9 * bpw / 8 + kv_gb * 1e9 / max(ctx, 1) * 64  # small KV read term
    est_tps = round(gpu["bandwidth_gbs"] * 1e9 / bytes_per_tok * 0.62, 1)      # 0.62 = empirical efficiency

    return {"verdict": verdict, "est_vram_gb": round(est, 2), "weights_gb": round(weights_gb, 2),
            "kv_gb": round(kv_gb, 2), "context": ctx, "est_tps": est_tps, "source": "estimated"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="registry slug or HF id")
    ap.add_argument("--gpu", default="rtx3060-12")
    ap.add_argument("--quant", default="Q4_K_M", choices=BPW)
    ap.add_argument("--context", type=int)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    hw, seeds = load_data()
    defaults = hw["fit_defaults"]
    ctx = args.context or defaults["context_tokens"]
    gpus = {g["id"]: g for g in hw["gpus"]}
    models = {m["id"]: m for m in seeds["models"]} | {m["hf_id"]: m for m in seeds["models"]}

    if args.all:
        out = {}
        for m in seeds["models"]:
            out[m["id"]] = {gid: {q: estimate(m, q, ctx, g, defaults) for q in ("Q4_K_M", "Q8_0")}
                            for gid, g in gpus.items() if g.get("reference")}
        print(json.dumps(out, indent=2))
        return 0

    m, g = models.get(args.model), gpus.get(args.gpu)
    if not m:
        sys.exit(f"unknown model {args.model!r} — add it to data/models.seed.yaml or run sync_hf.py")
    if not g:
        sys.exit(f"unknown gpu {args.gpu!r} — see data/hardware.yaml")
    r = estimate(m, args.quant, ctx, g, defaults)
    flags = " ".join(f"{k}✗" for k, v in g.get("flags", {}).items() if v is False)
    print(json.dumps({"model": m["id"], "gpu": g["id"], "quant": args.quant, **r,
                      "gpu_caveats": flags or None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
