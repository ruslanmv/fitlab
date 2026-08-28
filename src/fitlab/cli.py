"""fitlab CLI — `fitlab` with no arguments launches the wizard."""
import argparse
import json
import os
import sys
from urllib.parse import quote

from rich.console import Console
from rich.table import Table

from . import __version__, bench, estimate, hardware, registry

console = Console(highlight=False)
REPO = os.environ.get("FITLAB_REPO", "OWNER/llm-fitlab")

VERDICT_STYLE = {"fits": "bold green", "tight": "bold yellow",
                 "offload": "dark_orange", "no": "bold red", "-": "dim"}

# reference profiles for `fitlab check --gpu <id>` (mirrors data/hardware.yaml)
PROFILES = {
    "rtx3060-12": ("RTX 3060 12GB", 12, 360),
    "t4-16": ("T4 · Colab free", 15, 320),
    "rtx4070-12": ("RTX 4070 12GB", 12, 504),
    "rtx4060ti-16": ("RTX 4060 Ti 16GB", 16, 288),
    "apple-m-16": ("Apple M-series 16GB unified", 11, 120),
    "cpu-only": ("CPU only", 0, 0),
}


def ask(prompt: str, default: str = "") -> str:
    suffix = f" [dim]({default})[/]" if default else ""
    console.print(f"[green]?[/] {prompt}{suffix} ", end="")
    try:
        val = input().strip()
    except (EOFError, KeyboardInterrupt):
        console.print("\n[dim]aborted[/]")
        sys.exit(1)
    if val.lower() in {"q", "quit", "exit"}:
        console.print("[dim]bye — nothing was run[/]")
        sys.exit(0)
    return val or default


def banner():
    console.print(f"[green]// llm fitlab[/] [dim]— fit-check console · v{__version__}[/]")


def load_registry(refresh: bool = False) -> dict:
    reg, src = registry.load(refresh)
    label = {"live": "live (weekly)", "cache": "cached <24h", "stale-cache": "stale cache — offline?",
             "bundled-seed": "bundled seed — set FITLAB_REGISTRY_URL for weekly updates"}[src]
    console.print(f"[dim]registry {reg.get('generated_at', '?')} · "
                  f"{len(reg.get('models', {}))} models · {label}[/]")
    return reg


def hw_summary(hw: dict):
    vram = f"{hw['vram_gb']} GB VRAM" if hw["vram_gb"] else "no VRAM"
    console.print(f"[bold]your rig:[/] {hw['name']} · {vram} · ~{hw['bw']} GB/s"
                  + (f"  [dim]{hw['note']}[/]" if hw.get("note") else ""))


def ordered_models(reg: dict, category: str):
    cats = reg.get("categories", {})
    if category == "all":
        seen, out = set(), []
        for c in cats.values():
            for t in c.get("top", []):
                if t["id"] not in seen and t["id"] in reg["models"]:
                    seen.add(t["id"])
                    out.append(t["id"])
        return out
    return [t["id"] for t in cats.get(category, {}).get("top", []) if t["id"] in reg["models"]]


def fit_table(reg: dict, ids, hw: dict, quant: str, ctx: int, limit: int | None = None):
    tbl = Table(title=f"fit check · {hw['name']} · {quant} · {ctx // 1024}K ctx",
                title_style="bold", header_style="dim")
    for col in ("#", "model", "params", "est GB", "verdict", "~tok/s", "ollama tag"):
        tbl.add_column(col)
    rows = []
    for i, mid in enumerate(ids[:limit] if limit else ids, 1):
        m = reg["models"][mid]
        if not m.get("arch"):
            continue
        f = estimate.fit(m, quant, ctx, hw["vram_gb"], hw["bw"])
        rows.append((mid, m, f))
        tps = "—" if f["verdict"] == "no" else (f"~{f['est_tps']}" if f["est_tps"] else "n/a")
        tbl.add_row(str(i), m["hf_id"],
                    f"{m.get('active_params_b') or m['params_b']}B",
                    f"{f['est_gb']:.1f}",
                    f"[{VERDICT_STYLE[f['verdict']]}]{f['verdict'].upper()}[/]",
                    tps, m.get("ollama_tag") or "[dim]fit-only[/]")
    console.print(tbl)
    return rows


def submit_url(result: dict) -> str:
    title = f"[bench] {result['ollama_tag']} on {result['gpu']['name']}"
    body = "```json\n" + json.dumps(result, indent=2) + "\n```"
    return (f"https://github.com/{REPO}/issues/new?labels=benchmark-submission"
            f"&title={quote(title)}&body={quote(body)}")


def run_benchmarks(picked, hw, reps: int, ctx: int, submitter: str):
    if not bench.ollama_installed():
        console.print("[yellow]Ollama is not installed[/] — it is the engine FitLab benchmarks with.")
        if ask("Install now via the official script? y/N", "n").lower() == "y":
            if not bench.install_ollama():
                console.print(f"[red]install failed[/] — run manually: {bench.INSTALL_CMD}")
                return []
        else:
            console.print(f"[dim]install later with:[/] {bench.INSTALL_CMD}")
            return []
    results = []
    for mid, m, _f in picked:
        console.rule(f"[green]{m['ollama_tag']}[/]")
        try:
            result, path = bench.bench_model(
                m["ollama_tag"], hardware.schema_gpu(hw), model_id=mid, reps=reps,
                context=ctx, submitter=submitter, log=lambda s: console.print(f"[dim]{s}[/]"))
            results.append((result, path))
            console.print(f"[bold green]✓[/] {result['metrics']['gen_tps']} tok/s median · "
                          f"ttft {result['metrics']['ttft_ms']:.0f} ms · "
                          f"peak {result['metrics']['peak_vram_gb']} GB → [dim]{path}[/]")
        except Exception as e:
            console.print(f"[red]✗ {mid}: {e}[/]")
    if results:
        console.rule("[green]done")
        tbl = Table(header_style="dim")
        for col in ("model", "tok/s", "ttft ms", "load s", "peak GB"):
            tbl.add_column(col)
        for r, _ in results:
            mm = r["metrics"]
            tbl.add_row(r["ollama_tag"], str(mm["gen_tps"]), f"{mm['ttft_ms']:.0f}",
                        str(mm["load_s"]), str(mm["peak_vram_gb"]))
        console.print(tbl)
        console.print("\n[bold]share your numbers[/] — open to file a prefilled submission "
                      "(CI validates it and your rig joins the radar):")
        for r, _ in results:
            console.print(f"  [green]↗[/] {submit_url(r)[:120]}…" if len(submit_url(r)) > 120
                          else f"  [green]↗[/] {submit_url(r)}")
    return results


def wizard():
    banner()
    hw = hardware.detect()
    hw_summary(hw)
    reg = load_registry()
    console.print()

    cats = list(reg.get("categories", {}).keys()) + ["all"]
    for i, c in enumerate(cats, 1):
        title = reg["categories"][c]["title"] if c in reg.get("categories", {}) else "everything, deduplicated"
        console.print(f"  [green]{i}[/] {c} [dim]— {title}[/]")
    pick = ask(f"Category 1-{len(cats)}", "1")
    try:
        category = cats[int(pick) - 1]
    except (ValueError, IndexError):
        console.print("[red]not a valid choice[/]")
        sys.exit(1)

    ids = ordered_models(reg, category)
    rows = fit_table(reg, ids, hw, "Q4_K_M", 8192)
    benchable = [(mid, m, f) for mid, m, f in rows if m.get("ollama_tag") and f["verdict"] != "no"]

    sel = ask("Benchmark which? — e.g. 1,3 · a = all that fit · q = quit", "a")
    if sel.lower() == "a":
        picked = benchable
    else:
        picked = []
        for tok in sel.split(","):
            tok = tok.strip()
            if not tok.isdigit() or not (1 <= int(tok) <= len(rows)):
                console.print(f"[red]'{tok}' is not a row number[/]")
                sys.exit(1)
            mid, m, f = rows[int(tok) - 1]
            if not m.get("ollama_tag"):
                console.print(f"[yellow]{mid} has no ollama tag — fit-only, skipping[/]")
                continue
            picked.append((mid, m, f))
    if not picked:
        console.print("[dim]nothing to benchmark[/]")
        return
    console.print("picked: " + ", ".join(m["ollama_tag"] for _, m, _f in picked))

    reps = int(ask("Repetitions per model", "3") or 3)
    submitter = ask("GitHub handle for credit (optional)", "")
    est_pull = ", ".join(f"{f['weights_gb']:.0f} GB" for _, _m, f in picked[:4])
    console.print(f"[dim]heads-up: models download once via ollama (~{est_pull}{' …' if len(picked) > 4 else ''})[/]")
    if ask(f"Run {len(picked)} benchmark(s) now? Y/n", "y").lower() not in {"y", "yes"}:
        console.print("[dim]bye — nothing was run[/]")
        return
    run_benchmarks(picked, hw, reps, 4096, submitter)


def resolve_gpu(spec: str) -> dict:
    if spec == "auto":
        return hardware.detect()
    if spec in PROFILES:
        name, vram, bw = PROFILES[spec]
        return {"kind": "profile", "name": name, "vram_gb": vram, "bw": bw,
                "profile_id": spec, "driver": "", "cuda": "", "note": ""}
    console.print(f"[red]unknown gpu '{spec}'[/] — use auto or one of: {', '.join(PROFILES)}")
    sys.exit(1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fitlab",
                                 description="Will it run on my setup? Run `fitlab` with no arguments for the wizard.")
    ap.add_argument("--version", action="version", version=f"fitlab {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    sub.add_parser("detect", help="show detected hardware")
    sub.add_parser("update", help="refresh the weekly model registry now")

    p_check = sub.add_parser("check", help="fit table for a GPU (no benchmarking)")
    p_check.add_argument("--gpu", default="auto", help="auto or profile id, e.g. rtx3060-12")
    p_check.add_argument("--quant", default="Q4_K_M", choices=sorted(estimate.BPW))
    p_check.add_argument("--ctx", type=int, default=8192, choices=estimate.CTX_CHOICES)
    p_check.add_argument("--category", default="all")
    p_check.add_argument("--limit", type=int, default=None)

    p_bench = sub.add_parser("bench", help="benchmark one model non-interactively")
    p_bench.add_argument("--model", required=True, help="ollama tag (qwen3:8b) or registry id (qwen3-8b)")
    p_bench.add_argument("--reps", type=int, default=3)
    p_bench.add_argument("--context", type=int, default=4096)
    p_bench.add_argument("--out", default="fitlab-results")
    p_bench.add_argument("--submitter", default="")
    p_bench.add_argument("--source", default="community-local",
                         choices=["ci-kaggle", "ci-actions-cpu", "community-colab",
                                  "community-local", "maintainer"])
    p_bench.add_argument("--dry-run", action="store_true", help="show the fit + plan, don't run")

    args = ap.parse_args(argv)

    if args.cmd is None:
        wizard()
        return 0

    if args.cmd == "detect":
        banner()
        hw_summary(hardware.detect())
        return 0

    if args.cmd == "update":
        load_registry(refresh=True)
        return 0

    if args.cmd == "check":
        banner()
        hw = resolve_gpu(args.gpu)
        hw_summary(hw)
        reg = load_registry()
        ids = ordered_models(reg, args.category)
        if not ids:
            console.print(f"[red]unknown category[/] — one of: "
                          f"{', '.join(reg.get('categories', {}))}, all")
            return 1
        fit_table(reg, ids, hw, args.quant, args.ctx, args.limit)
        return 0

    if args.cmd == "bench":
        banner()
        hw = hardware.detect()
        hw_summary(hw)
        reg = load_registry()
        mid, m = None, None
        if args.model in reg.get("models", {}):
            mid, m = args.model, reg["models"][args.model]
            tag = m.get("ollama_tag")
            if not tag:
                console.print(f"[red]{mid} has no ollama tag in the registry[/]")
                return 1
        else:
            tag = args.model
        if m and m.get("arch"):
            f = estimate.fit(m, "Q4_K_M", args.context, hw["vram_gb"], hw["bw"])
            console.print(f"fit: [{VERDICT_STYLE[f['verdict']]}]{f['verdict'].upper()}[/] · "
                          f"est {f['est_gb']} GB / {hw['vram_gb']} GB")
        if args.dry_run:
            console.print(f"[dim]dry-run: would pull {tag}, run {args.reps}× "
                          f"(num_predict 256, seed 42, ctx {args.context}) → {args.out}/[/]")
            return 0
        try:
            result, path = bench.bench_model(tag, hardware.schema_gpu(hw), model_id=mid,
                                             reps=args.reps, context=args.context,
                                             source=args.source, submitter=args.submitter,
                                             out_dir=args.out,
                                             log=lambda s: console.print(f"[dim]{s}[/]"))
        except Exception as e:
            console.print(f"[red]benchmark failed: {e}[/]")
            return 1
        console.print(f"[bold green]✓[/] {result['metrics']['gen_tps']} tok/s median → {path}")
        console.print(f"[green]↗ submit:[/] {submit_url(result)}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
