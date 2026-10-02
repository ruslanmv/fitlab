#!/usr/bin/env python3
"""Refresh GeForce specs from NVIDIA; retain history and emit reviewable changes.

Network failures, malformed tables, and disappearing models never delete the last
known catalog. Runtime clients consume the bundled catalog without scraping.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fitlab.gpu_catalog import validate

SOURCES = {
    'desktop': 'https://www.nvidia.com/en-us/geforce/graphics-cards/compare/',
    'laptop': 'https://www.nvidia.com/en-us/geforce/laptops/compare/',
    'laptop30': 'https://www.nvidia.com/en-sg/geforce/laptops/compare/30-series/',
    'cuda': 'https://developer.nvidia.com/cuda/gpus',
}
ARCH = {'50': ('blackwell', '12.0'), '40': ('ada', '8.9'), '30': ('ampere', '8.6'),
        '20': ('turing', '7.5'), '16': ('turing', '7.5'), '10': ('pascal', '6.1')}


def fetch(url: str) -> str:
    for attempt in range(3):
        try:
            req = Request(url, headers={'User-Agent': 'FitLab GPU inventory (+https://github.com/ruslanmv/fitlab)'})
            with urlopen(req, timeout=30) as response:
                if 'text/html' not in response.headers.get('Content-Type', ''):
                    raise ValueError(f'Expected HTML from {url}')
                raw = response.read(4_000_001)
                if len(raw) > 4_000_000:
                    raise ValueError(f'Source too large: {url}')
                return raw.decode('utf-8')
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
    raise RuntimeError('unreachable')


def rows_of(table):
    """Expand HTML rowspan/colspan so a merged value cannot shift GPU columns."""
    grid, pending = [], {}
    for tr in table.find_all('tr'):
        row, col = [], 0
        def inherited():
            nonlocal col
            while col in pending:
                text, left = pending[col]
                row.append(text)
                if left == 1:
                    del pending[col]
                else:
                    pending[col] = (text, left - 1)
                col += 1
        for cell in tr.find_all(['td', 'th'], recursive=False):
            inherited()
            for sup in cell.find_all('sup'):
                sup.decompose()
            text = cell.get_text(' ', strip=True)
            for _ in range(int(cell.get('colspan', 1))):
                row.append(text)
                span = int(cell.get('rowspan', 1))
                if span > 1:
                    pending[col] = (text, span - 1)
                col += 1
        inherited()
        grid.append(row)
    return grid


def parse_compare(html: str, form: str, source_key: str | None = None) -> list[dict]:
    out = {}
    soup = BeautifulSoup(html, 'html.parser')
    for table in soup.find_all('table'):
        rows = rows_of(table)
        if not rows or not any(re.search(r'\b(?:RTX|GTX)\s*\d{4}', h) for h in rows[0]):
            continue
        headers = rows[0]
        fields = {r[0].lower(): r[1:] for r in rows[1:] if r}
        memory = next((v for k, v in fields.items() if k.startswith('standard memory config')), None)
        if memory is None or len(memory) != len(headers) - 1:
            raise ValueError('Missing or misaligned memory row')
        for col, header in enumerate(headers[1:]):
            header = re.sub(r'\bT\s+i\b', 'Ti', header, flags=re.I)
            m = re.search(r'\b(RTX|GTX)\s*(\d{4})(?:\s*(Ti))?(?:\s*(Super))?', header, re.I)
            if not m:
                raise ValueError(f'Unknown GPU header: {header}')
            model = f'{m[1].upper()} {m[2]}' + (' Ti' if m[3] else '') + (' SUPER' if m[4] else '')
            arch, sm = ARCH.get(m[2][:2], ('unknown', None))
            if arch == 'unknown':
                raise ValueError(f'New architecture needs review: {header}')
            mem = memory[col]
            sizes = list(dict.fromkeys(float(n) for n in re.findall(r'(\d+)\s*GB\b', mem, re.I)))
            types = list(dict.fromkeys(re.findall(r'GDDR\dX?', mem)))
            if not sizes or not types:
                raise ValueError(f'Unparseable memory: {header}: {mem}')
            def value(key):
                vals = fields.get(key.lower())
                if vals is None:
                    return None
                if len(vals) != len(headers) - 1:
                    raise ValueError(f'Misaligned {key}')
                return vals[col]
            bw_text = value('Memory Bandwidth')
            bw = float(re.match(r'\d+(?:\.\d+)?', bw_text)[0]) if bw_text and re.fullmatch(
                r'\d+(?:\.\d+)?\s*GB/(?:sec|s)', bw_text) else None
            cores = value('NVIDIA CUDA Cores') or value('NVIDIA CUDA ® Cores')
            cores = int(cores.replace(',', '')) if cores and re.fullmatch(r'[\d,]+', cores) else None
            for size in sizes:
                # Preserve existing memory-profile IDs; distinguish memory technology only
                # when the same name+capacity actually has multiple variants.
                suffix = '-laptop' if form == 'laptop' else ''
                memory_types = types if len(sizes) == 1 else [types[-1]]
                for memory_type in memory_types:
                    gid = re.sub(r'\s+', '', model.lower()) + suffix + f'-{size:g}'
                    preferred = 'GDDR6X' if model == 'RTX 4070' else memory_types[0]
                    if (len(memory_types) > 1 and memory_type != preferred) or '(G5)' in header or '(G6)' in header:
                        gid += '-' + memory_type.lower()
                    g = dict(id=gid, model=model + (' Laptop' if form == 'laptop' else ''),
                             name=f'GeForce {model}' + (' Laptop' if form == 'laptop' else '') +
                                  f' · {size:g} GB {memory_type}', vendor='nvidia', form_factor=form,
                             vram_gb=size, bandwidth_gbs=bw, arch=arch, sm=sm, memory_type=memory_type,
                             cuda_cores=cores, source_compute_capability=value('CUDA Capability'), sources=[SOURCES[source_key or form], SOURCES['cuda']],
                             flags={'bf16': arch in {'ampere', 'ada', 'blackwell'},
                                    'tensor_cores': m[1].upper() == 'RTX'},
                             notes='Reference specs; board and laptop power limits vary. '
                                   'Speed is estimated only when memory bandwidth is known.')
                    if gid in out and out[gid] != g:
                        raise ValueError(f'Conflicting source rows: {gid}')
                    out[gid] = g
    if form == 'desktop' and not {'blackwell', 'ada', 'ampere', 'turing'} <= {g['arch'] for g in out.values()}:
        raise ValueError('Missing a supported desktop generation')
    if len(out) < (35 if form == 'desktop' else 9 if source_key == 'laptop30' else 10):
        raise ValueError(f'Incomplete {form} comparison: {len(out)} variants')
    return list(out.values())


def parse_cuda(html: str) -> dict:
    soup = BeautifulSoup(html, 'html.parser')
    out = {}
    for tr in soup.select('table tr'):
        cells = tr.find_all(['td', 'th'], recursive=False)
        if not cells or not re.fullmatch(r'\d+\.\d+', cells[0].get_text(strip=True)):
            continue
        sm = cells[0].get_text(strip=True)
        for text in cells[2].stripped_strings:
            m = re.fullmatch(r'GeForce ((?:RTX|GTX) \d{4}(?: Ti)?(?: SUPER)?)', text, re.I)
            if m:
                out[m[1].lower()] = sm
    if len(out) < 20:
        raise ValueError('Incomplete CUDA GPU list')
    return out


def merge(old: dict, scraped: list[dict], cuda: dict, hashes: dict, now: str) -> tuple[dict, dict]:
    result = copy.deepcopy(old)
    existing = {g['id']: g for g in result['gpus']}
    changes, conflicts = [], []
    for g in scraped:
        g = copy.deepcopy(g)
        g['catalog_origin'] = 'nvidia-compare'
        base_model = g['model'].removesuffix(' Laptop').lower()
        cc = cuda.get(base_model)
        if cc:
            g['sm'] = cc
        source_cc = g.get('source_compute_capability')
        if g['form_factor'] == 'laptop' and source_cc and source_cc != g['sm']:
            conflicts.append({'id': g['id'], 'field': 'sm', 'comparison': source_cc, 'family': g['sm']})
            g['sm'] = None
            g['notes'] += ' Official comparison and family CUDA listing disagree on compute capability; runtime compatibility needs verification.'
        previous = existing.get(g['id'])
        if previous:
            # Maintainer-reviewed performance and capability fields are never overwritten
            # by absent scrape values. Generic capability flags are not kernel guarantees.
            for key in ('reference', 'aliases', 'bandwidth_source'):
                if key in previous:
                    g[key] = previous[key]
            if g['bandwidth_gbs'] is None:
                g['bandwidth_gbs'] = previous['bandwidth_gbs']
            if previous.get('flags'):
                g['flags'] = previous['flags']
            for key in ('vram_gb', 'sm', 'memory_type'):
                if previous.get(key) is not None and previous[key] != g[key]:
                    conflicts.append({'id': g['id'], 'field': key, 'old': previous[key], 'new': g[key]})
        if previous != g:
            changes.append({'id': g['id'], 'change': 'updated' if previous else 'added'})
        existing[g['id']] = g
    observed_models = {g['model'].removesuffix(' Laptop').lower() for g in existing.values()}
    orphan_models = sorted(set(cuda) - observed_models)
    scraped_ids = {g['id'] for g in scraped}
    missing = sorted(g['id'] for g in old['gpus'] if g.get('catalog_origin') == 'nvidia-compare'
                     and g['id'] not in scraped_ids)
    for gid in scraped_ids:
        existing[gid]['catalog_origin'] = 'nvidia-compare'
    result['gpus'] = sorted(existing.values(), key=lambda g: g['id'])
    if changes or result.get('source_hashes') != hashes:
        result['updated_at'] = now
        result['source_hashes'] = hashes
    validate(result)
    report = dict(changes=changes, conflicts=conflicts, orphan_models=orphan_models,
                  missing_from_sources=missing, sources=SOURCES)
    return result, report


def artifacts(catalog: dict) -> dict[Path, str]:
    validate(catalog)
    data = json.dumps(catalog, indent=2, ensure_ascii=False) + '\n'
    # Classic script works with file:// and hosted static pages, with no runtime CDN.
    js = '// Generated by scripts/gpu_catalog.py; edit the source catalog, then --build.\n'
    js += 'globalThis.FitLabHardware = ' + json.dumps(catalog, ensure_ascii=True, separators=(',', ':')) + ';\n'
    ui = (ROOT / 'web/hardware-ui.js').read_text(encoding='utf-8')
    return {ROOT / 'site/hardware-ui.js': ui, ROOT / 'huggingface-space/hardware-ui.js': ui,
            ROOT / 'data/gpu_catalog.json': data,
            ROOT / 'src/fitlab/data/gpu_catalog.json': data,
            ROOT / 'site/hardware.js': js, ROOT / 'huggingface-space/hardware.js': js}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--build', action='store_true', help='regenerate offline runtime copies')
    ap.add_argument('--check', action='store_true', help='fail if runtime copies differ; no network')
    ap.add_argument('--snapshots', type=Path, help='read desktop.html, laptop.html, cuda.html offline')
    ap.add_argument('--report', type=Path, default=ROOT / 'data/gpu_refresh_report.json')
    args = ap.parse_args()
    old = json.loads((ROOT / 'data/gpu_catalog.json').read_text())
    catalog = old
    report = None
    if not args.build and not args.check:
        pages = {k: (args.snapshots / f'{k}.html').read_text() if args.snapshots else fetch(u)
                 for k, u in SOURCES.items()}
        cuda = parse_cuda(pages['cuda'])
        catalog, report = merge(old, parse_compare(pages['desktop'], 'desktop') +
                                parse_compare(pages['laptop'], 'laptop') +
                                parse_compare(pages['laptop30'], 'laptop', 'laptop30'), cuda,
                                {k: hashlib.sha256(v.encode()).hexdigest() for k, v in pages.items()},
                                datetime.now(timezone.utc).date().isoformat())
    rendered = artifacts(catalog)  # validate EVERYTHING before writing any files
    if args.check:
        stale = [str(p.relative_to(ROOT)) for p, content in rendered.items()
                 if not p.exists() or p.read_text(encoding='utf-8') != content]
        if stale:
            raise ValueError('Stale catalog artifacts: ' + ', '.join(stale))
    else:
        for path, content in rendered.items():
            path.write_text(content, encoding='utf-8')
        if report is not None:
            args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(f'GPU catalog: {len(catalog["gpus"])} profiles; artifacts validated')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError) as e:
        print(f'GPU refresh failed; existing catalog retained: {e}', file=sys.stderr)
        raise SystemExit(1)
