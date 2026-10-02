"""Offline, versioned hardware inventory shared by the CLI and static clients."""
from __future__ import annotations

import json
import math
import re
from importlib.resources import files


def load() -> dict:
    return json.loads(files('fitlab').joinpath('data/gpu_catalog.json').read_text(encoding='utf-8'))


def normalize_name(name: str) -> str:
    """Normalize marketing prefixes and spacing, retaining Ti/SUPER/laptop qualifiers."""
    name = re.sub(r'\b(?:nvidia|geforce)\b', '', name, flags=re.I)
    name = re.sub(r'(?<=\d)(?=ti\b)', ' ', name, flags=re.I)
    name = re.sub(r'\b(?:gpu|\d+(?:\.\d+)?\s*g(?:b)?)\b', '', name, flags=re.I)
    name = re.sub(r'[^a-z0-9]+', ' ', name.lower()).strip()
    return name


def resolve(spec: str, vram_gb: float | None = None) -> dict | None:
    gpus = load()['gpus']
    exact = next((g for g in gpus if g['id'] == spec), None)
    if exact:
        return exact if vram_gb is None or abs(exact['vram_gb'] - vram_gb) < 0.5 else None
    # A bare number is shorthand only for an unambiguous RTX/GTX model.
    if re.fullmatch(r'\d{4}', spec.strip()):
        spec = ('RTX ' if int(spec) >= 2000 else 'GTX ') + spec
    match = normalize_name(spec)
    memory = re.search(r'(\d+(?:\.\d+)?)\s*g(?:b)?\b', spec, re.I)
    if vram_gb is None and memory:
        vram_gb = float(memory[1])
    candidates = [g for g in gpus if match in {
        normalize_name(g.get('model', g['name'])), normalize_name(g['name']),
        *(normalize_name(a) for a in g.get('aliases', []))}]
    if vram_gb is not None:
        candidates = [g for g in candidates if abs(g['vram_gb'] - vram_gb) < 0.5]
    return candidates[0] if len(candidates) == 1 else None


def validate(catalog: dict) -> None:
    if catalog.get('schema_version') != '1.0' or not catalog.get('gpus'):
        raise ValueError('Invalid or empty GPU catalog')
    ids = set()
    for g in catalog['gpus']:
        gid = g.get('id', '')
        if not re.fullmatch(r'[a-z0-9-]+', gid) or gid in ids:
            raise ValueError(f'Invalid or duplicate GPU id: {gid}')
        ids.add(gid)
        for field in ('vram_gb', 'bandwidth_gbs'):
            n = g.get(field)
            if n is None and field == 'bandwidth_gbs':
                continue
            if isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or n < 0 or n > (256 if field == 'vram_gb' else 10000):
                raise ValueError(f'{gid}: invalid {field}')
        if not isinstance(g.get('sources', []), list) or any(
                not isinstance(u, str) or not u.startswith('https://') for u in g.get('sources', [])):
            raise ValueError(f'{gid}: invalid source URLs')
        if not g.get('name') or not g.get('model') or g.get('form_factor') not in {
            'desktop', 'laptop', 'cloud', 'unified', 'cpu'}:
            raise ValueError(f'{gid}: missing identity')
        if g.get('vendor') == 'nvidia' and g['form_factor'] in {'desktop', 'laptop'}:
            if g['vram_gb'] <= 0 or not g.get('sources'):
                raise ValueError(f'{gid}: missing VRAM or provenance')
        if g.get('sm') not in (None, 'n/a') and not re.fullmatch(r'\d+\.\d+', g['sm']):
            raise ValueError(f'{gid}: invalid compute capability')
    if not {'rtx3060-12', 'rtx4060ti-16', 'rtx4070-12', 't4-16', 'cpu-only'} <= ids:
        raise ValueError('Missing stable reference profiles')
