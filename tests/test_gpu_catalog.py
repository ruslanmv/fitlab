"""Offline regression coverage for refresh safety, device identity, and client artifacts."""
import copy
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fitlab import gpu_catalog, hardware
from fitlab.cli import resolve_gpu

spec = importlib.util.spec_from_file_location('refresh_gpu_catalog', ROOT / 'scripts/gpu_catalog.py')
refresh = importlib.util.module_from_spec(spec)
spec.loader.exec_module(refresh)
FIXTURES = ROOT / 'tests/fixtures'


class InventoryTests(unittest.TestCase):
    def test_catalog_and_artifacts(self):
        catalog = gpu_catalog.load()
        gpu_catalog.validate(catalog)
        self.assertEqual(catalog, json.loads((ROOT / 'data/gpu_catalog.json').read_text()))
        for path, content in refresh.artifacts(catalog).items():
            self.assertEqual(path.read_text(), content, str(path))

    def test_exact_variants_and_aliases(self):
        self.assertEqual(gpu_catalog.resolve('NVIDIA GeForce RTX 4060Ti, 16GB')['id'], 'rtx4060ti-16')
        self.assertEqual(gpu_catalog.resolve('RTX 4060 Ti 8G')['id'], 'rtx4060ti-8')
        self.assertIsNone(gpu_catalog.resolve('RTX 4060 Ti'))
        self.assertEqual(gpu_catalog.resolve('4090')['id'], 'rtx4090-24')
        self.assertEqual(gpu_catalog.resolve('RTX 4070 Ti SUPER')['id'], 'rtx4070tisuper-16')
        self.assertEqual(gpu_catalog.resolve('RTX 4090 Laptop GPU')['id'], 'rtx4090-laptop-16')
        self.assertEqual(gpu_catalog.resolve('RTX 3060', 8)['id'], 'rtx3060-8')
        self.assertIsNone(gpu_catalog.resolve('AD104'))  # one die can serve several consumer SKUs
        self.assertIsNone(gpu_catalog.resolve('RTX 4050'))  # laptop-only name is explicit

    def test_custom_unknown_bandwidth(self):
        hw = resolve_gpu('My future GPU', 20)
        self.assertEqual(hw['vram_gb'], 20)
        self.assertIsNone(hw['bw'])
        self.assertNotEqual(hw['profile_id'], resolve_gpu('Other GPU', 20)['profile_id'])
        for v in [0, -1, float('nan'), float('inf'), 300]:
            with self.assertRaises(ValueError):
                resolve_gpu('custom', v)
        with self.assertRaises(ValueError):
            resolve_gpu('RTX 4060 Ti')

    def test_detect_vram_not_substring(self):
        def query(cmd):
            stdout = ('NVIDIA GeForce RTX 4060 Ti, 8188, 580.1\n' if '--query-gpu' in cmd
                      else 'CUDA Version: 12.8')
            return subprocess.CompletedProcess(cmd, 0, stdout, '')
        with patch.object(hardware, '_sh', side_effect=query):
            hw = hardware.detect()
        self.assertEqual(hw['profile_id'], 'rtx4060ti-8')
        self.assertEqual(hw['cuda'], '12.8')
        self.assertEqual(hw['vram_gb'], 8)

    def test_unknown_detection_and_unavailable_smi(self):
        with patch.object(hardware, '_sh', return_value=subprocess.CompletedProcess('', 0, 'RTX 6090, 32768, 580\n', '')):
            hw = hardware.detect()
        self.assertEqual(hw['vram_gb'], 32)
        self.assertIsNone(hw['profile_id'])
        self.assertIsNone(hw['bw'])
        with patch('subprocess.run', side_effect=FileNotFoundError):
            self.assertEqual(hardware._sh('nvidia-smi').returncode, 1)

    def test_standalone_cloud_benchmark_bundle(self):
        import shutil
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shutil.copy(ROOT / 'scripts/bench_gguf.py', root / 'bench_gguf.py')
            for name in ('__init__.py', 'hardware.py', 'gpu_catalog.py', 'data/gpu_catalog.json'):
                target = root / 'src/fitlab' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(ROOT / 'src/fitlab' / name, target)
            code = """
import runpy, subprocess
m = runpy.run_path('bench_gguf.py')
m['gpu_info']()  # imports from the downloaded bundle, with no installed fitlab
from fitlab import hardware
hardware._sh = lambda cmd: subprocess.CompletedProcess(cmd, 0, 'NVIDIA GeForce RTX 4060 Ti, 8192, 580\\n' if '--query-gpu' in cmd else '', '')
assert m['gpu_info']()['profile_id'] == 'rtx4060ti-8'
"""
            result = subprocess.run([sys.executable, '-I', '-c', code], cwd=tmp,
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_duplicate_and_invalid_vram_rejected(self):
        c = gpu_catalog.load()
        c['gpus'].append(copy.deepcopy(c['gpus'][0]))
        with self.assertRaises(ValueError):
            gpu_catalog.validate(c)
        c = gpu_catalog.load(); c['gpus'][0]['vram_gb'] = float('nan')
        with self.assertRaises(ValueError):
            gpu_catalog.validate(c)


class ScraperTests(unittest.TestCase):
    def scrape(self):
        return (refresh.parse_compare((FIXTURES / 'desktop.html').read_text(), 'desktop') +
                refresh.parse_compare((FIXTURES / 'laptop.html').read_text(), 'laptop') +
                refresh.parse_compare((FIXTURES / 'laptop30.html').read_text(), 'laptop', 'laptop30'))

    def test_real_tables(self):
        gs = {g['id']: g for g in self.scrape()}
        self.assertEqual(gs['rtx5090-32']['vram_gb'], 32)
        self.assertEqual(gs['rtx5090-laptop-24']['vram_gb'], 24)
        self.assertEqual(gs['rtx5060ti-8']['bandwidth_gbs'], 448)
        self.assertEqual(gs['rtx5050-8']['memory_type'], 'GDDR6')
        self.assertEqual(gs['rtx3050ti-laptop-4']['model'], 'RTX 3050 Ti Laptop')
        self.assertIn('gtx1650-4-gddr5', gs)
        self.assertIsNone(gs['rtx3080-12']['cuda_cores'])  # shared row lists multiple SKUs
        self.assertIsNone(gs['rtx4090-laptop-16']['bandwidth_gbs'])

    def test_grid_spans(self):
        from bs4 import BeautifulSoup
        table = BeautifulSoup('<table><tr><td rowspan="2">A</td><td colspan="2">B</td></tr>'
                              '<tr><td>C</td><td>D</td></tr></table>', 'html.parser').table
        self.assertEqual(refresh.rows_of(table), [['A', 'B', 'B'], ['A', 'C', 'D']])

    def test_fail_closed_and_no_network_build(self):
        with self.assertRaises(ValueError):
            refresh.parse_compare('<html>Access denied</html>', 'desktop')
        with self.assertRaises(ValueError):
            refresh.parse_cuda('<html></html>')
        malformed = (FIXTURES / 'desktop.html').read_text().replace('32 GB GDDR7', 'unknown memory')
        with self.assertRaises(ValueError):
            refresh.parse_compare(malformed, 'desktop')
        with tempfile.TemporaryDirectory() as tmp:
            before = (ROOT / 'data/gpu_catalog.json').read_bytes()
            cmd = [sys.executable, str(ROOT / 'scripts/gpu_catalog.py'), '--snapshots', tmp]
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)
            self.assertEqual(before, (ROOT / 'data/gpu_catalog.json').read_bytes())

    def test_merge_preserves_history_and_is_idempotent(self):
        old = gpu_catalog.load()
        cuda = refresh.parse_cuda((FIXTURES / 'cuda.html').read_text())
        new, report = refresh.merge(old, self.scrape(), cuda, {'test': 'hash'}, '2026-10-02')
        again, second = refresh.merge(new, self.scrape(), cuda, {'test': 'hash'}, '2026-11-02')
        self.assertEqual(new, again)
        self.assertEqual(second['changes'], [])
        self.assertEqual(next(g for g in new['gpus'] if g['id'] == 'rtx3060-12')['bandwidth_gbs'], 360)
        self.assertIn('gtx1080ti-11', {g['id'] for g in new['gpus']})
        laptop = next(g for g in new['gpus'] if g['id'] == 'rtx5090-laptop-24')
        self.assertIsNone(laptop['sm'])  # contradictory official sources are not guarantees
        self.assertTrue(report['conflicts'])
        partial = [g for g in self.scrape() if g['id'] != 'rtx5090-32']
        retained, r = refresh.merge(new, partial, {**cuda, 'rtx 6090': '13.0'}, {}, '2026-11-02')
        self.assertIn('rtx5090-32', {g['id'] for g in retained['gpus']})
        self.assertIn('rtx5090-32', r['missing_from_sources'])
        self.assertIn('rtx 6090', r['orphan_models'])


if __name__ == '__main__':
    unittest.main()
