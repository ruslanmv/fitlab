"""LLM ranking v1.2 — HomePilot suggestions follow Hugging Face likes."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('sync_hf', ROOT / 'scripts/sync_hf.py')
sync = importlib.util.module_from_spec(spec)
sys.modules['sync_hf'] = sync
spec.loader.exec_module(sync)


def model(mid, likes, downloads, verdict, *, tag='m:latest', task='text-generation', params=8.0, score=None):
    return {
        'id': mid,
        'hf_id': f'org/{mid}',
        'category': task,
        'status': 'auto',
        'params_b': params,
        'ollama_tag': tag,
        'capabilities': [],
        'hf_stats': {'likes': likes, 'downloads_30d': downloads},
        'fits': {'rtx3060-12': {'Q4_K_M': {'verdict': verdict, 'est_tps': 40}}},
        'scores': {'rtx3060-12': 0.5 if score is None else score},
    }


class PopularityTests(unittest.TestCase):
    def test_likes_outrank_downloads(self):
        loved = sync.popularity({'likes': 10_000, 'downloads_30d': 0})
        pulled = sync.popularity({'likes': 0, 'downloads_30d': 10_000_000})
        self.assertGreater(loved, pulled)

    def test_the_most_liked_model_that_fits_is_first(self):
        registry = {
            'famous': model('famous', 9000, 10, 'fits', score=0.4),
            'busy': model('busy', 100, 5_000_000, 'fits', score=0.9),
            'too-big': model('too-big', 50_000, 10, 'no', score=0.99),
            'no-install': model('no-install', 40_000, 10, 'fits', tag=None, score=0.95),
        }
        # Scores are applied from fits; the hand-written scores above are replaced.
        sync.apply_scores(registry, {}, 0.5, ['rtx3060-12'])
        lists = sync.build_lists(registry, {
            'local-12gb': {'title': '12 GB', 'reference_gpu': 'rtx3060-12'},
        })
        home = [t['id'] for t in lists['homepilot-12gb']['top']]
        self.assertEqual(home[0], 'famous')
        self.assertNotIn('too-big', home)
        self.assertNotIn('no-install', home)
        self.assertGreater(lists['homepilot-12gb']['top'][0]['likes'],
                           lists['homepilot-12gb']['top'][1]['likes'])

    def test_published_homepilot_lists_follow_likes(self):
        reg = json.loads((ROOT / 'data/registry.json').read_text())
        self.assertEqual(reg['ranking_version'], '1.2')
        for cid, cat in reg['categories'].items():
            if not cid.startswith('homepilot-'):
                continue
            rows = []
            for t in cat['top']:
                model = reg['models'][t['id']]
                self.assertTrue(model.get('ollama_tag'), t['id'])
                self.assertNotEqual(model['fits'][cat['reference_gpu']]['Q4_K_M']['verdict'], 'no')
                rows.append((t['likes'], t['score'], t['id']))
            self.assertEqual(rows, sorted(rows, key=lambda r: (-r[0], -r[1], r[2])), cid)
            self.assertGreater(rows[0][0], 0, cid)


if __name__ == '__main__':
    unittest.main()
