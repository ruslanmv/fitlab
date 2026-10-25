"""Media lane (image + video): verdicts, ranking and the published document — all offline."""
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import jsonschema
import yaml

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('build_media', ROOT / 'scripts/build_media.py')
media = importlib.util.module_from_spec(spec)
sys.modules['build_media'] = media
spec.loader.exec_module(media)

SEED = yaml.safe_load((ROOT / 'data/media.seed.yaml').read_text())
GPUS = {g['id']: g for g in json.loads((ROOT / 'data/gpu_catalog.json').read_text())['gpus']}
SCHEMA = json.loads((ROOT / 'data/schema/media.schema.json').read_text())


def offline_build(previous=None):
    return media.build(SEED, GPUS, previous or {}, offline=True, today='2026-10-09')


class VerdictTests(unittest.TestCase):
    need = {'min': 8, 'recommended': 12}

    def test_thresholds(self):
        self.assertEqual(media.verdict(12, self.need), 'fits')
        self.assertEqual(media.verdict(8, self.need), 'tight')
        self.assertEqual(media.verdict(5.6, self.need), 'offload')   # 0.7 × min
        self.assertEqual(media.verdict(5.5, self.need), 'no')


class RegistryTests(unittest.TestCase):
    def test_document_validates_against_its_schema(self):
        jsonschema.validate(offline_build(), SCHEMA)

    def test_published_copy_validates(self):
        jsonschema.validate(json.loads((ROOT / 'data/media_registry.json').read_text()), SCHEMA)

    def test_every_category_is_ordered_by_likes_and_only_lists_installable_models(self):
        reg = offline_build()
        for cid, cat in reg['categories'].items():
            rows = [(reg['models'][t['id']]['hf_stats']['likes'], t['score'], t['id']) for t in cat['top']]
            self.assertEqual(rows, sorted(rows, key=lambda r: (-r[0], -r[1], r[2])), cid)
            for t in cat['top']:
                model = reg['models'][t['id']]
                self.assertNotEqual(model['fits'][cat['reference_gpu']]['verdict'], 'no')
                self.assertTrue(model.get('homepilot'), cid)

    def test_more_likes_ranks_ahead_of_a_model_people_have_not_voted_for(self):
        # SDXL fits on 12 GB and HomePilot can install it. A huge download count must not
        # outrank a model people actually liked.
        previous = {
            'sdxl-base-1.0': {'hf_stats': {'downloads_30d': 10, 'likes': 8000, 'fetched_at': '2026-10-01'}},
            'sd15': {'hf_stats': {'downloads_30d': 10_000_000, 'likes': 5, 'fetched_at': '2026-10-01'}},
        }
        top = offline_build(previous)['categories']['image-12gb']['top']
        self.assertEqual(top[0]['id'], 'sdxl-base-1.0')
        self.assertGreater(top[0]['likes'], top[1]['likes'])

    def test_a_client_reproduces_the_score_from_components_and_weights(self):
        # The contract HomePilot relies on: score = Σ weight · component, with fit from its own VRAM.
        reg = offline_build()
        w, fit_score = reg['scoring']['weights'], reg['scoring']['fit_score']
        for m in reg['models'].values():
            for gid, s in m['scores'].items():
                c = m['components']
                expected = (w['fit'] * fit_score[m['fits'][gid]['verdict']] + w['capability'] * c['capability']
                            + w['momentum'] * c['momentum'] + w['speed'] * c['speed'])
                self.assertAlmostEqual(s, expected, places=3)

    def test_offline_keeps_the_previous_stats(self):
        previous = {'sdxl-base-1.0': {'hf_stats': {'downloads_30d': 1234, 'likes': 5, 'fetched_at': '2026-10-01'}}}
        reg = offline_build(previous)
        self.assertEqual(reg['models']['sdxl-base-1.0']['hf_stats']['downloads_30d'], 1234)
        self.assertEqual(reg['models']['sd15']['hf_stats']['downloads_30d'], 0)

    def test_more_vram_never_ranks_a_model_lower(self):
        reg = offline_build()
        for m in reg['models'].values():
            s = m['scores']
            self.assertLessEqual(s['rtx3060-12'], s['rtx4060ti-16'])
            self.assertLessEqual(s['rtx4060ti-16'], s['rtx4090-24'])

    def test_homepilot_install_ids_are_unique(self):
        ids = [m['homepilot']['model_id'] for m in SEED['models'] if m.get('homepilot')]
        self.assertEqual(len(ids), len(set(ids)))


if __name__ == '__main__':
    unittest.main()
