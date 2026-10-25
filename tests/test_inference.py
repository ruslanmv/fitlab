"""Inference lane — hosted models HomePilot can run, ranked by what people like."""
import datetime as dt
import importlib.util
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fitlab import inference  # noqa: E402

spec = importlib.util.spec_from_file_location('sync_inference', ROOT / 'scripts/sync_inference.py')
sync = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sync)

NOW = dt.datetime(2026, 10, 25, tzinfo=dt.timezone.utc)


def hub(repo, likes=100, downloads=1000, trending=1, *, task='text-to-image', status='live',
        created='2025-01-01T00:00:00.000Z', modified=None, tags=(), library='diffusers'):
    """A Hub API row as `list_models(expand=[...])` returns it."""
    return {
        'id': repo, 'likes': likes, 'downloads': downloads, 'trendingScore': trending,
        'createdAt': created, 'lastModified': modified or created, 'tags': list(tags),
        'library_name': library,
        'inferenceProviderMapping': [{'provider': 'fal-ai', 'status': status, 'task': task}],
    }


class Compatibility(unittest.TestCase):
    def test_a_live_provider_for_the_task_is_required(self):
        self.assertTrue(inference.compatible(hub('a/live'), 'image'))
        self.assertFalse(inference.compatible(hub('a/down', status='error'), 'image'))
        self.assertFalse(inference.compatible(hub('a/other-task', task='text-to-video'), 'image'))
        self.assertTrue(inference.compatible(hub('a/chat', task='conversational', library='transformers'), 'chat'))

    def test_adapters_and_gguf_are_not_runnable_models(self):
        self.assertFalse(inference.compatible(hub('a/lora', tags=['lora']), 'image'))
        self.assertFalse(inference.compatible(hub('a/peft', library='peft'), 'image'))
        self.assertFalse(inference.compatible(hub('a/q', tags=['gguf']), 'image'))
        self.assertFalse(inference.compatible(hub('a/ad', tags=['base_model:adapter:org/base']), 'image'))

    def test_older_dict_shaped_mapping_is_read_too(self):
        m = hub('a/x')
        m['inferenceProviderMapping'] = {'replicate': {'status': 'live', 'task': 'text-to-image'}}
        self.assertEqual(inference.live_providers(m, 'image'), ['replicate'])


class Providers(unittest.TestCase):
    def test_spellings_of_one_publisher_are_one(self):
        self.assertEqual(inference.normalize_provider('Meta-Llama/Llama-3.1-8B'), 'meta-llama')
        self.assertEqual(inference.normalize_provider('meta-llama/x'), 'meta-llama')
        self.assertEqual(inference.normalize_provider('facebook/x'), 'meta-llama')
        self.assertEqual(inference.provider_label('Qwen/Qwen3-8B'), 'Qwen')
        self.assertEqual(inference.provider_label('someone/model'), 'someone')


class Ranking(unittest.TestCase):
    MODELS = [
        hub('org/loved', likes=15000, downloads=500_000, trending=40, modified='2026-08-01T00:00:00Z'),
        hub('org/used', likes=200, downloads=9_000_000, trending=5),
        hub('org/hot', likes=150, downloads=2000, trending=280, created='2026-09-20T00:00:00Z'),
        hub('org/old-favourite', likes=16000, downloads=100_000, trending=1, created='2022-01-01T00:00:00Z'),
        hub('org/broken', likes=99999, status='error'),
    ]

    def ids(self, order, **kw):
        return [r['id'] for r in inference.rank(self.MODELS, 'image', order, now=NOW, **kw)]

    def test_most_liked_order_is_by_likes(self):
        self.assertEqual(self.ids('likes'), ['org/old-favourite', 'org/loved', 'org/used', 'org/hot'])

    def test_popular_weighs_likes_with_freshness(self):
        # Nearly the same likes, but one is maintained and trending: it leads the composite.
        self.assertEqual(self.ids('popular')[0], 'org/loved')

    def test_trending_and_rising(self):
        self.assertEqual(self.ids('trending')[0], 'org/hot')
        self.assertEqual(self.ids('rising'), ['org/hot'])          # only recent models

    def test_newest(self):
        self.assertEqual(self.ids('newest')[0], 'org/hot')

    def test_a_model_no_provider_runs_is_never_suggested(self):
        for order in inference.ORDERS:
            self.assertNotIn('org/broken', self.ids(order))

    def test_swapping_likes_swaps_the_order(self):
        a, b = hub('x/a', likes=900), hub('x/b', likes=100)
        first = [r['id'] for r in inference.rank([a, b], 'image', 'likes', now=NOW)]
        a['likes'], b['likes'] = 100, 900
        second = [r['id'] for r in inference.rank([a, b], 'image', 'likes', now=NOW)]
        self.assertEqual(first, ['x/a', 'x/b'])
        self.assertEqual(second, ['x/b', 'x/a'])

    def test_one_row_per_repo_and_raw_counts_kept(self):
        rows = inference.rank([hub('Org/Same', likes=5), hub('org/same', likes=50)], 'image', now=NOW)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['likes'], 50)
        self.assertEqual(set(rows[0]), {'id', 'provider', 'provider_label', 'likes', 'downloads',
                                        'trending_score', 'created_at', 'last_modified',
                                        'live_providers', 'score'})

    def test_limit_and_unknown_inputs(self):
        self.assertEqual(len(self.ids('likes', limit=2)), 2)
        with self.assertRaises(ValueError):
            inference.rank([], 'audio')
        with self.assertRaises(ValueError):
            inference.rank([], 'image', 'random')


class Diversity(unittest.TestCase):
    def test_share_of_the_biggest_publisher(self):
        rows = [{'provider': 'a'}, {'provider': 'a'}, {'provider': 'b'}, {'provider': 'c'}]
        self.assertEqual(inference.diversity(rows), {'publishers': 3, 'rows': 4, 'top_publisher_share': 0.5})
        self.assertEqual(inference.diversity([{'provider': 'a'}] * 3)['top_publisher_share'], 1.0)


class Publishing(unittest.TestCase):
    def test_a_run_with_nothing_compatible_keeps_the_previous_list(self):
        previous = {'kinds': {'video': {'lists': {'popular': [{'id': 'kept/one'}]}}}}
        doc = sync.build({'video': [hub('x/down', task='text-to-video', status='error')]}, NOW, 5, previous)
        self.assertEqual(doc['kinds']['video'], previous['kinds']['video'])

    def test_published_feed_shape(self):
        doc = json.loads((ROOT / 'data/inference_registry.json').read_text())
        self.assertTrue(doc['schema_version'].startswith('1'))
        self.assertEqual(doc['ranking_version'], inference.RANKING_VERSION)
        self.assertEqual(set(doc['kinds']), set(inference.KINDS))
        for kind, k in doc['kinds'].items():
            self.assertEqual(set(k['lists']), set(inference.ORDERS), kind)
            likes = [r['likes'] for r in k['lists']['likes']]
            self.assertEqual(likes, sorted(likes, reverse=True), kind)
            for row in k['lists']['popular']:
                self.assertTrue(row['live_providers'], (kind, row['id']))


if __name__ == '__main__':
    unittest.main()
