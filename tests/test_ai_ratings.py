from contextlib import redirect_stdout
from datetime import date, timedelta
import importlib.util
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from keiba_ai.ratings import RATING_FEATURES, rating_block


def _races(days=400, size=10, horses=80, seed=0):
    rng = np.random.default_rng(seed)
    ability = rng.normal(0, 1, horses)
    rows = dict(days=[], race_ids=[], horses=[], positions=[])
    for number in range(days):
        day = (date(2022, 1, 1) + timedelta(days=number)).isoformat()
        for race in range(2):
            field = rng.choice(horses, size, replace=False)
            utility = ability[field] + rng.gumbel(size=size)
            rank = np.empty(size, dtype=int)
            rank[np.argsort(-utility)] = np.arange(1, size + 1)
            for horse, position in zip(field, rank):
                rows['days'].append(day)
                rows['race_ids'].append(f'{day}-{race}')
                rows['horses'].append(f'h{horse}')
                rows['positions'].append(position)
    arrays = {key: np.asarray(value, dtype=object if key == 'horses' else None) for key, value in rows.items()}
    return arrays, ability


class RatingTests(unittest.TestCase):
    def test_ratings_recover_ability(self):
        rows, ability = _races()
        block = rating_block(rows['days'], rows['race_ids'], rows['horses'], np.ones(len(rows['days']), bool),
                             rows['positions'], half_life=10_000)
        last = {}
        for horse, value in zip(rows['horses'], block[:, 0]):
            last[horse] = value
        estimated = np.asarray([last[f'h{index}'] for index in range(len(ability))])
        self.assertGreater(np.corrcoef(estimated, ability)[0, 1], .9)

    def test_surprise_and_field_use_earlier_days_only(self):
        rows, _ = _races(days=60)
        finished = np.ones(len(rows['days']), bool)
        first = rating_block(rows['days'], rows['race_ids'], rows['horses'], finished, rows['positions'])
        last_day = rows['days'] == rows['days'][-1]
        changed = rows['positions'].copy()
        for race in np.unique(rows['race_ids'][last_day]):
            member = rows['race_ids'] == race
            changed[member] = changed[member].max() + 1 - changed[member]
        second = rating_block(rows['days'], rows['race_ids'], rows['horses'], finished, changed)
        np.testing.assert_array_equal(first[last_day], second[last_day])

    def test_surprises_are_centred_when_ratings_are_accurate(self):
        rows, ability = _races(days=300)
        block = rating_block(rows['days'], rows['race_ids'], rows['horses'], np.ones(len(rows['days']), bool),
                             rows['positions'], half_life=10_000)
        late = np.arange(len(block)) > len(block) * .7
        surprise = block[late, RATING_FEATURES.index('horse_recent5_surprise_mean')]
        self.assertTrue(np.isfinite(surprise).mean() > .9)
        # With accurate ratings, surprises are centred near zero.
        self.assertLess(abs(np.nanmean(surprise)), .05)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy')),
                     'Install requirements-ml.txt for the v3g experiment tests')
class ExperimentGTests(unittest.TestCase):
    def test_experiment_g_runs(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3g import main
        from test_ai_experiment_v3d import jockey_csv
        with TemporaryDirectory() as name:
            root = Path(name)
            jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025, 2026))
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(database), '--output-dir', str(root / 'experiment'),
                             '--validation-years', '2025', '--threads', '1', '--rounds', '200', '--patience', '20',
                             '--evaluation-samples', '64'])
            self.assertEqual(code, 0, output.getvalue())
            report = json.loads((root / 'experiment' / 'experiment.json').read_text(encoding='utf-8'))
            self.assertFalse(report['test_period_used'])
            self.assertEqual(set(report['groups']['central_flat']['summary']), {'speed', 'speed_rating'})
            self.assertEqual(set(report['groups']['local_flat']['summary']), {'speed_theta', 'speed_theta_rating'})
            self.assertIn('vs_base', report['groups']['local_flat']['summary']['speed_theta_rating'])
            self.assertGreater(report['groups']['central_flat']['rating_coverage']['horse_rating'], .99)


if __name__ == '__main__':
    unittest.main()
