from contextlib import redirect_stdout
import csv
from datetime import date, timedelta
import importlib.util
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from keiba_ai.experiment_v3 import ODDS_FEATURES, PLStructure, experiment_feature_names


def _total_loss(structure, scores):
    return float(structure.terms(scores, need_derivatives=False).sum())


class PLObjectiveTests(unittest.TestCase):
    def _structure(self, top_k=None, seed=0):
        rng = np.random.default_rng(seed)
        keys, positions = [], []
        for race, size in enumerate((5, 2, 8, 3)):
            order = rng.permutation(size) + 1
            keys.extend([f'r{race}'] * size)
            positions.extend(order.tolist())
        return PLStructure(np.asarray(keys), np.asarray(positions), top_k=top_k), len(keys), rng

    def test_gradient_matches_finite_differences(self):
        for top_k in (None, 3):
            structure, count, rng = self._structure(top_k=top_k)
            scores = rng.normal(size=count)
            _, grad, hess = structure.terms(scores)
            step = 1e-6
            for index in range(count):
                plus, minus = scores.copy(), scores.copy()
                plus[index] += step
                minus[index] -= step
                numeric = (_total_loss(structure, plus) - _total_loss(structure, minus)) / (2 * step)
                self.assertAlmostEqual(grad[index], numeric, places=5)
                wide = 1e-4
                plus[index] += wide - step
                minus[index] -= wide - step
                second = (_total_loss(structure, plus) - 2 * _total_loss(structure, scores)
                          + _total_loss(structure, minus)) / wide ** 2
                # The diagonal Hessian is exact for each horse's own score.
                self.assertAlmostEqual(hess[index], max(second, 1e-6), places=2)

    def test_loss_is_shift_invariant_and_matches_closed_form(self):
        structure = PLStructure(np.asarray(['a', 'a', 'a']), np.asarray([2, 1, 3]))
        scores = np.asarray([0.3, 1.2, -0.5])
        # Order by position: horse1 (1.2), horse0 (0.3), horse2 (-0.5); stages 1 and 2 of 2.
        first = 1.2 - np.log(np.exp(1.2) + np.exp(.3) + np.exp(-.5))
        second = .3 - np.log(np.exp(.3) + np.exp(-.5))
        expected = -(first + second) / 2
        self.assertAlmostEqual(structure.mean_loss(scores), expected, places=10)
        self.assertAlmostEqual(structure.mean_loss(scores + 100), expected, places=8)

    def test_rejects_ties_and_noncontiguous_races(self):
        with self.assertRaises(ValueError):
            PLStructure(np.asarray(['a', 'a']), np.asarray([1, 1]))
        with self.assertRaises(ValueError):
            PLStructure(np.asarray(['a', 'b', 'a']), np.asarray([1, 1, 2]))

    def test_feature_set_excludes_odds_and_bias(self):
        names = experiment_feature_names()
        self.assertEqual(len(names), 89)
        self.assertFalse(set(names) & set(ODDS_FEATURES))
        self.assertFalse(any(name.startswith('day_') or 'bias' in name for name in names))


def synthetic_csv(path, *, years=(2022, 2023, 2024, 2025), seed=7):
    """Races spread over every month, with a latent strength per horse."""
    rng = random.Random(seed)
    fields = ['race_id', 'region', 'page', 'race_date', 'horse_id', 'horse_number', 'frame', 'sex', 'age',
              'jockey_id', 'trainer_id', 'carried_weight_kg', 'finish_position', 'finish_status',
              'race_course', 'race_surface', 'race_distance_m', 'race_declared_field_size', 'snapshot_id',
              'race_id_verified', 'time_seconds', 'last_3f_seconds']
    strengths = {f'20180000{index:02d}': rng.gauss(0, 1) for index in range(1, 41)}
    horses = sorted(strengths)
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for year in years:
            for number in range(36):
                day = date(year, 1, 3) + timedelta(days=10 * number)
                field = rng.sample(horses, 10)
                noisy = sorted(field, key=lambda horse: -(strengths[horse] + rng.gauss(0, .8)))
                for horse_number, horse in enumerate(field, 1):
                    position = noisy.index(horse) + 1
                    writer.writerow(dict(
                        race_id=f'{year}05{number // 12 + 1:02d}{number % 12 + 1:02d}{1:02d}', region='central',
                        page='results', race_date=day.isoformat(), horse_id=horse, horse_number=horse_number,
                        frame=(horse_number + 1) // 2, sex='牡', age=4, jockey_id=f'{horse_number:05d}',
                        trainer_id=f'{int(horse[-2:]) % 7:05d}', carried_weight_kg=57,
                        finish_position=position, finish_status='finished', race_course='東京',
                        race_surface='芝', race_distance_m=1600, race_declared_field_size=10,
                        snapshot_id=f'snapshot-{year}-{number}', race_id_verified='True',
                        time_seconds=95 + position * .2, last_3f_seconds=33 + position / 10))


@unittest.skipUnless(importlib.util.find_spec('lightgbm') and importlib.util.find_spec('sklearn'),
                     'Install requirements-ml.txt for the v3 experiment tests')
class ExperimentTests(unittest.TestCase):
    def test_experiment_runs_m0_and_m1_without_touching_test_period(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3 import main
        with TemporaryDirectory() as name:
            root = Path(name)
            synthetic_csv(root / 'source.csv', years=(2022, 2023, 2024, 2025, 2026))
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(database), '--output-dir', str(root / 'experiment'),
                             '--groups', 'central_flat', '--validation-years', '2025',
                             '--train-start', '2022-04-01', '--threads', '1', '--m0-iterations', '30',
                             '--rounds', '200', '--patience', '20', '--validation-samples', '32',
                             '--evaluation-samples', '64', '--stage-weights', 'all', 'top3'])
            self.assertEqual(code, 0, output.getvalue())
            report = json.loads((root / 'experiment' / 'experiment.json').read_text(encoding='utf-8'))
            self.assertFalse(report['test_period_used'])
            group = report['groups']['central_flat']
            fold = group['folds'][0]
            self.assertEqual(fold['state'], 'trained')
            self.assertEqual(fold['validation_start'], '2025-01-01')
            self.assertEqual(fold['stop_start'], '2024-11-01')
            self.assertIn('all', fold['m1']['variants'])
            self.assertIn('top3', fold['m1']['variants'])
            self.assertLess(group['summary']['m1']['all'], group['summary']['uniform'])
            for value in (group['summary']['m0'], *group['summary']['m1'].values()):
                self.assertTrue(np.isfinite(value))

    def test_refuses_validation_in_test_period(self):
        from keiba_ai.experiment_v3 import main
        with TemporaryDirectory() as name:
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(Path(name) / 'missing.sqlite3'), '--output-dir', str(Path(name) / 'out'),
                             '--validation-years', '2026'])
            self.assertEqual(code, 2)
            self.assertIn('最終評価期間', output.getvalue())


if __name__ == '__main__':
    unittest.main()
