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

from keiba_ai.experiment_v3 import PLStructure
from keiba_ai.ranking import rank_distribution
from keiba_ai.ranking_v3 import (fit_temperatures, paired_bootstrap, staged_pl_loss,
                                 staged_rank_distribution)


class StagedTemperatureTests(unittest.TestCase):
    def test_single_temperature_matches_existing_distribution(self):
        scores = np.asarray([1.2, .4, 0., -.3, -1.1, .8])
        completion = np.asarray([.99, .97, 1., .95, .98, .99])
        expected = rank_distribution(scores, completion, temperature=.7, samples=40000, seed=3)
        actual = staged_rank_distribution(scores, completion, [.7], samples=40000, seed=5)
        self.assertLess(np.abs(expected - actual).max(), .02)
        np.testing.assert_allclose(actual.sum(axis=1), 1, atol=1e-10)
        np.testing.assert_allclose(actual[:, -1], 1 - completion, atol=1e-12)

    def test_staged_loss_with_equal_temperatures_matches_training_loss(self):
        rng = np.random.default_rng(1)
        keys = np.repeat(np.arange(30), 8)
        positions = np.concatenate([rng.permutation(8) + 1 for _ in range(30)])
        structure = PLStructure(keys, positions)
        scores = rng.normal(size=len(keys))
        self.assertAlmostEqual(staged_pl_loss(structure, scores, [1.0]), structure.mean_loss(scores), places=10)
        self.assertAlmostEqual(staged_pl_loss(structure, scores, [1.0, 1.0, 1.0]), structure.mean_loss(scores), places=10)
        self.assertAlmostEqual(staged_pl_loss(structure, scores, [2.0]), structure.mean_loss(scores / 2), places=10)

    def test_fit_recovers_stage_temperatures(self):
        rng = np.random.default_rng(2)
        races, size = 3000, 10
        true = np.asarray([.6, 1.0, 1.6])
        keys, positions, scores = [], [], []
        for race in range(races):
            strength = rng.normal(size=size)
            remaining = list(range(size))
            order = []
            for stage in range(size):
                temperature = true[min(stage, 2)]
                logits = strength[remaining] / temperature
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                order.append(remaining.pop(rng.choice(len(remaining), p=probabilities)))
            rank = np.empty(size, dtype=int)
            rank[order] = np.arange(1, size + 1)
            keys.extend([race] * size)
            positions.extend(rank.tolist())
            scores.extend(strength.tolist())
        structure = PLStructure(np.asarray(keys), np.asarray(positions))
        fitted, _ = fit_temperatures(structure, np.asarray(scores), 3)
        np.testing.assert_allclose(fitted, true, rtol=.15)

    def test_paired_bootstrap_sign(self):
        result = paired_bootstrap(np.full(200, -.01) + np.random.default_rng(0).normal(0, .001, 200))
        self.assertLess(result['upper95'], 0)
        self.assertGreater(result['better_share'], .9)


def flat_csv(path, *, years, seed=11):
    """Central (東京) and local (大井) races in every month with latent strengths."""
    rng = random.Random(seed)
    fields = ['race_id', 'region', 'page', 'race_date', 'horse_id', 'horse_number', 'frame', 'sex', 'age',
              'jockey_id', 'trainer_id', 'carried_weight_kg', 'finish_position', 'finish_status',
              'race_course', 'race_surface', 'race_distance_m', 'race_declared_field_size', 'snapshot_id',
              'race_id_verified', 'time_seconds', 'last_3f_seconds']
    pools = {region: {f'201{8 if region == "central" else 9}000{index:03d}': rng.gauss(0, 1) for index in range(40)}
             for region in ('central', 'local')}
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for year in years:
            for number in range(36):
                day = date(year, 1, 3) + timedelta(days=10 * number)
                for region, course, surface in (('central', '東京', '芝'), ('local', '大井', 'ダ')):
                    strengths = pools[region]
                    field = rng.sample(sorted(strengths), 10)
                    noisy = sorted(field, key=lambda horse: -(strengths[horse] + rng.gauss(0, .8)))
                    race_id = (f'{year}05{number // 12 + 1:02d}{number % 12 + 1:02d}01' if region == 'central'
                               else f'{year}44{day.month:02d}{day.day:02d}01')
                    for horse_number, horse in enumerate(field, 1):
                        position = noisy.index(horse) + 1
                        writer.writerow(dict(
                            race_id=race_id, region=region, page='results', race_date=day.isoformat(),
                            horse_id=horse, horse_number=horse_number, frame=(horse_number + 1) // 2, sex='牡',
                            age=4, jockey_id=f'{horse_number:05d}', trainer_id=f'{int(horse[-2:]) % 7:05d}',
                            carried_weight_kg=57, finish_position=position, finish_status='finished',
                            race_course=course, race_surface=surface, race_distance_m=1600,
                            race_declared_field_size=10, snapshot_id=f'snapshot-{region}-{year}-{number}',
                            race_id_verified='True', time_seconds=95 + position * .2,
                            last_3f_seconds=33 + position / 10))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'optuna', 'scipy')),
                     'Install requirements-ml.txt for the v3b experiment tests')
class ExperimentBTests(unittest.TestCase):
    def test_experiment_b_runs_tuning_and_reports_bootstrap(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3b import main
        with TemporaryDirectory() as name:
            root = Path(name)
            flat_csv(root / 'source.csv', years=(2022, 2023, 2024, 2025, 2026))
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(database), '--output-dir', str(root / 'experiment'),
                             '--validation-years', '2024', '2025', '--threads', '1', '--rounds', '60',
                             '--patience', '10', '--tuning-trials', '3', '--tuning-hours', '0.2',
                             '--evaluation-samples', '64'])
            self.assertEqual(code, 0, output.getvalue())
            report = json.loads((root / 'experiment' / 'experiment.json').read_text(encoding='utf-8'))
            self.assertFalse(report['test_period_used'])
            self.assertEqual(len(report['tuning']['trials']), 3)
            self.assertEqual(report['tuning']['trials'][0]['params']['structure'], 'separate')
            for group in ('central_flat', 'local_flat'):
                summary = report['groups'][group]['summary']
                self.assertIn('separate_default/single_t', summary)
                self.assertIn('pooled_default/staged_t', summary)
                comparison = summary['pooled_default/staged_t']['vs_baseline']['outcome']
                self.assertLessEqual(comparison['lower95'], comparison['upper95'])
                for fold in report['groups'][group]['folds']:
                    temperatures = fold['configs']['separate_default']['temperatures']
                    self.assertEqual(len(temperatures['staged']), 4)
            with (root / 'experiment' / 'race_losses.csv').open(encoding='utf-8') as file:
                rows = list(csv.DictReader(file))
            self.assertTrue(rows)
            self.assertTrue(all(float(row['outcome_log_loss']) > 0 for row in rows))


if __name__ == '__main__':
    unittest.main()
