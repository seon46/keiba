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

from keiba_ai.bias_v3 import BIAS_V3_FEATURES, _race_beta, day_features


class RaceBetaTests(unittest.TestCase):
    def test_average_race_beta_recovers_planted_bias(self):
        rng = np.random.default_rng(0)
        true = np.asarray([.8, -.4])
        estimates = []
        for _ in range(400):
            size = 12
            strength = rng.normal(size=size)
            u = np.column_stack([.5 - rng.random(size), .5 - rng.random(size)])
            utility = strength + u @ true + rng.gumbel(size=size)
            order = np.argsort(-utility)
            beta, variance = _race_beta(strength[order], u[order])
            self.assertTrue(np.all(variance > 0))
            estimates.append(beta)
        mean = np.mean(estimates, axis=0)
        np.testing.assert_allclose(mean, true, atol=.15)


class DayFeatureTests(unittest.TestCase):
    def _day(self):
        days = np.asarray(['2024-01-01'] * 9)
        races = np.asarray(['r1'] * 3 + ['r2'] * 3 + ['r3'] * 3)
        numbers = np.asarray([1.] * 3 + [2.] * 3 + [3.] * 3)
        course = np.asarray(['東京'] * 9, dtype=object)
        surface = np.asarray(['芝'] * 9, dtype=object)
        distance = np.full(9, 1600.)
        expected = np.asarray([0., .5, 1.] * 3)
        gate = np.asarray([0., .5, 1.] * 3)
        return days, races, numbers, course, surface, distance, expected, gate

    def test_input_side_uses_only_earlier_races_and_label_side_excludes_itself(self):
        args = self._day()
        betas = {'r1': (np.asarray([1., 0.]), np.asarray([.1, .1])),
                 'r2': (np.asarray([1., 0.]), np.asarray([.1, .1]))}
        features, label = day_features(*args, betas)
        front = BIAS_V3_FEATURES.index('bias3_day_front_dev')
        races_used = BIAS_V3_FEATURES.index('bias3_day_races')
        self.assertEqual(features[0, races_used], 0)
        self.assertEqual(features[0, front], 0)
        self.assertEqual(features[3, races_used], 1)
        self.assertGreater(features[3, front], 0)
        self.assertEqual(features[6, races_used], 2)
        # Changing r3's own beta changes neither its input features nor its label deviation.
        changed = dict(betas, r3=(np.asarray([-5., 0.]), np.asarray([.1, .1])))
        features2, label2 = day_features(*args, changed)
        np.testing.assert_array_equal(features[6:], features2[6:])
        np.testing.assert_array_equal(label[6:], label2[6:])
        # r1's label deviation uses the later race r2.
        self.assertGreater(label[0, 0], 0)

    def test_prior_uses_earlier_days_only(self):
        days, races, numbers, course, surface, distance, expected, gate = self._day()
        betas = {'r1': (np.asarray([2., 0.]), np.asarray([.1, .1])),
                 'r2': (np.asarray([2., 0.]), np.asarray([.1, .1]))}
        features, _ = day_features(days, races, numbers, course, surface, distance, expected, gate, betas)
        prior = BIAS_V3_FEATURES.index('bias3_prior_front')
        np.testing.assert_allclose(features[:, prior], 0)


def bias_csv(path, *, years, seed=9):
    """Flat races where on half of the days front runners get a planted advantage."""
    rng = random.Random(seed)
    fields = ['race_id', 'region', 'page', 'race_date', 'horse_id', 'horse_number', 'frame', 'sex', 'age',
              'jockey_id', 'trainer_id', 'carried_weight_kg', 'finish_position', 'finish_status',
              'race_course', 'race_surface', 'race_distance_m', 'race_declared_field_size', 'snapshot_id',
              'race_id_verified', 'time_seconds', 'last_3f_seconds', 'weight_kg', 'weight_change_kg',
              'race_track_condition', 'race_weather', 'corner_order', 'corner_order_corners']
    pools = {region: {f'201{8 if region == "central" else 9}000{index:03d}': rng.gauss(0, .4) for index in range(40)}
             for region in ('central', 'local')}
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for year in years:
            for number in range(36):
                day = date(year, 1, 3) + timedelta(days=10 * number)
                front_day = number % 2 == 0
                for region, course, surface in (('central', '東京', '芝'), ('local', '大井', 'ダ')):
                    for race_number in (1, 2, 3):
                        strengths = pools[region]
                        field = rng.sample(sorted(strengths), 10)
                        corners = dict(zip(field, rng.sample(range(1, 11), 10)))
                        def utility(horse):
                            bias = 1.5 * (.5 - (corners[horse] - 1) / 9) if front_day else 0
                            return strengths[horse] + bias + rng.gauss(0, .5)
                        noisy = sorted(field, key=lambda horse: -utility(horse))
                        race_id = (f'{year}05{number // 12 + 1:02d}{number % 12 + 1:02d}{race_number:02d}' if region == 'central'
                                   else f'{year}44{day.month:02d}{day.day:02d}{race_number:02d}')
                        for horse_number, horse in enumerate(field, 1):
                            position = noisy.index(horse) + 1
                            writer.writerow(dict(
                                race_id=race_id, region=region, page='results', race_date=day.isoformat(),
                                horse_id=horse, horse_number=horse_number, frame=(horse_number + 1) // 2, sex='牡',
                                age=4, jockey_id=f'{horse_number:05d}', trainer_id=f'{int(horse[-2:]) % 7:05d}',
                                carried_weight_kg=57, finish_position=position, finish_status='finished',
                                race_course=course, race_surface=surface, race_distance_m=1600,
                                race_declared_field_size=10,
                                snapshot_id=f'snapshot-{region}-{year}-{number}-{race_number}', race_id_verified='True',
                                time_seconds=95 + position * .2, last_3f_seconds=33 + position / 10, weight_kg=470,
                                weight_change_kg=0, race_track_condition='良', race_weather='晴',
                                corner_order=f'{corners[horse]}-{corners[horse]}', corner_order_corners='[3, 4]'))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy')),
                     'Install requirements-ml.txt for the v3i experiment tests')
class ExperimentITests(unittest.TestCase):
    def test_experiment_i_measures_planted_front_bias(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3i import main
        with TemporaryDirectory() as name:
            root = Path(name)
            bias_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025, 2026))
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
            for group in ('central_flat', 'local_flat'):
                result = report['groups'][group]
                self.assertGreater(result['geometry_coverage']['corner'], .9)
                fold = result['folds'][0]
                self.assertGreater(fold['races_with_beta'], 50)
                self.assertGreater(fold['beta_mean'][0], .2)
                self.assertGreater(fold['beta_sd'][0], .3)
                self.assertEqual(len(result['summary']), 6)
                self.assertIsNotNone(fold['configs']['bias_offset']['gammas'])


if __name__ == '__main__':
    unittest.main()
