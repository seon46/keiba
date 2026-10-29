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

from keiba_ai.experiment_v3c import GOING_FEATURES, WEIGHT_FEATURES, _race_dropout, _race_z


def announced_csv(path, *, years, seed=5):
    """Flat races where today's bodyweight change carries real information."""
    rng = random.Random(seed)
    fields = ['race_id', 'region', 'page', 'race_date', 'horse_id', 'horse_number', 'frame', 'sex', 'age',
              'jockey_id', 'trainer_id', 'carried_weight_kg', 'finish_position', 'finish_status',
              'race_course', 'race_surface', 'race_distance_m', 'race_declared_field_size', 'snapshot_id',
              'race_id_verified', 'time_seconds', 'last_3f_seconds', 'weight_kg', 'weight_change_kg',
              'race_track_condition', 'race_weather']
    pools = {region: {f'201{8 if region == "central" else 9}000{index:03d}': rng.gauss(0, .5) for index in range(40)}
             for region in ('central', 'local')}
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for year in years:
            for number in range(36):
                day = date(year, 1, 3) + timedelta(days=10 * number)
                going = rng.choice(['良', '良', '稍重', '重', '不良'])
                for region, course, surface in (('central', '東京', '芝'), ('local', '大井', 'ダ')):
                    strengths = pools[region]
                    field = rng.sample(sorted(strengths), 10)
                    changes = {horse: rng.choice([-12, -6, 0, 2, 4, 8]) for horse in field}
                    # A large loss of bodyweight today hurts performance.
                    noisy = sorted(field, key=lambda horse: -(strengths[horse] + (changes[horse] <= -6) * -1.5
                                                              + rng.gauss(0, .5)))
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
                            last_3f_seconds=33 + position / 10, weight_kg=470 + changes[horse],
                            weight_change_kg=changes[horse], race_track_condition=going, race_weather='晴'))


class HelperTests(unittest.TestCase):
    def test_race_z_is_within_race_and_keeps_missing(self):
        values = np.asarray([1., 2., 3., np.nan, 10., 10.])
        races = np.asarray(['a', 'a', 'a', 'a', 'b', 'b'])
        result = _race_z(values, races)
        np.testing.assert_allclose(result[:3], [-1.2247449, 0, 1.2247449], atol=1e-6)
        self.assertTrue(np.isnan(result[3]))
        np.testing.assert_allclose(result[4:], [0, 0])

    def test_dropout_blanks_whole_races_deterministically(self):
        races = np.repeat(np.asarray([f'r{index}' for index in range(2000)]), 3)
        first, second = _race_dropout(races, .1, 42), _race_dropout(races, .1, 42)
        np.testing.assert_array_equal(first, second)
        self.assertTrue((first.reshape(-1, 3).min(1) == first.reshape(-1, 3).max(1)).all())
        self.assertAlmostEqual(first.mean(), .1, delta=.03)

    def test_feature_lists_are_distinct(self):
        self.assertFalse(set(WEIGHT_FEATURES) & set(GOING_FEATURES))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy')),
                     'Install requirements-ml.txt for the v3c experiment tests')
class ExperimentCTests(unittest.TestCase):
    def test_bodyweight_signal_is_found_without_reading_the_outcome(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3c import main
        with TemporaryDirectory() as name:
            root = Path(name)
            announced_csv(root / 'source.csv', years=(2022, 2023, 2024, 2025, 2026))
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(database), '--output-dir', str(root / 'experiment'),
                             '--validation-years', '2025', '--threads', '1', '--rounds', '300', '--patience', '30',
                             '--evaluation-samples', '128'])
            self.assertEqual(code, 0, output.getvalue())
            report = json.loads((root / 'experiment' / 'experiment.json').read_text(encoding='utf-8'))
            self.assertFalse(report['test_period_used'])
            for group in ('central_flat', 'local_flat'):
                summary = report['groups'][group]['summary']
                self.assertEqual(set(summary), {'base', 'weight', 'going', 'both'})
                self.assertGreater(report['groups'][group]['coverage']['current_bodyweight_kg'], .99)
                self.assertLess(summary['weight']['race_outcome_log_loss'], summary['base']['race_outcome_log_loss'])
                fold = report['groups'][group]['folds'][0]['variants']['weight']
                self.assertIn('current_bodyweight_change_kg', fold['added_feature_gain_rank'])


if __name__ == '__main__':
    unittest.main()
