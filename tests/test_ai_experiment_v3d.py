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

from keiba_ai.experiment_v3d import jockey_timeline


def _simulate(races=1500, size=10, jockey_count=20, seed=0):
    rng = np.random.default_rng(seed)
    skill = rng.normal(0, .5, jockey_count)
    days, race_ids, jockeys, horses, positions, strength = [], [], [], [], [], []
    for race in range(races):
        day = (date(2022, 1, 1) + timedelta(days=race // 5)).isoformat()
        horse_strength = rng.normal(0, 1, size)
        riders = rng.choice(jockey_count, size, replace=False)
        utility = horse_strength + skill[riders] + rng.gumbel(size=size)
        order = np.argsort(-utility)
        rank = np.empty(size, dtype=int)
        rank[order] = np.arange(1, size + 1)
        for index in range(size):
            days.append(day)
            race_ids.append(f'r{race}')
            jockeys.append(f'j{riders[index]}')
            horses.append(f'h{race}-{index}')
            positions.append(rank[index])
            strength.append(horse_strength[index])
    return (np.asarray(days), np.asarray(race_ids), np.asarray(jockeys, dtype=object), np.asarray(horses, dtype=object),
            np.ones(len(days), dtype=bool), np.asarray(positions), np.asarray(strength), skill)


class JockeyTimelineTests(unittest.TestCase):
    def test_recovers_planted_skill_with_strength_offset(self):
        days, races, jockeys, horses, finished, positions, strength, skill = _simulate()
        result = jockey_timeline(days, races, jockeys, horses, finished, positions, strength,
                                 half_life=10_000, short_half_life=90, prior_sd=.5)
        last = {}
        for jockey, theta in zip(jockeys[-2000:], result['theta'][-2000:]):
            last[jockey] = theta
        estimated = np.asarray([last[f'j{index}'] for index in range(len(skill))])
        self.assertGreater(np.corrcoef(estimated, skill)[0, 1], .9)
        self.assertLess(np.abs((estimated - estimated.mean()) - (skill - skill.mean())).mean(), .12)

    def test_state_is_strictly_before_the_day(self):
        days, races, jockeys, horses, finished, positions, strength, _ = _simulate(races=200)
        first = jockey_timeline(days, races, jockeys, horses, finished, positions, strength)
        # Reverse the results of the last day only: that day's own features must not change.
        last_day = days == days[-1]
        changed = positions.copy()
        for race in np.unique(races[last_day]):
            rows = races == race
            changed[rows] = positions[rows].max() + 1 - positions[rows]
        second = jockey_timeline(days, races, jockeys, horses, finished, changed, strength)
        for key in ('theta', 'theta_recent', 'theta_sd'):
            np.testing.assert_array_equal(first[key], second[key])
        # Earlier days do influence later states.
        self.assertTrue(np.isfinite(first['theta'][last_day]).all())
        self.assertGreater(np.abs(first['theta'][last_day]).max(), 0)

    def test_new_jockey_starts_at_prior(self):
        days, races, jockeys, horses, finished, positions, strength, _ = _simulate(races=10)
        result = jockey_timeline(days, races, jockeys, horses, finished, positions, strength, prior_sd=.3)
        first_day = days == days[0]
        np.testing.assert_allclose(result['theta'][first_day], 0)
        np.testing.assert_allclose(result['theta_sd'][first_day], .3)
        np.testing.assert_allclose(result['rides'][first_day], 0)


def jockey_csv(path, *, years, seed=3):
    rng = random.Random(seed)
    fields = ['race_id', 'region', 'page', 'race_date', 'horse_id', 'horse_number', 'frame', 'sex', 'age',
              'jockey_id', 'trainer_id', 'carried_weight_kg', 'finish_position', 'finish_status',
              'race_course', 'race_surface', 'race_distance_m', 'race_declared_field_size', 'snapshot_id',
              'race_id_verified', 'time_seconds', 'last_3f_seconds', 'weight_kg', 'weight_change_kg',
              'race_track_condition', 'race_weather']
    pools = {region: {f'201{8 if region == "central" else 9}000{index:03d}': rng.gauss(0, .4) for index in range(40)}
             for region in ('central', 'local')}
    skills = {f'{index:05d}': rng.gauss(0, .8) for index in range(1, 16)}
    with path.open('w', encoding='utf-8', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for year in years:
            for number in range(36):
                day = date(year, 1, 3) + timedelta(days=10 * number)
                for region, course, surface in (('central', '東京', '芝'), ('local', '大井', 'ダ')):
                    strengths = pools[region]
                    field = rng.sample(sorted(strengths), 10)
                    riders = dict(zip(field, rng.sample(sorted(skills), 10)))
                    noisy = sorted(field, key=lambda horse: -(strengths[horse] + skills[riders[horse]] + rng.gauss(0, .5)))
                    race_id = (f'{year}05{number // 12 + 1:02d}{number % 12 + 1:02d}01' if region == 'central'
                               else f'{year}44{day.month:02d}{day.day:02d}01')
                    for horse_number, horse in enumerate(field, 1):
                        position = noisy.index(horse) + 1
                        writer.writerow(dict(
                            race_id=race_id, region=region, page='results', race_date=day.isoformat(),
                            horse_id=horse, horse_number=horse_number, frame=(horse_number + 1) // 2, sex='牡',
                            age=4, jockey_id=riders[horse], trainer_id=f'{int(horse[-2:]) % 7:05d}',
                            carried_weight_kg=57, finish_position=position, finish_status='finished',
                            race_course=course, race_surface=surface, race_distance_m=1600,
                            race_declared_field_size=10, snapshot_id=f'snapshot-{region}-{year}-{number}',
                            race_id_verified='True', time_seconds=95 + position * .2,
                            last_3f_seconds=33 + position / 10, weight_kg=470, weight_change_kg=0,
                            race_track_condition='良', race_weather='晴'))


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy')),
                     'Install requirements-ml.txt for the v3d experiment tests')
class ExperimentDTests(unittest.TestCase):
    def test_jockey_variants_run_and_theta_helps_when_skill_is_planted(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3d import main
        with TemporaryDirectory() as name:
            root = Path(name)
            jockey_csv(root / 'source.csv', years=(2021, 2022, 2023, 2024, 2025, 2026))
            with redirect_stdout(io.StringIO()):
                database = prepare(root / 'source.csv', root / 'prepared')
            output = io.StringIO()
            with redirect_stdout(output):
                code = main(['--dataset', str(database), '--output-dir', str(root / 'experiment'),
                             '--validation-years', '2025', '--threads', '1', '--rounds', '300', '--patience', '30',
                             '--evaluation-samples', '128', '--prior-sd', '.6'])
            self.assertEqual(code, 0, output.getvalue())
            report = json.loads((root / 'experiment' / 'experiment.json').read_text(encoding='utf-8'))
            self.assertFalse(report['test_period_used'])
            for group in ('central_flat', 'local_flat'):
                summary = report['groups'][group]['summary']
                self.assertEqual(set(summary), {'base', 'nojockey', 'theta_feature', 'theta_offset'})
                self.assertLess(summary['theta_offset']['race_outcome_log_loss'],
                                summary['nojockey']['race_outcome_log_loss'])
                fold = report['groups'][group]['folds'][0]
                self.assertGreater(fold['variants']['theta_offset']['gamma'], .3)
                self.assertGreater(fold['theta_summary']['theta_sd_across_validation_rows'], .1)


if __name__ == '__main__':
    unittest.main()
