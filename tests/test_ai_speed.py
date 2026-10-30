from contextlib import redirect_stdout
from datetime import date, timedelta
import importlib.util
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from keiba_ai.speed import SPEED_FEATURES, speed_feature_rows


def simulated_races(*, days=300, seed=1, variant_sd=1.0):
    """Two classes, one track; each day has a random track variant (sec/km)."""
    rng = random.Random(seed)
    ability = {f'h{index}': rng.gauss(0, 1) for index in range(120)}
    klass = {horse: ('open' if value > .3 else 'maiden') for horse, value in ability.items()}
    class_pace = {'open': 59.0, 'maiden': 61.0}
    races, truth = [], {}
    for number in range(days):
        day = (date(2022, 1, 1) + timedelta(days=number)).isoformat()
        variant = rng.gauss(0, variant_sd)
        for race_no, level in enumerate(('maiden', 'open', 'maiden')):
            pool = [horse for horse in ability if klass[horse] == level]
            field = rng.sample(pool, 10)
            paces = {horse: class_pace[level] - .4 * ability[horse] + variant + rng.gauss(0, .3) for horse in field}
            order = sorted(field, key=paces.get)
            race_id = f'{day}-{race_no}'
            rows = []
            for horse in field:
                rows.append(dict(day=day, race_id=race_id, horse_id=horse, course='東京', surface='芝', distance_m=1600,
                                 finish_status='finished', finish_position=order.index(horse) + 1,
                                 time_seconds=paces[horse] * 1.6, race_class=level, track_condition='良',
                                 carried_weight_kg=55))
                truth[(race_id, horse)] = (ability[horse], variant)
            races.append(rows)
    return races, ability, truth


class SpeedFigureTests(unittest.TestCase):
    def test_figures_track_ability_and_remove_the_day_variant(self):
        races, ability, truth = simulated_races(variant_sd=1.5)
        table, diagnostics = speed_feature_rows(races)
        self.assertGreater(diagnostics['variant_abs_mean'], .5)
        late = [row for race in races[-150:] for row in race]
        mean5 = np.asarray([table[(row['race_id'], row['horse_id'])][1] for row in late], dtype=float)
        target = np.asarray([ability[row['horse_id']] for row in late])
        known = np.isfinite(mean5)
        self.assertGreater(known.mean(), .9)
        self.assertGreater(np.corrcoef(mean5[known], target[known])[0, 1], .8)
        # The figure a horse earned on its previous start, against that day's true variant.
        last, figures, raw, variants = {}, [], [], []
        for race in races:
            for row in race:
                horse = row['horse_id']
                if horse in last and race[0]['day'] > races[300][0]['day']:
                    previous = last[horse]
                    figures.append(table[(row['race_id'], horse)][0])
                    raw.append(-previous['time_seconds'] / 1.6)
                    variants.append(truth[(previous['race_id'], horse)][1])
                last[horse] = row
        figures, raw, variants = map(np.asarray, (figures, raw, variants))
        self.assertLess(np.corrcoef(raw, variants)[0, 1], -.6)
        self.assertGreater(np.corrcoef(figures, variants)[0, 1], -.35)

    def test_rows_only_see_earlier_days(self):
        races, _, _ = simulated_races(days=40)
        first, _ = speed_feature_rows(races)
        last_day = races[-1][0]['day']
        changed = [[dict(row, time_seconds=row['time_seconds'] + 30) if row['day'] == last_day else row for row in race]
                   for race in races]
        second, _ = speed_feature_rows(changed)
        for race in races:
            if race[0]['day'] == last_day:
                for row in race:
                    key = (row['race_id'], row['horse_id'])
                    self.assertEqual(first[key], second[key])

    def test_first_start_has_no_figure_but_count_zero(self):
        races, _, _ = simulated_races(days=2)
        table, _ = speed_feature_rows(races)
        row = races[0][0]
        values = dict(zip(SPEED_FEATURES, table[(row['race_id'], row['horse_id'])]))
        self.assertIsNone(values['speed_fig_previous'])
        self.assertEqual(values['speed_fig_count'], 0.0)


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy')),
                     'Install requirements-ml.txt for the v3e experiment tests')
class ExperimentETests(unittest.TestCase):
    def test_speed_experiment_runs(self):
        from keiba_ai.cli import prepare
        from keiba_ai.experiment_v3e import main
        from test_ai_experiment_v3c import announced_csv
        with TemporaryDirectory() as name:
            root = Path(name)
            announced_csv(root / 'source.csv', years=(2022, 2023, 2024, 2025, 2026))
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
                self.assertEqual(set(report['groups'][group]['summary']), {'base', 'speed'})
                self.assertGreater(report['groups'][group]['speed_coverage']['speed_fig_previous'], .5)


if __name__ == '__main__':
    unittest.main()
