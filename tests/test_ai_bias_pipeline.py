"""Causal integration of day-frozen form and earlier-race track observations."""
from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.bias import BIAS_CONFIGS, BIAS_FEATURE_NAMES, DEFAULT_BIAS_CONFIG, BiasPriorState, IntradayBiasState
from keiba_ai.features import (CONTEXT_FEATURE_NAMES, FEATURE_NAMES, HistoryState,
                               _context_features, build_features, race_features)
from test_ai_features import corner_race, read_features, write_database


def source_race(day, number, *, surface='ダ', course='東京', popularity_reverse=True):
    result = corner_race(day, number, positions=tuple(range(1, 9)))
    for index, row in enumerate(result, 1):
        row.update(surface=surface, course=course, popularity=9-index if popularity_reverse else index,
                   track_condition='良', declared_field_size=8)
    return result


class BiasPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def build(self, name, races):
        database = self.root/name
        write_database(database, races)
        report = build_features(database)
        rows = read_features(database)
        return report, {(row[0], row[1], row[2]): json.loads(row[4]) for row in rows}, database

    def test_self_and_later_outcomes_cannot_change_a_races_inputs(self):
        day = '2024-01-02'
        races = [source_race(day, number) for number in (1, 2, 3)]
        changed = deepcopy(races)
        for race in changed[1:]:
            for row in race:
                row.update(finish_position=9-row['horse_number'], popularity=row['horse_number'],
                           track_condition='不良', weather='雨', odds=9999)
        _, original, _ = self.build('original.sqlite3', races)
        _, other, _ = self.build('changed.sqlite3', changed)
        first_two = [key for key in original if int(key[1][-2:]) <= 2]
        self.assertTrue(first_two)
        for key in first_two:
            self.assertEqual(original[key], other[key])
        self.assertTrue(any(original[key] != other[key] for key in original if key[1].endswith('03')))
        # Even a changed earlier same-day outcome cannot enter ordinary form.
        for key in original:
            self.assertEqual(original[key][:96], other[key][:96])

    def test_old_context_prefix_is_unchanged_and_same_day_guard_remains(self):
        previous = source_race('2024-01-01', 1)
        current = source_race('2024-01-02', 2)
        histories = HistoryState()
        histories.update_race(previous)
        expected = _context_features(current, histories)
        actual = race_features(current, histories)
        self.assertEqual(len(CONTEXT_FEATURE_NAMES), 96)
        self.assertEqual(FEATURE_NAMES[:96], CONTEXT_FEATURE_NAMES)
        self.assertEqual(len(FEATURE_NAMES), 104)
        self.assertEqual([row[:96] for row in actual], expected)
        self.assertEqual(histories.asof_day, '2024-01-01')
        histories.update_race(current)
        with self.assertRaises(ValueError):
            race_features(current, histories, intraday=IntradayBiasState('2024-01-02'), priors=BiasPriorState())

    def test_three_candidate_vectors_are_stored_before_the_result_is_added(self):
        races = [source_race('2024-01-02', number) for number in (1, 2)]
        report, features, database = self.build('variants.sqlite3', races)
        self.assertEqual(len(report['bias_configs']), 3)
        self.assertEqual(report['default_bias_config']['identifier'], DEFAULT_BIAS_CONFIG.identifier)
        self.assertEqual(report['races_with_intraday_bias'], 1)
        self.assertEqual(report['rows_with_intraday_bias'], 8)
        self.assertFalse(report['target_track_condition_used'])
        with sqlite3.connect(database) as connection:
            rows = connection.execute('SELECT race_id,horse_id,config_id,feature_json FROM bias_variants').fetchall()
            self.assertEqual(len(rows), 16*len(BIAS_CONFIGS))
            by_horse = {}
            for identifier, horse, config, encoded in rows:
                vector = json.loads(encoded)
                self.assertEqual(len(vector), len(BIAS_FEATURE_NAMES))
                by_horse.setdefault((identifier, horse), {})[config] = vector
                if config == DEFAULT_BIAS_CONFIG.identifier:
                    self.assertEqual(vector, features[('2024-01-02', identifier, horse)][96:])
            first = next(value for (identifier, horse), value in by_horse.items() if identifier.endswith('02'))
            self.assertGreater(len({tuple(vector) for vector in first.values()}), 1)
            original_count = len(rows)
        build_features(database)
        with sqlite3.connect(database) as connection:
            self.assertEqual(connection.execute('SELECT COUNT(*) FROM bias_variants').fetchone()[0], original_count)

    def test_training_and_prediction_share_frozen_history_and_original_gate_slots(self):
        first = source_race('2024-01-02', 1)
        target = source_race('2024-01-02', 2)
        target[-1].update(finish_status='scratched', finish_position=None)
        for row in target:
            row['declared_field_size'] = None
        _, built, _ = self.build('withdrawn.sqlite3', [target, first])
        intraday = IntradayBiasState('2024-01-02')
        intraday.add_result(first)
        active = target[:-1]
        direct = race_features(active, HistoryState(), intraday=intraday,
                               priors=BiasPriorState(), full_race=target)
        for row, vector in zip(active, direct):
            self.assertEqual(built[(row['day'], row['race_id'], row['horse_id'])], vector)
        named = dict(zip(FEATURE_NAMES, direct[-1]))
        self.assertGreater(named['day_inner_bias'], 0)
        self.assertAlmostEqual(named['horse_inner_bias_fit'], named['day_inner_bias']*(.5-6/7))

    def test_daily_state_is_reset_but_previous_day_prior_is_retained(self):
        races = [source_race('2024-01-01', 1), source_race('2024-01-02', 1)]
        for row in races[1]:
            row['race_id'] = '202405010201'
        _, built, _ = self.build('day-boundary.sqlite3', races)
        vector = next(value for (day, _, _), value in built.items() if day == '2024-01-02')
        named = dict(zip(FEATURE_NAMES, vector))
        self.assertEqual(named['day_bias_races'], 0)
        self.assertGreater(named['day_bias_prior_front'], 0)
        self.assertEqual(named['day_front_bias'], named['day_bias_prior_front'])


if __name__ == '__main__':
    unittest.main()
