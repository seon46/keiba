"""Causality and label tests using small fictional, normalized race rows."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.features import FEATURE_NAMES, CATEGORICAL_FEATURES, HistoryState, build_features, race_features


def race(day, number, *, positions=(1, 2, 3, 4), group='central_flat'):
    local = group.startswith('local_')
    return [
        dict(race_id=f'{day[:4]}050101{number:02d}', region='local' if local else 'central', day=day,
             horse_id=f'horse-{index}', horse_number=index, frame=index, sex='牡', age=4,
             jockey_id=f'jockey-{index}', trainer_id=f'trainer-{index}', carried_weight_kg=57,
             finish_position=position, finish_status='finished', course='水沢' if local else '東京',
             surface='ダ', distance_m=1600, declared_field_size=len(positions), group_name=group,
             odds=3.4, popularity=index, snapshot_id=f'snapshot-{number}')
        for index, position in enumerate(positions, 1)
    ]


def values(vector):
    return dict(zip(FEATURE_NAMES, vector))


def write_database(path, races):
    with sqlite3.connect(path) as connection:
        connection.executescript('''
            CREATE TABLE input_rows (
                region TEXT, race_id TEXT, day TEXT, horse_id TEXT, horse_number INTEGER,
                group_name TEXT, payload_json TEXT, PRIMARY KEY(region,race_id,horse_id)
            );
            CREATE TABLE race_issues (region TEXT,race_id TEXT,reason TEXT);
        ''')
        for items in races:
            for row in items:
                connection.execute('INSERT INTO input_rows VALUES (?,?,?,?,?,?,?)',
                                   (row['region'], row['race_id'], row['day'], row['horse_id'],
                                    row['horse_number'], row['group_name'], json.dumps(row)))


def read_features(path):
    with sqlite3.connect(path) as connection:
        return connection.execute(
            'SELECT day,race_id,horse_id,target,feature_json FROM features ORDER BY day,race_id,horse_id'
        ).fetchall()


class HistoryFeatureTests(unittest.TestCase):
    def test_past_form_and_prediction_share_numeric_schema(self):
        state = HistoryState()
        state.update_race(race('2024-01-01', 1))
        state.update_race(race('2024-01-02', 2, positions=(4, 1, 2, 3)))
        upcoming = race('2024-01-03', 3)
        for row in upcoming:
            row['distance_m'] = 1800
        result = values(race_features(upcoming, state)[0])
        self.assertEqual(result['horse_prior_starts'], 2)
        self.assertEqual(result['horse_prior_win_rate'], .5)
        self.assertEqual(result['horse_prior_top3_rate'], .5)
        self.assertEqual(result['horse_recent5_mean_rank_fraction'], .5)
        self.assertEqual(result['horse_previous_finish_position'], 4)
        self.assertEqual(result['horse_days_since_last_start'], 1)
        self.assertEqual(result['horse_distance_change_m'], 200)
        self.assertEqual(len(result), len(FEATURE_NAMES))
        self.assertTrue(set(CATEGORICAL_FEATURES).issubset(FEATURE_NAMES))
        self.assertFalse(result['horse_no_prior_history'])

    def test_current_results_times_odds_and_bodyweight_are_never_features(self):
        state = HistoryState()
        state.update_race(race('2024-01-01', 1))
        incoming = race('2024-01-02', 2)
        changed = deepcopy(incoming)
        for index, row in enumerate(changed):
            row.update(finish_position=100-index, finish_status='unknown', odds=10000,
                       popularity=99, weight_kg=9999, weight_change_kg=200,
                       time_seconds=1, last_3f_seconds=1, corner_order='1-1-1',
                       weather='雪', track_condition='不', margin='大差')
        self.assertEqual(race_features(incoming, state), race_features(changed, state))

    def test_same_or_future_history_is_refused(self):
        state = HistoryState()
        state.update_race(race('2024-01-02', 2))
        for day in ('2024-01-01', '2024-01-02'):
            with self.subTest(day=day), self.assertRaises(ValueError):
                race_features(race(day, 3), state)
        with self.assertRaises(ValueError):
            state.update_race(race('2024-01-01', 1))

    def test_withdrawn_runners_never_update_any_history(self):
        original = race('2024-01-01', 1)
        original[1]['finish_status'] = 'scratched'
        original[2]['finish_status'] = 'excluded'
        original[3]['finish_status'] = 'disqualified'
        state = HistoryState()
        state.update_race(original)
        upcoming = race('2024-01-02', 2)
        vectors = [values(vector) for vector in race_features(upcoming, state)]
        for index in (1, 2):
            self.assertEqual(vectors[index]['horse_prior_starts'], 0)
            self.assertEqual(vectors[index]['jockey_prior_starts'], 0)
            self.assertEqual(vectors[index]['trainer_prior_starts'], 0)
            self.assertTrue(vectors[index]['horse_no_prior_history'])
        self.assertEqual(vectors[3]['horse_prior_top3_rate'], 0)
        self.assertIsNone(vectors[3]['horse_previous_finish_position'])

    def test_transfer_keeps_observed_horse_form_without_pooling_jockey_rates(self):
        state = HistoryState()
        state.update_race(race('2024-01-01', 1))
        result = values(race_features(race('2024-01-02', 2, group='local_flat'), state)[0])
        self.assertEqual(result['horse_global_prior_starts'], 1)
        self.assertEqual(result['horse_prior_starts'], 0)
        self.assertTrue(result['horse_new_in_group'])
        self.assertFalse(result['horse_no_prior_history'])
        self.assertEqual(result['horse_recent5_top3_rate'], 1)
        self.assertEqual(result['jockey_prior_starts'], 0)

    def test_same_day_previous_race_order_is_not_claimed(self):
        first = race('2024-01-01', 1)
        second = race('2024-01-01', 2, positions=(4, 1, 2, 3))
        state = HistoryState()
        state.update_race(second)
        state.update_race(first)
        vector = values(race_features(race('2024-01-02', 3), state)[0])
        self.assertEqual(vector['horse_prior_starts'], 2)
        self.assertIsNone(vector['horse_previous_finish_position'])
        self.assertIsNone(vector['horse_distance_change_m'])
        self.assertEqual(vector['horse_days_since_last_start'], 1)

    def test_unknown_categories_and_missing_rider_ids_remain_unknown(self):
        incoming = race('2024-01-02', 2)
        incoming[0].update(course='未知の場', surface='不明', sex='不明', jockey_id=None, trainer_id='')
        vector = values(race_features(incoming, HistoryState())[0])
        for name in ('course_code', 'surface_code', 'sex_code', 'jockey_prior_top3_rate', 'trainer_prior_top3_rate'):
            self.assertIsNone(vector[name])
        self.assertTrue(vector['jockey_id_missing'])
        self.assertTrue(vector['trainer_id_missing'])


class StreamingFeatureBuildTests(unittest.TestCase):
    def setUp(self):
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)

    def build(self, name, races):
        path = self.directory / name
        write_database(path, races)
        summary = build_features(path)
        return summary, read_features(path), path

    def test_same_day_horse_jockey_and_trainer_outcomes_are_frozen(self):
        _, rows, _ = self.build('same-day.sqlite3', [race('2024-01-01', 1), race('2024-01-01', 2), race('2024-01-02', 3)])
        first = [values(json.loads(row[4])) for row in rows if row[0] == '2024-01-01']
        for vector in first:
            for field in ('horse_prior_starts', 'jockey_prior_starts', 'trainer_prior_starts'):
                self.assertEqual(vector[field], 0)
        second = [values(json.loads(row[4])) for row in rows if row[0] == '2024-01-02']
        self.assertTrue(all(vector['horse_prior_starts'] == 2 for vector in second))
        self.assertTrue(all(vector['jockey_prior_starts'] == 2 for vector in second))

    def test_changed_future_results_leave_earlier_features_unchanged(self):
        races = [race('2024-01-01', 1), race('2024-01-02', 2), race('2026-01-03', 3)]
        changed = deepcopy(races)
        for row, position in zip(changed[-1], (4, 3, 2, 1)):
            row.update(finish_position=position, odds=9999, popularity=99, corner_order='9-9')
        _, original_rows, _ = self.build('original.sqlite3', races)
        _, changed_rows, _ = self.build('changed.sqlite3', changed)
        self.assertEqual([row for row in original_rows if row[0] < '2026-01-03'],
                         [row for row in changed_rows if row[0] < '2026-01-03'])
        self.assertEqual([row[:3] + (row[4],) for row in original_rows],
                         [row[:3] + (row[4],) for row in changed_rows])

    def test_training_omits_withdrawn_and_labels_dnf_disqualification_zero(self):
        source = race('2024-01-01', 1, positions=(1, 2, 3, 4, 5, 6))
        for row, status in zip(source[1:], ('scratched', 'excluded', 'did_not_finish', 'disqualified', 'finished')):
            row['finish_status'] = status
        # Final ranked runners remain valid relative to actual starter count.
        source[-1]['finish_position'] = 2
        summary, rows, _ = self.build('status.sqlite3', [source])
        self.assertEqual(summary['rows'], 4)
        targets = {row[2]: row[3] for row in rows}
        self.assertEqual(targets, {'horse-1': 1, 'horse-4': 0, 'horse-5': 0, 'horse-6': 1})

    def test_dead_heat_can_produce_four_top3_labels_and_rebuild_has_no_stale_rows(self):
        summary, rows, path = self.build('ties.sqlite3', [race('2024-01-01', 1, positions=(1, 2, 3, 3, 5))])
        self.assertEqual(summary['groups']['central_flat']['positives'], 4)
        self.assertEqual(sum(row[3] for row in rows), 4)
        self.assertEqual(build_features(path)['rows'], 5)
        self.assertEqual(len(read_features(path)), 5)


if __name__ == '__main__':
    unittest.main()
