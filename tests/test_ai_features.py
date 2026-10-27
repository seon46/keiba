"""Causality and label tests using small fictional, normalized race rows."""

from copy import deepcopy
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from keiba_ai.features import BASE_FEATURE_NAMES, FORM_FEATURE_NAMES, RANK_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, FEATURE_NAMES, CATEGORICAL_FEATURES, HistoryState, build_features, race_features


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


def timed_race(day, number, *, positions=(1, 2, 3, 4), group='central_flat'):
    result = race(day, number, positions=positions, group=group)
    for index, row in enumerate(result):
        row.update(time_seconds=100 + index * 2, last_3f_seconds=35 + index)
    return result


def corner_race(day, number, *, positions=(1, 2, 3, 4), group='central_flat'):
    result = timed_race(day, number, positions=positions, group=group)
    for index, row in enumerate(result, 1):
        row['corner_positions'] = {str(corner): index for corner in (1, 2, 3, 4)}
        row['corner_sequence'] = [index] * 4
        row['corner_labels'] = [1, 2, 3, 4]
    return result


def context_race(day, number, *, positions=(1, 2, 3, 4), group='central_flat'):
    result = corner_race(day, number, positions=positions, group=group)
    for index, row in enumerate(result, 1):
        row.update(weather='雨', track_condition='重', track_moisture=2.5,
                   weight_kg=470+index*10, weight_change_kg=index-2,
                   odds=float(index*2), popularity=index,
                   race_class='1win', race_grade='G3', weight_rule='定量',
                   age_min=3, age_max=None, first_prize_man=800)
    return result


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
        for row in incoming:
            row.update(course='未知の場', surface='不明')
        incoming[0].update(sex='不明', jockey_id=None, trainer_id='')
        vector = values(race_features(incoming, HistoryState())[0])
        for name in ('course_code', 'surface_code', 'sex_code', 'jockey_prior_top3_rate', 'trainer_prior_top3_rate'):
            self.assertIsNone(vector[name])
        self.assertTrue(vector['jockey_id_missing'])
        self.assertTrue(vector['trainer_id_missing'])


class HistoricalTimingAndAptitudeTests(unittest.TestCase):
    def test_relative_times_and_sections_are_historical_and_distance_adjusted(self):
        state = HistoryState()
        state.update_race(timed_race('2024-01-01', 1))
        result = values(race_features(race('2024-01-02', 2), state)[1])
        self.assertAlmostEqual(result['horse_group_previous_speed_mps'], 1600 / 102)
        self.assertAlmostEqual(result['horse_group_recent5_winner_gap_per_km'], 1.25)
        self.assertAlmostEqual(result['horse_group_previous_mean_time_gap_per_km'], -.625)
        self.assertEqual(result['horse_group_recent5_last3f_seconds'], 36)
        self.assertAlmostEqual(result['horse_group_previous_last3f_vs_mean'], -.5)
        self.assertEqual(result['horse_group_recent5_timed_starts'], 1)
        self.assertEqual(result['horse_group_recent5_section_timed_starts'], 1)

    def test_dead_heat_winner_reference_and_missing_winner_time(self):
        tied = timed_race('2024-01-01', 1, positions=(1, 1, 3, 4))
        tied[1]['time_seconds'] = tied[0]['time_seconds']
        state = HistoryState()
        state.update_race(tied)
        vectors = [values(vector) for vector in race_features(race('2024-01-02', 2), state)]
        self.assertEqual(vectors[0]['horse_group_previous_winner_gap_per_km'], 0)
        self.assertEqual(vectors[1]['horse_group_previous_winner_gap_per_km'], 0)
        no_winner_time = timed_race('2024-01-01', 1)
        no_winner_time[0]['time_seconds'] = None
        state = HistoryState()
        state.update_race(no_winner_time)
        result = values(race_features(race('2024-01-02', 2), state)[1])
        self.assertIsNone(result['horse_group_previous_winner_gap_per_km'])
        self.assertIsNotNone(result['horse_group_previous_mean_time_gap_per_km'])

    def test_dnf_disqualification_and_invalid_clocks_never_become_zero_times(self):
        old = timed_race('2024-01-01', 1)
        old[2].update(finish_status='did_not_finish', time_seconds=1, last_3f_seconds=1)
        old[3].update(finish_status='disqualified', time_seconds=1, last_3f_seconds=1)
        state = HistoryState()
        state.update_race(old)
        vectors = [values(vector) for vector in race_features(race('2024-01-02', 2), state)]
        self.assertAlmostEqual(vectors[1]['horse_group_previous_mean_time_gap_per_km'], .625)
        self.assertAlmostEqual(vectors[1]['horse_group_previous_last3f_vs_mean'], .5)
        for index in (2, 3):
            self.assertIsNone(vectors[index]['horse_group_previous_speed_mps'])
            self.assertIsNone(vectors[index]['horse_group_previous_last3f_seconds'])
            self.assertEqual(vectors[index]['horse_group_recent5_timed_starts'], 0)
        invalid = timed_race('2024-01-01', 1)
        for row, value in zip(invalid, (None, 0, -1, float('inf'))):
            row.update(time_seconds=value, last_3f_seconds=value)
        state = HistoryState()
        state.update_race(invalid)
        for vector in race_features(race('2024-01-02', 2), state):
            result = values(vector)
            self.assertIsNone(result['horse_group_recent5_speed_mps'])
            self.assertIsNone(result['horse_group_recent5_last3f_seconds'])

    def test_single_timed_finisher_does_not_invent_mean_comparison(self):
        old = timed_race('2024-01-01', 1)
        for row in old[1:]:
            row.update(time_seconds=None, last_3f_seconds=None)
        state = HistoryState()
        state.update_race(old)
        result = values(race_features(race('2024-01-02', 2), state)[0])
        self.assertIsNotNone(result['horse_group_previous_speed_mps'])
        self.assertEqual(result['horse_group_previous_winner_gap_per_km'], 0)
        self.assertIsNone(result['horse_group_previous_mean_time_gap_per_km'])
        self.assertIsNone(result['horse_group_previous_last3f_vs_mean'])

    def test_aptitude_is_course_surface_and_200m_band_specific(self):
        state = HistoryState()
        state.update_race(timed_race('2024-01-01', 1))
        second = timed_race('2024-01-02', 2, positions=(4, 1, 2, 3))
        for row in second:
            row['distance_m'] = 1700
        state.update_race(second)
        upcoming = race('2024-01-03', 3)
        result = values(race_features(upcoming, state)[0])
        self.assertEqual(result['horse_course_surface_starts'], 2)
        self.assertEqual(result['horse_surface_distance_starts'], 2)
        self.assertEqual(result['horse_course_surface_distance_top3_rate'], .5)
        other_course = deepcopy(upcoming)
        for row in other_course:
            row['course'] = '中山'
        result = values(race_features(other_course, state)[0])
        self.assertEqual(result['horse_course_surface_starts'], 0)
        self.assertEqual(result['horse_surface_distance_starts'], 2)
        self.assertEqual(result['horse_course_surface_distance_starts'], 0)
        for row in upcoming:
            row.update(surface='芝', distance_m=1800)
        result = values(race_features(upcoming, state)[0])
        self.assertEqual(result['horse_course_surface_starts'], 0)
        self.assertEqual(result['horse_surface_distance_starts'], 0)
        self.assertIsNone(result['horse_course_surface_top3_rate'])

    def test_timing_and_aptitude_do_not_cross_flat_jump_or_banei_groups(self):
        state = HistoryState()
        state.update_race(timed_race('2024-01-01', 1))
        banei = timed_race('2024-01-02', 2, group='local_banei')
        for row in banei:
            row.update(course='帯広(ば)', surface=None, distance_m=200)
        state.update_race(banei)
        jumping = race('2024-01-03', 3, group='central_jump')
        flat = race('2024-01-03', 3)
        for row in jumping:
            row['surface'] = '障'
        jump_result = values(race_features(jumping, state)[0])
        flat_result = values(race_features(flat, state)[0])
        self.assertEqual(jump_result['horse_group_recent5_timed_starts'], 0)
        self.assertIsNone(jump_result['horse_group_previous_speed_mps'])
        self.assertEqual(jump_result['horse_course_surface_starts'], 0)
        self.assertEqual(flat_result['horse_group_previous_speed_mps'], 16)
        self.assertEqual(flat_result['horse_course_surface_starts'], 1)
        next_banei = deepcopy(banei)
        for row in next_banei:
            row['day'] = '2024-01-03'
        result = values(race_features(next_banei, state)[0])
        self.assertEqual(result['horse_group_previous_speed_mps'], 2)
        self.assertEqual(result['horse_course_surface_starts'], 1)

    def test_clock_only_changes_preserve_every_baseline_feature(self):
        original = timed_race('2024-01-01', 1)
        changed = deepcopy(original)
        for row in changed:
            row['time_seconds'] *= 1.1
            row['last_3f_seconds'] += 4
        first, second = HistoryState(), HistoryState()
        first.update_race(original)
        second.update_race(changed)
        upcoming = race('2024-01-02', 2)
        a, b = race_features(upcoming, first), race_features(upcoming, second)
        self.assertEqual(len(BASE_FEATURE_NAMES), 34)
        self.assertEqual(FEATURE_NAMES[:34], BASE_FEATURE_NAMES)
        self.assertEqual([row[:34] for row in a], [row[:34] for row in b])
        self.assertNotEqual([row[34:] for row in a], [row[34:] for row in b])

    def test_latest_missing_clock_stays_missing_and_recent_window_is_five_starts(self):
        state = HistoryState()
        for number in range(1, 7):
            old = timed_race(f'2024-01-{number:02d}', number)
            for row in old:
                row['time_seconds'] += number
            state.update_race(old)
        result = values(race_features(race('2024-01-07', 7), state)[0])
        self.assertEqual(result['horse_group_recent5_timed_starts'], 5)
        self.assertAlmostEqual(result['horse_group_recent5_speed_mps'], sum(1600 / (100+i) for i in range(2, 7)) / 5)
        last = timed_race('2024-01-07', 7)
        last[0].update(finish_status='did_not_finish', time_seconds=None, last_3f_seconds=None)
        state.update_race(last)
        result = values(race_features(race('2024-01-08', 8), state)[0])
        self.assertIsNone(result['horse_group_previous_speed_mps'])
        self.assertIsNotNone(result['horse_group_recent5_speed_mps'])
        self.assertEqual(result['horse_group_recent5_timed_starts'], 4)


class HistoricalCornerTests(unittest.TestCase):
    def test_positions_and_gains_are_normalized_by_historical_field_size(self):
        previous = corner_race('2024-01-01', 1)
        previous[0]['corner_positions'] = {'1': 4, '2': 4, '3': 4, '4': 2}
        state = HistoryState()
        state.update_race(previous)
        result = values(race_features(race('2024-01-02', 2), state)[0])
        self.assertEqual(result['horse_group_previous_first_corner_fraction'], 1)
        self.assertAlmostEqual(result['horse_group_previous_last_corner_fraction'], 1/3)
        self.assertAlmostEqual(result['horse_group_recent5_observed_corner_gain'], 2/3)
        self.assertAlmostEqual(result['horse_group_previous_third_to_fourth_gain'], 2/3)
        self.assertAlmostEqual(result['horse_group_previous_fourth_to_finish_gain'], 1/3)
        self.assertEqual(result['horse_group_recent5_corner_starts'], 1)
        self.assertEqual(result['horse_group_recent5_corner_leading_rate'], 0)
        self.assertEqual(result['horse_group_recent5_corner_front_rate'], 0)
        larger = corner_race('2024-01-01', 1, positions=tuple(range(1, 9)))
        larger[0]['corner_positions'] = {'1': 4, '4': 2}
        state = HistoryState()
        state.update_race(larger)
        result = values(race_features(race('2024-01-02', 2), state)[0])
        self.assertAlmostEqual(result['horse_group_previous_first_corner_fraction'], 3/7)
        self.assertAlmostEqual(result['horse_group_previous_fourth_to_finish_gain'], 1/7)

    def test_two_unlabelled_observations_do_not_claim_third_or_fourth_corner(self):
        old = corner_race('2024-01-01', 1)
        old[0].update(corner_positions={}, corner_sequence=[4, 2], corner_labels=None)
        state = HistoryState()
        state.update_race(old)
        result = values(race_features(race('2024-01-02', 2), state)[0])
        self.assertEqual(result['horse_group_previous_first_corner_fraction'], 1)
        self.assertAlmostEqual(result['horse_group_previous_observed_corner_gain'], 2/3)
        self.assertIsNone(result['horse_group_previous_third_to_fourth_gain'])
        self.assertIsNone(result['horse_group_previous_fourth_to_finish_gain'])
        explicit = deepcopy(old)
        explicit[0]['corner_labels'] = [3, 4]
        state = HistoryState()
        state.update_race(explicit)
        result = values(race_features(race('2024-01-02', 2), state)[0])
        self.assertAlmostEqual(result['horse_group_previous_third_to_fourth_gain'], 2/3)
        self.assertAlmostEqual(result['horse_group_previous_fourth_to_finish_gain'], 1/3)

    def test_parallel_group_leading_ranks_stay_tied_and_opponents_use_past_form(self):
        old = corner_race('2024-01-01', 1)
        old[1]['corner_positions'] = deepcopy(old[0]['corner_positions'])
        state = HistoryState()
        state.update_race(old)
        vectors = [values(vector) for vector in race_features(race('2024-01-02', 2), state)]
        self.assertEqual(vectors[0]['horse_group_previous_first_corner_fraction'],
                         vectors[1]['horse_group_previous_first_corner_fraction'])
        self.assertEqual(vectors[0]['horse_group_recent5_corner_leading_rate'], 1)
        self.assertEqual(vectors[1]['horse_group_recent5_corner_leading_rate'], 1)
        self.assertAlmostEqual(vectors[0]['opponents_recent5_corner_front_rate'], 1/3)
        self.assertAlmostEqual(vectors[2]['opponents_recent5_corner_front_rate'], 2/3)

    def test_missing_invalid_nonfinishing_corners_do_not_become_zero_position(self):
        old = corner_race('2024-01-01', 1)
        old[0].update(corner_positions={}, corner_sequence=[], corner_labels=None)
        old[1].update(finish_status='did_not_finish', finish_position=None)
        old[2].update(finish_status='disqualified', finish_position=None)
        old[3]['corner_positions'] = {'3': 100, '4': 1}
        state = HistoryState()
        state.update_race(old)
        vectors = [values(vector) for vector in race_features(race('2024-01-02', 2), state)]
        for result in vectors:
            self.assertEqual(result['horse_group_recent5_corner_starts'], 0)
            self.assertIsNone(result['horse_group_previous_first_corner_fraction'])
            self.assertIsNone(result['horse_group_recent5_corner_front_rate'])
            self.assertIsNone(result['opponents_recent5_corner_front_rate'])

    def test_corner_histories_do_not_cross_model_groups(self):
        state = HistoryState()
        state.update_race(corner_race('2024-01-01', 1))
        incoming = race('2024-01-02', 2, group='local_flat')
        result = values(race_features(incoming, state)[0])
        self.assertEqual(result['horse_global_prior_starts'], 1)
        self.assertEqual(result['horse_group_recent5_corner_starts'], 0)
        self.assertIsNone(result['horse_group_recent5_first_corner_fraction'])

    def test_current_corner_changes_are_unused_and_old_52_values_are_unchanged(self):
        old = corner_race('2024-01-01', 1)
        changed = deepcopy(old)
        for row in changed:
            row.update(corner_positions={'3': 4, '4': 1}, corner_sequence=[4, 1], corner_labels=[3, 4])
        a, b = HistoryState(), HistoryState()
        a.update_race(old)
        b.update_race(changed)
        upcoming = corner_race('2024-01-02', 2)
        first, second = race_features(upcoming, a), race_features(upcoming, b)
        self.assertEqual(len(FORM_FEATURE_NAMES), 52)
        self.assertEqual(FEATURE_NAMES[:52], FORM_FEATURE_NAMES)
        self.assertEqual([row[:52] for row in first], [row[:52] for row in second])
        self.assertNotEqual([row[52:] for row in first], [row[52:] for row in second])
        incoming_changed = deepcopy(upcoming)
        for row in incoming_changed:
            row.update(corner_positions={'3': 1, '4': 4}, corner_sequence=[1, 4], corner_labels=[3, 4])
        self.assertEqual(first, race_features(incoming_changed, a))

    def test_same_day_multiple_starts_do_not_claim_a_specific_previous_position(self):
        state = HistoryState()
        state.update_race(corner_race('2024-01-01', 1))
        state.update_race(corner_race('2024-01-01', 2))
        result = values(race_features(race('2024-01-02', 3), state)[0])
        self.assertEqual(result['horse_group_recent5_corner_starts'], 2)
        self.assertIsNone(result['horse_group_previous_first_corner_fraction'])
        self.assertIsNotNone(result['horse_group_recent5_first_corner_fraction'])


class HistoricalContextTests(unittest.TestCase):
    def test_fixed_category_vocabularies_cover_normalized_condition_enums(self):
        from keiba_ai import dataset, features
        from keiba_scraper.conditions import RACE_CLASSES_BY_REGION, RACE_GRADES_BY_REGION, WEIGHT_RULES
        for labels in RACE_CLASSES_BY_REGION.values():
            self.assertTrue(labels <= features._CLASSES.keys())
        for labels in RACE_GRADES_BY_REGION.values():
            self.assertTrue(labels <= features._GRADES.keys())
        self.assertTrue(WEIGHT_RULES <= features._WEIGHT_RULES.keys())
        self.assertTrue(dataset._WEATHER <= features._WEATHER.keys())
        self.assertTrue(dataset._GOING <= features._GOING.keys())
        for vocabulary in (features._CLASSES, features._GRADES, features._WEIGHT_RULES,
                           features._WEATHER, features._GOING):
            self.assertEqual(len(vocabulary), len(set(vocabulary.values())))
            self.assertTrue(all(0 <= code < 255 for code in vocabulary.values()))

    def test_observed_context_is_delayed_and_has_market_performance_residual(self):
        first = context_race('2024-01-01', 1, positions=(3, 1, 2, 4))
        second = context_race('2024-01-02', 2, positions=(2, 3, 1, 4))
        for row in second:
            row.update(track_condition='稍重', weather='曇', track_moisture=0,
                       weight_kg=row['weight_kg']+4, weight_change_kg=4)
        state = HistoryState()
        state.update_race(first)
        state.update_race(second)
        result = values(race_features(context_race('2024-01-03', 3), state)[0])
        self.assertEqual(result['horse_group_previous_bodyweight_kg'], 484)
        self.assertEqual(result['horse_group_recent5_bodyweight_kg'], 482)
        self.assertEqual(result['horse_group_previous_bodyweight_change_kg'], 4)
        self.assertEqual(result['horse_group_recent5_absolute_weight_change_kg'], 2.5)
        self.assertAlmostEqual(result['horse_group_previous_log_odds'], __import__('math').log(2))
        self.assertEqual(result['horse_group_previous_popularity_fraction'], 0)
        self.assertAlmostEqual(result['horse_group_previous_market_rank_gain'], -1/3)
        self.assertAlmostEqual(result['horse_group_recent5_market_rank_gain'], -.5)
        self.assertEqual(result['horse_group_previous_track_moisture'], 0)
        self.assertEqual(result['horse_group_recent5_track_moisture'], 1.25)
        self.assertEqual(result['horse_group_wet_starts'], 1)
        self.assertEqual(result['horse_group_wet_top3_rate'], 1)
        self.assertEqual(result['horse_group_dryish_starts'], 1)
        self.assertEqual(result['horse_group_dryish_top3_rate'], 1)
        self.assertEqual(result['horse_group_recent5_bodyweight_starts'], 2)
        self.assertEqual(result['horse_group_recent5_odds_starts'], 2)

    def test_current_measured_conditions_weights_and_market_are_not_inputs(self):
        state = HistoryState()
        state.update_race(context_race('2024-01-01', 1))
        upcoming = context_race('2024-01-02', 2)
        changed = deepcopy(upcoming)
        for row in changed:
            row.update(weather='雪', track_condition='不良', track_moisture=99,
                       weight_kg=9999, weight_change_kg=-999, odds=99999, popularity=99)
        self.assertEqual(race_features(upcoming, state), race_features(changed, state))

    def test_known_static_class_is_categorical_and_no_ordinal_class_change_is_invented(self):
        state = HistoryState()
        state.update_race(context_race('2024-01-01', 1))
        upcoming = context_race('2024-01-02', 2)
        same = values(race_features(upcoming, state)[0])
        self.assertEqual(same['horse_group_previous_same_class'], 1)
        self.assertEqual(same['horse_group_previous_same_grade'], 1)
        self.assertEqual(same['race_age_min'], 3)
        self.assertIsNone(same['race_age_max'])
        self.assertEqual(same['race_first_prize_man'], 800)
        changed = deepcopy(upcoming)
        for row in changed:
            row.update(race_class='A1', race_grade='Jpn1', weight_rule='ハンデ')
        different = values(race_features(changed, state)[0])
        self.assertEqual(different['horse_group_previous_same_class'], 0)
        self.assertEqual(different['horse_group_previous_same_grade'], 0)
        self.assertNotEqual(same['race_class_code'], different['race_class_code'])
        self.assertNotEqual(same['race_weight_rule_code'], different['race_weight_rule_code'])
        self.assertEqual(same['horse_group_previous_class_code'], different['horse_group_previous_class_code'])
        self.assertEqual([same[name] for name in RANK_FEATURE_NAMES], [different[name] for name in RANK_FEATURE_NAMES])
        self.assertTrue({'race_class_code','race_grade_code','race_weight_rule_code',
                         'horse_group_previous_class_code','horse_group_previous_grade_code',
                         'horse_group_previous_weather_code','horse_group_previous_going_code'} <= set(CATEGORICAL_FEATURES))

    def test_unknown_context_remains_missing_and_banei_weights_are_not_flat_clamped(self):
        incoming = race('2024-01-02', 2)
        missing = values(race_features(incoming, HistoryState())[0])
        for name in CONTEXT_FEATURE_NAMES[len(RANK_FEATURE_NAMES):]:
            if name.endswith('_starts'):
                self.assertEqual(missing[name], 0)
            else:
                self.assertIsNone(missing[name])
        banei = context_race('2024-01-01', 1, group='local_banei')
        for row in banei:
            row.update(course='帯広(ば)', surface=None, distance_m=200,
                       weight_kg=1150, weight_change_kg=0, odds=1.0, track_condition=None)
        state = HistoryState()
        state.update_race(banei)
        next_race = deepcopy(banei)
        for row in next_race:
            row['day'] = '2024-01-02'
        result = values(race_features(next_race, state)[0])
        self.assertEqual(result['horse_group_previous_bodyweight_kg'], 1150)
        self.assertEqual(result['horse_group_previous_bodyweight_change_kg'], 0)
        self.assertEqual(result['horse_group_previous_log_odds'], 0)
        self.assertEqual(result['horse_group_wet_starts'], 0)
        self.assertIsNone(result['horse_group_wet_top3_rate'])

    def test_nonfinisher_keeps_observed_context_but_never_invents_finish_rank_gain(self):
        old = context_race('2024-01-01', 1)
        old[1].update(finish_status='did_not_finish', finish_position=None)
        old[2].update(finish_status='disqualified', finish_position=None)
        state = HistoryState()
        state.update_race(old)
        vectors = [values(vector) for vector in race_features(context_race('2024-01-02', 2), state)]
        for index in (1, 2):
            result = vectors[index]
            self.assertIsNotNone(result['horse_group_previous_bodyweight_kg'])
            self.assertIsNotNone(result['horse_group_previous_log_odds'])
            self.assertIsNone(result['horse_group_previous_market_rank_gain'])
            self.assertEqual(result['horse_group_wet_starts'], 1)
            self.assertEqual(result['horse_group_wet_top3_rate'], 0)

    def test_context_histories_are_group_specific_and_past_changes_keep_old_66_columns(self):
        original = context_race('2024-01-01', 1)
        changed = deepcopy(original)
        for row in changed:
            row.update(weather='雪', track_condition='不良', track_moisture=10,
                       weight_kg=900, weight_change_kg=-20, odds=100, popularity=4)
        a, b = HistoryState(), HistoryState()
        a.update_race(original)
        b.update_race(changed)
        upcoming = context_race('2024-01-02', 2)
        first, second = race_features(upcoming, a), race_features(upcoming, b)
        self.assertEqual(len(RANK_FEATURE_NAMES), 66)
        self.assertEqual(FEATURE_NAMES[:66], RANK_FEATURE_NAMES)
        self.assertEqual([row[:66] for row in first], [row[:66] for row in second])
        self.assertNotEqual([row[66:] for row in first], [row[66:] for row in second])
        local = context_race('2024-01-02', 3, group='local_flat')
        result = values(race_features(local, a)[0])
        self.assertEqual(result['horse_global_prior_starts'], 1)
        self.assertEqual(result['horse_group_recent5_bodyweight_starts'], 0)
        self.assertIsNone(result['horse_group_previous_bodyweight_kg'])
        self.assertEqual(result['horse_group_wet_starts'], 0)

    def test_context_window_is_last_five_starts_and_missing_latest_is_not_backfilled(self):
        state = HistoryState()
        for number in range(1, 7):
            old = context_race(f'2024-01-{number:02d}', number)
            for row in old:
                row.update(weight_kg=400+number*10, odds=float(number))
            state.update_race(old)
        result = values(race_features(context_race('2024-01-07', 7), state)[0])
        self.assertEqual(result['horse_group_recent5_bodyweight_kg'], 440)
        self.assertEqual(result['horse_group_recent5_bodyweight_starts'], 5)
        last = context_race('2024-01-07', 7)
        last[0].update(weight_kg=None, odds=None, popularity=None, weight_change_kg=None)
        state.update_race(last)
        result = values(race_features(context_race('2024-01-08', 8), state)[0])
        self.assertIsNone(result['horse_group_previous_bodyweight_kg'])
        self.assertIsNone(result['horse_group_previous_log_odds'])
        self.assertIsNotNone(result['horse_group_recent5_bodyweight_kg'])
        self.assertEqual(result['horse_group_recent5_bodyweight_starts'], 4)


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
            row.update(finish_position=position, odds=9999, popularity=99, corner_order='9-9',
                       time_seconds=9999, last_3f_seconds=777)
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

    def test_same_day_clock_outcomes_are_not_used_by_any_race_that_day(self):
        first, second = timed_race('2024-01-01', 1), timed_race('2024-01-01', 2)
        second[0].update(time_seconds=500, last_3f_seconds=100)
        _, rows, _ = self.build('same-day-clock.sqlite3', [first, second, race('2024-01-02', 3)])
        for row in rows:
            vector = values(json.loads(row[4]))
            if row[0] == '2024-01-01':
                self.assertIsNone(vector['horse_group_recent5_speed_mps'])
                self.assertEqual(vector['horse_group_recent5_timed_starts'], 0)
            else:
                self.assertEqual(vector['horse_group_recent5_timed_starts'], 2)
                self.assertIsNone(vector['horse_group_previous_speed_mps'])

    def test_same_day_corner_outcomes_are_frozen_until_the_next_day(self):
        first, second = corner_race('2024-01-01', 1), corner_race('2024-01-01', 2)
        second[0]['corner_positions'] = {'1': 4, '4': 2}
        _, rows, _ = self.build('same-day-corners.sqlite3', [first, second, race('2024-01-02', 3)])
        for row in rows:
            vector = values(json.loads(row[4]))
            if row[0] == '2024-01-01':
                self.assertEqual(vector['horse_group_recent5_corner_starts'], 0)
                self.assertIsNone(vector['horse_group_recent5_first_corner_fraction'])
                self.assertIsNone(vector['opponents_recent5_corner_front_rate'])
            else:
                self.assertEqual(vector['horse_group_recent5_corner_starts'], 2)
                self.assertIsNone(vector['horse_group_previous_first_corner_fraction'])

    def test_future_corner_changes_only_affect_subsequent_dates(self):
        races = [corner_race('2024-01-01', 1), corner_race('2024-01-02', 2),
                 corner_race('2026-01-03', 3), corner_race('2026-01-04', 4)]
        changed = deepcopy(races)
        for row in changed[2]:
            row.update(corner_positions={'3': 4, '4': 1}, corner_sequence=[4, 1], corner_labels=[3, 4])
        _, original, _ = self.build('corner-original.sqlite3', races)
        _, alternative, _ = self.build('corner-changed.sqlite3', changed)
        self.assertEqual([row for row in original if row[0] <= '2026-01-03'],
                         [row for row in alternative if row[0] <= '2026-01-03'])
        before = [json.loads(row[4]) for row in original if row[0] == '2026-01-04']
        after = [json.loads(row[4]) for row in alternative if row[0] == '2026-01-04']
        self.assertEqual([row[:52] for row in before], [row[:52] for row in after])
        self.assertNotEqual([row[52:] for row in before], [row[52:] for row in after])

    def test_same_day_context_outcomes_are_not_used_until_the_next_day(self):
        first, second = context_race('2024-01-01', 1), context_race('2024-01-01', 2)
        for row in second:
            row.update(weight_kg=900, odds=100, track_condition='不良')
        _, rows, _ = self.build('same-day-context.sqlite3', [first, second, context_race('2024-01-02', 3)])
        for row in rows:
            vector = values(json.loads(row[4]))
            if row[0] == '2024-01-01':
                self.assertIsNone(vector['horse_group_recent5_bodyweight_kg'])
                self.assertIsNone(vector['horse_group_previous_log_odds'])
                self.assertEqual(vector['horse_group_wet_starts'], 0)
                self.assertIsNotNone(vector['race_class_code'])  # Pre-race condition is known.
            else:
                self.assertEqual(vector['horse_group_recent5_bodyweight_starts'], 2)
                self.assertEqual(vector['horse_group_wet_starts'], 2)
                self.assertIsNone(vector['horse_group_previous_bodyweight_kg'])

    def test_future_measured_context_changes_only_affect_later_dates(self):
        races = [context_race('2024-01-01', 1), context_race('2024-01-02', 2),
                 context_race('2026-01-03', 3), context_race('2026-01-04', 4)]
        changed = deepcopy(races)
        for row in changed[2]:
            row.update(weather='雪', track_condition='不良', track_moisture=8,
                       weight_kg=900, weight_change_kg=-20, odds=100, popularity=4)
        _, original, _ = self.build('context-original.sqlite3', races)
        _, alternative, _ = self.build('context-changed.sqlite3', changed)
        self.assertEqual([row for row in original if row[0] <= '2026-01-03'],
                         [row for row in alternative if row[0] <= '2026-01-03'])
        before = [json.loads(row[4]) for row in original if row[0] == '2026-01-04']
        after = [json.loads(row[4]) for row in alternative if row[0] == '2026-01-04']
        self.assertEqual([row[:66] for row in before], [row[:66] for row in after])
        self.assertNotEqual([row[66:] for row in before], [row[66:] for row in after])


if __name__ == '__main__':
    unittest.main()
