"""Causality and small-sample behavior of day-local track tendencies."""
from copy import deepcopy
from datetime import date
import unittest

from keiba_ai.bias import (
    BIAS_CONFIGS, BIAS_FEATURE_NAMES, BiasConfig, BiasPriorState,
    IntradayBiasState, bias_features, gate_position, race_order,
)
from keiba_ai.features import RANK_FEATURE_NAMES


def race(day='2024-10-11', number=1, *, course='東京', surface='芝',
         group='central_flat', meeting=1, meeting_day=2, heads=8, gain=True):
    local = group.startswith('local_')
    course_code = {'東京': '05', '中山': '06', '盛岡': '35', '水沢': '36', '帯広(ば)': '65'}[course]
    identifier = (f'{day[:4]}{course_code}{day[5:7]}{day[8:10]}{number:02d}' if local else
                  f'{day[:4]}{course_code}{meeting:02d}{meeting_day:02d}{number:02d}')
    return [dict(day=day, region='local' if local else 'central', race_id=identifier,
                 course=course, surface=surface, group_name=group, distance_m=1600,
                 horse_id=f'horse-{index}', horse_number=index, declared_field_size=heads,
                 finish_position=index, finish_status='finished',
                 popularity=heads + 1 - index if gain else index,
                 corner_positions={'4': index}, track_condition='良')
            for index in range(1, heads + 1)]


def forms(heads):
    values = [[None] * 96 for _ in range(heads)]
    corner = RANK_FEATURE_NAMES.index('horse_group_recent5_last_corner_fraction')
    front = RANK_FEATURE_NAMES.index('horse_group_recent5_corner_front_rate')
    for index, row in enumerate(values):
        row[corner] = index / max(heads - 1, 1)
        row[front] = 1 if index < 2 else 0
    return values


def named(vector):
    return dict(zip(BIAS_FEATURE_NAMES, vector))


class RaceOrderTests(unittest.TestCase):
    def test_race_number_requires_known_course_and_observed_date_consistency(self):
        self.assertEqual(race_order(race(number=7)), 7)
        self.assertEqual(race_order(race(number=7, group='local_flat', course='盛岡', surface='ダ')), 7)
        for field, value in (('course', '中山'), ('race_id', '202405000007'),
                             ('race_id', '202405010213'), ('race_id', '202305010207')):
            changed = race(number=7)
            for row in changed: row[field] = value
            self.assertIsNone(race_order(changed))
        local = race(group='local_flat', course='盛岡', surface='ダ')
        for row in local: row['day'] = '2024-10-12'
        self.assertIsNone(race_order(local))

    def test_mixed_day_or_surface_is_an_error_instead_of_silent_ordering(self):
        for field, value in (('day', '2024-10-12'), ('surface', 'ダ'), ('distance_m', 1800)):
            changed = race()
            changed[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                race_order(changed)

    def test_gate_position_keeps_scratched_number_gaps_and_declared_outer_gate(self):
        full = race(heads=10)
        full[-1]['finish_status'] = 'scratched'
        active = full[:-1]
        self.assertAlmostEqual(gate_position(active[-1], active), 8 / 9)
        for row in full: row['declared_field_size'] = None
        self.assertEqual(gate_position(active[-1], active), 1)
        self.assertAlmostEqual(gate_position(active[-1], full), 8 / 9)
        full[0]['declared_field_size'] = 7
        self.assertIsNone(gate_position(active[-1], full))


class IntradayBiasTests(unittest.TestCase):
    def test_sign_shrinkage_and_horse_interactions_are_consistent(self):
        state = IntradayBiasState('2024-10-11')
        self.assertTrue(state.add_result(race(number=1)))
        target = race(number=2)
        result = [named(row) for row in bias_features(target, state, BiasPriorState(), form_vectors=forms(8))]
        self.assertEqual(state.records[0].front, 2)
        self.assertEqual(state.records[0].inner, 2)
        self.assertAlmostEqual(result[0]['day_front_bias'], 2 * .8 / (4 + .8))
        self.assertEqual(result[0]['day_bias_races'], 1)
        self.assertEqual(result[0]['day_bias_prior_front'], 0)
        self.assertGreater(result[0]['horse_front_bias_fit'], 0)
        self.assertLess(result[-1]['horse_front_bias_fit'], 0)
        self.assertGreater(result[0]['horse_inner_bias_fit'], 0)
        self.assertLess(result[-1]['horse_inner_bias_fit'], 0)
        self.assertEqual(result[0]['field_front_runner_count'], 2)
        self.assertEqual(result[0]['meeting_day_index'], 2)

    def test_current_outcomes_odds_and_final_going_are_not_inputs(self):
        state = IntradayBiasState('2024-10-11')
        state.add_result(race())
        target = race(number=2)
        original = bias_features(target, state, BiasPriorState(), form_vectors=forms(8))
        changed = deepcopy(target)
        for row in changed:
            row.update(finish_position=99, finish_status='did_not_finish', popularity=99,
                       odds=1000, weight_kg=999, weather='雪', track_condition='不良',
                       corner_positions={'4': 99}, time_seconds=999)
        self.assertEqual(original, bias_features(changed, state, BiasPriorState(), form_vectors=forms(8)))

    def test_own_and_future_results_cannot_enter_target_state(self):
        for prior_number in (2, 3):
            state = IntradayBiasState('2024-10-11')
            state.add_result(race(number=prior_number))
            with self.subTest(prior_number=prior_number), self.assertRaises(ValueError):
                bias_features(race(number=2), state, BiasPriorState(), form_vectors=forms(8))

    def test_other_courses_surfaces_and_other_days_are_separate(self):
        state = IntradayBiasState('2024-10-11')
        state.add_result(race(number=3, course='中山'))
        state.add_result(race(number=3, surface='ダ'))
        output = named(bias_features(race(number=2), state, BiasPriorState(), form_vectors=forms(8))[0])
        self.assertEqual(output['day_bias_races'], 0)
        self.assertEqual(output['day_front_bias'], 0)
        with self.assertRaises(ValueError):
            state.add_result(race(day='2024-10-12'))
        with self.assertRaises(ValueError):
            bias_features(race(day='2024-10-12'), state, BiasPriorState(), form_vectors=forms(8))

    def test_small_fields_missing_corners_and_missing_popularity_are_not_zero_filled(self):
        for label in ('small', 'corners', 'popularity'):
            source = race(heads=5 if label == 'small' else 8)
            if label == 'corners':
                for row in source[:3]: row['corner_positions'] = {}
            if label == 'popularity':
                for row in source[:3]: row['popularity'] = None
            state = IntradayBiasState('2024-10-11')
            with self.subTest(label=label):
                self.assertFalse(state.add_result(source))
                self.assertEqual(state.valid_races, 0)
                result = named(bias_features(race(number=2), state, BiasPriorState(), form_vectors=forms(8))[0])
                self.assertEqual(result['day_bias_races'], 0)
        adequate = race()
        for row in adequate[:2]: row['corner_positions'] = {}
        state = IntradayBiasState('2024-10-11')
        self.assertTrue(state.add_result(adequate))  # Six of eight is sufficient.

    def test_unknown_order_and_unsupported_groups_do_not_create_intraday_observations(self):
        state = IntradayBiasState('2024-10-11')
        source = race()
        for row in source: row['race_id'] = 'unknown'
        self.assertFalse(state.add_result(source))
        for group, course in (('central_jump', '東京'), ('local_banei', '帯広(ば)')):
            unsupported = race(group=group, course=course)
            self.assertFalse(state.add_result(unsupported))
            result = named(bias_features(unsupported, state, BiasPriorState(), form_vectors=forms(8))[0])
            self.assertIsNone(result['day_front_bias'])
            self.assertEqual(result['day_bias_races'], 0)

    def test_duplicate_results_are_rejected_and_config_sweeps_do_not_mutate_state(self):
        state = IntradayBiasState('2024-10-11')
        state.add_result(race())
        original = state.records
        with self.assertRaises(ValueError): state.add_result(race())
        priors = BiasPriorState()
        predictions = [bias_features(race(number=2), state, priors, form_vectors=forms(8), config=config)
                       for config in BIAS_CONFIGS]
        self.assertEqual(state.records, original)
        self.assertIsNone(priors.latest_day)
        self.assertGreater(predictions[0][0][0], predictions[-1][0][0])
        self.assertEqual(predictions[1], bias_features(race(number=2), state, priors, form_vectors=forms(8)))

    def test_missing_past_style_stays_unknown_and_full_field_preserves_gate_fit(self):
        state = IntradayBiasState('2024-10-11')
        state.add_result(race())
        full = race(number=2, heads=10)
        full[-1]['finish_status'] = 'scratched'
        for row in full: row['declared_field_size'] = None
        active = full[:-1]
        vectors = [[None] * 96 for _ in active]
        result = [named(row) for row in bias_features(active, state, BiasPriorState(),
                                                     form_vectors=vectors, full_race=full)]
        self.assertIsNone(result[0]['horse_front_bias_fit'])
        self.assertIsNone(result[0]['field_front_runner_count'])
        self.assertAlmostEqual(result[-1]['horse_inner_bias_fit'], result[-1]['day_inner_bias'] * (.5 - 8 / 9))

    def test_recent_observed_going_drives_weight_without_using_target_final_going(self):
        state = IntradayBiasState('2024-10-11')
        first = race(number=1)
        second = race(number=2, gain=False)
        for row in second: row['track_condition'] = '重'
        state.add_result(first)
        state.add_result(second)
        result = named(bias_features(race(number=3), state, BiasPriorState(), form_vectors=forms(8))[0])
        expected = (2 * .8**2 * .5) / (4 + .8**2 * .5 + .8)
        self.assertAlmostEqual(result['day_front_bias'], expected)


class BiasPriorTests(unittest.TestCase):
    def test_first_race_uses_only_earlier_day_prior_and_previous_meeting_blend(self):
        prior = BiasPriorState()
        older = IntradayBiasState('2024-10-10')
        older.add_result(race(day='2024-10-10', meeting_day=1, gain=False))
        prior.update_day(older)
        previous = IntradayBiasState('2024-10-11')
        previous.add_result(race(meeting_day=2))
        prior.update_day(previous)
        target = race(day='2024-10-12', meeting_day=3)
        stats = prior.statistics(target)
        self.assertEqual(stats['front_count'], 2)
        self.assertEqual(stats['front_mean'], 1)
        self.assertEqual(stats['front_variance'], 2)
        self.assertEqual(prior.estimate(target), (1.5, 1.5))
        output = named(bias_features(target, IntradayBiasState('2024-10-12'), prior, form_vectors=forms(8))[0])
        self.assertEqual(output['day_bias_races'], 0)
        self.assertEqual(output['day_bias_prior_front'], 1.5)
        self.assertEqual(output['day_front_bias'], 1.5)
        self.assertEqual(prior.estimate(race(day='2024-10-12', meeting=2, meeting_day=1)), (1, 1))
        self.assertEqual(prior.estimate(race(day='2024-10-13', meeting_day=3)), (1, 1))

    def test_prior_never_accepts_same_or_future_day_history_and_double_updates(self):
        state = IntradayBiasState('2024-10-11')
        state.add_result(race())
        prior = BiasPriorState()
        prior.update_day(state)
        for day in ('2024-10-10', '2024-10-11'):
            with self.subTest(day=day), self.assertRaises(ValueError):
                bias_features(race(day=day), IntradayBiasState(day), prior, form_vectors=forms(8))
        with self.assertRaises(ValueError): prior.update_day(state)
        self.assertEqual(prior.latest_day, date(2024, 10, 11))

    def test_distance_surface_and_course_priors_are_separate(self):
        state = IntradayBiasState('2024-10-11')
        state.add_result(race())
        prior = BiasPriorState()
        prior.update_day(state)
        for field, value in (('distance_m', 2000), ('surface', 'ダ'), ('course', '中山')):
            target = race(day='2024-10-12')
            for row in target: row[field] = value
            with self.subTest(field=field): self.assertEqual(prior.estimate(target), (0, 0))

    def test_configs_are_finite_reproducible_and_validate_ranges(self):
        self.assertEqual([config.identifier for config in BIAS_CONFIGS], ['n3_d0.7', 'n4_d0.8', 'n5_d0.9'])
        for config in BIAS_CONFIGS: self.assertEqual(BiasConfig(**config.as_dict()), config)
        for values in ((0, .8), (4, 0), (4, 1.1), (float('inf'), .8), (4, float('nan')), (True, .8)):
            with self.subTest(values=values), self.assertRaises(ValueError): BiasConfig(*values)


if __name__ == '__main__':
    unittest.main()
