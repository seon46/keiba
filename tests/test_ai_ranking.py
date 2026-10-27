import importlib.util
import itertools
import unittest
import warnings


@unittest.skipUnless(importlib.util.find_spec('numpy'), 'Install requirements-ml.txt for ranking tests')
class RankingTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from keiba_ai.ranking import rank_distribution
        self.np = np
        self.distribution = rank_distribution

    def exact(self, scores, finish_probabilities, *, temperature=1.0, uniform=False):
        """Independent small-n reference: enumerate full orders and finish sets."""
        np = self.np
        count = len(scores)
        weights = np.exp(np.asarray(scores) / temperature)
        result = np.zeros((count, count + 1))
        for order in itertools.permutations(range(count)):
            order_probability = 1.0
            remaining = list(order)
            for horse in order:
                order_probability *= (1 / len(remaining) if uniform else
                                      weights[horse] / sum(weights[i] for i in remaining))
                remaining.remove(horse)
            for finished in itertools.product((False, True), repeat=count):
                probability = order_probability
                for horse, indicator in enumerate(finished):
                    p = finish_probabilities[horse]
                    probability *= p if indicator else 1 - p
                rank = 0
                for horse in order:
                    column = rank if finished[horse] else count
                    result[horse, column] += probability
                    rank += int(finished[horse])
        return result

    def occupancy(self, probabilities):
        """Enumerate finish sets; independent of strengths or PL simulation."""
        np = self.np
        result = np.zeros(len(probabilities))
        for finished in itertools.product((False, True), repeat=len(probabilities)):
            probability = 1.0
            for p, indicator in zip(probabilities, finished):
                probability *= p if indicator else 1 - p
            result[:sum(finished)] += probability
        return result

    def test_conditional_dp_preserves_joint_constraints_even_with_one_sample(self):
        np = self.np
        p = np.array([0.0, .000001, .35, .8, 1.0])
        for samples in (1, 7, 127):
            for smoothing in (0, .001, 1):
                with self.subTest(samples=samples, smoothing=smoothing):
                    result = self.distribution([9, 2, 0, -1, 4], p, temperature=.7,
                                               samples=samples, smoothing=smoothing)
                    self.assertEqual(result.shape, (5, 6))
                    self.assertTrue(np.isfinite(result).all())
                    self.assertTrue((result >= 0).all())
                    np.testing.assert_allclose(result.sum(axis=1), 1, atol=2e-14)
                    np.testing.assert_allclose(result[:, :-1].sum(axis=1), p, atol=2e-14)
                    np.testing.assert_array_equal(result[:, -1], 1-p)
                    np.testing.assert_allclose(result[:, :-1].sum(axis=0), self.occupancy(p), atol=2e-14)
                    self.assertTrue((np.diff(result[:, :-1].sum(axis=0)) <= 2e-14).all())
                    self.assertTrue((result[:, :-1].sum(axis=0) <= 1+2e-14).all())

    def test_all_finish_matrix_is_doubly_stochastic(self):
        np = self.np
        result = self.distribution([3, -1, 2, 0], np.ones(4), temperature=.9,
                                   samples=3, smoothing=.001)
        np.testing.assert_allclose(result[:, :-1].sum(axis=0), 1, atol=1e-14)
        np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-14)
        np.testing.assert_array_equal(result[:, -1], 0)
        self.assertTrue((result[:, :-1] > 0).all())

    def test_nonfinishers_are_not_assigned_fictional_or_vacant_ranks(self):
        np = self.np
        result = self.distribution([1e308, 0, -1e308], [0, 1, 1], temperature=1e-300,
                                   samples=17, smoothing=0)
        np.testing.assert_array_equal(result, [[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]])
        np.testing.assert_array_equal(self.distribution([0, 1], [0, 0], temperature=1),
                                      [[0, 0, 1], [0, 0, 1]])

    def test_one_horse_and_rare_nonfinish_are_exact_not_monte_carlo_events(self):
        np = self.np
        np.testing.assert_array_equal(self.distribution([42], [.37], temperature=1, samples=1),
                                      [[.37, .63]])
        p = np.array([.99999999, .999999999])
        result = self.distribution([0, 1], p, temperature=1, samples=1)
        np.testing.assert_array_equal(result[:, -1], 1-p)
        self.assertTrue((result[:, -1] > 0).all())

    def test_uniform_smoothing_is_an_exact_joint_law_with_the_same_finish_probabilities(self):
        np = self.np
        from keiba_ai.ranking import uniform_rank_distribution
        scores, p = [8, -3, 1], [.2, 1.0, .7]
        result = self.distribution(scores, p, temperature=.1, samples=1, smoothing=1)
        np.testing.assert_allclose(result, self.exact(scores, p, uniform=True), atol=1e-14)
        np.testing.assert_array_equal(uniform_rank_distribution(p), result)
        np.testing.assert_allclose(self.distribution([100, 1, -2], [1, 1, 1], temperature=1,
                                                   samples=1, smoothing=1)[:, :-1], 1/3, atol=1e-14)

    def test_mixture_equals_its_two_joint_components(self):
        np = self.np
        arguments = dict(temperature=.7, samples=37, seed=81)
        pl = self.distribution([1, 0, -1], [.8, .5, .9], smoothing=0, **arguments)
        uniform = self.distribution([1, 0, -1], [.8, .5, .9], smoothing=1, **arguments)
        mixed = self.distribution([1, 0, -1], [.8, .5, .9], smoothing=.2, **arguments)
        np.testing.assert_allclose(mixed, .8*pl+.2*uniform, atol=1e-14)

    def test_small_n_matches_enumerated_pl_theory(self):
        np = self.np
        scores = [1.2, -.5, .3]
        for p in ([1, 1, 1], [.8, .35, .95]):
            with self.subTest(p=p):
                result = self.distribution(scores, p, temperature=.8, samples=100000,
                                           smoothing=0, seed=792)
                np.testing.assert_allclose(result, self.exact(scores, p, temperature=.8), atol=.004)
                weights = np.exp(np.asarray(scores)/.8)
                if p == [1, 1, 1]:
                    np.testing.assert_allclose(result[:, 0], weights/weights.sum(), atol=.004)

    def test_equal_strengths_are_symmetric_in_monte_carlo_limit(self):
        np = self.np
        result = self.distribution([0]*5, [.8]*5, temperature=1, samples=50000, seed=92)
        expected = self.occupancy([.8]*5)/5
        np.testing.assert_allclose(result[:, :-1], np.tile(expected, (5, 1)), atol=.004)

    def test_extreme_scores_and_small_temperature_are_finite_and_keep_weak_horse_order(self):
        np = self.np
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            result = self.distribution([-1e308, 1e308, -1e307, 0], [1]*4,
                                       temperature=1e-300, samples=100, smoothing=0)
        np.testing.assert_array_equal(result[:, :-1],
                                      [[0, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]])

    def test_fixed_seed_repeats_and_preserves_inputs_and_global_rng(self):
        np = self.np
        scores, p = np.array([1., 3., -2.]), np.array([.8, .99, .4])
        original_scores, original_p = scores.copy(), p.copy()
        state = np.random.get_state()
        first = self.distribution(scores, p, temperature=.6, samples=513, seed=7)
        second = self.distribution(scores, p, temperature=.6, samples=513, seed=7)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(scores, original_scores)
        np.testing.assert_array_equal(p, original_p)
        after = np.random.get_state()
        self.assertEqual(state[0], after[0])
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])
        self.assertFalse(np.array_equal(first, self.distribution(scores, p, temperature=.6,
                                                               samples=513, seed=8)))

    def test_common_score_shift_and_joint_temperature_scale_preserve_draws(self):
        np = self.np
        p = [.5, .9, 1]
        original = self.distribution([2, 3, 4], p, temperature=.5, samples=97)
        shifted = self.distribution([1000002, 1000003, 1000004], p, temperature=.5, samples=97)
        scaled = self.distribution([20, 30, 40], p, temperature=5, samples=97)
        np.testing.assert_array_equal(original, shifted)
        np.testing.assert_array_equal(original, scaled)

    def test_bad_arrays_are_rejected(self):
        from keiba_ai.ranking import uniform_rank_distribution
        bad = [([], []), ([1], []), ([1, 2], [.5]), ([[1]], [[.5]]),
               ([float('nan')], [.5]), ([float('inf')], [.5]), ([0], [float('nan')]),
               ([0], [float('inf')]), ([0], [-.1]), ([0], [1.1]),
               ([complex(1, 2)], [.5]), (['1'], [.5]), ([None], [.5])]
        for scores, p in bad:
            with self.subTest(scores=scores, p=p), self.assertRaises(ValueError):
                self.distribution(scores, p, temperature=1)
        for p in ([], [[.5]], [float('nan')], [float('inf')], [-.1], [1.1], ['.5']):
            with self.subTest(p=p), self.assertRaises(ValueError):
                uniform_rank_distribution(p)

    def test_bad_controls_are_rejected(self):
        invalid = dict(temperature=[0, -1, float('nan'), float('inf'), True, '1', 10**400],
                       smoothing=[-.01, 1.01, float('nan'), float('inf'), True, '0.1', 10**400],
                       samples=[0, -1, 1.5, True, '1'],
                       seed=[-1, 2.5, True, '1'])
        for name, values in invalid.items():
            for value in values:
                arguments = dict(temperature=1)
                arguments[name] = value
                with self.subTest(name=name, value=value), self.assertRaises(ValueError):
                    self.distribution([0], [.5], **arguments)


if __name__ == '__main__':
    unittest.main()
