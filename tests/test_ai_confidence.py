import unittest

import numpy as np

from keiba_ai.experiment_confidence import order_measures, top3_orders
from keiba_ai.ranking_v3 import _staged_orders, stage_temperatures


class TopThreeOrderTests(unittest.TestCase):
    def test_matches_the_sampled_rank_distribution(self):
        scores = np.random.default_rng(0).normal(size=9)
        temperatures = [.7, .76, .92, 1.05]
        exact = top3_orders(scores, temperatures)
        self.assertAlmostEqual(exact.sum(), 1.0, places=12)
        orders = _staged_orders(scores - scores.max(), stage_temperatures(temperatures, 9),
                                np.random.default_rng(1), 200000)
        sampled = np.zeros_like(exact)
        np.add.at(sampled, (orders[:, 0], orders[:, 1], orders[:, 2]), 1)
        self.assertLess(np.abs(sampled / len(orders) - exact).max(), .002)

    def test_skills_and_certainty(self):
        positions = np.asarray([3, 1, 2, 4, 5, 6])
        # equal chances: no certainty and no skill
        self.assertEqual([round(value, 9) for value in order_measures(np.zeros(6), [1.0], positions)], [0, 0, 0, 0])
        # a confident and right prediction scores well on both
        certainty, trio_certainty, skill, trio_skill = order_measures(np.asarray([4., 6., 5., 0., 0., 0.]), [1.0],
                                                                      positions)
        self.assertGreater(certainty, .3)
        self.assertGreater(skill, certainty)
        self.assertGreater(trio_skill, .7)
        # the same confidence with the wrong horses gives a negative skill
        wrong = order_measures(np.asarray([4., 6., 5., 0., 0., 0.]), [1.0], np.asarray([4, 5, 6, 1, 2, 3]))
        self.assertLess(wrong[2], 0)
        # a dead heat for a place leaves the skills out
        self.assertEqual(order_measures(np.zeros(6), [1.0], np.asarray([1, 1, 3, 4, 5, 6]))[2:], (None, None))


if __name__ == '__main__':
    unittest.main()
