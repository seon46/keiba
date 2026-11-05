import unittest

import numpy as np


class BetTests(unittest.TestCase):
    def setUp(self):
        from keiba_ai.bets import RaceBets, simulate_top3
        rng = np.random.default_rng(0)
        self.count = 12
        scores = rng.normal(size=self.count) * 1.2
        top3 = simulate_top3(scores, np.full(self.count, .98), [1.0, 1.2, 1.4], samples=20000)
        self.bets = RaceBets(range(1, self.count + 1), [(i // 2) + 1 for i in range(self.count)], top3)

    def test_probabilities_sum_and_rules(self):
        for bet in ('win', 'quinella', 'exacta', 'trio', 'trifecta', 'bracket'):
            self.assertAlmostEqual(sum(self.bets.probabilities(bet).values()), 1.0, delta=.03, msg=bet)
        self.assertAlmostEqual(sum(self.bets.probabilities('place').values()), 3.0, delta=.08)  # 3 places paid
        self.assertAlmostEqual(sum(self.bets.probabilities('wide').values()), 3.0, delta=.08)

    def test_allocation_pays_the_same_whichever_hits(self):
        from keiba_ai.bets import allocate
        stakes, payout, total = allocate([1.6, 4.7], 1000)
        self.assertEqual(stakes, [700, 300])
        self.assertLessEqual(total, 1000)
        self.assertGreaterEqual(payout, total)
        self.assertIsNone(allocate([2.0] * 6, 500))  # budget below one unit per ticket

    def test_proposals_never_lose_when_they_hit(self):
        from keiba_ai.bets import propose
        odds = {}
        for bet in ('win', 'place', 'quinella', 'wide', 'exacta', 'trio', 'trifecta'):
            table = {}
            for key, p in self.bets.probabilities(bet).items():
                value = round(max(1.1, .75 / p), 1)
                table[key] = [value, value * 1.3] if bet in ('place', 'wide') else value
            odds[bet] = table
        proposals = propose(self.bets, odds, min_hit=.2, max_points=10, budget=1000)
        self.assertTrue(proposals)
        self.assertEqual([item['bet'] for item in proposals if not item['priced']], ['bracket'])  # no 枠連 odds given
        proposals = [item for item in proposals if item['priced']]
        for item in proposals:
            self.assertLessEqual(item['cost'], 1000)
            self.assertGreaterEqual(item['min_payout'], item['cost'])
            self.assertEqual(sum(ticket['stake'] for ticket in item['tickets']), item['cost'])
            self.assertGreaterEqual(item['hit_rate'], .2)
        returns = [item['expected_return'] for item in proposals]
        self.assertEqual(returns, sorted(returns, reverse=True))
        # without odds: 100 yen each, by hit rate
        plain = propose(self.bets, None, min_hit=.2)
        self.assertTrue(all(ticket['stake'] == 100 for item in plain for ticket in item['tickets']))


if __name__ == '__main__':
    unittest.main()
