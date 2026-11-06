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
        from keiba_ai.bets import combine
        mixes = combine(self.bets, odds, min_hit=.2, max_points=10, budget=1000)
        self.assertTrue(mixes)
        for mix in mixes:
            self.assertGreaterEqual(len({part['bet'] for part in mix['parts']}), 2)
            self.assertLessEqual(mix['cost'], 1000)
            self.assertLessEqual(mix['points'], 10)
            self.assertGreaterEqual(mix['min_payout'], mix['cost'])
            self.assertGreaterEqual(mix['hit_rate'], .2)
        self.assertEqual(combine(self.bets, None), [])
        # without odds: 100 yen each, by hit rate
        plain = propose(self.bets, None, min_hit=.2)
        self.assertTrue(all(ticket['stake'] == 100 for item in plain for ticket in item['tickets']))



class ConfidenceSummaryTests(unittest.TestCase):
    def test_tiers_and_roi(self):
        from keiba_ai.experiment_confidence import summarize
        rng = np.random.default_rng(1)
        table = []
        for i in range(300):
            top = float(rng.uniform(.1, .6))
            won = float(rng.random() < top)
            entry = dict(month=f'2025-{i % 12 + 1:02d}', pick_win=won, pick_top3=float(won or rng.random() < .5),
                         log_loss=2.0 - top, top_win=top, pick_cost=1.0, pick_return=won * 2.5, favourite_cost=1.0,
                         favourite_return=0.0)
            for threshold in (1.0, 1.2, 1.5):
                entry[f'value{threshold}_cost'] = 2.0
                entry[f'value{threshold}_return'] = won * 3.0
            table.append(entry)
        result = summarize(table, ('top_win',))
        tiers = result['candidates']['top_win']['tiers']
        self.assertEqual([tier['tier'] for tier in tiers], ['A', 'B', 'C'])
        self.assertGreater(tiers[0]['pick_win'], tiers[2]['pick_win'])  # confident tier wins more
        self.assertGreater(tiers[0]['roi']['pick']['roi'], tiers[2]['roi']['pick']['roi'])
        roi = result['overall']['roi']['pick']
        self.assertAlmostEqual(roi['roi'], sum(e['pick_return'] for e in table) / 300, places=3)
        self.assertLessEqual(roi['ci'][0], roi['roi'])
        self.assertGreaterEqual(roi['ci'][1], roi['roi'])


if __name__ == '__main__':
    unittest.main()
