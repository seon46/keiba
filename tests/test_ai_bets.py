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
        growths = [item['growth'] for item in proposals]
        self.assertEqual(growths, sorted(growths, reverse=True))
        # odds 25% below fair: no set grows the bankroll, so each keeps its smallest stake
        self.assertTrue(all(item['growth'] < 0 for item in proposals))
        self.assertTrue(all(item['cost'] == 100 * item['points'] for item in proposals))
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

    def test_kelly_stake_and_long_shot_parts(self):
        from keiba_ai.bets import combine, propose
        # odds 40% above fair: betting grows the bankroll, and the stake grows with the bankroll
        odds = {bet: {key: round(1.4 / p, 1) for key, p in self.bets.probabilities(bet).items() if p > .002}
                for bet in ('win', 'quinella', 'exacta', 'trio', 'trifecta')}
        priced = lambda items: [item for item in items if item['priced']]
        small = priced(propose(self.bets, odds, min_hit=.2, budget=10000, bankroll=20000))
        large = priced(propose(self.bets, odds, min_hit=.2, budget=10000, bankroll=1000000))
        self.assertTrue(small and all(item['growth'] > 0 for item in small))
        self.assertLess(max(item['cost'] for item in small), 10000)  # Kelly keeps it below the cap
        by_bet = {item['bet']: item['cost'] for item in small}
        self.assertTrue(any(item['cost'] >= by_bet.get(item['bet'], 0) for item in large))
        self.assertTrue(all(item['min_payout'] >= item['cost'] for item in small + large))
        # a mix may contain a part that alone hits rarely (3連単 / 馬単 sets)
        mixes = combine(self.bets, odds, min_hit=.3, max_points=10, budget=5000, bankroll=50000)
        self.assertTrue(mixes)
        self.assertEqual([mix['growth'] for mix in mixes], sorted((mix['growth'] for mix in mixes), reverse=True))
        rare = [part for mix in mixes for part in mix['parts']
                if self.bets.hit_rate(part['bet'], [ticket['key'] for ticket in part['tickets']]) < .3]
        self.assertTrue(rare)
        for mix in mixes:
            self.assertGreaterEqual(mix['hit_rate'], .3)
            self.assertGreaterEqual(mix['min_payout'], mix['cost'])
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



class MarketBlendTests(unittest.TestCase):
    def test_blend_uses_relative_temperatures_and_needs_every_runners_odds(self):
        from keiba_ai.bets import MARKET_BLEND, blend_with_market
        scores, temperatures = [1.0, 0.0, -1.0], [2.0, 3.0, 4.0]
        blended, stages = blend_with_market('local', scores, temperatures, [2.0, 4.0, [8.0, 9.0]])
        inverse = np.asarray([1 / 2, 1 / 4, 1 / 8])
        expected = MARKET_BLEND['local']['a'] * np.asarray(scores) + MARKET_BLEND['local']['b'] * np.log(inverse / inverse.sum())
        np.testing.assert_allclose(blended, expected)
        self.assertEqual(stages, [1.0, 1.5, 2.0])
        self.assertIsNone(blend_with_market('central', scores, temperatures, [2.0, None, 5.0]))


if __name__ == '__main__':
    unittest.main()
