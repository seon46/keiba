import unittest

import numpy as np


def synthetic(races=1500, runners=10, a=.8, b=.5, s0=.4, seed=0):
    """Races whose 1st-2nd-3rd follow a*model + b*market + noise (single temperature)."""
    rng = np.random.default_rng(seed)
    model = rng.normal(size=(races, runners))
    truth = rng.normal(size=(races, runners))
    market_strength = truth + rng.normal(scale=.6, size=(races, runners))
    market = market_strength - np.log(np.exp(market_strength).sum(axis=1, keepdims=True))
    model_score = truth + rng.normal(scale=.6, size=(races, runners))
    u = a * model_score + b * market + rng.normal(scale=s0, size=(races, runners))
    gumbel = -np.log(-np.log(rng.uniform(size=(races, runners))))
    order = np.argsort(-(u + gumbel), axis=1)
    win = np.exp(market)
    odds = 1 / (win * 1.25)  # a 20% take
    return dict(race_ids=[str(i) for i in range(races)], months=np.tile(np.arange(1, 13), races // 12 + 1)[:races],
                present=np.ones((races, runners), dtype=bool), model=model_score, market=market,
                spread=np.full((races, runners), .1), odds=odds, place=order[:, :3], temperatures=[1.0])


class UncertaintyFitTests(unittest.TestCase):
    def test_recovers_the_blend_and_prefers_it(self):
        from keiba_ai.experiment_uncertainty import VARIANTS, fit
        data = synthetic()
        params, full = fit(data)
        self.assertAlmostEqual(params['a'], .8, delta=.2)
        self.assertAlmostEqual(params['b'], .5, delta=.2)
        _, model_only = fit(data, fix=VARIANTS['model'])
        self.assertGreater(full, model_only)

    def test_kelly_backtest_and_predictive_probabilities(self):
        from keiba_ai.experiment_uncertainty import kelly_backtest, win_probabilities
        data = synthetic(races=600)
        p = win_probabilities(data, [.8, .5, np.log(.4), np.log(1.0)])
        self.assertTrue(np.allclose(p.sum(axis=1), 1))
        flat = win_probabilities(data, [.8, .5, np.log(3.0), np.log(1.0)])
        self.assertLess(flat.max(axis=1).mean(), p.max(axis=1).mean())  # more uncertainty, flatter
        result = kelly_backtest(data, p)
        self.assertEqual(result['races'], 600)
        self.assertLessEqual(result['bets'], 600)


if __name__ == '__main__':
    unittest.main()
