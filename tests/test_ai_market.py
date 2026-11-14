import unittest

import numpy as np


def synthetic(races=2400, runners=10, seed=0, curve=0.0, late_market=None, exp_effect=0.0):
    """1st-2nd-3rd from a*model + b*market (+ curve*market^2); 2nd-3rd may weigh the market differently;
    with ``exp_effect`` the market matters more for runners with little experience."""
    rng = np.random.default_rng(seed)
    truth = rng.normal(size=(races, runners))
    market_strength = truth + rng.normal(scale=.6, size=(races, runners))
    market = market_strength - np.log(np.exp(market_strength).sum(axis=1, keepdims=True))
    model = truth + rng.normal(scale=.6, size=(races, runners))
    experience = rng.uniform(0, 3, size=(races, runners))
    exp_z = (experience - experience.mean()) / experience.std()
    b = .8 - exp_effect * exp_z
    first = .4 * model + b * market + curve * market ** 2
    later = first if late_market is None else .4 * model + late_market * market
    order = np.zeros((races, 3), dtype=int)
    for race in range(races):
        left = list(range(runners))
        for stage in range(3):
            u = (first if stage == 0 else later)[race, left]
            p = np.exp(u - u.max())
            p /= p.sum()
            pick = left[rng.choice(len(left), p=p)]
            order[race, stage] = pick
            left.remove(pick)
    odds = 1 / (np.exp(market) * 1.25)
    data = dict(race_ids=[str(i) for i in range(races)], months=np.tile(np.arange(1, 13), races // 12 + 1)[:races],
                present=np.ones((races, runners), dtype=bool), model=model, market=market, odds=odds,
                spread=np.zeros((races, runners)), place=order, temperatures=[1.0], experience=experience)
    from keiba_ai.experiment_market import with_terms
    return with_terms(data)


class MarketBlendExperimentTests(unittest.TestCase):
    def test_blend_recovers_weights_and_stays_recommended_without_structure(self):
        from keiba_ai.experiment_market import evaluate_group, fit
        data = synthetic()
        params, _, _ = fit(data, 'blend')
        self.assertAlmostEqual(params['all']['s'], .4, delta=.12)
        self.assertAlmostEqual(params['all']['m'], .8, delta=.15)
        result = evaluate_group(None, None, 'x', data=data)
        self.assertEqual(result['recommended']['variant'], 'blend')
        self.assertEqual(set(result['recommended']['standardization']), {'field_mean', 'field_sd', 'exp_mean', 'exp_sd'})

    def test_detects_separate_weights_for_second_and_third(self):
        from keiba_ai.experiment_market import evaluate_group
        result = evaluate_group(None, None, 'x', data=synthetic(late_market=0.0, seed=1))
        self.assertIn(result['recommended']['variant'], ('stages', 'all'))
        stages = result['full_year']['stages']['params']
        self.assertGreater(stages['1st']['m'], stages['2nd_3rd']['m'] + .4)

    def test_experience_interaction_has_the_right_sign(self):
        from keiba_ai.experiment_market import fit
        params, _, _ = fit(synthetic(exp_effect=.4, seed=2), 'conditions')
        self.assertLess(params['all']['m_exp'], -.2)


if __name__ == '__main__':
    unittest.main()
