"""The saved-state queries must reproduce exactly what a full re-run gives for a new day's rows."""
from datetime import date, timedelta
import unittest

import numpy as np


def _history(seed=0, days=40, races_per_day=4, runners=8, horses=60, jockeys=12):
    rng = np.random.default_rng(seed)
    rows = dict(days=[], race_ids=[], horses=[], jockeys=[], positions=[], finished=[])
    start = date(2024, 1, 6)
    for d in range(days):
        day = (start + timedelta(days=3 * d)).isoformat()
        for r in range(races_per_day):
            field = rng.choice(horses, runners, replace=False)
            order = rng.permutation(runners)
            for slot, horse in enumerate(field):
                rows['days'].append(day)
                rows['race_ids'].append(f'{day}-{r}')
                rows['horses'].append(f'h{horse}')
                rows['jockeys'].append('' if rng.random() < .05 else f'j{rng.integers(jockeys)}')
                done = rng.random() > .05
                rows['finished'].append(done)
                rows['positions'].append(int(order[slot]) + 1)
    out = {key: np.asarray(value, dtype=object if key in ('horses', 'jockeys') else None) for key, value in rows.items()}
    # finish positions contiguous among finishers per race
    for race in np.unique(out['race_ids']):
        index = np.flatnonzero(out['race_ids'] == race)
        done = index[out['finished'][index]]
        ranks = np.argsort(np.argsort(out['positions'][done])) + 1
        out['positions'][index] = 0
        out['positions'][done] = ranks
    out['strength'] = rng.normal(size=len(out['days']))
    return out


class ResumableBlockTests(unittest.TestCase):
    def setUp(self):
        self.data = _history()
        days = self.data['days']
        self.last_day = max(days.tolist())
        self.history = days < self.last_day
        self.new = ~self.history

    def _split(self, key):
        return self.data[key][self.history], self.data[key][self.new]

    def test_jockey_timeline(self):
        from keiba_ai.experiment_v3d import jockey_timeline, timeline_query
        d = self.data
        full = jockey_timeline(d['days'], d['race_ids'], d['jockeys'], d['horses'], d['finished'], d['positions'],
                               d['strength'])
        h = self.history
        _, state = jockey_timeline(d['days'][h], d['race_ids'][h], d['jockeys'][h], d['horses'][h], d['finished'][h],
                                   d['positions'][h], d['strength'][h], return_state=True)
        queried = timeline_query(state, d['jockeys'][self.new], d['horses'][self.new], self.last_day)
        for key, values in queried.items():
            np.testing.assert_allclose(values, full[key][self.new], equal_nan=True, err_msg=key)

    def test_rating_block(self):
        from keiba_ai.ratings import rating_block, rating_query
        d = self.data
        full = rating_block(d['days'], d['race_ids'], d['horses'], d['finished'], d['positions'])
        h = self.history
        _, state = rating_block(d['days'][h], d['race_ids'][h], d['horses'][h], d['finished'][h], d['positions'][h],
                                return_state=True)
        queried = rating_query(state, d['race_ids'][self.new], d['horses'][self.new], self.last_day)
        np.testing.assert_allclose(queried, full[self.new], equal_nan=True, rtol=1e-6, atol=1e-6)

    def test_pedigree_block(self):
        from keiba_ai.pedigree import PATHS, _pedigree_block_reference, pedigree_block, pedigree_query
        d = self.data
        rng = np.random.default_rng(3)
        pedigrees = {}
        for horse in set(d['horses'].tolist()):
            if rng.random() < .9:
                pedigrees[horse] = tuple(None if rng.random() < .1 else f'a{rng.integers(30)}' for _ in PATHS)
        surfaces = rng.integers(0, 2, len(d['days'])).astype(float)
        distances = rng.choice([1200., 1600., 2000., 2400.], len(d['days']))
        args = (d['days'], d['race_ids'], d['horses'], d['finished'], d['positions'], d['strength'], surfaces, distances,
                pedigrees)
        full = pedigree_block(*args)
        np.testing.assert_allclose(full, _pedigree_block_reference(*args), equal_nan=True)
        h = self.history
        _, state = pedigree_block(*(value[h] if isinstance(value, np.ndarray) else value for value in args),
                                  return_state=True)
        queried = pedigree_query(state, d['race_ids'][self.new], d['horses'][self.new], surfaces[self.new],
                                 distances[self.new], pedigrees, self.last_day)
        np.testing.assert_allclose(queried, full[self.new], equal_nan=True, rtol=1e-6, atol=1e-6)

    def test_speed_figures(self):
        from keiba_ai.speed import SpeedFigures, speed_feature_rows
        d = self.data
        rng = np.random.default_rng(5)
        races = []
        for race in dict.fromkeys(d['race_ids'].tolist()):
            index = np.flatnonzero(d['race_ids'] == race)
            races.append([dict(day=d['days'][i], race_id=race, horse_id=d['horses'][i], course='東京', surface='芝',
                               distance_m=1600, finish_status='finished' if d['finished'][i] else 'did_not_finish',
                               finish_position=int(d['positions'][i]) or None,
                               time_seconds=95 + d['positions'][i] * .2 + rng.normal() * .3 if d['finished'][i] else None,
                               race_class='1勝', track_condition='良', carried_weight_kg=56) for i in index])
        full, _ = speed_feature_rows(races)
        history = [race for race in races if race[0]['day'] < self.last_day]
        new = [race for race in races if race[0]['day'] == self.last_day]
        figures = SpeedFigures()
        figures.feed(history)
        figures.finish()
        for race in new:
            for row, vector in zip(race, figures.vectors(race)):
                self.assertEqual(vector, full[(row['race_id'], row['horse_id'])])


if __name__ == '__main__':
    unittest.main()
