"""Track bias measured against horse strength (design chapter 5, Plackett-Luce form).

For race k, with strength offsets s_i and position vector
u_i = (0.5 - x_i, 0.5 - z_i)   (x: actual last-corner fraction, z: gate position),
beta_k = (front, inner) maximizes the PL likelihood of the finishing order under
scores s_i + beta . u_i (Newton steps with a small ridge).  Positive front means
horses near the front did better than their strength implied.

* Prior mu, tau^2 per (course, surface, 200 m band): running moments of
  beta over earlier days only.
* Input-side day deviation for race r: precision-weighted shrinkage of
  beta_k - mu over *earlier-numbered* races of the same day, course and surface.
* Label-side day deviation (training offsets only): the same over *every other*
  race of that day, course and surface (the race itself is excluded).
"""
from __future__ import annotations

from collections import defaultdict
import json
import math

BIAS_V3_FEATURES = (
    'bias3_day_front_dev', 'bias3_day_inner_dev', 'bias3_day_races',
    'bias3_prior_front', 'bias3_prior_inner',
    'bias3_front_fit', 'bias3_inner_fit', 'bias3_prior_front_fit', 'bias3_prior_inner_fit',
)


def load_geometry(database, group, identities):
    """Per row (aligned to identities): course, surface, distance, actual last-corner fraction, gate."""
    import numpy as np
    from itertools import groupby
    from .bias import gate_position
    from .features import _corner_form
    from .model import readonly
    wanted = {(race, horse): index for index, (race, horse, _) in enumerate(identities)}
    count = len(identities)
    course = np.full(count, '', dtype=object)
    surface = np.full(count, '', dtype=object)
    distance = np.full(count, np.nan)
    corner = np.full(count, np.nan)
    gate = np.full(count, np.nan)
    connection = readonly(database)
    try:
        cursor = connection.execute(
            'SELECT f.race_id,f.horse_id,p.payload_json FROM features f JOIN input_rows p ON p.region=f.region '
            'AND p.race_id=f.race_id AND p.horse_id=f.horse_id WHERE f.group_name=? ORDER BY f.day,f.region,f.race_id,f.horse_number',
            (group,))
        for race_id, rows in groupby(cursor, key=lambda row: row[0]):
            race = [(horse, json.loads(payload)) for _, horse, payload in rows]
            payloads = [payload for _, payload in race]
            size = len(payloads)
            for horse, payload in race:
                index = wanted.get((race_id, horse))
                if index is None:
                    continue
                course[index] = payload.get('course') or ''
                surface[index] = payload.get('surface') or ''
                value = payload.get('distance_m')
                distance[index] = float(value) if value else np.nan
                position = payload.get('finish_position') if payload.get('finish_status') == 'finished' else None
                fraction = _corner_form(payload, size, position)['last_corner_fraction']
                corner[index] = np.nan if fraction is None else fraction
                position_of_gate = gate_position(payload, payloads)
                gate[index] = np.nan if position_of_gate is None else position_of_gate
    finally:
        connection.close()
    return dict(course=course, surface=surface, distance=distance, corner=corner, gate=gate)


def _race_beta(scores, u, ridge=.1, steps=4):
    """scores, u ordered by finishing position (winner first)."""
    import numpy as np
    beta = np.zeros(2)
    m = len(scores)
    for _ in range(steps):
        v = scores + u @ beta
        v = v - v.max()
        e = np.exp(v)
        tail = np.cumsum(e[::-1])[::-1]
        gradient = ridge * beta
        hessian = ridge * np.eye(2)
        for j in range(m - 1):
            p = e[j:] / tail[j]
            mean = p @ u[j:]
            gradient += -u[j] + mean
            centred = u[j:] - mean
            hessian += (centred * p[:, None]).T @ centred
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            return None, None
        beta -= step
        if np.abs(step).max() < 1e-5:
            break
    try:
        covariance = np.linalg.inv(hessian)
    except np.linalg.LinAlgError:
        return None, None
    return beta, np.diag(covariance)


def race_betas(race_ids, finished, positions, corner, gate, strength, *, minimum=6, coverage=.7):
    """beta and its variance per race (dict race_id -> (beta, var)); rows must keep races contiguous."""
    import numpy as np
    out = {}
    boundaries = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1], True])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        rows = np.arange(start, end)
        done = rows[finished[rows]]
        if len(done) < minimum:
            continue
        known = done[np.isfinite(corner[done]) & np.isfinite(gate[done]) & np.isfinite(strength[done])]
        if len(known) < max(minimum, math.ceil(coverage * len(done))):
            continue
        if np.var(corner[known]) < .005 or np.var(gate[known]) < .005:
            continue
        order = known[np.argsort(positions[known], kind='stable')]
        u = np.column_stack([.5 - corner[order], .5 - gate[order]])
        beta, variance = _race_beta(strength[order], u)
        if beta is not None and np.all(np.isfinite(beta)) and np.all(variance > 0):
            out[race_ids[start]] = (beta, variance)
    return out


class _Moments:
    __slots__ = ('n', 'mean', 'm2', 'variance_mean')

    def __init__(self):
        self.n, self.mean, self.m2, self.variance_mean = 0, None, None, None

    def add(self, beta, variance):
        import numpy as np
        self.n += 1
        if self.mean is None:
            self.mean, self.m2, self.variance_mean = beta.copy(), np.zeros(2), variance.copy()
            return
        delta = beta - self.mean
        self.mean = self.mean + delta / self.n
        self.m2 = self.m2 + delta * (beta - self.mean)
        self.variance_mean = self.variance_mean + (variance - self.variance_mean) / self.n

    def prior(self):
        """(mu, tau^2): between-race variance net of estimation noise."""
        import numpy as np
        if self.n < 2:
            return np.zeros(2), np.full(2, .05)
        total = self.m2 / (self.n - 1)
        return self.mean, np.maximum(total - self.variance_mean, 1e-4)


def day_features(days, race_ids, race_numbers, course, surface, distance, expected_corner, gate, betas, *,
                 decay=.8, strength_scale=1.0):
    """Input-side features (earlier races only) and label-side deviations (all other races of the day).

    Returns (features array [rows x BIAS_V3_FEATURES], label_dev array [rows x 2]).
    """
    import numpy as np
    count = len(days)
    features = np.full((count, len(BIAS_V3_FEATURES)), np.nan, dtype=np.float32)
    label_dev = np.zeros((count, 2))
    moments = defaultdict(_Moments)
    # One record per race: (row slice, day, track, band, number, distance)
    boundaries = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1], True])
    races = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        d = distance[start]
        band = int(d // 200) if np.isfinite(d) else -1
        races.append((start, end, days[start], (course[start], surface[start]), band, race_numbers[start], d,
                      race_ids[start]))
    by_day = defaultdict(list)
    for race in races:
        by_day[race[2]].append(race)
    for day in sorted(by_day):
        day_races = by_day[day]
        priors = {}
        for start, end, _, track, band, number, d, race_id in day_races:
            key = (*track, band)
            if key not in priors:
                priors[key] = moments[key].prior()
        measured = defaultdict(list)  # track -> list of (number, distance, deviation, variance)
        for start, end, _, track, band, number, d, race_id in day_races:
            if race_id in betas:
                beta, variance = betas[race_id]
                mu, _ = priors[(*track, band)]
                measured[track].append((number, d, race_id, beta - mu, variance))
        for start, end, _, track, band, number, d, race_id in day_races:
            mu, tau2 = priors[(*track, band)]
            earlier = [item for item in measured[track]
                       if np.isfinite(number) and np.isfinite(item[0]) and item[0] < number]
            others = [item for item in measured[track] if item[2] != race_id]

            def shrink(items, use_decay):
                numerator, denominator = np.zeros(2), 1.0 / tau2
                for item_number, item_distance, _, deviation, variance in items:
                    weight = math.exp(-abs(d - item_distance) / 400) if np.isfinite(d) and np.isfinite(item_distance) else .5
                    if use_decay:
                        weight *= decay ** (number - item_number)
                    elif np.isfinite(number) and np.isfinite(item_number):
                        weight *= decay ** abs(number - item_number)
                    numerator += weight * deviation / variance
                    denominator += weight / variance
                return numerator / denominator

            dev = shrink(earlier, True) if earlier else np.zeros(2)
            label_dev[start:end] = shrink(others, False) if others else 0.0
            x_hat = expected_corner[start:end]
            z = gate[start:end]
            features[start:end, 0] = dev[0]
            features[start:end, 1] = dev[1]
            features[start:end, 2] = len(earlier)
            features[start:end, 3] = mu[0]
            features[start:end, 4] = mu[1]
            features[start:end, 5] = dev[0] * (.5 - x_hat)
            features[start:end, 6] = dev[1] * (.5 - z)
            features[start:end, 7] = mu[0] * (.5 - x_hat)
            features[start:end, 8] = mu[1] * (.5 - z)
        for start, end, _, track, band, number, d, race_id in day_races:
            if race_id in betas:
                beta, variance = betas[race_id]
                moments[(*track, band)].add(beta, variance)
    return features, label_dev
