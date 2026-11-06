"""Causal multi-player horse ratings and opponent-adjusted form (design 6.2 B).

A horse rating is the same day-by-day Gaussian Plackett-Luce update used for
jockeys (``experiment_v3d.jockey_timeline``) with the horse as the entity and
no offset.  Every row receives its rating *before* its own day.

From the pre-race ratings of each race:

* field level  = mean rating of the opponents (and of the best three of them)
* surprise     = (expected rank - actual rank) / (n - 1), expected rank from
                 pairwise PL probabilities ``sigmoid(r_j - r_i)``; positive
                 means the horse beat what its opposition implied.

A run's field level and surprise enter the horse's history only after its day,
so features built from them see earlier days only.
"""
from __future__ import annotations

from collections import defaultdict, deque

RATING_FEATURES = (
    'horse_rating', 'horse_rating_sd', 'horse_rating_recent', 'horse_rating_race_z',
    'horse_rating_minus_field_max', 'field_rating_top3_mean',
    'horse_recent5_surprise_mean', 'horse_previous_surprise',
    'horse_recent5_field_rating_mean', 'horse_previous_field_rating', 'field_level_change',
)


def rating_block(days, race_ids, horses, finished, positions, *, half_life=240.0, short_half_life=60.0,
                 prior_sd=1.0, return_state=False):
    import numpy as np
    from .experiment_v3d import jockey_timeline
    count = len(days)
    zeros = np.zeros(count)
    timeline, timeline_state = jockey_timeline(days, race_ids, horses, horses, finished, positions, zeros,
                                               half_life=half_life, short_half_life=short_half_life,
                                               prior_sd=prior_sd, return_state=True)
    rating, sd, recent = timeline['theta'], timeline['theta_sd'], timeline['theta_recent']
    out = np.full((count, len(RATING_FEATURES)), np.nan, dtype=np.float32)
    out[:, 0], out[:, 1], out[:, 2] = rating, sd, recent
    field_mean = np.full(count, np.nan)
    field_top3 = np.full(count, np.nan)
    surprise = np.full(count, np.nan)
    boundaries = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1], True]) if count else np.asarray([0])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        r = rating[start:end]
        n = end - start
        if n < 2:
            continue
        spread = r.std()
        out[start:end, 3] = (r - r.mean()) / spread if spread > 0 else 0.0
        for local in range(n):
            others = np.delete(r, local)
            out[start + local, 4] = r[local] - others.max()
            best = np.sort(others)[::-1][:3]
            field_top3[start + local] = best.mean()
            field_mean[start + local] = others.mean()
        out[start:end, 5] = np.sort(r)[::-1][:3].mean()
        done = finished[start:end]
        m = int(done.sum())
        if m >= 2:
            rr = r[done]
            beats = 1.0 / (1.0 + np.exp(-(rr[None, :] - rr[:, None])))  # P(j ahead of i)
            np.fill_diagonal(beats, 0.0)
            expected = 1.0 + beats.sum(axis=1)
            actual = positions[start:end][done].astype(float)
            # Ranks among finishers are 1..m (ties excluded upstream).
            surprise[np.flatnonzero(done) + start] = (expected - actual) / (m - 1)
    history = defaultdict(lambda: deque(maxlen=5))
    day_boundaries = np.flatnonzero(np.r_[True, days[1:] != days[:-1], True]) if count else np.asarray([0])
    for start, end in zip(day_boundaries[:-1], day_boundaries[1:]):
        for row in range(start, end):
            past = history.get(horses[row])
            if past:
                surprises = [item[0] for item in past if np.isfinite(item[0])]
                fields = [item[1] for item in past if np.isfinite(item[1])]
                out[row, 6] = np.mean(surprises) if surprises else np.nan
                out[row, 7] = past[-1][0]
                out[row, 8] = np.mean(fields) if fields else np.nan
                out[row, 9] = past[-1][1]
                if fields and np.isfinite(field_mean[row]):
                    out[row, 10] = field_mean[row] - np.mean(fields)
        for row in range(start, end):
            history[horses[row]].append((surprise[row], field_mean[row]))
    if return_state:
        return out, dict(timeline=timeline_state, history=history)
    return out


def _race_columns(out, field_mean, start, end, r):
    import numpy as np
    n = end - start
    if n < 2:
        return
    spread = r.std()
    out[start:end, 3] = (r - r.mean()) / spread if spread > 0 else 0.0
    for local in range(n):
        others = np.delete(r, local)
        out[start + local, 4] = r[local] - others.max()
        field_mean[start + local] = others.mean()
    out[start:end, 5] = np.sort(r)[::-1][:3].mean()


def rating_query(saved, race_ids, horses, day):
    """RATING_FEATURES for new rows of ``day`` (races contiguous) from ``rating_block(..., return_state=True)``."""
    import numpy as np
    from .experiment_v3d import timeline_query
    count = len(race_ids)
    queried = timeline_query(saved['timeline'], horses, horses, day)
    out = np.full((count, len(RATING_FEATURES)), np.nan, dtype=np.float32)
    rating = queried['theta']
    out[:, 0], out[:, 1], out[:, 2] = rating, queried['theta_sd'], queried['theta_recent']
    field_mean = np.full(count, np.nan)
    boundaries = np.flatnonzero(np.r_[True, race_ids[1:] != race_ids[:-1], True]) if count else np.asarray([0])
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        _race_columns(out, field_mean, start, end, rating[start:end])
    for row in range(count):
        past = saved['history'].get(horses[row])
        if past:
            surprises = [item[0] for item in past if np.isfinite(item[0])]
            fields = [item[1] for item in past if np.isfinite(item[1])]
            out[row, 6] = np.mean(surprises) if surprises else np.nan
            out[row, 7] = past[-1][0]
            out[row, 8] = np.mean(fields) if fields else np.nan
            out[row, 9] = past[-1][1]
            if fields and np.isfinite(field_mean[row]):
                out[row, 10] = field_mean[row] - np.mean(fields)
    return out
