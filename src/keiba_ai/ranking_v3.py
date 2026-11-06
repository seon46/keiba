"""Stage-wise temperature Plackett-Luce: fitting, sampling and per-race scoring.

A single temperature scales every selection stage equally.  Here stage j
(1st place, 2nd place, ...) uses its own temperature T_j; stages beyond the
last listed temperature reuse it.  With one temperature this is exactly the
ordinary Plackett-Luce law used by ``ranking.rank_distribution``.

Sampling draws the order sequentially (a fresh Gumbel vector per stage), then
integrates independent finishing indicators exactly, as ``ranking`` does.
"""
from __future__ import annotations

import numpy as np

from .ranking import _uniform_order_marginals


def stage_temperatures(temperatures, count):
    temperatures = np.asarray(temperatures, dtype=float)
    if temperatures.ndim != 1 or not len(temperatures) or (temperatures <= 0).any() or not np.isfinite(temperatures).all():
        raise ValueError('temperatures must be a nonempty vector of positive numbers.')
    index = np.minimum(np.arange(count), len(temperatures) - 1)
    return temperatures[index]


# --------------------------------------------------------------------------
# Fitting on observed finishing orders
# --------------------------------------------------------------------------

def staged_pl_loss(structure, scores, temperatures):
    """Mean per-race weighted PL negative log likelihood with stage temperatures.

    ``structure`` is an ``experiment_v3.PLStructure`` (its stage weights are
    reused, so the loss is comparable with the training objective at T=1).
    """
    scores = np.asarray(scores, dtype=float)
    padded = np.full((structure.races, structure.width), -np.inf)
    padded[structure.code, structure.slot] = scores
    padded = padded - padded.max(axis=1, keepdims=True)
    present = np.isfinite(padded)
    per_stage = stage_temperatures(temperatures, structure.width)
    weights = structure.weights
    active = weights > 0
    total = np.zeros(structure.races)
    for temperature in np.unique(per_stage):
        stages = per_stage == temperature
        scaled = np.where(present, padded / temperature, -np.inf)
        tail = np.logaddexp.accumulate(scaled[:, ::-1], axis=1)[:, ::-1]
        use = active & stages[None, :]
        contribution = np.where(use, scaled - np.where(use, tail, 0.0), 0.0)
        total -= (weights * contribution).sum(axis=1)
    return float(total[structure.counted].mean())


def fit_temperatures(structure, scores, stages=1, *, bounds=(.25, 6.0), shrink=.005):
    """Maximum-likelihood stage temperatures on a held-out period.

    With several stages a small penalty ``shrink * sum((log T_j - mean log T)^2)``
    keeps sparsely informed stages near the common temperature; a held-out
    period of a few hundred races otherwise lets single stages run to a bound.
    """
    from scipy.optimize import minimize

    def objective(logs):
        penalty = shrink * float(((logs - logs.mean()) ** 2).sum()) if stages > 1 else 0.0
        return staged_pl_loss(structure, scores, np.exp(logs)) + penalty

    start = np.zeros(stages)
    limits = [(np.log(bounds[0]), np.log(bounds[1]))] * stages
    result = minimize(objective, start, method='L-BFGS-B', bounds=limits)
    temperatures = np.exp(result.x)
    return [float(value) for value in temperatures], staged_pl_loss(structure, scores, temperatures)


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _staged_orders(scores, per_stage, generator, batch):
    count = len(scores)
    remaining = np.ones((batch, count), dtype=bool)
    orders = np.empty((batch, count), dtype=np.int64)
    rows = np.arange(batch)
    for position in range(count):
        uniforms = np.maximum(generator.random((batch, count)), np.finfo(float).tiny)
        keys = scores[None, :] / per_stage[position] - np.log(-np.log(uniforms))
        keys[~remaining] = -np.inf
        picks = np.argmax(keys, axis=1)
        orders[:, position] = picks
        remaining[rows, picks] = False
    return orders


def staged_rank_distribution(scores, finish_probabilities, temperatures, *, samples=1024,
                             seed=42, smoothing=.001):
    """Return n x (n+1) marginals: ranks 1..n and nonfinish, like ``ranking``."""
    scores = np.asarray(scores, dtype=float)
    probabilities = np.asarray(finish_probabilities, dtype=float)
    if scores.ndim != 1 or scores.shape != probabilities.shape or not len(scores):
        raise ValueError('scores and finish_probabilities must be nonempty vectors of equal length.')
    if not np.isfinite(scores).all() or ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('scores must be finite and probabilities in [0, 1].')
    if samples < 1 or not 0 <= smoothing <= 1:
        raise ValueError('samples must be positive and smoothing in [0, 1].')
    count = len(scores)
    per_stage = stage_temperatures(temperatures, count)
    result = np.zeros((count, count + 1))
    result[:, -1] = 1.0 - probabilities
    uniform = _uniform_order_marginals(probabilities)
    if smoothing == 1.0 or count == 1 or not probabilities.any():
        result[:, :-1] = uniform
        return result
    generator = np.random.default_rng(int(seed))
    rank_mass = np.zeros((count, count))
    remaining = int(samples)
    while remaining:
        batch = min(remaining, 2048)
        orders = _staged_orders(scores - scores.max(), per_stage, generator, batch)
        dp = np.zeros((batch, count + 1))
        dp[:, 0] = 1.0
        for position in range(count):
            horses = orders[:, position]
            finish = probabilities[horses, None]
            previous = dp[:, :position + 1].copy()
            np.add.at(rank_mass[:, :position + 1], horses, previous * finish)
            dp[:, :position + 1] = previous * (1.0 - finish)
            dp[:, 1:position + 2] += previous * finish
        remaining -= batch
    result[:, :-1] = (1.0 - smoothing) * (rank_mass / int(samples)) + smoothing * uniform
    return result


# --------------------------------------------------------------------------
# Per-race scoring
# --------------------------------------------------------------------------

def score_races(race_ids, positions, finished, race_numbers, scores, completion, temperatures, *,
                samples, seed=42):
    """Score contiguous races; returns per-race arrays and aggregate metrics."""
    from .rank_model import race_seed
    race_ids = list(race_ids)
    positions, finished = np.asarray(positions), np.asarray(finished, dtype=bool)
    scores, completion = np.asarray(scores, dtype=float), np.asarray(completion, dtype=float)
    race_numbers = np.asarray(race_numbers, dtype=float)
    boundaries = [0] + [index for index in range(1, len(race_ids)) if race_ids[index] != race_ids[index - 1]] + [len(race_ids)]
    keys, outcome, top3, win, pick, numbers = [], [], [], [], [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        count = end - start
        local = slice(start, end)
        probabilities = staged_rank_distribution(scores[local], completion[local], temperatures,
                                                 samples=samples, seed=race_seed(race_ids[start], seed))
        done = finished[local]
        outcomes = np.where(done, positions[local] - 1, count)
        truth = probabilities[np.arange(count), outcomes]
        outcome.append(float(-np.log(np.clip(truth, 1e-12, 1)).mean()))
        p3 = np.clip(probabilities[:, :min(3, count)].sum(axis=1), 1e-7, 1 - 1e-7)
        y3 = done & (positions[local] <= 3)
        top3.append(float(-(y3 * np.log(p3) + (~y3) * np.log(1 - p3)).mean()))
        winners = np.flatnonzero(done & (positions[local] == 1))
        win.append(float(-np.log(np.clip(probabilities[winners[0], 0], 1e-12, 1))) if len(winners) == 1 else np.nan)
        pick.append(float(y3[int(np.argmax(p3))]))
        keys.append(race_ids[start])
        numbers.append(race_numbers[start])
    outcome, top3, win, pick, numbers = map(np.asarray, (outcome, top3, win, pick, numbers))
    buckets = {}
    for name, mask in (('1_4', numbers <= 4), ('5_8', (numbers > 4) & (numbers <= 8)), ('9_plus', numbers > 8)):
        buckets[name] = dict(races=int(mask.sum()),
                             race_outcome_log_loss=float(outcome[mask].mean()) if mask.any() else None)
    metrics = dict(races=len(keys), race_outcome_log_loss=float(outcome.mean()),
                   top3_race_log_loss=float(top3.mean()), win_log_loss=float(np.nanmean(win)),
                   top_pick_top3_rate=float(pick.mean()), by_race_number=buckets)
    return dict(race_ids=keys, outcome=outcome, top3=top3, win=win, metrics=metrics)


def paired_bootstrap(differences, *, resamples=2000, seed=0):
    """Mean and 95% interval of per-race paired differences (negative = better)."""
    differences = np.asarray(differences, dtype=float)
    differences = differences[np.isfinite(differences)]
    if not len(differences):
        return None
    generator = np.random.default_rng(seed)
    means = np.empty(resamples)
    for index in range(resamples):
        means[index] = differences[generator.integers(0, len(differences), len(differences))].mean()
    return dict(mean=float(differences.mean()), lower95=float(np.percentile(means, 2.5)),
                upper95=float(np.percentile(means, 97.5)), races=int(len(differences)),
                better_share=float((differences < 0).mean()))
