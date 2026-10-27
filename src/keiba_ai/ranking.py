"""Jointly coherent, tie-free finishing-rank marginals.

Only the Plackett--Luce order is sampled.  Conditional on each sampled
Gumbel order, independent finishing indicators are integrated exactly with
a Poisson-binomial dynamic program.  Consequently rare nonfinishes do not
depend on whether a Monte Carlo draw happened to contain one.
"""
from __future__ import annotations

from numbers import Integral, Real

import numpy as np


def _vector(value, name: str) -> np.ndarray:
    try:
        original = np.asarray(value)
        if original.ndim != 1 or not original.size or original.dtype.kind not in 'biuf':
            raise ValueError
        result = np.array(original, dtype=float, copy=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f'{name} must be a nonempty, one-dimensional numeric array.') from exc
    if not np.isfinite(result).all():
        raise ValueError(f'{name} must contain only finite numbers.')
    return result


def _real(value, name: str, *, lower: float, upper: float | None = None,
          strict_lower: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f'{name} must be a finite number.')
    try:
        result = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f'{name} must be a finite number.') from exc
    if (not np.isfinite(result) or (result <= lower if strict_lower else result < lower)
            or (upper is not None and result > upper)):
        raise ValueError(f'{name} is outside its valid range.')
    return result


def _poisson_binomial(probabilities: np.ndarray) -> np.ndarray:
    """Distribution of the number of successes, including exact p=0/1."""
    mass = np.zeros(len(probabilities) + 1, dtype=float)
    mass[0] = 1.0
    for count, probability in enumerate(probabilities):
        previous = mass[:count + 1].copy()
        mass[:count + 1] = previous * (1.0 - probability)
        mass[1:count + 2] += previous * probability
    return mass


def _uniform_order_marginals(probabilities: np.ndarray) -> np.ndarray:
    """Same finishing indicators, with an independent uniform full order."""
    count = len(probabilities)
    ranks = np.zeros((count, count), dtype=float)
    denominators = np.arange(1, count + 1, dtype=float)
    for horse, probability in enumerate(probabilities):
        others = np.concatenate((probabilities[:horse], probabilities[horse + 1:]))
        weighted = _poisson_binomial(others) / denominators
        ranks[horse] = probability * np.cumsum(weighted[::-1])[::-1]
    return ranks


def uniform_rank_distribution(finish_probabilities) -> np.ndarray:
    """Exact strength-free baseline with independent supplied finishing p's.

    This is a uniform full order followed by deletion of nonfinishers, not
    a uniform choice over the n numerical ranks for each horse.  Its rank
    columns therefore have the same occupancy probabilities as the PL
    distribution, with no Monte Carlo error.
    """
    probabilities = _vector(finish_probabilities, 'finish_probabilities')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('finish_probabilities must be between zero and one.')
    result = np.zeros((len(probabilities), len(probabilities) + 1), dtype=float)
    result[:, :-1] = _uniform_order_marginals(probabilities)
    result[:, -1] = 1.0 - probabilities
    return result


def _orders(scores: np.ndarray, temperature: float, gumbels: np.ndarray) -> np.ndarray:
    """Order score/temperature + Gumbel without overflowing extreme scores.

    An adjacent score gap exceeding the entire sampled noise range cannot
    be crossed by these draws.  Capping only those gaps preserves their
    orders, including distinctions among several extremely weak horses.
    Subtracting a common huge score or clipping all weak scores together
    would lose those distinctions.
    """
    sorted_horses = np.argsort(scores, kind='stable')
    with np.errstate(over='ignore', under='ignore'):
        gaps = np.diff(scores[sorted_horses]) / temperature
    maximum_noise_gap = float(gumbels.max() - gumbels.min()) + 1.0
    scaled = np.empty(len(scores), dtype=float)
    scaled[sorted_horses[0]] = 0.0
    scaled[sorted_horses[1:]] = np.cumsum(np.minimum(gaps, maximum_noise_gap))
    return np.argsort(-(gumbels + scaled), axis=1, kind='stable')


def rank_distribution(scores, finish_probabilities, *, temperature: float,
                      samples: int = 4096, seed: int = 42,
                      smoothing: float = .001) -> np.ndarray:
    """Return n x (n+1) marginals for ranks 1..n and nonfinish/DQ.

    Higher scores mean stronger horses.  Finishing indicators are mutually
    independent and independent of the Gumbel order.  Nonfinishers receive
    no numerical rank; the remaining horses occupy consecutive ranks.

    ``smoothing`` mixes the sampled PL law with a uniform-order law having
    the *same* finishing probabilities.  Both components are joint laws,
    so each row sums to one, its last entry is exactly ``1-p_i``, and rank
    column r sums to Pr(number of finishers >= r), even with one sample.
    With p_i=1 for all horses, the rank matrix is doubly stochastic.
    Smoothing does not assign mass to impossible events (p_i=0 or 1);
    evaluation code must clip probabilities if finite log loss is needed
    for an observed event declared impossible by the supplied inputs.

    Scores and input probabilities are not modified.  A local generator
    makes a fixed seed reproducible without changing NumPy's global RNG.
    """
    scores = _vector(scores, 'scores')
    probabilities = _vector(finish_probabilities, 'finish_probabilities')
    if len(scores) != len(probabilities):
        raise ValueError('scores and finish_probabilities must have the same length.')
    if ((probabilities < 0) | (probabilities > 1)).any():
        raise ValueError('finish_probabilities must be between zero and one.')
    temperature = _real(temperature, 'temperature', lower=0.0, strict_lower=True)
    smoothing = _real(smoothing, 'smoothing', lower=0.0, upper=1.0)
    if (isinstance(samples, (bool, np.bool_)) or not isinstance(samples, Integral)
            or samples < 1):
        raise ValueError('samples must be a positive integer.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, Integral) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')

    count = len(scores)
    result = np.zeros((count, count + 1), dtype=float)
    result[:, -1] = 1.0 - probabilities
    uniform = _uniform_order_marginals(probabilities)
    if smoothing == 1.0 or count == 1 or not probabilities.any():
        result[:, :-1] = uniform
        return result

    generator = np.random.default_rng(int(seed))
    rank_mass = np.zeros((count, count), dtype=float)
    remaining = int(samples)
    while remaining:
        # Bound temporary storage for callers requesting many draws.
        batch = min(remaining, 2048)
        uniforms = generator.random((batch, count))
        # Generator.random is half-open; guard its possible exact zero.
        gumbels = -np.log(-np.log(np.maximum(uniforms, np.finfo(float).tiny)))
        orders = _orders(scores, temperature, gumbels)
        # dp[s,k] = probability of k finishers ahead in sample s.
        dp = np.zeros((batch, count + 1), dtype=float)
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
