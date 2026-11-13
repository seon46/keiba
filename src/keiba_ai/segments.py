"""Race segments (surface x distance x age x class) and segment-specific corrections of a fitted v4 group.

Approach A: every model is first fitted on the whole group as before.  For each segment with enough races,
the LightGBM+pedigree part (B) is continued with a few boosting rounds fitted on that segment only, starting
from B's scores (LightGBM ``init_score``), and each NN seed is fine-tuned on that segment, starting from its
weights.  Both stop on the segment's own stop period; when the first round / epoch does not improve the stop
loss, the segment keeps the group model unchanged.  How far a segment moves away from the group model is
thus decided by its own data, not by a weight hyperparameter.
"""
from __future__ import annotations

from dataclasses import asdict, replace

# Thresholds below which a segment keeps the group model.
MIN_FIT_RACES = 300
MIN_STOP_RACES = 50
# Continuation of B: smaller steps than the group fit, early stopped on the segment's stop races.
RESIDUAL_PARAMS = dict(learning_rate=.02, num_leaves=15, min_data_in_leaf=200, lambda_l2=20.0, feature_fraction=.7)
RESIDUAL_ROUNDS, RESIDUAL_PATIENCE = 2000, 100
# Fine-tuning of each NN seed: a fraction of the group learning rate, a few epochs.
NN_LEARNING_RATE_SCALE, NN_EPOCHS, NN_PATIENCE = .3, 6, 2

CENTRAL_CLASSES = {0: '新馬', 1: '未勝利', 2: '1勝', 3: '2・3勝', 4: '2・3勝', 5: 'OP・重賞'}
LOCAL_CLASSES = {6: 'A', 7: 'A', 8: 'A', 15: 'A', 9: 'B', 10: 'B', 11: 'B', 16: 'B',
                 12: 'C', 13: 'C', 14: 'C', 17: 'C'}


DISTANCES = (('短距離', 0, 1400), ('マイル', 1400, 1900), ('中距離', 1900, 2400), ('長距離', 2400, 10 ** 6))


def _distance(meters):
    import math
    if math.isnan(meters):
        return '距離不明'
    return next(name for name, low, high in DISTANCES if low <= meters < high)


def _age(minimum, maximum):
    import math
    if not math.isnan(maximum) and maximum <= 2:
        return '2歳'
    if not math.isnan(maximum) and maximum <= 3:
        return '3歳'
    if not math.isnan(minimum):
        return '古馬'
    return '年齢不明'


def segment_labels(x, group):
    """Label per row: central '芝・マイル・古馬・2・3勝', local 'マイル・古馬・C' (local is almost all dirt).
    Distance bands: 短距離 (-1399m), マイル (1400-1899m), 中距離 (1900-2399m), 長距離 (2400m-)."""
    import math
    import numpy as np
    from .features import FEATURE_NAMES
    index = {name: FEATURE_NAMES.index(name) for name in
             ('surface_code', 'distance_m', 'race_age_min', 'race_age_max', 'race_class_code', 'race_grade_code')}
    central = group.startswith('central')
    labels = np.empty(len(x), dtype=object)
    for row in range(len(x)):
        values = {name: float(x[row, column]) for name, column in index.items()}
        age = _age(values['race_age_min'], values['race_age_max'])
        distance = _distance(values['distance_m'])
        code = values['race_class_code']
        if central:
            surface = {0.0: '芝', 1.0: 'ダ'}.get(values['surface_code'], 'その他')
            klass = ('OP・重賞' if not math.isnan(values['race_grade_code'])
                     else CENTRAL_CLASSES.get(int(code), 'その他') if not math.isnan(code) else 'その他')
            labels[row] = f'{surface}・{distance}・{age}・{klass}'
        else:
            klass = ('重賞' if not math.isnan(values['race_grade_code'])
                     else LOCAL_CLASSES.get(int(code), 'その他') if not math.isnan(code) else 'その他')
            labels[row] = f'{distance}・{age}・{klass}'
    return labels


def residual_lightgbm(data, matrix, categorical, fit_rows, stop_rows, base_scores, params, *, report=None):
    """Boosting rounds on top of ``base_scores`` (all rows), fitted on ``fit_rows``; (booster, rounds, stop losses).
    ``rounds`` is 0 (booster None) when the segment's stop loss does not improve on the base scores."""
    from .experiment_v3b import _structure
    from .experiment_v3d import fit_pl
    base_loss = _structure(data, stop_rows).mean_loss(base_scores[stop_rows])
    booster, rounds = fit_pl(data, matrix, categorical, fit_rows, stop_rows, dict(params, **RESIDUAL_PARAMS),
                             rounds=RESIDUAL_ROUNDS, patience=RESIDUAL_PATIENCE, offset=base_scores)
    scores = base_scores[stop_rows] + booster.predict(matrix(stop_rows), num_iteration=rounds, raw_score=True)
    loss = _structure(data, stop_rows).mean_loss(scores)
    if loss >= base_loss - 1e-6:
        return None, 0, dict(base=base_loss, segment=base_loss)
    return booster, rounds, dict(base=base_loss, segment=loss)


def fine_tune_network(network, inputs, fit_rows, stop_rows, config, *, device, report=None):
    """Continue ``network`` on the segment; (network, info).  Keeps the starting weights unless an epoch
    lowers the segment's stop loss."""
    import copy
    from .neural import train
    tuned = replace(config, learning_rate=config.learning_rate * NN_LEARNING_RATE_SCALE, max_epochs=NN_EPOCHS,
                    patience=NN_PATIENCE)
    state = {key: value.detach().clone() for key, value in network.state_dict().items()}
    model, info = train(inputs, fit_rows, stop_rows, tuned, device=device, report=report, init_state=state)
    return model, dict(info, config=asdict(tuned))


def copy_state(network):
    return {key: value.detach().cpu().numpy() for key, value in network.state_dict().items()}
