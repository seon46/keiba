"""Chronological model selection and untouched final-period evaluation."""
from __future__ import annotations

from collections import defaultdict
from datetime import date
import hashlib
import json
from pathlib import Path
import sqlite3


def libraries():
    try:
        import numpy as np
        import joblib
        from sklearn.ensemble import HistGradientBoostingClassifier
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ValueError('先に python -m pip install -r requirements-ml.txt を実行してください。') from exc
    return np, joblib, HistGradientBoostingClassifier, threadpool_limits


def readonly(database):
    return sqlite3.connect(Path(database).resolve().as_uri() + '?mode=ro', uri=True)


def feature_revision():
    from . import dataset, features
    digest = hashlib.sha256()
    for module in (dataset, features):
        digest.update(module.__name__.encode() + b'\0')
        digest.update(Path(module.__file__).read_bytes().replace(b'\r\n',b'\n'))
    return digest.hexdigest()


def feature_sets():
    from .features import BASE_FEATURE_NAMES, FORM_FEATURE_NAMES, FEATURE_NAMES
    return {'baseline': list(BASE_FEATURE_NAMES), 'form': list(FORM_FEATURE_NAMES), 'enhanced': list(FEATURE_NAMES)}


def select_feature_columns(bundle):
    """Recover precisely the input columns selected on the validation period."""
    from .features import FEATURE_NAMES
    names = bundle.get('selected_feature_names', bundle['feature_names'])
    if not names or len(names) != len(set(names)) or any(name not in FEATURE_NAMES for name in names):
        raise ValueError('モデルに保存された特徴量の選択が不正です。')
    return [FEATURE_NAMES.index(name) for name in names]


def load_group(database, group):
    from .features import FEATURE_NAMES
    np, *_ = libraries()
    connection = readonly(database)
    try:
        count = connection.execute('SELECT COUNT(*) FROM features WHERE group_name=?', (group,)).fetchone()[0]
        x = np.empty((count, len(FEATURE_NAMES)), dtype=np.float32)
        y = np.empty(count, dtype=np.uint8)
        days = np.empty(count, dtype='U10')
        declared, popularity = np.empty(count), np.empty(count)
        identities = []
        rows = connection.execute(
            "SELECT f.day,f.race_id,f.horse_id,f.horse_number,f.target,f.feature_json,"
            "json_extract(p.payload_json,'$.declared_field_size'),json_extract(p.payload_json,'$.popularity') "
            "FROM features f JOIN input_rows p ON p.region=f.region AND p.race_id=f.race_id "
            "AND p.horse_id=f.horse_id WHERE f.group_name=? ORDER BY f.day,f.race_id,f.horse_number", (group,))
        for index, row in enumerate(rows):
            values = json.loads(row[5])
            if len(values) != len(FEATURE_NAMES):
                raise ValueError('特徴量の版が一致しません。prepareから作成し直してください。')
            x[index] = [float('nan') if value is None else value for value in values]
            y[index], days[index] = row[4], row[0]
            declared[index], popularity[index] = row[6] or float('nan'), row[7] or float('nan')
            identities.append((row[1], row[2], row[3]))
        return dict(x=x, y=y, days=days, declared=declared, popularity=popularity, identities=identities)
    finally:
        connection.close()


def evaluate(y, probabilities, identities, *, popularity=None):
    """Binary top-three probabilities are never normalized to sum to one."""
    np, *_ = libraries()
    y = np.asarray(y, dtype=float)
    p = np.asarray(probabilities, dtype=float)
    if (not len(y) or len(y) != len(p) or len(identities) != len(y) or not np.isin(y,[0,1]).all()
            or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any()):
        raise ValueError('評価用の確率または正解データが不正です。')
    clipped = np.clip(p, 1e-7, 1-1e-7)
    loss = -(y*np.log(clipped) + (1-y)*np.log(1-clipped))
    brier = (p-y)**2
    races = defaultdict(list)
    for index, identity in enumerate(identities):
        races[identity[0]].append(index)
    race_losses, race_briers, precision, first, favorite = [], [], [], [], []
    for indices in races.values():
        race_losses.append(float(loss[indices].mean()))
        race_briers.append(float(brier[indices].mean()))
        ordered = sorted(indices, key=lambda i: (-p[i], identities[i][2]))
        picks = ordered[:min(3, len(ordered))]
        precision.append(float(y[picks].mean()))
        first.append(float(y[ordered[0]]))
        if popularity is not None and np.isfinite(np.asarray(popularity)[indices]).all():
            picks = sorted(indices, key=lambda i: (popularity[i], identities[i][2]))[:3]
            favorite.append(float(y[picks].mean()))
    calibration = []
    bins = np.minimum((p*10).astype(int),9)
    for bin_number in range(10):
        mask = bins == bin_number
        if mask.any():
            calibration.append(dict(lower=bin_number/10, rows=int(mask.sum()),
                                    mean_prediction=float(p[mask].mean()), observed_rate=float(y[mask].mean())))
    return dict(rows=len(y), races=len(races), positives=int(y.sum()),
                log_loss=float(loss.mean()), brier=float(brier.mean()),
                race_log_loss=float(np.mean(race_losses)), race_brier=float(np.mean(race_briers)),
                top_pick_top3_rate=float(np.mean(first)), precision_at_3=float(np.mean(precision)),
                final_popularity_precision_at_3=float(np.mean(favorite)) if favorite else None,
                final_popularity_races=len(favorite), calibration=calibration)


def train(database: Path, output: Path, *, validation_start='2025-01-01', test_start='2026-01-01',
          threads=4, iterations=150, feature_set='auto', report=None):
    from .features import FEATURE_NAMES, CATEGORICAL_FEATURES
    np, joblib, Classifier, threadpool_limits = libraries()
    if any(date.fromisoformat(value).isoformat() != value for value in (validation_start,test_start)):
        raise ValueError('期間の境界はYYYY-MM-DDで指定してください。')
    if date.fromisoformat(validation_start) >= date.fromisoformat(test_start):
        raise ValueError('検証開始日は最終評価開始日より前にしてください。')
    if threads < 1 or iterations < 1:
        raise ValueError('threadsとiterationsは1以上にしてください。')
    profiles = feature_sets()
    if feature_set not in ('auto', *profiles):
        raise ValueError('feature_setはauto/baseline/form/enhancedで指定してください。')
    requested = list(profiles) if feature_set == 'auto' else [feature_set]
    report = report or (lambda **message: None)
    connection = readonly(database)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
        if json.loads(metadata['feature_revision']) != feature_revision():
            raise ValueError('特徴量コードが変わっています。prepareから新しい出力先に作成してください。')
        groups = [row[0] for row in connection.execute('SELECT DISTINCT group_name FROM features ORDER BY group_name')]
    finally:
        connection.close()
    output.mkdir(parents=True, exist_ok=True)
    evaluation = dict(target='finish_position <= 3', validation_start=validation_start,
                      test_start=test_start, groups={}, feature_names=list(FEATURE_NAMES),
                      feature_revision=feature_revision(), probability_normalization='none',
                      requested_feature_set=feature_set, selection_metric='validation.race_log_loss',
                      selection_period='validation',
                      interpretation='3着以内の確率。複勝の払戻条件とは異なります。')
    trained = 0
    for group in groups:
        report(event='training_group', group=group)
        data = load_group(database, group)
        masks = dict(train=data['days'] < validation_start,
                     validation=(data['days'] >= validation_start) & (data['days'] < test_start),
                     test=data['days'] >= test_start)
        sizes = {key: int(mask.sum()) for key, mask in masks.items()}
        if sizes['train'] < 200 or sizes['validation'] < 50 or sizes['test'] < 50 or any(
                len(np.unique(data['y'][mask])) != 2 for mask in masks.values()):
            evaluation['groups'][group] = dict(state='skipped', rows=sizes, reason='各期間の行数または正例・負例が不足')
            report(event='skipped_group', group=group, **evaluation['groups'][group])
            continue
        def metrics(mask, probabilities):
            indices = np.flatnonzero(mask)
            return evaluate(data['y'][mask], probabilities, [data['identities'][i] for i in indices],
                            popularity=data['popularity'][mask])
        candidates = []
        parameters = dict(max_iter=iterations, learning_rate=.07, min_samples_leaf=30,
                          l2_regularization=2.0, early_stopping=False, random_state=42)
        with threadpool_limits(limits=threads):
            for profile in requested:
                names = profiles[profile]
                columns = [FEATURE_NAMES.index(name) for name in names]
                categories = [name in CATEGORICAL_FEATURES for name in names]
                x_train = data['x'][masks['train']][:, columns]
                x_validation = data['x'][masks['validation']][:, columns]
                for leaves in (15, 31):
                    report(event='training_candidate', group=group, feature_set=profile, max_leaf_nodes=leaves)
                    estimator = Classifier(max_leaf_nodes=leaves, categorical_features=categories, **parameters)
                    estimator.fit(x_train, data['y'][masks['train']])
                    validation = metrics(masks['validation'], estimator.predict_proba(x_validation)[:, 1])
                    candidates.append(dict(feature_set=profile, max_leaf_nodes=leaves, validation=validation))
                del x_train, x_validation
            selected = min(candidates, key=lambda item: (item['validation']['race_log_loss'],
                           item['feature_set'] != 'baseline', item['max_leaf_nodes']))
            profile, leaves = selected['feature_set'], selected['max_leaf_nodes']
            selected_validation = selected['validation']
            names = profiles[profile]
            columns = [FEATURE_NAMES.index(name) for name in names]
            categories = [name in CATEGORICAL_FEATURES for name in names]
            estimator = Classifier(max_leaf_nodes=leaves, categorical_features=categories, **parameters)
            development = data['days'] < test_start
            estimator.fit(data['x'][development][:, columns], data['y'][development])
            test = metrics(masks['test'], estimator.predict_proba(data['x'][masks['test']][:, columns])[:, 1])
        # Compare the same known starter set as the model, including the same
        # assumptions about withdrawals at prediction time.
        field_size = data['x'][masks['test'],FEATURE_NAMES.index('field_size')]
        baseline = np.where(np.isfinite(field_size) & (field_size > 0), np.minimum(3/field_size, .999999),
                            float(data['y'][development].mean()))
        prior = metrics(masks['test'], baseline)
        trained_through = str(max(data['days'][development]))
        bundle = dict(estimator=estimator, group=group, feature_names=list(FEATURE_NAMES),
                      feature_revision=feature_revision(), trained_through=trained_through,
                      target='top3', schema_version=1, threads=threads, feature_set=profile,
                      selected_feature_names=names)
        joblib.dump(bundle, output / (group + '.joblib'), compress=3)
        comparison = {}
        for name in requested:
            best = min((candidate for candidate in candidates if candidate['feature_set'] == name),
                       key=lambda item: (item['validation']['race_log_loss'], item['max_leaf_nodes']))
            comparison[name] = dict(max_leaf_nodes=best['max_leaf_nodes'],
                                    feature_count=len(profiles[name]), metrics=best['validation'])
        evaluation['groups'][group] = dict(state='trained', rows=sizes, trained_through=trained_through,
            selected_feature_set=profile, selected_feature_names=names, validation_comparison=comparison,
            selected_max_leaf_nodes=leaves, validation=selected_validation, test=test, field_size_prior=prior,
            improves_prior_log_loss=test['race_log_loss'] < prior['race_log_loss'])
        trained += 1
        report(event='evaluated', group=group, feature_set=profile, test_races=test['races'],
               test_log_loss=test['race_log_loss'], prior_log_loss=prior['race_log_loss'])
    (output / 'evaluation.json').write_text(json.dumps(evaluation, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    if not trained:
        raise ValueError('学習できる区分がありません。期間別の行数を確認してください。')
    return evaluation
