"""Conditional rank regression and coherent race-level rank marginals."""
from __future__ import annotations

from collections import Counter
import csv
from itertools import groupby
import hashlib
import json
from pathlib import Path

from .model import evaluate, feature_revision, feature_sets, libraries, load_group, readonly


def race_seed(identifier, seed=42):
    return int.from_bytes(hashlib.blake2b(f'{seed}:{identifier}'.encode(),digest_size=8).digest(),'little')


def load_rank_group(database, group):
    """Keep histories intact but exclude tied/ambiguous races from rank fitting."""
    np, *_ = libraries()
    data = load_group(database,group)
    count=len(data['identities'])
    positions=np.zeros(count,dtype=np.int16)
    finished=np.zeros(count,dtype=bool)
    connection = readonly(database)
    try:
        rows = connection.execute(
            "SELECT json_extract(p.payload_json,'$.finish_position'),"
            "json_extract(p.payload_json,'$.finish_status') FROM features f JOIN input_rows p "
            "ON p.region=f.region AND p.race_id=f.race_id AND p.horse_id=f.horse_id "
            "WHERE f.group_name=? ORDER BY f.day,f.race_id,f.horse_number",(group,))
        observed=0
        for index,row in enumerate(rows):
            if index>=count:
                raise ValueError('AI用データが処理中に変更されました。')
            positions[index]=row[0] or 0
            finished[index]=row[1]=='finished'
            observed+=1
        if observed!=count:
            raise ValueError('AI用データが処理中に変更されました。')
    finally:
        connection.close()
    valid = np.ones(count,dtype=bool)
    exclusions = Counter()
    field_sizes = np.empty(count,dtype=np.int16)
    for identifier, iterator in groupby(range(count),key=lambda i:data['identities'][i][0]):
        indices = list(iterator)
        ranks = sorted(int(positions[i]) for i in indices if finished[i])
        field_sizes[indices] = len(indices)
        if ranks != list(range(1,len(ranks)+1)):
            valid[indices] = False
            exclusions['tied_or_noncontiguous_finish_positions'] += 1
    data['x'] = data['x'][valid]
    for key in ('y','days','declared','popularity'):
        data[key] = data[key][valid]
    data['identities'] = [value for i,value in enumerate(data['identities']) if valid[i]]
    data['positions'], data['finished'], data['field_sizes'] = positions[valid], finished[valid], field_sizes[valid]
    data['excluded_races'] = dict(exclusions)
    data['excluded_rows'] = int((~valid).sum())
    return data


def estimate_completion(estimator, constant, x):
    np, *_ = libraries()
    return np.full(len(x),constant,dtype=float) if estimator is None else estimator.predict_proba(x)[:,1]


def rank_metrics(data, mask, scores, completion, *, temperature, samples, seed=42,uniform=False,prediction_sink=None):
    from .ranking import rank_distribution
    np, *_ = libraries()
    indices = np.flatnonzero(mask)
    identities = [data['identities'][i] for i in indices]
    positions, finished = data['positions'][mask], data['finished'][mask]
    top3 = np.empty(len(indices))
    losses, briers, race_losses, race_briers, errors = [], [], [], [], []
    for identifier, iterator in groupby(range(len(indices)),key=lambda i:identities[i][0]):
        selected = list(iterator)
        n = len(selected)
        probabilities = rank_distribution(scores[selected],completion[selected],temperature=temperature,
                                          samples=samples,seed=race_seed(identifier,seed),smoothing=1 if uniform else .001)
        outcomes = np.where(finished[selected],positions[selected]-1,n)
        truth = probabilities[np.arange(n),outcomes]
        loss = -np.log(np.clip(truth,1e-12,1))
        brier = (probabilities**2).sum(axis=1)+1-2*truth
        losses.extend(loss.tolist()); briers.extend(brier.tolist())
        race_losses.append(float(loss.mean())); race_briers.append(float(brier.mean()))
        top3[selected] = np.clip(probabilities[:,:min(3,n)].sum(axis=1),0,1)
        if prediction_sink is not None:
            for local,index in enumerate(selected):
                row=dict(race_date=str(data['days'][indices[index]]),race_id=identifier,
                    horse_id=identities[index][1],horse_number=identities[index][2],starters=n,
                    actual_finish_position=int(positions[index]) if finished[index] else None,
                    actual_nonfinish=not bool(finished[index]),top3_probability=float(top3[index]),
                    nonfinish_probability=float(probabilities[local,-1]))
                row.update({f'rank_{rank}_probability':float(value)
                            for rank,value in enumerate(probabilities[local,:n],1)})
                prediction_sink(row)
        rank_mass = probabilities[:,:n].sum(axis=1)
        expected = (probabilities[:,:n]*np.arange(1,n+1)).sum(axis=1)/np.maximum(rank_mass,1e-12)
        errors.extend(abs(expected[j]-positions[selected[j]]) for j in range(n)
                      if finished[selected[j]] and rank_mass[j]>0)
    nonfinish = 1-completion
    return dict(rows=len(indices),races=len(race_losses),finished_rows=int(finished.sum()),
        nonfinished_rows=int((~finished).sum()),outcome_log_loss=float(np.mean(losses)),
        race_outcome_log_loss=float(np.mean(race_losses)),multiclass_brier=float(np.mean(briers)),
        race_multiclass_brier=float(np.mean(race_briers)),
        conditional_expected_rank_mae=float(np.mean(errors)) if errors else None,
        conditional_rank_rows=len(errors),
        nonfinish_brier=float(np.mean((nonfinish-(~finished))**2)),
        top3=evaluate(data['y'][mask],top3,identities,popularity=data['popularity'][mask]))


def train_rank(database: Path, output: Path, *, validation_start='2025-01-01',test_start='2026-01-01',
               threads=4,iterations=150,feature_set='auto',validation_samples=128,
               evaluation_samples=1024,report=None):
    from datetime import date
    from sklearn.ensemble import HistGradientBoostingRegressor
    from .features import FEATURE_NAMES,CATEGORICAL_FEATURES
    np,joblib,Classifier,threadpool_limits = libraries()
    if any(date.fromisoformat(value).isoformat()!=value for value in (validation_start,test_start)):
        raise ValueError('期間の境界はYYYY-MM-DDで指定してください。')
    if validation_start>=test_start or min(threads,iterations,validation_samples,evaluation_samples)<1:
        raise ValueError('期間の順序とthreads/iterations/samplesの正数指定を確認してください。')
    profiles = feature_sets()
    if feature_set not in ('auto',*profiles):
        raise ValueError('未対応のfeature_setです。')
    requested = ['form','enhanced'] if feature_set=='auto' else [feature_set]
    report = report or (lambda **message:None)
    connection = readonly(database)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
        if json.loads(metadata['feature_revision'])!=feature_revision():
            raise ValueError('特徴量の版が変わっています。fitでCSVから新しい出力先に作成してください。')
        groups = [row[0] for row in connection.execute('SELECT DISTINCT group_name FROM features ORDER BY group_name')]
    finally:
        connection.close()
    output.mkdir(parents=True,exist_ok=True)
    result = dict(target='rank_distribution',validation_start=validation_start,test_start=test_start,
        feature_names=list(FEATURE_NAMES),feature_revision=feature_revision(),groups={},
        requested_feature_set=feature_set,selection_period='validation',
        selection_metric='validation.race_outcome_log_loss',
        rank_definition='1..N (without dead heats); nonfinish includes DNF and disqualification',
        method='regression utilities + Plackett-Luce order + independent completion probabilities',
        validation_samples=validation_samples,evaluation_samples=evaluation_samples,prediction_samples=4096,
        smoothing=.001,seed=42)
    trained = 0
    for group in groups:
        report(event='training_rank_group',group=group)
        data = load_rank_group(database,group)
        masks = dict(train=data['days']<validation_start,
            validation=(data['days']>=validation_start)&(data['days']<test_start),test=data['days']>=test_start)
        sizes = {name:int(mask.sum()) for name,mask in masks.items()}
        finished_train = masks['train']&data['finished']
        info = dict(rows=sizes,excluded_rows=data['excluded_rows'],excluded_races=data['excluded_races'])
        if sizes['train']<200 or sizes['validation']<50 or sizes['test']<50 or finished_train.sum()<100:
            result['groups'][group]=dict(state='skipped',reason='順位モデルの期間別データが不足',**info)
            report(event='skipped_group',group=group,**result['groups'][group])
            continue
        fractions = (data['positions']-1)/np.maximum(data['field_sizes']-1,1)
        candidates=[]
        parameters=dict(max_iter=iterations,learning_rate=.07,min_samples_leaf=30,
            l2_regularization=2.0,early_stopping=False,random_state=42)
        def matrix(mask,columns):
            return data['x'][np.ix_(np.flatnonzero(mask),columns)]
        def fit_pair(mask,names,leaves):
            columns=[FEATURE_NAMES.index(name) for name in names]
            categories=[name in CATEGORICAL_FEATURES for name in names]
            completed=mask&data['finished']
            ranker=HistGradientBoostingRegressor(max_leaf_nodes=leaves,categorical_features=categories,**parameters)
            ranker.fit(matrix(completed,columns),fractions[completed])
            constant=float(data['finished'][mask].mean())
            finisher=None
            if len(np.unique(data['finished'][mask]))==2:
                finisher=Classifier(max_leaf_nodes=leaves,categorical_features=categories,**parameters)
                finisher.fit(matrix(mask,columns),data['finished'][mask].astype(np.uint8))
            return ranker,finisher,constant,columns
        with threadpool_limits(limits=threads):
            for profile in requested:
                for leaves in (15,31):
                    report(event='training_rank_candidate',group=group,feature_set=profile,max_leaf_nodes=leaves)
                    ranker,finisher,constant,columns=fit_pair(masks['train'],profiles[profile],leaves)
                    x=matrix(masks['validation'],columns)
                    scores=-ranker.predict(x)
                    completion=estimate_completion(finisher,constant,x)
                    for temperature in (.1,.2,.4):
                        metrics=rank_metrics(data,masks['validation'],scores,completion,temperature=temperature,
                                             samples=validation_samples)
                        candidates.append(dict(feature_set=profile,max_leaf_nodes=leaves,temperature=temperature,metrics=metrics))
            selected=min(candidates,key=lambda c:(c['metrics']['race_outcome_log_loss'],
                len(profiles[c['feature_set']]),c['max_leaf_nodes'],c['temperature']))
            profile=selected['feature_set']; names=profiles[profile]
            development=data['days']<test_start
            ranker,finisher,constant,columns=fit_pair(development,names,selected['max_leaf_nodes'])
            x=matrix(masks['test'],columns)
            maximum=int(data['field_sizes'][masks['test']].max())
            fields=['race_date','race_id','horse_id','horse_number','starters','actual_finish_position',
                    'actual_nonfinish','top3_probability','nonfinish_probability']
            rank_fields=[f'rank_{rank}_probability' for rank in range(1,maximum+1)]
            prediction_file=group+'_test_rank_probabilities.csv'
            with (output/prediction_file).open('w',encoding='utf-8',newline='') as file:
                writer=csv.DictWriter(file,fieldnames=fields+rank_fields)
                writer.writeheader()
                def write_prediction(row):
                    for name in rank_fields: row.setdefault(name,0.0)
                    writer.writerow(row)
                test=rank_metrics(data,masks['test'],-ranker.predict(x),estimate_completion(finisher,constant,x),
                    temperature=selected['temperature'],samples=evaluation_samples,prediction_sink=write_prediction)
            # Uniform rank order with the same development-period completion prior.
            prior=rank_metrics(data,masks['test'],np.zeros(len(x)),np.full(len(x),constant),
                               temperature=1,samples=evaluation_samples,uniform=True)
        through=str(max(data['days'][development]))
        bundle=dict(target='rank',schema_version=1,group=group,feature_names=list(FEATURE_NAMES),
            feature_revision=feature_revision(),selected_feature_names=names,feature_set=profile,
            rank_estimator=ranker,finish_estimator=finisher,finish_constant=constant,
            temperature=selected['temperature'],trained_through=through,threads=threads,
            samples=4096,smoothing=.001,seed=42)
        joblib.dump(bundle,output/(group+'.joblib'),compress=3)
        comparison={}
        for name in requested:
            best=min((c for c in candidates if c['feature_set']==name),key=lambda c:c['metrics']['race_outcome_log_loss'])
            comparison[name]=dict(feature_count=len(profiles[name]),**best)
        result['groups'][group]=dict(state='trained',**info,trained_through=through,
            test_predictions_file=prediction_file,
            selected_feature_set=profile,selected_feature_names=names,
            selected_max_leaf_nodes=selected['max_leaf_nodes'],selected_temperature=selected['temperature'],
            validation=selected['metrics'],validation_comparison=comparison,test=test,uniform_rank_prior=prior)
        trained+=1
        report(event='evaluated_rank',group=group,feature_set=profile,
               rank_log_loss=test['race_outcome_log_loss'],prior_log_loss=prior['race_outcome_log_loss'])
    (output/'evaluation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if not trained:
        raise ValueError('順位モデルを学習できる区分がありません。期間別の件数を確認してください。')
    return result
