"""Offline CSV preparation, top-three training, and entries prediction."""
from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import csv
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

from . import SCHEMA_VERSION
from .model import feature_revision, libraries, readonly, select_feature_columns, train


def emit(**message):
    print(json.dumps(message, ensure_ascii=False), flush=True)


@contextmanager
def fresh_directory(destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'出力先は既にあります。別の出力先を指定してください: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.keiba-ai-', dir=destination.parent))
    try:
        yield temporary
        if destination.exists():
            raise ValueError(f'出力先が既に作られています: {destination}')
        temporary.rename(destination)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def prepare(source, output):
    from .dataset import ingest_csv
    from .features import build_features, FEATURE_NAMES
    source = Path(source).resolve()
    original = source.stat()
    with fresh_directory(output) as temporary:
        database = temporary / 'features.sqlite3'
        quality = ingest_csv(source, database, report=emit)
        quality['database'] = str(Path(output).resolve() / 'features.sqlite3')
        emit(event='quality_checked', report=quality)
        built = build_features(database, report=emit)
        digest = hashlib.sha256()
        with source.open('rb') as file:
            for block in iter(lambda: file.read(1024*1024), b''):
                digest.update(block)
        current = source.stat()
        if (original.st_size, original.st_mtime_ns) != (current.st_size, current.st_mtime_ns):
            raise ValueError('CSVが処理中に変更されました。書き込み終了後に実行してください。')
        metadata = dict(schema_version=SCHEMA_VERSION, feature_names=list(FEATURE_NAMES),
                        feature_revision=feature_revision(), source_csv_sha256=digest.hexdigest(),
                        source_csv=str(source), source_bytes=original.st_size,
                        quality=quality, features=built)
        with closing(sqlite3.connect(database)) as connection:
            connection.execute('CREATE TABLE ai_metadata(key TEXT PRIMARY KEY,value_json TEXT NOT NULL)')
            connection.executemany('INSERT INTO ai_metadata VALUES (?,?)',
                                   [(key, json.dumps(value, ensure_ascii=False)) for key,value in metadata.items()])
            connection.commit()
            connection.execute('PRAGMA wal_checkpoint(TRUNCATE)')
        (temporary / 'quality_report.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    emit(event='prepared', output=str(Path(output) / 'features.sqlite3'))
    return Path(output) / 'features.sqlite3'


def predict(database, models, entries, output):
    from .dataset import normalize_row, iter_races
    from .features import FEATURE_NAMES, HistoryState, race_features
    np, joblib, _, threadpool_limits = libraries()
    output = Path(output).resolve()
    if output.exists():
        raise ValueError(f'出力先は既にあります: {output}')
    connection = readonly(database)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
        if json.loads(metadata['feature_revision']) != feature_revision():
            raise ValueError('特徴量コードが変わっています。prepareから作成し直してください。')
    finally:
        connection.close()
    grouped, seen = {}, set()
    with Path(entries).open(encoding='utf-8-sig',newline='') as file:
        for raw in csv.DictReader(file):
            row = normalize_row(raw, page='entries')
            key = (row['region'],row['race_id'],row['horse_id'])
            if key in seen:
                raise ValueError('出馬表に同じ馬・レースの重複があります。')
            seen.add(key)
            grouped.setdefault((row['day'],row['region'],row['race_id']),[]).append(row)
    if not grouped:
        raise ValueError('出馬表がありません。')
    results, loaded, cached_day, history = [], {}, None, None
    history_day = None
    for (day, region, race_id), race in sorted(grouped.items()):
        fields = ('day','group_name','course','surface','distance_m','declared_field_size','snapshot_id')
        if any(len({row.get(field) for row in race}) > 1 for field in fields):
            raise ValueError(f'出馬表の条件が一致しません: {race_id}')
        if len({row['horse_number'] for row in race}) != len(race):
            raise ValueError(f'出馬表の馬番が重複しています: {race_id}')
        declared = race[0].get('declared_field_size')
        if declared is not None and declared != len(race):
            raise ValueError(f'出馬表の頭数が不足しています: {race_id}')
        active = [row for row in race if row.get('finish_status') not in ('scratched','excluded')]
        if not active:
            continue
        if day != cached_day:
            history, history_day = HistoryState(), None
            for previous in iter_races(Path(database), before_date=day):
                history.update_race(previous)
                history_day = previous[0]['day']
            cached_day = day
        group = active[0]['group_name']
        if group not in loaded:
            path = Path(models) / (group+'.joblib')
            if not path.is_file():
                raise ValueError(f'この区分の学習済みモデルがありません: {group}')
            loaded[group] = joblib.load(path)
        bundle = loaded[group]
        if bundle.get('group') != group or bundle.get('target') != 'top3' or bundle.get('schema_version') != SCHEMA_VERSION:
            raise ValueError('モデルの区分または予測対象が一致しません。')
        if bundle['feature_names'] != list(FEATURE_NAMES) or bundle['feature_revision'] != feature_revision():
            raise ValueError('モデルと特徴量の版が一致しません。')
        if day <= bundle['trained_through']:
            raise ValueError(f'学習期間内の日付には予測できません: {day}')
        x = np.asarray([[np.nan if value is None else value for value in values]
                        for values in race_features(active,history)],dtype=np.float32)
        x = x[:, select_feature_columns(bundle)]
        with threadpool_limits(limits=bundle.get('threads',4)):
            probabilities = bundle['estimator'].predict_proba(x)[:,1]
        for row, probability in sorted(zip(active,probabilities),key=lambda pair:(-pair[1],pair[0]['horse_number'])):
            results.append(dict(race_date=day,region=region,race_id=race_id,horse_id=row['horse_id'],
                horse_number=row['horse_number'],top3_probability=float(probability),group=group,
                history_last_date=history_day,
                days_since_history=(date.fromisoformat(day)-date.fromisoformat(history_day)).days if history_day else None))
    if not results:
        raise ValueError('予測対象の馬がありません。')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',newline='',dir=output.parent,
                                         prefix='.keiba-prediction-',delete=False) as file:
            temporary = Path(file.name)
            writer = csv.DictWriter(file,fieldnames=list(results[0]))
            writer.writeheader()
            writer.writerows(results)
        os.link(temporary,output)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    emit(event='predicted',output=str(output),rows=len(results))


def main(argv=None):
    parser = argparse.ArgumentParser(description='保存済みCSVから3着以内の確率を学習・評価します。')
    commands = parser.add_subparsers(dest='command',required=True)
    for name in ('prepare','fit'):
        command = commands.add_parser(name)
        command.add_argument('--input',required=True,type=Path)
        command.add_argument('--output-dir',required=True,type=Path)
    training = commands.add_parser('train')
    training.add_argument('--dataset',required=True,type=Path)
    training.add_argument('--output-dir',required=True,type=Path)
    for command in (commands.choices['fit'],training):
        command.add_argument('--validation-start',default='2025-01-01')
        command.add_argument('--test-start',default='2026-01-01')
        command.add_argument('--threads',type=int,default=4)
        command.add_argument('--iterations',type=int,default=150)
        command.add_argument('--feature-set', choices=('auto','baseline','enhanced'), default='auto',
                             help='検証期間で従来特徴と拡張特徴を比較（既定: auto）')
    prediction = commands.add_parser('predict')
    prediction.add_argument('--dataset',required=True,type=Path)
    prediction.add_argument('--model-dir',required=True,type=Path)
    prediction.add_argument('--entries',required=True,type=Path)
    prediction.add_argument('--output',required=True,type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'prepare':
            prepare(args.input,args.output_dir)
        elif args.command in ('fit','train'):
            libraries()  # Report missing dependencies before scanning a large CSV.
            database = prepare(args.input,args.output_dir) if args.command == 'fit' else args.dataset
            destination = args.output_dir/'models' if args.command == 'fit' else args.output_dir
            with fresh_directory(destination) as temporary:
                train(database,temporary,validation_start=args.validation_start,test_start=args.test_start,
                      threads=args.threads,iterations=args.iterations,feature_set=args.feature_set,report=emit)
            emit(event='training_complete',report=str(destination/'evaluation.json'))
        else:
            predict(args.dataset,args.model_dir,args.entries,args.output)
        return 0
    except (ValueError,OSError,sqlite3.Error,KeyError,csv.Error) as exc:
        emit(event='error',message=str(exc))
        return 2
    except KeyboardInterrupt:
        emit(event='interrupted',message='停止しました。完成済みのprepare出力があればtrainから再実行できます。')
        return 130
