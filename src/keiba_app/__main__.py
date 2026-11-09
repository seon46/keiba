"""keiba_app: init / serve / nightly / predict / score."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys


def emit(**message):
    print(json.dumps(message, ensure_ascii=False, default=str), flush=True)


def main(argv=None):
    from .nightly import STEPS
    from .state import App
    parser = argparse.ArgumentParser(prog='keiba_app', description='競馬AIアプリ（順位分布モデル）')
    parser.add_argument('--data-dir', default='data', type=Path, help='スクレイパーのデータフォルダー（既定: data）')
    commands = parser.add_subparsers(dest='command', required=True)
    init = commands.add_parser('init', help='使うDBとモデルを登録します（コピーはしません）')
    init.add_argument('--dataset', required=True, type=Path)
    init.add_argument('--model-dir', required=True, type=Path)
    init.add_argument('--collected-through', help='収集済みの最終日（既定: DBのデータ最終日）')
    init.add_argument('--data-start', help='学習データの開始日（既定: 2021-10-11）')
    init.add_argument('--threads', type=int)
    serve = commands.add_parser('serve', help='ブラウザ用の画面を起動します')
    serve.add_argument('--host', default='127.0.0.1')
    serve.add_argument('--port', type=int, default=8765)
    serve.add_argument('--allow-host', action='append', default=[],
                       help='127.0.0.1・localhost・*.ts.net以外で開くときの名前（例: PCの名前）')
    nightly = commands.add_parser('nightly', help='成績の追加・DB更新・照合・学習し直し・当日の予想')
    nightly.add_argument('--force-fit', action='store_true', help='学習し直しの日でなくても学習します')
    nightly.add_argument('--only', nargs='+', choices=STEPS)
    nightly.add_argument('--skip', nargs='+', default=(), choices=STEPS)
    predict = commands.add_parser('predict', help='1レースを予想して記録します')
    predict.add_argument('--region', required=True, choices=('central', 'local'))
    predict.add_argument('--race-id', required=True)
    predict.add_argument('--refresh', action='store_true', help='出馬表を取り直します')
    commands.add_parser('score', help='結果が出たレースの予想を照合します')
    day = commands.add_parser('day', help='その日の全レースを予想して記録します（画面の「この日の全レースを予想」）')
    day.add_argument('--date', required=True)
    day.add_argument('--no-refresh', action='store_true', help='保存済みの出馬表を使います')
    discover = commands.add_parser('discover', help='その日のレース一覧を取得します')
    discover.add_argument('--date', required=True)
    args = parser.parse_args(argv)
    app = App(args.data_dir)
    try:
        if args.command == 'init':
            from .nightly import dataset_info, model_info
            dataset, model = args.dataset.resolve(), args.model_dir.resolve()
            info = dataset_info(dataset)
            summary = model_info(model)
            if summary is None:
                raise ValueError(f'モデルのフォルダーではありません（model.jsonがありません）: {model}')
            changes = dict(dataset=str(dataset), model_dir=str(model),
                           collected_through=args.collected_through or info['last_day'],
                           prepared_through=args.collected_through or info['last_day'],
                           last_fit=(summary.get('created_at') or '')[:10] or None)
            if args.data_start:
                changes['data_start'] = args.data_start
            if args.threads:
                changes['threads'] = args.threads
            state = app.write_state(**changes)
            emit(event='initialized', state=state, data_last_day=info['last_day'], model_train_end=summary.get('train_end'))
        elif args.command == 'serve':
            from .server import serve
            app.require_ready()
            serve(app, host=args.host, port=args.port, allowed_hosts=args.allow_host)
        elif args.command == 'nightly':
            from keiba_ai.priority import background
            from .nightly import run_nightly
            background()  # pages of the running app answer first
            result = run_nightly(app, force_fit=args.force_fit, only=args.only, skip=args.skip, log=print)
            emit(event='nightly_done', **result)
            return 0 if result['status'] in ('ok', 'warning') else 2
        elif args.command == 'predict':
            from .engine import Engine
            engine = Engine(app, report=emit)
            identifier = engine.predict_race(args.region, args.race_id, refresh=args.refresh)
            prediction = engine.logbook.prediction(identifier)
            for runner in sorted(prediction['runners'], key=lambda item: -(item.get('win_probability') or -1)):
                emit(horse_number=runner['horse_number'], horse_name=runner['horse_name'],
                     win=runner.get('win_probability'), top3=runner.get('top3_probability'),
                     expected_rank=runner.get('expected_rank'))
            emit(event='predicted', prediction_id=identifier)
        elif args.command in ('day', 'discover'):
            from keiba_ai.priority import background
            from .engine import Engine
            background()  # the app that started this job answers first
            engine = Engine(app)
            if args.command == 'day':
                result = engine.predict_day(args.date, refresh=not args.no_refresh,
                                            progress=lambda **message: emit(event='progress', **message))
                emit(event='result', result=dict(predicted=len(result['predicted']), skipped=result['skipped']))
            else:
                races = engine.discover(args.date)
                counts = {region: sum(race['region'] == region for race in races) for region in ('central', 'local')}
                emit(event='result', result=dict(races=len(races), counts=counts, errors=engine.discover_errors))
        elif args.command == 'score':
            from .engine import Engine
            emit(event='scored', counts=Engine(app).score_pending())
        return 0
    except (ValueError, KeyError, OSError, sqlite3.Error) as exc:
        emit(event='error', message=str(exc))
        return 2
    except KeyboardInterrupt:
        return 130


if __name__ == '__main__':
    sys.exit(main())
