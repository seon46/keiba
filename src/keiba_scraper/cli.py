from __future__ import annotations

import argparse
import csv
from datetime import date
import json
import os
from pathlib import Path
import sys
import tempfile
import sqlite3

from .client import Client, FetchError
from .jobs import JobQueue, Worker, historical_window
from .storage import Store, StorageError
from . import urls


def positive(value: str) -> float:
    number = float(value)
    if not 0 < number < float('inf'):
        raise argparse.ArgumentTypeError('有限の正数で指定してください。')
    return number


def nonnegative(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError('0以上で指定してください。')
    return number


def iso_date(value: str) -> str:
    try:
        if date.fromisoformat(value).isoformat() != value:
            raise ValueError(value)
        return value
    except ValueError as exc:
        raise argparse.ArgumentTypeError('YYYY-MM-DDで指定してください。') from exc


def identifier(value: str, validator) -> str:
    try:
        return validator(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parser() -> argparse.ArgumentParser:
    app = argparse.ArgumentParser(description='中央・地方のnetkeiba公開データを保存します。', allow_abbrev=False)
    app.add_argument('--data-dir', default='data', help='HTML原本とSQLiteの保存先（既定: data）')
    app.add_argument('--interval', type=positive, default=10, help='全HTTP要求の最小間隔・秒（既定: 10）')
    app.add_argument('--timeout', type=positive, default=30, help='通信タイムアウト・秒（既定: 30）')
    app.add_argument('--refresh', action='store_true', help='ページのキャッシュを使わず新しい観測を保存')
    app.add_argument('--offline', action='store_true', help='保存済みHTMLだけを使い、通信しない')
    commands = app.add_subparsers(dest='command', required=True,
                                  parser_class=lambda **kw: argparse.ArgumentParser(allow_abbrev=False, **kw))
    fetch = commands.add_parser('fetch', help='指定したレースの出馬表・結果を取得')
    fetch.add_argument('--region', choices=('central', 'local'), required=True)
    fetch.add_argument('--race-id', action='append', required=True,
                       type=lambda v: identifier(v, urls.race_id), help='12桁のレースID。複数回指定可')
    fetch.add_argument('--page', choices=('entries', 'results', 'both'), default='results')
    horse = commands.add_parser('horse', help='指定した馬の過去戦績を取得')
    horse.add_argument('--horse-id', action='append', required=True,
                       type=lambda v: identifier(v, urls.horse_id), help='馬ID。複数回指定可')
    listing = commands.add_parser('list', help='指定日のレースIDを表示')
    listing.add_argument('--date', required=True, type=iso_date)
    listing.add_argument('--region', choices=('central', 'local'), required=True)
    backfill = commands.add_parser('backfill', help='期間内の全開催日・全レースを継続収集（同条件で再開）')
    backfill.add_argument('--years', type=int, default=5, help='日本時間で昨日までの過去N年間（既定: 5）')
    backfill.add_argument('--start-date', type=iso_date)
    backfill.add_argument('--end-date', type=iso_date)
    backfill.add_argument('--region', choices=('central', 'local', 'both'), default='both')
    backfill.add_argument('--pages', nargs='+', choices=('results', 'entries'), default=['results'])
    backfill.add_argument('--include-horses', action='store_true', help='結果とは別に馬ページの全戦績も取得')
    backfill.add_argument('--plan-only', action='store_true', help='収集予定を保存し、通信しない')
    backfill.add_argument('--resume', metavar='JOB_ID', help='保存済みジョブを元の期間・条件で再開')
    backfill.add_argument('--max-tasks', type=nonnegative, default=0, help='今回実行するタスク数の上限（0: 制限なし）')
    backfill.add_argument('--max-hours', type=positive, help='今回の実行時間の上限（通信中の処理は完了まで待機）')
    backfill.add_argument('--min-free-mb', type=nonnegative, default=1024, help='空き容量が指定MB未満なら一時停止')
    status = commands.add_parser('status', help='収集進捗・未取得・失敗を表示（通信なし）')
    status.add_argument('--job-id')
    diagnose = commands.add_parser('diagnose', help='失敗全件と原因別件数を出力（通信なし）')
    diagnose.add_argument('--job-id', required=True)
    diagnose.add_argument('--output', type=Path, help='診断結果をUTF-8 JSONファイルに保存')
    diagnose.add_argument('--overwrite', action='store_true')
    retry = commands.add_parser('retry', help='失敗・取得制限のタスクを再実行待ちに戻す（通信なし）')
    retry.add_argument('--job-id', required=True)
    retry.add_argument('--error-type', action='append', help='指定した例外種別だけ戻す（複数指定可）')
    retry.add_argument('--kind', choices=('day', 'race', 'horse'))
    reparse = commands.add_parser('reparse', help='保存済みレース原本を再解析し、取り漏らしを補完（通信なし）')
    reparse.add_argument('--job-id', required=True)
    reparse.add_argument('--max-tasks', type=nonnegative, default=0)
    reparse.add_argument('--min-free-mb', type=nonnegative, default=1024)
    export = commands.add_parser('export', help='保存済みデータをCSV/JSONLへ出力')
    export.add_argument('--dataset', choices=('runners', 'horse-runs'), default='runners')
    export.add_argument('--page', choices=('entries', 'results'))
    export.add_argument('--region', choices=('central', 'local'))
    export.add_argument('--start-date', type=iso_date)
    export.add_argument('--end-date', type=iso_date)
    export.add_argument('--all-snapshots', action='store_true', help='最新だけでなく全取得履歴を出力')
    export.add_argument('--format', choices=('csv', 'jsonl'), default='csv')
    export.add_argument('--output', required=True, type=Path)
    export.add_argument('--overwrite', action='store_true', help='既存の出力ファイルを置き換える')
    return app


def report(**payload) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def output_destination(store: Store, args) -> Path:
    destination = args.output.resolve()
    protected = {store.directory / ('keiba.sqlite3' + suffix) for suffix in ('', '-wal', '-shm', '-journal')}
    if destination in protected or destination.is_relative_to(store.directory / 'raw'):
        raise ValueError('データベースやHTML原本を出力先にできません。')
    if destination.exists() and not args.overwrite:
        raise FileExistsError(f'出力先は既に存在します: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def write_diagnostic(store: Store, args, payload: dict) -> None:
    destination = output_destination(store, args)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=destination.parent,
                                         prefix='.keiba-diagnose-', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            json.dump(payload, file, ensure_ascii=False, indent=2)
            file.write('\n')
            file.flush()
            os.fsync(file.fileno())
        if args.overwrite:
            os.replace(temporary, destination)
        else:
            os.link(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def export_data(store: Store, args) -> int:
    """Two CSV scans share one SQLite snapshot; publication is atomic."""
    options = dict(dataset=args.dataset, page=args.page, region=args.region,
                   all_snapshots=args.all_snapshots, start_date=args.start_date, end_date=args.end_date)
    destination = output_destination(store, args)
    temporary = None
    rows = None
    count = 0
    store.db.execute('BEGIN')
    try:
        fields = store.export_fields(**options) if args.format == 'csv' else None
        rows = store.iter_export_rows(**options)
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                         dir=destination.parent, prefix='.keiba-export-', suffix='.tmp',
                                         delete=False) as file:
            temporary = Path(file.name)
            if args.format == 'csv':
                writer = csv.DictWriter(file, fieldnames=fields)
                writer.writeheader()
            for row in rows:
                if args.format == 'jsonl':
                    file.write(json.dumps(row, ensure_ascii=False) + '\n')
                else:
                    writer.writerow({key: json.dumps(value, ensure_ascii=False)
                                     if isinstance(value, (dict, list)) else value
                                     for key, value in row.items()})
                count += 1
            if not count:
                raise ValueError('出力対象のデータがありません。')
            file.flush()
            os.fsync(file.fileno())
        store.db.commit()
        if args.overwrite:
            os.replace(temporary, destination)
        else:
            # Same-directory hard link publishes a complete file exclusively.
            os.link(temporary, destination)
        return count
    except BaseException:
        store.db.rollback()
        raise
    finally:
        if rows is not None:
            rows.close()
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    app = parser()
    args = app.parse_args(argv)
    if args.offline and args.refresh:
        app.error('--offline と --refresh は同時に指定できません。')
    if args.command == 'export' and args.dataset == 'horse-runs' and (args.page or args.region):
        app.error('horse-runs には --page/--region フィルターはありません。')
    if args.command == 'reparse' and args.refresh:
        app.error('reparse は保存済み原本のみを使います。--refresh は指定できません。')
    if args.command == 'backfill' and args.resume:
        # Avoid silently ignoring changes to a resumed job's immutable scope.
        tokens = argv if argv is not None else sys.argv[1:]
        immutable = ('--start-date', '--end-date', '--years', '--region', '--pages', '--include-horses')
        if any(token.split('=')[0] in immutable for token in tokens):
            app.error('--resume は期間・地域・ページ・馬戦績の条件と同時に指定できません。')
    store, client = None, None
    try:
        from .parsers import parse_race, parse_horse_history

        store = Store(args.data_dir)
        if args.command == 'export':
            count = export_data(store, args)
            report(output=str(args.output), rows=count, dataset=args.dataset)
            return 0
        if args.command in ('backfill', 'status', 'retry', 'diagnose', 'reparse'):
            queue = JobQueue(store)
            if args.command == 'status':
                for summary in queue.summaries(args.job_id):
                    report(**summary)
                return 0
            if args.command == 'diagnose':
                store.db.execute('BEGIN')
                try:
                    diagnosis = queue.diagnostics(args.job_id)
                    store.db.commit()
                except BaseException:
                    store.db.rollback()
                    raise
                if args.output:
                    write_diagnostic(store, args, diagnosis)
                    report(event='diagnosis_saved', output=str(args.output),
                           failed_task_count=diagnosis['failed_task_count'], error_groups=diagnosis['error_groups'])
                else:
                    report(**diagnosis)
                return 0
            if args.command == 'reparse':
                from .maintenance import Reparser
                summary = Reparser(queue, report=report).run(args.job_id, max_tasks=args.max_tasks,
                                                            min_free_mb=args.min_free_mb)
                report(event='reparse_summary', **summary)
                return 2 if summary['failed'] else 0
            if args.command == 'retry':
                count = queue.retry(args.job_id, refresh=args.refresh, error_types=args.error_type, kind=args.kind)
                report(event='retry_scheduled', job_id=args.job_id, tasks=count, refresh=args.refresh)
                return 0
            if args.resume:
                job_id = args.resume
                queue.config(job_id)
            else:
                start, end = historical_window(args.years, start=args.start_date, end=args.end_date)
                regions = ('central', 'local') if args.region == 'both' else (args.region,)
                job_id = queue.plan(start, end, regions=regions, pages=tuple(args.pages),
                                    include_horses=args.include_horses)
            report(event='plan', **queue.summary(job_id))
            if args.plan_only:
                return 0
            client = Client(store, interval=args.interval, timeout=args.timeout, refresh=args.refresh)
            summary = Worker(queue, client, offline=args.offline, report=report).run(
                job_id, max_tasks=args.max_tasks, max_hours=args.max_hours, min_free_mb=args.min_free_mb)
            report(event='summary', **summary)
            if any(s.get('failed') or s.get('blocked') for s in summary['tasks'].values()):
                return 2
            return 0
        client = Client(store, interval=args.interval, timeout=args.timeout, refresh=args.refresh)

        def load(url: str):
            if args.offline:
                observation = store.cached(url)
                if observation is None:
                    raise FetchError(f'オフライン用の保存済みHTMLがありません: {url}')
                return observation, store.body(observation).decode(observation.encoding), True
            return client.fetch(url)

        if args.command == 'fetch':
            pages = ('entries', 'results') if args.page == 'both' else (args.page,)
            for race_id in dict.fromkeys(args.race_id):
                for page in pages:
                    observation, html, cached = load(urls.race_url(race_id, args.region, page))
                    parsed = parse_race(html, race_id=race_id, region=args.region, page=page)
                    store.save_race(observation, parsed)
                    report(race_id=race_id, region=args.region, page=page, runners=len(parsed['runners']),
                           cached=cached, retrieved_at=observation.retrieved_at, snapshot_id=observation.id)
        elif args.command == 'horse':
            for horse_id in dict.fromkeys(args.horse_id):
                observation, html, cached = load(urls.horse_url(horse_id))
                parsed = parse_horse_history(html, horse_id=horse_id)
                store.save_horse(observation, parsed)
                report(horse_id=horse_id, runs=len(parsed['runs']), cached=cached,
                       retrieved_at=observation.retrieved_at, snapshot_id=observation.id)
        elif args.command == 'list':
            observation, html, cached = load(urls.list_url(args.date, args.region))
            report(date=args.date, region=args.region, **urls.discover_day(html, day=args.date, region=args.region),
                   cached=cached, retrieved_at=observation.retrieved_at)
        return 0
    except (FetchError, StorageError, ValueError, OSError, sqlite3.Error) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print('取得を中断しました。保存済みデータと進捗は残っています。', file=sys.stderr)
        return 130
    finally:
        if client:
            client.close()
        if store:
            store.close()
