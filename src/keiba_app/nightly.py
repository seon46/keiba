"""Nightly update: collect results, rebuild the database, score predictions, retrain weekly, predict today."""
from __future__ import annotations

from datetime import date
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import time

from .engine import Engine
from .state import App, day_after, now, today

STEPS = ('collect', 'pedigree', 'prepare', 'score', 'fit', 'precompute')
PEDIGREE_DAYS = 60  # look for new horses in races of this many recent days


class Busy(RuntimeError):
    pass


class _Lock:
    def __init__(self, path, stale_hours=12):
        self.path, self.stale = Path(path), stale_hours * 3600

    def __enter__(self):
        if self.path.exists() and time.time() - self.path.stat().st_mtime > self.stale:
            self.path.unlink(missing_ok=True)
        try:
            handle = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            raise Busy('別の更新処理が実行中です。') from exc
        with os.fdopen(handle, 'w') as file:
            file.write(str(os.getpid()))
        return self

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


_DATASET_INFO = {}


def _index_days(dataset):
    """Index features by day so the status page's MAX(day) is instant.  Done here rather than in keiba_ai.features:
    that file's bytes are part of the feature revision, so changing it would invalidate every prepared database."""
    try:
        connection = sqlite3.connect(dataset, timeout=60)
        try:
            connection.execute('CREATE INDEX IF NOT EXISTS features_day ON features(day)')
            connection.commit()
        finally:
            connection.close()
    except sqlite3.Error:
        pass


def dataset_info(path):
    """Last day and feature revision of a prepared dataset.  Without an index on ``day`` the MAX() reads the
    whole (multi-GB) table, so the answer is cached until the file changes."""
    path = Path(path).resolve()
    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size)
    cached = _DATASET_INFO.get(str(path))
    if cached and cached[0] == key:
        return dict(cached[1])
    connection = sqlite3.connect(path.as_uri() + '?mode=ro', uri=True)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
        last_day = connection.execute('SELECT MAX(day) FROM features').fetchone()[0]
    finally:
        connection.close()
    info = dict(last_day=last_day, feature_revision=json.loads(metadata['feature_revision']))
    _DATASET_INFO[str(path)] = (key, info)
    return dict(info)


def model_info(model_dir):
    try:
        return json.loads((Path(model_dir) / 'model.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def check_model(summary):
    """Reasons a freshly fitted model must not replace the current one (empty = fine)."""
    problems = []
    if not summary:
        return ['model.jsonがありません。']
    for group in ('central_flat', 'local_flat'):
        result = summary.get('groups', {}).get(group)
        if not result:
            problems.append(f'{group}のモデルがありません。')
            continue
        if (result.get('best_iteration') or 0) < 20:
            problems.append(f'{group}: 学習の反復回数が少なすぎます（{result.get("best_iteration")}）。')
        temperatures = result.get('temperatures') or []
        if not temperatures or any(not (.2 < value < 6) for value in temperatures):
            problems.append(f'{group}: 温度が不正です。')
        if summary.get('pipeline') == 'v4-pipeline-1':
            weights = result.get('blend_weights') or []
            if len(weights) != 2 or not all(0 <= value <= 3 for value in weights) or sum(weights) <= 0:
                problems.append(f'{group}: 混ぜ合わせの重みが不正です。')
            if not result.get('networks') or any((network.get('best_epoch') or 0) < 1 for network in result['networks']):
                problems.append(f'{group}: ニューラルネットの学習に失敗しています。')
    return problems


def _prune(folder, keep, protected):
    protected = {Path(path).resolve() for path in protected if path}
    candidates = sorted((path for path in Path(folder).iterdir() if path.is_dir() and not path.name.startswith('.')),
                        key=lambda path: path.name, reverse=True)
    kept = 0
    for path in candidates:
        inside = {protected_path for protected_path in protected if protected_path == path or path in protected_path.parents}
        if inside or kept < keep:
            kept += 1
            continue
        shutil.rmtree(path, ignore_errors=True)


def run_nightly(app: App, *, force_fit=False, only=None, skip=(), python=None, engine=None, log=None):
    """Run the nightly steps; returns the list of step records."""
    python = python or sys.executable
    stamp = now().strftime('%Y%m%d-%H%M%S')
    log_path = app.home / 'logs' / f'nightly-{stamp}.log'
    log_file = log_path.open('a', encoding='utf-8')

    def write(line):
        log_file.write(f'{now().strftime("%H:%M:%S")} {line}\n')
        log_file.flush()
        if log:
            log(line)

    def subprocess_run(arguments):
        write('$ ' + ' '.join(str(argument) for argument in arguments))
        process = subprocess.Popen([str(argument) for argument in arguments], stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
        for line in process.stdout:
            write(line.rstrip('\n'))
        return process.wait()

    engine = engine or Engine(app, report=lambda **message: write(json.dumps(message, ensure_ascii=False)))
    logbook = engine.logbook
    run_id = logbook.start_run('nightly', now().isoformat(timespec='seconds'), log_path)
    steps = []
    selected = [step for step in STEPS if (only is None or step in only) and step not in skip]

    def record(name, status, started, **detail):
        steps.append(dict(step=name, status=status, seconds=round(time.time() - started, 1), **detail))
        logbook.update_run(run_id, steps=steps)
        write(f'[{name}] {status} {json.dumps(detail, ensure_ascii=False)}')

    overall = 'ok'
    try:
        with _Lock(app.home / 'nightly.lock'):
            state = app.require_ready()
            yesterday = day_after(today(), -1)
            current = dataset_info(state['dataset'])
            collected = state['collected_through'] or current['last_day']
            end = None
            # 1. collect results ------------------------------------------------
            if 'collect' in selected:
                started = time.time()
                start = day_after(collected)
                if start > yesterday:
                    record('collect', 'skipped', started, reason='収集済み', collected_through=collected)
                else:
                    code = subprocess_run([python, '-X', 'utf8', '-m', 'keiba_scraper', '--data-dir', app.data_dir,
                                           'backfill', '--start-date', start, '--end-date', yesterday,
                                           '--region', 'both'])
                    end = yesterday
                    if code == 0:
                        collected = yesterday
                        app.write_state(collected_through=collected)
                        record('collect', 'ok', started, start=start, end=yesterday)
                    else:
                        overall = 'warning'
                        record('collect', 'incomplete', started, start=start, end=yesterday, exit_code=code)
            # 1b. pedigrees of new horses (the v4 model uses them) -------------
            if 'pedigree' in selected:
                started = time.time()
                missing = engine.saved.missing_pedigrees(since=day_after(today(), -PEDIGREE_DAYS))
                if not missing:
                    record('pedigree', 'skipped', started, reason='新しい馬はいません')
                else:
                    failed = engine.scraper.pedigrees(missing, log=write)
                    left = len(engine.saved.missing_pedigrees(missing))
                    if failed:
                        overall = 'warning' if overall == 'ok' else overall
                    record('pedigree', 'incomplete' if failed else 'ok', started, horses=len(missing), missing=left)
            # 2. export + prepare ----------------------------------------------
            if 'prepare' in selected:
                started = time.time()
                state = app.read_state()
                target = end or collected
                prepared = state.get('prepared_through') or current['last_day']
                if prepared and target and prepared >= target:
                    record('prepare', 'skipped', started, reason='DBは最新', last_day=current['last_day'])
                else:
                    csv_path = app.home / 'db' / f'results-{stamp}.csv'
                    output = app.home / 'db' / stamp
                    code = subprocess_run([python, '-X', 'utf8', '-m', 'keiba_scraper', '--data-dir', app.data_dir,
                                           'export', '--page', 'results', '--start-date', state['data_start'],
                                           '--end-date', target, '--output', csv_path, '--overwrite'])
                    if code == 0:
                        code = subprocess_run([python, '-X', 'utf8', '-m', 'keiba_ai', 'prepare', '--input', csv_path,
                                               '--output-dir', output, '--race-db', app.scraper_db])
                    csv_path.unlink(missing_ok=True)
                    dataset = output / 'features.sqlite3'
                    if code == 0 and dataset.is_file():
                        _index_days(dataset)
                        previous = state['dataset']
                        current = dataset_info(dataset)
                        app.write_state(dataset=str(dataset), previous_dataset=previous, prepared_through=target)
                        engine.reset()
                        _prune(app.home / 'db', state['keep_databases'], [dataset, previous])
                        record('prepare', 'ok', started, dataset=str(dataset), last_day=current['last_day'])
                    else:
                        overall = 'error'
                        shutil.rmtree(output, ignore_errors=True)
                        record('prepare', 'error', started, exit_code=code)
            # 3. score logged predictions --------------------------------------
            if 'score' in selected:
                started = time.time()
                record('score', 'ok', started, counts=engine.score_pending())
            # 4. retrain -------------------------------------------------------
            if 'fit' in selected:
                started = time.time()
                state = app.read_state()
                model = model_info(state['model_dir'])
                stale = model is None or model.get('feature_revision') != current['feature_revision']
                due = state['last_fit'] is None or (
                    date.fromisoformat(today()) - date.fromisoformat(state['last_fit'])).days >= state['retrain_days']
                if not (force_fit or due or stale):
                    record('fit', 'skipped', started, reason='学習し直しの日ではありません', last_fit=state['last_fit'])
                else:
                    output = app.home / 'models' / stamp
                    if (model or {}).get('pipeline') == 'v4-pipeline-1':
                        command = ['keiba_ai.pipeline_v4', 'fit', '--race-db', app.scraper_db]
                        if state.get('train_start'):
                            command += ['--train-start', state['train_start']]
                    else:
                        command = ['keiba_ai.pipeline_v3', 'fit']
                    code = subprocess_run([python, '-X', 'utf8', '-m', *command, '--dataset', state['dataset'],
                                           '--output-dir', output, '--threads', state['threads']])
                    summary = model_info(output) if code == 0 else None
                    problems = check_model(summary) if code == 0 else [f'終了コード{code}']
                    if problems:
                        overall = 'error'
                        shutil.rmtree(output, ignore_errors=True)
                        record('fit', 'error', started, problems=problems)
                    else:
                        previous = state['model_dir']
                        app.write_state(model_dir=str(output), previous_model_dir=previous, last_fit=today())
                        engine.reset()
                        _prune(app.home / 'models', state['keep_models'], [output, previous])
                        record('fit', 'ok', started, model_dir=str(output), train_end=summary['train_end'])
            # 5. predict today's races -----------------------------------------
            if 'precompute' in selected:
                started = time.time()
                try:
                    engine.discover(today())
                    result = engine.predict_day(today(), origin='nightly',
                                                progress=lambda **message: write(message.get('message', '')))
                    record('precompute', 'ok', started, predicted=len(result['predicted']),
                           skipped=len(result['skipped']))
                except (ValueError, OSError, sqlite3.Error) as exc:
                    overall = 'warning' if overall == 'ok' else overall
                    record('precompute', 'error', started, message=str(exc)[:300])
    except Busy as exc:
        overall = 'busy'
        write(str(exc))
    except Exception as exc:  # recorded for the status page, then re-raised
        overall = 'error'
        write(f'ERROR: {exc!r}')
        logbook.update_run(run_id, status='error', steps=steps, finished_at=now().isoformat(timespec='seconds'))
        log_file.close()
        raise
    logbook.update_run(run_id, status=overall, steps=steps, finished_at=now().isoformat(timespec='seconds'))
    log_file.close()
    return dict(status=overall, steps=steps, log=str(log_path))


def progress(app: App, *, tail_bytes=1_500_000):
    """What the running nightly update is doing, read from its log (None when it is not running)."""
    if not (app.home / 'nightly.lock').exists():
        return None
    logs = sorted((app.home / 'logs').glob('nightly-*.log'), key=lambda path: path.stat().st_mtime)
    if not logs:
        return dict(step=None, message='開始しています')
    path = logs[-1]
    with path.open('rb') as file:
        size = file.seek(0, 2)
        file.seek(max(0, size - tail_bytes))
        lines = file.read().decode('utf-8', errors='replace').splitlines()
    step, plan, saved, last = None, None, 0, None
    names = {'backfill': '成績の収集', 'export': 'DBの作り直し', 'prepare': 'DBの作り直し', 'fit': '学習し直し'}
    for line in lines:
        text = line[9:] if len(line) > 9 and line[2] == ':' else line
        if not text.strip():
            continue
        last = line
        if text.startswith('$ '):
            for word, name in names.items():
                if f' {word} ' in text + ' ':
                    step, plan, saved = name, None, 0
                    break
            continue
        if text.startswith('[') and ']' in text:
            done = text[1:text.index(']')]
            step = {'collect': '血統の収集', 'pedigree': 'DBの作り直し', 'prepare': '予想の照合', 'score': '学習し直し',
                    'fit': '当日の先行予想'}.get(done, step)
            plan, saved = None, 0
            continue
        if text.startswith('{'):
            try:
                message = json.loads(text)
            except ValueError:
                continue
            if message.get('event') == 'plan':
                plan, saved = message, 0
                if (message.get('config') or {}).get('type') == 'pedigree':
                    step = '血統の収集'
            elif message.get('event') == 'saved':
                saved += 1
        elif '出馬表を取得中' in text or 'レースを予想中' in text:
            step = '当日の先行予想'
    result = dict(step=step, log=str(path), last_line=_describe(last), last_at=(last or '')[:8] if last else None)
    if plan is not None:
        if plan.get('total_horses') is not None:
            total = plan.get('remaining_horses') or 0
            result.update(done=saved, total=total, unit='頭')
        else:
            result.update(done=saved, total=None, unit='ページ')
    return result


def _describe(line):
    """A log line in words (the time, then what was saved)."""
    if not line:
        return ''
    stamp, text = (line[:8], line[9:]) if len(line) > 9 and line[2] == ':' else ('', line)
    try:
        message = json.loads(text) if text.startswith('{') else None
    except ValueError:
        message = None
    if isinstance(message, dict) and message.get('event') == 'saved':
        kind = message.get('kind')
        if kind == 'pedigree':
            text = f'血統を保存（馬ID {message.get("horse_id")}）'
        elif kind in ('race', 'day'):
            text = f'{"レース" if kind == "race" else "開催日"}を保存（{message.get("race_id") or message.get("day") or ""}）'
        else:
            text = f'{kind}を保存'
    elif isinstance(message, dict):
        text = message.get('event') or text
    return f'{stamp} {text}'.strip()[:200]
