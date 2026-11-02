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

STEPS = ('collect', 'prepare', 'score', 'fit', 'precompute')


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


def dataset_info(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + '?mode=ro', uri=True)
    try:
        metadata = dict(connection.execute('SELECT key,value_json FROM ai_metadata'))
        last_day = connection.execute('SELECT MAX(day) FROM features').fetchone()[0]
    finally:
        connection.close()
    return dict(last_day=last_day, feature_revision=json.loads(metadata['feature_revision']))


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
                    code = subprocess_run([python, '-X', 'utf8', '-m', 'keiba_ai.pipeline_v3', 'fit', '--dataset',
                                           state['dataset'], '--output-dir', output, '--threads', state['threads']])
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
