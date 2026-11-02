"""Local HTTP server: static UI and a small JSON API.  Heavy work runs in a single background worker."""
from __future__ import annotations

from collections import OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import itertools
import json
import mimetypes
from pathlib import Path
import queue
import subprocess
import sys
import threading
import traceback
from urllib.parse import parse_qs, urlsplit

from .engine import Engine
from .nightly import dataset_info, model_info
from .state import App, day_after, now, today

STATIC = Path(__file__).parent / 'static'
RACE_FIELDS = ('race_name', 'race_course', 'race_number', 'race_start_time', 'race_surface', 'race_distance_m',
               'race_track_condition', 'race_weather', 'race_class', 'race_grade', 'race_declared_field_size',
               'race_date', 'race_details', 'race_track_direction', 'race_type')
RUNNER_FIELDS = ('horse_id', 'horse_number', 'frame', 'horse_name', 'sex', 'age', 'carried_weight_kg', 'jockey_name',
                 'trainer_name', 'weight_kg', 'weight_change_kg', 'finish_status', 'finish_position', 'time_seconds',
                 'margin', 'popularity')


class Jobs:
    """One worker thread; jobs are kept (newest 100) so the page can poll their state."""

    def __init__(self):
        self.jobs = OrderedDict()
        self.counter = itertools.count(1)
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        threading.Thread(target=self._work, daemon=True).start()

    def submit(self, kind, label, function, **context):
        identifier = str(next(self.counter))
        job = dict(id=identifier, kind=kind, label=label, status='queued', message='待機中',
                   created_at=now().isoformat(timespec='seconds'), result=None, error=None, **context)
        with self.lock:
            self.jobs[identifier] = job
            while len(self.jobs) > 100:
                self.jobs.popitem(last=False)
        self.queue.put((identifier, function))
        return dict(job)

    def get(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            return dict(job) if job else None

    def active(self):
        with self.lock:
            return [dict(job) for job in self.jobs.values() if job['status'] in ('queued', 'running')]

    def _update(self, identifier, **changes):
        with self.lock:
            if identifier in self.jobs:
                self.jobs[identifier].update(changes)

    def _work(self):
        while True:
            identifier, function = self.queue.get()
            self._update(identifier, status='running', message='実行中')
            try:
                result = function(lambda **message: self._update(identifier, message=message.get('message', '実行中')))
                self._update(identifier, status='done', message='完了', result=result,
                             finished_at=now().isoformat(timespec='seconds'))
            except Exception as exc:  # shown on the page
                traceback.print_exc()
                self._update(identifier, status='error', message=str(exc), error=str(exc),
                             finished_at=now().isoformat(timespec='seconds'))


def _pick(row, fields):
    return {field: row.get(field) for field in fields if row.get(field) is not None}


class Api:
    def __init__(self, app: App, engine: Engine | None = None, python=None):
        self.app = app
        self.engine = engine or Engine(app)
        self.jobs = Jobs()
        self.python = python or sys.executable
        self.nightly_process = None

    # -- GET ---------------------------------------------------------------
    def status(self, _query):
        state = self.app.read_state()
        dataset = model = None
        if state.get('dataset') and Path(state['dataset']).is_file():
            try:
                dataset = dataset_info(state['dataset'])
            except Exception as exc:  # shown on the page
                dataset = dict(error=str(exc))
        if state.get('model_dir'):
            summary = model_info(state['model_dir'])
            if summary:
                model = dict(train_end=summary.get('train_end'), created_at=summary.get('created_at'),
                             seconds=summary.get('seconds'))
        running = (self.app.home / 'nightly.lock').exists()
        return dict(today=today(), state=state, dataset=dataset, model=model, runs=self.engine.logbook.runs(10),
                    nightly_running=running, jobs=self.jobs.active())

    def day(self, query):
        day = query.get('date') or today()
        listed = {(race['region'], race['race_id']): dict(race) for race in self.engine.logbook.day_races(day)}
        for key, saved in self.engine.saved.races_on(day).items():
            entry = listed.setdefault(key, dict(region=key[0], race_id=key[1]))
            entry.update({field: value for field, value in saved.items() if value is not None})
        marks = self.engine.logbook.predicted_races(day)
        from .sources import course_of, race_number_of
        races = []
        for (region, race_id), race in listed.items():
            race.setdefault('course', course_of(region, race_id))
            race['race_number'] = race.get('race_number') or race_number_of(race_id)
            race['predictions'] = marks.get(race_id, {}).get('count', 0)
            surface = race.get('surface') or ''
            race['supported'] = race.get('race_type') != 'banei' and '帯広' not in str(race['course']) and '障' not in surface
            races.append(race)
        races.sort(key=lambda race: (race['region'] != 'central', str(race['course']), race['race_number']))
        return dict(date=day, previous=day_after(day, -1), next=day_after(day), races=races,
                    discovered=bool(self.engine.logbook.day_races(day)))

    def race(self, query):
        region, race_id = query.get('region'), query.get('race_id')
        if region not in ('central', 'local') or not race_id:
            raise ValueError('region と race_id を指定してください。')
        entries = self.engine.saved.snapshot(region, race_id, 'entries')
        results = self.engine.saved.snapshot(region, race_id, 'results')
        base = entries or results
        race = _pick(base[0], RACE_FIELDS) if base else {}
        if entries:
            race['entries_retrieved_at'] = entries[0].get('retrieved_at')
        card = [_pick(row, RUNNER_FIELDS) for row in (entries or results or [])]
        finish = {str(row.get('horse_id')): _pick(row, ('finish_status', 'finish_position', 'time_seconds', 'margin',
                                                         'popularity'))
                  for row in results} if results else None
        predictions = self.engine.logbook.predictions_for_race(race_id)
        if not race and predictions:
            latest = predictions[0]
            race = dict(race_name=latest['race_name'], race_course=latest['course'], race_number=latest['race_number'],
                        race_start_time=latest['start_time'], race_surface=latest['surface'],
                        race_distance_m=latest['distance_m'], race_track_condition=latest['track_condition'],
                        race_date=latest['race_date'])
        return dict(region=region, race_id=race_id, race=race, card=card, results=finish, predictions=predictions)

    def job(self, query):
        job = self.jobs.get(query.get('id', ''))
        if job is None:
            raise KeyError('ジョブがありません。')
        return job

    def records(self, query):
        days = query.get('days')
        since = day_after(today(), -int(days)) if days and days != 'all' else None
        return self.engine.logbook.records(since=since, region=query.get('region') or None,
                                           include_after=query.get('after') == '1')

    def log(self, query):
        runs = self.engine.logbook.runs(50)
        run = next((item for item in runs if str(item['id']) == query.get('run')), runs[0] if runs else None)
        if run is None or not run.get('log_path') or not Path(run['log_path']).is_file():
            return dict(run=run, lines=[])
        lines = Path(run['log_path']).read_text(encoding='utf-8', errors='replace').splitlines()
        return dict(run=run, lines=lines[-300:])

    # -- POST --------------------------------------------------------------
    def predict_race(self, body):
        region, race_id = body.get('region'), body.get('race_id')
        refresh = bool(body.get('refresh'))
        if region not in ('central', 'local') or not race_id:
            raise ValueError('region と race_id を指定してください。')

        def work(progress):
            progress(message='最新の出馬表を取得中' if refresh else '出馬表を準備中')
            identifier = self.engine.predict_race(region, race_id, refresh=refresh)
            return dict(prediction_id=identifier)
        return self.jobs.submit('race', f'{race_id}の予想', work, race_id=race_id)

    def discover(self, body):
        day = body.get('date') or today()

        def work(progress):
            progress(message='レース一覧を取得中')
            races = self.engine.discover(day)
            counts = {region: sum(race['region'] == region for race in races) for region in ('central', 'local')}
            return dict(races=len(races), counts=counts, errors=self.engine.discover_errors)
        return self.jobs.submit('discover', f'{day}のレース一覧', work, date=day)

    def predict_day(self, body):
        day = body.get('date') or today()

        def work(progress):
            result = self.engine.predict_day(day, refresh=bool(body.get('refresh', True)), progress=progress)
            return dict(predicted=len(result['predicted']), skipped=result['skipped'])
        return self.jobs.submit('day', f'{day}の全レース', work, date=day)

    def nightly(self, body):
        if (self.app.home / 'nightly.lock').exists() or (
                self.nightly_process is not None and self.nightly_process.poll() is None):
            raise ValueError('更新処理はすでに実行中です。')
        arguments = [self.python, '-X', 'utf8', '-m', 'keiba_app', '--data-dir', str(self.app.data_dir), 'nightly']
        if body.get('force_fit'):
            arguments.append('--force-fit')
        if body.get('only'):
            arguments += ['--only'] + list(body['only'])
        self.nightly_process = subprocess.Popen(arguments, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dict(started=True)

    def changes(self, _query):
        """A token that changes whenever predictions, results, day lists, runs, the state or jobs change."""
        import hashlib
        parts = [self.engine.logbook.change_token()]
        for path in (self.app.state_path, self.app.home / 'nightly.lock'):
            try:
                parts.append(f'{path.name}:{path.stat().st_mtime_ns}')
            except OSError:
                parts.append(f'{path.name}:-')
        with self.jobs.lock:
            parts.extend(f'{job["id"]}:{job["status"]}' for job in self.jobs.jobs.values())
        return dict(token=hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16])

    GET = dict(status=status, day=day, race=race, job=job, records=records, log=log, changes=changes)
    POST = {'race/predict': predict_race, 'day/discover': discover, 'day/predict': predict_day, 'nightly': nightly}


def make_handler(api: Api):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'KeibaApp/1'

        def log_message(self, format, *args):  # quiet console
            pass

        def _send(self, code, body, content_type='application/json; charset=utf-8'):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode('utf-8')
            self.send_response(code)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

        def _api(self, table, name, argument):
            function = table.get(name)
            if function is None:
                return self._send(404, dict(error='見つかりません。'))
            try:
                return self._send(200, function(api, argument))
            except (ValueError, KeyError) as exc:
                return self._send(400, dict(error=str(exc).strip("'\"")))
            except Exception as exc:  # shown on the page
                traceback.print_exc()
                return self._send(500, dict(error=f'内部エラー: {exc}'))

        def do_GET(self):
            parts = urlsplit(self.path)
            if parts.path.startswith('/api/'):
                query = {key: values[-1] for key, values in parse_qs(parts.query).items()}
                return self._api(Api.GET, parts.path[5:], query)
            name = 'index.html' if parts.path in ('/', '/index.html') else parts.path.lstrip('/')
            if name.startswith('static/'):
                name = name[7:]
            path = (STATIC / name).resolve()
            if STATIC.resolve() not in path.parents or not path.is_file():
                return self._send(404, b'not found', 'text/plain; charset=utf-8')
            content_type = mimetypes.guess_type(path.name)[0] or 'application/octet-stream'
            if content_type.startswith('text/') or content_type.endswith('javascript'):
                content_type += '; charset=utf-8'
            return self._send(200, path.read_bytes(), content_type)

        def do_POST(self):
            parts = urlsplit(self.path)
            if not parts.path.startswith('/api/'):
                return self._send(404, dict(error='見つかりません。'))
            length = int(self.headers.get('Content-Length') or 0)
            try:
                body = json.loads(self.rfile.read(length) or b'{}')
            except ValueError:
                return self._send(400, dict(error='JSONが不正です。'))
            return self._api(Api.POST, parts.path[5:], body)

    return Handler


def serve(app: App, *, host='127.0.0.1', port=8765):
    api = Api(app)
    server = ThreadingHTTPServer((host, port), make_handler(api))
    print(json.dumps(dict(event='serving', url=f'http://{host}:{port}/'), ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
