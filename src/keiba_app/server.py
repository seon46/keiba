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


# Quick lane: one race at a time (odds before predictions); the long whole-day jobs have their own lane,
# so fetching odds or predicting one race never waits behind "predict the whole day".
LANES = dict(odds=('quick', 0), race=('quick', 1))


class Jobs:
    """Two worker threads (quick / long); jobs are kept (newest 100) so the page can poll their state."""

    def __init__(self):
        self.jobs = OrderedDict()
        self.counter = itertools.count(1)
        self.queues = dict(quick=queue.PriorityQueue(), long=queue.PriorityQueue())
        self.lock = threading.Lock()
        for lane in self.queues:
            threading.Thread(target=self._work, args=(lane,), daemon=True).start()

    def submit(self, kind, label, function, **context):
        number = next(self.counter)
        identifier = str(number)
        lane, rank = LANES.get(kind, ('long', 0))
        job = dict(id=identifier, kind=kind, label=label, status='queued', message='待機中', lane=lane, rank=rank,
                   created_at=now().isoformat(timespec='seconds'), result=None, error=None, **context)
        with self.lock:
            self.jobs[identifier] = job
            while len(self.jobs) > 100:
                self.jobs.popitem(last=False)
        self.queues[lane].put((rank, number, identifier, function))
        return dict(job)

    def get(self, identifier):
        with self.lock:
            job = self.jobs.get(identifier)
            return dict(job) if job else None

    def active(self):
        with self.lock:
            return [dict(job) for job in self.jobs.values() if job['status'] in ('queued', 'running')]

    def listing(self, limit=30):
        """Newest first; queued jobs carry their place in the line (1 = next)."""
        with self.lock:
            jobs = [dict(job) for job in self.jobs.values()]
        for lane in self.queues:
            waiting = sorted((job for job in jobs if job['status'] == 'queued' and job.get('lane') == lane),
                             key=lambda job: (job.get('rank', 0), int(job['id'])))
            for position, job in enumerate(waiting, 1):
                job['position'] = position
        return list(reversed(jobs))[:limit]

    def _update(self, identifier, **changes):
        with self.lock:
            if identifier in self.jobs:
                self.jobs[identifier].update(changes)

    def _work(self, lane):
        while True:
            _, _, identifier, function = self.queues[lane].get()
            self._update(identifier, status='running', message='実行中')
            try:
                result = function(lambda **message: self._update(identifier, message=message.get('message', '実行中')))
                self._update(identifier, status='done', message='完了', result=result,
                             finished_at=now().isoformat(timespec='seconds'))
            except Exception as exc:  # shown on the page
                traceback.print_exc()
                self._update(identifier, status='error', message=str(exc), error=str(exc),
                             finished_at=now().isoformat(timespec='seconds'))


def _race_label(region, race_id):
    from .sources import course_of, race_number_of
    try:
        return f'{course_of(region, race_id)}{race_number_of(race_id)}R'
    except (KeyError, ValueError, TypeError):
        return race_id


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
                             seconds=summary.get('seconds'), pipeline=summary.get('pipeline'))
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
        confidence = self.engine.day_confidence(day)
        from .sources import course_of, race_number_of
        races = []
        for (region, race_id), race in listed.items():
            race.setdefault('course', course_of(region, race_id))
            race['race_number'] = race.get('race_number') or race_number_of(race_id)
            race['predictions'] = marks.get(race_id, {}).get('count', 0)
            race['confidence'] = confidence.get(race_id)
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
        saved_odds = self.engine.saved.odds(region, race_id)
        odds = None if not saved_odds else dict(
            win=saved_odds['odds'].get('win', {}), place=saved_odds['odds'].get('place', {}),
            retrieved_at=saved_odds.get('retrieved_at'), official_at=saved_odds.get('official_at'))
        return dict(region=region, race_id=race_id, race=race, card=card, results=finish, predictions=predictions,
                    odds=odds)

    def bets(self, query):
        try:
            identifier = int(query.get('prediction', ''))
            min_hit = min(max(float(query.get('min_hit', .2)), 0), 1)
            max_points = min(max(int(query.get('max_points', 10)), 1), 30)
            budget = min(max(int(query.get('budget', 1000)) // 100 * 100, 100), 1_000_000)
        except ValueError as exc:
            raise ValueError('prediction・min_hit・max_points・budget の指定が不正です。') from exc
        return self.engine.bets(identifier, min_hit=min_hit, max_points=max_points, budget=budget)

    def jobs_overview(self, _query):
        from .nightly import progress
        return dict(jobs=self.jobs.listing(), nightly=progress(self.app), now=now().isoformat(timespec='seconds'))

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
        return self.jobs.submit('race', f'{_race_label(region, race_id)}の予想', work, race_id=race_id, region=region)

    def fetch_odds(self, body):
        region, race_id = body.get('region'), body.get('race_id')
        if region not in ('central', 'local') or not race_id:
            raise ValueError('region と race_id を指定してください。')

        def work(progress):
            progress(message='オッズを取得中')
            return dict(fetched=self.engine.fetch_odds(region, race_id))
        return self.jobs.submit('odds', f'{_race_label(region, race_id)}のオッズ', work, race_id=race_id, region=region)

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

    GET = dict(status=status, day=day, race=race, bets=bets, job=job, jobs=jobs_overview, records=records, log=log,
               changes=changes)
    POST = {'race/predict': predict_race, 'race/odds': fetch_odds, 'day/discover': discover, 'day/predict': predict_day,
            'nightly': nightly}


_DISCONNECTED = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)


class _Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], _DISCONNECTED):
            return
        super().handle_error(request, client_address)


LOCAL_HOSTS = ('127.0.0.1', 'localhost', '[::1]')


def host_allowed(host, extra=()):
    """Only this PC's loopback names, Tailscale names (*.ts.net) and hosts given with --allow-host.
    Rejecting other Host headers stops DNS-rebinding pages from reading or driving the API."""
    if not host:
        return False
    name = host.strip().lower()
    if name.startswith('['):
        name = name[:name.index(']') + 1] if ']' in name else name
    else:
        name = name.rsplit(':', 1)[0] if name.count(':') == 1 else name
    return name in LOCAL_HOSTS or name.endswith('.ts.net') or name in {value.lower() for value in extra}


def make_handler(api: Api, *, allowed_hosts=()):
    class Handler(BaseHTTPRequestHandler):
        server_version = 'KeibaApp/1'

        def _trusted(self, write=False):
            """Requests must name an allowed host; writes must also come from this app's own page."""
            if not host_allowed(self.headers.get('Host'), allowed_hosts):
                self._send(403, dict(error='このアドレスからは利用できません。'))
                return False
            if write:
                origin = self.headers.get('Origin')
                if origin and not host_allowed(urlsplit(origin).netloc, allowed_hosts):
                    self._send(403, dict(error='他のサイトからの操作は受け付けません。'))
                    return False
                if self.headers.get('Sec-Fetch-Site') not in (None, 'same-origin', 'none'):
                    self._send(403, dict(error='他のサイトからの操作は受け付けません。'))
                    return False
                if (self.headers.get('Content-Type') or '').split(';')[0].strip().lower() != 'application/json':
                    # A cross-site form or no-cors fetch cannot send this type without a CORS preflight,
                    # which this server never approves.
                    self._send(415, dict(error='JSONで送ってください。'))
                    return False
            return True

        def log_message(self, format, *args):  # quiet console
            pass

        def _send(self, code, body, content_type='application/json; charset=utf-8'):
            data = body if isinstance(body, bytes) else json.dumps(body, ensure_ascii=False, default=str).encode('utf-8')
            try:
                self.send_response(code)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('X-Content-Type-Options', 'nosniff')
                self.send_header('X-Frame-Options', 'DENY')
                self.send_header('Referrer-Policy', 'no-referrer')
                self.end_headers()
                self.wfile.write(data)
            except _DISCONNECTED:  # the browser left (reload / navigation) before the answer arrived
                self.close_connection = True

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
            if not self._trusted():
                return
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
            if not self._trusted(write=True):
                return
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


def serve(app: App, *, host='127.0.0.1', port=8765, allowed_hosts=()):
    api = Api(app)
    server = _Server((host, port), make_handler(api, allowed_hosts=allowed_hosts))
    print(json.dumps(dict(event='serving', url=f'http://{host}:{port}/'), ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
