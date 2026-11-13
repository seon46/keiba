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


def _tail(path, count, *, chunk=1 << 16):
    """The last ``count`` lines of a (possibly huge) text file, reading only its end."""
    with path.open('rb') as file:
        file.seek(0, 2)
        end = position = file.tell()
        data = b''
        while position > 0 and data.count(b'\n') <= count:
            position = max(0, position - chunk)
            file.seek(position)
            data = file.read(end - position)
    lines = data.decode('utf-8', errors='replace').splitlines()
    if position > 0:
        lines = lines[1:]  # the first line may be cut
    return lines[-count:]


def _race_label(region, race_id):
    from .sources import course_of, race_number_of
    try:
        return f'{course_of(region, race_id)}{race_number_of(race_id)}R'
    except (KeyError, ValueError, TypeError):
        return race_id


def _pick(row, fields):
    return {field: row.get(field) for field in fields if row.get(field) is not None}


class Api:
    def __init__(self, app: App, engine: Engine | None = None, python=None, separate=None):
        self.app = app
        # Whole-day jobs run in their own low-priority process (``keiba_app day`` / ``discover``) so pages and
        # one-race jobs keep the CPU and disk; with an injected engine (tests) they run in this process instead.
        self.separate = engine is None if separate is None else separate
        self.engine = engine or Engine(app)
        self.jobs = Jobs()
        self.python = python or sys.executable
        self.nightly_process = None
        self.request = threading.local()  # who sent the request being handled (set by the handler)
        self._seen = {}  # login -> last time noted as a visitor

    # -- users -------------------------------------------------------------
    def identity(self):
        """Who is asking: dict(login, name, admin, local, user).

        Through Tailscale Serve every request comes from this PC with the Tailscale-User-Login / -Name headers
        set by Tailscale (it removes any a client sends).  A request from this PC without them is this PC's own
        browser: the administrator.  Headers on a request from elsewhere are not trusted."""
        request = getattr(self.request, 'value', None) or {}
        login = request.get('login') if request.get('loopback') else None
        local = bool(request.get('loopback')) and not login
        user = self.engine.wallet.user(login) if login else None
        return dict(login=login, name=(user or {}).get('name') or request.get('name') or login, local=local,
                    admin=local or bool(user and user['admin']), user=user)

    def note_visitor(self):
        """Remember an unregistered Tailscale login (at most every ten minutes) for the administrator's list."""
        import time
        request = getattr(self.request, 'value', None) or {}
        login = request.get('login') if request.get('loopback') else None
        if not login or time.time() - self._seen.get(login, 0) < 600:
            return
        self._seen[login] = time.time()
        if self.engine.wallet.user(login) is None:
            self.engine.wallet.seen(login, request.get('name'))

    def _subject(self, values, *, require=True):
        """The user whose wallet a request is about: yourself, or (administrator) the one named by ``user``."""
        me = self.identity()
        wanted = values.get('user') or None
        if wanted and wanted != me['login']:
            if not me['admin']:
                raise ValueError('他のユーザーの資金は操作できません。')
            if self.engine.wallet.user(wanted) is None:
                raise KeyError('ユーザーがいません。')
            return wanted, me
        if me['user']:
            return me['login'], me
        if require:
            raise ValueError('ユーザーが登録されていません。資金のページで管理者に追加してもらってください。'
                             if me['login'] else 'ユーザーを選んでください。')
        return None, me

    def _child(self, arguments, progress):
        """Run ``python -m keiba_app <arguments>``; its {"event": "progress"} lines update the job, and its
        {"event": "result"} line is returned.  A failure raises with the child's own error message."""
        command = [self.python, '-X', 'utf8', '-m', 'keiba_app', '--data-dir', str(self.app.data_dir), *arguments]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                                   text=True, encoding='utf-8', errors='replace')
        result, error = None, None
        for line in process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if not isinstance(message, dict):
                continue
            if message.get('event') == 'progress':
                progress(message=str(message.get('message') or '実行中'))
            elif message.get('event') == 'result':
                result = message.get('result')
            elif message.get('event') == 'error':
                error = str(message.get('message') or '')
        code = process.wait()
        if code != 0 or result is None:
            raise ValueError(error or f'処理が終了コード{code}で止まりました。')
        return result

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
            bankroll = min(max(int(query.get('bankroll', 50000)), 1000), 100_000_000)
        except ValueError as exc:
            raise ValueError('prediction・min_hit・max_points・budget・bankroll の指定が不正です。') from exc
        source = dict(kind='setting')
        if query.get('wallet') == '1':
            login, _ = self._subject(query, require=False)
            user = self.engine.wallet.user(login) if login else None
            if user:
                balance = int(user['balance'])
                fraction = float(user['kelly_fraction'])
                source = dict(kind='wallet', user=login, name=user['name'], balance=balance, kelly_fraction=fraction)
                if balance >= 1000:
                    # fractional Kelly: stakes of full Kelly on a bankroll of fraction x balance
                    bankroll = max(1000, int(balance * fraction))
                    budget = max(100, min(budget, balance // 100 * 100))
                else:
                    source['too_small'] = True
        result = self.engine.bets(identifier, min_hit=min_hit, max_points=max_points, budget=budget, bankroll=bankroll,
                                  blend=query.get('blend', '1') != '0')
        result['bankroll_source'] = source
        return result

    def me(self, _query):
        me = self.identity()
        users = self.engine.wallet.users() if me['admin'] else ([me['user']] if me['user'] else [])
        return dict(me, users=[dict(login=user['login'], name=user['name'], balance=user['balance'],
                                    kelly_fraction=user['kelly_fraction']) for user in users])

    def wallet(self, query):
        """The funds page: settles what can be settled, then the chosen user's balance history and bets."""
        self.engine.wallet.settle(self.engine.saved)
        login, me = self._subject(query, require=False)
        if login is None and me['admin']:
            users = self.engine.wallet.users()
            login = users[0]['login'] if users else None
        users = self.engine.wallet.users() if me['admin'] else ([me['user']] if me['user'] else [])
        result = dict(me=dict(login=me['login'], name=me['name'], admin=me['admin'], local=me['local'],
                              registered=bool(me['user'])), users=users, user=None,
                      visitors=self.engine.wallet.visitors() if me['admin'] else [])
        if login:
            result.update(user=self.engine.wallet.user(login), bets=self.engine.wallet.bets(login),
                          **self.engine.wallet.history(login))
        return result

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
        include_after = query.get('after') == '1'
        day = query.get('day') or None
        if day == 'latest':  # the last race day with scored predictions
            day = next(iter(self.engine.logbook.scored_days(include_after)), None) or today()
        if day:
            try:
                from datetime import date
                date.fromisoformat(day)
            except ValueError as exc:
                raise ValueError(f'日付の形式が正しくありません: {day}') from exc
            since = None
        result = self.engine.logbook.records(since=since, day=day, region=query.get('region') or None,
                                             include_after=include_after)
        days = self.engine.logbook.scored_days(include_after)
        result['day'] = day
        if day:
            result['previous_day'] = next((other for other in days if other < day), None)
            result['next_day'] = next((other for other in reversed(days) if other > day), None)
        return result

    def log(self, query):
        runs = self.engine.logbook.runs(50)
        run = next((item for item in runs if str(item['id']) == query.get('run')), runs[0] if runs else None)
        if run is None or not run.get('log_path') or not Path(run['log_path']).is_file():
            return dict(run=run, lines=[])
        return dict(run=run, lines=_tail(Path(run['log_path']), 300))

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
            if self.separate:
                return self._child(['discover', '--date', day], progress)
            races = self.engine.discover(day)
            counts = {region: sum(race['region'] == region for race in races) for region in ('central', 'local')}
            return dict(races=len(races), counts=counts, errors=self.engine.discover_errors)
        return self.jobs.submit('discover', f'{day}のレース一覧', work, date=day)

    def predict_day(self, body):
        day = body.get('date') or today()

        def work(progress):
            refresh = bool(body.get('refresh', True))
            if self.separate:
                return self._child(['day', '--date', day] + ([] if refresh else ['--no-refresh']), progress)
            result = self.engine.predict_day(day, refresh=refresh, progress=progress)
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

    def add_user(self, body):
        if not self.identity()['admin']:
            raise ValueError('ユーザーの追加は管理者だけができます。')
        return self.engine.wallet.add_user(body.get('login'), body.get('name'), deposit=body.get('deposit') or 0,
                                           admin=bool(body.get('admin')))

    def forget_visitor(self, body):
        if not self.identity()['admin']:
            raise ValueError('管理者だけが操作できます。')
        self.engine.wallet.forget_visitor(body.get('login'))
        return dict(forgotten=body.get('login'))

    def update_user(self, body):
        login, me = self._subject(body)
        admin = body.get('admin')
        if admin is not None and not me['admin']:
            raise ValueError('管理者の設定は管理者だけが変えられます。')
        return self.engine.wallet.update_user(login, name=body.get('name'), admin=admin,
                                              kelly_fraction=body.get('kelly_fraction'))

    def remove_user(self, body):
        me = self.identity()
        if not me['admin']:
            raise ValueError('ユーザーの削除は管理者だけができます。')
        if body.get('user') == me['login']:
            raise ValueError('自分自身は削除できません。')
        self.engine.wallet.remove_user(body.get('user'))
        return dict(removed=body.get('user'))

    def move_money(self, body):
        login, _ = self._subject(body)
        return dict(balance=self.engine.wallet.move(login, body.get('kind'), body.get('amount'), body.get('note')))

    def delete_entry(self, body):
        login, _ = self._subject(body)
        self.engine.wallet.delete_entry(login, int(body.get('id') or 0))
        return dict(balance=self.engine.wallet.balance(login))

    def record_bet(self, body):
        """Record bought tickets: {prediction_id or region+race_id, tickets: [{bet, key, stake}], label}."""
        login, _ = self._subject(body)
        prediction = None
        if body.get('prediction_id'):
            prediction = self.engine.logbook.prediction(int(body['prediction_id']))
            if prediction is None:
                raise KeyError('予想がありません。')
            race = dict(region=prediction['region'], race_id=prediction['race_id'], race_date=prediction['race_date'],
                        course=prediction['course'], race_number=prediction['race_number'],
                        race_name=prediction['race_name'])
        else:
            region, race_id = body.get('region'), body.get('race_id')
            if region not in ('central', 'local') or not race_id:
                raise ValueError('region と race_id を指定してください。')
            rows = self.engine.saved.snapshot(region, race_id, 'entries') or self.engine.saved.snapshot(
                region, race_id, 'results')
            if not rows:
                raise ValueError('このレースの出馬表がありません。')
            first = rows[0]
            from .sources import course_of, race_number_of
            race = dict(region=region, race_id=race_id, race_date=first.get('race_date'),
                        course=first.get('race_course') or course_of(region, race_id),
                        race_number=first.get('race_number') or race_number_of(race_id),
                        race_name=first.get('race_name'))
            if not race['race_date']:
                raise ValueError('このレースの日付がわかりません。')
        bet = self.engine.wallet.record_bet(login, race, body.get('tickets'), label=body.get('label'),
                                            prediction_id=prediction and prediction['id'])
        self.engine.wallet.settle(self.engine.saved)  # a race already run settles at once
        return dict(bet=self.engine.wallet.bet(bet['id']), balance=self.engine.wallet.balance(login))

    def cancel_bet(self, body):
        login, _ = self._subject(body)
        self.engine.wallet.cancel_bet(login, int(body.get('id') or 0))
        return dict(balance=self.engine.wallet.balance(login))

    def void_bet(self, body):
        login, _ = self._subject(body)
        self.engine.wallet.void_bet(login, int(body.get('id') or 0))
        return dict(balance=self.engine.wallet.balance(login))

    def changes(self, _query):
        """A token that changes whenever predictions, results, day lists, runs, the state or jobs change."""
        import hashlib
        parts = [self.engine.logbook.change_token(), self.engine.wallet.change_token()]
        for path in (self.app.state_path, self.app.home / 'nightly.lock'):
            try:
                parts.append(f'{path.name}:{path.stat().st_mtime_ns}')
            except OSError:
                parts.append(f'{path.name}:-')
        with self.jobs.lock:
            parts.extend(f'{job["id"]}:{job["status"]}' for job in self.jobs.jobs.values())
        return dict(token=hashlib.sha1('|'.join(parts).encode()).hexdigest()[:16])

    GET = dict(status=status, day=day, race=race, bets=bets, job=job, jobs=jobs_overview, records=records, log=log,
               changes=changes, me=me, wallet=wallet)
    POST = {'race/predict': predict_race, 'race/odds': fetch_odds, 'day/discover': discover, 'day/predict': predict_day,
            'nightly': nightly, 'wallet/user': add_user, 'wallet/user/update': update_user,
            'wallet/user/remove': remove_user, 'wallet/visitor/forget': forget_visitor, 'wallet/move': move_money, 'wallet/entry/delete': delete_entry,
            'wallet/bet': record_bet, 'wallet/bet/cancel': cancel_bet, 'wallet/bet/void': void_bet}


_DISCONNECTED = (ConnectionAbortedError, ConnectionResetError, BrokenPipeError)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    # On Windows SO_REUSEADDR lets a second server bind the same port while the first one keeps answering,
    # so a "restarted" app silently keeps running old code.  Take the port exclusively there instead.
    allow_reuse_address = sys.platform != 'win32'

    def server_bind(self):
        import socket
        if sys.platform == 'win32' and hasattr(socket, 'SO_EXCLUSIVEADDRUSE'):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], _DISCONNECTED):
            return
        super().handle_error(request, client_address)


LOCAL_HOSTS = ('127.0.0.1', 'localhost', '[::1]')


def _loopback(address):
    import ipaddress
    try:
        ip = ipaddress.ip_address(str(address).split('%')[0])
    except ValueError:
        return False
    mapped = getattr(ip, 'ipv4_mapped', None)
    return (mapped or ip).is_loopback


def _header_text(value):
    """A header value; Tailscale sends non-ASCII names MIME-encoded (=?utf-8?q?...?=)."""
    if not value:
        return None
    from email.header import decode_header, make_header
    try:
        text = str(make_header(decode_header(value)))
    except (ValueError, LookupError):
        text = value
    return text.strip()[:200] or None


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
            api.request.value = dict(loopback=_loopback(self.client_address[0]),
                                     login=_header_text(self.headers.get('Tailscale-User-Login')),
                                     name=_header_text(self.headers.get('Tailscale-User-Name')))
            try:
                api.note_visitor()
            except Exception:  # never stop a request over the visitor list
                traceback.print_exc()
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
    try:
        server = _Server((host, port), lambda *args: None)
    except OSError as exc:
        raise ValueError(f'ポート{port}はすでに使われています。前に起動したアプリ（タスクスケジューラの「KeibaApp Server」など）'
                         f'が動いたままです。止めてから起動してください。（{exc}）') from exc
    api = Api(app)
    server.RequestHandlerClass = make_handler(api, allowed_hosts=allowed_hosts)
    print(json.dumps(dict(event='serving', url=f'http://{host}:{port}/'), ensure_ascii=False), flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
