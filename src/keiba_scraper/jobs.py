"""Durable historical collection; discoveries and completion share a transaction."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import shutil
import sqlite3
import threading
import time
import uuid
from zoneinfo import ZoneInfo

from . import urls
from .client import AccessRestricted, Client, FetchError
from .parsers import parse_horse_history, parse_race
from .storage import Store, StorageError


JST = ZoneInfo("Asia/Tokyo")


def historical_window(years: int = 5, *, today: date | None = None,
                      start: str | None = None, end: str | None = None) -> tuple[str, str]:
    """Inclusive rolling calendar years ending yesterday in Japan."""
    if not 1 <= years <= 50:
        raise ValueError("years は1〜50で指定してください。")
    today = today or datetime.now(JST).date()
    finish = date.fromisoformat(end) if end else today - timedelta(days=1)
    anchor = finish + timedelta(days=1)
    try:
        begin = anchor.replace(year=anchor.year - years)
    except ValueError:  # February 29 -> last valid day in the target year.
        begin = anchor.replace(year=anchor.year - years, day=28)
    if start:
        begin = date.fromisoformat(start)
    if begin > finish or finish >= today:
        raise ValueError("期間は開始日<=終了日、終了日は日本時間の昨日以前にしてください。")
    return begin.isoformat(), finish.isoformat()


@dataclass(frozen=True)
class Task:
    id: int
    kind: str
    region: str | None
    day: str | None
    race_id: str | None
    page: str | None
    horse_id: str | None
    attempts: int
    force_refresh: bool
    owner: str


class JobQueue:
    def __init__(self, store: Store):
        self.store, self.db = store, store.db
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS collection_jobs (
                job_id TEXT PRIMARY KEY, created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                start_date TEXT NOT NULL, end_date TEXT NOT NULL, config_json TEXT NOT NULL,
                state TEXT NOT NULL DEFAULT 'planned'
            );
            CREATE TABLE IF NOT EXISTS collection_tasks (
                id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES collection_jobs(job_id),
                kind TEXT NOT NULL, task_key TEXT NOT NULL, region TEXT, day TEXT,
                race_id TEXT, page TEXT, horse_id TEXT,
                state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                available_at REAL NOT NULL DEFAULT 0, owner TEXT, lease_expires REAL,
                force_refresh INTEGER NOT NULL DEFAULT 0,
                last_error TEXT, error_type TEXT, snapshot_id TEXT REFERENCES observations(id),
                items INTEGER NOT NULL DEFAULT 0, result_json TEXT,
                UNIQUE(job_id, kind, task_key)
            );
            CREATE INDEX IF NOT EXISTS collection_tasks_ready
                ON collection_tasks(job_id,state,available_at,day);
            CREATE TABLE IF NOT EXISTS collection_races (
                job_id TEXT NOT NULL REFERENCES collection_jobs(job_id), region TEXT NOT NULL,
                race_id TEXT NOT NULL, day TEXT NOT NULL, expected_field_size INTEGER,
                PRIMARY KEY(job_id,region,race_id)
            );
        """)
        # Additive migration for a collection DB created by an earlier version.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            columns = {row["name"] for row in self.db.execute("PRAGMA table_info(collection_races)")}
            if "expected_field_size" not in columns:
                self.db.execute("ALTER TABLE collection_races ADD COLUMN expected_field_size INTEGER")
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise

    def plan(self, start: str, end: str, *, regions: tuple[str, ...] = ("central", "local"),
             pages: tuple[str, ...] = ("results",), include_horses: bool = False) -> str:
        begin, finish = date.fromisoformat(start), date.fromisoformat(end)
        if begin > finish:
            raise ValueError("開始日は終了日以前にしてください。")
        regions, pages = tuple(sorted(set(regions))), tuple(sorted(set(pages)))
        if not regions or not set(regions) <= {"central", "local"}:
            raise ValueError("不正な地域です。")
        if not pages or not set(pages) <= {"results", "entries"}:
            raise ValueError("不正なページ種別です。")
        config = dict(version=1, start_date=start, end_date=end, regions=regions,
                      pages=pages, include_horses=include_horses)
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
        job_id = hashlib.sha256(encoded.encode()).hexdigest()[:16]
        now = datetime.now(timezone.utc).isoformat()
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO collection_jobs VALUES (?,?,?,?,?,?,?)",
                            (job_id, now, now, start, end, encoded, "planned"))
            days = (begin + timedelta(days=i) for i in range((finish - begin).days + 1))
            self.db.executemany(
                "INSERT OR IGNORE INTO collection_tasks(job_id,kind,task_key,region,day) "
                "VALUES (?,'day',?,?,?)",
                ((job_id, f"{region}:{day.isoformat()}", region, day.isoformat())
                 for day in days for region in regions),
            )
        return job_id

    def config(self, job_id: str) -> dict:
        row = self.db.execute("SELECT config_json FROM collection_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError(f"収集ジョブがありません: {job_id}")
        return json.loads(row[0])

    def claim(self, job_id: str, owner: str, *, now: float | None = None,
              lease_seconds: float = 900) -> Task | None:
        self.config(job_id)
        now = time.time() if now is None else now
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self.db.execute(
                "UPDATE collection_tasks SET state='pending',owner=NULL,lease_expires=NULL "
                "WHERE job_id=? AND state='running' AND lease_expires<=?", (job_id, now))
            row = self.db.execute(
                "SELECT * FROM collection_tasks WHERE job_id=? AND state='pending' AND available_at<=? "
                "ORDER BY COALESCE(day,'9999-12-31'),CASE kind WHEN 'day' THEN 0 WHEN 'race' THEN 1 ELSE 2 END,id LIMIT 1",
                (job_id, now)).fetchone()
            if row is None:
                self.db.commit()
                return None
            self.db.execute(
                "UPDATE collection_tasks SET state='running',attempts=attempts+1,owner=?,lease_expires=? WHERE id=?",
                (owner, now + lease_seconds, row["id"]))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
        return Task(row["id"], row["kind"], row["region"], row["day"], row["race_id"],
                    row["page"], row["horse_id"], row["attempts"] + 1,
                    bool(row["force_refresh"]), owner)

    def _owned(self, task: Task) -> str:
        row = self.db.execute(
            "SELECT job_id FROM collection_tasks WHERE id=? AND state='running' AND owner=?",
            (task.id, task.owner)).fetchone()
        if row is None:
            raise StorageError("タスクの所有権を失いました。他の収集プロセスを確認してください。")
        return row[0]

    def complete_day(self, task: Task, observation, discovery: dict) -> None:
        with self.db:
            job_id = self._owned(task)
            config = self.config(job_id)
            for race_id in discovery["race_ids"]:
                existing = self.db.execute(
                    "SELECT day FROM collection_races WHERE job_id=? AND region=? AND race_id=?",
                    (job_id, task.region, race_id)).fetchone()
                if existing and existing[0] != task.day:
                    raise ValueError("同じレースIDが異なる開催日に掲載されています。")
                self.db.execute("INSERT OR IGNORE INTO collection_races "
                                "(job_id,region,race_id,day,expected_field_size) VALUES (?,?,?,?,?)",
                                (job_id, task.region, race_id, task.day,
                                 discovery.get("field_sizes", {}).get(race_id)))
                self.db.executemany(
                    "INSERT OR IGNORE INTO collection_tasks(job_id,kind,task_key,region,day,race_id,page) "
                    "VALUES (?,'race',?,?,?,?,?)",
                    ((job_id, f"{task.region}:{race_id}:{page}", task.region, task.day, race_id, page)
                     for page in config["pages"]))
            self._finish(task, "done", snapshot_id=observation.id,
                         items=len(discovery["race_ids"]), result=discovery)

    def complete_race(self, task: Task, observation, parsed: dict) -> None:
        with self.db:
            job_id = self._owned(task)
            if self.config(job_id)["include_horses"]:
                self.db.executemany(
                    "INSERT OR IGNORE INTO collection_tasks(job_id,kind,task_key,horse_id) VALUES (?,'horse',?,?)",
                    ((job_id, runner["horse_id"], runner["horse_id"]) for runner in parsed["runners"]))
            self._finish(task, "done", snapshot_id=observation.id, items=len(parsed["runners"]),
                         result={"race_date": parsed["race"]["race_date"],
                                 "race_type": parsed["race"].get("race_type")})

    def complete_horse(self, task: Task, observation, parsed: dict) -> None:
        with self.db:
            self._owned(task)
            self._finish(task, "done", snapshot_id=observation.id, items=len(parsed["runs"]))

    def _finish(self, task: Task, state: str, *, snapshot_id=None, items=0, result=None,
                error: Exception | None = None, available_at: float = 0) -> None:
        self._owned(task)
        cursor = self.db.execute(
            "UPDATE collection_tasks SET state=?,owner=NULL,lease_expires=NULL,snapshot_id=?,items=?,result_json=?,"
            "last_error=?,error_type=?,available_at=?,force_refresh=CASE WHEN ?='done' THEN 0 ELSE force_refresh END "
            "WHERE id=? AND state='running' AND owner=?",
            (state, snapshot_id, items, json.dumps(result, ensure_ascii=False) if result is not None else None,
             str(error)[:2000] if error else None, type(error).__name__ if error else None,
             available_at, state, task.id, task.owner))
        if cursor.rowcount != 1:
            raise StorageError("タスクの所有権を失いました。取得結果の完了状態は更新しません。")

    def fail(self, task: Task, error: Exception, *, state: str = "failed", delay: float = 0) -> None:
        with self.db:
            self._finish(task, state, error=error, available_at=time.time() + delay)

    def release(self, task: Task) -> None:
        with self.db:
            self._finish(task, "pending")

    def set_state(self, job_id: str, state: str) -> None:
        with self.db:
            self.db.execute("UPDATE collection_jobs SET state=?,updated_at=? WHERE job_id=?",
                            (state, datetime.now(timezone.utc).isoformat(), job_id))

    def retry(self, job_id: str, *, refresh: bool = False) -> int:
        self.config(job_id)
        with self.db:
            cursor = self.db.execute(
                "UPDATE collection_tasks SET state='pending',attempts=0,available_at=0,force_refresh=?,"
                "last_error=NULL,error_type=NULL WHERE job_id=? AND state IN ('failed','blocked')",
                (int(refresh), job_id))
        return cursor.rowcount

    def next_wait(self, job_id: str) -> float | None:
        row = self.db.execute(
            "SELECT MIN(CASE WHEN state='pending' THEN available_at ELSE lease_expires END) "
            "FROM collection_tasks WHERE job_id=? AND state IN ('pending','running')", (job_id,)).fetchone()
        return max(0.1, row[0] - time.time()) if row[0] is not None else None

    def summaries(self, job_id: str | None = None) -> list[dict]:
        if job_id:
            self.config(job_id)
        rows = self.db.execute("SELECT * FROM collection_jobs " + ("WHERE job_id=? " if job_id else "") +
                               "ORDER BY created_at", (job_id,) if job_id else ()).fetchall()
        return [self.summary(row["job_id"]) for row in rows]

    def summary(self, job_id: str) -> dict:
        config = self.config(job_id)
        counts: dict[str, dict[str, int]] = {}
        for row in self.db.execute(
                "SELECT kind,state,COUNT(*) AS count FROM collection_tasks WHERE job_id=? GROUP BY kind,state", (job_id,)):
            counts.setdefault(row["kind"], {})[row["state"]] = row["count"]
        expected_days = (date.fromisoformat(config["end_date"]) - date.fromisoformat(config["start_date"])).days + 1
        expected_days *= len(config["regions"])
        outstanding = sum(n for states in counts.values() for state, n in states.items() if state != "done")
        days_done = counts.get("day", {}).get("done", 0)
        complete = outstanding == 0 and days_done == expected_days
        problems = [dict(row) for row in self.db.execute(
            "SELECT id,kind,region,day,race_id,page,horse_id,state,last_error,error_type,attempts "
            "FROM collection_tasks WHERE job_id=? AND state IN ('failed','blocked') ORDER BY id LIMIT 20", (job_id,))]
        unverified_empty = self.db.execute(
            "SELECT COUNT(*) FROM collection_tasks WHERE job_id=? AND kind='day' AND state='done' "
            "AND items=0 AND json_extract(result_json,'$.day_verified')=0", (job_id,)).fetchone()[0]
        discovered = self.db.execute("SELECT COUNT(*) FROM collection_races WHERE job_id=?", (job_id,)).fetchone()[0]
        items = self.db.execute("SELECT COALESCE(SUM(items),0) FROM collection_tasks WHERE job_id=? "
                                "AND kind='race' AND page='results' AND state='done'", (job_id,)).fetchone()[0]
        state = self.db.execute("SELECT state FROM collection_jobs WHERE job_id=?", (job_id,)).fetchone()[0]
        return dict(job_id=job_id, state=state, complete=complete, config=config, tasks=counts,
                    expected_day_tasks=expected_days, discovered_races=discovered,
                    result_runner_rows=items, empty_days_without_date_metadata=unverified_empty,
                    problems=problems)


class Worker:
    def __init__(self, queue: JobQueue, client: Client, *, offline: bool = False, report=None,
                 lease_seconds: float = 900, heartbeat_seconds: float = 60):
        if not 0 < heartbeat_seconds < lease_seconds:
            raise ValueError("heartbeat は正数でleaseより短くしてください。")
        self.queue, self.store, self.client = queue, queue.store, client
        self.offline = offline
        self.report = report or (lambda **payload: None)
        self.owner = uuid.uuid4().hex
        self.lease_seconds, self.heartbeat_seconds = lease_seconds, heartbeat_seconds

    @contextmanager
    def keep_lease(self, task: Task):
        """Renew on a separate SQLite connection while network calls are waiting."""
        stopped = threading.Event()

        def heartbeat():
            db = sqlite3.connect(self.store.directory / "keiba.sqlite3", timeout=5)
            try:
                while not stopped.wait(self.heartbeat_seconds):
                    try:
                        with db:
                            cursor = db.execute(
                                "UPDATE collection_tasks SET lease_expires=? "
                                "WHERE id=? AND state='running' AND owner=?",
                                (time.time() + self.lease_seconds, task.id, task.owner))
                        if cursor.rowcount != 1:
                            return
                    except sqlite3.Error:
                        # A temporary lock is retried; final writes still check ownership.
                        continue
            finally:
                db.close()

        thread = threading.Thread(target=heartbeat, daemon=True, name="keiba-task-lease")
        thread.start()
        try:
            yield
        finally:
            stopped.set()
            thread.join(timeout=6)

    def load(self, url: str, force_refresh: bool = False):
        if self.offline:
            observation = self.store.cached(url)
            if observation is None:
                raise StorageError(f"保存済みHTMLがありません: {url}")
            return observation, self.store.body(observation).decode(observation.encoding), True
        previous = self.client.refresh
        try:
            self.client.refresh = previous or force_refresh
            return self.client.fetch(url)
        finally:
            self.client.refresh = previous

    def process(self, task: Task) -> None:
        if task.kind == "day":
            observation, html, cached = self.load(urls.list_url(task.day, task.region), task.force_refresh)
            discovered = urls.discover_day(html, day=task.day, region=task.region)
            self.queue.complete_day(task, observation, discovered)
            items = len(discovered["race_ids"])
        elif task.kind == "race":
            observation, html, cached = self.load(urls.race_url(task.race_id, task.region, task.page), task.force_refresh)
            parsed = parse_race(html, race_id=task.race_id, region=task.region, page=task.page)
            race = parsed["race"]
            if not race.get("race_id_verified") or race.get("race_date") != task.day:
                raise ValueError(f"結果ページのIDまたは開催日を照合できません: 予定={task.day}, 実際={race.get('race_date')}")
            if race.get("declared_field_size") is not None and race["declared_field_size"] != len(parsed["runners"]):
                raise ValueError("掲載頭数と取得した出走行数が一致しません。原本を確認してください。")
            expected = self.queue.db.execute(
                "SELECT expected_field_size FROM collection_races WHERE job_id="
                "(SELECT job_id FROM collection_tasks WHERE id=?) AND region=? AND race_id=?",
                (task.id, task.region, task.race_id)).fetchone()
            if expected and expected[0] is not None and expected[0] != len(parsed["runners"]):
                raise ValueError("開催一覧の頭数と取得した出走行数が一致しません。原本を確認してください。")
            self.store.save_race(observation, parsed)
            self.queue.complete_race(task, observation, parsed)
            items = len(parsed["runners"])
        elif task.kind == "horse":
            observation, html, cached = self.load(urls.horse_url(task.horse_id), task.force_refresh)
            parsed = parse_horse_history(html, horse_id=task.horse_id)
            self.store.save_horse(observation, parsed)
            self.queue.complete_horse(task, observation, parsed)
            items = len(parsed["runs"])
        else:
            raise ValueError(f"不明なタスク: {task.kind}")
        self.report(event="saved", task_id=task.id, kind=task.kind, day=task.day, region=task.region,
                    race_id=task.race_id, page=task.page, horse_id=task.horse_id,
                    items=items, cached=cached, snapshot_id=observation.id, retrieved_at=observation.retrieved_at)

    def run(self, job_id: str, *, max_tasks: int = 0, max_hours: float | None = None,
            min_free_mb: int = 1024) -> dict:
        self.queue.config(job_id)
        summary = self.queue.summary(job_id)
        if any(states.get("blocked", 0) for states in summary["tasks"].values()):
            self.report(event="blocked", message="取得制限のあるタスクがあります。statusで確認し、条件解消後にretryしてください。")
            return summary
        deadline = time.monotonic() + max_hours * 3600 if max_hours else float("inf")
        count = 0
        active = None
        self.queue.set_state(job_id, "running")
        try:
            while (not max_tasks or count < max_tasks) and time.monotonic() < deadline:
                if shutil.disk_usage(self.store.directory).free < min_free_mb * 1024 * 1024:
                    self.report(event="paused", reason="disk_space", minimum_free_mb=min_free_mb)
                    break
                active = self.queue.claim(job_id, self.owner, lease_seconds=self.lease_seconds)
                if active is None:
                    delay = self.queue.next_wait(job_id)
                    if delay is None:
                        break
                    self.report(event="waiting", seconds=round(delay, 1))
                    time.sleep(min(delay, 30, max(0, deadline - time.monotonic())))
                    continue
                try:
                    with self.keep_lease(active):
                        self.process(active)
                except AccessRestricted as exc:
                    self.queue.fail(active, exc, state="blocked")
                    self.report(event="blocked", task_id=active.id, error=str(exc))
                    active = None
                    break
                except FetchError as exc:
                    state = "pending" if active.attempts < 3 else "failed"
                    self.queue.fail(active, exc, state=state, delay=60 * 2 ** (active.attempts - 1))
                    self.report(event="error", task_id=active.id, state=state, error=str(exc))
                except (ValueError, StorageError, OSError) as exc:
                    self.queue.fail(active, exc)
                    self.report(event="error", task_id=active.id, state="failed", error=str(exc))
                active = None
                count += 1
        except BaseException:
            if active is not None:
                try:
                    self.queue.release(active)
                except StorageError:
                    pass  # The new lease owner retains the task; preserve the original exception.
            self.queue.set_state(job_id, "paused")
            raise
        summary = self.queue.summary(job_id)
        states = summary["tasks"].values()
        state = ("complete" if summary["complete"] else "blocked" if any(s.get("blocked") for s in states)
                 else "needs_attention" if any(s.get("failed") for s in summary["tasks"].values()) else "paused")
        self.queue.set_state(job_id, state)
        return self.queue.summary(job_id)
