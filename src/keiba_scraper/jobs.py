"""Durable historical collection; discoveries and completion share a transaction."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import re
import shutil
import sqlite3
import threading
import time
import uuid

from . import urls
from .client import AccessRestricted, Client, FetchError
from .cancellations import is_confirmed_cancellation
from .parsers import parse_horse_history, parse_race
from .storage import Store, StorageError


# Japan has used UTC+9 without DST throughout the supported historical window.
# A fixed offset also works on Windows without a system IANA timezone database.
JST = timezone(timedelta(hours=9), name="JST")


class IncompleteDay(ValueError):
    """A daily list exposes some races but withholds another meeting."""


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

    def plan_pedigree(self, *, horse_ids=None, from_job_id: str | None = None) -> str:
        """Freeze one task per observed horse without fetching ancestor pages.

        The target digest is part of the immutable scope. If later race
        collection adds horses, planning again makes a separate job; resuming
        an existing job always retains its original list.
        """
        if (horse_ids is None) == (from_job_id is None):
            raise ValueError("血統の対象は --horse-id または --from-job-id の一方で指定してください。")
        source = self.config(from_job_id) if from_job_id is not None else None
        if source is not None and source.get("type") == "pedigree":
            raise ValueError("--from-job-id にはレース収集ジョブを指定してください。")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if source is None:
                targets = sorted({urls.horse_id(value) for value in horse_ids})
                # Existing columns are NOT NULL; explicit horse selection has
                # no race-date scope, represented only in the config below.
                start = end = "0001-01-01"
            else:
                start, end = source["start_date"], source["end_date"]
                rows = self.db.execute("""
                    WITH latest AS (
                        SELECT r.snapshot_id,r.metadata_json,cr.day,
                               ROW_NUMBER() OVER (
                                   PARTITION BY r.region,r.race_id
                                   ORDER BY o.retrieved_at DESC,o.rowid DESC
                               ) AS ordinal
                        FROM races r
                        JOIN observations o ON o.id=r.snapshot_id
                        JOIN collection_races cr ON cr.region=r.region AND cr.race_id=r.race_id
                        WHERE cr.job_id=? AND r.page='results' AND o.status=200
                    )
                    SELECT DISTINCT n.horse_id FROM latest r
                    JOIN runners n ON n.snapshot_id=r.snapshot_id
                    WHERE r.ordinal=1 AND json_extract(r.metadata_json,'$.race_date')=r.day
                          AND r.day BETWEEN ? AND ? AND n.horse_id IS NOT NULL
                    ORDER BY n.horse_id
                """, (from_job_id, start, end))
                targets = [urls.horse_id(row[0]) for row in rows]
            if not targets:
                raise ValueError("血統を取得する馬がありません。保存済みのレース結果とジョブを確認してください。")
            target_digest = hashlib.sha256("\n".join(targets).encode()).hexdigest()
            config = dict(version=1, type="pedigree", source_job_id=from_job_id,
                          horse_count=len(targets), target_sha256=target_digest,
                          generations=5, start_date=start if source is not None else None,
                          end_date=end if source is not None else None)
            encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
            job_id = "ped-" + hashlib.sha256(encoded.encode()).hexdigest()[:16]
            now = datetime.now(timezone.utc).isoformat()
            self.db.execute("INSERT OR IGNORE INTO collection_jobs VALUES (?,?,?,?,?,?,?)",
                            (job_id, now, now, start, end, encoded, "planned"))
            self.db.executemany(
                "INSERT OR IGNORE INTO collection_tasks(job_id,kind,task_key,horse_id) "
                "VALUES (?,'pedigree',?,?)", ((job_id, target, target) for target in targets))
            self.db.commit()
        except BaseException:
            self.db.rollback()
            raise
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
            incomplete = discovery.get("complete") is False
            error = None
            if incomplete:
                names = [meeting.get("name") or "不明" for meeting in discovery.get("unavailable_meetings", [])]
                error = IncompleteDay("開催一覧に未公開の区画があります: " + ", ".join(names))
            self._finish(task, "failed" if incomplete else "done", snapshot_id=observation.id,
                         items=len(discovery["race_ids"]), result=discovery, error=error)

    def complete_race(self, task: Task, observation, parsed: dict) -> None:
        with self.db:
            job_id = self._owned(task)
            if self.config(job_id)["include_horses"]:
                self.db.executemany(
                    "INSERT OR IGNORE INTO collection_tasks(job_id,kind,task_key,horse_id) VALUES (?,'horse',?,?)",
                    ((job_id, runner["horse_id"], runner["horse_id"]) for runner in parsed["runners"]))
            self._finish(task, "done", snapshot_id=observation.id, items=len(parsed["runners"]),
                         result={"race_date": parsed["race"]["race_date"],
                                 "race_type": parsed["race"].get("race_type"),
                                 "race_status": parsed["race"].get("race_status", "held"),
                                 "cancellation_reason": parsed["race"].get("cancellation_reason")})

    def complete_horse(self, task: Task, observation, parsed: dict) -> None:
        with self.db:
            self._owned(task)
            self._finish(task, "done", snapshot_id=observation.id, items=len(parsed["runs"]))

    def complete_pedigree(self, task: Task, observation, parsed: dict) -> None:
        if task.kind != "pedigree" or parsed.get("horse_id") != task.horse_id:
            raise ValueError("解析した血統の馬IDが収集予定と一致しません。")
        known = sum(bool(ancestor.get("name") or ancestor.get("horse_id"))
                    for ancestor in parsed["ancestors"])
        identified = sum(ancestor.get("horse_id") is not None for ancestor in parsed["ancestors"])
        with self.db:
            self._owned(task)
            self._finish(task, "done", snapshot_id=observation.id, items=known,
                         result={"known_ancestor_count": known, "identified_ancestor_count": identified,
                                 "missing_path_count": len(parsed.get("missing_paths", ()))})

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

    def retry(self, job_id: str, *, refresh: bool = False, error_types: list[str] | None = None,
              kind: str | None = None) -> int:
        self.config(job_id)
        conditions = ["job_id=?", "state IN ('failed','blocked')"]
        values = [int(refresh), job_id]
        if error_types:
            conditions.append("error_type IN (" + ",".join("?" for _ in error_types) + ")")
            values.extend(error_types)
        if kind:
            conditions.append("kind=?")
            values.append(kind)
        with self.db:
            cursor = self.db.execute(
                "UPDATE collection_tasks SET state='pending',attempts=0,available_at=0,force_refresh=?,"
                "last_error=NULL,error_type=NULL WHERE " + " AND ".join(conditions), values)
        return cursor.rowcount

    def validate_race(self, task_id: int, parsed: dict) -> None:
        task = self.db.execute("SELECT * FROM collection_tasks WHERE id=? AND kind='race'", (task_id,)).fetchone()
        if task is None:
            raise StorageError("照合対象のレースタスクがありません。")
        race, runners = parsed["race"], parsed["runners"]
        if any(race.get(key) != task[key] for key in ("race_id", "region", "page")):
            raise ValueError("解析したレースのID・地域・ページが収集予定と一致しません。")
        if not race.get("race_id_verified") or race.get("race_date") != task["day"]:
            raise ValueError(f"結果ページのIDまたは開催日を照合できません: 予定={task['day']}, 実際={race.get('race_date')}")
        if is_confirmed_cancellation(parsed):
            return  # Scheduled entries have no results when the source confirms cancellation.
        if not runners or race.get("race_status") == "cancelled":
            raise ValueError("空の結果表を開催中止と確認できません。原本を確認してください。")
        if race.get("declared_field_size") is not None and race["declared_field_size"] != len(runners):
            raise ValueError("掲載頭数と取得した出走行数が一致しません。原本を確認してください。")
        expected = self.db.execute(
            "SELECT expected_field_size FROM collection_races WHERE job_id=? AND region=? AND race_id=?",
            (task["job_id"], task["region"], task["race_id"])).fetchone()
        if expected and expected[0] is not None and expected[0] != len(runners):
            raise ValueError("開催一覧の頭数と取得した出走行数が一致しません。原本を確認してください。")

    def diagnostics(self, job_id: str) -> dict:
        """Include every failed task, unlike status's twenty examples. No HTTP."""
        summary = self.summary(job_id)
        problems, groups = [], {}
        for row in self.db.execute(
                "SELECT id,kind,region,day,race_id,page,horse_id,state,last_error,error_type,attempts,result_json "
                "FROM collection_tasks WHERE job_id=? AND state IN ('failed','blocked') ORDER BY id", (job_id,)):
            problem = dict(row)
            evidence = problem.pop("result_json")
            if evidence:
                problem["discovery_evidence"] = json.loads(evidence)
            if row["kind"] == "day":
                url = urls.list_url(row["day"], row["region"])
            elif row["kind"] == "race":
                url = urls.race_url(row["race_id"], row["region"], row["page"])
            elif row["kind"] == "horse":
                url = urls.horse_url(row["horse_id"])
            elif row["kind"] == "pedigree":
                from .pedigree import pedigree_url
                url = pedigree_url(row["horse_id"])
            else:
                url = None
            observation = self.store.cached(url) if url else None
            problem.update(source_url=url, cached_html=observation is not None)
            if observation:
                problem.update(snapshot_id=observation.id, retrieved_at=observation.retrieved_at,
                               raw_path=observation.raw_path)
            message = re.sub(r"https://\S+", "<URL>", row["last_error"] or "")
            message = re.sub(r"\b[0-9]{12}\b", "<RACE_ID>", message)
            message = re.sub(r"\b[0-9]{4}-[0-9]{2}-[0-9]{2}\b", "<DATE>", message)
            key = (row["kind"], row["error_type"], message)
            if key not in groups:
                groups[key] = dict(kind=row["kind"], error_type=row["error_type"], message=message,
                                   count=0, sample_task_ids=[])
            group = groups[key]
            group["count"] += 1
            if len(group["sample_task_ids"]) < 3:
                group["sample_task_ids"].append(row["id"])
            problems.append(problem)
        return dict(**summary, generated_at=datetime.now(timezone.utc).isoformat(),
                    failed_task_count=len(problems),
                    error_groups=sorted(groups.values(), key=lambda group: -group["count"]),
                    all_problems=problems)

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
        if config.get("type") == "pedigree":
            states = counts.get("pedigree", {})
            done = states.get("done", 0)
            total = config["horse_count"]
            complete = (done == total and sum(states.values()) == total and
                        all(kind == "pedigree" for kind in counts))
            problems = [dict(row) for row in self.db.execute(
                "SELECT id,kind,horse_id,state,last_error,error_type,attempts "
                "FROM collection_tasks WHERE job_id=? AND state IN ('failed','blocked') "
                "ORDER BY id LIMIT 20", (job_id,))]
            state = self.db.execute("SELECT state FROM collection_jobs WHERE job_id=?", (job_id,)).fetchone()[0]
            quality = self.db.execute(
                "SELECT COALESCE(SUM(items),0),COALESCE(SUM(CASE WHEN "
                "json_extract(result_json,'$.missing_path_count')>0 THEN 1 ELSE 0 END),0) "
                "FROM collection_tasks WHERE job_id=? AND kind='pedigree' AND state='done'", (job_id,)).fetchone()
            return dict(job_id=job_id, state=state, complete=complete, config=config, tasks=counts,
                        total_horses=total, completed_horses=done, remaining_horses=max(0,total-done),
                        known_ancestor_cells=quality[0], partial_pedigrees=quality[1], problems=problems)
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
        cancelled = self.db.execute("SELECT COUNT(*) FROM collection_tasks WHERE job_id=? AND kind='race' "
                                    "AND page='results' AND state='done' "
                                    "AND json_extract(result_json,'$.race_status')='cancelled'", (job_id,)).fetchone()[0]
        state = self.db.execute("SELECT state FROM collection_jobs WHERE job_id=?", (job_id,)).fetchone()[0]
        return dict(job_id=job_id, state=state, complete=complete, config=config, tasks=counts,
                    expected_day_tasks=expected_days, discovered_races=discovered,
                    result_runner_rows=items, cancelled_races=cancelled,
                    empty_days_without_date_metadata=unverified_empty,
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
            discovered = urls.discover_day(html, day=task.day, region=task.region, allow_partial=True)
            self.queue.complete_day(task, observation, discovered)
            items = len(discovered["race_ids"])
        elif task.kind == "race":
            observation, html, cached = self.load(urls.race_url(task.race_id, task.region, task.page), task.force_refresh)
            parsed = parse_race(html, race_id=task.race_id, region=task.region, page=task.page)
            self.queue.validate_race(task.id, parsed)
            self.store.save_race(observation, parsed)
            self.queue.complete_race(task, observation, parsed)
            items = len(parsed["runners"])
        elif task.kind == "horse":
            observation, html, cached = self.load(urls.horse_url(task.horse_id), task.force_refresh)
            parsed = parse_horse_history(html, horse_id=task.horse_id)
            self.store.save_horse(observation, parsed)
            self.queue.complete_horse(task, observation, parsed)
            items = len(parsed["runs"])
        elif task.kind == "pedigree":
            from .pedigree import parse_pedigree, pedigree_url
            observation, html, cached = self.load(pedigree_url(task.horse_id), task.force_refresh)
            parsed = parse_pedigree(html, horse_id=task.horse_id)
            if parsed.get("horse_id") != task.horse_id:
                raise ValueError("解析した血統の馬IDが収集予定と一致しません。")
            self.store.save_pedigree(observation, parsed)
            self.queue.complete_pedigree(task, observation, parsed)
            items = sum(bool(ancestor.get("name") or ancestor.get("horse_id"))
                        for ancestor in parsed["ancestors"])
        else:
            raise ValueError(f"不明なタスク: {task.kind}")
        incomplete = task.kind == "day" and discovered.get("complete") is False
        status_fields = ({"race_status": parsed["race"].get("race_status", "held")}
                         if task.kind == "race" else {})
        self.report(event="partial" if incomplete else "saved", task_id=task.id, kind=task.kind, day=task.day, region=task.region,
                    race_id=task.race_id, page=task.page, horse_id=task.horse_id,
                    items=items, cached=cached, snapshot_id=observation.id,
                    retrieved_at=observation.retrieved_at, **status_fields)

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
