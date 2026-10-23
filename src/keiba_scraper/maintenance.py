"""Resumable, offline re-parsing of successfully collected race snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

from . import parsers
from .jobs import JobQueue
from .storage import Observation, StorageError


class Reparser:
    def __init__(self, queue: JobQueue, *, report=None):
        self.queue, self.store, self.db = queue, queue.store, queue.db
        self.report = report or (lambda **payload: None)
        self.revision = hashlib.sha256(Path(parsers.__file__).read_bytes()).hexdigest()
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS snapshot_reparses (
                task_id INTEGER NOT NULL REFERENCES collection_tasks(id),
                snapshot_id TEXT NOT NULL REFERENCES observations(id),
                parser_revision TEXT NOT NULL, state TEXT NOT NULL,
                updated_at TEXT NOT NULL, items INTEGER NOT NULL DEFAULT 0, last_error TEXT,
                PRIMARY KEY(task_id,snapshot_id,parser_revision)
            );
        """)

    def summary(self, job_id: str) -> dict:
        self.queue.config(job_id)
        row = self.db.execute(
            "SELECT COUNT(*) AS total,"
            "COALESCE(SUM(CASE WHEN p.state='done' THEN 1 ELSE 0 END),0) AS done,"
            "COALESCE(SUM(CASE WHEN p.state='failed' THEN 1 ELSE 0 END),0) AS failed "
            "FROM collection_tasks t LEFT JOIN snapshot_reparses p ON p.task_id=t.id "
            "AND p.snapshot_id=t.snapshot_id AND p.parser_revision=? "
            "WHERE t.job_id=? AND t.kind='race' AND t.state='done' AND t.snapshot_id IS NOT NULL",
            (self.revision, job_id)).fetchone()
        remaining = row["total"] - row["done"]
        return dict(job_id=job_id, parser_revision=self.revision, total=row["total"],
                    done=row["done"], failed=row["failed"], pending=remaining-row["failed"],
                    remaining=remaining, complete=remaining == 0)

    def _record(self, row, state: str, *, items=0, error=None) -> None:
        self.db.execute(
            "INSERT INTO snapshot_reparses VALUES (?,?,?,?,?,?,?) "
            "ON CONFLICT(task_id,snapshot_id,parser_revision) DO UPDATE SET "
            "state=excluded.state,updated_at=excluded.updated_at,items=excluded.items,last_error=excluded.last_error",
            (row["task_id"], row["id"], self.revision, state,
             datetime.now(timezone.utc).isoformat(), items, str(error)[:2000] if error else None))

    def run(self, job_id: str, *, max_tasks: int = 0, min_free_mb: int = 1024) -> dict:
        self.queue.config(job_id)
        active = self.db.execute(
            "SELECT COUNT(*) FROM collection_tasks WHERE state='running' AND lease_expires>?",
            (time.time(),)).fetchone()[0]
        if active:
            raise StorageError("収集プロセスが処理中です。収集を停止してからreparseを実行してください。")
        self.report(event="reparse_plan", **self.summary(job_id))
        last_id, count = 0, 0
        while not max_tasks or count < max_tasks:
            if shutil.disk_usage(self.store.directory).free < min_free_mb * 1024 * 1024:
                self.report(event="paused", reason="disk_space", minimum_free_mb=min_free_mb)
                break
            limit = min(200, max_tasks-count) if max_tasks else 200
            rows = self.db.execute(
                "SELECT t.id AS task_id,t.region,t.day,t.race_id,t.page,t.result_json,o.* "
                "FROM collection_tasks t JOIN observations o ON o.id=t.snapshot_id "
                "LEFT JOIN snapshot_reparses p ON p.task_id=t.id AND p.snapshot_id=t.snapshot_id "
                "AND p.parser_revision=? WHERE t.job_id=? AND t.kind='race' AND t.state='done' "
                "AND (p.state IS NULL OR p.state!='done') AND t.id>? ORDER BY t.id LIMIT ?",
                (self.revision, job_id, last_id, limit)).fetchall()
            if not rows:
                break
            for row in rows:
                last_id = row["task_id"]
                observation = Observation(**{key: row[key] for key in Observation.__dataclass_fields__})
                try:
                    html = self.store.body(observation).decode(observation.encoding)
                    parsed = parsers.parse_race(html, race_id=row["race_id"],
                                                region=row["region"], page=row["page"])
                    self.queue.validate_race(row["task_id"], parsed)
                    # Validation completes before any previous parsed rows are replaced.
                    # If interrupted before the checkpoint, the same immutable snapshot
                    # is parsed again idempotently on the next run.
                    self.store.save_race(observation, parsed)
                    details = json.loads(row["result_json"] or "{}")
                    details.update(parser_revision=self.revision, race_type=parsed["race"].get("race_type"))
                    with self.db:
                        self.db.execute("UPDATE collection_tasks SET items=?,result_json=? "
                                        "WHERE id=? AND state='done' AND snapshot_id=?",
                                        (len(parsed["runners"]), json.dumps(details, ensure_ascii=False),
                                         row["task_id"], observation.id))
                        self._record(row, "done", items=len(parsed["runners"]))
                    self.report(event="reparsed", task_id=row["task_id"], race_id=row["race_id"],
                                items=len(parsed["runners"]), snapshot_id=observation.id,
                                retrieved_at=observation.retrieved_at)
                except (ValueError, StorageError, OSError) as exc:
                    with self.db:
                        self._record(row, "failed", error=exc)
                    self.report(event="reparse_error", task_id=row["task_id"], race_id=row["race_id"], error=str(exc))
                count += 1
        return self.summary(job_id)
