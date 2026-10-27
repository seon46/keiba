"""Immutable observations, parsed snapshots, and a shared request schedule."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import date, datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import time
import uuid
import zlib

from .cancellations import is_confirmed_cancellation


class StorageError(RuntimeError):
    pass


class HostBlocked(StorageError):
    pass


@dataclass(frozen=True)
class Observation:
    id: str
    url: str
    retrieved_at: str
    status: int
    encoding: str
    sha256: str
    raw_path: str


class Store:
    def __init__(self, directory: str | Path):
        self.directory = Path(directory).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.directory / "keiba.sqlite3", timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY, url TEXT NOT NULL, retrieved_at TEXT NOT NULL,
                status INTEGER NOT NULL, encoding TEXT NOT NULL,
                sha256 TEXT NOT NULL, raw_path TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS observations_url
                ON observations(url, retrieved_at DESC);
            CREATE TABLE IF NOT EXISTS races (
                snapshot_id TEXT PRIMARY KEY REFERENCES observations(id),
                race_id TEXT NOT NULL, region TEXT NOT NULL, page TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS races_identity
                ON races(region, race_id, page);
            CREATE TABLE IF NOT EXISTS runners (
                snapshot_id TEXT NOT NULL REFERENCES races(snapshot_id),
                row_index INTEGER NOT NULL, horse_id TEXT, payload_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, row_index)
            );
            CREATE TABLE IF NOT EXISTS horses (
                snapshot_id TEXT PRIMARY KEY REFERENCES observations(id),
                horse_id TEXT NOT NULL, metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS horses_identity ON horses(horse_id);
            CREATE TABLE IF NOT EXISTS horse_runs (
                snapshot_id TEXT NOT NULL REFERENCES horses(snapshot_id),
                row_index INTEGER NOT NULL, race_id TEXT, payload_json TEXT NOT NULL,
                PRIMARY KEY(snapshot_id, row_index)
            );
            CREATE TABLE IF NOT EXISTS pedigrees (
                snapshot_id TEXT PRIMARY KEY REFERENCES observations(id),
                horse_id TEXT NOT NULL, metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS pedigrees_identity ON pedigrees(horse_id);
            CREATE TABLE IF NOT EXISTS request_state (
                host TEXT PRIMARY KEY, next_request REAL NOT NULL DEFAULT 0,
                blocked_until REAL NOT NULL DEFAULT 0, reason TEXT
            );
        """)

    def close(self) -> None:
        self.db.close()

    def cached(self, url: str, *, statuses: tuple[int, ...] = (200,)) -> Observation | None:
        placeholders = ",".join("?" for _ in statuses)
        row = self.db.execute(
            f"SELECT * FROM observations WHERE url=? AND status IN ({placeholders}) "
            "ORDER BY retrieved_at DESC, rowid DESC LIMIT 1", (url, *statuses)
        ).fetchone()
        if row is None:
            return None
        observation = Observation(**dict(row))
        if not (self.directory / observation.raw_path).is_file():
            return None
        return observation

    def body(self, observation: Observation) -> bytes:
        path = self.directory / observation.raw_path
        body = path.read_bytes()
        if path.suffix == ".gz":
            try:
                body = gzip.decompress(body)
            except (OSError, EOFError, zlib.error) as exc:
                raise StorageError(
                    "保存済みgzip原本を展開できません。--refresh で再取得してください。"
                ) from exc
        if hashlib.sha256(body).hexdigest() != observation.sha256:
            raise StorageError("保存済みHTMLのチェックサムが一致しません。--refresh で再取得してください。")
        return body

    def observe(self, url: str, body: bytes, *, encoding: str, status: int = 200) -> Observation:
        digest = hashlib.sha256(body).hexdigest()
        raw_path = f"raw/{digest[:2]}/{digest}.html.gz"
        destination = self.directory / raw_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(f".{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as file:
                file.write(gzip.compress(body, mtime=0))
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(destination)
            # Make the original durable before committing its SQLite reference.
            if hasattr(os, "O_DIRECTORY"):
                descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(descriptor)
                finally:
                    os.close(descriptor)
        finally:
            temporary.unlink(missing_ok=True)
        observation = Observation(
            id=uuid.uuid4().hex, url=url,
            retrieved_at=datetime.now(timezone.utc).isoformat(timespec="microseconds"),
            status=status, encoding=encoding, sha256=digest, raw_path=raw_path,
        )
        with self.db:
            self.db.execute(
                "INSERT INTO observations VALUES (?,?,?,?,?,?,?)",
                tuple(observation.__dict__.values()),
            )
        return observation

    def save_race(self, observation: Observation, parsed: dict) -> None:
        race, runners = parsed["race"], parsed["runners"]
        if not runners and not is_confirmed_cancellation(parsed):
            raise StorageError("出走馬が0件のため保存しません。")
        if race.get("race_status") == "cancelled" and not is_confirmed_cancellation(parsed):
            raise StorageError("開催中止の根拠と空の結果表を確認できません。")
        with self.db:
            self.db.execute(
                "INSERT INTO races VALUES (?,?,?,?,?) ON CONFLICT(snapshot_id) "
                "DO UPDATE SET metadata_json=excluded.metadata_json",
                (observation.id, race["race_id"], race["region"], race["page"],
                 json.dumps(race, ensure_ascii=False)),
            )
            self.db.execute("DELETE FROM runners WHERE snapshot_id=?", (observation.id,))
            self.db.executemany(
                "INSERT INTO runners VALUES (?,?,?,?)",
                [(observation.id, i, horse.get("horse_id"), json.dumps(horse, ensure_ascii=False))
                 for i, horse in enumerate(runners)],
            )

    def save_horse(self, observation: Observation, parsed: dict) -> None:
        runs = parsed["runs"]
        if not runs:
            raise StorageError("過去戦績が0件のため保存しません。")
        with self.db:
            self.db.execute(
                "INSERT INTO horses VALUES (?,?,?) ON CONFLICT(snapshot_id) "
                "DO UPDATE SET metadata_json=excluded.metadata_json",
                (observation.id, parsed["horse_id"], json.dumps(
                    {k: v for k, v in parsed.items() if k != "runs"}, ensure_ascii=False)),
            )
            self.db.execute("DELETE FROM horse_runs WHERE snapshot_id=?", (observation.id,))
            self.db.executemany(
                "INSERT INTO horse_runs VALUES (?,?,?,?)",
                [(observation.id, i, run.get("race_id"), json.dumps(run, ensure_ascii=False))
                 for i, run in enumerate(runs)],
            )

    def save_pedigree(self, observation: Observation, parsed: dict) -> None:
        """Keep the complete path-labelled tree with its original observation."""
        from .urls import horse_id
        identifier = horse_id(parsed.get("horse_id", ""))
        if observation.status != 200 or observation.url != f"https://db.netkeiba.com/horse/ped/{identifier}/":
            raise StorageError("血統原本のURL・馬ID・HTTP状態が一致しません。")
        ancestors = parsed.get("ancestors")
        if not isinstance(ancestors, list) or len(ancestors) != 62:
            raise StorageError("5代血統の62枠を確認できません。")
        paths = set()
        for ancestor in ancestors:
            if not isinstance(ancestor, dict):
                raise StorageError("血統の祖先情報が不正です。")
            path = ancestor.get("path")
            if (not isinstance(path, str) or not re.fullmatch(r"[SD]{1,5}", path)
                    or path in paths or ancestor.get("generation") != len(path)):
                raise StorageError("血統の世代・父母の経路が重複または不正です。")
            paths.add(path)
            ancestor_id = ancestor.get("horse_id")
            if ancestor_id is not None and (not isinstance(ancestor_id, str)
                    or not re.fullmatch(r"[A-Za-z0-9]{10}", ancestor_id)):
                raise StorageError("血統の祖先IDが不正です。")
        known = sum(bool(ancestor.get("name") or ancestor.get("horse_id")) for ancestor in ancestors)
        if not known:
            raise StorageError("既知の祖先が0件のため血統を保存しません。")
        metadata = dict(parsed, known_ancestor_count=known,
                        identified_ancestor_count=sum(ancestor.get("horse_id") is not None for ancestor in ancestors))
        with self.db:
            self.db.execute(
                "INSERT INTO pedigrees VALUES (?,?,?) ON CONFLICT(snapshot_id) "
                "DO UPDATE SET metadata_json=excluded.metadata_json",
                (observation.id, identifier, json.dumps(metadata, ensure_ascii=False)),
            )

    def _export_query(self, *, dataset: str, page: str | None = None,
                      region: str | None = None, all_snapshots: bool = False,
                      start_date: str | None = None,
                      end_date: str | None = None) -> tuple[str, list[str]]:
        for value in (start_date, end_date):
            if value is not None:
                try:
                    valid = isinstance(value, str) and date.fromisoformat(value).isoformat() == value
                except ValueError:
                    valid = False
                if not valid:
                    raise StorageError("出力期間は YYYY-MM-DD の日付で指定してください。")
        if start_date is not None and end_date is not None and start_date > end_date:
            raise StorageError("出力期間の開始日は終了日以前にしてください。")
        if dataset == "pedigrees":
            if page or region or start_date or end_date:
                raise StorageError("血統は馬単位のため --page/--region/期間フィルターはありません。")
            selected_where = "" if all_snapshots else "WHERE p.rank=1"
            return f"""
                WITH selected AS (
                    SELECT p.*,o.retrieved_at,o.url,
                        ROW_NUMBER() OVER (PARTITION BY p.horse_id
                            ORDER BY o.retrieved_at DESC,o.rowid DESC) AS rank
                    FROM pedigrees p JOIN observations o ON o.id=p.snapshot_id
                )
                SELECT p.snapshot_id,p.metadata_json,p.retrieved_at,p.url,'{{}}' AS payload_json
                FROM selected p {selected_where}
                ORDER BY p.retrieved_at,p.snapshot_id
            """, []
        if dataset == "runners":
            partition, table, children = "p.region,p.race_id,p.page", "races", "runners"
            date_expression = "json_extract(p.metadata_json, '$.race_date')"
        elif dataset == "horse-runs":
            partition, table, children = "p.horse_id", "horses", "horse_runs"
            date_expression = "json_extract(c.payload_json, '$.race_date')"
        else:
            raise StorageError("未対応のデータセットです。")
        filters, arguments = [], []
        if dataset == "runners":
            if page:
                filters.append("p.page=?")
                arguments.append(page)
            if region:
                filters.append("p.region=?")
                arguments.append(region)
        where = " WHERE " + " AND ".join(filters) if filters else ""
        selected_filters = [] if all_snapshots else ["p.rank=1"]
        if start_date is not None:
            selected_filters.append(f"{date_expression}>=?")
            arguments.append(start_date)
        if end_date is not None:
            selected_filters.append(f"{date_expression}<=?")
            arguments.append(end_date)
        selected_where = " WHERE " + " AND ".join(selected_filters) if selected_filters else ""
        sql = f"""
            WITH selected AS (
                SELECT p.*,o.retrieved_at,o.url,
                    ROW_NUMBER() OVER (PARTITION BY {partition}
                        ORDER BY o.retrieved_at DESC,o.rowid DESC) AS rank
                FROM {table} p JOIN observations o ON o.id=p.snapshot_id {where}
            )
            SELECT p.snapshot_id,p.metadata_json,p.retrieved_at,p.url,c.payload_json
            FROM selected p JOIN {children} c ON c.snapshot_id=p.snapshot_id
            {selected_where}
            ORDER BY p.retrieved_at,p.snapshot_id,c.row_index
        """
        return sql, arguments

    def iter_export_rows(self, *, dataset: str, page: str | None = None,
                         region: str | None = None, all_snapshots: bool = False,
                         start_date: str | None = None,
                         end_date: str | None = None) -> Iterator[dict]:
        """Yield rows without accumulating the dataset in Python memory.

        Date bounds include both endpoints. Latest snapshots are selected before
        filtering dates, so a corrected snapshot does not revive older records.
        Unknown dates are retained only when no date boundary is requested.
        """
        sql, arguments = self._export_query(
            dataset=dataset, page=page, region=region, all_snapshots=all_snapshots,
            start_date=start_date, end_date=end_date,
        )
        cursor = self.db.execute(sql, arguments)
        try:
            for row in cursor:
                metadata = json.loads(row["metadata_json"])
                if dataset == "runners":
                    metadata = {
                        key if key in ("race_id", "region", "page") or key.startswith("race_")
                        else "race_" + key: value for key, value in metadata.items()
                    }
                yield {
                    **metadata, **json.loads(row["payload_json"]),
                    "source": "netkeiba", "source_url": row["url"],
                    "snapshot_id": row["snapshot_id"], "retrieved_at": row["retrieved_at"],
                }
        finally:
            cursor.close()

    def export_fields(self, *, dataset: str, page: str | None = None,
                      region: str | None = None, all_snapshots: bool = False,
                      start_date: str | None = None,
                      end_date: str | None = None) -> list[str]:
        """Find every output column using memory proportional to column count."""
        fields: dict[str, None] = {}
        for row in self.iter_export_rows(
            dataset=dataset, page=page, region=region, all_snapshots=all_snapshots,
            start_date=start_date, end_date=end_date,
        ):
            for field in row:
                fields.setdefault(field, None)
        return list(fields)

    def export_rows(self, *, dataset: str, page: str | None = None,
                    region: str | None = None, all_snapshots: bool = False,
                    start_date: str | None = None,
                    end_date: str | None = None) -> list[dict]:
        """Compatibility API; production exports should use iter_export_rows."""
        return list(self.iter_export_rows(
            dataset=dataset, page=page, region=region, all_snapshots=all_snapshots,
            start_date=start_date, end_date=end_date,
        ))

    def reserve_request(self, host: str, interval: float) -> float:
        """Reserve a global slot, including across multiple CLI processes."""
        now = time.time()
        self.db.execute("BEGIN IMMEDIATE")
        try:
            blocked = self.db.execute(
                "SELECT blocked_until,reason FROM request_state WHERE host=?", (host,)
            ).fetchone()
            if blocked and blocked["blocked_until"] > now:
                deadline = datetime.fromtimestamp(blocked["blocked_until"], timezone.utc).isoformat()
                raise HostBlocked(f"{host} は {deadline} まで取得を停止しています: {blocked['reason']}")
            row = self.db.execute("SELECT next_request FROM request_state WHERE host='*'").fetchone()
            slot = max(now, row[0] if row else now)
            self.db.execute(
                "INSERT INTO request_state(host,next_request) VALUES ('*',?) "
                "ON CONFLICT(host) DO UPDATE SET next_request=excluded.next_request", (slot + interval,)
            )
            self.db.commit()
            return slot - now
        except BaseException:
            self.db.rollback()
            raise

    def block(self, host: str, reason: str, seconds: float = 3600) -> None:
        with self.db:
            self.db.execute(
                "INSERT INTO request_state(host,blocked_until,reason) VALUES (?,?,?) "
                "ON CONFLICT(host) DO UPDATE SET blocked_until=max(blocked_until,excluded.blocked_until),"
                "reason=excluded.reason", (host, time.time() + max(3600, seconds), reason),
            )
