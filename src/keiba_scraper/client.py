"""Limited, cached requests; no login or access-restriction bypass."""

from __future__ import annotations

from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import re
import time
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import requests

from .storage import HostBlocked, Observation, Store


ALLOWED_HOSTS = {"www.netkeiba.com", "db.netkeiba.com", "race.netkeiba.com", "nar.netkeiba.com"}
USER_AGENT = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36")
ROBOTS_AGENT = "keiba-scraper"


class FetchError(RuntimeError):
    pass


class AccessRestricted(FetchError):
    pass


def validate_url(url: str) -> str:
    parts = urlsplit(url)
    if (parts.scheme != "https" or parts.hostname not in ALLOWED_HOSTS
            or parts.username is not None or parts.password is not None
            or parts.port not in (None, 443)):
        raise FetchError("取得先は指定した netkeiba の HTTPS ホストに限定しています。")
    return parts.hostname


def decode_body(body: bytes, content_type: str = "") -> tuple[str, str]:
    """Prefer explicit charsets; do not use requests' ISO-8859-1 default."""
    header = re.search(r"charset\s*=\s*[\"']?([\w-]+)", content_type, re.I)
    meta = re.search(br"charset\s*=\s*[\"']?([\w-]+)", body[:8192], re.I)
    candidates = ([header.group(1)] if header else []) + (
        [meta.group(1).decode("ascii")] if meta else []) + ["utf-8", "euc-jp", "cp932"]
    for encoding in dict.fromkeys(candidates):
        if encoding.lower() in ("iso-8859-1", "latin-1"):
            continue
        try:
            return body.decode(encoding), encoding
        except (LookupError, UnicodeDecodeError):
            continue
    raise FetchError("日本語HTMLの文字コードを判定できませんでした。原文を確認してください。")


def retry_seconds(value: str | None) -> float:
    if not value:
        return 3600
    try:
        return max(0, float(value))
    except ValueError:
        try:
            return max(0, (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return 3600


class Client:
    def __init__(self, store: Store, *, interval: float = 10, timeout: float = 30,
                 refresh: bool = False, retries: int = 2, priority: bool = False):
        self.store, self.interval, self.timeout, self.priority = store, interval, timeout, priority
        self.refresh, self.retries = refresh, retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.5"})
        self.robots: dict[str, RobotFileParser] = {}
        self.robots_retrieved_at: dict[str, datetime] = {}

    def close(self) -> None:
        self.session.close()

    def _check_block(self, host: str) -> None:
        """Recheck restrictions immediately before dispatching a reserved request."""
        row = self.store.db.execute(
            "SELECT blocked_until,reason FROM request_state WHERE host=?", (host,)
        ).fetchone()
        if row and row["blocked_until"] > time.time():
            deadline = datetime.fromtimestamp(row["blocked_until"], timezone.utc).isoformat()
            raise AccessRestricted(
                f"{host} は {deadline} まで取得を停止しています: {row['reason']}"
            )

    def _request(self, url: str, *, allow_missing: bool = False) -> tuple[bytes, str, int]:
        current = url
        for redirects in range(4):
            host = validate_url(current)
            for attempt in range(self.retries + 1):
                try:
                    delay = self.store.reserve_request(host, self.interval, priority=self.priority)
                except HostBlocked as exc:
                    raise AccessRestricted(str(exc)) from exc
                if delay:
                    time.sleep(delay)
                # Another collector may have received 403/429 while we waited.
                self._check_block(host)
                try:
                    response = self.session.get(current, timeout=self.timeout, stream=True,
                                                allow_redirects=False)
                except requests.exceptions.SSLError as exc:
                    raise FetchError("TLS証明書の検証に失敗しました。検証は無効化しません。") from exc
                except (requests.ConnectionError, requests.Timeout) as exc:
                    if attempt == self.retries:
                        raise FetchError(f"接続に失敗しました: {current}: {exc}") from exc
                    time.sleep(2 ** attempt)
                    continue
                with response:
                    status = response.status_code
                    if status in (403, 429):
                        reason = f"HTTP {status}; Retry-After={response.headers.get('Retry-After', 'unspecified')}"
                        self.store.block(host, reason, retry_seconds(response.headers.get("Retry-After")))
                        raise AccessRestricted(f"{host}: {reason}。再試行せず、取得を停止しました。")
                    if 300 <= status < 400:
                        location = response.headers.get("Location")
                        if not location:
                            raise FetchError(f"リダイレクト先がありません: HTTP {status}")
                        destination = urljoin(current, location)
                        validate_url(destination)
                        # Page redirects may lead to login or a different robots policy.
                        # This collector intentionally follows neither.
                        if (not allow_missing or urlsplit(destination).hostname != host
                                or urlsplit(destination).path != "/robots.txt"):
                            raise FetchError(f"ページがリダイレクトされました。取得条件を確認してください: {current}")
                        current = destination
                        break
                    if status >= 500 and attempt < self.retries:
                        time.sleep(2 ** attempt)
                        continue
                    if status != 200 and not (allow_missing and status in (404, 410)):
                        raise FetchError(f"取得に失敗しました: HTTP {status}: {current}")
                    chunks, size = [], 0
                    try:
                        for chunk in response.iter_content(65536):
                            size += len(chunk)
                            if size > 5_000_000:
                                raise FetchError("応答が5MBを超えたため取得を中止しました。")
                            chunks.append(chunk)
                    except requests.RequestException as exc:
                        raise FetchError(f"本文の受信に失敗しました: {current}") from exc
                    return b"".join(chunks), response.headers.get("Content-Type", ""), status
            else:
                raise FetchError(f"取得の再試行に失敗しました: {current}")
        raise FetchError("リダイレクトが多すぎるため中止しました。")

    def _check_robots(self, url: str) -> None:
        host = validate_url(url)
        now = datetime.now(timezone.utc)
        retrieved_at = self.robots_retrieved_at.get(host)
        if (host not in self.robots or retrieved_at is None
                or (now - retrieved_at).total_seconds() >= 86400):
            robots_url = f"https://{host}/robots.txt"
            cached = self.store.cached(robots_url, statuses=(200, 404, 410))
            if cached and (now - datetime.fromisoformat(cached.retrieved_at)).total_seconds() < 86400:
                body, status, encoding = self.store.body(cached), cached.status, cached.encoding
                text = body.decode(encoding)
                retrieved_at = datetime.fromisoformat(cached.retrieved_at)
            else:
                body, content_type, status = self._request(robots_url, allow_missing=True)
                text, encoding = decode_body(body, content_type)
                if status == 200 and re.search(r"<!doctype\s+html|<html\b|<body\b", text, re.I):
                    raise FetchError("robots.txt がHTMLを返したため、取得条件を確認できません。")
                observation = self.store.observe(robots_url, body, encoding=encoding, status=status)
                retrieved_at = datetime.fromisoformat(observation.retrieved_at)
            if status == 200 and re.search(r"<!doctype\s+html|<html\b|<body\b", text, re.I):
                raise FetchError("保存済みrobots.txtがHTMLのため、取得条件を確認できません。")
            rules = RobotFileParser(robots_url)
            rules.parse(text.splitlines() if status == 200 else [])
            self.robots[host] = rules
            self.robots_retrieved_at[host] = retrieved_at
        rules = self.robots[host]
        if not rules.can_fetch(ROBOTS_AGENT, url):
            raise AccessRestricted(f"robots.txt がこのURLの取得を拒否しています: {url}")
        crawl_delay = rules.crawl_delay(ROBOTS_AGENT)
        if crawl_delay:
            self.interval = max(self.interval, float(crawl_delay))
        request_rate = rules.request_rate(ROBOTS_AGENT)
        if request_rate:
            self.interval = max(self.interval, request_rate.seconds / request_rate.requests)

    def fetch(self, url: str, *, json_ok: bool = False) -> tuple[Observation, str, bool]:
        validate_url(url)
        cached = None if self.refresh else self.store.cached(url)
        if cached:
            return cached, self.store.body(cached).decode(cached.encoding), True
        self._check_robots(url)
        body, content_type, status = self._request(url)
        accepted = ("text/html", "application/xhtml+xml") + (("application/json", "text/javascript", "text/plain")
                                                              if json_ok else ())
        if content_type and not any(t in content_type.lower() for t in accepted):
            raise FetchError(f"HTMLではない応答を受信しました: {content_type}")
        text, encoding = decode_body(body, content_type)
        observation = self.store.observe(url, body, encoding=encoding, status=status)
        return observation, text, False
