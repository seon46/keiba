from __future__ import annotations

from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from keiba_scraper.client import AccessRestricted, Client, FetchError, decode_body, validate_url
from keiba_scraper.storage import Store, StorageError


URL = "https://race.netkeiba.com/race/result.html?race_id=202405021211"
ROBOTS = "https://race.netkeiba.com/robots.txt"


class Response:
    def __init__(self, status=200, body=b"<html>ok</html>", headers=None):
        self.status_code = status
        self.headers = headers or {"Content-Type": "text/html; charset=utf-8"}
        self.body = body

    def __enter__(self): return self
    def __exit__(self, *args): pass
    def iter_content(self, size): yield self.body


class ClientTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.client = Client(self.store, interval=0)

    def tearDown(self):
        self.client.close(); self.store.close(); self.temp.cleanup()

    def mock_responses(self, *responses):
        self.client.session.get = Mock(side_effect=responses)

    def test_cached_fetch_preserves_time_and_avoids_all_requests(self):
        self.mock_responses(Response(404), Response(body="<html>中央競馬</html>".encode()))
        first, text, cached = self.client.fetch(URL)
        second, second_text, reused = self.client.fetch(URL)
        self.assertFalse(cached); self.assertTrue(reused)
        self.assertEqual(first, second)
        self.assertEqual(text, second_text)
        self.assertEqual(self.client.session.get.call_count, 2)
        self.assertEqual(datetime.fromisoformat(first.retrieved_at).tzinfo, timezone.utc)

    def test_403_and_429_stop_without_retry_and_persist_block(self):
        for status in (403, 429):
            with self.subTest(status=status), TemporaryDirectory() as directory:
                store = Store(directory); client = Client(store, interval=0)
                store.observe(ROBOTS, b"not found", encoding="utf-8", status=404)
                client.session.get = Mock(return_value=Response(status, headers={"Retry-After": "7200"}))
                with self.assertRaises(AccessRestricted): client.fetch(URL)
                self.assertEqual(client.session.get.call_count, 1)
                other = Client(store, interval=0)
                other.session.get = Mock()
                with self.assertRaises(AccessRestricted): other.fetch(URL)
                other.session.get.assert_not_called()
                client.close(); other.close(); store.close()

    def test_robots_disallow_stops_before_page_request(self):
        self.mock_responses(Response(body=b"User-agent: *\nDisallow: /race/\n",
                                     headers={"Content-Type": "text/plain"}))
        with self.assertRaises(AccessRestricted): self.client.fetch(URL)
        self.assertEqual(self.client.session.get.call_count, 1)

    def test_long_running_client_rechecks_robots_at_original_observation_expiry(self):
        anchor = datetime.now(timezone.utc)
        original = self.store.observe(
            ROBOTS, b"User-agent: *\nAllow: /\n", encoding="utf-8",
        )
        observed_at = anchor - timedelta(hours=23)
        with self.store.db:
            self.store.db.execute("UPDATE observations SET retrieved_at=? WHERE id=?",
                                  (observed_at.isoformat(), original.id))
        self.mock_responses()
        self.client._check_robots(URL)
        self.assertEqual(self.client.robots_retrieved_at["race.netkeiba.com"], observed_at)
        self.client.session.get.assert_not_called()
        with patch("keiba_scraper.client.datetime", wraps=datetime) as clock:
            clock.now.return_value = anchor + timedelta(minutes=30)
            self.client._check_robots(URL)
            self.client.session.get.assert_not_called()
            # The cached rules are now 25 hours old, although this Client loaded
            # them only two hours ago. The site's changed policy must take effect.
            clock.now.return_value = anchor + timedelta(hours=2)
            self.mock_responses(Response(body=b"User-agent: *\nDisallow: /race/\n",
                                         headers={"Content-Type": "text/plain"}))
            with self.assertRaises(AccessRestricted):
                self.client.fetch(URL)
        self.assertEqual(self.client.session.get.call_count, 1)
        self.assertFalse(self.client.robots["race.netkeiba.com"].can_fetch("keiba-scraper", URL))

    def test_worker_block_during_reserved_wait_prevents_request_dispatch(self):
        self.store.observe(ROBOTS, b"not found", encoding="utf-8", status=404)
        other = Store(self.temp.name)
        try:
            other.reserve_request("race.netkeiba.com", 10)
            self.client.interval = 10
            self.mock_responses()
            with patch("keiba_scraper.client.time.sleep", side_effect=lambda delay: other.block(
                    "race.netkeiba.com", "HTTP 429 from another worker", 7200)) as sleeper:
                with self.assertRaises(AccessRestricted):
                    self.client.fetch(URL)
            sleeper.assert_called_once()
            self.assertGreater(sleeper.call_args.args[0], 0)
            self.client.session.get.assert_not_called()
        finally:
            other.close()

    def test_fresh_and_cached_html_robots_cannot_be_bypassed(self):
        self.mock_responses(Response(body=b"<!DOCTYPE HTML><HTML><BODY>blocked</BODY></HTML>"))
        with self.assertRaises(FetchError): self.client.fetch(URL)
        self.assertIsNone(self.store.cached(ROBOTS))
        self.store.observe(ROBOTS, b"<html>blocked</html>", encoding="utf-8")
        self.client.session.get.reset_mock()
        with self.assertRaises(FetchError): self.client.fetch(URL)
        self.client.session.get.assert_not_called()

    def test_redirect_to_another_page_is_not_followed(self):
        self.mock_responses(Response(404), Response(302, headers={"Location": "https://nar.netkeiba.com/race/x"}))
        with self.assertRaises(FetchError): self.client.fetch(URL)
        self.assertEqual(self.client.session.get.call_count, 2)

    def test_euc_jp_is_decoded_without_corruption(self):
        body = '<meta charset="EUC-JP"><p>東京競馬場</p>'.encode("euc-jp")
        text, encoding = decode_body(body, "text/html")
        self.assertIn("東京競馬場", text)
        self.assertEqual(encoding.lower(), "euc-jp")

    def test_untrusted_urls_are_rejected(self):
        for url in ("http://race.netkeiba.com/", "https://evil.test/", "https://race.netkeiba.com.evil.test/",
                    "https://user:password@race.netkeiba.com/", "https://race.netkeiba.com:1234/"):
            with self.subTest(url=url), self.assertRaises(FetchError): validate_url(url)

    def test_raw_checksum_detects_corruption(self):
        observation = self.store.observe(URL, b"original", encoding="utf-8")
        (Path(self.temp.name) / observation.raw_path).write_bytes(b"changed")
        with self.assertRaises(StorageError): self.store.body(observation)

    def test_reservations_are_shared_across_connections(self):
        other = Store(self.temp.name)
        try:
            with patch("keiba_scraper.storage.time.time", return_value=100):
                self.assertEqual(self.store.reserve_request("race.netkeiba.com", 10), 0)
                self.assertEqual(other.reserve_request("nar.netkeiba.com", 10), 10)
        finally: other.close()


class StorageAndCLITests(unittest.TestCase):
    def test_snapshots_and_labels_are_separate_and_export_latest(self):
        with TemporaryDirectory() as directory:
            store = Store(directory)
            entries = {"race": {"race_id": "202405021211", "region": "central", "page": "entries"},
                       "runners": [{"horse_id": "2021100001", "horse_number": 1}]}
            results = {"race": {"race_id": "202405021211", "region": "central", "page": "results"},
                       "runners": [{"horse_id": "2021100001", "horse_number": 1, "finish_position": 2}]}
            old = store.observe(URL, b"old", encoding="utf-8");store.save_race(old, results)
            new = store.observe(URL, b"new", encoding="utf-8");store.save_race(new, results)
            before = store.observe(URL.replace("result", "shutuba"), b"entries", encoding="utf-8")
            store.save_race(before, entries)
            self.assertEqual(len(store.export_rows(dataset="runners", page="results")), 1)
            self.assertEqual(len(store.export_rows(dataset="runners", all_snapshots=True)), 3)
            row = store.export_rows(dataset="runners", page="entries")[0]
            self.assertNotIn("finish_position", row)
            self.assertEqual(row["retrieved_at"], before.retrieved_at)
            store.close()

    def test_cli_offline_export_and_existing_file_protection(self):
        from keiba_scraper.cli import main
        with TemporaryDirectory() as directory:
            store = Store(directory)
            observation = store.observe(URL, b"fixture", encoding="utf-8")
            store.save_race(observation, {"race": {"race_id": "202405021211", "region": "central", "page": "results"},
                                         "runners": [{"horse_id": "2021100001", "finish_position": 1}]})
            store.close(); destination = Path(directory) / "dataset.csv"
            args = ["--data-dir", directory, "export", "--page", "results", "--output", str(destination)]
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                self.assertEqual(main(args), 0)
                content = destination.read_bytes()
                self.assertEqual(main(args), 2)
            self.assertEqual(destination.read_bytes(), content)
            self.assertIn(b"retrieved_at", content)


if __name__ == "__main__":
    unittest.main()
