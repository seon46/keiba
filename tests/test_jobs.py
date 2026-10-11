from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
import time
from unittest.mock import Mock, patch

from keiba_scraper.client import AccessRestricted, Client, FetchError
from keiba_scraper.jobs import historical_window, JobQueue, Worker
from keiba_scraper.storage import Store, StorageError
from keiba_scraper.urls import list_url, race_url


DAY = '2021-10-16'
RACE = '202105040301'
LISTING = '''<!-- block=race_list_sub (cp) -->
<div class="RaceList_Body RaceList_Top"><div class="RaceList_Box">
<dl class="RaceList_DataList"><dt class="RaceList_DataHeader">
<a href="../top/payback_list.html?kaisai_date=20211016">払戻一覧</a></dt>
<dd><ul><li class="RaceList_DataItem">
<a href="../race/result.html?race_id=202105040301"><div class="RaceData">10:15 芝1800m 1頭</div></a>
</li></ul></dd></dl></div></div>'''
RESULT = '''<html><head><title>テスト賞 結果 | 2021年10月16日 東京1R - netkeiba</title>
<link rel="canonical" href="https://race.netkeiba.com/race/result.html?race_id=202105040301"></head>
<body><h1 class="RaceName">テスト賞</h1><div class="RaceData01">10:15発走 / 芝1800m / 天候:晴 / 馬場:良</div>
<div class="RaceData02">1頭</div><table id="All_Result_Table"><tr><th>着順</th><th>馬番</th><th>馬名</th></tr>
<tr><td>1</td><td>1</td><td><a href="https://db.netkeiba.com/horse/2019100001/">架空の馬</a></td></tr>
</table></body></html>'''
EMPTY = ('<!-- block=race_list_sub (cp) -->'
         '<div class="RaceList_Body RaceList_Top"><div class="RaceList_Box"></div></div>')


class WindowTests(unittest.TestCase):
    def test_rolling_five_years_excludes_today_and_preserves_calendar_years(self):
        self.assertEqual(historical_window(today=date(2026, 10, 11)), ('2021-10-11', '2026-10-10'))
        self.assertEqual(historical_window(today=date(2024, 2, 29)), ('2019-02-28', '2024-02-28'))
        self.assertEqual(historical_window(today=date(2026, 10, 11), end='2024-06-30'),
                         ('2019-07-01', '2024-06-30'))
        with self.assertRaises(ValueError):
            historical_window(today=date(2026, 10, 11), end='2026-10-11')


class JobTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.store = Store(self.directory.name)
        self.queue = JobQueue(self.store)
        self.client = Client(self.store, interval=0)
        self.client.session.get = Mock(side_effect=AssertionError('Unexpected network call'))
        self.worker = Worker(self.queue, self.client, offline=True)

    def tearDown(self):
        self.client.close()
        self.store.close()
        self.directory.cleanup()

    def cache(self, url, html):
        return self.store.observe(url, html.encode(), encoding='utf-8')

    def one_day(self, listing=LISTING, result=RESULT):
        self.cache(list_url(DAY, 'central'), listing)
        self.cache(race_url(RACE, 'central', 'results'), result)
        return self.queue.plan(DAY, DAY, regions=('central',))

    def test_full_five_year_plan_is_idempotent_and_never_requests_data(self):
        job = self.queue.plan('2021-10-11', '2026-10-10')
        self.assertEqual(job, self.queue.plan('2021-10-11', '2026-10-10', regions=('local', 'central')))
        summary = self.queue.summary(job)
        self.assertEqual(summary['expected_day_tasks'], 3652)
        self.assertEqual(summary['tasks'], {'day': {'pending': 3652}})
        self.assertFalse(summary['complete'])
        self.client.session.get.assert_not_called()

    def test_resume_collects_discovered_races_without_duplicate_observations(self):
        job = self.one_day()
        first = self.worker.run(job, max_tasks=1, min_free_mb=0)
        self.assertEqual(first['tasks']['race'], {'pending': 1})
        self.assertFalse(first['complete'])
        resumed = Worker(self.queue, self.client, offline=True).run(job, min_free_mb=0)
        self.assertTrue(resumed['complete'])
        self.assertEqual(resumed['result_runner_rows'], 1)
        self.assertEqual(resumed['discovered_races'], 1)
        third = self.worker.run(job, min_free_mb=0)
        self.assertTrue(third['complete'])
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM runners').fetchone()[0], 1)
        self.client.session.get.assert_not_called()

    def test_no_meeting_is_completed_with_evidence_but_empty_response_is_failed(self):
        job = self.one_day(listing=EMPTY)
        summary = self.worker.run(job, min_free_mb=0)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['discovered_races'], 0)
        self.assertEqual(summary['empty_days_without_date_metadata'], 1)
        self.cache(list_url('2021-10-17', 'central'), '')
        bad = self.queue.plan('2021-10-17', '2021-10-17', regions=('central',))
        summary = self.worker.run(bad, min_free_mb=0)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['tasks']['day'], {'failed': 1})

    def test_wrong_date_never_saves_the_race(self):
        job = self.one_day(result=RESULT.replace('2021年10月16日', '2021年10月17日'))
        summary = self.worker.run(job, min_free_mb=0)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['tasks']['race'], {'failed': 1})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM races').fetchone()[0], 0)

    def test_listing_field_size_mismatch_never_saves_incomplete_races(self):
        job = self.one_day(listing=LISTING.replace('1頭', '2頭'))
        summary = self.worker.run(job, min_free_mb=0)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['tasks']['race'], {'failed': 1})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM races').fetchone()[0], 0)

    def test_parser_failure_can_reparse_cache_or_explicitly_refresh_original(self):
        job = self.one_day(result=RESULT.replace('2021年10月16日', '2021年10月17日'))
        self.worker.run(job, min_free_mb=0)
        self.assertEqual(self.queue.retry(job, refresh=True), 1)
        refreshed = self.cache(race_url(RACE, 'central', 'results'), RESULT)
        network_client = Mock(refresh=False)
        network_client.fetch.return_value = (refreshed, RESULT, False)
        flags = []
        network_client.fetch.side_effect = lambda url: (flags.append(network_client.refresh) or (refreshed, RESULT, False))
        summary = Worker(self.queue, network_client).run(job, min_free_mb=0)
        self.assertTrue(summary['complete'])
        self.assertEqual(flags, [True])
        self.assertFalse(network_client.refresh)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 3)

    def test_http_access_restriction_stops_entire_job_and_cannot_auto_resume(self):
        job = self.queue.plan(DAY, '2021-10-17')
        client = Mock(refresh=False)
        client.fetch.side_effect = AccessRestricted('HTTP 429')
        worker = Worker(self.queue, client)
        summary = worker.run(job, min_free_mb=0)
        self.assertEqual(client.fetch.call_count, 1)
        self.assertEqual(summary['state'], 'blocked')
        self.assertFalse(summary['complete'])
        worker.run(job, min_free_mb=0)
        self.assertEqual(client.fetch.call_count, 1)
        self.assertEqual(self.queue.retry(job), 1)

    def test_network_failure_has_bounded_attempts_and_is_not_complete(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        client = Mock(refresh=False)
        client.fetch.side_effect = FetchError('temporary failure')
        worker = Worker(self.queue, client)
        for attempt in range(3):
            with self.store.db:
                self.store.db.execute('UPDATE collection_tasks SET available_at=0 WHERE job_id=?', (job,))
            summary = worker.run(job, max_tasks=1, min_free_mb=0)
        self.assertEqual(client.fetch.call_count, 3)
        self.assertEqual(summary['tasks']['day'], {'failed': 1})
        self.assertFalse(summary['complete'])

    def test_expired_lease_recovery_never_steals_live_worker_task(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        other = Store(self.directory.name)
        try:
            second = JobQueue(other)
            first_task = self.queue.claim(job, 'first', now=100, lease_seconds=10)
            self.assertIsNone(second.claim(job, 'second', now=109))
            recovered = second.claim(job, 'second', now=111)
            self.assertEqual(recovered.id, first_task.id)
            self.assertEqual(recovered.attempts, 2)
            with self.assertRaises(StorageError):
                self.queue.release(first_task)
            second.release(recovered)
        finally:
            other.close()

    def test_ownership_change_between_check_and_update_cannot_overwrite_new_worker(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        task = self.queue.claim(job, 'first', now=100, lease_seconds=10)
        other = Store(self.directory.name)
        second = JobQueue(other)
        original = self.queue._owned
        try:
            def change_owner(active):
                identity = original(active)
                second.claim(job, 'second', now=111)
                return identity
            with patch.object(self.queue, '_owned', side_effect=change_owner):
                with self.assertRaises(StorageError):
                    self.queue.fail(task, ValueError('old worker'))
            row = self.store.db.execute('SELECT state,owner FROM collection_tasks WHERE id=?', (task.id,)).fetchone()
            self.assertEqual(tuple(row), ('running', 'second'))
        finally:
            other.close()

    def test_long_wait_renews_lease_so_another_worker_cannot_reclaim_task(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        worker = Worker(self.queue, self.client, lease_seconds=1, heartbeat_seconds=0.02)
        task = self.queue.claim(job, worker.owner, lease_seconds=1)
        other = Store(self.directory.name)
        try:
            second = JobQueue(other)
            with worker.keep_lease(task):
                time.sleep(1.2)  # Longer than the original lease; heartbeat keeps it live.
                self.assertIsNone(second.claim(job, 'other'))
            self.queue.release(task)
        finally:
            other.close()

    def test_old_manifest_schema_migrates_without_losing_discovered_races(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        with self.store.db:
            self.store.db.execute('DROP TABLE collection_races')
            self.store.db.execute('CREATE TABLE collection_races (job_id TEXT,region TEXT,race_id TEXT,day TEXT,'
                                  'PRIMARY KEY(job_id,region,race_id))')
            self.store.db.execute('INSERT INTO collection_races VALUES (?,?,?,?)', (job, 'central', RACE, DAY))
        migrated = JobQueue(self.store)
        self.assertEqual(migrated.summary(job)['discovered_races'], 1)
        job = self.one_day()
        self.assertTrue(Worker(migrated, self.client, offline=True).run(job, min_free_mb=0)['complete'])

    def test_interrupt_and_insufficient_disk_pause_without_losing_pending_tasks(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        client = Mock(refresh=False)
        client.fetch.side_effect = KeyboardInterrupt
        with self.assertRaises(KeyboardInterrupt):
            Worker(self.queue, client).run(job, min_free_mb=0)
        self.assertEqual(self.queue.summary(job)['tasks'], {'day': {'pending': 1}})
        with patch('keiba_scraper.jobs.shutil.disk_usage', return_value=Mock(free=0)):
            summary = self.worker.run(job, min_free_mb=1024)
        self.assertEqual(summary['state'], 'paused')
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['tasks'], {'day': {'pending': 1}})


if __name__ == '__main__':
    unittest.main()
