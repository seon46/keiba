from contextlib import redirect_stderr, redirect_stdout
from datetime import date, timedelta
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from keiba_scraper.cli import main
from keiba_scraper.client import Client, FetchError
from keiba_scraper.jobs import JobQueue, Worker
from keiba_scraper.maintenance import Reparser
from keiba_scraper.storage import Store, StorageError
from keiba_scraper.urls import list_url, race_url
from test_jobs import DAY, RACE, LISTING, RESULT


JOCKEY_RESULT = RESULT.replace('<th>馬名</th>', '<th>馬名</th><th>騎手</th>').replace(
    '</a></td></tr>', '</a></td><td><a href="https://db.netkeiba.com/jockey/result/recent/a03fd/">架空の騎手</a></td></tr>')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.queue = JobQueue(self.store)
        self.client = Client(self.store, interval=0)
        self.client.session.get = Mock(side_effect=AssertionError('Unexpected HTTP request'))

    def tearDown(self):
        self.client.close()
        self.store.close()
        self.temp.cleanup()

    def observe(self, url, html):
        return self.store.observe(url, html.encode(), encoding='utf-8')

    def successful_job(self, *, two_races=False):
        listing = LISTING
        if two_races:
            second_card = ('<li class="RaceList_DataItem"><a href="../race/result.html?race_id=202105040302">'
                           '<div class="RaceData">10:45 芝1800m 1頭</div></a></li>')
            listing = listing.replace('</ul>', second_card + '</ul>')
            self.observe(race_url('202105040302', 'central', 'results'), JOCKEY_RESULT.replace(RACE, '202105040302'))
        self.observe(list_url(DAY, 'central'), listing)
        observation = self.observe(race_url(RACE, 'central', 'results'), JOCKEY_RESULT)
        job = self.queue.plan(DAY, DAY, regions=('central',))
        self.assertTrue(Worker(self.queue, self.client, offline=True).run(job, min_free_mb=0)['complete'])
        with self.store.db:
            self.store.db.execute("UPDATE runners SET payload_json=json_set(payload_json,'$.jockey_id',NULL)")
        return job, observation

    def test_offline_reparse_repairs_ids_without_new_observations_and_is_resumable(self):
        job, observation = self.successful_job(two_races=True)
        count = self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0]
        reparser = Reparser(self.queue)
        with patch('requests.Session.request', side_effect=AssertionError('No network allowed')):
            first = reparser.run(job, max_tasks=1, min_free_mb=0)
            self.assertEqual((first['done'], first['pending'], first['complete']), (1, 1, False))
            second = reparser.run(job, min_free_mb=0)
        self.assertTrue(second['complete'])
        self.assertEqual(second['done'], 2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], count)
        rows = self.store.export_rows(dataset='runners', page='results')
        self.assertEqual([row['jockey_id'] for row in rows], ['a03fd', 'a03fd'])
        original_row = next(row for row in rows if row['snapshot_id'] == observation.id)
        self.assertEqual(original_row['retrieved_at'], observation.retrieved_at)
        with patch('keiba_scraper.maintenance.parsers.parse_race', side_effect=AssertionError('Already reparsed')):
            self.assertTrue(reparser.run(job, min_free_mb=0)['complete'])
        self.client.session.get.assert_not_called()

    def test_corrupt_original_records_failure_and_preserves_existing_parsed_rows(self):
        job, observation = self.successful_job()
        before = self.store.export_rows(dataset='runners')
        (self.store.directory / observation.raw_path).write_bytes(b'corrupt original')
        result = Reparser(self.queue).run(job, min_free_mb=0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['failed'], 1)
        self.assertEqual(self.store.export_rows(dataset='runners'), before)
        self.assertTrue(self.queue.summary(job)['complete'])

    def test_cancellation_parser_update_invalidates_reparse_checkpoint(self):
        from keiba_scraper import cancellations
        job, observation = self.successful_job()
        original = Reparser(self.queue)
        self.assertTrue(original.run(job, min_free_mb=0)['complete'])
        changed = Path(self.temp.name) / 'updated_cancellations.py'
        changed.write_bytes(Path(cancellations.__file__).read_bytes() + b'\n# parser update\n')
        with patch.object(cancellations, '__file__', str(changed)):
            updated = Reparser(self.queue)
            self.assertNotEqual(original.revision, updated.revision)
            self.assertEqual(updated.summary(job)['pending'], 1)
            self.assertTrue(updated.run(job, min_free_mb=0)['complete'])
        self.assertEqual(self.store.export_rows(dataset='runners')[0]['snapshot_id'], observation.id)
        self.client.session.get.assert_not_called()

    def test_wrong_date_never_overwrites_old_data_and_failed_reparse_can_retry(self):
        from keiba_scraper.parsers import parse_race
        job, observation = self.successful_job()
        before = self.store.export_rows(dataset='runners')
        wrong = parse_race(JOCKEY_RESULT, race_id=RACE, region='central', page='results')
        wrong['race']['race_date'] = '2021-10-17'
        with patch('keiba_scraper.maintenance.parsers.parse_race', return_value=wrong):
            failed = Reparser(self.queue).run(job, min_free_mb=0)
        self.assertEqual(failed['failed'], 1)
        self.assertEqual(self.store.export_rows(dataset='runners'), before)
        self.assertTrue(Reparser(self.queue).run(job, min_free_mb=0)['complete'])

    def test_reparse_refuses_to_run_during_live_collection(self):
        job, observation = self.successful_job()
        other = self.queue.plan('2021-10-17', '2021-10-17', regions=('central',))
        active = self.queue.claim(other, 'another-process')
        with self.assertRaises(StorageError):
            Reparser(self.queue).run(job, min_free_mb=0)
        self.queue.release(active)

    def test_partial_day_keeps_discovered_races_and_never_counts_as_complete(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        listing = self.observe(list_url(DAY, 'central'), LISTING)
        self.observe(race_url(RACE, 'central', 'results'), JOCKEY_RESULT)
        task = self.queue.claim(job, 'test')
        discovery = dict(race_ids=[RACE], complete=False, empty=False, day_verified=True,
                         field_sizes={RACE: 1}, unavailable_meetings=[{'name': None, 'reason': '公開予定'}])
        self.queue.complete_day(task, listing, discovery)
        summary = Worker(self.queue, self.client, offline=True).run(job, min_free_mb=0)
        self.assertEqual(summary['tasks'], {'day': {'failed': 1}, 'race': {'done': 1}})
        self.assertEqual(summary['discovered_races'], 1)
        self.assertFalse(summary['complete'])
        diagnosis = self.queue.diagnostics(job)
        self.assertEqual(diagnosis['all_problems'][0]['error_type'], 'IncompleteDay')
        self.assertEqual(diagnosis['all_problems'][0]['discovery_evidence'], discovery)
        self.client.session.get.assert_not_called()

    def test_diagnosis_includes_all_failures_and_groups_different_urls(self):
        end = (date.fromisoformat(DAY) + timedelta(days=25)).isoformat()
        job = self.queue.plan(DAY, end, regions=('central',))
        for index in range(26):
            task = self.queue.claim(job, 'test')
            self.queue.fail(task, FetchError('取得に失敗しました: HTTP 503: ' + list_url(task.day, task.region)))
        report = self.queue.diagnostics(job)
        self.assertEqual(len(report['problems']), 20)
        self.assertEqual(len(report['all_problems']), 26)
        self.assertEqual(report['failed_task_count'], 26)
        self.assertEqual(len(report['error_groups']), 1)
        self.assertEqual(report['error_groups'][0]['count'], 26)
        self.assertFalse(any(row['cached_html'] for row in report['all_problems']))

    def test_retry_filters_only_selected_failures_and_preserves_success(self):
        job, observation = self.successful_job()
        other = self.queue.plan('2021-10-17', '2021-10-18', regions=('central',))
        one, two = self.queue.claim(other, 'test'), self.queue.claim(other, 'test')
        self.queue.fail(one, FetchError('HTTP 503'))
        self.queue.fail(two, ValueError('unrecognized page'))
        self.assertEqual(self.queue.retry(other, error_types=['FetchError'], kind='day'), 1)
        self.assertEqual(self.queue.summary(other)['tasks'], {'day': {'failed': 1, 'pending': 1}})
        self.assertTrue(self.queue.summary(job)['complete'])
        self.assertEqual(self.queue.retry(job), 0)

    def test_diagnostic_file_is_utf8_atomic_and_protects_database(self):
        job = self.queue.plan(DAY, DAY, regions=('central',))
        task = self.queue.claim(job, 'test')
        self.queue.fail(task, ValueError('開催一覧のエラー'))
        output = Path(self.temp.name) / 'recovery_report.json'
        args = ['--data-dir', self.temp.name, 'diagnose', '--job-id', job, '--output', str(output)]
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(args), 0)
            original = output.read_bytes()
            self.assertEqual(main(args), 2)
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(main(['--data-dir', self.temp.name, 'diagnose', '--job-id', job,
                                   '--output', str(Path(self.temp.name) / 'keiba.sqlite3'), '--overwrite']), 2)
        payload = json.loads(original.decode('utf-8'))
        self.assertEqual(payload['all_problems'][0]['last_error'], '開催一覧のエラー')
        self.assertEqual(self.store.db.execute('PRAGMA integrity_check').fetchone()[0], 'ok')
        self.assertFalse(list(Path(self.temp.name).glob('.keiba-diagnose-*')))


if __name__ == '__main__':
    unittest.main()
