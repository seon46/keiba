import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock

from keiba_scraper.client import Client
from keiba_scraper.jobs import JobQueue, Worker
from keiba_scraper.maintenance import Reparser
from keiba_scraper.parsers import parse_race
from keiba_scraper.storage import Store, StorageError
from keiba_scraper.urls import list_url, race_url
from test_jobs import DAY, RACE, LISTING, RESULT


CANCELLED = f'''<html><head><title>架空賞 結果・払戻 | 2021年10月16日 東京1R - netkeiba</title>
<link rel="canonical" href="https://race.netkeiba.com/race/result.html?race_id={RACE}"></head>
<body><h1 class="RaceName">架空賞</h1><div class="RaceData01">10:15発走 / 芝1800m</div>
<div class="RaceData02">1頭</div><div class="ResultTableWrap">
<div class="Race_Infomation_Box">濃霧による視界不良により取り止めとなりました。</div>
<table id="All_Result_Table"><thead><tr><th>着順</th><th>馬番</th><th>馬名</th></tr></thead>
<tbody></tbody></table></div></body></html>'''


class CancelledStorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.store = Store(self.temp.name)
        self.queue = JobQueue(self.store)
        self.client = Client(self.store, interval=0)
        self.client.session.get = Mock(side_effect=AssertionError('Unexpected HTTP'))

    def tearDown(self):
        self.client.close()
        self.store.close()
        self.temp.cleanup()

    def cache(self, url, text):
        return self.store.observe(url, text.encode(), encoding='utf-8')

    def test_explicit_cancelled_race_is_accounted_for_without_fabricated_runners(self):
        self.cache(list_url(DAY, 'central'), LISTING)
        observation = self.cache(race_url(RACE, 'central', 'results'), CANCELLED)
        job = self.queue.plan(DAY, DAY, regions=('central',))
        result = Worker(self.queue, self.client, offline=True).run(job, min_free_mb=0)
        self.assertTrue(result['complete'])
        self.assertEqual(result['cancelled_races'], 1)
        self.assertEqual(result['result_runner_rows'], 0)
        self.assertEqual(result['tasks']['race'], {'done': 1})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM races').fetchone()[0], 1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM runners').fetchone()[0], 0)
        row = self.store.db.execute("SELECT items,result_json,snapshot_id FROM collection_tasks WHERE kind='race'").fetchone()
        self.assertEqual(row['items'], 0)
        self.assertEqual(row['snapshot_id'], observation.id)
        details = json.loads(row['result_json'])
        self.assertEqual(details['race_status'], 'cancelled')
        self.assertIn('濃霧', details['cancellation_reason'])
        self.assertTrue(Reparser(self.queue).run(job, min_free_mb=0)['complete'])
        self.assertEqual(self.queue.summary(job)['cancelled_races'], 1)
        self.client.session.get.assert_not_called()

    def test_unknown_empty_table_is_still_failed_and_not_marked_cancelled(self):
        unknown = CANCELLED.replace('濃霧による視界不良により取り止めとなりました。', '結果を準備中です。')
        self.cache(list_url(DAY, 'central'), LISTING)
        self.cache(race_url(RACE, 'central', 'results'), unknown)
        job = self.queue.plan(DAY, DAY, regions=('central',))
        result = Worker(self.queue, self.client, offline=True).run(job, min_free_mb=0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['cancelled_races'], 0)
        self.assertEqual(result['tasks']['race'], {'failed': 1})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM races').fetchone()[0], 0)

    def test_wrong_date_is_rejected_even_for_explicit_cancellation(self):
        self.cache(list_url(DAY, 'central'), LISTING)
        self.cache(race_url(RACE, 'central', 'results'), CANCELLED.replace('2021年10月16日', '2021年10月17日'))
        job = self.queue.plan(DAY, DAY, regions=('central',))
        result = Worker(self.queue, self.client, offline=True).run(job, min_free_mb=0)
        self.assertFalse(result['complete'])
        self.assertEqual(result['cancelled_races'], 0)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM races').fetchone()[0], 0)

    def test_status_flag_alone_cannot_bypass_empty_data_guard(self):
        observation = self.cache(race_url(RACE, 'central', 'results'), CANCELLED)
        forged = {'race': {'race_id': RACE, 'region': 'central', 'page': 'results', 'race_date': DAY,
                           'race_id_verified': True, 'race_status': 'cancelled'}, 'runners': []}
        with self.assertRaises(StorageError):
            self.store.save_race(observation, forged)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM races').fetchone()[0], 0)

    def test_latest_cancellation_never_revives_older_runner_rows(self):
        old = self.cache(race_url(RACE, 'central', 'results'), RESULT)
        self.store.save_race(old, parse_race(RESULT, race_id=RACE, region='central', page='results'))
        new = self.cache(race_url(RACE, 'central', 'results'), CANCELLED)
        self.store.save_race(new, parse_race(CANCELLED, race_id=RACE, region='central', page='results'))
        self.assertEqual(self.store.export_rows(dataset='runners', page='results'), [])
        history = self.store.export_rows(dataset='runners', page='results', all_snapshots=True)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['snapshot_id'], old.id)


if __name__ == '__main__':
    unittest.main()
