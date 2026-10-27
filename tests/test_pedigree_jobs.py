from itertools import product
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from keiba_scraper.client import AccessRestricted, Client
from keiba_scraper.jobs import JobQueue, Worker
from keiba_scraper.storage import Store, StorageError


HORSES = ('2019100001', '2019100002', '2019100003')


def pedigree_url(horse):
    return f'https://db.netkeiba.com/horse/ped/{horse}/'


def parsed_pedigree(horse):
    ancestors = []
    for generation in range(1, 6):
        for branch in product('SD', repeat=generation):
            path = ''.join(branch)
            identifier = '2010100001' if path == 'S' else None
            name = '父馬' if path == 'S' else ('母馬' if path == 'D' else None)
            ancestors.append(dict(path=path, generation=generation, horse_id=identifier,
                                  name=name, source_url=f'https://db.netkeiba.com/horse/{identifier}/'
                                  if identifier else None))
    return dict(horse_id=horse, horse_name='テスト馬', horse_id_verified=True,
                ancestors=ancestors, generations=5, parse_warnings=[],
                sire_id='2010100001', sire_name='父馬', dam_id=None, dam_name='母馬',
                damsire_id=None, damsire_name=None,
                missing_paths=[item['path'] for item in ancestors if item['name'] is None])


class PedigreeJobTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.store = Store(self.directory.name)
        self.addCleanup(self.store.close)
        self.queue = JobQueue(self.store)
        self.client = Client(self.store, interval=0)
        self.addCleanup(self.client.close)
        self.client.session.get = Mock(side_effect=AssertionError('Unexpected network call'))
        self.worker = Worker(self.queue, self.client, offline=True)

    def cache(self, horse, text='cached pedigree'):
        return self.store.observe(pedigree_url(horse), text.encode(), encoding='utf-8')

    def save_race(self, identifier, horses, *, day='2021-10-16', page='results', job=None):
        observation = self.store.observe(f'https://race.netkeiba.com/{page}/{identifier}',
                                         repr((identifier, horses, day, page)).encode(), encoding='utf-8')
        self.store.save_race(observation, dict(
            race=dict(race_id=identifier, region='central', page=page, race_date=day),
            runners=[dict(horse_id=horse) for horse in horses]))
        if job:
            with self.store.db:
                self.store.db.execute('INSERT OR IGNORE INTO collection_races(job_id,region,race_id,day) '
                                      'VALUES (?,\'central\',?,?)', (job, identifier, day))
        return observation

    def test_plan_deduplicates_targets_and_is_order_independent_without_http(self):
        first = self.queue.plan_pedigree(horse_ids=[HORSES[1], HORSES[0], HORSES[0]])
        second = self.queue.plan_pedigree(horse_ids=HORSES[:2])
        self.assertEqual(first, second)
        self.assertTrue(first.startswith('ped-'))
        summary = self.queue.summary(first)
        self.assertEqual(summary['tasks'], {'pedigree': {'pending': 2}})
        self.assertEqual(summary['total_horses'], 2)
        self.assertEqual(summary['remaining_horses'], 2)
        self.assertFalse(summary['complete'])
        self.client.session.get.assert_not_called()

    def test_invalid_or_empty_scope_does_not_create_a_job(self):
        for options in ({}, {'horse_ids': []}, {'horse_ids': ['invalid']},
                        {'horse_ids': HORSES, 'from_job_id': 'missing'}, {'from_job_id': 'missing'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.queue.plan_pedigree(**options)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM collection_jobs').fetchone()[0], 0)

    def test_source_job_uses_only_latest_results_of_its_races_and_preserves_original_job(self):
        source = self.queue.plan('2021-10-16', '2021-10-16', regions=('central',))
        self.save_race('202105040301', [HORSES[2]], job=source)
        self.save_race('202105040301', [HORSES[0]])  # corrected/latest snapshot replaces the earlier horse
        self.save_race('202105040302', [HORSES[1]], page='entries', job=source)
        self.save_race('202105040303', [HORSES[2]])  # another job's race
        self.save_race('202105040304', [HORSES[2]], day='2021-10-17', job=source)
        original = self.queue.summary(source)
        job = self.queue.plan_pedigree(from_job_id=source)
        targets = self.store.db.execute('SELECT horse_id FROM collection_tasks WHERE job_id=?', (job,)).fetchall()
        self.assertEqual([row[0] for row in targets], [HORSES[0]])
        self.assertEqual(self.queue.summary(source), original)
        self.assertEqual(job, self.queue.plan_pedigree(from_job_id=source))
        with self.assertRaises(ValueError):
            self.queue.plan_pedigree(from_job_id=job)

    def test_newly_observed_horses_create_a_new_scope_without_expanding_existing_job(self):
        source = self.queue.plan('2021-10-16', '2021-10-16', regions=('central',))
        self.save_race('202105040301', [HORSES[0]], job=source)
        frozen = self.queue.plan_pedigree(from_job_id=source)
        self.save_race('202105040301', list(HORSES[:2]))
        newer = self.queue.plan_pedigree(from_job_id=source)
        self.assertNotEqual(frozen, newer)
        self.assertEqual(self.queue.summary(frozen)['total_horses'], 1)
        self.assertEqual(self.queue.summary(newer)['total_horses'], 2)

    def test_empty_source_job_is_an_error_and_keeps_its_original_state(self):
        source = self.queue.plan('2021-10-16', '2021-10-16', regions=('central',))
        before = self.queue.summary(source)
        with self.assertRaises(ValueError):
            self.queue.plan_pedigree(from_job_id=source)
        self.assertEqual(self.queue.summary(source), before)

    def test_offline_batch_resumes_without_duplicate_observations_and_reports_partial_pedigrees(self):
        job = self.queue.plan_pedigree(horse_ids=HORSES[:2])
        for horse in HORSES[:2]: self.cache(horse)
        def parse(html, *, horse_id): return parsed_pedigree(horse_id)
        with patch('keiba_scraper.pedigree.parse_pedigree', side_effect=parse) as parser:
            partial = self.worker.run(job, max_tasks=1, min_free_mb=0)
            self.assertEqual(partial['completed_horses'], 1)
            self.assertFalse(partial['complete'])
            completed = self.worker.run(job, min_free_mb=0)
            self.assertTrue(completed['complete'])
            self.assertEqual(completed['completed_horses'], 2)
            self.assertEqual(completed['known_ancestor_cells'], 4)
            self.assertEqual(completed['partial_pedigrees'], 2)
            self.assertTrue(self.worker.run(job, min_free_mb=0)['complete'])
            self.assertEqual(parser.call_count, 2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM pedigrees').fetchone()[0], 2)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 2)
        self.client.session.get.assert_not_called()

    def test_parser_error_remains_failed_and_diagnostic_points_to_cached_pedigree(self):
        job = self.queue.plan_pedigree(horse_ids=HORSES[:1])
        observation = self.cache(HORSES[0], 'maintenance page')
        with patch('keiba_scraper.pedigree.parse_pedigree', side_effect=ValueError('not a pedigree')):
            summary = self.worker.run(job, min_free_mb=0)
        self.assertFalse(summary['complete'])
        self.assertEqual(summary['tasks'], {'pedigree': {'failed': 1}})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM pedigrees').fetchone()[0], 0)
        diagnosis = self.queue.diagnostics(job)
        problem = diagnosis['all_problems'][0]
        self.assertEqual(problem['source_url'], pedigree_url(HORSES[0]))
        self.assertEqual(problem['snapshot_id'], observation.id)
        self.assertTrue(problem['cached_html'])

    def test_wrong_parsed_identity_cannot_be_saved_or_marked_done(self):
        job = self.queue.plan_pedigree(horse_ids=HORSES[:1])
        self.cache(HORSES[0])
        with patch('keiba_scraper.pedigree.parse_pedigree', return_value=parsed_pedigree(HORSES[1])):
            summary = self.worker.run(job, min_free_mb=0)
        self.assertEqual(summary['tasks'], {'pedigree': {'failed': 1}})
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM pedigrees').fetchone()[0], 0)

    def test_offline_missing_cache_never_contacts_http_or_marks_done(self):
        job = self.queue.plan_pedigree(horse_ids=HORSES[:1])
        summary = self.worker.run(job, min_free_mb=0)
        self.assertEqual(summary['tasks'], {'pedigree': {'failed': 1}})
        self.assertEqual(summary['problems'][0]['error_type'], 'StorageError')
        self.client.session.get.assert_not_called()

    def test_access_restriction_stops_batch_until_explicit_retry(self):
        job = self.queue.plan_pedigree(horse_ids=HORSES[:2])
        client = Mock(refresh=False)
        client.fetch.side_effect = AccessRestricted('HTTP 429')
        worker = Worker(self.queue, client)
        summary = worker.run(job, min_free_mb=0)
        self.assertEqual(summary['state'], 'blocked')
        self.assertEqual(summary['tasks'], {'pedigree': {'blocked': 1, 'pending': 1}})
        self.assertFalse(summary['complete'])
        worker.run(job, min_free_mb=0)
        self.assertEqual(client.fetch.call_count, 1)
        self.assertEqual(self.queue.retry(job, kind='pedigree'), 1)

    def test_expired_pedigree_lease_recovers_without_stealing_live_work(self):
        job = self.queue.plan_pedigree(horse_ids=HORSES[:1])
        first = self.queue.claim(job, 'first', now=100, lease_seconds=10)
        self.assertIsNone(self.queue.claim(job, 'second', now=109))
        recovered = self.queue.claim(job, 'second', now=111)
        self.assertEqual(first.id, recovered.id)
        with self.assertRaises(StorageError): self.queue.release(first)
        self.queue.release(recovered)
        self.assertEqual(self.queue.summary(job)['remaining_horses'], 1)

    def test_interrupt_after_parsed_save_resumes_from_cache_without_duplicate_pedigree(self):
        job = self.queue.plan_pedigree(horse_ids=HORSES[:1])
        self.cache(HORSES[0])
        with patch('keiba_scraper.pedigree.parse_pedigree', return_value=parsed_pedigree(HORSES[0])):
            with patch.object(self.queue, 'complete_pedigree', side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt): self.worker.run(job, min_free_mb=0)
            self.assertEqual(self.queue.summary(job)['tasks'], {'pedigree': {'pending': 1}})
            self.assertTrue(self.worker.run(job, min_free_mb=0)['complete'])
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM pedigrees').fetchone()[0], 1)
        self.assertEqual(self.store.db.execute('SELECT COUNT(*) FROM observations').fetchone()[0], 1)


if __name__ == '__main__':
    unittest.main()
