from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from keiba_scraper.cli import main
from keiba_scraper.jobs import JobQueue
from keiba_scraper.storage import Store


class PedigreeCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_cli(self, *args):
        output, error = io.StringIO(), io.StringIO()
        with redirect_stdout(output), redirect_stderr(error):
            result = main(['--data-dir', str(self.directory), *args])
        return result, [json.loads(line) for line in output.getvalue().splitlines()], error.getvalue()

    def plan(self):
        code, rows, error = self.run_cli('pedigree', '--horse-id', '2019102383', '--plan-only')
        self.assertEqual(code, 0, error)
        return rows[-1]['job_id']

    def test_planning_status_and_diagnostic_are_offline_without_a_client(self):
        with patch('keiba_scraper.cli.Client', side_effect=AssertionError('no HTTP for planning/status')):
            job = self.plan()
            self.assertEqual(self.plan(), job)
            code, summaries, error = self.run_cli('status', '--job-id', job)
            self.assertEqual(code, 0, error)
            self.assertEqual(summaries[0]['total_horses'], 1)
            self.assertEqual(summaries[0]['tasks'], {'pedigree': {'pending': 1}})
            self.assertFalse(summaries[0]['complete'])
            code, summaries, error = self.run_cli('status')
            self.assertEqual(code, 0, error)
            self.assertEqual(len(summaries), 1)
            diagnosis = self.directory/'pedigree_diagnosis.json'
            code, _, error = self.run_cli('diagnose', '--job-id', job, '--output', str(diagnosis))
            self.assertEqual(code, 0, error)
            self.assertEqual(json.loads(diagnosis.read_text())['failed_task_count'], 0)
            code, rows, error = self.run_cli('retry', '--job-id', job, '--kind', 'pedigree')
            self.assertEqual(code, 0, error)
            self.assertEqual(rows[-1]['tasks'], 0)

    def test_resume_and_reparse_reject_wrong_job_type_before_client_creation(self):
        pedigree = self.plan()
        store = Store(self.directory)
        try:
            race = JobQueue(store).plan('2021-10-11', '2021-10-11')
        finally:
            store.close()
        with patch('keiba_scraper.cli.Client', side_effect=AssertionError('wrong job must fail before HTTP')):
            for command, job in (('backfill', pedigree), ('pedigree', race)):
                code, _, error = self.run_cli(command, '--resume', job, '--plan-only')
                self.assertEqual(code, 2, error)
            code, _, error = self.run_cli('reparse', '--job-id', pedigree)
            self.assertEqual(code, 2, error)

    def test_resume_plan_keeps_the_existing_target_list(self):
        job = self.plan()
        with patch('keiba_scraper.cli.Client', side_effect=AssertionError('plan only')):
            code, rows, error = self.run_cli('pedigree', '--resume', job, '--plan-only')
        self.assertEqual(code, 0, error)
        self.assertEqual(rows[-1]['job_id'], job)
        self.assertEqual(rows[-1]['total_horses'], 1)

    def test_scope_options_are_mutually_exclusive_and_race_export_filters_are_rejected(self):
        for args in (
            ['pedigree'],
            ['pedigree', '--horse-id', '2019102383', '--from-job-id', 'source'],
            ['pedigree', '--resume', 'ped-existing', '--horse-id', '2019102383'],
            ['export', '--dataset', 'pedigrees', '--region', 'central', '--output', 'unused.csv'],
            ['export', '--dataset', 'pedigrees', '--start-date', '2021-01-01', '--output', 'unused.csv'],
        ):
            with self.subTest(args=args), self.assertRaises(SystemExit) as exc:
                self.run_cli(*args)
            self.assertEqual(exc.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
