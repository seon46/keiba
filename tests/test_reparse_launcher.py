from contextlib import closing, redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from keiba_scraper.client import Client
from keiba_scraper.jobs import JobQueue, Worker
from keiba_scraper.storage import Store
from keiba_scraper.urls import list_url, race_url
from scripts import reparse_saved_data as launcher
from test_jobs import DAY, RACE, LISTING
from test_recovery import JOCKEY_RESULT


class ReparseLauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def test_real_reparse_updates_saved_data_and_writes_summary(self):
        repo = self.directory / 'repo with spaces'
        shutil.copytree(Path(__file__).resolve().parents[1] / 'src', repo / 'src')
        stale_package = self.directory / 'stale package' / 'keiba_scraper'
        stale_package.mkdir(parents=True)
        (stale_package / '__init__.py').write_text("raise AssertionError('Old installed package was used')")
        with closing(Store(repo / 'data')) as store:
            client = Client(store, interval=0)
            client.session.get = Mock(side_effect=AssertionError('Unexpected HTTP'))
            store.observe(list_url(DAY, 'central'), LISTING.encode(), encoding='utf-8')
            observation = store.observe(race_url(RACE, 'central', 'results'), JOCKEY_RESULT.encode(), encoding='utf-8')
            queue = JobQueue(store)
            job = queue.plan(DAY, DAY, regions=('central',))
            self.assertTrue(Worker(queue, client, offline=True).run(job, min_free_mb=0)['complete'])
            with store.db:
                store.db.execute("UPDATE runners SET payload_json=json_set(payload_json,'$.jockey_id',NULL)")
            observations = [tuple(row) for row in store.db.execute('SELECT * FROM observations ORDER BY id')]
            client.close()
        with patch.dict(os.environ, {'PYTHONPATH': str(stale_package.parent)}), redirect_stdout(io.StringIO()):
            summary = launcher.reparse(repo, job)
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['done'], 1)
        self.assertEqual(json.loads((repo / 'data/reparse_report.json').read_text())['complete'], True)
        self.assertIn('reparsed', (repo / 'data/reparse_console.log').read_text())
        with closing(Store(repo / 'data')) as store:
            row = store.export_rows(dataset='runners')[0]
            self.assertEqual(row['jockey_id'], 'a03fd')
            self.assertEqual((row['snapshot_id'], row['retrieved_at']), (observation.id, observation.retrieved_at))
            self.assertEqual([tuple(row) for row in store.db.execute('SELECT * FROM observations ORDER BY id')], observations)

    def test_zero_exit_with_incomplete_summary_does_not_claim_completion(self):
        (self.directory / 'data').mkdir()
        shutil.copytree(Path(__file__).resolve().parents[1] / 'src', self.directory / 'src')
        real_popen = subprocess.Popen
        paused = dict(event='reparse_summary', complete=False, done=0, pending=1, failed=0)
        script = 'print(' + repr(json.dumps(paused)) + ')'
        def paused_process(command, **kwargs):
            return real_popen([sys.executable, '-c', script], **kwargs)
        with patch.object(launcher.subprocess, 'Popen', side_effect=paused_process), redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, '未完了'):
                launcher.reparse(self.directory, 'test-job')
        self.assertFalse(json.loads((self.directory / 'data/reparse_report.json').read_text())['complete'])


@unittest.skipUnless(shutil.which('git'), 'Git is required for local update tests')
class LocalBundleUpdateTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.repo = self.root / 'source', self.root / 'user repo'
        self.source.mkdir()
        self.git(self.source, 'init', '-q', '-b', 'main')
        self.git(self.source, 'config', 'user.name', 'Test')
        self.git(self.source, 'config', 'user.email', 'test@example.invalid')
        (self.source / 'version').write_text('old')
        self.git(self.source, 'add', 'version')
        self.git(self.source, 'commit', '-qm', 'old')
        self.git(self.source, 'clone', '-q', '--local', str(self.source), str(self.repo))
        (self.repo / 'data').mkdir()
        (self.repo / 'data/keiba.sqlite3').write_bytes(b'existing user database marker')
        (self.source / 'version').write_text('new')
        self.git(self.source, 'commit', '-qam', 'new')
        self.bundle = self.root / 'download folder' / 'keiba-recovery.bundle'
        self.bundle.parent.mkdir()
        self.git(self.source, 'bundle', 'create', str(self.bundle), 'HEAD')

    @staticmethod
    def git(repo, *args):
        return subprocess.run(['git', '-C', str(repo), *args], check=True, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout.strip()

    def test_absolute_bundle_path_updates_repo_and_preserves_data(self):
        real_run = subprocess.run
        def quiet_run(command, **kwargs):
            return real_run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        with patch.object(launcher.subprocess, 'run', side_effect=quiet_run):
            launcher.update_code(self.repo, self.bundle)
        self.assertEqual(self.git(self.repo, 'rev-parse', 'HEAD'), self.git(self.source, 'rev-parse', 'HEAD'))
        self.assertEqual((self.repo / 'data/keiba.sqlite3').read_bytes(), b'existing user database marker')

    def test_non_fast_forward_stops_before_reparse_and_keeps_user_commit(self):
        self.git(self.repo, 'config', 'user.name', 'Test')
        self.git(self.repo, 'config', 'user.email', 'test@example.invalid')
        (self.repo / 'user-file').write_text('user change')
        self.git(self.repo, 'add', 'user-file')
        self.git(self.repo, 'commit', '-qm', 'user commit')
        before = self.git(self.repo, 'rev-parse', 'HEAD')
        real_run = subprocess.run
        def quiet_run(command, **kwargs):
            return real_run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
        with patch.object(launcher, 'reparse') as reparse, patch.object(launcher.subprocess, 'run', side_effect=quiet_run):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                result = launcher.main(['--repo', str(self.repo), '--bundle', str(self.bundle)])
            self.assertEqual(result, 2)
            reparse.assert_not_called()
        self.assertEqual(self.git(self.repo, 'rev-parse', 'HEAD'), before)
        self.assertEqual((self.repo / 'user-file').read_text(), 'user change')
        self.assertEqual((self.repo / 'data/keiba.sqlite3').read_bytes(), b'existing user database marker')


if __name__ == '__main__':
    unittest.main()
