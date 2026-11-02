import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
import unittest
import urllib.request


@unittest.skipUnless(all(importlib.util.find_spec(name) for name in ('lightgbm', 'sklearn', 'scipy', 'joblib')),
                     'Install requirements-ml.txt for the app tests')
class AppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app_fixtures import build
        cls.directory = TemporaryDirectory()
        cls.fixture = build(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def _app(self, name):
        from keiba_app.state import App
        app = App(self.fixture['data_dir'], home=Path(self.directory.name) / name)
        app.write_state(dataset=str(self.fixture['dataset']), model_dir=str(self.fixture['model_dir']),
                        collected_through='2025-12-31')
        return app

    def test_predict_score_records_and_api(self):
        from app_fixtures import FakeScraper, races_on, save_snapshot
        from keiba_app.engine import Engine
        from keiba_app.server import Api, make_handler
        from http.server import ThreadingHTTPServer
        app = self._app('app1')
        day = self.fixture['days_after'][2]
        races = races_on(self.fixture['rows'], day)
        listing = {}
        for (region, race_id), rows in races.items():
            save_snapshot(self.fixture['data_dir'], region, race_id, 'entries', rows, f'{day}T09:00:00+09:00')
            listing.setdefault((day, region), []).append(race_id)
        scraper = FakeScraper(listing)
        engine = Engine(app, scraper=scraper, samples=256)

        # One race from saved entries; results are not saved yet, so it waits for scoring.
        (region, race_id), rows = next(iter(sorted(races.items())))
        identifier = engine.predict_race(region, race_id)
        prediction = engine.logbook.prediction(identifier)
        self.assertEqual(prediction['starters'], 10)
        self.assertEqual(prediction['timing'], 'after')  # a 2025 race is in the past
        for runner in prediction['runners']:
            self.assertEqual(len(runner['distribution']), 11)
            self.assertAlmostEqual(sum(runner['distribution']), 1.0, places=6)
            self.assertTrue(runner['horse_name'])
        self.assertIsNone(prediction['outcome'])

        # Results arrive -> scoring
        for (region_, race_id_), race_rows in races.items():
            save_snapshot(self.fixture['data_dir'], region_, race_id_, 'results', race_rows, f'{day}T18:00:00+09:00')
        self.assertEqual(engine.score_pending(), {'scored': 1})
        outcome = engine.logbook.prediction(identifier)['outcome']
        self.assertEqual(outcome['status'], 'scored')
        self.assertLess(outcome['outcome_log_loss'], 5)
        self.assertGreater(outcome['uniform_log_loss'], 1)

        # Whole day (both regions) in one batch
        result = engine.predict_day(day, refresh=False)
        self.assertEqual(len(result['predicted']), len(races))
        records = engine.logbook.records(include_after=True)
        self.assertEqual(records['summary']['races'], len(races))
        self.assertEqual(engine.logbook.records()['summary']['races'], 0)  # only after-race predictions

        # A weekday-like day: the central list cannot be read, the local one still is
        weekday = FakeScraper({('2025-09-03', 'local'): ['202544090301']})
        engine_weekday = Engine(app, scraper=weekday, samples=64)
        listed_local = engine_weekday.discover('2025-09-03')
        self.assertEqual([race['region'] for race in listed_local], ['local'])
        self.assertIn('central', engine_weekday.discover_errors)

        # A day inside the training period is refused
        early_day = '2025-01-03'
        early = races_on(self.fixture['rows'], early_day)
        (early_region, early_id), early_rows = next(iter(early.items()))
        save_snapshot(self.fixture['data_dir'], early_region, early_id, 'entries', early_rows, f'{early_day}T09:00:00+09:00')
        with self.assertRaisesRegex(ValueError, '学習期間内'):
            engine.predict_race(early_region, early_id)

        # HTTP API
        api = Api(app, engine)
        server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(api))
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f'http://127.0.0.1:{server.server_address[1]}'
            get = lambda path: json.loads(urllib.request.urlopen(base + path).read())
            listed = get(f'/api/day?date={day}')
            self.assertEqual({race['race_id'] for race in listed['races']}, {race_id for _, race_id in races})
            self.assertTrue(all(race['predictions'] for race in listed['races']))
            detail = get(f'/api/race?region={region}&race_id={race_id}')
            self.assertEqual(len(detail['card']), 10)
            self.assertEqual(len(detail['predictions']), 2)
            self.assertIsNotNone(detail['results'])
            self.assertIn('summary', get('/api/records?days=all&after=1'))
            self.assertIn('state', get('/api/status'))
            before = get('/api/changes')['token']
            page = urllib.request.urlopen(base + '/').read().decode('utf-8')
            self.assertIn('順位分布', page)
            self.assertEqual(urllib.request.urlopen(base + '/static/app.js').status, 200)
            request = urllib.request.Request(base + '/api/race/predict', method='POST',
                                             data=json.dumps(dict(region=region, race_id=race_id)).encode(),
                                             headers={'Content-Type': 'application/json'})
            job = json.loads(urllib.request.urlopen(request).read())
            for _ in range(600):
                state = get(f'/api/job?id={job["id"]}')
                if state['status'] in ('done', 'error'):
                    break
                threading.Event().wait(.1)
            self.assertEqual(state['status'], 'done', state)
            self.assertNotEqual(get('/api/changes')['token'], before)
        finally:
            server.shutdown()
            server.server_close()

    def test_nightly_scores_and_checks(self):
        from app_fixtures import FakeScraper
        from keiba_app.engine import Engine
        from keiba_app.nightly import check_model, model_info, run_nightly
        app = self._app('app2')
        engine = Engine(app, scraper=FakeScraper(), samples=128)
        result = run_nightly(app, only=('score', 'precompute'), engine=engine)
        self.assertEqual([step['step'] for step in result['steps']], ['score', 'precompute'])
        self.assertEqual(result['status'], 'ok')
        self.assertEqual(engine.logbook.runs()[0]['status'], 'ok')
        self.assertEqual(check_model(model_info(self.fixture['model_dir'])), [])
        self.assertTrue(check_model({'groups': {}}))


if __name__ == '__main__':
    unittest.main()
