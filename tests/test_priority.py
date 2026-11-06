from tempfile import TemporaryDirectory
import threading
import time
import unittest


class PriorityReservationTests(unittest.TestCase):
    def test_priority_clients_hold_back_background_requests(self):
        from keiba_scraper.storage import Store
        with TemporaryDirectory() as name:
            store = Store(name)
            try:
                self.assertEqual(store.reserve_request('race.netkeiba.com', 10), 0)  # background, now
                wait = store.reserve_request('race.netkeiba.com', 10, priority=True)
                self.assertAlmostEqual(wait, 10, delta=.5)  # spacing is kept
                second = store.reserve_request('race.netkeiba.com', 10, priority=True)
                self.assertAlmostEqual(second, 20, delta=.5)  # priority pages go back to back
                background = store.reserve_request('db.netkeiba.com', 10)
                self.assertGreaterEqual(background, 39.5)  # waits for the hold after the last priority page
            finally:
                store.close()


class JobLaneTests(unittest.TestCase):
    def test_odds_first_and_day_jobs_in_their_own_lane(self):
        from keiba_app.server import Jobs
        jobs = Jobs()
        release = threading.Event()
        order = []

        def blocker(progress):
            release.wait(5)
            order.append('blocker')

        def make(name):
            def work(progress):
                order.append(name)
            return work
        jobs.submit('race', 'busy', blocker)           # occupies the quick lane
        time.sleep(.1)
        jobs.submit('race', 'race', make('race'))
        jobs.submit('odds', 'odds', make('odds'))      # submitted later, runs first
        day = jobs.submit('day', 'day', make('day'))   # other lane: not blocked
        for _ in range(50):
            if 'day' in order:
                break
            time.sleep(.05)
        self.assertEqual(order, ['day'])
        listing = {job['label']: job for job in jobs.listing()}
        self.assertEqual((listing['odds']['position'], listing['race']['position']), (1, 2))
        release.set()
        for _ in range(100):
            if len(order) == 4:
                break
            time.sleep(.05)
        self.assertEqual(order, ['day', 'blocker', 'odds', 'race'])
        self.assertEqual(jobs.get(day['id'])['status'], 'done')


if __name__ == '__main__':
    unittest.main()
