import gzip
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

FIXTURES = Path(__file__).parent / 'fixtures'
RACE = dict(region='central', race_id='202603030511', race_date='2026-11-08', course='福島', race_number=11,
            race_name='みちのくS')
RUNNERS = [dict(horse_number=number, frame=(number + 1) // 2, finish_status='finished') for number in range(1, 17)]


class FakeSaved:
    def __init__(self, payouts=None, runners=RUNNERS):
        self.table, self.runners = payouts, runners

    def payouts(self, region, race_id):
        return self.table, {}

    def snapshot(self, region, race_id, page):
        return self.runners


def central_payouts():
    from keiba_scraper.parsers import parse_payouts
    return parse_payouts((FIXTURES / 'payouts_central.html').read_text(encoding='utf-8'))


class WalletTests(unittest.TestCase):
    def setUp(self):
        from keiba_app.wallet import Wallet
        self.directory = TemporaryDirectory()
        self.wallet = Wallet(Path(self.directory.name) / 'app.sqlite3')
        self.wallet.add_user('a@example.com', 'A', deposit=10000)

    def tearDown(self):
        self.directory.cleanup()

    def test_bet_settles_hits_misses_and_refunds(self):
        bet = self.wallet.record_bet('a@example.com', RACE, [
            dict(bet='win', key='8', stake=200), dict(bet='trifecta', key='8>6>7', stake=100),
            dict(bet='quinella', key='6-8', stake=100), dict(bet='exacta', key='6>8', stake=100),
            dict(bet='wide', key='3-5', stake=100)])
        self.assertEqual(bet['cost'], 600)
        self.assertEqual(self.wallet.balance('a@example.com'), 9400)
        self.assertEqual(self.wallet.settle(FakeSaved(None)), 0)  # no payouts yet: stays open
        runners = [dict(row, finish_status='scratched') if row['horse_number'] == 3 else row for row in RUNNERS]
        self.assertEqual(self.wallet.settle(FakeSaved(central_payouts(), runners)), 1)
        settled = self.wallet.bet(bet['id'])
        self.assertEqual(settled['status'], 'settled')
        self.assertEqual(settled['payout'], 2 * 560 + 120110 + 8140)
        self.assertEqual(settled['refund'], 100)  # the wide ticket with the scratched horse 3
        self.assertEqual(self.wallet.balance('a@example.com'), 9400 + 2 * 560 + 120110 + 8140 + 100)
        history = self.wallet.history('a@example.com')
        self.assertEqual(history['summary']['staked'], 500)
        self.assertEqual(history['summary']['returned'], 2 * 560 + 120110 + 8140)
        self.assertEqual(history['series'][-1]['balance'], self.wallet.balance('a@example.com'))
        self.assertEqual(history['summary']['profit'], self.wallet.balance('a@example.com') - 10000)
        with self.assertRaises(ValueError):
            self.wallet.cancel_bet('a@example.com', bet['id'])
        self.wallet.void_bet('a@example.com', bet['id'])
        self.assertEqual(self.wallet.balance('a@example.com'), 10000)

    def test_unsold_bet_type_is_refunded_and_cancel_restores_balance(self):
        payouts = central_payouts()
        payouts.pop('bracket')
        bet = self.wallet.record_bet('a@example.com', RACE, [dict(bet='bracket', key='4-3', stake=300)])
        self.assertEqual(bet['tickets'][0]['key'], '3-4')
        self.wallet.settle(FakeSaved(payouts))
        self.assertEqual(self.wallet.bet(bet['id'])['refund'], 300)
        other = self.wallet.record_bet('a@example.com', RACE, [dict(bet='win', key='1', stake=100)])
        self.wallet.cancel_bet('a@example.com', other['id'])
        self.assertIsNone(self.wallet.bet(other['id']))
        self.assertEqual(self.wallet.balance('a@example.com'), 10000)

    def test_ticket_validation_and_money(self):
        for ticket in (dict(bet='win', key='8', stake=150), dict(bet='exacta', key='6-8', stake=100),
                       dict(bet='trio', key='1-1-2', stake=100), dict(bet='nope', key='1', stake=100)):
            with self.assertRaises(ValueError):
                self.wallet.record_bet('a@example.com', RACE, [ticket])
        self.assertEqual(self.wallet.move('a@example.com', 'withdraw', 3000), 7000)
        self.assertEqual(self.wallet.move('a@example.com', 'adjust', -500), 6500)
        with self.assertRaises(ValueError):
            self.wallet.move('a@example.com', 'deposit', -5)
        entry = self.wallet.history('a@example.com')['entries'][0]
        self.wallet.delete_entry('a@example.com', entry['id'])
        self.assertEqual(self.wallet.balance('a@example.com'), 7000)
        with self.assertRaises(ValueError):
            self.wallet.add_user('a@example.com', 'again')


class SavedPayoutTests(unittest.TestCase):
    def test_payouts_are_read_from_the_saved_results_page(self):
        from app_fixtures import save_snapshot
        from keiba_app.sources import SavedRaces
        with TemporaryDirectory() as directory:
            row = dict(horse_id='h1', horse_number='1', frame='1', sex='牡', age='3', jockey_id='j01', trainer_id='t01',
                       carried_weight_kg='56', weight_kg='480', weight_change_kg='0', finish_position='1',
                       finish_status='finished', time_seconds='70.1', race_date='2026-11-08', race_course='福島',
                       race_surface='芝', race_distance_m='1200', race_declared_field_size='16',
                       race_track_condition='良', race_weather='晴')
            save_snapshot(directory, 'central', RACE['race_id'], 'results', [row], '2026-11-08T17:00:00+09:00')
            (Path(directory) / 'raw').mkdir(exist_ok=True)
            (Path(directory) / 'raw' / 'none').write_bytes(gzip.compress((FIXTURES / 'payouts_central.html').read_bytes()))
            payouts, _ = SavedRaces(Path(directory) / 'keiba.sqlite3').payouts('central', RACE['race_id'])
        self.assertEqual(payouts['trifecta'], {'8>6>7': 120110})


class FakeEngine:
    def __init__(self, directory):
        from keiba_app.wallet import Wallet
        self.wallet = Wallet(Path(directory) / 'app.sqlite3')
        self.saved = FakeSaved(None)

        class Log:
            def prediction(self, identifier):
                return dict(id=identifier, **RACE) if identifier == 7 else None
        self.logbook = Log()

    def bets(self, identifier, **options):
        return dict(options)


class IdentityTests(unittest.TestCase):
    def setUp(self):
        from keiba_app.server import Api
        from keiba_app.state import App
        self.directory = TemporaryDirectory()
        self.engine = FakeEngine(self.directory.name)
        self.api = Api(App(self.directory.name), engine=self.engine)

    def tearDown(self):
        self.directory.cleanup()

    def as_(self, login=None, loopback=True):
        self.api.request.value = dict(loopback=loopback, login=login, name=None)

    def test_local_admin_adds_users_and_tailscale_users_see_only_themselves(self):
        self.as_()
        self.assertTrue(self.api.identity()['admin'])
        self.api.add_user(dict(login='b@example.com', name='B', deposit=20000))
        self.api.add_user(dict(login='c@example.com', name='C', deposit=5000))
        self.as_('b@example.com')
        me = self.api.identity()
        self.assertFalse(me['admin'])
        self.assertEqual(me['login'], 'b@example.com')
        page = self.api.wallet({})
        self.assertEqual([user['login'] for user in page['users']], ['b@example.com'])
        self.assertEqual(page['summary']['balance'], 20000)
        with self.assertRaises(ValueError):
            self.api.wallet({'user': 'c@example.com'})
        with self.assertRaises(ValueError):
            self.api.add_user(dict(login='d@example.com'))
        result = self.api.record_bet(dict(prediction_id=7, tickets=[dict(bet='win', key='8', stake=500)]))
        self.assertEqual(result['balance'], 19500)
        # headers from a request that did not come through Tailscale Serve are ignored
        self.as_('b@example.com', loopback=False)
        self.assertIsNone(self.api.identity()['login'])
        self.assertFalse(self.api.identity()['admin'])
        self.api.request.value = dict(loopback=True, login='stranger@example.com', name='Stranger')
        self.api.note_visitor()
        self.assertFalse(self.api.wallet({})['me']['registered'])
        self.as_()
        self.assertEqual([(v['login'], v['name']) for v in self.api.wallet({})['visitors']],
                         [('stranger@example.com', 'Stranger')])
        self.api.add_user(dict(login='stranger@example.com', name='S'))
        self.assertEqual(self.api.wallet({})['visitors'], [])
        self.as_('stranger@example.com')  # now a user
        self.assertEqual(self.api.move_money(dict(kind='deposit', amount=100))['balance'], 100)
        self.as_('nobody@example.com')
        with self.assertRaises(ValueError):
            self.api.move_money(dict(kind='deposit', amount=100))

    def test_bankroll_follows_the_users_balance(self):
        self.as_()
        self.api.add_user(dict(login='b@example.com', name='B', deposit=40000))
        self.as_('b@example.com')
        self.api.update_user(dict(kelly_fraction=.25))
        result = self.api.bets(dict(prediction='7', budget='1000', bankroll='50000', wallet='1'))
        self.assertEqual(result['bankroll'], 10000)
        self.assertEqual(result['bankroll_source']['kind'], 'wallet')
        self.assertEqual(self.api.bets(dict(prediction='7', bankroll='50000'))['bankroll'], 50000)
        self.as_()  # the administrator acting as B
        self.assertEqual(self.api.bets(dict(prediction='7', wallet='1', user='b@example.com'))['bankroll'], 10000)
        with self.assertRaises(ValueError):
            self.as_('b@example.com')
            self.api.update_user(dict(admin=True))


class HeaderTests(unittest.TestCase):
    def test_header_helpers(self):
        from keiba_app.server import _header_text, _loopback
        self.assertTrue(_loopback('127.0.0.1'))
        self.assertTrue(_loopback('::1'))
        self.assertTrue(_loopback('::ffff:127.0.0.1'))
        self.assertFalse(_loopback('100.64.0.5'))
        self.assertEqual(_header_text('=?utf-8?q?=E9=81=A0=E8=97=A4?='), '遠藤')
        self.assertEqual(_header_text('a@example.com'), 'a@example.com')


if __name__ == '__main__':
    unittest.main()
