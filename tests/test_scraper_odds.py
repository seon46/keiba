import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SAMPLE = {"status": "middle", "data": {"official_datetime": "2026-11-08 15:20:00", "odds": {
    "1": {"01": ["2.3", "", "1"], "02": ["15.8", "", "5"], "03": ["---.-", "", "0"]},
    "2": {"01": ["1.1", "1.3", "1"], "02": ["2.5", "4.1", "4"]},
    "4": {"0102": ["12.4", "", "3"]},
    "5": {"0102": ["3.1", "3.9", "2"]},
    "6": {"0201": ["40.1", "", "9"], "0102": ["20.5", "", "4"]},
    "7": {"010203": ["1,234.5", "", "120"]},
    "8": {"030201": ["9,876.0", "", "900"]}}}}


class OddsParserTests(unittest.TestCase):
    def test_parses_every_bet_type(self):
        from keiba_scraper.parsers import parse_odds_json
        parsed = parse_odds_json(json.dumps(SAMPLE), race_id='202605050811', region='central')
        odds = parsed['odds']
        self.assertEqual(odds['win'], {'1': 2.3, '2': 15.8})  # unsold / cancelled odds dropped
        self.assertEqual(odds['place']['2'], [2.5, 4.1])
        self.assertEqual(odds['quinella'], {'1-2': 12.4})
        self.assertEqual(odds['wide'], {'1-2': [3.1, 3.9]})
        self.assertEqual(odds['exacta'], {'2>1': 40.1, '1>2': 20.5})  # ordered keys keep the order
        self.assertEqual(odds['trio'], {'1-2-3': 1234.5})
        self.assertEqual(odds['trifecta'], {'3>2>1': 9876.0})
        self.assertEqual(parsed['popularity']['win']['1'], 1)
        self.assertEqual(parsed['official_at'], '2026-11-08 15:20:00')

    def test_rejects_unknown_shapes(self):
        from keiba_scraper.parsers import ParseError, parse_odds_json
        with self.assertRaises(ParseError):
            parse_odds_json('<html></html>', race_id='202605050811', region='central')
        with self.assertRaises(ParseError):
            parse_odds_json(json.dumps({"status": "yoso", "data": ""}), race_id='202605050811', region='central')
        with self.assertRaises(ParseError):
            parse_odds_json(json.dumps({"data": {"odds": {"4": {"012": ["1.0"]}}}}), race_id='202605050811',
                            region='central')

    def test_store_keeps_snapshots(self):
        from keiba_scraper.parsers import parse_odds_json
        from keiba_scraper.storage import Store
        with TemporaryDirectory() as name:
            store = Store(name)
            observation = store.observe('https://race.netkeiba.com/api/api_get_jra_odds.html?race_id=202605050811',
                                        json.dumps(SAMPLE).encode(), encoding='utf-8')
            store.save_odds(observation, parse_odds_json(json.dumps(SAMPLE), race_id='202605050811', region='central'))
            row = store.db.execute('SELECT race_id, metadata_json FROM odds').fetchone()
            store.close()
            self.assertEqual(row[0], '202605050811')
            self.assertEqual(json.loads(row[1])['odds']['win']['1'], 2.3)


if __name__ == '__main__':
    unittest.main()
