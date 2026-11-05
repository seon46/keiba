import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest


SAMPLE = {"status": "middle", "data": {"official_datetime": "2026-11-08 15:20:00", "odds": {
    "1": {"1": ["2.3", "", "1"], "2": ["15.8", "", "5"], "3": ["---.-", "", "0"]},
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
        with self.assertRaisesRegex(ParseError, '発売前'):  # real response of a race days before sales
            parse_odds_json(json.dumps({"status": "yoso", "data": {"odds": {"1": {"18": ["4.8", "", "1"]}}},
                                        "reason": "empty free odds schedule"}), race_id='202605050211',
                            region='central')
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


# Same structure as nar.netkeiba.com odds pages (b1: two horse tables, b4: cells with ids), invented numbers.
NAR_B1 = """<table class="RaceOdds_HorseList_Table"><tr><th>枠</th><th>馬番</th><th>印</th><th>選択</th><th>馬名</th><th>オッズ</th></tr>
<tr><td class="Waku1">1</td><td class="W31">1</td><td></td><td></td><td class="Horse_Name">A</td><td class="Odds">14.1</td></tr>
<tr><td class="Waku2">2</td><td class="W31">2</td><td></td><td></td><td class="Horse_Name">B</td><td class="Odds">3.7</td></tr></table>
<table class="RaceOdds_HorseList_Table"><tr><th>枠</th><th>馬番</th><th>印</th><th>選択</th><th>馬名</th><th>オッズ</th></tr>
<tr><td class="Waku1">1</td><td>1</td><td></td><td></td><td class="Horse_Name">A</td><td class="Odds">1.7 - 3.7</td></tr>
<tr><td class="Waku2">2</td><td>2</td><td></td><td></td><td class="Horse_Name">B</td><td class="Odds">1.1 - 2.0</td></tr></table>"""
NAR_B4 = """<table class="Odds_Table"><tr><th class="Waku1">1</th></tr>
<tr><td class="Waku_Normal">2</td><td class="Odds" id="chk_a5-47-4_b4_c0_1_2">18.3</td></tr>
<tr><td class="Waku_Normal">3</td><td class="Odds" id="chk_a5-47-4_b4_c0_1_3">285.0</td></tr></table>"""


class LocalOddsTests(unittest.TestCase):
    def test_win_place_and_quinella(self):
        from keiba_scraper.parsers import parse_local_odds
        b1 = parse_local_odds(NAR_B1, race_id='202647110504', kind='b1')['odds']
        self.assertEqual(b1['win'], {'1': 14.1, '2': 3.7})
        self.assertEqual(b1['place'], {'1': [1.7, 3.7], '2': [1.1, 2.0]})
        b4 = parse_local_odds(NAR_B4, race_id='202647110504', kind='b4')['odds']
        self.assertEqual(b4['quinella'], {'1-2': 18.3, '1-3': 285.0})
        exacta = NAR_B4.replace('_b4_c0_1_2', '_b6_c0_2_1').replace('_b4_c0_1_3', '_b6_c0_1_3')
        self.assertEqual(parse_local_odds(exacta, race_id='x', kind='b6')['odds']['exacta'], {'2>1': 18.3, '1>3': 285.0})
        wide = NAR_B4.replace('_b4_', '_b5_').replace('18.3', '1.3 - 1.4')
        self.assertEqual(parse_local_odds(wide, race_id='x', kind='b5')['odds']['wide'], {'1-2': [1.3, 1.4], '1-3': [285.0, 285.0]})
        bracket = NAR_B4.replace('_b4_c0_1_2', '_b3_c0_7_7').replace('_b4_', '_b3_')
        self.assertEqual(parse_local_odds(bracket, race_id='x', kind='b3')['odds']['bracket'], {'7-7': 18.3, '1-3': 285.0})

    def test_presale_pages_have_no_odds(self):
        from keiba_scraper.parsers import parse_local_odds
        import re
        presale = re.sub(r'(class="Odds">)[^<]*', lambda m: m.group(1) + '0.0 - 0.0' if ' - ' in m.group(0) else
                         m.group(1) + '0.0', NAR_B1)
        self.assertEqual(parse_local_odds(presale, race_id='202630110512', kind='b1')['odds'], {})
