import unittest
from unittest.mock import patch

from keiba_scraper.conditions import extract_conditions, parse_conditions


def header(name='テスト賞', title='テスト賞 出馬表 | 2026年10月11日 東京11R', conditions='', badge=''):
    return (f'<html><head><title>{title}</title></head><body>'
            f'<h1 class="RaceName">{name}{badge}</h1>'
            f'<div class="RaceData02">{conditions}</div>')


class ConditionsTests(unittest.TestCase):
    def test_central_class_names_and_historical_aliases(self):
        for text, expected in [('新馬', 'newcomer'), ('未勝利', 'maiden'),
                               ('1勝クラス', '1win'), ('500万下', '1win'),
                               ('2勝クラス', '2win'), ('1000万下', '2win'),
                               ('3勝クラス', '3win'), ('1600万下', '3win'),
                               ('オープン', 'open'), ('OP', 'open')]:
            with self.subTest(text=text):
                self.assertEqual(parse_conditions(conditions='サラ系3歳以上 '+text)['race_class'], expected)
                self.assertEqual(parse_conditions(name='3歳以上'+text)['race_class'], expected)
                self.assertEqual(parse_conditions(title=f'テスト賞({text}) 結果・払戻 | 2025年1月1日 東京1R')['race_class'], expected)

    def test_local_classes_remain_regional_labels_including_banei_and_monbetsu(self):
        for text, expected in [('A1', 'A1'), ('B2', 'B2'), ('C3', 'C3'), ('B4', 'B4'),
                               ('C4', 'C4'), ('C2一二三', 'C2'), ('C1-2', 'C1')]:
            with self.subTest(text=text):
                self.assertEqual(parse_conditions(name=text, region='local')['race_class'], expected)
        result = parse_conditions(name='オープン記念(C2)', region='local')
        self.assertEqual(result['race_class'], 'C2')
        self.assertIsNone(parse_conditions(name='A1', region='central')['race_class'])
        self.assertIsNone(parse_conditions(conditions='1勝クラス', region='local')['race_class'])
        for text in ('A1A2', 'A2・B1', 'A1以下', 'C9', 'C2 C9'):
            with self.subTest(text=text):
                self.assertIsNone(parse_conditions(conditions=text, region='local')['race_class'])

    def test_class_words_inside_commemorative_names_are_not_conditions(self):
        for text in ('オープン記念', '新馬主誕生日記念', '未勝利記念', '金沢C2カップ', 'G1優勝記念'):
            with self.subTest(text=text):
                parsed = parse_conditions(name=text, title=text+' 結果・払戻 | 2026年1月1日 金沢1R', region='local')
                self.assertIsNone(parsed['race_class'])
                self.assertIsNone(parsed['race_grade'])
        self.assertIsNone(parse_conditions(name='ダービー記念')['race_class'])

    def test_explicit_grade_spellings_and_regional_validation(self):
        for text, expected, region in [('G1', 'G1', 'central'), ('GⅡ', 'G2', 'central'),
                                       ('GIII', 'G3', 'central'), ('J.GI', 'JG1', 'central'),
                                       ('J・G2', 'JG2', 'central'), ('JG3', 'JG3', 'central'),
                                       ('L', 'L', 'central'), ('Jpn1', 'Jpn1', 'local'),
                                       ('JpnⅡ', 'Jpn2', 'local'), ('JpnIII', 'Jpn3', 'local'),
                                       ('G1', 'G1', 'local')]:
            with self.subTest(text=text):
                parsed = parse_conditions(title=f'テスト賞({text}) 結果・払戻 | 2026年1月1日 東京1R', region=region)
                self.assertEqual(parsed['race_grade'], expected)
                self.assertIsNone(parsed['race_class'])
        for text, region in [('Jpn1', 'central'), ('JG1', 'local'), ('L', 'local'), ('S1', 'local')]:
            with self.subTest(text=text, region=region):
                self.assertIsNone(parse_conditions(name=f'テスト賞({text})', region=region)['race_grade'])

    def test_only_known_current_race_grade_icons_are_accepted(self):
        self.assertEqual(parse_conditions(grade_tokens=['Icon_GradeType', 'Icon_GradeType2'])['race_grade'], 'G2')
        self.assertIsNone(parse_conditions(grade_tokens=['Icon_GradeType13'])['race_grade'])
        self.assertIsNone(parse_conditions(grade_tokens=['Icon_GradeType1'], region='local')['race_grade'])
        self.assertEqual(parse_conditions(grade_tokens=['Icon_GradeType19', 'Jpn1'], region='local')['race_grade'], 'Jpn1')

    def test_conflicts_are_unknown_rather_than_order_dependent(self):
        cases = [dict(name='3歳未勝利', conditions='新馬'),
                 dict(name='テスト賞(1勝クラス)', conditions='2勝クラス'),
                 dict(conditions='A1 B1', region='local')]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertIsNone(parse_conditions(**arguments)['race_class'])
        self.assertIsNone(parse_conditions(name='テスト賞(G1)', grade_tokens=['G2'])['race_grade'])
        contradictory = parse_conditions(name='テスト賞(G1)', conditions='3歳1勝クラス')
        self.assertIsNone(contradictory['race_class'])
        self.assertIsNone(contradictory['race_grade'])
        self.assertIsNone(parse_conditions(conditions='定量 ハンデ')['weight_rule'])

    def test_age_weight_and_prize_are_explicit_and_raw_is_preserved(self):
        raw = '  サラ系３歳以上\n オープン (国際) 牝(指) 別定 16頭 本賞金:5500,2200,1400,830,550万円 '
        parsed = parse_conditions(conditions=raw)
        self.assertEqual(parsed['conditions_raw'], ' '.join(raw.split()))
        self.assertEqual((parsed['age_min'], parsed['age_max'], parsed['weight_rule'], parsed['first_prize_man']),
                         (3, None, '別定', 5500.0))
        for text, expected in [('サラ系2歳 馬齢', (2, 2)), ('3歳以下', (None, 3)),
                               ('2・3歳', (2, 3)), ('3歳以上4歳以下', (3, 4)),
                               ('3歳 4歳', (None, None)), ('99歳', (None, None))]:
            with self.subTest(text=text):
                parsed = parse_conditions(conditions=text)
                self.assertEqual((parsed['age_min'], parsed['age_max']), expected)
        self.assertEqual(parse_conditions(conditions='ハンデキャップ')['weight_rule'], 'ハンデ')
        self.assertEqual(parse_conditions(conditions='本賞金:8000.0、2800.0、1600.0、800.0、400.0万円')['first_prize_man'], 8000)
        self.assertEqual(parse_conditions(conditions='賞金:30.5万円')['first_prize_man'], 30.5)
        for text in ('賞金不明', '本賞金:1000円', '本賞金:3,000万円', '本賞金:0万円',
                     '本賞金:100万円 賞金:200万円'):
            with self.subTest(text=text):
                self.assertIsNone(parse_conditions(conditions=text)['first_prize_man'])

    def test_birthday_names_do_not_fabricate_age_eligibility_or_prize(self):
        parsed = parse_conditions(name='20歳誕生日記念', title='20歳誕生日記念(賞金100万円)', region='local')
        self.assertIsNone(parsed['age_min'])
        self.assertIsNone(parsed['age_max'])
        self.assertIsNone(parsed['first_prize_man'])

    def test_extracts_nested_header_and_stops_before_unrelated_body_content(self):
        from keiba_scraper.conditions import _HeaderExtractor
        html = header(name='テスト賞', title='テスト賞(G2) 出馬表 | 2026年10月11日 東京11R',
                      conditions='<span>サラ系３歳以上</span><span>オープン</span><span>別定</span>'
                                 '<span>16頭</span><br><span>本賞金:5500,2200,1400,830,550万円</span>',
                      badge='<span class="Icon_GradeType Icon_GradeType2"></span>')
        html += '<div class="RaceName">別のレース(G1)</div><p>SHOULD_NOT_BE_READ</p>'
        original = _HeaderExtractor.handle_data
        def checked(parser, data):
            self.assertNotIn('SHOULD_NOT_BE_READ', data)
            return original(parser, data)
        with patch.object(_HeaderExtractor, 'handle_data', checked):
            parsed = extract_conditions(html, 'central')
        self.assertEqual((parsed['race_class'], parsed['race_grade'], parsed['age_min'], parsed['weight_rule']),
                         ('open', 'G2', 3, '別定'))
        self.assertEqual(parsed['first_prize_man'], 5500)
        self.assertIn('サラ系３歳以上 オープン 別定', parsed['conditions_raw'])

    def test_local_badge_and_unknown_or_incomplete_headers(self):
        parsed = extract_conditions(header(name='テスト杯', conditions='サラ系３歳以上 12頭',
                                            badge='<span class="Icon_Grade_None_Text Icon_GradeType Icon_GradeType19">Jpn1</span>'), 'local')
        self.assertEqual(parsed['race_grade'], 'Jpn1')
        self.assertEqual(parsed['age_min'], 3)
        for html in ('', '<html><p>ログインしてください</p></html>',
                     '<div class="other">3歳1勝クラス G1 別定</div>',
                     '<div class="RaceData02">3歳1勝クラス 別定',
                     '<div><span class="RaceData02">3歳1勝クラス 別定</div>',
                     '<h1 class="RaceName">テスト賞<aside><span class="Icon_GradeType Icon_GradeType1"></span></aside>'):
            with self.subTest(html=html):
                parsed = extract_conditions(html, 'central')
                self.assertEqual(parsed['conditions_raw'], '')
                self.assertTrue(all(value is None for key, value in parsed.items() if key != 'conditions_raw'))


if __name__ == '__main__':
    unittest.main()
