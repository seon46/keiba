"""Small fictional pages covering the public HTML formats and failure cases."""

import unittest

from keiba_scraper.parsers import ParseError, parse_horse_history, parse_race


RACE_ID = "202605040411"
HORSE = '<a href="https://db.netkeiba.com/horse/2021100001/">テスト馬</a>'
JOCKEY = '<a href="https://db.netkeiba.com/jockey/result/recent/00123/">騎手A</a>'
TRAINER = '<span>美浦</span><a href="https://db.netkeiba.com/trainer/00456/">調教師B</a>'


def document(body):
    return (
        '<html><head><title>テスト賞 出馬表 | 2026年10月11日 東京11R - netkeiba</title>'
        f'<link rel="canonical" href="https://race.netkeiba.com/race/shutuba.html?race_id={RACE_ID}">'
        '</head><body><h1 class="RaceName">テスト賞</h1>'
        '<div class="RaceData01">15:45発走 / 芝1800m (左 A) / 天候:晴 / 馬場:良</div>'
        '<div class="RaceData02">4回 東京 4日目 2頭</div>'
        + body + '</body></html>'
    )


def table(headers, rows, css="ShutubaTable", identifier=""):
    return (
        f'<table class="{css}" id="{identifier}"><thead><tr>'
        + ''.join(f'<th>{value}</th>' for value in headers)
        + '</tr></thead><tbody>'
        + ''.join('<tr>' + ''.join(f'<td>{value}</td>' for value in row) + '</tr>' for row in rows)
        + '</tbody></table>'
    )


ENTRY_HEADERS = ['枠', '馬<br>番', '馬名', '性齢', '斤量', '騎手', '厩舎', '馬体重(増減)', 'オッズ', '人気']
ENTRY_ROW = ['1', '2', HORSE, '牝4', '55.0', JOCKEY, TRAINER, '460(-4)', '8.5', '3']
RESULT_HEADERS = ['着順', '枠', '馬番', '馬名', '性齢', '斤量', '騎手', 'タイム', '着差',
                  '人気', '単勝オッズ', '後3F', 'コーナー通過順', '厩舎', '馬体重(増減)']
RESULT_ROW = ['1', '1', '2', HORSE, '牡3', '57', JOCKEY, '1:25.6', '',
              '2', '4.6', '36.8', '3-3-2-1', TRAINER, '499(+2)']


def corner_table(rows):
    return '<table class="RaceCommon_Table Corner_Num">' + ''.join(
        f'<tr><th><strong>{corner}</strong>コーナー</th><td>{order}</td></tr>'
        for corner, order in rows
    ) + '</table>'


def result_rows(numbers, scratched=()):
    rows = []
    for index, number in enumerate(numbers, 1):
        row = RESULT_ROW.copy()
        row[0] = '取消' if number in scratched else str(index)
        row[2] = str(number)
        row[3] = HORSE.replace('2021100001', str(2021100000 + number))
        row[12] = ''
        if number in scratched:
            row[7] = ''
        rows.append(row)
    return rows


class RaceParserTests(unittest.TestCase):
    def parse(self, body, page="entries", region="central"):
        return parse_race(document(body), race_id=RACE_ID, region=region, page=page)

    def test_entries_keep_ids_and_metadata(self):
        result = self.parse(table(ENTRY_HEADERS, [ENTRY_ROW]))
        race, runner = result['race'], result['runners'][0]
        self.assertEqual((race['race_date'], race['course'], race['distance_m']), ('2026-10-11', '東京', 1800))
        self.assertEqual((runner['horse_id'], runner['jockey_id'], runner['trainer_id']),
                         ('2021100001', '00123', '00456'))
        self.assertEqual((runner['sex'], runner['age'], runner['horse_number']), ('牝', 4, 2))
        self.assertEqual((runner['weight_kg'], runner['weight_change_kg']), (460, -4))
        self.assertEqual((runner['odds'], runner['popularity']), (8.5, 3))
        self.assertNotIn('finish_position', runner)

    def test_nar_malformed_nested_weight_cell_and_missing_values(self):
        body = (
            '<table class="ShutubaTable"><tr>'
            '<th>枠</th><th>馬番</th><th>馬名</th><th>性齢</th><th>斤量</th>'
            '<th>騎手</th><th>厩舎</th><th>馬体重(増減)</th><th>オッズ</th><th>人気</th>'
            '</tr><tr><td>1</td><td>2</td><td>' + HORSE + '</td><td><span>セ7</span></td>'
            '<td>57</td><td>' + JOCKEY + '</td><td>' + TRAINER + '</td>'
            '<td class="Weight"><td class="Popular">21.3</td><td>5</td></td></tr></table>'
        )
        runner = self.parse(body, region="local")['runners'][0]
        self.assertIsNone(runner['weight_kg'])
        self.assertIsNone(runner['weight_change_kg'])
        self.assertEqual((runner['odds'], runner['popularity'], runner['sex']), (21.3, 5, 'セ'))

    def test_central_blank_odds_header_and_reordered_columns(self):
        body = (
            '<table class="ShutubaTable"><tr><th>馬名</th><th>馬番</th>'
            '<th class="Popular">更新</th><th class="Popular Popular_Ninki">人気</th>'
            '<th>馬体重(増減)</th></tr><tr><td>' + HORSE + '</td><td>2</td>'
            '<td>---.-</td><td>**</td><td>480(0)</td></tr></table>'
        )
        runner = self.parse(body)['runners'][0]
        self.assertIsNone(runner['odds'])
        self.assertIsNone(runner['popularity'])
        self.assertEqual(runner['weight_change_kg'], 0)

    def test_entries_cancellation_is_not_finish_position_zero(self):
        body = table(ENTRY_HEADERS, [ENTRY_ROW]).replace('<tbody><tr>', '<tbody><tr class="HorseList Cancel">')
        runner = self.parse(body)['runners'][0]
        self.assertEqual(runner['finish_status'], 'scratched')
        self.assertIsNone(runner['finish_position'])

    def test_results_ignore_payout_and_convert_times(self):
        scratched = RESULT_ROW.copy()
        scratched[0] = '取消'
        scratched[2] = '3'
        scratched[3] = HORSE.replace('2021100001', '2021100002')
        scratched[7] = ''
        payout = table(['単勝', '馬名', '払戻'], [['2', HORSE, '460円']], css='Payout_Detail_Table')
        result = self.parse(payout + table(RESULT_HEADERS, [RESULT_ROW, scratched],
                                          css='RaceTable01', identifier='All_Result_Table'), page='results')
        self.assertEqual(len(result['runners']), 2)
        winner, withdrawn = result['runners']
        self.assertEqual((winner['finish_position'], winner['time_seconds']), (1, 85.6))
        self.assertEqual(winner['corner_order'], '3-3-2-1')
        self.assertIsNone(winner['margin'])
        self.assertEqual(withdrawn['finish_status'], 'scratched')
        self.assertIsNone(withdrawn['time_seconds'])

    def test_finish_statuses_and_ties(self):
        rows = []
        for index, position in enumerate(['1(同着)', '中止', '失格', '除外'], 1):
            row = RESULT_ROW.copy()
            row[0] = position
            row[2] = str(index)
            row[3] = HORSE.replace('2021100001', f'202110000{index}')
            rows.append(row)
        result = self.parse(table(RESULT_HEADERS, rows), page='results')
        self.assertEqual([r['finish_status'] for r in result['runners']],
                         ['finished', 'did_not_finish', 'disqualified', 'excluded'])

    def test_empty_login_missing_id_and_duplicate_rows_are_errors(self):
        missing_id = ENTRY_ROW.copy()
        missing_id[2] = 'テスト馬'
        for body in ('<h1>ログインしてください</h1>', table(ENTRY_HEADERS, []),
                     table(ENTRY_HEADERS, [missing_id]), table(ENTRY_HEADERS, [ENTRY_ROW, ENTRY_ROW])):
            with self.subTest(body=body[:60]), self.assertRaises(ParseError):
                self.parse(body)

    def test_results_without_confirmed_ranks_are_errors(self):
        unfinished = RESULT_ROW.copy()
        unfinished[0] = '--'
        for body in (table(ENTRY_HEADERS, [ENTRY_ROW]), table(RESULT_HEADERS, [unfinished]),
                     '<p>結果は未確定です</p>' + table(RESULT_HEADERS, [RESULT_ROW])):
            with self.subTest(body=body[:60]), self.assertRaises(ParseError):
                self.parse(body, page='results')

    def test_wrong_race_response_is_error(self):
        with self.assertRaises(ParseError):
            parse_race(document(table(ENTRY_HEADERS, [ENTRY_ROW])),
                       race_id='202605040412', region='central', page='entries')

    def test_header_horselist_class_is_not_a_runner(self):
        body = table(ENTRY_HEADERS, [ENTRY_ROW]).replace('<thead><tr>', '<thead><tr class="HorseList">')
        result = self.parse(body)
        self.assertEqual(len(result['runners']), 1)
        self.assertTrue(result['race']['race_id_verified'])

    def test_og_url_identity_and_absent_metadata(self):
        body = table(ENTRY_HEADERS, [ENTRY_ROW])
        valid = f'<meta property="og:url" content="https://race.netkeiba.com/race/result.html?race_id={RACE_ID}">' + body
        parsed = parse_race(valid, race_id=RACE_ID, region='central', page='entries')
        self.assertTrue(parsed['race']['race_id_verified'])
        parsed = parse_race(body, race_id=RACE_ID, region='central', page='entries')
        self.assertFalse(parsed['race']['race_id_verified'])
        with self.assertRaises(ParseError):
            parse_race(valid, race_id='202605040412', region='central', page='entries')

    def test_banei_results_preserve_b_ids_sled_weight_and_cancellation(self):
        first = RESULT_ROW.copy()
        first[3] = HORSE.replace('2021100001', 'B201800249')
        first[5] = '620.0'
        first[6] = JOCKEY.replace('00123', 'B0059')
        first[7] = '2:11.3'
        first[13] = TRAINER.replace('00456', 'B0057')
        first[14] = '894(-5)'
        withdrawn = first.copy()
        withdrawn[0] = '取消'
        withdrawn[2] = '3'
        withdrawn[3] = HORSE.replace('2021100001', 'B201700029')
        withdrawn[7] = ''
        html = document(table(RESULT_HEADERS, [first, withdrawn]))
        html = html.replace('東京11R', '帯広(ば)1R').replace(
            '15:45発走 / 芝1800m (左 A) / 天候:晴 / 馬場:良',
            '14:35発走 / 直200m / 天候:曇 / 水分量:3.7')
        result = parse_race(html, race_id=RACE_ID, region='local', page='results')
        race, runners = result['race'], result['runners']
        self.assertEqual(race['race_type'], 'banei')
        self.assertEqual((race['distance_m'], race['track_direction'], race['track_moisture']), (200, '直線', 3.7))
        self.assertNotIn('surface', race)
        self.assertEqual((runners[0]['horse_id'], runners[0]['jockey_id'], runners[0]['trainer_id']),
                         ('B201800249', 'B0059', 'B0057'))
        self.assertEqual(runners[0]['carried_weight_kg'], 620.0)
        self.assertEqual(runners[0]['weight_kg'], 894)
        self.assertIsNone(runners[1]['finish_position'])
        self.assertEqual(runners[1]['finish_status'], 'scratched')

    def test_local_personnel_mixed_ids_are_retained_without_guessing(self):
        row = ENTRY_ROW.copy()
        row[5] = JOCKEY.replace('00123', 'a03fd')
        row[6] = TRAINER.replace('https://db.netkeiba.com/trainer/00456/',
                                 'https://db.netkeiba.com/trainer/result/recent/a024c')
        runner = self.parse(table(ENTRY_HEADERS, [row]), region='local')['runners'][0]
        self.assertEqual((runner['jockey_id'], runner['trainer_id']), ('a03fd', 'a024c'))
        row[5] = '<a href="https://db.netkeiba.com/jockey/result/recent/">騎手A</a>'
        runner = self.parse(table(ENTRY_HEADERS, [row]), region='local')['runners'][0]
        self.assertIsNone(runner['jockey_id'])
        self.assertEqual(runner['jockey_name'], '騎手A')

    def test_known_weight_survives_unknown_previous_measurement(self):
        for text, expected in [('480(前計不)', (480, None)), ('480(計不)', (480, None)),
                               ('４８０（前計不）', (480, None)), ('480()', (480, None)),
                               ('480(-2)', (480, -2)), ('480(0)', (480, 0)),
                               ('計不', (None, None)), ('不明480', (None, None))]:
            with self.subTest(text=text):
                row = ENTRY_ROW.copy()
                row[7] = text
                runner = self.parse(table(ENTRY_HEADERS, [row]), region='local')['runners'][0]
                self.assertEqual((runner['weight_kg'], runner['weight_change_kg']), expected)

    def test_local_corner_groups_share_leading_rank_and_preserve_original_structure(self):
        body = table(RESULT_HEADERS, result_rows([1, 8, 2, 4], scratched=[4]))
        body += corner_table([
            (1, '(<span>1</span>,8),2'), (2, '(1,8)-2'),
            (3, '2,(1,8)'), (4, '(1,8,2)'),
        ])
        result = self.parse(body, region='local', page='results')
        race = result['race']
        runners = {runner['horse_number']: runner for runner in result['runners']}
        self.assertEqual(race['corner_passages'][0]['groups'], [[1, 8], [2]])
        self.assertIn('-', race['corner_passages'][1]['raw'])
        self.assertEqual(race['corner_position_basis'], 'parallel_group_leading_rank')
        self.assertEqual(runners[1]['corner_positions'], {'1': 1, '2': 1, '3': 2, '4': 1})
        self.assertEqual(runners[8]['corner_positions'], runners[1]['corner_positions'])
        self.assertEqual(runners[2]['corner_positions'], {'1': 3, '2': 3, '3': 1, '4': 1})
        self.assertEqual(runners[8]['corner_group_ids'], {'1': 1, '2': 1, '3': 2, '4': 1})
        self.assertEqual(runners[1]['corner_order'], '1-1-2-1')
        self.assertEqual(runners[1]['corner_order_corners'], [1, 2, 3, 4])
        self.assertEqual(runners[1]['corner_order_source'], 'corner_table')
        self.assertEqual(runners[4]['finish_status'], 'scratched')
        self.assertIsNone(runners[4]['corner_order'])
        self.assertNotIn('corner_positions', runners[4])

    def test_missing_corner_for_a_horse_is_not_imputed(self):
        body = table(RESULT_HEADERS, result_rows([1, 2, 3], scratched=[3]))
        body += corner_table([(3, '1,2'), (4, '2')])
        result = self.parse(body, region='local', page='results')
        runners = {runner['horse_number']: runner for runner in result['runners']}
        self.assertEqual(runners[1]['corner_positions'], {'3': 1})
        self.assertIsNone(runners[1]['corner_order'])
        self.assertEqual(runners[2]['corner_order'], '2-1')
        self.assertEqual(runners[2]['corner_order_corners'], [3, 4])
        self.assertTrue(any('missing_finished_runners' in warning
                            for warning in result['race']['parse_warnings']))

    def test_broken_or_unknown_corner_values_are_warned_without_fabricating_order(self):
        for passage in ['(1,2', '1,99', '1,1,2', '1,x,2', '1,,2', '1 2', '1,2,3']:
            with self.subTest(passage=passage):
                body = table(RESULT_HEADERS, result_rows([1, 2, 3], scratched=[3]))
                body += corner_table([(3, passage), (4, '1,2')])
                result = self.parse(body, region='local', page='results')
                self.assertTrue(any('corner_table_invalid' in warning
                                    for warning in result['race']['parse_warnings']))
                self.assertEqual([row['corner'] for row in result['race']['corner_passages']], [4])
                self.assertEqual(result['runners'][0]['corner_positions'], {'4': 1})
                self.assertIsNone(result['runners'][0]['corner_order'])

    def test_duplicate_corner_rows_are_ambiguous_and_primary_column_is_preserved(self):
        body = table(RESULT_HEADERS, result_rows([1, 2]))
        body += corner_table([(3, '1,2'), (3, '2,1'), (4, '1,2')])
        result = self.parse(body, region='local', page='results')
        self.assertTrue(any('duplicate_corner' in warning for warning in result['race']['parse_warnings']))
        self.assertEqual([row['corner'] for row in result['race']['corner_passages']], [4])
        self.assertIsNone(result['runners'][0]['corner_order'])

        body = table(RESULT_HEADERS, [RESULT_ROW]) + corner_table([(3, '2'), (4, '2')])
        for region in ('central', 'local'):
            with self.subTest(region=region):
                result = self.parse(body, region=region, page='results')
                self.assertEqual(result['runners'][0]['corner_order'], '3-3-2-1')
                self.assertNotIn('corner_order_source', result['runners'][0])

    def test_truncated_corner_html_is_warned_before_dom_repairs_it(self):
        for fragment in (
            '<table class="Corner_Num"><tr><th>3コーナー</th><td>1,2',
            '<table class="Corner_Num"><tr><th>3コーナー</th><td>1,2</tr></table>',
        ):
            with self.subTest(fragment=fragment):
                body = table(RESULT_HEADERS, result_rows([1, 2])) + fragment
                result = self.parse(body, region='local', page='results')
                self.assertTrue(any('corner_table_malformed_html' in warning
                                    for warning in result['race']['parse_warnings']))
                self.assertNotIn('corner_passages', result['race'])
                self.assertNotIn('corner_positions', result['runners'][0])
                self.assertIsNone(result['runners'][0]['corner_order'])


class HorseHistoryParserTests(unittest.TestCase):
    def test_history_retains_overseas_ids_and_ignores_paid_indices(self):
        headers = ['日付', '開催', 'レース名', '頭数', '枠番', '馬番', 'オッズ', '人気',
                   '着順', '騎手', '斤量', '距離', '馬場', '馬場指数', 'タイム',
                   '通過', '上り', '馬体重']
        rows = [
            ['2025/04/05', '海外', '<a href="/race/2025J0010109/">海外賞</a>', '10', '2', '3',
             '**', '−', '取消', JOCKEY, '57', 'ダ2000', '良', '**', '', '', '', '計不'],
            ['2025/03/02', '1東京8', '<a href="/race/202505010811/">テスト賞</a>', '12', '3', '4',
             '2.2', '1', '2', JOCKEY, '57', 'ダ1600', '稍', '-16', '1:38.6', '2-2-2', '35.9', '492(+5)'],
        ]
        html = '<h1>テスト馬</h1>' + table(headers, rows, css='db_h_race_results')
        result = parse_horse_history(html, horse_id='2021100001')
        self.assertEqual(result['horse_name'], 'テスト馬')
        cancelled, previous = result['runs']
        self.assertEqual(cancelled['race_id'], '2025J0010109')
        self.assertIsNone(cancelled['odds'])
        self.assertEqual(cancelled['finish_status'], 'scratched')
        self.assertEqual((previous['course'], previous['race_date'], previous['distance_m']),
                         ('東京', '2025-03-02', 1600))
        self.assertEqual(previous['time_seconds'], 98.6)
        self.assertNotIn('track_index', previous)

    def test_profile_without_public_history_is_error(self):
        with self.assertRaises(ParseError):
            parse_horse_history('<h1>テスト馬</h1><div>戦績を読み込み中</div>', horse_id='2021100001')

    def test_local_history_retains_mixed_jockey_id_and_known_weight(self):
        html = '<h1>テスト馬</h1>' + table(
            ['日付', 'レース名', '着順', '騎手', '馬体重'],
            [['2025/03/02', '<a href="/race/202535030201/">テスト賞</a>', '1',
              JOCKEY.replace('00123', 'a02e2'), '480(前計不)']], css='db_h_race_results')
        run = parse_horse_history(html, horse_id='2021100001')['runs'][0]
        self.assertEqual(run['jockey_id'], 'a02e2')
        self.assertEqual(run['weight_kg'], 480)
        self.assertIsNone(run['weight_change_kg'])

    def test_history_name_and_response_identity(self):
        html = (
            '<link rel="canonical" href="https://db.netkeiba.com/horse/result/2021100001/">'
            '<h1>テスト馬 (Test Horse)の競走成績</h1>'
            + table(['日付', 'レース名', '着順'],
                    [['2025/03/02', '<a href="/race/202505010811/">テスト賞</a>', '1']],
                    css='db_h_race_results')
        )
        self.assertEqual(parse_horse_history(html, horse_id='2021100001')['horse_name'], 'テスト馬')
        with self.assertRaises(ParseError):
            parse_horse_history(html, horse_id='2021100002')


if __name__ == '__main__':
    unittest.main()
