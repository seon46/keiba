import unittest

from keiba_scraper.urls import discover_day, discover_races, horse_url, list_url, race_url


def daily_fragment(cards, *, day='20211016', marker='cp'):
    return (
        f'<!-- block=race_list_sub ({marker}) -->'
        '<div class="RaceList_Body RaceList_Top"><div class="RaceList_Box">'
        '<dl class="RaceList_DataList"><dt class="RaceList_DataHeader">'
        f'<a href="../top/payback_list.html?kaisai_date={day}">払戻一覧</a>'
        '</dt><dd class="RaceList_Data"><ul>' + ''.join(cards)
        + '</ul></dd></dl></div></div>'
    )


def race_card(identifier, *, host=None):
    prefix = f'https://{host}' if host else '..'
    return (f'<li class="RaceList_DataItem"><a href="{prefix}/race/result.html?race_id={identifier}">'
            '<div class="RaceData">10:15 ダ1600m 12頭</div></a>'
            f'<a href="../race/movie.html?race_id={identifier}">映像</a></li>')


def unpublished_fragment(public_cards=(), *, reason='開催2~3日前 公開予定です。'):
    unpublished = (
        '<dl class="RaceList_DataList" style="display:none">'
        '<dt class="RaceList_DataHeader"><p class="RaceList_DataTitle">'
        '<small>-回</small> 帯広(ば) <small>-日目</small></p></dt>'
        '<dd class="RaceList_Data RaceList_Data_Infomation"><ul></ul>'
        '<div class="Race_Infomation_Box">' + reason + '</div></dd></dl>'
    )
    if public_cards:
        html = daily_fragment(public_cards, day='20250427', marker='d')
        return html.rsplit('</div></div>', 1)[0] + unpublished + '</div></div>'
    return ('<!-- block=race_list_sub (d) -->'
            '<div class="RaceList_Body RaceList_Top"><div class="RaceList_Box">'
            + unpublished + '</div></div>')


class URLTests(unittest.TestCase):
    def test_exact_source_urls_and_identifier_validation(self):
        self.assertEqual(race_url('202405021211', 'central', 'results'),
                         'https://race.netkeiba.com/race/result.html?race_id=202405021211')
        self.assertIn('nar.netkeiba.com/race/shutuba.html', race_url('202635101212', 'local', 'entries'))
        self.assertEqual(horse_url('2019102383'), 'https://db.netkeiba.com/horse/result/2019102383/')
        for value in ('2025J0010109', '../202405021211', '202405021211&other=1', '２０２４０５０２１２１１'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                race_url(value, 'central', 'results')

    def test_date_validation_and_no_unbounded_discovery(self):
        self.assertIn('kaisai_date=20240526', list_url('2024-05-26', 'central'))
        with self.assertRaises(ValueError): list_url('2024-02-30', 'local')
        with self.assertRaises(ValueError): discover_races('<html>login required</html>')

    def test_discovery_deduplicates_links_without_deriving_ids(self):
        html = '''<a href="/race/shutuba.html?race_id=202405021211">A</a>
        <a href="/race/result.html?race_id=202405021211">B</a>
        <a href="/race/shutuba.html?race_id=202405021212">C</a>
        <a href="/race/?race_id=bad">D</a>'''
        self.assertEqual(discover_races(html), ['202405021211', '202405021212'])

    def test_banei_horse_id_remains_a_source_identifier(self):
        self.assertEqual(horse_url('B201800249'), 'https://db.netkeiba.com/horse/result/B201800249/')
        for value in ('B20180024', 'B201800249/../x', 'b201800249'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                horse_url(value)

    def test_daily_discovery_scopes_cards_and_counts_all_meetings(self):
        html = daily_fragment([race_card('202105040301'), race_card('202105040302')])
        html += '<aside><a href="/race/result.html?race_id=202406010101&kaisai_date=20240106">おすすめ</a></aside>'
        result = discover_day(html, day='2021-10-16', region='central')
        self.assertEqual(result['race_ids'], ['202105040301', '202105040302'])
        self.assertEqual(result['card_count'], 2)
        self.assertTrue(result['day_verified'])
        self.assertTrue(result['region_verified'])
        self.assertTrue(result['complete'])
        self.assertEqual(result['unavailable_meetings'], [])
        self.assertEqual(result['field_sizes'], {'202105040301': 12, '202105040302': 12})

    def test_declared_count_comes_from_metadata_not_numeric_race_title(self):
        card = race_card('202147102805').replace(
            '<div class="RaceData">',
            '<div class="RaceList_ItemTitle">テスト競馬465頭の記念</div><div class="RaceData">')
        card = card.replace('12頭', '9頭')
        result = discover_day(daily_fragment([card], day='20211028', marker='d'),
                              day='2021-10-28', region='local')
        self.assertEqual(result['field_sizes'], {'202147102805': 9})
        ambiguous = card.replace('9頭', '9頭 10頭')
        with self.assertRaises(ValueError):
            discover_day(daily_fragment([ambiguous], day='20211028', marker='d'),
                         day='2021-10-28', region='local')

    def test_known_empty_day_retains_unverified_date(self):
        html = ('<meta charset="UTF-8"><!-- block=race_list_sub (cp) -->'
                '<div class="RaceList_Body RaceList_Top" id="RaceTopRace">'
                '<div class="RaceList_Box clearfix"></div></div>')
        result = discover_day(html, day='2024-05-27', region='central')
        self.assertTrue(result['empty'])
        self.assertEqual(result['race_ids'], [])
        self.assertFalse(result['day_verified'])
        self.assertEqual(result['empty_evidence'], 'empty_list_fragment')

    def test_known_local_empty_selector_is_not_missing_data(self):
        html = ('<meta charset="UTF-8"><!-- block=race_list_sub (d) -->'
                '<div class="RaceList_Body RaceList_Top"><div>'
                '<ul class="RaceList_ProvinceSelect"></ul></div>'
                '<div class="RaceList_Box"></div><!-- /.RaceList_Box -->'
                '</div><!-- .RaceList_Body RaceList_Top -->')
        result = discover_day(html, day='2025-02-03', region='local')
        self.assertTrue(result['empty'])
        self.assertTrue(result['complete'])
        self.assertFalse(result['day_verified'])
        self.assertEqual(result['empty_evidence'], 'empty_list_fragment')
        for unknown in (html.replace('<ul class="RaceList_ProvinceSelect"></ul>', '<div></div>'),
                        html.replace('</ul>', '<li>読み込み中</li></ul>'), html[:-60]):
            with self.subTest(html=unknown[-100:]), self.assertRaises(ValueError):
                discover_day(unknown, day='2025-02-03', region='local')

    def test_explicitly_unpublished_meeting_requires_partial_opt_in(self):
        html = unpublished_fragment([race_card('202536042701')])
        with self.assertRaisesRegex(ValueError, '未公開'):
            discover_day(html, day='2025-04-27', region='local')
        result = discover_day(html, day='2025-04-27', region='local', allow_partial=True)
        self.assertEqual(result['race_ids'], ['202536042701'])
        self.assertEqual(result['card_count'], 1)
        self.assertEqual(result['meeting_count'], 2)
        self.assertFalse(result['empty'])
        self.assertFalse(result['complete'])
        self.assertTrue(result['day_verified'])
        self.assertEqual(result['unavailable_meetings'], [{
            'name': '帯広(ば)', 'reason': '開催2~3日前 公開予定です。',
            'meeting_index': 2, 'source_state': 'not_published',
        }])

    def test_only_unpublished_meetings_are_never_an_empty_complete_day(self):
        result = discover_day(unpublished_fragment(), day='2025-04-27', region='local', allow_partial=True)
        self.assertEqual(result['race_ids'], [])
        self.assertFalse(result['empty'])
        self.assertFalse(result['complete'])
        self.assertFalse(result['day_verified'])
        self.assertEqual(result['unavailable_meetings'][0]['name'], '帯広(ば)')

    def test_partial_mode_does_not_allow_unknown_or_invalid_sections(self):
        malformed = [
            unpublished_fragment(reason='読み込み中です。'),
            unpublished_fragment().replace('class="RaceList_DataHeader"', 'class="UnknownHeader"'),
            unpublished_fragment(['<li class="RaceList_DataItem">壊れたカード</li>']),
            unpublished_fragment([race_card('invalid')]),
        ]
        for html in malformed:
            with self.subTest(html=html[-100:]), self.assertRaises(ValueError):
                discover_day(html, day='2025-04-27', region='local', allow_partial=True)
        dated = unpublished_fragment([race_card('202536042701')])
        with self.assertRaises(ValueError):
            discover_day(dated, day='2025-04-28', region='local', allow_partial=True)
        with self.assertRaises(ValueError):
            discover_day(dated, day='2025-04-27', region='central', allow_partial=True)

    def test_explicit_empty_day_and_unknown_empty_are_distinct(self):
        box = ('<div class="RaceList_Body RaceList_Top"><div class="RaceList_Box">'
               '<p>この日の開催はありません</p></div></div>')
        html = '<link rel="canonical" href="https://race.netkeiba.com/top/race_list_sub.html?kaisai_date=20240527">' + box
        result = discover_day(html, day='2024-05-27', region='central')
        self.assertTrue(result['empty'])
        self.assertTrue(result['day_verified'])
        self.assertEqual(result['empty_evidence'], 'explicit_no_meeting')
        unknown = box.replace('<p>この日の開催はありません</p>', '')
        with self.assertRaises(ValueError):
            discover_day(unknown, day='2024-05-27', region='central')

    def test_daily_mismatched_date_or_region_is_error(self):
        html = daily_fragment([race_card('202105040301')])
        with self.assertRaises(ValueError):
            discover_day(html, day='2021-10-17', region='central')
        with self.assertRaises(ValueError):
            discover_day(html, day='2021-10-16', region='local')
        local = daily_fragment([race_card('202135101101', host='nar.netkeiba.com')],
                               day='20211011', marker='d')
        self.assertEqual(discover_day(local, day='2021-10-11', region='local')['race_ids'], ['202135101101'])

    def test_missing_invalid_duplicate_or_foreign_cards_are_errors(self):
        bad_cards = [
            ['<li class="RaceList_DataItem">読み込み中</li>'],
            [race_card('not-an-id')],
            [race_card('202105040301'), race_card('202105040301')],
            [race_card('202135101101', host='nar.netkeiba.com')],
        ]
        for cards in bad_cards:
            with self.subTest(cards=cards), self.assertRaises(ValueError):
                discover_day(daily_fragment(cards), day='2021-10-16', region='central')

    def test_empty_blocked_broken_and_header_only_pages_are_errors(self):
        good = daily_fragment([race_card('202105040301')])
        for html in ('', '<html>403 Forbidden</html>', good[:-6],
                     daily_fragment([]), '<html><body>ログインしてください</body></html>'):
            with self.subTest(html=html[:40]), self.assertRaises(ValueError):
                discover_day(html, day='2021-10-16', region='central')


if __name__ == '__main__':
    unittest.main()
