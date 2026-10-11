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
        self.assertEqual(result['field_sizes'], {'202105040301': 12, '202105040302': 12})

    def test_known_empty_day_retains_unverified_date(self):
        html = ('<meta charset="UTF-8"><!-- block=race_list_sub (cp) -->'
                '<div class="RaceList_Body RaceList_Top" id="RaceTopRace">'
                '<div class="RaceList_Box clearfix"></div></div>')
        result = discover_day(html, day='2024-05-27', region='central')
        self.assertTrue(result['empty'])
        self.assertEqual(result['race_ids'], [])
        self.assertFalse(result['day_verified'])
        self.assertEqual(result['empty_evidence'], 'empty_list_fragment')

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
