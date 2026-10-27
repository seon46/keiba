import unittest

from bs4 import BeautifulSoup

from keiba_scraper.parsers import ParseError
from keiba_scraper.pedigree import parse_pedigree, pedigree_url


HORSE = '2019102383'
REAL_NAMES = {
    'S': ('2012102013', 'キタサンブラック'),
    'D': ('2013110100', 'チェストケローズ'),
    'SS': ('2001103312', 'ブラックタイド'),
    'SSS': ('000a00033a', 'サンデーサイレンス'),
    'DS': ('000a011ffb', 'Uncle Mo (米)'),
}


def identifier(path):
    return str(2000000000 + len(path) * 1000 + int(path.replace('S', '0').replace('D', '1'), 2))


def document(*, unknown_prefix=None, linkless=None):
    rows = []
    for row in range(32):
        cells = []
        for generation in range(1, 6):
            height = 2 ** (5 - generation)
            path = ''.join('D' if bit == '1' else 'S'
                           for bit in f'{row // height:0{generation}b}')
            if unknown_prefix and path.startswith(unknown_prefix):
                if path == unknown_prefix and row % height == 0:
                    cells.append(f'<td rowspan="{height}" colspan="{6-generation}">不明</td>')
                continue
            if row % height:
                continue
            horse_id, name = REAL_NAMES.get(path, (identifier(path), '祖先' + path))
            if path == linkless:
                content = name
            else:
                content = f'<a href="/horse/{horse_id}/"><span>{name}</span></a>'
            content += f'<br>1980 鹿毛<br>[<a href="/horse/ped/{horse_id}/">血統</a>]'
            cells.append(f'<td rowspan="{height}"><div class="parent"><div>{content}</div><div>9-g</div></div></td>')
        rows.append('<tr>' + ''.join(cells) + '</tr>')
    return (f'<html><head><link rel="canonical" href="https://db.netkeiba.com/horse/{HORSE}/">'
            f'<meta property="og:url" content="https://db.netkeiba.com//horse/{HORSE}/"></head>'
            '<body><div class="horse_title"><h1>ウィルソンテソーロ</h1></div>'
            '<table class="blood_table detail" summary="5代血統表">' + ''.join(rows) + '</table></body></html>')


class PedigreeParserTests(unittest.TestCase):
    def test_all_generations_are_assigned_by_rowspan_not_link_order(self):
        result = parse_pedigree(document(), HORSE)
        self.assertEqual(result['horse_id'], HORSE)
        self.assertEqual(result['horse_name'], 'ウィルソンテソーロ')
        self.assertEqual(result['sire_name'], 'キタサンブラック')
        self.assertEqual(result['dam_name'], 'チェストケローズ')
        self.assertEqual(result['damsire_name'], 'Uncle Mo (米)')
        self.assertEqual(result['generations'], 5)
        self.assertEqual(result['missing_paths'], [])
        self.assertEqual(result['missing_id_paths'], [])
        self.assertEqual(len(result['ancestors']), 62)
        indexed = {item['path']: item for item in result['ancestors']}
        for generation in range(1, 6):
            nodes = [node for node in indexed.values() if node['generation'] == generation]
            self.assertEqual(len(nodes), 2**generation)
            for node in nodes:
                self.assertEqual(node['generation'], len(node['path']))
        self.assertEqual(indexed['SSS']['horse_id'], '000a00033a')
        self.assertEqual(indexed['SSS']['birth_year'], 1980)
        self.assertEqual(indexed['SSS']['source_url'], 'https://db.netkeiba.com/horse/000a00033a/')
        self.assertEqual(indexed['SDDSD']['name'], '祖先SDDSD')

    def test_unknown_subtree_colspan_is_preserved_without_inventing_names(self):
        result = parse_pedigree(document(unknown_prefix='D'), HORSE)
        self.assertEqual(len(result['missing_paths']), 31)
        self.assertTrue(all(path.startswith('D') for path in result['missing_paths']))
        self.assertIsNone(result['dam_id'])
        self.assertIsNone(result['dam_name'])
        self.assertIsNone(result['damsire_name'])
        self.assertEqual(result['sire_name'], 'キタサンブラック')
        self.assertEqual(len(result['ancestors']), 62)

    def test_named_linkless_ancestor_keeps_name_but_never_guesses_id(self):
        result = parse_pedigree(document(linkless='SSSS'), HORSE)
        node = next(node for node in result['ancestors'] if node['path'] == 'SSSS')
        self.assertEqual(node['name'], '祖先SSSS')
        self.assertIsNone(node['horse_id'])
        self.assertIsNone(node['source_url'])
        self.assertIn('SSSS', result['missing_id_paths'])
        self.assertNotIn('SSSS', result['missing_paths'])

    def test_plain_parent_without_inner_div_can_be_unknown(self):
        soup = BeautifulSoup(document(), 'html.parser')
        cell = soup.select_one('table.blood_table tr td')
        cell.clear()
        cell.append(BeautifulSoup('<div class="parent">不明</div>', 'html.parser').div)
        result = parse_pedigree(str(soup), HORSE)
        self.assertIsNone(result['sire_name'])
        self.assertIsNone(result['sire_id'])
        self.assertIn('S', result['missing_paths'])

    def test_plain_parent_without_inner_div_keeps_named_ancestor(self):
        soup = BeautifulSoup(document(), 'html.parser')
        cell = soup.select_one('table.blood_table tr td')
        cell.clear()
        cell.append(BeautifulSoup('<div class="parent">キタサンブラック<br>2012 鹿毛</div>',
                                 'html.parser').div)
        result = parse_pedigree(str(soup), HORSE)
        self.assertEqual(result['sire_name'], 'キタサンブラック')
        self.assertIsNone(result['sire_id'])
        node = next(node for node in result['ancestors'] if node['path'] == 'S')
        self.assertEqual(node['birth_year'], 2012)
        self.assertIn('S', result['missing_id_paths'])
        self.assertNotIn('S', result['missing_paths'])

    def test_repeated_ancestor_retains_each_lineage_position(self):
        html = document().replace('/horse/' + identifier('DSDDD') + '/', '/horse/000a00033a/')
        result = parse_pedigree(html, HORSE)
        matches = [node['path'] for node in result['ancestors'] if node['horse_id'] == '000a00033a']
        self.assertEqual(matches, ['SSS', 'DSDDD'])

    def test_wrong_or_conflicting_document_identity_is_rejected(self):
        with self.assertRaises(ParseError):
            parse_pedigree(document(), '2019102384')
        html = document().replace('https://db.netkeiba.com/horse/' + HORSE + '/',
                                  'https://db.netkeiba.com/horse/2019102384/', 1)
        with self.assertRaises(ParseError):
            parse_pedigree(html, HORSE)

    def test_active_pedigree_navigation_can_identify_a_page_without_canonical(self):
        soup = BeautifulSoup(document(), 'html.parser')
        for tag in soup.select('link[rel="canonical"], meta[property="og:url"]'):
            tag.decompose()
        without_identity = str(soup)
        with self.assertRaises(ParseError):
            parse_pedigree(without_identity, HORSE)
        navigation = f'<ul class="db_detail_menu"><li><a class="active" href="/horse/ped/{HORSE}/">血統</a></li></ul>'
        result = parse_pedigree(without_identity.replace('<body>', '<body>' + navigation), HORSE)
        self.assertTrue(result['horse_id_verified'])

    def test_recommended_horse_links_do_not_override_document_identity(self):
        recommendation = '<aside><a href="/horse/2019102384/">別の馬</a></aside>'
        result = parse_pedigree(document().replace('</body>', recommendation + '</body>'), HORSE)
        self.assertEqual(result['horse_id'], HORSE)
        self.assertEqual(len(result['ancestors']), 62)

    def test_incomplete_table_or_malformed_span_never_shifts_ancestry(self):
        examples = [document().replace('</table>', ''),
                    document().replace('rowspan="16"', 'rowspan="17"', 1),
                    document().replace('rowspan="16"', 'rowspan="bad"', 1),
                    document().replace('rowspan="16"', 'rowspan="0"', 1)]
        for html in examples:
            with self.subTest(html=html[:100]), self.assertRaises(ParseError):
                parse_pedigree(html, HORSE)
        soup = BeautifulSoup(document(), 'html.parser')
        soup.select('table tr')[-1].decompose()
        with self.assertRaises(ParseError):
            parse_pedigree(str(soup), HORSE)

    def test_known_cell_cannot_stand_for_multiple_generations(self):
        html = document(unknown_prefix='D').replace('colspan="5">不明', 'colspan="5">既知の祖先')
        with self.assertRaises(ParseError):
            parse_pedigree(html, HORSE)

    def test_error_pages_and_duplicate_pedigree_tables_are_rejected(self):
        for html in ('<title>メンテナンス中</title>', '<h1>ログインしてください</h1>',
                     document() + '<table class="blood_table"></table>'):
            with self.subTest(html=html[:100]), self.assertRaises(ParseError):
                parse_pedigree(html, HORSE)

    def test_pedigree_url_uses_race_horse_identifiers_only(self):
        self.assertEqual(pedigree_url(HORSE), f'https://db.netkeiba.com/horse/ped/{HORSE}/')
        self.assertEqual(pedigree_url('B123456789'), 'https://db.netkeiba.com/horse/ped/B123456789/')
        for value in ('000a00033a', '../../etc', '1234', '2019102383?x=1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                pedigree_url(value)


if __name__ == '__main__':
    unittest.main()
