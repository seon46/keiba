from pathlib import Path
import unittest

FIXTURES = Path(__file__).parent / 'fixtures'


class PayoutParserTests(unittest.TestCase):
    def parse(self, name):
        from keiba_scraper.parsers import parse_payouts
        return parse_payouts((FIXTURES / name).read_text(encoding='utf-8'))

    def test_central_payouts(self):
        payouts = self.parse('payouts_central.html')
        self.assertEqual(payouts['win'], {'8': 560})
        self.assertEqual(payouts['place'], {'8': 270, '6': 510, '7': 280})
        self.assertEqual(payouts['quinella'], {'6-8': 8140})
        self.assertEqual(payouts['wide'], {'6-8': 3080, '7-8': 1430, '6-7': 3130})
        self.assertEqual(payouts['exacta'], {'8>6': 14570})
        self.assertEqual(payouts['trio'], {'6-7-8': 24670})
        self.assertEqual(payouts['trifecta'], {'8>6>7': 120110})

    def test_local_payouts_sort_unordered_keys_numerically(self):
        payouts = self.parse('payouts_local.html')
        self.assertEqual(payouts['quinella'], {'10-11': 840})
        self.assertEqual(payouts['trio'], {'1-10-11': 730})
        self.assertEqual(payouts['trifecta'], {'11>10>1': 6820})
        self.assertEqual(payouts['bracket'], {'8-8': 830})

    def test_page_without_payouts(self):
        from keiba_scraper.parsers import parse_payouts
        self.assertEqual(parse_payouts('<html><body>no table</body></html>'), {})


if __name__ == '__main__':
    unittest.main()
