"""Conservative acceptance of source notices; synthetic pages contain no data."""

from copy import deepcopy
import unittest

from bs4 import BeautifulSoup

from keiba_scraper.cancellations import (
    CANCELLATION_SELECTOR,
    cancellation_evidence,
    is_confirmed_cancellation,
)


RACE_ID = "202135112211"
URL = f"https://nar.netkeiba.com/race/result.html?race_id={RACE_ID}"
TITLE = "テスト賞 結果・払戻 | 2021年11月22日 盛岡11R 地方競馬レース情報 - netkeiba"
REASON = "濃霧による視界不良により取り止めとなりました。"


def race_metadata():
    return {"race_id": RACE_ID, "region": "local", "page": "results",
            "race_date": "2021-11-22", "race_number": 11, "race_id_verified": True}


def page(reason=REASON, *, title=TITLE, url=URL, rows="", extra_table=""):
    return (
        f'<html><head><title>{title}</title><link rel="canonical" href="{url}"></head><body>'
        '<div class="ResultTableWrap"><div class="Race_Infomation_Box">' + reason + '</div>'
        '<table id="All_Result_Table"><thead><tr><th>着順</th><th>馬番</th><th>馬名</th></tr></thead>'
        '<tbody>' + rows + '</tbody>' + extra_table + '</table></div></body></html>'
    )


def detect(html, race=None):
    soup = BeautifulSoup(html, "html.parser")
    return cancellation_evidence(soup, soup.select_one("#All_Result_Table"),
                                 race_metadata() if race is None else race)


def confirmed_payload():
    return {"race": {**race_metadata(), **detect(page())}, "runners": []}


class CancellationEvidenceTests(unittest.TestCase):
    def test_verified_completed_cancellation_notices(self):
        reasons = [REASON, "走路状況の悪化により取り止めになりました。",
                   "アクシデントのため取りやめになりました。", "天候不良により取止めとなりました。"]
        for reason in reasons:
            with self.subTest(reason=reason):
                evidence = detect(page(reason))
                self.assertEqual(evidence, {
                    "race_status": "cancelled", "cancellation_reason": reason,
                    "cancellation_evidence": {"selector": CANCELLATION_SELECTOR, "text": reason},
                })
                self.assertTrue(is_confirmed_cancellation({"race": {**race_metadata(), **evidence}, "runners": []}))

    def test_ordinary_empty_pending_ambiguous_and_unknown_notices_are_not_cancellations(self):
        for reason in ("", "結果は未確定です。", "レース終了後に公開します。", "公開予定です。",
                       "取り止めになる予定です。", "取り止めとなりましたか？", "取り止めではありません。",
                       "未確定:取り止めとなりました。", "中止", "通信に失敗しました。"):
            with self.subTest(reason=reason):
                self.assertIsNone(detect(page(reason)))

    def test_unrelated_or_nested_notice_is_not_accepted(self):
        normal = page()
        notice = '<div class="Race_Infomation_Box">' + REASON + '</div>'
        outside = normal.replace(notice, '').replace('</body>', notice + '</body>')
        nested = normal.replace(notice, '<div class="OtherPanel">' + notice + '</div>')
        foreign_wrapper = normal.replace('class="ResultTableWrap"', 'class="OtherPanel"')
        duplicated = normal.replace(notice, notice + notice)
        for html in (outside, nested, foreign_wrapper, duplicated):
            with self.subTest(html=html[-100:]):
                self.assertIsNone(detect(html))

    def test_actual_or_malformed_result_rows_contradict_cancellation(self):
        for rows in ('<tr><td>1</td><td>2</td><td><a href="/horse/2019000001/">テスト馬</a></td></tr>',
                     '<tr><td>取消</td></tr>', '<tr></tr>', '<div>読み込み中</div>', '未確定'):
            with self.subTest(rows=rows):
                self.assertIsNone(detect(page(rows=rows)))
        self.assertIsNone(detect(page(extra_table='<tfoot><tr><td>1</td></tr></tfoot>')))
        sibling = page().replace('</table></div>', '</table><table><tr><td>実走行</td></tr></table></div>')
        self.assertIsNone(detect(sibling))

    def test_table_body_and_identity_must_be_unambiguous(self):
        html = page()
        for changed in (html.replace('<tbody></tbody>', ''),
                        html.replace('<tbody></tbody>', '<tbody></tbody><tbody></tbody>'),
                        html.replace('id="All_Result_Table"', 'id="OtherResult"'),
                        html.replace('</body>', '<table id="All_Result_Table"><tbody></tbody></table></body>')):
            with self.subTest(html=changed[-120:]):
                self.assertIsNone(detect(changed))
        soup = BeautifulSoup(html, 'html.parser')
        cloned_table = BeautifulSoup(str(soup.select_one('#All_Result_Table')), 'html.parser').table
        self.assertIsNone(cancellation_evidence(soup, cloned_table, race_metadata()))

    def test_source_race_date_and_region_must_match(self):
        for changed in (page(url=URL.replace(RACE_ID, '202135112212')),
                        page(url=URL.replace('nar.netkeiba.com', 'race.netkeiba.com')),
                        page(url=URL.replace('result.html', 'shutuba.html')),
                        page(url='https://nar.netkeiba.com/race/result.html'),
                        page(title=TITLE.replace('11月22日', '11月23日')),
                        page(title=TITLE.replace('盛岡11R', '盛岡12R')),
                        page().replace(f'<link rel="canonical" href="{URL}">', '')):
            with self.subTest(html=changed[:150]):
                self.assertIsNone(detect(changed))
        conflict = page().replace('</head>', '<meta property="og:url" content="' + URL.replace(RACE_ID, '202135112212') + '"></head>')
        self.assertIsNone(detect(conflict))

    def test_date_in_race_name_is_not_the_event_date(self):
        misleading = '2023年7月7日の記念 ' + TITLE
        self.assertIsNotNone(detect(page(title=misleading)))
        self.assertIsNone(detect(page(title='2021年11月22日の記念 | 開催日未確認')))

    def test_supplied_context_needs_results_explicit_id_and_strict_date(self):
        changes = [dict(page='entries'), dict(race_id_verified=False), dict(race_id_verified=1),
                   dict(race_id='202135112212'), dict(race_id='２０２１３５１１２２１１'),
                   dict(race_date=None), dict(race_date='2021-11-2'), dict(race_date='2021-02-29'),
                   dict(region='unknown')]
        for change in changes:
            with self.subTest(change=change):
                self.assertIsNone(detect(page(), {**race_metadata(), **change}))
        for missing in ('race_id', 'race_date', 'race_id_verified'):
            metadata = race_metadata()
            metadata.pop(missing)
            with self.subTest(missing=missing):
                self.assertIsNone(detect(page(), metadata))


class ConfirmedCancellationPredicateTests(unittest.TestCase):
    def test_flag_alone_or_incomplete_evidence_does_not_permit_empty_race(self):
        payload = confirmed_payload()
        self.assertTrue(is_confirmed_cancellation(payload))
        changes = [dict(race_status='finished'), dict(race_id_verified=False), dict(page='entries'),
                   dict(race_id='2021J0112211'), dict(race_date='2021-11-2'), dict(race_date='2021-02-29'),
                   dict(cancellation_reason=''), dict(cancellation_reason='公開予定です。'),
                   dict(cancellation_evidence=None), dict(cancellation_evidence={}),
                   dict(cancellation_evidence={'selector': 'body', 'text': REASON}),
                   dict(cancellation_evidence={'selector': CANCELLATION_SELECTOR, 'text': '別の文言'})]
        for change in changes:
            changed = deepcopy(payload)
            changed['race'].update(change)
            with self.subTest(change=change):
                self.assertFalse(is_confirmed_cancellation(changed))
        for missing in ('cancellation_reason', 'cancellation_evidence', 'race_date', 'race_id'):
            changed = deepcopy(payload)
            changed['race'].pop(missing)
            with self.subTest(missing=missing):
                self.assertFalse(is_confirmed_cancellation(changed))

    def test_nonempty_or_missing_runner_list_is_not_empty_cancellation(self):
        payload = confirmed_payload()
        for runners in ([{'finish_position': 1}], None, (), 0):
            with self.subTest(runners=runners):
                self.assertFalse(is_confirmed_cancellation({**payload, 'runners': runners}))
        for malformed in (None, {}, {'race': None, 'runners': []}):
            with self.subTest(malformed=malformed):
                self.assertFalse(is_confirmed_cancellation(malformed))


if __name__ == '__main__':
    unittest.main()
