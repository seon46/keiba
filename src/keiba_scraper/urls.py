from __future__ import annotations

from datetime import date
from html.parser import HTMLParser
import re
from urllib.parse import parse_qs, urlencode, urljoin, urlsplit

from bs4 import BeautifulSoup, Comment


def race_id(value: str) -> str:
    if not re.fullmatch(r"[0-9]{12}", value):
        raise ValueError("国内レースIDは12桁の数字で指定してください。")
    return value


def horse_id(value: str) -> str:
    if not re.fullmatch(r"(?:[0-9]{10}|B[0-9]{9})", value):
        raise ValueError("馬IDは10桁の数字、またはばんえいのBと9桁の数字で指定してください。")
    return value


def race_url(identifier: str, region: str, page: str) -> str:
    race_id(identifier)
    if region not in ("central", "local") or page not in ("entries", "results"):
        raise ValueError("未対応の地域またはページ種別です。")
    host = "race.netkeiba.com" if region == "central" else "nar.netkeiba.com"
    filename = "shutuba.html" if page == "entries" else "result.html"
    return f"https://{host}/race/{filename}?{urlencode({'race_id': identifier})}"


def horse_url(identifier: str) -> str:
    return f"https://db.netkeiba.com/horse/result/{horse_id(identifier)}/"


def list_url(day: str, region: str) -> str:
    compact = date.fromisoformat(day).strftime("%Y%m%d")
    if region not in ("central", "local"):
        raise ValueError("未対応の地域です。")
    host = "race.netkeiba.com" if region == "central" else "nar.netkeiba.com"
    return f"https://{host}/top/race_list_sub.html?{urlencode({'kaisai_date': compact})}"


def discover_races(html: str) -> list[str]:
    soup = BeautifulSoup(html, "html.parser")
    identifiers = set()
    for anchor in soup.select("a[href]"):
        query = parse_qs(urlsplit(anchor["href"]).query)
        for value in query.get("race_id", []):
            if re.fullmatch(r"[0-9]{12}", value):
                identifiers.add(value)
    if not identifiers:
        raise ValueError("レースIDを取得できませんでした。開催なし・動的ページ・サイト変更を確認してください。")
    return sorted(identifiers)


class _ListStructure(HTMLParser):
    """Check list-container endings before BeautifulSoup repairs truncation."""

    structural_tags = {"div", "dl", "dd", "ul", "li"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.broken = False

    def handle_starttag(self, tag, attrs):
        if tag in self.structural_tags:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if tag in self.structural_tags:
            if self.stack and self.stack[-1] == tag:
                self.stack.pop()
            else:
                self.broken = True


def discover_day(html: str, *, day: str, region: str) -> dict:
    """Discover every race card in one official daily-list response.

    The empty central list fragment does not contain its requested date. Its
    known complete structure is accepted, with ``day_verified=False`` so the
    caller can retain that limitation alongside the request URL and raw body.
    Neither dates nor regions are decoded from the twelve-digit race IDs.
    """
    source_url = list_url(day, region)
    compact = date.fromisoformat(day).strftime("%Y%m%d")
    host = urlsplit(source_url).hostname
    if not html.strip():
        raise ValueError("日別一覧の応答が空です。開催なしとは判断できません。")
    structure = _ListStructure()
    structure.feed(html)
    structure.close()
    if structure.broken or structure.stack:
        raise ValueError("日別一覧のコンテナーが途切れているか、閉じタグが壊れています。")
    soup = BeautifulSoup(html, "html.parser")
    visible = " ".join(soup.get_text(" ", strip=True).split())
    if re.search(r"Access\s+Denied|403\s+Forbidden|Too\s+Many\s+Requests|"
                 r"Just\s+a\s+moment|Checking\s+your\s+browser|"
                 r"通信制限|アクセス制限|アクセスが集中|ログインしてください", visible, re.I):
        raise ValueError("日別一覧ではなくアクセス制限・認証・エラーページが返っています。")
    bodies = soup.select(".RaceList_Body.RaceList_Top")
    if len(bodies) != 1:
        raise ValueError("日別一覧の既知のコンテナーを一意に確認できません。")
    body = bodies[0]
    boxes = body.select(".RaceList_Box")
    if len(boxes) != 1:
        raise ValueError("日別一覧のレース領域を一意に確認できません。")
    box = boxes[0]

    markers = set()
    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        match = re.search(r"block\s*=\s*race_list_sub\s*\((cp|d)\)", str(comment))
        if match:
            markers.add(match[1])
    expected_marker = "cp" if region == "central" else "d"
    if markers and markers != {expected_marker}:
        raise ValueError("日別一覧の中央・地方の識別情報が指定地域と一致しません。")

    observed_days = set()
    identity_tags = soup.select('link[rel="canonical"], meta[property="og:url"]')
    identity_tags += body.select(".RaceList_DataHeader a[href], .RaceList_ProvinceSelect a[href]")
    for tag in identity_tags:
        url = tag.get("href") or tag.get("content") or ""
        values = parse_qs(urlsplit(url).query).get("kaisai_date", [])
        observed_days.update(values)
    for tag in [body, box, *body.select("input[name='kaisai_date'], [data-kaisai-date]")]:
        value = tag.get("data-kaisai-date") or (tag.get("value") if tag.get("name") == "kaisai_date" else None)
        if value:
            observed_days.add(str(value).replace("-", ""))
    if soup.title:
        match = re.search(r"(\d{4})年(\d{1,2})月(\d{1,2})日", soup.title.get_text())
        if match:
            try:
                observed_days.add(date(*(int(value) for value in match.groups())).strftime("%Y%m%d"))
            except ValueError as exc:
                raise ValueError("日別一覧のタイトルに無効な日付があります。") from exc
    if observed_days and observed_days != {compact}:
        raise ValueError(f"日別一覧の日付が指定日 {day} と一致しません: {sorted(observed_days)}")

    meetings = box.select(".RaceList_DataList")
    cards = box.select(".RaceList_DataItem")
    if not cards:
        if meetings:
            raise ValueError("開催セクションにレースカードがありません。壊れた一覧を確認してください。")
        explicit_empty = bool(re.search(r"(?:開催(?:は|が)ありません|開催はございません|開催がない日)",
                                        body.get_text(" ", strip=True)))
        # BeautifulSoup repairs missing end tags. A strict source-level match
        # prevents a truncated empty-looking response from being skipped.
        complete_empty_fragment = bool(re.search(
            r'<div\b[^>]*class=[\"\'][^\"\']*RaceList_Body[^\"\']*[\"\'][^>]*>\s*'
            r'<div\b[^>]*class=[\"\'][^\"\']*RaceList_Box[^\"\']*[\"\'][^>]*>\s*'
            r'</div>\s*</div>\s*$', html, re.S))
        known_empty = markers == {expected_marker} and complete_empty_fragment and not box.get_text(strip=True)
        if not explicit_empty and not known_empty:
            raise ValueError("レースカードがなく、正常な開催なし応答も確認できません。")
        return {"race_ids": [], "empty": True, "day_verified": bool(observed_days),
                "region_verified": bool(markers), "card_count": 0, "meeting_count": 0,
                "field_sizes": {},
                "empty_evidence": "explicit_no_meeting" if explicit_empty else "empty_list_fragment"}

    if not meetings:
        raise ValueError("レースカードに対応する開催セクションがありません。")
    for meeting in meetings:
        if not meeting.select_one(".RaceList_DataHeader") or not meeting.select(".RaceList_DataItem"):
            raise ValueError("開催ヘッダーまたは開催内のレースカードが欠落しています。")

    identifiers = []
    field_sizes = {}
    absolute_host_verified = False
    for card in cards:
        meeting = card.find_parent(class_="RaceList_DataList")
        if meeting is None or meeting not in meetings:
            raise ValueError("開催セクションの外にレースカードがあります。")
        card_ids = set()
        for anchor in card.select("a[href]"):
            parts = urlsplit(urljoin(source_url, anchor["href"]))
            # Movie/odds/recommended links do not identify a list's primary race.
            if parts.hostname != host or parts.path not in ("/race/shutuba.html", "/race/result.html"):
                continue
            query = parse_qs(parts.query)
            link_days = query.get("kaisai_date", [])
            if link_days and set(link_days) != {compact}:
                raise ValueError("レースカードのリンク日付が指定日と一致しません。")
            values = query.get("race_id", [])
            if len(values) != 1 or not re.fullmatch(r"[0-9]{12}", values[0]):
                raise ValueError("レースカードに有効な国内レースIDがありません。")
            card_ids.add(values[0])
            if urlsplit(anchor["href"]).hostname == host:
                absolute_host_verified = True
        if len(card_ids) != 1:
            raise ValueError("レースカードを1つのレースIDに対応付けできません。")
        identifier = next(iter(card_ids))
        identifiers.append(identifier)
        declared = re.findall(r"(\d+)\s*頭", card.get_text(" ", strip=True))
        if len(declared) > 1:
            raise ValueError("レースカードの宣言頭数を一意に確認できません。")
        field_sizes[identifier] = int(declared[0]) if declared else None
    if len(set(identifiers)) != len(cards):
        raise ValueError("レースカード数と一意なレースID数が一致しません。")
    return {"race_ids": sorted(identifiers), "empty": False,
            "day_verified": bool(observed_days),
            "region_verified": bool(markers) or absolute_host_verified,
            "card_count": len(cards), "meeting_count": len(meetings),
            "field_sizes": field_sizes, "empty_evidence": None}
