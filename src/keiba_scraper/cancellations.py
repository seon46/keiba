"""Conservative evidence for races explicitly called off by the source page.

An empty result table alone never establishes a cancellation. This module does
not infer cancellation from weather, calendar dates, runner counts, or HTTP
errors, and does not make network requests.
"""

from __future__ import annotations

from datetime import date
import re
from urllib.parse import parse_qs, urlsplit

from bs4 import BeautifulSoup, Tag


CANCELLATION_SELECTOR = ".ResultTableWrap > .Race_Infomation_Box"
_CANCELLATION_NOTICE = re.compile(
    r".*?(?:取り止め|取りやめ|取止め|取止)(?:となりました|になりました)[。．.]?"
)
_TITLE_DATE = re.compile(
    r"\|\s*([0-9]{4})年([0-9]{1,2})月([0-9]{1,2})日\s+"
    r"([^\s|]+?)([0-9]+)R(?=\s|$)"
)


def _valid_race_context(race: object) -> bool:
    if not isinstance(race, dict):
        return False
    if race.get("page") != "results" or race.get("race_id_verified") is not True:
        return False
    if race.get("region") not in ("central", "local"):
        return False
    identifier = race.get("race_id")
    if not isinstance(identifier, str) or not re.fullmatch(r"[0-9]{12}", identifier):
        return False
    day = race.get("race_date")
    if not isinstance(day, str) or not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day):
        return False
    try:
        return date.fromisoformat(day).isoformat() == day
    except ValueError:
        return False


def _confirmed_notice(text: object) -> bool:
    if not isinstance(text, str) or not text or text != " ".join(text.split()):
        return False
    if re.search(r"未確定|未確認|暫定", text):
        return False
    return _CANCELLATION_NOTICE.fullmatch(text) is not None


def _source_identity_matches(soup: BeautifulSoup, race: dict) -> bool:
    """Check source identity again rather than relying only on a supplied flag."""
    expected_host = "race.netkeiba.com" if race["region"] == "central" else "nar.netkeiba.com"
    metadata = soup.select('link[rel="canonical"], meta[property="og:url"]')
    if not metadata:
        return False
    for tag in metadata:
        value = tag.get("href") or tag.get("content")
        if not isinstance(value, str):
            return False
        try:
            url = urlsplit(value)
            identifiers = parse_qs(url.query, keep_blank_values=True).get("race_id", [])
        except ValueError:
            return False
        if (url.scheme != "https" or url.hostname != expected_host
                or url.path != "/race/result.html" or identifiers != [race["race_id"]]):
            return False
    if soup.title is None:
        return False
    # Race names can themselves contain a date. Only the final dated race
    # segment after | identifies the event, matching parse_race's metadata.
    matches = list(_TITLE_DATE.finditer(soup.title.get_text(" ", strip=True)))
    if not matches:
        return False
    match = matches[-1]
    try:
        source_day = date(*(int(value) for value in match.groups()[:3])).isoformat()
    except ValueError:
        return False
    if source_day != race["race_date"]:
        return False
    if race.get("race_number") is not None and race["race_number"] != int(match[5]):
        return False
    return True


def cancellation_evidence(soup: BeautifulSoup, table: Tag, race: dict) -> dict | None:
    """Return cancellation metadata only for a dated, identified empty result.

    The expected result table and the completed cancellation notice must be
    direct children of the same ResultTableWrap, as in verified source pages.
    Unknown or contradictory evidence returns None for the parser to reject.
    """
    if not _valid_race_context(race) or not isinstance(soup, BeautifulSoup):
        return None
    if not _source_identity_matches(soup, race):
        return None
    tables = soup.select("#All_Result_Table")
    if len(tables) != 1 or tables[0] is not table or not isinstance(table, Tag) or table.name != "table":
        return None
    wrapper = table.parent
    if not isinstance(wrapper, Tag) or "ResultTableWrap" not in wrapper.get("class", []):
        return None
    bodies = table.find_all("tbody", recursive=False)
    if len(bodies) != 1 or bodies[0].find(True) is not None or bodies[0].get_text(strip=True):
        return None
    # Also reject malformed real rows placed outside tbody or in a footer.
    if wrapper.find("td") is not None:
        return None
    notices = wrapper.find_all(class_="Race_Infomation_Box", recursive=False)
    if len(notices) != 1:
        return None
    reason = " ".join(notices[0].get_text(" ", strip=True).split())
    if not _confirmed_notice(reason):
        return None
    return {
        "race_status": "cancelled",
        "cancellation_reason": reason,
        "cancellation_evidence": {"selector": CANCELLATION_SELECTOR, "text": reason},
    }


def is_confirmed_cancellation(parsed: object) -> bool:
    """Validate the complete cancellation marker used by storage and jobs.

    This checks parser output, not remote HTML authenticity; source identity and
    DOM evidence are established by cancellation_evidence before this marker is
    produced. A status flag alone never permits an empty saved race.
    """
    if not isinstance(parsed, dict):
        return False
    runners = parsed.get("runners")
    race = parsed.get("race")
    if not isinstance(runners, list) or runners or not _valid_race_context(race):
        return False
    if race.get("race_status") != "cancelled":
        return False
    reason = race.get("cancellation_reason")
    evidence = race.get("cancellation_evidence")
    return (isinstance(evidence, dict) and evidence.get("selector") == CANCELLATION_SELECTOR
            and evidence.get("text") == reason and _confirmed_notice(reason))
