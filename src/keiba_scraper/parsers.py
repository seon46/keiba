"""Parse publicly visible netkeiba race and horse-history HTML.

No network requests are made here. Hidden/member-only fields are not inferred.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date
from urllib.parse import parse_qs, unquote, urlsplit

from bs4 import BeautifulSoup, NavigableString, Tag


class ParseError(ValueError):
    """The response does not contain the requested usable public table."""


def _text(tag: Tag | None) -> str:
    if tag is None:
        return ""
    return " ".join(tag.get_text(" ", strip=True).split())


def _cell_text(cell: Tag | None) -> str:
    """Exclude nested cells found in some malformed NAR entry-table HTML."""
    if cell is None:
        return ""
    parts = []
    for node in cell.descendants:
        if isinstance(node, NavigableString):
            if node.find_parent(["td", "th"]) is cell:
                if not node.find_parent(["script", "style"]):
                    parts.append(str(node))
    return " ".join(" ".join(parts).split())


def _normalized(value: str) -> str:
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", value))


def _number(value: str, *, integer: bool = False) -> int | float | None:
    value = _normalized(value).replace(",", "")
    if not re.fullmatch(r"[+-]?\d+(?:\.\d+)?", value):
        return None
    if integer:
        return int(value) if re.fullmatch(r"[+-]?\d+", value) else None
    return float(value)


def _time(value: str) -> float | None:
    value = _normalized(value)
    match = re.fullmatch(r"(?:(\d+):)?(\d{1,2}(?:\.\d+)?)", value)
    if not match:
        return None
    seconds = float(match[2])
    if match[1] is not None and seconds >= 60:
        return None
    return round(int(match[1] or 0) * 60 + seconds, 3)


def _date(value: str) -> str | None:
    match = re.search(r"(\d{4})(?:年|[/-])(\d{1,2})(?:月|[/-])(\d{1,2})", value)
    if not match:
        return None
    try:
        return date(*(int(part) for part in match.groups())).isoformat()
    except ValueError:
        return None


def _field(cell: Tag) -> str | None:
    label = _normalized(_cell_text(cell))
    exact = {
        "枠": "frame", "枠番": "frame", "馬番": "horse_number",
        "馬名": "horse_name", "性齢": "sex_age", "斤量": "carried_weight_kg",
        "騎手": "jockey_name", "厩舎": "trainer_name", "調教師": "trainer_name",
        "着順": "finish_position", "タイム": "time_seconds", "着差": "margin",
        "人気": "popularity", "オッズ": "odds", "単勝オッズ": "odds",
        "後3F": "last_3f_seconds", "上り": "last_3f_seconds", "上がり": "last_3f_seconds",
        "コーナー通過順": "corner_order", "通過": "corner_order",
        "日付": "race_date", "開催": "course", "レース名": "race_name",
        "距離": "distance", "天気": "weather", "馬場": "track_condition",
        "頭数": "field_size", "R": "race_number",
    }
    if label in exact:
        return exact[label]
    if label.startswith("馬体重"):
        return "weight"
    classes = set(cell.get("class", []))
    if "Popular_Ninki" in classes:
        return "popularity"
    if "Popular" in classes:
        return "odds"
    return None


def _headers(table: Tag) -> list[str | None]:
    """Map the first actual header row, retaining hidden/unknown columns."""
    for row in table.find_all("tr"):
        cells = row.find_all("th", recursive=False)
        if not cells:
            continue
        headers = []
        for cell in cells:
            field = _field(cell)
            try:
                span = max(1, int(cell.get("colspan", 1)))
            except (TypeError, ValueError):
                span = 1
            headers.extend([field] + [None] * (span - 1))
        if "horse_name" in headers or "race_name" in headers:
            return headers
    return []


def _rows(table: Tag, headers: list[str | None]):
    for row in table.find_all("tr"):
        if row.find_parent("table") is not table:
            continue
        cells = [cell for cell in row.find_all("td") if cell.find_parent("tr") is row]
        if not cells:
            continue
        mapped = {field: cells[i] for i, field in enumerate(headers)
                  if field is not None and i < len(cells)}
        yield row, mapped


def _entity(cell: Tag | None, entity: str) -> tuple[str | None, str | None]:
    if cell is None:
        return None, None
    pattern = rf"/{entity}/(?:(?:result/(?:recent/)?)|profile/)?([A-Za-z]?\d+)(?:/|$)"
    for anchor in cell.find_all("a", href=True):
        path = unquote(urlsplit(anchor["href"]).path)
        match = re.search(pattern, path)
        if match:
            return match[1], _text(anchor) or anchor.get("title") or None
    return None, _cell_text(cell) or None


def _finish(value: str) -> tuple[int | None, str]:
    value = _normalized(value)
    statuses = (("取消", "scratched"), ("除外", "excluded"),
                ("中止", "did_not_finish"), ("失格", "disqualified"),
                ("不成立", "abandoned"))
    for token, status in statuses:
        if token in value:
            return None, status
    match = re.fullmatch(r"(\d+)(?:(?:\(?同着\)?)|(?:\([^)]+\)))?", value)
    if match and int(match[1]) > 0:
        return int(match[1]), "finished"
    return None, "unknown"


def _weight(value: str) -> dict:
    value = _normalized(value)
    match = re.fullmatch(r"(\d+)(?:\(([+-]?\d+)\))?", value)
    return {
        "weight_kg": int(match[1]) if match else None,
        "weight_change_kg": int(match[2]) if match and match[2] else None,
    }


def _distance(value: str) -> dict:
    match = re.search(r"(障(?:芝|ダ)?|芝|ダート|ダ|直)\s*(\d{3,4})", value)
    if not match:
        return {}
    if match[1] == "直":
        # ばんえい's 直 is a course-direction label, not a turf/dirt surface.
        return {"distance_m": int(match[2]), "track_direction": "直線"}
    return {"surface": match[1], "distance_m": int(match[2])}


def _runner(row: Tag, cells: dict[str, Tag], *, page: str) -> dict:
    horse_id, horse_name = _entity(cells.get("horse_name"), "horse")
    if not horse_id:
        raise ParseError("A runner is missing a linked horse ID; refusing a name-only record.")
    result = {"horse_id": horse_id, "horse_name": horse_name}
    for field in ("frame", "horse_number", "popularity"):
        if field in cells:
            result[field] = _number(_cell_text(cells[field]), integer=True)
    for field in ("carried_weight_kg", "odds", "last_3f_seconds"):
        if field in cells:
            result[field] = _number(_cell_text(cells[field]))
    for entity in ("jockey", "trainer"):
        cell = cells.get(f"{entity}_name")
        if cell is not None:
            identifier, name = _entity(cell, entity)
            result[f"{entity}_id"] = identifier
            result[f"{entity}_name"] = name
    if "sex_age" in cells:
        match = re.search(r"([牡牝セ騸])\s*(\d+)", _cell_text(cells["sex_age"]))
        result["sex"] = match[1] if match else None
        result["age"] = int(match[2]) if match else None
    if "weight" in cells:
        result.update(_weight(_cell_text(cells["weight"])))
    if page == "results":
        position, status = _finish(_cell_text(cells.get("finish_position")))
        result.update(finish_position=position, finish_status=status)
        if "time_seconds" in cells:
            result["time_seconds"] = _time(_cell_text(cells["time_seconds"]))
        for field in ("margin", "corner_order"):
            if field in cells:
                value = _cell_text(cells[field])
                result[field] = value if value and not re.fullmatch(r"[-−*]+", value) else None
    else:
        # Only explicit status text/classes: the prediction-mark dropdown has 消.
        status_node = row.select_one(".CancelTxt, .CancelText, .Scratched, .Excluded")
        status = _finish(_text(status_node))[1]
        if status == "unknown":
            status = _finish(_cell_text(cells.get("horse_name")))[1]
        if status == "unknown" and "Cancel" in row.get("class", []):
            status = "scratched"
        if status != "unknown":
            result.update(finish_position=None, finish_status=status)
    return result


def _check_race_identity(soup: BeautifulSoup, race_id: str) -> bool:
    verified = False
    # Navigation also links to other races, so only document identity metadata
    # can establish that this is the race requested by the caller.
    for tag in soup.select('link[rel="canonical"], meta[property="og:url"]'):
        url = tag.get("href") or tag.get("content")
        if not url:
            continue
        parsed = urlsplit(url)
        found = parse_qs(parsed.query).get("race_id", [None])[0]
        if not found:
            match = re.search(r"/race/([A-Za-z0-9]{12})(?:/|$)", parsed.path)
            found = match[1] if match else None
        if found:
            if found != race_id:
                raise ParseError(f"Response race ID {found} differs from requested {race_id}.")
            verified = True
    return verified


def _race_metadata(soup: BeautifulSoup, race_id: str, region: str, page: str) -> dict:
    title = _text(soup.title)
    result = {"race_id": race_id, "region": region, "page": page, "title": title}
    name = _text(soup.select_one(".RaceName"))
    if name:
        result["race_name"] = name
    race_date = _date(title)
    if race_date:
        result["race_date"] = race_date
    course = re.search(r"\d{4}年\d{1,2}月\d{1,2}日\s+([^\s|]+?)(\d+)R", title)
    if course:
        result.update(course=course[1], race_number=int(course[2]))
        if "帯広(ば)" in course[1] or "ばんえい" in course[1]:
            result["race_type"] = "banei"
    detail = _text(soup.select_one(".RaceData01"))
    if detail:
        result["race_details"] = detail
        result.update(_distance(detail))
        start = re.search(r"(\d{1,2}:\d{2})\s*発走", detail)
        if start:
            result["start_time"] = start[1]
        for field, label in (("weather", "天候"), ("track_condition", "馬場")):
            match = re.search(rf"{label}\s*[:：]\s*([^\s/]+)", detail)
            if match:
                value = match[1]
                result[field] = None if value in ("-", "−", "--") else value
        direction = re.search(r"[（(]\s*(左|右|直線)", detail)
        if direction:
            result["track_direction"] = direction[1]
        moisture = re.search(r"水分量\s*[:：]\s*(\d+(?:\.\d+)?)", detail)
        if moisture:
            result["track_moisture"] = float(moisture[1])
    additional = _text(soup.select_one(".RaceData02"))
    count = re.search(r"(\d+)\s*頭", additional)
    if count:
        result["declared_field_size"] = int(count[1])
    return result


def parse_race(html: str, *, race_id: str, region: str, page: str) -> dict:
    """Return a race and its linked runners, or raise instead of saving empty data."""
    if page not in ("entries", "results"):
        raise ValueError("page must be 'entries' or 'results'")
    soup = BeautifulSoup(html, "html.parser")
    identity_verified = _check_race_identity(soup, race_id)
    candidates = []
    for table in soup.find_all("table"):
        headers = _headers(table)
        if "horse_name" not in headers or "horse_number" not in headers:
            continue
        if page == "results" and "finish_position" not in headers:
            continue
        if page == "entries" and "finish_position" in headers:
            continue
        score = int(table.get("id") == "All_Result_Table") * 2
        score += int("ShutubaTable" in table.get("class", [])) * 2
        candidates.append((score, table, headers))
    if not candidates:
        raise ParseError("No requested public race table; response may be a login, block, or unfinished page.")
    _, table, headers = max(candidates, key=lambda item: item[0])
    runners = []
    for row, cells in _rows(table, headers):
        if "horse_name" not in cells:
            continue
        runners.append(_runner(row, cells, page=page))
    if not runners:
        raise ParseError("The public race table has no runners.")
    identities = [runner["horse_id"] for runner in runners]
    if len(set(identities)) != len(identities):
        raise ParseError("Duplicate horse IDs in the race table.")
    if page == "results":
        if any(runner["finish_status"] == "unknown" for runner in runners):
            raise ParseError("Result rows include unconfirmed or unrecognized finish positions.")
        if not any(runner["finish_position"] is not None for runner in runners):
            raise ParseError("No finishing positions are available; results are unfinished or abandoned.")
        visible_text = _text(soup)
        if re.search(r"(?:結果(?:は|が)?未確定|レース結果は確定していません|着順(?:は|が)?未確定)", visible_text):
            raise ParseError("The response explicitly marks its results as unconfirmed.")
    race = _race_metadata(soup, race_id, region, page)
    race["race_id_verified"] = identity_verified
    return {"race": race, "runners": runners}


def parse_horse_history(html: str, *, horse_id: str) -> dict:
    """Parse the public /horse/result/{horse_id}/ table, including overseas IDs."""
    soup = BeautifulSoup(html, "html.parser")
    canonical = soup.select_one('link[rel="canonical"]')
    if canonical and canonical.get("href"):
        match = re.search(r"/horse/(?:result/)?([A-Za-z]?\d+)(?:/|$)", urlsplit(canonical["href"]).path)
        if match and match[1] != horse_id:
            raise ParseError("Response horse ID differs from the requested horse ID.")
    table = soup.select_one("table.db_h_race_results")
    if table is None:
        raise ParseError("No public horse-history table; use the horse result page rather than its JS profile.")
    headers = _headers(table)
    if not {"race_date", "race_name", "finish_position"}.issubset(headers):
        raise ParseError("Horse-history columns are missing or unsupported.")
    runs = []
    for _, cells in _rows(table, headers):
        race_cell = cells.get("race_name")
        if race_cell is None:
            continue
        race_id = None
        for anchor in race_cell.find_all("a", href=True):
            match = re.search(r"/race/([A-Za-z0-9]{12})(?:/|$)", urlsplit(anchor["href"]).path)
            if match:
                race_id = match[1]
                break
        if not race_id:
            raise ParseError("A horse-history row is missing its linked race ID.")
        run = {"race_id": race_id, "race_name": _cell_text(race_cell),
               "race_date": _date(_cell_text(cells.get("race_date")))}
        if run["race_date"] is None:
            raise ParseError("A horse-history row has an unrecognized race date.")
        for field in ("frame", "horse_number", "popularity", "field_size", "race_number"):
            if field in cells:
                run[field] = _number(_cell_text(cells[field]), integer=True)
        for field in ("odds", "carried_weight_kg", "last_3f_seconds"):
            if field in cells:
                run[field] = _number(_cell_text(cells[field]))
        if "course" in cells:
            label = _cell_text(cells["course"])
            run["course"] = re.sub(r"^\d+|\d+$", "", label)
            run["meeting_label"] = label
            if "帯広(ば)" in label or "ばんえい" in label:
                run["race_type"] = "banei"
        for field in ("weather", "track_condition", "corner_order", "margin"):
            if field in cells:
                value = _cell_text(cells[field])
                run[field] = value if value and not re.fullmatch(r"[-−*]+", value) else None
        position, status = _finish(_cell_text(cells.get("finish_position")))
        run.update(finish_position=position, finish_status=status)
        if "time_seconds" in cells:
            run["time_seconds"] = _time(_cell_text(cells["time_seconds"]))
        if "distance" in cells:
            run.update(_distance(_cell_text(cells["distance"])))
        if "weight" in cells:
            run.update(_weight(_cell_text(cells["weight"])))
        if "jockey_name" in cells:
            run["jockey_id"], run["jockey_name"] = _entity(cells["jockey_name"], "jockey")
        runs.append(run)
    if not runs:
        raise ParseError("The public horse-history table has no runs.")
    heading = soup.select_one(".horse_title h1, .HorseTitle h1, h1")
    horse_name = _text(heading) or _text(soup.title).split("|")[0].strip()
    horse_name = re.sub(r"の(?:競走成績|競走馬データ)$", "", horse_name).strip()
    horse_name = re.sub(r"\s*\([^)]*\)$", "", horse_name).strip()
    return {"horse_id": horse_id, "horse_name": horse_name, "runs": runs}
