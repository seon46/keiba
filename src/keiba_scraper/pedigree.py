"""Parse the public five-generation pedigree without following ancestor links.

Paths are relative to the requested horse: S is sire, D is dam, and DS is
the dam's sire. Repeated ancestors retain each path; they are not deduplicated.
"""
from __future__ import annotations

from html.parser import HTMLParser
import re
from urllib.parse import urljoin, urlsplit

from bs4 import BeautifulSoup

from .parsers import ParseError
from .urls import horse_id as validate_horse_id


GENERATIONS = 5
_ROW_COUNT = 2 ** GENERATIONS
_ANCESTOR_ID = r"[A-Za-z0-9]{10}"
_UNKNOWN = re.compile(r"(?:[?？*＊\-−―ー]+|不明|未登録|登録なし|unknown|none|null)?", re.I)


def pedigree_url(horse_id: str) -> str:
    return f"https://db.netkeiba.com/horse/ped/{validate_horse_id(horse_id)}/"


def _text(tag) -> str:
    return " ".join(tag.get_text(" ", strip=True).split()) if tag is not None else ""


def _horse_link(value: str, *, sections=("",)) -> tuple[str, str] | None:
    url = urlsplit(urljoin("https://db.netkeiba.com/", value))
    if (url.scheme not in ("https", "http") or url.hostname != "db.netkeiba.com"
            or url.username is not None or url.password is not None):
        return None
    # Actual og:url contains //horse/ in some pages.
    path = re.sub(r"/+", "/", url.path)
    section = "|".join(re.escape(section) for section in sections if section)
    prefix = rf"(?:(?:{section})/)?" if "" in sections else rf"(?:{section})/"
    match = re.fullmatch(rf"/horse/{prefix}({_ANCESTOR_ID})/?", path)
    if match is None:
        return None
    identifier = match[1]
    return identifier, f"https://db.netkeiba.com/horse/{identifier}/"


class _TableStructure(HTMLParser):
    """Reject truncated or overlapping table tags before Soup repairs them."""

    structural = {"table", "tbody", "thead", "tfoot", "tr", "td", "th"}

    def __init__(self):
        super().__init__()
        self.stack = []
        self.seen = 0
        self.closed = 0
        self.broken = False

    def handle_starttag(self, tag, attrs):
        if not self.stack:
            classes = (dict(attrs).get("class") or "").split()
            if tag != "table" or "blood_table" not in classes:
                return
            self.seen += 1
        elif tag == "table":
            self.broken = True
        if tag in self.structural:
            self.stack.append(tag)

    def handle_endtag(self, tag):
        if not self.stack or tag not in self.structural:
            return
        if self.stack[-1] != tag:
            self.broken = True
            return
        self.stack.pop()
        if tag == "table" and not self.stack:
            self.closed += 1


def _identity(soup, requested):
    identities = []
    for tag in soup.select('link[rel="canonical"], meta[property="og:url"]'):
        value = tag.get("href") or tag.get("content") or ""
        reference = _horse_link(value, sections=("", "ped"))
        if reference is None:
            raise ParseError("血統ページの識別URLを確認できません。")
        identities.append(reference[0])
    for tag in soup.select('.db_detail_menu a.active[href]'):
        reference = _horse_link(tag["href"], sections=("ped",))
        if reference is not None:
            identities.append(reference[0])
    if not identities or set(identities) != {requested}:
        raise ParseError("血統ページの馬IDが指定した馬と一致しないか、照合できません。")


def _cell_data(cell) -> dict:
    raw = _text(cell)
    links = []
    for anchor in cell.select("a[href]"):
        reference = _horse_link(anchor["href"])
        if reference is not None:
            links.append((reference, _text(anchor)))
    if len({reference[0] for reference, _ in links}) > 1:
        raise ParseError("1つの血統セルに複数の祖先IDがあります。")
    if links:
        (identifier, url), name = links[0]
        if not name or _UNKNOWN.fullmatch(name):
            raise ParseError("リンク付き祖先の馬名が欠落しています。")
    else:
        identifier, url = None, None
        # Linkless, but explicitly named ancestors remain useful observations.
        parent = cell.select_one(".parent")
        label = (parent.find("div", recursive=False) or parent) if parent is not None else cell
        text = " ".join(str(value).strip() for value in label.strings
                        if value.find_parent(["script", "style"]) is None)
        # The first line is the name in this page's observed cell layout.
        first = label.get_text("\n", strip=True).split("\n", 1)[0] if text else ""
        name = None if _UNKNOWN.fullmatch(first.strip()) else first.strip()
    year = None
    if name and raw.startswith(name):
        remaining = raw[len(name):].strip()
        match = re.match(r"((?:1[6-9]|20)[0-9]{2})(?:\s|$)", remaining)
        if match:
            year = int(match[1])
    return dict(horse_id=identifier, name=name, source_url=url,
                birth_year=year, raw_text=raw or None)


def _grid(table):
    rows = [row for row in table.find_all("tr") if row.find_parent("table") is table]
    if len(rows) != _ROW_COUNT:
        raise ParseError("5代血統表の行数が32行ではありません。未知の形式を推測しません。")
    grid = [[None] * GENERATIONS for _ in rows]
    placements = {}
    for y, row in enumerate(rows):
        x = 0
        for cell in row.find_all(["td", "th"], recursive=False):
            while x < GENERATIONS and grid[y][x] is not None:
                x += 1
            try:
                height = int(cell.get("rowspan", 1))
                width = int(cell.get("colspan", 1))
            except (TypeError, ValueError) as exc:
                raise ParseError("血統表のrowspan/colspanが整数ではありません。") from exc
            if height < 1 or width < 1 or y + height > _ROW_COUNT or x + width > GENERATIONS:
                raise ParseError("血統表のセル範囲が5代の領域外です。")
            information = _cell_data(cell)
            if width > 1 and information["name"] is not None:
                raise ParseError("複数世代にまたがる既知の祖先セルを解釈できません。")
            key = id(cell)
            placements[key] = (x, y, width, height, information)
            for gy in range(y, y + height):
                for gx in range(x, x + width):
                    if grid[gy][gx] is not None:
                        raise ParseError("血統表のセルが重なっています。")
                    grid[gy][gx] = key
            x += width
    if any(key is None for row in grid for key in row):
        raise ParseError("血統表のセルが欠けており、祖先の位置を特定できません。")
    return grid, placements


def parse_pedigree(html: str, horse_id: str) -> dict:
    """Read all 62 lineage positions, keeping explicit unknown ancestors null.

    Public five-generation tables have 32 leaf rows and five generation
    columns. A known ancestor must occupy precisely its lineage's rowspan.
    Unknown rectangular subtrees can use colspan; they remain unknown.
    """
    validate_horse_id(horse_id)
    structure = _TableStructure()
    structure.feed(html)
    structure.close()
    if structure.broken or structure.stack or structure.seen != 1 or structure.closed != 1:
        raise ParseError("完全な血統表を一意に確認できません。")
    soup = BeautifulSoup(html, "html.parser")
    _identity(soup, horse_id)
    tables = soup.select("table.blood_table")
    if len(tables) != 1:
        raise ParseError("公開血統表がないか、複数あります。")
    table = tables[0]
    grid, placements = _grid(table)
    ancestors = []
    for generation in range(1, GENERATIONS + 1):
        height = 2 ** (GENERATIONS - generation)
        for branch in range(2 ** generation):
            path = "".join("D" if bit == "1" else "S"
                           for bit in f"{branch:0{generation}b}")
            y, x = branch * height, generation - 1
            keys = {grid[row][x] for row in range(y, y + height)}
            if len(keys) != 1:
                raise ParseError("世代に対応する祖先セルの配置が不正です。")
            px, py, width, span, data = placements[keys.pop()]
            if data["name"] is not None and (px, py, width, span) != (x, y, 1, height):
                raise ParseError("既知の祖先のrowspanが世代と一致しません。")
            ancestors.append(dict(path=path, generation=generation, **data))
    if not any(ancestor["name"] for ancestor in ancestors):
        raise ParseError("祖先名が1件もないため、利用可能な血統表として保存しません。")
    indexed = {ancestor["path"]: ancestor for ancestor in ancestors}
    result = dict(horse_id=horse_id, horse_id_verified=True,
                  horse_name=_text(soup.select_one(".horse_title h1, .HorseTitle h1")) or None,
                  generations=GENERATIONS, ancestors=ancestors,
                  missing_paths=[item["path"] for item in ancestors if item["name"] is None],
                  missing_id_paths=[item["path"] for item in ancestors if item["horse_id"] is None],
                  parse_warnings=[])
    for label, path in (("sire", "S"), ("dam", "D"), ("damsire", "DS")):
        result[label + "_id"] = indexed[path]["horse_id"]
        result[label + "_name"] = indexed[path]["name"]
    if result["missing_paths"]:
        result["parse_warnings"].append(f"祖先名不明: {len(result['missing_paths'])}/62")
    if result["missing_id_paths"]:
        result["parse_warnings"].append(f"祖先ID不明: {len(result['missing_id_paths'])}/62")
    return result
