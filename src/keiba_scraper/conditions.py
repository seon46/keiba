"""Conservative race conditions from the current race's saved header only.

This module has no third-party or network dependencies.  Regional classes
remain labels, not a universal ordering of race strength.
"""
from __future__ import annotations

from html.parser import HTMLParser
import math
import re
import unicodedata


RACE_CLASSES_BY_REGION = {
    'central': frozenset({'newcomer', 'maiden', '1win', '2win', '3win', 'open'}),
    'local': frozenset({'newcomer', 'maiden', 'open'} |
                       {f'{letter}{number}' for letter in 'ABC' for number in range(1, 5)}),
}
RACE_GRADES_BY_REGION = {
    'central': frozenset({'G1', 'G2', 'G3', 'JG1', 'JG2', 'JG3', 'L'}),
    'local': frozenset({'G1', 'G2', 'G3', 'Jpn1', 'Jpn2', 'Jpn3'}),
}
WEIGHT_RULES = frozenset({'定量', '別定', 'ハンデ', '馬齢'})

# Only these known JRA header icons are interpreted.  Other numbered
# icons include unrelated attributes, such as international eligibility.
_CENTRAL_GRADE_ICONS = {'Icon_GradeType1': 'G1', 'Icon_GradeType2': 'G2',
                        'Icon_GradeType3': 'G3'}
_GRADE_PATTERN = r'(?:J[.・]?G|JPN|G)[.・]?(?:III|II|I|[123])|L'
_VOID_TAGS = frozenset({'area', 'base', 'br', 'col', 'embed', 'hr', 'img', 'input',
                        'link', 'meta', 'param', 'source', 'track', 'wbr'})


def _normalized(value: str) -> str:
    return ' '.join(unicodedata.normalize('NFKC', value or '').split())


def _single(values):
    return next(iter(values)) if len(values) == 1 and None not in values else None


def _title_name(title: str) -> str:
    head = re.split(r'[|｜]', _normalized(title), maxsplit=1)[0]
    return re.sub(r'\s*(?:出馬表|結果・払戻|結果|レース結果|オッズ(?:・購入)?)\s*$', '', head).strip()


def _canonical_grade(token: str) -> str | None:
    normalized = re.sub(r'[.・\s]', '', _normalized(token).upper())
    if normalized in ('L', 'リステッド'):
        return 'L'
    match = re.fullmatch(r'(JPN|JG|G)(III|II|I|[123])', normalized)
    if not match:
        return None
    number = {'I': '1', 'II': '2', 'III': '3'}.get(match[2], match[2])
    return ('Jpn' if match[1] == 'JPN' else match[1]) + number


def _name_fragments(value: str) -> list[str]:
    """Only full condition labels and parentheses, never arbitrary name words."""
    value = _normalized(value)
    fragments = re.findall(r'\(([^()]*)\)', value)
    plain = re.sub(r'\([^()]*\)', '', value).strip()
    previous = None
    while previous != plain:
        previous = plain
        plain = re.sub(r'^(?:サラ系|障害|\d+(?:[・、]\d+)*歳(?:以上|以下|上)?)\s*', '', plain)
    fragments.append(plain)
    return fragments


def _classes(value: str, region: str, *, condition_text: bool) -> set[str | None]:
    values: set[str | None] = set()
    fragments = [_normalized(value)] if condition_text else _name_fragments(value)
    labels = {'新馬': 'newcomer', '未勝利': 'maiden', 'オープン': 'open', 'OP': 'open',
              '1勝': '1win', '1勝クラス': '1win', '500万下': '1win', '500万円以下': '1win',
              '2勝': '2win', '2勝クラス': '2win', '1000万下': '2win', '1000万円以下': '2win',
              '3勝': '3win', '3勝クラス': '3win', '1600万下': '3win', '1600万円以下': '3win'}
    for text in fragments:
        if condition_text:
            for pattern, label in ((r'新馬', 'newcomer'), (r'未勝利', 'maiden'),
                                   (r'オープン|(?<![A-Z0-9])OP(?![A-Z0-9])', 'open')):
                if re.search(pattern, text, re.IGNORECASE):
                    values.add(label)
            if region == 'central':
                for match in re.finditer(r'(?<!\d)([123])勝(?:クラス)?(?!\d)', text):
                    values.add(match[1] + 'win')
                for amount, label in (('500', '1win'), ('1000', '2win'), ('1600', '3win')):
                    if re.search(rf'(?<!\d){amount}万(?:下|円以下)(?!\d)', text):
                        values.add(label)
        else:
            label = labels.get(text.upper() if text.upper() == 'OP' else text)
            if label in RACE_CLASSES_BY_REGION[region]:
                values.add(label)
        if region == 'local' and (condition_text or re.fullmatch(
                r'[ABC][1-9]\d*(?:[ABC][1-9]\d*|[一二三四五六七八九十0-9・、,\s()-])*', text)):
            for match in re.finditer(r'(?<![A-Z])([ABC][1-9]\d*)(?!\d)', text):
                label = match[1]
                ranged = re.match(r'\s*(?:以上|以下|未満|[~〜～])', text[match.end():])
                values.add(label if label in RACE_CLASSES_BY_REGION['local'] and not ranged else None)
    return values


def _grades(name: str, title: str, conditions: str, region: str, grade_tokens) -> set[str | None]:
    tokens = []
    for value in (name, _title_name(title)):
        tokens.extend(_name_fragments(value))
        # Some grade badges have visible text immediately after the name.
        tokens.extend(match[1] for match in re.finditer(
            rf'(?:^|\s)({_GRADE_PATTERN})(?=\s|$)', _normalized(value).upper()))
    tokens.extend(match[1] for match in re.finditer(
        rf'(?:^|[\s(])({_GRADE_PATTERN})(?=[\s)]|$)', _normalized(conditions).upper()))
    if 'リステッド' in _normalized(conditions):
        tokens.append('L')
    if grade_tokens is None:
        grade_tokens = ()
    if isinstance(grade_tokens, str):
        grade_tokens = (grade_tokens,)
    tokens.extend(token for token in grade_tokens if isinstance(token, str))
    result: set[str | None] = set()
    for token in tokens:
        grade = _canonical_grade(token)
        if grade is None and region == 'central':
            grade = _CENTRAL_GRADE_ICONS.get(token)
        if grade is not None:
            result.add(grade if grade in RACE_GRADES_BY_REGION[region] else None)
    return result


def _ages(conditions: str) -> tuple[int | None, int | None]:
    bounds = []
    for match in re.finditer(r'(?<!\d)([1-9]\d?)(?:[・、]([1-9]\d?))?歳\s*(以上|以下|上)?', conditions):
        first, second = int(match[1]), int(match[2]) if match[2] else None
        if first > 30 or (second is not None and (second > 30 or second < first)):
            return None, None
        if second is not None:
            bounds.append((first, second))
        elif match[3] in ('以上', '上'):
            bounds.append((first, None))
        elif match[3] == '以下':
            bounds.append((None, first))
        else:
            bounds.append((first, first))
    lows = [low for low, _ in bounds if low is not None]
    highs = [high for _, high in bounds if high is not None]
    low, high = max(lows) if lows else None, min(highs) if highs else None
    return (None, None) if low is not None and high is not None and low > high else (low, high)


def _first_prize(conditions: str) -> float | None:
    amounts = set()
    for match in re.finditer(r'(?:本)?賞金\s*:\s*(\d+(?:\.\d+)?(?:\s*[,、]\s*\d+(?:\.\d+)?)*)\s*万円', conditions):
        sequence = match[1]
        # A bare comma-grouped amount could instead be several prize values.
        # Do not silently turn an ambiguous 3,000 into three million yen.
        if re.fullmatch(r'\d{1,3}(?:,\d{3})+', sequence):
            amounts.add(None)
            continue
        amount = float(re.split(r'[,、]', sequence, maxsplit=1)[0])
        amounts.add(amount if math.isfinite(amount) and amount > 0 else None)
    return _single(amounts)


def parse_conditions(name: str = '', title: str = '', conditions: str = '',
                     region: str = 'central', grade_tokens=()) -> dict:
    """Parse explicit current-race conditions, retaining unknowns as None.

    Grade tokens must come from the current RaceName, not site navigation
    or recommended races.  A contradiction leaves the affected field
    unknown.  Grade alone never fabricates an open-class label.
    """
    raw = ' '.join((conditions or '').split())
    result = dict(race_class=None, race_grade=None, weight_rule=None,
                  age_min=None, age_max=None, first_prize_man=None, conditions_raw=raw)
    if region not in RACE_CLASSES_BY_REGION:
        return result
    normalized = _normalized(conditions)
    classes: set[str | None] = set()
    classes.update(_classes(normalized, region, condition_text=True))
    classes.update(_classes(name, region, condition_text=False))
    classes.update(_classes(_title_name(title), region, condition_text=False))
    result['race_class'] = _single(classes)
    result['race_grade'] = _single(_grades(name, title, normalized, region, grade_tokens))
    if result['race_grade'] is not None and result['race_class'] not in (None, 'open'):
        result['race_grade'] = result['race_class'] = None
    rules = {rule for rule in WEIGHT_RULES if rule in normalized}
    # ハンデキャップ is a spelling of the same explicit condition.
    result['weight_rule'] = _single(rules)
    result['age_min'], result['age_max'] = _ages(normalized)
    result['first_prize_man'] = _first_prize(normalized)
    return result


class _HeaderDone(Exception):
    pass


class _HeaderExtractor(HTMLParser):
    """Collect only three header fields, then stop at RaceData02's end."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.active: dict[str, int] = {}
        self.parts = {'title': [], 'name': [], 'conditions': []}
        self.seen = set()
        self.closed = set()
        self.grade_tokens = []
        self.grade_depths = set()
        self.conditions_closed = False

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = set((attributes.get('class') or '').split())
        if tag not in _VOID_TAGS:
            self.stack.append(tag)
        depth = len(self.stack)
        for kind, matches in (('title', tag == 'title'), ('name', 'RaceName' in classes),
                              ('conditions', 'RaceData02' in classes)):
            if matches and kind not in self.seen:
                self.active[kind] = depth
                self.seen.add(kind)
        if 'name' in self.active and 'Icon_GradeType' in classes:
            self.grade_tokens.extend(classes)
            self.grade_depths.add(depth)
        if tag == 'br':
            for kind in self.active:
                self.parts[kind].append(' ')

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in _VOID_TAGS:
            self.handle_endtag(tag)

    def handle_data(self, data):
        if any(tag in ('script', 'style') for tag in self.stack):
            return
        for kind in self.active:
            self.parts[kind].append(data)
        if self.grade_depths:
            self.grade_tokens.append(data.strip())

    def handle_endtag(self, tag):
        if tag not in self.stack:
            return
        index = len(self.stack) - 1 - self.stack[::-1].index(tag)
        self.stack[index:] = []
        self.grade_depths = {depth for depth in self.grade_depths if depth <= index}
        completed = [kind for kind, depth in self.active.items() if depth > index]
        for kind in self.active:
            self.parts[kind].append(' ')
        for kind in completed:
            if self.active[kind] == index + 1:
                self.closed.add(kind)
            del self.active[kind]
        if 'conditions' in completed:
            self.conditions_closed = 'conditions' in self.closed
            raise _HeaderDone


def extract_conditions(html: str, region: str) -> dict:
    """Read saved header fields without building or scanning the full DOM.

    Parsing stops at the end of the first RaceData02.  Unclosed condition
    markup is not treated as a complete header.  A page with no usable
    conditions remains valid input and returns unknown fields.
    """
    parser = _HeaderExtractor()
    try:
        parser.feed(html)
        parser.close()
    except _HeaderDone:
        pass
    text = {kind: ' '.join(''.join(parts).split()) for kind, parts in parser.parts.items()}
    if not parser.conditions_closed:
        text['conditions'] = ''
    for kind in ('name', 'title'):
        if kind not in parser.closed:
            text[kind] = ''
    grade_tokens = parser.grade_tokens if 'name' in parser.closed else ()
    return parse_conditions(text['name'], text['title'], text['conditions'], region, grade_tokens)
