"""Which spellings are the same name.

This mirrors mycomap.org's services/nameVariants.ts, rule for rule. Both are held
to one shared fixture (tests/data/name-variants.json here, test-support/
name-variants.json there; the two copies must stay identical), so change the
rules, the TypeScript and that file together.

House style for a temporary code is `Genus sp. 'CODE'`: straight single quotes,
a hyphen between an epithet and its code ('vulgare-CA01'), two digits at least
('IN07').

Vision merges only what the rule calls "same" (spellings that differ in how they
are written: quotes, `sp.`, spacing, capitals, odd characters, a hyphen). What
needs a person ("check": a code also in use as a plain name, IN7 / IN07, a
described code) is left as separate labels and reported by `mv name-spellings`.
The name stored in the manifest is never rewritten: labels are worked out when
records are loaded (evaluate.load_records).

Where Python and JavaScript differ, this file follows JavaScript:
- whitespace is JavaScript's `\\s` (JS_SPACE below), not Python's;
- digits and letters in the patterns are ASCII, and `$` is the very end (`\\Z`);
- ordering (`sort_key`) follows what `localeCompare` does with the names we
  have: spaces, then punctuation, then digits, then letters; accents, case and
  quote glyphs only break ties, lower case first. It is a fixed table, not the
  machine's locale, so the order is the same everywhere.
"""

from __future__ import annotations

import math
import re
import sqlite3
import unicodedata
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from typing import Iterable, Mapping, NamedTuple

# Differences that leave no doubt the spellings are one name.
MECHANICAL = frozenset({"quotes", "sp_prefix", "unquoted_code", "spacing", "capitals",
                        "characters", "hyphen"})

# What JavaScript's \s and trim() call white space.
JS_SPACE = ("\t\n\x0b\x0c\r \xa0\u1680\u2000\u2001\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
            "\u2009\u200a\u2028\u2029\u202f\u205f\u3000\ufeff")
_S = "[" + JS_SPACE + "]"
_NOT_S = "[^" + JS_SPACE + "]"

_QUOTES = re.compile("[`'\u2018\u2019\u2032\"\u201c\u201d\ufffd]")
_QUOTE_GLYPHS = re.compile("[`\u2018\u2019\u2032\"\u201c\u201d]")
_SPACES = re.compile(_S + "+")
_GENUS = "([A-Za-z][a-z]+(?:-[a-z]+)?)"
_REGION_CODE = "[A-Z]{2,4}[0-9]+[a-z]?"

_QUOTED = re.compile("^" + _GENUS + r" (?:sp\.? ?)?' ?(?:sp[-. ] ?)?([^']+?) ?'?\Z")
_UNQUOTED_SP = re.compile("^" + _GENUS + r" sp\.? (" + _NOT_S + r".*)\Z")
_UNQUOTED_EPITHET_CODE = re.compile(
    "^" + _GENUS + " ([a-z]+(?:-[a-z]+)*)[ -](" + _REGION_CODE + r")\Z")
_UNQUOTED_CODE = re.compile("^" + _GENUS + r" ([A-Z][A-Za-z]*[0-9]+[a-z]?)\Z")
_EPITHET_AND_CODE = re.compile("^[a-z]+(?:-[a-z]+)* " + _REGION_CODE + r"\Z")
_HOUSE_STYLE = re.compile(r"^[A-Z][a-z]+(?:-[a-z]+)? sp\. '[^']+'\Z")

_NAME_START = re.compile("^[A-Za-z][a-z-]+ " + _NOT_S)
_NOT_A_NAME = re.compile(
    r"^(uncultured|unidentified|environmental|fungi|fungal|sample|sequence|mixed)\b",
    re.ASCII | re.IGNORECASE)
_ONLY_SP = re.compile(r"^sp\.?\Z", re.ASCII | re.IGNORECASE)
_PAD = re.compile(r"([A-Za-z]+)([0-9]+)([a-z]?)\Z")
_CODE_KEY_NUMBER = re.compile(r"-?0*([0-9]+)([a-z]?)\Z")
_UNPADDED = re.compile(r"0*([0-9]+)([a-z]?)\Z")
_LAST_DIGITS = re.compile(r"([0-9]+)[a-z]?\Z")
_EPITHET_ONLY = re.compile(r"^[a-z]+(?:-[a-z]+)*\Z")
_SP_INSIDE_QUOTES = re.compile("'" + _S + "*sp[-. ]")
_ONE_DIGIT = re.compile(r"[A-Za-z][0-9][a-z]?\Z")


def _text(raw) -> str:
    return "" if raw is None else str(raw)


def fold_name(raw) -> str:
    """Quote glyphs, non-breaking and doubled spaces, and compatibility characters made plain."""
    s = unicodedata.normalize("NFKC", _text(raw))
    return _SPACES.sub(" ", _QUOTES.sub("'", s)).strip(JS_SPACE)


def is_name_like(raw) -> bool:
    """False for what is not a name to fix: blank, a link, a note ("delete", "sample
    mixup"), a GenBank placeholder."""
    s = fold_name(raw)
    if not _NAME_START.search(s) or "://" in s or len(s) > 255:
        return False
    return not _NOT_A_NAME.search(s)


@dataclass(frozen=True)
class Code:
    genus: str
    code: str


@dataclass(frozen=True)
class NameParts:
    folded: str            # odd characters and spacing folded; never written anywhere
    key: str               # spellings of one name share this; never written anywhere
    code: Code | None      # genus and code when the name is a temporary code
    house_style: str       # a temporary code rewritten, anything else folded
    is_house_style: bool   # already written exactly in house style


def _capital(genus: str) -> str:
    return genus[:1].upper() + genus[1:]


def _pad_code(code: str) -> str:
    """'IN7' -> 'IN07'; two digits or more are left alone. Same rule as Temp Code Transfer."""
    return _PAD.sub(lambda m: m[1] + m[2].rjust(2, "0") + m[3], code, count=1)


def _code_key(code: str) -> str:
    # 'flavoconia-01' and 'flavoconia01', 'IN7' and 'IN07' are one code.
    return _CODE_KEY_NUMBER.sub(r"\g<1>\g<2>", code.lower().replace(" ", "-"), count=1)


def _read_code(folded: str) -> Code | None:
    m = _QUOTED.search(folded)
    if m:
        inner = m[2].strip(JS_SPACE)
        if inner and not _ONLY_SP.search(inner):
            return Code(_capital(m[1]), inner)
    if "'" in folded:
        return None
    m = _UNQUOTED_SP.search(folded)
    if m and re.search("[A-Z0-9]", m[2]):
        return Code(_capital(m[1]), m[2])
    m = _UNQUOTED_EPITHET_CODE.search(folded)
    if m:
        return Code(_capital(m[1]), f"{m[2]}-{m[3]}")
    m = _UNQUOTED_CODE.search(folded)
    if m:
        return Code(_capital(m[1]), m[2])
    return None


def parse_name(raw) -> NameParts:
    return _parse(_text(raw))


@lru_cache(maxsize=None)      # the same few thousand names, every time records are loaded
def _parse(text: str) -> NameParts:
    folded = fold_name(text)
    code = _read_code(folded)
    if not code:
        return NameParts(folded, folded.lower(), None, folded, text == folded)
    is_house_style = bool(_HOUSE_STYLE.search(text))
    # A name already in house style is never rewritten: its spaces may be meant
    # ('IN01 x IN02', 'Green Madrone').
    house = text if is_house_style else \
        f"{code.genus} sp. '{_pad_code(code.code.replace(' ', '-'))}'"
    return NameParts(folded, f"{code.genus.lower()} sp. '{_code_key(code.code)}'", code, house,
                     is_house_style)


def variant_key(raw) -> str:
    return parse_name(raw).key


def _digits(s: str) -> str:
    m = _LAST_DIGITS.search(s)
    return m[1] if m else ""


def _number(digits: str) -> float:
    return float(digits) if digits else 0.0       # JavaScript's Number("") is 0


def _unpadded(s: str) -> str:
    return _UNPADDED.sub(r"\g<1>\g<2>", s, count=1)


def reasons_for(spelling, target) -> list[str]:
    """How one spelling differs from the spelling it should become."""
    raw = _text(spelling)
    src, to = parse_name(raw), parse_name(target)
    reasons: list[str] = []

    def add(reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    nfkc = unicodedata.normalize("NFKC", raw)
    if raw != nfkc or "\ufffd" in raw or "\xa0" in raw:
        add("characters")
    if _QUOTE_GLYPHS.search(raw):
        add("quotes")
    if nfkc.replace("\xa0", " ") != _SPACES.sub(" ", nfkc).strip(JS_SPACE):
        add("spacing")
    if src.code and to.code:
        if "'" not in src.folded:
            add("unquoted_code")
        elif " sp. '" not in src.folded or _SP_INSIDE_QUOTES.search(src.folded):
            add("sp_prefix")
        a, b = src.code.code, to.code.code

        def joined(s: str) -> str:
            return re.sub("[ -]", "", _unpadded(s))

        if re.search("^[a-z]", src.folded) or (a != b and a.lower() == b.lower()):
            add("capitals")
        if _digits(a) != _digits(b) and _number(_digits(a)) == _number(_digits(b)):
            add("zero_padding")
        if (_unpadded(a).lower() != _unpadded(b).lower()
                and joined(a).lower() == joined(b).lower()):
            # An epithet and its code written with a space is plain spacing; any
            # other space, or a hyphen that comes and goes, is a hyphen difference.
            def words(s: str) -> str:
                return _unpadded(s).lower().replace(" ", "-")

            if words(a) != words(b):
                add("hyphen")
            elif not _EPITHET_AND_CODE.search(a) and not _EPITHET_AND_CODE.search(b):
                add("descriptive_code")
            else:
                add("spacing")
        # Words in a code that are not an epithet and its code: a description, a cross.
        if " " in a and not _EPITHET_AND_CODE.search(a):
            add("descriptive_code")
    elif src.code or to.code:
        add("name_or_code")
    elif src.folded != to.folded and src.folded.lower() == to.folded.lower():
        add("capitals")
    return reasons


class NameCount(NamedTuple):
    name: str | None
    source: str
    count: int


@dataclass
class VariantSpelling:
    name: str
    records: int
    by_source: dict[str, int]
    keep: bool                    # already the spelling to keep
    reasons: list[str]


@dataclass
class VariantGroup:
    key: str
    confidence: str               # "same": only the writing differs. "check": a person decides
    proposed_name: str | None     # the spelling to keep; None when a person has to choose it
    proposed_is_new: bool         # no record carries the proposed spelling yet
    reasons: list[str]
    spellings: list[VariantSpelling]
    records: int
    records_to_fix: int

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class _Tally:
    name: str
    parts: NameParts
    records: int = 0
    by_source: dict[str, int] = field(default_factory=dict)


# The order localeCompare gives the characters found in names (checked against Node,
# en-US). Characters in one group sort as the same letter and only break ties, in
# the order written: a space and a non-breaking space, the quote glyphs, a and A.
_ORDER = ([" \u00a0", "_", "-", ",", ";", ":", "!", "?", ".", "'\u2018\u2019", '"\u201c\u201d',
           "(", ")", "[", "]", "{", "}", "@", "*", "/", "\\", "&", "#", "%", "`", "^", "+",
           "\u00d7", "<", "=", ">", "|", "~", "$"] + list("0123456789")
          + [c + c.upper() for c in "abcdefghijklmnopqrstuvwxyz"])
_RANK = {c: (i, j) for i, group in enumerate(_ORDER) for j, c in enumerate(group)}


def sort_key(name: str):
    """Names in the order localeCompare puts them: by letters first (spaces and
    punctuation before digits before letters), then accents, then case and glyph
    with lower case first, then the characters themselves. A fixed table, not a
    locale; a character outside the table sorts after the letters."""
    letters, accents, forms = [], [], []
    for ch in unicodedata.normalize("NFD", name):
        if unicodedata.combining(ch):
            accents.append((len(letters), ord(ch)))
            continue
        letter, form = _RANK.get(ch, (len(_ORDER) + ord(ch), 0))
        letters.append(letter)
        forms.append(form)
    return letters, accents, forms, name


def _most_records(t: _Tally):
    return -t.records, sort_key(t.name)


def _count(value) -> float | int:
    """JavaScript's `Number(count) || 0`."""
    try:
        n = float(value)
    except (TypeError, ValueError):
        return 0
    if not math.isfinite(n):
        return 0
    return int(n) if n == int(n) else n


def _tally(rows: Iterable) -> dict[str, list[_Tally]]:
    by_spelling: dict[str, _Tally] = {}
    for row in rows:
        raw, source, count = row
        name, count = _text(raw), _count(count)
        if count <= 0 or not is_name_like(name):
            continue
        t = by_spelling.get(name)
        if t is None:
            t = by_spelling[name] = _Tally(name, parse_name(name))
        t.records += count
        t.by_source[source] = t.by_source.get(source, 0) + count
    by_key: dict[str, list[_Tally]] = {}
    for t in by_spelling.values():
        by_key.setdefault(t.parts.key, []).append(t)
    return by_key


def _code_of(t: _Tally) -> str:
    return t.parts.code.code if t.parts.code else ""


def _pick_proposed(members: list[_Tally]) -> str:
    is_code = any(m.parts.code for m in members)
    if is_code:
        clean = [m for m in members if m.parts.is_house_style]
    else:
        clean = [m for m in members if m.name == m.parts.folded and re.search("^[A-Z]", m.name)]
    # Of the spellings already in house style: hyphenated ('x-PNW10', not
    # 'x PNW10'), then two digits ('IN07', not 'IN7'), then the most used.
    if clean:
        return min(clean, key=lambda m: (int(" " in _code_of(m)),
                                         int(bool(_ONE_DIGIT.search(_code_of(m)))),
                                         _most_records(m))).name
    top = min(members, key=_most_records)
    return top.parts.house_style if is_code else _capital(top.parts.folded)


def group_name_variants(rows: Iterable) -> list[VariantGroup]:
    """Every name that needs a fix: spellings that share a key, a temporary code
    written in an old format, and a code that is also in use as a plain name
    (Craterellus neotubaeformis / Craterellus sp. 'neotubaeformis').

    `rows` are (name, source, count), e.g. NameCount."""
    by_key = _tally(rows)
    groups: list[VariantGroup] = []
    used: set[str] = set()

    for key, members in by_key.items():
        code = next((m.parts.code for m in members if m.parts.code), None)
        if not code or not _EPITHET_ONLY.search(code.code):
            continue
        plain_key = f"{code.genus.lower()} {code.code}"
        plain = by_key.get(plain_key)
        if not plain:
            continue
        used.update((key, plain_key))
        both = sorted(members + plain, key=_most_records)
        records = sum(m.records for m in both)
        groups.append(VariantGroup(
            key, "check", None, False, ["name_or_code"],
            [VariantSpelling(m.name, m.records, m.by_source, False, []) for m in both],
            records, records))

    for key, members in by_key.items():
        if key in used:
            continue
        proposed = _pick_proposed(members)
        off = [m for m in members if m.name != proposed]
        if not off:
            continue
        spellings = [VariantSpelling(m.name, m.records, m.by_source, m.name == proposed,
                                     [] if m.name == proposed else reasons_for(m.name, proposed))
                     for m in sorted(members, key=_most_records)]
        reasons = list(dict.fromkeys(r for s in spellings for r in s.reasons))
        groups.append(VariantGroup(
            key, "same" if all(r in MECHANICAL for r in reasons) else "check", proposed,
            not any(m.name == proposed for m in members), reasons, spellings,
            sum(m.records for m in members), sum(m.records for m in off)))
    return sorted(groups, key=lambda g: (-g.records_to_fix, sort_key(g.key)))


# --- labels: what Vision does with the groups -----------------------------------------

def label_map(counts: Mapping[str, int], groups: list[VariantGroup] | None = None
              ) -> dict[str, str]:
    """{spelling: label} for every counted name. Spellings of a "same" group all get
    the group's proposed name; every other name, and every spelling in a "check"
    group, is its own label, unchanged."""
    if groups is None:
        groups = group_name_variants((name, "", n) for name, n in counts.items())
    labels = {name: name for name in counts}
    for g in groups:
        if g.confidence == "same":
            for s in g.spellings:
                labels[s.name] = g.proposed_name
    return labels


def name_counts(conn: sqlite3.Connection) -> list[NameCount]:
    """Every name in the manifest with how many records carry it, per source. All
    records count (every region, test and reference alike), so a name gets the
    same label wherever it is used."""
    return [NameCount(*row) for row in conn.execute(
        "select scientific_name, source, count(*) from records "
        "where coalesce(scientific_name, '') <> '' group by 1, 2 order by 1, 2")]


def counted(rows: Iterable) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, _source, n in rows:
        out[name] = out.get(name, 0) + n
    return out


def manifest_labels(conn: sqlite3.Connection, extra: Iterable[str] = ()) -> dict[str, str]:
    """The label of every name in the manifest. `extra` are names met elsewhere (an
    earlier prediction) to be labelled by the same rule, as if one record had them."""
    counts = counted(name_counts(conn))
    for name in extra:
        if name:
            counts.setdefault(name, 1)
    return label_map(counts)


def summary(rows: list, groups: list[VariantGroup]) -> dict:
    """What `mv name-spellings` reports: how many groups, how many are merged in
    Vision and how many wait for a person."""
    counts = counted(rows)
    labels = label_map(counts, groups)
    same = [g for g in groups if g.confidence == "same"]
    check = [g for g in groups if g.confidence == "check"]
    return {
        "names": len(counts),
        "labels": len(set(labels.values())),
        "groups": len(groups),
        "records_to_fix": sum(g.records_to_fix for g in groups),
        "merged_in_vision": {
            "groups": len(same),
            "spellings": sum(len(g.spellings) for g in same),
            "records": sum(g.records for g in same),
            "records_relabelled": sum(g.records_to_fix for g in same),
        },
        "left_for_a_person": {
            "groups": len(check),
            "spellings": sum(len(g.spellings) for g in check),
            "records": sum(g.records for g in check),
        },
    }
