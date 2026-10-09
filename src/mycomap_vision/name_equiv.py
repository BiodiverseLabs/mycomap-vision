"""When two names count as the same, for scoring only (never for training labels).

Three readings of "the answer named the right species", each its own column, and two of
the right genus:

    strict    the same name, its spellings folded together (names.py: the labels
              Vision trains and scores with, rule for rule the same as .org). A
              temporary code is a species like any other: the same code in the same
              genus.
    s.l.      strict, or genera of one group (genus_groups.json: recently split genera
              such as Cortinarius s.l.) with the same described epithet ('pacificus'
              under Calonarius, Thaxterogaster and Cortinarius), its Latin gender ending
              aside ('rimosa' / 'rimosum'). Not a temporary code: codes are numbered
              within a genus, so Calonarius sp. 'IN06' and Cortinarius sp. 'IN06' are
              usually two taxa.
    complex   BETA. s.l., or genera of one group with the same epithet stem: a described
              name and the provisional names split from it ('fallax' and 'fallax-PNW03'),
              or two provisional names on one stem ('schweinitzii-IN01' and '-IN02'), or a
              species and its subspecies. Gender endings aside here too; the code itself
              is never folded. A bare code ('CA04') has no stem and only ever matches
              itself. Report it as its own beta column next to strict, never in its
              place.

    genus strict   the same genus;   genus s.l.   genera of one group.

The API is small on purpose (other scorers call it): species_match, genus_match,
genus_group, parts, gender_fold. Steve, 2026-10-09: "we'll have to think about this
more, but make a beta"; the stem rule is the part most likely to change (docs/PLAN.md).
Steve, 2026-10-08: gender endings count as the same name; temporary codes "are just as
good as names" (about half the records).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from . import names

GROUPS_FILE = Path(__file__).with_name("genus_groups.json")
_QUALIFIERS = {"cf", "cf.", "aff", "aff.", "nr", "nr.", "sp", "sp.", "s.l.", "s.str."}
_EPITHET = re.compile(r"^[a-z]+(?:-[a-z]+)*$")
# The epithet at the start of a provisional code: 'fallax-PNW03', 'conchatus OH01',
# 'rooseveltensis'. A bare code ('CA04', 'IN-07', 'PNW01b') has none.
_CODE_STEM = re.compile(r"^([a-z]+(?:-[a-z]+)*?)(?:[- ]?[A-Za-z]*[0-9].*)?$")
MIN_STEM = 4              # 'alba' is an epithet; 'ca', 'pnw' are code letters
# Latin adjective endings that follow the genus's gender, by declension: -us/-a/-um
# ('rimosus', 'rimosa', 'rimosum') and -is/-e ('viridis', 'viride').
_GENDER_ENDINGS = (("us", "1"), ("um", "1"), ("a", "1"), ("is", "2"), ("e", "2"))


@lru_cache(maxsize=1)
def _group_of() -> dict[str, str]:
    data = json.loads(GROUPS_FILE.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for group, genera in data["groups"].items():
        for g in genera:
            if g.lower() in out:
                raise ValueError(f"{g} is in two genus groups")
            out[g.lower()] = group
    return out


def gender_fold(epithet: str) -> str:
    """An epithet with its gender ending set aside: 'rimosa', 'rimosus' and 'rimosum'
    -> 'rimos:1'; 'viridis', 'viride' -> 'virid:2'; 'ruber', 'rubra', 'rubrum' -> 'rubr:1'
    (an -er stem drops its e). Endings of different declensions stay apart."""
    w = epithet.lower()
    for end, declension in _GENDER_ENDINGS:
        if w.endswith(end):
            stem = w[:-len(end)]
            break
    else:
        if not w.endswith("er"):
            return w
        stem, declension = w, "1"
    if stem.endswith("er"):
        stem = stem[:-2] + "r"
    return f"{stem}:{declension}"


def _fold_words(species_part: str) -> str:
    return " ".join(gender_fold(w) if _EPITHET.match(w) else w for w in species_part.split())


def genus_group(genus: str | None) -> str:
    """The group a genus belongs to ('Cortinarius s.l.'), else the genus itself."""
    g = (genus or "").strip()
    return _group_of().get(g.lower(), g)


@dataclass(frozen=True)
class Parts:
    genus: str             # as written, capitalised
    species_part: str      # what follows the genus, spelling-folded; "" for a genus alone
    stem: str | None       # the epithet stem, gender-folded; None for a bare code or no epithet
    provisional: bool      # a temporary code ('CA04', 'fallax-PNW03'), not a described name
    folded: str            # species_part with each epithet gender-folded (s.l. compares it)


@lru_cache(maxsize=None)
def parts(name: str) -> Parts:
    p = names.parse_name(name or "")
    if p.code:
        # Matched as written: an epithet is lower case ('fallax-PNW03'), a code is not
        # ('CA04', 'PNW01'). A short lower-case start ('ca04') is a code written small.
        m = _CODE_STEM.match(p.code.code.strip())
        stem = m[1].lower() if m and m[1] and len(m[1]) >= MIN_STEM else None
        part = p.key.split(" ", 1)[1]
        return Parts(p.code.genus, part, stem and gender_fold(stem), True, part)
    words = p.folded.split()
    if not words:
        return Parts("", "", None, False, "")
    genus = words[0][:1].upper() + words[0][1:]
    rest = [w for w in words[1:] if w.lower() not in _QUALIFIERS]
    epithet = rest[0].lower() if rest and _EPITHET.match(rest[0].lower()) else None
    part = " ".join(w.lower() for w in rest)
    return Parts(genus, part, epithet and gender_fold(epithet), False, _fold_words(part))


def species_match(answer: str, truth: str) -> dict[str, bool]:
    """{"strict", "sl", "complex"} for a species answer against the true name. Each
    implies the next (strict -> s.l. -> complex)."""
    if not answer or not truth:
        return {"strict": False, "sl": False, "complex": False}
    strict = names.variant_key(answer) == names.variant_key(truth)
    a, t = parts(answer), parts(truth)
    same_group = bool(a.genus) and genus_group(a.genus) == genus_group(t.genus)
    sl = strict or (same_group and not a.provisional and not t.provisional
                    and bool(a.folded) and a.folded == t.folded)
    complex_ = sl or (same_group and a.stem is not None and a.stem == t.stem)
    return {"strict": strict, "sl": sl, "complex": complex_}


def genus_match(answer: str, truth: str) -> dict[str, bool]:
    """{"strict", "sl"} for a genus answer against the true genus."""
    if not answer or not truth:
        return {"strict": False, "sl": False}
    strict = answer.strip().lower() == truth.strip().lower()
    return {"strict": strict, "sl": strict or genus_group(answer) == genus_group(truth)}
