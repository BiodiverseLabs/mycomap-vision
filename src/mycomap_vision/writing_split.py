"""One name, several writings (name_writing_split), split in two (Steve, 2026-11-18).

Of the queue's items most are an old way of writing a temporary code beside today's
(Mycena "sp-IN10" and Mycena sp. 'IN10', Lepiota 'castaneidisca PNW13' and Lepiota sp.
'castaneidisca-PNW13'): nothing about the identification changes, and mycomap.org's own Names
to fix lists them already, under the reason "Old format". Those are "format" here: one panel
says where they are fixed on .org, and each closes by itself once no record Vision serves
carries an old writing (feedback_org.close_done_formats). The rest need a person: "look".

The rule is .org's (services/nameVariants.ts and nameFixRecommend.ts), through Vision's copy of
it in names.py, so the two sites agree on what counts as format:

  format   every writing has one variant key; every difference is mechanical (quotes, the
           sp- prefix, a missing sp. or quotes, spacing, capitals, odd characters, a hyphen)
           and not only spacing, capitals, odd characters or a hyphen (.org's "Spacing"); and
           the name to keep is a code in house style, Genus sp. 'CODE', with no space inside
  look     anything else: a number written two ways (IN5 / IN05, .org's "Number rewritten"), a
           described code ('acrid orange', its space may be meant), a rank word (rubescens
           alba / rubescens var. alba), a code that is also a plain name, a stray character

The proposal is MycoMap's current format, never only the most used writing: the seed proposed
the most used one, which in some items is the old format (Lactarius "acrid orange" (32) beside
Lactarius sp. 'acrid-orange' (13)). It is worked out when an item is read (Store._item); the
stored item is never rewritten. .org's code registry (renamed, undecided) is not known here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import names

FORMAT, LOOK = "format", "look"
LISTS = (FORMAT, LOOK)

# Reasons .org files under "Spacing" rather than "Old format" (nameFixRecommend.ts SPACING).
SPACING = frozenset({"spacing", "capitals", "characters", "hyphen"})
_COUNT = re.compile(r"\s*\(\s*([\d,]+)\s*\)\s*\Z")
_CLEAN_CODE = re.compile(r"\A[A-Za-z0-9-]+\Z")
_RANK_WORD = re.compile(r"\s+(?:var\.|subsp\.|ssp\.|f\.|forma|subvar\.)(?=\s)", re.IGNORECASE)


def writings(current: str | None) -> list[tuple[str, int | None]]:
    """"Name A (17) | Name B (1)" -> [("Name A", 17), ("Name B", 1)], each writing once."""
    out: list[tuple[str, int | None]] = []
    for part in (current or "").split(" | "):
        m = _COUNT.search(part)
        name = _COUNT.sub("", part).strip()
        if name and name not in [n for n, _ in out]:
            out.append((name, int(m[1].replace(",", "")) if m else None))
    return out


@dataclass(frozen=True)
class Split:
    kind: str                    # format | look (the list it is shown in)
    reason: str                  # format | spacing | number | described | rank | name_or_code
                                 # | other
    proposal: str                # the writing to keep, in MycoMap's current format
    reasons: list[str] = field(default_factory=list)   # names.reasons_for, every writing


def _current_format(name: str) -> str:
    p = names.parse_name(name)
    return p.house_style if p.code else name


def classify(current: str | None, proposed: str | None) -> Split:
    """Which list an item belongs in, and the writing it should propose."""
    ws = writings(current)
    stored = (proposed or "").strip()
    fallback = _current_format(stored) if stored else (ws[0][0] if ws else "")
    if len(ws) < 2:
        return Split(LOOK, "other", fallback)
    keys = {names.variant_key(n) for n, _ in ws}
    if len(keys) > 1:
        coded = {bool(names.parse_name(n).code) for n, _ in ws}
        unranked = {names.variant_key(_RANK_WORD.sub("", names.fold_name(n))) for n, _ in ws}
        reason = ("name_or_code" if coded == {True, False}
                  else "rank" if len(unranked) == 1 else "other")
        return Split(LOOK, reason, fallback)
    groups = names.group_name_variants((n, "seed", c or 1) for n, c in ws)
    if len(groups) != 1 or not groups[0].proposed_name:
        return Split(LOOK, "other", fallback)
    g = groups[0]
    keep = g.proposed_name
    p = names.parse_name(keep)
    reasons = list(g.reasons)
    if "zero_padding" in reasons:
        reason = "number"
        # Two current-format writings whose numbers differ by a leading zero ('IN5', 'IN05')
        # are two names on .org (nameFixRecommend.splitPaddedNames, Steve, Nov 2026): neither
        # is proposed for the other, so the item keeps what it proposed.
        written = {names.parse_name(n).code.code for n, _ in ws
                   if names.parse_name(n).is_house_style and names.parse_name(n).code}
        if len(written) > 1:
            keep = fallback
    elif "descriptive_code" in reasons:
        reason = "described"
    elif "name_or_code" in reasons:
        reason = "name_or_code"
    elif g.confidence != "same":
        reason = "other"
    elif all(r in SPACING for r in reasons):
        reason = "spacing"
    elif p.code and p.is_house_style and _CLEAN_CODE.search(p.code.code):
        reason = "format"
    else:
        reason = "other"
    return Split(FORMAT if reason == "format" else LOOK, reason, keep, reasons)


def of_item(item: dict) -> Split | None:
    """The split of a name_writing_split item (None for any other)."""
    if item.get("item_type") != "name_writing_split":
        return None
    return classify(item.get("current"), item.get("proposed"))
