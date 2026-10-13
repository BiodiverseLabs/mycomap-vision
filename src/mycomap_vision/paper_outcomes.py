"""Per-find outcomes for `mv paper-score`: every comparator's answer on a find turned into
the same record, so one set of tables and tests serves Vision, iNaturalist's computer
vision, the expert identifiers and the published classifiers alike.

An outcome is {cell: True | False | None} over the cells of the standard table:
`species_strict@k`, `species_sl@k`, `species_complex@k`, `genus_strict@k`, `genus_sl@k`
for k in 1, 3, 5, 10 (a looser level counts every stricter hit; name_equiv, beta), plus
`answered`. None means the find is not scorable for that cell by that comparator (an
expert who named no species; a provisional truth for a model without provisional codes);
False includes "no answer" and "no reference for the true species" (paper rule: such
finds count as wrong).

Readers here take the comparisons lane's files (E14 finds.jsonl, E31 scores.jsonl, format
1) and the published-model answers (`mv external run`'s <model>.jsonl). Each line is
joined to a find by `record_key`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

K = (1, 3, 5, 10)
LEVELS = {"species": ("strict", "sl", "complex"), "genus": ("strict", "sl")}
CELLS = tuple(f"{rank}_{lv}@{k}" for rank, lvs in LEVELS.items() for lv in lvs for k in K)
FORMAT = 1


@dataclass
class Find:
    """A scored find: its answer key and the attributes tables, tests and slices use."""
    key: str
    observer: str
    species: str                  # "" for a one-word name
    genus: str
    family: str
    name_kind: str                # formal | provisional | one-word
    depth: int                    # reference records of the true species (0 = none)
    photos: int
    attrs: dict = field(default_factory=dict)       # slices: season, source, licence ...
    near_duplicate: dict = field(default_factory=dict)   # {"bytes"|"dhash"|"cosine": bool}

    @property
    def is_near_duplicate(self) -> bool:
        return any(self.near_duplicate.values())


def empty(answered: bool = False) -> dict:
    return {**{c: False for c in CELLS}, "answered": answered}


def from_ranked(species: list[str], genus: list[str], find: Find,
                species_scorable: bool = True) -> dict:
    """Outcome from ranked name lists (best first). A cell's level is hit at k when any of
    the top k answers matches the truth at that level or a stricter one; strict is the
    exact name (as the standard table's strict row)."""
    from . import name_equiv
    out = empty(answered=bool(species or genus))
    for rank, answers, truth, match in (
            ("species", species, find.species, name_equiv.species_match),
            ("genus", genus, find.genus, name_equiv.genus_match)):
        levels = LEVELS[rank]
        if not truth or (rank == "species" and not species_scorable):
            for lv in levels:
                for k in K:
                    out[f"{rank}_{lv}@{k}"] = None
            continue
        first = {lv: None for lv in levels}
        for i, answer in enumerate(answers[:max(K)]):
            m = match(answer, truth)
            m["strict"] = answer == truth
            for j, lv in enumerate(levels):
                if first[lv] is None and any(m[x] for x in levels[:j + 1]):
                    first[lv] = i
        for lv, pos in first.items():
            for k in K:
                out[f"{rank}_{lv}@{k}"] = pos is not None and pos < k
    return out


# --- the comparisons lane's files ----------------------------------------------------------------

def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def inat_outcome(line: dict, find: Find, score: str) -> dict:
    """iNat CV's outcome on a find from an E14 finds.jsonl line (`score`: "vision", photos
    only, or "combined", with the public point and date). Strict cells come from the lane's
    own hits, matched by iNat taxon id (a truth iNat has no taxon for, and no suggestion,
    are misses); s.l. and complex come from the ranked names (outside models: complex on
    formal names only). Species cells are None unless the truth is a formal species."""
    hits = line[f"{score}_hits"]
    ranked = line[score]
    names = {rank: [r[1] for r in ranked.get(rank) or []] for rank in ("species", "genus")}
    formal = bool(line["truth"].get("formal_species"))
    out = from_ranked(names["species"], names["genus"], find, species_scorable=formal)
    answered = line.get("answered") or {}
    # combined_raw is the same answer as combined, on iNat's raw place-aware scale.
    out["answered"] = bool(answered.get(score, answered.get(score.split("_")[0])))
    for rank in ("species", "genus"):
        h = hits.get(rank)
        if h is None:
            for lv in LEVELS[rank]:
                for k in K:
                    out[f"{rank}_{lv}@{k}"] = None
            continue
        for k in K:
            strict = bool(h[f"top{k}"])
            out[f"{rank}_strict@{k}"] = strict
            for lv in LEVELS[rank][1:]:      # a looser level counts every stricter hit
                out[f"{rank}_{lv}@{k}"] = bool(out[f"{rank}_{lv}@{k}"]) or strict
    return out


def inat_confidence(line: dict, score: str) -> dict:
    """iNat's own confidence in its top answer per rank (its score, 0-100, as a fraction),
    for the risk-coverage comparison; iNat gives no probability for the true name."""
    out = {}
    for rank in ("species", "genus"):
        ranked = (line.get(score) or {}).get(rank) or []
        out[rank] = {"conf": (float(ranked[0][2]) / 100.0) if ranked else 0.0}
    return out


EXPERT_VIEWS = ("first", "best", "community")


def expert_outcome(line: dict, tier: str = "1", view: str = "first", strict: bool = False
                   ) -> dict | None:
    """The experts' outcome on a find from an E31 scores.jsonl line, or None when the line
    is not scored (status not ok; with `strict`, the observer had named a species before
    the cutoff). `first` (primary, one opinion per find): at species, the earliest
    before-cutoff species-level ID by an identifier of `tier` (by made_at); at genus, the
    earliest such ID that can be scored at genus. `best`: any identifier of `tier` with a
    before-cutoff ID right at that rank. Either is None at a rank no such identifier
    reached (stopping at genus is coverage, not error). `community`: the community taxon as
    it stood at the cutoff. Experts give one name, so only the @1 cells are set; s.l. and
    complex equal strict (the lane scores taxa, not spellings)."""
    if view not in EXPERT_VIEWS:
        raise ValueError(f"expert view is one of {EXPERT_VIEWS}")
    if line.get("status") != "ok":
        return None
    if strict and line.get("observer_named_species_before_cutoff"):
        return None
    out = {c: None for c in CELLS}
    if view == "community":
        com = line.get("community") or {}
        out["answered"] = bool(com.get("present"))
        sp = com.get("species_right") if com.get("at_species") else None
        ge = com.get("genus_right") if com.get("present") else None
    else:
        ids = [i for i in line.get("ids") or [] if i.get("role") == "identifier"
               and i.get("before_cutoff") and str(i.get("tier")) == str(tier)]
        out["answered"] = bool(ids)
        ids = sorted(ids, key=lambda i: str(i.get("made_at") or ""))
        sp_ids = [i["species_right"] for i in ids if i.get("species_right") is not None]
        ge_ids = [i["genus_right"] for i in ids if i.get("genus_right") is not None]
        if view == "first":
            sp = sp_ids[0] if sp_ids else None
            ge = ge_ids[0] if ge_ids else None
        else:
            sp = any(sp_ids) if sp_ids else None
            ge = any(ge_ids) if ge_ids else None
    for rank, v in (("species", sp), ("genus", ge)):
        if v is None:
            continue
        for lv in LEVELS[rank]:
            out[f"{rank}_{lv}@1"] = bool(v)
    return out


def published_outcome(line: dict, find: Find) -> dict:
    """A published classifier's outcome from `mv external run`'s <model>.jsonl (its answer
    already in Vision's names, published.vision_names)."""
    res = line["result"]
    return from_ranked([a["name"] for a in res.get("species") or []],
                       [a["name"] for a in res.get("genus") or []], find)


def check_format(lines: list[dict], what: str) -> None:
    """Refuse a file that is not format 1 of the agreed shape."""
    need = {"e14": ("record_key", "truth", "vision", "combined", "vision_hits",
                    "combined_hits", "answered", "photo_ids"),
            "e31": ("record_key", "status", "ids", "community")}[what]
    for i, line in enumerate(lines[:50]):
        missing = [f for f in need if f not in line]
        if missing:
            raise ValueError(f"{what} line {i + 1} lacks {missing}: not format {FORMAT}")
