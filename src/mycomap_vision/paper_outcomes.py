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


# --- the answer key: the one place a species or genus answer is judged ----------------------------

@dataclass(frozen=True)
class AnswerKey:
    """Steve's answer key (name_equiv: species rank, one name per writing, a hybrid code
    right on either parent; with `taxonomy`, the canonical step to the iNat taxon at the
    frozen snapshot: an answer in iNat's newer names, or as an iNat taxon id, is not marked
    wrong for the taxonomy's version, and a genus truth accepts its label genus and its
    resolved iNat genus). EVERY comparator's species and genus answers are judged here,
    through name_equiv.species_credit and name_equiv.genus_match, so no comparator is scored
    by a rule of its own; the comparisons lane's verdicts are diagnostics only. `rule`
    "exact" is v1's published rule (a sensitivity row; no canonical step)."""
    rule: str = ""
    taxonomy: object = None          # name_equiv.Taxonomy (the pinned snapshot), or None

    def __post_init__(self):
        from . import name_equiv
        if not self.rule:
            object.__setattr__(self, "rule", name_equiv.ANSWER_KEY)
        if self.rule not in name_equiv.ANSWER_KEYS:
            raise ValueError(f"answer key {self.rule!r}: one of {name_equiv.ANSWER_KEYS}")
        if self.rule == "exact":
            object.__setattr__(self, "taxonomy", None)

    def _side(self, answer, taxon, species: bool):
        """What the key reads for an answer: its iNat taxon id when the taxonomy places
        it, else its name."""
        tx = self.taxonomy
        if tx is None or taxon is None:
            return answer
        try:
            tid = int(taxon)
        except (TypeError, ValueError):
            return answer
        placed = tx.species_id(tid) if species else tx.current(tid)
        return tid if placed is not None else answer

    def species(self, answer: str, truth: str, taxon=None) -> bool:
        from . import name_equiv
        if not truth or (not answer and taxon is None):
            return False
        return bool(name_equiv.species_credit(self._side(answer, taxon, True), truth, self.rule,
                                              self.taxonomy))

    def genus(self, answer: str, truth: str, taxon=None, truth_species: str | None = None
              ) -> bool:
        from . import name_equiv
        if not truth or (not answer and taxon is None):
            return False
        if self.rule == "exact":
            return answer.strip().lower() == truth.strip().lower()
        return bool(name_equiv.genus_match(self._side(answer, taxon, False), truth,
                                           self.taxonomy, truth_species)["strict"])

    def species_levels(self, answer: str, truth: str) -> dict:
        from . import name_equiv
        return name_equiv.species_match(answer, truth, self.taxonomy)

    def genus_levels(self, answer: str, truth: str, truth_species: str | None = None) -> dict:
        from . import name_equiv
        return name_equiv.genus_match(answer, truth, self.taxonomy, truth_species)

    def collapse(self, answers: list, taxa: list | None = None) -> tuple[list, list | None]:
        """The ranked answers with those sharing a key kept once, at the first place (and
        their taxa alongside)."""
        from . import name_equiv
        kept = name_equiv.collapse(answers, self.rule, self.taxonomy)
        if not taxa:
            return kept, None
        first = {}
        for i, a in enumerate(answers):
            first.setdefault(a, i)
        return kept, [taxa[first[a]] if first[a] < len(taxa) else None for a in kept]

    def resolves(self, truth: str) -> bool | None:
        """Does a species truth resolve one-to-one to an iNat species (PLAN rule 7)? None
        without a taxonomy."""
        return None if self.taxonomy is None or not truth else bool(
            self.taxonomy.resolves(truth))


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


def species_rank(name: str) -> str:
    """A species name with any infraspecific part dropped ('Amanita muscaria var.
    guessowii' -> 'Amanita muscaria'; a trinomial without a marker likewise)."""
    return " ".join((name or "").split()[:2])


def from_ranked(species: list[str], genus: list[str], find: Find,
                species_scorable: bool = True, same=None, collapse: bool = False,
                key: AnswerKey | None = None, species_taxa: list | None = None,
                genus_taxa: list | None = None) -> dict:
    """Outcome from ranked name lists (best first), judged by the answer key (`key`;
    default: the current key). A cell's level is hit at k when any of the top k answers
    matches the truth at that level or a stricter one. Strict is the key's credit; a
    published model's answer may first be translated through the reviewed crosswalk
    (`same(answer, truth)`: then the truth's own label is what the key judges). Answers
    sharing a key count once, at their first place. `species_taxa` / `genus_taxa`: the
    answers' iNat taxon ids, for the canonical step. With `collapse` under the "exact" key,
    truth and answers are taken at species rank (species_rank), as v1's E15 rows were."""
    key = key or AnswerKey()
    if collapse and key.rule == "exact":
        find = Find(find.key, find.observer, species_rank(find.species), find.genus,
                    find.family, find.name_kind, find.depth, find.photos, find.attrs,
                    find.near_duplicate)
        seen, collapsed = set(), []
        for a in species:                       # one entry per species, in order
            c = species_rank(a)
            if c not in seen:
                seen.add(c)
                collapsed.append(c)
        species, species_taxa = collapsed, None
    elif key.rule != "exact":
        species, species_taxa = key.collapse(list(species), species_taxa)
    out = empty(answered=bool(species or genus))
    for rank, answers, taxa, truth, match in (
            ("species", species, species_taxa, find.species, key.species_levels),
            ("genus", genus, genus_taxa, find.genus,
             lambda a, t: key.genus_levels(a, t, find.species or None))):
        levels = LEVELS[rank]
        if not truth or (rank == "species" and not species_scorable):
            for lv in levels:
                for k in K:
                    out[f"{rank}_{lv}@{k}"] = None
            continue
        judge = (key.species if rank == "species" else
                 lambda a, t, x=None: key.genus(a, t, x, find.species or None))
        first = {lv: None for lv in levels}
        for i, answer in enumerate(answers[:max(K)]):
            taxon = taxa[i] if taxa and i < len(taxa) else None
            m = match(answer, truth)
            m["strict"] = judge(answer, truth, taxon) or (
                rank == "species" and same is not None and bool(answer)
                and same(answer, truth) and judge(truth, truth))
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


def inat_outcome(line: dict, find: Find, score: str, key: AnswerKey | None = None) -> dict:
    """iNat CV's outcome on a find from an E14 finds.jsonl line (`score`: "vision", photos
    only, or "combined", with the public point and date), judged by the answer key from
    iNat's own ranked names and taxon ids. The lane's taxon-id hits are a diagnostic only
    (inat_hits_disagree). Species cells are None unless the truth is a formal species."""
    ranked = line[score]
    names = {rank: [r[1] for r in ranked.get(rank) or []] for rank in ("species", "genus")}
    taxa = {rank: [r[0] for r in ranked.get(rank) or []] for rank in ("species", "genus")}
    formal = bool(line["truth"].get("formal_species"))
    out = from_ranked(names["species"], names["genus"], find, species_scorable=formal,
                      key=key, species_taxa=taxa["species"], genus_taxa=taxa["genus"])
    answered = line.get("answered") or {}
    # combined_raw is the same answer as combined, on iNat's raw place-aware scale.
    out["answered"] = bool(answered.get(score, answered.get(score.split("_")[0])))
    return out


def inat_hits_disagree(line: dict, outcome: dict, score: str) -> dict:
    """Per rank, whether the lane's taxon-id top-1 hit disagrees with the answer key's
    verdict (a diagnostic, never scored)."""
    hits = line.get(f"{score}_hits") or line.get(f"{score.split('_')[0]}_hits") or {}
    out = {}
    for rank in ("species", "genus"):
        h, v = hits.get(rank), outcome.get(f"{rank}_strict@1")
        if h is not None and v is not None:
            out[rank] = bool(h.get("top1")) != bool(v)
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


def _taxon(entry: dict | None, rank: str):
    """(name, iNat taxon id) of an identification at `rank`, or None (no ID at that rank).
    Refuses an entry with a verdict but no name: the answer key judges names."""
    if not entry:
        return None
    t = entry.get(f"{rank}_taxon")
    if t is None:
        if entry.get(f"{rank}_right") is not None or entry.get(f"at_{rank}"):
            raise ValueError(f"an E31 identification at {rank} carries no taxon name: the "
                             "answer key judges names, so the lane's file needs "
                             f"'{rank}_taxon' ({{id, name}}) on every ID")
        return None
    return t.get("name") or "", t.get("id")


def expert_outcome(line: dict, tier: str = "1", view: str = "first", strict: bool = False,
                   find: Find | None = None, key: AnswerKey | None = None) -> dict | None:
    """The experts' outcome on `find` from an E31 scores.jsonl line, judged by the answer
    key on each identification's taxon name (and iNat taxon id), or None when the line is
    not scored (status not ok; with `strict`, the observer had named a species before the
    cutoff). `tier` is "1", "2" or pooled "1+2". `first` (primary, one opinion per find):
    at species, the earliest before-cutoff species-level ID by an identifier of `tier` (by
    made_at); at genus, the earliest ID that can be judged at genus. `best`: any identifier
    of `tier` with a before-cutoff ID right at that rank. Either is None at a rank no such
    identifier reached (stopping at genus is coverage, not error). `community`: the
    community taxon as it stood at the cutoff. Experts give one name, so only the @1 cells
    are set; s.l. and complex equal strict. The lane's own species_right / genus_right are
    not used (expert_verdicts_disagree compares them)."""
    if view not in EXPERT_VIEWS:
        raise ValueError(f"expert view is one of {EXPERT_VIEWS}")
    if find is None:
        raise ValueError("expert_outcome judges names against the find's truth: pass find")
    if line.get("status") != "ok":
        return None
    if strict and line.get("observer_named_species_before_cutoff"):
        return None
    key = key or AnswerKey()
    out = {c: None for c in CELLS}

    def right(entry, rank):
        t = _taxon(entry, rank)
        truth = find.species if rank == "species" else find.genus
        if t is None or not truth:
            return None
        if rank == "species":
            return key.species(t[0], truth, t[1])
        return key.genus(t[0], truth, t[1], find.species or None)

    if view == "community":
        com = line.get("community") or {}
        out["answered"] = bool(com.get("present"))
        sp = right(com, "species") if com.get("present") else None
        ge = right(com, "genus") if com.get("present") else None
    else:
        tiers = set(str(tier).split("+"))     # "1+2": every identifier with >= 1,000 IDs
        ids = [i for i in line.get("ids") or [] if i.get("role") == "identifier"
               and i.get("before_cutoff") and str(i.get("tier")) in tiers]
        out["answered"] = bool(ids)
        ids = sorted(ids, key=lambda i: str(i.get("made_at") or ""))
        sp_ids = [v for v in (right(i, "species") for i in ids) if v is not None]
        ge_ids = [v for v in (right(i, "genus") for i in ids) if v is not None]
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


def expert_verdicts_disagree(line: dict, outcome: dict | None, tier: str, view: str) -> dict:
    """Per rank, whether the lane's own verdict for the same view (its first ID's
    species_right / genus_right, or the community's) disagrees with the answer key's (a
    diagnostic, never scored)."""
    if not outcome:
        return {}
    if view == "community":
        src = line.get("community") or {}
    else:
        tiers = set(str(tier).split("+"))
        ids = sorted([i for i in line.get("ids") or [] if i.get("role") == "identifier"
                      and i.get("before_cutoff") and str(i.get("tier")) in tiers],
                     key=lambda i: str(i.get("made_at") or ""))
        src = {}
        for rank in ("species", "genus"):
            v = [i.get(f"{rank}_right") for i in ids if i.get(f"{rank}_right") is not None]
            if v:
                src[f"{rank}_right"] = v[0] if view == "first" else any(v)
    out = {}
    for rank in ("species", "genus"):
        lane, ours = src.get(f"{rank}_right"), outcome.get(f"{rank}_strict@1")
        if lane is not None and ours is not None:
            out[rank] = bool(lane) != bool(ours)
    return out


def published_outcome(line: dict, find: Find, same=None, collapse: bool = False,
                      key: AnswerKey | None = None) -> dict:
    """A published classifier's outcome from `mv external run`'s <model>.jsonl (its answer
    already in Vision's names, published.vision_names); `same`: the crosswalk rule; judged
    by the answer key."""
    res = line["result"]
    return from_ranked([a["name"] for a in res.get("species") or []],
                       [a["name"] for a in res.get("genus") or []], find, same=same,
                       collapse=collapse, key=key)


E31_FORMAT = 2         # from 2 on, every identification carries its taxon names and ids
                       # (3: the lane's verdicts by the shared key too; a superset)


def check_format(lines: list[dict], what: str) -> None:
    """Refuse a file that is not of the agreed shape: E14 format 1; E31 format 2, whose
    identifications carry their taxon names and ids (format 1 had verdicts only)."""
    need = {"e14": ("record_key", "truth", "vision", "combined", "vision_hits",
                    "combined_hits", "answered", "photo_ids"),
            "e31": ("record_key", "status", "ids", "community")}[what]
    for i, line in enumerate(lines[:50]):
        missing = [f for f in need if f not in line]
        if missing:
            raise ValueError(f"{what} line {i + 1} lacks {missing}: not format {FORMAT}")
        if what == "e31" and line.get("status") == "ok" and not (
                isinstance(line.get("format"), int) and line["format"] >= E31_FORMAT):
            raise ValueError(f"e31 line {i + 1} is format {line.get('format')!r}, not "
                             f"{E31_FORMAT} or later: the answer key judges each "
                             "identification's taxon name, which format 1 lines do not carry")
