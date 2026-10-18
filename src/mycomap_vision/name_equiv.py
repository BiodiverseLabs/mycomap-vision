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

The answer key (Steve, 2026-10-18; V2-REQUIREMENTS 3a), applied the same way to every
comparator (Vision, iNat CV, expert IDs, published models) and to the truth:

    species rank   a variety, subspecies or form counts as its species ('Amanita muscaria
                   var. flavivolvata' is 'Amanita muscaria'); the infraspecific name is an
                   extra suggestion, never the answer key. 'group' / 'complex' names are
                   not cut: they are fixed at source.
    one writing    a name's writings are one name: rank words, code padding (IN7 / IN07),
                   hyphens and spaces in codes, hyphenated epithets (luteo-olivaceum).
    hybrid codes   a truth 'X sp. 'IN01 x VT01'' is answered right by either parent code.

answer_key gives a name's key; species_credit says whether an answer counts for a truth;
first_credited finds the first answer in a ranked list that counts (answers sharing a key
are one answer, so the list is collapsed first). Scorers call these, never their own
comparison. ANSWER_KEY names this rule in reports; "exact" (the label as written, v1's
published rule) stays available to reproduce earlier numbers.

The API is small on purpose (other scorers call it): species_match, genus_match,
genus_group, parts, gender_fold, answer_key, species_credit, first_credited, and the
optional frozen-iNat-taxonomy step: Taxonomy and canonical (a name or a taxon id on either side). Steve, 2026-10-09: "we'll have to think about this
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


ANSWER_KEY = "species-rank-2026-10-18"      # the rule below; "exact" = v1's published rule
ANSWER_KEYS = (ANSWER_KEY, "exact")
_RANK_WORD = re.compile(r"\s+(?:var\.?|subsp\.?|ssp\.?|f\.|forma)\s+", re.IGNORECASE)
_NOT_INFRA = {"group", "complex", "agg.", "s.l.", "s.str.", "sensu", "aff.", "cf.", "nr."}
_HYBRID = re.compile(r"^(.+?)\s*[- ]x[- ]\s*(.+)$", re.IGNORECASE)


@lru_cache(maxsize=None)
def answer_key(name: str, rule: str = ANSWER_KEY) -> str:
    """The key a species name is scored by. `rule` "exact": the name as written. The
    default: provisional codes by their spelling key (names.variant_key); formal names cut
    to genus + epithet (rank words dropped, a bare trinomial's third word dropped, a
    hyphen inside an epithet dropped), then spelling-folded. '' for no name."""
    n = (name or "").strip()
    if not n or rule == "exact":
        return n
    if rule != ANSWER_KEY:
        raise ValueError(f"unknown answer key {rule!r} (known: {ANSWER_KEYS})")
    if names.parse_name(n).code:
        return names.variant_key(n)
    words = _RANK_WORD.sub(" ", n).split()
    if (len(words) >= 3 and words[1][:1].islower() and words[1].lower() not in _QUALIFIERS
            and words[2][:1].islower() and words[2].lower() not in _NOT_INFRA
            and _EPITHET.match(words[2].lower())):
        words = words[:2]
    if len(words) >= 2 and words[1][:1].islower() and _EPITHET.match(words[1].lower()):
        words[1] = words[1].replace("-", "")
    return names.variant_key(" ".join(words))


def hybrid_parents(name: str) -> tuple[str, ...]:
    """A hybrid code's two parent codes as names ('Physisporinus sp. 'IN01 x VT01'' ->
    'Physisporinus sp. 'IN01'', 'Physisporinus sp. 'VT01''), else ()."""
    p = names.parse_name(name or "")
    if not p.code:
        return ()
    m = _HYBRID.match(p.code.code.strip())
    if not m:
        return ()
    return tuple(f"{p.code.genus} sp. '{x.strip()}'" for x in m.groups())


SPECIES_LEVEL = 10        # iNat rank_level: species 10, varieties and forms below
GENUS_LEVEL = 20


class Taxonomy:
    """A frozen iNaturalist taxonomy (the paper's snapshot), for the optional canonical step
    (the paper coordinator, 2026-10-18): truth and answer are each resolved to an iNat
    species, following a retired taxon to its single current successor, so a comparator that
    answers in iNat's newer names (Candolleomyces, Neoboletus, Calonarius) is not marked
    wrong for the taxonomy's version. One-to-one only: when two labels (different answer
    keys) resolve to the same iNat species, neither is resolved and both stay names. A name
    iNat doesn't know (a provisional code) stays its string.

    `taxa`: {id: {name, rank_level, is_active, ancestor_ids, current_synonymous_taxon_ids?}},
    e.g. expert_ids.load_open_taxa plus the frozen successors (Taxonomy.load); `labels`: the
    label vocabulary (label_vocabulary), for the one-to-one rule; `extras`: answer-side taxa
    the snapshot lacks or knows less about (a comparator's newer ids, or a retired id's
    successors from a later lookup): they add to and update the snapshot for reading an ANSWER
    only; a label and a truth's genus always read the snapshot alone, so extras never change
    how a label resolves; `cite`:
    the files and their sha256, for reports.

    resolves(label) is False for a name that does not resolve one-to-one (a code, unknown to
    iNat, or kept apart): such a truth leaves the iNat- and expert-paired tests (PLAN rule 7)
    rather than counting as their miss."""

    def __init__(self, taxa: dict[int, dict], labels=(), cite: dict | None = None,
                 extras: dict[int, dict] | None = None):
        self.base = taxa
        # Answers read the snapshot updated by the extras (a retired taxon's successors that
        # only a later lookup knows); labels read the snapshot alone (current, _level and
        # genus_names with base_only), so the extras cannot change a label's resolution.
        self.extras = extras or {}
        self.taxa = dict(taxa)                          # set_labels applies the extras
        self.cite = cite or {}
        self.by_name: dict[str, list[int]] = {}
        for tid, t in taxa.items():
            if t.get("name"):
                self.by_name.setdefault(t["name"].lower(), []).append(int(tid))
        self.ambiguous: set[int] = set()
        self.shared_by: dict[int, set[str]] = {}
        self.set_labels(labels)

    @classmethod
    def load(cls, taxa_csv_gz, labels=(), successors=None) -> "Taxonomy":
        """The paper's frozen snapshot: iNat open data's taxa.csv.gz (fungi and protozoa,
        expert_ids.load_open_taxa) plus a TSV of retired taxa's current successors
        (taxon_id, current_ids comma-separated; the open data has none), both cited by
        sha256."""
        import csv
        import hashlib
        from .expert_ids import load_open_taxa
        taxa = load_open_taxa(Path(taxa_csv_gz))
        cite = {"taxa": Path(taxa_csv_gz).name,
                "taxa_sha256": hashlib.sha256(Path(taxa_csv_gz).read_bytes()).hexdigest()}
        if successors is not None:
            with open(successors, encoding="utf-8", newline="") as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    tid = int(r["taxon_id"])
                    ids = [int(x) for x in (r["current_ids"] or "").split(",") if x.strip()]
                    if tid in taxa:
                        taxa[tid]["current_synonymous_taxon_ids"] = ids
            cite.update(successors=Path(successors).name, successors_sha256=hashlib.sha256(
                Path(successors).read_bytes()).hexdigest())
        return cls(taxa, labels, cite)

    def reach(self, labels) -> set[int]:
        """Snapshot taxa a label lookup can reach (the paper coordinator, 2026-10-18): a
        taxon named by a label, its rank-stripped form, its binomial, its genus word or any
        other capitalised word, plus those taxa's successors and ancestry. The extras never
        override these, not even for reading an answer."""
        hit = set()
        for lab in labels:
            words = _RANK_WORD.sub(" ", lab).split()
            if not words:
                continue
            forms = {lab, " ".join(words), " ".join(words[:2]), words[0]}
            forms |= {w.strip("'\"()") for w in words[1:] if w[:1].isupper()}
            for f in forms:
                hit.update(self.by_name.get(f.lower(), []))
        out, todo = set(), list(hit)
        while todo:
            tid = todo.pop()
            if tid in out or tid not in self.base:
                continue
            out.add(tid)
            t = self.base[tid]
            todo.extend(int(x) for x in t.get("current_synonymous_taxon_ids") or [])
            todo.extend(int(x) for x in t.get("ancestor_ids") or [])
        return out

    def set_labels(self, labels) -> None:
        labels = list(labels)
        reached = self.reach(labels)
        # extras add taxa and update the snapshot's entries no label reaches (a retired id an
        # identifier used, whose successor only a later lookup knows)
        self.taxa = {**self.extras, **{tid: ({**t, **self.extras[tid]}
                                             if tid in self.extras and tid not in reached else t)
                                       for tid, t in self.base.items()}}
        self.reached = reached
        claims: dict[int, set[str]] = {}
        for lab in labels:
            sid = self._resolve_name(lab)
            if sid is not None:
                claims.setdefault(sid, set()).add(answer_key(lab))
        self.shared_by = {sid: ks for sid, ks in claims.items() if len(ks) > 1}
        self.ambiguous = set(self.shared_by)

    def current(self, tid: int, base_only: bool = False) -> int | None:
        """An active taxon id: the id itself, or a retired one's single current successor
        (within the snapshot alone when base_only: how labels resolve)."""
        taxa = self.base if base_only else self.taxa
        seen = set()
        while tid is not None and tid not in seen:
            seen.add(tid)
            t = taxa.get(tid)
            if t is None:
                return None
            if t.get("is_active", True):
                return tid
            succ = t.get("current_synonymous_taxon_ids") or []
            tid = int(succ[0]) if len(succ) == 1 else None
        return None

    def _level(self, tid: int, level: int, base_only: bool = False) -> int | None:
        """The taxon at `level` (species, genus) on tid's own line, or None."""
        taxa = self.base if base_only else self.taxa
        t = taxa.get(tid) or {}
        if t.get("rank_level") == level:
            return tid
        if t.get("rank_level") is not None and t["rank_level"] > level:
            return None
        for a in reversed(t.get("ancestor_ids") or []):
            if (taxa.get(a) or {}).get("rank_level") == level:
                return a
        return None

    def _resolve_name(self, name: str) -> int | None:
        """The current iNat species of a formal name (rank words dropped; a variety or
        subspecies to its species), or None (a code, unknown, or a homonym)."""
        n = (name or "").strip()
        if not n or names.parse_name(n).code:
            return None
        words = _RANK_WORD.sub(" ", n).split()
        if any(w.lower() in _NOT_INFRA for w in words[2:]):
            return None            # a group or complex is no species: it stays a name
        for cand in (" ".join(words), " ".join(words[:2])):
            ids = self.by_name.get(cand.lower(), [])
            current = {c for c in (self.current(i, base_only=True) for i in ids) if c is not None}
            species = {s for s in (self._level(c, SPECIES_LEVEL, base_only=True)
                                   for c in current) if s}
            if len(species) == 1:
                return species.pop()
        return None

    def species_id(self, x, one_to_one: bool = True) -> int | None:
        """A name or a taxon id -> its current iNat species id; one-to-one only unless
        `one_to_one` is False (genus reads a kept-apart label's species: its genus is not in
        doubt)."""
        if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit()):
            cur = self.current(int(x))
            sid = self._level(cur, SPECIES_LEVEL) if cur is not None else None
        else:
            sid = self._resolve_name(x)
        return None if sid is None or (one_to_one and sid in self.ambiguous) else sid

    def resolves(self, label: str) -> bool:
        """Does this label resolve one-to-one to an iNat species? A truth that does not is
        left out of the iNat- and expert-paired tests (PLAN rule 7), never scored a miss."""
        return self.species_id(label) is not None

    def genus_names(self, x, base_only: bool = False) -> set[str]:
        """Genus names a name or taxon id stands for: as written, and iNat's current genus
        of it (a retired genus followed to its successor; a species' genus). A name always
        reads the snapshot alone; an id reads the extras too unless base_only (a truth)."""
        taxa = self.base if base_only else self.taxa
        out = set()
        if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit()):
            cur = self.current(int(x), base_only)
            if cur is None and int(x) in taxa:
                # retired with no single successor (a split): its own line's genus stands
                g = self._level(int(x), GENUS_LEVEL, base_only)
                if g is not None and taxa[g].get("name"):
                    out.add(taxa[g]["name"].lower())
        else:
            base_only = True
            taxa = self.base
            out.add(x.strip().split()[0].lower())
            ids = self.by_name.get(x.strip().lower(), [])     # a name: the snapshot alone
            cur_ids = {c for c in (self.current(i, base_only=True) for i in ids) if c is not None}
            cur = cur_ids.pop() if len(cur_ids) == 1 else None
        if cur is not None:
            g = self._level(cur, GENUS_LEVEL, base_only)
            if g is not None and taxa[g].get("name"):
                out.add(taxa[g]["name"].lower())
        return out

    def name_of(self, x) -> str:
        """A taxon id as its current name (a name passes through)."""
        if isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit()):
            cur = self.current(int(x))
            return (self.taxa.get(cur) or {}).get("name", "") if cur is not None else ""
        return x


def canonical(x, taxonomy: Taxonomy | None = None, rule: str = ANSWER_KEY) -> str:
    """The key a species name or iNat taxon id is scored by, for every comparator: with a
    taxonomy and rule ANSWER_KEY, its current iNat species when that is one-to-one
    ('inat:<id>'); otherwise answer_key of the name (a taxon id without a taxonomy, or one
    that does not resolve, gets 'taxon:<id>', which matches nothing else)."""
    is_id = isinstance(x, int) or (isinstance(x, str) and x.strip().isdigit())
    if taxonomy is not None and rule != "exact":
        sid = taxonomy.species_id(x)
        if sid is not None:
            return f"inat:{sid}"
    if is_id:
        return f"taxon:{int(x)}"
    return answer_key(x, rule)


def species_credit(answer, truth, rule: str = ANSWER_KEY,
                   taxonomy: Taxonomy | None = None) -> bool:
    """Does `answer` count as naming the species `truth`? The same function for every
    comparator: a model, iNat's computer vision, an expert's identification. Either side
    may be a name or (with a taxonomy) an iNat taxon id."""
    if answer in (None, "") or truth in (None, ""):
        return False
    a = canonical(answer, taxonomy, rule)
    if a == canonical(truth, taxonomy, rule):
        return True
    return rule != "exact" and isinstance(truth, str) and any(
        a == canonical(p, taxonomy, rule) for p in hybrid_parents(truth))


def collapse(answers, rule: str = ANSWER_KEY, taxonomy: Taxonomy | None = None) -> list:
    """A ranked answer list with answers that share a key kept once, at the first place."""
    seen, out = set(), []
    for a in answers:
        k = canonical(a, taxonomy, rule)
        if k and k not in seen:
            seen.add(k)
            out.append(a)
    return out


def first_credited(answers, truth, rule: str = ANSWER_KEY,
                   taxonomy: Taxonomy | None = None) -> int | None:
    """The position (0-based) of the first answer that counts for `truth` in the collapsed
    ranked list (collapse), or None. Top-k is right when this is < k."""
    for i, a in enumerate(collapse(answers, rule, taxonomy)):
        if species_credit(a, truth, rule, taxonomy):
            return i
    return None


def species_match(answer, truth, taxonomy: Taxonomy | None = None) -> dict[str, bool]:
    """{"strict", "sl", "complex"} for a species answer against the true name. Each
    implies the next (strict -> s.l. -> complex). Strict is the answer key
    (species_credit); either side may be an iNat taxon id when a taxonomy is given."""
    if answer in (None, "") or truth in (None, ""):
        return {"strict": False, "sl": False, "complex": False}
    strict = species_credit(answer, truth, taxonomy=taxonomy)
    if taxonomy is not None:
        answer, truth = taxonomy.name_of(answer), taxonomy.name_of(truth)
        if not answer or not truth:
            return {"strict": strict, "sl": strict, "complex": strict}
    a, t = parts(answer), parts(truth)
    same_group = bool(a.genus) and genus_group(a.genus) == genus_group(t.genus)
    sl = strict or (same_group and not a.provisional and not t.provisional
                    and bool(a.folded) and a.folded == t.folded)
    complex_ = sl or (same_group and a.stem is not None and a.stem == t.stem)
    return {"strict": strict, "sl": sl, "complex": complex_}


def genus_match(answer, truth, taxonomy: Taxonomy | None = None,
                truth_species: str | None = None) -> dict[str, bool]:
    """{"strict", "sl"} for a genus answer against the true genus. With a taxonomy (PLAN
    rule 4), the answer may be a name or an iNat taxon id and stands for its current genus
    too, and the truth accepts its label genus, that genus's current name, and the genus of
    its resolved species (`truth_species`): Psathyrella candolleana's truth genus is both
    Psathyrella and Candolleomyces, so neither vocabulary is marked wrong for the version."""
    if answer in (None, "") or truth in (None, ""):
        return {"strict": False, "sl": False}
    if taxonomy is None:
        strict = answer.strip().lower() == truth.strip().lower()
        return {"strict": strict, "sl": strict or genus_group(answer) == genus_group(truth)}
    got = taxonomy.genus_names(answer)
    want = taxonomy.genus_names(truth)
    if truth_species:
        # the genus word of the label as written ('Candolleomyces candolleanus' filed under
        # genus Psathyrella), and the genus of its iNat species even when the label is kept
        # apart for species (a kept-apart pair shares one iNat species, so one genus)
        want.add(truth_species.strip().split()[0].lower())
        sid = taxonomy.species_id(truth_species, one_to_one=False)
        if sid is not None:
            want |= taxonomy.genus_names(sid, base_only=True)      # a truth: snapshot alone
    strict = bool(got & want)
    sl = strict or any(genus_group(a.capitalize()) == genus_group(b.capitalize())
                       for a in got for b in want)
    return {"strict": strict, "sl": sl}


def label_vocabulary(release) -> list[str]:
    """The label vocabulary for the one-to-one rule (PLAN rule 4, research a267d31): every
    distinct label on the release's included records in train and val, all ranks, after the
    erratum the release was opened with (release.use_errata), sorted. The same list for every
    comparator."""
    return sorted({r["label"] for r in release.records(("train", "val")) if r["label"]})


def write_label_vocabulary(release, path) -> dict:
    """Write label_vocabulary as one label per line (UTF-8, LF; never overwritten) and give
    its sha256, for pinning."""
    import hashlib
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: a pinned vocabulary is never rewritten")
    labels = label_vocabulary(release)
    body = ("\n".join(labels) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    cite = release.cite()
    return {"file": path.name, "sha256": hashlib.sha256(body).hexdigest(), "labels": len(labels),
            "release": cite["dataset_release"],
            "errata": (cite.get("errata") or {}).get("file")}


def read_label_vocabulary(path) -> list[str]:
    return [x for x in Path(path).read_text(encoding="utf-8").split("\n") if x]
