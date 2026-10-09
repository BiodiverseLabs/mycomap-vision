"""Scores for a held-out benchmark (heldout.py): the numbers that go in the paper.

For each model (a backbone and method, or iNat's computer vision) on the records of
one split that it answered and that have an answer key:

- top-1 and top-5 at species, genus and family, with Wilson 95% intervals (and for
  top-1 a 95% interval from a bootstrap over observers, whose records are not
  independent);
- the same on the records every model answered, and McNemar's paired test between
  each two models on the records both answered (exact binomial up to 1,000
  discordant pairs, then the chi-square with continuity correction);
- calibration: the top answer's stated confidence in tenths against how often it
  was right, with the expected calibration error;
- breakdowns: how many reference records the true species and genus have (0, 1-4,
  5-19, 20-99, 100+), whether the true species is in the reference at all, how many
  photos the record has, east or west of -100 degrees longitude, whether the
  reference holds a record by the same iNat observer on the same day, whether the
  true name is formal, provisional (a temporary code) or one word, whether Vision held
  the record before the freeze (was_reference), and near-duplicates: a photo whose
  nearest reference photo has cosine >= 0.99, or a photo file identical (sha256) to a
  reference record's copy;
- likely sets, when the stored identify results carry them (feat/likely-sets):
  per rank, how often the true name is in the set and the set's mean size;
- a label audit: the answer key's categories and sources (heldout.answer_key), the
  records whose title is one word (name may be stale: a .com refresh), those whose
  index name is in another genus than the title, and label hygiene: answer names
  that match a known name only once writing is set aside (normalised), and names
  Vision doesn't know whose genus is unknown or that sit close to a known name
  (possible typos: listed, never fixed).

Names are judged by Vision's labels (heldout.Labeller). iNat's answers are judged in
iNat's taxonomy by taxon id where the answer key was found on iNat
(inat_cv.resolve_truth), and by name otherwise.

The dev split is for tuning and exploring. On a sealed benchmark (heldout_sets.sealed)
a report on the test split is the paper's number: it says so loudly and is recorded in
heldout_test_looks each time. A development benchmark's test split is not sealed.

A model is a backbone, a method, the place it was given (for a method that uses one)
and the photo size. Each Vision model is scored on its newest reference, or on the one
`reference_hash` names (a "before" run after a relabel); when its references disagree
and the newest covers under 90% of the records another covers, the report refuses to
choose for you.

The JSON and the per-record CSV carry no coordinates: only east or west of -100.
"""

from __future__ import annotations

import csv
import difflib
import itertools
import json
import math
import re
import sqlite3

import numpy as np
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Iterable

from . import config, heldout, metrics, names
from .heldout import (NO_ANSWER, RANKS, SOURCE_OTHER_GENUS, SOURCES, TRUTH_ONE_WORD,
                      TRUTH_STATUSES, Labeller, Truth)

INAT = "external:inat-cv"
EXTERNAL = "external:"          # an outside model (iNat, external.py): no Vision reference
DEPTH_BUCKETS = [(0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 99, "20-99"),
                 (100, 10**9, "100+")]
PHOTO_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4-5"), (6, 10**9, "6+")]
EAST_WEST_LONGITUDE = -100.0
CONFIDENCE_BINS = 10
NEAR_DUPLICATE = 0.99              # cosine to the nearest reference photo
REFERENCE_COVERAGE = 0.9           # the newest reference must cover this share of another's
BOOTSTRAP_REPS = 1000
EXACT_MCNEMAR_UP_TO = 1000
EPITHET = re.compile(r"[a-z][a-z-]+\Z")
BREAKDOWNS = ("species reference records", "genus reference records",
              "true species in reference", "photos", "east or west of -100",
              "same observer and day in reference", "name kind",
              "known to Vision before the freeze", "nearest reference photo cosine >= 0.99",
              "photo file identical to a reference copy")


# --- statistics ------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    """Wilson score 95% interval for k successes in n."""
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4))


def rate(k: int, n: int) -> dict:
    return {"n": n, "right": k, "rate": round(k / n, 4) if n else None, "ci95": wilson(k, n)}


def observer_bootstrap(hits: list[tuple[object, bool]], reps: int = BOOTSTRAP_REPS,
                       seed: int = 0) -> tuple[float, float] | None:
    """95% percentile interval of the rate, resampling observers (with all their records)
    rather than records: one observer's records share a place, a camera and a habit."""
    if not hits:
        return None
    groups: dict = defaultdict(lambda: [0, 0])
    for who, hit in hits:
        g = groups[who]
        g[0] += 1
        g[1] += bool(hit)
    n = np.array([g[0] for g in groups.values()])
    k = np.array([g[1] for g in groups.values()])
    pick = np.random.default_rng(seed).integers(0, len(n), size=(reps, len(n)))
    rates = k[pick].sum(axis=1) / n[pick].sum(axis=1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return (round(float(lo), 4), round(float(hi), 4))


def mcnemar(b: int, c: int) -> float:
    """Two-sided p-value of McNemar's test from the discordant pairs: b (only the first
    right) and c (only the second right). Exact binomial up to EXACT_MCNEMAR_UP_TO pairs."""
    n = b + c
    if n == 0:
        return 1.0
    if n <= EXACT_MCNEMAR_UP_TO:
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
        return min(1.0, float(Fraction(2 * tail, 2 ** n)))
    chi2 = (abs(b - c) - 1) ** 2 / n
    return math.erfc(math.sqrt(chi2 / 2))


def calibration(pairs: list[tuple[float, bool]], bins: int = CONFIDENCE_BINS) -> dict:
    """Stated confidence of the top answer, in equal bins, against how often it was right."""
    rows = []
    ece = 0.0
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        inside = [(c, ok) for c, ok in pairs if lo <= c < hi or (b == bins - 1 and c == 1.0)]
        if not inside:
            continue
        conf = sum(c for c, _ in inside) / len(inside)
        acc = sum(ok for _, ok in inside) / len(inside)
        ece += len(inside) / len(pairs) * abs(conf - acc)
        rows.append({"confidence": f"{lo:.1f}-{hi:.1f}", "n": len(inside),
                     "mean_confidence": round(conf, 4), "accuracy": round(acc, 4),
                     "ci95": wilson(sum(ok for _, ok in inside), len(inside))})
    return {"n": len(pairs), "bins": rows, "ece": round(ece, 4) if pairs else None}


def bucket(n: int, buckets) -> str:
    for lo, hi, label in buckets:
        if lo <= n <= hi:
            return label
    return buckets[-1][2]


# --- judging one answer -----------------------------------------------------------------

def judge(result: dict, truth: Truth, labeller: Labeller, by_taxon_id: bool) -> dict:
    """{rank: (right at 1, right in 5, top name, top confidence)} for the ranks the answer
    key has. With by_taxon_id (iNat's answers), a rank whose answer was found on iNat is
    judged by taxon id, the others by name."""
    taxa = result.get("truth_taxa") or {}
    out = {}
    for rank in RANKS:
        want = getattr(truth, rank)
        if not want:
            continue
        top = result.get(rank) or []
        if by_taxon_id and taxa.get(rank) is not None:
            hits = [c.get("taxon_id") == taxa[rank] for c in top]
        else:
            hits = [labeller.same(c["name"], want) for c in top]
        first = top[0] if top else {}
        out[rank] = (bool(hits[:1] and hits[0]), any(hits[:5]), first.get("name"),
                     first.get("confidence"))
    return out


# --- what a record is -------------------------------------------------------------------

def name_kind(truth_name: str, truth: Truth) -> str:
    if not truth.species:
        return "one word"
    return "provisional" if names.parse_name(truth_name).code else "formal"


def features(rec: heldout.HeldOutRecord, truth: Truth, reference: dict | None,
             nearest: float | None = None, identical: bool | None = None) -> dict:
    """The breakdowns a record falls into (BREAKDOWNS). `nearest`: its best cosine to a
    reference photo; `identical`: a photo file the same as a reference copy. No
    coordinate leaves here."""
    counts = (reference or {}).get("rank_counts") or {}
    days = (reference or {}).get("observer_day_set")
    sp = counts.get("species", {}).get(truth.species, 0) if truth.species else None
    ge = counts.get("genus", {}).get(truth.genus, 0) if truth.genus else None
    lng = rec.org_longitude if rec.org_longitude is not None else rec.inat_longitude
    if days is None or rec.user_id is None or not rec.observed_on:
        same = "unknown"
    else:
        same = "yes" if f"{rec.user_id}|{rec.observed_on}" in days else "no"
    known = reference is not None
    return {
        "species reference records": bucket(sp, DEPTH_BUCKETS) if known and sp is not None
        else "n/a",
        "genus reference records": bucket(ge, DEPTH_BUCKETS) if known and ge is not None
        else "n/a",
        "true species in reference": ("n/a" if sp is None or not known
                                      else "seen" if sp else "unseen"),
        "photos": bucket(len(rec.photos), PHOTO_BUCKETS) if rec.photos else "0",
        "east or west of -100": ("unknown" if lng is None
                                 else "west" if lng < EAST_WEST_LONGITUDE else "east"),
        "same observer and day in reference": same,
        "name kind": name_kind(rec.truth_name or "", truth),
        "known to Vision before the freeze": "yes" if rec.was_reference else "no",
        "nearest reference photo cosine >= 0.99": ("unknown" if nearest is None else
                                                   "yes" if nearest >= NEAR_DUPLICATE else "no"),
        "photo file identical to a reference copy": ("unknown" if identical is None else
                                                     "yes" if identical else "no"),
    }


# --- label hygiene ---------------------------------------------------------------------------

def label_hygiene(records: list[heldout.HeldOutRecord], truths: dict[str, Truth],
                  labeller: Labeller) -> dict:
    """Answer names Vision doesn't know as written: normalised (a known label once
    writing is set aside: a code's format, a missing `var.`) or listed for a person (an
    unknown genus, or close to a known name: a possible typo)."""
    by_genus: dict[str, list[str]] = defaultdict(list)
    for lab in labeller.known:
        words = lab.split()
        if len(words) > 1:
            by_genus[words[0]].append(lab)
    added: Counter = Counter()
    unknown_genus: dict[str, dict] = {}
    near: dict[str, dict] = {}
    missing_rank_word: Counter = Counter()
    for rec in records:
        t = truths.get(rec.observation_id)
        if t is None:
            continue
        if t.normalised:
            added[(rec.truth_name, t.label)] += 1
            continue
        if t.label in labeller.known:
            continue
        genus = t.label.split()[0] if t.label else ""
        if genus and genus not in labeller.known_genera:
            g = unknown_genus.setdefault(genus, {"genus": genus, "records": 0, "names": set(),
                                                 "closest_known_genus": None})
            g["records"] += 1
            g["names"].add(t.label)
            close = difflib.get_close_matches(genus, labeller.known_genera, n=1, cutoff=0.8)
            g["closest_known_genus"] = close[0] if close else None
            continue
        # A temporary code ('Candolleomyces sp. 'FL05'') is its own name: never a missing
        # rank word, and close to another code by design, not by a typo.
        if t.species and not names.parse_name(t.label).code:
            words = t.label.split()
            if (len(words) == 3 and all(EPITHET.match(w) for w in words[1:])
                    and words[2] not in heldout.RANK_WORDS):
                missing_rank_word[t.label] += 1
            close = difflib.get_close_matches(t.label, by_genus.get(genus, []), n=1, cutoff=0.85)
            if close:
                n = near.setdefault(t.label, {"name": t.label, "records": 0,
                                              "closest_known_name": close[0]})
                n["records"] += 1
    return {
        "normalised": [{"answer_key": k, "label": v, "records": n}
                       for (k, v), n in added.most_common()],
        "unknown_genus": [{**g, "names": sorted(g["names"])}
                          for g in sorted(unknown_genus.values(), key=lambda g: -g["records"])],
        "close_to_a_known_name": sorted(near.values(), key=lambda n: -n["records"]),
        "rank_word_missing_no_known_form": [{"name": k, "records": n}
                                            for k, n in missing_rank_word.most_common()],
    }


def label_audit(records: list[heldout.HeldOutRecord]) -> dict:
    status = Counter(r.truth_status for r in records)
    source = Counter(r.name_source for r in records)
    return {
        "records": len(records),
        "categories": {s: status[s] for s in TRUTH_STATUSES if status[s]},
        "answer_source": {s: source[s] for s in SOURCES if source[s]},
        "without_an_answer": sum(status[s] for s in NO_ANSWER),
        "name_may_be_stale_needs_com_refresh": [
            {"observation_id": r.observation_id, "name": r.truth_name}
            for r in records if r.truth_status == TRUTH_ONE_WORD],
        "index_name_in_another_genus": [
            {"observation_id": r.observation_id, "index_name": r.index_name,
             "title": r.truth_name}
            for r in records if r.name_source == SOURCE_OTHER_GENUS],
    }


def likely_metrics(answers: dict[str, dict], truths: dict[str, Truth],
                   labeller: Labeller) -> dict | None:
    """Per rank, from stored identify results that carry likely sets (result["likely"]
    [rank]["names"]): how often the true name is in the set, and its mean size, in the
    shape of likely.set_metrics. None when no result carries them."""
    tally = {rank: [0, 0, 0] for rank in RANKS}         # n, covered, total size
    for oid, result in answers.items():
        sets = (result or {}).get("likely")
        if not sets or oid not in truths:
            continue
        for rank in RANKS:
            want = getattr(truths[oid], rank)
            entry = sets.get(rank)
            if not want or not entry:
                continue
            got = [c.get("name") for c in entry.get("names") or []]
            tally[rank][0] += 1
            tally[rank][1] += any(labeller.same(n, want) for n in got)
            tally[rank][2] += len(got)
    out = {rank: {"n": n, "coverage": round(k / n, 4), "mean_size": round(s / n, 2)}
           for rank, (n, k, s) in tally.items() if n}
    return out or None


# --- the report ------------------------------------------------------------------------------

def name_equivalence(judged: dict, truths: dict[str, Truth]) -> dict | None:
    """BETA, beside strict and never in its place: each model's top answer judged by
    name_equiv (feat/name-equivalence) at species (strict, s.l., complex) and genus
    (strict, s.l.). None while that module isn't merged."""
    try:
        from . import name_equiv
    except ImportError:
        return None
    out = {}
    for m, recs in judged.items():
        tally = {"species": Counter(), "genus": Counter()}
        for oid, ranks in recs.items():
            for rank, match in (("species", name_equiv.species_match),
                                ("genus", name_equiv.genus_match)):
                if rank not in ranks:
                    continue
                tally[rank]["n"] += 1
                for kind, ok in match(ranks[rank][2] or "", getattr(truths[oid], rank)).items():
                    tally[rank][kind] += bool(ok)
        out[model_name(m)] = {rank: {kind: rate(c[kind], c["n"]) for kind in c if kind != "n"}
                              for rank, c in tally.items() if c["n"]}
    return out


def chosen_references(conn: sqlite3.Connection, name: str, ids: set[str],
                      reference_hash: str | None = None) -> dict[tuple, str]:
    """The reference each Vision model (backbone, method, place, size) is scored on:
    `reference_hash` for the models that ran against it, else each model's newest.
    Refuses to pick the newest when it covers under REFERENCE_COVERAGE of the records
    another reference of the model covers (an unfinished rerun would hide the rest)."""
    covered: dict[tuple, Counter] = defaultdict(Counter)
    for oid, backbone, method, place, size, ref_hash in conn.execute(
            "select observation_id, backbone, method, place, size, reference_hash "
            "from heldout_predictions where benchmark = ? and backbone != ?", (name, INAT)):
        if oid in ids:
            covered[(backbone, method, place, size)][ref_hash] += 1
    chosen: dict[tuple, str] = {}
    for backbone, method, place, size, ref_hash in conn.execute(
            "select backbone, method, place, size, reference_hash from heldout_runs "
            "where benchmark = ? order by created_at", (name,)):
        if reference_hash is None or ref_hash == reference_hash:
            chosen[(backbone, method, place, size)] = ref_hash
    if reference_hash is None:
        for key, ref_hash in chosen.items():
            mine = covered[key][ref_hash]
            others = {h: n for h, n in covered[key].items() if h != ref_hash}
            best = max(others.items(), key=lambda kv: kv[1], default=(None, 0))
            if best[1] and mine < REFERENCE_COVERAGE * best[1]:
                raise ValueError(
                    f"{model_name(key)}: its newest reference {ref_hash} answered {mine:,} of "
                    f"these records, reference {best[0]} answered {best[1]:,}; name the one "
                    "to score with --reference-hash (or finish the newer run)")
    return chosen


def load_predictions(conn: sqlite3.Connection, name: str, ids: set[str],
                     chosen: dict[tuple, str]) -> tuple[dict, dict, dict]:
    """({model: {id: result}} with each Vision model's answers from its chosen reference
    only, how many answers from other references were left out per model, and {model:
    {id: nearest reference photo cosine}})."""
    preds: dict[tuple, dict] = defaultdict(dict)
    nearest: dict[tuple, dict] = defaultdict(dict)
    other = Counter()
    for oid, backbone, method, place, size, ref_hash, sim, result in conn.execute(
            "select observation_id, backbone, method, place, size, reference_hash, "
            "max_similarity, result_json from heldout_predictions where benchmark = ?",
            (name,)):
        if oid not in ids:
            continue
        key = (backbone, method, place, size)
        if backbone != INAT and ref_hash != chosen.get(key):
            other[model_name(key)] += 1
            continue
        preds[key][oid] = json.loads(result)
        if sim is not None:
            nearest[key][oid] = sim
    return dict(preds), dict(other), dict(nearest)


def load_answers(conn: sqlite3.Connection, name: str, key: tuple, ref_hash: str,
                 ids: set[str]) -> dict[str, dict]:
    """The full identify results a Vision model stored (heldout.answers_db)."""
    import gzip
    adb = heldout.answers_db(conn, name)
    try:
        return {oid: json.loads(gzip.decompress(blob)) for oid, blob in adb.execute(
            "select observation_id, result_gz from answers where backbone = ? and method = ? "
            "and place = ? and size = ? and reference_hash = ?", (*key, ref_hash))
            if oid in ids}
    finally:
        adb.close()


def identical_to_reference(conn: sqlite3.Connection, name: str) -> set[str]:
    """The set's records with a photo file identical (sha256) to a copy of a reference
    record's photo: the same picture under another observation."""
    reference = {r[0] for r in conn.execute(
        "select distinct c.sha256 from photo_copies c join observation_photos op "
        "on op.photo_id = c.photo_id join records r on r.observation_id = op.observation_id "
        "where c.sha256 is not null")}
    return {oid for oid, sha in conn.execute(
        "select observation_id, sha256 from heldout_photos where benchmark = ? and "
        "status = 'done' and sha256 is not null", (name,)) if sha in reference}


def reference_run(conn: sqlite3.Connection, name: str, backbone: str | None,
                  chosen: dict[tuple, str]) -> dict | None:
    """The reference summary (heldout.reference_summary) the breakdowns use: of the chosen
    reference of `backbone`, or of the newest chosen run of any backbone."""
    # An external model (external.py) answers from no reference of Vision's: its run row
    # holds a checkpoint, never the reference counts the breakdowns need.
    hashes = [h for (b, *_k), h in chosen.items()
              if (backbone is None or b == backbone) and not b.startswith(EXTERNAL)]
    if not hashes:
        return None
    row = conn.execute(
        "select backbone, reference_hash, reference_json from heldout_runs where benchmark = ? "
        f"and reference_hash in ({','.join('?' * len(hashes))}) "
        "and (? is null or backbone = ?) and backbone not like 'external:%' "
        "order by created_at desc limit 1",
        (name, *hashes, backbone, backbone)).fetchone()
    if not row:
        return None
    ref = json.loads(row[2])
    ref["backbone"], ref["hash"] = row[0], row[1]
    ref["observer_day_set"] = set(ref.pop("observer_days", []))
    return ref


def score(models: dict, truths: dict[str, Truth], labeller: Labeller) -> dict:
    """{model: {id: judge(...)}} for records with an answer key."""
    out = {}
    for key, results in models.items():
        out[key] = {oid: judge(res, truths[oid], labeller, by_taxon_id=key[0] == INAT)
                    for oid, res in results.items() if oid in truths}
    return out


def summary(judged: dict[str, dict], ids: Iterable[str] | None = None,
            observers: dict[str, object] | None = None) -> dict:
    """Per rank: top-1 and top-5 with Wilson intervals, on `ids` (default: all judged).
    With `observers` ({id: observer}), top-1 also gets an observer bootstrap interval."""
    want = set(ids) if ids is not None else None
    tally = {rank: Counter() for rank in RANKS}
    hits: dict[str, list] = defaultdict(list)
    for oid, ranks in judged.items():
        if want is not None and oid not in want:
            continue
        for rank, (hit1, hit5, _name, _conf) in ranks.items():
            tally[rank]["n"] += 1
            tally[rank]["top1"] += hit1
            tally[rank]["top5"] += hit5
            if observers is not None:
                hits[rank].append((observers.get(oid) or f"record {oid}", hit1))
    out = {}
    for rank, c in tally.items():
        if not c["n"]:
            continue
        out[rank] = {"top1": rate(c["top1"], c["n"]), "top5": rate(c["top5"], c["n"])}
        if observers is not None:
            out[rank]["top1"]["ci95_observer_bootstrap"] = observer_bootstrap(hits[rank])
    return out


def species_macro_f1(judged_m: dict, truths: dict[str, Truth], labeller: Labeller) -> dict:
    """Species macro-F1 of a model's top-1 answers (metrics.macro_f1; the Picek group's
    headline number) over the records with a species answer key. An answer the strict
    judgement counts as right takes the truth's label; others their own label."""
    pairs = []
    for oid, ranks in judged_m.items():
        if "species" not in ranks:
            continue
        hit1, _h5, name, _conf = ranks["species"]
        want = truths[oid].species
        pairs.append((want, want if hit1 else labeller.label(name) or ""))
    f1 = metrics.macro_f1(*zip(*pairs)) if pairs else None
    return {"n": len(pairs), "macro_f1": None if f1 is None else round(f1, 4)}


def per_image(answers: dict[str, dict], truths: dict[str, Truth], labeller: Labeller) -> dict:
    """Each photo's own answer (the identify result's per_photo), judged like the record's:
    top-1 per rank over photos, and species macro-F1 over photos. Next to the record-level
    numbers this is the per-image / per-observation pair the Picek group reports."""
    tally = Counter()
    pairs = []
    for oid, result in answers.items():
        truth = truths.get(oid)
        if truth is None:
            continue
        for photo in result.get("per_photo") or []:
            for rank in ("species", "genus", "family"):
                want = getattr(truth, rank)
                if not want:
                    continue
                top = (photo.get("ranks") or {}).get(rank) or []
                name = top[0]["name"] if top else ""
                hit = bool(name) and labeller.same(name, want)
                tally[f"{rank}_n"] += 1
                tally[f"{rank}_top1"] += hit
                if rank == "species":
                    pairs.append((want, want if hit else labeller.label(name) or ""))
    out = {rank: rate(tally[f"{rank}_top1"], tally[f"{rank}_n"])
           for rank in ("species", "genus", "family") if tally[f"{rank}_n"]}
    f1 = metrics.macro_f1(*zip(*pairs)) if pairs else None
    out["species_macro_f1"] = None if f1 is None else round(f1, 4)
    return out


def paired(judged: dict, a: tuple, b: tuple) -> dict:
    out = {}
    for rank in RANKS:
        both = [oid for oid in judged[a] if oid in judged[b]
                and rank in judged[a][oid] and rank in judged[b][oid]]
        if not both:
            continue
        ra = [judged[a][o][rank][0] for o in both]
        rb = [judged[b][o][rank][0] for o in both]
        only_a = sum(x and not y for x, y in zip(ra, rb))
        only_b = sum(y and not x for x, y in zip(ra, rb))
        out[rank] = {"n": len(both), "a_top1": round(sum(ra) / len(both), 4),
                     "b_top1": round(sum(rb) / len(both), 4), "only_a_right": only_a,
                     "only_b_right": only_b, "mcnemar_p": round(mcnemar(only_a, only_b), 6)}
    return out


def model_name(key: tuple) -> str:
    """'backbone/method', '@place' for a method given one, '[size]' unless large."""
    backbone, method, place, size = (tuple(key) + ("", ""))[:4]
    out = f"{backbone}/{method}"
    if backbone != INAT and place not in ("", heldout.NO_PLACE):
        out += f"@{place}"
    if size not in ("", "large"):
        out += f"[{size}]"
    return out


def report(conn: sqlite3.Connection, name: str, split: str | None = "dev",
           subset: Iterable[str] | None = None, reference_backbone: str | None = None,
           reference_hash: str | None = None, out_dir: Path | None = None,
           stamp: str | None = None, log=print) -> dict:
    """Score the set's `split` (dev unless told otherwise), optionally only `subset`.
    Writes report-<split>-<stamp>.json and .csv; a report on a sealed benchmark's test
    split is recorded as a look."""
    heldout.ensure_schema(conn)
    splits = heldout.splits_of(conn, name)
    if splits and split not in splits:
        raise ValueError(f"name the split to score: {' or '.join(splits)} (dev to tune and "
                         "explore; test is the sealed paper number)")
    if not splits:
        split = None
    ids = heldout.benchmark_ids(conn, name, subset, split=split)
    records = heldout.load_benchmark(conn, name, ids)
    state = heldout.set_state(conn, name)
    chosen = chosen_references(conn, name, set(ids), reference_hash)
    if reference_hash and not chosen:
        raise ValueError(f"no run of {name} against reference {reference_hash}")
    preds, stale, nearest = load_predictions(conn, name, set(ids), chosen)
    extra = [r.truth_name for r in records if r.truth_name]
    extra += [c["name"] for key, by in preds.items() if key[0] != INAT
              for res in by.values() for rank in RANKS for c in res.get(rank) or []]
    labeller = Labeller(conn, extra=extra)
    truths_all = {r.observation_id: labeller.truth(r.truth_name) for r in records if r.truth_name}
    guests_left_out = sorted(o for o, t in truths_all.items() if t.guest)
    truths = {o: t for o, t in truths_all.items() if not t.guest}
    ref = reference_run(conn, name, reference_backbone, chosen)
    near: dict[str, float] = {}             # from the reference backbone's own answers
    for key, sims in nearest.items():
        if ref and key[0] == ref["backbone"]:
            for oid, s in sims.items():
                near[oid] = max(s, near.get(oid, s))
    identical = identical_to_reference(conn, name)
    fetched = {r.observation_id for r in records if r.photos}
    feats = {r.observation_id: features(
        r, truths[r.observation_id], ref, near.get(r.observation_id),
        (r.observation_id in identical) if r.observation_id in fetched else None)
        for r in records if r.observation_id in truths}
    observers = {r.observation_id: r.user_id for r in records}
    judged = score(preds, truths, labeller)
    models = sorted(judged)
    common = set.intersection(*(set(judged[m]) for m in models)) if models else set()

    out: dict = {
        "benchmark": name, "split": split, **state, "records": len(records),
        "with_an_answer": len(truths_all), "guests_left_out": len(guests_left_out),
        "scored_records": len(truths), "code_version": config.code_version(),
        "reference": ({k: ref[k] for k in ("backbone", "hash", "records", "photos", "species")}
                      if ref else None),
        "references": {model_name(k): h for k, h in sorted(chosen.items())},
        "other_reference_answers_left_out": stale,
        "models": {}, "on_records_every_model_answered": {}, "paired": [],
        "calibration": {}, "likely_sets": {}, "breakdowns": {},
        "label_audit": label_audit(records),
        "label_hygiene": label_hygiene(records, truths, labeller),
    }
    full: dict[tuple, dict] = {}            # the stored identify results, per Vision model
    for m in models:
        out["models"][model_name(m)] = {"records": len(judged[m]),
                                        **summary(judged[m], observers=observers),
                                        "species_macro_f1": species_macro_f1(judged[m], truths,
                                                                             labeller)}
        if len(models) > 1:
            out["on_records_every_model_answered"][model_name(m)] = summary(judged[m], common)
        out["calibration"][model_name(m)] = {
            rank: calibration([(conf, hit1) for ranks in judged[m].values()
                               for r, (hit1, _h5, _n, conf) in ranks.items()
                               if r == rank and conf is not None])
            for rank in ("species", "genus")}
        by_dim = {}
        for dim in BREAKDOWNS:
            groups: dict[str, dict] = defaultdict(dict)
            for oid, ranks in judged[m].items():
                groups[feats[oid][dim]][oid] = ranks
            by_dim[dim] = {value: {rank: s["top1"] for rank, s in summary(g).items()
                                   if rank in ("species", "genus")}
                           for value, g in sorted(groups.items())}
        out["breakdowns"][model_name(m)] = by_dim
        if m in chosen:
            full[m] = load_answers(conn, name, m, chosen[m], set(judged[m]))
            out["models"][model_name(m)]["per_image"] = per_image(full[m], truths, labeller)
            likely = likely_metrics(full[m], truths, labeller)
            if likely:
                out["likely_sets"][model_name(m)] = likely
    out["on_records_every_model_answered"] = (
        {"records": len(common), **out["on_records_every_model_answered"]}
        if len(models) > 1 else None)
    for a, b in itertools.combinations(models, 2):
        out["paired"].append({"a": model_name(a), "b": model_name(b), **paired(judged, a, b)})
    out["beta_name_equivalence"] = name_equivalence(judged, truths)
    from .heldout_summary import standard_summary
    out = {"summary": standard_summary(
        {k: out[k] for k in ("benchmark", "split", "sealed", "released_at", "records",
                             "scored_records")}, preds, judged, truths, feats, labeller, full),
        **out}

    stamp = stamp or heldout.now_iso().replace(":", "").replace("-", "")[:15]
    out_dir = out_dir or heldout.bench_dir(conn, name) / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    base = f"report-{split or 'all'}-{stamp}"
    csv_path = write_csv(out_dir / f"{base}.csv", records, truths, feats, judged)
    json_path = out_dir / f"{base}.json"
    out["files"] = {"json": str(json_path), "csv": str(csv_path)}
    if split == "test" and out["sealed"]:
        out["test_look"] = record_test_look(conn, name, models, len(truths), str(json_path))
        log(f"*** SEALED TEST SPLIT of {name}: look number {out['test_look']} at test, recorded "
            "in heldout_test_looks. This is the paper's number: do not tune on it. ***")
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def record_test_look(conn: sqlite3.Connection, name: str, models: list, records: int,
                     path: str) -> int:
    with conn:
        conn.execute("insert into heldout_test_looks values (?, ?, ?, ?, ?, ?)",
                     (name, heldout.now_iso(), config.code_version(),
                      json.dumps([model_name(m) for m in models]), records, path))
    return conn.execute("select count(*) from heldout_test_looks where benchmark = ?",
                        (name,)).fetchone()[0]


CSV_FIXED = ["observation_id", "split", "model", "answer_key", "label", "answer_key_status",
             "answer_source", *BREAKDOWNS]


def write_csv(path: Path, records: list[heldout.HeldOutRecord], truths: dict[str, Truth],
              feats: dict[str, dict], judged: dict) -> Path:
    """One row per record and model: the answer key, the breakdowns, and per rank the
    top answer, its confidence and whether it was right at 1 and in 5."""
    cols = CSV_FIXED + [f"{rank}_{c}" for rank in RANKS
                        for c in ("truth", "top1", "confidence", "right_top1", "right_top5")]
    by_id = {r.observation_id: r for r in records}
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for m in sorted(judged):
            for oid in sorted(judged[m], key=lambda o: (len(o), o)):
                rec, t, ranks = by_id[oid], truths[oid], judged[m][oid]
                row = [oid, rec.split or "", model_name(m), rec.truth_name, t.label,
                       rec.truth_status, rec.name_source, *(feats[oid][d] for d in BREAKDOWNS)]
                for rank in RANKS:
                    if rank in ranks:
                        hit1, hit5, top, conf = ranks[rank]
                        row += [getattr(t, rank), top or "", "" if conf is None else conf,
                                int(hit1), int(hit5)]
                    else:
                        row += ["", "", "", "", ""]
                w.writerow(row)
    return path
