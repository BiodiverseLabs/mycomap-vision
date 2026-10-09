"""Leave-one-out scoring of the reference set, to find records whose label the photos
dispute (experiment exp/loo-mislabel-scan, docs/experiments/2026-10-09-loo-mislabel-scan.md).

Every reference record is identified by nearest+mean (methods.NearestAndMean, the site's
default) against all the other reference records, as if it were a new upload. Left out
with it: every record of the same observer on the same day (a second record of one
collection is a near-duplicate, not independent evidence) and any column holding one of
its own photos (the same photo filed under two observations). What is left out is masked,
not rebuilt: the scores equal a NearestAndMean fitted without those records
(tests/test_loo.py checks that), so a whole pass over ~590k photos needs no refit.

The flagging rules (`classify`) work on per-record summaries only:

- (a) probably wrong label: the model names another species with confidence, most of the
  record's nearest records carry that name, the label's species is well supported by other
  records that are themselves recognised, and the other species is rarely mistaken for the
  label (otherwise it is (b) or (d));
- (b) the same species under two names: the pair's names are related (a writing variant,
  a provisional code beside a formal name in one genus, the s.l./complex rules of
  name_equiv) or the two labels' records are predicted as each other most of the time;
- (c) wrong photos: the record matches nothing well, or its photos disagree with each
  other about the genus;
- (d) hard look-alikes: both names well supported and confused in both directions,
  while each keeps most of its own records. Not mislabels.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

K = 2            # NearestAndMean.k
WEIGHT = 0.6     # NearestAndMean.weight


@dataclass
class Layout:
    """The reference as columns: photos sorted by unit (an evaluate.Index group: a species,
    or a one-word name), each record's photos contiguous inside its unit."""
    units: list[str]
    col_rows: np.ndarray        # (P,) rows into the vector matrix (= index.cols)
    unit_of_col: np.ndarray     # (P,)
    starts: np.ndarray          # (U,) first column of each unit
    counts: np.ndarray          # (U,) photos per unit
    rec_starts: np.ndarray      # (N,) first column of each record, records in column order
    rec_unit: np.ndarray        # (N,)
    rec_group: np.ndarray       # (N,) leave-out group (observer and day)
    group_recs: list            # group -> record indices
    same_row_cols: dict         # vector row -> columns holding it, for rows in >1 column

    @property
    def n_records(self) -> int:
        return len(self.rec_starts)

    def record_cols(self, r: int) -> np.ndarray:
        hi = self.rec_starts[r + 1] if r + 1 < len(self.rec_starts) else len(self.col_rows)
        return np.arange(self.rec_starts[r], hi)


def leave_out_key(observer: str | None, day: str | None, oid: str) -> tuple:
    """Records of one observer on one day are left out together; a record missing
    either is its own group."""
    return ("day", observer, day) if observer and day else ("record", oid)


def build_layout(records: list, index) -> tuple[Layout, list]:
    """The Layout of `records` (evaluate.Record with photo_rows) and the records in
    layout order. Checks that its columns are exactly index.cols."""
    pos = {u: i for i, u in enumerate(index.species)}
    by_unit: dict[int, list] = defaultdict(list)
    for r in records:
        by_unit[pos[r.unit]].append(r)
    ordered = [r for u in range(len(index.species)) for r in by_unit[u]]
    cols, rec_starts, rec_unit, unit_of_col = [], [], [], []
    for r in ordered:
        rec_starts.append(len(cols))
        rec_unit.append(pos[r.unit])
        cols.extend(r.photo_rows)
        unit_of_col.extend([pos[r.unit]] * len(r.photo_rows))
    col_rows = np.asarray(cols, dtype=np.int64)
    if not np.array_equal(col_rows, index.cols):
        raise ValueError("record order does not reproduce the index's columns")
    keys: dict[tuple, int] = {}
    rec_group = np.asarray([keys.setdefault(leave_out_key(r.observer, r.observed_on,
                                                          r.observation_id), len(keys))
                            for r in ordered], dtype=np.int64)
    group_recs: list[list[int]] = [[] for _ in keys]
    for i, g in enumerate(rec_group.tolist()):
        group_recs[g].append(i)
    where: dict[int, list[int]] = defaultdict(list)
    for c, row in enumerate(col_rows.tolist()):
        where[row].append(c)
    same = {row: np.asarray(c) for row, c in where.items() if len(c) > 1}
    counts = np.diff(np.append(index.starts, len(col_rows)))
    layout = Layout(list(index.species), col_rows, np.asarray(unit_of_col, dtype=np.int64),
                    np.asarray(index.starts, dtype=np.int64), counts,
                    np.asarray(rec_starts, dtype=np.int64), np.asarray(rec_unit, np.int64),
                    rec_group, [np.asarray(g, dtype=np.int64) for g in group_recs], same)
    return layout, ordered


def left_out_cols(layout: Layout, r: int) -> np.ndarray:
    """Columns hidden from record r: its observer-day group's records, and every column
    holding one of its own photos."""
    parts = [layout.record_cols(g) for g in layout.group_recs[layout.rec_group[r]]]
    for row in layout.col_rows[layout.record_cols(r)].tolist():
        if row in layout.same_row_cols:
            parts.append(layout.same_row_cols[row])
    return np.unique(np.concatenate(parts))


def unit_sums(ref: np.ndarray, layout: Layout) -> np.ndarray:
    """(U, dim) float32 sum of each unit's photo vectors (columns of `ref`)."""
    out = np.empty((len(layout.starts), ref.shape[1]), dtype=np.float32)
    step = 2048
    for s in range(0, len(layout.starts), step):
        e = min(s + step, len(layout.starts))
        lo = layout.starts[s]
        hi = layout.starts[e] if e < len(layout.starts) else ref.shape[0]
        block = np.asarray(ref[lo:hi], dtype=np.float32)
        out[s:e] = np.add.reduceat(block, layout.starts[s:e] - lo, axis=0)
    return out


def without(ref: np.ndarray, layout: Layout, sums: np.ndarray, cols: np.ndarray):
    """(sums, means) of the reference with columns `cols` left out: for score_queries with
    adjust_means=False when one removal set serves many queries."""
    out = sums.copy()
    if len(cols):
        np.subtract.at(out, layout.unit_of_col[cols], np.asarray(ref[cols], np.float32))
    return out, as_served(out)


def as_served(means: np.ndarray) -> np.ndarray:
    """Unit-length means as NearestAndMean scores them (stored in float16)."""
    m = means / np.linalg.norm(means, axis=-1, keepdims=True).clip(1e-12)
    return m.astype(np.float16).astype(np.float32)


@dataclass
class BlockScores:
    unit: np.ndarray        # (n records, U) nearest+mean score, -inf where nothing is left
    photo: np.ndarray       # (photos, U) the same per query photo
    photo_owner: np.ndarray  # (photos,) which of the block's records each photo belongs to
    record_max: np.ndarray  # (photos, N) best match of each photo within each record
    available: np.ndarray   # (n records, U) photos each unit keeps once the group is out


def score_records(ref: np.ndarray, layout: Layout, recs: list[int], sums: np.ndarray,
                  means: np.ndarray, k: int = K, weight: float = WEIGHT) -> BlockScores:
    """nearest+mean scores of reference records `recs`, each against the reference with
    its left_out_cols hidden. `ref` (P, dim) holds the reference vectors in column order
    (float32), `sums` = unit_sums, `means` = as_served(sums)."""
    q_cols = [layout.record_cols(r) for r in recs]
    owner = np.concatenate([np.full(len(c), j) for j, c in enumerate(q_cols)])
    q = np.asarray(ref[np.concatenate(q_cols)], dtype=np.float32)
    hidden = [left_out_cols(layout, r) for r in recs]
    return score_queries(q, owner, hidden, ref, layout, sums, means, k=k, weight=weight)


def score_queries(q: np.ndarray, owner: np.ndarray, hidden: list[np.ndarray],
                  ref: np.ndarray, layout: Layout, sums: np.ndarray, means: np.ndarray,
                  sims: np.ndarray | None = None, k: int = K, weight: float = WEIGHT,
                  adjust_means: bool = True) -> BlockScores:
    """nearest+mean scores of query records against the reference with some columns
    hidden. q: (photos, dim) float32 query photos; owner: which query record each photo
    is (0..n-1, contiguous); hidden[j]: the columns hidden from record j. `sims`: q @ ref.T
    when already made (it is overwritten). adjust_means=False: `sums`/`means` already
    leave the hidden columns out (one removal set for many queries)."""
    if k != 2:
        raise ValueError("the masked top-k is written for k = 2 (NearestAndMean.k)")
    if sims is None:
        sims = q @ np.asarray(ref, dtype=np.float32).T
    U = len(layout.starts)
    available = np.tile(layout.counts, (len(hidden), 1))
    for j, cols in enumerate(hidden):
        rows = np.flatnonzero(owner == j)
        if len(cols):
            sims[rows[0]:rows[-1] + 1, cols] = -np.inf
            available[j] -= np.bincount(layout.unit_of_col[cols], minlength=U)
    record_max = np.maximum.reduceat(sims, layout.rec_starts, axis=1)
    m1 = np.maximum.reduceat(sims, layout.starts, axis=1)
    top = sims >= m1[:, layout.unit_of_col]
    ties = np.add.reduceat(top, layout.starts, axis=1, dtype=np.int32)
    sims[top] = -np.inf
    m2 = np.maximum.reduceat(sims, layout.starts, axis=1)
    del sims, top
    m2 = np.where(ties >= 2, m1, m2)
    avail_p = available[owner]
    nearest = np.where(avail_p >= 2, (m1 + m2) / 2, m1)
    mean_sim = q @ means.T
    for j, cols in enumerate(hidden):
        if not len(cols) or not adjust_means:
            continue
        rows = np.flatnonzero(owner == j)
        for u in np.unique(layout.unit_of_col[cols]).tolist():
            if available[j, u] <= 0:
                mean_sim[rows, u] = -np.inf
                continue
            gone = np.asarray(ref[cols[layout.unit_of_col[cols] == u]], np.float32).sum(axis=0)
            mean_sim[rows, u] = q[rows] @ as_served(sums[u] - gone)
    gone_p = avail_p <= 0
    nearest[gone_p] = -np.inf
    mean_sim[gone_p] = -np.inf
    photo = weight * nearest + (1 - weight) * mean_sim
    unit = np.stack([photo[owner == j].mean(axis=0) for j in range(len(hidden))])
    return BlockScores(unit.astype(np.float32), photo.astype(np.float32), owner,
                       record_max, available)


# --- per-record summaries --------------------------------------------------------------

def softmax_conf(scores: np.ndarray, temperature: float) -> np.ndarray:
    """Confidence as identify.softmax_confidence gives it, over the finite scores (0 for
    the rest)."""
    out = np.zeros(len(scores), dtype=np.float64)
    fin = np.isfinite(scores)
    if fin.any():
        z = (scores[fin].astype(np.float64) - scores[fin].max()) / temperature
        e = np.exp(z)
        out[fin] = e / e.sum()
    return out


def rank_of(scores: np.ndarray, i: int) -> int:
    """1-based place of scores[i] among the finite scores; 0 when it has none."""
    if i < 0 or not np.isfinite(scores[i]):
        return 0
    return int((scores[np.isfinite(scores)] > scores[i]).sum()) + 1


def genus_scores(unit_scores: np.ndarray, unit_genus: np.ndarray, n_genera: int) -> np.ndarray:
    """Best unit score inside each genus (evaluate.rank_scores at genus)."""
    out = np.full(n_genera, -np.inf, dtype=np.float32)
    keep = unit_genus >= 0
    np.maximum.at(out, unit_genus[keep], unit_scores[keep])
    return out


# --- flagging ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rules:
    """Thresholds, fixed before the scan's results were looked at (see the experiment)."""
    confident: float = 0.7        # calibrated confidence of the other species
    margin: float = 0.01          # or: its score beats the label's by this much
    neighbours_agree: int = 6     # of the 10 nearest records, at least this many carry it
    label_records: int = 3        # label species keeps at least this many records elsewhere
    label_recognised: float = 0.5  # and at least this share of them are top-1 the label
    back_rate: float = 0.2        # the other species' records predicted as the label: (b)/(d)
    same_cluster: float = 0.5     # both directions at least this share: one species (b)
    pair_records: int = 3         # a loosely related pair is systematic from this many records
    no_match: float = 0.55        # best photo match to any other record below this: (c)


STRONG_RELATIONS = ("same name, other writing", "same species s.l.")


def related_names(a: str, b: str, name_key=None, equiv=None) -> str:
    """How two labels' names relate, '' when they don't: a writing variant (name_key), the
    s.l./complex rules (name_equiv), the same epithet in another genus, or a provisional
    name beside another name in its genus."""
    if name_key is not None and name_key(a) == name_key(b):
        return "same name, other writing"
    if equiv is not None and equiv.parts(a).genus and equiv.parts(b).genus:
        m = equiv.species_match(a, b)
        if m.get("sl"):
            return "same species s.l."
        if m.get("complex"):
            return "same complex"
    if equiv is None:
        return ""
    pa, pb = equiv.parts(a), equiv.parts(b)
    if not (pa.genus and pb.genus):
        return ""
    if (pa.genus != pb.genus and not (pa.provisional or pb.provisional)
            and pa.folded and pa.folded == pb.folded):
        return "same epithet, other genus"
    if pa.genus == pb.genus and pa.provisional != pb.provisional:
        return "provisional beside formal, one genus"
    return ""


def pair_rates(label: np.ndarray, pred: np.ndarray) -> tuple[dict, dict]:
    """({(a, b): share of label-a records whose top-1 is b}, {(a, b): how many}) for
    a != b, over records whose label is a species (>= 0)."""
    n = Counter(label[label >= 0].tolist())
    hits = Counter(zip(label.tolist(), pred.tolist()))
    pairs = {(a, b): c for (a, b), c in hits.items() if a >= 0 and b >= 0 and a != b}
    return {ab: c / n[ab[0]] for ab, c in pairs.items()}, pairs


def classify(rec: dict, rules: Rules, rates: dict, counts: dict, self_rate: dict,
             support: dict, relation: str) -> str:
    """The category of one record ('a', 'b', 'c', 'd', or '' for nothing to flag).

    rec: label (unit index, -1 for a one-word name), pred (top-1 species unit), conf (its
    calibrated confidence), margin (pred score minus label score; inf when the label has
    nothing left), nb_pred (of its 10 nearest records, how many carry pred), best_match
    (its photos' best match to any other record, mean over photos), photo_genera (distinct
    genera its photos point to at top-1, among photos that match well), label_left (the
    label's records outside its observer-day group; default support - 1).
    rates, counts: pair_rates; self_rate[u]: share of u's records top-1 u; support[u]: u's
    records; relation: related_names of the two labels. A writing variant or an s.l. match
    is (b) at once; a looser relation only when the pair is systematic."""
    if rec["best_match"] < rules.no_match:
        return "c"
    if rec.get("photo_genera", 1) > 1 and rec["pred"] != rec["label"]:
        return "c"
    a, b = rec["label"], rec["pred"]
    if a < 0 or b < 0 or a == b:
        return ""
    sure = rec["conf"] >= rules.confident or rec["margin"] >= rules.margin
    if not sure or rec["nb_pred"] < rules.neighbours_agree:
        return ""
    ab, ba = rates.get((a, b), 0.0), rates.get((b, a), 0.0)
    systematic = (counts.get((a, b), 0) + counts.get((b, a), 0) >= rules.pair_records
                  or max(ab, ba) >= rules.back_rate)
    if relation in STRONG_RELATIONS or (relation and systematic):
        return "b"
    if ab >= rules.same_cluster and ba >= rules.same_cluster:
        return "b"
    if ba >= rules.back_rate:
        well = (support.get(a, 0) >= rules.label_records
                and support.get(b, 0) >= rules.label_records)
        return "d" if well else ""
    if (rec.get("label_left", support.get(a, 0) - 1) >= rules.label_records
            and self_rate.get(a, 0.0) >= rules.label_recognised):
        return "a"
    return ""


def strength(rec: dict, rates: dict, self_rate: dict) -> float:
    """How strongly the evidence says the label is wrong, 0-1: the other species'
    confidence, the share of nearest records carrying it, how well the label's species is
    recognised elsewhere, and how rarely the other species is taken for the label."""
    a, b = rec["label"], rec["pred"]
    return float(rec["conf"] * (rec["nb_pred"] / 10) * self_rate.get(a, 0.0)
                 * (1 - rates.get((b, a), 0.0)))
