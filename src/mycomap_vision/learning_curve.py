"""Learning curve: how accuracy grows with a species' reference records (exp/learning-curve).

Read-only analysis on a manifest COPY and stored embeddings; CPU and numpy only (no
torch: numpy + torch segfaults on big CPU matmuls in this venv, so methods.Scorer is
not used here).

Two experiments re-score the held-out development split against subsets of the
reference index, without embedding anything again:

- fractions: every species keeps 25 / 50 / 75 / 100% of its reference RECORDS (whole
  records, at least one where it has any), several seeds. What a smaller (or, read
  backwards, a growing) reference does to everyone at once.
- truth caps ("add N more"): only the TRUE species of each record is cut to N records
  (1, 2, 3, 5, 10, ...); every other species keeps all of its records. What one more
  sequenced record of this species is worth while everything else stays as today:
  the number that says which species to sequence next.

Both score `nearest` (mean over query photos of the best match in each species) and
`nearest+mean` (0.6 x mean of the two best matches + 0.4 x similarity to the species'
average photo), exactly as methods.py does on the full reference. The trick that makes
subsets cheap: each query photo's similarities are reduced once to each reference
RECORD's best and second-best photo; the species' best two photos over any set of kept
records are the best two of those pairs, and a species' average photo is the
normalised sum of its kept records' photo sums.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

NEG = np.float32(-np.inf)
WEIGHT, K = 0.6, 2            # nearest+mean as methods.NearestAndMean
METHODS = ("nearest", "nearest+mean")
BANDS = [(0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, 10**9, "100+")]


def band_of(n: int) -> str:
    for lo, hi, label in BANDS:
        if lo <= n <= hi:
            return label
    return BANDS[-1][2]


# --- reductions ----------------------------------------------------------------------

def record_best_two(sims: np.ndarray, rec_col_starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(q, R) best and second-best similarity of each query photo to each record's photos
    (columns grouped by record, `rec_col_starts` where each record's run begins). A
    record with one photo has second = -inf; two photos tied at the best give second =
    best."""
    best = np.maximum.reduceat(sims, rec_col_starts, axis=1)
    lengths = np.diff(np.append(rec_col_starts, sims.shape[1]))
    rep = np.repeat(best, lengths, axis=1)
    eq = sims == rep
    ties = np.add.reduceat(eq, rec_col_starts, axis=1, dtype=np.int32)
    second = np.maximum.reduceat(np.where(eq, NEG, sims), rec_col_starts, axis=1)
    second = np.where(ties >= 2, best, second)
    return best, second


def species_top(best: np.ndarray, second: np.ndarray, kept: np.ndarray,
                unit_rec_starts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(q, S) best match and mean of the two best matches within each species, over the
    kept records only (records grouped by species). A species with no kept record gets
    -inf in both; one with a single kept photo gets its best match as the top-2 mean
    (methods.top_k_species_scores divides by min(photos, k))."""
    a = np.where(kept, best, NEG)
    b = np.where(kept, second, NEG)
    m1 = np.maximum.reduceat(a, unit_rec_starts, axis=1)
    lengths = np.diff(np.append(unit_rec_starts, best.shape[1]))
    eq = (a == np.repeat(m1, lengths, axis=1)) & kept
    ties = np.add.reduceat(eq, unit_rec_starts, axis=1, dtype=np.int32)
    m2a = np.maximum.reduceat(np.where(eq, NEG, a), unit_rec_starts, axis=1)
    b_at = np.maximum.reduceat(np.where(eq, b, NEG), unit_rec_starts, axis=1)
    m2 = np.where(ties >= 2, m1, np.maximum(m2a, b_at))
    with np.errstate(invalid="ignore"):
        top2 = np.where(np.isfinite(m2), (m1 + m2) / 2, m1)
    return m1, top2.astype(np.float32)


def kept_means(rec_sums: np.ndarray, kept: np.ndarray, unit_rec_starts: np.ndarray) -> np.ndarray:
    """(S, d) unit-length average photo of each species over its kept records (zeros for
    a species with none)."""
    sums = np.add.reduceat(np.where(kept[:, None], rec_sums, 0.0).astype(np.float32),
                           unit_rec_starts, axis=0)
    norm = np.linalg.norm(sums, axis=1, keepdims=True)
    return np.where(norm > 0, sums / np.where(norm > 0, norm, 1.0), 0.0).astype(np.float32)


def blend(top2: np.ndarray, mean_sim: np.ndarray) -> np.ndarray:
    return WEIGHT * top2 + (1 - WEIGHT) * mean_sim


# --- subsets -------------------------------------------------------------------------

def record_order(n_records: int, seed: int) -> np.ndarray:
    """One random key per record for a seed: within a species, records are kept in key
    order, so every fraction and cap of one seed is nested in the larger ones."""
    return np.random.default_rng(seed).random(n_records)


def rank_within_unit(keys: np.ndarray, base: np.ndarray, rec_unit: np.ndarray) -> np.ndarray:
    """Each base record's 0-based position among its species' base records in key order;
    a large number for a record outside the base."""
    k = np.where(base, keys, np.inf)
    order = np.lexsort((k, rec_unit))
    pos = np.empty(len(keys), dtype=np.int64)
    starts = np.flatnonzero(np.diff(rec_unit[order], prepend=-1) != 0)
    run = np.arange(len(order)) - np.repeat(starts, np.diff(np.append(starts, len(order))))
    pos[order] = run
    return np.where(base, pos, 1 << 40)


def keep_count(n_base: np.ndarray, fraction: float | None = None,
               cap: int | None = None) -> np.ndarray:
    """Records a species keeps: round(fraction x n), at least one where it has any; or
    min(cap, n)."""
    if fraction is not None:
        k = np.floor(fraction * n_base + 0.5).astype(np.int64)
        return np.where(n_base > 0, np.maximum(k, 1), 0)
    return np.minimum(n_base, int(cap))


def subset_mask(keys: np.ndarray, base: np.ndarray, rec_unit: np.ndarray, n_units: int,
                fraction: float | None = None, cap: int | None = None) -> np.ndarray:
    pos = rank_within_unit(keys, base, rec_unit)
    n_base = np.bincount(rec_unit[base], minlength=n_units)
    k = keep_count(n_base, fraction, cap)
    return base & (pos < k[rec_unit])


# --- saturating curves -----------------------------------------------------------------

def fit_saturating(ns: np.ndarray, acc: np.ndarray, weights: np.ndarray | None = None) -> dict:
    """acc(N) = a x N / (N + k) (a ceiling a, half of it reached at N = k), by weighted
    least squares: a grid over k, a in closed form. The knee is where one more record
    adds less than one point (d acc / dN < 0.01): N = sqrt(100 a k) - k."""
    ns, acc = np.asarray(ns, float), np.asarray(acc, float)
    w = np.ones_like(acc) if weights is None else np.asarray(weights, float)
    best = None
    for k in np.geomspace(0.05, 2000, 4000):
        x = ns / (ns + k)
        a = float((w * x * acc).sum() / max((w * x * x).sum(), 1e-12))
        sse = float((w * (acc - a * x) ** 2).sum())
        if best is None or sse < best[0]:
            best = (sse, a, float(k))
    _sse, a, k = best
    knee = max(0.0, math.sqrt(100 * a * k) - k)
    return {"ceiling": a, "half_at": k, "knee": knee, "n90": 9 * k,
            "rmse": math.sqrt(best[0] / max(w.sum(), 1e-12))}


def saturating(n, fit: dict):
    n = np.asarray(n, float)
    return fit["ceiling"] * n / (n + fit["half_at"])


# --- the run -------------------------------------------------------------------------

@dataclass
class Layout:
    units: list[str]                  # index.species order
    unit_rec_starts: np.ndarray       # (S,) first record of each unit
    rec_unit: np.ndarray              # (R,)
    rec_obs: list[str]                # (R,) observation ids (never written out)
    rec_col_starts: np.ndarray        # (R,) first column of each record
    cols: np.ndarray                  # (C,) vector rows in column order
    species_of_unit: np.ndarray       # (S,) species label index or -1
    genus_of_unit: np.ndarray         # (S,) genus label index or -1
    genus_names: list[str] = field(default_factory=list)


def build_layout(records, index) -> Layout:
    """Records grouped by index unit (index.species order), photos grouped by record:
    the same columns build_index makes (records within a unit in list order)."""
    by_unit: dict[str, list] = defaultdict(list)
    for r in records:
        by_unit[r.unit].append(r)
    rec_unit, rec_obs, rec_col_starts, cols, unit_rec_starts = [], [], [], [], []
    for u, name in enumerate(index.species):
        unit_rec_starts.append(len(rec_unit))
        for r in by_unit[name]:
            rec_unit.append(u)
            rec_obs.append(r.observation_id)
            rec_col_starts.append(len(cols))
            cols.extend(r.photo_rows)
    cols = np.asarray(cols, dtype=np.int64)
    assert np.array_equal(cols, index.cols), "record layout differs from the index's columns"
    return Layout(list(index.species), np.asarray(unit_rec_starts, np.int64),
                  np.asarray(rec_unit, np.int64), rec_obs, np.asarray(rec_col_starts, np.int64),
                  cols, np.asarray(index.label_of["species"]), np.asarray(index.label_of["genus"]),
                  list(index.labels["genus"]))


def reference_hash(backbone: str, records) -> str:
    """heldout.reference_summary's hash: the records and their labels."""
    lines = sorted(f"{r.observation_id}\t{r.species}\t{r.genus}\t{r.family}" for r in records)
    h = hashlib.sha1(f"{backbone}|".encode())
    h.update("\n".join(lines).encode())
    return h.hexdigest()[:12]


def bad_reference_ids(sources_tsv: Path) -> tuple[set[str], int]:
    """Ids .org holds only as non-iNat records (MO, MyCoPortal, .com Sequences, GenBank)
    and that Vision took for iNat ids: their iNat photos are some other observation's.
    Ids green as both an iNat and a non-iNat record are left in (counted)."""
    inat, other = set(), set()
    with open(sources_tsv, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 2 or not parts[0].strip():
                continue
            (inat if parts[1].strip() == "iNaturalist" else other).add(parts[0].strip())
    return other - inat, len(other & inat)


def top_list(scores: np.ndarray, allowed: np.ndarray, k: int) -> list[tuple[int, float]]:
    s = np.where(allowed & np.isfinite(scores), scores, NEG)
    k = min(k, int(np.isfinite(s).sum()))
    if k <= 0:
        return []
    idx = np.argpartition(-s, k - 1)[:k]
    idx = idx[np.argsort(-s[idx], kind="stable")]
    return [(int(i), float(s[i])) for i in idx]


def genus_scores(scores: np.ndarray, lay: Layout) -> np.ndarray:
    out = np.full(len(lay.genus_names), NEG, dtype=np.float32)
    keep = lay.genus_of_unit >= 0
    np.maximum.at(out, lay.genus_of_unit[keep], scores[keep])
    return out


SEED_BASE = 1000              # seed s draws its record order from default_rng(SEED_BASE + s)


def score(args) -> None:
    """One process: score the split (or one shard of it) under every condition."""
    import sqlite3
    from . import heldout
    from .embed import load_embeddings
    from .evaluate import build_index, load_records, with_rows
    from .config import code_version
    from .serving import map_embeddings

    t0 = time.time()
    log = lambda *a: print(f"[{time.time() - t0:7.1f}s]", *a, flush=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(args.manifest)
    bconn = sqlite3.connect(args.benchmark_db) if args.benchmark_db else conn
    bb = args.backbone
    ids, vecs = map_embeddings(conn, bb, Path(args.embeddings_root) / bb)
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    records = with_rows(load_records(conn, {int(p): int(p) for p in ids.tolist()}), row_of)
    index = build_index(records)
    lay = build_layout(records, index)
    ref_hash = reference_hash(bb, records)
    log(f"reference {ref_hash}: {len(records):,} records, {len(lay.cols):,} photos, "
        f"{len(lay.units):,} units")

    bad_ids, both = bad_reference_ids(Path(args.sources)) if args.sources else (set(), 0)
    is_bad = np.array([o in bad_ids for o in lay.rec_obs], dtype=bool)
    log(f"non-iNat ids in .org: {len(bad_ids):,} ({both} also iNat, left in); "
        f"in this reference: {int(is_bad.sum()):,} records")
    clean = ~is_bad
    R, S = len(lay.rec_obs), len(lay.units)

    # Reference vectors (float32, column order) and per-record photo sums.
    cache = Path(args.cache) if args.cache else None
    tag = f"{bb}-{ref_hash}"
    if cache and (cache / f"ref-{tag}.npy").is_file():
        ref = np.load(cache / f"ref-{tag}.npy", mmap_mode="r")
        rec_sums = np.load(cache / f"recsums-{tag}.npy", mmap_mode="r")
    else:
        ref = np.empty((len(lay.cols), vecs.shape[1]), dtype=np.float32)
        for s in range(0, len(lay.cols), 65536):
            ref[s:s + 65536] = vecs[lay.cols[s:s + 65536]]
        rec_sums = np.add.reduceat(ref, lay.rec_col_starts, axis=0)
        if cache:
            # Shared by the shard processes through the page cache (never committed).
            cache.mkdir(parents=True, exist_ok=True)
            np.save(cache / f"ref-{tag}.npy", ref)
            np.save(cache / f"recsums-{tag}.npy", rec_sums)
    log("reference vectors ready")

    # Held-out development records.
    split_ids = heldout.benchmark_ids(bconn, args.benchmark, split=args.split)
    hrecs = [r for r in heldout.load_benchmark(bconn, args.benchmark, split_ids, "large")
             if r.photos]
    edb = sqlite3.connect(args.bench_index)
    pids, pvecs = load_embeddings(edb, bb, Path(args.bench_vectors))
    edb.close()
    qvec = {int(p): pvecs[i] for i, p in enumerate(pids.tolist())}
    labeller = heldout.Labeller(conn, extra=[r.truth_name for r in hrecs if r.truth_name])
    key_of_unit = [heldout.name_key(labeller.label(u)) for u in lay.units]
    units_by_key: dict[str, list[int]] = defaultdict(list)
    for u, k in enumerate(key_of_unit):
        if lay.species_of_unit[u] >= 0:
            units_by_key[k].append(u)
    genus_by_key: dict[str, list[int]] = defaultdict(list)
    for g, name in enumerate(lay.genus_names):
        genus_by_key[heldout.name_key(labeller.label(name))].append(g)
    ref_obs = set(lay.rec_obs)
    ref_photos = set(int(ids[c]) for c in lay.cols.tolist())
    queries = []        # (oid, photo vectors, truth dict)
    skipped = Counter()
    for r in hrecs:
        if not r.truth_name:
            skipped["no answer key"] += 1
            continue
        t = labeller.truth(r.truth_name)
        if t.guest:
            skipped["guest"] += 1
            continue
        if r.observation_id in ref_obs:
            skipped["already a reference record"] += 1
            continue
        p = [pid for pid, _s, _p in r.photos]
        if ref_photos & set(p):
            skipped["photo is a reference photo"] += 1
            continue
        v = [qvec[x] for x in p if x in qvec]
        if not v:
            skipped["no vectors"] += 1
            continue
        tu = units_by_key.get(heldout.name_key(labeller.label(t.species)), []) if t.species else []
        tg = genus_by_key.get(heldout.name_key(labeller.label(t.genus)), []) if t.genus else []
        queries.append((r.observation_id, np.stack(v).astype(np.float32),
                        {"species": t.species, "genus": t.genus, "units": tu, "genera": tg}))
    if args.limit:
        queries = queries[:args.limit]
    if args.shards > 1:
        queries = queries[args.shard::args.shards]
    log(f"{len(queries):,} {args.split} records to score ({dict(skipped)})")

    # Conditions on the whole reference.
    seeds = list(range(args.seeds))
    keys = {s: record_order(R, SEED_BASE + s) for s in seeds}
    conds: dict[str, np.ndarray] = {"full-clean": clean.copy()}
    if is_bad.any():
        conds["full-with-bad"] = np.ones(R, bool)
    for f in args.fractions:
        for s in seeds:
            conds[f"frac{int(f * 100)}-s{s}"] = subset_mask(keys[s], clean, lay.rec_unit, S, fraction=f)
    for c in args.global_caps:
        for s in seeds[:3]:
            conds[f"gcap{c}-s{s}"] = subset_mask(keys[s], clean, lay.rec_unit, S, cap=c)
    counts = {name: np.bincount(lay.rec_unit[m], minlength=S) for name, m in conds.items()}
    protos = {}
    for name, m in conds.items():
        f = cache / f"proto-{tag}-{name}-{args.seeds}.npy" if cache else None
        if f is not None and f.is_file():
            protos[name] = np.load(f, mmap_mode="r")
            continue
        protos[name] = kept_means(rec_sums, m, lay.unit_rec_starts)
        if f is not None:
            np.save(f, protos[name])
    if args.prep_only:
        log("caches ready")
        return
    log(f"{len(conds)} reference conditions; {len(args.caps)} truth caps x {len(seeds)} seeds")

    species_ok = lay.species_of_unit >= 0
    results = {name: {m: {} for m in METHODS} for name in conds}
    capped = {m: defaultdict(dict) for m in METHODS}     # method -> oid -> {(cap, seed): score}
    base_others = {}                                     # oid -> species/genus lists without truth
    truth_info = {}
    clean_pos = {s: rank_within_unit(keys[s], clean, lay.rec_unit) for s in seeds}

    def score_photos(best, second, qv, name):
        m1, top2 = species_top(best, second, conds[name], lay.unit_rec_starts)
        mean_sim = qv @ protos[name].T
        return m1, blend(top2, mean_sim)

    batch, bstart = [], 0
    order = list(range(len(queries)))

    def flush(batch_ids):
        qv = np.concatenate([queries[i][1] for i in batch_ids])
        owner = np.concatenate([[j] * len(queries[i][1]) for j, i in enumerate(batch_ids)])
        best = np.empty((len(qv), R), np.float32)
        second = np.empty((len(qv), R), np.float32)
        # Similarities in blocks of whole records.
        step = 50000
        r0 = 0
        while r0 < R:
            r1 = min(R, r0 + step)
            c0 = lay.rec_col_starts[r0]
            c1 = lay.rec_col_starts[r1] if r1 < R else len(lay.cols)
            sims = qv @ ref[c0:c1].T
            b, s2 = record_best_two(sims, lay.rec_col_starts[r0:r1] - c0)
            best[:, r0:r1], second[:, r0:r1] = b, s2
            r0 = r1
        seg = np.flatnonzero(np.diff(owner, prepend=-1) != 0)
        for name in conds:
            m1, nm = score_photos(best, second, qv, name)
            for meth, ph in (("nearest", m1), ("nearest+mean", nm)):
                with np.errstate(invalid="ignore"):
                    per_rec = np.add.reduceat(ph, seg, axis=0) / np.diff(np.append(seg, len(qv)))[:, None]
                for j, i in enumerate(batch_ids):
                    oid, _v, t = queries[i]
                    sc = per_rec[j]
                    truth_units = t["units"]
                    entry = {"species": top_list(sc, species_ok, args.top),
                             "genus": top_list(genus_scores(sc, lay), np.ones(len(lay.genus_names), bool),
                                               args.top),
                             "truth_score": max((float(sc[u]) for u in truth_units), default=None)}
                    results[name][meth][oid] = entry
                    if name == "full-clean":
                        # The lists without the truth's own unit(s), for the truth caps.
                        sc2 = sc.copy()
                        sc2[truth_units] = NEG
                        base_others.setdefault(oid, {})[meth] = {
                            "species": top_list(sc2, species_ok, args.top + 2),
                            "genus": top_list(genus_scores(sc2, lay),
                                              np.ones(len(lay.genus_names), bool), args.top + 2)}
        # Truth caps: only the true species is cut down (clean base).
        for j, i in enumerate(batch_ids):
            oid, v, t = queries[i]
            rows = np.flatnonzero(owner == j)
            if len(t["units"]) != 1:
                continue
            u = t["units"][0]
            lo = lay.unit_rec_starts[u]
            hi = lay.unit_rec_starts[u + 1] if u + 1 < S else R
            recs = np.arange(lo, hi)[clean[lo:hi]]
            truth_info[oid] = {"unit": u, "n_clean": int(len(recs)),
                               "n_with_bad": int(hi - lo)}
            if not len(recs):
                continue
            B, S2 = best[rows][:, recs], second[rows][:, recs]
            for s in seeds:
                rk = recs[np.argsort(clean_pos[s][recs], kind="stable")]
                ordr = np.searchsorted(recs, rk)
                for cap in args.caps:
                    if cap > len(recs) and cap != args.caps[-1]:
                        continue
                    sel = ordr[:min(cap, len(recs))]
                    bb_, ss_ = B[:, sel], S2[:, sel]
                    m1 = bb_.max(axis=1)
                    vals = np.concatenate([bb_, ss_], axis=1)
                    top2v = np.sort(vals, axis=1)[:, -2:]
                    t2 = np.where(np.isfinite(top2v[:, 0]), top2v.mean(axis=1), top2v[:, 1])
                    sm = rec_sums[rk[:min(cap, len(recs))]].sum(axis=0)
                    sm = sm / max(np.linalg.norm(sm), 1e-12)
                    ms = v @ sm
                    capped["nearest"][oid][(min(cap, len(recs)), s)] = float(m1.mean())
                    capped["nearest+mean"][oid][(min(cap, len(recs)), s)] = float(
                        blend(t2, ms).mean())

    ids_in_batch = []
    photos_in_batch = 0
    for i in order:
        ids_in_batch.append(i)
        photos_in_batch += len(queries[i][1])
        if photos_in_batch >= args.batch_photos:
            flush(ids_in_batch)
            log(f"  {i + 1:,}/{len(queries):,} records scored")
            ids_in_batch, photos_in_batch = [], 0
    if ids_in_batch:
        flush(ids_in_batch)
    log("scoring done")

    rec_unit_counts = {name: c for name, c in counts.items()}
    # Genus reference counts per condition (records), for genus bands.
    genus_counts = {}
    for name, c in counts.items():
        g = np.zeros(len(lay.genus_names), np.int64)
        keep = lay.genus_of_unit >= 0
        np.add.at(g, lay.genus_of_unit[keep], c[keep])
        genus_counts[name] = g
    payload = {
        "reference_hash": ref_hash, "backbone": bb, "benchmark": args.benchmark,
        "split": args.split, "records": len(records), "photos": int(len(lay.cols)),
        "units": lay.units, "species_of_unit": lay.species_of_unit,
        "genus_of_unit": lay.genus_of_unit, "genus_names": lay.genus_names,
        "bad_in_reference": int(is_bad.sum()), "bad_ids_total": len(bad_ids),
        "bad_also_inat": both, "skipped": dict(skipped),
        "conditions": list(conds), "unit_counts": rec_unit_counts, "genus_counts": genus_counts,
        "results": results, "capped": {m: dict(v) for m, v in capped.items()},
        "base_others": base_others, "truth_info": truth_info,
        "truths": {q[0]: q[2] for q in queries}, "n_photos": {q[0]: len(q[1]) for q in queries},
        "caps": args.caps, "seeds": seeds, "fractions": args.fractions,
        "global_caps": args.global_caps,
        "provenance": {"code_version": code_version(), "seed_base": SEED_BASE},
    }
    fname = f"results-{args.shard}of{args.shards}.pkl" if args.shards > 1 else "results.pkl"
    with open(out_dir / fname, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"wrote {out_dir / fname}")


# --- the one command ---------------------------------------------------------------------

REPRODUCIBILITY = "exploratory-pre-freeze"   # until the "Dataset release v1" freeze (Steve)


def snapshot(src: Path, dst: Path) -> None:
    """A consistent copy of a SQLite file (its WAL included), the source opened read-only.
    Every run works on its own copies: the loaders add tables to the file they open, and
    a shared manifest changes during the day."""
    import sqlite3
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    source = sqlite3.connect(Path(src).resolve().as_uri() + "?mode=ro", uri=True)
    target = sqlite3.connect(tmp)
    try:
        source.backup(target)
    finally:
        target.close()
        source.close()
    tmp.replace(dst)


def frequencies(conn, bconn, benchmark: str, occurrence: Path | None) -> dict:
    """How often each species arrives in the benchmark (all splits: MycoMap's own stream
    of newly DNA-validated records) and, where the occurrence store knows it, how many
    North American iNat observations it has. Keyed by Vision's species label."""
    from . import heldout
    recs = heldout.load_benchmark(bconn, benchmark)
    lab = heldout.Labeller(conn, extra=[r.truth_name for r in recs if r.truth_name])
    pool, genus_of = Counter(), {}
    for r in recs:
        if r.truth_name:
            t = lab.truth(r.truth_name)
            if t.species and not t.guest:
                pool[t.species] += 1
                genus_of[t.species] = t.genus
    inat: dict[str, int] = {}
    if occurrence is not None and Path(occurrence).is_file():
        from .occurrence import OccurrenceStore
        store = OccurrenceStore.load(Path(occurrence))
        labels = set(pool)
        for (n,) in conn.execute("select distinct scientific_name from records"):
            if n:
                t = lab.truth(n)
                if t.species:
                    labels.add(t.species)
                    genus_of.setdefault(t.species, t.genus)
        for name in labels:
            m = store.resolve(name, genus_of.get(name, ""))
            if m.species_unit >= 0:
                inat[name] = int(store.unit_total[m.species_unit])
    return {"pool": dict(pool), "inat": inat}


def resolve_inputs(args) -> dict:
    """Where a run reads from: a manifest (with the data folder's embeddings) or a pulled
    release folder (its manifest and embeddings; the held-out tables then come from
    --benchmark-db, since a release manifest carries none)."""
    from . import config
    from .taxonomy import CACHE as TAXONOMY_CACHE
    data_dir = Path(args.data_dir) if args.data_dir else config.DATA_DIR
    if args.release:
        rel = Path(args.release)
        meta = rel / "release.json"
        if not args.benchmark_db:
            raise SystemExit("--release needs --benchmark-db (a release manifest has no "
                             "benchmark tables)")
        inputs = {"manifest": rel / "manifest.sqlite", "embeddings": rel / "embeddings",
                  "taxonomy": rel / TAXONOMY_CACHE,
                  "release": (json.loads(meta.read_text(encoding="utf-8")).get("id")
                              if meta.is_file() else rel.name)}
    else:
        manifest = Path(args.manifest)
        inputs = {"manifest": manifest,
                  "embeddings": (Path(args.embeddings_root) if args.embeddings_root
                                 else data_dir / "embeddings"),
                  "taxonomy": manifest.parent / TAXONOMY_CACHE, "release": None}
    bench = (Path(args.benchmark_dir) if args.benchmark_dir
             else data_dir / "benchmarks" / args.benchmark)
    inputs.update(data_dir=data_dir, benchmark_db=Path(args.benchmark_db) if args.benchmark_db else None,
                  bench_index=bench / "embeddings" / "large" / "index.sqlite",
                  bench_vectors=bench / "embeddings" / "large" / args.backbone,
                  occurrence=(Path(args.occurrence) if args.occurrence
                              else data_dir / "occurrence" / "inat-fungi-na.npz"))
    return inputs


def score_command(args, work: Path, scores: Path, cache: Path, inputs: dict) -> list[str]:
    """The `score` arguments every shard shares (fixed seeds: SEED_BASE + 0..seeds-1)."""
    cmd = ["--manifest", str(work / "manifest.sqlite"), "--embeddings-root", str(inputs["embeddings"]),
           "--backbone", args.backbone, "--benchmark", args.benchmark, "--split", args.split,
           "--bench-index", str(work / "bench-index.sqlite"),
           "--bench-vectors", str(inputs["bench_vectors"]), "--out", str(scores),
           "--cache", str(cache), "--seeds", str(args.seeds),
           "--batch-photos", str(args.batch_photos), "--top", str(args.top)]
    if inputs["benchmark_db"] is not None:
        cmd += ["--benchmark-db", str(work / "benchmark.sqlite")]
    if args.sources:
        cmd += ["--sources", str(args.sources)]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return cmd + ["--fractions", *map(str, args.fractions),
                  "--global-caps", *map(str, args.global_caps), "--caps", *map(str, args.caps)]


def run_all(args) -> None:
    """The whole experiment, re-runnable: snapshot the inputs, build the shared reference
    arrays, score the split in parallel shard processes (fixed seeds), write the report."""
    import os
    import shutil
    import sqlite3
    import subprocess
    from . import config

    out = Path(args.out)
    work, scores = out / "work", out / "scores"
    cache = Path(args.cache) if args.cache else out / "cache"
    inputs = resolve_inputs(args)
    man = work / "manifest.sqlite"
    if not (args.reuse_snapshot and man.is_file()):
        print(f"snapshot {inputs['manifest']} -> {man}", flush=True)
        snapshot(inputs["manifest"], man)
        if inputs["benchmark_db"] is not None:
            snapshot(inputs["benchmark_db"], work / "benchmark.sqlite")
        if inputs["taxonomy"].is_file():
            tax = work / inputs["taxonomy"].relative_to(inputs["taxonomy"].parents[1])
            tax.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(inputs["taxonomy"], tax)
        shutil.copy2(inputs["bench_index"], work / "bench-index.sqlite")
    snapshot_at = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(man.stat().st_mtime))

    common = score_command(args, work, scores, cache, inputs)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    threads = str(max(1, (os.cpu_count() or 4) // max(args.workers, 1)))
    env.update(OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads)
    me = [sys.executable, "-m", "mycomap_vision.learning_curve", "score"]
    scores.mkdir(parents=True, exist_ok=True)
    for old in scores.glob("results*.pkl"):
        old.unlink()
    with open(out / "prep.log", "w", encoding="utf-8") as log:
        if subprocess.run(me + common + ["--prep-only"], env=env, stdout=log,
                          stderr=subprocess.STDOUT).returncode:
            raise SystemExit(f"prep failed: see {out / 'prep.log'}")
    procs = []
    for i in range(args.workers):
        log = open(out / f"score-{i}.log", "w", encoding="utf-8")
        shard = ["--shards", str(args.workers), "--shard", str(i)] if args.workers > 1 else []
        procs.append((subprocess.Popen(me + common + shard, env=env, stdout=log,
                                       stderr=subprocess.STDOUT), log, i))
    failed = []
    for proc, log, i in procs:
        if proc.wait():
            failed.append(i)
        log.close()
    if failed:
        raise SystemExit(f"shards {failed} failed: see {out}/score-<n>.log")

    conn = sqlite3.connect(man)
    bconn = sqlite3.connect(work / "benchmark.sqlite") if inputs["benchmark_db"] else conn
    occ = inputs["occurrence"]
    freq = frequencies(conn, bconn, args.benchmark, occ)
    (work / "frequencies.json").write_text(json.dumps(freq), encoding="utf-8")
    from .learning_curve_report import report
    provenance = {"reproducibility": REPRODUCIBILITY, "code_version": config.code_version(),
                  "release": inputs["release"], "snapshot_at": snapshot_at,
                  "seed_base": SEED_BASE, "seeds": args.seeds, "workers": args.workers,
                  "limit": args.limit or None,
                  "occurrence_store": occ.name if occ.is_file() else None,
                  "exclusions": Path(args.sources).name if args.sources else None}
    report(scores, out, freq["pool"], freq["inat"], provenance=provenance)
    print(f"report: {out / 'report.md'}", flush=True)


def _score_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--manifest", required=True, help="a COPY of the manifest (tables are added to it)")
    p.add_argument("--embeddings-root", required=True, help="folder holding <backbone>/shard-*.npy")
    p.add_argument("--benchmark-db", help="where the held-out tables are (default: --manifest)")
    p.add_argument("--bench-index", required=True, help="a copy of the benchmark's embeddings index")
    p.add_argument("--bench-vectors", required=True)
    p.add_argument("--sources", help=".org observation sources TSV: non-iNat ids are left out")
    p.add_argument("--out", required=True)
    p.add_argument("--cache", help="folder for the reference arrays shared by shard processes")
    p.add_argument("--prep-only", action="store_true", help="build the caches and stop")
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard", type=int, default=0)


def _shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backbone", default="bioclip-2-ft-20261007-165400")
    p.add_argument("--benchmark", default="heldout-2026-10-08")
    p.add_argument("--split", default="dev")
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--fractions", type=float, nargs="*", default=[0.25, 0.5, 0.75])
    p.add_argument("--global-caps", type=int, nargs="*", default=[1, 3, 10, 30])
    p.add_argument("--caps", type=int, nargs="*",
                   default=[1, 2, 3, 4, 5, 6, 8, 10, 11, 15, 20, 21, 25, 30, 40, 50, 51, 55, 60,
                            75, 100, 110, 150, 200, 10**6])
    p.add_argument("--top", type=int, default=10)
    p.add_argument("--batch-photos", type=int, default=300)
    p.add_argument("--limit", type=int, default=0, help="score only the first N records (smoke runs)")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mv learning-curve", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="step", required=True)
    r = sub.add_parser("run", help="the whole experiment: snapshot, score (parallel), report")
    src = r.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="a manifest (snapshotted, never written)")
    src.add_argument("--release", help="a pulled release folder (manifest.sqlite + embeddings/)")
    r.add_argument("--benchmark-db", help="manifest holding the held-out tables (needed with "
                                          "--release; default: --manifest)")
    r.add_argument("--benchmark-dir", help="the benchmark's folder (default: <data>/benchmarks/<name>)")
    r.add_argument("--data-dir", help="default: MV_DATA_DIR or the repo's data folder")
    r.add_argument("--embeddings-root", help="default: <data>/embeddings (with --manifest)")
    r.add_argument("--sources", help=".org observation sources TSV: non-iNat ids (the wrong-photo "
                                     "records) are left out; omit once a release has dropped them")
    r.add_argument("--occurrence", help="iNat occurrence store (default: "
                                        "<data>/occurrence/inat-fungi-na.npz)")
    r.add_argument("--out", required=True)
    r.add_argument("--cache", help="default: <out>/cache")
    r.add_argument("--workers", type=int, default=4)
    r.add_argument("--reuse-snapshot", action="store_true", help="keep <out>/work's snapshot")
    _shared_args(r)
    s = sub.add_parser("score", help="(used by run) one process: score the split or one shard")
    _score_args(s)
    _shared_args(s)
    rp = sub.add_parser("report", help="write the report again from <out>/scores")
    rp.add_argument("--out", required=True)
    return p


def main(argv=None) -> None:
    a = parser().parse_args(argv)
    if a.step == "run":
        run_all(a)
    elif a.step == "score":
        score(a)
    else:
        from .learning_curve_report import report
        out = Path(a.out)
        freq = json.loads((out / "work" / "frequencies.json").read_text(encoding="utf-8"))
        old = out / "report.json"
        prov = json.loads(old.read_text(encoding="utf-8")).get("provenance") if old.is_file() else None
        report(out / "scores", out, freq["pool"], freq["inat"], provenance=prov)


if __name__ == "__main__":
    main()
