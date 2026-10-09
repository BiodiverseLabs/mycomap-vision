"""Observation sets: score a record's photos as a set (experiment, exp/observation-sets).

Vision scores photos one at a time and combines them by fixed rules (methods.py):
`nearest` is each query photo's best match within a species, averaged over the photos;
`nearest+mean` blends each photo's two best matches with the species' average vector.
An observation is a SET of views (cap, gills, stipe, habitat, slip), and so is every
reference record. This module scores at set level, with no training:

    nearest            (a) "query coverage": the mean over the query's photos of each
                       photo's best match among the species' reference photos. That is
                       exactly what `nearest` computes, so (a) IS nearest.
    obs-forward        (b) per REFERENCE OBSERVATION R: the mean over query photos of
                       each one's best match within R (all photos must find their match
                       in one record, not in a different record each); species = its
                       best observation.
    obs-chamfer        (b) bidirectional: (forward + backward) / 2, backward being the
                       mean over R's photos of each one's best match among the query
                       photos; species = its best observation.
    obs-chamfer-top2   the same, species = mean of its two best observations (a species
                       with one record keeps that one), the record-level version of
                       nearest+mean's top-2.
    blend:<m>@w        (c) w x nearest+mean + (1 - w) x <m>.

Everything is computed from the stored photo vectors with torch, on the GPU when there
is one. The reference matrix is scored in float16 like methods.Scorer on the GPU, so
nearest and nearest+mean reproduce the stored held-out answers.

Results are judged with the held-out report's own rules (heldout_report.judge, the
answer key = the .com observation name, name_equiv for s.l. / complex) and printed in
the standard summary format (heldout_summary). The manifest is opened read-only and
nothing is written to it: answers go to a JSON / CSV under the benchmark's reports.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import evaluate, heldout, heldout_report, heldout_summary
from .methods import species_means

BACKBONE = "bioclip-2-ft-20261007-165400"
NM_K, NM_WEIGHT = 2, 0.6            # methods.NearestAndMean
STEP1 = ("nearest", "nearest+mean", "obs-forward", "obs-chamfer", "obs-chamfer-top2")
BLEND_WEIGHTS = (0.25, 0.5, 0.75)


def read_only(path: Path) -> sqlite3.Connection:
    """The manifest as it is, read-only: any write raises."""
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


# --- the reference, as photos grouped by record grouped by species ----------------------

@dataclass
class Reference:
    backbone: str
    index: evaluate.Index
    vectors: np.ndarray        # (P, d) float16 in index.cols order (species, then record)
    photo_ids: np.ndarray      # (P,)
    col_rec: np.ndarray        # (P,) record number of each column; a record's columns are contiguous
    rec_unit: np.ndarray       # (R,) the index group (species or one-word unit) of each record
    rec_obs: np.ndarray        # (R,) iNat observation id
    rec_validated: np.ndarray  # (R,) validation date ('' when unknown)
    rec_observer: np.ndarray   # (R,)
    source_records: list | None = None   # the evaluate.Records, in index order (for +prior)

    @property
    def records(self) -> int:
        return len(self.rec_obs)


def build_reference(backbone: str, ids: np.ndarray, vecs: np.ndarray,
                    records: list[evaluate.Record]) -> Reference:
    """`records` with photo_rows indexing `vecs` (evaluate.with_rows)."""
    index = evaluate.build_index(records)
    # build_index lays out each group's records in turn, each record's photos in a run
    # (a photo row can belong to two records, so rows don't name records): follow it.
    by_unit: dict[str, list] = {}
    for r in records:
        by_unit.setdefault(r.unit, []).append(r)
    recs = [r for s in index.species for r in by_unit[s]]
    cols = np.fromiter((row for r in recs for row in r.photo_rows), dtype=np.int64)
    if not np.array_equal(cols, index.cols):
        raise ValueError("the reference's columns differ from evaluate.build_index's")
    col_rec = np.repeat(np.arange(len(recs), dtype=np.int64), [len(r.photo_rows) for r in recs])
    unit_of = {s: i for i, s in enumerate(index.species)}
    return Reference(
        backbone, index, np.ascontiguousarray(vecs[index.cols]), ids[index.cols],
        col_rec, np.array([unit_of[r.unit] for r in recs], dtype=np.int64),
        np.array([r.observation_id for r in recs]),
        np.array([r.validated_on or "" for r in recs]),
        np.array([r.observer or "" for r in recs]), recs)


def load_reference(conn: sqlite3.Connection, backbone: str = BACKBONE,
                   exclude: set[str] | None = None) -> Reference:
    """Every record Vision would serve for `backbone` (as identify.Identifier builds it),
    less the observation ids in `exclude` (e.g. the label audit's records whose ids were
    not iNat ids, so their photos are of some other iNat observation)."""
    from .embed import load_embeddings
    ids, vecs = load_embeddings(conn, backbone)
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    records = evaluate.with_rows(evaluate.load_records(conn, {int(p): int(p) for p in ids.tolist()}),
                                 row_of)
    if exclude:
        records = [r for r in records if r.observation_id not in exclude]
    return build_reference(backbone, ids, vecs, records)


def read_id_list(path: Path) -> set[str]:
    """One observation id per line (blank lines and '#' comments ignored)."""
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def reference_hash(ref: Reference) -> str:
    """The reference's identity by heldout.reference_summary's rule: sha1 of the backbone and
    each record's observation id with its species, genus and family labels, sorted. Two runs
    with the same hash answered from the same records under the same labels."""
    import hashlib
    idx = ref.index

    def lab(rank, u):
        i = idx.label_of[rank][u]
        return idx.labels[rank][i] if i >= 0 else ""
    lines = sorted(f"{o}\t{lab('species', u)}\t{lab('genus', u)}\t{lab('family', u)}"
                   for o, u in zip(ref.rec_obs.tolist(), ref.rec_unit.tolist()))
    h = hashlib.sha1(f"{ref.backbone}|".encode())
    h.update("\n".join(lines).encode())
    return h.hexdigest()[:12]


def file_sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_manifest(src: Path, dest: Path) -> dict:
    """A consistent copy of a (live) manifest, read through a read-only connection with
    SQLite's backup API. Put it in the same folder as the source: the taxonomy cache and the
    benchmark folders are found beside the manifest."""
    dest = Path(dest)
    if dest.exists():
        raise FileExistsError(f"{dest} exists: a snapshot is never overwritten")
    s = read_only(src)
    d = sqlite3.connect(dest)
    try:
        s.backup(d)
    finally:
        d.close()
        s.close()
    return {"source": str(src), "snapshot": str(dest), "taken_at": heldout.now_iso(),
            "sha256": file_sha256(dest), "bytes": dest.stat().st_size}


# --- set scores (torch, on the GPU when there is one) -------------------------------------

def _torch():
    import torch
    return torch


def seg_max(x, seg, n: int):
    """(Q, N) -> (Q, n): the max of x over each segment id in `seg` (N,)."""
    torch = _torch()
    out = torch.full((x.shape[0], n), float("-inf"), dtype=x.dtype, device=x.device)
    return out.scatter_reduce_(1, seg.expand(x.shape[0], -1), x, "amax", include_self=True)


def seg_mean(x, seg, counts):
    """(N,) -> (n,): the mean of x over each segment."""
    torch = _torch()
    out = torch.zeros(len(counts), dtype=x.dtype, device=x.device)
    return out.index_add_(0, seg, x) / counts.to(x.dtype)


def seg_topk_mean(x, seg, starts, counts, k: int):
    """(Q, N) -> (Q, n): the mean of the k largest x in each contiguous segment (all of
    them when it has fewer). Segments are contiguous and in id order."""
    torch = _torch()
    key = seg.to(torch.float64) * 4.0 - x.to(torch.float64)        # x within [-1, 1]
    ranked = torch.gather(x, 1, torch.argsort(key, dim=1, stable=True))
    total = torch.zeros((x.shape[0], len(counts)), dtype=x.dtype, device=x.device)
    for j in range(k):
        has = counts > j
        total[:, has] += ranked[:, starts[has] + j]
    return total / torch.clamp(counts, max=k).to(x.dtype)


def starts_of(seg: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.bincount(seg, minlength=n)
    return np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64), counts


class SetEngine:
    """Scores a record's photo vectors against the reference: every Step 1 method at once."""

    def __init__(self, ref: Reference, device: str | None = None):
        torch = _torch()
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev = self.device
        n_units = len(ref.index.species)
        self.n_units, self.n_recs = n_units, ref.records
        self.ref = torch.from_numpy(np.asarray(ref.vectors, dtype=np.float16)).to(dev)
        # On the CPU, float16 matmuls are slow and not what serving does: float32 there.
        if dev == "cpu":
            self.ref = self.ref.float()
        means = species_means(ref.vectors, _identity_index(ref))
        self.means = torch.from_numpy(means.astype(np.float16)).to(dev)
        if dev == "cpu":
            self.means = self.means.float()
        col_unit = np.repeat(np.arange(n_units, dtype=np.int64),
                             np.diff(np.append(ref.index.starts, len(ref.index.cols))))
        self.col_unit = torch.from_numpy(col_unit).to(dev)
        self.unit_starts = torch.from_numpy(ref.index.starts.astype(np.int64)).to(dev)
        self.unit_counts = torch.from_numpy(np.diff(np.append(ref.index.starts,
                                                              len(ref.index.cols)))).to(dev)
        self.col_rec = torch.from_numpy(ref.col_rec).to(dev)
        self.rec_photos = torch.from_numpy(np.bincount(ref.col_rec, minlength=ref.records)).to(dev)
        self.rec_unit = torch.from_numpy(ref.rec_unit).to(dev)
        rs, rc = starts_of(ref.rec_unit, n_units)
        self.unit_rec_starts = torch.from_numpy(rs).to(dev)
        self.unit_rec_counts = torch.from_numpy(rc).to(dev)

    def sims(self, query: np.ndarray):
        q = self.torch.from_numpy(np.asarray(query, dtype=np.float16)).to(self.device)
        if self.device == "cpu":
            return q.float() @ self.ref.T
        return (q @ self.ref.T).float()

    def record_sets(self, sims):
        """(forward, backward) per reference observation, each (R,)."""
        forward = seg_max(sims, self.col_rec, self.n_recs).mean(dim=0)
        backward = seg_mean(sims.max(dim=0).values, self.col_rec, self.rec_photos)
        return forward, backward

    def per_unit(self, rec_scores, how: str):
        """Reference-observation scores (R,) -> species scores (n_units,): its best record,
        or the mean of its two best."""
        x = rec_scores[None, :]
        if how == "max":
            return seg_max(x, self.rec_unit, self.n_units)[0]
        return seg_topk_mean(x, self.rec_unit, self.unit_rec_starts, self.unit_rec_counts,
                             int(how[3:]))[0]

    def scores(self, query: np.ndarray) -> dict[str, np.ndarray]:
        """{method: (n_units,) scores, higher better} for one record's photos."""
        sims = self.sims(query)
        q = self.torch.from_numpy(np.asarray(query, dtype=np.float16)).to(self.device)
        if self.device == "cpu":
            q = q.float()
        nearest = seg_max(sims, self.col_unit, self.n_units).mean(dim=0)
        top2 = seg_topk_mean(sims, self.col_unit, self.unit_starts, self.unit_counts,
                             NM_K).mean(dim=0)
        mean = (q @ self.means.T).float().mean(dim=0)
        fwd, bwd = self.record_sets(sims)
        chamfer = (fwd + bwd) / 2
        out = {"nearest": nearest,
               "nearest+mean": NM_WEIGHT * top2 + (1 - NM_WEIGHT) * mean,
               "obs-forward": self.per_unit(fwd, "max"),
               "obs-chamfer": self.per_unit(chamfer, "max"),
               "obs-chamfer-top2": self.per_unit(chamfer, "top2")}
        return {k: v.cpu().numpy().astype(np.float32) for k, v in out.items()}


def _identity_index(ref: Reference):
    """ref.index with cols pointing into ref.vectors (already in cols order)."""
    idx = ref.index
    return evaluate.Index(idx.species, np.arange(len(idx.cols)), idx.starts, idx.ref_count,
                          idx.labels, idx.label_of)


def blend(nm: np.ndarray, other: np.ndarray, weight: float) -> np.ndarray:
    """weight x nearest+mean + (1 - weight) x other. A group that only one side scores
    (-inf on the other) keeps -inf."""
    return weight * nm + (1 - weight) * other


# --- answers, judged by the held-out report's rules ----------------------------------------

def ranked(scores: np.ndarray, index: evaluate.Index, top: int = heldout.TOP_K) -> dict:
    """The top `top` names per rank, in the shape heldout.summarise stores."""
    out = {"top_k": top}
    for rank in heldout.RANKS:
        rs = evaluate.rank_scores(scores, index, rank)
        names = index.labels[rank]
        finite = np.isfinite(rs)
        k = min(top, int(finite.sum()))
        order = np.argsort(-np.where(finite, rs, -np.inf), kind="stable")[:k]
        out[rank] = [{"name": names[i], "score": round(float(rs[i]), 5)} for i in order]
    return out


def query_vectors(conn: sqlite3.Connection, name: str, backbone: str,
                  size: str = "large") -> dict[int, np.ndarray]:
    """The benchmark's own photo vectors, read without writing (heldout.load_vectors
    opens its index for writing)."""
    from .embed import load_embeddings
    folder = heldout.bench_dir(conn, name) / "embeddings" / size
    edb = read_only(folder / "index.sqlite")
    try:
        ids, vecs = load_embeddings(edb, backbone, folder / backbone)
    finally:
        edb.close()
    return {int(p): vecs[i] for i, p in enumerate(ids.tolist())}


def benchmark_queries(conn: sqlite3.Connection, name: str, split: str, ref: Reference,
                      size: str = "large") -> list[tuple[heldout.HeldOutRecord, np.ndarray]]:
    """(record, photo vectors) for each record of `split` that predict would answer:
    has photos, is not a reference record, shares no photo with the reference."""
    ids = heldout.benchmark_ids(conn, name, split=split)
    vectors = query_vectors(conn, name, ref.backbone, size)
    in_ref = set(ref.rec_obs.tolist())
    ref_photos = set(int(p) for p in ref.photo_ids.tolist())
    out = []
    for rec in heldout.load_benchmark(conn, name, ids, size):
        pids = [pid for pid, _s, _p in rec.photos]
        if not pids or rec.observation_id in in_ref or ref_photos & set(pids):
            continue
        vecs = [vectors[p] for p in pids if p in vectors]
        if vecs:
            out.append((rec, np.stack(vecs)))
    return out


def model_key(method: str, backbone: str = BACKBONE, place: str = heldout.NO_PLACE) -> tuple:
    return (backbone, method, place, "large")


def stored_answers(conn: sqlite3.Connection, name: str, ids: set[str]) -> dict[tuple, dict]:
    """The benchmark's stored answers (nearest, nearest+mean, nearest+prior@org)."""
    chosen = heldout_report.chosen_references(conn, name, ids)
    preds, _other, _near = heldout_report.load_predictions(conn, name, ids, chosen)
    return {k: v for k, v in preds.items() if k[0] == BACKBONE}


def rank_counts(ref: Reference) -> dict:
    """Reference records per name at each rank (what heldout.reference_summary stores)."""
    from collections import Counter
    out = {}
    per_unit = np.bincount(ref.rec_unit, minlength=len(ref.index.species))
    for rank in heldout.RANKS:
        c = Counter()
        for u, n in enumerate(per_unit.tolist()):
            lab = ref.index.label_of[rank][u]
            if n and lab >= 0:
                c[ref.index.labels[rank][lab]] += n
        out[rank] = dict(c)
    return out


def judge_all(conn: sqlite3.Connection, name: str, records: list[heldout.HeldOutRecord],
              results: dict[tuple, dict[str, dict]], baseline: tuple,
              ref: Reference | None = None) -> dict:
    """The standard summary (heldout_summary) for every model in `results`, plus paired
    species / genus top-1 counts against `baseline` with McNemar p, overall and by the
    true species' reference records. With `ref`, depth bands count its records (else the
    stored run's reference)."""
    extra = [r.truth_name for r in records if r.truth_name]
    extra += [c["name"] for by in results.values() for res in by.values()
              for rank in heldout.RANKS for c in res.get(rank) or []]
    labeller = heldout.Labeller(conn, extra=extra)
    truths = {r.observation_id: labeller.truth(r.truth_name) for r in records if r.truth_name}
    truths = {o: t for o, t in truths.items() if not t.guest}
    ids = {r.observation_id for r in records}
    chosen = heldout_report.chosen_references(conn, name, ids)
    reference = heldout_report.reference_run(conn, name, BACKBONE, chosen)
    if ref is not None:
        reference = {**(reference or {}), "rank_counts": rank_counts(ref)}
    feats = {r.observation_id: heldout_report.features(r, truths[r.observation_id], reference)
             for r in records if r.observation_id in truths}
    judged = heldout_report.score(results, truths, labeller)
    head = {"benchmark": name, "split": records[0].split if records else None, "sealed": False,
            "released_at": None, "records": len(records), "scored_records": len(truths)}
    summary = heldout_summary.standard_summary(head, results, judged, truths, feats, labeller)
    paired = {}
    for key in results:
        if key == baseline:
            continue
        p = heldout_report.paired(judged, key, baseline)
        bands = {}
        for _lo, _hi, label in heldout_report.DEPTH_BUCKETS:
            sub = {o for o in judged[key] if feats[o]["species reference records"] == label}
            j = {key: {o: judged[key][o] for o in sub if o in judged[baseline]},
                 baseline: {o: judged[baseline][o] for o in sub if o in judged[key]}}
            bp = heldout_report.paired(j, key, baseline).get("species")
            if bp:
                bands[label] = {"n": bp["n"], "fixed": bp["only_a_right"],
                                "broken": bp["only_b_right"], "mcnemar_p": bp["mcnemar_p"]}
        paired[heldout_report.model_name(key)] = {
            rank: {"n": v["n"], "fixed": v["only_a_right"], "broken": v["only_b_right"],
                   "mcnemar_p": v["mcnemar_p"]} for rank, v in p.items()
            if rank in ("species", "genus")} | {"species_by_depth": bands}
    return {"summary": summary, "paired_vs": heldout_report.model_name(baseline),
            "paired": paired, "judged": judged, "feats": feats}


def format_paired(out: dict) -> str:
    lines = [f"Paired top-1 vs {out['paired_vs']} (fixed / broken, McNemar p)"]
    for model, p in out["paired"].items():
        sp, ge = p.get("species"), p.get("genus")
        cell = (lambda d: f"+{d['fixed']}/-{d['broken']} p={d['mcnemar_p']:.3g}" if d else "-")
        bands = "  ".join(f"{b}: +{d['fixed']}/-{d['broken']}"
                          for b, d in p["species_by_depth"].items())
        lines.append(f"  {model}\n    species {cell(sp)}   genus {cell(ge)}\n    {bands}")
    return "\n".join(lines)


def variant_table(out: dict) -> list[dict]:
    """Species and genus top-1 / top-5 (strict) per model, for the every-variant list."""
    rows = []
    for model, m in out["summary"]["models"].items():
        sp = m["rows"]["species strict"]
        ge = m["rows"]["genus strict"]
        rows.append({"model": model, "species_top1": sp["top1"]["rate"],
                     "species_top5": sp["top5"]["rate"], "genus_top1": ge["top1"]["rate"],
                     "n": sp["n"]})
    return rows


def write_report(out: dict, out_dir: Path, stem: str, extra: dict | None = None) -> Path:
    """JSON (summary, paired, variants) and a per-record CSV of top-1 hits, no coordinates."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.json"
    body = {"summary": out["summary"], "paired_vs": out["paired_vs"], "paired": out["paired"],
            "variants": variant_table(out), **(extra or {})}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    import csv
    models = sorted(out["judged"])
    with open(out_dir / f"{stem}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["observation_id", "species_refs", "model", "species_top1", "species_top5",
                    "genus_top1", "top_species"])
        for m in models:
            for oid, ranks in sorted(out["judged"][m].items()):
                sp, ge = ranks.get("species"), ranks.get("genus")
                w.writerow([oid, out["feats"][oid]["species reference records"],
                            heldout_report.model_name(m), int(sp[0]) if sp else "",
                            int(sp[1]) if sp else "", int(ge[0]) if ge else "",
                            sp[2] if sp else ""])
    return path


# --- Step 1 run ------------------------------------------------------------------------

PRIOR_TEMPERATURE = 0.02      # methods.AsLogProb's default, as METHODS["nearest+prior"]


def fit_prior(ref: Reference):
    """The range-and-season prior of `nearest+prior` (prior.RangeSeasonPrior) on `ref`."""
    from .prior import RangeSeasonPrior
    prior = RangeSeasonPrior()
    prior.fit(ref.source_records or [], ref.index.species)
    return prior


def with_prior(nearest: np.ndarray, prior, context) -> np.ndarray:
    """nearest+prior's score: nearest as log-probabilities plus the range-and-season score."""
    from .methods import log_softmax
    return log_softmax(nearest / PRIOR_TEMPERATURE) + prior.log_prior(context)


def run_step1(conn: sqlite3.Connection, name: str, split: str = "dev", backbone: str = BACKBONE,
              out_dir: Path | None = None, stem: str | None = None,
              weights=BLEND_WEIGHTS, limit: int | None = None,
              exclude: set[str] | None = None, log=print) -> dict:
    t0 = time.time()
    ref = load_reference(conn, backbone, exclude)
    log(f"  reference: {ref.records:,} records, {len(ref.photo_ids):,} photos, "
        f"{len(ref.index.species):,} groups ({time.time() - t0:.0f} s)")
    engine = SetEngine(ref)
    prior = fit_prior(ref)
    queries = benchmark_queries(conn, name, split, ref)[:limit]
    log(f"  {split}: {len(queries):,} records to answer")
    results: dict[tuple, dict[str, dict]] = {}

    def put(method, oid, scores):
        results.setdefault(model_key(method, backbone), {})[oid] = ranked(scores, ref.index)
    for i, (rec, q) in enumerate(queries):
        s = engine.scores(q)
        for m, v in s.items():
            put(m, rec.observation_id, v)
        results.setdefault(model_key("nearest+prior", backbone, "org"), {})[rec.observation_id] = \
            ranked(with_prior(s["nearest"], prior, heldout.context_for(rec, "org")), ref.index)
        for m in STEP1[2:]:
            for w in weights:
                put(f"blend:{m}@{w}", rec.observation_id, blend(s["nearest+mean"], s[m], w))
        if (i + 1) % 500 == 0:
            log(f"  {i + 1:,} records ({time.time() - t0:.0f} s)")
    ids = {r.observation_id for r, _ in queries}
    if not exclude:                 # the stored answers were made against the full reference
        stored = stored_answers(conn, name, ids)
        for (b, method, place, size), by in stored.items():
            results[(b, f"{method} (stored)" if place == heldout.NO_PLACE else method,
                     place, size)] = by
    out = judge_all(conn, name, [r for r, _ in queries], results,
                    baseline=model_key("nearest+mean", backbone), ref=ref)
    out["reference"] = {"records": ref.records, "photos": int(len(ref.photo_ids)),
                        "species": len(ref.index.species), "hash": reference_hash(ref),
                        "excluded_ids": len(exclude or ())}
    out["seconds"] = round(time.time() - t0, 1)
    if out_dir is not None:
        stem = stem or f"obsets-step1-{split}-{heldout.now_iso().replace(':', '').replace('-', '')[:15]}"
        out["file"] = str(write_report(out, out_dir, stem, {"seconds": out["seconds"],
                                                            "reference": out["reference"]}))
    return out


# --- an independent check of a blend chosen on dev: the reference's own newest weeks -----

def subset_reference(ref: Reference, keep: np.ndarray) -> Reference:
    """`ref` with only the records numbered in `keep`."""
    from .evaluate import Record
    recs = []
    row = 0
    rows_of = np.split(np.arange(len(ref.col_rec)), np.flatnonzero(np.diff(ref.col_rec)) + 1)
    names = ref.index.species
    labels_of = {s: {rank: (ref.index.labels[rank][ref.index.label_of[rank][u]]
                            if ref.index.label_of[rank][u] >= 0 else "")
                     for rank in ("species", "genus", "family")} for u, s in enumerate(names)}
    vec_rows = []
    for r in sorted(keep.tolist()):
        unit = names[ref.rec_unit[r]]
        lab = labels_of[unit]
        cols = rows_of[r]
        recs.append(Record(str(ref.rec_obs[r]), lab["species"], lab["genus"], lab["family"],
                           str(ref.rec_validated[r]) or None, str(ref.rec_observer[r]),
                           list(range(row, row + len(cols))),
                           taxon="" if lab["species"] else unit))
        vec_rows.append(cols)
        row += len(cols)
    cols = np.concatenate(vec_rows)
    return build_reference(ref.backbone, ref.photo_ids[cols], ref.vectors[cols], recs)


def time_slice_check(ref: Reference, until: str, weights=(0.5, 0.75, 0.9), log=print) -> dict:
    """Score the reference records validated after `until` against those validated up to
    it (the comparison 4ef7b0 layout: 1,152 records after 2026-09-07), by Vision's labels
    (strict), for nearest, nearest+mean, obs-chamfer-top2 and its blends. A blend weight
    chosen on dev should hold up here, where nothing was tuned."""
    dated = ref.rec_validated != ""
    old = np.flatnonzero(~dated | (ref.rec_validated <= until))
    new = np.flatnonzero(dated & (ref.rec_validated > until))
    base = subset_reference(ref, old)
    engine = SetEngine(base)
    unit_of = {s: i for i, s in enumerate(base.index.species)}
    rows_of = np.split(np.arange(len(ref.col_rec)), np.flatnonzero(np.diff(ref.col_rec)) + 1)
    sp_label = base.index.label_of["species"]
    hits: dict[str, list] = {}
    depth: list[str] = []
    for r in new.tolist():
        truth = ref.index.species[ref.rec_unit[r]]
        if ref.index.label_of["species"][ref.rec_unit[r]] < 0:
            continue                                   # a one-word record: no species answer
        s = engine.scores(ref.vectors[rows_of[r]])
        s.pop("obs-forward"), s.pop("obs-chamfer")
        for w in weights:
            s[f"blend:obs-chamfer-top2@{w}"] = blend(s["nearest+mean"], s["obs-chamfer-top2"], w)
        n = base.index.ref_count.get(truth, 0)
        depth.append(heldout_report.bucket(n, heldout_report.DEPTH_BUCKETS))
        for m, v in s.items():
            v = np.where(sp_label >= 0, v, -np.inf)
            hits.setdefault(m, []).append(unit_of.get(truth, -1) == int(np.argmax(v)))
    out = {"until": until, "records": len(depth), "methods": {}}
    nm = np.array(hits["nearest+mean"])
    for m, h in hits.items():
        h = np.array(h)
        fixed, broken = int((h & ~nm).sum()), int((~h & nm).sum())
        out["methods"][m] = {
            "species_top1": round(float(h.mean()), 4), "fixed": fixed, "broken": broken,
            "mcnemar_p": round(heldout_report.mcnemar(fixed, broken), 6),
            "by_depth": {b: round(float(h[np.array(depth) == b].mean()), 4)
                         for _l, _h, b in heldout_report.DEPTH_BUCKETS
                         if (np.array(depth) == b).any()}}
    out["depth_n"] = {b: int((np.array(depth) == b).sum()) for _l, _h, b in heldout_report.DEPTH_BUCKETS}
    return out


# --- the whole experiment as one command ---------------------------------------------------

REPRODUCIBILITY = "exploratory-pre-freeze"


def run_all(manifest: Path, name: str = "heldout-2026-10-08", split: str = "dev",
            exclude_file: Path | None = None, out_dir: Path | None = None,
            log=print) -> dict:
    """Step 1, Step 2 (with the projection, and attention only) and the time-slice check
    on one manifest (a stated snapshot or a release's), with the grid fixed in code
    (BLEND_WEIGHTS, obsets_head.TrainConfig, time_slice_check's weights) and fixed seeds.
    Writes obsets-all-<split>[-clean].json beside the per-step reports: every number with the
    manifest's sha256, the reference hash, the code commit and the excluded list's sha256."""
    from . import config, obsets_head
    torch = _torch()
    manifest = Path(manifest)
    conn = read_only(manifest)
    exclude = read_id_list(exclude_file) if exclude_file else None
    tag = "-clean" if exclude else ""
    out_dir = Path(out_dir) if out_dir else heldout.bench_dir(conn, name) / "reports"
    t0 = time.time()
    prov = {"reproducibility": REPRODUCIBILITY, "code_commit": config.code_version(),
            "manifest": str(manifest), "manifest_sha256": file_sha256(manifest),
            "benchmark": name, "split": split,
            "exclude_file": str(exclude_file) if exclude_file else None,
            "exclude_sha256": file_sha256(Path(exclude_file)) if exclude_file else None,
            "excluded_ids": len(exclude or ()), "blend_weights": list(BLEND_WEIGHTS),
            "head_config": obsets_head.TrainConfig().__dict__,
            "command": (f"mv obsets all --manifest {manifest} --name {name} --split {split}"
                        + (f" --exclude {exclude_file}" if exclude_file else ""))}
    out = {"provenance": prov}
    log(f"== step1 ({time.time() - t0:.0f} s)")
    s1 = run_step1(conn, name, split, out_dir=out_dir, stem=f"obsets-step1-{split}{tag}",
                   exclude=exclude, log=log)
    out["step1"] = {"file": s1["file"], "reference": s1["reference"], "seconds": s1["seconds"]}
    del s1
    torch.cuda.empty_cache()
    for label, project in (("step2", True), ("step2-attnonly", False)):
        log(f"== {label} ({time.time() - t0:.0f} s)")
        s2 = obsets_head.run_step2(conn, name, split,
                                   cfg=obsets_head.TrainConfig(project=project),
                                   out_dir=out_dir, stem=f"obsets-{label}-{split}{tag}",
                                   exclude=exclude, log=log)
        out[label] = {"file": s2["file"], "reference": s2["reference"],
                      "training": {k: v for k, v in s2["training"].items() if k != "history"},
                      "seconds": s2["seconds"]}
        del s2
        torch.cuda.empty_cache()
    log(f"== time slice ({time.time() - t0:.0f} s)")
    ref = load_reference(conn, BACKBONE, exclude)
    out["time_slice"] = time_slice_check(ref, "2026-09-07")
    out["time_slice"]["reference_hash"] = reference_hash(ref)
    del ref
    out["seconds"] = round(time.time() - t0, 1)
    path = out_dir / f"obsets-all-{split}{tag}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    out["file"] = str(path)
    return out
