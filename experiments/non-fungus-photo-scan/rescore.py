"""Re-score the held-out dev split against reference indexes with photos or records left out.

Standalone and read-only toward the shared data: the manifest is a scratch copy (Identifier
and load_records may create tables in it), the benchmark's photo vectors are read with a
read-only connection, nothing is written to heldout_runs / heldout_predictions or the
benchmark's answers. The index is rebuilt in memory for each variant by filtering what
evaluate.load_records returns, exactly where the served index gets its records.

Output (private): data/audits/non-fungus-scan/rescore-dev.json and .txt.
Usage: python rescore.py <scratch manifest copy> [method ...]
"""
import sys

# CPU only, always: with torch importable, methods.Scorer puts the reference vectors on
# the GPU whenever CUDA is visible (an empty CUDA_VISIBLE_DEVICES does not reach a Windows
# child), and numpy + torch in one process can crash on large CPU matmuls.
sys.modules["torch"] = None
import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

from mycomap_vision import config, heldout, identify
from mycomap_vision.heldout_report import features, judge, paired
from mycomap_vision.heldout_summary import format_summary, standard_summary
from mycomap_vision.heldout import Labeller

from zs_score import OUT

BENCH = "heldout-2026-10-08"
BB = "bioclip-2-ft-20261007-165400"
REAL_BENCH = config.DATA_DIR / "benchmarks" / BENCH
ORIG_LOAD = identify.load_records


def bench_vectors() -> dict[int, np.ndarray]:
    """The benchmark's large-photo vectors for BB, read-only (heldout.load_vectors would
    run the schema script on the benchmark's index)."""
    root = REAL_BENCH / "embeddings" / "large"
    edb = sqlite3.connect((root / "index.sqlite").resolve().as_uri() + "?mode=ro", uri=True)
    shards = sorted(r[0] for r in edb.execute(
        "select distinct shard from embeddings where backbone = ?", (BB,)))
    edb.close()
    out = {}
    for s in shards:
        ids = np.load(root / BB / f"shard-{s:05d}.ids.npy")
        vecs = np.load(root / BB / f"shard-{s:05d}.npy")
        out.update({int(p): vecs[i] for i, p in enumerate(ids.tolist())})
    return out


SHARED_LIST = OUT.parent / "non-inat-reference-records-2026-10-09.tsv"


def mislinked() -> set[str]:
    """The list of reference records that are not iNat records (Mushroom Observer,
    MyCoPortal, .com Sequences: their numeric id was fetched from iNat as if it were an
    iNat id). Default: the list shared with the label audit and the record-sources fix;
    MV_NONINAT_LIST names another (run_all.py passes the one its rule built)."""
    import os
    path = Path(os.environ.get("MV_NONINAT_LIST") or SHARED_LIST)
    lines = path.read_text(
        encoding="utf-8").splitlines()[1:]
    return {line.split()[0] for line in lines if line.strip()}


def variants() -> dict[str, tuple[set[int], set[str]]]:
    """name -> (photo ids left out, observation ids left out)."""
    F = pickle.load(open(OUT / "frame.pkl", "rb"))
    C = F["classes"]
    J = [C.index(c) for c in F["junk"]]
    top = F["bioclip-2"].argmax(1)
    bad = mislinked()
    flagged = {int(p) for p in F["photo_id"][np.isin(top, J)]}
    flagged_inat = {int(F["photo_id"][i]) for i in np.flatnonzero(np.isin(top, J))
                    if F["rows"][i][0] not in bad}
    return {
        "current": (set(), set()),
        "zero-shot flagged photos out": (flagged, set()),
        "non-iNat records out": (set(), bad),
        "non-iNat records out + flagged photos out": (flagged_inat, bad),
    }


# variant -> the variant it is compared with
BASE_OF = {"zero-shot flagged photos out": "current", "non-iNat records out": "current",
           "non-iNat records out + flagged photos out": "non-iNat records out"}


class Prepared:
    """What scoring an index needs, made once per index: its species starts, each reference
    column's species, the species with two or more photos, and the species means widened to
    float32 (methods.Scorer would widen all 18,000 of them again on every call)."""

    def __init__(self, ident):
        self.ident = ident
        self.starts = ident.index.starts
        counts = np.diff(np.append(self.starts, len(ident.index.cols)))
        self.group = np.repeat(np.arange(len(counts)), counts)
        self.two = counts >= 2
        self.means = np.asarray(ident.model.means.ref, dtype=np.float32)
        self.k, self.weight = ident.model.k, ident.model.weight
        assert self.k == 2


def scores_from_sims(prep: Prepared, sims: np.ndarray, query: np.ndarray, method: str) -> np.ndarray:
    """The species scores identify_vectors ranks, from the query's similarities to the
    index's reference photos (columns in ident.index.cols order), without the per-photo
    and specimen work. nearest: mean over photos of each photo's best match in the
    species. nearest+mean: methods.NearestAndMean (k = 2) with the two best matches per
    species from two reduceat passes instead of a full sort (equal on ties: a species with
    two photos at the best score gets that score twice, as the sort gives)."""
    best = np.maximum.reduceat(sims, prep.starts, axis=1)
    if method == "nearest":
        return best.mean(axis=0)
    assert method == "nearest+mean"
    at_best = sims == best[:, prep.group]
    n_best = np.add.reduceat(at_best, prep.starts, axis=1, dtype=np.int32)   # bool would OR
    second = np.maximum.reduceat(np.where(at_best, -np.inf, sims), prep.starts, axis=1)
    second = np.where(n_best >= 2, best, second)
    top2 = np.where(prep.two, (best + second) / 2, best).mean(axis=0)
    mean = (query @ prep.means.T).mean(axis=0)
    return prep.weight * top2 + (1 - prep.weight) * mean


def all_sims(query: np.ndarray, mapped, chunk: int = 65_536) -> np.ndarray:
    """(photos, every stored vector) float32 similarities, widening the memory-mapped float16
    vectors a block at a time (as serving.MappedSelection does), so no float32 copy of the
    whole reference is held."""
    out = np.empty((len(query), mapped.shape[0]), dtype=np.float32)
    for s in range(0, mapped.shape[0], chunk):
        rows = np.arange(s, min(s + chunk, mapped.shape[0]))
        out[:, rows[0]:rows[-1] + 1] = query @ np.asarray(mapped[rows], dtype=np.float32).T
    return out


def use_filter(photos_out: set[int], records_out: set[str]) -> None:
    def load(conn, photo_row, *a, **kw):
        keep = {p: r for p, r in photo_row.items() if p not in photos_out}
        return [r for r in ORIG_LOAD(conn, keep, *a, **kw) if r.observation_id not in records_out]
    identify.load_records = load


def main() -> None:
    conn = sqlite3.connect(Path(sys.argv[1]))
    methods = sys.argv[2:] or ["nearest", "nearest+mean"]
    heldout.bench_dir = lambda _c, _n: REAL_BENCH          # the copy lives elsewhere
    ids = heldout.benchmark_ids(conn, BENCH, split="dev")
    records = [r for r in heldout.load_benchmark(conn, BENCH, ids, "large") if r.photos]
    vectors = bench_vectors()
    labeller = Labeller(conn, extra=[r.truth_name for r in records if r.truth_name])
    truths = {r.observation_id: labeller.truth(r.truth_name) for r in records if r.truth_name}
    truths = {o: t for o, t in truths.items() if not t.guest}
    cache = OUT / "model-cache"
    # Every variant's index is a subset of the same stored vectors (map_embeddings order), so
    # each batch of dev photos is scored against all of them once, in float32 like
    # serving.MappedSelection, and each variant takes its own columns.
    from mycomap_vision.serving import map_embeddings
    _ids, mapped = map_embeddings(conn, BB)
    idents, refinfo, feats = {}, {}, None
    for vname, (p_out, r_out) in variants().items():
        use_filter(p_out, r_out)
        ident = identify.Identifier(conn, BB, "nearest+mean", photo_info=False, model_cache=cache)
        idents[vname] = (Prepared(ident), r_out)
        refinfo[vname] = {"records": ident.records, "photos": int(len(ident.col_photo)),
                          "species": len(ident.index.species),
                          "reference_hash": heldout.reference_summary(conn, ident)["hash"]}
        print(f"index {vname}: {refinfo[vname]}", flush=True)
        if feats is None:     # depth buckets from today's full index, the same for all
            rc = {"rank_counts": {k: dict(v) for k, v in ident.rank_counts.items()}}
            feats = {r.observation_id: features(r, truths[r.observation_id], rc)
                     for r in records if r.observation_id in truths}
    identify.load_records = ORIG_LOAD
    todo = [(r, np.stack([vectors[p] for p, _s, _p in r.photos if p in vectors]).astype(np.float32))
            for r in records if any(p in vectors for p, _s, _p in r.photos)]
    preds = {(BB, f"{m} | {v}", "", "large"): {} for v in idents for m in methods}
    def score_index(vname, prep, r_out, batch, S):
        """One index's answers for one batch (numpy releases the GIL, so the four indexes
        run side by side)."""
        ident = prep.ident
        Sv = S[:, ident.index.cols]
        at, out = 0, []
        for rec, q in batch:
            sims = Sv[at:at + len(q)]
            at += len(q)
            if rec.observation_id in r_out:
                continue
            for m in methods:
                ranks = ident._ranks(scores_from_sims(prep, sims, q, m), 10)
                out.append(((BB, f"{m} | {vname}", "", "large"), rec.observation_id,
                            heldout.summarise(ranks, top=10)))
        return out

    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(len(idents))
    for b0 in range(0, len(todo), 32):
        batch = todo[b0:b0 + 32]
        Q = np.concatenate([q for _r, q in batch])
        S = all_sims(Q, mapped)
        if b0 == 0:   # the widened means score as the served scorer does
            q0 = batch[0][1]
            for prep, _r in idents.values():
                assert np.allclose(q0 @ prep.means.T, prep.ident.model.means.sims(q0), atol=1e-5)
        for out in pool.map(lambda kv: score_index(kv[0], kv[1][0], kv[1][1], batch, S),
                            list(idents.items())):
            for key, oid, res in out:
                preds[key][oid] = res
        del S
        if (b0 // 32) % 5 == 4 or b0 + 32 >= len(todo):
            print(f"  {min(b0 + 32, len(todo)):,}/{len(todo):,} records", flush=True)
    pool.shutdown()
    judged = {key: {o: judge(res, truths[o], labeller, False) for o, res in by.items()
                    if o in truths} for key, by in preds.items()}
    head = {"benchmark": BENCH, "split": "dev", "sealed": False, "released_at": None,
            "records": len(ids), "scored_records": len(truths)}
    s = standard_summary(head, preds, judged, truths, feats, labeller)
    text = [format_summary(s), "", "Paired (same method, vs the variant in brackets): species / "
            "genus top-1 fixed (+) and broken (-), McNemar p"]
    pairs = []
    for key in judged:
        method, vname = key[1].split(" | ")
        if vname not in BASE_OF:
            continue
        base = (BB, f"{method} | {BASE_OF[vname]}", "", "large")
        p = paired(judged, base, key)
        pairs.append({"model": key[1], "vs": base[1], **p})
        text.append(f"  {key[1] + ' [' + BASE_OF[vname] + ']':<85}" + "  ".join(
            f"{rank} +{p[rank]['only_b_right']}/-{p[rank]['only_a_right']} "
            f"(p={p[rank]['mcnemar_p']:.3f})" for rank in ("species", "genus") if rank in p))
    (OUT / "rescore-dev.txt").write_text("\n".join(text), encoding="utf-8")
    (OUT / "rescore-dev.json").write_text(json.dumps(
        {"summary": s, "paired": pairs, "index": refinfo}, indent=2), encoding="utf-8")
    with open(OUT / "rescore-dev-preds.pkl", "wb") as f:
        pickle.dump({"preds": preds, "judged": judged}, f)
    print("\n".join(text))


if __name__ == "__main__":
    main()
