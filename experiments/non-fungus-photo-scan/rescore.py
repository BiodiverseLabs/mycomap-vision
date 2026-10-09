"""Re-score the held-out dev split against reference indexes with photos or records left out.

Standalone and read-only toward the shared data: the manifest is a scratch copy (Identifier
and load_records may create tables in it), the benchmark's photo vectors are read with a
read-only connection, nothing is written to heldout_runs / heldout_predictions or the
benchmark's answers. The index is rebuilt in memory for each variant by filtering what
evaluate.load_records returns, exactly where the served index gets its records.

Output (private): data/audits/non-fungus-scan/rescore-dev.json and .txt.
Usage: python rescore.py <scratch manifest copy> [method ...]
"""
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


def mislinked() -> set[str]:
    """The agreed list of reference records that are not iNat records (Mushroom Observer,
    MyCoPortal, .com Sequences: their numeric id was fetched from iNat as if it were an
    iNat id), shared with the label audit and the record-sources fix."""
    lines = (OUT.parent / "non-inat-reference-records-2026-10-09.tsv").read_text(
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


def fast_scores(ident, query: np.ndarray) -> np.ndarray:
    """The species scores identify_vectors ranks, without the per-photo and specimen work.
    nearest: mean over photos of each photo's best match in the species. nearest+mean: the
    same blend as methods.NearestAndMean (k = 2), with the two best matches per species from
    two reduceat passes instead of a full sort (equal on ties: a species with two photos at
    the best score gets that score twice, as the sort gives)."""
    starts = ident.index.starts
    sims = ident.nearest.photo_sims(query).astype(np.float32)
    best = np.maximum.reduceat(sims, starts, axis=1)
    if ident.method == "nearest":
        return best.mean(axis=0)
    model = ident.model
    assert ident.method == "nearest+mean" and model.k == 2
    counts = np.diff(np.append(starts, sims.shape[1]))
    group = np.repeat(np.arange(len(counts)), counts)
    at_best = sims == best[:, group]
    n_best = np.add.reduceat(at_best, starts, axis=1)
    second = np.maximum.reduceat(np.where(at_best, -np.inf, sims), starts, axis=1)
    second = np.where(n_best >= 2, best, second)
    top2 = np.where(counts >= 2, (best + second) / 2, best).mean(axis=0)
    mean = model.means.sims(query).mean(axis=0)
    return model.weight * top2 + (1 - model.weight) * mean


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
    preds, judged, refinfo, feats = {}, {}, {}, None
    for vname, (p_out, r_out) in variants().items():
        use_filter(p_out, r_out)
        for method in methods:
            ident = identify.Identifier(conn, BB, method, photo_info=False, model_cache=cache)
            key = (BB, f"{method} | {vname}", "", "large")
            refinfo[f"{method} | {vname}"] = {"records": ident.records,
                                              "photos": int(len(ident.col_photo)),
                                              "species": len(ident.index.species),
                                              "reference_hash":
                                                  heldout.reference_summary(conn, ident)["hash"]}
            if feats is None:     # depth buckets from today's full index, the same for all
                ref = {"rank_counts": {k: dict(v) for k, v in ident.rank_counts.items()}}
                feats = {r.observation_id: features(r, truths[r.observation_id], ref)
                         for r in records if r.observation_id in truths}
            by = {}
            for rec in records:
                vecs = [vectors[p] for p, _s, _p in rec.photos if p in vectors]
                if not vecs or rec.observation_id in r_out:
                    continue
                ranks = ident._ranks(fast_scores(ident, np.stack(vecs)), 10)
                by[rec.observation_id] = heldout.summarise(ranks, top=10)
            preds[key] = by
            judged[key] = {o: judge(res, truths[o], labeller, False)
                           for o, res in by.items() if o in truths}
            print(f"{key[1]}: {len(by):,} answered, index {refinfo[key[1]]}", flush=True)
            del ident
    identify.load_records = ORIG_LOAD
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
