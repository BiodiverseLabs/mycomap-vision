"""Held-out DEV with the flagged reference records left out (exp/loo-mislabel-scan).

    python scripts/loo_dev_effect.py --scan DIR/scan --out DIR \
        --remove top1pct=DIR/remove-top1pct.txt --remove top2pct=DIR/remove-top2pct.txt

Scores every dev record of heldout-2026-10-08 with nearest+mean against the scan's
reference snapshot, unchanged and with each removal list left out (one similarity product
per record serves every variant), and prints Steve's standard tables per variant plus
fixed/broken counts against the unchanged reference. DEV only: the test split is never
loaded. Run with MV_MANIFEST_PATH on the scan's manifest copy and OPENBLAS_NUM_THREADS=1.
"""

from __future__ import annotations

import sys

sys.modules["torch"] = None      # noqa: E402

import argparse
import csv
import json
import pickle
import sqlite3
import time
from collections import Counter
from pathlib import Path

import numpy as np

BENCH = "heldout-2026-10-08"
BACKBONE = "bioclip-2-ft-20261007-165400"
BLOCK_PHOTOS = 48
TOP = 10

_W: dict = {}


def _init(scan: str, variants: dict) -> None:
    from mycomap_vision import loo
    scan = Path(scan)
    with open(scan / "layout.pkl", "rb") as f:
        st = pickle.load(f)
    layout = st["layout"]
    ref = np.load(scan / "ref32.npy", mmap_mode="r")
    sums = np.load(scan / "sums.npy")
    pos = {o: i for i, o in enumerate(st["records"]["observation_id"])}
    _W.update(layout=layout, ref=ref, labels=st["labels"],
              temp=st["meta"]["temperature"], variants={})
    for name, ids in variants.items():
        cols = (np.concatenate([layout.record_cols(pos[o]) for o in ids if o in pos])
                if ids else np.zeros(0, dtype=np.int64))
        s, m = loo.without(ref, layout, sums, cols)
        _W["variants"][name] = (cols, s, m)


def _ranks(unit: np.ndarray) -> dict:
    from mycomap_vision import loo
    lab = _W["labels"]
    us, ug = np.asarray(lab["unit_species"]), np.asarray(lab["unit_genus"])
    out = {"top_k": TOP}
    sp = np.full(len(lab["species"]), -np.inf, dtype=np.float32)
    sp[us[us >= 0]] = unit[us >= 0]
    for rank, scores, names in (("species", sp, lab["species"]),
                                ("genus", loo.genus_scores(unit, ug, len(lab["genus"])),
                                 lab["genus"])):
        conf = loo.softmax_conf(scores, _W["temp"][rank])
        order = [i for i in np.argsort(-scores)[:TOP] if np.isfinite(scores[i])]
        out[rank] = [{"name": names[i], "score": round(float(scores[i]), 4),
                      "confidence": round(float(conf[i]), 4)} for i in order]
    return out


def _score(block):
    from mycomap_vision import loo
    oids, q, owner = block
    ref, layout = _W["ref"], _W["layout"]
    base = np.empty((len(q), ref.shape[0]), dtype=np.float32)
    step = 65_536
    for s in range(0, ref.shape[0], step):
        base[:, s:s + step] = q @ np.asarray(ref[s:s + step]).T
    out = {}
    for name, (cols, sums, means) in _W["variants"].items():
        b = loo.score_queries(q, owner, [cols] * len(oids), ref, layout, sums, means,
                              sims=base.copy(), adjust_means=False)
        out[name] = {o: _ranks(b.unit[j]) for j, o in enumerate(oids)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--remove", action="append", default=[], help="name=path of ids")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--bench", type=Path, default=None,
                    help="the benchmark folder (default: data/benchmarks/heldout-2026-10-08)")
    a = ap.parse_args()
    import multiprocessing as mp

    from mycomap_vision import config, heldout, heldout_report, heldout_summary, name_equiv
    from mycomap_vision.embed import load_embeddings
    conn = sqlite3.connect(str(config.MANIFEST_PATH))
    bench = a.bench or config.DATA_DIR / "benchmarks" / BENCH
    with open(bench / "dev.csv", encoding="utf-8") as f:
        dev_ids = [r["observation_id"] for r in csv.DictReader(f)]
    records = heldout.load_benchmark(conn, BENCH, dev_ids, "large")
    if any(r.split not in (None, "dev") for r in records):
        raise SystemExit("a non-dev record was loaded")
    edb = sqlite3.connect(f"file:{bench / 'embeddings' / 'large' / 'index.sqlite'}?mode=ro",
                          uri=True)
    pids, vecs = load_embeddings(edb, BACKBONE, bench / "embeddings" / "large" / BACKBONE)
    edb.close()
    vec_of = {int(p): i for i, p in enumerate(pids.tolist())}
    with open(a.scan / "layout.pkl", "rb") as f:
        st = pickle.load(f)
    ref_obs = set(st["records"]["observation_id"])
    ref_photos = {p for ps in st["records"]["photo_ids"] for p in ps}
    meta = st["meta"]
    stats, todo = Counter(), []
    for r in records:
        pids_r = [p for p, _s, _p in r.photos]
        if not r.truth_name:
            stats["no answer key"] += 1
        elif r.observation_id in ref_obs:
            stats["already a reference record (skipped)"] += 1
        elif ref_photos & set(pids_r):
            stats["photo also a reference photo (skipped)"] += 1
        elif not any(p in vec_of for p in pids_r):
            stats["no vectors"] += 1
        else:
            todo.append((r.observation_id, np.stack([vecs[vec_of[p]] for p in pids_r
                                                      if p in vec_of]).astype(np.float32)))
    stats["scored"] = len(todo)
    print(json.dumps(dict(stats)), flush=True)
    blocks, cur, n = [], [], 0
    for oid, q in todo:
        cur.append((oid, q))
        n += len(q)
        if n >= BLOCK_PHOTOS:
            blocks.append(cur)
            cur, n = [], 0
    if cur:
        blocks.append(cur)
    blocks = [([o for o, _ in b], np.concatenate([q for _, q in b]),
               np.concatenate([np.full(len(q), j) for j, (_, q) in enumerate(b)]))
              for b in blocks]
    variants = {"unchanged": []}
    for spec in a.remove:
        name, path = spec.split("=", 1)
        variants[name] = [x for x in Path(path).read_text().split() if x]
    results = {v: {} for v in variants}
    t0 = time.time()
    with mp.get_context("spawn").Pool(a.workers, initializer=_init,
                                      initargs=(str(a.scan), variants)) as pool:
        for i, out in enumerate(pool.imap_unordered(_score, blocks), 1):
            for v, res in out.items():
                results[v].update(res)
            if i % 20 == 0:
                print(f"  {i}/{len(blocks)} blocks, {(time.time() - t0) / 60:.1f} min", flush=True)
    # Standard tables (heldout_summary), depth bands from the unchanged reference.
    labeller = heldout.Labeller(conn, extra=[r.truth_name for r in records if r.truth_name])
    truths = {r.observation_id: labeller.truth(r.truth_name) for r in records if r.truth_name}
    sp_count = Counter(st["records"]["species"])
    feats = {o: {"species reference records":
                 heldout_report.bucket(sp_count.get(t.species, 0), heldout_report.DEPTH_BUCKETS)
                 if t.species else "n/a"} for o, t in truths.items()}
    report = {"benchmark": BENCH, "split": "dev", "reference_hash": meta["reference_hash"],
              "method": "nearest+mean", "records": stats, "variants": {},
              "code_version": config.code_version(),
              "reproducibility": "exploratory-pre-freeze"}
    judged_all = {}
    for v, res in results.items():
        judged = {o: heldout_report.judge(r, truths[o], labeller, False)
                  for o, r in res.items() if o in truths}
        judged_all[v] = judged
        report["variants"][v] = {
            "removed_records": len(variants[v]),
            **heldout_summary.ladder(res, set(res), truths, labeller, False, name_equiv),
            "species_by_true_species_reference_records":
                heldout_summary.by_reference_depth(judged, feats)}
    base = judged_all["unchanged"]
    for v, judged in judged_all.items():
        fb = {}
        for rank in ("species", "genus"):
            both = [o for o in judged if rank in judged[o] and rank in base.get(o, {})]
            fb[rank] = {"fixed": sum(judged[o][rank][0] and not base[o][rank][0] for o in both),
                        "broken": sum(base[o][rank][0] and not judged[o][rank][0] for o in both),
                        "n": len(both)}
        report["variants"][v]["vs_unchanged"] = fb
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "dev-effect.json").write_text(json.dumps(report, indent=2, default=str))
    lines = [f"{BENCH} / dev (development, not sealed), nearest+mean, reference "
             f"{meta['reference_hash']}; {stats['scored']:,} records scored"]
    for v, m in report["variants"].items():
        lines += ["", f"{v}  (removed {m['removed_records']:,} reference records; "
                      f"n = {m['records']:,} records)"]
        heldout_summary._table(lines, m)
        lines.append("    species, by the true species' reference records (unchanged reference)")
        lines.append(f"    {'':<24}{'top 1':>8}{'top 5':>8}{'n':>9}")
        for bucket, row in m["species_by_true_species_reference_records"].items():
            lines.append(f"    {bucket + ' records':<24}{heldout_summary._cell(row['top1'])}"
                         f"{heldout_summary._cell(row['top5'])}{row['n']:>9,}")
        for rank, fb in m["vs_unchanged"].items():
            lines.append(f"    {rank} top-1 vs unchanged: {fb['fixed']} fixed, "
                         f"{fb['broken']} broken (n = {fb['n']:,})")
    text = "\n".join(lines)
    (a.out / "dev-effect.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
