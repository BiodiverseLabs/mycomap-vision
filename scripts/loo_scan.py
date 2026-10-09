"""Leave-one-out scan of the reference set (exp/loo-mislabel-scan). CPU only.

    python scripts/loo_scan.py prep  --out DIR      # reference -> DIR (vectors, layout)
    python scripts/loo_scan.py score --out DIR --workers 8

Run with MV_MANIFEST_PATH pointing at a COPY of the manifest (the shared one changes
during the day) and OPENBLAS_NUM_THREADS=1. torch is never imported: in the shared
.venv, numpy's OpenBLAS segfaults on big products once torch is loaded.
Outputs stay in DIR (under data/, never committed).
"""

from __future__ import annotations

import sys

sys.modules["torch"] = None      # noqa: E402  (see the module notes)

import argparse
import json
import os
import pickle
import sqlite3
import time
from pathlib import Path

import numpy as np

BACKBONE = "bioclip-2-ft-20261007-165400"
METHOD = "nearest+mean"
BLOCK_PHOTOS = 48
CHUNK_BLOCKS = 40
TOP = 10


def prep(out: Path, embeddings: Path | None = None) -> None:
    from mycomap_vision import config, evaluate, heldout, loo
    from mycomap_vision.identify import Identifier
    from mycomap_vision.serving import map_embeddings
    conn = sqlite3.connect(str(config.MANIFEST_PATH))
    ids, vecs = map_embeddings(conn, BACKBONE, embeddings)
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    records = evaluate.with_rows(evaluate.load_records(conn, {int(p): int(p) for p in
                                                              ids.tolist()}), row_of)
    index = evaluate.build_index(records)
    layout, ordered = loo.build_layout(records, index)
    # The same reference the site and the held-out benchmark use, and its hash.
    ident = Identifier(conn, BACKBONE, "nearest", embeddings_root=embeddings, photo_info=False)
    ref = heldout.reference_summary(conn, ident)
    if not np.array_equal(ids[ident.index.cols], ids[layout.col_rows]):
        raise SystemExit("the scan's columns differ from the Identifier's")
    del ident
    cal = evaluate.latest_calibration(conn, BACKBONE, METHOD)
    ref32 = np.lib.format.open_memmap(out / "ref32.npy", mode="w+", dtype=np.float32,
                                      shape=(len(layout.col_rows), vecs.shape[1]))
    step = 65_536
    for s in range(0, len(layout.col_rows), step):
        ref32[s:s + step] = np.asarray(vecs[layout.col_rows[s:s + step]], dtype=np.float32)
    ref32.flush()
    sums = loo.unit_sums(ref32, layout)
    np.save(out / "sums.npy", sums)
    meta = {
        "backbone": BACKBONE, "method": METHOD, "reference_hash": ref["hash"],
        "records_hash": ref["json"]["records_hash"], "records": len(ordered),
        "photos": int(len(layout.col_rows)), "units": len(layout.units),
        "species": len(index.labels["species"]),
        "manifest_newest_export": ref["json"]["manifest_newest_export"],
        "temperature": cal["temperatures"] if cal else None,
        "calibration_comparison": cal["comparison_id"] if cal else None,
        "leave_out_groups": len(layout.group_recs),
        "rows_in_two_columns": len(layout.same_row_cols),
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_version_prep": config.code_version(),
        "manifest": file_digest(Path(config.MANIFEST_PATH)),
        "embeddings_root": str(embeddings or config.DATA_DIR / "embeddings" / BACKBONE),
    }
    recs = {"observation_id": [r.observation_id for r in ordered],
            "species": [r.species for r in ordered], "genus": [r.genus for r in ordered],
            "family": [r.family for r in ordered], "unit": [r.unit for r in ordered],
            "observer": [r.observer or "" for r in ordered],
            "observed_on": [r.observed_on or "" for r in ordered],
            "projects": [list(r.projects) for r in ordered],
            "photo_ids": [ids[r.photo_rows].tolist() for r in ordered]}
    labels = {"species": index.labels["species"], "genus": index.labels["genus"],
              "family": index.labels["family"],
              "unit_species": index.label_of["species"].tolist(),
              "unit_genus": index.label_of["genus"].tolist(),
              "unit_family": index.label_of["family"].tolist()}
    with open(out / "layout.pkl", "wb") as f:
        pickle.dump({"layout": layout, "records": recs, "labels": labels, "meta": meta}, f)
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def file_digest(path: Path) -> dict:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": h.hexdigest()}


# --- scoring ---------------------------------------------------------------------------

_W: dict = {}


def _init(out: str) -> None:
    out = Path(out)
    with open(out / "layout.pkl", "rb") as f:
        state = pickle.load(f)
    from mycomap_vision import loo
    _W["layout"] = state["layout"]
    _W["labels"] = state["labels"]
    _W["temp"] = (state["meta"]["temperature"] or {}).get("species", 0.02)
    _W["ref"] = np.load(out / "ref32.npy", mmap_mode="r")
    _W["sums"] = np.load(out / "sums.npy")
    _W["means"] = loo.as_served(_W["sums"])
    _W["out"] = out


def blocks_of(layout, n_records: int) -> list[list[int]]:
    blocks, cur, photos = [], [], 0
    for r in range(n_records):
        cur.append(r)
        photos += len(layout.record_cols(r))
        if photos >= BLOCK_PHOTOS:
            blocks.append(cur)
            cur, photos = [], 0
    if cur:
        blocks.append(cur)
    return blocks


def _score_chunk(args) -> tuple[int, int, float]:
    from mycomap_vision import loo
    chunk_id, blocks = args
    path = _W["out"] / "chunks" / f"chunk-{chunk_id:05d}.npz"
    if path.exists():
        return chunk_id, 0, 0.0
    t0 = time.time()
    layout, labels = _W["layout"], _W["labels"]
    unit_species = np.asarray(labels["unit_species"])
    unit_genus = np.asarray(labels["unit_genus"])
    is_sp = unit_species >= 0
    n_gen = len(labels["genus"])
    cols = {k: [] for k in ("rec", "top_unit", "top_score", "top_conf", "own_score",
                            "own_rank", "own_conf", "own_avail", "gen_top", "gen_score",
                            "own_gen_rank", "nb_rec", "nb_score", "best_match", "worst_match",
                            "n_photos")}
    pcols = {k: [] for k in ("rec", "top_unit", "top_score", "own_score", "best_rec",
                             "best_sim", "top_genus")}
    for recs in blocks:
        b = loo.score_records(_W["ref"], layout, recs, _W["sums"], _W["means"])
        for j, r in enumerate(recs):
            s = b.unit[j]
            sp = np.where(is_sp, s, -np.inf)
            order = np.argsort(-sp)[:TOP]
            order = order[np.isfinite(sp[order])]
            conf = loo.softmax_conf(sp[is_sp], _W["temp"])
            conf_all = np.zeros(len(s))
            conf_all[is_sp] = conf
            own = int(layout.rec_unit[r])
            g = loo.genus_scores(s, unit_genus, n_gen)
            g_order = np.argsort(-g)[:3]
            rows = np.flatnonzero(b.photo_owner == j)
            rec_score = b.record_max[rows].mean(axis=0)
            nb = np.argpartition(-rec_score, TOP)[:TOP]
            nb = nb[np.argsort(-rec_score[nb])]
            per_photo_best = b.record_max[rows].max(axis=1)
            cols["rec"].append(r)
            cols["top_unit"].append(np.pad(order, (0, TOP - len(order)), constant_values=-1))
            cols["top_score"].append(np.pad(sp[order], (0, TOP - len(order)),
                                            constant_values=-np.inf))
            cols["top_conf"].append(np.pad(conf_all[order], (0, TOP - len(order))))
            cols["own_score"].append(s[own])
            cols["own_rank"].append(loo.rank_of(sp, own) if is_sp[own] else 0)
            cols["own_conf"].append(conf_all[own])
            cols["own_avail"].append(b.available[j, own])
            cols["gen_top"].append(g_order)
            cols["gen_score"].append(g[g_order])
            og = int(unit_genus[own])
            cols["own_gen_rank"].append(loo.rank_of(g, og) if og >= 0 else 0)
            cols["nb_rec"].append(nb)
            cols["nb_score"].append(rec_score[nb])
            cols["best_match"].append(per_photo_best.mean())
            cols["worst_match"].append(per_photo_best.min())
            cols["n_photos"].append(len(rows))
            for i in rows.tolist():
                ps = np.where(is_sp, b.photo[i], -np.inf)
                t = int(np.argmax(ps))
                pcols["rec"].append(r)
                pcols["top_unit"].append(t)
                pcols["top_score"].append(ps[t])
                pcols["own_score"].append(b.photo[i, own])
                br = int(np.argmax(b.record_max[i]))
                pcols["best_rec"].append(br)
                pcols["best_sim"].append(b.record_max[i, br])
                pcols["top_genus"].append(int(unit_genus[t]))
    arrays = {k: np.asarray(v) for k, v in cols.items()}
    arrays.update({f"photo_{k}": np.asarray(v) for k, v in pcols.items()})
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)
    return chunk_id, sum(len(x) for x in blocks), time.time() - t0


def score(out: Path, workers: int, limit: int | None) -> None:
    import multiprocessing as mp
    with open(out / "layout.pkl", "rb") as f:
        layout = pickle.load(f)["layout"]
    blocks = blocks_of(layout, layout.n_records)
    chunks = [(i, blocks[s:s + CHUNK_BLOCKS])
              for i, s in enumerate(range(0, len(blocks), CHUNK_BLOCKS))]
    if limit:
        chunks = chunks[:limit]
    (out / "chunks").mkdir(exist_ok=True)
    from mycomap_vision import config
    with open(out / "score_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                            "code_version": config.code_version(), "workers": workers}) + "\n")
    todo = [c for c in chunks if not (out / "chunks" / f"chunk-{c[0]:05d}.npz").exists()]
    print(f"{len(blocks):,} blocks in {len(chunks):,} chunks; {len(todo):,} to score "
          f"with {workers} workers", flush=True)
    t0, done_recs = time.time(), 0
    with mp.get_context("spawn").Pool(workers, initializer=_init, initargs=(str(out),)) as pool:
        for n, (cid, nrec, secs) in enumerate(pool.imap_unordered(_score_chunk, todo), 1):
            done_recs += nrec
            el = time.time() - t0
            eta = el / n * (len(todo) - n)
            print(f"chunk {cid} ({nrec} records, {secs:.0f} s); {n}/{len(todo)} chunks, "
                  f"{done_recs:,} records, {el / 60:.1f} min, ETA {eta / 60:.0f} min",
                  flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["prep", "score"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="score only the first N chunks")
    ap.add_argument("--embeddings", type=Path, default=None,
                    help="the backbone's embedding shards (default: data/embeddings/<backbone>)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    if os.environ.get("OPENBLAS_NUM_THREADS") != "1":
        print("note: set OPENBLAS_NUM_THREADS=1 (one BLAS thread per worker)", flush=True)
    prep(a.out, a.embeddings) if a.step == "prep" else score(a.out, a.workers, a.limit)


if __name__ == "__main__":
    main()
