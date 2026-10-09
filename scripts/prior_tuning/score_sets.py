"""Score the held-out dev and test records with nearest and nearest+mean, CPU only,
against a SNAPSHOT of the manifest (never the shared one). Writes per-record full
species score vectors + metadata (true .org coordinates inside: PRIVATE, scratch only).

usage: python score_sets.py <snapshot.sqlite> <out_dir> [--limit N]
"""
import hashlib
import os
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # the GPU belongs to observation-sets
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
sys.modules["torch"] = None   # CPU numpy only: torch + OpenBLAS threads crashed (access violation)

import numpy as np

from mycomap_vision import heldout, config
from mycomap_vision.evaluate import build_index, load_records, with_rows
from mycomap_vision.identify import Specimens
from mycomap_vision.methods import Scorer, reference_rows, species_means, top_k_species_scores
from mycomap_vision.embed import load_embeddings

BENCH = config.DATA_DIR / "benchmarks" / "heldout-2026-10-08"
NAME = "heldout-2026-10-08"
BACKBONE = "bioclip-2-ft-20261007-165400"
EXPECTED_REF_HASH = "5dbfdb1d24a5"
heldout.bench_dir = lambda conn, name: BENCH          # read the real benchmark folder


def top2_species(sims: np.ndarray, starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Same as methods.top_k_species_scores(sims, index, 2), in O(N): the mean of each
    photo's two best matches per species (one when the species has one photo)."""
    best = np.maximum.reduceat(sims, starts, axis=1)
    seg = np.repeat(np.arange(len(starts)), counts)
    is_best = sims == best[:, seg]
    n_best = np.add.reduceat(is_best.astype(np.int32), starts, axis=1)
    masked = np.where(is_best, -np.inf, sims)
    second = np.maximum.reduceat(masked, starts, axis=1)
    second = np.where(n_best >= 2, best, second)
    two = np.where(counts >= 2, (best + second) / 2, best)
    return two.mean(axis=0), best.mean(axis=0)


def main():
    snap, out = Path(sys.argv[1]), Path(sys.argv[2])
    limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
    out.mkdir(parents=True, exist_ok=True)
    conn = __import__("sqlite3").connect(snap)
    t0 = time.time()
    ids, vecs = load_embeddings(conn, BACKBONE, config.DATA_DIR / "embeddings" / BACKBONE)
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    records = with_rows(load_records(conn, {int(p): int(p) for p in ids.tolist()}), row_of)
    index = build_index(records)
    # The reference hash, as heldout.reference_summary makes it.
    col = Specimens(records, index.cols.tolist())
    obs = [str(o) for o in col.obs.tolist()]
    lines = sorted(f"{o}\t" + "\t".join(str(col.names[i]) for i in col.taxa[r])
                   for r, o in enumerate(obs))
    h = hashlib.sha1(f"{BACKBONE}|".encode())
    h.update("\n".join(lines).encode())
    ref_hash = h.hexdigest()[:12]
    print(f"reference: {len(records):,} records, {len(index.cols):,} photos, "
          f"{len(index.species):,} groups, hash {ref_hash} (expected {EXPECTED_REF_HASH}) "
          f"[{time.time() - t0:.0f}s]", flush=True)
    if ref_hash != EXPECTED_REF_HASH:
        print("WARNING: reference differs from the one the benchmark answers used", flush=True)
    ref_obs = set(obs)
    ref_photos = set(int(p) for p in ids[index.cols].tolist())
    # float32 once (1.8 GB): Scorer widens float16 blocks on every query otherwise (2 s each).
    scorer = Scorer(np.ascontiguousarray(vecs[index.cols], dtype=np.float32))
    # methods.species_means, as a plain loop: np.add.reduceat there crashed this process
    # (access violation, 4 of 4 runs).
    ref = scorer.ref
    starts_ = index.starts.tolist() + [len(index.cols)]
    sums = np.empty((len(index.species), ref.shape[1]), dtype=np.float32)
    for j in range(len(index.species)):
        sums[j] = ref[starts_[j]:starts_[j + 1]].sum(axis=0)
    sums /= np.linalg.norm(sums, axis=1, keepdims=True).clip(1e-12)
    means = Scorer(sums.astype(np.float16))
    del vecs
    starts = index.starts
    counts = np.diff(np.append(starts, len(index.cols)))
    print(f"fitted [{time.time() - t0:.0f}s]", flush=True)

    bench = heldout.load_benchmark(conn, NAME, None, "large")
    qv = heldout.load_vectors(conn, NAME, BACKBONE, "large")
    labeller = heldout.Labeller(conn, extra=[r.truth_name for r in bench if r.truth_name])
    for split in ("dev", "test"):
        recs = [r for r in bench if r.split == split and r.photos]
        if limit:
            recs = recs[:limit]
        rows = {k: [] for k in ("oid", "near", "nm", "nphotos", "maxsim")}
        meta = []
        skipped = {}
        cache = {}

        def prefetch(start):
            """Similarities for the next 32 usable records in one matrix product."""
            block, rows_ = [], []
            for rec in recs[start:start + 32]:
                pids = [p for p, _s, _p in rec.photos]
                v = [qv[p] for p in pids if p in qv]
                if v and rec.observation_id not in ref_obs and not (ref_photos & set(pids)):
                    rows_.append((rec.observation_id, len(block), len(block) + len(v)))
                    block.extend(v)
            cache.clear()
            if block:
                allsims = scorer.sims(np.stack(block))
                msims = means.sims(np.stack(block))
                for oid, a, b in rows_:
                    cache[oid] = (allsims[a:b], msims[a:b])
        for i, rec in enumerate(recs):
            if i % 32 == 0:
                prefetch(i)
            if rec.observation_id in ref_obs:
                skipped["in reference"] = skipped.get("in reference", 0) + 1
                continue
            pids = [p for p, _s, _p in rec.photos]
            if ref_photos & set(pids):
                skipped["photo in reference"] = skipped.get("photo in reference", 0) + 1
                continue
            v = [qv[p] for p in pids if p in qv]
            if not v:
                skipped["no vectors"] = skipped.get("no vectors", 0) + 1
                continue
            q = np.stack(v)
            sims, msims = cache[rec.observation_id]
            two, near = top2_species(sims, starts, counts)
            if i < 3:      # check the O(N) top-2 against the method's own
                ref2 = top_k_species_scores(sims, index, 2)
                assert np.allclose(ref2, two, atol=1e-6), "top-2 mismatch"
            nm = 0.6 * two + 0.4 * msims.mean(axis=0)
            rows["oid"].append(rec.observation_id)
            rows["near"].append(near.astype(np.float32))
            rows["nm"].append(nm.astype(np.float32))
            rows["nphotos"].append(len(v))
            rows["maxsim"].append(float(sims.max()))
            t = labeller.truth(rec.truth_name) if rec.truth_name else None
            meta.append((rec, t))
            if (i + 1) % 250 == 0:
                print(f"  {split}: {i + 1:,}/{len(recs):,} [{time.time() - t0:.0f}s]", flush=True)
        S = lambda xs: np.array(["" if x is None else str(x) for x in xs], dtype=str)
        F = lambda xs: np.array([np.nan if x is None else float(x) for x in xs])
        g = lambda rank: S(index.labels[rank][j] if j >= 0 else ""
                           for j in index.label_of[rank].tolist())
        np.savez_compressed(
            out / f"{split}-scores.npz",
            observation_id=S(rows["oid"]), species=S(index.species),
            nearest=np.stack(rows["near"]), nearest_mean=np.stack(rows["nm"]),
            n_photos=np.array(rows["nphotos"]), max_sim=np.array(rows["maxsim"]),
            truth_name=S(r.truth_name for r, _ in meta),
            truth=S(t.species if t else "" for _, t in meta),
            truth_genus=S(t.genus if t else "" for _, t in meta),
            truth_family=S(t.family if t else "" for _, t in meta),
            truth_guest=np.array([bool(t and t.guest) for _, t in meta]),
            org_lat=F(r.org_latitude for r, _ in meta), org_lng=F(r.org_longitude for r, _ in meta),
            inat_lat=F(r.inat_latitude for r, _ in meta),
            inat_lng=F(r.inat_longitude for r, _ in meta),
            observed_on=S(r.observed_on for r, _ in meta), uuid=S(r.uuid for r, _ in meta),
            user_id=S(r.user_id for r, _ in meta),
            group_genus=g("genus"), group_family=g("family"),
            group_is_species=(index.label_of["species"] >= 0),
            ref_count=np.array([index.ref_count[s] for s in index.species]),
            ref_hash=np.array(ref_hash))
        print(f"{split}: {len(rows['oid']):,} scored, skipped {skipped} "
              f"[{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
