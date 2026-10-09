"""rescore's batched scoring ranks the same as identify_vectors: both methods, the first
`n` dev records (default 8), on the CPU path. Also checks that the batched float32
similarities equal the served scorer's.
Usage: python check_fast.py <scratch manifest copy> [n]
"""
import sqlite3
import sys

# CPU only (see rescore.py).
sys.modules["torch"] = None
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

from mycomap_vision import heldout, identify  # noqa: E402
from mycomap_vision.serving import map_embeddings  # noqa: E402

import rescore  # noqa: E402

conn = sqlite3.connect(Path(sys.argv[1]))
n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
heldout.bench_dir = lambda _c, _n: rescore.REAL_BENCH
ids = heldout.benchmark_ids(conn, rescore.BENCH, split="dev")
recs = [r for r in heldout.load_benchmark(conn, rescore.BENCH, ids, "large") if r.photos][:n]
vec = rescore.bench_vectors()
_ids, mapped = map_embeddings(conn, rescore.BB)
# Ties: duplicate photos at a species' best score, checked against the method's own sort.
from types import SimpleNamespace  # noqa: E402

from mycomap_vision.methods import top_k_species_scores  # noqa: E402

rng = np.random.default_rng(0)
counts = np.array([1, 2, 3, 5, 2, 1, 4])
starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
sims = rng.random((3, counts.sum())).astype(np.float32)
sims[:, 2] = sims[:, 1] = sims[:, 1:3].max(axis=1)[:, None].ravel()   # a tie at species 1
sims[0, 3:6] = 0.9                                                         # a triple tie
fake = SimpleNamespace(starts=starts, group=np.repeat(np.arange(len(counts)), counts),
                       two=counts >= 2, means=np.zeros((len(counts), 1), np.float32), k=2,
                       weight=1.0)
want = top_k_species_scores(sims, SimpleNamespace(starts=starts, cols=np.arange(counts.sum())), 2)
got = rescore.scores_from_sims(fake, sims, np.zeros((3, 1), np.float32), "nearest+mean")
print("ties: top-2 equal to the sort:", bool(np.allclose(want, got)), flush=True)

for m in ("nearest", "nearest+mean"):
    ident = identify.Identifier(conn, rescore.BB, m, photo_info=False,
                                model_cache=rescore.OUT / "model-cache")
    ref = np.asarray(mapped[ident.index.cols], dtype=np.float32)
    prep = rescore.Prepared(ident) if m == "nearest+mean" else SimpleNamespace(
        starts=ident.index.starts)
    same, worst = 0, 0.0
    for r in recs:
        q = np.stack([vec[p] for p, _s, _p in r.photos if p in vec]).astype(np.float32)
        served = ident.nearest.photo_sims(q)
        batched = q @ ref.T
        worst = max(worst, float(np.abs(served - batched).max()))
        # identify_vectors ranks these same scores; it also re-scores every photo alone,
        # which on the CPU path costs minutes per record for nearest+mean.
        a = ident._ranks(ident._species_scores(q, None), 10)
        b = ident._ranks(rescore.scores_from_sims(prep, batched, q, m), 10)
        same += all([c["name"] for c in a[k]] == [c["name"] for c in b[k]]
                    for k in ("species", "genus", "family"))
    print(f"{m}: {same}/{len(recs)} identical top-10 at every rank; "
          f"largest similarity difference {worst:.2e}", flush=True)
