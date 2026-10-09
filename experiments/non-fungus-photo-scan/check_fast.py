"""fast_scores ranks the same as identify_vectors (both methods, 40 dev records)."""
import sqlite3, sys
from pathlib import Path
import numpy as np
from mycomap_vision import heldout, identify
import rescore
conn = sqlite3.connect(Path(sys.argv[1]))
heldout.bench_dir = lambda _c, _n: rescore.REAL_BENCH
ids = heldout.benchmark_ids(conn, rescore.BENCH, split="dev")
recs = [r for r in heldout.load_benchmark(conn, rescore.BENCH, ids, "large") if r.photos][:40]
vec = rescore.bench_vectors()
for m in ("nearest", "nearest+mean"):
    ident = identify.Identifier(conn, rescore.BB, m, photo_info=False, model_cache=rescore.OUT / "model-cache")
    same = 0
    for r in recs:
        q = np.stack([vec[p] for p, _s, _p in r.photos if p in vec])
        a = ident.identify_vectors(q, top_k=10)["ranks"]
        b = ident._ranks(rescore.fast_scores(ident, q), 10)
        same += all([c["name"] for c in a[k]] == [c["name"] for c in b[k]] for k in ("species", "genus", "family"))
    print(m, f"{same}/{len(recs)} identical top-10 at every rank", flush=True)
