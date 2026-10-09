"""Join the shard files score.py --shard k/n wrote into scores-<tag>.npz (query order of
prep.pkl, which report.py doesn't need but keeps runs comparable row by row)."""

import sys
from pathlib import Path

import numpy as np

data, tag, n = Path(sys.argv[1]), sys.argv[2], int(sys.argv[3])
parts = [np.load(data / f"scores-{tag}.shard{k}of{n}.npz") for k in range(n)]
order = np.argsort(np.concatenate([np.arange(k, k + n * len(p["oid"]), n)[:len(p["oid"])]
                                   for k, p in enumerate(parts)]), kind="stable")
out = {key: np.concatenate([p[key] for p in parts])[order] for key in parts[0].files}
np.savez_compressed(data / f"scores-{tag}.npz", **out)
print(f"scores-{tag}.npz: {len(out['oid']):,} queries from {n} shards")
