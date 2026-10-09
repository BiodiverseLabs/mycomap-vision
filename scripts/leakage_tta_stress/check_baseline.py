"""Check: the base condition reproduces the stored held-out predictions made on the same
reference (heldout_predictions at the snapshot's reference hash), method by method."""

from __future__ import annotations

import json
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

data = Path(sys.argv[1] if len(sys.argv) > 1 else "data/stress")
tag = sys.argv[2] if len(sys.argv) > 2 else "main"
prep = pickle.load(open(data / "prep.pkl", "rb"))
z = np.load(data / f"scores-{tag}.npz")
conn = sqlite3.connect(f"file:{data / 'manifest-snapshot.sqlite'}?mode=ro", uri=True)
labels = prep["labels"]
for method, place in (("nearest", "-"), ("nearest+mean", "-"), ("nearest+prior", "org")):
    key = f"{method}@org" if place == "org" else method
    stored = {oid: json.loads(js) for oid, js in conn.execute(
        "select observation_id, result_json from heldout_predictions where benchmark = "
        "'heldout-2026-10-08' and backbone = 'bioclip-2-ft-20261007-165400' and method = ? "
        "and place = ? and reference_hash = ?", (method, place, prep["ref_hash"]))}
    for rank in ("species", "genus"):
        mine = z[f"top|base|{key}|{rank}"]
        same1 = same10 = n = 0
        for i, oid in enumerate(z["oid"]):
            if oid not in stored:
                continue
            theirs = [c["name"] for c in stored[oid][rank]][:10]
            ours = [labels[rank][j] for j in mine[i] if j >= 0][:len(theirs)]
            n += 1
            same1 += bool(theirs) and bool(ours) and theirs[0] == ours[0]
            same10 += theirs == ours
        print(f"{key:20s} {rank:8s} n={n:5d}  top-1 same {same1 / max(n, 1):.4f}  "
              f"whole top-10 list same {same10 / max(n, 1):.4f}")
