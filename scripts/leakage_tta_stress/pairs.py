"""Paired top-1 counts between two models of report runs (fixed / broken, McNemar), by
species reference band. python pairs.py <data> tagA:cond:method tagB:cond:method"""

import pickle
import sqlite3
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from mycomap_vision import heldout
from mycomap_vision.heldout_report import DEPTH_BUCKETS, bucket, judge, mcnemar

data = Path(sys.argv[1])
prep = pickle.load(open(data / "prep.pkl", "rb"))
conn = sqlite3.connect(data / "manifest-snapshot.sqlite") if (data / "manifest-snapshot.sqlite").exists() \
    else sqlite3.connect("data/stress/manifest-snapshot.sqlite")
labeller = heldout.Labeller(conn, extra=[q["truth_name"] for q in prep["queries"] if q["truth_name"]])
truth = {q["oid"]: q["truth"] for q in prep["queries"]}
labels = prep["labels"]["species"]


def hits(spec):
    tag, cond, method = spec.split(":")
    z = np.load(data / f"scores-{tag}.npz")
    top = z[f"top|{cond}|{method}|species"]
    out = {}
    for i, oid in enumerate(z["oid"].tolist()):
        t = truth.get(oid)
        if t is None or not t.species:
            continue
        out[oid] = judge({"species": [{"name": labels[j]} for j in top[i] if j >= 0]}, t,
                         labeller, False)["species"][0]
    return out, dict(zip(z["oid"].tolist(), z["fact|truth_refs"].tolist()))


a, refs = hits(sys.argv[2])
for spec in sys.argv[3:]:
    b, _ = hits(spec)
    c = Counter()
    for oid in a:
        band = bucket(int(refs[oid]), DEPTH_BUCKETS)
        for key in ("all", band):
            c[(key, "fixed")] += (not a[oid]) and b[oid]
            c[(key, "broken")] += a[oid] and not b[oid]
    print(f"{spec} vs {sys.argv[2]}:")
    for key in ["all"] + [x[2] for x in DEPTH_BUCKETS]:
        f, br = c[(key, "fixed")], c[(key, "broken")]
        print(f"   {key:6s} +{f} / -{br}  net {f - br:+d}  p {mcnemar(f, br):.4f}")
