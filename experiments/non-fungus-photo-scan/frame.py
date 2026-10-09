"""One row per reference photo: record, label, the record's photo count and position, the
true species' reference-record count, and the zero-shot class probabilities from both
backbones. Saved as a pickle under data/audits/non-fungus-scan/ (private).

The reference set is exactly what Identifier serves: evaluate.load_records over the
photos with stored vectors (green, unconflicted, North American, permitted, not held out).
Usage: python frame.py <scratch manifest copy>
"""
import pickle
import sqlite3
import sys
from pathlib import Path

import numpy as np

from mycomap_vision.evaluate import load_records

from zs_score import OUT


def main() -> None:
    conn = sqlite3.connect(Path(sys.argv[1]))   # a scratch copy: load_records may create tables
    zs = {bb: np.load(OUT / f"zs-{bb}.npz") for bb in ("bioclip-2", "bioclip-2-ft-20261007-165400")}
    base = zs["bioclip-2"]
    ids = base["photo_id"]
    recs = load_records(conn, {int(p): int(p) for p in ids.tolist()})
    unit_count: dict[str, int] = {}
    for r in recs:
        unit_count[r.unit] = unit_count.get(r.unit, 0) + 1
    pos = {}
    for oid, pid, p in conn.execute("select observation_id, photo_id, position from observation_photos"):
        pos[(oid, pid)] = p
    lic = dict(conn.execute("select photo_id, license_class from photos"))
    rows = []
    for r in recs:
        for k, pid in enumerate(r.photo_rows):
            rows.append((r.observation_id, pid, k, len(r.photo_rows), r.species, r.genus, r.family,
                         r.unit, unit_count[r.unit], pos.get((r.observation_id, pid)),
                         lic.get(pid)))
    pid_arr = np.array([x[1] for x in rows], np.int64)
    out = {"rows": rows, "classes": [str(c) for c in base["classes"]],
           "junk": [str(c) for c in base["junk"]], "photo_id": pid_arr}
    for bb, z in zs.items():
        row_of = {int(p): i for i, p in enumerate(z["photo_id"].tolist())}
        out[bb] = z["probs"][[row_of[int(p)] for p in pid_arr]].astype(np.float32)
    with open(OUT / "frame.pkl", "wb") as f:
        pickle.dump(out, f)
    print(f"{len(recs):,} reference records, {len(rows):,} reference photos, "
          f"{len(unit_count):,} units")


if __name__ == "__main__":
    main()
