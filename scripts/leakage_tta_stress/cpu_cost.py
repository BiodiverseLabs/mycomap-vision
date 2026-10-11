"""Serving cost of test-time augmentation on the CPU (the Vision box has no GPU).

    MV_DATA_DIR=<main data> python scripts/leakage_tta_stress/cpu_cost.py --data data/stress

Two parts, each in its own process because torch and numpy's large CPU matrix products
must not share one (this environment segfaults):
  --part tower   the image tower on 4 CPU threads: seconds per photo for 1, 2, 4 and 8
                 views, measured in interleaved rounds (median), so background load
                 affects every setting alike;
  --part score   the similarity pass against the whole reference (numpy, float16
                 reference widened per block, as the server does): seconds per query
                 vector, which score-averaging multiplies by the number of views.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
import time
from pathlib import Path


def tower(a) -> dict:
    import pickle

    import torch
    torch.set_num_threads(4)
    torch.cuda.is_available = lambda: False          # build the model on the CPU
    import tta_embed as T

    from mycomap_vision.embed import decode
    from mycomap_vision.models import load_backbone
    model = load_backbone(T.BACKBONE)
    prep = pickle.load(open(a.data / "prep.pkl", "rb"))
    pids = [p for q in prep["queries"] for p in q["photo_ids"]][: a.photos]
    conn = sqlite3.connect(f"file:{a.data / 'manifest-snapshot.sqlite'}?mode=ro", uri=True)
    where = dict(((int(p), (s, r)) for p, s, r in conn.execute(
        "select photo_id, store, path from heldout_photos where benchmark = "
        "'heldout-2026-10-08' and size = 'large' and status = 'done'")))
    imgs = [decode((Path(where[p][0]) / where[p][1]).read_bytes()) for p in pids]
    sets = {"1 view": ("orig",), "2 views": ("orig", "hflip"),
            "4 views": ("orig", "hflip", "cc80", "hflip_cc80"), "8 views": T.VIEWS}
    model.encode([model.prepare(imgs[0])])              # warm up
    times = {k: [] for k in sets}
    for _ in range(a.rounds):
        for k, names in sets.items():
            t = time.perf_counter()
            for img in imgs:
                small = T.reduced(img)
                model.encode([model.prepare(T.view(img, v, small)) for v in names])
            times[k].append((time.perf_counter() - t) / len(imgs))
    return {"threads": 4, "photos": len(imgs), "rounds": a.rounds,
            "seconds_per_photo_median": {k: round(statistics.median(v), 3)
                                         for k, v in times.items()}}


def score(a) -> dict:
    import numpy as np
    ref = np.load(a.data / "ref_vecs.npy", mmap_mode="r")
    q = np.random.default_rng(0).normal(size=(8, ref.shape[1])).astype(np.float32)
    out = {}
    for n in (1, 2, 4, 8):
        runs = []
        for _ in range(a.rounds):
            t = time.perf_counter()
            for s in range(0, len(ref), 65536):
                q[:n] @ np.asarray(ref[s:s + 65536], dtype=np.float32).T
            runs.append(time.perf_counter() - t)
        out[f"{n} query vectors"] = round(statistics.median(runs), 3)
    return {"reference_photos": int(len(ref)), "seconds_median": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/stress"))
    ap.add_argument("--part", choices=("tower", "score"), required=True)
    ap.add_argument("--photos", type=int, default=8)
    ap.add_argument("--rounds", type=int, default=3)
    a = ap.parse_args()
    res = tower(a) if a.part == "tower" else score(a)
    path = a.data / "tta-cpu-cost.json"
    have = json.loads(path.read_text()) if path.exists() else {}
    have[a.part] = res
    path.write_text(json.dumps(have, indent=2))
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
