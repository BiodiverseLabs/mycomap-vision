"""Step 3 (GPU, its own process): embed the dev query photos under fixed test-time views.

    MV_DATA_DIR=<main data> python scripts/leakage_tta_stress/tta_embed.py --data data/stress

Views, each made from the decoded photo before the model's own preprocessing (resize
shortest side + centre crop + normalise): orig, hflip, cc80 (centre 80% of width and
height), tl80 / tr80 / bl80 / br80 (corner 80% crops), hflip_cc80. Colour is never
touched (colour identifies fungi; finetune.py doesn't augment it either).

Writes <data>/tta-vectors.npz (photo_ids, views, vectors (photos, views, dim) float16,
unit length) and <data>/tta-timing.json: GPU photos/s per view, and CPU seconds per
photo for 1 view and for all views on a small sample (the serving cost; the box answers
on CPU). Reads photos from the benchmark's own folder; writes nothing else.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

BACKBONE = "bioclip-2-ft-20261007-165400"
VIEWS = ("orig", "hflip", "cc80", "tl80", "tr80", "bl80", "br80", "hflip_cc80")


def crop(img: Image.Image, where: str, f: float = 0.8) -> Image.Image:
    w, h = img.size
    cw, ch = int(round(w * f)), int(round(h * f))
    x = {"c": (w - cw) // 2, "l": 0, "r": w - cw}
    y = {"c": (h - ch) // 2, "t": 0, "b": h - ch}
    if where == "cc":
        x0, y0 = x["c"], y["c"]
    else:
        x0, y0 = x[where[1]], y[where[0]]
    return img.crop((x0, y0, x0 + cw, y0 + ch))


def view(img: Image.Image, name: str) -> Image.Image:
    if name == "orig":
        return img
    if name == "hflip":
        return ImageOps.mirror(img)
    if name == "hflip_cc80":
        return crop(ImageOps.mirror(img), "cc")
    return crop(img, name[:2])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/stress"))
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cpu-sample", type=int, default=24)
    a = ap.parse_args()
    import torch

    from mycomap_vision.embed import decode, normalise
    from mycomap_vision.models import load_backbone

    prep = pickle.load(open(a.data / "prep.pkl", "rb"))
    pids = [p for q in prep["queries"] for p in q["photo_ids"]]
    pids = pids[: a.limit or None]
    conn = sqlite3.connect(f"file:{a.data / 'manifest-snapshot.sqlite'}?mode=ro", uri=True)
    where = {int(p): (s, path) for p, s, path in conn.execute(
        "select photo_id, store, path from heldout_photos where benchmark = "
        "'heldout-2026-10-08' and size = 'large' and status = 'done'")}
    model = load_backbone(BACKBONE)
    print(f"{len(pids):,} photos x {len(VIEWS)} views on {model.device}", flush=True)

    def load(pid):
        store, rel = where[pid]
        img = decode((Path(store) / rel).read_bytes())
        return [model.prepare(view(img, v)) for v in VIEWS]

    out = np.zeros((len(pids), len(VIEWS), model.dim), dtype=np.float16)
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=8) as pool:
        batches = [pids[i:i + a.batch] for i in range(0, len(pids), a.batch)]
        ahead = [pool.submit(load, p) for p in batches[0]]
        at = 0
        for bi, batch in enumerate(batches):
            loaded = [f.result() for f in ahead]
            ahead = ([pool.submit(load, p) for p in batches[bi + 1]]
                     if bi + 1 < len(batches) else [])
            x = [t for views in loaded for t in views]
            v = normalise(model.encode(x)).reshape(len(batch), len(VIEWS), -1)
            out[at:at + len(batch)] = v
            at += len(batch)
            if bi % 50 == 0:
                el = time.time() - t0
                print(f"  {at:,}/{len(pids):,} photos, {at * len(VIEWS) / el:.0f} views/s",
                      flush=True)
    gpu_s = time.time() - t0
    np.savez(a.data / "tta-vectors.npz", photo_ids=np.array(pids, dtype=np.int64),
             views=np.array(VIEWS), vectors=out)

    # Serving cost on CPU (the box has no GPU): a fresh model on the CPU, small sample.
    timing = {"gpu": {"device": str(model.device), "photos": len(pids), "views": len(VIEWS),
                      "seconds": round(gpu_s, 1),
                      "views_per_second": round(len(pids) * len(VIEWS) / gpu_s, 1)}}
    del model
    torch.cuda.empty_cache()
    torch.set_num_threads(4)           # the 4 GB box has 2 vCPUs; the laptop uses 4 here
    orig_cuda = torch.cuda.is_available
    torch.cuda.is_available = lambda: False
    try:
        cpu_model = load_backbone(BACKBONE)
    finally:
        torch.cuda.is_available = orig_cuda
    sample = pids[: a.cpu_sample]
    imgs = [decode((Path(where[p][0]) / where[p][1]).read_bytes()) for p in sample]
    res = {}
    for label, names in (("1 view", ("orig",)), ("2 views (orig+hflip)", ("orig", "hflip")),
                         ("4 views", ("orig", "hflip", "cc80", "hflip_cc80")),
                         ("8 views", VIEWS)):
        t = time.time()
        for img in imgs:
            cpu_model.encode([cpu_model.prepare(view(img, v)) for v in names])
        res[label] = round((time.time() - t) / len(imgs), 3)
    timing["cpu_seconds_per_photo"] = {"threads": 4, "photos": len(imgs), **res}
    (a.data / "tta-timing.json").write_text(json.dumps(timing, indent=2))
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
