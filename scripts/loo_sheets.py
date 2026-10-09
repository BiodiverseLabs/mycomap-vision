"""Contact sheets for hand-checking flagged records (exp/loo-mislabel-scan).

    python scripts/loo_sheets.py --scan DIR/scan --review DIR/review-a-wrong-label.csv \
        --top 500 --sample 50 --seed 20261009 --out DIR/handcheck

Draws a seeded random sample from the top `--top` rows of a review list and, per record,
one row: its own photos (up to 3), two photos of the label's best-recognised records, and
two of the nearest records carrying the suggested name. Five records per sheet. Sheets and
sample.csv stay under data/ (never committed); photos are training copies, never published.
"""

from __future__ import annotations

import sys

sys.modules["torch"] = None      # noqa: E402

import argparse
import csv
import pickle
import random
import sqlite3
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from mycomap_vision import config

H = 190


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", type=Path, required=True)
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--sample", type=int, default=50)
    ap.add_argument("--seed", type=int, default=20261009)
    ap.add_argument("--skip-known", action="store_true",
                    help="sample only rows with no known_issue tag")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with open(a.review, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if a.skip_known:
        rows = [r for r in rows if not r["known_issue"]]
    pool = rows[:a.top]
    rng = random.Random(a.seed)
    sample = rng.sample(pool, min(a.sample, len(pool)))
    sample.sort(key=lambda r: -float(r["strength"]))
    with open(a.scan / "layout.pkl", "rb") as f:
        st = pickle.load(f)
    recs, layout = st["records"], st["layout"]
    pos = {o: i for i, o in enumerate(recs["observation_id"])}
    units = layout.units
    upos = {u: i for i, u in enumerate(units)}
    conn = sqlite3.connect(f"file:{config.MANIFEST_PATH}?mode=ro", uri=True)
    path_of = {}
    for pid, store, path in conn.execute(
            "select photo_id, store, path from photo_copies where size = 'large'"):
        path_of[pid] = Path(store) / path
    # Per record: is its LOO top-1 its own label (to pick typical label photos).
    chunks = sorted((a.scan / "chunks").glob("chunk-*.npz"))
    top1 = np.full(layout.n_records, -1)
    conf = np.zeros(layout.n_records)
    nb = np.full((layout.n_records, 10), -1)
    for p in chunks:
        with np.load(p) as z:
            top1[z["rec"]] = z["top_unit"][:, 0]
            conf[z["rec"]] = z["top_conf"][:, 0]
            nb[z["rec"]] = z["nb_rec"]
    by_unit: dict[int, list[int]] = {}
    for i, u in enumerate(layout.rec_unit.tolist()):
        by_unit.setdefault(u, []).append(i)

    def thumbs(rec_idx: int, n: int) -> list[Image.Image]:
        out = []
        for pid in recs["photo_ids"][rec_idx][:n]:
            p = path_of.get(int(pid))
            try:
                im = Image.open(p).convert("RGB")
                im.thumbnail((H * 2, H))
                out.append(im)
            except Exception:
                pass
        return out

    a.out.mkdir(parents=True, exist_ok=True)
    W = 1700
    for s in range(0, len(sample), 5):
        sheet = Image.new("RGB", (W, 5 * (H + 34)), "white")
        d = ImageDraw.Draw(sheet)
        for k, row in enumerate(sample[s:s + 5]):
            y = k * (H + 34)
            r = pos[row["observation_id"]]
            lu, su = upos[row["current_label"]], upos[row["suggested_name"]]
            typical = sorted((i for i in by_unit.get(lu, []) if top1[i] == lu and
                              layout.rec_group[i] != layout.rec_group[r]),
                             key=lambda i: -conf[i])[:2]
            near = [i for i in nb[r].tolist() if layout.rec_unit[i] == su][:2]
            d.text((5, y + 2), f"#{s + k + 1}  LABEL: {row['current_label']}   ->   "
                   f"SUGGESTED: {row['suggested_name']}   (conf {float(row['confidence']):.2f}, "
                   f"nb {row['nearest10_suggested']}/10)", fill="black")
            x = 5
            for im in thumbs(r, 3):
                sheet.paste(im, (x, y + 16))
                x += im.width + 4
            x += 12
            d.text((x, y + 16), "label:", fill="blue")
            for i in typical:
                for im in thumbs(i, 1):
                    sheet.paste(im, (x, y + 30))
                    x += im.width + 4
            x += 12
            d.text((x, y + 16), "suggested:", fill="red")
            for i in near:
                for im in thumbs(i, 1):
                    sheet.paste(im, (x, y + 30))
                    x += im.width + 4
        sheet.save(a.out / f"sheet-{s // 5 + 1:02d}.jpg", quality=85)
    with open(a.out / "sample.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["n"] + list(sample[0]))
        w.writeheader()
        for n, row in enumerate(sample, 1):
            w.writerow({"n": n, **row})
    print(f"{len(sample)} records, {(len(sample) + 4) // 5} sheets in {a.out}")


if __name__ == "__main__":
    main()
