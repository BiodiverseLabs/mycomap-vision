"""Contact sheets for hand-checking: numbered thumbnails of reference photos, read from the
project's own photo store (local copy, else the private bucket). Nothing goes to iNat.
Sheets are written under data/audits/non-fungus-scan/sheets/ (private, never committed).

make_sheets(manifest, name, photo_ids) -> list of sheet paths; tile k on a sheet is
photo_ids[k] (numbered from 0 across sheets).
"""
import io
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageDraw

from mycomap_vision.storage import open_store

from zs_score import OUT

TILE, COLS, ROWS = 220, 6, 5


def _copies(conn: sqlite3.Connection, ids: list[int]) -> dict[int, tuple[str, str]]:
    q = ",".join("?" * len(ids))
    out = {}
    for pid, store, size, path in conn.execute(
            f"select photo_id, store, size, path from photo_copies where photo_id in ({q}) "
            "order by size = 'large'", ids):
        out[pid] = (store, path)     # large wins (ordered last)
    return out


def make_sheets(manifest: Path, name: str, ids: list[int]) -> list[Path]:
    conn = sqlite3.connect(Path(manifest).resolve().as_uri() + "?mode=ro", uri=True)
    where = _copies(conn, ids)
    stores = {}

    def load(pid):
        try:
            store, path = where[pid]
            if store not in stores:
                stores[store] = open_store(store, Path(store))
            im = Image.open(io.BytesIO(stores[store].get(path))).convert("RGB")
            im.thumbnail((TILE, TILE))
            return im
        except Exception:
            return None
    with ThreadPoolExecutor(8) as ex:
        thumbs = list(ex.map(load, ids))
    d = OUT / "sheets"
    d.mkdir(parents=True, exist_ok=True)
    per = COLS * ROWS
    paths = []
    for s in range(0, len(ids), per):
        sheet = Image.new("RGB", (COLS * TILE, ROWS * TILE), "white")
        draw = ImageDraw.Draw(sheet)
        for k, im in enumerate(thumbs[s:s + per]):
            x, y = (k % COLS) * TILE, (k // COLS) * TILE
            if im is not None:
                sheet.paste(im, (x + (TILE - im.width) // 2, y + (TILE - im.height) // 2))
            draw.rectangle([x, y, x + 34, y + 18], fill="yellow")
            draw.text((x + 3, y + 3), str(s + k), fill="black")
        p = d / f"{name}-{s // per:02d}.jpg"
        sheet.save(p, quality=85)
        paths.append(p)
    return paths
