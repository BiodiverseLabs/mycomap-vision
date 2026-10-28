"""Research > Excluded photos > Modified photos (Steve, 2026-10-27): every field-slip photo whose
Recovered Labels crops go back into Vision, before (the original, as people left it out) and
after (each kept crop, exactly the square the model reads), with "where Vision looked" on both.

**Which photos.** A Recovered Labels item (recovered_labels.py) that keeps at least one crop,
on a photo a person still leaves out as a field slip: the crops Vision uses
(recovered_labels.accepted_crops). Open items are listed too (Steve, 2026-10-27: "lets have it
use open crops. Most look good. We can refine them with time and exclude/modify them, which
would get picked up with the next round"): an open item's crops are used (the queue's default
is Include), and a later exclusion or box edit makes the crop's map stale, made again on the
next preload pass. Reviewed items first, newest decision first; open items after, newest first.

**What is shown** is what the rest of the page shows (excluded_looked.py, api.py): a photo
anyone may see, by its public URL, never who decided or their note; any other photo a
placeholder with no photo, crops or maps; a photo the paper's seal holds back shows without
maps. A crop is drawn from the original image by CSS (no cropped file is made), its square from
recovered_labels.effective_box. Each crop says whether its box is the detector's or a
reviewer's (recovered_labels.box_source, the export's rule).

**Maps** are the kept ones only (the preload makes them: excluded_looked.Preloader): the photo's
under its key, a crop's under `<photo key>#crop<n>`; each is a current map, a "no map possible"
with its reason, or neither ("map coming"). Nothing is computed when the page is viewed.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from . import feedback as fb
from . import recovered_labels as rl
from .excluded_looked import crop_key

FILTERS = {"all": "All", "edited": "Box edited by a reviewer", "detector": "Detector's box"}
PAGE_SIZE = 12
PAGE_MAX = 48


def card_of(item: dict) -> dict | None:
    """An item's card data from the store alone: its kept crops (squares, in the frame the
    detector saw, and as fractions of the photo: the box the crop's vector and map are made
    from, as accepted_crops gives it) and how it was decided; None when it keeps no crop."""
    keep = set(rl.kept(item))
    if not keep:
        return None
    ev = item.get("evidence") or {}
    w, h = int(ev.get("width") or 0), int(ev.get("height") or 0)
    if not (w and h):
        return None
    now = rl.boxes_now(item)
    last = rl.last_of(item)
    crops = [{"n": n, "box": now[n], "source": rl.box_source(item, n),
              "fraction": [round(now[n][0] / w, 6), round(now[n][1] / h, 6),
                           round(now[n][2] / w, 6), round(now[n][3] / h, 6)]}
             for n in sorted(now) if n in keep]
    return {"item_id": item["id"], "photo_key": item["subject"],
            "record_key": (item.get("record_keys") or [None])[0], "width": w, "height": h,
            "decided": item["status"] != "open",
            "decided_at": (last or {}).get("at") if item["status"] != "open" else None,
            "edited": any(c["source"] == "reviewer" for c in crops), "crops": crops,
            "_order": ((last or {}).get("id") or 0, item["id"])}


def ordered(items: list[dict]) -> list[dict]:
    """Every item's card that keeps a crop: reviewed first, newest decision first; open after,
    newest item first."""
    rows = [c for c in map(card_of, items) if c is not None]
    rows.sort(key=lambda r: (not r["decided"], -r["_order"][0], -r["_order"][1]))
    return rows


def page(rows: list[dict], slip_keys: set[str], filter: str = "all", offset: int = 0,
         limit: int = PAGE_SIZE, mapped: set[str] | None = None) -> dict:
    """A page of `ordered` cards on photos still left out as a field slip (`slip_keys`). With
    `mapped` (Maps.mapped's keys), only photos with a kept map of the photo or of a crop.
    ValueError for an unknown filter. Cards are copies, without the sort key."""
    if filter not in FILTERS:
        raise ValueError(f"no such filter: {filter}")
    rows = [r for r in rows if r["photo_key"] in slip_keys]
    counts = {"all": len(rows), "edited": sum(1 for r in rows if r["edited"])}
    counts["detector"] = counts["all"] - counts["edited"]
    chosen = [r for r in rows if (filter == "all" or r["edited"] == (filter == "edited"))
              and (mapped is None or r["photo_key"] in mapped
                   or any(crop_key(r["photo_key"], c["n"]) in mapped for c in r["crops"]))]
    start = max(0, offset)
    cards = [{k: v for k, v in r.items() if k != "_order"} | {"crops": [dict(c) for c in r["crops"]]}
             for r in chosen[start:start + min(max(1, limit), PAGE_MAX)]]
    return {"total": len(chosen), "offset": start, "counts": counts, "cards": cards,
            "include_open": True,
            "filters": [{"code": k, "label": v, "count": counts[k]} for k, v in FILTERS.items()]}


def map_view(doc: dict | None) -> dict | None:
    """What the page draws of a kept map: the map, its seen square, the name and rank."""
    if not doc:
        return None
    return {"map": doc["map"], "name": doc.get("name"), "rank": doc.get("rank"),
            "top": doc.get("top"), "made_at": doc.get("made_at")}


class Source:
    """The Recovered labels items as `ordered` cards, read again only when the store's decisions
    or items change (checked at most every `every` seconds, on a read-only connection of its
    own, as excluded_looked.Listing does): tens of thousands of items are seeded."""

    def __init__(self, store: fb.Store, every: float = 30.0, clock=time.monotonic):
        self.store, self.every, self.clock = store, every, clock
        self._at, self._version, self._rows = -1e9, None, []
        self._conn: sqlite3.Connection | None = None
        self.lock = threading.Lock()

    def cards(self) -> list[dict]:
        with self.lock:
            if self.clock() - self._at >= self.every:
                self._at = self.clock()
                if self._conn is None:
                    self._conn = sqlite3.connect(
                        f"file:{Path(self.store.path).as_posix()}?mode=ro", uri=True,
                        check_same_thread=False)
                version = self._conn.execute(
                    "select (select max(id) from decisions), (select count(*) from decisions), "
                    "(select max(id) from items where item_type = ?)", (rl.ITEM_TYPE,)).fetchone()
                if version != self._version:
                    self._rows = ordered(self.store.items_of(rl.ITEM_TYPE))
                    self._version = version
            return self._rows
