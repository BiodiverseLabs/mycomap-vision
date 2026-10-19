"""How often an uploaded photo is a reference photo (Steve, 2026-10-19): a count-only meter.

Identifying from a link leaves out every twin of the observation (twins.py). Uploads leave
nothing out, so a photo that is itself one of Vision's reference photos (the uploader's own
DNA record, or its MO <-> iNat cross-post) finds itself. Before deciding whether uploads
should leave such matches out, this counts how often it happens. Answers are unchanged.

A photo matches when its dHash (dhash.dhash64, as the twin table's were made) is within
errata.TWIN_MAX_BITS of a reference photo's in the served twin table, template hashes (slip
forms, microscope fields) left out. Without a table nothing is hashed and an upload counts
only as an upload (`metered` 0).

Only counters are kept: per UTC day and path (`site` /api/identify, `api` /api/v1/identify,
`session` a session photo), uploads, metered uploads, uploads with a match, photos, metered
photos and matching photos. No photo, hash, person or reference id is kept. They live in
the usage store (quotas.py's vision-usage.sqlite) and are shown only to mycomap.org's
server key in /api/health (`upload_matches`).
"""

from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import errata, twins

PATHS = ("site", "api", "session")
SHOWN_DAYS = 7
SCHEMA = """
create table if not exists upload_matches (
  day             text not null,       -- YYYY-MM-DD, UTC
  path            text not null,       -- site | api | session
  uploads         integer not null default 0,
  metered         integer not null default 0,   -- uploads checked against a twin table
  matched         integer not null default 0,   -- of those, uploads with a matching photo
  photos          integer not null default 0,
  photos_metered  integer not null default 0,
  photos_matched  integer not null default 0,
  primary key (day, path)
);
"""
COUNTERS = ("uploads", "metered", "matched", "photos", "photos_metered", "photos_matched")


class ReferenceHashes:
    """The usable reference dHashes of a twin table: every photo's, templates left out."""

    def __init__(self, path: Path):
        with np.load(Path(path), allow_pickle=False) as z:
            hashes = z["dhash"].astype(np.uint64)
            templates = z["templates"].astype(np.uint64)
        self.hashes = np.unique(hashes[~np.isin(hashes, templates)])
        self.templates = templates

    def near(self, h: int) -> bool:
        h = np.uint64(int(h))
        if np.isin(h, self.templates) or not len(self.hashes):
            return False
        return bool((np.bitwise_count(self.hashes ^ h) <= errata.TWIN_MAX_BITS).any())


class UploadMeter:
    """`record` never raises: a meter must not fail an identification."""

    def __init__(self, path: Path, table: Callable[[], Path | None],
                 clock: Callable[[], float] = time.time, note: Callable[[str], None] = print):
        self.path, self.table, self.clock, self.note = Path(path), table, clock, note
        self._lock = threading.Lock()
        self._loaded: tuple | None = None
        self._refs: ReferenceHashes | None = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("pragma journal_mode = wal")
        return db

    def references(self) -> ReferenceHashes | None:
        """The served twin table's hashes, reloaded when the table changes; None without one."""
        path = self.table()
        version = twins.version(path)
        with self._lock:
            if self._loaded != version:
                self._refs = ReferenceHashes(path) if version[1] is not None else None
                self._loaded = version
            return self._refs

    def hashing(self) -> bool:
        """Whether uploads should be hashed now (only when there is a table to meter with)."""
        try:
            return self.references() is not None
        except Exception as e:  # noqa: BLE001 - a broken table meters nothing
            self.note(f"[upload-meter] twin table unreadable: {type(e).__name__}: {e}")
            return False

    def record(self, path: str, photos: int, hashes: list | None = None) -> None:
        """Count one upload of `photos` photos on `path`; `hashes` (one dHash per photo, None
        for a photo that couldn't be hashed) when it was hashed."""
        try:
            refs = self.references() if hashes is not None else None
            row = dict.fromkeys(COUNTERS, 0)
            row.update(uploads=1, photos=photos)
            if refs is not None:
                near = [h is not None and refs.near(h) for h in hashes]
                row.update(metered=1, matched=int(any(near)),
                           photos_metered=sum(h is not None for h in hashes),
                           photos_matched=sum(near))
            day = datetime.fromtimestamp(self.clock(), timezone.utc).strftime("%Y-%m-%d")
            with self._lock, self._db() as db:
                db.execute(
                    f"insert into upload_matches (day, path, {', '.join(COUNTERS)}) "
                    f"values (?, ?, {', '.join('?' for _ in COUNTERS)}) "
                    "on conflict (day, path) do update set "
                    + ", ".join(f"{c} = {c} + excluded.{c}" for c in COUNTERS),
                    (day, path, *(row[c] for c in COUNTERS)))
        except Exception as e:  # noqa: BLE001 - never fail an identification
            self.note(f"[upload-meter] not counted: {type(e).__name__}: {e}")

    def summary(self, days: int = SHOWN_DAYS) -> dict:
        """The last `days` UTC days, newest first, each with its paths and their sum, and the
        totals over them: what /api/health shows mycomap.org."""
        today = datetime.fromtimestamp(self.clock(), timezone.utc).date()
        since = (today - timedelta(days=days - 1)).isoformat()
        with self._db() as db:
            rows = db.execute(f"select day, path, {', '.join(COUNTERS)} from upload_matches "
                              "where day >= ? order by day desc, path", (since,)).fetchall()
        by_day: dict[str, dict] = {}
        total = dict.fromkeys(COUNTERS, 0)
        for day, path, *values in rows:
            d = by_day.setdefault(day, {"day": day, **dict.fromkeys(COUNTERS, 0), "paths": {}})
            d["paths"][path] = dict(zip(COUNTERS, values))
            for c, v in zip(COUNTERS, values):
                d[c] += v
                total[c] += v
        return {"table": self._loaded is not None and self._loaded[1] is not None,
                "max_bits": errata.TWIN_MAX_BITS, "days": list(by_day.values()),
                "total": total}
