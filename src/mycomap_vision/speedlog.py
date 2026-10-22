"""How long Vision takes per photo, one row per day (Steve, 2026-10-22: a daily chart that
keeps going, for How it works).

Every batch the model queue runs (workqueue.ModelQueue) adds the photos it embedded and
the seconds it spent embedding them and searching the reference set. A day's figure is
its total seconds over its total photos, so a busy hour weighs as much as its photos. The
day is UTC. Only the averages are served: the photo counts stay on the box.

Recording never fails or slows an identification: a write that fails is noted and
dropped. Kept beside the other state files, across releases.
"""
from __future__ import annotations

import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

FILE = "speed-daily.sqlite"
SCHEMA = """
create table if not exists speed_daily (
  day text primary key,          -- UTC, YYYY-MM-DD
  seconds real not null,         -- embedding and search time, summed
  photos integer not null        -- photos embedded
);
"""


def day_of(at: float) -> str:
    return datetime.fromtimestamp(at, timezone.utc).strftime("%Y-%m-%d")


class SpeedLog:
    def __init__(self, path: Path, note=print, clock=time.time):
        self.path, self.note, self.clock = Path(path), note, clock
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("pragma journal_mode=wal")
        self._conn.executescript(SCHEMA)

    def record(self, photos: int, seconds: float) -> None:
        """Add one batch. Never raises."""
        if photos <= 0 or seconds < 0:
            return
        try:
            with self._lock, self._conn:
                self._conn.execute(
                    "insert into speed_daily values (?, ?, ?) on conflict(day) do update set "
                    "seconds = seconds + excluded.seconds, photos = photos + excluded.photos",
                    (day_of(self.clock()), float(seconds), int(photos)))
        except Exception as e:  # noqa: BLE001 - a lost row never costs an answer
            self.note(f"[speed] not recorded: {type(e).__name__}: {e}")

    def daily(self, days: int = 365) -> list[dict]:
        """[{day, seconds_per_photo}], oldest first, for the last `days` days with photos."""
        since = day_of(self.clock() - max(1, days) * 86400)
        with self._lock:
            rows = self._conn.execute(
                "select day, seconds, photos from speed_daily where day > ? and photos > 0 "
                "order by day", (since,)).fetchall()
        return [{"day": d, "seconds_per_photo": round(s / n, 2)} for d, s, n in rows]
