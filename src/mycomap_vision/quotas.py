"""Quotas for the public API (docs/api.md §6.4–6.6): Vision counts and enforces; mycomap.org
owns the plans and limits (in each credential) and shows the use Vision reports.

Steve, 2026-10-11:
- the unit is one **identification**: a /identify call, or a session counted once, at its
  first photo (extra photos never cost more; they're counted as photos for capacity);
- one pool per member: the app and the member's developer tokens all count against the
  member's opaque `subject`; a token with its own admin override counts alone;
- a session re-opened for the same find (same pool, same `client_ref`) within 24 hours of
  the one that was counted is free: the app re-opens one after a restart, the 30-minute
  idle end or the 2-hour cap (Steve, 2026-10-12). Its photos still count as photos.

Counts are running totals per calendar month (UTC), kept in SQLite on the box so a restart
loses nothing. A refusal is never counted as use. Modes (MV_QUOTAS): `off` (nothing
counted, `usage` null), `count` (counted, reported and shown, never refused: for a trial),
`enforce` (past the limit, 429 quota_exceeded until the month resets).

`report_rows` gives the rows mycomap.org's POST /api/vision/usage takes, kind "user" with
`sub` and kind "key" with `keyId`; .org keeps the greatest value per row, so sending the
same running totals again is harmless.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from .apiv1 import Credential, V1Error

MODES = ("off", "count", "enforce")
DEFAULT_MONTHLY = 100              # a credential that names no limit: the free plan's
REOPEN_FREE_SECONDS = 86400        # a find re-opened within a day isn't counted again
SCHEMA = """
create table if not exists usage (
  period          text not null,       -- YYYY-MM, UTC
  kind            text not null,       -- user | key
  ref             text not null,       -- the member's subject, or the token's key id
  identifications integer not null default 0,
  photos          integer not null default 0,
  refused         text not null default '{}',   -- {code: n}
  last_used_at    text,
  primary key (period, kind, ref)
);
create table if not exists counted_finds (
  kind            text not null,
  ref             text not null,
  client_ref      text not null,       -- the app's id for the find (its observation)
  counted_at      real not null,       -- when its session was counted (unix seconds)
  primary key (kind, ref, client_ref)
);
"""


def period_of(at: float) -> str:
    d = datetime.fromtimestamp(at, timezone.utc)
    return f"{d.year}-{d.month:02d}"


def period_end(period: str) -> datetime:
    """When the month `period` ends (the first instant of the next one), UTC."""
    y, m = map(int, period.split("-"))
    return datetime(y + (m == 12), m % 12 + 1, 1, tzinfo=timezone.utc)


def rows_for(cred: Credential) -> list[tuple[str, str]]:
    """The rows a call by `cred` counts in, the pool it is limited by first. The app and
    pooled tokens count in the member's row (and a token also in its own, so .org can show
    each key's use); a token with its own override counts only in its own; mycomap.org's
    server key counts nowhere."""
    if cred.kind == "app" and cred.subject:
        return [("user", cred.subject)]
    if cred.kind == "key":
        if cred.limit_source == "override" or not cred.subject:
            return [("key", cred.key_id)]
        return [("user", cred.subject), ("key", cred.key_id)]
    return []


class Quotas:
    def __init__(self, path: Path, mode: str = "off", default_monthly: int = DEFAULT_MONTHLY,
                 clock: Callable[[], float] = time.time):
        if mode not in MODES:
            raise ValueError(f"MV_QUOTAS must be one of {', '.join(MODES)}, not {mode!r}")
        self.mode, self.default_monthly, self.clock = mode, default_monthly, clock
        self.path = Path(path)
        self._lock = threading.Lock()
        if mode != "off":
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self._db() as db:
                db.executescript(SCHEMA)

    def _db(self) -> sqlite3.Connection:
        db = sqlite3.connect(self.path, timeout=30)
        db.execute("pragma journal_mode = wal")
        return db

    @property
    def on(self) -> bool:
        return self.mode != "off"

    def limit_of(self, cred: Credential) -> int | None:
        """The monthly limit for `cred`'s pool: None is unlimited."""
        if cred.limit is None:
            return self.default_monthly
        return cred.limit.get("monthly")

    def _used(self, db, period: str, kind: str, ref: str) -> int:
        row = db.execute("select identifications from usage where period = ? and kind = ? "
                         "and ref = ?", (period, kind, ref)).fetchone()
        return row[0] if row else 0

    def usage(self, cred: Credential) -> dict | None:
        """The `usage` block of an answer (docs/api.md §4.1), or None when quotas are off
        or the caller has no pool (mycomap.org's own server key)."""
        rows = rows_for(cred)
        if not self.on or not rows:
            return None
        period = period_of(self.clock())
        with self._lock, self._db() as db:
            used = self._used(db, period, *rows[0])
        return {"plan": cred.plan, "used": used, "limit": self.limit_of(cred),
                "period": period,
                "resets_at": period_end(period).isoformat().replace("+00:00", "Z")}

    def check(self, cred: Credential) -> None:
        """Before an identification: 429 quota_exceeded when the pool's month is used up
        (enforce only). The refusal is counted as a refusal, never as use."""
        if self.mode != "enforce":
            return
        u = self.usage(cred)
        if u is None or u["limit"] is None or u["used"] < u["limit"]:
            return
        self.refused(cred, "quota_exceeded")
        wait = max(1, int(period_end(u["period"]).timestamp() - self.clock()))
        raise V1Error(429, "quota_exceeded", f"this month's {u['limit']} identifications are "
                      f"used; they renew {u['resets_at'][:10]}", retry_after=wait,
                      limit=u["limit"], used=u["used"], resets_at=u["resets_at"])

    def count(self, cred: Credential, identifications: int, photos: int) -> None:
        """Count use after an answer was given."""
        rows = rows_for(cred)
        if not self.on or not rows or not (identifications or photos):
            return
        now = self.clock()
        period, at = period_of(now), datetime.fromtimestamp(now, timezone.utc).isoformat(
            timespec="seconds").replace("+00:00", "Z")
        with self._lock, self._db() as db:
            for kind, ref in rows:
                db.execute("insert into usage (period, kind, ref, identifications, photos, "
                           "last_used_at) values (?, ?, ?, ?, ?, ?) on conflict "
                           "(period, kind, ref) do update set identifications = "
                           "identifications + excluded.identifications, photos = photos + "
                           "excluded.photos, last_used_at = excluded.last_used_at",
                           (period, kind, ref, identifications, photos, at))

    def counted_before(self, cred: Credential, client_ref: str | None) -> bool:
        """Whether a session for this find was counted in the last REOPEN_FREE_SECONDS, so
        a re-opened one is free."""
        rows = rows_for(cred)
        if not self.on or not rows or not client_ref:
            return False
        kind, ref = rows[0]
        with self._lock, self._db() as db:
            row = db.execute("select counted_at from counted_finds where kind = ? and ref = ? "
                             "and client_ref = ?", (kind, ref, client_ref)).fetchone()
        return bool(row) and self.clock() - row[0] < REOPEN_FREE_SECONDS

    def remember_counted(self, cred: Credential, client_ref: str | None) -> None:
        rows = rows_for(cred)
        if not self.on or not rows or not client_ref:
            return
        kind, ref = rows[0]
        with self._lock, self._db() as db:
            db.execute("insert into counted_finds (kind, ref, client_ref, counted_at) "
                       "values (?, ?, ?, ?) on conflict (kind, ref, client_ref) do update "
                       "set counted_at = excluded.counted_at",
                       (kind, ref, client_ref, self.clock()))
            # Forget finds counted long ago (the table stays small).
            db.execute("delete from counted_finds where counted_at < ?",
                       (self.clock() - 7 * REOPEN_FREE_SECONDS,))

    def refused(self, cred: Credential, code: str) -> None:
        rows = rows_for(cred)
        if not self.on or not rows:
            return
        period = period_of(self.clock())
        with self._lock, self._db() as db:
            for kind, ref in rows:
                row = db.execute("select refused from usage where period = ? and kind = ? "
                                 "and ref = ?", (period, kind, ref)).fetchone()
                refused = json.loads(row[0]) if row else {}
                refused[code] = refused.get(code, 0) + 1
                db.execute("insert into usage (period, kind, ref, refused) values (?, ?, ?, ?) "
                           "on conflict (period, kind, ref) do update set refused = "
                           "excluded.refused", (period, kind, ref, json.dumps(refused)))

    def periods_to_report(self) -> list[str]:
        """This month, and last month for 2 days after it ends (late counts settle)."""
        now = self.clock()
        this = period_of(now)
        last = period_of(now - 2 * 86400)
        return [this] if last == this else [last, this]

    def report_rows(self, period: str) -> list[dict]:
        """Rows for mycomap.org's POST /api/vision/usage: running totals for `period`."""
        if not self.on:
            return []
        with self._lock, self._db() as db:
            rows = db.execute("select kind, ref, identifications, photos, refused, "
                              "last_used_at from usage where period = ? order by kind, ref",
                              (period,)).fetchall()
        out = []
        for kind, ref, ids, photos, refused, last in rows:
            row = {"kind": kind, "identifications": ids, "photos": photos,
                   "refused": json.loads(refused or "{}"), "last_used_at": last}
            row["sub" if kind == "user" else "keyId"] = ref
            out.append(row)
        return out


MAX_ROWS = 5000                    # mycomap.org's limit per report


def report(quotas: Quotas, post: Callable[[dict], None], clock: Callable[[], float] = time.time,
           note=print) -> int:
    """Send the running totals to mycomap.org (post(body) raises on failure). Returns the
    rows sent. Totals are absolute, so a report that fails is simply sent again next time."""
    sent = 0
    for period in quotas.periods_to_report():
        rows = quotas.report_rows(period)
        for i in range(0, len(rows), MAX_ROWS):
            post({"period": period,
                  "reported_at": datetime.fromtimestamp(clock(), timezone.utc).isoformat(
                      timespec="seconds").replace("+00:00", "Z"),
                  "rows": rows[i:i + MAX_ROWS]})
            sent += len(rows[i:i + MAX_ROWS])
    return sent


def org_poster(base_url: str, key: str, timeout: float = 15.0,
               session_factory: Callable | None = None) -> Callable[[dict], None]:
    """The real `post` for `report`: mycomap.org's POST /api/vision/usage.

    A report goes every 5 minutes, longer than Cloudflare and nginx keep an idle connection
    open, so the kept-alive one is sometimes already closed when a report goes out on it (29-30
    Oct: 9 reports lost to "RemoteDisconnected"). A connection error (closed, reset, refused) is
    tried once more on a fresh connection; the same running totals sent twice are harmless. An
    HTTP error status or a read timeout is not tried again."""
    import requests
    make = session_factory or requests.Session
    url = base_url.rstrip("/") + "/api/vision/usage"
    kept = {"session": make()}

    def send(body: dict):
        return kept["session"].post(url, json=body, timeout=timeout,
                                    headers={"Authorization": f"Bearer {key}"})

    def post(body: dict) -> None:
        try:
            r = send(body)
        except requests.ConnectionError:
            try:
                kept["session"].close()
            except Exception:  # noqa: BLE001 - the stale session is dropped either way
                pass
            kept["session"] = make()
            r = send(body)
        r.raise_for_status()
    return post
