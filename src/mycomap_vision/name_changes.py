"""mycomap.org's name-change log, read into the nightly layer (Steve, 2026-10-18: "one name
decision everywhere").

Every Name Update run on mycomap.org that renames records, whoever ran it and wherever the
decision came from (Name Update's page, Names to fix, or a Human Feedback request from here,
feedback_org.py), is logged on .org with the records it renamed. Vision reads the log
(GET /api/vision/name-changes?after=<cursor>, the Vision key) and renames those records in
the NIGHTLY LAYER's manifest straight away: when a request it sent is found done, every
10 minutes beside the request check, and before each nightly update, so the nightly's
"renamed" count is what .org changed since, not these.

Only the layer copy is written (<release root>/state/nightly/<release id>/manifest.sqlite).
A served release's own files, the frozen dataset releases (v1, v1-cc, v1-test), the pinned
label vocabulary and the errata are never opened for writing: a rename reaches live data and
the next release, never a frozen one (tests/test_name_changes.py holds them byte-identical).

A record is renamed only while it still carries the change's old name, so a later change
(the nightly, another rename) is never overwritten; one Vision does not hold (a sequence, a
record not green) is counted and skipped. The cursor lives in <release root>/state/ beside
the Human Feedback queue, across releases: a new release already holds the names of its
day, so the log is never replayed onto it. Each rename is kept in the layer's org_renames
table, whose newest id rebuilds the served index (version()).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable

import requests

PAGE = 100
MAX_PAGES = 1000

STATE_SCHEMA = """
create table if not exists name_change_cursor (
  id integer primary key check (id = 1), after integer not null, read_at real
);
create table if not exists name_changes_read (
  change_id integer primary key, at text, kind text, old_name text, new_name text,
  source text, request_id integer, decision_ref text, records integer not null,
  renamed integer not null, not_held integer not null, name_moved integer not null,
  read_at real not null
);
"""

LAYER_SCHEMA = """
create table if not exists org_renames (
  id integer primary key, change_id integer not null, record_key text not null,
  old_name text not null, new_name text not null, at real not null,
  unique (change_id, record_key)
);
"""


def record_key(ref: str) -> str | None:
    """The manifest's key of a log record: inat:123 -> '123', mo:45 -> 'mo:45'; None for a
    sequence (seq:), a local record (mp:, local:) or anything else Vision does not hold."""
    source, _, number = (ref or "").partition(":")
    if not number.isdigit():
        return None
    if source == "inat":
        return number
    if source == "mo":
        return f"mo:{number}"
    return None


def version(conn: sqlite3.Connection) -> int:
    """Changes whenever a rename from the log reaches the layer (the index is then rebuilt)."""
    try:
        return int(conn.execute("select coalesce(max(id), 0) from org_renames").fetchone()[0])
    except sqlite3.OperationalError:                # never read into this manifest: none
        return 0


def apply_change(conn: sqlite3.Connection, change: dict, records: list[dict]) -> dict:
    """Rename a change's records in the layer manifest `conn`, each only while it still
    carries the old name the log gives for it."""
    conn.executescript(LAYER_SCHEMA)
    out = {"records": len(records), "renamed": 0, "not_held": 0, "name_moved": 0}
    now = time.time()
    with conn:
        for r in records:
            key = record_key(r.get("ref"))
            old, new = r.get("old_name") or change.get("old_name"), r.get("new_name") or change.get(
                "new_name")
            held = key is not None and conn.execute(
                "select scientific_name from records where observation_id = ?", (key,)).fetchone()
            if not held or not old or not new:
                out["not_held"] += 1
                continue
            cur = conn.execute("update records set scientific_name = ? where observation_id = ? "
                               "and scientific_name = ?", (new, key, old))
            if cur.rowcount:
                conn.execute("insert or ignore into org_renames (change_id, record_key, old_name, "
                             "new_name, at) values (?,?,?,?,?)",
                             (int(change["id"]), key, old, new, now))
                out["renamed"] += 1
            else:
                out["name_moved"] += 1              # renamed since, or already the new name
    return out


class Reader:
    """Reads the log after the kept cursor and applies it to the layer manifest. `org` is
    feedback_api.OrgClient (or a fake with name_changes / name_change_records); `layer_db()`
    opens the layer manifest. One run at a time; a run never raises for .org."""

    def __init__(self, state_path: Path, org, layer_db: Callable[[], sqlite3.Connection],
                 note=print):
        self.state_path = Path(state_path)
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.org, self.layer_db, self.note = org, layer_db, note
        self._lock = threading.Lock()
        with self._state() as s:
            s.executescript(STATE_SCHEMA)

    def _state(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.state_path, timeout=30)
        c.row_factory = sqlite3.Row
        return c

    def cursor(self) -> int:
        with self._state() as s:
            row = s.execute("select after from name_change_cursor where id = 1").fetchone()
        return int(row[0]) if row else 0

    def _records(self, change: dict) -> list[dict]:
        records = list(change.get("records") or [])
        if not change.get("records_truncated"):
            return records
        records, after = [], 0
        for _ in range(MAX_PAGES):
            page = self.org.name_change_records(int(change["id"]), after)
            if not page:
                raise RuntimeError(f"mycomap.org gave no records for change {change['id']}")
            records += page.get("records") or []
            nxt = int(page.get("next") or after)
            if nxt <= after or not page.get("records"):
                return records
            after = nxt
        return records

    def run(self) -> dict:
        """Read and apply every change after the cursor. {"read", "renamed", ...}; with
        "available": False when .org has no log yet (or no key), "error" when it failed."""
        if not self._lock.acquire(blocking=False):
            return {"busy": True}
        try:
            return self._run()
        finally:
            self._lock.release()

    def _run(self) -> dict:
        out = {"read": 0, "renamed": 0, "not_held": 0, "name_moved": 0}
        after = self.cursor()
        conn = None
        try:
            for _ in range(MAX_PAGES):
                page = self.org.name_changes(after, PAGE)
                if page is None:
                    return {**out, "available": False, "after": after}
                changes = page.get("changes") or []
                for change in changes:
                    cid = int(change["id"])
                    if cid <= after:
                        continue                    # never back: the log only grows
                    records = self._records(change)
                    conn = conn or self.layer_db()
                    done = apply_change(conn, change, records)
                    for k in ("renamed", "not_held", "name_moved"):
                        out[k] += done[k]
                    out["read"] += 1
                    self._keep(change, done, cid)
                    after = cid
                if len(changes) < PAGE:             # a short page is the end of the log
                    break
        except (requests.RequestException, RuntimeError, ValueError, KeyError) as e:
            out["error"] = f"{type(e).__name__}: {e}"[:300]
        finally:
            if conn is not None:
                conn.close()
        if out["read"] or out.get("error"):
            self.note(f"[name changes] {json.dumps(out)}")
        return {**out, "after": after}

    def _keep(self, change: dict, done: dict, cid: int) -> None:
        now = time.time()
        with self._state() as s:
            s.execute("insert or replace into name_changes_read values (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (cid, change.get("at"), change.get("kind"), change.get("old_name"),
                       change.get("new_name"), change.get("source"), change.get("request_id"),
                       change.get("decision_ref"), done["records"], done["renamed"],
                       done["not_held"], done["name_moved"], now))
            s.execute("insert into name_change_cursor (id, after, read_at) values (1, ?, ?) "
                      "on conflict (id) do update set after = excluded.after, "
                      "read_at = excluded.read_at", (cid, now))
