"""Records held out for a benchmark: never trained on, indexed, released or added at night.

A sealed benchmark's records (heldout.py: the paper's test set) are worth something
only while Vision has never seen them. Each is listed in the manifest's
benchmark_holdouts table, and every path that brings a record into Vision checks
that list:

    records.save_records        an export or the nightly update never stores one
    nightly.changes             ... nor counts one as new
    evaluate.load_records       reference sets, comparisons, fine-tuning, the served
                                index (and so a release and the nightly layer)
    photos.pending_photos       no reference photo is downloaded for one
    embed.photos_to_embed       ... or embedded
    release.publish             refuses a manifest whose records hold one
    aws.check_trainer_request   refuses to ship such a manifest to a trainer
    trainer.run_job             ... and the trainer refuses it too

The benchmark reads its own records from its own tables (heldout.py), so it can still
score them. A development benchmark (heldout-2026-10-08, Steve 2026-10-08) is frozen
without holding its records out: they join training and the reference index as usual.
`mv holdout release` lifts a benchmark's exclusion (logged in benchmark_holdout_releases);
the records then come in with the next export.

A release ships the manifest, this table with it, so the server box's nightly update
leaves the records out too once it runs this code on a release made after the freeze.
"""

from __future__ import annotations

import csv
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

SCHEMA = """
create table if not exists benchmark_holdouts (
  observation_id text not null,     -- records.observation_id (an iNat id)
  benchmark      text not null,     -- the frozen set it belongs to (heldout_sets.name)
  added_at       text not null,
  primary key (observation_id, benchmark)
);
-- Every exclusion lifted (mv holdout release), so it can be shown when and how many.
create table if not exists benchmark_holdout_releases (
  benchmark      text not null,
  records        integer not null,
  released_at    text not null
);
"""

BENCHMARK_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}\Z")


class HeldOutLeak(RuntimeError):
    """Held-out benchmark records are where only reference records may be."""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def not_held_out(column: str = "r.observation_id") -> str:
    """SQL: `column` is no benchmark's held-out record."""
    return f"{column} not in (select observation_id from benchmark_holdouts)"


def held_out_ids(conn: sqlite3.Connection) -> set[str]:
    ensure_schema(conn)
    return {r[0] for r in conn.execute("select observation_id from benchmark_holdouts")}


def check_name(benchmark: str) -> str:
    if not BENCHMARK_NAME.match(benchmark or ""):
        raise ValueError(f"benchmark names are letters, digits, '.', '-' or '_': {benchmark!r}")
    return benchmark


def clean_ids(ids: Iterable) -> list[str]:
    """Ids as the records table writes them: text, trimmed, no blanks, no repeats."""
    out, seen = [], set()
    for i in ids:
        s = str(i if i is not None else "").strip()
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def add(conn: sqlite3.Connection, benchmark: str, ids: Iterable,
        added_at: str | None = None) -> dict:
    """List `ids` as held out for `benchmark`. Adding them again changes nothing."""
    check_name(benchmark)
    ensure_schema(conn)
    ids = clean_ids(ids)
    if not ids:
        raise ValueError("no observation ids to hold out")
    with conn:
        added = insert(conn, benchmark, ids, added_at)
    return {"benchmark": benchmark, "ids": len(ids), "added": added,
            "already": len(ids) - added, "in_records": len(leaked(conn, ids))}


def insert(conn: sqlite3.Connection, benchmark: str, ids: list[str],
           added_at: str | None = None) -> int:
    """The rows of add(), inside the caller's transaction (heldout.freeze adds them with
    the set itself). Returns how many were new."""
    at = added_at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    before = conn.total_changes
    conn.executemany("insert or ignore into benchmark_holdouts values (?, ?, ?)",
                     [(i, benchmark, at) for i in ids])
    return conn.total_changes - before


def is_held_out(conn: sqlite3.Connection, benchmark: str) -> bool:
    """Whether the benchmark's records are held out (a sealed set) right now."""
    ensure_schema(conn)
    return conn.execute("select 1 from benchmark_holdouts where benchmark = ? limit 1",
                        (benchmark,)).fetchone() is not None


def release(conn: sqlite3.Connection, benchmark: str, at: str | None = None) -> dict:
    """Lift a benchmark's exclusion: its records may join training and the reference
    index with the next export. A record another benchmark still holds out stays out."""
    check_name(benchmark)
    ensure_schema(conn)
    at = at or datetime.now(timezone.utc).isoformat(timespec="seconds")
    with conn:
        n = conn.execute("delete from benchmark_holdouts where benchmark = ?",
                         (benchmark,)).rowcount
        if n:
            conn.execute("insert into benchmark_holdout_releases values (?, ?, ?)",
                         (benchmark, n, at))
    still = conn.execute("select count(*) from benchmark_holdouts").fetchone()[0]
    return {"benchmark": benchmark, "released": n, "held_out_by_other_benchmarks": still}


def read_ids_csv(path: Path) -> list[str]:
    """The observation_id column of a CSV (UTF-8, a byte-order mark allowed)."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "observation_id" not in (reader.fieldnames or []):
            raise ValueError(f"{path} has no observation_id column")
        return clean_ids(r["observation_id"] for r in reader)


def drop_held_out(conn: sqlite3.Connection, recs: list[dict]) -> tuple[list[dict], int]:
    """Export rows without the held-out records, and how many were dropped."""
    held = held_out_ids(conn)
    if not held:
        return recs, 0
    kept = [r for r in recs if str(r["observation_id"]) not in held]
    return kept, len(recs) - len(kept)


def leaked(conn: sqlite3.Connection, ids: Iterable[str] | None = None) -> list[str]:
    """Held-out ids (of `ids`, or of every benchmark) that the records table holds."""
    ensure_schema(conn)
    rows = conn.execute("select distinct r.observation_id from records r join benchmark_holdouts h "
                        "on h.observation_id = r.observation_id order by 1").fetchall()
    found = [r[0] for r in rows]
    if ids is not None:
        want = set(ids)
        found = [i for i in found if i in want]
    return found


def check_clean(conn: sqlite3.Connection, what: str) -> None:
    """Refuse `what` (a release, a trainer run) while the records table holds a held-out
    record. Only code from before benchmark_holdouts can have stored one: an export with
    this code drops it again (records.save_records)."""
    bad = leaked(conn)
    if bad:
        raise HeldOutLeak(
            f"{what} refused: the records table holds {len(bad):,} held-out benchmark "
            f"records (first: {', '.join(bad[:5])}). Run `mv export-records` with this "
            "code, which drops them, then try again.")


def summary(conn: sqlite3.Connection) -> list[dict]:
    """Per benchmark: how many records are held out, since when, and how many of them
    the records table holds (should be 0)."""
    ensure_schema(conn)
    rows = conn.execute("select benchmark, count(*), min(added_at), max(added_at) "
                        "from benchmark_holdouts group by 1 order by 1").fetchall()
    out = []
    for name, n, first, last in rows:
        in_records = conn.execute(
            "select count(*) from benchmark_holdouts h join records r "
            "on r.observation_id = h.observation_id where h.benchmark = ?", (name,)).fetchone()[0]
        out.append({"benchmark": name, "records": n, "first_added": first, "last_added": last,
                    "in_records": in_records})
    return out
