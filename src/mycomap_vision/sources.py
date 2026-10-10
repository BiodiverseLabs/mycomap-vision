"""Where each record comes from, and what that allows. The source is always explicit.

mycomap.org's observations come from several places, and their ids look alike: an iNat
observation number, a Mushroom Observer (MO) observation number, a MyCoPortal occurrence
id and a legacy .com sequence id are all plain numbers. Until 2026-10-09 the export
guessed "a numeric id is an iNat id", so Vision fetched whatever iNat observation shared
an MO or MyCoPortal record's number (a mallard as a tooth fungus). Now:

- The export reads .org's `observations.source` and stores it, normalised, in
  `records.source` (inat | mo | mycoportal | com_sequence | genbank | unknown), with
  .org's own spelling in `records.org_source` and the id at the source in
  `records.source_id`.
- A record's key (`records.observation_id`) is the iNat id for an iNat record, and
  `<source>:<id>` for every other source, so an iNat record and an MO record with the
  same number are two records (72 such pairs on .org on 2026-10-09).
- Everything that fetches, downloads, embeds or uses photos asks `records.source`, never
  the shape of an id. A record whose source is not one Vision takes photos from
  (PHOTO_SOURCES) gets no photos at all.

Steve, 2026-10-09: "We want to include our validated MO records" (with their own MO
photos) and "Let's not include MyCoPortal records as they rarely have field images".
.com "Sequences" and GenBank accession records have no photos of their own.

A manifest written by older code still holds the guessed sources. The first export with
this code migrates it (migrate_legacy, inside records.save_records), and records that
in the same transaction under MIGRATION in `manifest_migrations`; code that must not run
on the old guesses (a training launch) checks `migrated(conn)`.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

INAT = "inat"
MO = "mo"
MYCOPORTAL = "mycoportal"
COM_SEQUENCE = "com_sequence"
GENBANK = "genbank"
UNKNOWN = "unknown"

SOURCES = (INAT, MO, MYCOPORTAL, COM_SEQUENCE, GENBANK, UNKNOWN)

# .org's spellings of observations.source (lower-cased, spaces collapsed).
_ORG_SPELLINGS = {
    "inaturalist": INAT,
    "mo observations": MO,
    "mushroom observer": MO,
    "mycoportal": MYCOPORTAL,          # .org writes both 'MyCoPortal' and 'MycoPortal'
    "sequences": COM_SEQUENCE,
    "genbank accessions": GENBANK,
}

# Sources whose records may have photos in Vision: fetched, downloaded, embedded, used.
PHOTO_SOURCES = (INAT, MO)

# Why each other source is left out of Vision entirely.
EXCLUDED = {
    MYCOPORTAL: "MyCoPortal: rarely field images (Steve, 2026-10-09)",
    COM_SEQUENCE: ".com Sequences: no photos of their own",
    GENBANK: "GenBank accessions: no photos of their own",
    UNKNOWN: "no known source: no photos",
}

# The SQL list of PHOTO_SOURCES, for `r.source in (...)`.
PHOTO_SOURCES_SQL = "(" + ", ".join(f"'{s}'" for s in PHOTO_SOURCES) + ")"

# Where a record's own page is, by source.
RECORD_URLS = {
    INAT: "https://www.inaturalist.org/observations/{id}",
    MO: "https://mushroomobserver.org/obs/{id}",
}


def normalise(org_source: str | None) -> str:
    """.org's observations.source -> one of SOURCES (UNKNOWN for anything else)."""
    key = " ".join((org_source or "").split()).lower()
    return _ORG_SPELLINGS.get(key, UNKNOWN)


def record_key(source: str, source_id: str) -> str:
    """The manifest key of a record: the iNat id for iNat, `<source>:<id>` otherwise."""
    if source not in SOURCES:
        raise ValueError(f"unknown source {source!r}")
    source_id = str(source_id).strip()
    return source_id if source == INAT else f"{source}:{source_id}"


def record_url(source: str, source_id: str | None) -> str | None:
    """The record's page at its source, or None (MyCoPortal and the rest have none here)."""
    template = RECORD_URLS.get(source)
    return template.format(id=source_id) if template and source_id else None


# ---------------------------------------------------------------------------
# The migration marker

MIGRATION = "record-sources-v1"

SCHEMA = """
-- One row per data migration applied to this manifest (schema changes are additive and
-- need none). 'record-sources-v1': records.source is .org's, not a guess from the id.
create table if not exists manifest_migrations (
  name          text primary key,
  applied_at    text not null,
  code_version  text,
  detail        text                -- JSON: what the migration changed
);

-- Photos taken off a record because they never belonged to it: the record's source was
-- not iNat, yet older code fetched the iNat observation with the same number. One row
-- per (old record key, photo); the photo's files and vectors are named so they can be
-- deleted later, and nothing of them is used meanwhile.
create table if not exists source_removals (
  observation_id  text not null,    -- the old (guessed) record key: an iNat number
  photo_id        integer,          -- null: the iNat observation had no photos
  org_source      text,             -- what .org says the record really is
  new_key         text,             -- the record's key now (e.g. 'mo:12345')
  reason          text not null,
  photo_json      text,             -- the photos row as it was (owner, licence, url, hash)
  copies_json     text,             -- its photo_copies rows (store, size, path, sha256)
  embeddings_json text,             -- its embeddings rows (backbone, shard, row)
  removed_at      text not null,
  migration       text not null
);
create index if not exists source_removals_key_idx on source_removals(observation_id);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def migrated(conn: sqlite3.Connection) -> bool:
    """True when this manifest's records.source comes from .org (MIGRATION applied).
    A manifest without the marker table is not migrated."""
    has = conn.execute("select 1 from sqlite_master where type = 'table' "
                       "and name = 'manifest_migrations'").fetchone()
    return bool(has and conn.execute("select 1 from manifest_migrations where name = ?",
                                     (MIGRATION,)).fetchone())


class NotMigrated(RuntimeError):
    """The manifest's sources are still guessed from ids (older code): nothing may be
    fetched, downloaded or embedded for its records until an export migrates it."""


def require_migrated(conn: sqlite3.Connection, what: str) -> None:
    if not migrated(conn):
        raise NotMigrated(
            f"Refused {what}: this manifest's record sources were guessed from ids by older "
            f"code (MO and MyCoPortal numbers taken for iNat ids). Run an export (`mv export`, "
            f"or `mv record-sources --apply`) first; it migrates the manifest ({MIGRATION}).")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _rows(conn: sqlite3.Connection, sql: str, args: tuple) -> list[dict]:
    cur = conn.execute(sql, args)
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def plan_legacy(conn: sqlite3.Connection, records: list[dict]) -> dict:
    """What migrate_legacy would do with this export, without doing it.

    A legacy record key is wrong-sourced when .org has no iNat record with that id but
    has a record of another source with it. Its iNat details and photo links were
    fetched for the wrong observation."""
    live_inat = {r["observation_id"] for r in records if r["source"] == INAT}
    other_by_id: dict[str, dict] = {}
    for r in records:
        if r["source"] != INAT:
            other_by_id.setdefault(str(r["source_id"]), r)
    old = [k for (k,) in conn.execute("select observation_id from records")]
    wrong = {}
    for key in old:
        if key in live_inat or key not in other_by_id:
            continue
        wrong[key] = other_by_id[key]
    photos = conn.execute(
        "select count(*) from observation_photos where observation_id in "
        "(select value from json_each(?))", (json.dumps(sorted(wrong)),)).fetchone()[0]
    by_source: dict[str, int] = {}
    for r in wrong.values():
        by_source[r["source"]] = by_source.get(r["source"], 0) + 1
    return {"wrong_records": wrong, "wrong_photos": photos, "by_source": by_source}


def migrate_legacy(conn: sqlite3.Connection, records: list[dict], code_version: str | None = None,
                   now: str | None = None) -> dict:
    """Take the iNat data fetched for wrong-sourced legacy records off them, with an
    audit row per photo, and set the MIGRATION marker. Runs inside the caller's
    transaction (records.save_records), once per manifest.

    For each wrong-sourced key: its observation_photos links, its inat_observations row
    and the embeddings of photos no other record links to are removed; each photo is
    written to source_removals first (photos row, copies, embeddings as they were).
    The photos and photo_copies rows stay (they say what files we hold and under which
    licence) but nothing links to them, so nothing downloads, embeds or uses them.

    Set-based: observation_photos and embeddings are keyed for lookups by record and by
    backbone, not by photo, so each is read once against temp tables (a full manifest
    has ~650k links and ~1.5M vectors).
    """
    ensure_schema(conn)
    if migrated(conn):
        return {"already": True}
    now = now or _now()
    plan = plan_legacy(conn, records)
    wrong = plan["wrong_records"]
    conn.execute("create temp table if not exists mig_keys "
                 "(observation_id text primary key, org_source text, new_key text)")
    conn.execute("create temp table if not exists mig_links "
                 "(observation_id text, photo_id integer, position integer)")
    conn.execute("delete from mig_keys")
    conn.execute("delete from mig_links")
    conn.executemany("insert into mig_keys values (?, ?, ?)",
                     [(k, r["org_source"], r["observation_id"]) for k, r in wrong.items()])
    conn.execute("insert into mig_links select op.observation_id, op.photo_id, op.position "
                 "from observation_photos op join mig_keys k on k.observation_id = op.observation_id")
    conn.execute("create index if not exists temp.mig_links_photo on mig_links(photo_id)")
    # Photos also linked to a record that keeps them: their vectors stay.
    shared = {pid for (pid,) in conn.execute(
        "select distinct op.photo_id from observation_photos op join mig_links w "
        "on w.photo_id = op.photo_id where op.observation_id not in "
        "(select observation_id from mig_keys)")}

    def grouped(sql: str) -> dict[int, list[dict]]:
        out: dict[int, list[dict]] = {}
        for row in _rows(conn, sql, ()):
            out.setdefault(row["photo_id"], []).append(row)
        return out

    photo_rows = grouped("select p.* from photos p where p.photo_id in "
                         "(select photo_id from mig_links)")
    copies = grouped("select c.* from photo_copies c where c.photo_id in "
                     "(select photo_id from mig_links)")
    vectors = (grouped("select e.* from embeddings e join (select distinct photo_id from "
                       "mig_links) w on w.photo_id = e.photo_id")
               if _has_table(conn, "embeddings") else {})
    links: dict[str, list[int]] = {}
    for key, pid in conn.execute("select observation_id, photo_id from mig_links "
                                 "order by observation_id, position"):
        links.setdefault(key, []).append(pid)

    audit = []
    drop_vectors = set()
    for key, live in sorted(wrong.items()):
        reason = (f"org source {live['org_source']!r}: photos were iNat observation "
                  f"{key}'s, not this record's")
        if not links.get(key):
            audit.append((key, None, live["org_source"], live["observation_id"], reason,
                          None, None, None, now, MIGRATION))
        for pid in links.get(key, []):
            kept = pid in shared
            photo = photo_rows.get(pid, [None])[0]
            audit.append((key, pid, live["org_source"], live["observation_id"], reason,
                          json.dumps(photo), json.dumps(copies.get(pid, [])),
                          json.dumps([] if kept else vectors.get(pid, [])), now, MIGRATION))
            if not kept:
                drop_vectors.add(pid)
    conn.executemany(
        "insert into source_removals (observation_id, photo_id, org_source, new_key, reason, "
        "photo_json, copies_json, embeddings_json, removed_at, migration) "
        "values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", audit)
    removed_vectors = sum(len(vectors.get(p, [])) for p in drop_vectors)
    if drop_vectors and _has_table(conn, "embeddings"):
        conn.execute("delete from embeddings where photo_id in (select photo_id from mig_links) "
                     "and photo_id in (select value from json_each(?))",
                     (json.dumps(sorted(drop_vectors)),))
    conn.execute("delete from observation_photos where observation_id in "
                 "(select observation_id from mig_keys)")
    conn.execute("delete from inat_observations where observation_id in "
                 "(select observation_id from mig_keys)")
    detail = {"wrong_records": len(wrong), "by_source": plan["by_source"],
              "photos_unlinked": sum(1 for a in audit if a[1] is not None),
              "embeddings_removed": removed_vectors}
    conn.execute("insert into manifest_migrations (name, applied_at, code_version, detail) "
                 "values (?, ?, ?, ?)", (MIGRATION, now, code_version, json.dumps(detail)))
    conn.execute("drop table mig_links")
    conn.execute("drop table mig_keys")
    return detail


def _has_table(conn: sqlite3.Connection, name: str) -> bool:
    return bool(conn.execute("select 1 from sqlite_master where type = 'table' and name = ?",
                             (name,)).fetchone())
