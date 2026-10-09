"""The local manifest: which records, observations and photos we hold, and where each came from.

Every photo keeps its owner and the license it carried when we last checked, so a
contributor's photos can be listed, asked about, or dropped at any time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .holdouts import SCHEMA as HOLDOUTS_SCHEMA
from .permissions import SCHEMA as PERMISSIONS_SCHEMA
from .sources import SCHEMA as SOURCES_SCHEMA

SCHEMA = """
create table if not exists records (
  observation_id   text primary key,   -- the key: the iNat id for iNat, '<source>:<id>' otherwise (sources.record_key)
  source           text not null,      -- inat | mo | mycoportal | com_sequence | genbank | unknown (sources.py);
                                       -- older code guessed 'inat' from a numeric id: see sources.migrated
  org_source       text,               -- .org's observations.source as spelt there
  source_id        text,               -- the id at the source (.org observations.observation_id)
  scientific_name  text,               -- the DNA-validated name on .org: the training label
  phylum text, class text, "order" text, family text, genus text, species text, infraspecies text,
  latitude real, longitude real,
  observed_on      text,
  state text, country text, continent text,
  north_america    integer not null,
  sequence_id      integer,
  green_projects   text,               -- JSON list of the projects that marked the record green
  validated_on     text,               -- earliest green validation date (ISO), for weekly test sets
  label_conflict   integer not null default 0,  -- .org holds more than one name for the record
  names_json       text,
  exported_at      text not null,
  first_seen_at    text                -- export that first listed it; null = before tracking
);

create table if not exists inat_observations (
  observation_id   text primary key,
  status           text not null,      -- 'ok' | 'missing' (deleted, private or not returned)
  uuid text, quality_grade text, geoprivacy text, obscured integer, taxon_geoprivacy text,
  positional_accuracy integer, observed_on text, inat_latitude real, inat_longitude real,
  user_id integer, user_login text, user_name text,
  taxon_id integer, taxon_name text, taxon_rank text, taxon_ancestor_ids text,
  photo_count integer,
  fetched_at text not null
);

create table if not exists photos (
  photo_id         integer primary key,  -- iNat photo id; an MO image is MO_PHOTO_ID_BASE + its id (mo.py)
  source           text not null default 'inat',  -- where the photo is from: 'inat' | 'mo'
  source_photo_id  integer,              -- the id at the source (an MO image id)
  owner_user_id    integer,              -- iNat user id; null for MO (permissions are iNat accounts)
  source_owner_id  integer,              -- the owner's id at a non-iNat source (an MO user id)
  owner_login      text,
  owner_name       text,
  license_code     text,                 -- '' = all rights reserved
  license_class    text not null,        -- 'open' | 'nc' | 'arr' (see licenses.py)
  license_text     text,                 -- the licence as the source names it (MO)
  ok_for_export    integer,              -- MO's ok_for_export flag (null for iNat)
  attribution      text,
  source_url       text not null,        -- the URL iNat gave (square size)
  host             text,
  first_seen_at    text not null,
  license_checked_at text not null,
  status           text not null default 'pending',  -- pending | done | missing | error
  local_path       text,                 -- relative path inside `store`
  store            text,                 -- where the file is: a folder or s3://bucket/prefix/
  size             text,
  bytes            integer,
  sha256           text,
  attempts         integer not null default 0,
  error            text,
  downloaded_at    text
);
create index if not exists photos_status_idx on photos(status, host);
create index if not exists photos_owner_idx on photos(owner_login);

-- Mushroom Observer observations of MO records (mo.py): what MO's images API answered.
-- No MO location is stored, so none can be shown.
create table if not exists mo_observations (
  observation_id   text primary key,     -- records.observation_id ('mo:<n>')
  mo_id            integer not null,     -- the MO observation number
  status           text not null,        -- 'ok' | 'missing' (MO answered no images)
  owner_id integer, owner_login text, owner_name text,
  image_count      integer,
  fetched_at       text not null
);

create table if not exists observation_photos (
  observation_id   text not null,
  photo_id         integer not null,
  position         integer,
  primary key (observation_id, photo_id)
);

-- Every held copy of a photo: one per size per store (a local folder or s3://bucket/).
-- A photo can be medium on a laptop and large in S3 at the same time.
create table if not exists photo_copies (
  photo_id       integer not null,
  store          text not null,
  size           text not null,
  path           text not null,        -- relative to the store
  bytes          integer,
  sha256         text,
  downloaded_at  text,
  primary key (photo_id, store, size)
);

-- A row each time a photo's license is seen to change (including the first sighting).
create table if not exists license_history (
  photo_id         integer not null,
  license_code     text,
  seen_at          text not null
);
""" + PERMISSIONS_SCHEMA + HOLDOUTS_SCHEMA + SOURCES_SCHEMA


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma foreign_keys=on")
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    _backfill_copies(conn, local_store=str(path.parent))
    if new:
        # A manifest made by this code never held a guessed source (sources.py).
        mark_new_manifest(conn)
    return conn


def mark_new_manifest(conn: sqlite3.Connection) -> None:
    from .sources import MIGRATION
    with conn:
        conn.execute("insert or ignore into manifest_migrations (name, applied_at, detail) "
                     "values (?, datetime('now'), '{\"new_manifest\": true}')", (MIGRATION,))


def _backfill_copies(conn: sqlite3.Connection, local_store: str) -> None:
    """Manifests from before photo_copies held one copy per photo in the photos row.
    Copy those into photo_copies once; a row with no store was downloaded to the
    manifest's own data folder."""
    if conn.execute("select 1 from photo_copies limit 1").fetchone():
        return
    with conn:
        conn.execute(
            "insert or ignore into photo_copies (photo_id, store, size, path, bytes, sha256, "
            "downloaded_at) select photo_id, coalesce(store, ?), size, local_path, bytes, "
            "sha256, downloaded_at from photos where status = 'done' and local_path is not null "
            "and size is not null", (local_store,))


# Columns added after a manifest may already exist; `create table if not exists`
# does not add them, so they are added here.
_LATER_COLUMNS = {
    "photos": {"store": "text",
               # Every photo before MO photos came from iNat (even the ones fetched for the
               # wrong record were iNat's): 'inat' is a fact for those rows, not a guess.
               "source": "text not null default 'inat'", "source_photo_id": "integer",
               "source_owner_id": "integer", "license_text": "text", "ok_for_export": "integer"},
    "records": {"first_seen_at": "text", "org_source": "text", "source_id": "text"},
}


def snapshot(conn: sqlite3.Connection, dest: Path) -> Path:
    """A consistent copy of the manifest while it is in use (safe to upload)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    tmp.unlink(missing_ok=True)
    out = sqlite3.connect(tmp)
    try:
        conn.backup(out)
    finally:
        out.close()
    tmp.replace(dest)
    return dest


# A benchmark's own tables (heldout.py: its answer key, .org's true coordinates for its
# records, its answers) stay on this machine. benchmark_holdouts is not one of them: the
# box's nightly update needs it.
BENCHMARK_TABLES = "heldout_%"


def shippable_snapshot(conn: sqlite3.Connection, dest: Path) -> Path:
    """snapshot() for a copy that leaves this machine (a release, a trainer or downloader
    run): without the benchmark tables, vacuumed so nothing of them stays in free pages."""
    snapshot(conn, dest)
    out = sqlite3.connect(dest)
    try:
        doomed = [r[0] for r in out.execute(
            "select name from sqlite_master where type = 'table' and name like ?",
            (BENCHMARK_TABLES,))]
        for name in doomed:
            out.execute(f'drop table "{name}"')
        out.commit()
        if doomed:
            out.execute("vacuum")
    finally:
        out.close()
    return dest


def upgrade(conn: sqlite3.Connection) -> None:
    """Bring a manifest opened without connect() (the API, a release copy made by older
    code) up to this schema: missing tables and columns are added, nothing else. Does
    nothing (and commits nothing) when it is already current."""
    have = {r[0] for r in conn.execute("select name from sqlite_master where type = 'table'")}
    missing_tables = {"records", "photos", "mo_observations", "manifest_migrations",
                      "source_removals"} - have
    missing_cols = any(
        name not in {r[1] for r in conn.execute(f"pragma table_info({table})")}
        for table, cols in _LATER_COLUMNS.items() for name in cols)
    if missing_tables:
        conn.executescript(SCHEMA)
    if missing_tables or missing_cols:
        _add_missing_columns(conn)


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, cols in _LATER_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"pragma table_info({table})")}
        for name, decl in cols.items():
            if name not in have:
                conn.execute(f"alter table {table} add column {name} {decl}")
    conn.commit()
