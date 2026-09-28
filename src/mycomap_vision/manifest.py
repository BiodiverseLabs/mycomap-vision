"""The local manifest: which records, observations and photos we hold, and where each came from.

Every photo keeps its owner and the license it carried when we last checked, so a
contributor's photos can be listed, asked about, or dropped at any time.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA = """
create table if not exists records (
  observation_id   text primary key,   -- .org observations.observation_id (an iNat id for iNat records)
  source           text not null,      -- 'inat' when the id is numeric, else 'other'
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
  photo_id         integer primary key,  -- iNat photo id
  owner_user_id    integer,
  owner_login      text,
  owner_name       text,
  license_code     text,                 -- '' = all rights reserved
  license_class    text not null,        -- 'open' | 'nc' | 'arr' (see licenses.py)
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

create table if not exists observation_photos (
  observation_id   text not null,
  photo_id         integer not null,
  position         integer,
  primary key (observation_id, photo_id)
);

-- A row each time a photo's license is seen to change (including the first sighting).
create table if not exists license_history (
  photo_id         integer not null,
  license_code     text,
  seen_at          text not null
);
"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("pragma journal_mode=wal")
    conn.execute("pragma foreign_keys=on")
    conn.executescript(SCHEMA)
    _add_missing_columns(conn)
    return conn


# Columns added after a manifest may already exist; `create table if not exists`
# does not add them, so they are added here.
_LATER_COLUMNS = {"photos": {"store": "text"}, "records": {"first_seen_at": "text"}}


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


def _add_missing_columns(conn: sqlite3.Connection) -> None:
    for table, cols in _LATER_COLUMNS.items():
        have = {r[1] for r in conn.execute(f"pragma table_info({table})")}
        for name, decl in cols.items():
            if name not in have:
                conn.execute(f"alter table {table} add column {name} {decl}")
    conn.commit()
