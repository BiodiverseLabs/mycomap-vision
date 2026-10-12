"""Where a green record is, when .org doesn't say (Steve, 2026-10-12).

.org marks a record North American from its continent field. A record with no continent,
country or coordinates on .org was simply counted "not North America" and never fetched,
though nobody knew where it was (892 at the v1 audit: 555 iNat, 337 MO). This asks each
record's source, read-only and anonymously, and keeps its answer:

- iNat: the observation's public `place_ids` (v2 API, 1 request a second, 200 ids a
  request); North America is iNat place 97394 (checked 2026-10-12). An obscured record's
  places are still public at country and continent level.
- MO: the country at the end of the record's location name (mopoints.py reads it with the
  public point), against NORTH_AMERICA_COUNTRIES.

The answer goes in `place_fallback`, and `apply` sets records.north_america = 1 for a record
.org leaves placeless whose source puts it in North America. records.save_records applies
it after every export, so a re-export keeps it, and everything that reads the flag (the
fetches, the photo download, embedding, the dataset release) treats the record like any
other North American one. Nothing here reads or stores .org's own point.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import requests

from . import config
from .inat import chunks
from .ratelimit import MinInterval

INAT_API = "https://api.inaturalist.org/v2/observations"
INAT_NORTH_AMERICA = 97394
BATCH = 200
NORTH_AMERICA_COUNTRIES = frozenset(c.lower() for c in (
    "USA", "United States", "Canada", "Mexico", "Greenland", "Bermuda",
    "Saint Pierre and Miquelon", "Bahamas", "Cuba", "Cayman Islands", "Jamaica", "Haiti",
    "Dominican Republic", "Puerto Rico", "U.S. Virgin Islands", "US Virgin Islands",
    "Virgin Islands", "British Virgin Islands", "Anguilla", "Saint Kitts and Nevis",
    "Antigua and Barbuda", "Montserrat", "Guadeloupe", "Dominica", "Martinique",
    "Saint Lucia", "Saint Vincent and the Grenadines", "Barbados", "Grenada",
    "Trinidad and Tobago", "Turks and Caicos Islands", "Belize", "Guatemala", "Honduras",
    "El Salvador", "Nicaragua", "Costa Rica", "Panama"))

SCHEMA = """
create table if not exists place_fallback (
  observation_id   text primary key,     -- records.observation_id
  source           text not null,        -- 'inat' | 'mo'
  north_america    integer not null,     -- what the source's own place says
  how              text not null,        -- 'inat_place_ids' | 'mo_location_name' | 'not_returned'
  country          text,                 -- MO: the location name's country
  read_at          text not null
);
"""

PLACELESS_SQL = ("r.north_america = 0 and coalesce(r.country, '') = '' and "
                 "coalesce(r.continent, '') = '' and r.latitude is null")


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)


def placeless(conn: sqlite3.Connection, source: str) -> list[tuple[str, str]]:
    """(record key, source id) of `source` records .org gives no place at all."""
    return [(k, s) for k, s in conn.execute(
        f"select r.observation_id, r.source_id from records r where r.source = ? and "
        f"{PLACELESS_SQL} order by r.observation_id", (source,))]


def country_of(location_name: str | None) -> str | None:
    if not location_name:
        return None
    return location_name.rsplit(",", 1)[-1].strip() or None


def in_north_america(country: str | None) -> bool:
    return bool(country) and country.lower() in NORTH_AMERICA_COUNTRIES


def fetch_inat(conn: sqlite3.Connection, session=None, limiter=None, log=print) -> dict:
    ensure_schema(conn)
    todo = [k for k, _ in placeless(conn, "inat")]
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
    limiter = limiter or MinInterval(1.0)
    na = 0
    for batch in chunks(todo, BATCH):
        params = {"id": ",".join(batch), "per_page": str(BATCH), "fields": "id,place_ids"}
        limiter.wait()
        resp = session.get(INAT_API, params=params, timeout=60)
        if resp.status_code != 200:
            raise RuntimeError(f"iNat answered {resp.status_code}: {resp.text[:300]}")
        places = {str(o["id"]): set(o.get("place_ids") or [])
                  for o in resp.json().get("results") or []}
        at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with conn:
            for key in batch:
                got = places.get(key)
                is_na = int(got is not None and INAT_NORTH_AMERICA in got)
                na += is_na
                conn.execute("insert or replace into place_fallback values (?,?,?,?,?,?)",
                             (key, "inat", is_na,
                              "inat_place_ids" if got is not None else "not_returned", None, at))
    log(f"  iNat: {len(todo):,} placeless records asked, {na:,} in North America")
    return {"inat_asked": len(todo), "inat_north_america": na}


def decide_mo(conn: sqlite3.Connection) -> dict:
    """MO's answer from the location names mopoints.py read (run `mv fetch-mo-points`)."""
    ensure_schema(conn)
    from . import mopoints
    mopoints.ensure_schema(conn)
    asked = na = unread = 0
    with conn:
        for key, _ in placeless(conn, "mo"):
            row = conn.execute("select location_name, fetched_at, status from mo_public_points "
                               "where observation_id = ?", (key,)).fetchone()
            if row is None:
                unread += 1
                continue
            name, at, status = row
            country = country_of(name)
            is_na = int(status == "ok" and in_north_america(country))
            asked += 1
            na += is_na
            conn.execute("insert or replace into place_fallback values (?,?,?,?,?,?)",
                         (key, "mo", is_na, "mo_location_name" if status == "ok"
                          else "not_returned", country, at))
    return {"mo_decided": asked, "mo_north_america": na, "mo_points_not_read": unread}


def apply(conn: sqlite3.Connection) -> int:
    """Mark North American the placeless records whose source says so. Returns how many
    records changed now (0 when already applied)."""
    if not conn.execute("select 1 from sqlite_master where type = 'table' and "
                        "name = 'place_fallback'").fetchone():
        return 0
    with conn:
        return conn.execute(
            f"update records set north_america = 1 where observation_id in (select "
            f"observation_id from place_fallback where north_america = 1) and "
            f"{PLACELESS_SQL.replace('r.', '')}").rowcount


def from_source_place(conn: sqlite3.Connection) -> int:
    """Records counted North American on their source's word (for a release's record)."""
    if not conn.execute("select 1 from sqlite_master where type = 'table' and "
                        "name = 'place_fallback'").fetchone():
        return 0
    return conn.execute("select count(*) from records r join place_fallback f using "
                        "(observation_id) where f.north_america = 1 and r.north_america = 1"
                        ).fetchone()[0]
