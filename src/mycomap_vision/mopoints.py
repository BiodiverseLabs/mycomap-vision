"""Mushroom Observer records' PUBLIC point, for dataset releases (Steve, 2026-10-12, D1 of
docs/atlas-environment.md): what anyone sees on MO, frozen into a release's private part so
a later environment sidecar (v1-env) is sampled only where the source itself shows the find.

    GET https://mushroomobserver.org/api2/observations?id=<n,...>&detail=high&format=json

answers per observation `latitude` / `longitude` (null when the observer gave no GPS, and
for an anonymous request when the GPS is hidden), `gps_hidden`, and `location` with the
box edges `latitude_north/south`, `longitude_east/west`. Read anonymously (no account, no
key), so a hidden GPS stays hidden; never .org's own point. Same pace and cache as mo.py:
MO's 20 requests a minute, at least 5 s apart (mo.Pacer), each answer kept gzipped in
data/raw/mo/, nothing asked twice unless `refresh`.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config, sources
from .inat import chunks
from .mo import MissingObservation, Pacer, _NOT_FOUND

API = "https://mushroomobserver.org/api2/observations"
BATCH = 50

SCHEMA = """
create table if not exists mo_public_points (
  observation_id   text primary key,     -- records.observation_id ('mo:<n>')
  latitude real, longitude real,         -- MO's shown GPS; null when absent or hidden
  gps_hidden       integer,
  box_north real, box_south real, box_east real, box_west real,   -- MO's location box
  location_name    text,                 -- MO's public location name (its country: placefallback.py)
  status           text not null,        -- 'ok' | 'missing' (MO has no such observation)
  fetched_at       text not null
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    cols = {r[1] for r in conn.execute("pragma table_info(mo_public_points)")}
    if "location_name" not in cols:              # a table written before 2026-10-12
        conn.execute("alter table mo_public_points add column location_name text")


def _num(v):
    try:
        return None if v in (None, "") else float(v)
    except (TypeError, ValueError):
        return None


def parse(obs: dict) -> dict:
    """One api2 observation (detail=high) -> its public point fields."""
    loc = obs.get("location") if isinstance(obs.get("location"), dict) else {}
    hidden = bool(obs.get("gps_hidden"))
    lat, lng = _num(obs.get("latitude")), _num(obs.get("longitude"))
    if hidden:                       # anonymous answers carry none; never keep one anyway
        lat = lng = None
    return {"mo_id": int(obs["id"]), "latitude": lat, "longitude": lng,
            "gps_hidden": int(hidden),
            "box_north": _num(loc.get("latitude_north")), "box_south": _num(loc.get("latitude_south")),
            "box_east": _num(loc.get("longitude_east")), "box_west": _num(loc.get("longitude_west")),
            "location_name": loc.get("name") or None}


def pending(conn: sqlite3.Connection, refresh: bool = False) -> dict[int, str]:
    """MO number -> record key, for MO records whose public point isn't read yet."""
    sources.require_migrated(conn, "reading Mushroom Observer public points")
    ensure_schema(conn)
    where = f"r.source = '{sources.MO}'" + ("" if refresh else " and p.observation_id is null")
    out = {}
    for key, sid in conn.execute(
            "select r.observation_id, r.source_id from records r left join mo_public_points p "
            f"on p.observation_id = r.observation_id where {where} "
            "order by cast(r.source_id as integer)"):
        try:
            out[int(sid)] = key
        except (TypeError, ValueError):
            continue
    return out


def fetch_batch(session, mo_ids: list[int], pacer: Pacer) -> list[dict]:
    out, page, pages = [], 1, 1
    while page <= pages:
        params = {"id": ",".join(str(i) for i in mo_ids), "detail": "high", "format": "json",
                  "page": str(page)}
        for attempt in range(6):
            pacer.wait()
            try:
                resp = session.get(API, params=params, timeout=60)
            except requests.RequestException:
                pacer.limiter.pause(30 * (attempt + 1))
                continue
            if resp.status_code == 200:
                break
            if resp.status_code == 429 or resp.status_code >= 500:
                pacer.limiter.pause(60 * (attempt + 1))
                continue
            raise RuntimeError(f"MO answered {resp.status_code}: {resp.text[:300]}")
        else:
            raise RuntimeError("MO kept failing; stopped. Re-run to resume.")
        body = resp.json()
        pacer.ran(body.get("run_time"))
        for err in body.get("errors") or []:
            m = _NOT_FOUND.search(str(err.get("details") or ""))
            if err.get("code") == "API2::ObjectNotFoundByID" and m:
                raise MissingObservation(int(m.group(1)))
        if body.get("errors"):
            raise RuntimeError(f"MO answered errors: {json.dumps(body['errors'])[:300]}")
        out.extend(body.get("results") or [])
        pages = int(body.get("number_of_pages") or 1)
        page += 1
    return out


def save(conn: sqlite3.Connection, keys: dict[int, str], answers: list[dict],
         fetched_at: str) -> dict:
    """Store what MO answered for these observations; one it didn't answer is 'missing'."""
    ensure_schema(conn)
    got = {p["mo_id"]: p for p in map(parse, answers) if p["mo_id"] in keys}
    with conn:
        for n, key in keys.items():
            p = got.get(n)
            conn.execute(
                "insert or replace into mo_public_points values (?,?,?,?,?,?,?,?,?,?,?)",
                (key, *(p[k] for k in ("latitude", "longitude", "gps_hidden", "box_north",
                                       "box_south", "box_east", "box_west", "location_name")),
                 "ok", fetched_at) if p else (key, None, None, None, None, None, None, None,
                                              None, "missing", fetched_at))
    return {"ok": len(got), "missing": len(keys) - len(got)}


def fetch_all(conn: sqlite3.Connection, refresh: bool = False, limit: int | None = None,
              raw_dir: Path | None = None, session=None, pacer: Pacer | None = None,
              fetch=fetch_batch, log=print) -> dict:
    config.ensure_dirs()
    todo = pending(conn, refresh)
    ids = sorted(todo)[:limit] if limit is not None else sorted(todo)
    raw_dir = raw_dir or config.RAW_DIR / "mo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
    pacer = pacer or Pacer()
    totals = {"requested": len(ids), "ok": 0, "missing": 0, "batches": 0}
    started = time.monotonic()
    for batch in chunks(ids, BATCH):
        ask = list(batch)
        while True:
            try:
                answers = fetch(session, ask, pacer) if ask else []
                break
            except MissingObservation as e:
                if e.mo_id not in ask:
                    raise
                ask.remove(e.mo_id)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with gzip.open(raw_dir / f"observations-{batch[0]}-{fetched_at[:10]}.json.gz", "wt",
                       encoding="utf-8") as f:
            json.dump(answers, f)
        stats = save(conn, {n: todo[n] for n in batch}, answers, fetched_at)
        totals["ok"] += stats["ok"]
        totals["missing"] += stats["missing"]
        totals["batches"] += 1
        if totals["batches"] % 10 == 0:
            log(f"  {totals['ok'] + totals['missing']:,}/{len(ids):,} MO public points, "
                f"{time.monotonic() - started:.0f}s")
    return totals
