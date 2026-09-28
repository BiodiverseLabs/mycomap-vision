"""Fetch iNat observation metadata (photos, licenses, owners) for the green records.

Uses API v2 with a field selection (~1 KB per observation) at one request per
second, below iNat's requested 60 requests a minute.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from datetime import datetime, timezone
from typing import Iterable, Iterator

import requests

from . import config
from .licenses import host_of, license_class
from .ratelimit import MinInterval

API = "https://api.inaturalist.org/v2/observations"
BATCH = 200
FIELDS = (
    "(id:!t,uuid:!t,quality_grade:!t,geoprivacy:!t,obscured:!t,taxon_geoprivacy:!t,"
    "positional_accuracy:!t,observed_on:!t,location:!t,"
    "user:(id:!t,login:!t,name:!t),"
    "taxon:(id:!t,name:!t,rank:!t,ancestor_ids:!t),"
    "observation_photos:(position:!t,photo:(id:!t,license_code:!t,url:!t,attribution:!t)))"
)


def chunks(items: list[str], n: int) -> Iterator[list[str]]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def _latlng(location: str | None) -> tuple[float | None, float | None]:
    if not location or "," not in location:
        return None, None
    try:
        lat, lng = location.split(",", 1)
        return float(lat), float(lng)
    except ValueError:
        return None, None


def parse_observation(obs: dict) -> tuple[dict, list[dict]]:
    """One v2 result -> (observation row, photo rows in position order)."""
    user = obs.get("user") or {}
    taxon = obs.get("taxon") or {}
    lat, lng = _latlng(obs.get("location"))
    photos = []
    for op in sorted(obs.get("observation_photos") or [], key=lambda p: p.get("position") or 0):
        p = op.get("photo") or {}
        if not p.get("id") or not p.get("url"):
            continue
        photos.append({
            "photo_id": int(p["id"]),
            "position": op.get("position"),
            "license_code": (p.get("license_code") or "").lower(),
            "license_class": license_class(p.get("license_code")),
            "attribution": p.get("attribution"),
            "source_url": p["url"],
            "host": host_of(p["url"]),
            "owner_user_id": user.get("id"),
            "owner_login": user.get("login"),
            "owner_name": user.get("name"),
        })
    row = {
        "observation_id": str(obs["id"]),
        "status": "ok",
        "uuid": obs.get("uuid"),
        "quality_grade": obs.get("quality_grade"),
        "geoprivacy": obs.get("geoprivacy"),
        "obscured": int(bool(obs.get("obscured"))),
        "taxon_geoprivacy": obs.get("taxon_geoprivacy"),
        "positional_accuracy": obs.get("positional_accuracy"),
        "observed_on": obs.get("observed_on"),
        "inat_latitude": lat,
        "inat_longitude": lng,
        "user_id": user.get("id"),
        "user_login": user.get("login"),
        "user_name": user.get("name"),
        "taxon_id": taxon.get("id"),
        "taxon_name": taxon.get("name"),
        "taxon_rank": taxon.get("rank"),
        "taxon_ancestor_ids": json.dumps(taxon.get("ancestor_ids") or []),
        "photo_count": len(photos),
    }
    return row, photos


_OBS_COLS = [
    "observation_id", "status", "uuid", "quality_grade", "geoprivacy", "obscured",
    "taxon_geoprivacy", "positional_accuracy", "observed_on", "inat_latitude",
    "inat_longitude", "user_id", "user_login", "user_name", "taxon_id", "taxon_name",
    "taxon_rank", "taxon_ancestor_ids", "photo_count", "fetched_at",
]


def save_batch(conn: sqlite3.Connection, requested: Iterable[str], results: list[dict],
               fetched_at: str) -> dict:
    """Store one API batch. Requested ids iNat didn't return are marked missing."""
    requested = list(requested)
    seen: set[str] = set()
    stats = {"ok": 0, "missing": 0, "photos_new": 0, "license_changes": 0}
    cols = ", ".join(_OBS_COLS)
    marks = ", ".join("?" for _ in _OBS_COLS)
    updates = ", ".join(f"{c} = excluded.{c}" for c in _OBS_COLS if c != "observation_id")
    upsert_obs = (f"insert into inat_observations ({cols}) values ({marks}) "
                  f"on conflict(observation_id) do update set {updates}")
    with conn:
        for obs in results:
            row, photos = parse_observation(obs)
            row["fetched_at"] = fetched_at
            seen.add(row["observation_id"])
            conn.execute(upsert_obs, [row[c] for c in _OBS_COLS])
            stats["ok"] += 1
            # Replace the photo list: owners can remove or reorder photos.
            conn.execute("delete from observation_photos where observation_id = ?",
                         (row["observation_id"],))
            for p in photos:
                conn.execute(
                    "insert or replace into observation_photos (observation_id, photo_id, position) "
                    "values (?, ?, ?)", (row["observation_id"], p["photo_id"], p["position"]))
                prev = conn.execute("select license_code from photos where photo_id = ?",
                                    (p["photo_id"],)).fetchone()
                if prev is None:
                    conn.execute(
                        "insert into photos (photo_id, owner_user_id, owner_login, owner_name, "
                        "license_code, license_class, attribution, source_url, host, "
                        "first_seen_at, license_checked_at) values (?,?,?,?,?,?,?,?,?,?,?)",
                        (p["photo_id"], p["owner_user_id"], p["owner_login"], p["owner_name"],
                         p["license_code"], p["license_class"], p["attribution"],
                         p["source_url"], p["host"], fetched_at, fetched_at))
                    conn.execute("insert into license_history values (?, ?, ?)",
                                 (p["photo_id"], p["license_code"], fetched_at))
                    stats["photos_new"] += 1
                else:
                    if (prev["license_code"] or "") != p["license_code"]:
                        conn.execute("insert into license_history values (?, ?, ?)",
                                     (p["photo_id"], p["license_code"], fetched_at))
                        stats["license_changes"] += 1
                    # A license change can move the photo between hosts; keep the URL current.
                    conn.execute(
                        "update photos set owner_user_id = ?, owner_login = ?, owner_name = ?, "
                        "license_code = ?, license_class = ?, attribution = ?, source_url = ?, "
                        "host = ?, license_checked_at = ? where photo_id = ?",
                        (p["owner_user_id"], p["owner_login"], p["owner_name"], p["license_code"],
                         p["license_class"], p["attribution"], p["source_url"], p["host"],
                         fetched_at, p["photo_id"]))
        for oid in requested:
            if oid not in seen:
                conn.execute(
                    "insert into inat_observations (observation_id, status, fetched_at) "
                    "values (?, 'missing', ?) on conflict(observation_id) do update set "
                    "status = 'missing', fetched_at = excluded.fetched_at", (oid, fetched_at))
                stats["missing"] += 1
    return stats


def pending_ids(conn: sqlite3.Connection, refresh: bool, north_america_only: bool) -> list[str]:
    where = ["r.source = 'inat'"]
    if north_america_only:
        where.append("r.north_america = 1")
    if not refresh:
        where.append("o.observation_id is null")
    sql = ("select r.observation_id from records r left join inat_observations o "
           "on o.observation_id = r.observation_id where " + " and ".join(where) +
           " order by cast(r.observation_id as integer)")
    return [row[0] for row in conn.execute(sql)]


def fetch_batch(session: requests.Session, ids: list[str], limiter: MinInterval) -> list[dict]:
    params = {"id": ",".join(ids), "per_page": str(BATCH), "fields": FIELDS}
    for attempt in range(6):
        limiter.wait()
        try:
            resp = session.get(API, params=params, timeout=60)
        except requests.RequestException:
            limiter.pause(10 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json().get("results") or []
        if resp.status_code == 429 or resp.status_code >= 500:
            limiter.pause(30 * (attempt + 1))
            continue
        raise RuntimeError(f"iNat answered {resp.status_code}: {resp.text[:300]}")
    raise RuntimeError("iNat kept failing; stopped. Re-run to resume.")


def fetch_all(conn: sqlite3.Connection, refresh: bool = False, north_america_only: bool = True,
              limit: int | None = None, log=print) -> dict:
    config.ensure_dirs()
    ids = pending_ids(conn, refresh, north_america_only)
    if limit is not None:
        ids = ids[:limit]
    raw_dir = config.RAW_DIR / "inat"
    raw_dir.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    limiter = MinInterval(1.0)
    totals = {"ok": 0, "missing": 0, "photos_new": 0, "license_changes": 0, "batches": 0}
    started = time.monotonic()
    for batch in chunks(ids, BATCH):
        results = fetch_batch(session, batch, limiter)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with gzip.open(raw_dir / f"obs-{batch[0]}-{fetched_at[:10]}.json.gz", "wt",
                       encoding="utf-8") as f:
            json.dump(results, f)
        stats = save_batch(conn, batch, results, fetched_at)
        for k, v in stats.items():
            totals[k] += v
        totals["batches"] += 1
        if totals["batches"] % 25 == 0:
            done = totals["ok"] + totals["missing"]
            log(f"  {done:,}/{len(ids):,} observations, {totals['photos_new']:,} new photos, "
                f"{time.monotonic() - started:.0f}s")
    totals["requested"] = len(ids)
    return totals
