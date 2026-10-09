"""Mushroom Observer photos for DNA-validated MO records: read-only, within MO's limits.

Steve, 2026-10-09: "We want to include our validated MO records", with their own MO
photos. An MO record (records.source = 'mo', key 'mo:<n>', sources.py) gets the images
MO lists for observation <n>, read from MO's images API:

    GET https://mushroomobserver.org/api2/images?observation=<n,...>&detail=high&format=json

which answers, per image: id, owner (id, login_name, legal_name), copyright_holder,
license, ok_for_export, the file URLs and the observations it belongs to. No MO location
is read or stored (the record's place is .org's, used for scores and never shown, as
for iNat records).

MO's limits (README_API, read 2026-10-09): anonymous traffic, the API included, is
limited to 20 requests a minute, "1 request every 5 seconds on average"; wait at least
5 seconds or the last answer's run_time, whichever is longer. Every request here, and
every image download (photos.py, the mushroomobserver.org host policy), keeps that
pace and sends config.USER_AGENT. MO says the API is not meant for bulk image work, so
we ask for observations in batches and cache every answer (data/raw/mo/), and nothing
is fetched twice unless asked (`refresh`).

Photos: an MO image is stored in `photos` with photos.source = 'mo', its MO id in
source_photo_id and photo_id = MO_PHOTO_ID_BASE + that id, so it never collides with an
iNat photo id. Its owner goes in source_owner_id / owner_login / owner_name, never in
owner_user_id (the iNat account the photo-permission answers are keyed by). Its licence
is kept as MO names it (license_text) and mapped to a licence code and class
(license_of); what may be used and shown is decided in permissions.py.
"""

from __future__ import annotations

import gzip
import json
import sqlite3
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import requests

from . import config, sources
from .inat import chunks
from .licenses import host_of, license_class
from .ratelimit import MinInterval

API = "https://mushroomobserver.org/api2/images"
MIN_INTERVAL = 5.0      # MO README_API: 20 requests/min, at least 5 s apart
BATCH = 40              # observations per request
MO_PHOTO_ID_BASE = 10 ** 12

# MO's licence names (lower-cased) -> a licence code (licenses.license_class reads it).
# Anything else counts as all rights reserved until a person adds it here.
_LICENSES = {
    "creative commons wikipedia compatible v3.0": "cc-by-sa",
    "creative commons attribution-sharealike 3.0": "cc-by-sa",
    "creative commons non-commercial v3.0": "cc-by-nc-sa",
    "creative commons non-commercial v2.5": "cc-by-nc-sa",
    "creative commons attribution-noncommercial-sharealike 3.0": "cc-by-nc-sa",
    "public domain": "pd",
}


def license_of(text: str | None) -> tuple[str, str]:
    """(licence code, class) for an MO licence name. Unknown or reserved: ('', 'arr')."""
    code = _LICENSES.get(" ".join((text or "").split()).lower(), "")
    return code, license_class(code)


def photo_id_of(mo_image_id: int) -> int:
    return MO_PHOTO_ID_BASE + int(mo_image_id)


def _thumb_url(img: dict) -> str | None:
    """The image's smallest file URL (our source_url, like iNat's square one)."""
    for url in img.get("files") or []:
        if "/images/thumb/" in url:
            return url
    return None


def parse_image(img: dict) -> dict | None:
    """One images-API result -> a photos row (None when it has no usable file URL)."""
    if not img.get("id"):
        return None
    url = _thumb_url(img)
    if not url or host_of(url) != "mushroomobserver.org":
        return None
    owner = img.get("owner") or img.get("user") or {}     # api2 nests it under owner
    code, cls = license_of(img.get("license"))
    holder = (img.get("copyright_holder") or "").strip()
    return {
        "photo_id": photo_id_of(img["id"]),
        "source_photo_id": int(img["id"]),
        "source_owner_id": owner.get("id"),
        "owner_login": owner.get("login_name") or owner.get("login"),
        "owner_name": owner.get("legal_name") or owner.get("name"),
        "license_code": code,
        "license_class": cls,
        "license_text": img.get("license"),
        "ok_for_export": None if img.get("ok_for_export") is None else int(bool(img["ok_for_export"])),
        "attribution": f"(c) {holder}, {img.get('license') or 'licence not stated'}"
                       if holder else None,
        "source_url": url,
        "host": host_of(url),
        "observation_ids": [int(o) for o in img.get("observation_ids") or []],
    }


def save_batch(conn: sqlite3.Connection, keys: dict[int, str], images: list[dict],
               fetched_at: str) -> dict:
    """Store one batch: `keys` maps each requested MO observation number to its record
    key. Images are linked to every requested observation they belong to; a requested
    observation with no images is 'missing'."""
    stats = {"ok": 0, "missing": 0, "photos_new": 0, "license_changes": 0, "photos": 0}
    per_obs: dict[int, list[dict]] = {n: [] for n in keys}
    for img in images:
        row = parse_image(img)
        if row is None:
            continue
        for n in row["observation_ids"]:
            if n in per_obs:
                per_obs[n].append(row)
    with conn:
        for n, key in keys.items():
            rows = sorted(per_obs[n], key=lambda r: r["source_photo_id"])
            owners = Counter((r["source_owner_id"], r["owner_login"], r["owner_name"])
                             for r in rows if r["owner_login"])
            owner = owners.most_common(1)[0][0] if owners else (None, None, None)
            status = "ok" if rows else "missing"
            stats[status] += 1
            conn.execute(
                "insert into mo_observations (observation_id, mo_id, status, owner_id, "
                "owner_login, owner_name, image_count, fetched_at) values (?,?,?,?,?,?,?,?) "
                "on conflict(observation_id) do update set mo_id = excluded.mo_id, "
                "status = excluded.status, owner_id = excluded.owner_id, "
                "owner_login = excluded.owner_login, owner_name = excluded.owner_name, "
                "image_count = excluded.image_count, fetched_at = excluded.fetched_at",
                (key, n, status, *owner, len(rows), fetched_at))
            conn.execute("delete from observation_photos where observation_id = ?", (key,))
            for pos, r in enumerate(rows):
                conn.execute("insert or replace into observation_photos (observation_id, "
                             "photo_id, position) values (?, ?, ?)", (key, r["photo_id"], pos))
                stats["photos"] += 1
                prev = conn.execute("select license_code, source from photos where photo_id = ?",
                                    (r["photo_id"],)).fetchone()
                if prev is not None and prev[1] != sources.MO:
                    raise RuntimeError(f"photo id {r['photo_id']} is already a {prev[1]} photo")
                if prev is None:
                    conn.execute(
                        "insert into photos (photo_id, source, source_photo_id, owner_user_id, "
                        "source_owner_id, owner_login, owner_name, license_code, license_class, "
                        "license_text, ok_for_export, attribution, source_url, host, "
                        "first_seen_at, license_checked_at) "
                        "values (?, 'mo', ?, null, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (r["photo_id"], r["source_photo_id"], r["source_owner_id"],
                         r["owner_login"], r["owner_name"], r["license_code"],
                         r["license_class"], r["license_text"], r["ok_for_export"],
                         r["attribution"], r["source_url"], r["host"], fetched_at, fetched_at))
                    conn.execute("insert into license_history values (?, ?, ?)",
                                 (r["photo_id"], r["license_code"], fetched_at))
                    stats["photos_new"] += 1
                else:
                    if (prev[0] or "") != r["license_code"]:
                        conn.execute("insert into license_history values (?, ?, ?)",
                                     (r["photo_id"], r["license_code"], fetched_at))
                        stats["license_changes"] += 1
                    conn.execute(
                        "update photos set source_owner_id = ?, owner_login = ?, owner_name = ?, "
                        "license_code = ?, license_class = ?, license_text = ?, ok_for_export = ?, "
                        "attribution = ?, source_url = ?, host = ?, license_checked_at = ? "
                        "where photo_id = ?",
                        (r["source_owner_id"], r["owner_login"], r["owner_name"],
                         r["license_code"], r["license_class"], r["license_text"],
                         r["ok_for_export"], r["attribution"], r["source_url"], r["host"],
                         fetched_at, r["photo_id"]))
    return stats


def pending(conn: sqlite3.Connection, refresh: bool = False,
            north_america_only: bool = True) -> dict[int, str]:
    """MO observation number -> record key, for MO records (by records.source only) not
    read from MO yet. Refused on a manifest whose sources are still guessed."""
    sources.require_migrated(conn, "fetching Mushroom Observer images")
    where = [f"r.source = '{sources.MO}'"]
    if north_america_only:
        where.append("r.north_america = 1")
    if not refresh:
        where.append("m.observation_id is null")
    out = {}
    for key, sid in conn.execute(
            "select r.observation_id, r.source_id from records r left join mo_observations m "
            "on m.observation_id = r.observation_id where " + " and ".join(where) +
            " order by cast(r.source_id as integer)"):
        try:
            out[int(sid)] = key
        except (TypeError, ValueError):
            continue               # an MO record whose MO number isn't a number: nothing to ask
    return out


class Pacer:
    """MO's pace: at least MIN_INTERVAL between requests, and at least the last answer's
    run_time (README_API)."""

    def __init__(self, interval: float = MIN_INTERVAL):
        self.limiter = MinInterval(interval)

    def wait(self) -> None:
        self.limiter.wait()

    def ran(self, run_time: float | None) -> None:
        if run_time and run_time > self.limiter.interval:
            self.limiter.pause(run_time)


def fetch_batch(session: requests.Session, mo_ids: list[int], pacer: Pacer) -> list[dict]:
    """Every image of these observations, all pages."""
    out: list[dict] = []
    page, pages = 1, 1
    while page <= pages:
        params = {"observation": ",".join(str(i) for i in mo_ids), "detail": "high",
                  "format": "json", "page": str(page)}
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
        if body.get("errors"):
            raise RuntimeError(f"MO answered errors: {json.dumps(body['errors'])[:300]}")
        out.extend(body.get("results") or [])
        pages = int(body.get("number_of_pages") or 1)
        page += 1
    return out


def fetch_all(conn: sqlite3.Connection, refresh: bool = False, north_america_only: bool = True,
              limit: int | None = None, raw_dir: Path | None = None, session=None,
              pacer: Pacer | None = None, fetch=fetch_batch, log=print) -> dict:
    """MO's images for MO records not read before (all of them with `refresh`). Each
    answer is kept gzipped in `raw_dir` (default <data>/raw/mo)."""
    config.ensure_dirs()
    todo = pending(conn, refresh, north_america_only)
    ids = sorted(todo)
    if limit is not None:
        ids = ids[:limit]
    raw_dir = raw_dir or config.RAW_DIR / "mo"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = config.USER_AGENT
    pacer = pacer or Pacer()
    totals = {"requested": len(ids), "ok": 0, "missing": 0, "photos_new": 0,
              "license_changes": 0, "photos": 0, "batches": 0}
    started = time.monotonic()
    for batch in chunks(ids, BATCH):
        images = fetch(session, batch, pacer)
        fetched_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with gzip.open(raw_dir / f"images-{batch[0]}-{fetched_at[:10]}.json.gz", "wt",
                       encoding="utf-8") as f:
            json.dump(images, f)
        stats = save_batch(conn, {n: todo[n] for n in batch}, images, fetched_at)
        for k, v in stats.items():
            totals[k] += v
        totals["batches"] += 1
        if totals["batches"] % 10 == 0:
            log(f"  {totals['ok'] + totals['missing']:,}/{len(ids):,} MO observations, "
                f"{totals['photos']:,} photos, {time.monotonic() - started:.0f}s")
    return totals


def licence_report(conn: sqlite3.Connection) -> dict:
    """MO photos linked to MO records, by licence as MO names it and by class, with how
    many MO marks not ok_for_export."""
    rows = conn.execute(
        "select p.license_text, p.license_class, coalesce(p.ok_for_export, -1), count(*) "
        "from photos p join observation_photos op on op.photo_id = p.photo_id "
        "join records r on r.observation_id = op.observation_id and r.source = 'mo' "
        "where p.source = 'mo' group by 1, 2, 3").fetchall()
    by_text: Counter = Counter()
    by_class: Counter = Counter()
    not_export = 0
    for text, cls, ok, n in rows:
        by_text[text or "(none)"] += n
        by_class[cls] += n
        not_export += n if ok == 0 else 0
    return {"by_licence": dict(by_text.most_common()), "by_class": dict(by_class),
            "not_ok_for_export": not_export, "photos": sum(by_class.values())}
