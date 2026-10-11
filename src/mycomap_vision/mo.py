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
import re
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
# Anything else counts as all rights reserved until a person adds it here. Every name
# MO gave for the DNA-validated MO records' photos on 2026-10-09 is listed.
_LICENSES = {
    "creative commons wikipedia compatible v3.0": "cc-by-sa",
    "creative commons attribution-sharealike 3.0": "cc-by-sa",
    "creative commons attribution sharealike v4.0 (wikipedia compatible)": "cc-by-sa",
    "creative commons attribution v4.0 (wikipedia compatible)": "cc-by",
    "creative commons non-commercial v3.0": "cc-by-nc-sa",
    "creative commons non-commercial v2.5": "cc-by-nc-sa",
    "creative commons attribution-noncommercial-sharealike 3.0": "cc-by-nc-sa",
    "creative commons attribution non-commercial sharealike v4.0": "cc-by-nc-sa",
    "creative commons attribution non-commercial v4.0": "cc-by-nc",
    "creative commons attribution non-commercial noderivs v.4.0": "cc-by-nc-nd",
    "public domain": "pd",
    "public domain (wikipedia compatible)": "pd",
}


def license_of(text: str | None) -> tuple[str, str]:
    """(licence code, class) for an MO licence name. Unknown or reserved: ('', 'arr')."""
    code = _LICENSES.get(" ".join((text or "").split()).lower(), "")
    return code, license_class(code)


def relicense(conn: sqlite3.Connection, now: str | None = None) -> dict:
    """Re-read every MO photo's licence code and class from the name MO gave
    (license_text), after names are added to _LICENSES. A change goes to license_history."""
    now = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    changed = 0
    with conn:
        for pid, text, code in conn.execute(
                "select photo_id, license_text, license_code from photos where source = 'mo'"
                ).fetchall():
            new_code, new_class = license_of(text)
            if (code or "") != new_code:
                conn.execute("update photos set license_code = ?, license_class = ? "
                             "where photo_id = ?", (new_code, new_class, pid))
                conn.execute("insert into license_history values (?, ?, ?)", (pid, new_code, now))
                changed += 1
    return {"changed": changed}


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


class MissingObservation(LookupError):
    """MO says one of the requested observations doesn't exist (deleted, or merged away)."""

    def __init__(self, mo_id: int):
        super().__init__(f"MO has no observation {mo_id}")
        self.mo_id = mo_id


_NOT_FOUND = re.compile(r"Observation #(\d+) does not exist")


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
    totals["gone_on_mo"] = 0
    for batch in chunks(ids, BATCH):
        # MO refuses a whole request for one observation it no longer has: that one is
        # left out (it gets no photos, status 'missing') and the rest asked again.
        ask = list(batch)
        while True:
            try:
                images = fetch(session, ask, pacer) if ask else []
                break
            except MissingObservation as e:
                if e.mo_id not in ask:
                    raise
                ask.remove(e.mo_id)
                totals["gone_on_mo"] += 1
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


# ---------------------------------------------------------------------------
# Image files from a zip (Steve, 2026-10-09: "just give me a list and Alan can give me a
# zip"). MO's image files come from a person with access to MO's image store, not from
# MO's public site, so nothing is downloaded here: image_list() writes the wanted
# images, import_zip() takes the files back by image id.

IMAGE_LIST_COLUMNS = ["mo_image_id", "mo_observation_id", "license", "url_960"]
_ZIP_NAME = re.compile(r"(?:^|/)(\d+)\.(jpe?g|png|webp)$", re.IGNORECASE)


def image_list(conn: sqlite3.Connection, store_location: str | None = None,
               size: str = "large", north_america_only: bool = True) -> list[dict]:
    """The MO images Vision wants: every MO photo of an MO record that MO marks ok for
    export, one row per image (with one of its MO observations), leaving out images
    already held at `size` in `store_location`."""
    na = "and r.north_america = 1" if north_america_only else ""
    rows = conn.execute(f"""
      select p.source_photo_id, min(r.source_id), p.license_text
      from photos p
      join observation_photos op on op.photo_id = p.photo_id
      join records r on r.observation_id = op.observation_id and r.source = '{sources.MO}' {na}
      where p.source = '{sources.MO}' and coalesce(p.ok_for_export, 0) = 1
        and not exists (select 1 from photo_copies c where c.photo_id = p.photo_id
                        and c.size = ? and (? is null or c.store = ?))
      group by p.source_photo_id order by p.source_photo_id
    """, (size, store_location, store_location)).fetchall()
    return [{"mo_image_id": int(i), "mo_observation_id": int(o), "license": lic or "",
             "url_960": f"https://mushroomobserver.org/images/960/{int(i)}.jpg"}
            for i, o, lic in rows]


def _shrink(body: bytes, max_side: int) -> bytes:
    """A JPEG no longer than `max_side` px on its long side (EXIF rotation applied)."""
    import io

    from PIL import Image, ImageOps
    img = ImageOps.exif_transpose(Image.open(io.BytesIO(body))).convert("RGB")
    if max(img.size) <= max_side:
        return body
    img.thumbnail((max_side, max_side))
    out = io.BytesIO()
    img.save(out, "JPEG", quality=90)
    return out.getvalue()


def import_zip(conn: sqlite3.Connection, zip_path: Path, store, size: str = "large",
               max_side: int | None = 1280, log=print) -> dict:
    """Store the image files in a zip (named <MO image id>.jpg, any folders) as copies of
    the MO photos the manifest already lists, with their hash. A file for an image the
    manifest doesn't list as an MO photo of an MO record is refused, so nothing but the
    listed images comes in. `max_side`: larger files are shrunk to it (MO's originals
    run to 5,000+ px; the identifier works at a few hundred)."""
    import hashlib
    import zipfile

    from .licenses import looks_like_image, photo_relpath
    from .photos import Result, save_result
    wanted = {int(r["mo_image_id"]) for r in image_list(conn, None, "__any__", False)}
    stats = Counter()
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            m = _ZIP_NAME.search(info.filename)
            if not m:
                stats["not an image name"] += 1
                continue
            image_id = int(m.group(1))
            if image_id not in wanted:
                stats["not on the list (refused)"] += 1
                continue
            body = zf.read(info)
            if not looks_like_image(body[:16]):
                stats["not an image"] += 1
                continue
            if max_side:
                body = _shrink(body, max_side)
            pid = photo_id_of(image_id)
            rel = photo_relpath(pid, size, "jpg" if max_side else m.group(2).lower())
            store.put(rel, body)
            with conn:
                save_result(conn, Result(pid, "done", rel, len(body),
                                         hashlib.sha256(body).hexdigest()),
                            size, now, store.location)
            stats["imported"] += 1
            if stats["imported"] % 2000 == 0:
                log(f"  {stats['imported']:,} imported")
    stats["still wanted"] = len(image_list(conn, store.location, size))
    return dict(stats)
