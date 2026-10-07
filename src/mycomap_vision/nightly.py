"""The nightly update: keep the site's reference set in step with mycomap.org between releases.

Every night (MV_NIGHTLY_AT, default 03:00 in MV_NIGHTLY_TZ, America/New_York) the
server asks mycomap.org for every green record (GET /api/vision/green-records, the
key in permissions.org_key()), compares the answer with what it serves, and then:

    records no longer green      leave the reference set that night
    records renamed on .org      carry their new name that night
    records new to the list      iNat details -> photos -> one vector per photo,
                                 embedded on the box's CPU with the served model,
                                 then they join the reference set

What a night changes lives in a layer beside the release, never inside it
(<MV_RELEASE_ROOT>/state/nightly/<release id>/):

    manifest.sqlite             the release's manifest, copied once, then kept current;
                                the server reads and writes this copy, not the release's
    taxonomy/inat_genera.sqlite the release's taxonomy cache, plus genera new since
    embeddings/<backbone>/      the shards the nights added (numbered after the
                                release's; nightly_layer records where they start)
    photos/                     photos only until they are embedded, then deleted
    raw/                        iNat's answers, gzipped

The served index is the release plus this layer. A new release starts a fresh layer
(its manifest already holds everything) and the old one is deleted when the server
starts on it. The layer is meant to stay small: when the photos a backbone gained
pass MV_NIGHTLY_WARN_SHARE (20%) of the release's, `mv nightly` and the server log
say it is time for a new release.

Nothing changes when mycomap.org's answer is missing, malformed, empty or short of
the total it announced, and an answer that would remove more records than
MV_NIGHTLY_MAX_REMOVED (500, or 2% of the list when that is larger) is refused as
suspicious; `mv nightly --now --accept-removals` takes it. Every run is recorded in
nightly_runs.

The server runs the update itself (api.create_app) because it already holds the
model: a second process on the 4 GB box would hold a second copy. `mv nightly`
reports, `--plan` shows what tonight would change without changing it, and
`--now` asks the running server to update within a minute.

Read-only toward every outside system, like the rest of Vision: it reads
mycomap.org and iNat, within iNat's limits (inat.py, photos.py).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone, tzinfo
from pathlib import Path
from typing import Callable

import requests

from . import config, inat, photos, refresh, taxonomy
from .embed import SCHEMA as EMBED_SCHEMA
from .embed import embed_photos, next_shard, photos_to_embed
from .manifest import snapshot
from .records import build_records, save_records
from .release import RELEASE_ID, current_release
from .storage import LocalStore

PATH = "/api/vision/green-records"
PENDING_PATH = "/api/vision/pending-records"
PAGE_LIMIT = 5000
MAX_PAGES = 1000                 # 5 million records: far past any real answer
LOCK_STALE_SECONDS = 6 * 3600    # a lock this old was left by a run that died

SCHEMA = """
-- Where each backbone's nightly shards start, and how many photos the release had.
create table if not exists nightly_layer (
  backbone       text primary key,
  first_shard    integer not null,
  release_rows   integer not null,
  created_at     text not null
);

-- Every nightly run, refused and failed ones included.
create table if not exists nightly_runs (
  id             integer primary key autoincrement,
  started_at     text not null,
  finished_at    text,
  ok             integer,          -- 1 done, 0 refused or failed, null still running
  changed        integer not null default 0,  -- the served reference set changed
  generated_at   text,             -- mycomap.org's clock when it answered
  report         text,             -- JSON: what each step did
  error          text
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(SCHEMA)


class NightlyRefused(RuntimeError):
    """mycomap.org's answer was not taken; nothing was changed."""


class NightlyBusy(RuntimeError):
    """Another nightly run holds the layer."""


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Settings

def enabled() -> bool:
    return (config.setting("MV_NIGHTLY") or "").lower() in ("1", "true", "yes", "on")


def _number(name: str, default: float) -> float:
    raw = config.setting(name)
    try:
        return float(raw) if raw else default
    except ValueError:
        raise ValueError(f"{name} must be a number, not {raw!r}") from None


_AT = re.compile(r"^([01]?\d|2[0-3]):([0-5]\d)$")


@dataclass(frozen=True)
class Settings:
    at: str = "03:00"                    # local time of the nightly run
    tz: str = "America/New_York"
    max_photos: int = 3000               # photos embedded in one night (~0.7 s each on the box)
    size: str = "large"                  # the size the release's vectors were made from
    max_removed: int = 500               # removals refused above max(this, share of the list)
    max_removed_share: float = 0.02
    warn_share: float = 0.20             # layer / release photos that call for a new release
    predict: int = 300                   # advance predictions a night (0: none)

    def __post_init__(self):
        if not _AT.match(self.at):
            raise ValueError(f"MV_NIGHTLY_AT must be HH:MM, not {self.at!r}")

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(at=config.setting("MV_NIGHTLY_AT", "03:00"),
                   tz=config.setting("MV_NIGHTLY_TZ", "America/New_York"),
                   max_photos=int(_number("MV_NIGHTLY_MAX_PHOTOS", 3000)),
                   size=config.setting("MV_NIGHTLY_PHOTO_SIZE", "large"),
                   max_removed=int(_number("MV_NIGHTLY_MAX_REMOVED", 500)),
                   max_removed_share=_number("MV_NIGHTLY_MAX_REMOVED_SHARE", 0.02),
                   warn_share=_number("MV_NIGHTLY_WARN_SHARE", 0.20),
                   predict=int(_number("MV_NIGHTLY_PREDICT", 300)))

    def zone(self) -> tzinfo:
        from zoneinfo import ZoneInfo
        return ZoneInfo(self.tz)

    def removal_limit(self, records_before: int) -> int:
        return max(self.max_removed, int(records_before * self.max_removed_share))


# ---------------------------------------------------------------------------
# The layer

@dataclass(frozen=True)
class Layer:
    root: Path                 # <release root>/state/nightly/<release id>
    release_dir: Path          # <release root>/releases/<release id>

    @property
    def release_id(self) -> str:
        return self.root.name

    @property
    def manifest(self) -> Path:
        return self.root / "manifest.sqlite"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def photos(self) -> Path:
        return self.root / "photos"

    @property
    def raw(self) -> Path:
        return self.root / "raw"

    @property
    def taxonomy_cache(self) -> Path:
        return self.root / taxonomy.CACHE

    @property
    def run_now(self) -> Path:
        return self.root / "run-now"

    @property
    def accept_removals(self) -> Path:
        return self.root / "accept-removals"

    @property
    def lock(self) -> Path:
        return self.root / "nightly.lock"


def layer_for(release_root: Path, release_id: str | None = None) -> Layer:
    rid = release_id or current_release(release_root)
    if not rid:
        raise ValueError(f"no current release in {release_root}: run `mv pull-release` first")
    return Layer(release_root / "state" / "nightly" / rid, release_root / "releases" / rid)


def prepare(release_root: Path, release_id: str | None = None, prune: bool = False,
            log=print) -> Layer:
    """The layer of the current release, made the first time: a copy of the release's
    manifest and taxonomy cache, and the shard each backbone's nightly vectors start
    at. With `prune`, the layers of other releases are deleted (only the server does
    this, at startup: a CLI run may happen while a server still serves the old one)."""
    layer = layer_for(release_root, release_id)
    if not layer.manifest.is_file():
        source = layer.release_dir / "manifest.sqlite"
        if not source.is_file():
            raise ValueError(f"release {layer.release_id} has no manifest.sqlite")
        layer.root.mkdir(parents=True, exist_ok=True)
        tmp = layer.root / "manifest.copying.sqlite"
        src = sqlite3.connect(source)
        try:
            snapshot(src, tmp)
        finally:
            src.close()
        cache = layer.release_dir / taxonomy.CACHE
        if cache.is_file():
            layer.taxonomy_cache.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache, layer.taxonomy_cache)
        conn = sqlite3.connect(tmp)
        try:
            start_layer(conn)
        finally:
            conn.close()
        os.replace(tmp, layer.manifest)
        log(f"[nightly] new layer for release {layer.release_id} (manifest copied)")
    if prune:
        for old in (layer.root.parent.iterdir() if layer.root.parent.is_dir() else []):
            if old.is_dir() and old.name != layer.release_id and RELEASE_ID.match(old.name):
                shutil.rmtree(old, ignore_errors=True)
                log(f"[nightly] removed the layer of release {old.name}")
    return layer


def start_layer(conn: sqlite3.Connection) -> None:
    """Note, for every backbone the release embedded, where nightly shards begin."""
    ensure_schema(conn)
    at = iso(now_utc())
    with conn:
        for (backbone, n) in conn.execute(
                "select backbone, count(*) from embeddings group by backbone").fetchall():
            conn.execute("insert or ignore into nightly_layer values (?, ?, ?, ?)",
                         (backbone, next_shard(conn, backbone), n, at))


def served_manifest() -> Path:
    """The manifest commands should open: on a server box with the nightly update on,
    the current release's layer copy once the server has made it; else the usual one."""
    if enabled() and config.RELEASE_ROOT:
        try:
            layer = layer_for(config.RELEASE_ROOT)
        except ValueError:
            return config.MANIFEST_PATH
        if layer.manifest.is_file():
            return layer.manifest
    return config.MANIFEST_PATH


def layer_version(conn: sqlite3.Connection) -> int:
    """Changes whenever a night changed the reference set (so the index is rebuilt even
    when only names changed, which leaves the number of vectors alone)."""
    has = conn.execute("select 1 from sqlite_master where type = 'table' and "
                       "name = 'nightly_runs'").fetchone()
    if not has:
        return 0
    return int(conn.execute("select coalesce(max(id), 0) from nightly_runs "
                            "where changed = 1").fetchone()[0])


def layer_sizes(conn: sqlite3.Connection, warn_share: float) -> dict:
    """Per backbone: the release's photos, the photos the nights added, their share."""
    ensure_schema(conn)
    out = {}
    for backbone, first, release_rows in conn.execute(
            "select backbone, first_shard, release_rows from nightly_layer order by backbone"
    ).fetchall():
        added = conn.execute("select count(*) from embeddings where backbone = ? and shard >= ?",
                             (backbone, first)).fetchone()[0]
        share = added / release_rows if release_rows else 0.0
        out[backbone] = {"release_photos": release_rows, "nightly_photos": added,
                         "share": round(share, 4), "new_release_due": share >= warn_share}
    return out


# ---------------------------------------------------------------------------
# Reading mycomap.org

def _fail(reason: str) -> NightlyRefused:
    return NightlyRefused(f"{reason}; nothing was changed")


def fetch_green(base_url: str, key: str, session: requests.Session | None = None,
                limit: int = PAGE_LIMIT, timeout: float = 120, log=print) -> tuple[list[dict], str]:
    """Every green record from mycomap.org, page by page: (rows, mycomap.org's clock).

    Refuses (NightlyRefused, with a reason that never includes the key) a failed or
    malformed page, a cursor that repeats, an empty list, and a list short of (or past)
    the total the first page announced by more than 1% (records turning green while
    the pages are read move it a little)."""
    return fetch_pages(base_url, key, PATH, "green", session=session, limit=limit,
                       timeout=timeout, log=log)


def fetch_pending(base_url: str, key: str, session: requests.Session | None = None,
                  limit: int = PAGE_LIMIT, timeout: float = 120,
                  log=print) -> tuple[list[dict], str]:
    """Every record awaiting validation (GET /api/vision/pending-records), for the
    advance predictions; checked as fetch_green, but an empty list is an answer."""
    return fetch_pages(base_url, key, PENDING_PATH, "pending", session=session, limit=limit,
                       timeout=timeout, allow_empty=True, log=log)


def fetch_pages(base_url: str, key: str, path: str, what: str,
                session: requests.Session | None = None, limit: int = PAGE_LIMIT,
                timeout: float = 120, allow_empty: bool = False,
                log=print) -> tuple[list[dict], str]:
    session = session or requests.Session()
    url = base_url.rstrip("/") + path
    headers = {"Authorization": f"Bearer {key}", "User-Agent": config.USER_AGENT}
    rows: list[dict] = []
    after, total, generated = "", None, None
    cursors: set[str] = set()
    for page in range(MAX_PAGES):
        params = {"limit": str(limit)}
        if after:
            params["after"] = after
        try:
            resp = session.get(url, params=params, headers=headers, timeout=timeout)
        except requests.RequestException as e:
            raise _fail(f"could not reach {base_url}: {type(e).__name__}") from None
        if resp.status_code == 401:
            raise _fail("mycomap.org refused the key (401): check MV_ORG_VISION_KEY")
        if resp.status_code == 503:
            raise _fail("mycomap.org has no VISION_API_KEY set (503)")
        if resp.status_code != 200:
            raise _fail(f"mycomap.org answered {resp.status_code} on page {page + 1}")
        try:
            payload = resp.json()
        except ValueError:
            raise _fail(f"page {page + 1} was not JSON") from None
        records = payload.get("records") if isinstance(payload, dict) else None
        nxt = payload.get("next") if isinstance(payload, dict) else None
        if not isinstance(records, list) or payload.get("count") != len(records) \
                or not (nxt is None or isinstance(nxt, str)):
            raise _fail(f"page {page + 1} is not a {what}-records page")
        if any(not isinstance(r, dict) or r.get("observation_id") in (None, "") for r in records):
            raise _fail(f"page {page + 1} holds a record without an observation_id")
        if page == 0:
            total, generated = payload.get("total"), payload.get("generatedAt")
            if not isinstance(total, int) or total < 0 or not isinstance(generated, str):
                raise _fail("the first page has no total or generatedAt")
        rows.extend(records)
        if nxt is None:
            break
        if nxt in cursors or not records:
            raise _fail(f"page {page + 1} repeats a cursor")
        cursors.add(nxt)
        after = nxt
    else:
        raise _fail(f"more than {MAX_PAGES} pages")
    if not rows and not allow_empty:
        raise _fail(f"mycomap.org listed no {what} records")
    if abs(len(rows) - total) > max(50, total // 100):
        raise _fail(f"mycomap.org announced {total:,} {what} rows but sent {len(rows):,}")
    log(f"[nightly] {len(rows):,} {what} rows from mycomap.org in {page + 1} pages")
    return rows, generated


# ---------------------------------------------------------------------------
# What a night changes

def changes(conn: sqlite3.Connection, recs: list[dict]) -> dict:
    """What saving `recs` would do: new, removed and renamed records (with examples)."""
    before = dict(conn.execute("select observation_id, scientific_name from records"))
    now = {r["observation_id"]: r["scientific_name"] for r in recs}
    new = sorted(set(now) - set(before))
    removed = sorted(set(before) - set(now))
    renamed = sorted(o for o in set(now) & set(before) if now[o] != before[o])
    return {"before": len(before), "after": len(now), "new": len(new), "removed": len(removed),
            "renamed": len(renamed), "examples": {"new": new[:10], "removed": removed[:10],
                                                   "renamed": renamed[:10]}}


class Locked:
    """A backbone whose encode() waits for `lock`, so the nightly embedding takes turns
    with identifications on the one CPU model instead of running beside them."""

    def __init__(self, inner, lock: threading.Lock):
        self._inner, self._lock = inner, lock
        self.name, self.dim = inner.name, inner.dim

    def encode(self, images):
        with self._lock:
            return self._inner.encode(images)

    def __getattr__(self, name):          # prepare, device: whatever the model has
        return getattr(self._inner, name)


@dataclass
class Steps:
    """The outside calls, injectable for tests."""
    fetch_inat: Callable[..., dict] = inat.fetch_all
    taxonomy: Callable[..., dict] = refresh.new_genera_taxonomy
    download: Callable[..., dict] = photos.download_all


@dataclass
class RunLock:
    path: Path
    stale_seconds: float = LOCK_STALE_SECONDS

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            except FileExistsError:
                try:
                    age = time.time() - self.path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age < self.stale_seconds:
                    raise NightlyBusy(f"another nightly run holds {self.path} "
                                      f"(started {age / 60:.0f} min ago)") from None
                self.path.unlink(missing_ok=True)           # left by a run that died
                continue
            with os.fdopen(fd, "w") as f:
                f.write(f"{os.getpid()} {iso(now_utc())}\n")
            return self
        raise NightlyBusy(f"could not take {self.path}")

    def __exit__(self, *exc):
        self.path.unlink(missing_ok=True)


def run_once(conn: sqlite3.Connection, layer: Layer,
             fetch: Callable[[], tuple[list[dict], str]],
             backbones: dict[str, Callable[[], object]], settings: Settings | None = None,
             encode_lock: threading.Lock | None = None, accept_removals: bool = False,
             steps: Steps | None = None, log=print) -> dict:
    """One night. `conn` is the layer's manifest (opened with manifest.connect);
    `fetch` reads mycomap.org (fetch_green); `backbones` maps each served backbone to a
    way to get its loaded model. Returns the report recorded in nightly_runs; never
    raises for a refused answer or a failed step (the report says what happened)."""
    settings = settings or Settings()
    steps = steps or Steps()
    ensure_schema(conn)
    started = now_utc()
    with conn:
        run_id = conn.execute("insert into nightly_runs (started_at) values (?)",
                              (iso(started),)).lastrowid
    report: dict = {"release": layer.release_id}
    changed, ok, error, generated = False, True, None, None
    try:
        with RunLock(layer.lock):
            rows, generated = fetch()
            exported_at = iso(now_utc())
            recs = build_records(rows, exported_at)
            plan = changes(conn, recs)
            report["records"] = {k: plan[k] for k in ("before", "after", "new", "removed",
                                                      "renamed")}
            limit = settings.removal_limit(plan["before"])
            if plan["removed"] > limit and not accept_removals:
                raise NightlyRefused(
                    f"REFUSED: the answer removes {plan['removed']:,} of {plan['before']:,} "
                    f"records (more than {limit:,}); nothing was changed. If they really left "
                    "the green list, run `mv nightly --now --accept-removals`.")
            saved = save_records(conn, recs)
            changed = changed or any(saved[k] for k in ("new", "removed", "renamed"))
            log(f"[nightly] records: {saved['new']:,} new, {saved['removed']:,} removed, "
                f"{saved['renamed']:,} renamed")
            report["inat"] = _step(lambda: steps.fetch_inat(conn, raw_dir=layer.raw / "inat",
                                                            log=log), "iNat details", log)
            report["taxonomy"] = _step(lambda: steps.taxonomy(
                conn, max_minutes=5.0, cache=layer.taxonomy_cache, log=log), "taxonomy", log)
            report["photos"], report["embedded"] = {}, {}
            store = LocalStore(layer.photos)
            served = [b for b in _layer_backbones(conn) if b in backbones]
            # Only records that joined after the layer began: a release that embedded a
            # sample of the records must not have the nights fetch all the others.
            since = layer_started(conn)
            for name in served:
                report["photos"][name] = _step(lambda: steps.download(
                    conn, store, size=settings.size, limit=settings.max_photos,
                    first_seen_since=since, unembedded_for=name, log=log),
                    f"photos for {name}", log)
                todo = photos_to_embed(conn, name, settings.size, store.location,
                                       str(layer.photos))[:settings.max_photos]
                if not todo:
                    report["embedded"][name] = 0
                    continue
                model = backbones[name]()
                if encode_lock is not None:
                    model = Locked(model, encode_lock)
                stats = embed_photos(conn, store, model, todo, layer.embeddings / name,
                                     batch_size=8, readers=2, log=log)
                report["embedded"][name] = stats.embedded
                if stats.skipped:
                    report.setdefault("unreadable", {})[name] = len(stats.skipped)
                changed = changed or stats.embedded > 0
            report["photos_deleted"] = drop_embedded_photos(conn, store, served)
            report["layer"] = layer_sizes(conn, settings.warn_share)
            for name, size in report["layer"].items():
                if size["new_release_due"]:
                    log(f"[nightly] WARNING: {name} gained {size['nightly_photos']:,} photos, "
                        f"{size['share']:.0%} of the release's: time for a new release")
    except NightlyRefused as e:
        ok, error = False, str(e)
        log(f"[nightly] {e}")
    except Exception as e:  # noqa: BLE001 - recorded; the server carries on serving
        ok, error = False, f"{type(e).__name__}: {e}"
        log(f"[nightly] failed: {error}")
    report["seconds"] = round((now_utc() - started).total_seconds(), 1)
    with conn:
        conn.execute("update nightly_runs set finished_at = ?, ok = ?, changed = ?, "
                     "generated_at = ?, report = ?, error = ? where id = ?",
                     (iso(now_utc()), int(ok), int(changed), generated, json.dumps(report),
                      error[:1000] if error else None, run_id))
    return {"id": run_id, "ok": ok, "changed": changed, "error": error, **report}


def _step(work: Callable[[], dict], what: str, log) -> dict:
    """A step whose failure (iNat down, a network error) the next night makes good:
    logged and reported, and the night goes on with what it has."""
    try:
        return work()
    except Exception as e:  # noqa: BLE001
        err = f"{type(e).__name__}: {e}"
        log(f"[nightly] {what} failed, carrying on: {err}")
        return {"error": err[:500]}


def _layer_backbones(conn: sqlite3.Connection) -> list[str]:
    return [r[0] for r in conn.execute("select backbone from nightly_layer order by backbone")]


def layer_started(conn: sqlite3.Connection) -> str | None:
    return conn.execute("select min(created_at) from nightly_layer").fetchone()[0]


def drop_embedded_photos(conn: sqlite3.Connection, store: LocalStore, backbones: list[str]) -> int:
    """Delete the layer's photo files once every served backbone has their vector; their
    copy rows go too, so a photo no backbone still needs is never fetched again."""
    if not backbones:
        return 0
    held = conn.execute("select photo_id, path from photo_copies where store = ?",
                        (store.location,)).fetchall()
    done = []
    for pid, rel in held:
        n = conn.execute(f"select count(*) from embeddings where photo_id = ? and backbone in "
                         f"({','.join('?' * len(backbones))})", (pid, *backbones)).fetchone()[0]
        if n == len(backbones):
            (store.root / rel).unlink(missing_ok=True)
            done.append(pid)
    with conn:
        conn.executemany("delete from photo_copies where store = ? and photo_id = ?",
                         [(store.location, pid) for pid in done])
    return len(done)


# ---------------------------------------------------------------------------
# Advance predictions

def advance_predictions(conn: sqlite3.Connection, run_id: int,
                        fetch: Callable[[], tuple[list[dict], str]],
                        served: Callable[[], tuple[object, object]], fetcher, limit: int,
                        log=print) -> dict:
    """Predict up to `limit` of the records awaiting validation (newest first) with the
    served index and model (`served()` gives both), from their iNat photos
    (prospective.py: held in memory, never stored), and add what happened to nightly
    run `run_id`. Each prediction is scored by
    `mv prospective` once its record turns green; one made after that never counts.
    Only records mycomap.org still lists as pending tonight are predicted."""
    from . import prospective
    try:
        identifier, backbone_model = served()
        rows, _generated = fetch()
        seen = iso(now_utc())
        prospective.save_candidates(conn, rows, seen)
        ids = prospective.unpredicted(conn, identifier.backbone, identifier.method, limit,
                                      seen_since=seen)
        stats = prospective.predict_pending(conn, identifier, backbone_model, ids, fetcher,
                                            log=log)
        out = {"model": f"{identifier.backbone}/{identifier.method}", "pending": len(rows),
               "asked": len(ids), **stats}
        log(f"[nightly] advance predictions: {stats.get('predicted', 0):,} of "
            f"{len(ids):,} asked ({len(rows):,} pending)")
    except Exception as e:  # noqa: BLE001 - the reference set is already updated
        out = {"error": f"{type(e).__name__}: {e}"[:500]}
        log(f"[nightly] advance predictions failed: {out['error']}")
    with conn:
        row = conn.execute("select report from nightly_runs where id = ?", (run_id,)).fetchone()
        report = json.loads(row[0]) if row and row[0] else {}
        report["predictions"] = out
        conn.execute("update nightly_runs set report = ? where id = ?",
                     (json.dumps(report), run_id))
    return out


# ---------------------------------------------------------------------------
# When

def next_run(now: datetime, at: str, zone: tzinfo) -> datetime:
    """The next `at` (HH:MM, local to `zone`) strictly after `now`, in UTC."""
    hour, minute = (int(x) for x in at.split(":"))
    local = now.astimezone(zone)
    day: date = local.date()
    for _ in range(3):
        cand = datetime(day.year, day.month, day.day, hour, minute, tzinfo=zone)
        if cand > local:
            return cand.astimezone(timezone.utc)
        day += timedelta(days=1)
    raise AssertionError("unreachable")


@dataclass
class Schedule:
    """Decides, each tick, whether the nightly run is due: at the set time, or when
    `mv nightly --now` left the run-now flag. Testable without threads."""
    layer: Layer
    settings: Settings
    zone: tzinfo
    next_at: datetime | None = None
    last: dict | None = field(default=None)

    def due(self, now: datetime) -> tuple[bool, bool]:
        """(run now?, accept removals?) The flags are consumed when they cause a run."""
        if self.next_at is None:
            self.next_at = next_run(now, self.settings.at, self.zone)
        asked = self.layer.run_now.exists()
        if not asked and now < self.next_at:
            return False, False
        accept = self.layer.accept_removals.exists()
        self.layer.run_now.unlink(missing_ok=True)
        self.layer.accept_removals.unlink(missing_ok=True)
        self.next_at = next_run(now, self.settings.at, self.zone)
        return True, accept


def start(schedule: Schedule, job: Callable[[bool], dict], note=print, tick: float = 60.0,
          clock: Callable[[], datetime] = now_utc) -> threading.Thread:
    """Run `job(accept_removals)` whenever the schedule says so, on a daemon thread."""
    def loop():
        schedule.due(clock())
        note(f"[nightly] next run {iso(schedule.next_at)} "
             f"({schedule.settings.at} {schedule.settings.tz})")
        while True:
            try:
                run, accept = schedule.due(clock())
                if run:
                    schedule.last = job(accept)
                    note(f"[nightly] next run {iso(schedule.next_at)}")
            except Exception as e:  # noqa: BLE001 - never stops the server
                note(f"[nightly] {type(e).__name__}: {e}")
            threading.Event().wait(tick)
    t = threading.Thread(target=loop, name="nightly", daemon=True)
    t.start()
    return t


# ---------------------------------------------------------------------------
# Reporting

def health(conn: sqlite3.Connection, settings: Settings, zone: tzinfo | None = None,
           now: datetime | None = None) -> dict:
    """The nightly part of /api/health (public, so no error text, which can name hosts or
    paths: `mv nightly` on the box has that). nightly_notes.py turns it into lines."""
    ensure_schema(conn)
    last = conn.execute("select started_at, ok, report, error from nightly_runs "
                        "order by id desc limit 1").fetchone()
    ok_at = conn.execute("select max(finished_at) from nightly_runs where ok = 1").fetchone()[0]
    layer = {b: {k: s[k] for k in ("release_photos", "nightly_photos", "share", "new_release_due")}
             for b, s in layer_sizes(conn, settings.warn_share).items()}
    out = {"enabled": True, "last_ok_at": ok_at, "layer": layer,
           "new_release_due": any(s["new_release_due"] for s in layer.values()),
           "next_run": iso(next_run(now or now_utc(), settings.at, zone)) if zone else None}
    if last is None:
        return {**out, "state": "never", "last_run_at": None}
    started, ok, report, error = last
    report = json.loads(report) if report else {}
    records = report.get("records") or {}
    out.update(state="running" if ok is None else ("ok" if ok else "failed"), last_run_at=started,
               # Every refusal of mycomap.org's answer says so (fetch_green, run_once).
               refused=bool(error) and "nothing was changed" in error,
               changed={"new": records.get("new", 0), "removed": records.get("removed", 0),
                        "renamed": records.get("renamed", 0),
                        "embedded": sum((report.get("embedded") or {}).values()),
                        "predicted": (report.get("predictions") or {}).get("predicted", 0)})
    return out


def status(conn: sqlite3.Connection, settings: Settings, now: datetime | None = None,
           zone: tzinfo | None = None, runs: int = 5) -> dict:
    """The last runs, the layer's size per backbone and when the next run is."""
    ensure_schema(conn)
    rows = conn.execute("select id, started_at, finished_at, ok, changed, generated_at, report, "
                        "error from nightly_runs order by id desc limit ?", (runs,)).fetchall()
    last = [{"id": r[0], "started_at": r[1], "finished_at": r[2],
             "state": "running" if r[3] is None else ("ok" if r[3] else "failed"),
             "changed": bool(r[4]), "generated_at": r[5],
             "report": json.loads(r[6]) if r[6] else None, "error": r[7]} for r in rows]
    ok_at = conn.execute("select max(finished_at) from nightly_runs where ok = 1").fetchone()[0]
    out = {"at": settings.at, "tz": settings.tz, "last_ok_at": ok_at, "runs": last,
           "layer": layer_sizes(conn, settings.warn_share)}
    if zone is not None:
        out["next_run"] = iso(next_run(now or now_utc(), settings.at, zone))
    return out
