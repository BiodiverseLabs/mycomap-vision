"""Download photos for the green records, within iNat's and MO's limits, with a hash for each.

Hosts are limited separately:
- the AWS Open Data bucket (CC-licensed photos) is built for bulk download;
- iNat's static host (all-rights-reserved photos) gets iNat's media rule:
  under 5 GB an hour and 24 GB a day. We stay at 4 GB and 20 GB.
- Mushroom Observer (mushroomobserver.org) gets MO's rule for anonymous traffic: one
  request at a time, at least 5 seconds apart (20 a minute, README_API).

Which URL a photo is fetched from depends on photos.source (licenses.photo_url), never
on the look of its id or URL.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

import requests

from . import config, holdouts, sources
from .licenses import (MO_HOST, OPEN_DATA_HOST, STATIC_HOST, host_of, looks_like_image,
                       photo_extension, photo_relpath, photo_url, taken_down)
from .ratelimit import ByteBudget, MinInterval
from .storage import PhotoStore

GB = 1024 ** 3
MAX_ATTEMPTS = 4


@dataclass
class HostPolicy:
    concurrency: int
    min_interval: float
    budgets: list[ByteBudget] = field(default_factory=list)


STATIC_HOUR_GB = 4
STATIC_DAY_GB = 20


def default_policies(static_day_gb: float = STATIC_DAY_GB) -> dict[str, HostPolicy]:
    """`static_day_gb` raises the static host's day cap for one run, when a person has
    decided to (see CLAUDE.md); the hourly cap, which paces the load on iNat's
    server, is never raised."""
    if static_day_gb < STATIC_HOUR_GB:
        raise ValueError(f"a day cap under the hourly cap ({STATIC_HOUR_GB} GB) makes no sense")
    return {
        OPEN_DATA_HOST: HostPolicy(concurrency=8, min_interval=0.02),
        STATIC_HOST: HostPolicy(concurrency=2, min_interval=0.25, budgets=[
            ByteBudget(STATIC_HOUR_GB * GB, 3600), ByteBudget(int(static_day_gb * GB), 86400)]),
        MO_HOST: HostPolicy(concurrency=1, min_interval=MO_MIN_INTERVAL),
    }


# MO's README_API: anonymous traffic at most 20 requests a minute, 5 s apart on average.
MO_MIN_INTERVAL = 5.0
# Where an MO image URL may redirect to (once): MO's own hosts, over HTTPS.
MO_REDIRECT_HOSTS = (MO_HOST,)
MO_REDIRECT_SUFFIX = ".mushroomobserver.org"


def mo_redirect_ok(location: str) -> bool:
    """MO serves images itself or sends one redirect to its image host; anything else
    (another site, plain http, a private address) is refused."""
    from urllib.parse import urlparse
    u = urlparse(location or "")
    host = (u.hostname or "").lower()
    return u.scheme == "https" and (host in MO_REDIRECT_HOSTS or host.endswith(MO_REDIRECT_SUFFIX))


# Any other host gets the cautious static policy.
FALLBACK = HostPolicy(concurrency=1, min_interval=1.0, budgets=[ByteBudget(4 * GB, 3600)])


class HostGate:
    """Concurrency, pacing and byte budgets for one host."""

    def __init__(self, policy: HostPolicy):
        self.policy = policy
        self.sem = threading.Semaphore(policy.concurrency)
        self.pacer = MinInterval(policy.min_interval)

    def budget_wait(self) -> float:
        return max((b.wait_time() for b in self.policy.budgets), default=0.0)

    def consume(self, n: int) -> None:
        for b in self.policy.budgets:
            b.add(n)


def seed_budgets(conn: sqlite3.Connection, policies: dict[str, HostPolicy],
                 now: datetime | None = None) -> dict[str, int]:
    """Count each host's downloads still inside its budget windows (from any earlier run,
    any store, and a benchmark's own photos, heldout.py): a restarted downloader must not
    get a fresh day's budget, and two downloaders share one. Returns the bytes counted
    per host."""
    now = now or datetime.now(timezone.utc)
    sources = ["select c.downloaded_at, c.bytes from photo_copies c join photos p on "
               "p.photo_id = c.photo_id where p.host = ? and c.downloaded_at is not null "
               "and c.bytes is not null"]
    if conn.execute("select 1 from sqlite_master where type = 'table' "
                    "and name = 'heldout_photos'").fetchone():
        sources.append("select downloaded_at, bytes from heldout_photos where host = ? "
                       "and downloaded_at is not null and bytes is not null")
    counted = {}
    for host, policy in policies.items():
        if not policy.budgets:
            continue
        longest = max(b.window for b in policy.budgets)
        total = 0
        for when, n in conn.execute(" union all ".join(sources), (host,) * len(sources)):
            try:
                t = datetime.fromisoformat(when)
            except ValueError:
                continue
            if t.tzinfo is None:
                t = t.replace(tzinfo=timezone.utc)
            ago = (now - t).total_seconds()
            if ago < longest:
                for b in policy.budgets:
                    b.add_past(ago, n)
                total += n
        counted[host] = total
    return counted


@dataclass
class Result:
    photo_id: int
    status: str            # done | missing | error
    local_path: str | None = None
    bytes: int | None = None
    sha256: str | None = None
    error: str | None = None


def download_one(session: requests.Session, gate: HostGate, photo_id: int, source_url: str,
                 size: str, store: PhotoStore, stop: threading.Event,
                 source: str = "inat") -> Result:
    """Fetch one photo at `size`. `source` is photos.source: it picks the URL rule, and
    an MO image may follow at most one redirect, to an MO host over HTTPS."""
    if source == "inat" and taken_down(source_url):
        return Result(photo_id, "missing", error="taken down on iNat (copyright)")
    try:
        url = photo_url(source, source_url, size)
    except ValueError as e:
        return Result(photo_id, "error", error=str(e))
    rel = photo_relpath(photo_id, size, photo_extension(source, source_url))
    with gate.sem:
        while (w := gate.budget_wait()) > 0:
            if stop.wait(min(w, 60)):
                return Result(photo_id, "error", error="stopped")
        gate.pacer.wait()
        try:
            if source == "mo":
                resp = session.get(url, timeout=60, allow_redirects=False)
                if resp.is_redirect:
                    location = requests.compat.urljoin(url, resp.headers.get("location", ""))
                    if not mo_redirect_ok(location):
                        return Result(photo_id, "error",
                                      error=f"redirect refused: {host_of(location) or 'no host'}")
                    gate.pacer.wait()
                    resp = session.get(location, timeout=60, allow_redirects=False)
                    if resp.is_redirect:
                        return Result(photo_id, "error", error="redirect chain refused")
            else:
                resp = session.get(url, timeout=60)
        except requests.RequestException as e:
            return Result(photo_id, "error", error=f"request: {e.__class__.__name__}")
        body = resp.content
        gate.consume(len(body))
    if resp.status_code in (403, 404, 410):
        return Result(photo_id, "missing", error=f"http {resp.status_code}")
    if resp.status_code == 429:
        gate.pacer.pause(120)
        return Result(photo_id, "error", error="http 429")
    if resp.status_code != 200:
        return Result(photo_id, "error", error=f"http {resp.status_code}")
    if not looks_like_image(body[:16]):
        return Result(photo_id, "error", error="not an image")
    try:
        store.put(rel, body)
    except Exception as e:  # a failed store write is retried like a failed download
        return Result(photo_id, "error", error=f"store: {e.__class__.__name__}")
    return Result(photo_id, "done", local_path=rel, bytes=len(body),
                  sha256=hashlib.sha256(body).hexdigest())


def pending_photos(conn: sqlite3.Connection, north_america_only: bool, limit: int | None,
                   size: str = "medium", store_location: str | None = None,
                   random_order: bool = False, record_sample: int | None = None,
                   first_seen_since: str | None = None, held_at: str | None = None,
                   hosts: tuple[str, ...] | None = None, unembedded_for: str | None = None):
    """Photos of green records with no copy at `size` (in `store_location`, when given),
    failed fewest times first. Photos iNat no longer has, and ones that failed
    MAX_ATTEMPTS times, are left out. A copy at another size or in another store
    doesn't count, so a laptop sample and the S3 set are kept independently.

    `held_at` keeps only photos the store already holds at that other size (a
    sample fetched again at a new size); `hosts` keeps only photos served from
    those hosts (e.g. the open-data bucket, when another machine is using the
    capped host's budget).

    `unembedded_for` keeps only photos with no vector yet from that backbone (the
    nightly update: the release's photos are embedded but not held on the box).

    A benchmark's held-out records (holdouts.py) get no reference photos; the
    benchmark keeps its own (heldout.py).

    Only records of a source Vision takes photos from (sources.PHOTO_SOURCES) get
    photos, and only photos of that same source: an iNat photo linked to an MO record
    is never downloaded. MO photos are fetched from MO one at a time, 5 s apart (MO's
    pace). The first set came as a dump from MO's image store (mo.image_list /
    mo.import_zip; Steve 2026-10-09: "we will want to download MO images in the future
    but just upload dump this initial set"); like every photo, one held in the store is
    never fetched again, so runs fetch only new MO records' photos (a new store is
    filled from one we hold with copy_photos, never from the source). Refused
    on a manifest whose sources are still guessed.
    """
    sources.require_migrated(conn, "downloading photos")
    holdouts.ensure_schema(conn)
    na = "and r.north_america = 1" if north_america_only else ""
    params: list = []
    if first_seen_since:
        # Only records that joined the list at or after this export (e.g. a refresh's new ones).
        na += " and r.first_seen_at >= ?"
        params.append(first_seen_since)
    if record_sample:
        # All photos of a random set of records: a small but complete reference set.
        na += (" and r.observation_id in (select observation_id from records"
               f" where north_america = 1 order by random() limit {int(record_sample)})")
    extra, tail = "", []
    if held_at:
        extra += (" and exists (select 1 from photo_copies h where h.photo_id = p.photo_id"
                  " and h.size = ? and (? is null or h.store = ?))")
        tail += [held_at, store_location, store_location]
    if hosts:
        extra += f" and p.host in ({','.join('?' * len(hosts))})"
        tail += list(hosts)
    if unembedded_for:
        extra += (" and not exists (select 1 from embeddings e where e.backbone = ?"
                  " and e.photo_id = p.photo_id)")
        tail.append(unembedded_for)
    sql = f"""
      select distinct p.photo_id, p.source_url, p.host, p.attempts, p.source
      from photos p
      join observation_photos op on op.photo_id = p.photo_id
      join records r on r.observation_id = op.observation_id {na}
        and r.source in {sources.PHOTO_SOURCES_SQL} and p.source = r.source
      where p.status != 'missing' and p.attempts < {MAX_ATTEMPTS}
        and {holdouts.not_held_out('r.observation_id')}
        and not exists (select 1 from photo_copies c where c.photo_id = p.photo_id
                        and c.size = ? and (? is null or c.store = ?)){extra}
      order by {"random()" if random_order else "p.attempts, p.photo_id"}
    """
    if limit is not None:
        sql += f" limit {int(limit)}"
    return conn.execute(sql, (*params, size, store_location, store_location, *tail)).fetchall()


def save_result(conn: sqlite3.Connection, r: Result, size: str, now: str,
                store_location: str | None = None) -> None:
    """Record one download. A success adds a copy (the photos row keeps the latest one
    for display); only failures count toward MAX_ATTEMPTS."""
    if r.status == "done":
        conn.execute(
            "insert or replace into photo_copies (photo_id, store, size, path, bytes, sha256, "
            "downloaded_at) values (?, ?, ?, ?, ?, ?, ?)",
            (r.photo_id, store_location or "", size, r.local_path, r.bytes, r.sha256, now))
        conn.execute(
            "update photos set status = 'done', local_path = ?, store = ?, size = ?, bytes = ?, "
            "sha256 = ?, error = null, downloaded_at = ? where photo_id = ?",
            (r.local_path, store_location, size, r.bytes, r.sha256, now, r.photo_id))
    elif r.error == "stopped":
        return
    else:
        conn.execute(
            "update photos set status = ?, error = ?, attempts = attempts + 1 where photo_id = ?",
            (r.status, r.error, r.photo_id))


def copy_photos(conn: sqlite3.Connection, source: PhotoStore, dest: PhotoStore, size: str,
                held_at: str | None = None, workers: int = 16, log=print) -> dict:
    """Copy photos we already hold in one store into another (e.g. S3 -> laptop), so no
    photo is fetched from iNat twice. Each file must match the hash recorded when it was
    downloaded; a mismatch is reported and not recorded. `held_at` keeps only photos
    the destination already holds at that other size (a sample at a new size)."""
    from concurrent.futures import ThreadPoolExecutor as Pool
    extra, params = "", [source.location, size, dest.location, size]
    if held_at:
        extra = (" and exists (select 1 from photo_copies h where h.photo_id = s.photo_id"
                 " and h.store = ? and h.size = ?)")
        params += [dest.location, held_at]
    rows = conn.execute(
        "select s.photo_id, s.path, s.bytes, s.sha256 from photo_copies s "
        "where s.store = ? and s.size = ? and not exists (select 1 from photo_copies d "
        "where d.photo_id = s.photo_id and d.store = ? and d.size = ?)" + extra
        + " order by s.photo_id", params).fetchall()
    totals = {"queued": len(rows), "copied": 0, "mismatch": 0, "error": 0, "bytes": 0}

    def one(row):
        pid, path, _n, sha = row
        try:
            body = source.get(path)
        except Exception as e:
            return pid, path, None, f"read: {e.__class__.__name__}"
        if sha and hashlib.sha256(body).hexdigest() != sha:
            return pid, path, None, "hash mismatch"
        try:
            dest.put(path, body)
        except Exception as e:
            return pid, path, None, f"write: {e.__class__.__name__}"
        return pid, path, body, None

    with Pool(max_workers=workers) as pool:
        for i, (pid, path, body, err) in enumerate(pool.map(one, rows), 1):
            if err:
                totals["mismatch" if err == "hash mismatch" else "error"] += 1
                log(f"  photo {pid}: {err}")
            else:
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with conn:
                    conn.execute(
                        "insert or replace into photo_copies (photo_id, store, size, path, "
                        "bytes, sha256, downloaded_at) values (?, ?, ?, ?, ?, ?, ?)",
                        (pid, dest.location, size, path, len(body),
                         hashlib.sha256(body).hexdigest(), now))
                totals["copied"] += 1
                totals["bytes"] += len(body)
            if i % 1000 == 0:
                log(f"  {i:,}/{len(rows):,} photos")
    return totals


def download_all(conn: sqlite3.Connection, store: PhotoStore, size: str = "medium",
                 north_america_only: bool = True, limit: int | None = None,
                 max_hours: float | None = None,
                 checkpoint: Callable[[], None] | None = None,
                 checkpoint_every: float = 600, random_order: bool = False,
                 record_sample: int | None = None, first_seen_since: str | None = None,
                 held_at: str | None = None, hosts: tuple[str, ...] | None = None,
                 policies: dict[str, HostPolicy] | None = None, poll: float = 30,
                 unembedded_for: str | None = None, log=print) -> dict:
    """Download pending photos into `store`. `checkpoint` (e.g. copy the manifest to S3)
    runs every `checkpoint_every` seconds and once at the end.

    Each host has its own lane: at most its concurrency in flight, so a host waiting
    for its byte budget (the capped static host, for hours) can't hold the threads
    the others need. Budgets start from what earlier runs fetched (seed_budgets)."""
    config.ensure_dirs()
    rows = pending_photos(conn, north_america_only, limit, size, store.location, random_order,
                          record_sample, first_seen_since, held_at, hosts, unembedded_for)
    return run_downloads(conn, rows, store, size,
                         lambda r, now: save_result(conn, r, size, now, store.location),
                         policies=policies, max_hours=max_hours, checkpoint=checkpoint,
                         checkpoint_every=checkpoint_every, poll=poll, log=log)


def run_downloads(conn: sqlite3.Connection, rows, store: PhotoStore, size: str,
                  save: Callable[[Result, str], None],
                  policies: dict[str, HostPolicy] | None = None,
                  max_hours: float | None = None,
                  checkpoint: Callable[[], None] | None = None,
                  checkpoint_every: float = 600, poll: float = 30, log=print) -> dict:
    """Download `rows` (photo_id, source_url, host, source) into `store` at `size`, each host in
    its own lane within its limits; `save(result, now)` records each one (inside a
    transaction on `conn`). Shared by download_all and the benchmark (heldout.py)."""
    policies = policies or default_policies()
    seeded = seed_budgets(conn, policies)
    for host, n in seeded.items():
        if n:
            log(f"  {host}: {n / GB:.1f} GB already fetched inside its budget windows")
    gates: dict[str, HostGate] = {}
    session = requests.Session()
    session.headers["User-Agent"] = config.USER_AGENT
    adapter = requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=16)
    session.mount("https://", adapter)
    stop = threading.Event()
    deadline = time.monotonic() + max_hours * 3600 if max_hours else None
    totals = {"queued": len(rows), "done": 0, "missing": 0, "error": 0, "bytes": 0}
    started = time.monotonic()

    def gate_for(host: str) -> HostGate:
        if host not in gates:
            gates[host] = HostGate(policies.get(host, FALLBACK))
        return gates[host]

    lanes: dict[str, deque] = {}
    for row in rows:
        lanes.setdefault(row["host"], deque()).append(row)

    def cap(host: str) -> int:
        return policies.get(host, FALLBACK).concurrency

    # One thread per lane slot, so every host can run at its own concurrency at once.
    workers = max(1, sum(cap(h) for h in lanes))
    pending: dict[Future, str] = {}
    in_flight: dict[str, int] = {h: 0 for h in lanes}
    finished = 0
    last_checkpoint = time.monotonic()
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            def top_up():
                for host, lane in lanes.items():
                    while lane and in_flight[host] < cap(host) and not stop.is_set():
                        row = lane.popleft()
                        fut = pool.submit(download_one, session, gate_for(host),
                                          row["photo_id"], row["source_url"], size, store, stop,
                                          row["source"])
                        pending[fut] = host
                        in_flight[host] += 1
            top_up()
            while pending:
                done, _ = wait(list(pending), timeout=poll, return_when=FIRST_COMPLETED)
                now = datetime.now(timezone.utc).isoformat(timespec="seconds")
                with conn:
                    for fut in done:
                        in_flight[pending.pop(fut)] -= 1
                        r = fut.result()
                        save(r, now)
                        if r.error == "stopped":
                            continue
                        totals[r.status] += 1
                        totals["bytes"] += r.bytes or 0
                        finished += 1
                        if finished % 2000 == 0:
                            log(f"  {finished:,}/{len(rows):,} photos, "
                                f"{totals['bytes'] / GB:.1f} GB, {totals['missing']:,} missing, "
                                f"{totals['error']:,} errors, {time.monotonic() - started:.0f}s")
                if deadline and time.monotonic() > deadline:
                    stop.set()
                if checkpoint and time.monotonic() - last_checkpoint > checkpoint_every:
                    checkpoint()
                    last_checkpoint = time.monotonic()
                top_up()
    except KeyboardInterrupt:
        stop.set()
        log("Stopping; progress so far is saved. Re-run to resume.")
    finally:
        if checkpoint:
            checkpoint()
    return totals
