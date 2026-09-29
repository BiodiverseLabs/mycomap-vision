"""Download photos for the green records, within iNat's limits, with a hash for each.

Hosts are limited separately:
- the AWS Open Data bucket (CC-licensed photos) is built for bulk download;
- iNat's static host (all-rights-reserved photos) gets iNat's media rule:
  under 5 GB an hour and 24 GB a day. We stay at 4 GB and 20 GB.
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

from . import config
from .licenses import (OPEN_DATA_HOST, STATIC_HOST, looks_like_image, photo_relpath, sized_url,
                       taken_down, url_extension)
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
    }


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
    any store): a restarted downloader must not get a fresh day's budget. Returns the
    bytes counted per host."""
    now = now or datetime.now(timezone.utc)
    counted = {}
    for host, policy in policies.items():
        if not policy.budgets:
            continue
        longest = max(b.window for b in policy.budgets)
        total = 0
        for when, n in conn.execute(
                "select c.downloaded_at, c.bytes from photo_copies c join photos p on "
                "p.photo_id = c.photo_id where p.host = ? and c.downloaded_at is not null "
                "and c.bytes is not null", (host,)):
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
                 size: str, store: PhotoStore, stop: threading.Event) -> Result:
    if taken_down(source_url):
        return Result(photo_id, "missing", error="taken down on iNat (copyright)")
    try:
        url = sized_url(source_url, size)
    except ValueError as e:
        return Result(photo_id, "error", error=str(e))
    rel = photo_relpath(photo_id, size, url_extension(source_url))
    with gate.sem:
        while (w := gate.budget_wait()) > 0:
            if stop.wait(min(w, 60)):
                return Result(photo_id, "error", error="stopped")
        gate.pacer.wait()
        try:
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
                   hosts: tuple[str, ...] | None = None):
    """Photos of green records with no copy at `size` (in `store_location`, when given),
    failed fewest times first. Photos iNat no longer has, and ones that failed
    MAX_ATTEMPTS times, are left out. A copy at another size or in another store
    doesn't count, so a laptop sample and the S3 set are kept independently.

    `held_at` keeps only photos the store already holds at that other size (a
    sample fetched again at a new size); `hosts` keeps only photos served from
    those hosts (e.g. the open-data bucket, when another machine is using the
    capped host's budget).
    """
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
    sql = f"""
      select distinct p.photo_id, p.source_url, p.host, p.attempts
      from photos p
      join observation_photos op on op.photo_id = p.photo_id
      join records r on r.observation_id = op.observation_id {na}
      where p.status != 'missing' and p.attempts < {MAX_ATTEMPTS}
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


def download_all(conn: sqlite3.Connection, store: PhotoStore, size: str = "medium",
                 north_america_only: bool = True, limit: int | None = None,
                 max_hours: float | None = None,
                 checkpoint: Callable[[], None] | None = None,
                 checkpoint_every: float = 600, random_order: bool = False,
                 record_sample: int | None = None, first_seen_since: str | None = None,
                 held_at: str | None = None, hosts: tuple[str, ...] | None = None,
                 policies: dict[str, HostPolicy] | None = None, poll: float = 30,
                 log=print) -> dict:
    """Download pending photos into `store`. `checkpoint` (e.g. copy the manifest to S3)
    runs every `checkpoint_every` seconds and once at the end.

    Each host has its own lane: at most its concurrency in flight, so a host waiting
    for its byte budget (the capped static host, for hours) can't hold the threads
    the others need. Budgets start from what earlier runs fetched (seed_budgets)."""
    config.ensure_dirs()
    rows = pending_photos(conn, north_america_only, limit, size, store.location, random_order,
                          record_sample, first_seen_since, held_at, hosts)
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
                                          row["photo_id"], row["source_url"], size, store, stop)
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
                        save_result(conn, r, size, now, store.location)
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
