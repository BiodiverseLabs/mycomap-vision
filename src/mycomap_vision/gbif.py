"""A synonym crosswalk through GBIF, for scoring outside models only (external_report.py).

The Danish models name Danish records in the Danish checklist's names; we name North
American records in ours. Where the two differ only by synonymy ('Lepista nuda' and
'Collybia nuda', 'Piptoporus betulinus' and 'Fomitopsis betulina'), a fair comparison
counts them as one species. Steve's coordinator (2026-10-09): yes, a GBIF crosswalk, for
scoring only, never Vision's labels, and applied the same way to every model scored
(Vision too: it may make a Vision answer count as right).

Each formal species name (never a temporary code or a one-word name) is matched exactly,
with no fuzzy matching, in two checklists GBIF serves:

    gbif  the GBIF Backbone (GET api.gbif.org/v1/species/match, strict, kingdom hint
          Fungi). It lags recent combinations: it has no 'Collybia nuda'.
    col   the Catalogue of Life eXtended Release through GBIF's checklist match
          (GET api.gbif.org/v2/species/match, checklistKey COL_XR), which has them.

A name's keys are its accepted usage in each checklist (the accepted name's key for a
synonym, its own for an accepted name), at species rank or below. Two names are one
species when they share a key in either checklist. Names neither checklist matches
exactly have no key and are compared by Vision's labels alone.

Answers are cached (<data>/external/gbif/match.sqlite), so a report reads the cache only
and makes no request; `mv external crosswalk` fills it, one request at a time, paced
(default 4 a second across both checklists), with the project's User-Agent, backing off on
429 and 5xx. GBIF is only read. No iNat call is made here.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable

import requests

from . import config
from .ratelimit import MinInterval

API = "https://api.gbif.org"
COL_XR = "7ddf754f-d193-4cc9-b351-99906754a03b"     # Catalogue of Life eXtended Release
SOURCES = ("gbif", "col")
SPECIES_RANKS = {"SPECIES", "SUBSPECIES", "VARIETY", "FORM", "INFRASPECIFIC_NAME"}


def cache_path() -> Path:
    return config.DATA_DIR / "external" / "gbif" / "match.sqlite"


def accepted_key(source: str, body: dict) -> str | None:
    """'<source>:<accepted key>' for an exact match at species rank or below, else None."""
    if source == "gbif":
        if body.get("matchType") != "EXACT" or body.get("rank") not in SPECIES_RANKS:
            return None
        key = body.get("acceptedUsageKey") or body.get("usageKey")
    elif source == "col":
        usage = body.get("usage") or {}
        if (body.get("diagnostics") or {}).get("matchType") != "EXACT" \
                or usage.get("rank") not in SPECIES_RANKS:
            return None
        key = (body.get("acceptedUsage") or {}).get("key") or usage.get("key")
    else:
        raise ValueError(f"unknown checklist {source!r}")
    return f"{source}:{key}" if key else None


class Matcher:
    """GBIF's name match, cached in sqlite and paced; a cached name is never asked again."""

    def __init__(self, path: Path | None = None, session: requests.Session | None = None,
                 interval: float = 0.25, sleep: Callable[[float], None] = time.sleep):
        self.path = path or cache_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, check_same_thread=False)
        self.db.execute("create table if not exists matches (source text not null, name text "
                        "not null, body text not null, fetched_at text not null, "
                        "primary key (source, name))")
        self.lock = threading.Lock()
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.pace = MinInterval(interval, sleep=sleep)
        self.calls = 0

    def cached(self, source: str, name: str) -> dict | None:
        with self.lock:
            row = self.db.execute("select body from matches where source = ? and name = ?",
                                  (source, name)).fetchone()
        return json.loads(row[0]) if row else None

    def _request(self, source: str, name: str) -> dict:
        if source == "gbif":
            url, params = f"{API}/v1/species/match", {"name": name, "kingdom": "Fungi",
                                                      "strict": "true"}
        else:
            url, params = f"{API}/v2/species/match", {"scientificName": name,
                                                      "checklistKey": COL_XR}
        for attempt in range(6):
            self.pace.wait()
            self.calls += 1
            try:
                r = self.session.get(url, params=params, timeout=30)
            except requests.RequestException:
                self.pace.pause(5 * (attempt + 1))
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 or r.status_code >= 500:
                self.pace.pause(15 * (attempt + 1))
                continue
            raise RuntimeError(f"GBIF answered {r.status_code} for {name!r}: {r.text[:200]}")
        raise RuntimeError("GBIF kept failing; stopped. Run again to resume (answers are cached).")

    def match(self, source: str, name: str) -> dict:
        body = self.cached(source, name)
        if body is None:
            body = self._request(source, name)
            with self.lock, self.db:
                self.db.execute("insert or replace into matches values (?, ?, ?, ?)",
                                (source, name, json.dumps(body),
                                 datetime.now(timezone.utc).isoformat(timespec="seconds")))
        return body

    def fill(self, names_: Iterable[str], sources: Iterable[str] = SOURCES, log=print) -> dict:
        """Ask every name not cached yet, the checklists side by side (one pace shared)."""
        from concurrent.futures import ThreadPoolExecutor
        todo = [(s, n) for n in sorted(set(names_)) for s in sources
                if self.cached(s, n) is None]
        stats = Counter(asked=len(todo))
        done = 0
        with ThreadPoolExecutor(max_workers=2) as pool:
            for _ in pool.map(lambda sn: self.match(*sn), todo):
                done += 1
                if done % 500 == 0:
                    log(f"  {done:,}/{len(todo):,} GBIF matches")
        stats["calls"] = self.calls
        return dict(stats)

    def close(self) -> None:
        self.db.close()


class Crosswalk:
    """{name: accepted keys} for scoring: `same(a, b)` when two names share a key."""

    def __init__(self, keys: dict[str, frozenset[str]] | None = None):
        self.keys = keys or {}

    @classmethod
    def from_cache(cls, matcher: Matcher, names_: Iterable[str],
                   sources: Iterable[str] = SOURCES) -> "Crosswalk":
        """Keys of `names_` from the cache only (a name not asked yet has none)."""
        keys = {}
        for n in set(names_):
            got = set()
            for s in sources:
                body = matcher.cached(s, n)
                k = accepted_key(s, body) if body else None
                if k:
                    got.add(k)
            if got:
                keys[n] = frozenset(got)
        return cls(keys)

    def of(self, name: str | None) -> frozenset[str]:
        return self.keys.get(name or "", frozenset())

    def same(self, a: str | None, b: str | None) -> bool:
        return bool(self.of(a) & self.of(b))

    def summary(self, names_: Iterable[str]) -> dict:
        names_ = set(names_)
        by = Counter()
        for n in names_:
            ks = self.of(n)
            by["matched"] += bool(ks)
            for s in SOURCES:
                by[f"matched in {s}"] += any(k.startswith(s + ":") for k in ks)
        return {"names": len(names_), **dict(by)}


def merged_groups(cw: Crosswalk, names_: Iterable[str]) -> list[list[str]]:
    """Names that the crosswalk joins although they are written differently, for a person
    to look over (the report lists a sample)."""
    by_key: dict[str, set[str]] = defaultdict(set)
    for n in names_:
        for k in cw.of(n):
            by_key[k].add(n)
    return sorted({tuple(sorted(v)) for v in by_key.values() if len(v) > 1})
