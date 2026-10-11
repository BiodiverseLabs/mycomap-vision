"""Place search for the Identify page's map (Steve, 2026-10-11): a place name to points.

The browser asks Vision (/api/geocode/search, behind the sign-in gate); Vision asks
OpenStreetMap's Nominatim, within its usage policy (https://operations.osmfoundation.org/
policies/nominatim/): one request at a time and at most one a second for the whole
server, our own User-Agent, answers cached. Read-only; nothing about the person is sent
but the words they typed. Same answer shape as mycomap.org's /api/geocode/search.
"""
from __future__ import annotations

import math
import threading
import time
from collections import OrderedDict
from typing import Callable
from urllib.parse import urlencode

from . import config

NOMINATIM = "https://nominatim.openstreetmap.org/search"
MIN_QUERY = 3
MAX_QUERY = 200
MAX_RESULTS = 5


class GeocodeError(Exception):
    """A refusal the API answers as {"detail": ...} with this status."""

    def __init__(self, status: int, detail: str, retry_after: int | None = None):
        super().__init__(detail)
        self.status, self.detail, self.retry_after = status, detail, retry_after


def nominatim_fetch(url: str) -> object:
    import requests
    r = requests.get(url, headers={"User-Agent": config.USER_AGENT, "Accept": "application/json"},
                     timeout=10)
    r.raise_for_status()
    return r.json()


class PlaceSearch:
    """Searches, cached for `ttl` seconds (at most `max_entries` queries kept), with
    upstream calls at least `min_interval` apart. A search that would wait longer than
    `max_wait` for its turn is refused (429) rather than queued."""

    def __init__(self, fetch: Callable[[str], object] = nominatim_fetch, *,
                 ttl: float = 24 * 3600, max_entries: int = 500, min_interval: float = 1.1,
                 max_wait: float = 5.0, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.fetch, self.ttl, self.max_entries = fetch, ttl, max_entries
        self.min_interval, self.max_wait, self.clock, self.sleep = min_interval, max_wait, clock, sleep
        self._cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._lock = threading.Lock()
        self._next_slot = 0.0
        self.upstream_calls = 0

    def search(self, q: str) -> list[dict]:
        q = " ".join((q or "").split())
        if len(q) < MIN_QUERY:
            return []
        if len(q) > MAX_QUERY:
            raise GeocodeError(400, "Search for a shorter place name.")
        key = q.lower()
        with self._lock:
            now = self.clock()
            hit = self._cache.get(key)
            if hit and now - hit[0] < self.ttl:
                self._cache.move_to_end(key)
                return hit[1]
            wait = self._next_slot - now
            if wait > self.max_wait:
                raise GeocodeError(429, "Too many place searches right now; try again in a few seconds.",
                                   retry_after=math.ceil(wait))
            self._next_slot = max(now, self._next_slot) + self.min_interval
        if wait > 0:
            self.sleep(wait)
        url = f"{NOMINATIM}?{urlencode({'q': q, 'format': 'json', 'limit': MAX_RESULTS})}"
        try:
            self.upstream_calls += 1
            data = self.fetch(url)
        except Exception as e:  # noqa: BLE001 (any upstream failure is the same to the page)
            raise GeocodeError(503, "Place search isn't answering; try again in a minute.") from e
        places = [p for p in (_place(d) for d in (data if isinstance(data, list) else [])) if p]
        with self._lock:
            self._cache[key] = (self.clock(), places[:MAX_RESULTS])
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return places[:MAX_RESULTS]


def _place(d: object) -> dict | None:
    """One Nominatim result as {display_name, lat, lon}; None when it isn't a usable point."""
    if not isinstance(d, dict):
        return None
    try:
        lat, lon = float(d["lat"]), float(d["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    return {"display_name": str(d.get("display_name") or ""), "lat": lat, "lon": lon}
