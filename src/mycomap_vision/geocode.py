"""Place search for the Identify page's map (Steve, 2026-10-11): a place name to points.

The browser asks Vision (/api/geocode/search, behind the sign-in gate); Vision asks
OpenStreetMap's Nominatim, within its usage policy (https://operations.osmfoundation.org/
policies/nominatim/): one request at a time and at most one a second for the whole
server, our own User-Agent, answers cached. Read-only; nothing about the person is sent
but the words they typed. Same answer shape as mycomap.org's /api/geocode/search.
"""
from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections import OrderedDict
from typing import Callable
from urllib.parse import urlencode

from . import config
from .guards import RateLimiter

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


# --- Suggestions as you type (Steve, 2026-10-20) ------------------------------------------
# Nominatim's policy forbids autocomplete, so suggestions come from Photon instead
# (https://photon.komoot.io, OpenStreetMap data, built for search-as-you-type; komoot's
# public instance asks for fair use and throttles heavy use). Asked by the server only, with
# our User-Agent and a contact address, never with anything about the person: no location
# bias, no address, just the words typed. Any refusal (too short, too many, Photon down or
# saying no) is an empty list and a status, and the page says to press Search, which still
# asks Nominatim (PlaceSearch above).

PHOTON = "https://photon.komoot.io/api/"
CONTACT = "info@mycomap.org"
# A whole word naming a county: Photon ranks counties far down unless asked for that layer
# ("Perry County, OH" otherwise gives Perry Township and courthouses).
_COUNTY_WORD = re.compile(r"\b(county|parish)\b", re.I)
_US_COUNTY_SUFFIX = re.compile(r"\b(county|parish|borough|census area|municipality)$", re.I)


def photon_fetch(url: str) -> object:
    import requests
    r = requests.get(url, headers={"User-Agent": config.USER_AGENT, "From": CONTACT,
                                   "Accept": "application/json"}, timeout=5)
    r.raise_for_status()
    return r.json()


class PlaceSuggest:
    """Suggestions for what has been typed so far, as [{display_name, lat, lon}] (the shape
    of PlaceSearch) plus a status: "ok", "short" (under MIN_QUERY letters), "long",
    "limited" (this address, or the whole server, asked too often) or "unavailable"
    (Photon failed or refused; it is left alone for `backoff` seconds, or as long as it
    asks). Answers are cached for `ttl` seconds, failures never."""

    def __init__(self, fetch: Callable[[str], object] = photon_fetch, *,
                 per_ip: tuple[int, float] = (30, 60.0), upstream: tuple[int, float] = (60, 60.0),
                 ttl: float = 24 * 3600, max_entries: int = 2000, backoff: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.fetch, self.ttl, self.max_entries, self.backoff, self.clock = fetch, ttl, max_entries, backoff, clock
        self.per_ip = RateLimiter(*per_ip, clock=clock)
        self.upstream = RateLimiter(*upstream, clock=clock)
        self._cache: OrderedDict[str, tuple[float, list[dict]]] = OrderedDict()
        self._lock = threading.Lock()
        self._paused_until = 0.0
        self.upstream_calls = 0

    def suggest(self, q: str, client: str = "unknown") -> tuple[list[dict], str]:
        q = " ".join((q or "").split())
        if len(q.replace(" ", "")) < MIN_QUERY:      # letters, not spaces
            return [], "short"
        if len(q) > MAX_QUERY:
            return [], "long"
        if self.per_ip.check(client):
            return [], "limited"
        key = q.lower()
        with self._lock:
            now = self.clock()
            hit = self._cache.get(key)
            if hit and now - hit[0] < self.ttl:
                self._cache.move_to_end(key)
                return hit[1], "ok"
            if now < self._paused_until:
                return [], "unavailable"
        if self.upstream.check("all"):
            return [], "limited"
        params = [("q", q), ("limit", MAX_RESULTS + 3), ("lang", "en")]
        if _COUNTY_WORD.search(q):
            params.append(("layer", "county"))
        try:
            self.upstream_calls += 1
            data = self.fetch(f"{PHOTON}?{urlencode(params)}")
        except Exception as e:  # noqa: BLE001 (any failure: the page falls back to Search)
            self._pause(e)
            return [], "unavailable"
        places, seen = [], set()
        features = data.get("features") if isinstance(data, dict) else None
        for f in features if isinstance(features, list) else []:
            p = _suggestion(f)
            if p and p["display_name"].lower() not in seen:
                seen.add(p["display_name"].lower())
                places.append(p)
        places = places[:MAX_RESULTS]
        with self._lock:
            self._cache[key] = (self.clock(), places)
            self._cache.move_to_end(key)
            while len(self._cache) > self.max_entries:
                self._cache.popitem(last=False)
        return places, "ok"

    def _pause(self, e: Exception) -> None:
        """Leave Photon alone for a while after it fails; as long as it asks (Retry-After,
        at most 10 minutes) when it says too many."""
        wait = self.backoff
        headers = getattr(getattr(e, "response", None), "headers", None) or {}
        try:
            wait = max(wait, min(600.0, float(headers.get("Retry-After") or 0)))
        except (TypeError, ValueError):
            pass
        with self._lock:
            self._paused_until = self.clock() + wait


def _suggestion(f: object) -> dict | None:
    """One Photon feature as {display_name, lat, lon}: its name, then the town, county,
    state and country it is in. None when it has no name or no usable point."""
    if not isinstance(f, dict) or not isinstance(f.get("properties"), dict):
        return None
    props = f["properties"]
    try:
        lon, lat = (float(c) for c in f["geometry"]["coordinates"][:2])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    us = str(props.get("countrycode") or "").upper() == "US"

    def county(name: object) -> str:
        """A US county as people write it: Photon names Perry County, Ohio just "Perry"."""
        name = str(name or "").strip()
        return f"{name} County" if name and us and not _US_COUNTY_SUFFIX.search(name) else name

    kind = props.get("type")
    name = county(props.get("name")) if kind == "county" else str(props.get("name") or "").strip()
    if not name:
        return None
    parts = [name]
    if kind not in ("city", "county", "state", "country"):
        parts.append(props.get("city"))
    if kind not in ("county", "state", "country"):
        parts.append(county(props.get("county")))
    if kind not in ("state", "country"):
        parts.append(props.get("state"))
    if kind != "country":
        parts.append(props.get("country"))
    shown: list[str] = []
    for part in (str(x or "").strip() for x in parts):
        if part and part.lower() not in (s.lower() for s in shown):
            shown.append(part)
    return {"display_name": ", ".join(shown), "lat": lat, "lon": lon}


class GeocodeQueryRedactor(logging.Filter):
    """Keeps what people type out of the server's access log: uvicorn logs each request's
    path with its query, and suggestions are one request per pause in typing. The path
    stays (so the requests can still be counted), the words go."""

    def filter(self, record: logging.LogRecord) -> bool:
        args = record.args
        if isinstance(args, tuple) and len(args) >= 3 and isinstance(args[2], str) \
                and args[2].startswith("/api/geocode/") and "?" in args[2]:
            record.args = (*args[:2], args[2].split("?", 1)[0] + "?q=(hidden)", *args[3:])
        return True


def redact_access_log() -> None:
    log = logging.getLogger("uvicorn.access")
    if not any(isinstance(f, GeocodeQueryRedactor) for f in log.filters):
        log.addFilter(GeocodeQueryRedactor())
