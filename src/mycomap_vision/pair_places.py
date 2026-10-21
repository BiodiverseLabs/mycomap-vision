"""Where each copy of a record pair is, as its own site lists it (Steve, 2026-10-21): the
location name on Mushroom Observer, the place on iNaturalist. Shown beside the two copies'
photos in Human Feedback's "One find, two places".

Text only, never a coordinate: the queue keeps none (feedback.rows_from_twin_locations), and
an answer's latitude / longitude are dropped as it is read. MO's name comes from Vision's
cache of MO's public points (mopoints.py) when it holds one, else from MO's API; iNat's from
iNat's API. One request per record, paced, each answer kept an hour in memory.
"""

from __future__ import annotations

import threading
import time
from typing import Callable

import requests

from . import config
from .ratelimit import MinInterval

MO_API = "https://mushroomobserver.org/api2/observations"
INAT_API = "https://api.inaturalist.org/v2/observations"
TTL = 3600.0


def site_of(key: str) -> str:
    return "MO" if key.startswith("mo:") else "iNat"


def mo_place(answer: dict) -> str | None:
    """MO api2 (detail=high) -> the observation's location name."""
    results = answer.get("results") or []
    loc = results[0].get("location") if results and isinstance(results[0], dict) else None
    name = loc.get("name") if isinstance(loc, dict) else None
    return (name or "").strip() or None


def inat_place(answer: dict) -> str | None:
    """iNat v2 (fields=place_guess) -> the place as the observation lists it."""
    results = answer.get("results") or []
    guess = results[0].get("place_guess") if results and isinstance(results[0], dict) else None
    return (guess or "").strip() or None


class PairPlaces:
    """`cached_mo(key)`: MO's location name from Vision's cache, or None. `get(url, params)`
    returns the parsed JSON answer (raises on a failure)."""

    def __init__(self, cached_mo: Callable[[str], str | None] | None = None,
                 get: Callable[[str, dict], dict] | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 mo_interval: float = 5.0, inat_interval: float = 1.0):
        self.cached_mo = cached_mo
        self.get = get or self._get
        self.clock = clock
        self._seen: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._mo_pace = MinInterval(mo_interval)      # MO: at least 5 s apart (mo.py)
        self._inat_pace = MinInterval(inat_interval)

    @staticmethod
    def _get(url: str, params: dict) -> dict:
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": config.USER_AGENT})
        r.raise_for_status()
        return r.json()

    def one(self, key: str) -> dict:
        """{"site", "text", "status"}: status "ok" (text is the place), "none" (the site lists
        no place) or "unavailable" (it did not answer just now; not kept)."""
        with self._lock:
            hit = self._seen.get(key)
        if hit and self.clock() - hit[0] < TTL:
            return hit[1]
        site = site_of(key)
        text = None
        try:
            if site == "MO":
                text = self.cached_mo(key) if self.cached_mo else None
                if text is None:
                    self._mo_pace.wait()
                    text = mo_place(self.get(MO_API, {"id": key[3:], "detail": "high",
                                                      "format": "json"}))
            else:
                self._inat_pace.wait()
                text = inat_place(self.get(INAT_API, {"id": key, "fields": "place_guess"}))
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return {"site": site, "text": None, "status": "unavailable"}
        out = {"site": site, "text": text, "status": "ok" if text else "none"}
        with self._lock:
            self._seen[key] = (self.clock(), out)
        return out

    def of(self, keys: list[str]) -> dict[str, dict]:
        return {k: self.one(k) for k in keys}
