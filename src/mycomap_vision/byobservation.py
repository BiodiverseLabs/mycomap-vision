"""Identify by observation number (docs/api.md §3.2): an iNaturalist or Mushroom Observer
observation's public photos, place and date, read from that site and kept in memory only.

Every read here is polite, for any number of callers:

- one observation at a time across the whole server (a lock), and at most MAX_WAITING
  callers queued behind it; more get Busy (503 `busy`) rather than tie up a thread;
- each host keeps its own pace: iNat's API one request a second (its asked-for 60 a
  minute), its photo hosts as photos.py paces them, and Mushroom Observer 5 s between
  every request, photos included (its README_API: anonymous traffic, 20 a minute), and
  at least the last answer's run_time;
- config.USER_AGENT on every request, HTTPS only, and photos only from those sites'
  photo hosts (one redirect at most, to MO's own image host: photos.mo_redirect_ok).

Place and date are what the site publishes: iNat's public location (obscured as iNat
obscures it), MO's GPS when not hidden, else the centre of its named place when that
place is small (MO_AREA_MAX_DEGREES). Nothing fetched is written anywhere.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse

import requests

from . import config
from .dates import real_date
from .licenses import MO_HOST, OPEN_DATA_HOST, STATIC_HOST, photo_url
from .mo import photo_id_of
from .photos import MO_MIN_INTERVAL, mo_redirect_ok
from .sources import INAT, MO, record_url

SOURCES = (INAT, MO)
INAT_API = "https://api.inaturalist.org/v2/observations"
INAT_FIELDS = ("(id:!t,observed_on:!t,location:!t,obscured:!t,"
               "taxon:(name:!t,rank:!t),"
               "observation_photos:(position:!t,photo:(id:!t,url:!t)))")
MO_API = "https://mushroomobserver.org/api2/observations"
INAT_PHOTO_HOSTS = (OPEN_DATA_HOST, STATIC_HOST)
PHOTO_SIZE = "large"            # iNat 1024 px, MO 960 px: what the reference was embedded from
MAX_PHOTO_BYTES = 15 * 2**20
MO_AREA_MAX_DEGREES = 0.5       # an MO place whose box is wider than this gives no place
MAX_WAITING = 3                 # callers queued behind the one fetching
TIMEOUT = (10, 30)              # connect, read (seconds)

# Seconds between requests to each host.
INTERVALS = {"api.inaturalist.org": 1.0, OPEN_DATA_HOST: 0.02, STATIC_HOST: 0.25,
             MO_HOST: MO_MIN_INTERVAL}


class NotFound(LookupError):
    """The site has no such observation (404 observation_not_found)."""


class Unavailable(RuntimeError):
    """The site didn't answer usefully (502 source_unavailable)."""

    def __init__(self, message: str, retry_after: int = 30):
        super().__init__(message)
        self.retry_after = retry_after


class Busy(RuntimeError):
    """Too many callers already waiting for the one fetch at a time (503 busy)."""

    def __init__(self, retry_after: int):
        super().__init__("busy fetching other observations")
        self.retry_after = retry_after


@dataclass
class Observation:
    source: str
    source_id: str
    photo_urls: list[str]               # in the site's order, at PHOTO_SIZE
    photo_ids: list[int]                # as Vision's photos table numbers them
    latitude: float | None = None
    longitude: float | None = None
    observed_on: str | None = None
    place: str | None = None            # "coordinates", "obscured", "area" or None
    name_at_source: str | None = None
    rank_at_source: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def url(self) -> str | None:
        return record_url(self.source, self.source_id)


def valid_id(raw: str) -> str | None:
    """The observation number as digits, or None when it isn't one."""
    raw = (raw or "").strip()
    return raw if raw.isdigit() and 1 <= len(raw) <= 12 and int(raw) > 0 else None


def parse_inat(obs: dict) -> Observation:
    """One iNat v2 result -> Observation (photos at PHOTO_SIZE, in position order)."""
    urls, ids = [], []
    for op in sorted(obs.get("observation_photos") or [], key=lambda p: p.get("position") or 0):
        p = op.get("photo") or {}
        try:
            url = photo_url(INAT, p.get("url") or "", PHOTO_SIZE)
        except ValueError:
            continue
        if not p.get("id") or urlparse(url).hostname not in INAT_PHOTO_HOSTS:
            continue
        urls.append(url)
        ids.append(int(p["id"]))
    lat = lng = None
    loc = obs.get("location")
    if loc and "," in loc:
        try:
            lat, lng = (float(x) for x in loc.split(",", 1))
        except ValueError:
            lat = lng = None
    place = None if lat is None else ("obscured" if obs.get("obscured") else "coordinates")
    taxon = obs.get("taxon") or {}
    return Observation(INAT, str(obs["id"]), urls, ids, lat, lng,
                       real_date(obs.get("observed_on")), place,
                       taxon.get("name"), taxon.get("rank"))


def parse_mo(obs: dict) -> Observation:
    """One MO api2 observation (detail=high) -> Observation: the primary image first, then
    the rest, at PHOTO_SIZE. GPS unless hidden, else a small named place's centre."""
    images, seen = [], set()
    for img in [obs.get("primary_image"), *(obs.get("images") or [])]:
        if img and img.get("id") and img["id"] not in seen:
            seen.add(img["id"])
            images.append(img)
    urls, ids = [], []
    for img in images:
        source_url = img.get("original_url") or f"https://{MO_HOST}/images/orig/{img['id']}.jpg"
        try:
            urls.append(photo_url(MO, source_url, PHOTO_SIZE))
        except ValueError:
            continue
        ids.append(photo_id_of(int(img["id"])))
    lat = lng = None
    place = None
    if not obs.get("gps_hidden") and obs.get("latitude") is not None \
            and obs.get("longitude") is not None:
        lat, lng, place = float(obs["latitude"]), float(obs["longitude"]), "coordinates"
    else:
        box = obs.get("location") or {}
        try:
            n, s = float(box["latitude_north"]), float(box["latitude_south"])
            e, w = float(box["longitude_east"]), float(box["longitude_west"])
        except (KeyError, TypeError, ValueError):
            n = s = e = w = None
        if n is not None and w <= e and 0 <= n - s <= MO_AREA_MAX_DEGREES \
                and e - w <= MO_AREA_MAX_DEGREES:
            lat, lng, place = (n + s) / 2, (e + w) / 2, "area"
    name = obs.get("consensus") or {}
    return Observation(MO, str(obs["id"]), urls, ids, lat, lng, real_date(obs.get("date")),
                       place, name.get("name"), name.get("rank"))


class Fetcher:
    """Reads observations and their photos politely (module docstring). `session`,
    `clock` and `sleep` are replaced in tests."""

    def __init__(self, session: requests.Session | None = None, clock=time.monotonic,
                 sleep=time.sleep, intervals: dict[str, float] | None = None,
                 max_waiting: int = MAX_WAITING):
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.clock, self.sleep = clock, sleep
        self.intervals = {**INTERVALS, **(intervals or {})}
        self.max_waiting = max_waiting
        self.next_at: dict[str, float] = {}
        self.busy = threading.Lock()            # one observation at a time
        self.count = threading.Lock()
        self.waiting = 0
        self.last_seconds = 10.0                # how long the last fetch took (Retry-After)

    # -- one at a time ---------------------------------------------------------------

    def fetch(self, source: str, source_id: str, max_photos: int,
              admit=None) -> tuple[Observation, list[bytes]]:
        """The observation and up to `max_photos` of its photos (bytes), or NotFound,
        Unavailable, Busy. `admit(observation)`, called before any photo is read, says
        how many to read (0: none) or raises to refuse: a caller over its limits costs
        the photo hosts nothing."""
        with self.count:
            if self.waiting > self.max_waiting:
                raise Busy(int(self.last_seconds * self.waiting) + 1)
            self.waiting += 1
        try:
            with self.busy:
                start = self.clock()
                try:
                    obs = self.observation(source, source_id)
                    n = min(max_photos, len(obs.photo_urls))
                    if admit is not None:
                        n = min(n, admit(obs))
                    bodies = self.photos(obs, n) if n > 0 else []
                finally:
                    self.last_seconds = max(1.0, self.clock() - start)
                return obs, bodies
        finally:
            with self.count:
                self.waiting -= 1

    # -- reads -----------------------------------------------------------------------

    def observation(self, source: str, source_id: str) -> Observation:
        if source == INAT:
            body = self._json(INAT_API, {"id": source_id, "fields": INAT_FIELDS})
            results = body.get("results") or []
            if not results:
                raise NotFound(f"iNaturalist has no observation {source_id}")
            return parse_inat(results[0])
        if source == MO:
            body = self._json(MO_API, {"id": source_id, "detail": "high", "format": "json"})
            run_time = body.get("run_time")
            if isinstance(run_time, (int, float)) and run_time > self.intervals[MO_HOST]:
                self.next_at[MO_HOST] = max(self.next_at.get(MO_HOST, 0),
                                            self.clock() + float(run_time))
            for err in body.get("errors") or []:
                if "ObjectNotFound" in str(err.get("code")):
                    raise NotFound(f"Mushroom Observer has no observation {source_id}")
            if body.get("errors"):
                raise Unavailable("Mushroom Observer answered with an error")
            results = body.get("results") or []
            if not results:
                raise NotFound(f"Mushroom Observer has no observation {source_id}")
            return parse_mo(results[0])
        raise ValueError(f"unknown source {source!r}")

    def photos(self, obs: Observation, max_photos: int) -> list[bytes]:
        """Up to `max_photos` photos as bytes; a photo the host won't give is skipped."""
        out = []
        for url in obs.photo_urls[:max_photos]:
            body = self._photo(url, obs.source)
            if body is not None:
                out.append(body)
        return out

    # -- HTTP ------------------------------------------------------------------------

    def _pace(self, host: str) -> None:
        if host.endswith("." + MO_HOST):          # MO's image host keeps MO's pace
            host = MO_HOST
        wait =self.next_at.get(host, 0) - self.clock()
        if wait > 0:
            self.sleep(wait)
        self.next_at[host] = max(self.next_at.get(host, 0), self.clock()) + \
            self.intervals.get(host, 1.0)

    def _get(self, url: str, **kw) -> requests.Response:
        host = urlparse(url).hostname or ""
        self._pace(host)
        try:
            return self.session.get(url, timeout=TIMEOUT, **kw)
        except requests.RequestException as e:
            raise Unavailable(f"{host} did not answer") from e

    def _json(self, url: str, params: dict) -> dict:
        resp = self._get(url, params=params)
        host = urlparse(url).hostname
        if resp.status_code == 404:
            raise NotFound(f"{host} has no such observation")
        if resp.status_code == 429:
            raise Unavailable(f"{host} asked us to slow down", retry_after=60)
        if resp.status_code != 200:
            raise Unavailable(f"{host} answered {resp.status_code}")
        try:
            return resp.json()
        except ValueError as e:
            raise Unavailable(f"{host} answered something unreadable") from e

    def _photo(self, url: str, source: str) -> bytes | None:
        resp = self._get(url, stream=True, allow_redirects=False)
        if resp.is_redirect or resp.status_code in (301, 302, 303, 307, 308):
            target = urljoin(url, resp.headers.get("Location") or "")
            resp.close()
            if source != MO or not mo_redirect_ok(target):
                return None
            resp = self._get(target, stream=True, allow_redirects=False)
        try:
            if resp.status_code == 429:
                raise Unavailable(f"{urlparse(url).hostname} asked us to slow down",
                                  retry_after=60)
            if resp.status_code != 200:
                return None
            body = b""
            for chunk in resp.iter_content(64 * 1024):
                body += chunk
                if len(body) > MAX_PHOTO_BYTES:
                    return None
            return body
        except requests.RequestException:
            return None
        finally:
            resp.close()
