"""Observation sessions for the app (docs/api.md §3.3).

The app opens a session when someone starts photographing a find, and sends each photo
as it is taken. Each photo is embedded once. Its vectors (this model's, and the gate's
base vectors), its gate verdict and its photo-type tag are kept here; the photo itself
never is. Every change
(a photo added or removed, place or date set) re-scores all of the session's stored
vectors together, so the combined answer is exactly what identifying all its photos at
once would give, and the app never merges answers itself.

Sessions live in this process's memory: a restart or deploy loses them, and the app
opens a new one (404 session_gone). Lifetime (Steve, 2026-10-11): `idle_ttl` after the
last call (30 min), `max_age` from creation at most (2 h). Memory is capped in bytes
(`max_bytes`, MV_SESSIONS_MAX_MB, 64 MB): past it the session idle longest is dropped,
and a new session is refused while every one was active in the last `busy_window` s.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

MAX_PHOTOS = 10
MAX_REF_LENGTH = 64


class Gone(LookupError):
    """No such session for this caller: unknown, expired, lost to a restart, or theirs."""


class Full(RuntimeError):
    """No room for another session right now."""


@dataclass
class Photo:
    ref: str
    vector: np.ndarray                    # this model's unit vector
    base: np.ndarray | None               # the gate's base-model vector, if the gate runs
    verdict: dict | None                  # the gate's verdict, when judged
    gate_fingerprint: str | None          # which gate judged it (re-judged when it changes)
    found: tuple = (None, None, None)     # its EXIF (latitude, longitude, date), for gaps
    photo_type: dict | None = None        # its photo-type tag (phototypes.py), when tagged
    type_fingerprint: str | None = None   # which tagger tagged it (tagged again when it changes)

    @property
    def nbytes(self) -> int:
        return self.vector.nbytes + (self.base.nbytes if self.base is not None else 0) + 256


@dataclass
class Session:
    id: str
    owner: str
    backbone: str
    method: str
    top_k: int
    entered: dict                         # {latitude, longitude, observed_on} as given
    client_ref: str | None
    keep_photos: bool
    created: float
    last: float
    photos: list[Photo] = field(default_factory=list)
    combined: dict | None = None          # the last answer, as sent
    counted: bool = False                 # counted against quota (at its first photo)
    include: set = field(default_factory=set)  # photo_refs counted whatever their photo type
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _answer_bytes: int = 0

    def photo(self, ref: str) -> Photo | None:
        return next((p for p in self.photos if p.ref == ref), None)

    def index_of(self, ref: str) -> int | None:
        return next((i for i, p in enumerate(self.photos) if p.ref == ref), None)

    def remember(self, combined: dict | None) -> None:
        self.combined = combined
        self._answer_bytes = len(json.dumps(combined)) if combined is not None else 0

    @property
    def nbytes(self) -> int:
        return 1024 + sum(p.nbytes for p in self.photos) + self._answer_bytes


class Sessions:
    def __init__(self, idle_ttl: float = 1800, max_age: float = 7200,
                 max_bytes: int = 64 * 2**20, busy_window: float = 120,
                 clock: Callable[[], float] = time.monotonic):
        self.idle_ttl, self.max_age, self.max_bytes = idle_ttl, max_age, max_bytes
        self.busy_window, self.clock = busy_window, clock
        self._by_id: dict[str, Session] = {}
        self._lock = threading.Lock()

    def _alive(self, s: Session, now: float) -> bool:
        return now - s.last < self.idle_ttl and now - s.created < self.max_age

    def _sweep(self, now: float) -> None:
        for sid in [sid for sid, s in self._by_id.items() if not self._alive(s, now)]:
            del self._by_id[sid]

    def create(self, owner: str, **fields) -> Session:
        now = self.clock()
        with self._lock:
            self._sweep(now)
            self._make_room(now, 4096)
            s = Session(id="s_" + secrets.token_urlsafe(16), owner=owner, created=now,
                        last=now, **fields)
            self._by_id[s.id] = s
            return s

    def get(self, sid: str, owner: str) -> Session:
        """The caller's live session (its idle clock restarted), else Gone."""
        now = self.clock()
        with self._lock:
            s = self._by_id.get(sid)
            if s is None or s.owner != owner or not self._alive(s, now):
                if s is not None and not self._alive(s, now):
                    del self._by_id[sid]
                raise Gone(sid)
            s.last = now
            return s

    def delete(self, sid: str, owner: str) -> None:
        with self._lock:
            s = self._by_id.get(sid)
            if s is None or s.owner != owner:
                raise Gone(sid)
            del self._by_id[sid]

    def make_room(self, extra: int) -> None:
        """Before a session grows by `extra` bytes: drop idle sessions past the cap, or Full."""
        with self._lock:
            self._make_room(self.clock(), extra)

    def _make_room(self, now: float, extra: int) -> None:
        self._sweep(now)
        used = sum(s.nbytes for s in self._by_id.values())
        if used + extra <= self.max_bytes:
            return
        for s in sorted(self._by_id.values(), key=lambda s: s.last):
            if now - s.last < self.busy_window:
                break                        # the rest are newer still
            del self._by_id[s.id]
            used -= s.nbytes
            if used + extra <= self.max_bytes:
                return
        raise Full("no room for another session right now")

    def expires_in(self, s: Session) -> float:
        """Seconds until `s` ends if nothing more happens."""
        now = self.clock()
        return max(0.0, min(self.idle_ttl - (now - s.last), self.max_age - (now - s.created)))

    def state(self) -> dict:
        with self._lock:
            self._sweep(self.clock())
            return {"open": len(self._by_id),
                    "photos": sum(len(s.photos) for s in self._by_id.values()),
                    "bytes": sum(s.nbytes for s in self._by_id.values()),
                    "max_bytes": self.max_bytes}
