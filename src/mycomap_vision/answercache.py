"""Answers already given for an iNat or MO observation, kept so the same observation is
not fetched and scored again until what Vision answers from changes.

One row per (observation, model, options): the answer as of an answers version
(`answers_version`: the served release and its reference set as served, which moves when
a release, a night's additions or removals, a new calibration or a photographer's answer
changes the reference). A row for an older version is never served; the next answer
replaces it. The observation's own photos are not checked again while the version holds:
a photo added at the source after the answer is not seen until the version moves.

Durable on the box beside vision-usage.sqlite; observation answers hold no one's
identity, only public observations' scores.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path

SCHEMA = """
create table if not exists observation_answers (
    answer_key text primary key,      -- kind:source:id:backbone/method:options
    version    text not null,         -- answers_version when it was answered
    payload    text not null,         -- JSON: what the route needs to answer again
    answered_at real not null
);
"""


def answers_version(release_version: str | None, index_version: tuple) -> str:
    """An opaque id for what Vision answers from now: the release, plus the served index's
    version (reference photos, calibration run, photographers' answers, the nightly
    layer). Equal ids give equal answers for equal photos and options."""
    raw = json.dumps([release_version, [str(v) for v in index_version]])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def answer_key(kind: str, source: str, number: str, backbone: str, method: str,
               options: dict) -> str:
    opts = ",".join(f"{k}={options[k]}" for k in sorted(options))
    return f"{kind}:{source}:{number}:{backbone}/{method}:{opts}"


class AnswerCache:
    """Thread-safe; `path=None` keeps the rows in memory (tests, a laptop without state)."""

    def __init__(self, path: Path | None, clock=time.time):
        self.clock = clock
        self._lock = threading.Lock()
        if path is None:
            self._conn = sqlite3.connect(":memory:", check_same_thread=False)
        else:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
            self._conn.execute("pragma journal_mode = wal")
        with self._lock:
            self._conn.executescript(SCHEMA)

    def get(self, key: str, version: str) -> dict | None:
        """The answer kept for `key` if it was given under `version`, else None."""
        with self._lock:
            row = self._conn.execute(
                "select version, payload from observation_answers where answer_key = ?",
                (key,)).fetchone()
        if row is None or row[0] != version:
            return None
        try:
            return json.loads(row[1])
        except ValueError:
            return None

    def put(self, key: str, version: str, payload: dict) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=_plain)
        with self._lock:
            self._conn.execute(
                "insert into observation_answers (answer_key, version, payload, answered_at) "
                "values (?, ?, ?, ?) on conflict(answer_key) do update set "
                "version = excluded.version, payload = excluded.payload, "
                "answered_at = excluded.answered_at",
                (key, version, body, self.clock()))
            self._conn.commit()

    def drop_older_than(self, version: str) -> int:
        """Removes rows of any other version (housekeeping; they are never served)."""
        with self._lock:
            n = self._conn.execute("delete from observation_answers where version != ?",
                                   (version,)).rowcount
            self._conn.commit()
        return n


def without_confusions(result: dict) -> dict:
    """A copy of an answer without its "often confused with" notes: they follow
    mycomap.org's table, which moves on its own, so they are added afresh when served."""
    out = json.loads(json.dumps(result, default=_plain))
    lists = [(out.get("ranks") or {}).get("species") or [],
             ((out.get("likely") or {}).get("species") or {}).get("names") or []]
    lists += [((p or {}).get("ranks") or {}).get("species") or [] for p in out.get("per_photo") or []]
    for candidates in lists:
        for c in candidates:
            if isinstance(c, dict):
                c.pop("confused_with", None)
    return out


def _plain(value):
    """numpy scalars and arrays in an answer, as JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")
