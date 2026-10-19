"""Answers already given for an iNat or MO observation, kept so the same observation is
not fetched and scored again: Vision runs an observation once, and every later ask (the
site's Identify page, the v1 API, mycomap.org's Validate Records) gets that answer back
until someone asks for a fresh one (`rerun`, the "Run again" button). Steve, 2026-10-19:
never rerun automatically, not for a night's new references and not for a new release.

One row per (observation, model, options). Each row says when it was answered, who asked
(`asked_by`: the site, mycomap.org, the app or a developer key) and the answers version it
was answered under (`answers_version`: the served release and its reference set, which
moves when a release, a night's additions or removals, a new calibration or a
photographer's answer changes the reference). A row under an older version is still
served, saying so (`current` false), and the page offers to run it again. The
observation's own photos are not checked again: a photo added at the source after the
answer is not seen until someone runs it again.

Durable on the box beside vision-usage.sqlite; observation answers hold no one's
identity, only public observations' scores.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

SCHEMA = """
create table if not exists observation_answers (
    answer_key text primary key,      -- kind:source:id:backbone/method:options
    version    text not null,         -- answers_version when it was answered
    payload    text not null,         -- JSON: what the route needs to answer again
    answered_at real not null,
    asked_by   text                   -- site | mycomap.org | app | api (null: before 10/19)
);
"""

ASKERS = ("site", "mycomap.org", "app", "api")


@dataclass(frozen=True)
class Kept:
    """A kept answer and when, by whom and under which answers version it was given."""
    payload: dict
    version: str
    answered_at: float
    asked_by: str | None

    def about(self, current_version: str | None) -> dict:
        """The run's details, as answers carry them (`answered`): when (UTC, ISO 8601),
        who asked, and whether Vision still answers from what it was answered from."""
        return {"at": iso_utc(self.answered_at), "by": self.asked_by,
                "answers_version": self.version,
                "current": current_version is not None and self.version == current_version}


def iso_utc(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z")


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
            cols = {r[1] for r in self._conn.execute("pragma table_info(observation_answers)")}
            if "asked_by" not in cols:                  # a store made before 2026-10-19
                self._conn.execute("alter table observation_answers add column asked_by text")
                self._conn.commit()

    def held(self, key: str) -> Kept | None:
        """The answer kept for `key`, under whichever version it was given; None if none."""
        with self._lock:
            row = self._conn.execute(
                "select version, payload, answered_at, asked_by from observation_answers "
                "where answer_key = ?", (key,)).fetchone()
        if row is None:
            return None
        try:
            payload = json.loads(row[1])
        except ValueError:
            return None
        return Kept(payload, row[0], float(row[2]), row[3])

    def get(self, key: str, version: str) -> dict | None:
        """The answer kept for `key` if it was given under `version`, else None."""
        kept = self.held(key)
        return kept.payload if kept is not None and kept.version == version else None

    def put(self, key: str, version: str, payload: dict, asked_by: str | None = None) -> None:
        body = json.dumps(payload, separators=(",", ":"), default=_plain)
        with self._lock:
            self._conn.execute(
                "insert into observation_answers (answer_key, version, payload, answered_at, "
                "asked_by) values (?, ?, ?, ?, ?) on conflict(answer_key) do update set "
                "version = excluded.version, payload = excluded.payload, "
                "answered_at = excluded.answered_at, asked_by = excluded.asked_by",
                (key, version, body, self.clock(), asked_by))
            self._conn.commit()

    def drop_older_than(self, version: str) -> int:
        """Removes rows of any other version. Nothing calls it on the box: older answers
        are served until someone runs them again (Steve, 2026-10-19)."""
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


def specimens_in(answer) -> list[dict]:
    """Every reference specimen an answer shows (identify.Identifier._specimens dicts),
    wherever it sits: the answer's own list, per photo, under a candidate, or spare."""
    found: list[dict] = []

    def walk(node):
        if isinstance(node, dict):
            if "observation_id" in node and "photo_withheld" in node:
                found.append(node)
                return
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(answer)
    return found


def reshow_photos(answer, photos_of, photo_lookup, may_show, url_of) -> None:
    """Shows each specimen's photo only if it may be shown now (CLAUDE.md: a photo whose
    owner declined or withdrew since is never shown), in place. A kept answer was given
    under the permissions of its day; an older one is checked again before it is served.

    `photos_of(observation_id)`: the reference record's photo ids in the index served now
    (empty when the record has left it); `photo_lookup(ids)`: {id: PhotoInfo};
    `may_show(info)`: the showing rule now; `url_of(info)`: the URL the page shows. The
    photo shown before stays if it still may be shown, else another of the record's
    photos that may, else none."""
    for s in specimens_in(answer):
        ids = list(photos_of(str(s.get("observation_id") or "")))
        infos = photo_lookup(ids) if ids else {}
        allowed = [infos[i] for i in ids if infos.get(i) is not None and may_show(infos[i])]
        urls = [url_of(i) for i in allowed]
        if s.get("photo_url") in urls:
            continue
        pick = allowed[0] if allowed else None
        s["photo_url"] = urls[0] if pick is not None else None
        s["photo_owner"] = pick.owner_login if pick is not None else None
        s["photo_withheld"] = pick is None


def _plain(value):
    """numpy scalars and arrays in an answer, as JSON."""
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"{type(value).__name__} is not JSON")
