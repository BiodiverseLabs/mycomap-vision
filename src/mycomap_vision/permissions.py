"""Photographers' answers from mycomap.org: which all-rights-reserved photos may be shown or used.

MycoMap asks the owners of all-rights-reserved (ARR) photos for permission on
mycomap.org; the answers live there. This module keeps a copy in the manifest,
pulled from mycomap.org's GET /api/vision/photo-permissions (read-only, with a
service key), and applies two rules:

- Showing (an example match, a credit): a CC-licensed photo follows its licence;
  an ARR photo needs its owner's grant, read recently (MV_PERMISSIONS_MAX_AGE_SECONDS,
  default one hour). When the answers can't be refreshed, ARR photos stop showing.
- Using (reference set, comparisons, training): an ARR photo whose owner said no
  is left out (evaluate.load_records). Owners who haven't answered stay in while
  permission is being sought (Steve, 2026-09-28).

Only a well-formed answer replaces the copy. A failed, malformed or suspicious one
changes nothing: the last good list stays, withdrawals included, and the failure is
recorded and logged. mycomap.org keeps one answer per person and never deletes one
(a person's status only moves between granted and withdrawn), and its answer has no
count or version to say "the list really is shorter"; so an answer that leaves out
anyone the last good list has, an empty one included, is refused as suspicious. If
people really were removed there, `mv permissions --sync --accept-shrink` takes it.

The last good list is also saved outside the manifest (state_path(): on the server
box <MV_RELEASE_ROOT>/state/photo-permissions.json), because a new release brings its
own manifest; the server puts the newer of the two back at startup, so a withdrawal
holds across restarts and releases even if mycomap.org can't be reached then.

Settings: MV_ORG_BASE_URL (e.g. https://mycomap.org) and MV_ORG_VISION_KEY or
MV_ORG_VISION_KEY_FILE (the key set as VISION_API_KEY on mycomap.org). Never logged.
MV_PERMISSIONS_STATE overrides where the last good list is saved.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable

import requests

from . import config

SCHEMA = """
-- Photographers' answers on mycomap.org: the latest snapshot of
-- GET /api/vision/photo-permissions, replaced whole on every successful sync.
create table if not exists photo_permissions (
  inat_user_id     integer primary key,  -- photos.owner_user_id
  inat_login       text,
  status           text not null,        -- 'granted' | 'withdrawn'
  terms_version    text,
  granted_at       text,
  withdrawn_at     text,
  updated_at       text
);

-- Every sync attempt, so staleness and failures are visible.
create table if not exists permission_syncs (
  id               integer primary key autoincrement,
  attempted_at     text not null,
  ok               integer not null,
  people           integer,
  generated_at     text,                 -- mycomap.org's clock when it answered
  error            text
);
"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """For connections not opened through manifest.connect (the API, older copies)."""
    have = conn.execute("select count(*) from sqlite_master where type = 'table' and "
                        "name in ('photo_permissions', 'permission_syncs')").fetchone()[0]
    if have < 2:
        conn.executescript(SCHEMA)


log = logging.getLogger("mycomap_vision.permissions")

PATH = "/api/vision/photo-permissions"
STATUSES = ("granted", "withdrawn")
DEFAULT_MAX_AGE = timedelta(hours=1)


class PermissionSyncError(RuntimeError):
    pass


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def iso(t: datetime) -> str:
    return t.astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_time(value: str | None) -> datetime | None:
    if not value:
        return None
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def org_key() -> str | None:
    """The key mycomap.org knows as VISION_API_KEY: MV_ORG_VISION_KEY, or the file
    MV_ORG_VISION_KEY_FILE names (the server box keeps secrets in their own files
    under /etc/mycomap-vision, not in vision.env)."""
    key = config.setting("MV_ORG_VISION_KEY")
    if key:
        return key
    path = config.setting("MV_ORG_VISION_KEY_FILE")
    if not path:
        return None
    try:
        return open(path, encoding="utf-8").read().strip() or None
    except OSError:
        return None


def max_age() -> timedelta:
    raw = config.setting("MV_PERMISSIONS_MAX_AGE_SECONDS")
    return timedelta(seconds=int(raw)) if raw and raw.isdigit() else DEFAULT_MAX_AGE


# ---------------------------------------------------------------------------
# Pulling the answers

def fetch(base_url: str, key: str, session: requests.Session | None = None,
          timeout: float = 30) -> dict:
    """One read of mycomap.org's answers. Raises PermissionSyncError with a reason
    that never includes the key."""
    session = session or requests.Session()
    url = base_url.rstrip("/") + PATH
    try:
        resp = session.get(url, headers={"Authorization": f"Bearer {key}",
                                         "User-Agent": config.USER_AGENT}, timeout=timeout)
    except requests.RequestException as e:
        raise PermissionSyncError(f"could not reach {base_url}: {type(e).__name__}") from None
    if resp.status_code == 401:
        raise PermissionSyncError("mycomap.org refused the key (401): check MV_ORG_VISION_KEY")
    if resp.status_code == 503:
        raise PermissionSyncError("mycomap.org has no VISION_API_KEY set (503)")
    if resp.status_code != 200:
        raise PermissionSyncError(f"mycomap.org answered {resp.status_code}")
    try:
        return resp.json()
    except ValueError:
        raise PermissionSyncError("mycomap.org's answer was not JSON") from None


def clean_rows(payload: dict) -> list[tuple]:
    """Validate the answer into rows for photo_permissions. A malformed answer is
    refused whole rather than half-applied: it must carry mycomap.org's clock
    (generatedAt) and a list of well-formed rows."""
    rows = payload.get("permissions") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise PermissionSyncError("answer has no permissions list")
    if not isinstance(payload.get("generatedAt"), str) or not payload["generatedAt"].strip():
        raise PermissionSyncError("answer has no generatedAt: not the photo-permissions answer")
    out, seen = [], set()
    for p in rows:
        uid = p.get("inatUserId") if isinstance(p, dict) else None
        status = p.get("status") if isinstance(p, dict) else None
        if not isinstance(uid, int) or uid <= 0 or status not in STATUSES or uid in seen:
            raise PermissionSyncError(f"malformed permission row: {p!r}"[:200])
        seen.add(uid)
        out.append((uid, p.get("inatLogin"), status, p.get("termsVersion"), p.get("grantedAt"),
                    p.get("withdrawnAt"), p.get("updatedAt")))
    return out


def save_snapshot(conn: sqlite3.Connection, payload: dict, attempted_at: datetime,
                  accept_shrink: bool = False) -> dict:
    """Replace the copy with a well-formed answer. Refuses (PermissionSyncError) one that
    leaves out anyone the copy has, unless `accept_shrink` (see the module notes)."""
    rows = clean_rows(payload)
    had = {int(u) for (u,) in conn.execute("select inat_user_id from photo_permissions")}
    left_out = had - {r[0] for r in rows}
    if left_out and not accept_shrink:
        withdrawn = sum(1 for (u,) in conn.execute(
            "select inat_user_id from photo_permissions where status = 'withdrawn'")
            if int(u) in left_out)
        raise PermissionSyncError(
            f"REFUSED a suspicious answer: it leaves out {len(left_out)} of the {len(had)} people "
            f"in the last good list ({withdrawn} of them withdrawn)"
            + (", it is empty" if not rows else "")
            + ". mycomap.org never deletes an answer, so the last good list is kept and its "
            "withdrawals still apply. If people really were removed there, run "
            "`mv permissions --sync --accept-shrink`.")
    with conn:
        conn.execute("delete from photo_permissions")
        conn.executemany("insert into photo_permissions values (?, ?, ?, ?, ?, ?, ?)", rows)
        conn.execute("insert into permission_syncs (attempted_at, ok, people, generated_at) "
                     "values (?, 1, ?, ?)", (iso(attempted_at), len(rows),
                                             payload.get("generatedAt")))
    granted = sum(1 for r in rows if r[2] == "granted")
    return {"people": len(rows), "granted": granted, "withdrawn": len(rows) - granted}


def record_failure(conn: sqlite3.Connection, attempted_at: datetime, error: str) -> None:
    with conn:
        conn.execute("insert into permission_syncs (attempted_at, ok, error) values (?, 0, ?)",
                      (iso(attempted_at), error[:500]))


_STATE_DEFAULT = object()


def state_path() -> Path | None:
    """Where the last good list is saved outside the manifest: MV_PERMISSIONS_STATE, or on
    the server box <MV_RELEASE_ROOT>/state/photo-permissions.json. None elsewhere (a
    laptop keeps one manifest, which is copy enough)."""
    explicit = config.setting("MV_PERMISSIONS_STATE")
    if explicit:
        return Path(explicit)
    return config.RELEASE_ROOT / "state" / "photo-permissions.json" if config.RELEASE_ROOT else None


def save_state(path: Path, payload: dict, attempted_at: datetime) -> None:
    """Write the last good answer atomically (a crash leaves the previous file whole)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"saved_at": iso(attempted_at), "answer": payload}), encoding="utf-8")
    os.replace(tmp, path)


def restore_last_good(conn: sqlite3.Connection, path=_STATE_DEFAULT,
                      note: Callable[[str], None] = print) -> bool:
    """At startup: when the saved last good list is newer than the manifest's copy (a new
    release brings an older one), put it back, with the time it was read. True if it was."""
    path = state_path() if path is _STATE_DEFAULT else path
    if not path or not Path(path).is_file():
        return False
    try:
        saved = json.loads(Path(path).read_text(encoding="utf-8"))
        at = parse_time(saved["saved_at"])
        clean_rows(saved["answer"])
    except (OSError, ValueError, KeyError, TypeError, PermissionSyncError) as e:
        note(f"[permissions] the saved last good list ({path}) is unreadable, not used: "
             f"{type(e).__name__}: {e}")
        return False
    here = last_good_sync(conn)
    if here is not None and here >= at:
        return False
    # The saved list is the newest good one this deployment has read, so it replaces
    # an older copy even where that copy has more people.
    stats = save_snapshot(conn, saved["answer"], at, accept_shrink=True)
    note(f"[permissions] restored the last good list read {iso(at)} ({stats['granted']} granted, "
         f"{stats['withdrawn']} withdrawn): newer than this manifest's copy")
    return True


def sync(conn: sqlite3.Connection, base_url: str | None = None, key: str | None = None,
         fetcher: Callable[[str, str], dict] = fetch, now: Callable[[], datetime] = now_utc,
         accept_shrink: bool = False, state=_STATE_DEFAULT) -> dict:
    """Pull the answers and replace the local copy. On failure (unreachable, refused,
    malformed, suspicious) the previous copy is kept (it ages out for showing, while its
    withdrawals still apply), and the failure is recorded and logged. A good answer is
    also saved to `state` (default: state_path())."""
    base_url = base_url or config.required("MV_ORG_BASE_URL", "mycomap.org's address, "
                                           "e.g. https://mycomap.org")
    key = key or org_key()
    if not key:
        raise RuntimeError("Set MV_ORG_VISION_KEY or MV_ORG_VISION_KEY_FILE (the key set as "
                           "VISION_API_KEY on mycomap.org); see .env.example.")
    attempted = now()
    try:
        payload = fetcher(base_url, key)
        stats = save_snapshot(conn, payload, attempted, accept_shrink=accept_shrink)
    except PermissionSyncError as e:
        record_failure(conn, attempted, str(e))
        log.error("photo permissions not updated, the last good list is kept: %s", e)
        raise
    path = state_path() if state is _STATE_DEFAULT else state
    if path:
        save_state(Path(path), payload, attempted)
    return stats


def last_good_sync(conn: sqlite3.Connection) -> datetime | None:
    row = conn.execute("select max(attempted_at) from permission_syncs where ok = 1").fetchone()
    return parse_time(row[0]) if row and row[0] else None


def status_report(conn: sqlite3.Connection) -> dict:
    counts = dict(conn.execute("select status, count(*) from photo_permissions group by 1"))
    last = conn.execute("select attempted_at, ok, error from permission_syncs "
                        "order by id desc limit 1").fetchone()
    good = last_good_sync(conn)
    return {"granted": counts.get("granted", 0), "withdrawn": counts.get("withdrawn", 0),
            "last_good_sync": iso(good) if good else None,
            "last_attempt": ({"attempted_at": last[0], "ok": bool(last[1]), "error": last[2]}
                             if last else None),
            "fresh_for_showing": bool(good and now_utc() - good <= max_age())}


# ---------------------------------------------------------------------------
# The rules

@dataclass
class PermissionView:
    """The answers as of the last good sync, for deciding what to show."""
    granted: frozenset[int] = frozenset()
    withdrawn: frozenset[int] = frozenset()
    synced_at: datetime | None = None
    max_age: timedelta = field(default=DEFAULT_MAX_AGE)

    @classmethod
    def load(cls, conn: sqlite3.Connection, max_age_: timedelta | None = None) -> "PermissionView":
        g, w = set(), set()
        for uid, status in conn.execute("select inat_user_id, status from photo_permissions"):
            (g if status == "granted" else w).add(int(uid))
        return cls(frozenset(g), frozenset(w), last_good_sync(conn),
                   max_age_ if max_age_ is not None else max_age())

    def fresh(self, at: datetime | None = None) -> bool:
        return self.synced_at is not None and (at or now_utc()) - self.synced_at <= self.max_age

    def may_show(self, license_class: str | None, owner_user_id: int | None,
                 at: datetime | None = None) -> bool:
        """A CC photo follows its licence. An all-rights-reserved photo (or one whose
        licence we don't know) needs a grant from a recent sync."""
        if license_class in ("open", "nc"):
            return True
        return owner_user_id is not None and int(owner_user_id) in self.granted and self.fresh(at)

    def fingerprint(self) -> str:
        """Changes whenever the set of people who said no changes (the reference set
        must then be rebuilt)."""
        h = hashlib.sha1(",".join(str(u) for u in sorted(self.withdrawn)).encode())
        return h.hexdigest()[:12]


# Used by evaluate.load_records: photos left out of the reference set, comparisons
# and training. Kept here so the rule lives next to the showing rule.
EXCLUDED_FROM_USE_SQL = """
  not exists (
    select 1 from photos xp join photo_permissions xpp on xpp.inat_user_id = xp.owner_user_id
    where xp.photo_id = op.photo_id and xp.license_class = 'arr' and xpp.status = 'withdrawn'
  )
"""
