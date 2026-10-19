"""Your collections (Steve, 2026-10-18): a signed-in person's own DNA-verified records among
the references, shown to that person only, under the ID area of an Identify result.

mycomap.org says who the person is on iNaturalist and Mushroom Observer
(GET /api/vision/linked-accounts?user=<sub>, verified connections only); Vision matches its
reference records on the owner each source reports (`inat_observations.user_id`,
`mo_observations.owner_id`, a login only where an account has no id). The collector field
on mycomap.org is a display name on most iNat rows, so it is never used.

- The person is the `sub` in our own signed session cookie. Nothing here takes a user id
  from a browser, and an answer from mycomap.org for another `sub` is refused.
- Their accounts are kept in memory for TTL_SECONDS per `sub` (a failed read for
  FAILED_TTL_SECONDS), never on disk and never logged.
- Their records are scored against the find's photo vectors directly (a few hundred
  columns at most for nearly everyone), with the same rule as the closest-specimens list:
  for each photo, its best match among the record's photos, averaged over the photos.
- The closest-specimens list ("from other collectors") then leaves their records out and
  still shows SHOWN: every site answer is scored with POOL specimens and trimmed here.
- Kept answers (answercache) are shared between people, so they hold only the
  unpersonalised answer, its spare specimens and the photo vectors; this module makes the
  personal part per request from those.
"""

from __future__ import annotations

import base64
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

import numpy as np
import requests

from . import config
from .sources import INAT, MO

log = logging.getLogger("mycomap_vision.yours")

PATH = "/api/vision/linked-accounts"
SHOWN = 6                    # specimens in each list on the result card
POOL = 2 * SHOWN             # scored, so the list without yours can still show SHOWN
TTL_SECONDS = 600
FAILED_TTL_SECONDS = 60
MAX_PEOPLE = 2000            # people whose accounts are kept at once


class LinkedError(RuntimeError):
    """mycomap.org's answer can't be used; the reason never holds the key or the user."""


@dataclass(frozen=True)
class Linked:
    """One person's verified accounts on iNaturalist and Mushroom Observer."""
    inat_ids: frozenset = frozenset()
    inat_logins: frozenset = frozenset()     # lower case, only for accounts without an id
    mo_ids: frozenset = frozenset()
    mo_logins: frozenset = frozenset()
    unverified_inat: bool = False
    unverified_mo: bool = False

    @property
    def linked(self) -> bool:
        return bool(self.inat_ids or self.inat_logins or self.mo_ids or self.mo_logins)


def _accounts(rows) -> tuple[frozenset, frozenset]:
    if not isinstance(rows, list):
        raise LinkedError("accounts are not a list")
    ids, logins = set(), set()
    for a in rows:
        if not isinstance(a, dict):
            raise LinkedError("an account is not an object")
        uid, login = a.get("user_id"), a.get("login")
        if isinstance(uid, int) and not isinstance(uid, bool) and uid > 0:
            ids.add(uid)
        elif isinstance(login, str) and login.strip():
            logins.add(login.strip().lower())
    return frozenset(ids), frozenset(logins)


def parse_linked(payload, sub: str) -> Linked:
    """mycomap.org's answer for `sub`, checked. An answer about anyone else is refused."""
    if not isinstance(payload, dict):
        raise LinkedError("not a JSON object")
    if payload.get("user") != sub:
        raise LinkedError("the answer is for another user")
    inat_ids, inat_logins = _accounts(payload.get("inat", []))
    mo_ids, mo_logins = _accounts(payload.get("mo", []))
    unverified = payload.get("unverified") or {}
    return Linked(inat_ids, inat_logins, mo_ids, mo_logins,
                  bool(unverified.get("inat")), bool(unverified.get("mo")))


def fetch_linked(base_url: str, key: str, sub: str, session: requests.Session | None = None,
                 timeout: float = 5) -> Linked:
    """One read from mycomap.org. The endpoint's own 404 (`not_found`: no such active user)
    is a person with no accounts; any other 404 (the route not deployed yet) is a failure,
    so nobody is told to connect an account they have."""
    session = session or requests.Session()
    try:
        r = session.get(base_url.rstrip("/") + PATH, params={"user": sub}, timeout=timeout,
                        headers={"Authorization": f"Bearer {key}",
                                 "User-Agent": config.USER_AGENT})
    except requests.RequestException as e:
        raise LinkedError(f"unreachable: {type(e).__name__}") from None
    if r.status_code == 404:
        try:
            code = (r.json() or {}).get("code")
        except (ValueError, AttributeError):
            code = None
        if code == "not_found":
            return Linked()
        raise LinkedError("HTTP 404 (no linked-accounts route)")
    if r.status_code != 200:
        raise LinkedError(f"HTTP {r.status_code}")
    try:
        payload = r.json()
    except ValueError:
        raise LinkedError("not JSON") from None
    return parse_linked(payload, sub)


class LinkedCache:
    """Each person's accounts, kept in memory a short while. `get` gives None when they
    can't be read (then the page shows no "Your collections" and the usual list)."""

    def __init__(self, fetcher: Callable[[str], Linked], clock=time.monotonic,
                 ttl: float = TTL_SECONDS, failed_ttl: float = FAILED_TTL_SECONDS,
                 max_people: int = MAX_PEOPLE):
        self.fetcher, self.clock = fetcher, clock
        self.ttl, self.failed_ttl, self.max_people = ttl, failed_ttl, max_people
        self._lock = threading.Lock()
        self._kept: OrderedDict[str, tuple[float, Linked | None]] = OrderedDict()

    @classmethod
    def from_settings(cls, session: requests.Session | None = None) -> "LinkedCache | None":
        from .permissions import org_key
        base_url, key = config.setting("MV_ORG_BASE_URL"), org_key()
        if not base_url or not key:
            return None
        return cls(lambda sub: fetch_linked(base_url, key, sub, session=session))

    def get(self, sub: str) -> Linked | None:
        now = self.clock()
        with self._lock:
            kept = self._kept.get(sub)
            if kept is not None and kept[0] > now:
                return kept[1]
        try:
            linked, ttl = self.fetcher(sub), self.ttl
        except LinkedError as e:
            log.warning("linked accounts not read: %s", e)
            linked, ttl = None, self.failed_ttl
        with self._lock:
            self._kept[sub] = (now + ttl, linked)
            self._kept.move_to_end(sub)
            while len(self._kept) > self.max_people:
                self._kept.popitem(last=False)
        return linked


class OwnerIndex:
    """Reference records (by their index in the served Specimens) grouped by the owner their
    source reports: the iNat user, the MO user."""

    def __init__(self, by_id: dict, by_login: dict):
        self.by_id, self.by_login = by_id, by_login

    @classmethod
    def from_manifest(cls, conn, record_keys: np.ndarray, record_sources: np.ndarray) -> "OwnerIndex":
        at = {str(k): i for i, k in enumerate(record_keys.tolist())}
        by_id: dict = {}
        by_login: dict = {}
        queries = [(INAT, "select observation_id, user_id, user_login from inat_observations"),
                   (MO, "select observation_id, owner_id, owner_login from mo_observations")]
        for source, sql in queries:
            try:
                rows = conn.execute(sql).fetchall()
            except Exception:                 # an older or cut-down manifest: no owners there
                continue
            for key, uid, login in rows:
                i = at.get(str(key))
                if i is None or record_sources[i] != source:
                    continue
                if uid is not None:
                    by_id.setdefault((source, int(uid)), []).append(i)
                if login:
                    by_login.setdefault((source, str(login).strip().lower()), []).append(i)
        pack = lambda d: {k: np.asarray(sorted(set(v)), dtype=np.int64) for k, v in d.items()}
        return cls(pack(by_id), pack(by_login))

    def records(self, linked: Linked) -> np.ndarray:
        """The person's reference records: by account id, else by login."""
        parts = [self.by_id.get((INAT, i)) for i in linked.inat_ids]
        parts += [self.by_login.get((INAT, n)) for n in linked.inat_logins]
        parts += [self.by_id.get((MO, i)) for i in linked.mo_ids]
        parts += [self.by_login.get((MO, n)) for n in linked.mo_logins]
        parts = [p for p in parts if p is not None]
        return np.unique(np.concatenate(parts)) if parts else np.empty(0, dtype=np.int64)


# --- kept answers --------------------------------------------------------------------

def encode_vectors(vecs: np.ndarray) -> dict:
    v = np.ascontiguousarray(vecs, dtype=np.float32)
    return {"shape": list(v.shape), "b64": base64.b64encode(v.tobytes()).decode("ascii")}


def decode_vectors(doc) -> np.ndarray | None:
    try:
        v = np.frombuffer(base64.b64decode(doc["b64"]), dtype=np.float32)
        return v.reshape(doc["shape"])
    except (TypeError, KeyError, ValueError):
        return None


def split_pool(result: dict) -> tuple[dict, list]:
    """(the answer with SHOWN specimens, the spare ones after them)."""
    pool = result.get("specimens") or []
    return {**result, "specimens": pool[:SHOWN]}, pool[SHOWN:]


# --- the personal part, made per request ----------------------------------------------

def not_a_fungus(result: dict) -> bool:
    gate = result.get("gate") or {}
    return gate.get("verdict") == "not_fungus"


def personalize(result: dict, spare: list, linked: Linked | None, own_keys: set,
                own_records: int, own_specimens: list) -> dict:
    """The answer one person sees. Nobody signed in (or their accounts unreadable, or a find
    that isn't a fungus): the plain answer, SHOWN specimens, no `yours`. Otherwise `yours`
    (their records, most alike first) and the closest specimens without their records,
    still SHOWN long; a per-photo match that is theirs is marked `yours`. Never changes
    `result` or `spare` (they may be a kept answer)."""
    if linked is None or not_a_fungus(result):
        return {**result, "specimens": (result.get("specimens") or [])[:SHOWN]}
    pool = [*(result.get("specimens") or []), *spare]
    others = [s for s in pool if s.get("observation_id") not in own_keys][:SHOWN]
    per_photo = [{**p, "specimens": [({**s, "yours": True} if s.get("observation_id") in own_keys
                                      else s) for s in p.get("specimens") or []]}
                 for p in result.get("per_photo") or []]
    return {**result, "specimens": others, "per_photo": per_photo,
            "yours": {"linked": linked.linked, "records": int(own_records),
                      "specimens": own_specimens[:SHOWN]}}


def for_person(ident, result: dict, spare: list, vecs: np.ndarray | None,
               hidden: np.ndarray | None, linked: Linked | None,
               photo_lookup=None, may_show=None) -> dict:
    """personalize() with the person's records found and scored in `ident`."""
    if linked is None or not_a_fungus(result):
        return personalize(result, spare, None, set(), 0, [])
    if not linked.linked:                    # signed in, nothing connected: say how
        return personalize(result, spare, linked, set(), 0, [])
    if vecs is None or ident is None:        # nothing to score with: no panel
        return personalize(result, spare, None, set(), 0, [])
    owners = getattr(ident, "owners", None)
    recs = owners.records(linked) if owners is not None else np.empty(0, np.int64)
    if not len(recs):
        return personalize(result, spare, linked, set(), 0, [])
    keys = {str(ident.col_record.obs[r]) for r in recs.tolist()}
    # The photos the answer used: a photo the fungus gate left out is left out here too.
    left = set((result.get("gate") or {}).get("left_out") or [])
    kept = [i for i in range(len(vecs)) if i not in left]
    mine = ident.own_specimens(np.asarray(vecs)[kept], recs, hidden, SHOWN,
                               photo_lookup=photo_lookup, may_show=may_show)
    for s in mine:
        s["matched_query_photo"] = kept[s["matched_query_photo"]]
    return personalize(result, spare, linked, keys, len(recs), mine)
