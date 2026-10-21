"""The names behind a Human Feedback name item, for the reviewer to judge it (Steve,
2026-10-21): each name the item is about, how iNaturalist holds it (its taxon, linked; active
or not; the current name an inactive one points to; the same name written another way), and
MycoMap's own synonym list around it (mycomap.org, GET /api/vision/feedback/synonyms).

"Names iNaturalist no longer uses" listed names whose iNat taxon is inactive with no current
name at all (Callistosporium luteoolivaceum: an unused inactive taxon of that spelling, while
iNat's active one is spelled luteo-olivaceum and MycoMap's synonym list retires the hyphen).
Without both sides shown, an item like that reads as a name renamed to itself.

iNat is asked at most once a second (CLAUDE.md), one autocomplete per name plus one for an
inactive name's current taxa and one for a near spelling; answers are kept a day in memory
and a failure not at all. A request asks iNat for a few names only (NAME_BUDGET); a name
past it is linked to iNat's taxon search instead, and asked when the item is opened again.
"""

from __future__ import annotations

import re
import threading
import time
from typing import Callable
from urllib.parse import quote

import requests

from . import config
from .ratelimit import MinInterval

AUTOCOMPLETE = "https://api.inaturalist.org/v1/taxa/autocomplete"
TAXA = "https://api.inaturalist.org/v1/taxa/"
TAXON_PAGE = "https://www.inaturalist.org/taxa/{id}"
SEARCH_PAGE = "https://www.inaturalist.org/taxa/search?q={q}"
FUNGI = 47170
TTL = 24 * 3600.0
NAME_BUDGET = 4             # names asked of iNat per request beyond the cached ones
MAX_SYNONYM_ROWS = 40
NAME_TYPES = ("name_retired", "name_inactive_on_inat")

# iNat writes an infraspecific name without its marker ('Amanita muscaria guessowii').
_MARKERS = re.compile(r"\s+(?:var\.|subsp\.|ssp\.|f\.|forma|subvar\.)\s+")
_FORMAL = re.compile(r"^[A-Z][a-z]+(?:-[a-z]+)?(?: [a-z]+(?:-[a-z]+)*)+\Z")


def inat_form(name: str) -> str:
    return _MARKERS.sub(" ", " ".join((name or "").split()))


def loose(name: str) -> str:
    """One key for writings of one name: letters and digits only ('luteo-olivaceum' and
    'luteoolivaceum' agree; so do "sp-IN36" and sp. 'IN36')."""
    return re.sub(r"[^a-z0-9]", "", inat_form(name).lower())


def exact(name: str) -> str:
    """The name as written, spaces folded: two writings stay two names."""
    return " ".join((name or "").split())


def search_url(name: str) -> str:
    return SEARCH_PAGE.format(q=quote(inat_form(name)))


def taxon_view(t: dict) -> dict:
    return {"id": int(t["id"]), "name": t.get("name"), "rank": t.get("rank"),
            "active": t.get("is_active") is not False,
            "observations": t.get("observations_count"),
            "changes": t.get("taxon_changes_count"),
            "url": TAXON_PAGE.format(id=int(t["id"]))}


def in_fungi(t: dict) -> bool:
    return int(t.get("id") or 0) == FUNGI or FUNGI in (t.get("ancestor_ids") or [])


def not_asked(name: str) -> dict:
    return {"status": "not_asked", "asked": inat_form(name), "taxon": None, "current": [],
            "near": [], "search_url": search_url(name)}


def describe(answer: dict) -> str:
    """One line on how iNat holds the name, for the reviewer."""
    s = answer.get("status")
    if s == "active":
        return "iNaturalist uses this name."
    if s == "inactive":
        cur = answer.get("current") or []
        if len(cur) == 1:
            also = " (itself inactive on iNat)" if not cur[0].get("active", True) else ""
            return f"Inactive on iNaturalist; its current name is {cur[0]['name']}{also}."
        if cur:
            return ("Inactive on iNaturalist; it was split into "
                    + ", ".join(c["name"] for c in cur) + ".")
        t = answer.get("taxon") or {}
        unused = not t.get("observations") and not t.get("changes")
        return ("Inactive on iNaturalist with no current name"
                + (" (a taxon iNat never used: no observations, no taxon change)" if unused
                   else "") + ".")
    if s == "not_found":
        return "iNaturalist has no fungal taxon of this name."
    if s == "unavailable":
        return "iNaturalist did not answer just now."
    return "Not looked up yet."


class InatNames:
    """How iNat holds a name. `get(url, params)` returns the parsed JSON (raises on a
    failure); injected by the tests."""

    def __init__(self, get: Callable[[str, dict], dict] | None = None,
                 clock: Callable[[], float] = time.monotonic, interval: float = 1.0):
        self.get = get or self._get
        self.clock = clock
        self._seen: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()
        self._pace = MinInterval(interval)

    @staticmethod
    def _get(url: str, params: dict) -> dict:
        r = requests.get(url, params=params, timeout=10,
                         headers={"User-Agent": config.USER_AGENT})
        r.raise_for_status()
        return r.json()

    def _ask(self, url: str, params: dict) -> list[dict]:
        self._pace.wait()
        return list(self.get(url, params).get("results") or [])

    def cached(self, name: str) -> dict | None:
        with self._lock:
            hit = self._seen.get(exact(name))
        return hit[1] if hit and self.clock() - hit[0] < TTL else None

    def one(self, name: str) -> dict:
        """{"status": active|inactive|not_found|unavailable, "asked", "taxon", "current",
        "near", "search_url"}: the taxon of exactly this name (an active one first), the
        current taxa an inactive one points to, and active taxa of the same name written
        another way (a hyphen, a marker) when no active taxon has this writing."""
        hit = self.cached(name)
        if hit is not None:
            return hit
        asked = inat_form(name)
        out = {"status": "not_found", "asked": asked, "taxon": None, "current": [],
               "near": [], "search_url": search_url(name)}
        try:
            found = [t for t in self._ask(AUTOCOMPLETE, {"q": asked, "is_active": "any",
                                                         "per_page": 30})
                     if str(t.get("name", "")).lower() == asked.lower() and in_fungi(t)]
            found.sort(key=lambda t: t.get("is_active") is False)      # an active one first
            if found:
                t = found[0]
                out["taxon"] = taxon_view(t)
                out["status"] = "active" if out["taxon"]["active"] else "inactive"
                ids = [int(i) for i in (t.get("current_synonymous_taxon_ids") or [])][:10]
                if out["status"] == "inactive" and ids:
                    out["current"] = [taxon_view(c) for c in
                                      self._ask(TAXA + ",".join(map(str, ids)), {})]
            if out["status"] != "active" and not out["current"] and _FORMAL.match(asked):
                genus, *rest = asked.split()
                stem = f"{genus} {rest[-1][:5]}"
                out["near"] = [taxon_view(t) for t in
                               self._ask(AUTOCOMPLETE, {"q": stem, "is_active": "true",
                                                        "per_page": 30})
                               if in_fungi(t) and t.get("is_active") is not False
                               and loose(t.get("name", "")) == loose(asked)
                               and str(t.get("name", "")).lower() != asked.lower()]
        except (requests.RequestException, ValueError, KeyError, TypeError):
            return {**out, "status": "unavailable"}
        with self._lock:
            if len(self._seen) > 5000:
                self._seen.clear()
            self._seen[exact(name)] = (self.clock(), out)
        return out


def item_names(item: dict) -> list[tuple[str, str]]:
    """(name, role) of a name item: the name on the records, then the proposed one(s)."""
    out: list[tuple[str, str]] = []
    cur = (item.get("current") or item.get("subject") or "").strip()
    if cur:
        out.append((cur, "records"))
    for p in (item.get("proposed") or "").split(";"):
        p = p.strip()
        if p and exact(p) not in {exact(n) for n, _ in out}:
            out.append((p, "proposed"))
    return out


def context(item: dict, inat: InatNames, synonyms: Callable[[str], dict | None] | None,
            budget: int = NAME_BUDGET) -> dict:
    """{"names": [{"name", "roles", "inat", "about"}], "synonyms": {"status", "rows"}}: the
    item's names and every name on MycoMap's synonym list around them, each with how iNat
    holds it. The item's own names are always asked; synonyms while the budget lasts."""
    names: dict[str, dict] = {}

    def add(name: str, role: str) -> None:
        k = exact(name)
        if not k:
            return
        if k not in names:
            names[k] = {"name": name, "roles": []}
        if role not in names[k]["roles"]:
            names[k]["roles"].append(role)

    own = item_names(item)
    for n, role in own:
        add(n, role)
    rows: list[dict] = []
    status = "unavailable"
    if synonyms is not None:
        seen_rows = set()
        try:
            for n, _ in own:
                got = synonyms(n)
                if got is None:
                    continue
                status = "ok"
                for r in got.get("rows") or []:
                    key = (r.get("retired"), r.get("accepted"))
                    if key in seen_rows or not all(key):
                        continue
                    seen_rows.add(key)
                    rows.append({"retired": key[0], "accepted": key[1]})
        except (requests.RequestException, ValueError):
            status = "unavailable"
    for r in rows[:MAX_SYNONYM_ROWS]:
        add(r["retired"], "synonym list")
        add(r["accepted"], "synonym list")
    asked = 0
    for k, entry in names.items():
        mine = any(r in ("records", "proposed") for r in entry["roles"])
        hit = inat.cached(entry["name"])
        if hit is None and (mine or asked < budget):
            hit = inat.one(entry["name"])
            asked += 0 if mine else 1
        entry["inat"] = hit or not_asked(entry["name"])
        entry["about"] = describe(entry["inat"])
    return {"names": list(names.values()),
            "synonyms": {"status": status, "rows": rows[:MAX_SYNONYM_ROWS],
                         "more": max(0, len(rows) - MAX_SYNONYM_ROWS)}}
