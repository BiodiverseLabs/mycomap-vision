"""The Human Feedback photo review (Steve, 2026-10-18): the "Photos that show no specimen" queue
(feedback.TYPES["non_specimen_photo"]) reviewed one RECORD at a time, for speed. Each card holds
every photo of the record, the flagged ones marked with why and how strongly; a reviewer keeps
or leaves out each photo with one click and saves the record. Decisions stay per photo
(feedback.Store.decide on each photo's item, applies="vision"), so the nightly exclusions,
the change list and the fairness rules are those of every other queue.

Tabs, one per kind of flag, with their counts: a record shows in each tab one of its photos is
flagged for, with that tab's photos outlined.

Which photos a reviewer sees: a CC photo, or one its owner let Vision show, by its public URL
(Vision's photo rule, identify.py). Every other photo, all rights reserved included, is shown to
Human Feedback reviewers with access to the item (Steve, 2026-10-18), on these signed-in pages
only: served by /api/feedback/photo/, which checks the reviewer's access to a queued record of
the photo, with Cache-Control: private, no-store and no public URL. Never a photo whose owner
withdrew or declined (photo_permissions), nor an MO photo its owner marked not for export, and
never on a public page, Identify or the API.

The image comes from Vision's own copy when this machine can read it, else from the photo's
source (iNat's photo hosts, Mushroom Observer) through SourceFetcher: HTTPS to those hosts only,
paced like every other photo download (photos.py), its own small byte budget on iNat's static
host, a timeout, and a small in-memory cache, so a page of thumbnails costs the source little.
"""

from __future__ import annotations

import io
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Callable

from . import feedback as fb

ITEM_TYPE = "non_specimen_photo"

TABS: tuple[tuple[str, str], ...] = (
    ("micrograph", "Micrographs and microscope set-ups"),
    ("slip", "Field slips and labels"),
    ("habitat", "Habitat and scenes"),
    ("people", "People"),
    ("gate", "Not a fungus (gate)"),
    ("other", "Other"),
)
TAB_CODES = tuple(c for c, _ in TABS)

# Why a photo is left out, one per photo (Steve, 2026-10-18: a record's photos can be of
# several kinds). A flagged photo's reason starts at its flag's (suggested_reason). Dried
# specimens and spore prints have their own reasons (Steve, 2026-10-19) instead of "Other",
# and so do UV reactions, chemical spot tests (KOH, FeSO4 and like colour reactions),
# phylogenetic trees or BLAST results, and drawings of microscopy or of the specimen (Steve,
# 2026-10-21), and species descriptions (a reason only: no flag or tab finds them). A code
# never changes once decisions use it; only its label may.
REASONS: tuple[tuple[str, str], ...] = (
    ("micrograph", "Micrograph or microscope"),
    ("slip", "Field slip or label"),
    ("habitat", "Habitat or scene"),
    ("people", "People"),
    ("not_fungus", "Not a fungus"),
    ("dried", "Dried specimen"),
    ("spore_print", "Spore print"),
    ("uv", "UV reaction"),
    ("chemical", "Chemical spot test"),
    ("phylo_tree", "Phylogenetic tree / BLAST"),   # BLAST result screens too (Steve, 2026-10-21)
    ("drawing_micro", "Drawing – microscopy"),
    ("drawing_specimen", "Drawing – specimen"),
    ("species_description", "Species description"),   # a photo of a written description
    ("other", "Other"),
)
REASON_CODES = tuple(c for c, _ in REASONS)
REASON_OF_TAB = {"micrograph": "micrograph", "slip": "slip", "habitat": "habitat",
                 "people": "people", "gate": "not_fungus", "other": "other"}


def suggested_reason(flags: list[dict], tab: str = "all") -> str | None:
    """The reason a flagged photo's picker starts at: its flag for the open tab when it has
    one, else its strongest flag. None for a photo no rule flagged (the reviewer picks)."""
    if not flags:
        return None
    here = [f for f in flags if f["tab"] == tab] or flags
    best = max(here, key=lambda f: f.get("strength") or 0.0)
    return REASON_OF_TAB.get(best["tab"], "other")

# The zero-shot views (the miss review's s6) by the tab they belong to.
VIEW_TABS = {"habitat": "habitat", "scene": "habitat", "landscape": "habitat",
             "slip": "slip", "label": "slip", "paper": "slip", "text": "slip",
             "people": "people", "person": "people", "hand": "people",
             "micrograph": "micrograph", "microscope": "micrograph"}

_PROBE = re.compile(r"micrograph probe p=([0-9.]+)")
_VIEW = re.compile(r"view (\w+) p=([0-9.]+)")
_GATE = re.compile(r"gate not_fungus p\(fungus\)=([0-9.]+)")


def parse_flags(rule: str) -> list[dict]:
    """The flags a seeded rule names ('R6 micrograph probe p=0.62; gate not_fungus
    p(fungus)=0.01'), each {tab, kind, p, strength}: strength orders the queue (the probe's or
    view's p; for the gate, 1 - p(fungus)). A rule none of these match is one 'other' flag."""
    out = []
    for part in (rule or "").split(";"):
        if m := _PROBE.search(part):
            p = float(m.group(1))
            out.append({"tab": "micrograph", "kind": "micrograph probe", "p": p, "strength": p})
        elif m := _GATE.search(part):
            p = float(m.group(1))
            out.append({"tab": "gate", "kind": "not-a-fungus gate", "p": p,
                        "strength": round(1 - p, 6)})
        elif m := _VIEW.search(part):
            p = float(m.group(2))
            out.append({"tab": VIEW_TABS.get(m.group(1).lower(), "other"),
                        "kind": f"view {m.group(1)}", "p": p, "strength": p})
    if not out:
        out.append({"tab": "other", "kind": (rule or "flagged").strip() or "flagged", "p": None,
                    "strength": 0.0})
    return out


record_of_link = fb.record_of_link          # one rule for record links (feedback.py)


def record_of_item(item: dict) -> str | None:
    for k in item.get("record_keys") or []:
        return k
    for link in item.get("links") or []:
        if key := record_of_link(link):
            return key
    return None


def record_counts(items: list[dict]) -> dict[str, int]:
    """Observations behind the queue's photos: "open" has at least one flagged photo still
    open, "decided" has every flagged photo decided (Steve, 2026-10-18: photo and observation
    counts on the queue's card)."""
    open_, seen = set(), set()
    for it in items:
        key = record_of_item(it)
        if key is None:
            continue
        seen.add(key)
        if it["status"] == "open":
            open_.add(key)
    return {"open": len(open_), "decided": len(seen - open_)}


def cards(items: list[dict], tab: str = "all", status: str = "open", sort: str = "p",
          min_flagged: int = 1, query: str = "") -> dict:
    """Group the queue's photo items by record. `tab` is one of TAB_CODES or 'all'; a record is
    in a tab when one of its photos is flagged for it, and `status` 'open' keeps records with an
    open photo in the tab ('decided': every photo in the tab decided; 'any': both). Sorted by the strongest
    flag in the tab (sort 'p', highest first; 'p_asc' lowest first; 'flagged' most flagged
    photos first). `query` keeps records whose key or label contains it (the tab counts stay
    the queue's). Gives {"tabs": [{code, title, open, decided}], "total", "records"}."""
    if tab != "all" and tab not in TAB_CODES:
        raise fb.FeedbackError(f"unknown tab {tab!r} ({', '.join(TAB_CODES)} or all)")
    if status not in ("open", "decided", "any"):
        raise fb.FeedbackError("status is open, decided or any")
    if sort not in ("p", "p_asc", "flagged"):
        raise fb.FeedbackError("sort is p, p_asc or flagged")
    records: dict[str, dict] = {}
    for it in items:
        key = record_of_item(it)
        if key is None:
            continue
        rec = records.setdefault(key, {"record_key": key, "label": it.get("current") or "",
                                       "link": fb.link(key), "flagged": []})
        last = it.get("last_decision") or None
        rec["flagged"].append({"item_id": it["id"], "photo_key": it["subject"],
                               "flags": parse_flags(it.get("rule") or ""),
                               "status": it["status"],
                               "decision": last and {"decision": last["decision"],
                                                     "reason": last.get("value"),
                                                     "by": last.get("by_name") or last.get("by_user")},
                               "rule": it.get("rule") or ""})

    def in_tab(photo: dict, t: str) -> bool:
        return t == "all" or any(f["tab"] == t for f in photo["flags"])

    def state(rec: dict, t: str) -> str | None:
        ps = [p for p in rec["flagged"] if in_tab(p, t)]
        if not ps:
            return None
        return "open" if any(p["status"] == "open" for p in ps) else "decided"

    tabs = []
    for code, title in TABS:
        c = {"open": 0, "decided": 0}
        for rec in records.values():
            s = state(rec, code)
            if s:
                c[s] += 1
        tabs.append({"code": code, "title": title, **c})
    chosen = []
    query = (query or "").strip().lower()
    for rec in records.values():
        if query and query not in rec["record_key"].lower() and query not in rec["label"].lower():
            continue
        st = state(rec, tab)
        if st is None or (status != "any" and st != status):
            continue
        mine = [p for p in rec["flagged"] if in_tab(p, tab)]
        if len(mine) < max(1, int(min_flagged or 1)):
            continue
        strongest = max((f["strength"] for p in mine for f in p["flags"]
                         if tab == "all" or f["tab"] == tab), default=0.0)
        chosen.append({**rec, "tab": tab, "in_tab": [p["photo_key"] for p in mine],
                       "strongest": strongest, "flagged_in_tab": len(mine)})
    if sort == "flagged":
        chosen.sort(key=lambda r: (-r["flagged_in_tab"], -r["strongest"], r["record_key"]))
    else:
        sign = -1 if sort == "p" else 1
        chosen.sort(key=lambda r: (sign * r["strongest"], r["record_key"]))
    return {"tabs": tabs, "total": len(chosen), "records": chosen}


# --- the record's photos, and the admin-only copy -------------------------------------------------

@dataclass
class PhotoRow:
    photo_id: int
    key: str
    position: int
    source: str
    license_class: str
    owner_user_id: int | None
    ok_for_export: int | None
    source_url: str


PHOTOS_SQL = ("select p.photo_id, op.position, p.source, p.source_photo_id, p.license_class, "
              "p.owner_user_id, p.ok_for_export, p.source_url from observation_photos op "
              "join photos p on p.photo_id = op.photo_id")


def photo_key_of(photo_id: int, source: str, source_photo_id: int | None) -> str:
    if source == "mo":
        return f"mo:{source_photo_id or photo_id - fb.MO_PHOTO_ID_OFFSET}"
    return f"inat:{source_photo_id or photo_id}"


def _row(r) -> PhotoRow:
    return PhotoRow(int(r[0]), photo_key_of(int(r[0]), r[2], r[3]), int(r[1] or 0), r[2],
                    r[4], r[5], r[6], r[7])


class ReviewPhotos:
    """The record's photos from the manifest Vision serves from, and Vision's own copies.

    `query(sql, params)` runs against that manifest (api.py's locked connection);
    `permission_view()` the photo permissions now; `read(store, relpath)` gives a copy's
    bytes (dataset_release.default_store_reader)."""

    THUMB_PX = 480
    GRID_PX = 320              # the Summary grid: iNat's "small" (240 px) / MO's 320 px
    FULL_PX = 1600
    SOURCE_SIZE = {"grid": "small", "thumb": "medium", "full": "large"}
    PX = {"grid": GRID_PX, "thumb": THUMB_PX, "full": FULL_PX}

    def __init__(self, query: Callable[[str, tuple], list], permission_view: Callable,
                 read: Callable[[str, str], bytes] | None = None,
                 source: "SourceFetcher | None" = None, cache_items: int = 400,
                 cache_bytes: int = 96 * 1024 * 1024):
        self.query = query
        self.permission_view = permission_view
        if read is None:
            from .dataset_release import default_store_reader
            read = default_store_reader()
        self.read = read
        self.source = source or SourceFetcher()
        self._cache: "OrderedDict[tuple[str, str], tuple[bytes, str]]" = OrderedDict()
        self._cache_items = cache_items
        self._cache_bytes = cache_bytes
        self._cached_bytes = 0
        self._cache_lock = threading.Lock()

    def records_of(self, row: PhotoRow) -> set[str]:
        """The records (observation ids) the photo belongs to."""
        return {r[0] for r in self.query("select observation_id from observation_photos "
                                         "where photo_id = ?", (row.photo_id,))}

    def of_record(self, record_key: str) -> list[PhotoRow]:
        return [_row(r) for r in self.query(PHOTOS_SQL + " where op.observation_id = ? "
                                            "order by op.position, p.photo_id", (record_key,))]

    def many(self, tiles: list[tuple[str, str]]) -> dict[str, dict]:
        """photo key -> {row, owner_login, license_code, record_photos} for a page of tiles,
        each (photo key, its record key). By the record (observation_photos' key starts with
        it) and the photo id (photos' key): observation_photos has no photo-id index, and a
        look-up by photo id alone reads the whole table."""
        want = {k: (pid, rk) for k, rk in tiles if (pid := fb.photo_id_of(k)) is not None}
        if not want:
            return {}
        records = sorted({rk for _, rk in want.values()})
        rmarks = ",".join("?" * len(records))
        positions, counts = {}, {}
        for rk, pid, pos in self.query(
                f"select observation_id, photo_id, position from observation_photos "
                f"where observation_id in ({rmarks})", tuple(records)):
            counts[rk] = counts.get(rk, 0) + 1
            positions[(rk, int(pid))] = pos
        ids = sorted({pid for pid, _ in want.values()})
        marks = ",".join("?" * len(ids))
        photos = {int(r[0]): r for r in self.query(
            f"select photo_id, source, source_photo_id, license_class, owner_user_id, "
            f"ok_for_export, source_url, owner_login, license_code from photos "
            f"where photo_id in ({marks})", tuple(ids))}
        out = {}
        for key, (pid, rk) in want.items():
            r = photos.get(pid)
            if r is None or (rk, pid) not in positions:
                continue
            row = PhotoRow(pid, photo_key_of(pid, r[1], r[2]), int(positions[(rk, pid)] or 0),
                           r[1], r[3], r[4], r[5], r[6])
            out[key] = {"row": row, "owner_login": r[7], "license_code": r[8],
                        "record_photos": counts.get(rk)}
        return out

    def one(self, photo_key: str) -> PhotoRow | None:
        pid = fb.photo_id_of(photo_key)
        if pid is None:
            return None
        rows = self.query(PHOTOS_SQL + " where p.photo_id = ? limit 1", (pid,))
        return _row(rows[0]) if rows else None

    @staticmethod
    def declined(row: PhotoRow, view) -> bool:
        """The owner said no: withdrew or declined Vision's use (iNat), or marked the photo not
        for export (MO)."""
        if row.source == "inat":
            return row.owner_user_id is not None and int(row.owner_user_id) in view.withdrawn
        return row.source == "mo" and not row.ok_for_export

    def for_viewer(self, rows: list[PhotoRow], reviewer: bool = True,
                   thumb: str = "thumb") -> list[dict]:
        """Each photo as a reviewer of the record sees it: 'public' (its public URL), 'private'
        (served behind sign-in by /api/feedback/photo/) or 'hidden' (a placeholder: the owner
        declined, or `reviewer` is False)."""
        from .licenses import photo_url
        view = self.permission_view()
        out = []
        for r in rows:
            d = {"photo_key": r.key, "position": r.position, "license_class": r.license_class}
            if view.may_show(r.license_class, r.owner_user_id):
                try:
                    d.update(shown="public",
                             thumb=photo_url(r.source, r.source_url, self.SOURCE_SIZE[thumb]),
                             full=photo_url(r.source, r.source_url, "large"))
                except ValueError:
                    d.update(shown="hidden", hidden_reason="no usable photo URL")
            elif self.declined(r, view):
                d.update(shown="hidden", hidden_reason="the owner declined")
            elif reviewer:
                d.update(shown="private", thumb=f"/api/feedback/photo/{r.key}?size={thumb}",
                         full=f"/api/feedback/photo/{r.key}?size=full")
            else:
                d.update(shown="hidden", hidden_reason="all rights reserved")
            out.append(d)
        return out

    def image(self, row: PhotoRow, size: str) -> tuple[bytes, str] | None:
        """The photo at `size` ('thumb' or 'full') as JPEG: Vision's own copy when readable
        here, else fetched from its source; None when neither works (the page shows its
        placeholder). Kept in a small in-memory cache."""
        key = (row.key, size)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
        try:
            got = self.copy_bytes(row, size)
        except Exception:                           # a laptop path, an S3 copy we can't read
            got = None
        if got is None:
            from .licenses import photo_url
            try:
                url = photo_url(row.source, row.source_url, self.SOURCE_SIZE[size])
            except ValueError:
                return None
            data = self.source.get(url, row.source)
            if not data:
                return None
            try:
                got = _scaled(data, self.PX[size])
            except ValueError:
                return None
        with self._cache_lock:
            self._cache[key] = got
            self._cached_bytes += len(got[0])
            while self._cache and (len(self._cache) > self._cache_items
                                   or self._cached_bytes > self._cache_bytes):
                _, old = self._cache.popitem(last=False)
                self._cached_bytes -= len(old[0])
        return got

    def copy_bytes(self, row: PhotoRow, size: str) -> tuple[bytes, str] | None:
        """Vision's own copy of the photo (the smallest copy that is big enough), scaled down
        to the size asked for, as JPEG. None when no copy is held."""
        copies = self.query("select store, size, path from photo_copies where photo_id = ?",
                            (row.photo_id,))
        if not copies:
            return None
        rank = {"medium": 0, "large": 1, "small": 2, "original": 3} if size in ("thumb", "grid") else \
               {"large": 0, "original": 1, "medium": 2, "small": 3}
        last_error = None
        for store, _, path in sorted(copies, key=lambda c: rank.get(c[1], 9)):
            try:
                data = self.read(store, path)
            except Exception as e:                  # a missing file, an unreachable S3
                last_error = e
                continue
            if data:
                return _scaled(data, self.PX[size])
        if last_error is not None:
            raise last_error
        return None


class SourceFetcher:
    """Photo bytes from their source, politely: HTTPS to iNat's photo hosts and Mushroom
    Observer only (one redirect, to MO's own image host), config.USER_AGENT, each host paced
    as photos.py paces it (MO: 5 s apart), a timeout, and a size cap. iNat's static host (all
    rights reserved photos) gets its own budget, well inside the 4 GB/hour and 20 GB/day the
    downloaders keep to, so review pages never eat their share. Over budget, busy, or any
    failure: None, and the page shows a placeholder."""

    STATIC_HOUR_BYTES = 250 * 1024 * 1024
    STATIC_DAY_BYTES = 1024 * 1024 * 1024
    MAX_BYTES = 8 * 1024 * 1024
    TIMEOUT = 10.0
    WAIT = 8.0                                      # longest a page waits for a host's turn

    def __init__(self, session=None, clock=None, sleep=None):
        import time as _time
        import requests
        from . import config
        from .licenses import MO_HOST, OPEN_DATA_HOST, STATIC_HOST
        from .photos import MO_MIN_INTERVAL
        from .ratelimit import ByteBudget, MinInterval
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        clock = clock or _time.monotonic
        sleep = sleep or _time.sleep
        self.static_host = STATIC_HOST
        self.hosts = {OPEN_DATA_HOST: MinInterval(0.02, clock, sleep),
                      STATIC_HOST: MinInterval(0.25, clock, sleep),
                      MO_HOST: MinInterval(MO_MIN_INTERVAL, clock, sleep)}
        self.slots = {h: threading.Semaphore(1 if h == MO_HOST else 2) for h in self.hosts}
        self.budgets = [ByteBudget(self.STATIC_HOUR_BYTES, 3600, clock),
                        ByteBudget(self.STATIC_DAY_BYTES, 86400, clock)]

    def allowed(self, url: str) -> str | None:
        from urllib.parse import urlparse
        u = urlparse(url or "")
        host = (u.hostname or "").lower()
        return host if u.scheme == "https" and host in self.hosts else None

    def get(self, url: str, source: str) -> bytes | None:
        host = self.allowed(url)
        if host is None:
            return None
        if host == self.static_host and max(b.wait_time() for b in self.budgets) > 0:
            return None
        if not self.slots[host].acquire(timeout=self.WAIT):
            return None
        try:
            self.hosts[host].wait()
            data = self._read(url, source)
        finally:
            self.slots[host].release()
        if data and host == self.static_host:
            for b in self.budgets:
                b.add(len(data))
        return data

    def _read(self, url: str, source: str) -> bytes | None:
        import requests
        from urllib.parse import urljoin
        from .photos import mo_redirect_ok
        try:
            resp = self.session.get(url, stream=True, allow_redirects=False,
                                    timeout=self.TIMEOUT)
            if resp.status_code in (301, 302, 303, 307, 308):
                target = urljoin(url, resp.headers.get("Location") or "")
                resp.close()
                if source != "mo" or not mo_redirect_ok(target):
                    return None
                resp = self.session.get(target, stream=True, allow_redirects=False,
                                        timeout=self.TIMEOUT)
            try:
                if resp.status_code != 200:
                    return None
                body = b""
                for chunk in resp.iter_content(64 * 1024):
                    body += chunk
                    if len(body) > self.MAX_BYTES:
                        return None
                return body
            finally:
                resp.close()
        except requests.RequestException:
            return None


def _scaled(data: bytes, px: int) -> tuple[bytes, str]:
    from PIL import Image
    try:
        im = Image.open(io.BytesIO(data))
        im.thumbnail((px, px))
        buf = io.BytesIO()
        im.convert("RGB").save(buf, "JPEG", quality=85)
        return buf.getvalue(), "image/jpeg"
    except Exception:                               # not an image PIL reads: refuse to guess
        raise ValueError("the stored copy is not a readable image") from None


def decide_record(store: fb.Store, record_key: str, photos: list[dict], person: dict,
                  record_photo_keys: set[str], note: str | None = None,
                  projects: set[str] | None = None) -> dict:
    """Store a record's photo decisions: each {photo_key, decision: 'keep'|'exclude', reason}
    on that photo's queue item. A photo no rule flagged that the reviewer leaves out gets an
    item of its own first (rule 'reviewer: left out in photo review', not model-flagged), so
    every exclusion has an item, a decision and who made it. Keeping an unflagged photo stores
    nothing. Refuses a photo that is not the record's, an unknown decision, and an exclusion
    without one of REASON_CODES."""
    items = {it["subject"]: it for it in store.items_of(ITEM_TYPE, projects)
             if record_of_item(it) == record_key}
    if not items:
        raise KeyError(record_key)
    template = next(iter(items.values()))
    known = record_photo_keys | set(items)
    plan = []
    for p in photos:
        key, decision = str(p.get("photo_key") or ""), str(p.get("decision") or "")
        reason = (p.get("reason") or "").strip() or None
        if key not in known:
            raise fb.FeedbackError(f"{key!r} is not a photo of {record_key}")
        if decision not in ("keep", "exclude"):
            raise fb.FeedbackError("a photo's decision is keep or exclude")
        if decision == "exclude" and reason not in REASON_CODES:
            raise fb.FeedbackError(f"say why {key} is left out ({', '.join(REASON_CODES)})")
        plan.append((key, decision, reason))
    if not plan:
        raise fb.FeedbackError("no photo decisions")
    decided = added = 0
    for key, decision, reason in plan:
        item = items.get(key)
        if item is None:
            if decision == "keep":
                continue
            store.seed([{"item_type": ITEM_TYPE, "subject": key,
                         "rule": "reviewer: left out in photo review", "model_flagged": False,
                         "links": [fb.link(record_key)], "current": template.get("current"),
                         "proposed": "leave the photo out of Vision",
                         "projects": template.get("projects") or []}], "photo-review")
            item = next((it for it in store.items_of(ITEM_TYPE, None) if it["subject"] == key),
                        None)
            if item is None:            # the seal refused it (feedback.SealGate)
                raise fb.FeedbackError(f"{record_key} is in the paper's sealed test: its photos "
                                       "are not reviewed until the seal is lifted")
            added += 1
        store.decide(item["id"], "exclude_photo" if decision == "exclude" else "keep_photo",
                     person, value=reason if decision == "exclude" else None, note=note,
                     projects=projects)
        decided += 1
    return {"decided": decided, "added": added}


# --- the Summary view (Steve, 2026-10-21): the flagged photos alone, to scan and batch ---------

def flagged_tiles(items: list[dict], tab: str = "all", status: str = "open", sort: str = "p",
                  query: str = "") -> list[dict]:
    """The queue's flagged photos in `tab`, one tile each (not grouped by record), sorted like the
    record view: by the strongest flag in the tab (p: highest first, p_asc: lowest first), or
    'flagged': the records with most flagged photos first, their photos together."""
    if tab != "all" and tab not in TAB_CODES:
        raise fb.FeedbackError(f"unknown tab {tab!r} ({', '.join(TAB_CODES)} or all)")
    if status not in ("open", "decided"):
        raise fb.FeedbackError("status is open or decided")
    if sort not in ("p", "p_asc", "flagged"):
        raise fb.FeedbackError("sort is p, p_asc or flagged")
    query = (query or "").strip().lower()
    tiles, per_record = [], {}
    for it in items:
        if it["status"] != status:
            continue
        key = record_of_item(it)
        if key is None:
            continue
        label = it.get("current") or ""
        if query and query not in key.lower() and query not in label.lower():
            continue
        flags = parse_flags(it.get("rule") or "")
        mine = [f for f in flags if tab == "all" or f["tab"] == tab]
        if not mine:
            continue
        last = it.get("last_decision") or None
        tiles.append({"item_id": it["id"], "photo_key": it["subject"], "record_key": key,
                      "label": label, "link": fb.link(key), "flags": mine,
                      "strength": max(f["strength"] for f in mine),
                      "suggested_reason": suggested_reason(flags, tab), "status": it["status"],
                      "decision": last and {"decision": last["decision"],
                                            "reason": last.get("value")}})
        per_record[key] = per_record.get(key, 0) + 1
    if sort == "flagged":
        tiles.sort(key=lambda t: (-per_record[t["record_key"]], t["record_key"], -t["strength"]))
    else:
        sign = -1 if sort == "p" else 1
        tiles.sort(key=lambda t: (sign * t["strength"], t["record_key"], t["photo_key"]))
    return tiles


def batch_decide(store: fb.Store, decisions: list[dict], person: dict, note: str | None = None,
                 projects: set[str] | None = None) -> dict:
    """The Summary's saves (Steve, 2026-10-21: a tile's Include/Exclude flipper IS its saved
    decision; the default Exclude is saved once the tile has been shown): each
    {item_id, decision: exclude|keep, reason, note?} on that photo's queue item. Every entry is
    checked before anything is written (a photo item of this queue the reviewer may see, a
    known decision, a reason for each exclusion, and the paper's seal on its record), so a
    batch is written whole or not at all. Keeps are written too: a kept photo does not come
    back. Idempotent: an entry equal to the item's latest decision (same decision and reason)
    writes nothing, so re-sends and flips back add no duplicate rows; the latest row wins."""
    if not decisions:
        raise fb.FeedbackError("nothing to save")
    plan, seen = [], set()
    for d in decisions:
        try:
            item_id = int(d.get("item_id"))
        except (TypeError, ValueError):
            raise fb.FeedbackError(f"not an item id: {d.get('item_id')!r}") from None
        if item_id in seen:
            raise fb.FeedbackError(f"item {item_id} is in the batch twice")
        seen.add(item_id)
        try:
            item = store.get(item_id, projects)
        except KeyError:
            raise fb.FeedbackError(f"item {item_id} is not one you can review") from None
        if item["item_type"] != ITEM_TYPE:
            raise fb.FeedbackError(f"item {item_id} is not a photo of this queue")
        decision = str(d.get("decision") or "")
        reason = (d.get("reason") or "").strip() or None
        if decision not in ("keep", "exclude"):
            raise fb.FeedbackError("a photo's decision is keep or exclude")
        if decision == "exclude" and reason not in REASON_CODES:
            raise fb.FeedbackError(f"say why {item['subject']} is left out "
                                   f"({', '.join(REASON_CODES)})")
        if store.gate is not None and store.gate.judge_row(item)[0] != "ok":
            raise fb.FeedbackError(f"{item['subject']} is on a record in the paper's sealed test")
        code = "exclude_photo" if decision == "exclude" else "keep_photo"
        value = reason if decision == "exclude" else None
        latest = (item.get("decisions") or [None])[-1]
        same = (latest is not None and item["status"] == "decided"
                and latest["decision"] == code and (latest.get("value") or None) == value)
        plan.append((item_id, decision, code, value, d.get("note") or note, same))
    for item_id, _, code, value, why, same in plan:
        if not same:
            store.decide(item_id, code, person, value=value, note=why, projects=projects)
    return {"excluded": sum(1 for p in plan if p[1] == "exclude" and not p[5]),
            "kept": sum(1 for p in plan if p[1] == "keep" and not p[5]),
            "unchanged": sum(1 for p in plan if p[5]),
            "saved": [p[0] for p in plan]}
