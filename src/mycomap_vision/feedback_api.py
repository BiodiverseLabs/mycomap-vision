"""The Human Feedback page's API (feedback.py): /api/feedback/...

Who may use it goes by the mycomap.org site role alone, not project seats (Steve, 2026-11-20:
"any .org member should be able to work with photos. Only contributors or higher should see
and be able to alter things on the taxonomy tab"; "members dont matter. role does.
contributor, lab contributor, project manager, admin"):

- a Contributor (project_assistant), Lab Contributor, Project Leader or Admin (.org's staff
  flag, or its admin flag) sees and decides every item of both tabs;
- any other signed-in mycomap.org account decides the Photos tab's items (feedback.PUBLIC_ITEMS)
  and sees no Taxonomy item (403). It sees the photos as feedback_public.py cuts them: only
  photos anyone may see, and the photos it may not see are left out of its lists and refused
  in its writes (ShownPhotos), so a Save page never marks a photo it could not look at;
- a signed-out visitor browses the Photos tab read-only, cut the same way (401 on writes).

Roles come from mycomap.org (GET /api/vision/feedback/roles, behind the Vision key), cached a
few minutes; MV_FEEDBACK_ADMINS (comma-separated mycomap.org user ids) names admins even
while .org does not answer. The admin downloads (changes.tsv, marking a decision applied) stay
with .org's admin flag (Admin and Project Leader).

What a reviewer sees beyond the queued item comes live from mycomap.org
(GET /api/vision/feedback/record?observation=inat:123): the record's name today, its projects,
links to its .org pages (record, Validate Records, Name Update), and every associated sequence
with its latest BLAST summary and links to the sequence page and full BLAST results (opened in
a new tab by the page). Edits happen on .org through its own flows; this page stores the
decision. "Refresh" re-reads the record from .org, re-asks Vision about it (its stored vectors,
the record itself left out of the reference) and says whether they still disagree.
"""

from __future__ import annotations

import csv
import io
import threading
import time
from typing import Callable

import requests
from fastapi import Body, FastAPI, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response

from . import config, feedback as fb, feedback_org as fo, label_review as lr, name_equiv
from . import feedback_lists
from . import photo_audit
from . import feedback_public as fp
from . import name_context as nc
from . import photo_review as pr
from . import photo_views as pv
from . import recovered_labels as rl
from . import writing_split as ws
from .pair_places import PairPlaces

ROLE_TTL = 300.0

# The refusals of the role rule (module docstring), in words a person can act on.
TAXONOMY_ONLY = ("The Taxonomy queues are for mycomap.org Contributors, Lab Contributors, "
                 "Project Leaders and admins")
PHOTO_NOT_SHOWN = ("This photo is not shown publicly: it is reviewed by mycomap.org "
                   "Contributors, Lab Contributors, Project Leaders and admins")


# Licences shown as they are (permissions.PermissionView.may_show).
CC_CLASSES = ("open", "nc")


class ShownPhotos:
    """Which photos a viewer below Contributor may see (feedback_public.py): the photo rule's
    may_show on the photo's licence and owner now (photo_review.ReviewPhotos.for_viewer's
    "public"). Licence and owner are read from the manifest once per photo and kept for
    `ttl` seconds (a new release may change them); the permissions are read on every ask,
    so a grant shows at once. A photo the manifest no longer has is not shown."""

    CHUNK = 900

    def __init__(self, query: Callable[[str, tuple], list], permission_view: Callable,
                 ttl: float = 3600.0, clock: Callable[[], float] = time.monotonic):
        self.query = query
        self.permission_view = permission_view
        self.ttl = ttl
        self.clock = clock
        # photo id -> True (Creative Commons: always shown), its owner's iNat id (shown with
        # their grant), 0 (no owner: never), None (not in the manifest): one small value each,
        # since the photo queue holds 150,000 photos.
        self.info: dict[int, bool | int | None] = {}
        self.read_at = clock()
        self.generation = 0
        self.lock = threading.Lock()

    def _known(self, pids: set[int]) -> None:
        with self.lock:
            if self.clock() - self.read_at > self.ttl:
                self.info = {}
                self.read_at = self.clock()
                self.generation += 1
            need = sorted(p for p in pids if p not in self.info)
            for i in range(0, len(need), self.CHUNK):
                chunk = need[i:i + self.CHUNK]
                got = {int(r[0]): (True if r[1] in CC_CLASSES else int(r[2] or 0))
                       for r in self.query(
                    "select photo_id, license_class, owner_user_id from photos where photo_id in "
                    f"({','.join('?' * len(chunk))})", tuple(chunk))}
                for p in chunk:
                    self.info[p] = got.get(p)

    def mark(self) -> tuple:
        """Changes when what is shown may have: the permissions, or a fresh manifest read."""
        view = self.permission_view()
        return (self.generation, hash(view.granted), view.fresh())

    def shown(self, photo_keys) -> set[str]:
        """The keys of `photo_keys` this viewer may see."""
        pids = {k: fb.photo_id_of(k) for k in photo_keys}
        self._known({p for p in pids.values() if p is not None})
        view = self.permission_view()
        fresh = view.fresh()
        return {k for k, p in pids.items()
                if p is not None and ((i := self.info.get(p)) is True
                                      or (i and fresh and i in view.granted))}


class ViewCache:
    """Values cached until `version()` changes, built once per key and version: a request that
    finds the key being built waits for that build instead of starting its own. Every Save
    changes the version, and without this each concurrent request (the tiles, the summary,
    every private photo's access check) rebuilt at once on the one store connection: the
    summary took 14 s and a photo 106 s (2026-10-21)."""

    def __init__(self, version: Callable[[], tuple], limit: int = 64):
        self.version = version
        self.limit = limit
        self.views: dict = {}
        self.building: dict = {}           # key -> its build lock
        self.lock = threading.Lock()

    def _fresh(self, key, v) -> tuple | None:
        with self.lock:
            hit = self.views.get(key)
        return hit if hit is not None and hit[0] == v else None

    def __call__(self, key, build: Callable[[], object]):
        if hit := self._fresh(key, self.version()):
            return hit[1]
        with self.lock:
            building = self.building.setdefault(key, threading.Lock())
        with building:
            v = self.version()
            if hit := self._fresh(key, v):    # built by the request this one waited for
                return hit[1]
            value = build()
            with self.lock:
                if len(self.views) > self.limit:
                    self.views.clear()
                self.views[key] = (v, value)
        return value


class OrgClient:
    """mycomap.org's Human Feedback endpoints, behind the Vision key (permissions.org_key)."""

    def __init__(self, base_url: str | None = None, key: str | None = None,
                 session: requests.Session | None = None, timeout: float = 15.0):
        from .permissions import org_key
        self.base = (base_url or config.setting("MV_ORG_BASE_URL")
                     or fo.ORG_SITE).rstrip("/")
        self.key = key if key is not None else org_key()
        self.http = session or requests.Session()
        self.timeout = timeout

    def _get(self, path: str, params: dict) -> dict | None:
        if not self.key:
            return None
        r = self.http.get(self.base + path, params=params, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.key}"})
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def roles(self, user_id: str) -> dict | None:
        """{"admin": bool, "staff": bool, "projects": [{"id", "slug", "name", "role":
        "leader"|"contributor"}]}, or None when .org does not answer roles yet."""
        return self._get("/api/vision/feedback/roles", {"user": user_id})

    def record(self, record_key: str) -> dict | None:
        """The record as .org holds it now (see the module docstring), or None."""
        obs = record_key if record_key.startswith("mo:") else f"inat:{record_key}"
        return self._get("/api/vision/feedback/record", {"observation": obs})

    def synonyms(self, name: str) -> dict | None:
        """MycoMap's synonym list around a name: {"name", "rows": [{"retired", "accepted"}]}
        (name_context.py), or None when .org does not answer it yet."""
        return self._get("/api/vision/feedback/synonyms", {"name": name})

    # --- Name Update requests (feedback_org.py) and the name-change log (name_changes.py) ---
    def _send(self, method: str, path: str, body: dict | None = None) -> tuple[int | None, dict]:
        """(status, JSON body) of a write; (None, {}) without a key. A 4xx is an answer, not
        an error: the caller tells a refusal from a conflict."""
        if not self.key:
            return None, {}
        r = self.http.request(method, self.base + path, json=body, timeout=self.timeout,
                              headers={"Authorization": f"Bearer {self.key}"})
        try:
            data = r.json()
        except ValueError:
            data = {}
        return r.status_code, data if isinstance(data, dict) else {}

    def name_update_post(self, payload: dict) -> tuple[int | None, dict]:
        """Open a Name Update request from a decision: 202 new, 200 the same decision again,
        409 conflict / open_request, 422 names refused, 400 a bad body."""
        return self._send("POST", "/api/vision/name-updates", payload)

    def name_update_get(self, request_id: int) -> dict | None:
        return self._get(f"/api/vision/name-updates/{int(request_id)}", {})

    def name_update_find(self, decision_id: str) -> dict | None:
        return self._get("/api/vision/name-updates", {"decision": decision_id})

    def name_update_delete(self, request_id: int) -> tuple[int | None, dict]:
        """Withdraw a request nobody has run (409 once a run has started)."""
        return self._send("DELETE", f"/api/vision/name-updates/{int(request_id)}")

    def name_changes(self, after: int, limit: int = 100) -> dict | None:
        return self._get("/api/vision/name-changes", {"after": after, "limit": limit})

    def name_change_records(self, change_id: int, after: int = 0) -> dict | None:
        return self._get(f"/api/vision/name-changes/{int(change_id)}/records", {"after": after})


def blast_link(seq: dict, staff: bool) -> dict | None:
    """Where a sequence's BLAST link opens for this viewer (Steve, 2026-10-18; .org's
    blastHrefFor): staff (admins, project leaders, Contributors, lab contributors) the admin
    BLAST results form for the sequence, where the record can be updated; anyone else the
    public job page, never the admin form. Opened in a new window by the page."""
    blast = seq.get("blast") or {}
    if staff and blast.get("admin_results_url"):
        return {"href": blast["admin_results_url"], "kind": "admin"}
    if blast.get("url"):
        return {"href": blast["url"], "kind": "public"}
    return None


def item_url(item_id: int) -> str | None:
    """The item on Vision's Human Feedback page, for .org's link back (MV_PUBLIC_ORIGIN)."""
    origin = (config.setting("MV_PUBLIC_ORIGIN") or "").rstrip("/")
    return f"{origin}/feedback?item={int(item_id)}" if origin.startswith("http") else None


# The Summary's largest page (its page-size choices are 150, 300 and 500; Steve, 2026-10-21).
SUMMARY_MAX_PAGE = 500


def add_routes(app: FastAPI, *, person: Callable[[Request], dict | None], store: fb.Store,
               org: OrgClient | None = None,
               rescore: Callable[[str, dict | None], dict] | None = None,
               photos: "pr.ReviewPhotos | None" = None,
               name_info: "fo.NameInfo | None" = None,
               on_name_update_done: Callable[[], object] | None = None,
               places: "PairPlaces | None" = None,
               inat_names: "nc.InatNames | None" = None,
               note=print) -> fo.OrgRequests:
    """The routes. Returns the Name Update requests (feedback_org.OrgRequests), whose
    poll_once the server runs every feedback_org.POLL_SECONDS."""
    org = org or OrgClient()
    requests_ = fo.OrgRequests(store, org, item_url, on_done=on_name_update_done, note=note)
    cache: dict[str, tuple[float, dict]] = {}
    lock = threading.Lock()
    allow = {x.strip() for x in (config.setting("MV_FEEDBACK_ADMINS") or "").split(",")
             if x.strip()}

    def roles_of(who: dict) -> dict:
        """.org's roles for a signed-in person (cached ROLE_TTL); {"admin": False} when .org
        does not answer. MV_FEEDBACK_ADMINS are admins whatever .org says."""
        sub = str(who["sub"])
        with lock:
            hit = cache.get(sub)
        if hit and time.monotonic() - hit[0] < ROLE_TTL:
            roles = hit[1]
        else:
            try:
                roles = org.roles(sub)
            except requests.RequestException:
                roles = None
            roles = roles or {"admin": False, "staff": False, "projects": []}
            with lock:
                cache[sub] = (time.monotonic(), roles)
        if sub in allow:
            roles = {**roles, "admin": True, "staff": True}
        return roles

    def viewer(request: Request) -> tuple[dict | None, set[str] | None, dict, bool]:
        """(person, projects, roles, cut) by the site role (module docstring). A Contributor or
        higher: projects None (every item of both tabs), cut False. Anyone else, signed in or
        not: fb.PUBLIC_ITEMS (the Photos tab's items) and cut True, seeing what
        feedback_public.py leaves. Who may write is writer()'s and staff_only()'s."""
        who = person(request)
        if not who:
            return None, fb.PUBLIC_ITEMS, {}, True
        roles = roles_of(who)
        if roles.get("admin") or roles.get("staff"):
            return who, None, roles, False
        return who, fb.PUBLIC_ITEMS, roles, True

    # Research > Excluded photos lists People photos only to reviewers (excluded_looked.py):
    # Contributors and up, whose [3] (cut) is False.
    app.state.feedback_viewer = viewer

    def writer(request: Request) -> tuple[dict, set[str] | None, bool]:
        """(person, projects, cut) of a signed-in person: any account writes the Photos tab."""
        who, projects, _, cut = viewer(request)
        if not who:
            raise HTTPException(401, "sign in with your mycomap.org account")
        return who, projects, cut

    def staff_only(request: Request) -> tuple[dict, None]:
        """A write on the Taxonomy tab: Contributors and up only."""
        who, projects, cut = writer(request)
        if cut:
            raise HTTPException(403, TAXONOMY_ONLY)
        return who, projects

    def admin_only(request: Request) -> dict:
        who, _, _ = writer(request)
        if not roles_of(who).get("admin"):
            raise HTTPException(403, "admins only")
        return who

    shown_photos = (ShownPhotos(photos.query, photos.permission_view)
                    if photos is not None else None)
    app.state.feedback_shown_photos = shown_photos

    def check_photo_writes(cut: bool, item_ids=(), photo_keys=()) -> None:
        """Refuse a cut viewer's write that touches a Taxonomy item or a photo it may not see
        (403); unknown items are left to the write itself to refuse."""
        if not cut:
            return
        ids = sorted({int(i) for i in item_ids if isinstance(i, int) or str(i).isdigit()})
        keys = {str(k) for k in photo_keys if k}
        for i in range(0, len(ids), 900):
            chunk = ids[i:i + 900]
            for t, subject in store.conn.execute(
                    "select item_type, subject from items where id in "
                    f"({','.join('?' * len(chunk))})", tuple(chunk)):
                if t not in fb.PUBLIC_PHOTO_TYPES:
                    raise HTTPException(403, TAXONOMY_ONLY)
                keys.add(subject)
        if keys and shown_photos is not None and shown_photos.shown(keys) != keys:
            raise HTTPException(403, PHOTO_NOT_SHOWN)

    def batch_ids(payload: dict) -> list:
        return [d.get("item_id") for d in payload.get("decisions") or [] if isinstance(d, dict)]

    def bad(e: Exception):
        raise HTTPException(400, str(e)) from e

    def version() -> tuple:
        return tuple(store.conn.execute(
            "select (select count(*) from items), (select max(id) from items), "
            "(select max(id) from decisions), (select count(*) from decisions), "
            "(select coalesce(sum(case when status = 'open' then 1 else 0 end), 0) from items), "
            # a seed may change open items' rules (another flag) without adding a row
            "(select max(at) from seeds)"
        ).fetchone())

    def projkey(projects):
        if isinstance(projects, fb.PublicItems):        # the photos it may see change apart
            return ("photos-shown", shown_photos.mark() if shown_photos else None)
        return None if projects is None else tuple(sorted(projects))

    # Counts and item lists cached until the store changes (any item or decision added):
    # the page's first load reads 16,000+ photo items for its counts and lists.
    cached = ViewCache(version)

    def fill(item_type: str, items: list[dict]) -> list[dict]:
        if photos is not None and item_type in pr.NAMED_PHOTO_QUEUES:
            return pr.fill_names(items, photos.names_of)
        return items

    # Each queue read whole once, then kept across saves: a decision or a batch reads again
    # only the items it changed (feedback_lists.py). 150,024 photo items took 12-14 s to read
    # after every Save page (2026-11-20).
    lists = feedback_lists.ItemLists(store, fill)
    app.state.feedback_lists = lists

    def items_cached(item_type: str, projects):
        """The queue's items this viewer may see: for fb.PUBLIC_ITEMS only the photos it may
        look at (ShownPhotos), so its lists, counts and Save page hold no hidden photo."""
        def build():
            items = lists.of(item_type, projects)
            if isinstance(projects, fb.PublicItems) and shown_photos is not None:
                ok = shown_photos.shown({i["subject"] for i in items})
                items = [i for i in items if i["subject"] in ok]
            return items
        return cached(("items", item_type, projkey(projects)), build)

    # The rescoring audit's scores (photo_audit.py), read apart from the queue's items: p only,
    # for a few tabs at a time, all of the run loaded now (photo_audit.ScoreCache); a page's
    # image-only and comparator scores are read for its tiles alone. They never change an item,
    # so the item lists and their cache are not touched by them.
    audit_scores = photo_audit.ScoreCache(store.conn)
    app.state.photo_audit_scores = audit_scores

    def audit_for(tab: str) -> "pr.Audit | None":
        return audit_scores.get(tab)

    def with_audit_details(page: list[dict], audit: "pr.Audit | None") -> list[dict]:
        if audit is None:
            return page
        got = photo_audit.details(store.conn, audit.run, audit.tab,
                                  [t["item_id"] for t in page if t.get("audit")])
        out = []
        for t in page:
            if t.get("audit"):
                pi, cp = got.get(t["item_id"], (None, None))
                t = {**t, "audit": {**t["audit"],
                                    "p_image": None if pi is None else round(pi, 4),
                                    "comparator_p": None if cp is None else round(cp, 4)}}
            out.append(t)
        return out

    @app.get("/api/feedback/me")
    def fb_me(request: Request):
        """Who this viewer is here: read_only signed out; taxonomy for Contributors and up;
        all_photos when every photo is shown to them (the others see only public ones)."""
        who, projects, roles, cut = viewer(request)
        return {"user": who and {"id": who["sub"], "name": who.get("name")},
                "admin": bool(roles.get("admin")), "staff": bool(roles.get("staff")),
                "projects": [], "read_only": who is None, "taxonomy": not cut,
                "all_photos": not cut}

    @app.get("/api/feedback/catalogue")
    def fb_catalogue(request: Request):
        viewer(request)
        out = fb.type_catalogue()
        for t in out:                # which options open a Name Update (feedback_org.KINDS)
            for o in t["options"]:
                o["name_update"] = fo.KINDS.get((t["type"], o["code"]))
                if o["name_update"] == "rename":     # the shared option, on a whole name
                    o["label"] = "Rename every record of this name to…"
        return out

    @app.get("/api/feedback/summary")
    def fb_summary(request: Request):
        """Per queue, open and decided items; for a photo queue also the observations they
        are in (records_open, records_decided). Cached until the store changes."""
        _, projects, _, _ = viewer(request)

        def build():
            if isinstance(projects, fb.PublicItems):
                # The Photos tab only, counted from the photos this viewer may see.
                out = [t for t in store.summary(None) if t["type"] in fb.PUBLIC_PHOTO_TYPES]
                for t in out:
                    items = items_cached(t["type"], projects)
                    t["open"] = sum(1 for i in items if i["status"] == "open")
                    t["decided"] = len(items) - t["open"]
            else:
                out = store.summary(projects)
            for t in out:
                if t["granularity"] == "photo":
                    c = pr.record_counts(items_cached(t["type"], projects))
                    t["records_open"], t["records_decided"] = c["open"], c["decided"]
            return out
        return cached(("summary", projkey(projects)), build)

    def with_writings(items: list[dict]) -> None:
        """Each writing of a name_writing_split item with its records now (Vision's served
        records, exactly as written; null without them) beside the seed's count, and the
        proposal's: one read of the served records for the page."""
        split = [i for i in items if i["item_type"] == "name_writing_split"]
        if not split:
            return
        wanted = sorted({n for i in split for n, _ in ws.writings(i["current"])}
                        | {i["proposed"] for i in split if i.get("proposed")})
        info = name_info(wanted) if name_info else {}

        def now(n):
            return (info.get(n) or {}).get("exact")
        for i in split:
            i["writings"] = [{"name": n, "seeded": c, "records": now(n),
                              "keep": n == i.get("proposed")}
                             for n, c in ws.writings(i["current"])]
            i["proposed_records"] = now(i["proposed"]) if i.get("proposed") else None

    # .org's Names to fix, where the "format" writings are accepted (Name Update's page).
    names_to_fix_url = f"{getattr(org, 'base', None) or fo.ORG_SITE}/admin/name-update"

    @app.get("/api/feedback/items")
    def fb_items(request: Request, type: str, status: str = "open", q: str = "",
                 offset: int = 0, limit: int = 25, list_: str = Query("", alias="list")):
        """A page of a queue; `list` one of a split queue's lists (name_writing_split:
        "format" or "look", writing_split.py)."""
        _, projects, _, ro = viewer(request)
        if ro and type not in fb.PUBLIC_PHOTO_TYPES:
            raise HTTPException(403, TAXONOMY_ONLY)
        if status not in ("open", "decided"):
            bad(ValueError("status is open or decided"))
        try:
            out = store.items(type, status, max(0, offset), min(max(1, limit), 100),
                              projects, q.strip(), list_=list_.strip())
            out["items"] = [fb.reviewer_view(i) for i in out["items"]]
            chips = requests_.chips([i["id"] for i in out["items"]])
            for i in out["items"]:
                i["org_chip"] = chips.get(i["id"])
            with_writings(out["items"])
            if type == "name_writing_split":
                out["names_to_fix_url"] = names_to_fix_url
            return fp.read_only(out) if ro else out
        except fb.FeedbackError as e:
            bad(e)

    def item_or_404(item_id: int, projects):
        try:
            return store.get(item_id, projects)
        except KeyError:
            if isinstance(projects, fb.PublicItems):
                row = store.conn.execute("select item_type from items where id = ?",
                                         (item_id,)).fetchone()
                if row is not None and row[0] not in fb.PUBLIC_PHOTO_TYPES:
                    raise HTTPException(403, TAXONOMY_ONLY) from None
            raise HTTPException(404, "no such item") from None

    @app.get("/api/feedback/items/{item_id}")
    def fb_item(request: Request, item_id: int):
        who, projects, roles, ro = viewer(request)
        item = item_or_404(item_id, projects)
        live = []
        for key in item["record_keys"][:4]:
            try:
                rec = org.record(key) or {"unavailable": True}
            except requests.RequestException as e:
                rec = {"unavailable": True, "error": str(e)[:200]}
            for seq in rec.get("sequences") or []:
                seq["blast_link"] = blast_link(seq, bool(roles.get("staff")))
            rec = {"record_key": key, **rec}
            live.append(fp.live_record(rec) if ro else rec)
        item["live"] = live
        if item["granularity"] == "record pair" and photos is not None:
            # Both copies' photos side by side (Steve, 2026-10-21: one look tells whether the
            # two records are one find). A record Vision has no photos of gets an empty list.
            item["photos"] = {key: [fp.photo(p) if ro else p for p in
                                    photos.for_viewer(photos.of_record(key), reviewer=not ro)]
                              for key in item["record_keys"][:4]}
        if not ro:                                     # the Name Update as .org has it now
            requests_.check_item(item_id)
        view = fo.request_view(requests_.for_item(item_id))
        item["org_request"] = fp.org_request(view) if ro else view
        with_writings([item])
        # An open name item .org has already renamed (feedback_org.already_done): the page
        # offers "Already done on mycomap.org" first and greys out what would ask .org.
        item["already_done"] = (fo.already_done(item, name_info)
                                if item["status"] == "open" else None)
        out = fb.reviewer_view(item)
        if ro:
            return fp.read_only(out)                   # blind items stay blind: no reveal
        if out["masked"]:
            out["revealed"] = store.revealed(item_id, str(who["sub"]))
        return out

    @app.get("/api/feedback/items/{item_id}/places")
    def fb_item_places(request: Request, item_id: int):
        """Where each copy of a record pair is, as MO and iNat list it (pair_places.py): the
        place text only, never a coordinate. Asked apart from the item, so its photos show
        before MO and iNat answer."""
        _, projects, _, _ = viewer(request)
        item = item_or_404(item_id, projects)
        if item["granularity"] != "record pair" or places is None:
            return {"places": {}}
        return {"places": places.of(item["record_keys"][:4])}

    @app.get("/api/feedback/items/{item_id}/names")
    def fb_item_names(request: Request, item_id: int):
        """The names a name item is about, each with how iNaturalist holds it (linked), and
        MycoMap's synonym list around them (name_context.py). Asked apart from the item, so
        the item opens before iNat answers."""
        _, projects, _, _ = viewer(request)
        item = item_or_404(item_id, projects)
        if item["item_type"] not in nc.NAME_TYPES or inat_names is None:
            return {"names": [], "synonyms": {"status": "none", "rows": [], "more": 0}}

        def synonyms(name: str) -> dict | None:
            ask = getattr(org, "synonyms", None)
            return ask(name) if ask else None
        return nc.context(item, inat_names, synonyms)

    def with_request(item: dict) -> dict:
        return item | {"org_request": fo.request_view(requests_.for_item(item["id"]))}

    def plan_or_400(item: dict, decision: str, value):
        try:
            return fo.plan(item, decision, value, name_info)
        except fb.FeedbackError as e:
            bad(e)

    @app.post("/api/feedback/items/{item_id}/org-preview")
    def fb_org_preview(request: Request, item_id: int, payload: dict = Body(...)):
        """What saving this decision would ask mycomap.org for, before it is saved: the old
        names, the new name and anything to check ({"opens": false} for a decision that opens
        no Name Update)."""
        _, projects = staff_only(request)
        item = item_or_404(item_id, projects)
        p = plan_or_400(item, str(payload.get("decision") or ""), payload.get("value"))
        if p is None:
            return {"opens": False}
        wanted = p.old_names + [p.new_name]
        info = name_info(wanted) if name_info else {}
        return {"opens": True, **p.body(), "warnings": p.warnings,
                "records": {n: c for n, c in fo.preview_counts(wanted, info).items()
                            if n in info}}

    @app.post("/api/feedback/items/{item_id}/org-request")
    def fb_org_send_again(request: Request, item_id: int):
        """Send the latest decision's Name Update again (not sent yet, not taken, refused)."""
        _, projects = staff_only(request)
        item = item_or_404(item_id, projects)
        d = requests_.latest_decision(item_id)
        if d is None or item["status"] != "decided":
            raise HTTPException(409, "decide first")
        row = requests_.row(d["id"])
        if row is not None and row["status"] not in fo.SEND_AGAIN:
            raise HTTPException(409, "this Name Update is already on mycomap.org")
        if requests_.send(d["id"], name_info=name_info) is None:
            raise HTTPException(409, "this decision opens no Name Update")
        return with_request(fb.reviewer_view(store.get(item_id, projects)))

    @app.post("/api/feedback/items/{item_id}/decision")
    def fb_decide(request: Request, item_id: int, payload: dict = Body(...)):
        who, projects, cut = writer(request)
        item = item_or_404(item_id, projects)
        check_photo_writes(cut, [item_id])
        decision = str(payload.get("decision") or "")
        # A whole-name decision must make a Name Update before it is saved (feedback_org.py).
        opens = plan_or_400(item, decision, payload.get("value")) is not None
        if decision == fb.DONE_AT_SOURCE.code:     # a false "already done" is never saved
            try:
                fo.refuse_done_at_source(item, name_info)
            except fb.FeedbackError as e:
                bad(e)
        try:
            done = store.decide(
                item_id, decision, who, value=payload.get("value"),
                note=payload.get("note"), photo_keys=list(payload.get("photo_keys") or []),
                projects=projects)
        except fb.FeedbackError as e:
            bad(e)
        if opens:                    # saved either way; .org down leaves "not sent yet"
            requests_.send(done["decisions"][-1]["id"], name_info=name_info)
        out = with_request(fb.reviewer_view(store.get(item_id, projects)))
        return fp.read_only(out) if cut else out

    @app.post("/api/feedback/items/{item_id}/reopen")
    def fb_reopen(request: Request, item_id: int):
        _, projects, cut = writer(request)
        item_or_404(item_id, projects)
        check_photo_writes(cut, [item_id])
        try:                         # a Name Update nobody has run is withdrawn first
            requests_.withdraw_for_reopen(item_id)
        except fo.Unwithdrawable as e:
            raise HTTPException(409, str(e)) from None
        out = with_request(fb.reviewer_view(store.reopen(item_id, projects)))
        return fp.read_only(out) if cut else out

    @app.post("/api/feedback/items/{item_id}/reveal")
    def fb_reveal(request: Request, item_id: int):
        """"Show Vision's answer" on a blind item before deciding (Steve, 2026-10-18). Gives
        Vision's top 5 for the record asked afresh (the record left out of the reference), the
        same way for every item of the queue, so it never tells a check-sample item from a
        flagged one: no rule, evidence or outcome. The reveal is recorded, and a decision made
        after it is marked seen_answer and kept out of the blind measure."""
        who, projects = staff_only(request)
        item = item_or_404(item_id, projects)
        if not fb.reviewer_view(item)["masked"]:
            raise HTTPException(409, "nothing is hidden on this item")
        key = (item["record_keys"] or [item["subject"]])[0]
        got = rescore(key, who) if rescore else {"available": False}
        answer = {"available": bool(got.get("available")), "top5": got.get("top5") or [],
                  "reason": got.get("reason") if not got.get("available") else None}
        return store.reveal(item_id, who, answer)

    @app.post("/api/feedback/items/{item_id}/refresh")
    def fb_refresh(request: Request, item_id: int):
        """Re-read each record from .org and re-ask Vision: do they still disagree?"""
        who, projects = staff_only(request)
        item = item_or_404(item_id, projects)
        if fb.reviewer_view(item)["masked"]:
            raise HTTPException(409, "decide first: this queue is reviewed without Vision's "
                                     "answer")
        out = []
        for key in item["record_keys"][:4]:
            try:
                now = org.record(key) or {}
            except requests.RequestException:
                now = {}
            name = now.get("name") or item.get("current")
            answer = rescore(key, who) if rescore else {"available": False}
            top5 = answer.get("top5") or []
            still = (None if not top5 or not name
                     else not name_equiv.species_credit(top5[0][0], name))
            store.record_refresh(item_id, who, now.get("name"), top5, still)
            out.append({"record_key": key, "name_now": now.get("name"), "vision": answer,
                        "still_disagrees": still})
        return {"item": fb.reviewer_view(store.get(item_id, projects)), "refreshed": out}

    # --- the photo review: one card per record (photo_review.py) ----------------------------
    def need_photos() -> pr.ReviewPhotos:
        if photos is None:
            raise HTTPException(503, "photo review needs the served manifest")
        return photos

    def card(rec: dict, admin: bool, ro: bool = False) -> dict:
        rows = need_photos().of_record(rec["record_key"])
        shown = {p["photo_key"]: p for p in need_photos().for_viewer(rows, reviewer=not ro,
                                                                      thumb="grid")}
        flagged = {f["photo_key"]: f for f in rec["flagged"]}
        out = []
        for key in [r.key for r in rows] + [k for k in flagged if k not in shown]:
            ph = shown.get(key) or {"photo_key": key, "position": None, "shown": "hidden",
                                    "hidden_reason": "not in Vision's photo list now"}
            f = flagged.get(key)
            out.append({**ph, "flagged": f is not None, "in_tab": key in rec["in_tab"],
                        "item_id": f and f["item_id"], "flags": f["flags"] if f else [],
                        "suggested_reason": pr.suggested_reason(f["flags"], rec.get("tab", "all"))
                        if f else None,
                        "suggested_decision": pr.suggested_decision(f["flags"]) if f else None,
                        "band": pr.band_of(f["flags"]) if f else None,
                        "status": f["status"] if f else None,
                        "outcome": f.get("outcome") if f else None,
                        "decision": f["decision"] if f else None})
        out = {k: v for k, v in rec.items() if k not in ("flagged", "in_tab")} | {"photos": out}
        return fp.read_only(out) if ro else out

    @app.get("/api/feedback/photo-review/flagged")
    def fb_photo_review_flagged(request: Request, tab: str = "all", status: str = "open",
                                sort: str = "p", q: str = "", offset: int = 0, limit: int = 150,
                                band: str = "all", outcome: str = "all", show: str = "all"):
        """The Summary view (Steve, 2026-10-21): the queue's flagged photos in the tab, one tile
        each, paged (at most SUMMARY_MAX_PAGE), with the record's name and link, the flags, the
        suggested reason, and a small thumbnail: public photos straight from their source, others
        from /api/feedback/photo/…?size=grid (private, no-store). `outcome` keeps the photos whose
        flag in the tab came to it (photo_review.outcome_of); "outcomes" counts the photos each
        would keep (Steve, 2026-10-30). `show` keeps the photos that stayed in the tab's
        category or moved out of it; "shows" counts each (Steve, 2026-11-20)."""
        _, projects, _, ro = viewer(request)
        counts: dict = {}
        show_counts: dict = {}
        audit = audit_for(tab) if status == "decided" else None
        try:
            tiles = pr.flagged_tiles(items_cached(pr.ITEM_TYPE, projects), tab, status, sort, q,
                                     band, outcome, counts, audit, show, show_counts)
        except fb.FeedbackError as e:
            bad(e)
        start = max(0, offset)
        page = with_audit_details(tiles[start:start + min(max(1, limit), SUMMARY_MAX_PAGE)], audit)
        out = photo_tiles(page, ro)
        return {"total": len(tiles), "offset": start, "tiles": out, "outcomes": counts,
                "shows": show_counts, "reasons": [{"code": c, "label": t} for c, t in pr.REASONS]}

    @app.get("/api/feedback/photo-review/audit")
    def fb_photo_review_audit(request: Request):
        """The rescoring audit loaded now (photo_audit.py; Steve, 2026-11-20): its run and the
        tabs it scored, each with its High cut and the served comparator shown beside it. The
        Summary offers "Strongest disagreement first" and "High score, not in this tab" on the
        Decided list of these tabs only."""
        viewer(request)
        run = photo_audit.latest_run(store.conn)
        if run is None:
            return {"run": None, "tabs": []}
        return {"run": {k: run[k] for k in ("run_id", "loaded_at", "rows", "items")},
                "tabs": [{"code": t, "high": run["high"].get(t, photo_audit.HIGH_FALLBACK),
                          "comparator": run["comparators"].get(t)} for t in run["tabs"]]}

    def photo_tiles(page: list[dict], ro: bool, thumb: str = "grid") -> list[dict]:
        """A page of tiles with each photo as this viewer may see it (Summary, Photo views,
        Recovered labels)."""
        info = need_photos().many([(t["photo_key"], t["record_key"]) for t in page])
        shown = {d["photo_key"]: d for d in need_photos().for_viewer(
            [i["row"] for i in info.values()], reviewer=not ro, thumb=thumb)}
        out = []
        for t in page:
            i = info.get(t["photo_key"])
            ph = shown.get(t["photo_key"]) or {"photo_key": t["photo_key"], "position": None,
                                               "shown": "hidden",
                                               "hidden_reason": "not in Vision's photo list now"}
            tile = t | ph | {"owner_login": i and i["owner_login"],
                             "license_code": i and i["license_code"],
                             "record_photos": i and i["record_photos"]}
            out.append(fp.read_only(fp.photo(tile)) if ro else tile)
        return out

    @app.get("/api/feedback/photo-views")
    def fb_photo_views(request: Request, tab: str = "all", status: str = "open", q: str = "",
                       offset: int = 0, limit: int = 60):
        """Photo views (Steve, 2026-10-21): the queue's photos in a view's tab, one tile each,
        paged (at most 200), with the suggested and confirmed tags and a small thumbnail (public
        photos from their source, others from /api/feedback/photo/…?size=grid)."""
        _, projects, _, ro = viewer(request)
        items = items_cached(pv.ITEM_TYPE, projects)
        try:
            tiles = pv.tiles(items, tab, status, q)
        except fb.FeedbackError as e:
            bad(e)
        start = max(0, offset)
        page = tiles[start:start + min(max(1, limit), 200)]
        out = photo_tiles(page, ro)
        return {"tabs": pv.tab_counts(items), "total": len(tiles), "offset": start,
                "views": [{"code": c, "label": t, "suggested": c in pv.SUGGESTED}
                          for c, t in pv.VIEWS],
                "tiles": out}

    @app.post("/api/feedback/photo-views/batch")
    def fb_photo_views_batch(request: Request, payload: dict = Body(...)):
        """Save a page of Photo views: [{item_id, views: [...]}, or {item_id, no_specimen:
        true}], written whole or not at all (photo_views.batch_decide)."""
        who, projects, cut = writer(request)
        check_photo_writes(cut, batch_ids(payload))
        try:
            return pv.batch_decide(store, list(payload.get("decisions") or []), who,
                                   projects=projects)
        except fb.FeedbackError as e:
            bad(e)

    # --- Recovered labels: specimen crops of excluded slip photos (recovered_labels.py) -----
    @app.get("/api/feedback/recovered-labels")
    def fb_recovered_labels(request: Request, tranche: str = "all", status: str = "open",
                            sort: str = "riskiest", q: str = "", offset: int = 0,
                            limit: int = 30):
        """Recovered labels (Steve, 2026-10-26): one tile per excluded slip photo with the crop
        boxes the detector proposes, paged (at most 100), smallest specimen first, or the
        priority list's order (tranche "priority": records that would lose every photo). The photo
        as this viewer may see it (the photo rule); a read-only viewer gets the boxes on the
        original only, and the page shows them no enlarged crop. "still_excluded" False: a
        person has since kept the photo, or left it out for another reason, so its crops
        count for nothing."""
        _, projects, _, ro = viewer(request)
        items = items_cached(rl.ITEM_TYPE, projects)
        slips = rl.photo_reasons(items_cached(pr.ITEM_TYPE, None))
        # The priority tranche (Steve, 2026-11-01): photos of records that would otherwise lose
        # every photo, marked by their own list (recovered_labels.seed_priority).
        prio = cached(("recovered-priority",), lambda: rl.priorities(store))
        try:
            tiles = rl.tiles(items, tranche, status, sort, q, slips, prio)
        except fb.FeedbackError as e:
            bad(e)
        start = max(0, offset)
        page = tiles[start:start + min(max(1, limit), 100)]
        return {"total": len(tiles), "offset": start, "tranches": rl.tranche_counts(items),
                "priority": rl.priority_counts(items, prio),
                "reasons": [{"code": c, "label": t} for c, t in pr.REASONS],
                "tiles": photo_tiles(page, ro, thumb="full")}

    @app.post("/api/feedback/recovered-labels/batch")
    def fb_recovered_labels_batch(request: Request, payload: dict = Body(...)):
        """Save a page of Recovered labels: [{item_id, include: [crop numbers kept], boxes?:
        {n: [l, t, r, b]}, reason?}], written whole or not at all (recovered_labels.batch_decide)."""
        who, projects, cut = writer(request)
        check_photo_writes(cut, batch_ids(payload))
        try:
            return rl.batch_decide(store, list(payload.get("decisions") or []), who,
                                   projects=projects)
        except fb.FeedbackError as e:
            bad(e)

    @app.post("/api/feedback/photo-review/batch")
    def fb_photo_review_batch(request: Request, payload: dict = Body(...)):
        """The Summary's one commit: [{item_id, decision: exclude|keep, reason}], written whole
        or not at all (photo_review.batch_decide)."""
        who, projects, cut = writer(request)
        check_photo_writes(cut, batch_ids(payload))
        try:
            return pr.batch_decide(store, list(payload.get("decisions") or []), who,
                                   note=payload.get("note"), projects=projects)
        except fb.FeedbackError as e:
            bad(e)

    @app.get("/api/feedback/photo-review")
    def fb_photo_review(request: Request, tab: str = "all", status: str = "open",
                        sort: str = "p", min_flagged: int = 1, q: str = "", offset: int = 0,
                        limit: int = 10, band: str = "all", outcome: str = "all",
                        show: str = "all"):
        _, projects, _, ro = viewer(request)
        try:
            out = pr.cards(items_cached(pr.ITEM_TYPE, projects), tab, status, sort, min_flagged,
                           q, band, outcome, show)
        except fb.FeedbackError as e:
            bad(e)
        page = out["records"][max(0, offset):max(0, offset) + min(max(1, limit), 50)]
        return {"tabs": out["tabs"], "total": out["total"], "outcomes": out["outcomes"],
                "shows": out["shows"], "reasons": [
                    {"code": c, "label": t} for c, t in pr.REASONS],
                "records": [card(r, projects is None, ro) for r in page]}

    @app.post("/api/feedback/photo-review/{record_key}/decision")
    def fb_photo_review_decide(request: Request, record_key: str, payload: dict = Body(...)):
        who, projects, cut = writer(request)
        keys = {r.key for r in need_photos().of_record(record_key)}
        check_photo_writes(cut, photo_keys=[p.get("photo_key") for p in payload.get("photos") or []
                                            if isinstance(p, dict)])
        try:
            done = pr.decide_record(store, record_key, list(payload.get("photos") or []), who,
                                    keys, note=payload.get("note"), projects=projects)
        except KeyError:
            raise HTTPException(404, "no such record in the photo review") from None
        except fb.FeedbackError as e:
            bad(e)
        rec = next((r for r in pr.cards(items_cached(pr.ITEM_TYPE, projects), "all",
                                        "any")["records"] if r["record_key"] == record_key), None)
        return {**done, "record": card(rec, projects is None, cut) if rec else None}

    # --- the label review: the blind "Photos unlike the label" queue as cards (label_review) --
    def label_card(c: dict, admin: bool, ro: bool = False) -> dict:
        rows = need_photos().of_record(c["record_key"])
        excluded = set(((c.get("decision") or {}).get("photo_keys")) or [])
        out = c | {"photos": [p | {"excluded": p["photo_key"] in excluded}
                              for p in need_photos().for_viewer(rows, reviewer=not ro,
                                                                thumb="grid")]}
        return fp.read_only(out | {"photos": [fp.photo(p) for p in out["photos"]]}) if ro else out

    @app.get("/api/feedback/label-review")
    def fb_label_review(request: Request, status: str = "open", q: str = "", offset: int = 0,
                        limit: int = 10):
        who, projects, _, ro = viewer(request)
        if ro:
            raise HTTPException(403, TAXONOMY_ONLY)
        try:
            out = lr.cards(store.items_of(lr.ITEM_TYPE, projects), status, q)
        except fb.FeedbackError as e:
            bad(e)
        page = out["records"][max(0, offset):max(0, offset) + min(max(1, limit), 50)]
        for c in page:
            if c["masked"] and not ro:                 # blind stays blind read-only
                c["revealed"] = store.revealed(c["item_id"], str(who["sub"]))
        return {"total": out["total"],
                "reasons": [{"code": c, "label": t} for c, t in lr.REASONS],
                "outcomes": [{"code": o.code, "label": o.label, "applies": o.applies,
                              "needs_value": o.needs_value}
                             for o in fb.TYPES[lr.ITEM_TYPE].options],
                "records": [label_card(c, projects is None, ro) for c in page]}

    @app.post("/api/feedback/label-review/{item_id}/decision")
    def fb_label_review_decide(request: Request, item_id: int, payload: dict = Body(...)):
        who, projects = staff_only(request)
        try:
            item = store.get(item_id, projects)
        except KeyError:
            raise HTTPException(404, "no such item") from None
        if item["item_type"] != lr.ITEM_TYPE:
            raise HTTPException(404, "no such item in the label review")
        key = (item["record_keys"] or [item["subject"]])[0]
        keys = {r.key for r in need_photos().of_record(key)}
        try:
            lr.decide(store, item_id, str(payload.get("outcome") or ""), who, keys,
                      photos=list(payload.get("photos") or []), value=payload.get("value"),
                      reason=payload.get("reason"), note=payload.get("note"), projects=projects)
        except fb.FeedbackError as e:
            bad(e)
        card = next((c for c in lr.cards(store.items_of(lr.ITEM_TYPE, projects), "decided")
                     ["records"] if c["item_id"] == item_id), None)
        return {"record": label_card(card, projects is None) if card else None}

    # Built once a minute: every private photo on a page asks at once.
    reachable = ViewCache(lambda: (int(time.monotonic() // 60),))

    def records_reachable(projects: set[str] | None) -> set[str]:
        """The records of every queue item this reviewer can open (cached a minute)."""
        def build() -> set[str]:
            out: set[str] = set()
            for r in store._visible("select * from items", (), projects):
                it = fb.Store._item(r)
                out.update(it["record_keys"] or [])
                if (k := pr.record_of_item(it)) is not None:
                    out.add(k)
            return out
        return reachable(None if projects is None else frozenset(projects), build)

    @app.get("/api/feedback/photo/{photo_key}")
    def fb_photo(request: Request, photo_key: str, size: str = "thumb"):
        """A photo for the review pages (Steve, 2026-10-18): to any Human Feedback reviewer
        with access to a queued record of the photo, private, never stored by the browser or a
        proxy, and never a photo whose owner declined. From Vision's copy or its source. A
        read-only viewer gets only a photo anyone may see (feedback_public.py), never an
        all-rights-reserved one."""
        _, projects, _, ro = viewer(request)
        if size not in ("grid", "thumb", "full"):
            bad(ValueError("size is grid, thumb or full"))
        row = need_photos().one(photo_key)
        if row is None:
            raise HTTPException(404, "no such photo")
        if not (need_photos().records_of(row) & records_reachable(projects)):
            raise HTTPException(403, "this photo is not on a record you review")
        view = need_photos().permission_view()
        if need_photos().declined(row, view):
            raise HTTPException(403, "the photo's owner declined")
        if ro and not view.may_show(row.license_class, row.owner_user_id):
            raise HTTPException(403, "this photo is shown only to the record's reviewers")
        got = need_photos().image(row, size)
        if got is None:
            raise HTTPException(404, "the photo could not be read just now")
        data, media = got
        return Response(data, media_type=media, headers={
            "Cache-Control": "private, no-store", "Pragma": "no-cache",
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})

    @app.get("/api/feedback/changes.tsv")
    def fb_changes(request: Request, applies: str = "org", all: bool = False):
        admin_only(request)
        rows = store.changes(applies=applies or None, pending_only=not all)
        buf = io.StringIO()
        cols = ["decision_id", "item_type", "subject", "decision", "value", "applies",
                "current", "proposed", "records", "record_keys", "photo_keys", "rule",
                "blind", "seen_answer", "by", "at", "applied", "name_update"]
        w = csv.writer(buf, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for c in rows:
            # A whole-name decision already on .org as a Name Update: not to be retyped there.
            req = fo.request_view(requests_.row(c["id"]))
            w.writerow([c["id"], c["item_type"], c["subject"], c["decision"], c["value"] or "",
                        c["applies"], c["current"] or "", c["proposed"] or "", c["records"],
                        ",".join(c["record_keys"]), ",".join(c["photo_keys"]), c["rule"],
                        c["blind"], int(c["seen_answer"]), c["by_name"] or c["by_user"],
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(c["at"])), c["applied"],
                        "" if req is None else f"request {req['request_id'] or '-'}: {req['status']}"])
        return Response(buf.getvalue(), media_type="text/tab-separated-values",
                        headers={"Content-Disposition": 'attachment; filename="feedback-changes.tsv"'})

    @app.post("/api/feedback/decisions/{decision_id}/applied")
    def fb_applied(request: Request, decision_id: int):
        who = admin_only(request)
        store.mark_applied(decision_id, who)
        return JSONResponse({"ok": True})

    return requests_
