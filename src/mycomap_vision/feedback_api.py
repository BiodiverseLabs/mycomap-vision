"""The Human Feedback page's API (feedback.py): /api/feedback/...

Who may use it: a signed-in mycomap.org admin (every item), project leader or contributor (the
items of their projects). Roles come from mycomap.org (GET /api/vision/feedback/roles, behind
the Vision key), cached a few minutes; until .org answers it, MV_FEEDBACK_ADMINS (comma-
separated mycomap.org user ids) names the admins and nobody else gets in.

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
from fastapi import Body, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from . import config, feedback as fb, feedback_org as fo, label_review as lr, name_equiv
from . import name_context as nc
from . import photo_review as pr
from .pair_places import PairPlaces

ROLE_TTL = 300.0


class OrgClient:
    """mycomap.org's Human Feedback endpoints, behind the Vision key (permissions.org_key)."""

    def __init__(self, base_url: str | None = None, key: str | None = None,
                 session: requests.Session | None = None, timeout: float = 15.0):
        from .permissions import org_key
        self.base = (base_url or config.setting("MV_ORG_BASE_URL")
                     or "https://www.mycomap.org").rstrip("/")
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

    def access(request: Request) -> tuple[dict, set[str] | None]:
        """(person, projects): projects None for an admin (every item)."""
        who, projects, _ = access_roles(request)
        return who, projects

    def access_roles(request: Request) -> tuple[dict, set[str] | None, dict]:
        """(person, projects, roles); roles["staff"] is .org's staff flag (blastHrefFor)."""
        who = person(request)
        if not who:
            raise HTTPException(401, "sign in with your mycomap.org account")
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
            roles = roles or {"admin": False, "projects": []}
            with lock:
                cache[sub] = (time.monotonic(), roles)
        if sub in allow:
            roles = {**roles, "admin": True, "staff": True}
        if roles.get("admin"):
            return who, None, roles
        # Records name their projects (green_projects: .org's validation_project_N), so the
        # project's name is the key; its id only when .org gives no name.
        projects = {str(p.get("name") or p["id"]) for p in roles.get("projects") or []
                    if p.get("role") in ("leader", "contributor")}
        if not projects:
            raise HTTPException(403, "Human Feedback is for mycomap.org admins, project "
                                     "leaders and contributors")
        return who, projects, roles

    def admin_only(request: Request) -> dict:
        who, projects = access(request)
        if projects is not None:
            raise HTTPException(403, "admins only")
        return who

    def bad(e: Exception):
        raise HTTPException(400, str(e)) from e

    # Counts and item lists cached until the store changes (any item or decision added):
    # the page's first load reads 16,000+ photo items for its counts and lists.
    views: dict = {}
    views_lock = threading.Lock()

    def version() -> tuple:
        return tuple(store.conn.execute(
            "select (select count(*) from items), (select max(id) from items), "
            "(select max(id) from decisions), (select count(*) from decisions), "
            "(select coalesce(sum(case when status = 'open' then 1 else 0 end), 0) from items)"
        ).fetchone())

    def projkey(projects):
        return None if projects is None else tuple(sorted(projects))

    def cached(key, build):
        v = version()
        with views_lock:
            hit = views.get(key)
            if hit and hit[0] == v:
                return hit[1]
        value = build()
        with views_lock:
            if len(views) > 64:
                views.clear()
            views[key] = (v, value)
        return value

    def items_cached(item_type: str, projects):
        return cached(("items", item_type, projkey(projects)),
                      lambda: store.items_of(item_type, projects))

    @app.get("/api/feedback/me")
    def fb_me(request: Request):
        who, projects, roles = access_roles(request)
        return {"user": {"id": who["sub"], "name": who.get("name")},
                "admin": projects is None, "staff": bool(roles.get("staff")),
                "projects": sorted(projects or [])}

    @app.get("/api/feedback/catalogue")
    def fb_catalogue(request: Request):
        access(request)
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
        _, projects = access(request)

        def build():
            out = store.summary(projects)
            for t in out:
                if t["granularity"] == "photo":
                    c = pr.record_counts(items_cached(t["type"], projects))
                    t["records_open"], t["records_decided"] = c["open"], c["decided"]
            return out
        return cached(("summary", projkey(projects)), build)

    @app.get("/api/feedback/items")
    def fb_items(request: Request, type: str, status: str = "open", q: str = "",
                 offset: int = 0, limit: int = 25):
        _, projects = access(request)
        if status not in ("open", "decided"):
            bad(ValueError("status is open or decided"))
        try:
            out = store.items(type, status, max(0, offset), min(max(1, limit), 100),
                              projects, q.strip())
            out["items"] = [fb.reviewer_view(i) for i in out["items"]]
            chips = requests_.chips([i["id"] for i in out["items"]])
            for i in out["items"]:
                i["org_chip"] = chips.get(i["id"])
            return out
        except fb.FeedbackError as e:
            bad(e)

    def item_or_404(item_id: int, projects):
        try:
            return store.get(item_id, projects)
        except KeyError:
            raise HTTPException(404, "no such item") from None

    @app.get("/api/feedback/items/{item_id}")
    def fb_item(request: Request, item_id: int):
        _, projects, roles = access_roles(request)
        item = item_or_404(item_id, projects)
        live = []
        for key in item["record_keys"][:4]:
            try:
                rec = org.record(key) or {"unavailable": True}
            except requests.RequestException as e:
                rec = {"unavailable": True, "error": str(e)[:200]}
            for seq in rec.get("sequences") or []:
                seq["blast_link"] = blast_link(seq, bool(roles.get("staff")))
            live.append({"record_key": key, **rec})
        item["live"] = live
        if item["granularity"] == "record pair" and photos is not None:
            # Both copies' photos side by side (Steve, 2026-10-21: one look tells whether the
            # two records are one find). A record Vision has no photos of gets an empty list.
            item["photos"] = {key: photos.for_viewer(photos.of_record(key))
                              for key in item["record_keys"][:4]}
        requests_.check_item(item_id)                  # the Name Update as .org has it now
        item["org_request"] = fo.request_view(requests_.for_item(item_id))
        out = fb.reviewer_view(item)
        if out["masked"]:
            out["revealed"] = store.revealed(item_id, str(person(request)["sub"]))
        return out

    @app.get("/api/feedback/items/{item_id}/places")
    def fb_item_places(request: Request, item_id: int):
        """Where each copy of a record pair is, as MO and iNat list it (pair_places.py): the
        place text only, never a coordinate. Asked apart from the item, so its photos show
        before MO and iNat answer."""
        _, projects = access(request)
        item = item_or_404(item_id, projects)
        if item["granularity"] != "record pair" or places is None:
            return {"places": {}}
        return {"places": places.of(item["record_keys"][:4])}

    @app.get("/api/feedback/items/{item_id}/names")
    def fb_item_names(request: Request, item_id: int):
        """The names a name item is about, each with how iNaturalist holds it (linked), and
        MycoMap's synonym list around them (name_context.py). Asked apart from the item, so
        the item opens before iNat answers."""
        _, projects = access(request)
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
        _, projects = access(request)
        item = item_or_404(item_id, projects)
        p = plan_or_400(item, str(payload.get("decision") or ""), payload.get("value"))
        if p is None:
            return {"opens": False}
        info = name_info(p.old_names + [p.new_name]) if name_info else {}
        return {"opens": True, **p.body(), "warnings": p.warnings,
                "records": {n: (info.get(n) or {}).get("records") for n in info}}

    @app.post("/api/feedback/items/{item_id}/org-request")
    def fb_org_send_again(request: Request, item_id: int):
        """Send the latest decision's Name Update again (not sent yet, not taken, refused)."""
        _, projects = access(request)
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
        who, projects = access(request)
        item = item_or_404(item_id, projects)
        decision = str(payload.get("decision") or "")
        # A whole-name decision must make a Name Update before it is saved (feedback_org.py).
        opens = plan_or_400(item, decision, payload.get("value")) is not None
        try:
            done = store.decide(
                item_id, decision, who, value=payload.get("value"),
                note=payload.get("note"), photo_keys=list(payload.get("photo_keys") or []),
                projects=projects)
        except fb.FeedbackError as e:
            bad(e)
        if opens:                    # saved either way; .org down leaves "not sent yet"
            requests_.send(done["decisions"][-1]["id"], name_info=name_info)
        return with_request(fb.reviewer_view(store.get(item_id, projects)))

    @app.post("/api/feedback/items/{item_id}/reopen")
    def fb_reopen(request: Request, item_id: int):
        _, projects = access(request)
        item_or_404(item_id, projects)
        try:                         # a Name Update nobody has run is withdrawn first
            requests_.withdraw_for_reopen(item_id)
        except fo.Unwithdrawable as e:
            raise HTTPException(409, str(e)) from None
        return with_request(fb.reviewer_view(store.reopen(item_id, projects)))

    @app.post("/api/feedback/items/{item_id}/reveal")
    def fb_reveal(request: Request, item_id: int):
        """"Show Vision's answer" on a blind item before deciding (Steve, 2026-10-18). Gives
        Vision's top 5 for the record asked afresh (the record left out of the reference), the
        same way for every item of the queue, so it never tells a check-sample item from a
        flagged one: no rule, evidence or outcome. The reveal is recorded, and a decision made
        after it is marked seen_answer and kept out of the blind measure."""
        who, projects = access(request)
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
        who, projects = access(request)
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

    def card(rec: dict, admin: bool) -> dict:
        rows = need_photos().of_record(rec["record_key"])
        shown = {p["photo_key"]: p for p in need_photos().for_viewer(rows, thumb="grid")}
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
                        "status": f["status"] if f else None,
                        "decision": f["decision"] if f else None})
        return {k: v for k, v in rec.items() if k not in ("flagged", "in_tab")} | {"photos": out}

    @app.get("/api/feedback/photo-review/flagged")
    def fb_photo_review_flagged(request: Request, tab: str = "all", status: str = "open",
                                sort: str = "p", q: str = "", offset: int = 0, limit: int = 150):
        """The Summary view (Steve, 2026-10-21): the queue's flagged photos in the tab, one tile
        each, paged (at most 200), with the record's name and link, the flags, the suggested
        reason, and a small thumbnail: public photos straight from their source, others
        from /api/feedback/photo/…?size=grid (private, no-store)."""
        _, projects = access(request)
        try:
            tiles = pr.flagged_tiles(items_cached(pr.ITEM_TYPE, projects), tab, status, sort, q)
        except fb.FeedbackError as e:
            bad(e)
        start = max(0, offset)
        page = tiles[start:start + min(max(1, limit), 200)]
        info = need_photos().many([(t["photo_key"], t["record_key"]) for t in page])
        shown = {d["photo_key"]: d for d in need_photos().for_viewer(
            [i["row"] for i in info.values()], thumb="grid")}
        out = []
        for t in page:
            i = info.get(t["photo_key"])
            ph = shown.get(t["photo_key"]) or {"photo_key": t["photo_key"], "position": None,
                                               "shown": "hidden",
                                               "hidden_reason": "not in Vision's photo list now"}
            out.append(t | ph | {"owner_login": i and i["owner_login"],
                                 "license_code": i and i["license_code"],
                                 "record_photos": i and i["record_photos"]})
        return {"total": len(tiles), "offset": start, "tiles": out,
                "reasons": [{"code": c, "label": t} for c, t in pr.REASONS]}

    @app.post("/api/feedback/photo-review/batch")
    def fb_photo_review_batch(request: Request, payload: dict = Body(...)):
        """The Summary's one commit: [{item_id, decision: exclude|keep, reason}], written whole
        or not at all (photo_review.batch_decide)."""
        who, projects = access(request)
        try:
            return pr.batch_decide(store, list(payload.get("decisions") or []), who,
                                   note=payload.get("note"), projects=projects)
        except fb.FeedbackError as e:
            bad(e)

    @app.get("/api/feedback/photo-review")
    def fb_photo_review(request: Request, tab: str = "all", status: str = "open",
                        sort: str = "p", min_flagged: int = 1, q: str = "", offset: int = 0,
                        limit: int = 10):
        _, projects = access(request)
        try:
            out = pr.cards(items_cached(pr.ITEM_TYPE, projects), tab, status, sort, min_flagged,
                           q)
        except fb.FeedbackError as e:
            bad(e)
        page = out["records"][max(0, offset):max(0, offset) + min(max(1, limit), 50)]
        return {"tabs": out["tabs"], "total": out["total"], "reasons": [
                    {"code": c, "label": t} for c, t in pr.REASONS],
                "records": [card(r, projects is None) for r in page]}

    @app.post("/api/feedback/photo-review/{record_key}/decision")
    def fb_photo_review_decide(request: Request, record_key: str, payload: dict = Body(...)):
        who, projects = access(request)
        keys = {r.key for r in need_photos().of_record(record_key)}
        try:
            done = pr.decide_record(store, record_key, list(payload.get("photos") or []), who,
                                    keys, note=payload.get("note"), projects=projects)
        except KeyError:
            raise HTTPException(404, "no such record in the photo review") from None
        except fb.FeedbackError as e:
            bad(e)
        rec = next((r for r in pr.cards(store.items_of(pr.ITEM_TYPE, projects), "all",
                                        "any")["records"] if r["record_key"] == record_key), None)
        return {**done, "record": card(rec, projects is None) if rec else None}

    # --- the label review: the blind "Photos unlike the label" queue as cards (label_review) --
    def label_card(c: dict, admin: bool) -> dict:
        rows = need_photos().of_record(c["record_key"])
        excluded = set(((c.get("decision") or {}).get("photo_keys")) or [])
        return c | {"photos": [p | {"excluded": p["photo_key"] in excluded}
                               for p in need_photos().for_viewer(rows, thumb="grid")]}

    @app.get("/api/feedback/label-review")
    def fb_label_review(request: Request, status: str = "open", q: str = "", offset: int = 0,
                        limit: int = 10):
        _, projects = access(request)
        try:
            out = lr.cards(store.items_of(lr.ITEM_TYPE, projects), status, q)
        except fb.FeedbackError as e:
            bad(e)
        page = out["records"][max(0, offset):max(0, offset) + min(max(1, limit), 50)]
        me = str(person(request)["sub"])
        for c in page:
            if c["masked"]:
                c["revealed"] = store.revealed(c["item_id"], me)
        return {"total": out["total"],
                "reasons": [{"code": c, "label": t} for c, t in lr.REASONS],
                "outcomes": [{"code": o.code, "label": o.label, "applies": o.applies,
                              "needs_value": o.needs_value}
                             for o in fb.TYPES[lr.ITEM_TYPE].options],
                "records": [label_card(c, projects is None) for c in page]}

    @app.post("/api/feedback/label-review/{item_id}/decision")
    def fb_label_review_decide(request: Request, item_id: int, payload: dict = Body(...)):
        who, projects = access(request)
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

    reachable: dict[frozenset | None, tuple[float, set[str]]] = {}

    def records_reachable(projects: set[str] | None) -> set[str]:
        """The records of every queue item this reviewer can open (cached a minute)."""
        key = None if projects is None else frozenset(projects)
        with lock:
            hit = reachable.get(key)
        if hit and time.monotonic() - hit[0] < 60:
            return hit[1]
        out: set[str] = set()
        for r in store._visible("select * from items", (), projects):
            it = fb.Store._item(r)
            out.update(it["record_keys"] or [])
            if (k := pr.record_of_item(it)) is not None:
                out.add(k)
        with lock:
            reachable[key] = (time.monotonic(), out)
        return out

    @app.get("/api/feedback/photo/{photo_key}")
    def fb_photo(request: Request, photo_key: str, size: str = "thumb"):
        """A photo for the review pages (Steve, 2026-10-18): to any Human Feedback reviewer
        with access to a queued record of the photo, private, never stored by the browser or a
        proxy, and never a photo whose owner declined. From Vision's copy or its source."""
        _, projects = access(request)
        if size not in ("grid", "thumb", "full"):
            bad(ValueError("size is grid, thumb or full"))
        row = need_photos().one(photo_key)
        if row is None:
            raise HTTPException(404, "no such photo")
        if not (need_photos().records_of(row) & records_reachable(projects)):
            raise HTTPException(403, "this photo is not on a record you review")
        if need_photos().declined(row, need_photos().permission_view()):
            raise HTTPException(403, "the photo's owner declined")
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
