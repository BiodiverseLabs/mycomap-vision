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

from . import config, feedback as fb, label_review as lr, name_equiv, photo_review as pr

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


def add_routes(app: FastAPI, *, person: Callable[[Request], dict | None], store: fb.Store,
               org: OrgClient | None = None,
               rescore: Callable[[str], dict] | None = None,
               photos: "pr.ReviewPhotos | None" = None) -> None:
    org = org or OrgClient()
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

    @app.get("/api/feedback/me")
    def fb_me(request: Request):
        who, projects, roles = access_roles(request)
        return {"user": {"id": who["sub"], "name": who.get("name")},
                "admin": projects is None, "staff": bool(roles.get("staff")),
                "projects": sorted(projects or [])}

    @app.get("/api/feedback/catalogue")
    def fb_catalogue(request: Request):
        access(request)
        return fb.type_catalogue()

    @app.get("/api/feedback/summary")
    def fb_summary(request: Request):
        _, projects = access(request)
        return store.summary(projects)

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
        return fb.reviewer_view(item)

    @app.post("/api/feedback/items/{item_id}/decision")
    def fb_decide(request: Request, item_id: int, payload: dict = Body(...)):
        who, projects = access(request)
        item_or_404(item_id, projects)
        try:
            return fb.reviewer_view(store.decide(
                item_id, str(payload.get("decision") or ""), who, value=payload.get("value"),
                note=payload.get("note"), photo_keys=list(payload.get("photo_keys") or []),
                projects=projects))
        except fb.FeedbackError as e:
            bad(e)

    @app.post("/api/feedback/items/{item_id}/reopen")
    def fb_reopen(request: Request, item_id: int):
        _, projects = access(request)
        item_or_404(item_id, projects)
        return fb.reviewer_view(store.reopen(item_id, projects))

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
            answer = rescore(key) if rescore else {"available": False}
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
        shown = {p["photo_key"]: p for p in need_photos().for_viewer(rows, admin)}
        flagged = {f["photo_key"]: f for f in rec["flagged"]}
        out = []
        for key in [r.key for r in rows] + [k for k in flagged if k not in shown]:
            ph = shown.get(key) or {"photo_key": key, "position": None, "shown": "hidden",
                                    "hidden_reason": "not in Vision's photo list now"}
            f = flagged.get(key)
            out.append({**ph, "flagged": f is not None, "in_tab": key in rec["in_tab"],
                        "item_id": f and f["item_id"], "flags": f["flags"] if f else [],
                        "status": f["status"] if f else None,
                        "decision": f["decision"] if f else None})
        return {k: v for k, v in rec.items() if k not in ("flagged", "in_tab")} | {"photos": out}

    @app.get("/api/feedback/photo-review")
    def fb_photo_review(request: Request, tab: str = "all", status: str = "open",
                        sort: str = "p", min_flagged: int = 1, q: str = "", offset: int = 0,
                        limit: int = 10):
        _, projects = access(request)
        try:
            out = pr.cards(store.items_of(pr.ITEM_TYPE, projects), tab, status, sort, min_flagged,
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
                               for p in need_photos().for_viewer(rows, admin)]}

    @app.get("/api/feedback/label-review")
    def fb_label_review(request: Request, status: str = "open", q: str = "", offset: int = 0,
                        limit: int = 10):
        _, projects = access(request)
        try:
            out = lr.cards(store.items_of(lr.ITEM_TYPE, projects), status, q)
        except fb.FeedbackError as e:
            bad(e)
        page = out["records"][max(0, offset):max(0, offset) + min(max(1, limit), 50)]
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

    @app.get("/api/feedback/photo/{photo_key}")
    def fb_photo(request: Request, photo_key: str, size: str = "thumb"):
        """Vision's own copy of a photo, for ADMINS on the review pages only (Steve,
        2026-10-18): private, never stored by the browser or a proxy, and never a photo whose
        owner declined."""
        _, projects = access(request)
        if projects is not None:
            raise HTTPException(403, "admins only")
        if size not in ("thumb", "full"):
            bad(ValueError("size is thumb or full"))
        row = need_photos().one(photo_key)
        if row is None:
            raise HTTPException(404, "no such photo")
        if need_photos().declined(row, need_photos().permission_view()):
            raise HTTPException(403, "the photo's owner declined")
        try:
            got = need_photos().copy_bytes(row, size)
        except Exception:
            got = None
        if got is None:
            raise HTTPException(404, "Vision holds no readable copy of this photo")
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
                "blind", "by", "at", "applied"]
        w = csv.writer(buf, delimiter="\t", lineterminator="\n")
        w.writerow(cols)
        for c in rows:
            w.writerow([c["id"], c["item_type"], c["subject"], c["decision"], c["value"] or "",
                        c["applies"], c["current"] or "", c["proposed"] or "", c["records"],
                        ",".join(c["record_keys"]), ",".join(c["photo_keys"]), c["rule"],
                        c["blind"], c["by_name"] or c["by_user"],
                        time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(c["at"])), c["applied"]])
        return Response(buf.getvalue(), media_type="text/tab-separated-values",
                        headers={"Content-Disposition": 'attachment; filename="feedback-changes.tsv"'})

    @app.post("/api/feedback/decisions/{decision_id}/applied")
    def fb_applied(request: Request, decision_id: int):
        who = admin_only(request)
        store.mark_applied(decision_id, who)
        return JSONResponse({"ok": True})
