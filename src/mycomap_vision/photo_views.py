"""Human Feedback "Photo views" (Steve, 2026-10-21): what each photo shows, as tags a reviewer
confirms or corrects (cap top, gills, stem and base, in situ, several specimens, …). Not an
exclusion: every photo here stays in Vision. The photo-type tagger (the research repo's
experiments/photo-type-tagger) suggests tags, each a probe checked >= 90% by eye on photos it
never trained on, and seeds them as items of the "photo_view" queue, one per photo, with the
suggestions in the rule ("R8 gills p=0.97; R8 in_situ p=0.99").

The decision model: a reviewer saves the photo's whole tag set, "confirm_views" with the set as
a JSON list in the decision's value (an empty list says the photo shows none of them). A
correction is the same decision with another set; the latest decision is the photo's tags. A
photo that shows no specimen at all is "no_specimen": it goes to "Photos that show no
specimen" as an item of its own (rule "reviewer: no specimen (photo views)"), where the
exclusion is decided, never here.

Tabs: one per view, a photo being in a tab when a suggestion names that view, or (decided)
when its confirmed set has it; "all" holds every photo."""

from __future__ import annotations

import json
import re

from . import feedback as fb
from . import photo_review as pr

ITEM_TYPE = "photo_view"
RULE_TAG = "R8"

# Every view a reviewer can set (Steve's list, 2026-10-21). The tagger suggests only the ones
# whose probe passed its check (SUGGESTED); the others are there to add by hand.
VIEWS: tuple[tuple[str, str], ...] = (
    ("cap_top", "Cap top"),
    ("gills", "Gills"),
    ("pores", "Pores"),
    ("teeth", "Teeth"),
    ("smooth_underside", "Smooth underside"),
    ("stem_base", "Stem and base"),
    ("cross_section", "Cross-section"),
    ("in_situ", "In situ (growing)"),
    ("in_hand", "In hand"),
    ("several", "Several specimens"),
    ("young", "Young / button"),
    ("mixed", "Mixed collection"),
    ("substrate", "Substrate close-up"),
)
VIEW_CODES = tuple(c for c, _ in VIEWS)
SUGGESTED = ("cap_top", "gills", "stem_base", "in_situ", "several")
NO_SPECIMEN_RULE = "reviewer: no specimen (photo views)"
_FLAG = re.compile(r"\b" + RULE_TAG + r" (\w+) p=([0-9.]+)")


def suggestions(rule: str) -> dict[str, float]:
    """The views a rule suggests, each with the probe's p."""
    return {m.group(1): float(m.group(2)) for m in _FLAG.finditer(rule or "")
            if m.group(1) in VIEW_CODES}


def confirmed(item: dict) -> list[str] | None:
    """The tag set of the item's latest decision, or None while undecided / not a set."""
    last = item.get("last_decision") or None
    if not last or last.get("decision") != "confirm_views":
        return None
    try:
        return [v for v in json.loads(last.get("value") or "[]") if v in VIEW_CODES]
    except ValueError:
        return None


def tiles(items: list[dict], tab: str = "all", status: str = "open", query: str = "") -> list[dict]:
    """The queue's photos in `tab`, one tile each: suggestions (strongest first), the confirmed
    set when decided, sorted by the tab's suggestion p (highest first), then record."""
    if tab != "all" and tab not in VIEW_CODES:
        raise fb.FeedbackError(f"unknown view {tab!r}")
    if status not in ("open", "decided"):
        raise fb.FeedbackError("status is open or decided")
    query = (query or "").strip().lower()
    out = []
    for it in items:
        if it["status"] != status:
            continue
        key = pr.record_of_item(it)
        if key is None:
            continue
        label = it.get("current") or ""
        if query and query not in key.lower() and query not in label.lower():
            continue
        sug = suggestions(it.get("rule") or "")
        conf = confirmed(it)
        if tab != "all" and tab not in sug and tab not in (conf or []):
            continue
        last = it.get("last_decision") or None
        out.append({"item_id": it["id"], "photo_key": it["subject"], "record_key": key,
                    "label": label, "link": fb.link(key), "status": it["status"],
                    "suggested": [{"view": v, "p": p} for v, p in
                                  sorted(sug.items(), key=lambda kv: -kv[1])],
                    "confirmed": conf,
                    "decision": last and {"decision": last["decision"], "views": conf},
                    "strength": sug.get(tab, max(sug.values(), default=0.0))})
    out.sort(key=lambda t: (-t["strength"], t["record_key"], t["photo_key"]))
    return out


def tab_counts(items: list[dict]) -> list[dict]:
    """Per view: photos open and decided in its tab."""
    out = []
    for code, title in (("all", "All photos"),) + VIEWS:
        c = {"open": 0, "decided": 0}
        for it in items:
            if code != "all":
                sug = suggestions(it.get("rule") or "")
                if code not in sug and code not in (confirmed(it) or []):
                    continue
            c["open" if it["status"] == "open" else "decided"] += 1
        out.append({"code": code, "title": title, **c})
    return out


def batch_decide(store: fb.Store, decisions: list[dict], person: dict,
                 projects: set[str] | None = None) -> dict:
    """Save a page: each {item_id, views: [codes]} confirms that set, or {item_id,
    no_specimen: true} sends the photo to the exclusion review. Every entry is checked before
    anything is written (an item of this queue the reviewer may see, known views, the paper's
    seal), so a batch is written whole or not at all. Idempotent: an entry equal to the item's
    latest decision writes nothing."""
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
            raise fb.FeedbackError(f"item {item_id} is not a photo of the Photo views queue")
        if store.gate is not None and store.gate.judge_row(item)[0] != "ok":
            raise fb.FeedbackError(f"item {item_id} is under the paper's seal")
        if d.get("no_specimen"):
            plan.append((item, "no_specimen", None))
            continue
        views = d.get("views")
        if not isinstance(views, list) or any(v not in VIEW_CODES for v in views):
            raise fb.FeedbackError(f"views are a list of {', '.join(VIEW_CODES)}")
        ordered = [c for c in VIEW_CODES if c in set(views)]          # one order, no repeats
        plan.append((item, "confirm_views", json.dumps(ordered)))
    saved = unchanged = sent = 0
    for item, decision, value in plan:
        last = item["decisions"][-1] if item["decisions"] else None
        if last and last["decision"] == decision and (last["value"] or None) == value:
            unchanged += 1
            continue
        store.decide(item["id"], decision, person, value=value, projects=projects)
        saved += 1
        if decision == "no_specimen":
            store.seed([{"item_type": pr.ITEM_TYPE, "subject": item["subject"],
                         "rule": NO_SPECIMEN_RULE, "granularity": "photo",
                         "record_keys": item["record_keys"], "links": item["links"],
                         "current": item["current"], "projects": item["projects"]}],
                       "photo-views")
            sent += 1
    return {"saved": saved, "unchanged": unchanged, "sent_to_exclusion_review": sent}
