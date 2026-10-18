"""The Human Feedback label review (the coordinator, for Steve, 2026-10-18): the "Photos unlike
the label" queue (feedback.TYPES["photos_unlike_label"]) as one card per record, like the photo
review (photo_review.py): every photo of the record, one-click keep / leave out per photo, and
one outcome per record.

The queue is reviewed BLIND (feedback.BLIND_TYPES): finds Vision answered right are mixed in,
and until a card is decided it carries the record, its label and its photos only. No Vision
answer, rule, score or evidence, no blind mark, and an order that does not follow seeding
(feedback.blind_order). Once decided, the card shows what the reviewer_view unmasks.

Outcomes are the queue's own options, one decision per record:
    exclude_photos  the photos the reviewer left out stay out of Vision (applies "vision")
    hard_example    label and photos are right (a note)
    rename          the record's name is wrong: the name to use (applies "org")
Leaving photos out is its own outcome: a card that keeps every photo is a hard example or a
rename, and one that leaves photos out says why (REASONS).
"""

from __future__ import annotations

import json

from . import feedback as fb

ITEM_TYPE = "photos_unlike_label"

REASONS: tuple[tuple[str, str], ...] = (
    ("other_organism", "Shows another organism"),
    ("no_specimen", "No usable specimen"),
    ("other", "Other"),
)
REASON_CODES = tuple(c for c, _ in REASONS)
OUTCOMES = ("exclude_photos", "hard_example", "rename")


def cards(items: list[dict], status: str = "open", query: str = "") -> dict:
    """The queue's items as cards, each the reviewer_view of its item reduced to what the card
    shows. Open cards in the blind order; decided ones newest first."""
    if status not in ("open", "decided"):
        raise fb.FeedbackError("status is open or decided")
    query = (query or "").strip().lower()
    chosen = []
    for it in items:
        if it["status"] != status:
            continue
        key = (it.get("record_keys") or [it["subject"]])[0]
        label = it.get("current") or ""
        if query and query not in key.lower() and query not in label.lower():
            continue
        view = fb.reviewer_view({k: v for k, v in it.items() if k != "last_decision"})
        card = {"item_id": it["id"], "record_key": key, "label": label, "link": fb.link(key),
                "status": it["status"], "masked": view["masked"]}
        if not view["masked"]:
            last = it.get("last_decision") or {}
            card |= {"rule": view.get("rule"), "evidence": view.get("evidence") or {},
                     "decision": last and {
                         "decision": last["decision"], "value": last.get("value"),
                         "photo_keys": json.loads(last.get("photo_keys") or "[]"),
                         "note": last.get("note"), "by": last.get("by_name") or last.get("by_user"),
                         "at": last.get("at")}}
        chosen.append((fb.blind_order(it["subject"]), card))
    chosen = [c for _, c in sorted(chosen, key=lambda x: x[0])]
    if status == "decided":
        chosen.sort(key=lambda c: -((c.get("decision") or {}).get("at") or 0))
    return {"total": len(chosen), "records": chosen}


def decide(store: fb.Store, item_id: int, outcome: str, person: dict, record_photo_keys: set[str],
           photos: list[dict] | None = None, value: str | None = None, reason: str | None = None,
           note: str | None = None, projects: set[str] | None = None) -> dict:
    """Store one record's outcome. `photos`: [{photo_key, decision: keep|exclude}] for that
    record; exclude_photos needs at least one left out and a reason, and the other outcomes
    none left out; rename needs the name. Refuses another record's photo."""
    item = store.get(item_id, projects)
    if item["item_type"] != ITEM_TYPE:
        raise KeyError(item_id)
    if outcome not in OUTCOMES:
        raise fb.FeedbackError(f"the outcome is one of {', '.join(OUTCOMES)}")
    out = []
    for p in photos or []:
        key, d = str(p.get("photo_key") or ""), str(p.get("decision") or "")
        if key not in record_photo_keys:
            raise fb.FeedbackError(f"{key!r} is not a photo of this record")
        if d not in ("keep", "exclude"):
            raise fb.FeedbackError("a photo's decision is keep or exclude")
        if d == "exclude":
            out.append(key)
    if outcome == "exclude_photos":
        if not out:
            raise fb.FeedbackError("choose the photos to leave out")
        if reason not in REASON_CODES:
            raise fb.FeedbackError(f"say why they are left out ({', '.join(REASON_CODES)})")
    elif out:
        raise fb.FeedbackError("leaving photos out is its own outcome: choose 'Leave out the "
                               "chosen photos', or keep every photo")
    text = "; ".join(x for x in (f"reason: {reason}" if outcome == "exclude_photos" else "",
                                 (note or "").strip()) if x) or None
    return store.decide(item_id, outcome, person, value=value, note=text,
                        photo_keys=sorted(out), projects=projects)
