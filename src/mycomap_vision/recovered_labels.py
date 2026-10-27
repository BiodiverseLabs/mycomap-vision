"""Human Feedback "Recovered labels" (Steve, 2026-10-26): a specimen crop for a field-slip photo
a person left out.

People keep excluding field-slip ("label") photos as before ("Photos that show no specimen",
reason "slip"); nothing here changes that decision. For each such photo the specimen detector
(slipcrop.py, run as a batch job: `mv feedback slip-crops`) looks for a specimen on the whole
photo and proposes crop boxes; `mv feedback seed --slip-crops` queues them here, one item per
photo. A crop goes back into Vision unless a person rejects it (Steve's rule: the queue's
default is Include), and only while the photo stays excluded as a slip: kept, or excluded for
another reason, and its crops count for nothing.

The crop is COORDINATES on the original photo, never a new image file (no-derivatives licences;
the public read-only viewer must never be shown a derivative). The page draws the box on the
original and, for reviewers, shows the box's part of the same image by CSS. Embedding applies
the box in memory (embed.py); the manifest holds the boxes Vision uses (feedback_photo_crops,
synced like the exclusions).

Decisions: "include_crops" with the crop numbers kept as a JSON list in the value, or
"reject_crops" (none kept). The latest decision is the photo's crops; an open item is every
crop. Both apply in Vision. A reviewer may move or resize a crop (Steve, 2026-10-27): the value
is then {"include": [...], "boxes": {"<n>": [l, t, r, b]}}, the reviewer's boxes in the photo's
pixels, carried into later decisions until changed again. From the same tile a reviewer may
change the photo's exclusion reason, written on its "Photos that show no specimen" item as the
photo review writes it; only a field slip gives crops.

Paper (the coordinator's rule, 2026-10-26): crops never change the E34/E35 exclusion lists
(`mv feedback export-photo-exclusions` is byte-identical with or without them). They have an
export of their own (`mv feedback export-photo-crops`) that a training run reads only with
--photo-crops, pinned by sha256, and only in a crop arm Steve declares before any result.
"""

from __future__ import annotations

import csv
import json

from . import feedback as fb
from . import photo_review as pr

ITEM_TYPE = "recovered_label"
SLIP_REASON = "slip"
RULE = "slipcrop: specimen found on a field-slip photo a person left out"

# The slip tranches: tagger R10's slip p on the photo's own flag when it was seeded, in
# Steve's bands for slip flags (2026-10-26: high >= 0.6, medium 0.4-0.6, low 0.2-0.4); crops
# start with high. "rule" holds slips an earlier rule found, or below 0.2.
TRANCHES: tuple[tuple[str, str, float | None, float | None], ...] = (
    ("high", "High (p ≥ 0.60)", 0.60, None),
    ("medium", "Medium (0.40–0.60)", 0.40, 0.60),
    ("low", "Low (0.20–0.40)", 0.20, 0.40),
    ("rule", "Earlier rules (no tagger p)", None, None),
)
TRANCHE_CODES = tuple(c for c, *_ in TRANCHES)
SORTS = ("riskiest", "largest", "newest")
# The tagger whose slip p places a photo: R10 (the HF-retrained probe, 2026-10-26).
TAGGER = "R10"

SEED_COLUMNS = ("photo_key", "record_key", "label", "slip_p", "tranche", "width", "height",
                "slip_area", "specimen_area", "on_slip_area", "crops", "version")


def tranche_of(p: float | None) -> str:
    """The tranche a slip p falls in ("rule" with no p, or below the lowest band)."""
    if p is None:
        return "rule"
    for code, _, lo, hi in TRANCHES:
        if lo is not None and p >= lo and (hi is None or p < hi):
            return code
    return "rule"


def slip_p(rule: str, tagger: str = TAGGER) -> float | None:
    """The slip p `tagger` gave the photo ("R10 v2 slip p=0.83 (low)"); its last when several."""
    ps = [f["p"] for part in (rule or "").split(";") if part.strip().startswith(tagger + " ")
          for f in pr.parse_flags(part) if f.get("v2") and f["tab"] == "slip"]
    return ps[-1] if ps else None


def slip_excluded(photo_items: list[dict]) -> dict[str, dict]:
    """photo key -> its item, of the "Photos that show no specimen" items (Store.items_of)
    whose latest decision leaves the photo out as a field slip."""
    out = {}
    for it in photo_items:
        last = it.get("last_decision") or {}
        if last.get("decision") == "exclude_photo" and last.get("value") == SLIP_REASON:
            out[it["subject"]] = it
    return out


def slip_exclusions(store: fb.Store) -> dict[str, dict]:
    """Every photo a person leaves out as a field slip now, with its item."""
    return slip_excluded(store.items_of(pr.ITEM_TYPE))


# --- seeding --------------------------------------------------------------------------------------

def seed_row(photo: dict, found: dict) -> dict:
    """One line of the seed TSV: the slip photo (photo_key, record_key, label, slip_p) and the
    detector's proposal (slipcrop.Detector.detect)."""
    return {"photo_key": photo["photo_key"], "record_key": photo["record_key"],
            "label": photo.get("label") or "",
            "slip_p": "" if photo.get("slip_p") is None else f"{photo['slip_p']:.3f}",
            "tranche": tranche_of(photo.get("slip_p")),
            "width": found["width"], "height": found["height"],
            "slip_area": found["slip_area"], "specimen_area": found["specimen_area"],
            "on_slip_area": found["on_slip_area"],
            "crops": json.dumps([c["box"] for c in found["crops"]], separators=(",", ":")),
            "version": found["version"]}


def valid_box(box, w: int, h: int) -> bool:
    return (isinstance(box, list) and len(box) == 4 and all(isinstance(v, (int, float))
                                                           for v in box)
            and 0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h)


def rows_from_slip_crops(path) -> tuple[list[dict], dict]:
    """The slip-crops TSV as queue rows, and counts: a photo where the detector found no
    specimen is not queued ("no_specimen"); a box outside its photo refuses the file."""
    rows, none = [], 0
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            w, h = int(r["width"]), int(r["height"])
            boxes = json.loads(r["crops"] or "[]")
            if not boxes:
                none += 1
                continue
            for b in boxes:
                if not valid_box(b, w, h):
                    raise fb.FeedbackError(f"{r['photo_key']}: crop {b} is not inside its "
                                           f"{w} x {h} photo")
            p = float(r["slip_p"]) if r.get("slip_p") else None
            rows.append({
                "item_type": ITEM_TYPE, "subject": r["photo_key"], "granularity": "photo",
                "rule": RULE + (f" (slip p={p:.2f})" if p is not None else ""),
                "model_flagged": True, "record_keys": [r["record_key"]],
                "links": [fb.link(r["record_key"])], "current": r.get("label") or "",
                "proposed": "include the specimen crop(s) in Vision",
                "evidence": {"version": r["version"], "width": w, "height": h,
                             "slip_p": p, "tranche": r.get("tranche") or tranche_of(p),
                             "slip_area": float(r["slip_area"] or 0),
                             "specimen_area": float(r["specimen_area"] or 0),
                             "on_slip_area": float(r.get("on_slip_area") or 0),
                             "crops": [{"n": i + 1, "box": b} for i, b in enumerate(boxes)]}})
    return rows, {"no_specimen": none}


# --- what a decision means --------------------------------------------------------------------

def crops_of(item: dict) -> list[dict]:
    return list((item.get("evidence") or {}).get("crops") or [])


def read_value(decision: dict | None) -> tuple[list[int] | None, dict[int, list[int]]]:
    """A Recovered labels decision's (crop numbers kept, reviewer's boxes by crop number).
    None kept for no decision; [] for reject_crops or an unreadable value."""
    if not decision:
        return None, {}
    if decision["decision"] == "reject_crops":
        return [], {}
    try:
        v = json.loads(decision.get("value") or "[]")
    except ValueError:
        return [], {}
    if isinstance(v, dict):
        return (list(v.get("include") or []),
                {int(n): list(b) for n, b in (v.get("boxes") or {}).items()})
    return list(v), {}


def last_of(item: dict) -> dict | None:
    """The item's latest decision, from items_of ("last_decision") or get ("decisions")."""
    if "last_decision" in item:
        return item.get("last_decision") or None
    return (item.get("decisions") or [None])[-1]


def kept(item: dict) -> list[int]:
    """The crop numbers the item keeps now: every crop while open (the default Include), else
    its latest decision's."""
    every = [c["n"] for c in crops_of(item)]
    chosen, _ = read_value(last_of(item))
    if item["status"] == "open" or chosen is None:
        return every
    return [n for n in every if n in set(chosen)]


def boxes_now(item: dict) -> dict[int, list[int]]:
    """Each crop's box now: the reviewer's (the latest decision's), else the detector's."""
    _, edited = read_value(last_of(item))
    return {c["n"]: list(edited.get(c["n"], c["box"])) for c in crops_of(item)}


def accepted_crops(store: fb.Store) -> dict[str, dict]:
    """photo key -> {"width", "height", "boxes": [(n, [l, t, r, b] as fractions of the photo)],
    "version"} for every photo a person still leaves out as a field slip and whose item keeps a
    crop: the crops Vision uses."""
    slips = slip_exclusions(store)
    out = {}
    for it in store.items_of(ITEM_TYPE):
        if it["subject"] not in slips:
            continue
        ev = it.get("evidence") or {}
        w, h = int(ev.get("width") or 0), int(ev.get("height") or 0)
        keep = set(kept(it))
        now = boxes_now(it)
        boxes = [(n, [round(b[0] / w, 6), round(b[1] / h, 6), round(b[2] / w, 6),
                      round(b[3] / h, 6)])
                 for n, b in now.items() if n in keep and w and h]
        if boxes:
            out[it["subject"]] = {"width": w, "height": h, "boxes": boxes,
                                  "version": ev.get("version") or ""}
    return out


# --- the queue's page -------------------------------------------------------------------------

def photo_reasons(photo_items: list[dict]) -> dict[str, str]:
    """photo key -> its exclusion reason now ("keep" when kept), from the "Photos that show no
    specimen" items (Store.items_of); a photo nobody decided is not in it."""
    out = {}
    for it in photo_items:
        last = it.get("last_decision") or {}
        if last.get("decision") == "exclude_photo":
            out[it["subject"]] = last.get("value") or "other"
        elif last.get("decision") == "keep_photo":
            out[it["subject"]] = KEEP
    return out


def tiles(items: list[dict], tranche: str = "all", status: str = "open", sort: str = "riskiest",
          query: str = "", slips: set[str] | dict | None = None) -> list[dict]:
    """The queue's photos, one tile each with its crops (each with its box now, the detector's
    "proposed" box, and the decision it carries). Sorted riskiest first (the smallest
    specimen: where false finds sit), or largest, or newest. With `slips` (photo key -> its
    exclusion reason, photo_reasons; or the set of photos left out as slips), each tile says
    its "reason", and a photo no longer left out as a slip is shown as such
    ("still_excluded": False): its crops count for nothing."""
    if tranche != "all" and tranche not in TRANCHE_CODES:
        raise fb.FeedbackError(f"tranche is all or one of {', '.join(TRANCHE_CODES)}")
    if status not in ("open", "decided"):
        raise fb.FeedbackError("status is open or decided")
    if sort not in SORTS:
        raise fb.FeedbackError(f"sort is one of {', '.join(SORTS)}")
    query = (query or "").strip().lower()
    out = []
    for it in items:
        if it["status"] != status:
            continue
        ev = it.get("evidence") or {}
        if tranche != "all" and ev.get("tranche") != tranche:
            continue
        key = pr.record_of_item(it)
        if key is None:
            continue
        label = it.get("current") or ""
        if query and query not in key.lower() and query not in label.lower() \
                and query not in it["subject"].lower():
            continue
        keep = set(kept(it))
        now = boxes_now(it)
        decided = it["status"] == "decided"
        crops = [{"n": c["n"], "box": now[c["n"]], "proposed": c["box"],
                  "area": round((now[c["n"]][2] - now[c["n"]][0])
                                * (now[c["n"]][3] - now[c["n"]][1])
                                / max(1, ev.get("width", 1) * ev.get("height", 1)), 4),
                  "decision": ("include" if c["n"] in keep else "reject") if decided else None}
                 for c in crops_of(it)]
        if isinstance(slips, dict):
            reason = slips.get(it["subject"])
            still = reason == SLIP_REASON
        else:
            reason = None if slips is None else (SLIP_REASON if it["subject"] in slips else None)
            still = None if slips is None else it["subject"] in slips
        out.append({"item_id": it["id"], "photo_key": it["subject"], "record_key": key,
                    "label": label, "link": fb.link(key), "slip_p": ev.get("slip_p"),
                    "tranche": ev.get("tranche") or "rule", "width": ev.get("width"),
                    "height": ev.get("height"), "specimen_area": ev.get("specimen_area") or 0.0,
                    "crops": crops, "status": it["status"],
                    "still_excluded": still, "reason": reason,
                    "created_at": it.get("created_at") or 0})
    if sort == "riskiest":
        out.sort(key=lambda t: (t["specimen_area"], t["photo_key"]))
    elif sort == "largest":
        out.sort(key=lambda t: (-t["specimen_area"], t["photo_key"]))
    else:
        out.sort(key=lambda t: (-t["created_at"], t["photo_key"]))
    for t in out:
        del t["created_at"]
    return out


def tranche_counts(items: list[dict]) -> list[dict]:
    """Per tranche: photos open and decided (every item is in exactly one)."""
    out = []
    for code in TRANCHE_CODES:
        c = {"open": 0, "decided": 0}
        for it in items:
            if ((it.get("evidence") or {}).get("tranche") or "rule") == code:
                c["open" if it["status"] == "open" else "decided"] += 1
        out.append({"code": code, **c})
    return out


KEEP = "keep"            # a tile's reason picker: keep the whole photo in Vision


def min_side(w: int, h: int) -> int:
    """The smallest side a reviewer's box may have (the page's minSide)."""
    return max(16, round(0.04 * min(w, h)))


def check_box(box, w: int, h: int) -> list[int]:
    """A reviewer's box as whole pixels inside its w x h photo, at least min_side a side."""
    if not (isinstance(box, list) and len(box) == 4
            and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in box)):
        raise fb.FeedbackError(f"a box is [left, top, right, bottom] in pixels, not {box!r}")
    b = [int(round(v)) for v in box]
    m = min_side(w, h)
    if not (0 <= b[0] and 0 <= b[1] and b[2] <= w and b[3] <= h
            and b[2] - b[0] >= m and b[3] - b[1] >= m):
        raise fb.FeedbackError(f"box {b} is not inside the {w} x {h} photo with sides of at "
                               f"least {m} px")
    return b


def crop_value(chosen: list[int], boxes: dict[int, list[int]]) -> str:
    """include_crops' value: the kept numbers, with the reviewer's boxes when there are any."""
    if not boxes:
        return json.dumps(chosen)
    return json.dumps({"include": chosen, "boxes": {str(n): boxes[n] for n in sorted(boxes)}},
                      separators=(",", ":"))


def batch_decide(store: fb.Store, decisions: list[dict], person: dict,
                 projects: set[str] | None = None) -> dict:
    """Save a page: each {item_id, include: [crop numbers kept], boxes?: {n: [l, t, r, b]},
    reason?} (an empty include rejects every crop). Include is a decision too, once the
    reviewer has seen the tile.

    `boxes` are the reviewer's moved or resized crops (photo pixels), for kept crops only; a
    box back on the detector's is no longer an edit, and boxes saved before carry over.
    `reason` changes the photo's exclusion: one of photo_review.REASONS, written as
    "exclude_photo" on its "Photos that show no specimen" item, or "keep" ("keep_photo"); only
    a field slip gives crops.

    Every entry is checked before anything is written (an item of this queue the reviewer may
    see, crop numbers it has, boxes inside the photo, a known reason, the paper's seal), so a
    batch is written whole or not at all. Idempotent: an entry equal to the latest decisions
    writes nothing."""
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
            raise fb.FeedbackError(f"item {item_id} is not a photo of the Recovered labels queue")
        if store.gate is not None and store.gate.judge_row(item)[0] != "ok":
            raise fb.FeedbackError(f"item {item_id} is under the paper's seal")
        include = d.get("include")
        have = {c["n"]: c["box"] for c in crops_of(item)}
        if not isinstance(include, list) or any(not isinstance(n, int) or isinstance(n, bool)
                                                or n not in have for n in include):
            raise fb.FeedbackError(f"include is a list of the crop numbers of item {item_id} "
                                   f"({', '.join(map(str, sorted(have)))})")
        chosen = sorted(set(include))
        ev = item.get("evidence") or {}
        w, h = int(ev.get("width") or 0), int(ev.get("height") or 0)
        given = d.get("boxes") or {}
        if not isinstance(given, dict):
            raise fb.FeedbackError("boxes are {crop number: [left, top, right, bottom]}")
        _, edited = read_value(last_of(item))
        edited = {n: b for n, b in edited.items() if n in chosen}      # carried over
        for n, b in given.items():
            try:
                n = int(n)
            except (TypeError, ValueError):
                raise fb.FeedbackError(f"not a crop number: {n!r}") from None
            if n not in chosen:
                raise fb.FeedbackError(f"crop {n} of item {item_id} is not kept: a box is only "
                                       "for a kept crop")
            edited[n] = check_box(b, w, h)
        edited = {n: b for n, b in edited.items() if b != list(have[n])}
        reason = d.get("reason")
        photo = None
        if reason is not None:
            if reason != KEEP and reason not in pr.REASON_CODES:
                raise fb.FeedbackError(f"reason is one of {', '.join(pr.REASON_CODES)} or keep")
            row = store.conn.execute("select id from items where item_type = ? and subject = ?",
                                     (pr.ITEM_TYPE, item["subject"])).fetchone()
            if row is None:
                raise fb.FeedbackError(f"{item['subject']} has no exclusion to change")
            try:
                photo = store.get(row["id"], projects)
            except KeyError:
                raise fb.FeedbackError(f"{item['subject']}'s exclusion is not one you can "
                                       "review") from None
            if store.gate is not None and store.gate.judge_row(photo)[0] != "ok":
                raise fb.FeedbackError(f"{item['subject']} is under the paper's seal")
        plan.append((item, ("include_crops", crop_value(chosen, edited)) if chosen
                     else ("reject_crops", None), photo, reason))
    saved = unchanged = reasons = 0
    for item, (decision, value), photo, reason in plan:
        if photo is not None:
            code, why = ("keep_photo", None) if reason == KEEP else ("exclude_photo", reason)
            last = (photo.get("decisions") or [None])[-1]
            if not (photo["status"] == "decided" and last and last["decision"] == code
                    and (last.get("value") or None) == why):
                store.decide(photo["id"], code, person, value=why, note="recovered labels",
                             projects=projects)
                reasons += 1
        last = item["decisions"][-1] if item["decisions"] else None
        if (item["status"] == "decided" and last and last["decision"] == decision
                and (last["value"] or None) == value):
            unchanged += 1
            continue
        store.decide(item["id"], decision, person, value=value, projects=projects)
        saved += 1
    return {"saved": saved, "unchanged": unchanged, "reasons_changed": reasons,
            "included": sum(1 for p in plan if p[1][0] == "include_crops"),
            "rejected": sum(1 for p in plan if p[1][0] == "reject_crops")}


# --- the batch job: `mv feedback slip-crops` (laptop GPU; slipcrop.Detector) ------------------

def slip_photos_in_copy(conn) -> list[dict]:
    """The photos a person leaves out as a field slip in a store COPY (a sqlite3 connection,
    read-only), each {photo_key, record_key, label, slip_p, tranche, rule}, and none that
    already has a Recovered labels item there. Newest first: the latest decisions are the
    tagger's tranches."""
    queued = {r[0] for r in conn.execute("select subject from items where item_type = ?",
                                         (ITEM_TYPE,))}
    out = []
    for r in conn.execute(
            "select i.subject, i.rule, i.record_keys, i.links, i.current, d.decision, d.value "
            "from items i join decisions d on d.item_id = i.id "
            "where i.item_type = ? and d.id = (select max(x.id) from decisions x "
            "where x.item_id = i.id) order by d.id desc", (pr.ITEM_TYPE,)):
        key, rule, rks, links, label, decision, value = tuple(r)
        if decision != "exclude_photo" or value != SLIP_REASON or key in queued:
            continue
        rec = pr.record_of_item({"record_keys": json.loads(rks or "[]"),
                                 "links": json.loads(links or "[]")})
        if rec is None:
            continue
        p = slip_p(rule)
        out.append({"photo_key": key, "record_key": rec, "label": label or "", "slip_p": p,
                    "tranche": tranche_of(p), "rule": rule})
    return out


TRAINABLE = ("train", "val")


def v1_check(splits: dict[str, str], gate: "fb.SealGate"):
    """Whether the job may read a photo (the Paper Coordinator, 2026-10-26): only one a person
    left out of a v1 train or val record (`splits`: release v1's record key -> split), never a
    v1-test record before the unseal, and its record must pass the paper's seal (SealGate) as
    well. Gives check(photo) -> None when allowed, else why not."""
    def check(photo: dict) -> str | None:
        split = splits.get(photo.get("record_key"))
        if split == "test":
            return "v1 test"
        if split not in TRAINABLE:
            return "not a v1 train or val record"
        verdict = gate.judge(photo["record_key"])
        return None if verdict == "ok" else f"seal: {verdict}"
    return check


def job_photos(picks: list[dict], check) -> tuple[list[dict], dict]:
    """The picks the job may read (v1_check), and how many were refused, by reason."""
    ok, refused = [], {}
    for p in picks:
        why = check(p)
        if why is None:
            ok.append(p)
        else:
            refused[why] = refused.get(why, 0) + 1
    return ok, refused


def run_slip_crops(photos: list[dict], image, detect, out_path, check,
                   limit: int | None = None, log=print) -> dict:
    """Run the detector on each photo and append its seed line to `out_path` (a header first
    when the file is new). `image(photo)` gives the photo as a PIL image (EXIF applied) or
    None; `detect(img)` is slipcrop.Detector.detect. `check` is v1_check: a photo it refuses
    stops the run before that photo is read (job_photos should have left it out). A photo
    already in the file is skipped, so a stopped run carries on where it left off."""
    import time
    from pathlib import Path
    out_path = Path(out_path)
    done: set[str] = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8", newline="") as f:
            done = {r["photo_key"] for r in csv.DictReader(f, delimiter="\t")}
    todo = [p for p in photos if p["photo_key"] not in done][:limit]
    counts = {"photos": len(todo),
              "already_done": sum(1 for p in photos if p["photo_key"] in done),
              "with_crops": 0, "no_specimen": 0, "unreadable": 0}
    new = not out_path.exists()
    started = time.monotonic()
    with open(out_path, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SEED_COLUMNS, delimiter="\t", lineterminator="\n")
        if new:
            w.writeheader()
        for i, p in enumerate(todo, 1):
            if (why := check(p)) is not None:
                raise fb.FeedbackError(f"{p['photo_key']} may not be read ({why}): the crop job "
                                       "reads only v1 train or val photos under the seal")
            img = image(p)
            if img is None:
                counts["unreadable"] += 1
                continue
            found = detect(img)
            w.writerow(seed_row(p, found))
            f.flush()
            counts["with_crops" if found["crops"] else "no_specimen"] += 1
            if i % 50 == 0:
                log(f"  {i:,}/{len(todo):,} photos, "
                    f"{(time.monotonic() - started) / i:.1f} s a photo")
    counts["seconds"] = round(time.monotonic() - started, 1)
    return counts
