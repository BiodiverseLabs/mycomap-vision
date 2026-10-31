"""The photo review's Outcome filter (Steve, 2026-10-30: "filter for things that were originally
flagged but then changed to included or another category"). A decided photo's outcome in a tab
comes from the flags it was queued with for that tab (never the tab its exclusion reason adds,
never a person's own "reviewer:" item) and its latest decision:

- Kept as flagged: left out for this tab's reason;
- Changed to included: flagged for this tab, but the person kept it;
- Moved to another category: flagged for this tab, left out for another reason;
- Not flagged for this tab: in the tab only by the reason a person gave.

The filter works with the tabs, Open/Decided, sort, confidence, search and paging, never changes
the tab counts, and each option's count is what the list's total becomes when it is picked."""

import pytest
from test_feedback_api import as_user
from test_photo_review import photo_item, site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr

INAT = "https://www.inaturalist.org/observations/"


def item(i, record, rule, status="decided", decision=None, reason=None, label="A x"):
    last = None
    if decision:
        last = {"decision": decision, "value": reason, "by_name": "S"}
    return {"id": i, "subject": f"inat:{i}", "rule": rule, "status": status,
            "links": [INAT + record], "current": label, "last_decision": last}


SLIP = "R9 v2 slip p=0.90 (high)"


def queue():
    """Record 10: a slip kept as flagged (1), a slip changed to included (2). Record 11: a slip
    left out as habitat (3), a micrograph-probe photo left out as a slip (4). Record 12: an open
    slip (5), a slip left out as blurry (6, medium band). Record 13: a photo a person left out
    as a slip with no flag at all (7), and a photo tagger v2 read as a specimen, kept (8)."""
    return [
        item(1, "10", SLIP, decision="exclude_photo", reason="slip"),
        item(2, "10", "R9 v2 slip p=0.80 (high)", decision="keep_photo"),
        item(3, "11", "R9 v2 slip p=0.70 (high)", decision="exclude_photo", reason="habitat",
             label="Habitat one"),
        item(4, "11", "R6 micrograph probe p=0.60", decision="exclude_photo", reason="slip",
             label="Habitat one"),
        item(5, "12", "R9 v2 slip p=0.95 (high)", status="open"),
        item(6, "12", "R9 v2 slip p=0.60 (medium)", decision="exclude_photo", reason="blurry"),
        item(7, "13", "reviewer: left out in photo review", decision="exclude_photo",
             reason="slip"),
        item(8, "13", "R9 v2 specimen p=0.90 (high)", decision="keep_photo"),
    ]


def keys(tiles):
    return sorted(t["photo_key"] for t in tiles)


# --- each outcome, photo by photo -------------------------------------------------------------

def test_each_decided_photo_on_the_slips_tab_has_the_outcome_of_its_original_flag():
    tiles = {t["photo_key"]: t["outcome"] for t in
             pr.flagged_tiles(queue(), "slip", status="decided")}
    assert tiles == {"inat:1": "as_flagged",     # flagged slip, left out as a slip
                     "inat:2": "included",       # flagged slip, kept
                     "inat:3": "moved",          # flagged slip, left out as habitat
                     "inat:4": "not_flagged",    # a micrograph flag: here by its reason only
                     "inat:6": "moved",          # flagged slip, left out as blurry
                     "inat:7": "not_flagged"}    # a person's own item, no flag


def test_a_photo_flagged_for_slips_but_excluded_as_habitat_is_moved_on_the_slips_tab():
    moved = pr.flagged_tiles(queue(), "slip", status="decided", outcome="moved")
    assert "inat:3" in keys(moved)
    for other in ("as_flagged", "included", "not_flagged"):
        assert "inat:3" not in keys(pr.flagged_tiles(queue(), "slip", status="decided",
                                                     outcome=other))
    # On the habitat tab it was never flagged: there only by the reason the person gave.
    assert keys(pr.flagged_tiles(queue(), "habitat", status="decided",
                                 outcome="not_flagged")) == ["inat:3"]
    assert pr.flagged_tiles(queue(), "habitat", status="decided", outcome="moved") == []


def test_each_outcome_keeps_exactly_its_photos():
    def got(o):
        return keys(pr.flagged_tiles(queue(), "slip", status="decided", outcome=o))
    assert got("as_flagged") == ["inat:1"]
    assert got("included") == ["inat:2"]
    assert got("moved") == ["inat:3", "inat:6"]
    assert got("not_flagged") == ["inat:4", "inat:7"]
    assert got("all") == ["inat:1", "inat:2", "inat:3", "inat:4", "inat:6", "inat:7"]


def test_the_original_flag_ignores_the_tab_an_exclusion_reason_adds():
    four = next(i for i in queue() if i["id"] == 4)
    assert [f["tab"] for f in pr.tab_flags(four)] == ["micrograph", "slip"]
    assert [f["tab"] for f in pr.original_flags(four)] == ["micrograph"]
    assert pr.item_outcome(four, "micrograph") == "moved"
    assert pr.item_outcome(four, "slip") == "not_flagged"
    seven = next(i for i in queue() if i["id"] == 7)
    assert pr.original_flags(seven) == []                 # a person's own item: no flag


def test_a_specimen_flag_kept_is_kept_as_flagged_and_left_out_is_moved():
    eight = next(i for i in queue() if i["id"] == 8)
    assert pr.item_outcome(eight, "specimen") == "as_flagged"
    out = dict(eight, last_decision={"decision": "exclude_photo", "value": "slip"})
    assert pr.item_outcome(out, "specimen") == "moved"


def test_in_all_any_of_its_flags_counts_and_open_photos_have_no_outcome():
    by = {i["id"]: i for i in queue()}
    assert pr.item_outcome(by[1], "all") == "as_flagged"
    assert pr.item_outcome(by[3], "all") == "moved"
    assert pr.item_outcome(by[2], "all") == "included"
    assert pr.item_outcome(by[7], "all") == "not_flagged"
    assert pr.item_outcome(by[5], "slip") is None
    # Older probes and views: a flag for the reason's tab among several is kept as flagged.
    both = item(9, "14", "R6 micrograph probe p=0.6; R6 view habitat p=0.3",
                decision="exclude_photo", reason="habitat")
    assert pr.item_outcome(both, "all") == "as_flagged"
    assert pr.item_outcome(both, "micrograph") == "moved"


def test_a_retired_reason_counts_as_its_successor():
    old = item(9, "14", "drawing probe p=0.99", decision="exclude_photo", reason="drawing_micro")
    assert pr.item_outcome(old, "drawing") == "as_flagged"


# --- with the page's other controls -----------------------------------------------------------

def test_the_filter_keeps_sort_band_search_and_the_open_view_working():
    moved = pr.flagged_tiles(queue(), "slip", status="decided", outcome="moved")
    assert [t["photo_key"] for t in moved] == ["inat:3", "inat:6"]          # p 0.70, 0.60
    assert [t["photo_key"] for t in pr.flagged_tiles(queue(), "slip", status="decided",
                                                     outcome="moved", sort="p_asc")
            ] == ["inat:6", "inat:3"]
    assert keys(pr.flagged_tiles(queue(), "slip", status="decided", outcome="moved",
                                 band="medium")) == ["inat:6"]
    assert keys(pr.flagged_tiles(queue(), "slip", status="decided", outcome="moved",
                                 query="habitat one")) == ["inat:3"]
    # Open photos have no outcome yet.
    assert pr.flagged_tiles(queue(), "slip", status="open", outcome="moved") == []
    assert keys(pr.flagged_tiles(queue(), "slip", status="open")) == ["inat:5"]


def test_the_summary_counts_are_what_each_option_would_show():
    counts = {}
    pr.flagged_tiles(queue(), "slip", status="decided", outcome="included", counts=counts)
    assert counts == {"as_flagged": 1, "included": 1, "moved": 2, "not_flagged": 2}
    for code, n in counts.items():
        assert len(pr.flagged_tiles(queue(), "slip", status="decided", outcome=code)) == n
    banded = {}
    pr.flagged_tiles(queue(), "slip", status="decided", band="medium", counts=banded)
    assert banded == {"as_flagged": 0, "included": 0, "moved": 1, "not_flagged": 0}
    none_open = {}
    pr.flagged_tiles(queue(), "slip", status="open", counts=none_open)
    assert set(none_open.values()) == {0}


def test_records_are_kept_by_any_photo_in_the_tab_with_the_outcome():
    def recs(o, **kw):
        return [r["record_key"] for r in
                pr.cards(queue(), "slip", status="decided", outcome=o, **kw)["records"]]
    assert recs("all") == ["10", "11", "13"]       # 12 still has an open slip photo
    assert recs("as_flagged") == ["10"]
    assert recs("included") == ["10"]
    assert recs("moved") == ["11"]
    assert recs("not_flagged") == ["11", "13"]
    assert recs("moved", min_flagged=2) == ["11"] and recs("included", min_flagged=3) == []
    out = pr.cards(queue(), "slip", status="decided", outcome="moved")
    assert out["outcomes"] == {"as_flagged": 1, "included": 1, "moved": 1, "not_flagged": 2}
    photo = {p["photo_key"]: p["outcome"] for p in out["records"][0]["flagged"]}
    assert photo == {"inat:3": "moved", "inat:4": "not_flagged"}


def test_the_tab_counts_never_change_with_the_outcome():
    every = pr.cards(queue(), "slip", status="decided")["tabs"]
    for code in pr.OUTCOME_CODES:
        assert pr.cards(queue(), "slip", status="decided", outcome=code)["tabs"] == every


def test_an_unknown_outcome_is_refused():
    with pytest.raises(fb.FeedbackError, match="outcome"):
        pr.cards(queue(), "slip", outcome="nope")
    with pytest.raises(fb.FeedbackError, match="outcome"):
        pr.flagged_tiles(queue(), "slip", outcome="nope")


# --- over the real app ------------------------------------------------------------------------

def test_the_api_filters_by_outcome_after_real_decisions_and_counts_each(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([photo_item("inat:2000", "102", "R9 v2 slip p=0.91 (high)"),
                 photo_item("inat:2001", "102", "R9 v2 slip p=0.81 (high)"),
                 photo_item("inat:2002", "103", "R9 v2 slip p=0.71 (high)")], "slips")
    as_user(c, "admin-1")
    ids = {t["photo_key"]: t["item_id"] for t in
           c.get("/api/feedback/photo-review/flagged", params={"tab": "slip"}).json()["tiles"]}
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:2000"], "decision": "exclude", "reason": "slip"},
        {"item_id": ids["inat:2001"], "decision": "exclude", "reason": "slip"},
        {"item_id": ids["inat:2002"], "decision": "exclude", "reason": "habitat"}]})
    assert r.status_code == 200
    # The person changes their mind on 2001: kept. The latest decision is the outcome.
    assert c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:2001"], "decision": "keep"}]}).status_code == 200

    def flagged(**p):
        return c.get("/api/feedback/photo-review/flagged",
                     params={"tab": "slip", "status": "decided", **p}).json()
    every = flagged()
    assert every["outcomes"] == {"as_flagged": 1, "included": 1, "moved": 1, "not_flagged": 0}
    assert {t["photo_key"]: t["outcome"] for t in every["tiles"]} == {
        "inat:2000": "as_flagged", "inat:2001": "included", "inat:2002": "moved"}
    assert [t["photo_key"] for t in flagged(outcome="moved")["tiles"]] == ["inat:2002"]
    assert flagged(outcome="moved")["total"] == 1
    assert [t["photo_key"] for t in flagged(outcome="included")["tiles"]] == ["inat:2001"]
    page = flagged(outcome="all", limit=1, offset=1)
    assert page["total"] == 3 and len(page["tiles"]) == 1

    def records(**p):
        return c.get("/api/feedback/photo-review",
                     params={"tab": "slip", "status": "decided", **p}).json()
    rec = records(outcome="moved")
    assert [x["record_key"] for x in rec["records"]] == ["103"]
    assert rec["outcomes"] == {"as_flagged": 1, "included": 1, "moved": 1, "not_flagged": 0}
    card = rec["records"][0]
    assert next(p for p in card["photos"] if p["photo_key"] == "inat:2002")["outcome"] == "moved"
    assert rec["tabs"] == records()["tabs"]                 # the tab counts never move
    assert [x["record_key"] for x in records(outcome="included")["records"]] == ["102"]
    # On the habitat tab the slip left out as habitat was never flagged there.
    hab = c.get("/api/feedback/photo-review/flagged",
                params={"tab": "habitat", "status": "decided", "outcome": "not_flagged"}).json()
    assert [t["photo_key"] for t in hab["tiles"]] == ["inat:2002"]
    for path in ("/api/feedback/photo-review/flagged", "/api/feedback/photo-review"):
        assert c.get(path, params={"outcome": "nope"}).status_code == 400


def test_a_read_only_viewer_gets_the_same_filter(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    item_id = next(i["id"] for i in fstore.items_of("non_specimen_photo")
                   if i["subject"] == "inat:1000")
    fstore.decide(item_id, "exclude_photo", {"sub": "admin-1", "name": "Steve"}, value="slip")
    c.cookies.clear()
    body = c.get("/api/feedback/photo-review/flagged",
                 params={"tab": "micrograph", "status": "decided", "outcome": "moved"}).json()
    assert [t["photo_key"] for t in body["tiles"]] == ["inat:1000"]
    assert body["outcomes"]["moved"] == 1
