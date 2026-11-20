"""The Decided view's Show box (Steve, 2026-11-20: "Show only this category / other categories -
showing only things that remain in the bucket or have been moved elsewhere"). A decided photo in
a tab either stayed in its category (left out for the tab's reason: kept as flagged, or put
there by a person's reason) or left it (left out for another reason, or kept: Steve, a kept
photo has left the bucket too). Show combines with Outcome, and each choice's count is what the
list becomes when it is picked."""

import pytest
from test_feedback_api import as_user
from test_photo_review import photo_item, site_with_photos
from test_photo_review_outcome import keys, queue

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr


def tiles(**kw):
    return keys(pr.flagged_tiles(queue(), "slip", status="decided", **kw))


def test_still_in_this_category_is_every_photo_left_out_for_the_tabs_reason():
    assert tiles(show="stayed") == ["inat:1", "inat:4", "inat:7"]


def test_moved_to_other_categories_is_every_photo_left_out_for_another_reason_or_kept():
    assert tiles(show="left") == ["inat:2", "inat:3", "inat:6"]     # kept, habitat, blurry


def test_the_two_together_are_every_decided_photo_and_never_overlap():
    assert sorted(tiles(show="stayed") + tiles(show="left")) == tiles()
    assert not set(tiles(show="stayed")) & set(tiles(show="left"))


def test_show_combines_with_outcome():
    assert tiles(show="left", outcome="moved") == ["inat:3", "inat:6"]
    assert tiles(show="left", outcome="as_flagged") == []
    assert tiles(show="stayed", outcome="not_flagged") == ["inat:4", "inat:7"]


def test_each_shows_count_is_what_picking_it_would_list():
    counts, shows = {}, {}
    pr.flagged_tiles(queue(), "slip", status="decided", counts=counts, show_counts=shows)
    assert shows == {"stayed": 3, "left": 3}
    for code, n in shows.items():
        assert len(tiles(show=code)) == n
    # Picking an outcome narrows the Show counts; picking a Show narrows the outcome counts.
    narrowed = {}
    pr.flagged_tiles(queue(), "slip", status="decided", outcome="moved", show_counts=narrowed)
    assert narrowed == {"stayed": 0, "left": 2}
    by_outcome = {}
    pr.flagged_tiles(queue(), "slip", status="decided", show="stayed", counts=by_outcome)
    assert by_outcome == {"as_flagged": 1, "included": 0, "moved": 0, "not_flagged": 2}


def test_records_are_kept_by_a_photo_that_matches_show_and_outcome_together():
    def recs(**kw):
        out = pr.cards(queue(), "slip", status="decided", **kw)
        return [r["record_key"] for r in out["records"]], out["shows"]
    got, shows = recs(show="stayed")
    assert got == ["10", "11", "13"] and shows == {"stayed": 3, "left": 2}
    assert recs(show="left")[0] == ["10", "11"]
    # Record 11 has a moved photo (3) and a stayed one (4); no single photo is both.
    assert recs(show="stayed", outcome="moved")[0] == []
    every = pr.cards(queue(), "slip", status="decided")["tabs"]
    assert pr.cards(queue(), "slip", status="decided", show="left")["tabs"] == every


def test_an_unknown_show_is_refused():
    with pytest.raises(fb.FeedbackError, match="show"):
        pr.cards(queue(), "slip", show="nope")
    with pytest.raises(fb.FeedbackError, match="show"):
        pr.flagged_tiles(queue(), "slip", show="nope")


def test_the_api_shows_what_stayed_and_what_moved_after_real_decisions(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([photo_item("inat:2000", "102", "R9 v2 slip p=0.91 (high)"),
                 photo_item("inat:2001", "102", "R9 v2 slip p=0.81 (high)"),
                 photo_item("inat:2002", "103", "R9 v2 slip p=0.71 (high)")], "slips")
    as_user(c, "admin-1")
    ids = {t["photo_key"]: t["item_id"] for t in
           c.get("/api/feedback/photo-review/flagged", params={"tab": "slip"}).json()["tiles"]}
    assert c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:2000"], "decision": "exclude", "reason": "slip"},
        {"item_id": ids["inat:2001"], "decision": "keep"},
        {"item_id": ids["inat:2002"], "decision": "exclude", "reason": "multiple_species"}]}
    ).status_code == 200

    def flagged(**p):
        return c.get("/api/feedback/photo-review/flagged",
                     params={"tab": "slip", "status": "decided", **p}).json()
    assert flagged()["shows"] == {"stayed": 1, "left": 2}
    assert [t["photo_key"] for t in flagged(show="stayed")["tiles"]] == ["inat:2000"]
    assert sorted(t["photo_key"] for t in flagged(show="left")["tiles"]) == ["inat:2001", "inat:2002"]
    rec = c.get("/api/feedback/photo-review",
                params={"tab": "slip", "status": "decided", "show": "left"}).json()
    assert [x["record_key"] for x in rec["records"]] == ["102", "103"]
    assert rec["shows"] == {"stayed": 1, "left": 2}
    for path in ("/api/feedback/photo-review/flagged", "/api/feedback/photo-review"):
        assert c.get(path, params={"show": "nope"}).status_code == 400
    # A read-only visitor gets the same filter.
    c.cookies.clear()
    assert [t["photo_key"] for t in flagged(show="stayed")["tiles"]] == ["inat:2000"]
