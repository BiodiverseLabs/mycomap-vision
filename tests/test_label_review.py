"""The Human Feedback label review (label_review.py): the blind "Photos unlike the label" queue
as one card per record. Until a card is decided it shows the record, its label and its photos
only: no Vision answer, rule or evidence, no blind mark, and an order that does not follow
seeding (the blind sample is seeded last). One outcome per record: leave out the chosen
photos (with a reason), a hard example, or a rename."""

import pytest
from test_feedback_api import as_user, make_site

from mycomap_vision import feedback as fb
from mycomap_vision import label_review as lr

SECRET_WORDS = ("Vision:", "M1 truth", "blind", "v1_val_outcome")


def flagged(subject, label="A x", projects=("p1",)):
    return {"item_type": "photos_unlike_label", "subject": subject, "rule": "M1 truth ranked >300",
            "record_keys": [subject], "current": label, "proposed": "",
            "projects": list(projects), "model_flagged": True,
            "evidence": {"evidence": "Vision: B y 0.71; C z 0.40", "in_v1": "val",
                         "v1_val_outcome": "miss"}}


# --- the order of a blind queue ---------------------------------------------------------------

def test_a_blind_queue_is_not_listed_in_seeding_order(tmp_path):
    store = fb.Store(tmp_path / "f.sqlite")
    store.seed([flagged(str(500 + i)) for i in range(8)], "flagged")
    store.seed(fb.blind_sample([{"record_key": str(900 + i), "label": "A x"} for i in range(4)],
                               "photos_unlike_label", 4, seed=1), "blind")
    shown = [i["subject"] for i in store.items("photos_unlike_label", limit=100)["items"]]
    assert shown == sorted(shown, key=fb.blind_order)
    blind_at = [shown.index(str(900 + i)) for i in range(4)]
    assert sorted(blind_at) != [8, 9, 10, 11]                 # not all at the end
    cards = lr.cards(store.items_of("photos_unlike_label"))["records"]
    assert [c["record_key"] for c in cards] == shown
    # A queue that is not blind keeps its order (most records first, then oldest).
    store.seed([{"item_type": "group_label", "subject": s, "rule": "R2", "records": 1}
                for s in ("z group", "a group")], "groups")
    assert [i["subject"] for i in store.items("group_label")["items"]] == ["z group", "a group"]


# --- what a card shows ------------------------------------------------------------------------

def site_with_queue(conn, tmp_path):
    c, fstore, org = make_site(conn, tmp_path)
    fstore.seed([flagged("102")], "more")
    fstore.seed([{**b, "projects": ["p1"]} for b in fb.blind_sample(
        [{"record_key": "101", "label": "A x"}], "photos_unlike_label", 1, seed=3)], "blind")
    return c, fstore


def test_an_open_card_shows_the_record_label_and_photos_and_nothing_of_visions_answer(conn,
                                                                                    tmp_path):
    c, _ = site_with_queue(conn, tmp_path)
    as_user(c, "admin-1")
    r = c.get("/api/feedback/label-review")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and {o["code"] for o in body["outcomes"]} == set(lr.OUTCOMES)
    text = r.text
    for word in SECRET_WORDS:
        assert word not in text
    shapes = {frozenset(k for k in card if k != "photos") for card in body["records"]}
    assert len(shapes) == 1                                   # blind and flagged look the same
    card = next(x for x in body["records"] if x["record_key"] == "102")
    assert card["masked"] and card["label"] == "A x"
    assert [p["photo_key"] for p in card["photos"]] == ["inat:1002"]
    assert card["photos"][0]["shown"] == "public"


def test_outcomes_are_checked_and_a_decided_card_shows_what_vision_said(conn, tmp_path):
    c, fstore = site_with_queue(conn, tmp_path)
    as_user(c, "leader-1")
    item = next(i for i in fstore.items_of("photos_unlike_label") if i["subject"] == "102")
    url = f"/api/feedback/label-review/{item['id']}/decision"
    leave_out = [{"photo_key": "inat:1002", "decision": "exclude"}]
    for body, why in (
            ({"outcome": "exclude_photos", "photos": [], "reason": "no_specimen"}, "choose"),
            ({"outcome": "exclude_photos", "photos": leave_out}, "say why"),
            ({"outcome": "hard_example", "photos": leave_out}, "its own outcome"),
            ({"outcome": "rename"}, "needs a value"),
            ({"outcome": "maybe"}, "outcome is one of"),
            ({"outcome": "hard_example",
              "photos": [{"photo_key": "inat:1000", "decision": "keep"}]}, "not a photo")):
        r = c.post(url, json=body)
        assert r.status_code == 400 and why in r.json()["detail"], body
    r = c.post(url, json={"outcome": "exclude_photos", "photos": leave_out,
                          "reason": "other_organism", "note": "a slime mould"})
    assert r.status_code == 200
    card = r.json()["record"]
    assert not card["masked"] and "Vision: B y" in card["evidence"]["evidence"]
    assert card["decision"]["photo_keys"] == ["inat:1002"] and card["photos"][0]["excluded"]
    assert card["decision"]["note"] == "reason: other_organism; a slime mould"
    assert fstore.excluded_photos() == {"inat:1002"}
    assert "102" not in {x["record_key"] for x in c.get("/api/feedback/label-review").json()
                         ["records"]}
    decided = c.get("/api/feedback/label-review", params={"status": "decided"}).json()
    assert [x["record_key"] for x in decided["records"]] == ["102"]
    # A rename is stored with the name and goes to .org.
    other = next(i for i in fstore.items_of("photos_unlike_label") if i["subject"] == "100")
    r = c.post(f"/api/feedback/label-review/{other['id']}/decision",
               json={"outcome": "rename", "value": "B y"})
    assert r.status_code == 200
    assert {(x["subject"], x["value"]) for x in fstore.changes(applies="org")} == {("100", "B y")}


def test_who_may_review_and_which_items_are_cards(conn, tmp_path):
    c, fstore = site_with_queue(conn, tmp_path)
    c.cookies.clear()
    assert c.get("/api/feedback/label-review").status_code == 401
    as_user(c, "member-1")                                    # another project's contributor
    assert c.get("/api/feedback/label-review").json()["records"] == []
    item = next(i for i in fstore.items_of("photos_unlike_label") if i["subject"] == "102")
    assert c.post(f"/api/feedback/label-review/{item['id']}/decision",
                  json={"outcome": "hard_example"}).status_code == 404
    as_user(c, "admin-1")
    group = fstore.items("group_label")["items"][0]["id"]
    assert c.post(f"/api/feedback/label-review/{group}/decision",
                  json={"outcome": "hard_example"}).status_code == 404
    with pytest.raises(fb.FeedbackError):
        lr.cards([], status="later")
