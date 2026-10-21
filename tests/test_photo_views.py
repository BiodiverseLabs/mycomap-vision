"""Human Feedback "Photo views" (photo_views.py; Steve, 2026-10-21): tags for what each photo
shows, suggested by the photo-type tagger and confirmed or corrected by a reviewer. A photo's
tags are its latest "confirm_views" decision (the whole set); "no_specimen" sends it to the
exclusion review instead of deciding anything here; a page is saved whole or not at all."""

import json

import pytest
from test_feedback_api import as_user
from test_photo_review import site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr
from mycomap_vision import photo_views as pv

INAT = "https://www.inaturalist.org/observations/"
ME = {"sub": "admin-1", "name": "A"}


def view_item(key, record, rule, projects=("p1",)):
    return {"item_type": "photo_view", "subject": key, "rule": rule, "granularity": "photo",
            "links": [INAT + record], "current": "A x", "projects": list(projects)}


@pytest.fixture
def store(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([view_item("inat:1", "10", "R8 gills p=0.97; R8 in_situ p=0.99"),
            view_item("inat:2", "10", "R8 cap_top p=0.95"),
            view_item("inat:3", "11", "R8 several p=0.98; R8 nonsense p=0.99")], "views")
    return s


def ids(s):
    return {i["subject"]: i["id"] for i in s.items_of(pv.ITEM_TYPE)}


def test_a_rule_suggests_known_views_only_and_tiles_sort_by_the_tabs_view():
    assert pv.suggestions("R8 gills p=0.97; R8 in_situ p=0.99; R8 nonsense p=0.5") == {
        "gills": 0.97, "in_situ": 0.99}
    assert set(pv.SUGGESTED) <= set(pv.VIEW_CODES)


def test_tabs_hold_photos_suggested_for_a_view_or_confirmed_with_it(store):
    t = pv.tiles(store.items_of(pv.ITEM_TYPE), "gills")
    assert [x["photo_key"] for x in t] == ["inat:1"]
    assert [s["view"] for s in t[0]["suggested"]] == ["in_situ", "gills"]       # strongest first
    store.decide(ids(store)["inat:2"], "confirm_views", ME, value=json.dumps(["gills"]))
    done = pv.tiles(store.items_of(pv.ITEM_TYPE), "gills", status="decided")
    assert [x["photo_key"] for x in done] == ["inat:2"] and done[0]["confirmed"] == ["gills"]
    counts = {c["code"]: (c["open"], c["decided"]) for c in pv.tab_counts(store.items_of(pv.ITEM_TYPE))}
    assert counts["all"] == (2, 1) and counts["gills"] == (1, 1) and counts["cap_top"] == (0, 1)
    with pytest.raises(fb.FeedbackError):
        pv.tiles([], "nope")


def test_a_page_save_writes_each_photos_whole_tag_set_and_corrections_replace_it(store):
    i = ids(store)
    out = pv.batch_decide(store, [{"item_id": i["inat:1"], "views": ["in_situ", "gills", "gills"]},
                                  {"item_id": i["inat:2"], "views": []}], ME)
    assert out == {"saved": 2, "unchanged": 0, "sent_to_exclusion_review": 0}
    by = {x["subject"]: x for x in store.items_of(pv.ITEM_TYPE)}
    assert pv.confirmed(by["inat:1"]) == ["gills", "in_situ"]       # one order, no repeats
    assert pv.confirmed(by["inat:2"]) == [] and by["inat:2"]["status"] == "decided"
    again = pv.batch_decide(store, [{"item_id": i["inat:1"], "views": ["gills", "in_situ"]}], ME)
    assert (again["saved"], again["unchanged"]) == (0, 1)                # idempotent
    pv.batch_decide(store, [{"item_id": i["inat:1"], "views": ["cap_top"]}], ME)
    assert pv.confirmed(next(x for x in store.items_of(pv.ITEM_TYPE)
                             if x["subject"] == "inat:1")) == ["cap_top"]


def test_a_bad_entry_writes_nothing_from_the_page(store):
    i = ids(store)
    for bad in ({"item_id": i["inat:2"], "views": ["wings"]},          # not a view
                {"item_id": i["inat:2"], "views": "gills"},             # not a list
                {"item_id": 99999, "views": []}):                       # no such item
        with pytest.raises(fb.FeedbackError):
            pv.batch_decide(store, [{"item_id": i["inat:1"], "views": ["gills"]}, bad], ME)
    assert store.conn.execute("select count(*) from decisions").fetchone()[0] == 0
    store.seed([{"item_type": "non_specimen_photo", "subject": "inat:7", "rule": "R6",
                 "links": [INAT + "12"]}], "other")
    other = store.items_of("non_specimen_photo")[0]["id"]
    with pytest.raises(fb.FeedbackError, match="not a photo of the Photo views queue"):
        pv.batch_decide(store, [{"item_id": other, "views": []}], ME)


def test_no_specimen_sends_the_photo_to_the_exclusion_review_and_excludes_nothing(store):
    i = ids(store)
    out = pv.batch_decide(store, [{"item_id": i["inat:3"], "no_specimen": True}], ME)
    assert out["sent_to_exclusion_review"] == 1
    sent = store.items_of(pr.ITEM_TYPE)
    assert [(x["subject"], x["status"], x["rule"]) for x in sent] == [
        ("inat:3", "open", pv.NO_SPECIMEN_RULE)]
    assert pr.record_of_item(sent[0]) == "11"
    assert store.excluded_photos() == set()                       # the exclusion is decided there
    assert pv.batch_decide(store, [{"item_id": i["inat:3"], "no_specimen": True}], ME)["unchanged"] == 1


def test_the_queue_is_its_own_section_and_its_decisions_apply_nowhere():
    t = fb.TYPES[pv.ITEM_TYPE]
    assert fb.SECTION_OF[pv.ITEM_TYPE] == "views" and "views" in fb.SECTIONS
    assert {o.code: o.applies for o in t.options} == {"confirm_views": "note", "no_specimen": "note"}


def test_the_api_pages_tiles_and_saves_a_page_for_reviewers_with_access(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([view_item("inat:1000", "100", "R8 gills p=0.97"),
                 view_item("inat:1100", "100", "R8 in_situ p=0.99")], "views")
    as_user(c, "admin-1")
    r = c.get("/api/feedback/photo-views", params={"tab": "all"})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2 and [t["photo_key"] for t in body["tiles"]] == ["inat:1100", "inat:1000"]
    assert {v["code"] for v in body["views"] if v["suggested"]} == set(pv.SUGGESTED)
    assert body["tiles"][1]["shown"] == "public" and body["tiles"][0]["shown"] == "private"
    ids_ = {t["photo_key"]: t["item_id"] for t in body["tiles"]}
    saved = c.post("/api/feedback/photo-views/batch", json={"decisions": [
        {"item_id": ids_["inat:1000"], "views": ["gills", "stem_base"]}]})
    assert saved.status_code == 200 and saved.json()["saved"] == 1
    bad = c.post("/api/feedback/photo-views/batch", json={"decisions": [
        {"item_id": ids_["inat:1100"], "views": ["wings"]}]})
    assert bad.status_code == 400
    done = c.get("/api/feedback/photo-views", params={"tab": "stem_base", "status": "decided"}).json()
    assert [t["confirmed"] for t in done["tiles"]] == [["gills", "stem_base"]]
    as_user(c, "member-1")                                     # another project's contributor
    assert c.get("/api/feedback/photo-views").json()["total"] == 0
    assert c.post("/api/feedback/photo-views/batch", json={"decisions": [
        {"item_id": ids_["inat:1100"], "views": []}]}).status_code == 400
