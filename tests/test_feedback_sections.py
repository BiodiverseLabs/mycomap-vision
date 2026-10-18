"""Human Feedback in two parts (Steve, 2026-10-18): data cleanup (mechanical fixes a rule
picked) and data validation (is the record's label right?). And "Show Vision's answer" on a
blind item: it gives Vision's top 5 for the record asked afresh, the same way for a flagged and
a check-sample item, records who looked, and keeps decisions made after looking out of the
blind measure."""

from test_feedback_api import as_user, make_site

from mycomap_vision import feedback as fb
from mycomap_vision import label_review as lr


def test_every_queue_is_in_one_of_the_two_sections_and_the_blind_ones_are_validation():
    assert set(fb.SECTION_OF) == set(fb.TYPES)
    assert set(fb.SECTION_OF.values()) == set(fb.SECTIONS) == {"cleanup", "validation"}
    assert all(fb.SECTION_OF[t] == "validation" for t in fb.BLIND_TYPES)
    assert fb.SECTION_OF["non_specimen_photo"] == "cleanup"
    by_type = {t["type"]: t for t in fb.type_catalogue()}
    assert by_type["photos_unlike_label"]["section"] == "validation"
    assert by_type["photos_unlike_label"]["blind"] and not by_type["group_label"]["blind"]


def site_with_blind_queue(conn, tmp_path):
    c, fstore, org = make_site(conn, tmp_path)
    fstore.seed([{**b, "projects": ["p1"]} for b in fb.blind_sample(
        [{"record_key": "104", "label": "B y"}], "photos_unlike_label", 1, seed=3)], "blind")
    return c, fstore


def item(fstore, subject):
    return next(i for i in fstore.items_of("photos_unlike_label") if i["subject"] == subject)


def test_the_summary_says_each_queues_section(conn, tmp_path):
    c, _ = site_with_blind_queue(conn, tmp_path)
    as_user(c, "admin-1")
    by_type = {s["type"]: s for s in c.get("/api/feedback/summary").json()}
    assert by_type["photos_unlike_label"]["section"] == "validation"
    assert by_type["group_label"]["section"] == "validation"


def test_showing_visions_answer_looks_the_same_for_flagged_and_check_sample_items(conn,
                                                                                tmp_path):
    c, fstore = site_with_blind_queue(conn, tmp_path)
    c.cookies.clear()
    flagged, blind = item(fstore, "100"), item(fstore, "104")
    assert c.post(f"/api/feedback/items/{flagged['id']}/reveal").status_code == 401
    as_user(c, "leader-1")
    answers = []
    for it in (flagged, blind):
        r = c.post(f"/api/feedback/items/{it['id']}/reveal")
        assert r.status_code == 200
        body = r.json()
        assert set(body) == {"at", "answer"}
        assert set(body["answer"]) == {"available", "top5", "reason"}
        for word in ("blind", "M1", "rule", "evidence", "outcome", "seed"):
            assert word not in r.text
        answers.append(body["answer"])
    assert answers[0]["available"] and answers[0]["top5"][0][0] in ("A x", "B y")
    assert answers[1]["available"] and answers[1]["top5"]
    # The card remembers it for this person, and only for them.
    cards = c.get("/api/feedback/label-review").json()["records"]
    assert all(card["revealed"] for card in cards if card["record_key"] in ("100", "104"))
    as_user(c, "admin-1")
    cards = c.get("/api/feedback/label-review").json()["records"]
    assert all(card["revealed"] is None for card in cards)
    detail = c.get(f"/api/feedback/items/{flagged['id']}").json()
    assert detail["masked"] and detail["revealed"] is None


def test_decisions_after_looking_are_marked_and_others_are_not(conn, tmp_path):
    c, fstore = site_with_blind_queue(conn, tmp_path)
    flagged, blind = item(fstore, "100"), item(fstore, "104")
    as_user(c, "leader-1")
    c.post(f"/api/feedback/items/{flagged['id']}/reveal")
    for it in (flagged, blind):
        assert c.post(f"/api/feedback/label-review/{it['id']}/decision",
                      json={"outcome": "hard_example"}).status_code == 200
    seen = {ch["subject"]: ch["seen_answer"] for ch in fstore.changes(applies="note")}
    assert seen == {"100": True, "104": False}
    as_user(c, "admin-1")
    tsv = c.get("/api/feedback/changes.tsv", params={"applies": "note"}).text.splitlines()
    head = tsv[0].split("\t")
    rows = {r.split("\t")[head.index("subject")]: r.split("\t")[head.index("seen_answer")]
            for r in tsv[1:]}
    assert rows == {"100": "1", "104": "0"}
    # Another person's look does not mark this person's decision.
    fstore.reveal(blind["id"], {"sub": "admin-1"}, {"available": False})
    assert {ch["subject"]: ch["seen_answer"] for ch in fstore.changes(applies="note")} == seen


def test_only_hidden_answers_can_be_revealed(conn, tmp_path):
    c, fstore = site_with_blind_queue(conn, tmp_path)
    as_user(c, "admin-1")
    group = fstore.items("group_label")["items"][0]["id"]
    assert c.post(f"/api/feedback/items/{group}/reveal").status_code == 409
    flagged = item(fstore, "100")
    c.post(f"/api/feedback/label-review/{flagged['id']}/decision", json={"outcome": "hard_example"})
    assert c.post(f"/api/feedback/items/{flagged['id']}/reveal").status_code == 409
    assert lr.ITEM_TYPE in fb.BLIND_TYPES
