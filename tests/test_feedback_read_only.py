"""Human Feedback, read-only to anyone (Steve, 2026-10-22): signed out, or signed in without a
reviewer's role, a person browses the queues, the Summary and record views and item detail,
and changes nothing. They see only items of no project, each decision's outcome and date but
never who made it or their note, blind items still blind, and only photos anyone may see.
Reviewers keep exactly what they had (test_feedback_api.py, test_photo_review.py)."""

import json

import pytest
from test_feedback_api import as_user, make_site
from test_photo_review import photo_item
from test_signin import gated_site_with_pages

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_public as fp

# Keys (as JSON writes them) and values that would say who decided, or what they wrote.
PERSONAL = ('"by_user":', '"by_name":', '"by":', '"note":', '"seen_answer":', '"applied_by":',
            '"revealed":', "admin-1", "Reviewer Ann", "my private note")
READS = ("/api/feedback/me", "/api/feedback/catalogue", "/api/feedback/summary",
         "/api/feedback/photo-review/flagged", "/api/feedback/photo-review",
         "/api/feedback/photo-views", "/api/feedback/label-review",
         "/api/feedback/items?type=non_specimen_photo",
         "/api/feedback/items?type=non_specimen_photo&status=decided")


def public_site(conn, tmp_path):
    """make_site's two project items, plus photo items of no project: record 100's CC photo
    1000 and its all-rights-reserved 1100 (an owner nobody asked), and a blind label check of
    no project. An admin has decided the 1000 item, with a note."""
    c, fstore, org = make_site(conn, tmp_path)
    row = conn.execute("select source_url from photos where photo_id = 1000").fetchone()
    conn.execute("insert into photos (photo_id, source, license_class, license_code, "
                 "owner_user_id, source_url, first_seen_at, license_checked_at, status) "
                 "values (1100, 'inat', 'arr', '', 55, ?, 'now', 'now', 'done')",
                 (row[0].replace("1000", "1100"),))
    conn.execute("insert into observation_photos values ('100', 1100, 1)")
    path = conn.execute("select store, path from photo_copies where photo_id = 1000").fetchone()
    conn.execute("insert into photo_copies (photo_id, store, size, path) values (1100, ?, "
                 "'large', ?)", path)
    conn.commit()
    fstore.seed([photo_item("inat:1000", "100", "R6 micrograph probe p=0.80", projects=()),
                 photo_item("inat:1100", "100", "R6 view habitat p=0.40", projects=()),
                 photo_item("inat:1002", "102", "R6 micrograph probe p=0.70", projects=("p1",)),
                 {"item_type": "photos_unlike_label", "subject": "104", "rule": "M1",
                  "record_keys": ["104"], "current": "B y", "projects": []}], "public")
    decided = item(fstore, "non_specimen_photo", "inat:1000")
    fstore.decide(decided, "exclude_photo", {"sub": "admin-1", "name": "Reviewer Ann"},
                  value="micrograph", note="my private note")
    c.cookies.clear()
    return c, fstore, org


def item(fstore, item_type, subject):
    return next(i["id"] for i in fstore.items_of(item_type) if i["subject"] == subject)


@pytest.fixture
def site(conn, tmp_path):
    return public_site(conn, tmp_path)


def test_a_signed_out_visitor_browses_every_read_and_no_reviewer_detail(site):
    c, fstore, _ = site
    for path in READS:
        r = c.get(path)
        assert r.status_code == 200, (path, r.text)
        for word in PERSONAL:
            assert word not in r.text, (path, word)
    assert c.get("/api/feedback/me").json() == {"user": None, "admin": False, "staff": False,
                                                "projects": [], "read_only": True}
    got = c.get(f"/api/feedback/items/{item(fstore, 'non_specimen_photo', 'inat:1000')}")
    assert got.status_code == 200
    body = got.json()
    for word in PERSONAL:
        assert word not in got.text, word
    # The outcome and its date stay.
    [d] = body["decisions"]
    assert d["decision"] == "exclude_photo" and d["value"] == "micrograph" and d["at"]


def test_a_signed_in_person_who_reviews_nothing_reads_the_same_as_a_visitor(site):
    c, fstore, _ = site
    as_user(c, "nobody-1")
    me = c.get("/api/feedback/me").json()
    assert me["read_only"] is True and me["user"]["id"] == "nobody-1" and me["projects"] == []
    r = c.get("/api/feedback/items", params={"type": "non_specimen_photo", "status": "decided"})
    assert r.status_code == 200 and '"by_user":' not in r.text and "my private note" not in r.text


def test_only_items_of_no_project_are_listed_or_opened_read_only(site):
    c, fstore, _ = site
    by_type = {s["type"]: s["open"] + s["decided"] for s in c.get("/api/feedback/summary").json()}
    assert by_type["non_specimen_photo"] == 2 and by_type["photos_unlike_label"] == 1
    assert by_type["group_label"] == 0
    flagged = c.get("/api/feedback/photo-review/flagged", params={"status": "open"}).json()
    assert {t["photo_key"] for t in flagged["tiles"]} == {"inat:1100"}
    for subject, kind in (("100", "photos_unlike_label"), ("Laccaria laccata group", "group_label"),
                          ("inat:1002", "non_specimen_photo")):
        assert c.get(f"/api/feedback/items/{item(fstore, kind, subject)}").status_code == 404
    listed = c.get("/api/feedback/items", params={"type": "photos_unlike_label"}).json()
    assert [i["subject"] for i in listed["items"]] == ["104"]


def test_a_reviewer_still_sees_their_projects_items_and_who_decided(site):
    c, fstore, _ = site
    as_user(c, "admin-1")
    got = c.get(f"/api/feedback/items/{item(fstore, 'non_specimen_photo', 'inat:1000')}").json()
    assert got["decisions"][0]["by_name"] == "Reviewer Ann"
    assert got["decisions"][0]["note"] == "my private note"
    as_user(c, "member-1")                      # a contributor sees their project, not the public items
    by_type = {s["type"]: s["open"] for s in c.get("/api/feedback/summary").json()}
    assert by_type["group_label"] == 1 and by_type["non_specimen_photo"] == 0
    assert c.get("/api/feedback/me").json()["read_only"] is False


WRITES = [("/api/feedback/items/{i}/decision", {"decision": "hard_example"}),
          ("/api/feedback/items/{i}/reopen", None),
          ("/api/feedback/items/{i}/reveal", None),
          ("/api/feedback/items/{i}/refresh", None),
          ("/api/feedback/items/{i}/org-preview", {"decision": "hard_example"}),
          ("/api/feedback/items/{i}/org-request", None),
          ("/api/feedback/photo-review/batch", {"decisions": []}),
          ("/api/feedback/photo-views/batch", {"decisions": []}),
          ("/api/feedback/photo-review/100/decision", {"photos": []}),
          ("/api/feedback/label-review/{i}/decision", {"outcome": "hard_example"}),
          ("/api/feedback/decisions/1/applied", None)]


def test_every_write_refuses_a_read_only_viewer_and_a_reviewer_still_decides(site):
    c, fstore, _ = site
    i = item(fstore, "photos_unlike_label", "104")
    for who, status in ((None, 401), ("nobody-1", 403)):
        if who:
            as_user(c, who)
        for path, body in WRITES:
            r = c.post(path.format(i=i), json=body) if body is not None else c.post(path.format(i=i))
            assert r.status_code == status, (who, path, r.status_code)
    assert c.get("/api/feedback/changes.tsv").status_code == 403
    assert fstore.get(i)["decisions"] == [] and fstore.get(i)["status"] == "open"
    as_user(c, "admin-1")
    r = c.post(f"/api/feedback/items/{i}/decision", json={"decision": "hard_example"})
    assert r.status_code == 200 and r.json()["status"] == "decided"


def test_a_restricted_photo_is_a_placeholder_and_its_bytes_are_never_served_read_only(site):
    c, fstore, _ = site
    cards = c.get("/api/feedback/photo-review", params={"status": "any"}).json()["records"]
    [card] = [x for x in cards if x["record_key"] == "100"]
    shown = {p["photo_key"]: p for p in card["photos"]}
    assert shown["inat:1000"]["shown"] == "public"
    assert shown["inat:1100"]["shown"] == "hidden"
    assert "/api/feedback/photo/" not in json.dumps(cards)
    tiles = c.get("/api/feedback/photo-review/flagged").json()["tiles"]
    [arr] = [t for t in tiles if t["photo_key"] == "inat:1100"]
    assert arr["shown"] == "hidden" and "owner_login" not in arr and "thumb" not in arr
    for who in (None, "nobody-1"):
        if who:
            as_user(c, who)
        for size in ("grid", "thumb", "full"):
            r = c.get(f"/api/feedback/photo/inat:1100?size={size}")
            assert r.status_code == 403 and not r.content.startswith(b"\xff\xd8"), (who, size)
    as_user(c, "admin-1")                       # a reviewer still gets it, privately
    r = c.get("/api/feedback/photo/inat:1100?size=grid")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"


def test_a_blind_item_stays_blind_read_only(site):
    c, fstore, _ = site
    i = item(fstore, "photos_unlike_label", "104")
    fstore.conn.execute("update items set blind = 1, proposed = 'Vision says C z', "
                        "rule = 'M1 vision disagrees' where id = ?", (i,))
    fstore.conn.commit()
    got = c.get(f"/api/feedback/items/{i}").json()
    assert got["masked"] is True and got["proposed"] == "" and got["rule"] == "label check"
    assert "blind" not in got and "revealed" not in got and "Vision says" not in json.dumps(got)
    cards = c.get("/api/feedback/label-review").json()["records"]
    assert all("revealed" not in x and "rule" not in x for x in cards if x["masked"])
    assert c.post(f"/api/feedback/items/{i}/reveal").status_code == 401


def test_mycomap_orgs_record_reaches_a_read_only_viewer_without_admin_links(site):
    c, fstore, _ = site
    i = item(fstore, "photos_unlike_label", "104")
    [live] = c.get(f"/api/feedback/items/{i}").json()["live"]
    assert live["name"] == "B y" and "links" not in live
    assert "admin" not in json.dumps(live)
    assert [s["blast_link"] for s in live["sequences"]][0] == {
        "href": "https://www.mycomap.org/mycoblast/9", "kind": "public"}


def test_the_page_and_its_reads_open_on_a_site_that_needs_sign_in(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    assert c.get("/feedback").status_code == 200
    assert c.get("/api/feedback/summary").status_code == 200
    assert c.post("/api/feedback/photo-review/batch", json={"decisions": []}).status_code == 401
    assert c.get("/api/feedback/changes.tsv").status_code == 401


def test_the_read_only_cut_drops_who_decided_at_any_depth_and_keeps_the_outcome():
    got = fp.read_only({"decisions": [{"decision": "x", "value": "v", "at": 1, "by_user": "7",
                                       "by_name": "Ann", "note": "n", "seen_answer": 1}],
                        "tiles": [{"decision": {"decision": "keep", "by": "Ann"}}]})
    assert got == {"decisions": [{"decision": "x", "value": "v", "at": 1}],
                   "tiles": [{"decision": {"decision": "keep"}}]}
    assert fp.photo({"shown": "hidden", "owner_login": "x"}) == {"shown": "hidden"}
    assert fp.photo({"shown": "public", "owner_login": "x"})["owner_login"] == "x"


def test_items_of_no_project_are_the_only_ones_a_public_view_holds(tmp_path):
    s = fb.Store(tmp_path / "hf.sqlite")
    s.seed([{"item_type": "group_label", "subject": "a", "rule": "R", "projects": []},
            {"item_type": "group_label", "subject": "b", "rule": "R", "projects": ["p1"]}], "t")
    assert [i["subject"] for i in s.items_of("group_label", fb.PUBLIC_ITEMS)] == ["a"]
    assert [i["subject"] for i in s.items_of("group_label", {"p1"})] == ["b"]
    assert len(s.items_of("group_label", None)) == 2
