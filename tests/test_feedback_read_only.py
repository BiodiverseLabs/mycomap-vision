"""Human Feedback by the mycomap.org site role (Steve, 2026-11-20: "any .org member should be
able to work with photos. Only contributors or higher should see and be able to alter things
on the taxonomy tab"; "members dont matter. role does. contributor, lab contributor, project
manager, admin"; earlier read-only rule, 2026-10-22).

Signed out, a person browses the Photos tab and changes nothing; signed in below Contributor,
they also decide its photos. Either way they see every photo item whatever its project, but
only the photos anyone may see (the others are left out of their lists, and refused in their
writes), each decision's outcome and date but never who made it or their note, and no
Taxonomy item at all. Contributors and up see and decide everything, whatever their project
seats (test_feedback_api.py, test_photo_review.py)."""

import json

import pytest
from test_feedback_api import as_user, make_site
from test_photo_review import photo_item
from test_signin import gated_site_with_pages

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_api
from mycomap_vision import feedback_public as fp

# Keys (as JSON writes them) and values that would say who decided, or what they wrote.
PERSONAL = ('"by_user":', '"by_name":', '"by":', '"note":', '"seen_answer":', '"applied_by":',
            '"revealed":', "admin-1", "Reviewer Ann", "my private note")
READS = ("/api/feedback/me", "/api/feedback/catalogue", "/api/feedback/summary",
         "/api/feedback/photo-review/flagged", "/api/feedback/photo-review",
         "/api/feedback/photo-views",
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


CUT_ME = {"admin": False, "staff": False, "projects": [], "taxonomy": False, "all_photos": False}


def test_a_signed_out_visitor_browses_the_photos_tab_and_no_reviewer_detail(site):
    c, fstore, _ = site
    for path in READS:
        r = c.get(path)
        assert r.status_code == 200, (path, r.text)
        for word in PERSONAL:
            assert word not in r.text, (path, word)
    assert c.get("/api/feedback/me").json() == {"user": None, "read_only": True, **CUT_ME}
    got = c.get(f"/api/feedback/items/{item(fstore, 'non_specimen_photo', 'inat:1000')}")
    assert got.status_code == 200
    body = got.json()
    for word in PERSONAL:
        assert word not in got.text, word
    # The outcome and its date stay.
    [d] = body["decisions"]
    assert d["decision"] == "exclude_photo" and d["value"] == "micrograph" and d["at"]


def test_an_account_below_contributor_reads_what_a_visitor_reads_but_is_not_read_only(site):
    c, fstore, _ = site
    for who in ("nobody-1", "member-1"):          # member-1 holds a seat on p2: no matter
        as_user(c, who)
        me = c.get("/api/feedback/me").json()
        assert me == {"user": {"id": who, "name": me["user"]["name"]}, "read_only": False,
                      **CUT_ME}, who
        for path in READS:
            r = c.get(path)
            assert r.status_code == 200, (who, path, r.text)
            for word in PERSONAL:
                assert word not in r.text, (who, path, word)


def test_below_contributor_the_photos_tab_holds_only_photos_they_may_see_and_no_taxonomy(site):
    c, fstore, _ = site
    for who in (None, "nobody-1"):
        if who:
            as_user(c, who)
        summary = c.get("/api/feedback/summary").json()
        assert {t["type"] for t in summary} == fb.PUBLIC_PHOTO_TYPES, who
        by_type = {t["type"]: (t["open"], t["decided"]) for t in summary}
        # 1000 decided, 1002 open (a p1 item, CC); the all-rights-reserved 1100 left out.
        assert by_type["non_specimen_photo"] == (1, 1), who
        flagged = c.get("/api/feedback/photo-review/flagged", params={"status": "open"}).json()
        assert [t["photo_key"] for t in flagged["tiles"]] == ["inat:1002"], who
        assert flagged["tiles"][0]["shown"] == "public"
        project_photo = item(fstore, "non_specimen_photo", "inat:1002")
        assert c.get(f"/api/feedback/items/{project_photo}").status_code == 200
        for subject, kind in (("100", "photos_unlike_label"), ("104", "photos_unlike_label"),
                              ("Laccaria laccata group", "group_label")):
            r = c.get(f"/api/feedback/items/{item(fstore, kind, subject)}")
            assert r.status_code == 403 and r.json()["detail"] == feedback_api.TAXONOMY_ONLY
        assert c.get("/api/feedback/items", params={"type": "photos_unlike_label"}).status_code == 403
        assert c.get("/api/feedback/label-review").status_code == 403


def test_contributors_and_up_see_every_item_every_photo_and_who_decided(site):
    c, fstore, _ = site
    as_user(c, "admin-1")
    got = c.get(f"/api/feedback/items/{item(fstore, 'non_specimen_photo', 'inat:1000')}").json()
    assert got["decisions"][0]["by_name"] == "Reviewer Ann"
    assert got["decisions"][0]["note"] == "my private note"
    as_user(c, "leader-1")                      # a Contributor seated on p1 only
    me = c.get("/api/feedback/me").json()
    assert me["read_only"] is False and me["taxonomy"] is True and me["all_photos"] is True
    by_type = {s["type"]: s["open"] for s in c.get("/api/feedback/summary").json()}
    assert by_type["group_label"] == 1 and by_type["non_specimen_photo"] == 2
    assert by_type["photos_unlike_label"] == 2
    tiles = {t["photo_key"]: t for t in c.get("/api/feedback/photo-review/flagged").json()["tiles"]}
    assert set(tiles) == {"inat:1100", "inat:1002"} and tiles["inat:1100"]["shown"] == "private"


# The Taxonomy tab's writes, and the admins' one.
TAXONOMY_WRITES = [("/api/feedback/items/{i}/decision", {"decision": "hard_example"}),
                   ("/api/feedback/items/{i}/reopen", None),
                   ("/api/feedback/items/{i}/reveal", None),
                   ("/api/feedback/items/{i}/refresh", None),
                   ("/api/feedback/items/{i}/org-preview", {"decision": "hard_example"}),
                   ("/api/feedback/items/{i}/org-request", None),
                   ("/api/feedback/label-review/{i}/decision", {"outcome": "hard_example"}),
                   ("/api/feedback/decisions/1/applied", None)]
PHOTO_WRITES = [("/api/feedback/photo-review/batch", {"decisions": []}),
                ("/api/feedback/photo-views/batch", {"decisions": []}),
                ("/api/feedback/recovered-labels/batch", {"decisions": []}),
                ("/api/feedback/photo-review/100/decision", {"photos": []})]


def post(c, path, body, i):
    return c.post(path.format(i=i), json=body) if body is not None else c.post(path.format(i=i))


def test_every_write_refuses_a_visitor_and_taxonomy_refuses_accounts_below_contributor(site):
    c, fstore, _ = site
    i = item(fstore, "photos_unlike_label", "104")
    for path, body in TAXONOMY_WRITES + PHOTO_WRITES:
        assert post(c, path, body, i).status_code == 401, path
    for who in ("nobody-1", "member-1"):
        as_user(c, who)
        for path, body in TAXONOMY_WRITES:
            assert post(c, path, body, i).status_code == 403, (who, path)
        for path, body in PHOTO_WRITES:            # let in; an empty page is the write's to refuse
            assert post(c, path, body, i).status_code == 400, (who, path)
    assert c.get("/api/feedback/changes.tsv").status_code == 403
    assert fstore.get(i)["decisions"] == [] and fstore.get(i)["status"] == "open"
    as_user(c, "leader-1")
    r = c.post(f"/api/feedback/items/{i}/decision", json={"decision": "hard_example"})
    assert r.status_code == 200 and r.json()["status"] == "decided"


def test_an_account_below_contributor_decides_the_photos_it_sees_and_no_hidden_one(site):
    c, fstore, _ = site
    cc = item(fstore, "non_specimen_photo", "inat:1002")
    arr = item(fstore, "non_specimen_photo", "inat:1100")
    as_user(c, "nobody-1")
    # A Save page holding the hidden photo is refused whole: it never marks what it hid.
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": cc, "decision": "keep"}, {"item_id": arr, "decision": "keep"}]})
    assert r.status_code == 403 and r.json()["detail"] == feedback_api.PHOTO_NOT_SHOWN
    assert fstore.get(cc)["status"] == "open" and fstore.get(arr)["status"] == "open"
    r = c.post("/api/feedback/photo-review/100/decision",
               json={"photos": [{"photo_key": "inat:1100", "decision": "keep"}]})
    assert r.status_code == 403
    r = c.post(f"/api/feedback/items/{arr}/decision", json={"decision": "keep_photo"})
    assert r.status_code == 403
    # A Taxonomy item in a photo batch is refused too.
    label = item(fstore, "photos_unlike_label", "104")
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": label, "decision": "keep"}]})
    assert r.status_code == 403 and r.json()["detail"] == feedback_api.TAXONOMY_ONLY
    # The photo it sees is its to decide, and what comes back still names nobody.
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": cc, "decision": "exclude", "reason": "micrograph"}]})
    assert r.status_code == 200, r.text
    got = fstore.get(cc)
    assert got["status"] == "decided" and got["decisions"][-1]["by_user"] == "nobody-1"
    r = c.post(f"/api/feedback/items/{cc}/reopen")
    assert r.status_code == 200 and r.json()["status"] == "open"
    for word in ('"by_user":', '"by_name":', '"note":'):
        assert word not in r.text
    r = c.post("/api/feedback/photo-review/102/decision",
               json={"photos": [{"photo_key": "inat:1002", "decision": "keep"}]})
    assert r.status_code == 200, r.text
    assert fstore.get(cc)["status"] == "decided"
    assert "/api/feedback/photo/" not in r.text and '"by_user":' not in r.text
    as_user(c, "leader-1")                       # a Contributor decides the hidden one
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": arr, "decision": "keep"}]})
    assert r.status_code == 200 and fstore.get(arr)["status"] == "decided"


def test_a_photo_granted_by_its_owner_joins_the_list_at_once(site, conn):
    c, fstore, _ = site
    as_user(c, "nobody-1")
    keys = lambda: {t["photo_key"] for t in c.get("/api/feedback/photo-review/flagged").json()["tiles"]}  # noqa: E731
    assert "inat:1100" not in keys()
    conn.execute("insert into photo_permissions (inat_user_id, status) values (55, 'granted')")
    from datetime import datetime, timezone
    conn.execute("insert into permission_syncs (attempted_at, ok) values (?, 1)",
                 (datetime.now(timezone.utc).isoformat(),))
    conn.commit()
    assert "inat:1100" in keys()


def test_a_restricted_photo_is_a_placeholder_and_its_bytes_are_never_served_below_contributor(site):
    c, fstore, _ = site
    cards = c.get("/api/feedback/photo-review", params={"status": "any"}).json()["records"]
    [card] = [x for x in cards if x["record_key"] == "100"]
    shown = {p["photo_key"]: p for p in card["photos"]}
    assert shown["inat:1000"]["shown"] == "public"
    assert shown["inat:1100"]["shown"] == "hidden" and shown["inat:1100"]["item_id"] is None
    assert "/api/feedback/photo/" not in json.dumps(cards)
    tiles = c.get("/api/feedback/photo-review/flagged").json()["tiles"]
    assert "inat:1100" not in {t["photo_key"] for t in tiles}
    for who in (None, "nobody-1"):
        if who:
            as_user(c, who)
        for size in ("grid", "thumb", "full"):
            r = c.get(f"/api/feedback/photo/inat:1100?size={size}")
            assert r.status_code == 403 and not r.content.startswith(b"\xff\xd8"), (who, size)
    as_user(c, "leader-1")                       # a Contributor still gets it, privately
    r = c.get("/api/feedback/photo/inat:1100?size=grid")
    assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"


def test_a_blind_item_is_a_taxonomy_item_and_refused_below_contributor(site):
    c, fstore, _ = site
    i = item(fstore, "photos_unlike_label", "104")
    fstore.conn.execute("update items set blind = 1, proposed = 'Vision says C z', "
                        "rule = 'M1 vision disagrees' where id = ?", (i,))
    fstore.conn.commit()
    for who, write in ((None, 401), ("nobody-1", 403)):
        if who:
            as_user(c, who)
        r = c.get(f"/api/feedback/items/{i}")
        assert r.status_code == 403 and "Vision says" not in r.text, who
        assert c.get("/api/feedback/label-review").status_code == 403
        assert c.post(f"/api/feedback/items/{i}/reveal").status_code == write
    as_user(c, "leader-1")
    got = c.get(f"/api/feedback/items/{i}").json()
    assert got["masked"] is True and "Vision says" not in json.dumps(got)


def test_mycomap_orgs_record_reaches_an_account_below_contributor_without_admin_links(site):
    c, fstore, _ = site
    fstore.seed([{**photo_item("inat:1003", "100", "R6 micrograph probe p=0.6"),
                  "record_keys": ["100"]}], "live")
    as_user(c, "nobody-1")
    i = item(fstore, "non_specimen_photo", "inat:1003")
    [live] = c.get(f"/api/feedback/items/{i}").json()["live"]
    assert live["name"] == "A x" and "links" not in live
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


def test_the_photos_tab_scope_holds_every_photo_item_and_no_record_or_name_item(tmp_path):
    s = fb.Store(tmp_path / "hf.sqlite")
    s.seed([{"item_type": "group_label", "subject": "a", "rule": "R", "projects": []},
            {"item_type": "group_label", "subject": "b", "rule": "R", "projects": ["p1"]},
            {"item_type": "photo_view", "subject": "inat:9", "rule": "R8", "granularity": "photo",
             "projects": ["p1"]}], "t")
    assert s.items_of("group_label", fb.PUBLIC_ITEMS) == []     # not even one of no project
    assert [i["subject"] for i in s.items_of("photo_view", fb.PUBLIC_ITEMS)] == ["inat:9"]
    assert [i["subject"] for i in s.items_of("group_label", {"p1"})] == ["b"]
    assert s.items_of("photo_view", {"p2"}) == []             # a reviewer keeps their own scope
    assert len(s.items_of("group_label", None)) == 2
    # Recovered labels too (Steve, 2026-10-26): read-only, the crop outlines on the original only
    assert fb.PUBLIC_PHOTO_TYPES == {"non_specimen_photo", "photo_view", "recovered_label"}


def test_shown_photos_follows_the_photo_rule_and_rereads_the_manifest_after_its_ttl():
    from datetime import datetime, timedelta, timezone

    from mycomap_vision.permissions import PermissionView
    manifest = {1: ("open", 5), 2: ("nc", None), 3: ("arr", 7), 4: ("arr", None), 5: (None, 7)}
    asked = []

    def query(sql, params):
        asked.append(len(params))
        return [(p, *manifest[p]) for p in params if p in manifest]
    now = datetime.now(timezone.utc)
    view = {"v": PermissionView(frozenset({7}), frozenset(), now, timedelta(days=2))}
    clock = {"t": 0.0}
    shown = feedback_api.ShownPhotos(query, lambda: view["v"], ttl=60, clock=lambda: clock["t"])
    keys = {f"inat:{i}" for i in range(1, 7)}            # inat:6 is not in the manifest
    assert shown.shown(keys) == {"inat:1", "inat:2", "inat:3", "inat:5"}
    # A stale sync shows no all-rights-reserved photo (may_show), read from the cache.
    view["v"] = PermissionView(frozenset({7}), frozenset(), now - timedelta(days=9), timedelta(days=2))
    assert shown.shown(keys) == {"inat:1", "inat:2"} and asked == [6]
    manifest[1] = ("arr", 8)                             # relicensed in a new release
    assert "inat:1" in shown.shown(keys)                 # kept until the TTL
    clock["t"] = 61
    assert "inat:1" not in shown.shown(keys) and asked == [6, 6]
    for k in ("inat:1", "inat:2", "inat:3", "inat:4", "inat:5", "inat:6"):
        pid = int(k.split(":")[1])
        info = manifest.get(pid)
        want = info is not None and view["v"].may_show(*info)
        assert (k in shown.shown([k])) == want, k
