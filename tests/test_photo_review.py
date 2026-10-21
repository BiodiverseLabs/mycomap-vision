"""The Human Feedback photo review (photo_review.py; Steve, 2026-10-18): the "Photos that show
no specimen" queue one record at a time, tabs per kind of flag with their counts, per-photo
keep / leave-out decisions with a reason, and the photo rule: CC or owner-permitted photos by
their public URL; all-rights-reserved photos to ADMINS ONLY, from Vision's own copy, private and
never stored; never a photo whose owner declined."""

import io

import pytest
from test_feedback_api import as_user, make_site

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr

INAT = "https://www.inaturalist.org/observations/"


def photo_item(photo_key, record, rule, label="A x", projects=("p1",)):
    return {"item_type": "non_specimen_photo", "subject": photo_key, "rule": rule,
            "granularity": "photo", "links": [INAT + record], "current": label,
            "projects": list(projects)}


# --- the rules themselves ---------------------------------------------------------------------

def test_each_flag_goes_to_its_tab_and_the_gate_counts_by_how_unlike_a_fungus():
    flags = pr.parse_flags("R6 micrograph probe p=0.62; gate not_fungus p(fungus)=0.01")
    assert [(f["tab"], f["strength"]) for f in flags] == [("micrograph", 0.62), ("gate", 0.99)]
    assert pr.parse_flags("R6 view habitat p=0.20")[0]["tab"] == "habitat"
    assert pr.parse_flags("R6 view slip p=0.40")[0]["tab"] == "slip"
    assert pr.parse_flags("R6 view people p=0.30")[0]["tab"] == "people"
    assert pr.parse_flags("R6 view dried p=0.05")[0]["tab"] == "other"
    assert pr.parse_flags("something new")[0]["tab"] == "other"


def test_a_photo_item_belongs_to_the_record_its_link_names():
    assert pr.record_of_link("https://mushroomobserver.org/obs/102739") == "mo:102739"
    assert pr.record_of_link(INAT + "100048248") == "100048248"
    assert pr.record_of_link("https://example.org/x") is None


def items_for_cards():
    s = [
        {"id": 1, "subject": "inat:1", "rule": "R6 micrograph probe p=0.90", "status": "open",
         "links": [INAT + "10"], "current": "A x"},
        {"id": 2, "subject": "inat:2", "rule": "R6 view habitat p=0.30", "status": "open",
         "links": [INAT + "10"], "current": "A x"},
        {"id": 3, "subject": "inat:3", "rule": "R6 micrograph probe p=0.55", "status": "open",
         "links": [INAT + "11"], "current": "B y"},
        {"id": 4, "subject": "inat:4", "rule": "R6 micrograph probe p=0.70", "status": "decided",
         "links": [INAT + "12"], "current": "C z",
         "last_decision": {"decision": "exclude_photo", "value": "micrograph", "by_name": "S"}},
    ]
    return s


def test_records_show_in_each_tab_they_have_a_flagged_photo_for_with_counts():
    out = pr.cards(items_for_cards(), "micrograph")
    tabs = {t["code"]: (t["open"], t["decided"]) for t in out["tabs"]}
    assert tabs["micrograph"] == (2, 1) and tabs["habitat"] == (1, 0) and tabs["gate"] == (0, 0)
    assert [r["record_key"] for r in out["records"]] == ["10", "11"]     # highest p first
    assert out["records"][0]["in_tab"] == ["inat:1"]                      # that tab's photos
    habitat = pr.cards(items_for_cards(), "habitat")["records"]
    assert [r["record_key"] for r in habitat] == ["10"] and habitat[0]["in_tab"] == ["inat:2"]


def test_sorting_filtering_and_the_decided_view():
    assert [r["record_key"] for r in pr.cards(items_for_cards(), "micrograph", sort="p_asc")
            ["records"]] == ["11", "10"]
    assert [r["record_key"] for r in pr.cards(items_for_cards(), "all", min_flagged=2)
            ["records"]] == ["10"]
    found = pr.cards(items_for_cards(), "all", query="b Y")
    assert [r["record_key"] for r in found["records"]] == ["11"]
    assert [r["record_key"] for r in pr.cards(items_for_cards(), query="10")["records"]] == ["10"]
    assert {t["code"]: t["open"] for t in found["tabs"]}["micrograph"] == 2  # the queue's counts
    done = pr.cards(items_for_cards(), "micrograph", status="decided")["records"]
    assert [r["record_key"] for r in done] == ["12"]
    assert done[0]["flagged"][0]["decision"]["reason"] == "micrograph"
    for bad in ({"tab": "nope"}, {"status": "x"}, {"sort": "x"}):
        with pytest.raises(fb.FeedbackError):
            pr.cards(items_for_cards(), **bad)


def test_record_decisions_are_per_photo_with_a_reason_and_audited(tmp_path):
    store = fb.Store(tmp_path / "f.sqlite")
    store.seed([photo_item("inat:1", "10", "R6 micrograph probe p=0.9"),
                photo_item("inat:2", "10", "R6 view habitat p=0.3")], "t")
    who = {"sub": "admin-1", "name": "Steve"}
    keys = {"inat:1", "inat:2", "inat:5"}
    with pytest.raises(fb.FeedbackError, match="say why"):
        pr.decide_record(store, "10", [{"photo_key": "inat:1", "decision": "exclude"}], who, keys)
    with pytest.raises(fb.FeedbackError, match="not a photo of"):
        pr.decide_record(store, "10", [{"photo_key": "inat:9", "decision": "keep"}], who, keys)
    with pytest.raises(KeyError):
        pr.decide_record(store, "99", [{"photo_key": "inat:1", "decision": "keep"}], who, keys)
    out = pr.decide_record(store, "10", [
        {"photo_key": "inat:1", "decision": "exclude", "reason": "micrograph"},
        {"photo_key": "inat:2", "decision": "keep"},
        {"photo_key": "inat:5", "decision": "exclude", "reason": "slip"}], who, keys)
    assert out == {"decided": 3, "added": 1}
    assert store.excluded_photos() == {"inat:1", "inat:5"}
    added = next(i for i in store.items_of("non_specimen_photo") if i["subject"] == "inat:5")
    assert added["rule"].startswith("reviewer:") and not added["model_flagged"]
    assert added["projects"] == ["p1"]                    # the record's projects, like the rest
    tsv = {c["subject"]: c for c in store.changes(applies="vision")}
    assert tsv["inat:1"]["value"] == "micrograph" and tsv["inat:1"]["by_name"] == "Steve"
    # Keeping an unflagged photo stores nothing.
    assert pr.decide_record(store, "10", [{"photo_key": "inat:5", "decision": "keep"}], who,
                            keys)["decided"] == 1         # inat:5 has an item now: re-decided
    assert store.excluded_photos() == {"inat:1"}


# --- over the real app ------------------------------------------------------------------------

def site_with_photos(conn, tmp_path):
    """Records 100 and 101 with flagged photos. 1000 is CC; record 100 also has 1100, an
    all-rights-reserved photo nobody asked about; record 101's photo 1001 is all rights
    reserved by an owner who withdrew."""
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
    conn.execute("update photos set license_class = 'arr', license_code = '', owner_user_id = 66 "
                 "where photo_id = 1001")
    conn.execute("insert into photo_permissions (inat_user_id, status) values (66, 'withdrawn')")
    conn.commit()
    fstore.seed([photo_item("inat:1000", "100", "R6 micrograph probe p=0.80"),
                 photo_item("inat:1100", "100", "R6 view habitat p=0.40"),
                 photo_item("inat:1001", "101", "R6 micrograph probe p=0.60")], "photos")
    return c, fstore


def photo_of(card, key):
    return next(p for p in card["photos"] if p["photo_key"] == key)


def test_admins_get_cards_and_see_all_rights_reserved_photos_privately(conn, tmp_path):
    c, _ = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    r = c.get("/api/feedback/photo-review", params={"tab": "micrograph"}).json()
    assert {t["code"]: t["open"] for t in r["tabs"]}["micrograph"] == 2
    assert [x["record_key"] for x in r["records"]] == ["100", "101"]
    card = r["records"][0]
    assert [p["photo_key"] for p in card["photos"]] == ["inat:1000", "inat:1100"]
    cc, arr = photo_of(card, "inat:1000"), photo_of(card, "inat:1100")
    assert cc["shown"] == "public" and cc["thumb"].startswith("https://") and cc["in_tab"]
    assert arr["shown"] == "private" and arr["thumb"] == "/api/feedback/photo/inat:1100?size=grid"
    assert arr["flagged"] and not arr["in_tab"]               # flagged for habitat, not here
    img = c.get(arr["thumb"])
    assert img.status_code == 200 and img.headers["content-type"] == "image/jpeg"
    assert img.headers["cache-control"] == "private, no-store"
    assert img.content[:2] == b"\xff\xd8"


def test_any_reviewer_with_access_to_the_record_sees_its_photos_and_nobody_else(conn,
                                                                                 tmp_path):
    c, _ = site_with_photos(conn, tmp_path)
    c.cookies.clear()
    assert c.get("/api/feedback/photo/inat:1100").status_code == 401
    assert c.get("/api/feedback/photo/inat:1100",
                 headers={"Authorization": "Bearer some-v1-key"}).status_code == 401
    as_user(c, "leader-1")                                    # a leader of the item's project
    card = c.get("/api/feedback/photo-review").json()["records"][0]
    arr = photo_of(card, "inat:1100")
    assert arr["shown"] == "private" and arr["thumb"] == "/api/feedback/photo/inat:1100?size=grid"
    img = c.get(arr["thumb"])
    assert img.status_code == 200 and img.headers["cache-control"] == "private, no-store"
    as_user(c, "member-1")                                    # another project's contributor
    assert c.get("/api/feedback/photo-review").json()["records"] == []
    r = c.get("/api/feedback/photo/inat:1100")
    assert r.status_code == 403 and "not on a record you review" in r.json()["detail"]
    as_user(c, "admin-1")                                     # a photo on no queued record
    assert c.get("/api/feedback/photo/inat:1003").status_code == 403


def test_a_declined_owners_photo_is_never_served_even_to_admins(conn, tmp_path):
    c, _ = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    assert c.get("/api/feedback/photo/inat:1001").status_code == 403
    r = c.get("/api/feedback/photo-review", params={"tab": "micrograph"}).json()
    card = next(x for x in r["records"] if x["record_key"] == "101")
    p = photo_of(card, "inat:1001")
    assert p["shown"] == "hidden" and p["hidden_reason"] == "the owner declined"
    assert c.get("/api/feedback/photo/inat:424242").status_code == 404


def test_saving_a_record_decides_each_photo_and_feeds_the_nightly_exclusions(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "leader-1")
    url = "/api/feedback/photo-review/100/decision"
    assert c.post(url, json={"photos": [{"photo_key": "inat:1000", "decision": "exclude"}]}
                  ).status_code == 400
    assert c.post(url, json={"photos": [{"photo_key": "inat:1001", "decision": "keep"}]}
                  ).status_code == 400                         # not this record's photo
    r = c.post(url, json={"photos": [
        {"photo_key": "inat:1000", "decision": "exclude", "reason": "micrograph"},
        {"photo_key": "inat:1100", "decision": "keep"}]})
    assert r.status_code == 200 and r.json()["decided"] == 2
    assert all(p["status"] == "decided" for p in r.json()["record"]["photos"] if p["flagged"])
    assert fstore.excluded_photos() == {"inat:1000"}
    left = c.get("/api/feedback/photo-review").json()
    assert [x["record_key"] for x in left["records"]] == ["101"]
    as_user(c, "member-1")
    assert c.post(url, json={"photos": [{"photo_key": "inat:1000", "decision": "keep"}]}
                  ).status_code == 404


# --- one reason per photo, starting at its flag (Steve, 2026-10-18) ---------------------------

def test_each_flagged_photo_suggests_the_reason_of_its_own_flag():
    slip = pr.parse_flags("R6 view slip p=0.40")
    habitat = pr.parse_flags("R6 view habitat p=0.30")
    both = pr.parse_flags("R6 micrograph probe p=0.55; gate not_fungus p(fungus)=0.01")
    assert pr.suggested_reason(slip) == "slip"
    assert pr.suggested_reason(habitat) == "habitat"
    assert pr.suggested_reason(both) == "not_fungus"                  # strongest flag: 0.99
    assert pr.suggested_reason(both, "micrograph") == "micrograph"   # the open tab's own flag
    assert pr.suggested_reason(pr.parse_flags("R6 view dried p=0.05")) == "other"
    assert pr.suggested_reason([]) is None
    assert set(pr.REASON_OF_TAB) == set(pr.TAB_CODES)
    assert set(pr.REASON_OF_TAB.values()) <= set(pr.REASON_CODES)


def test_photos_of_one_record_are_left_out_for_different_reasons(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    card = c.get("/api/feedback/photo-review").json()["records"][0]
    assert {p["photo_key"]: p["suggested_reason"] for p in card["photos"]} == {
        "inat:1000": "micrograph", "inat:1100": "habitat"}
    r = c.post("/api/feedback/photo-review/100/decision", json={"photos": [
        {"photo_key": "inat:1000", "decision": "exclude", "reason": "micrograph"},
        {"photo_key": "inat:1100", "decision": "exclude", "reason": "not_fungus"}]})
    assert r.status_code == 200
    reasons = {ch["subject"]: ch["value"] for ch in fstore.changes(applies="vision")}
    assert reasons == {"inat:1000": "micrograph", "inat:1100": "not_fungus"}


def test_dried_specimens_and_spore_prints_are_reasons_of_their_own(conn, tmp_path):
    """Steve, 2026-10-19: a dried specimen or a spore print is left out as itself, not 'Other'."""
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    offered = {r["code"]: r["label"] for r in c.get("/api/feedback/photo-review").json()["reasons"]}
    assert offered["dried"] == "Dried specimen" and offered["spore_print"] == "Spore print"
    assert list(offered)[-1] == "other"                           # "Other" stays last
    r = c.post("/api/feedback/photo-review/100/decision", json={"photos": [
        {"photo_key": "inat:1000", "decision": "exclude", "reason": "dried"},
        {"photo_key": "inat:1100", "decision": "exclude", "reason": "spore_print"}]})
    assert r.status_code == 200
    reasons = {ch["subject"]: ch["value"] for ch in fstore.changes(applies="vision")}
    assert reasons == {"inat:1000": "dried", "inat:1100": "spore_print"}



def test_uv_chemical_tests_trees_and_drawings_are_reasons_of_their_own(conn, tmp_path):
    """Steve, 2026-10-21: a UV reaction, a chemical spot test (KOH, FeSO4 and the like), a
    phylogenetic tree, or a drawing of microscopy or of the specimen is left out as itself,
    not 'Other'."""
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    offered = {r["code"]: r["label"] for r in c.get("/api/feedback/photo-review").json()["reasons"]}
    assert offered["uv"] == "UV reaction" and offered["chemical"] == "Chemical spot test"
    assert offered["phylo_tree"] == "Phylogenetic tree"
    assert offered["drawing_micro"] == "Drawing – microscopy"
    assert offered["drawing_specimen"] == "Drawing – specimen"
    assert list(offered)[-1] == "other"                           # "Other" stays last
    for code in ("uv", "chemical", "phylo_tree", "drawing_micro", "drawing_specimen"):
        r = c.post("/api/feedback/photo-review/100/decision", json={"photos": [
            {"photo_key": "inat:1000", "decision": "exclude", "reason": code}]})
        assert r.status_code == 200, code
        reasons = {ch["subject"]: ch["value"] for ch in fstore.changes(applies="vision")}
        assert reasons == {"inat:1000": code}

# --- photos the box holds no readable copy of come from their source ----------------------------

class FakeResp:
    def __init__(self, status=200, body=b"", location=None):
        self.status_code, self.body = status, body
        self.headers = {"Location": location} if location else {}

    def iter_content(self, n):
        for i in range(0, len(self.body), n):
            yield self.body[i:i + n]

    def close(self):
        pass


class FakeSession:
    def __init__(self, answers):
        self.answers, self.calls, self.headers = answers, [], {}

    def get(self, url, **kw):
        self.calls.append(url)
        a = self.answers.get(url)
        return a if a is not None else FakeResp(404)


def png(color=(10, 200, 10)):
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(buf, "PNG")
    return buf.getvalue()


def test_without_a_readable_copy_the_photo_is_fetched_from_its_source_once(conn, tmp_path,
                                                                         monkeypatch):
    session = FakeSession({})
    monkeypatch.setattr(pr.SourceFetcher, "__init__", lambda self, *a, **k: real_init(
        self, session=session, sleep=lambda s: None))
    c, _ = site_with_photos(conn, tmp_path)
    conn.execute("update photo_copies set store = 's3://not-readable/' where photo_id = 1100")
    conn.commit()
    url = conn.execute("select source_url from photos where photo_id = 1100").fetchone()[0]
    medium = url.replace("square", "medium")
    session.answers[medium] = FakeResp(200, png())
    as_user(c, "leader-1")
    for _ in range(2):
        r = c.get("/api/feedback/photo/inat:1100?size=thumb")
        assert r.status_code == 200 and r.headers["cache-control"] == "private, no-store"
        assert r.content[:2] == b"\xff\xd8"
    assert session.calls == [medium]                          # the second came from the cache


real_init = pr.SourceFetcher.__init__


def test_a_source_that_fails_gives_the_placeholder_not_an_error(conn, tmp_path, monkeypatch):
    session = FakeSession({})                                 # every source answer is a 404
    monkeypatch.setattr(pr.SourceFetcher, "__init__", lambda self, *a, **k: real_init(
        self, session=session, sleep=lambda s: None))
    c, _ = site_with_photos(conn, tmp_path)
    conn.execute("delete from photo_copies where photo_id = 1100")
    conn.commit()
    as_user(c, "admin-1")
    assert c.get("/api/feedback/photo/inat:1100").status_code == 404
    assert len(session.calls) == 1


def test_the_source_fetcher_reads_only_photo_hosts_over_https_within_its_budget():
    url = "https://static.inaturalist.org/photos/5/medium.jpg"
    session = FakeSession({url: FakeResp(200, b"x" * 1000)})
    f = pr.SourceFetcher(session=session, sleep=lambda s: None)
    assert session.headers["User-Agent"].startswith("MycoMap-Vision")
    assert f.get("http://static.inaturalist.org/photos/5/medium.jpg", "inat") is None
    assert f.get("https://example.org/photos/5/medium.jpg", "inat") is None
    assert session.calls == []
    assert f.get(url, "inat") == b"x" * 1000
    for b in f.budgets:                                       # the review's own static budget
        b.add(b.limit)
    assert f.get(url, "inat") is None and len(session.calls) == 1
    # MO: one redirect, to MO's own image host only.
    mo = "https://mushroomobserver.org/images/640/7.jpg"
    session.answers[mo] = FakeResp(302, location="https://evil.example/7.jpg")
    session.answers["https://evil.example/7.jpg"] = FakeResp(200, b"not ours")
    assert f.get(mo, "mo") is None
    assert "https://evil.example/7.jpg" not in session.calls
    session.answers[mo] = FakeResp(302, location="https://images.mushroomobserver.org/640/7.jpg")
    session.answers["https://images.mushroomobserver.org/640/7.jpg"] = FakeResp(200, b"mo")
    assert f.get(mo, "mo") == b"mo"


def test_the_queue_card_counts_photos_and_the_observations_they_are_in(conn, tmp_path):
    items = items_for_cards()                  # 3 open photos in records 10 (2) and 11; 12 decided
    assert pr.record_counts(items) == {"open": 2, "decided": 1}
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    s = {t["type"]: t for t in c.get("/api/feedback/summary").json()}["non_specimen_photo"]
    assert (s["open"], s["records_open"], s["records_decided"]) == (3, 2, 0)
    c.post("/api/feedback/photo-review/101/decision",
           json={"photos": [{"photo_key": "inat:1001", "decision": "keep"}]})
    s = {t["type"]: t for t in c.get("/api/feedback/summary").json()}["non_specimen_photo"]
    assert (s["open"], s["decided"], s["records_open"], s["records_decided"]) == (2, 1, 1, 1)
    assert "records_open" not in {t["type"]: t for t in
                                  c.get("/api/feedback/summary").json()}["group_label"]
