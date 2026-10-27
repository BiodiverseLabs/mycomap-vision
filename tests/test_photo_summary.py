"""The photo review's Summary view (Steve, 2026-10-21): the flagged photos alone, one tile each,
to scan and save in one batch; and the page's first load kept cheap (counts and item lists
cached until the store changes). A batch writes exactly what was shown, excluded or kept, and
nothing before the commit; it is whole or nothing, and the paper's seal still holds."""

import pytest
from test_feedback_api import as_user
from test_photo_review import items_for_cards, photo_item, site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr

SEAL = "2026-10-12T13:17:51+00:00"


def test_tiles_are_the_flagged_photos_of_the_tab_sorted_like_the_record_view():
    tiles = pr.flagged_tiles(items_for_cards(), "micrograph")
    assert [t["photo_key"] for t in tiles] == ["inat:1", "inat:3"]        # open only, p 0.90, 0.55
    assert tiles[0]["record_key"] == "10" and tiles[0]["suggested_reason"] == "micrograph"
    assert [t["photo_key"] for t in pr.flagged_tiles(items_for_cards(), "micrograph",
                                                     sort="p_asc")] == ["inat:3", "inat:1"]
    assert [t["photo_key"] for t in pr.flagged_tiles(items_for_cards(), "all", sort="flagged")
            ][:2] == ["inat:1", "inat:2"]                                    # record 10 has 2
    assert [t["photo_key"] for t in pr.flagged_tiles(items_for_cards(), "micrograph",
                                                     status="decided")] == ["inat:4"]
    assert [t["photo_key"] for t in pr.flagged_tiles(items_for_cards(), "all", query="b y")
            ] == ["inat:3"]
    with pytest.raises(fb.FeedbackError):
        pr.flagged_tiles(items_for_cards(), "nope")


def test_the_summary_api_pages_tiles_with_small_thumbs_and_record_details(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    r = c.get("/api/feedback/photo-review/flagged", params={"tab": "all", "limit": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 3 and len(body["tiles"]) == 2 and body["offset"] == 0
    first = body["tiles"][0]
    assert first["photo_key"] == "inat:1000" and first["label"] == "A x"
    assert first["link"] == "https://www.inaturalist.org/observations/100"
    assert first["shown"] == "public" and "/small." in first["thumb"]       # 240 px, not 500
    assert first["record_photos"] == 2 and first["suggested_reason"] == "micrograph"
    rest = c.get("/api/feedback/photo-review/flagged",
                 params={"tab": "all", "offset": 2, "limit": 2}).json()["tiles"]
    assert len(rest) == 1
    private = next(t for t in body["tiles"] + rest if t["photo_key"] == "inat:1100")
    assert private["shown"] == "private" and private["thumb"].endswith("?size=grid")
    img = c.get(private["thumb"])
    assert img.status_code == 200 and img.headers["cache-control"] == "private, no-store"
    as_user(c, "member-1")                                     # another project's contributor
    assert c.get("/api/feedback/photo-review/flagged").json()["total"] == 0


def test_a_batch_writes_exactly_the_shown_split_and_nothing_before_it(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "leader-1")
    ids = {t["photo_key"]: t["item_id"] for t in
           c.get("/api/feedback/photo-review/flagged").json()["tiles"]}
    assert fstore.changes(applies="vision") == [] and fstore.changes(applies="note") == []
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:1000"], "decision": "exclude", "reason": "micrograph"},
        {"item_id": ids["inat:1001"], "decision": "exclude", "reason": "micrograph"},
        {"item_id": ids["inat:1100"], "decision": "keep"}]})            # kept in the lightbox
    assert r.status_code == 200
    assert (r.json()["excluded"], r.json()["kept"], r.json()["unchanged"]) == (2, 1, 0)
    assert fstore.excluded_photos() == {"inat:1000", "inat:1001"}
    kept = {ch["subject"]: ch["decision"] for ch in fstore.changes(applies="note")}
    assert kept == {"inat:1100": "keep_photo"}
    assert c.get("/api/feedback/photo-review/flagged").json()["total"] == 0  # decided: gone
    done = c.get("/api/feedback/photo-review/flagged", params={"status": "decided"}).json()
    assert {t["photo_key"]: t["decision"]["decision"] for t in done["tiles"]} == {
        "inat:1000": "exclude_photo", "inat:1001": "exclude_photo", "inat:1100": "keep_photo"}



def test_the_summary_offers_and_saves_uv_chemical_tests_trees_and_drawings(conn, tmp_path):
    """Steve, 2026-10-21: the Summary's reason picker has UV reaction, Chemical spot test,
    Phylogenetic tree / BLAST and Drawing (one reason since 2026-10-27), and a batch saves each
    as the photo's reason."""
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    body = c.get("/api/feedback/photo-review/flagged").json()
    offered = {r["code"]: r["label"] for r in body["reasons"]}
    assert offered["uv"] == "UV reaction" and offered["chemical"] == "Chemical spot test"
    assert offered["phylo_tree"] == "Phylogenetic tree / BLAST"
    assert offered["drawing"] == "Drawing"
    assert "drawing_micro" not in offered and "drawing_specimen" not in offered
    assert offered["species_description"] == "Species description"
    assert offered["blurry"] == "Blurry or badly exposed"
    ids = {t["photo_key"]: t["item_id"] for t in body["tiles"]}
    for code in ("uv", "chemical", "phylo_tree", "drawing", "species_description", "blurry"):
        r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
            {"item_id": ids["inat:1000"], "decision": "exclude", "reason": code}]})
        assert r.status_code == 200, code
        reasons = {ch["subject"]: ch["value"] for ch in fstore.changes(applies="vision")}
        assert reasons == {"inat:1000": code}

def test_a_bad_batch_writes_nothing(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    ids = [t["item_id"] for t in c.get("/api/feedback/photo-review/flagged").json()["tiles"]]
    group = fstore.items("group_label")["items"][0]["id"]
    for bad, why in (
            ([{"item_id": ids[0], "decision": "exclude", "reason": "micrograph"},
              {"item_id": ids[1], "decision": "exclude"}], "say why"),
            ([{"item_id": ids[0], "decision": "maybe"}], "keep or exclude"),
            ([{"item_id": group, "decision": "keep"}], "not a photo of this queue"),
            ([{"item_id": ids[0], "decision": "keep"}, {"item_id": ids[0], "decision": "keep"}],
             "twice"),
            ([{"item_id": 999999, "decision": "keep"}], "not one you can review"),
            ([], "nothing to save")):
        r = c.post("/api/feedback/photo-review/batch", json={"decisions": bad})
        assert r.status_code == 400 and why in r.json()["detail"], bad
    assert fstore.conn.execute("select count(*) from decisions").fetchone()[0] == 0
    as_user(c, "member-1")                                     # not their project's items
    r = c.post("/api/feedback/photo-review/batch",
               json={"decisions": [{"item_id": ids[0], "decision": "keep"}]})
    assert r.status_code == 400 and "not one you can review" in r.json()["detail"]


def test_the_seal_still_refuses_a_batch_on_a_sealed_record(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([photo_item("inat:1", "30", "R6 micrograph probe p=0.9")], "t")
    item = s.items_of("non_specimen_photo")[0]["id"]
    s.gate = fb.SealGate({"30": "2026-10-15"}.get, frozenset(), SEAL)
    with pytest.raises(fb.FeedbackError, match="sealed test"):
        pr.batch_decide(s, [{"item_id": item, "decision": "keep"}], {"sub": "a"})
    assert s.conn.execute("select count(*) from decisions").fetchone()[0] == 0


def test_counts_are_cached_until_the_store_changes(conn, tmp_path, monkeypatch):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    calls = []
    real = fb.Store.summary
    monkeypatch.setattr(fb.Store, "summary", lambda self, p=None: calls.append(1) or real(self, p))
    first = {t["type"]: t for t in c.get("/api/feedback/summary").json()}
    c.get("/api/feedback/summary")
    assert len(calls) == 1                                    # the second came from the cache
    tile = c.get("/api/feedback/photo-review/flagged").json()["tiles"][0]
    c.post("/api/feedback/photo-review/batch",
           json={"decisions": [{"item_id": tile["item_id"], "decision": "keep"}]})
    after = {t["type"]: t for t in c.get("/api/feedback/summary").json()}
    assert len(calls) == 2                                    # a decision: counted afresh
    assert after["non_specimen_photo"]["open"] == first["non_specimen_photo"]["open"] - 1


def test_flips_save_at_once_and_resending_the_same_state_adds_no_rows(conn, tmp_path):
    """Steve, 2026-10-21: a tile's flipper is its saved decision; the default Exclude is saved
    when the tile has been shown, a flip back to Include saves too, and the latest wins."""
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    tile = next(t for t in c.get("/api/feedback/photo-review/flagged").json()["tiles"]
                if t["photo_key"] == "inat:1000")
    shown = {"item_id": tile["item_id"], "decision": "exclude", "reason": "micrograph",
             "note": "summary: shown"}
    url = "/api/feedback/photo-review/batch"
    assert c.post(url, json={"decisions": [shown]}).json()["excluded"] == 1
    again = c.post(url, json={"decisions": [shown]}).json()          # a re-send: nothing new
    assert (again["excluded"], again["unchanged"]) == (0, 1)
    flip = c.post(url, json={"decisions": [{"item_id": tile["item_id"], "decision": "keep",
                                            "note": "summary: flipped"}]}).json()
    assert flip["kept"] == 1
    rows = [(d["decision"], d["note"]) for d in fstore.get(tile["item_id"])["decisions"]]
    assert rows == [("exclude_photo", "summary: shown"), ("keep_photo", "summary: flipped")]
    assert "inat:1000" not in fstore.excluded_photos()                 # the latest wins
    back = c.post(url, json={"decisions": [shown | {"note": "summary: flipped"}]}).json()
    assert back["excluded"] == 1 and "inat:1000" in fstore.excluded_photos()
    # A new reason is a new decision.
    other = c.post(url, json={"decisions": [shown | {"reason": "slip"}]}).json()
    assert other["excluded"] == 1 and len(fstore.get(tile["item_id"])["decisions"]) == 4


def test_a_decided_photo_reclassified_full_screen_takes_its_new_decision(conn, tmp_path):
    """Steve, 2026-10-21: a decided photo can be reclassified from the viewer. The batch writes
    the new decision on the decided item; the latest wins for the nightly exclusions."""
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    ids = {t["photo_key"]: t["item_id"] for t in c.get("/api/feedback/photo-review/flagged").json()["tiles"]}
    c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:1000"], "decision": "exclude", "reason": "micrograph", "note": "summary: shown"},
        {"item_id": ids["inat:1100"], "decision": "keep", "note": "summary: flipped"}]})
    assert fstore.excluded_photos() == {"inat:1000"}
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:1000"], "decision": "exclude", "reason": "slip", "note": "summary: flipped"},
        {"item_id": ids["inat:1100"], "decision": "exclude", "reason": "uv", "note": "summary: flipped"}]})
    assert r.status_code == 200 and r.json()["excluded"] == 2
    assert fstore.excluded_photos() == {"inat:1000", "inat:1100"}
    done = {t["photo_key"]: t["decision"] for t in
            c.get("/api/feedback/photo-review/flagged", params={"status": "decided"}).json()["tiles"]}
    assert done["inat:1000"]["reason"] == "slip" and done["inat:1100"]["reason"] == "uv"
    c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": ids["inat:1000"], "decision": "keep", "note": "summary: flipped"}]})
    assert fstore.excluded_photos() == {"inat:1100"}                # included again: leaves the list


def test_the_summary_serves_pages_of_up_to_500_photos(conn, tmp_path):
    """Steve, 2026-10-21: 150, 300 or 500 photos per page; the server serves up to 500."""
    from mycomap_vision import feedback_api
    from test_photo_review import photo_item
    assert feedback_api.SUMMARY_MAX_PAGE == 500
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([photo_item(f"inat:{9000 + i}", "100", f"R6 micrograph probe p=0.{i % 90 + 10}")
                 for i in range(520)], "many")
    as_user(c, "admin-1")
    page = lambda n: c.get("/api/feedback/photo-review/flagged", params={"limit": n}).json()  # noqa: E731
    assert len(page(300)["tiles"]) == 300
    assert len(page(500)["tiles"]) == 500
    big = page(1000)
    assert big["total"] == 523 and len(big["tiles"]) == 500                # capped at 500
