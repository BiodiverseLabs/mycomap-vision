"""The photo-type tagger's tabs in "Photos that show no specimen" (Steve, 2026-10-21): UV
reactions, phylogenetic trees and BLAST, drawings. A seed adds its flag to a photo already
queued and open (it then shows in both tabs) and never touches or reopens a decided one; a photo
already left out for a tab's reason shows on that tab's Completed list; the page sees a seed's
new flags without a restart; and the Summary's own tab list stays the server's."""

import re
from pathlib import Path

from test_feedback_api import as_user
from test_photo_review import items_for_cards, photo_item, site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr

ADMIN = {"sub": "admin-1", "name": "A"}


def test_each_tagger_flag_goes_to_its_own_tab():
    assert [f["tab"] for f in pr.parse_flags("R7 uv probe p=0.99")] == ["uv"]
    assert [f["tab"] for f in pr.parse_flags("R7 tree probe p=0.97")] == ["tree"]
    assert [f["tab"] for f in pr.parse_flags("R7 drawing probe p=1.00")] == ["drawing"]
    both = pr.parse_flags("R6 micrograph probe p=0.62; R7 uv probe p=0.99")
    assert [(f["tab"], f["p"]) for f in both] == [("micrograph", 0.62), ("uv", 0.99)]
    assert pr.suggested_reason(pr.parse_flags("R7 tree probe p=0.97"), "tree") == "phylo_tree"
    assert pr.suggested_reason(pr.parse_flags("R7 drawing probe p=0.97"), "drawing") == "drawing"
    assert pr.suggested_reason(pr.parse_flags("R7 uv probe p=0.99"), "uv") == "uv"
    assert {"uv", "tree", "drawing"} <= set(pr.TAB_CODES) and pr.TAB_CODES[-1] == "other"


def test_a_seed_adds_its_flag_to_an_open_photo_and_never_reopens_a_decided_one(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([photo_item("inat:1", "10", "R6 micrograph probe p=0.90"),
            photo_item("inat:2", "10", "R6 view habitat p=0.30")], "first")
    two = next(i for i in s.items_of(pr.ITEM_TYPE) if i["subject"] == "inat:2")
    s.decide(two["id"], "keep_photo", ADMIN)
    out = s.seed([photo_item("inat:1", "10", "R7 uv probe p=0.99"),
                  photo_item("inat:2", "10", "R7 uv probe p=0.98"),
                  photo_item("inat:3", "11", "R7 uv probe p=0.99")], "tagger")
    assert (out["added"], out["flags_added"], out["already_decided"]) == (1, 1, 1)
    by = {i["subject"]: i for i in s.items_of(pr.ITEM_TYPE)}
    assert by["inat:1"]["rule"] == "R6 micrograph probe p=0.90; R7 uv probe p=0.99"
    assert by["inat:1"]["status"] == "open"
    assert by["inat:2"]["rule"] == "R6 view habitat p=0.30"          # decided: untouched
    assert by["inat:2"]["status"] == "decided"
    again = s.seed([photo_item("inat:1", "10", "R7 uv probe p=0.99")], "tagger")
    assert (again["added"], again["flags_added"]) == (0, 0)           # never doubled
    assert s.items_of(pr.ITEM_TYPE)[0]["rule"].count("uv probe") == 1
    tabs = {t["code"]: (t["open"], t["decided"]) for t in pr.cards(s.items_of(pr.ITEM_TYPE))["tabs"]}
    assert tabs["uv"] == (2, 0) and tabs["micrograph"] == (1, 0)    # record 10 is in both


def test_other_queues_still_leave_a_queued_item_as_it_is(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    row = {"item_type": "group_label", "subject": "Laccaria laccata group", "rule": "R2"}
    s.seed([row], "a")
    out = s.seed([row | {"rule": "R9"}], "b")
    assert out == {"seed": "b", "rows": 1, "added": 0}
    assert s.items("group_label")["items"][0]["rule"] == "R2"


def test_a_photo_left_out_for_a_tabs_reason_shows_on_that_tabs_completed_list():
    items = items_for_cards() + [
        {"id": 5, "subject": "inat:5", "rule": "R6 micrograph probe p=0.65", "status": "decided",
         "links": ["https://www.inaturalist.org/observations/13"], "current": "D w",
         "last_decision": {"decision": "exclude_photo", "value": "uv", "by_name": "S"}},
        {"id": 6, "subject": "inat:6", "rule": "R6 view habitat p=0.20", "status": "decided",
         "links": ["https://www.inaturalist.org/observations/14"], "current": "E v",
         "last_decision": {"decision": "keep_photo", "value": None, "by_name": "S"}}]
    tabs = {t["code"]: (t["open"], t["decided"]) for t in pr.cards(items, "uv")["tabs"]}
    assert tabs["uv"] == (0, 1)
    done = pr.cards(items, "uv", status="decided")["records"]
    assert [r["record_key"] for r in done] == ["13"]
    assert pr.cards(items, "uv")["records"] == []                     # nothing open there
    tiles = pr.flagged_tiles(items, "uv", status="decided")
    assert [t["photo_key"] for t in tiles] == ["inat:5"]
    assert tiles[0]["decision"]["reason"] == "uv"
    # still in its flag's tab too, and a kept photo joins no reason's tab
    assert "inat:5" in [t["photo_key"] for t in pr.flagged_tiles(items, "micrograph", status="decided")]
    assert all("inat:6" not in [t["photo_key"] for t in pr.flagged_tiles(items, c, status="decided")]
               for c in pr.TAB_CODES if c != "habitat")


def test_the_page_shows_a_seeds_new_flags_without_a_restart(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    as_user(c, "admin-1")
    assert c.get("/api/feedback/photo-review/flagged", params={"tab": "uv"}).json()["total"] == 0
    c.get("/api/feedback/photo-review", params={"tab": "uv"})          # cache the views
    fstore.seed([photo_item("inat:1000", "100", "R7 uv probe p=0.99")], "tagger")
    tiles = c.get("/api/feedback/photo-review/flagged", params={"tab": "uv"}).json()["tiles"]
    assert [t["photo_key"] for t in tiles] == ["inat:1000"]
    assert tiles[0]["suggested_reason"] == "uv"
    tabs = {t["code"]: t["open"] for t in
            c.get("/api/feedback/photo-review", params={"tab": "uv"}).json()["tabs"]}
    assert tabs["uv"] == 1 and tabs["micrograph"] == 2


def test_the_summarys_tab_list_is_the_servers():
    src = (Path(__file__).parents[1] / "web" / "src" / "lib" / "photoSummary.ts").read_text(
        encoding="utf-8")
    block = src[src.index("export const SUMMARY_TABS"):]
    block = block[:block.index("];")]
    assert re.findall(r'code: "(\w+)", title: "([^"]+)"', block) == list(pr.TABS)
    api = (Path(__file__).parents[1] / "web" / "src" / "lib" / "api.ts").read_text(encoding="utf-8")
    union = api[api.index("export type PhotoReviewTab"):]
    union = union[:union.index(";")]
    assert set(re.findall(r'"(\w+)"', union)) == set(pr.TAB_CODES) | {"all"}
