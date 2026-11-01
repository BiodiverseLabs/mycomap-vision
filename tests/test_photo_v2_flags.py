"""Tagger v2's flags in "Photos that show no specimen" (Steve, 2026-10-21: "so I don't have to
manually do all the field slips out of microscopy, and we have more levels and results for all
categories"). A v2 flag ("R9 v2 slip p=0.83 (high)") decides the photo's tab over the older
flags it carries, so a field slip the micrograph probe caught moves to Field slips; its band
(high / medium / low) filters every tab; a photo v2 reads as a specimen photo starts as
Include, and every other one as Exclude, low band included (Steve, 2026-10-21)."""

import pytest
from test_feedback_api import as_user
from test_photo_review import photo_item, site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import photo_review as pr

INAT = "https://www.inaturalist.org/observations/"


def item(i, key, record, rule, status="open"):
    return {"id": i, "subject": key, "rule": rule, "status": status, "links": [INAT + record],
            "current": "A x"}


def items():
    return [
        item(1, "inat:1", "10", "R6 micrograph probe p=0.90; R9 v2 slip p=0.96 (high)"),
        item(2, "inat:2", "10", "R6 micrograph probe p=0.80; R9 v2 micrograph p=0.88 (high)"),
        item(3, "inat:3", "11", "R6 micrograph probe p=0.70; R9 v2 specimen p=0.91"),
        item(4, "inat:4", "12", "R9 v2 uv p=0.41 (low)"),
        item(5, "inat:5", "12", "R9 v2 uv p=0.62 (medium)"),
        item(6, "inat:6", "13", "R6 view habitat p=0.30"),                   # no v2 flag
    ]


def test_a_v2_flag_puts_the_photo_in_its_own_tab_over_the_older_flags():
    f = pr.parse_flags("R6 micrograph probe p=0.90; R9 v2 slip p=0.96 (high)")
    assert [(x["tab"], x.get("band")) for x in f] == [("micrograph", None), ("slip", "high")]
    assert [x["tab"] for x in pr.placing_flags("R6 micrograph probe p=0.90; R9 v2 slip p=0.96 (high)")] == ["slip"]
    # a later v2 flag (a newer tagger) wins
    assert [x["tab"] for x in pr.placing_flags("R9 v2 slip p=0.6 (medium); R9 v2 dried p=0.7 (high)")] == ["dried"]
    assert [x["tab"] for x in pr.placing_flags("R6 view habitat p=0.30")] == ["habitat"]
    micro = [t["photo_key"] for t in pr.flagged_tiles(items(), "micrograph")]
    slips = [t["photo_key"] for t in pr.flagged_tiles(items(), "slip")]
    assert micro == ["inat:2"] and slips == ["inat:1"]                     # the slip left Microscopy
    tabs = {t["code"]: t["open"] for t in pr.cards(items())["tabs"]}
    assert tabs["micrograph"] == 1 and tabs["slip"] == 1 and tabs["specimen"] == 1


def test_bands_filter_every_tab_and_the_tabs_count_them():
    assert [t["photo_key"] for t in pr.flagged_tiles(items(), "uv", band="low")] == ["inat:4"]
    assert [t["photo_key"] for t in pr.flagged_tiles(items(), "uv", band="medium")] == ["inat:5"]
    assert [t["photo_key"] for t in pr.flagged_tiles(items(), "all", band="high")] == ["inat:1", "inat:2"]
    assert pr.flagged_tiles(items(), "habitat", band="high") == []           # no v2 flag: All only
    assert [t["photo_key"] for t in pr.flagged_tiles(items(), "habitat")] == ["inat:6"]
    uv = next(t for t in pr.cards(items(), "uv")["tabs"] if t["code"] == "uv")
    assert uv["bands"] == {"high": 0, "medium": 1, "low": 1, "very_low": 0}
    assert [r["record_key"] for r in pr.cards(items(), "uv", band="low")["records"]] == ["12"]
    with pytest.raises(fb.FeedbackError):
        pr.flagged_tiles(items(), "uv", band="certain")


# The bottom band (Steve, 2026-11-01: "a new Low category bottom confidence after the retrain"),
# filled first by the UV retrain's re-score of the open UV tab ("R11.1 v2 uv p=... (very_low)").

def test_very_low_is_the_bottom_band_and_any_v2_flag_can_carry_it():
    assert pr.BANDS[-1] == "very_low" and pr.BANDS[:3] == ("high", "medium", "low")
    f = pr.parse_flags("R11 v2 uv p=0.45 (low); R11.1 v2 uv p=0.0400 (very_low)")
    assert [(x["tab"], x["band"], x["p"]) for x in f] == [("uv", "low", 0.45), ("uv", "very_low", 0.04)]
    assert pr.band_of(pr.placing_flags("R11 v2 uv p=0.45 (low); R11.1 v2 uv p=0.0400 (very_low)"), "uv") == "very_low"
    # another tab may use it later
    assert pr.band_of(pr.parse_flags("R12 v2 slip p=0.05 (very_low)"), "slip") == "very_low"


def test_a_uv_rescore_rebands_and_reranks_open_photos_and_never_touches_decided_ones(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([photo_item("inat:1", "10", "R11 v2 uv p=0.450 (low)"),
            photo_item("inat:2", "11", "R11 v2 uv p=0.600 (low)"),
            photo_item("inat:3", "12", "R11 v2 uv p=0.550 (low)")], "r11")
    three = next(i for i in s.items_of(pr.ITEM_TYPE) if i["subject"] == "inat:3")
    s.decide(three["id"], "keep_photo", {"sub": "a"})
    before = next(i for i in s.items_of(pr.ITEM_TYPE) if i["subject"] == "inat:3")
    res = s.seed([photo_item("inat:1", "10", "R11.1 v2 uv p=0.9100 (high)"),
                  photo_item("inat:2", "11", "R11.1 v2 uv p=0.0300 (very_low)"),
                  photo_item("inat:3", "12", "R11.1 v2 uv p=0.0100 (very_low)")], "uv-r11.1")
    assert (res["added"], res["flags_added"], res["already_decided"]) == (0, 2, 1)
    items = s.items_of(pr.ITEM_TYPE)
    open_ = pr.flagged_tiles(items, "uv")
    assert [(t["photo_key"], t["band"]) for t in open_] == [("inat:1", "high"), ("inat:2", "very_low")]  # new rank
    assert [t["photo_key"] for t in pr.flagged_tiles(items, "uv", band="very_low")] == ["inat:2"]
    assert pr.flagged_tiles(items, "uv", band="low") == []
    uv = next(t for t in pr.cards(items, "uv")["tabs"] if t["code"] == "uv")
    assert uv["bands"] == {"high": 1, "medium": 0, "low": 0, "very_low": 1}
    after = next(i for i in items if i["subject"] == "inat:3")
    assert after["rule"] == before["rule"] == "R11 v2 uv p=0.550 (low)"        # decided: unchanged
    assert pr.item_outcome(after, "uv") == "included"
    done = pr.flagged_tiles(items, "uv", status="decided")
    assert [(t["photo_key"], t["band"]) for t in done] == [("inat:3", "low")]
    # Steve, 2026-11-01: "Very low starts as Include"; the high one still starts as Exclude
    vl = next(t for t in open_ if t["photo_key"] == "inat:2")
    assert (vl["suggested_decision"], vl["suggested_reason"]) == ("keep", "uv")
    hi = next(t for t in open_ if t["photo_key"] == "inat:1")
    assert (hi["suggested_decision"], hi["suggested_reason"]) == ("exclude", "uv")


def test_very_low_photos_start_as_include_and_every_other_band_as_exclude():
    def start(rule):
        return pr.suggested_decision(pr.placing_flags(rule))
    assert start("R11.1 v2 uv p=0.0400 (very_low)") == "keep"
    assert start("R12 v2 slip p=0.0500 (very_low)") == "keep"                 # any tab's very_low
    for band in ("high", "medium", "low"):
        assert start(f"R11.1 v2 uv p=0.5000 ({band})") == "exclude"
    assert start("R9 v2 uv p=0.41") == "exclude"                              # no band
    assert start("R9 v2 specimen p=0.91") == "keep"                           # as before
    assert start("R6 micrograph probe p=0.90") == "exclude"                   # no v2 flag
    # only the latest v2 flag counts: a very_low re-score after a low flag starts as Include,
    # a newer low flag after a very_low one as Exclude
    assert start("R11 v2 uv p=0.45 (low); R11.1 v2 uv p=0.0400 (very_low)") == "keep"
    assert start("R11.1 v2 uv p=0.0400 (very_low); R12 v2 uv p=0.30 (low)") == "exclude"


def test_only_specimen_photos_start_as_include_a_low_band_photo_in_a_group_starts_excluded():
    t = {x["photo_key"]: x for x in pr.flagged_tiles(items(), "all")}
    assert t["inat:3"]["suggested_decision"] == "keep" and t["inat:3"]["suggested_reason"] is None
    assert t["inat:4"]["suggested_decision"] == "exclude" and t["inat:4"]["suggested_reason"] == "uv"
    assert t["inat:5"]["suggested_decision"] == "exclude"
    assert t["inat:1"]["suggested_decision"] == "exclude" and t["inat:1"]["suggested_reason"] == "slip"
    assert t["inat:6"]["suggested_decision"] == "exclude"                   # v1 flags as before


def test_a_v2_seed_moves_open_photos_and_never_touches_decided_ones(tmp_path):
    s = fb.Store(tmp_path / "f.sqlite")
    s.seed([photo_item("inat:1", "10", "R6 micrograph probe p=0.90"),
            photo_item("inat:2", "10", "R6 micrograph probe p=0.80")], "first")
    two = next(i for i in s.items_of(pr.ITEM_TYPE) if i["subject"] == "inat:2")
    s.decide(two["id"], "exclude_photo", {"sub": "a"}, value="micrograph")
    s.seed([photo_item("inat:1", "10", "R9 v2 slip p=0.960 (high)"),
            photo_item("inat:2", "10", "R9 v2 slip p=0.950 (high)"),
            photo_item("inat:9", "14", "R9 v2 dried p=0.800 (high)")], "v2")
    open_ = pr.flagged_tiles(s.items_of(pr.ITEM_TYPE), "slip")
    assert [t["photo_key"] for t in open_] == ["inat:1"]
    assert pr.flagged_tiles(s.items_of(pr.ITEM_TYPE), "micrograph") == []
    done = pr.flagged_tiles(s.items_of(pr.ITEM_TYPE), "micrograph", status="decided")
    assert [t["photo_key"] for t in done] == ["inat:2"]                       # stays where decided
    assert [t["photo_key"] for t in pr.flagged_tiles(s.items_of(pr.ITEM_TYPE), "dried")] == ["inat:9"]


def test_the_api_filters_by_band_and_refuses_an_unknown_one(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([photo_item("inat:1000", "100", "R9 v2 slip p=0.900 (high)"),
                 photo_item("inat:1100", "100", "R9 v2 specimen p=0.800")], "v2")
    as_user(c, "admin-1")
    hi = c.get("/api/feedback/photo-review/flagged", params={"tab": "slip", "band": "high"}).json()
    assert [t["photo_key"] for t in hi["tiles"]] == ["inat:1000"] and hi["tiles"][0]["band"] == "high"
    sp = c.get("/api/feedback/photo-review/flagged", params={"tab": "specimen"}).json()
    assert [(t["photo_key"], t["suggested_decision"]) for t in sp["tiles"]] == [("inat:1100", "keep")]
    cards = c.get("/api/feedback/photo-review", params={"tab": "slip", "band": "high"}).json()
    assert cards["total"] == 1
    assert next(t for t in cards["tabs"] if t["code"] == "slip")["bands"]["high"] == 1
    photo = next(p for p in cards["records"][0]["photos"] if p["photo_key"] == "inat:1000")
    assert photo["band"] == "high" and photo["suggested_decision"] == "exclude"
    assert c.get("/api/feedback/photo-review/flagged", params={"band": "certain"}).status_code == 400


def test_the_api_filters_by_the_very_low_band_and_counts_it(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([photo_item("inat:1000", "100", "R11 v2 uv p=0.500 (low)"),
                 photo_item("inat:1000", "100", "R11.1 v2 uv p=0.0400 (very_low)"),
                 photo_item("inat:1001", "101", "R11.1 v2 uv p=0.8000 (high)")], "uv")
    as_user(c, "admin-1")
    vl = c.get("/api/feedback/photo-review/flagged", params={"tab": "uv", "band": "very_low"})
    assert vl.status_code == 200
    assert [(t["photo_key"], t["band"]) for t in vl.json()["tiles"]] == [("inat:1000", "very_low")]
    assert c.get("/api/feedback/photo-review/flagged", params={"tab": "uv", "band": "low"}).json()["tiles"] == []
    cards = c.get("/api/feedback/photo-review", params={"tab": "uv", "band": "very_low"}).json()
    assert [r["record_key"] for r in cards["records"]] == ["100"]
    assert next(t for t in cards["tabs"] if t["code"] == "uv")["bands"] == {
        "high": 1, "medium": 0, "low": 0, "very_low": 1}
    assert c.get("/api/feedback/photo-review", params={"band": "very-low"}).status_code == 400


def test_the_api_starts_very_low_photos_as_include_in_the_summary_and_full_views(conn, tmp_path):
    c, fstore = site_with_photos(conn, tmp_path)
    fstore.seed([photo_item("inat:1000", "100", "R11.1 v2 uv p=0.0400 (very_low)"),
                 photo_item("inat:1001", "101", "R11.1 v2 uv p=0.8000 (high)")], "uv")
    as_user(c, "admin-1")
    tiles = {t["photo_key"]: t for t in
             c.get("/api/feedback/photo-review/flagged", params={"tab": "uv"}).json()["tiles"]}
    assert (tiles["inat:1000"]["suggested_decision"], tiles["inat:1000"]["suggested_reason"]) == ("keep", "uv")
    assert tiles["inat:1001"]["suggested_decision"] == "exclude"
    cards = {r["record_key"]: r for r in
             c.get("/api/feedback/photo-review", params={"tab": "uv"}).json()["records"]}
    vl = next(p for p in cards["100"]["photos"] if p["photo_key"] == "inat:1000")
    assert (vl["band"], vl["suggested_decision"], vl["suggested_reason"]) == ("very_low", "keep", "uv")
    hi = next(p for p in cards["101"]["photos"] if p["photo_key"] == "inat:1001")
    assert (hi["band"], hi["suggested_decision"]) == ("high", "exclude")
