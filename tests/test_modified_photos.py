"""Research > Excluded photos > Modified photos (modified_photos.py, api.py; Steve, 2026-10-27):
each field-slip photo whose Recovered labels crops go back into Vision, before and after, with
"where Vision looked" on both. Which photos are listed (open crops too: Steve, "lets have it use
open crops"), in what order, whose box each crop has, what is shown to whom (as on the rest of
the page: a photo anyone may see, the seal, never who decided), which kept map each photo and
crop shows, and "no map possible" (2026-10-28): kept with its reason, shown, and not tried
again by the preload while what is served stays the same."""

import json
from datetime import datetime, timedelta, timezone

import pytest
from test_excluded_looked_api import ANN, LOOKED, PERSONAL, site
from test_excluded_preload import any_time, keep_crop, tick  # noqa: F401 - fixtures
from test_run_links import person, visitor

from mycomap_vision import excluded_looked as xl
from mycomap_vision import feedback as fb
from mycomap_vision import looked
from mycomap_vision import modified_photos as mp
from mycomap_vision import recovered_labels as rl

MODIFIED = "/api/research/excluded-photos/modified"
QUARTER = [0.25, 0.25, 0.75, 0.75]


def rl_row(key, record, boxes, w=40, h=40):
    """A Recovered labels item as the seed makes it (recovered_labels.rows_from_slip_crops)."""
    return {"item_type": rl.ITEM_TYPE, "subject": key, "granularity": "photo",
            "rule": rl.RULE + " (slip p=0.80)", "model_flagged": True, "record_keys": [record],
            "links": [fb.link(record)], "current": "", "projects": [],
            "proposed": "include the specimen crop(s) in Vision",
            "evidence": {"version": "t1", "width": w, "height": h, "slip_p": 0.8,
                         "tranche": "high", "slip_area": 0.3, "specimen_area": 0.2,
                         "on_slip_area": 0.0,
                         "crops": [{"n": i + 1, "box": b} for i, b in enumerate(boxes)]}}


def store_of(tmp_path):
    return fb.Store(tmp_path / "state" / "human-feedback.sqlite")


def rl_ids(s):
    return {i["subject"]: i["id"] for i in s.items_of(rl.ITEM_TYPE)}


def seeded(conn, tmp_path, maker=None):
    """site() (slips 1003, record 103, and 1005, record 105, past the seal) plus Recovered labels
    items: 1003 open, one crop; 1005 reviewed, two crops, keeps crop 2; 1001 (kept in Vision,
    not a slip) open; 1000 (a micrograph) open."""
    app, maker, _ = site(conn, tmp_path, maker=maker)
    s = store_of(tmp_path)
    s.seed([rl_row("inat:1003", "103", [[10, 10, 30, 30]]),
            rl_row("inat:1005", "105", [[0, 0, 20, 20], [20, 20, 40, 40]]),
            rl_row("inat:1001", "101", [[0, 0, 20, 20]]),
            rl_row("inat:1000", "100", [[0, 0, 20, 20]])], "slipcrops")
    s.decide(rl_ids(s)["inat:1005"], "include_crops", ANN, value="[2]", note="my private note")
    return app, maker, s


def body(app, **params):
    res = visitor(app).get(MODIFIED, params=params)
    assert res.status_code == 200, res.text
    return res.json()


def keys(b):
    return [c["photo_key"] for c in b["cards"]]


# --- which photos, in what order -----------------------------------------------------------------

def test_open_and_reviewed_slip_photos_are_listed_to_anyone_reviewed_first(conn, tmp_path):
    app, _, _ = seeded(conn, tmp_path)
    b = body(app)
    # 1001 is kept in Vision and 1000 left out as a micrograph: not field slips, not listed.
    assert keys(b) == ["inat:1005", "inat:1003"] and b["total"] == 2
    reviewed, open_ = b["cards"]
    assert reviewed["decided"] is True and [k["n"] for k in reviewed["crops"]] == [2]
    assert open_["decided"] is False and [k["n"] for k in open_["crops"]] == [1]
    assert b["include_open"] is True


def test_a_photo_whose_crops_were_all_rejected_is_not_listed(conn, tmp_path):
    app, _, s = seeded(conn, tmp_path)
    s.decide(rl_ids(s)["inat:1003"], "reject_crops", ANN)
    assert keys(body(app)) == ["inat:1005"]


def test_who_decided_and_their_note_are_never_shown(conn, tmp_path):
    app, _, _ = seeded(conn, tmp_path)
    text = visitor(app).get(MODIFIED).text
    assert not [p for p in PERSONAL if p in text]


def test_each_crop_is_the_square_the_model_reads_and_says_whose_box_it_is(conn, tmp_path):
    app, _, s = seeded(conn, tmp_path)
    # A reviewer moves 1003's box (a rectangle: the model reads its central square).
    s.decide(rl_ids(s)["inat:1003"], "include_crops", ANN,
             value=json.dumps({"include": [1], "boxes": {"1": [0, 0, 40, 20]}}))
    b = body(app)
    crop = next(c for c in b["cards"] if c["photo_key"] == "inat:1003")["crops"][0]
    assert crop["box"] == [10, 0, 30, 20] and crop["source"] == "reviewer"
    assert crop["fraction"] == [0.25, 0.0, 0.75, 0.5]
    assert b["counts"] == {"all": 2, "edited": 1, "detector": 1}
    assert keys(body(app, filter="edited")) == ["inat:1003"]
    assert keys(body(app, filter="detector")) == ["inat:1005"]
    assert visitor(app).get(MODIFIED, params={"filter": "mine"}).status_code == 400


def test_a_page_at_a_time():
    items = [{"id": i, "subject": f"inat:{i}", "status": "open", "record_keys": [str(i)],
              "evidence": {"width": 40, "height": 40, "crops": [{"n": 1, "box": [0, 0, 20, 20]}]}}
             for i in range(1, 6)]
    rows = mp.ordered(items)
    got = mp.page(rows, {f"inat:{i}" for i in range(1, 6)}, offset=1, limit=2)
    assert [c["photo_key"] for c in got["cards"]] == ["inat:4", "inat:3"] and got["total"] == 5
    assert "_order" not in got["cards"][0]
    assert mp.page(rows, set(), limit=10_000)["total"] == 0
    assert len(mp.page(rows, {f"inat:{i}" for i in range(1, 6)}, limit=10_000)["cards"]) == 5
    got["cards"][0]["crops"][0]["map"] = "x"                  # a page never changes the cache
    assert "map" not in rows[1]["crops"][0]


# --- what is shown, and to whom --------------------------------------------------------------------

def test_a_photo_not_anyone_may_see_is_a_placeholder_with_no_crops_or_maps(conn, tmp_path):
    app, _, s = seeded(conn, tmp_path)
    # 1100 is all rights reserved; a person now leaves it out as a field slip.
    item = next(i for i in s.items_of("non_specimen_photo") if i["subject"] == "inat:1100")
    s.decide(item["id"], "exclude_photo", ANN, value="slip")
    s.seed([rl_row("inat:1100", "100", [[0, 0, 20, 20]])], "more")
    card = next(c for c in body(app)["cards"] if c["photo_key"] == "inat:1100")
    assert card["shown"] == "hidden" and card["photo_url"] is None and card["crops"] == []
    assert card["owner_login"] is None and card["map"] is None


def test_the_seal_holds_back_the_maps_of_records_past_it(conn, tmp_path, any_time):
    app, maker, _ = seeded(conn, tmp_path)
    # A map of 1005 kept anyway (made before the seal rule, say): still never shown.
    app.state.excluded_preload.skip_reason = lambda t: None
    tick(app)
    assert "B y" in [c["name"] for c in maker.calls] and len(maker.calls) == 2
    card = next(c for c in body(app)["cards"] if c["photo_key"] == "inat:1005")
    assert card["held"] is True and card["shown"] == "public" and card["photo_url"]
    assert card["map"] is None and all(k["map"] is None for k in card["crops"])


def test_switched_off_there_is_nothing_here(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, excluded=False)
    assert visitor(app).get(MODIFIED).status_code == 404


# --- the maps on both sides ----------------------------------------------------------------------

def test_before_and_after_show_the_kept_maps_once_made_and_map_coming_before(conn, tmp_path,
                                                                           any_time):
    app, _, _ = seeded(conn, tmp_path)
    app.state.test_root = tmp_path / "emb"
    card = body(app)["cards"][1]
    assert card["photo_key"] == "inat:1003" and card["map"] is None and card["no_map"] is None
    assert card["crops"][0]["map"] is None                       # map coming
    keep_crop(conn, app, 1003, 1, tuple(QUARTER))               # the sync and the nightly
    tick(app)
    card = body(app)["cards"][1]
    assert card["map"]["rank"] == 3 and card["map"]["name"] == "B y"
    assert card["crops"][0]["map"]["map"]["grid"] == 2
    assert keys(body(app, maps=1)) == ["inat:1003"]


def test_a_crops_map_made_for_its_old_box_is_not_shown_after_a_reviewer_moves_it(conn, tmp_path,
                                                                                any_time):
    app, _, s = seeded(conn, tmp_path)
    app.state.test_root = tmp_path / "emb"
    keep_crop(conn, app, 1003, 1, tuple(QUARTER))
    tick(app)
    s.decide(rl_ids(s)["inat:1003"], "include_crops", ANN,
             value=json.dumps({"include": [1], "boxes": {"1": [0, 0, 20, 20]}}))
    card = next(c for c in body(app)["cards"] if c["photo_key"] == "inat:1003")
    assert card["crops"][0]["source"] == "reviewer" and card["crops"][0]["map"] is None
    assert card["map"] is not None                               # the photo's own still holds


# --- no map possible (2026-10-28) ------------------------------------------------------------------

def never(*args):
    raise looked.NeverMap("no other DNA-verified record of B y to compare with")


def test_no_map_possible_is_kept_shown_with_why_and_not_tried_again(conn, tmp_path, any_time):
    calls = []

    def maker(*args):
        calls.append(1)
        never()
    app, _, _ = seeded(conn, tmp_path, maker=maker)
    tick(app)
    assert len(calls) == 1
    h = visitor(app).get("/api/health").json()["excluded_preload"]
    assert h["failed"] == 0 and h["no_map_found"] == 1 and h["skipped"] == {"held by the seal": 1}
    got = visitor(app).get(LOOKED, params={"photo": "inat:1003"})
    assert got.status_code == 200 and got.json()["status"] == "no_map"
    assert got.json()["reason"] == "no other DNA-verified record of B y to compare with"
    card = body(app)["cards"][1]
    assert card["no_map"] == got.json()["reason"] and card["map"] is None
    # The next pass counts it, and makes nothing; nor does a person's ask.
    app.state.excluded_preload.pass_ = None
    tick(app)
    assert len(calls) == 1
    assert app.state.excluded_preload.health()["no_map_possible"] == {"slips": 1, "crops": 0}
    res = person(app, "m").post(LOOKED, params={"photo": "inat:1003"})
    assert res.status_code == 200 and res.json()["status"] == "no_map" and len(calls) == 1


def test_no_map_possible_is_tried_again_when_the_records_name_changes(conn, tmp_path, any_time):
    calls = []

    def maker(*args):
        calls.append(1)
        never()
    app, _, _ = seeded(conn, tmp_path, maker=maker)
    tick(app)
    conn.execute("update records set species = 'A x' where observation_id = '103'")
    conn.commit()
    app.state.excluded_preload.pass_ = None
    tick(app)
    assert len(calls) == 2


def test_no_map_possible_holds_for_a_week_then_is_tried_again():
    stamp = {"map_version": 1, "backbone": "m1", "method": "x", "release": "r1", "name": "A x",
             "box": None}
    made = datetime(2026, 10, 28, 9, tzinfo=timezone.utc)
    doc = {"reason": "r", "stamp": stamp, "made_at": made.strftime("%Y-%m-%dT%H:%M:%SZ")}
    assert xl.no_map_current(doc, stamp, made + timedelta(days=6, hours=23))
    assert not xl.no_map_current(doc, stamp, made + timedelta(days=xl.NO_MAP_DAYS))
    assert not xl.no_map_current(doc, {**stamp, "release": "r2"}, made)
    assert not xl.no_map_current(None, stamp, made)


def test_a_map_made_later_replaces_a_kept_no_map(tmp_path):
    maps = xl.Maps(tmp_path / "m.sqlite")
    maps.keep_no_map("inat:1", "v1", "why", {"name": "A x"}, "2026-10-28T09:00:00Z")
    assert maps.no_map("inat:1")["reason"] == "why" and set(maps.no_maps()) == {"inat:1"}
    maps.keep("inat:1", "v1", {"made_at": "2026-10-29T09:00:00Z", "map": {}})
    assert maps.no_map("inat:1") is None and maps.no_maps() == {}


def test_a_map_that_failed_for_another_reason_is_still_refused_and_tried_again(conn, tmp_path,
                                                                             any_time):
    def busy(*args):
        raise looked.NoMap("the photo could not be read just now")
    app, _, _ = seeded(conn, tmp_path, maker=busy)
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003"}).status_code == 409
    tick(app)
    assert app.state.excluded_preload.health()["failed"] == 1
    assert body(app)["cards"][1]["no_map"] is None


# --- the preload's log and its give-up --------------------------------------------------------------

def test_every_failed_map_is_logged_with_why_and_given_up_after_three_tries():
    from test_excluded_preload import hit, preloader
    p, w, made = preloader(slips=[hit(1)])
    said = []
    p.note = said.append
    for attempt in range(1, 5):
        p.pass_ = None
        p.step()
        if w.jobs:
            _, _, done = w.jobs.pop()
            done(False, "ReadTimeout: the photo took too long", False)
    assert made == ["inat:1"] * xl.GIVE_UP_AFTER
    fails = [s for s in said if s.startswith("[excluded preload] map failed: inat:1: ")]
    assert len(fails) == 3 and "ReadTimeout: the photo took too long" in fails[0]
    assert fails[-1].endswith("given up until the server restarts)")
    h = p.health()
    assert h["failed"] == 3 and h["given_up"] == 1 and h["skipped"] == {"given up": 1}


def test_a_no_map_outcome_is_logged_and_counted_apart_from_failures():
    from test_excluded_preload import hit, preloader
    p, w, _ = preloader(slips=[hit(1)])
    said = []
    p.note = said.append
    p.step()
    _, _, done = w.jobs.pop()
    done(False, "NeverMap: no other DNA-verified record of A x to compare with", True)
    assert said[-1] == ("[excluded preload] no map possible: inat:1: NeverMap: no other "
                        "DNA-verified record of A x to compare with")
    h = p.health()
    assert h["failed"] == 0 and h["no_map_found"] == 1 and h["given_up"] == 0
    assert h["no_map_possible"] == {"slips": 1, "crops": 0}
