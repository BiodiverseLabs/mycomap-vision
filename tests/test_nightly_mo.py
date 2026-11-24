"""The nightly Mushroom Observer step (Steve, 2026-11-24: "sure").

Without it, an MO record that turned green after the layer began never got MO's details on the
box (only `mv fetch-mo`, on the laptop, read them), so it had no photos and no model served it:
1,140 North-America records by 24 Nov 2026 (reports/v2-beta-deploy/GAPS.md, gap 2). With
MV_NIGHTLY_MO=1 each night asks MO, at MO's pace, about every green MO record with no MO answer,
whatever its first-seen date, at most MV_NIGHTLY_MAX_MO a night; their photos then go through
the night's photo steps like a new record's, under MO's use rule. No test here reaches MO: every
MO answer is played."""

import gzip
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from test_models_and_scoreboard import Const, seed_two_species
from test_nightly import QUIET, RID, FakeOutside, row, seed_rows
from test_sources import mo_image

from mycomap_vision import config, manifest, mo, nightly, nightly_notes
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.identify import Identifier
from mycomap_vision.inat import save_batch
from mycomap_vision.photos import pending_photos
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

from conftest import inat_obs

OPEN = "inaturalist-open-data.s3.amazonaws.com"
ON = nightly.Settings(mo=True)
CC = "Creative Commons Wikipedia Compatible v3.0"
ARR = "Copyright Reserved"
BEFORE_LAYER = "2026-09-01T00:00:00+00:00"


def mo_row(n, name="C z", when="10/1/2026"):
    return {"source": "MO Observations", "observation_id": str(n), "scientific_name": name,
            "genus": name.split()[0], "family": "F", "continent": "North America",
            "validation_status_1": "yes", "validation_date_1": when}


def pid(image_id):
    return mo.photo_id_of(image_id)


class MOOutside(FakeOutside):
    """FakeOutside, with Mushroom Observer played: `images` maps an MO observation number to
    its images (test_sources.mo_image). The real mo.fetch_all runs on it, with MO's answers
    played and no pause; `asked` notes each batch of observation numbers."""

    def __init__(self, reds, photo_reds=None, images=None, fail_after=None):
        super().__init__(reds, photo_reds)
        self.images, self.asked, self.fail_after, self.calls = images or {}, [], fail_after, []

    def fetch_mo(self, conn, raw_dir=None, limit=None, max_minutes=None, log=print):
        self.calls.append({"limit": limit, "max_minutes": max_minutes, "raw_dir": raw_dir})

        def answer(session, ids, pacer):
            if self.fail_after is not None and len(self.asked) >= self.fail_after:
                raise RuntimeError("MO answered 503: busy")
            self.asked.append(list(ids))
            return [img for n in ids for img in self.images.get(n, [])]
        return mo.fetch_all(conn, raw_dir=raw_dir, limit=limit, session=object(),
                            pacer=mo.Pacer(0), fetch=answer, max_minutes=max_minutes, log=log)

    def steps(self):
        s = super().steps()
        s.fetch_mo = self.fetch_mo
        return s


def run(layer, rows, outside, settings=ON, backbones=None, photo_review=None):
    conn = manifest.connect(layer.manifest)
    try:
        return nightly.run_once(conn, layer, lambda: (rows, "2026-11-24T08:00:00.000Z"),
                                backbones or {"m1": lambda: Const("m1")}, settings,
                                steps=outside.steps(), photo_review=photo_review, **QUIET)
    finally:
        conn.close()


def served(layer, release_root, backbone="m1"):
    """The served index of `backbone`: observation id -> its reference photo ids."""
    conn = sqlite3.connect(layer.manifest)
    try:
        ident = Identifier(conn, backbone, "nearest",
                           release_root / "releases" / RID / "embeddings" / backbone,
                           layer_root=layer.embeddings)
    finally:
        conn.close()
    out: dict = {}
    for col, photo in enumerate(ident.col_photo.tolist()):
        out.setdefault(ident.col_record.obs[col], set()).add(int(photo))
    return out


@pytest.fixture
def box(tmp_path):
    """The server box of test_nightly: one release, the two-species set, backbone m1."""
    root = tmp_path / "srv"
    rel = root / "releases" / RID
    rel.mkdir(parents=True)
    conn = manifest.connect(rel / "manifest.sqlite")
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, rel / "embeddings" / "m1", **QUIET)
    conn.commit()
    conn.close()
    (root / "current.txt").write_text(RID + "\n", encoding="utf-8")
    return root


@pytest.fixture
def layer(box):
    return nightly.prepare(box, **QUIET)


# --- tonight's MO records ------------------------------------------------------------------------

def test_a_green_mo_record_with_no_mo_details_is_fetched_at_night_and_joins_the_reference(
        box, layer):
    outside = MOOutside({}, {pid(610775): 130}, {500: [mo_image(610775, [500])]})
    report = run(layer, seed_rows() + [mo_row(500)], outside)
    assert report["ok"] and report["changed"]
    assert outside.asked == [[500]]
    m = report["mo"]
    assert (m["requested"], m["added"], m["photos_new"], m["still_pending"]) == (1, 1, 1, 0)
    assert outside.downloads == [pid(610775)] and report["embedded"] == {"m1": 1}
    assert served(layer, box)["mo:500"] == {pid(610775)}
    # MO's answer is kept with the layer's raw answers, as iNat's are.
    [kept] = list((layer.raw / "mo").glob("images-500-*.json.gz"))
    assert json.loads(gzip.open(kept, "rt").read())[0]["id"] == 610775
    # Asked once: the next night has nothing to ask MO.
    again = run(layer, seed_rows() + [mo_row(500)], outside)
    assert outside.asked == [[500]] and again["mo"]["requested"] == 0
    assert outside.downloads == [pid(610775)]


def test_the_backfill_takes_mo_records_first_seen_before_the_layer_began(box, layer):
    # On the box every MO record that joined since 20 Oct lacked details; any older one with
    # none is taken too: MO's answer is new tonight, and so are its photos.
    conn = manifest.connect(layer.manifest)
    save_records(conn, build_records(seed_rows() + [mo_row(400)], BEFORE_LAYER))
    conn.close()
    outside = MOOutside({}, {pid(7001): 130}, {400: [mo_image(7001, [400])]})
    report = run(layer, seed_rows() + [mo_row(400)], outside)
    assert report["records"]["new"] == 0                      # not new to the green list
    assert outside.asked == [[400]] and report["mo"]["added"] == 1
    assert outside.downloads == [pid(7001)] and "mo:400" in served(layer, box)


def test_only_north_america_mo_records_are_asked_of_mo(box, layer):
    away = {**mo_row(501), "continent": "Europe"}
    outside = MOOutside({}, {pid(1): 130}, {500: [mo_image(1, [500])]})
    report = run(layer, seed_rows() + [mo_row(500), away], outside)
    assert outside.asked == [[500]] and report["mo"]["still_pending"] == 0


def test_the_nightly_mo_cap_is_respected_and_the_rest_follow_on_later_nights(box, layer):
    images = {n: [mo_image(n * 10, [n])] for n in (500, 501, 502)}
    outside = MOOutside({}, {pid(n * 10): 130 for n in images}, images)
    rows = seed_rows() + [mo_row(n) for n in images]
    first = run(layer, rows, outside, nightly.Settings(mo=True, max_mo=2))
    assert outside.calls[0]["limit"] == 2
    assert outside.asked == [[500, 501]]
    assert (first["mo"]["requested"], first["mo"]["still_pending"]) == (2, 1)
    second = run(layer, rows, outside, nightly.Settings(mo=True, max_mo=2))
    assert outside.asked == [[500, 501], [502]] and second["mo"]["still_pending"] == 0
    none = MOOutside({}, {}, images)
    run(layer, rows + [mo_row(503)], none, nightly.Settings(mo=True, max_mo=0))
    assert none.calls == []                                   # 0: no MO step at all


def test_the_mo_step_is_bounded_in_time_and_the_rest_follow(conn, tmp_path):
    save_records(conn, build_records([mo_row(n) for n in range(1, 82)], "t"))
    ticks = iter([0.0] + [0.0, 0.0] + [31 * 60.0] * 10)       # 30 min pass after two batches
    asked = []

    def answer(session, ids, pacer):
        asked.append(list(ids))
        return []
    out = mo.fetch_all(conn, raw_dir=tmp_path, session=object(), pacer=mo.Pacer(0),
                       fetch=answer, max_minutes=30, clock=lambda: next(ticks), **QUIET)
    assert [len(a) for a in asked] == [40, 40] and out["stopped_for_time"] == 1
    assert len(mo.pending(conn)) == 1
    assert nightly.MO_MAX_MINUTES <= 30


# --- MO's pace and limits ------------------------------------------------------------------------

class FakeClock:
    def __init__(self):
        self.t = 1000.0

    def now(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_the_nightly_mo_step_keeps_mos_pace_and_sends_our_user_agent(box, layer, monkeypatch):
    """The night's MO step is the real mo.fetch_all: one request at a time, at least 5 s apart
    and no sooner than MO's last run_time, 40 observations a request, with config.USER_AGENT.
    MO's HTTP answers are played; nothing leaves the test."""
    clock = FakeClock()
    sent = []

    class Clocked(mo.MinInterval):
        def __init__(self, interval):
            super().__init__(interval, clock=clock.now, sleep=clock.sleep)

    class Answer:
        status_code = 200
        text = ""

        def __init__(self, ids):
            self.ids = ids

        def json(self):
            return {"results": [mo_image(n * 10, [n]) for n in self.ids], "number_of_pages": 1,
                    "run_time": 8.0 if len(sent) == 2 else 0.5}

    class Session:
        def __init__(self):
            self.headers = {}

        def get(self, url, params=None, timeout=None):
            ids = [int(i) for i in params["observation"].split(",")]
            sent.append({"at": clock.now(), "url": url, "ids": ids,
                         "ua": self.headers.get("User-Agent")})
            return Answer(ids)

    monkeypatch.setattr(mo, "MinInterval", Clocked)
    monkeypatch.setattr(mo.requests, "Session", Session)
    numbers = list(range(1, 92))                               # 91 records: 3 requests
    outside = FakeOutside({})
    steps = outside.steps()                                    # fetch_mo: the real mo.fetch_all
    assert steps.fetch_mo is mo.fetch_all
    conn = manifest.connect(layer.manifest)
    try:
        report = nightly.run_once(conn, layer, lambda: (seed_rows() + [mo_row(n) for n in numbers],
                                                        "2026-11-24T08:00:00.000Z"),
                                  {}, ON, steps=steps, **QUIET)
    finally:
        conn.close()
    assert report["ok"] and report["mo"]["added"] == 91
    assert [len(s["ids"]) for s in sent] == [40, 40, 11]
    assert all(s["url"] == mo.API and s["ua"] == config.USER_AGENT for s in sent)
    gaps = [b["at"] - a["at"] for a, b in zip(sent, sent[1:])]
    assert gaps[0] >= mo.MIN_INTERVAL >= 5 and gaps[1] >= 8.0  # MO's slow answer waited out


# --- failures ----------------------------------------------------------------------------------

def test_a_failed_mo_step_is_reported_and_the_nights_other_work_stands(box, layer):
    images = {n: [mo_image(n * 10, [n])] for n in range(500, 545)}  # two batches
    outside = MOOutside({300: 130}, {pid(n * 10): 140 for n in images}, images, fail_after=1)
    report = run(layer, seed_rows() + [row(300, "C z")] + [mo_row(n) for n in images], outside)
    assert report["ok"] and report["changed"]
    assert "MO answered 503" in report["mo"]["error"]
    # The batch saved before the failure stays and is served; the rest follow next night.
    assert report["mo"]["added"] == 40 and report["mo"]["still_pending"] == 5
    assert report["records"]["new"] == 46 and report["embedded"] == {"m1": 41}
    assert "300" in served(layer, box)
    conn = sqlite3.connect(layer.manifest)
    try:
        health = nightly.health(conn, ON)
        status = nightly.status(conn, ON)
    finally:
        conn.close()
    assert health["mo"]["failed"] and health["mo"]["added"] == 40
    assert "503" not in json.dumps(health)                    # public: counts only
    assert "503" in status["runs"][0]["report"]["mo"]["error"]  # `mv nightly` has the reason
    assert status["mo"] == {"on": True, "max_per_night": 2000, "max_photos_per_night": 1500,
                            "waiting": 5}
    [line] = [t for lvl, t in nightly_notes.notes({**health, "state": "ok"}) if "Mushroom" in t]
    assert "it failed" in line


def test_a_crashing_mo_step_never_undoes_the_night(box, layer):
    class Broken(MOOutside):
        def fetch_mo(self, *a, **kw):
            raise ConnectionError("no route to MO")
    report = run(layer, seed_rows() + [row(300, "C z"), mo_row(500)], Broken({300: 130}))
    assert report["ok"] and report["embedded"] == {"m1": 1}
    assert report["mo"]["error"].startswith("ConnectionError") and report["mo"]["added"] == 0


# --- off by default ------------------------------------------------------------------------------

def test_the_mo_step_is_off_unless_switched_on(box, layer, monkeypatch):
    monkeypatch.delenv("MV_NIGHTLY_MO", raising=False)
    monkeypatch.delenv("MV_NIGHTLY_MAX_MO", raising=False)
    assert nightly.Settings().mo is False and nightly.Settings.from_env().mo is False
    outside = MOOutside({}, {}, {500: [mo_image(1, [500])]})
    report = run(layer, seed_rows() + [mo_row(500)], outside, settings=nightly.Settings())
    assert outside.calls == [] and "mo" not in report and report["ok"]
    conn = sqlite3.connect(layer.manifest)
    try:
        assert "mo" not in nightly.health(conn, nightly.Settings())
        assert nightly.status(conn, nightly.Settings())["mo"] == {
            "on": False, "max_per_night": 2000, "max_photos_per_night": 1500}
    finally:
        conn.close()
    monkeypatch.setenv("MV_NIGHTLY_MO", "1")
    monkeypatch.setenv("MV_NIGHTLY_MAX_MO", "1500")
    s = nightly.Settings.from_env()
    assert (s.mo, s.max_mo) == (True, 1500)
    monkeypatch.setenv("MV_NIGHTLY_MAX_MO", "-1")
    with pytest.raises(ValueError, match="MV_NIGHTLY_MAX_MO"):
        nightly.Settings.from_env()


def test_health_and_mv_nightly_show_the_mo_step_in_counts_only(box, layer):
    conn = sqlite3.connect(layer.manifest)
    try:
        assert nightly.health(conn, ON)["mo"] == {"on": True, "max_per_night": 2000,
                                                  "max_photos_per_night": 1500}
    finally:
        conn.close()
    outside = MOOutside({}, {pid(987654): 130}, {876543: [mo_image(987654, [876543])]})
    run(layer, seed_rows() + [mo_row(876543)], outside)
    conn = sqlite3.connect(layer.manifest)
    try:
        block = nightly.health(conn, ON)
    finally:
        conn.close()
    assert block["mo"] == {"on": True, "max_per_night": 2000, "max_photos_per_night": 1500,
                           "requested": 1, "added": 1,
                           "ok": 1, "missing": 0, "gone_on_mo": 0, "photos_new": 1,
                           "batches": 1, "still_pending": 0, "failed": False,
                           "stopped_for_time": False}
    assert "876543" not in json.dumps(block) and "987654" not in json.dumps(block)
    lines = dict((t, lvl) for lvl, t in nightly_notes.notes(block))
    assert ("Mushroom Observer: 1 records gained MO details, 1 new photos, 0 records still "
            "without") in lines


# --- MO's use rule -------------------------------------------------------------------------------

def test_mo_photos_not_for_export_or_all_rights_reserved_are_never_fetched_or_served(box, layer):
    images = {500: [mo_image(11, [500]), mo_image(12, [500], ok_for_export=False),
                    mo_image(13, [500], license=ARR)],
              501: [mo_image(21, [501], license=ARR)]}
    outside = MOOutside({}, {pid(i): 130 for i in (11, 12, 13, 21)}, images)
    report = run(layer, seed_rows() + [mo_row(500), mo_row(501)], outside)
    assert report["mo"]["added"] == 2 and report["mo"]["photos_new"] == 4
    assert outside.downloads == [pid(11)]                    # never asked of MO's image host
    index = served(layer, box)
    assert index["mo:500"] == {pid(11)} and "mo:501" not in index


def test_a_photo_added_later_to_a_record_from_before_the_layer_is_fetched_and_its_old_ones_not(
        box, layer):
    # A sample release: record 200 and its photo 2000 came with it and were never embedded
    # (not tonight's job). A photo the record gained since the layer began is.
    conn = manifest.connect(layer.manifest)
    save_records(conn, build_records(seed_rows() + [row(200, "D w")], BEFORE_LAYER))
    save_batch(conn, ["200"], [inat_obs(200, photos=[(0, 2000, "cc0", OPEN)])], BEFORE_LAYER)
    later = (datetime.now(timezone.utc) + timedelta(minutes=1)).isoformat(timespec="seconds")
    save_batch(conn, ["200"], [inat_obs(200, photos=[(0, 2000, "cc0", OPEN),
                                                     (1, 2001, "cc0", OPEN)])], later)
    conn.close()
    outside = FakeOutside({200: 140})
    report = run(layer, seed_rows() + [row(200, "D w")], outside, settings=nightly.Settings())
    assert outside.downloads == [2001] and report["embedded"] == {"m1": 1}


def test_pending_photos_takes_a_photo_by_its_own_first_seen_date(conn):
    save_records(conn, build_records([row(1, "A x"), row(2, "B y")], "2026-01-01T00:00:00+00:00"))
    save_batch(conn, ["1"], [inat_obs(1, photos=[(0, 11, "cc0", OPEN)])], "2026-01-01T00:00:00+00:00")
    save_batch(conn, ["2"], [inat_obs(2, photos=[(0, 21, "cc0", OPEN)])], "2026-06-01T00:00:00+00:00")
    since = "2026-03-01T00:00:00+00:00"
    ids = lambda **kw: [r["photo_id"] for r in pending_photos(conn, True, None, "large", **kw)]
    assert ids(first_seen_since=since) == []                  # both records are older
    assert ids(first_seen_since=since, new_photos_since=since) == [21]
    assert ids(new_photos_since=since) == [21]


# --- the photo review ----------------------------------------------------------------------------

def test_the_nightly_photo_review_sees_the_new_mo_photos_and_never_a_not_for_export_one(
        box, layer, tmp_path):
    from test_nightly_photo_review import MICRO_HIGH, Fork, make_store, review
    store = make_store(tmp_path, layer)
    images = {500: [mo_image(31, [500]), mo_image(32, [500], ok_for_export=False)]}
    outside = MOOutside({}, {pid(31): MICRO_HIGH, pid(32): MICRO_HIGH}, images)
    report = run(layer, seed_rows() + [mo_row(500, when="10/1/2026")], outside,
                 backbones={"m1": lambda: Fork("m1")}, photo_review=review(store))
    assert report["photo_review"]["tagged"] == 1 and report["photo_review"]["queued"] == 1
    subjects = [r[0] for r in store.conn.execute(
        "select subject from items where item_type = 'non_specimen_photo'")]
    assert subjects == ["mo:31"]


# --- a backbone that already holds the MO photos' vectors (v2) ----------------------------------

@pytest.fixture
def two_model_box(tmp_path):
    """A release with backbones m1 and m2, where m2 (like v2 on the box, from its release
    shards) already holds vectors for MO photos whose MO rows the layer lacks: one usable, one
    all rights reserved."""
    root = tmp_path / "srv2"
    rel = root / "releases" / RID
    rel.mkdir(parents=True)
    conn = manifest.connect(rel / "manifest.sqlite")
    store = seed_two_species(conn, tmp_path)
    for name in ("m1", "m2"):
        todo = photos_to_embed(conn, name, "large", store.location, "local")
        embed_photos(conn, store, Const(name), todo, rel / "embeddings" / name, **QUIET)
    # m2's vectors of MO image 41 (usable) and 42 (all rights reserved), made where the photos
    # were held; the layer's manifest has no MO answer, photo or link for them.
    from test_nightly import png
    held = LocalStore(tmp_path / "mo-held")
    for image_id in (41, 42):
        held.put(f"mo/{image_id}.png", png(130))
    embed_photos(conn, held, Const("m2"), [(pid(41), "mo/41.png"), (pid(42), "mo/42.png")],
                 rel / "embeddings" / "m2", **QUIET)
    conn.commit()
    conn.close()
    (root / "current.txt").write_text(RID + "\n", encoding="utf-8")
    return root


def test_a_backbone_with_the_mo_photos_vectors_serves_the_record_without_embedding_again(
        two_model_box):
    layer = nightly.prepare(two_model_box, **QUIET)
    before = served(layer, two_model_box, "m2")
    assert not any(k.startswith("mo:") for k in before)
    images = {500: [mo_image(41, [500]), mo_image(42, [500], license=ARR)]}
    outside = MOOutside({}, {pid(41): 130, pid(42): 130}, images)
    report = run(layer, seed_rows() + [mo_row(500)], outside,
                 backbones={"m1": lambda: Const("m1"), "m2": lambda: Const("m2")})
    assert report["ok"] and report["changed"]
    assert report["embedded"] == {"m1": 1, "m2": 0}            # m2 embeds nothing
    assert outside.downloads == [pid(41)]                      # fetched once, for m1 only
    assert served(layer, two_model_box, "m2")["mo:500"] == {pid(41)}   # never the ARR one
    assert served(layer, two_model_box, "m1")["mo:500"] == {pid(41)}
    assert report["photos_deleted"] == 1                       # both backbones hold it now


def test_mo_details_alone_change_the_reference_set_so_the_index_is_rebuilt(two_model_box):
    # Only m2 served (its vectors came with the release): nothing is embedded tonight, yet the
    # record is new to its index, so the night counts as a change and the server rebuilds.
    layer = nightly.prepare(two_model_box, **QUIET)
    outside = MOOutside({}, {pid(41): 130}, {500: [mo_image(41, [500])]})
    rows = seed_rows() + [mo_row(500)]
    conn = sqlite3.connect(layer.manifest)
    save_records(conn, build_records(rows, BEFORE_LAYER))       # not new to the green list
    conn.commit()
    before = nightly.layer_version(conn)
    conn.close()
    report = run(layer, rows, outside, backbones={"m2": lambda: Const("m2")})
    assert report["records"]["new"] == 0 and report["embedded"] == {"m2": 0}
    assert outside.downloads == [] and report["changed"]
    conn = sqlite3.connect(layer.manifest)
    try:
        assert nightly.layer_version(conn) > before
    finally:
        conn.close()
    assert served(layer, two_model_box, "m2")["mo:500"] == {pid(41)}


def test_mo_photos_a_night_are_capped_for_every_backbone_together_and_the_rest_follow(
        two_model_box):
    # MO's image host takes one request every 5 s: a 5,000-photo backlog is 7 hours, so the
    # night takes at most MV_NIGHTLY_MAX_MO_PHOTOS of them, whichever backbone needs them;
    # iNat photos are not held back.
    layer = nightly.prepare(two_model_box, **QUIET)
    images = {n: [mo_image(n * 10, [n])] for n in (500, 501, 502)}
    outside = MOOutside({300: 130}, {pid(n * 10): 130 for n in images}, images)
    rows = seed_rows() + [row(300, "C z")] + [mo_row(n) for n in images]
    capped = nightly.Settings(mo=True, max_mo_photos=2)
    both = {"m1": lambda: Const("m1"), "m2": lambda: Const("m2")}
    first = run(layer, rows, outside, capped, backbones=both)
    assert sorted(outside.downloads) == [3000, pid(5000), pid(5010)]
    assert first["embedded"] == {"m1": 3, "m2": 3}
    run(layer, rows, outside, capped, backbones=both)
    assert sorted(outside.downloads) == [3000, pid(5000), pid(5010), pid(5020)]
    none = MOOutside({}, {}, {503: [mo_image(5030, [503])]})
    run(layer, rows + [mo_row(503)], none, nightly.Settings(mo=True, max_mo_photos=0),
        backbones=both)
    assert none.downloads == []                               # 0: no MO photos at all
