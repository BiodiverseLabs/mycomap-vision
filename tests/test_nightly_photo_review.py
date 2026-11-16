"""Every nightly photo goes to the photo review (Steve, 2026-11-07): "whenever there is a nightly
update, all of our putative exclusions are added. They are not excluded until human validation.
But it should be a dynamic process for each record that is pushed in. Tagger runs for all. Human
review still required to save them".

With MV_NIGHTLY_PHOTO_REVIEW=1 each photo a night embeds is tagged by the served photo-type
tagger from the same pass, and each non-specimen one in the chosen bands is queued on the
"Photos that show no specimen" review through the box's seal gate. Nothing is excluded until a
person decides, nothing is queued twice, a decided photo never reopens, and a failure here never
fails the night."""

import json
import sqlite3
import threading

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_models_and_scoreboard import Const
from test_nightly import QUIET, RID, FakeOutside, box, png, row, seed_rows  # noqa: F401

from mycomap_vision import feedback as fb
from mycomap_vision import manifest, nightly, nightly_notes, nightly_review, phototypes
from mycomap_vision import photo_review as pr
from mycomap_vision import prospective
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos

SEAL = prospective.SEAL_DEFAULT                          # 2026-10-12T13:17:51+00:00
BASE = "bioclip-2-336"
CLASSES = ("specimen", "micrograph", "slip")
BANDS = {"micrograph": [[0.7, "high"], [0.5, "medium"], [0.35, "low"]], "slip": [[0.35, "high"]]}
# A photo's red level -> its frozen base vector -> its tag (W = 3 I).
SPECIMEN, MICRO_HIGH, MICRO_MEDIUM, MICRO_LOW, SLIP_HIGH = 200, 130, 140, 150, 160
BASES = {SPECIMEN: [1, 0, 0], MICRO_HIGH: [0, 1, 0], MICRO_MEDIUM: [0, 0.8, 0.6],
         MICRO_LOW: [0.55, 0.7, 0.45], SLIP_HIGH: [0, 0, 1]}


def tagger():
    return phototypes.PhotoTagger(3 * np.eye(3), np.zeros(3), CLASSES, BANDS, 0.35, BASE,
                                  "v2.1-336")


def red_of(im) -> int:
    return int(round(float(np.asarray(im, dtype=np.float32)[..., 0].mean())))


class Frozen(Const):
    """The frozen base model itself (the catch-up's): its vectors are the base vectors."""

    def __init__(self, name=BASE):
        super().__init__(name)
        self.dim = 3

    def encode(self, images):
        return np.array([BASES[red_of(im)] for im in images], float)


class Fork(Const):
    """The served fine-tune: its own vectors, and its frozen base's from the same pass."""
    base_name = BASE

    def __init__(self, name="m1"):
        super().__init__(name)
        self.forks = 0

    def encode_with_base(self, images):
        self.forks += 1
        return self.encode(images), np.array([BASES[red_of(im)] for im in images], float)


def gate(layer, seal=SEAL):
    """The box's seal gate on the layer's manifest (its own connection, as the server's)."""
    if seal is None:
        return fb.SealGate(lambda k: None, frozenset(), None)
    return fb.SealGate.from_manifest(sqlite3.connect(layer.manifest, check_same_thread=False),
                                     seal=seal)


def make_store(tmp_path, layer, seal=SEAL):
    store = fb.Store(tmp_path / "human-feedback.sqlite")
    store.gate = gate(layer, seal)
    return store


def review(store, bands=nightly_review.DEFAULT_BANDS, t=None):
    return nightly_review.NightlyReview(t or tagger(), store, tuple(bands))


def night(layer, rows, outside, photo_review=None, model=Fork):
    conn = manifest.connect(layer.manifest)
    try:
        return nightly.run_once(conn, layer, lambda: (rows, "2026-10-08T07:00:00.000Z"),
                                {"m1": lambda: model("m1")}, steps=outside.steps(),
                                photo_review=photo_review, **QUIET)
    finally:
        conn.close()


def items(store):
    return {r["subject"]: dict(r) for r in store.conn.execute(
        "select subject, status, rule, links, projects, evidence, seed, granularity, "
        "model_flagged, record_keys from items where item_type = 'non_specimen_photo'")}


NEW = {300: MICRO_HIGH, 301: SPECIMEN, 302: SLIP_HIGH, 303: MICRO_MEDIUM, 304: MICRO_LOW}


def new_rows(when="10/1/2026", reds=NEW):
    return seed_rows() + [row(o, "C z", when) for o in reds]


@pytest.fixture
def layer(box):  # noqa: F811
    return nightly.prepare(box, **QUIET)


# --- a night ------------------------------------------------------------------------------------

def test_a_night_with_the_tagger_on_queues_its_non_specimen_photos_and_not_its_specimen_ones(
        layer, tmp_path):
    store = make_store(tmp_path, layer)
    report = night(layer, new_rows(), FakeOutside(NEW), review(store))
    assert report["ok"] and report["embedded"] == {"m1": 5}
    got = items(store)
    assert set(got) == {"inat:3000", "inat:3020", "inat:3030", "inat:3040"}   # not 3010
    one = got["inat:3000"]
    assert one["status"] == "open" and one["granularity"] == "photo"
    assert one["rule"] == "nightly v2.1-336 micrograph p=0.909 (high)"
    assert json.loads(one["links"]) == ["https://www.inaturalist.org/observations/300"]
    assert json.loads(one["record_keys"]) == [] and one["model_flagged"] == 0
    assert one["seed"] == f"nightly-photo-review-{report['id']}"
    ev = json.loads(one["evidence"])
    assert "v2.1-336" in ev["evidence"] and "until a person decides" in ev["evidence"]
    assert (ev["type"], ev["band"]) == ("micrograph", "high")
    r = report["photo_review"]
    assert (r["tagged"], r["queued"], r["specimen"], r["already_queued"], r["already_decided"],
            r["refused_sealed"], r["refused_unknown"]) == (5, 4, 1, 0, 0, 0, 0)
    assert r["tagger"] == "v2.1-336"


def test_the_queued_photo_carries_its_records_green_projects(layer, tmp_path):
    store = make_store(tmp_path, layer)
    rows = seed_rows() + [dict(row(300, "C z"), validation_project_1="Run151"),
                          dict(row(300, "C z"), validation_project_1="CA FunDiS")]
    night(layer, rows, FakeOutside({300: MICRO_HIGH}), review(store))
    assert json.loads(items(store)["inat:3000"]["projects"]) == ["CA FunDiS", "Run151"]


def test_the_photo_review_places_a_nightly_flag_in_its_tab_and_band():
    flags = pr.parse_flags("nightly v2.1-336 micrograph p=0.909 (high)")
    assert flags == [{"tab": "micrograph", "kind": "v2 micrograph", "p": 0.909,
                      "strength": 0.909, "v2": True, "band": "high"}]
    assert pr.parse_flags("nightly v2.1-336 not_fungus p=0.400 (low)")[0]["tab"] == "gate"
    # The rules already on the queue read as before.
    assert pr.parse_flags("R11.1 v2 uv p=0.043 (very_low)")[0]["band"] == "very_low"
    assert pr.parse_flags("R9.1 tagger v2.1")[0]["tab"] == "other"
    assert pr.suggested_decision(pr.placing_flags(
        "R6 micrograph probe p=0.51; nightly v2.1-336 slip p=0.910 (high)")) == "exclude"


def test_only_the_bands_asked_for_are_queued(layer, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW", "1")
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW_BANDS", "medium")
    store = make_store(tmp_path, layer)
    rev = nightly_review.NightlyReview.from_env(tagger(), store)
    assert rev.bands == ("medium",)
    report = night(layer, new_rows(), FakeOutside(NEW), rev)
    assert set(items(store)) == {"inat:3030"}
    assert report["photo_review"]["out_of_band"] == 3 and report["photo_review"]["queued"] == 1


def test_the_bands_default_to_high_medium_and_low_and_a_bad_one_is_reported(monkeypatch):
    monkeypatch.delenv("MV_NIGHTLY_PHOTO_REVIEW_BANDS", raising=False)
    assert nightly_review.bands_from_env() == ("high", "medium", "low")
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW_BANDS", " High , low ")
    assert nightly_review.bands_from_env() == ("high", "low")
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW_BANDS", "high,sky")
    with pytest.raises(ValueError):
        nightly_review.bands_from_env()
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW", "1")
    rev = nightly_review.NightlyReview.from_env(tagger(), None)
    assert "MV_NIGHTLY_PHOTO_REVIEW_BANDS" in rev.problem


def test_the_review_is_off_unless_switched_on(monkeypatch):
    monkeypatch.delenv("MV_NIGHTLY_PHOTO_REVIEW", raising=False)
    assert nightly_review.NightlyReview.from_env(tagger(), object()) is None
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW", "0")
    assert nightly_review.NightlyReview.from_env(tagger(), object()) is None


def test_switched_on_without_a_tagger_the_night_says_so_and_tags_nothing(layer, tmp_path,
                                                                        monkeypatch):
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW", "1")
    store = make_store(tmp_path, layer)
    model = Fork("m1")
    report = night(layer, new_rows(), FakeOutside(NEW),
                   nightly_review.NightlyReview.from_env(None, store), model=lambda n: model)
    assert report["ok"] and report["embedded"] == {"m1": 5}
    assert "no photo-type tagger" in report["photo_review"]["error"]
    assert items(store) == {} and model.forks == 0         # no extra pass either


def test_without_the_review_a_night_makes_no_extra_pass_and_queues_nothing(layer, tmp_path):
    store = make_store(tmp_path, layer)
    model = Fork("m1")
    report = night(layer, new_rows(), FakeOutside(NEW), None, model=lambda n: model)
    assert report["ok"] and "photo_review" not in report
    assert model.forks == 0 and items(store) == {}


def test_a_second_night_queues_no_duplicates(layer, tmp_path):
    store = make_store(tmp_path, layer)
    night(layer, new_rows(), FakeOutside(NEW), review(store))
    before = items(store)
    report = night(layer, new_rows(), FakeOutside(NEW), review(store))
    assert items(store) == before
    assert report["photo_review"]["queued"] == 0 and report["photo_review"]["candidates"] == 0
    # A tag queued, should it be tried again, finds its open item and changes nothing.
    conn = manifest.connect(layer.manifest)
    conn.execute("update photo_type_tags set queue_state = null")
    conn.commit()
    out = nightly_review.queue_tags(conn, store, nightly_review.DEFAULT_BANDS, "again", "again",
                                    **QUIET)
    conn.close()
    assert (out["queued"], out["already_queued"], out["flags_added"]) == (0, 4, 0)
    assert items(store) == before


def test_a_decided_photo_is_never_reopened_and_an_open_one_gains_the_nightly_flag(layer,
                                                                                  tmp_path):
    store = make_store(tmp_path, layer)
    seal, store.gate = store.gate, None         # queued by an older seed, before these records
    store.seed([{"item_type": "non_specimen_photo", "subject": "inat:3000", "rule":
                 "R9 v2 micrograph p=0.800 (high)", "granularity": "photo",
                 "links": ["https://www.inaturalist.org/observations/300"]},
                {"item_type": "non_specimen_photo", "subject": "inat:3020", "rule":
                 "R6 view slip p=0.9", "granularity": "photo",
                 "links": ["https://www.inaturalist.org/observations/302"]}], "older")
    store.gate = seal
    first = {s: i for s, i in items(store).items()}
    item = store.conn.execute("select id from items where subject = 'inat:3000'").fetchone()[0]
    store.decide(item, "keep_photo", {"sub": "1", "name": "Steve"})
    report = night(layer, new_rows(), FakeOutside(NEW), review(store))
    got = items(store)
    assert got["inat:3000"]["status"] == "decided"
    assert got["inat:3000"]["rule"] == first["inat:3000"]["rule"]          # unchanged
    assert got["inat:3020"]["status"] == "open"
    assert got["inat:3020"]["rule"] == "R6 view slip p=0.9; nightly v2.1-336 slip p=0.909 (high)"
    r = report["photo_review"]
    assert (r["queued"], r["already_queued"], r["flags_added"], r["already_decided"]) == \
           (2, 1, 1, 1)
    assert store.excluded_photos() == set()                 # kept is no exclusion either


def test_a_post_seal_record_is_refused_and_queued_once_the_seal_is_off(layer, tmp_path):
    store = make_store(tmp_path, layer)
    reds = {300: MICRO_HIGH, 301: SLIP_HIGH}
    rows = seed_rows() + [row(300, "C z", "10/20/2026"), row(301, "C z", "10/12/2026")]
    report = night(layer, rows, FakeOutside(reds), review(store))
    r = report["photo_review"]
    assert (r["tagged"], r["queued"], r["refused_sealed"], r["refused_unknown"]) == (2, 0, 1, 1)
    assert items(store) == {}
    # The seal is switched off after the paper's sealed run: the next night queues the kept
    # tags without reading a photo again.
    model = Fork("m1")
    report = night(layer, rows, FakeOutside(reds), review(make_store(tmp_path, layer, None)),
                   model=lambda n: model)
    assert model.forks == 0 and report["photo_review"]["tagged"] == 0
    assert report["photo_review"]["queued"] == 2
    assert set(items(store)) == {"inat:3000", "inat:3010"}


def test_nothing_is_excluded_until_a_person_decides(layer, tmp_path):
    store = make_store(tmp_path, layer)
    night(layer, new_rows(), FakeOutside(NEW), review(store))
    assert len(items(store)) == 4 and store.excluded_photos() == set()
    conn = manifest.connect(layer.manifest)
    assert fb.sync_exclusions(store, conn)["photos"] == 0
    item = store.conn.execute("select id from items where subject = 'inat:3000'").fetchone()[0]
    store.decide(item, "exclude_photo", {"sub": "1", "name": "Steve"}, value="micrograph")
    assert store.excluded_photos() == {"inat:3000"}
    assert fb.sync_exclusions(store, conn)["photos"] == 1
    conn.close()


# --- failures never fail the night ----------------------------------------------------------------

class BrokenFork(Fork):
    def encode_with_base(self, images):
        raise RuntimeError("fork broke")


def test_a_failing_fork_leaves_the_night_ok_and_every_vector_made(layer, tmp_path):
    store = make_store(tmp_path, layer)
    report = night(layer, new_rows(), FakeOutside(NEW), review(store), model=BrokenFork)
    assert report["ok"] and report["error"] is None and report["embedded"] == {"m1": 5}
    assert report["base_vector_errors"] == {"m1": 1}
    assert report["photo_review"]["tagged"] == 0 and items(store) == {}


def test_a_failing_tagger_or_queue_leaves_the_night_ok(layer, tmp_path, monkeypatch):
    store = make_store(tmp_path, layer)
    t = tagger()
    monkeypatch.setattr(t, "tag", lambda v: (_ for _ in ()).throw(ValueError("tagger broke")))
    report = night(layer, new_rows(), FakeOutside(NEW), review(store, t=t))
    assert report["ok"] and report["embedded"] == {"m1": 5}
    assert report["photo_review"]["tag_errors"] == 1
    assert "tagger broke" in report["photo_review"]["error"]

    def broken_seed(*a, **k):
        raise sqlite3.OperationalError("database is locked")
    store2 = make_store(tmp_path / "b", layer)
    monkeypatch.setattr(store2, "seed", broken_seed)
    report = night(layer, new_rows() + [row(305, "C z")], FakeOutside({**NEW, 305: MICRO_HIGH}),
                   review(store2))
    assert report["ok"] and report["embedded"] == {"m1": 1}
    assert "database is locked" in report["photo_review"]["error"]


def test_embedding_with_base_vectors_gives_the_same_vectors(tmp_path, conn):
    from mycomap_vision.photos import Result, save_result
    from mycomap_vision.records import build_records, save_records
    from mycomap_vision.storage import LocalStore
    save_records(conn, build_records([row(1, "A x")], "t"))
    store = LocalStore(tmp_path / "s")
    todo = []
    for i, red in enumerate((MICRO_HIGH, SPECIMEN, SLIP_HIGH)):
        store.put(f"p/{i}.png", png(red))
        todo.append((i, f"p/{i}.png"))
    seen = []
    a = embed_photos(conn, store, Fork("a"), todo, tmp_path / "a", **QUIET)
    b = embed_photos(conn, store, Fork("b"), todo, tmp_path / "b", **QUIET,
                     with_base=lambda ids, v: seen.append((list(ids), np.asarray(v).tolist())))
    assert a.embedded == b.embedded == 3
    assert np.array_equal(np.load(tmp_path / "a" / "shard-00000.npy"),
                          np.load(tmp_path / "b" / "shard-00000.npy"))
    assert seen == [([0, 1, 2], [BASES[MICRO_HIGH], BASES[SPECIMEN], BASES[SLIP_HIGH]])]


def test_the_fork_runs_under_the_nightly_model_lock():
    lock = threading.Lock()
    locked = nightly.Locked(Fork("m1"), lock)
    done = threading.Event()
    lock.acquire()
    t = threading.Thread(target=lambda: (locked.encode_with_base([png_image(MICRO_HIGH)]),
                                         done.set()), daemon=True)
    t.start()
    assert not done.wait(0.2)                                # waits for identifications
    lock.release()
    assert done.wait(5)
    assert nightly.Locked(Const("m1"), lock).encode_with_base([png_image(1)]) is None


def png_image(red):
    from PIL import Image
    return Image.new("RGB", (8, 8), (red, 0, 0))


def test_a_frozen_served_backbone_is_its_own_base_and_another_gives_none():
    same, why = nightly_review.with_base_model(Const(BASE), BASE)
    v, base = same.encode_with_base([png_image(MICRO_HIGH)])
    assert why is None and np.array_equal(v, base)
    none, why = nightly_review.with_base_model(Const("other"), BASE)
    assert none is None and BASE in why


# --- health and `mv nightly` ----------------------------------------------------------------------

def test_health_and_mv_nightly_show_the_reviews_counts(layer, tmp_path):
    store = make_store(tmp_path, layer)
    night(layer, new_rows(), FakeOutside(NEW), review(store))
    conn = manifest.connect(layer.manifest)
    try:
        h = nightly.health(conn, nightly.Settings())
        last = nightly.status(conn, nightly.Settings())["runs"][0]["report"]
    finally:
        conn.close()
    assert h["photo_review"] == {"tagged": 5, "queued": 4, "already_queued": 0,
                                 "already_decided": 0, "refused_sealed": 0, "refused_unknown": 0,
                                 "failed": False}
    assert last["photo_review"]["queued"] == 4
    lines = dict((t.split(":")[0], lvl) for lvl, t in nightly_notes.notes(
        {"enabled": True, "state": "ok", "last_ok_at": h["last_ok_at"],
         "last_run_at": h["last_run_at"], "photo_review": h["photo_review"]},
        now=nightly.now_utc()))
    assert lines["photo review"] == "ok"
    text = [t for _, t in nightly_notes.notes({"enabled": True, "state": "ok",
                                               "photo_review": h["photo_review"]})]
    assert any("5 photos tagged, 4 queued" in t for t in text)


# --- the server, with the setting -----------------------------------------------------------------

def site(box, layer, rows, outside, store):  # noqa: F811
    return TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Fork(name), limits=open_limits(), background=False,
        layer=layer, nightly_fetch=lambda: (rows, "2026-10-08T07:00:00.000Z"),
        nightly_steps=outside.steps(), photo_tagger=tagger(), fungus_gate=None,
        feedback_store=store, note=lambda s: None))


def test_the_server_queues_tonights_photos_when_switched_on(box, layer, tmp_path,  # noqa: F811
                                                            monkeypatch):
    monkeypatch.setenv("MV_NIGHTLY_PHOTO_REVIEW", "1")
    store = fb.Store(tmp_path / "hf.sqlite")
    client = site(box, layer, new_rows(), FakeOutside(NEW), store)
    report = client.app.state.run_nightly()
    assert report["ok"] and report["photo_review"]["queued"] == 4
    assert store.gate is not None                            # the box's seal gate
    assert set(items(store)) == {"inat:3000", "inat:3020", "inat:3030", "inat:3040"}


def test_the_server_queues_nothing_when_not_switched_on(box, layer, tmp_path,  # noqa: F811
                                                        monkeypatch):
    monkeypatch.delenv("MV_NIGHTLY_PHOTO_REVIEW", raising=False)
    store = fb.Store(tmp_path / "hf.sqlite")
    client = site(box, layer, new_rows(), FakeOutside(NEW), store)
    report = client.app.state.run_nightly()
    assert report["ok"] and "photo_review" not in report and items(store) == {}


# --- the catch-up -----------------------------------------------------------------------------------

def arrived(layer, rows, reds):
    """Records and photos the nights added before the review was on (tagged by none)."""
    night(layer, rows, FakeOutside(reds), None)


def fake_fetch(reds):
    fetched = []

    def fetch(rows, work_dir):
        out = {}
        for r in rows:
            f = work_dir / f"{r['photo_id']}.png"
            f.write_bytes(png(reds[r["photo_id"] // 10]))
            out[r["photo_id"]] = f
            fetched.append(r["photo_id"])
        return out
    return fetch, fetched


def snapshot(store, layer):
    c = sqlite3.connect(layer.manifest)
    try:
        tags = c.execute("select count(*) from sqlite_master where name = 'photo_type_tags'"
                         ).fetchone()[0] and c.execute("select count(*) from photo_type_tags"
                                                       ).fetchone()[0]
    finally:
        c.close()
    return (store.conn.execute("select count(*) from items").fetchone()[0],
            store.conn.execute("select count(*) from seeds").fetchone()[0], tags)


def test_the_catch_up_dry_run_changes_nothing_and_says_what_the_seal_refuses(layer, tmp_path):
    reds = {300: MICRO_HIGH, 301: SPECIMEN, 302: SLIP_HIGH}
    rows = seed_rows() + [row(300, "C z", "10/14/2026"), row(301, "C z", "10/20/2026"),
                          row(302, "C z", "10/12/2026")]
    arrived(layer, rows, reds)
    store = make_store(tmp_path, layer)
    before = snapshot(store, layer)
    fetch, fetched = fake_fetch(reds)
    conn = manifest.connect(layer.manifest)
    try:
        plan = nightly_review.catch_up(conn, store, tagger(), "2026-10-12", dry_run=True,
                                       load_model=lambda: pytest.fail("no model in a dry run"),
                                       fetch=fetch, **QUIET)
        # With the seal on, a real run fetches and reads nothing, and queues nothing.
        real = nightly_review.catch_up(conn, store, tagger(), "2026-10-12", dry_run=False,
                                       load_model=lambda: pytest.fail("nothing to read"),
                                       fetch=fetch, work_dir=tmp_path / "w", **QUIET)
    finally:
        conn.close()
    assert plan["dry_run"] and (plan["records"], plan["photos"]) == (3, 3)
    assert (plan["to_tag"], plan["refused_sealed"], plan["refused_unknown"]) == (0, 2, 1)
    assert plan["cpu_hours"] == 0 and plan["seal"] == SEAL
    assert snapshot(store, layer)[:2] == before[:2] and fetched == []
    assert real["queued"] == 0 and real["tagged"] == 0


def test_the_catch_up_tags_and_queues_once_the_seal_is_off_and_resumes(layer, tmp_path):
    reds = {300: MICRO_HIGH, 301: SPECIMEN, 302: SLIP_HIGH, 303: MICRO_MEDIUM}
    rows = seed_rows() + [row(o, "C z", "10/20/2026") for o in reds]
    arrived(layer, rows, reds)
    store = make_store(tmp_path, layer, None)                     # MV_PROSPECTIVE_SEAL=off
    store.seed([{"item_type": "non_specimen_photo", "subject": "inat:3030", "rule": "R9 v2 x",
                 "links": ["https://www.inaturalist.org/observations/303"]}], "older")
    fetch, fetched = fake_fetch(reds)
    conn = manifest.connect(layer.manifest)
    try:
        plan = nightly_review.catch_up(conn, store, tagger(), "2026-10-13", dry_run=True,
                                       fetch=fetch, **QUIET)
        assert (plan["to_tag"], plan["already_queued"], plan["refused_sealed"]) == (3, 1, 0)
        assert nightly_review.catch_up(conn, store, tagger(), "2026-10-13", dry_run=True,
                                       limit=2, **QUIET)["this_run"] == 2
        assert plan["cpu_seconds"] > 0 and plan["seconds_per_photo"] > 0
        out = nightly_review.catch_up(conn, store, tagger(), "2026-10-13", dry_run=False,
                                      limit=2, load_model=Frozen, fetch=fetch,
                                      work_dir=tmp_path / "w", **QUIET)
        assert out["tagged"] == 2 and sorted(fetched) == [3000, 3010]
        assert not list((tmp_path / "w").iterdir())               # every file deleted
        again = nightly_review.catch_up(conn, store, tagger(), "2026-10-13", dry_run=False,
                                        load_model=Frozen, fetch=fetch,
                                        work_dir=tmp_path / "w", **QUIET)
    finally:
        conn.close()
    assert again["tagged"] == 1 and sorted(fetched) == [3000, 3010, 3020]   # resumed
    got = items(store)
    assert set(got) == {"inat:3000", "inat:3020", "inat:3030"}
    assert got["inat:3030"]["rule"] == "R9 v2 x"               # queued before: left as it was
    assert got["inat:3000"]["seed"].startswith("nightly-photo-review-catch-up-")


def test_the_catch_up_waits_for_a_nightly_run(layer, tmp_path):
    arrived(layer, seed_rows() + [row(300, "C z", "10/20/2026")], {300: MICRO_HIGH})
    store = make_store(tmp_path, layer, None)
    conn = manifest.connect(layer.manifest)
    try:
        with pytest.raises(RuntimeError, match="nightly run"):
            nightly_review.catch_up(conn, store, tagger(), "2026-10-13", dry_run=False,
                                    busy=lambda: True, **QUIET)
    finally:
        conn.close()


def test_the_catch_up_dry_run_counts_what_it_would_queue_from_kept_tags(layer, tmp_path):
    reds = {300: MICRO_HIGH, 301: SPECIMEN, 302: SLIP_HIGH}
    rows = seed_rows() + [row(o, "C z", "10/20/2026") for o in reds]
    sealed = make_store(tmp_path, layer)
    night(layer, rows, FakeOutside(reds), review(sealed))        # tagged, refused by the seal
    store = make_store(tmp_path, layer, None)                     # the seal switched off
    before = snapshot(store, layer)
    conn = manifest.connect(layer.manifest)
    try:
        states = conn.execute("select queue_state from photo_type_tags order by photo_id"
                              ).fetchall()
        plan = nightly_review.catch_up(conn, store, tagger(), "2026-10-13", dry_run=True,
                                       limit=1, **QUIET)
        assert conn.execute("select queue_state from photo_type_tags order by photo_id"
                            ).fetchall() == states
    finally:
        conn.close()
    assert snapshot(store, layer) == before and items(store) == {}
    assert (plan["to_tag"], plan["this_run"], plan["tagged_not_queued"]) == (0, 0, 2)
    assert plan["kept_tags_would_queue"] == 2 and plan["kept_tags_refused"] == 0


def test_mv_feedback_seed_nightly_tags_dry_run_goes_through_the_seal(layer, tmp_path,
                                                                    monkeypatch):
    from types import SimpleNamespace

    from mycomap_vision import cli
    arrived(layer, seed_rows() + [row(300, "C z", "10/20/2026")], {300: MICRO_HIGH})
    monkeypatch.setattr(phototypes.PhotoTagger, "from_setting", classmethod(lambda c: tagger()))
    store = fb.Store(tmp_path / "hf.sqlite")
    args = SimpleNamespace(since="2026-10-13", dry_run=True, limit=3000, bands=None, threads=2)
    conn = manifest.connect(layer.manifest)
    try:
        plan = cli._seed_nightly_tags(conn, store, args)
        assert (plan["refused_sealed"], plan["to_tag"], plan["dry_run"]) == (1, 0, True)
        monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
        plan = cli._seed_nightly_tags(conn, store, args)
        assert (plan["refused_sealed"], plan["to_tag"], plan["seal"]) == (0, 1, None)
        with pytest.raises(SystemExit):
            cli._seed_nightly_tags(conn, store, SimpleNamespace(**{**vars(args),
                                                                 "since": "13 Oct"}))
        with pytest.raises(SystemExit):
            cli._seed_nightly_tags(conn, store, SimpleNamespace(**{**vars(args),
                                                                 "bands": "high,sky"}))
    finally:
        conn.close()
    assert items(store) == {}
