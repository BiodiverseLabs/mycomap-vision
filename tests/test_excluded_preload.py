"""The Excluded photos preload (excluded_looked.Preloader, api.py; Steve, 2026-10-27, the approved
plan): every slip photo's and kept crop's "where Vision looked" map made ahead of anyone asking.
What it makes and skips (the seal, photos not shown), one batch photo at a time behind every
identification and every person's ask (Identify waits at most one map), its window, its off
switch, crops only once embedded and slips after them, the stamp that makes a kept map stale,
and its progress in /api/health."""

import io
import threading
import time
from datetime import datetime, timezone

import pytest
from PIL import Image
from test_excluded_looked_api import LOOKED, Maker, site
from test_run_links import person, visitor

from mycomap_vision import excluded_looked as xl
from mycomap_vision import looked
from mycomap_vision.embed import crops_to_embed, embed_photos
from mycomap_vision.feedback import ensure_crops_schema
from test_models_and_scoreboard import Const

AT = lambda h, m=0: datetime(2026, 10, 27, h, m, tzinfo=timezone.utc)   # noqa: E731


# --- the worker: people first, Identify waits at most one map ----------------------------------

def job(rid, make, batch=False):
    return looked.Job(rid, [looked.Task(0, lambda: None, make)], lambda e: None, batch=batch)


def test_a_persons_ask_goes_ahead_of_every_queued_preload_photo():
    gate, order = threading.Event(), []
    w = looked.Worker(gate.is_set, threading.Lock(), poll=0.005, note=lambda s: None)

    def make(name):
        return lambda got: order.append(name) or {"m": 1}
    w.submit(job("b1", make("b1"), batch=True))       # taken first, waits for idle
    time.sleep(0.05)
    w.submit(job("b2", make("b2"), batch=True))
    w.submit(job("p1", make("p1")))                   # a person's ask
    w.submit(job("b3", make("b3"), batch=True))
    w.submit(job("p2", make("p2")))
    gate.set()
    for _ in range(200):
        if len(order) == 5:
            break
        time.sleep(0.01)
    assert order == ["b1", "p1", "p2", "b2", "b3"]


def test_preload_photos_never_make_a_person_wait_for_room():
    w = looked.Worker(lambda: False, threading.Lock(), max_waiting_photos=2, poll=0.005,
                      note=lambda s: None)
    for i in range(5):
        w.submit(job(f"b{i}", lambda got: {}, batch=True))
    assert w.waiting_photos() == 0 and w.batch_pending()
    w.submit(job("p1", lambda got: {}))
    w.submit(job("p2", lambda got: {}))
    with pytest.raises(looked.Busy):                  # people's own limit still holds
        w.submit(job("p3", lambda got: {}))


def test_an_identification_waits_at_most_for_the_one_map_being_made():
    """The preload feeds the worker one photo after another; an identification that arrives
    mid-map waits for that map alone, and no map starts while it runs."""
    lock, busy = threading.Lock(), threading.Event()
    starts, MAP = [], 0.2

    def make(got):
        starts.append(time.monotonic())
        time.sleep(MAP)
        return {"m": 1}
    w = looked.Worker(lambda: not busy.is_set(), lock, poll=0.005, note=lambda s: None)
    stop = threading.Event()

    def feeder():                                     # what Preloader.step does, one at a time
        i = 0
        while not stop.is_set():
            if not w.batch_pending():
                w.submit(job(f"b{i}", make, batch=True))
                i += 1
            time.sleep(0.002)
    threading.Thread(target=feeder, daemon=True).start()
    while len(starts) < 3:
        time.sleep(0.01)
    time.sleep(MAP / 3)                               # mid-map
    arrived = time.monotonic()
    busy.set()                                        # the model queue has an identification
    with lock:                                        # ... which takes the model lock
        got_it = time.monotonic()
        waited = got_it - arrived
        began = len(starts)
        time.sleep(0.5)                               # the identification runs
        during = len(starts) - began
    # A queued identification that has not reached the lock yet holds the next map back too.
    queued = time.monotonic()
    time.sleep(3 * MAP)
    held_back = sum(1 for s in starts if s > queued)
    busy.clear()
    stop.set()
    assert waited <= MAP + 0.1, waited
    assert during == 0                                # nothing starts while it runs
    assert sum(1 for s in starts if arrived < s <= got_it) <= 1   # at most the one in hand
    assert held_back <= 1, held_back


# --- the Preloader's own rules, with fakes ------------------------------------------------------

class FakeWorker:
    def __init__(self):
        self.jobs, self.pending, self.last_seconds = [], False, 4.2

    def batch_pending(self):
        return self.pending

    def submit(self, j):
        self.jobs.append(j)


def hit(n, category="slip", record=None):
    return xl.Excluded(f"inat:{n}", record or str(n), category)


STAMP = {"map_version": 1, "backbone": "m1", "method": "x", "release": "r1", "name": "A x",
         "box": None}


def preloader(slips=(), crops=(), waiting=0, kept=None, skip=None, window=None, enabled=True,
              crop_wait=True, nightly=False, now=AT(9)):
    kept = {} if kept is None else kept
    w = FakeWorker()
    made = []

    def make_job(t, done):
        made.append(t.key)
        return ("job", t.key, done)
    p = xl.Preloader(
        enabled=enabled, window=window, crop_wait=crop_wait, worker=w,
        targets=lambda: ([xl.Target(h) for h in slips], list(crops), waiting),
        stamp_now=lambda t: {**STAMP, "box": list(t.box) if t.box else None},
        stamps=lambda: dict(kept), stamp_of=lambda k: kept.get(k),
        skip=skip or (lambda t: None), job=make_job, nightly_running=lambda: nightly,
        now=lambda: now, note=lambda s: None)
    return p, w, made


def run(p, w, ticks=20):
    for _ in range(ticks):
        p.step()
        if w.jobs and not w.pending:                 # the worker finishes each at once
            _, key, done = w.jobs[-1]
            done(True)
            w.jobs.pop()


def test_it_makes_each_missing_map_one_photo_at_a_time():
    p, w, made = preloader(slips=[hit(1), hit(2), hit(3)])
    w.pending = True
    assert p.step() == "making a map" and not made     # the worker has one in hand: wait
    w.pending = False
    p.step()
    assert made == ["inat:1"] and len(w.jobs) == 1
    w.pending = True
    p.step()
    assert made == ["inat:1"]                          # never a second while one is pending


def test_current_maps_are_counted_done_and_stale_ones_remade():
    kept = {"inat:1": dict(STAMP), "inat:2": {**STAMP, "name": "an old name"},
            "inat:3": {**STAMP, "release": "r0"}}
    p, w, made = preloader(slips=[hit(1), hit(2), hit(3), hit(4)], kept=kept)
    run(p, w)
    assert made == ["inat:2", "inat:3", "inat:4"]
    h = p.health()
    assert h["slips"] == {"total": 4, "done": 4, "remaining": 0} and h["made"] == 3
    assert h["doing"] == "done" and h["last_map_seconds"] == 4.2 and h["last_map_at"]


def test_held_and_hidden_photos_are_skipped_and_counted():
    why = {"inat:2": "held by the seal", "inat:3": "not shown"}
    p, w, made = preloader(slips=[hit(1), hit(2), hit(3)], skip=lambda t: why.get(t.key))
    run(p, w)
    assert made == ["inat:1"]
    assert p.health()["skipped"] == {"held by the seal": 1, "not shown": 1}


def test_slips_wait_for_crops_to_be_embedded_and_embedded_crops_go_first():
    crop = xl.Target(hit(1), 1, (0.1, 0.1, 0.9, 0.9))
    p, w, made = preloader(slips=[hit(1), hit(2)], crops=[crop], waiting=5)
    run(p, w)
    assert made == ["inat:1#crop1"] and p.state == "waiting for crops to be embedded"
    assert p.health()["crops_waiting"] == 5
    p2, w2, made2 = preloader(slips=[hit(1), hit(2)], crops=[crop], waiting=5, crop_wait=False)
    run(p2, w2)
    assert made2 == ["inat:1", "inat:2", "inat:1#crop1"]          # slips first, then crops


def test_once_the_night_has_embedded_the_crops_the_next_pass_lets_the_slips_go():
    waiting = [3]
    clock = [0.0]
    w = FakeWorker()
    made = []
    p = xl.Preloader(enabled=True, window=None, crop_wait=True, worker=w,
                     targets=lambda: ([xl.Target(hit(1))], [], waiting[0]),
                     stamp_now=lambda t: dict(STAMP), stamps=dict, stamp_of=lambda k: None,
                     skip=lambda t: None, job=lambda t, done: made.append(t.key) or ("j", t.key, done),
                     nightly_running=lambda: False, rescan_seconds=3600, clock=lambda: clock[0],
                     note=lambda s: None)
    assert p.step() == "waiting for crops to be embedded"
    waiting[0] = 0                                    # the nightly embedded them
    clock[0] = 3599
    assert p.step() == "waiting for crops to be embedded" and not made
    clock[0] = 3600
    assert p.step() == "making a map" and made == ["inat:1"]


def test_a_crops_map_is_stale_when_its_box_changes():
    crop = xl.Target(hit(1), 1, (0.2, 0.2, 0.8, 0.8))
    kept = {"inat:1#crop1": {**STAMP, "box": [0.1, 0.1, 0.9, 0.9]}}
    p, w, made = preloader(crops=[crop], kept=kept)
    run(p, w)
    assert made == ["inat:1#crop1"]


def test_it_runs_only_in_its_window_never_beside_the_nightly_and_not_when_off():
    window = xl.parse_window("07:45-12:45")
    for when, expect in ((AT(7, 44), []), (AT(7, 45), ["inat:1"]), (AT(12, 44), ["inat:1"]),
                         (AT(12, 45), []), (AT(23), [])):
        p, w, made = preloader(slips=[hit(1)], window=window, now=when)
        run(p, w, 2)
        assert made == expect, when
    p, w, made = preloader(slips=[hit(1)], window=None, now=AT(23))        # "any"
    run(p, w, 2)
    assert made == ["inat:1"]
    p, w, made = preloader(slips=[hit(1)], nightly=True)
    assert p.step() == "waiting for the nightly update" and not made
    p, w, made = preloader(slips=[hit(1)], enabled=False)
    assert p.step() == "off" and not made and p.health()["enabled"] is False


def test_the_window_is_one_setting_and_wraps_past_midnight():
    assert xl.parse_window(None) == xl.parse_window(xl.PRELOAD_WINDOW) == (465, 765)
    assert xl.parse_window("any") is None
    night = xl.parse_window("22:00-03:00")
    assert xl.in_window(night, AT(23)) and xl.in_window(night, AT(2, 59))
    assert not xl.in_window(night, AT(3)) and not xl.in_window(night, AT(12))
    for bad in ("7-9", "25:00-03:00", "09:00-09:00", "soon"):
        with pytest.raises(ValueError):
            xl.parse_window(bad)


def test_a_failed_map_is_counted_and_the_pass_goes_on():
    p, w, made = preloader(slips=[hit(1), hit(2)])
    p.step()
    _, _, done = w.jobs.pop()
    done(False)
    run(p, w)
    assert made == ["inat:1", "inat:2"]
    assert p.health()["failed"] == 1 and p.health()["skipped"] == {"failed": 1}


def test_a_map_without_a_stamp_is_never_current():
    assert not xl.is_current({"name": "A x"}, STAMP) and not xl.is_current(None, STAMP)
    assert xl.is_current({"stamp": {**STAMP, "calibration": "c9"}}, STAMP)


# --- on the server ---------------------------------------------------------------------------------

@pytest.fixture
def any_time(monkeypatch):
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD", "1")          # off the box it is off by default
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD_WINDOW", "any")


def tick(app, n=30):
    for _ in range(n):
        app.state.excluded_preload.step()


def test_the_server_preloads_public_unsealed_slip_photos_only(conn, tmp_path, any_time):
    app, maker, _ = site(conn, tmp_path)
    tick(app)
    # Slips: 1003 (made) and 1005 (record 105, past the seal: never made).
    assert [c["name"] for c in maker.calls] == ["B y"]
    res = visitor(app).get(LOOKED, params={"photo": "inat:1003"})
    assert res.status_code == 200 and res.json()["stamp"]["backbone"] == "m1"
    assert visitor(app).get(LOOKED, params={"photo": "inat:1005"}).status_code == 403
    h = visitor(app).get("/api/health").json()["excluded_preload"]
    assert h["enabled"] is True and h["window"] == "any" and h["doing"] == "done"
    assert h["slips"] == {"total": 2, "done": 1, "remaining": 0}
    assert h["skipped"] == {"held by the seal": 1} and h["made"] == 1
    # Asked for again afterwards: kept, never made again.
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003"}).status_code == 200
    assert len(maker.calls) == 1


def test_a_map_goes_stale_when_its_records_name_changes(conn, tmp_path, any_time):
    app, maker, _ = site(conn, tmp_path)
    tick(app)
    conn.execute("update records set species = 'A x' where observation_id = '103'")
    conn.commit()
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003"}).json() == {"status": "none"}
    app.state.excluded_preload.pass_ = None          # the next pass (hourly on the box)
    tick(app)
    assert len(maker.calls) == 2
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003"}).json()["stamp"]["name"] == "A x"


def test_switched_off_it_makes_nothing(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD", "0")
    app, maker, _ = site(conn, tmp_path)
    tick(app)
    assert not maker.calls
    assert visitor(app).get("/api/health").json()["excluded_preload"]["doing"] == "off"


def test_it_is_on_by_default_on_the_box_only(conn, tmp_path, monkeypatch):
    monkeypatch.delenv("MV_EXCLUDED_PRELOAD", raising=False)
    app, maker, _ = site(conn, tmp_path)
    assert app.state.excluded_preload.enabled is False           # a laptop: no preload
    tick(app)
    assert not maker.calls
    assert xl.preload_on(None, on_box=True) and xl.preload_on("1", on_box=False)
    assert not xl.preload_on("0", on_box=True) and not xl.preload_on(None, on_box=False)


def test_the_health_keeps_the_two_state_keys_deploy_reads(conn, tmp_path, any_time):
    app, _, _ = site(conn, tmp_path)
    tick(app)
    import json
    block = visitor(app).get("/api/health").json()["excluded_preload"]
    assert '"state":' not in json.dumps(block, separators=(",", ":"))
    assert block["doing"] == "done"


def test_outside_its_window_it_waits(conn, tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    start = (now.hour + 2) % 24
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD", "1")
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD_WINDOW", f"{start:02d}:00-{(start + 1) % 24:02d}:00")
    app, maker, _ = site(conn, tmp_path)
    tick(app)
    assert not maker.calls and app.state.excluded_preload.state == "outside its window"


def keep_crop(conn, app, photo_id=1003, n=1, box=(0.25, 0.25, 0.75, 0.75), embed=True):
    ensure_crops_schema(conn)
    conn.execute("insert into feedback_photo_crops values (?,?,?,?,?,?,?,?)",
                 (photo_id, n, f"inat:{photo_id}", *box, "t1"))
    conn.commit()
    if embed:
        store = app.state.test_store
        todo = crops_to_embed(conn, "m1", "large", store.location, north_america_only=False)
        assert todo
        embed_photos(conn, store, Const("m1"), todo, app.state.test_root / "m1",
                     log=lambda s: None)
        conn.commit()


def test_an_embedded_crop_gets_its_own_map_of_the_box_alone(conn, tmp_path, any_time):
    app, maker, _ = site(conn, tmp_path)
    app.state.test_root = tmp_path / "emb"
    keep_crop(conn, app)
    tick(app)
    assert len(maker.calls) == 2                     # the slip photo, then its crop
    whole, crop = (Image.open(io.BytesIO(c["photo"])) for c in maker.calls)
    assert abs(crop.width - whole.width / 2) <= 2 and abs(crop.height - whole.height / 2) <= 2
    assert maker.calls[1]["hidden_records"] == {"103"}        # its record left out, as ever
    got = visitor(app).get(LOOKED, params={"photo": "inat:1003", "crop": 1}).json()
    assert got["status"] == "ready" and got["crop_of_photo"] == {
        "n": 1, "box": [0.25, 0.25, 0.75, 0.75]}
    assert got["stamp"]["box"] == [0.25, 0.25, 0.75, 0.75]
    h = visitor(app).get("/api/health").json()["excluded_preload"]
    assert h["crops"] == {"total": 1, "done": 1, "remaining": 0}
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003", "crop": 2}).status_code == 404


def test_a_crop_not_yet_embedded_has_no_map_and_holds_the_slips_back(conn, tmp_path, any_time,
                                                                      monkeypatch):
    app, maker, _ = site(conn, tmp_path)
    keep_crop(conn, app, embed=False)
    monkeypatch.setattr(app.state.excluded_preload, "targets",
                        lambda real=app.state.excluded_preload.targets: (*real()[:2], 1))
    tick(app)
    assert not maker.calls
    assert app.state.excluded_preload.state == "waiting for crops to be embedded"
    assert visitor(app).get(LOOKED, params={"photo": "inat:1003", "crop": 1}).status_code == 404
    assert person(app, "m").post(LOOKED, params={"photo": "inat:1003", "crop": 1}
                                 ).status_code == 404
