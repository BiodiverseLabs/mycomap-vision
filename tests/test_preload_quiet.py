"""Preload maps give way to a stream of identifications (Steve, 2026-11-07, "Make maps give way
to .org's queue"). mycomap.org's auto-validation asks for one record at a time and sends the next
ask as soon as the last is answered; before this a preload map started in every such gap, so
each identification waited ~12 s behind one. Now a preload (batch) photo starts only after the
model queue has been quiet for QUIET_SECONDS, a person's ask still goes ahead as before, and a map
already being made is never stopped."""

import threading
import time

import numpy as np
import pytest
from test_excluded_looked_api import site
from test_run_links import visitor

from mycomap_vision import excluded_looked as xl
from mycomap_vision import looked, workqueue


def ident(q: workqueue.ModelQueue, during=lambda: None):
    """One identification through the real model queue, answered by its worker."""
    def embed():
        during()
        return np.zeros((1, 2))
    return q.run(workqueue.Job(key="k", photos=1, embed=embed,
                               search=lambda v: np.zeros((1, 1)), finish=lambda v, s: {}),
                 timeout=5)


def job(rid, make, batch=False):
    return looked.Job(rid, [looked.Task(0, lambda: None, make)], lambda e: None, batch=batch)


def wait_for(cond, seconds=3.0):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.005)
    return cond()


# --- the model queue says how long it has been quiet ----------------------------------------

def test_the_queue_is_quiet_forever_before_its_first_identification_and_not_while_one_runs():
    now = [100.0]
    q = workqueue.ModelQueue(threading.Lock(), clock=lambda: now[0])
    assert q.quiet_seconds() == float("inf")
    inside = []
    ident(q, during=lambda: inside.append(q.quiet_seconds()))
    assert inside == [0.0]                                # running: not quiet
    assert wait_for(q.idle)
    assert q.quiet_seconds() == 0.0                       # just answered
    now[0] += 42.5
    assert q.quiet_seconds() == 42.5


def test_quiet_is_counted_from_when_the_last_identification_arrived_or_was_answered():
    now = [0.0]
    q = workqueue.ModelQueue(threading.Lock(), clock=lambda: now[0])

    def slow():                                           # arrives at 0, answered at 30
        now[0] = 30.0
    ident(q, during=slow)
    assert wait_for(q.idle)
    now[0] = 50.0
    assert q.quiet_seconds() == 20.0                      # from the answer, not the arrival
    q2 = workqueue.ModelQueue(threading.Lock(), clock=lambda: now[0], start=False)
    t = threading.Thread(target=lambda: ident(q2), daemon=True)
    t.start()
    assert wait_for(lambda: not q2.idle())
    now[0] = 500.0
    assert q2.quiet_seconds() == 0.0                      # waiting in the queue: never quiet
    with q2._cond:
        batch = q2._take_batch()
    q2.run_batch(batch)                                   # answer it, so its caller returns
    t.join(5)


# --- the worker: a preload photo waits for quiet ----------------------------------------------

def test_a_preload_map_does_not_start_within_the_quiet_period_and_does_once_it_passes():
    now = [1000.0]
    q = workqueue.ModelQueue(threading.Lock(), clock=lambda: now[0])
    made = []
    w = looked.Worker(q.idle, q.lock, poll=0.005, note=lambda s: None,
                      quiet=lambda: q.quiet_seconds() >= looked.QUIET_SECONDS)
    ident(q)                                              # .org's identification, answered
    w.submit(job("b1", lambda got: made.append("b1") or {}, batch=True))
    time.sleep(0.15)
    assert made == []                                     # idle, but not quiet
    now[0] += looked.QUIET_SECONDS - 1
    time.sleep(0.15)
    assert made == []                                     # 59 s: still waiting
    now[0] += 1
    assert wait_for(lambda: made == ["b1"])               # 60 s quiet: made


def test_no_preload_map_slips_in_between_back_to_back_identifications():
    """The 7 Nov journal: one map between every two of .org's identifications. Now none."""
    q = workqueue.ModelQueue(threading.Lock())
    QUIET, starts, answered = 0.3, [], []
    w = looked.Worker(q.idle, q.lock, poll=0.005, note=lambda s: None,
                      quiet=lambda: q.quiet_seconds() >= QUIET)
    stop = threading.Event()

    def feeder():                                         # Preloader.step, one photo at a time
        i = 0
        while not stop.is_set():
            if not w.batch_pending():
                w.submit(job(f"b{i}", lambda got: starts.append(time.monotonic()) or {},
                             batch=True))
                i += 1
            time.sleep(0.002)
    threading.Thread(target=feeder, daemon=True).start()
    began = time.monotonic()
    while time.monotonic() - began < 1.0:                 # a stream: the next ask at once
        ident(q, during=lambda: time.sleep(0.01))
        answered.append(time.monotonic())
        time.sleep(0.02)                                  # .org reads the answer, asks again
    last = answered[-1]
    assert wait_for(lambda: any(s > last for s in starts), 2.0)   # quiet again: maps resume
    stop.set()
    assert len(answered) >= 10
    assert not any(began <= s <= last for s in starts), "a map started inside the stream"
    assert min(s for s in starts if s > last) - last >= QUIET - 0.05


def test_a_persons_ask_still_goes_ahead_while_a_preload_photo_waits_for_quiet():
    quiet = threading.Event()
    order = []
    w = looked.Worker(lambda: True, threading.Lock(), poll=0.005, note=lambda s: None,
                      quiet=quiet.is_set)
    b = job("b1", lambda got: order.append("b1") or {}, batch=True)
    w.submit(b)
    time.sleep(0.05)
    w.submit(job("p1", lambda got: order.append("p1") or {}))     # idle, not quiet: made now
    assert wait_for(lambda: order == ["p1"])
    assert not b.finished.is_set() and w.batch_pending()          # the preload photo waits on
    quiet.set()
    assert wait_for(lambda: order == ["p1", "b1"]) and b.finished.wait(2)


def test_a_preload_photo_in_hand_steps_back_for_a_person_without_being_lost_or_made_twice():
    order, reads = [], []
    w = looked.Worker(lambda: True, threading.Lock(), background=False, note=lambda s: None,
                      sleep=lambda s: None)
    b = looked.Job("b1", [looked.Task(0, lambda: reads.append("b1"),
                                      lambda got: order.append("b1") or {})],
                   lambda e: None, batch=True)
    with w._cond:
        w._queue.append(b)
    calls = {"n": 0}

    def quiet_until_a_person_asks():
        calls["n"] += 1
        if calls["n"] > 50:                                # it never stepped back
            raise AssertionError("the preload photo kept the person waiting")
        if calls["n"] == 2:                                # read done, then .org asks again
            with w._cond:
                w._queue.insert(0, job("p1", lambda got: order.append("p1") or {}))
        return calls["n"] == 1
    w.quiet = quiet_until_a_person_asks
    w._run_next()                                          # takes b1, reads it, steps back
    assert order == [] and not b.finished.is_set()
    assert [j.rid for j in w._queue] == ["p1", "b1"] and w._current is None
    w.quiet = lambda: True
    w._run_next()
    w._run_next()
    assert order == ["p1", "b1"] and b.finished.is_set() and reads == ["b1", "b1"]


def test_with_no_quiet_rule_a_preload_photo_goes_whenever_the_queue_is_idle():
    made = []
    w = looked.Worker(lambda: True, threading.Lock(), background=False, note=lambda s: None)
    w.submit(job("b1", lambda got: made.append("b1") or {}, batch=True))
    assert made == ["b1"]


# --- the Preloader says so and submits nothing --------------------------------------------------

class FakeWorker:
    def __init__(self):
        self.jobs, self.pending, self.last_seconds = [], False, 4.2

    def batch_pending(self):
        return self.pending

    def submit(self, j):
        self.jobs.append(j)


def test_the_preloader_gives_way_while_identifications_keep_coming_and_says_why():
    quiet = {"on": False}
    w = FakeWorker()
    p = xl.Preloader(
        enabled=True, window=None, crop_wait=True, worker=w,
        targets=lambda: ([xl.Target(xl.Excluded("inat:1", "1", "slip"))], [], 0),
        stamp_now=lambda t: {"map_version": 1, "backbone": "m1", "method": "x",
                             "release": "r1", "name": "A x", "box": None},
        stamps=dict, stamp_of=lambda k: None, skip=lambda t: None,
        job=lambda t, done: ("job", t.key, done), nightly_running=lambda: False,
        quiet=lambda: quiet["on"], stall_minutes=0, note=lambda s: None)
    assert p.step() == "giving way to identifications" and w.jobs == []
    assert "identifications kept coming" in p.why()
    quiet["on"] = True
    assert p.step() == "making a map" and len(w.jobs) == 1


# --- on the server -------------------------------------------------------------------------------

@pytest.fixture
def any_time(monkeypatch):
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD", "1")
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD_WINDOW", "any")


def tick(app, n=30):
    for _ in range(n):
        app.state.excluded_preload.step()


def test_the_server_makes_no_preload_map_within_a_minute_of_an_identification(
        conn, tmp_path, any_time, monkeypatch):
    monkeypatch.delenv("MV_EXCLUDED_PRELOAD_QUIET_SECONDS", raising=False)
    app, maker, _ = site(conn, tmp_path)
    q = app.state.model_queue
    with q._cond:
        q._last_seen = q.clock() - 30                     # .org's last identify, 30 s ago
    tick(app)
    assert maker.calls == []
    h = visitor(app).get("/api/health").json()["excluded_preload"]
    assert h["doing"] == "giving way to identifications"
    with q._cond:
        q._last_seen = q.clock() - 61                     # a quiet minute
    tick(app)
    assert [c["name"] for c in maker.calls] == ["B y"]


def test_the_quiet_period_is_a_setting_and_0_means_whenever_idle(
        conn, tmp_path, any_time, monkeypatch):
    monkeypatch.setenv("MV_EXCLUDED_PRELOAD_QUIET_SECONDS", "0")
    app, maker, _ = site(conn, tmp_path)
    q = app.state.model_queue
    with q._cond:
        q._last_seen = q.clock()                          # answered this instant
    tick(app)
    assert [c["name"] for c in maker.calls] == ["B y"]
