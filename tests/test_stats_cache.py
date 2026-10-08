"""The Data page's statistics: computed rarely, never holding up identifications."""
import threading
import time

from test_api import app_with_model, open_limits

from mycomap_vision.api import Cached
from mycomap_vision.records import build_records, save_records


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_a_fresh_value_is_answered_from_memory():
    calls = []
    c = Cached(lambda: calls.append(1) or {"n": len(calls)}, max_age=60, clock=Clock())
    assert c.get() == {"n": 1} and c.get() == {"n": 1} and len(calls) == 1


def test_a_stale_value_is_answered_at_once_while_it_is_recomputed_behind():
    clock, release, calls = Clock(), threading.Event(), []

    def compute():
        calls.append(1)
        if len(calls) > 1:
            release.wait(5)                  # the refresh takes a while
        return {"n": len(calls)}
    c = Cached(compute, max_age=60, clock=clock)
    assert c.get() == {"n": 1}
    clock.t = 61
    started = time.monotonic()
    assert c.get() == {"n": 1}                # stale, answered without waiting
    assert time.monotonic() - started < 1
    release.set()
    for _ in range(100):
        if c.get() == {"n": 2}:
            break
        time.sleep(0.02)
    assert c.get() == {"n": 2} and len(calls) == 2


def test_a_failed_refresh_keeps_the_last_figures():
    clock, said, calls = Clock(), [], []

    def compute():
        calls.append(1)
        if len(calls) > 1:
            raise RuntimeError("database busy")
        return {"n": 1}
    c = Cached(compute, max_age=60, clock=clock, note=said.append)
    c.get()
    clock.t = 61
    c.get()
    for _ in range(100):
        if said:
            break
        time.sleep(0.02)
    assert c.get() == {"n": 1} and "database busy" in said[0]


def test_the_data_page_shows_only_the_served_models_and_no_download_bookkeeping(conn, tmp_path):
    client = app_with_model(conn, tmp_path, limits=open_limits(allowed_backbones={"m1"}))
    conn.execute("insert into embeddings values ('test-backbone', 1000, 0, 0, 'now')")
    conn.commit()
    client.app.state.stats_cache.max_age = 0                 # recount on this call
    client.get("/api/stats")
    time.sleep(0.2)
    s = client.get("/api/stats").json()
    assert list(s["embedded"]) == ["m1"]
    assert "photos_by_status" not in s and "photos_by_size" not in s


def test_the_figures_are_counted_once_an_hour_but_photographers_answers_are_current(conn,
                                                                                      tmp_path):
    client = app_with_model(conn, tmp_path)
    first = client.get("/api/stats").json()
    save_records(conn, build_records([{"observation_id": "999", "scientific_name": "Z z",
                                       "continent": "North America",
                                       "validation_status_1": "yes"}], "t2"))
    conn.execute("insert into photo_permissions (inat_user_id, status) values (7, 'granted')")
    conn.commit()
    again = client.get("/api/stats").json()
    assert again["records"] == first["records"]                     # not counted again yet
    assert again["permissions"]["granted"] == first["permissions"]["granted"] + 1
