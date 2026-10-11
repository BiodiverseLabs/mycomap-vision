"""The queue in front of the model (workqueue.py): interactive first, one search per batch,
honest refusals, and answers identical to scoring one request at a time."""

import threading
from collections import Counter

import numpy as np
import pytest
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import apiv1
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.workqueue import Busy, Job, ModelQueue


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def queue(**kw):
    kw.setdefault("clock", Clock())
    return ModelQueue(threading.Lock(), start=False, note=lambda s: None, **kw)


def job(q, key="idx", photos=1, interactive=True, searches=None, fail=None, timer=None,
        embed_s=0.0):
    """A job whose vectors are its photo numbers; finish returns the sims rows it got."""
    vecs = np.arange(photos, dtype=np.float32)[:, None] + 100 * len(q._jobs)

    def embed():
        if fail == "embed":
            raise ValueError("unreadable")
        q.clock.t += embed_s
        return vecs

    def search(v):
        searches.append(v.copy()) if searches is not None else None
        q.clock.t += 2.0
        return np.concatenate([v, v * 10], axis=1)

    def finish(v, rows):
        if fail == "finish":
            raise RuntimeError("bad answer")
        return {"vecs": v.ravel().tolist(), "rows": rows.tolist()}
    j = Job(key=key, photos=photos, embed=embed, search=search, finish=finish,
            interactive=interactive, timer=timer)
    j.seq, j.arrived = len(q._jobs) + 100 * q.batches, q.clock()
    q._jobs.append(j)
    return j


def test_jobs_waiting_on_the_same_index_share_one_search():
    q, searches = queue(), []
    a, b, c = (job(q, photos=n, searches=searches) for n in (1, 2, 3))
    q.run_batch(q._take_batch())
    assert len(searches) == 1 and len(searches[0]) == 6
    # Each job gets exactly its own rows back.
    for j in (a, b, c):
        out = j.future.result(0)
        assert [r[0] for r in out["rows"]] == out["vecs"]


def test_interactive_calls_go_first_and_other_indexes_wait():
    q = queue()
    dev = job(q, key="idx", interactive=False)
    other = job(q, key="old-idx", interactive=False)
    app = job(q, key="idx")
    first = q._take_batch()
    assert first[0] is app and dev in first and other not in first
    assert q._take_batch() == [other]


def test_a_batch_stops_at_its_photo_limit():
    q = queue(max_batch_photos=5)
    a, b, c = job(q, photos=3), job(q, photos=3), job(q, photos=2)
    assert q._take_batch() == [a, c]
    assert q._take_batch() == [b]


def test_one_jobs_failure_fails_only_that_job():
    q = queue()
    bad_photo, bad_answer, good = job(q, fail="embed"), job(q, fail="finish"), job(q)
    q.run_batch(q._take_batch())
    with pytest.raises(ValueError):
        bad_photo.future.result(0)
    with pytest.raises(RuntimeError):
        bad_answer.future.result(0)
    assert "rows" in good.future.result(0)


def test_a_job_is_refused_when_the_work_ahead_passes_the_limit():
    q = queue(max_wait=10, embed_seconds=1.6, search_seconds=2.0)
    q.admit()                                    # nothing ahead: always admitted
    job(q, photos=3, interactive=False)          # 3 x 1.6 + 2 = 6.8 s
    q.admit()
    job(q, photos=3, interactive=False)          # 6 x 1.6 + 2 = 11.6 s
    with pytest.raises(Busy) as e:
        q.admit(interactive=False)
    assert e.value.wait_seconds == pytest.approx(11.6)
    # The app goes ahead of developer calls, so their work isn't ahead of it.
    q.admit(interactive=True)


def test_timing_splits_own_work_from_waiting_for_the_batch():
    q = queue()
    ta, tb = apiv1.Timer(clock=q.clock), apiv1.Timer(clock=q.clock)
    job(q, photos=1, timer=ta, embed_s=1.5)
    job(q, photos=2, timer=tb, embed_s=3.0)
    q.run_batch(q._take_batch())
    a, b = ta.block(), tb.block()
    assert a["embed_ms"] == 1500 and b["embed_ms"] == 3000
    assert a["score_ms"] == b["score_ms"] == 2000              # the one shared search
    assert a["queue_ms"] == 3000                               # waited for b's photos
    assert b["queue_ms"] == 1500                               # waited for a's
    for t in (a, b):
        assert t["total_ms"] == t["queue_ms"] + t["embed_ms"] + t["score_ms"]
    assert (a["photos_embedded"], b["photos_embedded"]) == (1, 2)


def test_the_estimates_follow_what_batches_take():
    q = queue(embed_seconds=1.6, search_seconds=2.0)
    job(q, photos=2, embed_s=10.0)               # throttled: 5 s a photo
    q.run_batch(q._take_batch())
    assert q.embed_seconds > 1.6 * 1.5


def test_the_worker_answers_callers_waiting_on_it():
    q = ModelQueue(threading.Lock(), note=lambda s: None)
    done = []

    def caller(n):
        j = Job(key="idx", photos=1, embed=lambda: np.ones((1, 2), np.float32) * n,
                search=lambda v: v.copy(), finish=lambda v, rows: float(rows[0, 0]))
        done.append(q.run(j, timeout=10))
    threads = [threading.Thread(target=caller, args=(n,)) for n in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert sorted(done) == [0.0, 1.0, 2.0, 3.0, 4.0]


def test_a_batched_search_answers_exactly_as_one_at_a_time(conn, tmp_path):
    from mycomap_vision.identify import Identifier, map_embeddings
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    root = tmp_path / "emb" / "m1"
    embed_photos(conn, store, Const("m1"), todo, root, log=lambda s: None)
    conn.commit()
    ident = Identifier(conn, "m1", "nearest+mean", embeddings_root=root, photo_info=False)
    _, vecs = map_embeddings(conn, "m1", root)
    a = np.asarray(vecs[[0, 3]], dtype=np.float32)
    b = np.asarray(vecs[[5]], dtype=np.float32)
    alone = [ident.identify_vectors(a), ident.identify_vectors(b)]
    sims = ident.nearest.photo_sims(np.concatenate([a, b]))
    together = [ident.identify_vectors(a, sims=sims[:2]), ident.identify_vectors(b, sims=sims[2:])]
    assert together == alone


def test_health_reports_the_queue(conn, tmp_path):
    from test_api import app_with_model
    client = app_with_model(conn, tmp_path)
    q = client.get("/api/health").json()["queue"]
    assert set(q) >= {"queued_jobs", "queued_photos", "embed_seconds_per_photo", "batches"}


def test_the_site_and_v1_say_busy_with_the_estimated_wait(conn, tmp_path):
    from test_api import app_with_model, post_photos
    from test_api_v1_identify import app_token, bearer, identify, v1_app
    client = v1_app(conn, tmp_path)
    mq = client.app.state.model_queue
    mq.max_wait = -1                             # anything ahead is too much
    res = identify(client, bearer(app_token()))
    assert res.status_code == 503 and res.json()["code"] == "busy"
    assert int(res.headers["retry-after"]) >= 1
    assert res.json()["estimated_wait_seconds"] == int(res.headers["retry-after"])
    site = post_photos(client, [247], "m1/nearest")
    assert site.status_code == 503 and "retry-after" in site.headers


def test_a_job_scoring_stored_vectors_reports_them_as_reused():
    q = queue()
    t = apiv1.Timer(clock=q.clock)
    j = Job(key="idx", photos=1, embed=lambda: np.ones((4, 2), np.float32),
            search=lambda v: v.copy(), finish=lambda v, rows: len(rows), timer=t)
    j.seq, j.arrived = 0, 0.0
    q._jobs.append(j)
    q.run_batch(q._take_batch())
    assert j.future.result(0) == 4
    assert (t.photos_embedded, t.photos_reused) == (1, 3)
