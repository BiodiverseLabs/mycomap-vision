"""The photo queues read in the background (feedback_lists.Warmer; Steve, 2026-11-20: "the first
tab is still slow"): at startup once the preload is done, after a seed, and after another
process's write, so the first person to open a tab gets the kept list instead of the 11 s whole
read. A whole read reads every photo queue in one pass, holds the store's lock one id range at a
time, and keeps the items whose rows did not change; its lists are exactly a fresh read. Nothing
here waits on real time except as a generous upper bound."""

import functools
import random
import threading
import time

import test_feedback_api
from test_feedback_api import as_user
from test_feedback_lists import NAMED, Names, check, fill_with, make_queue, open_ids, photo_rows
from test_photo_review import site_with_photos

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_lists as fl
from mycomap_vision import photo_review as pr
from mycomap_vision import photo_views as pv
from mycomap_vision import recovered_labels as rl
from mycomap_vision.api import create_app

STEVE = {"sub": "u1", "name": "Steve"}
WAIT = 60          # seconds: only ever reached when something is broken


def warmer_for(store, names=None, **kw):
    lists = fl.ItemLists(store, fill_with(names or Names()))
    kw.setdefault("poll", 3600.0)               # no look-ups by the clock: only triggers
    kw.setdefault("pause", 0.0)
    warmer = fl.Warmer(lists, note=lambda s: None, **kw)
    store.listeners.append(warmer.trigger)
    return lists, warmer


def fresh(store, names):
    return {t: fill_with(names)(t, store.items_of(t)) for t in NAMED}


# --- the whole read -----------------------------------------------------------------------
def test_one_pass_reads_every_photo_queue(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists = fl.ItemLists(s, fill_with(names))
    assert set(fl.WARM_TYPES) == {pr.ITEM_TYPE, pv.ITEM_TYPE, rl.ITEM_TYPE}
    assert lists.of(pr.ITEM_TYPE) == fresh(s, names)[pr.ITEM_TYPE]
    assert lists.full_reads == 1 and set(lists.lists) == set(fl.WARM_TYPES)
    for t in (pv.ITEM_TYPE, rl.ITEM_TYPE):
        assert lists.of(t) == fresh(s, names)[t] and lists.last[t] == "same"
    assert lists.full_reads == 1


def test_a_whole_read_keeps_unchanged_items_and_builds_only_new_changed_and_patched_rows(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists = fl.ItemLists(s, fill_with(names))
    lists.of(pr.ITEM_TYPE)
    total = sum(len(lists.lists[t].raw) for t in fl.WARM_TYPES)
    assert (lists.rows_built, lists.rows_kept) == (total, 0)        # a restart builds them all
    patched = open_ids(s, 1)[0]
    s.decide(patched, "keep_photo", STEVE)
    check(lists, s, names, "patch")
    flagged = open_ids(s, 1)[0]                                    # a seed: 3 new, 1 flag added
    rng = random.Random(3)
    s.seed(photo_rows(rng, 3, 40_000) + [{**photo_rows(rng, 1)[0], "subject": s.get(flagged)["subject"],
                                          "rule": "R11.1 v2 uv p=0.043 (very_low)"}], "nightly")
    built, kept = lists.rows_built, lists.rows_kept
    check(lists, s, names, "full")
    assert lists.rows_built - built == 3 + 1 + 1                    # new, flag added, patched
    assert lists.rows_kept - kept == total - 1 - 1
    other = fb.Store(s.path)                                       # by hand, another connection
    other.conn.execute("update items set evidence = '{\"by\": \"hand\"}' where id = ?", (patched,))
    other.conn.commit()
    built = lists.rows_built
    check(lists, s, names, "full")
    assert lists.rows_built - built == 1
    assert next(i for i in lists.of(pr.ITEM_TYPE) if i["id"] == patched)["evidence"] == {"by": "hand"}


def test_a_whole_read_holds_the_store_lock_one_id_range_at_a_time(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists = fl.ItemLists(s, fill_with(names), read_step=25)
    calls, saved = [], []
    real = s._rows

    def rows(sql, params):
        out = real(sql, params)
        calls.append(params)
        if len(calls) == 3:                    # mid-read: another thread saves, and finishes
            t = threading.Thread(target=lambda: saved.append(
                s.decide(open_ids(s, 1)[0], "keep_photo", STEVE)))
            t.start()
            t.join(WAIT)
            assert saved, "a save waited for the whole read"
        return out
    s._rows = rows
    lists.of(pr.ITEM_TYPE)
    s._rows = real
    assert len(calls) > 5 and saved
    check(lists, s, names, "patch")            # the save made mid-read: found by the next look


# --- the warm-up --------------------------------------------------------------------------
def test_the_startup_warm_up_waits_for_the_preload_then_the_first_request_reads_nothing_whole(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    preload = threading.Event()
    lists, warmer = warmer_for(s, names, after=preload.wait)
    warmer.start()
    assert not warmer.idle(timeout=0.2) and lists.full_reads == 0   # waits for the preload
    preload.set()
    assert warmer.idle(WAIT)
    assert lists.full_reads == 1 and warmer.health()["last"]["reasons"] == ["startup"]
    for t in NAMED:                                                  # the first request
        assert lists.of(t) == fresh(s, names)[t] and lists.last[t] == "same"
    assert lists.full_reads == 1


def test_a_seed_warms_the_lists_before_anyone_asks(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists, warmer = warmer_for(s, names)
    warmer.start()
    assert warmer.idle(WAIT) and lists.full_reads == 1
    s.seed(photo_rows(random.Random(4), 5, 50_000), "nightly-photo-review")
    assert warmer.idle(WAIT)
    assert lists.full_reads == 2 and warmer.health()["last"]["reasons"] == ["seed"]
    for t in NAMED:
        assert lists.of(t) == fresh(s, names)[t] and lists.last[t] == "same"
    s.reopen(next(i["id"] for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "decided"))
    assert warmer.idle(WAIT)                                         # a reopen: patched ahead
    assert lists.full_reads == 2 and warmer.health()["last"]["hows"][pr.ITEM_TYPE] == "patch"
    assert lists.of(pr.ITEM_TYPE) == fresh(s, names)[pr.ITEM_TYPE]
    assert lists.last[pr.ITEM_TYPE] == "same"


def test_requests_during_a_warm_up_wait_for_it_and_triggers_during_it_make_one_more_run(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists, warmer = warmer_for(s, names)
    entered, release = threading.Event(), threading.Event()
    real = s.latest_ranges

    def slow(*a, **kw):
        entered.set()
        assert release.wait(WAIT)
        yield from real(*a, **kw)
    s.latest_ranges = slow
    warmer.start()
    assert entered.wait(WAIT)
    got = {}
    asks = [threading.Thread(target=lambda k=k, t=t: got.__setitem__(k, lists.of(t)))
            for k, t in enumerate(NAMED * 3)]
    for a in asks:
        a.start()
    for _ in range(4):
        warmer.trigger("seed")
    release.set()
    for a in asks:
        a.join(WAIT)
    assert warmer.idle(WAIT)
    assert lists.full_reads == 1                                     # one whole read for all
    want = fresh(s, names)
    assert all(got[k] == want[t] for k, t in enumerate(NAMED * 3))
    runs = warmer.health()["runs"]
    assert runs == 2 and warmer.runs[-1]["whole"] is False           # the 4 triggers: one run


def test_another_process_write_is_found_by_the_look_and_warmed(tmp_path):
    s = make_queue(tmp_path)
    names = Names()
    lists, warmer = warmer_for(s, names, poll=0.05)
    warmer.start()
    assert warmer.idle(WAIT) and not lists.needs_whole()
    other = fb.Store(s.path)
    other.decide(open_ids(s, 1)[0], "keep_photo", STEVE)            # `mv feedback`, by hand
    assert lists.needs_whole()
    for _ in range(WAIT * 20):                                       # the next look finds it
        if lists.full_reads == 2:
            break
        time.sleep(0.05)
    assert lists.full_reads == 2
    assert warmer.idle(WAIT) and not lists.needs_whole()
    check(lists, s, names, "same")


def test_every_write_the_lists_cannot_patch_tells_the_listeners(tmp_path):
    s = make_queue(tmp_path)
    heard = []
    s.listeners += [heard.append, lambda e: 1 / 0]                  # a broken listener: ignored
    s.seed(photo_rows(random.Random(5), 1, 60_000), "x")
    item = next(i for i in s.items_of(pr.ITEM_TYPE) if i["status"] == "decided")
    s.reopen(item["id"])
    never = next(i for i in s.items_of(pr.ITEM_TYPE) if i["last_decision"] is None)
    assert s.close_by_system(never["id"], "keep_photo", "mycomap.org", "mycomap.org")
    assert not s.close_by_system(never["id"], "keep_photo", "mycomap.org", "mycomap.org")
    path = tmp_path / "priority.tsv"
    path.write_text("rank\tphoto_key\trecord_key\tgroup\tlost_species\n", encoding="utf-8")
    rl.seed_priority(s, path)                                       # its own seeds row
    assert heard == ["seed", "reopen", "system close", "seed"]


# --- through the app ----------------------------------------------------------------------
def test_with_feedback_warm_the_first_tab_after_a_restart_takes_the_kept_list(conn, tmp_path,
                                                                             monkeypatch):
    monkeypatch.setattr(test_feedback_api, "create_app",
                        functools.partial(create_app, feedback_warm=True))
    c, fstore = site_with_photos(conn, tmp_path)
    lists, warmer = c.app.state.feedback_lists, c.app.state.feedback_warmer
    assert warmer.idle(WAIT)
    reads = lists.full_reads
    assert reads >= 1
    as_user(c, "admin-1")
    tiles = c.get("/api/feedback/photo-review/flagged").json()["tiles"]
    assert len(tiles) == 3 and lists.last[pr.ITEM_TYPE] in ("same", "patch")
    assert lists.full_reads == reads
    r = c.post("/api/feedback/photo-review/batch", json={"decisions": [
        {"item_id": tiles[0]["item_id"], "decision": "exclude", "reason": "micrograph"}]})
    assert r.status_code == 200
    assert c.get("/api/feedback/photo-review/flagged").json()["total"] == 2
    assert lists.full_reads == reads


def test_without_feedback_warm_nothing_reads_in_the_background(conn, tmp_path):
    c, _ = site_with_photos(conn, tmp_path)
    assert not hasattr(c.app.state, "feedback_warmer")
    assert c.app.state.feedback_lists.full_reads == 0


def test_the_collector_paused_for_a_whole_read_is_always_turned_back_on(tmp_path):
    import gc
    s = make_queue(tmp_path)
    lists = fl.ItemLists(s, fill_with(Names()), read_step=50)
    lists.of(pr.ITEM_TYPE)
    assert gc.isenabled()
    real = s.latest_ranges

    def broken(*a, **kw):
        it = real(*a, **kw)
        yield next(it)
        raise OSError("disk I/O error")
    s.latest_ranges = broken
    s.seed(photo_rows(random.Random(8), 1, 70_000), "nightly")
    try:
        lists.of(pr.ITEM_TYPE)
    except OSError:
        pass
    else:
        raise AssertionError("the read error was swallowed")
    assert gc.isenabled()
    s.latest_ranges = real
    check(lists, s, Names(), "full")                     # the next request reads it whole
