"""Switching the box to a new release (nightly.py): the fresh layer gets the last layer's
"often confused with" table and, when its records are old, one catch-up night soon after
the server starts, instead of serving the laptop's days-old export until the next 3 a.m.
(2026-10-18 and -19: 368 new and 887 renamed records missing for most of a day, and no
confused-with notes until `mv confused-with --fetch`)."""

import os
import sqlite3
import threading
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from test_confusions import A_X, B_Y, answer, write_table
from test_nightly import QUIET, RID, FakeOutside, box, night, row, seed_rows  # noqa: F401
from test_api import open_limits
from test_models_and_scoreboard import Const

from mycomap_vision import confusions, manifest, nightly
from mycomap_vision.api import create_app, first_confused_with

NEW_RID = "20261101-000000-next"
NOW = datetime(2026, 10, 19, 15, 15, tzinfo=timezone.utc)
UTC_ZONE = timezone.utc


def switch_release(box):
    """`mv pull-release` of a new release: its own folder, named in current.txt."""
    rel = box / "releases" / NEW_RID
    rel.mkdir(parents=True)
    for p in (box / "releases" / RID).rglob("*"):
        if p.is_file():
            dest = rel / p.relative_to(box / "releases" / RID)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(p.read_bytes())
    (box / "current.txt").write_text(NEW_RID + "\n", encoding="utf-8")


def set_exported(layer, when: datetime):
    c = sqlite3.connect(layer.manifest)
    with c:
        c.execute("update records set exported_at = ?", (when.isoformat(),))
    c.close()


def reason(layer, settings=None, now=NOW):
    c = manifest.connect(layer.manifest)
    try:
        return nightly.catch_up_reason(c, settings or nightly.Settings(), now)
    finally:
        c.close()


# --- the confused-with table ---------------------------------------------------------------

def test_a_new_releases_layer_carries_over_the_often_confused_with_table(box):
    old = nightly.prepare(box, **QUIET)
    write_table(old.root, A_X, B_Y)
    held = (old.root / confusions.CACHE).read_bytes()
    switch_release(box)
    said = []
    new = nightly.prepare(box, prune=True, log=said.append)
    assert not old.root.exists()
    assert (new.root / confusions.CACHE).read_bytes() == held
    assert confusions.health(new.root / confusions.CACHE)["cached"]   # its state came too
    assert confusions.load(new.root / confusions.CACHE).confused_with("A x")
    assert any("carried over from release " + RID in s for s in said)


def test_a_layer_keeps_its_own_table_and_the_first_layer_has_none(box):
    first = nightly.prepare(box, **QUIET)
    assert not (first.root / confusions.CACHE).exists()            # nothing to carry
    write_table(first.root, A_X)
    switch_release(box)
    new = nightly.prepare(box, **QUIET)
    write_table(new.root, B_Y)                                     # the new layer's own night
    mine = (new.root / confusions.CACHE).read_bytes()
    assert nightly.carry_confused_with(new, **QUIET) is None
    assert (new.root / confusions.CACHE).read_bytes() == mine


def test_the_newest_other_layer_with_a_table_is_the_one_carried(box):
    root = box / "state" / "nightly"
    for rid, rows in (("20261001-000000-older", (B_Y,)), ("20261015-000000-newer", (A_X,)),
                      ("20261020-000000-none", ())):
        (root / rid).mkdir(parents=True)
        if rows:
            write_table(root / rid, *rows)
    layer = nightly.prepare(box, **QUIET)
    assert nightly.carry_confused_with(layer, **QUIET) is None      # prepare already did
    table = confusions.load(layer.root / confusions.CACHE)
    assert table.confused_with("A x") and not table.confused_with("B y")


def test_the_server_reads_orgs_table_at_startup_when_its_layer_has_none(box):
    layer = nightly.prepare(box, **QUIET)
    asked = []

    def misid(cache, labels=None, log=print):
        asked.append(cache)
        return confusions.refresh(cache, lambda: answer(A_X), **QUIET)
    served = confusions.Served(confusions.cache_path(layer.manifest), note=lambda s: None)
    first_confused_with(served, layer, nightly.Steps(misidentifications=misid), lambda s: None)
    assert asked == [layer.root / confusions.CACHE]
    assert served.table() is not None and served.table().confused_with("A x")
    first_confused_with(served, layer, nightly.Steps(misidentifications=misid), lambda s: None)
    assert len(asked) == 1                                          # it has one now


def test_a_laptop_server_never_reads_orgs_table_at_startup(tmp_path):
    def misid(cache, labels=None, log=print):
        raise AssertionError("read without a layer")
    served = confusions.Served(tmp_path / confusions.CACHE, note=lambda s: None)
    first_confused_with(served, None, nightly.Steps(misidentifications=misid), lambda s: None)
    assert served.table() is None


def test_a_failed_startup_read_leaves_the_notes_off_and_the_server_up(box):
    layer = nightly.prepare(box, **QUIET)

    def misid(cache, labels=None, log=print):
        raise RuntimeError("down")
    said = []
    served = confusions.Served(confusions.cache_path(layer.manifest), note=said.append)
    first_confused_with(served, layer, nightly.Steps(misidentifications=misid), said.append)
    assert served.table() is None and any("startup read failed" in s for s in said)


# --- when a catch-up is due --------------------------------------------------------------

def test_a_fresh_layer_of_an_old_export_is_due_a_catch_up(box):
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, NOW - timedelta(days=6))                  # the 10-13 export, on 10-19
    why = reason(layer)
    assert why and "6.0 days ago" in why


def test_a_fresh_layer_of_a_recent_export_waits_for_the_night(box):
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, NOW - timedelta(hours=3))
    assert reason(layer) is None
    assert reason(layer, nightly.Settings(catch_up_after_hours=2)) is not None


def test_a_layer_without_a_known_export_time_is_due(box):
    layer = nightly.prepare(box, **QUIET)                        # seed rows say "t"
    assert "unknown time" in reason(layer)


def test_a_layer_that_had_a_good_night_is_never_due_again(box):
    layer = nightly.prepare(box, **QUIET)
    assert night(layer, seed_rows())["ok"]
    set_exported(layer, NOW - timedelta(days=6))       # a night's rows are fresh: not the test
    assert reason(layer) is None


def test_a_layer_whose_nights_were_all_refused_is_still_due(box):
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, NOW - timedelta(days=6))
    strict = nightly.Settings(max_removed=1, max_removed_share=0.0)
    assert not night(layer, seed_rows()[:4], settings=strict)["ok"]   # the removal guard
    assert reason(layer) is not None


def test_the_catch_up_can_be_turned_off(box, monkeypatch):
    layer = nightly.prepare(box, **QUIET)
    monkeypatch.setenv("MV_NIGHTLY_CATCHUP_AFTER_HOURS", "0")
    monkeypatch.setenv("MV_NIGHTLY_CATCHUP_DELAY_MINUTES", "2.5")
    settings = nightly.Settings.from_env()
    assert settings.catch_up_delay_minutes == 2.5
    assert reason(layer, settings) is None


# --- the catch-up run ----------------------------------------------------------------------

def armed(layer, ready=None, settings=None):
    sched = nightly.Schedule(layer, settings or nightly.Settings(at="03:00"), UTC_ZONE)
    c = manifest.connect(layer.manifest)
    try:
        why = nightly.arm_catch_up(sched, c, ready=ready, now=NOW, **{"note": lambda s: None})
    finally:
        c.close()
    return sched, why


def test_the_catch_up_runs_once_after_the_delay_and_only_once_the_preload_is_done(box):
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, NOW - timedelta(days=6))
    loading = [True]
    sched, why = armed(layer, ready=lambda: not loading[0])
    assert why and sched.catch_up_at == NOW + timedelta(minutes=10)
    assert sched.due(NOW + timedelta(minutes=9)) == (False, False)
    assert sched.due(NOW + timedelta(minutes=11)) == (False, False)      # preload still on
    loading[0] = False
    assert sched.due(NOW + timedelta(minutes=12)) == (True, False)       # removals refused
    assert sched.due(NOW + timedelta(minutes=13)) == (False, False)      # once
    assert sched.next_at == datetime(2026, 10, 20, 3, 0, tzinfo=timezone.utc)  # night kept


def test_the_catch_up_takes_removals_only_when_asked(box):
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, NOW - timedelta(days=6))
    sched, _ = armed(layer)
    layer.accept_removals.touch()
    assert sched.due(NOW + timedelta(minutes=10)) == (True, True)
    assert not layer.accept_removals.exists()


def test_a_run_asked_for_before_the_catch_up_is_the_catch_up(box):
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, NOW - timedelta(days=6))
    sched, _ = armed(layer)
    layer.run_now.touch()
    assert sched.due(NOW + timedelta(minutes=1)) == (True, False)
    assert sched.catch_up_at is None
    assert sched.due(NOW + timedelta(minutes=20)) == (False, False)


def test_a_recent_layer_gets_no_catch_up(box):
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, NOW - timedelta(hours=1))
    sched, why = armed(layer)
    assert why is None and sched.catch_up_at is None
    assert sched.due(NOW + timedelta(hours=1)) == (False, False)


def test_the_catch_up_brings_in_whats_new_since_the_export_and_respects_the_photo_cap(box):
    old = nightly.prepare(box, **QUIET)
    night(old, seed_rows() + [row(300, "C z")], FakeOutside({300: 130}))   # the old release
    switch_release(box)
    new = nightly.prepare(box, prune=True, **QUIET)
    set_exported(new, NOW - timedelta(days=6))
    assert "300" not in names(new)                       # the switch dropped it
    sched, _ = armed(new)
    run, accept = sched.due(NOW + timedelta(minutes=10))
    assert run
    rows = seed_rows() + [row(300, "C z"), row(301, "C z")]
    rows[0] = dict(rows[0], scientific_name="A renamed")
    report = night(new, rows, FakeOutside({300: 130, 301: 131}), accept_removals=accept,
                   settings=nightly.Settings(max_photos=1))
    assert report["ok"] and report["records"]["new"] == 2 and report["records"]["renamed"] == 1
    assert report["embedded"] == {"m1": 1}               # MV_NIGHTLY_MAX_PHOTOS holds
    assert names(new)["300"] == "C z" and names(new)["100"] == "A renamed"
    assert reason(new) is None                           # the next restart doesn't redo it


def names(layer):
    c = sqlite3.connect(layer.manifest)
    try:
        return dict(c.execute("select observation_id, scientific_name from records"))
    finally:
        c.close()


def test_a_catch_up_due_at_startup_is_run_on_the_first_tick(box):
    """start() used to call due() once only to learn the next run, dropping a run that was
    due then: a run-now flag left before a restart, or a catch-up with no delay."""
    layer = nightly.prepare(box, **QUIET)
    layer.run_now.touch()
    sched = nightly.Schedule(layer, nightly.Settings(at="03:00"), UTC_ZONE)
    ran = threading.Event()
    nightly.start(sched, lambda accept: ran.set() or {}, note=lambda s: None, tick=0.01,
                  clock=lambda: NOW)
    assert ran.wait(5)


def test_the_server_arms_the_catch_up_behind_its_preload(box, monkeypatch):
    monkeypatch.delenv("MV_ORG_BASE_URL", raising=False)
    monkeypatch.setattr(nightly.Settings, "zone", lambda self: UTC_ZONE)
    started = []
    monkeypatch.setattr(nightly, "start", lambda schedule, job, note=print, **_:
                        started.append(schedule))
    layer = nightly.prepare(box, **QUIET)
    set_exported(layer, nightly.now_utc() - timedelta(days=6))
    outside = FakeOutside({})
    app = create_app(layer.manifest, box / "releases" / RID / "embeddings",
                     backbone_loader=lambda name: Const(name), limits=open_limits(),
                     background=True, preload=[], layer=layer,
                     nightly_fetch=lambda: (seed_rows(), "2026-10-19T07:00:00Z"),
                     nightly_steps=outside.steps(), note=lambda s: None)
    with TestClient(app):
        [sched] = started
        assert sched.catch_up_at is not None and sched.ready is not None and sched.ready()


# --- the lock ------------------------------------------------------------------------------

def test_a_lock_left_by_a_process_that_is_gone_is_stale_at_once(tmp_path):
    lock = tmp_path / "nightly.lock"
    lock.write_text("4242 2026-10-19T15:20:00+00:00\n")            # a restart mid-run
    with pytest.raises(nightly.NightlyBusy):
        with nightly.RunLock(lock, alive=lambda pid: True):
            pass
    with nightly.RunLock(lock, alive=lambda pid: False):
        assert lock.read_text().split()[0] == str(os.getpid())
    assert not lock.exists()


def test_a_lock_whose_holder_cannot_be_read_waits_for_its_age(tmp_path):
    lock = tmp_path / "nightly.lock"
    lock.write_text("garbled")
    with pytest.raises(nightly.NightlyBusy):
        with nightly.RunLock(lock, alive=lambda pid: False):
            pass
