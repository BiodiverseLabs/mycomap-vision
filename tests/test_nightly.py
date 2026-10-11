import io
import os
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from conftest import inat_obs
from test_api import open_limits, post_photos
from test_models_and_scoreboard import OPEN, Const, seed_two_species

from mycomap_vision import manifest, nightly
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.identify import Identifier
from mycomap_vision.inat import save_batch
from mycomap_vision.photos import Result, pending_photos, save_result
from mycomap_vision.records import build_records, save_records

RID = "20261001-000000-test"
SEED = [("A x", "1/1/2026"), ("A x", "1/2/2026"), ("A x", "1/3/2026"), ("B y", "1/1/2026"),
        ("B y", "1/2/2026"), ("B y", "1/3/2026"), ("A x", "9/20/2026"), ("B y", "9/21/2026")]
QUIET = dict(log=lambda s: None)
# mycomap.org's own answer when VISION_API_KEY is unset (requireVisionKey).
NO_KEY = {"code": "not_configured", "message": "VISION_API_KEY is not set on this deployment."}


@pytest.fixture(autouse=True)
def no_retry_pauses(monkeypatch):
    """A failing page is asked again after pauses of minutes; not in tests."""
    monkeypatch.setattr(nightly, "PAGE_RETRY_PAUSES", (0.0, 0.0))


def row(oid, name, when="10/1/2026"):
    return {"source": "iNaturalist", "observation_id": str(oid), "scientific_name": name, "genus": name.split()[0],
            "family": "F", "continent": "North America", "validation_status_1": "yes",
            "validation_date_1": when}


def seed_rows():
    return [row(100 + i, sp, when) for i, (sp, when) in enumerate(SEED)]


@pytest.fixture
def box(tmp_path):
    """A server box: one release (the two-species set, embedded with backbone m1) under
    <root>/releases/<RID>, named in current.txt."""
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


def png(red):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (red, 0, 0)).save(buf, "PNG")
    return buf.getvalue()


class FakeOutside:
    """iNat and its photos for the nightly steps: each new record gets one photo whose
    red level is in `reds` (by observation id); every photo asked for is noted."""

    def __init__(self, reds):
        self.reds, self.downloads = reds, []

    def fetch_inat(self, conn, raw_dir=None, log=print):
        from mycomap_vision.inat import pending_ids
        ids = pending_ids(conn, False, True)
        save_batch(conn, ids, [inat_obs(int(o), photos=[(0, int(o) * 10, "cc0", OPEN)])
                               for o in ids], "t-night")
        return {"ok": len(ids)}

    def taxonomy(self, conn, max_minutes=None, cache=None, log=print):
        return {"asked": 0}

    def download(self, conn, store, size, limit=None, first_seen_since=None,
                 unembedded_for=None, log=print):
        rows = pending_photos(conn, True, limit, size, store.location,
                              first_seen_since=first_seen_since, unembedded_for=unembedded_for)
        with conn:
            for r in rows:
                pid = r["photo_id"]
                rel = f"n/{pid}.png"
                store.put(rel, png(self.reds[pid // 10]))
                save_result(conn, Result(pid, "done", rel, 1, "h"), size, "now", store.location)
                self.downloads.append(pid)
        return {"done": len(rows)}

    def steps(self):
        return nightly.Steps(fetch_inat=self.fetch_inat, taxonomy=self.taxonomy,
                             download=self.download)


def night(layer, rows, outside=None, accept_removals=False, settings=None):
    """One nightly run on the layer with `rows` as mycomap.org's answer."""
    outside = outside or FakeOutside({})
    conn = manifest.connect(layer.manifest)
    try:
        return nightly.run_once(conn, layer, lambda: (rows, "2026-10-08T07:00:00.000Z"),
                                {"m1": lambda: Const("m1")}, settings,
                                accept_removals=accept_removals, steps=outside.steps(), **QUIET)
    finally:
        conn.close()


def names_in(path):
    c = sqlite3.connect(path)
    try:
        return dict(c.execute("select observation_id, scientific_name from records"))
    finally:
        c.close()


# --- the layer -----------------------------------------------------------------------

def test_the_layer_is_a_copy_of_the_release_made_once_and_the_release_is_never_written(box):
    layer = nightly.prepare(box, **QUIET)
    release_bytes = (box / "releases" / RID / "manifest.sqlite").read_bytes()
    report = night(layer, seed_rows()[:-1] + [row(107, "B renamed")])
    assert report["ok"] and report["changed"]
    assert (box / "releases" / RID / "manifest.sqlite").read_bytes() == release_bytes
    again = nightly.prepare(box, **QUIET)                   # not copied over the night's work
    assert names_in(again.manifest)["107"] == "B renamed"


def test_a_new_release_gets_a_fresh_layer_and_only_the_server_deletes_the_old_one(box, tmp_path):
    old = nightly.prepare(box, **QUIET)
    new_rid = "20261101-000000-next"
    os.rename(box / "releases" / RID, box / "releases" / new_rid)
    (box / "current.txt").write_text(new_rid, encoding="utf-8")
    new = nightly.prepare(box, prune=False, **QUIET)
    assert new.root != old.root and old.root.is_dir()     # a CLI run leaves it for the server
    nightly.prepare(box, prune=True, **QUIET)
    assert not old.root.exists() and new.manifest.is_file()


def test_nightly_shards_start_after_the_releases_and_are_read_from_the_layer(box):
    layer = nightly.prepare(box, **QUIET)
    conn = sqlite3.connect(layer.manifest)
    [(first, release_rows)] = conn.execute(
        "select first_shard, release_rows from nightly_layer where backbone = 'm1'").fetchall()
    conn.close()
    assert release_rows == 8 and first == 1                # the release wrote shard 0
    night(layer, seed_rows() + [row(300, "C z")], FakeOutside({300: 130}))
    assert (layer.embeddings / "m1" / "shard-00001.npy").is_file()
    assert not (box / "releases" / RID / "embeddings" / "m1" / "shard-00001.npy").exists()
    conn = sqlite3.connect(layer.manifest)
    ident = Identifier(conn, "m1", "nearest", box / "releases" / RID / "embeddings" / "m1",
                       layer_root=layer.embeddings)
    conn.close()
    assert ident.embedded == 9 and "C z" in ident.index.species


# --- what a night changes ------------------------------------------------------------

def test_a_night_adds_new_records_drops_unvalidated_ones_and_renames(box):
    layer = nightly.prepare(box, **QUIET)
    rows = [r for r in seed_rows() if r["observation_id"] != "101"]
    rows[0]["scientific_name"] = "A renamed"
    rows.append(row(300, "C z"))
    outside = FakeOutside({300: 130})
    report = night(layer, rows, outside)
    assert report["records"] == {"before": 8, "after": 8, "new": 1, "removed": 1, "renamed": 1}
    assert report["embedded"] == {"m1": 1}
    names = names_in(layer.manifest)
    assert "101" not in names and names["100"] == "A renamed" and names["300"] == "C z"


def test_only_records_new_since_the_layer_began_are_fetched_never_the_releases_unembedded(box):
    layer = nightly.prepare(box, **QUIET)
    # The release holds a record it never embedded (a sample release): not tonight's job.
    conn = manifest.connect(layer.manifest)
    save_records(conn, build_records(seed_rows() + [row(200, "D w")], "2026-09-01T00:00:00+00:00"))
    save_batch(conn, ["200"], [inat_obs(200, photos=[(0, 2000, "cc0", OPEN)])], "t")
    conn.close()
    outside = FakeOutside({300: 130})
    night(layer, seed_rows() + [row(200, "D w"), row(300, "C z")], outside)
    assert outside.downloads == [3000]


def test_embedded_photos_are_deleted_from_the_box_and_never_fetched_again(box):
    layer = nightly.prepare(box, **QUIET)
    outside = FakeOutside({300: 130})
    report = night(layer, seed_rows() + [row(300, "C z")], outside)
    assert report["photos_deleted"] == 1
    assert not any(p.is_file() for p in layer.photos.rglob("*"))
    night(layer, seed_rows() + [row(300, "C z")], outside)
    assert outside.downloads == [3000]                     # the vector is kept, not the photo


def test_a_night_with_nothing_new_changes_nothing(box):
    layer = nightly.prepare(box, **QUIET)
    report = night(layer, seed_rows())
    assert report["ok"] and not report["changed"]
    conn = sqlite3.connect(layer.manifest)
    assert nightly.layer_version(conn) == 0
    conn.close()


def test_an_answer_removing_too_many_records_is_refused_until_accepted(box):
    layer = nightly.prepare(box, **QUIET)
    strict = nightly.Settings(max_removed=2, max_removed_share=0.0)
    report = night(layer, seed_rows()[:5], settings=strict)       # removes 3 of 8
    assert not report["ok"] and "REFUSED" in report["error"] and "--accept-removals" in report["error"]
    assert len(names_in(layer.manifest)) == 8
    report = night(layer, seed_rows()[:5], settings=strict, accept_removals=True)
    assert report["ok"] and len(names_in(layer.manifest)) == 5


def test_the_removal_limit_is_a_floor_or_a_share_of_the_list_whichever_is_larger():
    s = nightly.Settings()
    assert s.removal_limit(1_000) == 500
    assert s.removal_limit(165_000) == 3_300


def test_a_failed_night_is_recorded_and_changes_nothing(box):
    layer = nightly.prepare(box, **QUIET)
    conn = manifest.connect(layer.manifest)

    def broken():
        raise nightly.NightlyRefused("mycomap.org answered 500 on page 1; nothing was changed")
    report = nightly.run_once(conn, layer, broken, {"m1": lambda: Const("m1")}, **QUIET)
    status = nightly.status(conn, nightly.Settings())
    conn.close()
    assert not report["ok"] and len(names_in(layer.manifest)) == 8
    assert status["runs"][0]["state"] == "failed" and "500" in status["runs"][0]["error"]
    assert status["last_ok_at"] is None


def test_the_layer_says_when_a_new_release_is_due(box):
    layer = nightly.prepare(box, **QUIET)
    rows = seed_rows() + [row(300, "C z"), row(301, "C z")]
    night(layer, rows, FakeOutside({300: 130, 301: 131}))
    conn = sqlite3.connect(layer.manifest)
    sizes = nightly.layer_sizes(conn, warn_share=0.20)
    conn.close()
    assert sizes["m1"] == {"release_photos": 8, "nightly_photos": 2, "share": 0.25,
                           "new_release_due": True}


def test_two_runs_never_work_on_the_layer_at_once(tmp_path):
    lock = tmp_path / "nightly.lock"
    with nightly.RunLock(lock):
        with pytest.raises(nightly.NightlyBusy):
            with nightly.RunLock(lock):
                pass
    assert not lock.exists()
    lock.write_text("123 long ago")
    old = time.time() - nightly.LOCK_STALE_SECONDS - 60
    os.utime(lock, (old, old))
    with nightly.RunLock(lock):                             # left by a run that died
        pass


def test_nightly_embedding_takes_turns_with_identifications_on_the_model():
    lock = threading.Lock()
    model = nightly.Locked(Const("m1"), lock)
    done = threading.Event()
    lock.acquire()
    t = threading.Thread(target=lambda: (model.encode([Image.new("RGB", (4, 4))]), done.set()))
    t.start()
    assert not done.wait(0.2)
    lock.release()
    assert done.wait(5)
    assert model.name == "m1" and model.dim == 2


# --- the site, the same night ---------------------------------------------------------

def site(box, layer, rows, outside):
    return TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name), limits=open_limits(), background=False,
        layer=layer, nightly_fetch=lambda: (rows, "2026-10-08T07:00:00.000Z"),
        nightly_steps=outside.steps(), note=lambda s: None))


def top_species(client, red):
    res = post_photos(client, [red], "m1/nearest")
    assert res.status_code == 200
    return res.json()["results"][0]["ranks"]["species"][0]["name"]


def test_new_renamed_and_unvalidated_records_change_the_site_the_same_night(box):
    layer = nightly.prepare(box, **QUIET)
    rows = [r for r in seed_rows() if r["scientific_name"] != "B y"]       # B y un-validated
    rows = [dict(r, scientific_name="A renamed") if r["scientific_name"] == "A x" else r
            for r in rows] + [row(300, "C z")]
    client = site(box, layer, rows, FakeOutside({300: 130}))
    assert top_species(client, 130) in ("A x", "B y")       # before: no C z to find
    assert top_species(client, 12) == "B y"
    report = client.app.state.run_nightly()
    assert report["ok"] and report["changed"]
    assert top_species(client, 130) == "C z"
    assert top_species(client, 248) == "A renamed"
    assert top_species(client, 12) != "B y"


def test_a_rename_alone_rebuilds_the_index(box):
    layer = nightly.prepare(box, **QUIET)
    rows = [dict(r, scientific_name="B new") if r["scientific_name"] == "B y" else r
            for r in seed_rows()]
    client = site(box, layer, rows, FakeOutside({}))
    assert top_species(client, 12) == "B y"
    client.app.state.run_nightly()
    assert top_species(client, 12) == "B new"


def test_without_the_nightly_update_the_server_has_no_nightly_job(box):
    client = TestClient(create_app(box / "releases" / RID / "manifest.sqlite",
                                   box / "releases" / RID / "embeddings",
                                   backbone_loader=lambda name: Const(name), limits=open_limits(),
                                   background=False, note=lambda s: None))
    assert client.app.state.run_nightly is None


# --- reading mycomap.org --------------------------------------------------------------

class Page:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


class Pages:
    """mycomap.org's green-records pages, answered by cursor."""

    def __init__(self, pages):
        self.pages, self.calls = pages, []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), headers))
        after = (params or {}).get("after", "")
        got = self.pages[after]
        if isinstance(got, Exception):
            raise got
        return got


def page(records, nxt, total=None):
    body = {"generatedAt": "2026-10-08T07:00:00.000Z", "count": len(records), "next": nxt,
            "records": records}
    if total is not None:
        body["total"] = total
    return Page(200, body)


def test_every_page_is_read_with_the_key_following_the_cursor():
    a, b = seed_rows()[:5], seed_rows()[5:]
    s = Pages({"": page(a, "104", total=8), "104": page(b, None)})
    rows, generated = nightly.fetch_green("https://mycomap.org/", "k" * 40, session=s, limit=5,
                                          **QUIET)
    assert [r["observation_id"] for r in rows] == [r["observation_id"] for r in a + b]
    assert generated == "2026-10-08T07:00:00.000Z"
    assert [c[1] for c in s.calls] == [{"limit": "5"}, {"limit": "5", "after": "104"}]
    assert s.calls[0][0] == "https://mycomap.org/api/vision/green-records"
    assert s.calls[0][2]["Authorization"] == "Bearer " + "k" * 40


@pytest.mark.parametrize("pages, reason", [
    ({"": Page(401)}, "refused the key"),
    ({"": Page(503, NO_KEY)}, "no VISION_API_KEY"),
    ({"": Page(503)}, "answered 503"),           # a busy server or a proxy, not the key
    ({"": Page(500)}, "answered 500"),
    ({"": Page(200)}, "not JSON"),
    ({"": Page(200, {"permissions": []})}, "not a green-records page"),
    ({"": page([], None, total=0)}, "no green records"),
    ({"": page(seed_rows(), None)}, "no total"),
    ({"": page(seed_rows()[:2], None, total=500)}, "announced 500"),
    ({"": page(seed_rows()[:2], "x", total=4), "x": page(seed_rows()[2:4], "x")}, "repeats"),
    ({"": page([{"scientific_name": "A x"}], None, total=1)}, "without an observation_id"),
])
def test_a_bad_answer_is_refused_whole_and_never_repeats_the_key(pages, reason):
    with pytest.raises(nightly.NightlyRefused, match=reason) as e:
        nightly.fetch_green("https://mycomap.org", "secret-key-" + "x" * 30, session=Pages(pages),
                            **QUIET)
    assert "secret-key" not in str(e.value) and "nothing was changed" in str(e.value)


class Flaky:
    """Fails the first `fails` requests (an exception or a response), then answers."""

    def __init__(self, fails, then):
        self.fails, self.then, self.calls = list(fails), then, 0

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls += 1
        if self.fails:
            got = self.fails.pop(0)
            if isinstance(got, Exception):
                raise got
            return got
        return self.then


def test_a_page_that_fails_is_asked_again_after_a_pause():
    import requests
    slept = []
    s = Flaky([requests.ReadTimeout("slow"), Page(500)], page(seed_rows(), None, total=8))
    rows, _ = nightly.fetch_pending("https://mycomap.org", "k" * 40, session=s, pauses=(30, 120),
                                    sleep=slept.append, **QUIET)
    assert len(rows) == 8 and s.calls == 3 and slept == [30, 120]


def test_a_page_that_keeps_failing_is_refused_after_its_tries():
    s = Flaky([Page(500), Page(502), Page(504)], page(seed_rows(), None, total=8))
    with pytest.raises(nightly.NightlyRefused, match="answered 504 on page 1, 3 tries"):
        nightly.fetch_green("https://mycomap.org", "k" * 40, session=s, pauses=(1, 1),
                            sleep=lambda s: None, **QUIET)


@pytest.mark.parametrize("answer", [Page(401), Page(400), Page(503, NO_KEY)])
def test_a_refused_key_a_bad_request_or_a_missing_key_is_not_asked_again(answer):
    s = Flaky([answer], page(seed_rows(), None, total=8))
    with pytest.raises(nightly.NightlyRefused):
        nightly.fetch_green("https://mycomap.org", "k" * 40, session=s, pauses=(1, 1),
                            sleep=lambda s: None, **QUIET)
    assert s.calls == 1


def test_an_unreachable_mycomap_org_is_refused():
    import requests
    with pytest.raises(nightly.NightlyRefused, match="could not reach"):
        nightly.fetch_green("https://mycomap.org", "k" * 40,
                            session=Pages({"": requests.ConnectionError("down")}), **QUIET)


# --- when ------------------------------------------------------------------------------

EST = timezone(timedelta(hours=-5))


def test_the_next_run_is_the_next_set_time_in_the_set_zone():
    at = lambda s: datetime.fromisoformat(s)  # noqa: E731
    assert nightly.next_run(at("2026-01-10T07:59:00+00:00"), "03:00", EST) == \
        at("2026-01-10T08:00:00+00:00")
    assert nightly.next_run(at("2026-01-10T08:00:00+00:00"), "03:00", EST) == \
        at("2026-01-11T08:00:00+00:00")


def test_3_am_eastern_follows_daylight_saving_time():
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo("America/New_York")
    except Exception:  # noqa: BLE001 - no tz database here (Windows without tzdata)
        pytest.skip("no America/New_York in this Python's time zone database")
    summer = nightly.next_run(datetime(2026, 7, 1, 12, tzinfo=timezone.utc), "03:00", zone)
    winter = nightly.next_run(datetime(2026, 12, 1, 12, tzinfo=timezone.utc), "03:00", zone)
    assert summer.hour == 7 and winter.hour == 8


def test_the_schedule_runs_at_the_time_or_when_asked_and_takes_removals_only_when_asked(box):
    layer = nightly.prepare(box, **QUIET)
    sched = nightly.Schedule(layer, nightly.Settings(at="03:00"), EST)
    t0 = datetime(2026, 1, 10, 7, 0, tzinfo=timezone.utc)                # 02:00 EST
    assert sched.due(t0) == (False, False)
    layer.run_now.touch()
    layer.accept_removals.touch()
    assert sched.due(t0) == (True, True)
    assert not layer.run_now.exists() and not layer.accept_removals.exists()
    assert sched.due(t0 + timedelta(minutes=59, seconds=59)) == (False, False)
    assert sched.due(t0 + timedelta(hours=1)) == (True, False)           # 03:00 EST
    assert sched.due(t0 + timedelta(hours=2)) == (False, False)


def test_bad_settings_are_refused():
    with pytest.raises(ValueError, match="HH:MM"):
        nightly.Settings(at="3am")


# --- commands on the box ---------------------------------------------------------------

def test_commands_on_the_box_open_the_layer_copy_once_the_server_made_it(box, monkeypatch):
    from mycomap_vision import config
    monkeypatch.setattr(config, "RELEASE_ROOT", box)
    monkeypatch.setenv("MV_NIGHTLY", "1")
    assert nightly.served_manifest() == config.MANIFEST_PATH          # no layer yet
    layer = nightly.prepare(box, **QUIET)
    assert nightly.served_manifest() == layer.manifest
    monkeypatch.setenv("MV_NIGHTLY", "0")
    assert nightly.served_manifest() == config.MANIFEST_PATH


def test_unembedded_for_keeps_photos_without_that_backbones_vector(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    conn.executescript("create table if not exists embeddings (backbone text, photo_id integer, "
                       "shard integer, row integer, created_at text, primary key (backbone, photo_id))")
    conn.execute("insert into embeddings values ('m1', 1000, 0, 0, 'now')")
    ids = {r["photo_id"] for r in pending_photos(conn, True, None, "large", "elsewhere",
                                                 unembedded_for="m1")}
    assert 1000 not in ids and 1001 in ids
    assert np.isin(1000, [r["photo_id"] for r in pending_photos(conn, True, None, "large",
                                                                "elsewhere")])


def test_mv_nightly_plan_prints_tonights_changes_and_writes_nothing(box, monkeypatch, capsys):
    import json
    from mycomap_vision import cli, config
    layer = nightly.prepare(box, **QUIET)
    monkeypatch.setattr(config, "RELEASE_ROOT", box)
    monkeypatch.setattr(config, "DATA_DIR", box / "releases" / RID)
    monkeypatch.setenv("MV_NIGHTLY", "1")
    monkeypatch.setenv("MV_ORG_BASE_URL", "https://mycomap.org")
    monkeypatch.setenv("MV_ORG_VISION_KEY", "k" * 40)
    rows = seed_rows()[1:] + [row(300, "C z")]
    monkeypatch.setattr(nightly, "fetch_green", lambda base, key: (rows, "g"))
    before = names_in(layer.manifest)
    assert cli.main(["nightly", "--plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert (plan["new"], plan["removed"], plan["renamed"]) == (1, 1, 0)
    assert plan["removals_refused"] is False
    assert names_in(layer.manifest) == before


def test_mv_nightly_now_asks_the_server_and_reports_runs(box, monkeypatch, capsys):
    import json
    from mycomap_vision import cli, config
    layer = nightly.prepare(box, **QUIET)
    night(layer, seed_rows())
    monkeypatch.setattr(config, "RELEASE_ROOT", box)
    monkeypatch.setattr(config, "DATA_DIR", box / "releases" / RID)
    monkeypatch.setenv("MV_NIGHTLY", "1")
    assert cli.main(["nightly", "--now", "--accept-removals"]) == 0
    assert layer.run_now.exists() and layer.accept_removals.exists()
    capsys.readouterr()
    assert cli.main(["nightly"]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["runs"][0]["state"] == "ok" and status["layer"]["m1"]["release_photos"] == 8
