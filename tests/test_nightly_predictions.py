import time

import pytest
from fastapi.testclient import TestClient

from conftest import inat_obs
from test_api import open_limits
from test_models_and_scoreboard import Const
from test_nightly import (RID, FakeOutside, Page, Pages, QUIET, box,  # noqa: F401
                          night, page, row, seed_rows)
from test_prospective import FakeFetcher, FakeIdentifier

from mycomap_vision import manifest, nightly, prospective
from mycomap_vision.api import create_app


def pending(oid, continent="North America"):
    return {"observation_id": str(oid), "scientific_name": "pending sp.", "continent": continent,
            "country": None, "latitude": None, "longitude": None}


def run_row(conn):
    with conn:
        return conn.execute("insert into nightly_runs (started_at, report) values ('t', '{}')"
                            ).lastrowid


# --- reading what awaits validation -------------------------------------------------

def test_pending_records_are_read_from_their_own_endpoint_and_may_be_none():
    s = Pages({"": page([], None, total=0)})
    rows, _ = nightly.fetch_pending("https://mycomap.org", "k" * 40, session=s, **QUIET)
    assert rows == [] and s.calls[0][0] == "https://mycomap.org/api/vision/pending-records"
    with pytest.raises(nightly.NightlyRefused, match="no green records"):
        nightly.fetch_green("https://mycomap.org", "k" * 40, session=s, **QUIET)


def test_a_bad_pending_answer_is_refused_like_a_green_one():
    with pytest.raises(nightly.NightlyRefused, match="pending-records page"):
        nightly.fetch_pending("https://mycomap.org", "k" * 40,
                              session=Pages({"": Page(200, {"records": "x"})}), **QUIET)


def test_only_records_still_pending_tonight_are_predicted(conn):
    prospective.save_candidates(conn, [pending(10), pending(11)], "2026-10-01T00:00:00+00:00")
    prospective.save_candidates(conn, [pending(11)], "2026-10-08T07:00:00+00:00")
    assert prospective.unpredicted(conn, "m1", "nearest") == ["11", "10"]
    assert prospective.unpredicted(conn, "m1", "nearest",
                                   seen_since="2026-10-08T07:00:00+00:00") == ["11"]


# --- a night's predictions -------------------------------------------------------------

def test_a_night_predicts_the_newest_pending_records_up_to_its_limit_and_records_it(conn):
    nightly.ensure_schema(conn)
    run = run_row(conn)
    rows = [pending(10), pending(11), pending(12), pending(13, continent="Europe")]
    fetcher = FakeFetcher({str(i): inat_obs(i, photos=[(0, i, "cc0", "x")]) for i in (10, 11, 12)})
    out = nightly.advance_predictions(conn, run, lambda: (rows, "g"),
                                      lambda: (FakeIdentifier(), object()), fetcher, 2, **QUIET)
    assert out == {"model": "m1/hybrid", "pending": 4, "asked": 2, "predicted": 2}
    assert sorted(r[0] for r in conn.execute("select observation_id from predictions")) == \
        ["11", "12"]
    import json
    report = json.loads(conn.execute("select report from nightly_runs where id = ?",
                                     (run,)).fetchone()[0])
    assert report["predictions"]["predicted"] == 2


def test_a_failed_prediction_step_is_recorded_and_never_raised(conn):
    nightly.ensure_schema(conn)
    run = run_row(conn)

    def down():
        raise nightly.NightlyRefused("mycomap.org answered 502 on page 1; nothing was changed")
    out = nightly.advance_predictions(conn, run, down, lambda: (FakeIdentifier(), object()),
                                      FakeFetcher({}), 300, **QUIET)
    assert "502" in out["error"]
    out = nightly.advance_predictions(conn, run, lambda: ([], "g"),
                                      lambda: (_ for _ in ()).throw(RuntimeError("no index")),
                                      FakeFetcher({}), 300, **QUIET)
    assert "no index" in out["error"]


# --- on the server -----------------------------------------------------------------------

def server(box, layer, rows, pending_rows, fetcher, predict=300):
    client = TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name), limits=open_limits(), background=False,
        layer=layer, nightly_fetch=lambda: (rows, "g"),
        nightly_fetch_pending=lambda: (pending_rows, "g"),
        nightly_steps=FakeOutside({300: 130}).steps(), photo_fetcher=lambda: fetcher,
        nightly_settings=nightly.Settings(predict=predict), preload=["m1/nearest"],
        note=lambda s: None))
    for _ in range(200):
        if client.get("/api/health").json()["preload"]["state"] != "loading":
            break
        time.sleep(0.05)
    return client


def test_the_server_predicts_pending_records_after_the_night_with_its_default_model(box):
    layer = nightly.prepare(box, **QUIET)
    fetcher = FakeFetcher({"400": inat_obs(400, photos=[(0, 4000, "cc0", "x")])})
    client = server(box, layer, seed_rows(), [pending(400)], fetcher)
    report = client.app.state.run_nightly()
    assert report["predictions"] == {"model": "m1/nearest", "pending": 1, "asked": 1,
                                     "predicted": 1}
    assert client.get("/api/health").json()["nightly"]["changed"]["predicted"] == 1


def test_predictions_can_be_turned_off(box):
    layer = nightly.prepare(box, **QUIET)
    client = server(box, layer, seed_rows(), [pending(400)], FakeFetcher({}), predict=0)
    assert "predictions" not in client.app.state.run_nightly()


def test_a_prediction_made_before_the_record_turned_green_is_scored(box):
    layer = nightly.prepare(box, **QUIET)
    fetcher = FakeFetcher({"400": inat_obs(400, photos=[(0, 4000, "cc0", "x")])})
    server(box, layer, seed_rows(), [pending(400)], fetcher).app.state.run_nightly()
    conn = manifest.connect(layer.manifest)       # nights are a day apart, not a second
    with conn:
        conn.execute("update predictions set predicted_at = '2026-10-07T07:05:00+00:00'")
    conn.close()
    # The next night: 400 has its DNA answer (the fake photo is red, like "A x").
    night(layer, seed_rows() + [row(400, "A x")], FakeOutside({400: 200}))
    conn = manifest.connect(layer.manifest)
    [scored] = prospective.report(conn)
    conn.close()
    assert (scored["backbone"], scored["method"], scored["resolved"]) == ("m1", "nearest", 1)
    assert scored["species_top1"] == 1.0
