"""Each record's PUBLIC point, frozen into a release's private part (Steve, 2026-10-12, D1 of
docs/atlas-environment.md): what iNat or MO itself shows, never .org's true point; hidden
GPS stays hidden; never in release.sqlite, a trainer copy or the public variant; a frozen
release refuses an included record whose public point was never read."""

import gzip
import json

import pytest
from test_dataset_release import LAT, LON, build, world  # noqa: F401  (the fixture)

from mycomap_vision import dataset_release as dr
from mycomap_vision import mopoints
from mycomap_vision.mo import Pacer

FREEZE = "Steve 2026-10-20 (test)"


def inat(**kw):
    return {"latitude": 45.0, "longitude": -122.0, "obscured": 0, "geoprivacy": None,
            "taxon_geoprivacy": None, "positional_accuracy": 12, "fetched_at": "t", **kw}


def mo(**kw):
    return {"latitude": None, "longitude": None, "gps_hidden": 0, "box_north": 45.2,
            "box_south": 45.0, "box_east": -122.0, "box_west": -122.4, "fetched_at": "t", **kw}


def test_an_inat_record_keeps_the_point_the_anonymous_api_showed():
    p = dr.public_place("1", "inat", inat(), None)
    assert (p["public_lat_micro"], p["public_lon_micro"]) == (45_000_000, -122_000_000)
    assert p["public_kind"] == "inat_open" and p["positional_accuracy_m"] == 12
    assert p["source"] == "inat" and p["read_at"] == "t"
    for flag in ({"obscured": 1}, {"geoprivacy": "obscured"}, {"taxon_geoprivacy": "obscured"}):
        assert dr.public_place("1", "inat", inat(**flag), None)["public_kind"] == "inat_obscured"


def test_a_private_inat_record_has_no_public_point_and_an_unread_one_has_no_row():
    p = dr.public_place("1", "inat", inat(latitude=None, longitude=None, geoprivacy="private"),
                        None)
    assert p["public_kind"] == "none" and p["public_lat_micro"] is None
    assert dr.public_place("1", "inat", None, None) is None


def test_an_mo_record_uses_its_shown_gps_else_its_box_centre():
    gps = dr.public_place("mo:4", "mo", None, mo(latitude=45.1, longitude=-122.2))
    assert gps["public_kind"] == "mo_gps" and gps["public_lat_micro"] == 45_100_000
    assert gps["box_n_micro"] is None
    box = dr.public_place("mo:4", "mo", None, mo())
    assert box["public_kind"] == "mo_box_centre"
    assert (box["public_lat_micro"], box["public_lon_micro"]) == (45_100_000, -122_200_000)
    assert 15_000 < box["positional_accuracy_m"] < 20_000          # half the box's diagonal
    assert box["box_n_micro"] == 45_200_000 and box["box_w_micro"] == -122_400_000
    none = dr.public_place("mo:4", "mo", None, mo(box_north=None))
    assert none["public_kind"] == "none"
    assert dr.public_place("mo:4", "mo", None, None) is None


def test_a_hidden_mo_gps_is_never_kept_even_if_it_was_answered():
    p = dr.public_place("mo:4", "mo", None, mo(latitude=45.1, longitude=-122.2, gps_hidden=1))
    assert p["public_kind"] == "mo_box_centre" and p["public_lat_micro"] == 45_100_000
    parsed = mopoints.parse({"id": 4, "latitude": "45.1", "longitude": "-122.2",
                             "gps_hidden": True, "location": {}})
    assert parsed["latitude"] is None and parsed["longitude"] is None


def test_a_box_across_the_antimeridian_is_centred_on_it():
    p = dr.public_place("mo:4", "mo", None, mo(box_east=-179.0, box_west=179.0))
    assert p["public_lon_micro"] in (180_000_000, -180_000_000)


class FakeSession:
    def __init__(self, answers):
        self.answers, self.asked = answers, []
        self.headers = {}

    def get(self, url, params, timeout):
        self.asked.append(params)
        body = {"results": [self.answers[int(i)] for i in params["id"].split(",")
                            if int(i) in self.answers],
                "number_of_pages": 1, "run_time": 0.1}
        return type("R", (), {"status_code": 200, "json": lambda self: body, "text": ""})()


def test_fetching_mo_points_stores_what_mo_shows_and_caches_the_answers(world, tmp_path):
    conn = world["conn"]
    conn.execute("delete from mo_public_points")
    answers = {4: {"id": 4, "latitude": None, "longitude": None, "gps_hidden": False,
                   "location": {"latitude_north": "45.2", "latitude_south": "45.0",
                                "longitude_east": "-122.0", "longitude_west": "-122.4"}}}
    session = FakeSession(answers)
    out = mopoints.fetch_all(conn, raw_dir=tmp_path / "raw", session=session,
                             pacer=Pacer(interval=0), log=lambda s: None)
    assert out["ok"] == 1 and out["missing"] == 2          # mo:15 and mo:16 unanswered
    row = conn.execute("select box_north, gps_hidden, status from mo_public_points "
                       "where observation_id = 'mo:4'").fetchone()
    assert tuple(row) == (45.2, 0, "ok")
    assert all(p["detail"] == "high" and "api_key" not in p for p in session.asked)
    cached = list((tmp_path / "raw").glob("observations-*.json.gz"))
    assert cached and json.load(gzip.open(cached[0], "rt"))[0]["id"] == 4
    assert mopoints.fetch_all(conn, raw_dir=tmp_path / "raw", session=session,
                              pacer=Pacer(interval=0), log=lambda s: None)["requested"] == 0


def test_the_release_freezes_public_points_privately_beside_the_true_point(world):
    conn = world["conn"]
    conn.execute("update inat_observations set inat_latitude = 45.31, inat_longitude = -122.61, "
                 "obscured = 1 where observation_id = '1'")
    conn.commit()
    m = build(world, name="v1", freeze_approved=FREEZE, no_loo_exclusions=FREEZE)
    rel = dr.load_release("v1", world["root"])
    try:
        pub = rel.public_places()
        assert pub["1"]["public_kind"] == "inat_obscured"
        assert pub["1"]["public_lat_micro"] == 45_310_000
        true = rel.private.execute("select lat_micro from record_places where record_key = '1'"
                                   ).fetchone()[0]
        assert true == dr.micro(LAT) != pub["1"]["public_lat_micro"]      # never .org's point
        assert pub["mo:4"]["public_kind"] == "mo_box_centre"
        assert pub["2"]["public_kind"] == "none"                          # no iNat point here
        tables = {r[0] for r in rel.db.execute("select name from sqlite_master")}
        assert "record_public_places" not in tables                      # private part only
    finally:
        rel.close()
    assert m["counts"]["public_places_by_kind"]["inat_obscured"] == 1
    # The public variant is built from release.sqlite: no public-point table, no µ° values.
    dr.public_variant("v1", world["root"])
    cc = world["root"] / "v1-cc"
    import sqlite3
    priv = sqlite3.connect(f"file:{(cc / dr.PRIVATE_DB).as_posix()}?mode=ro", uri=True)
    assert priv.execute("select count(*) from record_public_places").fetchone()[0] == 0
    assert priv.execute("select count(*) from record_places").fetchone()[0] == 0
    priv.close()
    assert b"record_public_places" not in (cc / dr.RELEASE_DB).read_bytes()


def test_a_frozen_release_refuses_a_record_whose_public_point_was_never_read(world):
    world["conn"].execute("delete from mo_public_points")
    world["conn"].commit()
    with pytest.raises(dr.ReleaseError, match="no public point read"):
        build(world, name="v1", freeze_approved=FREEZE, no_loo_exclusions=FREEZE)
    m = build(world)                                      # a draft says so and goes on
    assert m["public_places_missing"] == 1


def test_a_trainer_copy_cannot_read_the_public_points(world, tmp_path):
    import shutil
    build(world, name="v1", freeze_approved=FREEZE, no_loo_exclusions=FREEZE)
    copy = tmp_path / "copy"
    shutil.copytree(world["root"] / "v1", copy / "v1")
    (copy / "v1" / dr.PRIVATE_DB).chmod(0o600)
    (copy / "v1" / dr.PRIVATE_DB).unlink()
    rel = dr.load_release("v1", copy, without_private=True)
    with pytest.raises(dr.ReleaseError, match="without private.sqlite"):
        rel.public_places()
    rel.close()
