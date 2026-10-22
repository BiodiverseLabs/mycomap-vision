"""How it works' "How good is it" (howgood.py, GET /api/how-good): the served model's newest
open run, cut to what the page draws, and never a number from the paper's sealed test."""
import json

from test_api import app_with_model
from test_signin import gated_site_with_pages, sign_in

from mycomap_vision import evaluate, howgood

BANDS = ["0", "1-4", "5-19", "20-99", "100+"]


def scores(n, top1, top3, top5, top10):
    return {"n": n, "top1": top1, "top3": top3, "top5": top5, "top10": top10}


def report(standard=True):
    """A run report shaped like the box's run 169 (numbers rounded)."""
    rep = {"backbone": "m1", "method": "nearest",
           "all_photos": {"species": {"all": {"n": 12798, "top1": 0.5917, "top5": 0.8015}},
                          "groups": {"project": [], "observer": [{"name": "someone"}]}},
           "first_photo_only": {"species": {"all": {"n": 12798, "top1": 0.5309, "top5": 0.76}}}}
    if standard:
        rep["all_photos"]["standard"] = {
            "k": [1, 3, 5, 10], "bands": BANDS,
            "species": {"all": scores(12798, 0.592, 0.753, 0.802, 0.851),
                        "100+": scores(2158, 0.773, 0.912, 0.946, 0.967),
                        "0": scores(649, 0.0, 0.0, 0.0, 0.0),
                        "20-99": scores(5736, 0.73, 0.884, 0.921, 0.95),
                        "1-4": scores(1391, 0.199, 0.388, 0.463, 0.595),
                        "5-19": scores(2864, 0.502, 0.717, 0.799, 0.884)},
            "genus": {"all": scores(12865, 0.858, 0.939, 0.957, 0.973)},
            "family": {"all": scores(12759, 0.906, 0.963, 0.976, 0.985)}}
    return rep


def add_run(conn, backbone="m1", method="nearest", cutoff="2026-08-17", test_days=28,
            comparison=None, rep=None, species=0.592, release=None):
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    rep = dict(rep if rep is not None else report())
    if release:
        rep["dataset_release"] = {"dataset_release": release, "draft": False}
    cur = conn.execute(
        "insert into eval_runs (comparison_id, backbone, method, cutoff, test_days, n_reference, "
        "n_test, record_set, species_top1, genus_top1, family_top1, species_top1_first_photo, "
        "report_json, code_version, created_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (comparison or f"c-{cutoff}", backbone, method, cutoff, test_days, 155648, 12874, "abc",
         species, 0.858, 0.906, 0.531, json.dumps(rep), "x", "2026-10-19T00:00:00+00:00"))
    conn.commit()
    return cur.lastrowid


def test_the_page_gets_top_1_3_5_10_depth_and_first_photo_of_the_run_it_shows(conn):
    run = add_run(conn)
    out = howgood.how_good(conn, ["m1"], "nearest")
    assert out["run"] == {"id": run, "backbone": "m1", "method": "nearest",
                          "cutoff": "2026-08-17", "test_days": 28, "n_test": 12874,
                          "n_reference": 155648}
    assert out["top1"] == {"species": 0.592, "genus": 0.858, "family": 0.906}
    assert out["top_k"]["k"] == [1, 3, 5, 10]
    assert out["top_k"]["species"] == scores(12798, 0.592, 0.753, 0.802, 0.851)
    assert out["top_k"]["genus"]["top10"] == 0.973
    # Species by the true species' reference records, in the standard bands' order.
    assert [b["band"] for b in out["species_by_depth"]] == BANDS
    assert out["species_by_depth"][1] == {"band": "1-4", **scores(1391, 0.199, 0.388, 0.463, 0.595)}
    # The same records from their first photo only, beside all their photos.
    assert out["first_photo"] == {"n": 12798, "first_photo_top1": 0.5309,
                                  "all_photos_n": 12798, "all_photos_top1": 0.5917}
    assert out["inat"] is None
    # Nothing of the report beyond what the page draws: no observer names, no groups.
    assert "someone" not in json.dumps(out)


def test_a_run_scored_before_the_standard_summary_shows_no_made_up_ladder(conn):
    add_run(conn, rep=report(standard=False))
    out = howgood.how_good(conn, ["m1"], "nearest")
    assert out["top_k"] is None and out["species_by_depth"] is None
    assert out["top1"]["species"] == 0.592
    assert out["first_photo"]["first_photo_top1"] == 0.5309


def test_a_run_without_a_first_photo_answer_has_no_first_photo_comparison(conn):
    rep = report()
    rep["first_photo_only"] = {}
    add_run(conn, rep=rep)
    assert howgood.how_good(conn, ["m1"], "nearest")["first_photo"] is None


def test_the_run_is_the_newest_of_a_served_model_with_the_served_method(conn):
    add_run(conn, comparison="c1")
    newer = add_run(conn, comparison="c2")
    add_run(conn, comparison="c3", backbone="m2")              # not served
    add_run(conn, comparison="c4", method="species-mean")      # not the served method
    assert howgood.how_good(conn, ["m1"], "nearest")["run"]["id"] == newer
    assert howgood.how_good(conn, ["m9"], "nearest") == {"run": None}


def test_no_sealed_test_number_is_ever_shown(conn):
    shown = add_run(conn, comparison="c1", release="v1", test_days=0)    # v1's validation split
    add_run(conn, comparison="c2", cutoff="2026-10-12", release="v1")    # the sealed test
    add_run(conn, comparison="c3", cutoff="2026-09-20", test_days=28)    # live, reaches the seal
    add_run(conn, comparison="c4", cutoff="2026-09-01", test_days=0)     # every record after
    assert howgood.how_good(conn, ["m1"], "nearest")["run"]["id"] == shown
    add_run(conn, comparison="c0", cutoff="2026-10-20", backbone="m0")
    assert howgood.pick(evaluate.scoreboard(conn), ["m0"], "nearest") == (None, None)


def test_inat_comes_from_the_same_records_with_its_location_when_it_had_one(conn):
    add_run(conn, comparison="c2", backbone=howgood.INAT, method="vision-max", species=0.5)   # listed first
    add_run(conn, comparison="c1", backbone=howgood.INAT, method="combined-max", species=0.9)
    add_run(conn, comparison="c2")
    assert howgood.how_good(conn, ["m1"], "nearest")["inat"]["method"] == "vision-max"
    add_run(conn, comparison="c2", backbone=howgood.INAT, method="combined-max", species=0.45)
    inat = howgood.how_good(conn, ["m1"], "nearest")["inat"]
    assert inat == {"method": "combined-max", "n_test": 12874, "species_top1": 0.45,
                    "genus_top1": 0.858, "family_top1": 0.906}


def test_the_route_answers_for_the_served_model_and_method(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    assert client.get("/api/how-good").json() == {"run": None}
    method = client.get("/api/models").json()["default_method"]
    run = add_run(conn, method=method)
    out = client.get("/api/how-good").json()
    assert out["run"]["id"] == run and out["top_k"]["species"]["top3"] == 0.753


def test_with_the_site_signed_in_the_page_data_answers_members_and_not_strangers(conn, tmp_path):
    """How it works stays behind sign-in (MV_SIGNIN=all): its new data answers a signed-in
    member, and nobody signed out, like the page."""
    c = gated_site_with_pages(conn, tmp_path)
    add_run(conn)
    for path in ("/api/how-good", "/api/speed/daily"):
        assert c.get(path).status_code == 401, path
    sign_in(c)
    assert c.get("/api/how-good").status_code == 200
    assert c.get("/api/speed/daily").json() == {"days": []}
