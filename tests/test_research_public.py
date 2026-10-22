"""The Research section is open to anyone (Steve, 2026-10-22): its pages and the data they
load answer without sign-in, cut to what may be public (research_public.py); the paper
draft and the experiment registry stay members only."""

import json
import re

from test_benchmark_io import report as benchmark_report
from test_signin import gated_site_with_pages, sign_in

from mycomap_vision import benchmark_io, evaluate, research_public
from mycomap_vision.signin import SigninConfig

RESEARCH_PAGES = ("/research", "/research/results", "/research/benchmarks",
                  "/research/benchmarks/sealed-test", "/research/protocols", "/research/data",
                  "/research/experiments", "/research/experiments/some-entry",
                  "/research/for-researchers", "/research/api", "/models", "/data")
TWINS = "C:\\Users\\someone\\Projects\\mv-retrain-data\\record-scoring\\twins-train-b7bac93.tsv"
ANY_PATH = re.compile(r"[A-Za-z]:\\\\|\\\\\\\\|/(?:Users|home|srv|var|tmp)/|someone")


def add_run(conn, cutoff="2026-08-17", test_days=0, release="v1") -> int:
    """A scoreboard run shaped like the box's runs 168/169: a twin table named by its
    laptop path, and results by project and by observer."""
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    groups = {"project": [{"name": "Run155", "n": 10, "species_top1": 0.6, "genus_top1": 0.9}],
              "observer": [{"name": "someones-login", "n": 7, "species_top1": 0.7,
                            "genus_top1": 0.9}]}
    rep = {"backbone": "m1", "method": "nearest",
           "all_photos": {"species": {"all": {"n": 10, "top1": 0.6, "top5": 0.8}},
                          "groups": groups},
           "first_photo_only": {"species": {"all": {"n": 10, "top1": 0.5, "top5": 0.7}},
                                "groups": groups},
           "twins": {"file": TWINS, "records": 4190},
           "answer_taxonomy": {"twins": {"file": TWINS},
                               "taxa": {"file": "occurrence/inat-open-data/taxa.csv.gz"}},
           "note": f"scored with {TWINS} on the laptop"}
    if release:
        rep["dataset_release"] = {"dataset_release": release, "draft": False}
    cur = conn.execute(
        "insert into eval_runs (comparison_id, backbone, method, cutoff, test_days, n_reference, "
        "n_test, record_set, species_top1, genus_top1, family_top1, species_top1_first_photo, "
        "report_json, code_version, created_at) values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (f"c-{cutoff}", "m1", "nearest", cutoff, test_days, 100, 10, "abc", 0.6, 0.9, 0.95, 0.5,
         json.dumps(rep), "x", "2026-10-19T00:00:00+00:00"))
    conn.commit()
    return cur.lastrowid


def test_a_signed_out_visitor_reads_every_research_page_and_the_data_they_load(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    run = add_run(conn)
    for path in RESEARCH_PAGES:
        res = c.get(path)
        assert res.status_code == 200, path
        assert "<title>site</title>" in res.text, path
    for path in ("/api/stats", "/api/models", "/api/scoreboard", f"/api/scoreboard/{run}",
                 "/api/prospective", "/api/benchmarks"):
        assert c.get(path).status_code == 200, path
    assert c.get("/api/scoreboard/99999").status_code == 404


def test_the_paper_draft_and_the_experiment_registry_still_need_sign_in(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    for path in ("/api/paper", "/api/experiments", "/api/experiments/some-entry"):
        assert c.get(path).status_code == 401, path
    # Opening the Research data reads only: nothing is written there signed out.
    assert c.post("/api/scoreboard").status_code == 401
    assert c.post("/api/benchmarks").status_code == 401


def test_in_every_sign_in_mode_the_paper_is_members_only_and_the_research_data_open():
    # The paper draft is members only in every mode; the Research data is open in every mode.
    for mode in ("identify", "all"):
        cfg = SigninConfig(mode=mode, origin="http://x", issuer="https://mycomap.org",
                           public_key=object(), secret=b"s" * 48)
        assert cfg.needs_session("GET", "/api/paper")
        assert cfg.needs_session("GET", "/api/experiments/x")
        assert not cfg.needs_session("GET", "/api/scoreboard/12")
        assert not cfg.needs_session("HEAD", "/research/results")
        assert cfg.needs_session("GET", "/api/scoreboard/12/x") == (mode == "all")
        assert cfg.needs_session("DELETE", "/api/scoreboard/12") == (mode == "all")


def test_no_research_answer_carries_a_file_path_and_signed_out_none_names_an_observer(conn,
                                                                                     tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    run = add_run(conn)
    signed_out = c.get(f"/api/scoreboard/{run}")
    assert not ANY_PATH.search(signed_out.text), signed_out.text
    body = signed_out.json()
    assert body["twins"]["file"] == "twins-train-b7bac93.tsv"
    assert body["answer_taxonomy"]["taxa"]["file"] == "occurrence/inat-open-data/taxa.csv.gz"
    assert body["note"] == "scored with twins-train-b7bac93.tsv on the laptop"
    for part in ("all_photos", "first_photo_only"):
        assert list(body[part]["groups"]) == ["project"]
    for path in ("/api/scoreboard", "/api/stats", "/api/models", "/api/prospective",
                 "/api/benchmarks"):
        assert not ANY_PATH.search(c.get(path).text), path
    # A signed-in member still sees the breakdown by observer, never the path.
    sign_in(c)
    member = c.get(f"/api/scoreboard/{run}")
    assert member.json()["all_photos"]["groups"]["observer"][0]["name"] == "someones-login"
    assert not ANY_PATH.search(member.text.replace("someones-login", ""))


def test_photographer_permission_counts_are_in_no_answer(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    conn.execute("insert into photo_permissions (inat_user_id, status) values (7, 'granted')")
    conn.execute("insert into photo_permissions (inat_user_id, status) values (8, 'withdrawn')")
    conn.execute("insert into permission_syncs (attempted_at, ok, error) values "
                 "('2026-10-22T00:00:00+00:00', 0, 'HTTPSConnectionPool(host=private.example)')")
    conn.commit()
    for who in ("signed out", "member"):
        if who == "member":
            sign_in(c)
        p = c.get("/api/stats").json()["permissions"]
        assert set(p) == {"last_good_sync", "fresh_for_showing", "last_attempt"}, who
        assert p["last_attempt"] == {"attempted_at": "2026-10-22T00:00:00+00:00", "ok": False}


def test_nothing_from_the_sealed_test_reaches_a_signed_out_visitor(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    val = add_run(conn)
    sealed = add_run(conn, cutoff="2026-10-12", release="v1")
    live_reaching = add_run(conn, cutoff="2026-09-20", test_days=28, release=None)
    benchmark_io.import_summary(conn, benchmark_io.export_summary(benchmark_report("val"), "a"))
    test = {**benchmark_report("test"), "sealed": True}
    benchmark_io.import_summary(conn, benchmark_io.export_summary(test, "a"))
    conn.commit()
    ids = [r["id"] for r in c.get("/api/scoreboard").json()["runs"]]
    assert ids == [val]
    for run in (sealed, live_reaching):
        assert c.get(f"/api/scoreboard/{run}").status_code == 404
    assert [b["split"] for b in c.get("/api/benchmarks").json()["benchmarks"]] == ["val"]
    assert all("sealed" not in m for m in c.get("/api/prospective").json()["models"])
    sign_in(c)            # members see them, as before
    assert sorted(r["id"] for r in c.get("/api/scoreboard").json()["runs"]) == \
        sorted([val, sealed, live_reaching])
    assert c.get(f"/api/scoreboard/{sealed}").status_code == 200
    assert len(c.get("/api/benchmarks").json()["benchmarks"]) == 2


def test_a_run_is_sealed_when_its_test_can_reach_past_the_snapshot():
    sealed = research_public.sealed_run
    assert not sealed({"cutoff": "2026-08-17", "test_days": 0, "dataset_release": "v1"})
    assert sealed({"cutoff": "2026-10-12", "test_days": 28, "dataset_release": "v1"})
    assert not sealed({"cutoff": "2026-09-07", "test_days": 28, "dataset_release": None})
    assert sealed({"cutoff": "2026-09-20", "test_days": 28, "dataset_release": None})
    assert sealed({"cutoff": "2026-09-07", "test_days": 0, "dataset_release": None})


def test_file_paths_are_cut_to_their_name_and_nothing_else_changes():
    strip = research_public.strip_paths
    assert strip("C:\\Users\\a\\b\\t.tsv") == "t.tsv"
    assert strip("C:/Users/a/b/t.tsv") == "t.tsv"
    assert strip("/home/ubuntu/data/manifest.sqlite") == "manifest.sqlite"
    assert strip("\\\\host\\share\\x.json") == "x.json"
    for kept in ("https://vision.mycomap.org/research/results", "occurrence/x/taxa.csv.gz",
                 "nearest+mean-records-balanced+place", "public (record_public_places)",
                 "1-4 refs", "names iNat knows", "2026-10-19T21:57:19+00:00"):
        assert strip(kept) == kept
    assert strip({"a": [{"file": "/srv/x/y.tsv"}], "n": 3}) == {"a": [{"file": "y.tsv"}], "n": 3}
