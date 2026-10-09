import json

import numpy as np
import pytest

from mycomap_vision import occtune
from mycomap_vision.occprior import (OccParams, OccurrencePrior, TuningRefused,
                                     check_store_excludes, load_params)
from mycomap_vision.occtune import (ScoredSet, check_not_sealed, grid_search, load_scored_set,
                                    save_scored_set, tune)
from mycomap_vision.occurrence import OccurrenceStore, loo_path
from occurrence_fixtures import EAST, WEST, build_store, obs, standard_observations

SPECIES = ["Amanita", "Amanita occidentalis", "Amanita orientalis", "Amanita rara",
           "Westia pacifica"]
SMALL_GRID = {"radius_km": [1500.0], "out_of_range_penalty": [0.0, 2.0, 6.0],
              "density_weight": [0.0, 0.5], "season_weight": [0.0, 0.5],
              "photo_temperature": [0.02]}


def validation_set() -> ScoredSet:
    """100 records. In the east, Amanita orientalis whose photos look a touch more like
    Westia pacifica, which lives only 3,400 km west: photo-only gets them wrong. In the
    west, Westia and occidentalis that the photos get right, and Amanita rara that
    looks like Westia (both live there, so place can't help)."""
    rows = []

    def add(n, truth, lat, lon, date, best, second, margin=0.01):
        for _ in range(n):
            s = np.full(len(SPECIES), 0.40, dtype=np.float32)
            s[SPECIES.index(best)] = 0.80
            s[SPECIES.index(second)] = 0.80 - margin
            rows.append((truth, lat, lon, date, s))
    add(40, "Amanita orientalis", *EAST, "2026-10-01", "Westia pacifica", "Amanita orientalis")
    add(25, "Westia pacifica", *WEST, "2026-05-01", "Westia pacifica", "Amanita rara", 0.03)
    add(20, "Amanita occidentalis", *WEST, "2026-04-15", "Amanita occidentalis",
        "Amanita rara", 0.03)
    add(15, "Amanita rara", *WEST, "2026-06-01", "Westia pacifica", "Amanita rara", 0.03)
    ids = [f"{1000 + i}" for i in range(len(rows))]
    return ScoredSet(ids, SPECIES, np.stack([r[4] for r in rows]), "similarity",
                     truth=[r[0] for r in rows], truth_genus=[r[0].split()[0] for r in rows],
                     latitude=[r[1] for r in rows], longitude=[r[2] for r in rows],
                     observed_on=[r[3] for r in rows], uuids=[f"val-{i}" for i in ids],
                     group_genus=[s.split()[0] for s in SPECIES],
                     group_is_species=[len(s.split()) > 1 for s in SPECIES])


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    s = validation_set()
    return build_store(tmp_path_factory.mktemp("occ"), exclude=s.uuids)[0]


def prior_for(store, s):
    p = OccurrencePrior(store, OccParams())
    p.fit([], s.species)
    return p


def test_tuning_chooses_a_penalty_that_puts_far_away_lookalikes_down(store):
    s = validation_set()
    only_penalty = {**SMALL_GRID, "density_weight": [0.0], "season_weight": [0.0]}
    res = grid_search(prior_for(store, s), s, only_penalty, log=lambda *a: None)
    assert res["photo_only"]["species_top1"] == 0.45
    assert res["best"]["species_top1"] == 0.85
    assert res["best"]["out_of_range_penalty"] == 2.0     # the gentlest that does it


def test_ties_go_to_the_gentler_setting(store):
    # Density or season alone also puts the far lookalike down here (it lives elsewhere
    # and fruits in another season): then no penalty, and only one gentle term, is chosen.
    s = validation_set()
    res = grid_search(prior_for(store, s), s, SMALL_GRID, log=lambda *a: None)
    best = res["best"]
    assert best["species_top1"] == 0.85
    assert best["out_of_range_penalty"] == 0.0
    assert best["density_weight"] + best["season_weight"] == 0.5


def test_the_tuned_confidence_is_graded_not_saturated(store):
    s = validation_set()
    cal = grid_search(prior_for(store, s), s, SMALL_GRID, log=lambda *a: None)["calibration"]
    assert cal["species"]["n"] == 100
    # 15 of 100 are wrong whatever the setting: a fitted temperature can't call
    # everything certain, unlike the cosine default on log-probabilities.
    assert cal["species"]["share_stated_99"] < 0.5
    assert cal["species"]["temperature"] > 0.02


def test_tuning_refuses_records_in_a_sealed_set(tmp_path, conn):
    (tmp_path / "dev.csv").write_text("observation_id\n1000\n1001\n", encoding="utf-8")
    (tmp_path / "paper.csv").write_text("observation_id,lat\n5\n1001,1\n", encoding="utf-8")
    sealed = occtune.sealed_files(str(tmp_path / "dev.csv"), [tmp_path / "paper.csv"])
    with pytest.raises(TuningRefused, match="sealed set"):
        check_not_sealed(["1000", "1001"], sealed, conn)
    check_not_sealed(["1000"], sealed, conn)


def test_the_2026_10_08_test_csv_is_no_longer_sealed_by_itself(tmp_path):
    # Steve 2026-10-08: the held-out set's test.csv may be scored and checked.
    (tmp_path / "dev.csv").write_text("observation_id\n1000\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("observation_id\n1000\n", encoding="utf-8")
    assert occtune.sealed_files(str(tmp_path / "dev.csv"), []) == []


def test_tuning_refuses_registered_benchmark_holdouts(conn):
    # The manifest's own table (holdouts.py): every registered id is a sealed holdout.
    conn.execute("create table if not exists benchmark_holdouts (observation_id text not null, "
                 "benchmark text not null, added_at text not null, "
                 "primary key (observation_id, benchmark))")
    conn.executemany("insert into benchmark_holdouts values (?, 'paper', '2026-10-09')",
                     [("1001",), ("1002",)])
    check_not_sealed(["1000"], [], conn)                # an unregistered record is allowed
    for frozen in ("1001", "1002"):
        with pytest.raises(TuningRefused, match="frozen benchmark"):
            check_not_sealed(["1000", frozen], [], conn)


def test_a_holdout_table_with_a_split_column_lets_the_tuning_split_through(conn):
    conn.execute("drop table if exists benchmark_holdouts")
    conn.execute("create table benchmark_holdouts (observation_id text, split text)")
    conn.executemany("insert into benchmark_holdouts values (?, ?)",
                     [("1000", "dev"), ("1001", "test"), ("1002", None)])
    check_not_sealed(["1000"], [], conn)
    for frozen in ("1001", "1002"):
        with pytest.raises(TuningRefused, match="frozen benchmark"):
            check_not_sealed(["1000", frozen], [], conn)


def test_a_holdout_table_that_cannot_be_checked_refuses_everything(conn):
    conn.execute("drop table if exists benchmark_holdouts")
    conn.execute("create table benchmark_holdouts (inat text)")
    with pytest.raises(TuningRefused, match="no observation_id"):
        check_not_sealed(["1000"], [], conn)


def test_tuning_refuses_a_store_that_would_count_records_towards_themselves(tmp_path, store):
    counts_them = build_store(tmp_path)[0]              # built without the exclusions
    assert check_store_excludes(counts_them, ["val-1000", "val-1001"])[
        "taken_out_at_scoring"] == 2                    # its leave-one-out index takes them out
    loo_path(counts_them.path).unlink()                 # without the index: refused
    counts_them = OccurrenceStore.load(counts_them.path)
    with pytest.raises(TuningRefused, match="counts 2 of the validation"):
        check_store_excludes(counts_them, ["val-1000", "val-1001"])
    assert check_store_excludes(store, ["val-1000", "val-1001"])["excluded_at_build"] == 2
    with pytest.raises(TuningRefused, match="no known iNat uuid"):
        check_store_excludes(store, ["val-1000", None])
    assert check_store_excludes(store, ["val-1000", None], allow_missing=True)["missing_uuid"] == 1


def test_each_validation_record_is_scored_without_its_own_observation(tmp_path, store):
    # A store that counts the validation records themselves, at their own spots, must
    # give the same tuning result as one built without them.
    s = validation_set()
    taxon = {"Amanita orientalis": 104, "Westia pacifica": 132, "Amanita occidentalis": 105,
             "Amanita rara": 106}
    rows = standard_observations() + [
        obs(u, taxon[t], lat, lon, date) for u, t, lat, lon, date
        in zip(s.uuids, s.truth, s.latitude, s.longitude, s.observed_on)]
    counts_them = build_store(tmp_path, rows)[0]
    grid = {**SMALL_GRID, "density_weight": [0.0, 0.5, 1.0], "season_weight": [0.0, 0.5]}
    with_loo = grid_search(prior_for(counts_them, s), s, grid, log=lambda *a: None)
    clean = grid_search(prior_for(store, s), s, grid, log=lambda *a: None)
    assert [t["species_right"] for t in with_loo["table"]] == \
        [t["species_right"] for t in clean["table"]]
    assert with_loo["best"] == clean["best"]


def test_scores_round_trip_through_their_file(tmp_path):
    s = validation_set()
    save_scored_set(tmp_path / "s.npz", s)
    back = load_scored_set(tmp_path / "s.npz")
    assert back.observation_ids == s.observation_ids and back.species == s.species
    assert np.allclose(back.scores, s.scores) and back.truth == s.truth
    assert back.latitude == s.latitude and back.uuids == s.uuids


def test_tuning_on_a_dev_csv_saves_the_chosen_values_and_where_they_came_from(tmp_path, conn,
                                                                              store):
    s = validation_set()
    save_scored_set(tmp_path / "scores.npz", s)
    (tmp_path / "dev.csv").write_text(
        "observation_id\n" + "\n".join(s.observation_ids) + "\n", encoding="utf-8")
    (tmp_path / "test.csv").write_text("observation_id\n99\n", encoding="utf-8")
    out = tmp_path / "params.json"
    res = tune(conn, str(tmp_path / "dev.csv"), tmp_path / "scores.npz", store_path=store.path,
               out=out, grid=SMALL_GRID, log=lambda *a: None)
    assert res["best"]["species_top1"] == 0.85
    saved = json.loads(out.read_text(encoding="utf-8"))
    for k in ("radius_km", "out_of_range_penalty", "density_weight", "season_weight"):
        assert saved[k] == res["best"][k]
    assert set(saved["confidence_temperature"]) == {"species", "genus", "family"}
    prov = saved["provenance"]
    assert prov["records_csv"].endswith("dev.csv") and prov["n_records"] == 100
    assert prov["sealed_checked"] == []
    assert prov["store_exclusion_check"]["excluded_at_build"] == 100
    assert prov["photo_only"]["species_top1"] == 0.45
    assert load_params(out).season_weight == res["best"]["season_weight"]


def test_a_records_csv_without_scores_is_refused(tmp_path, conn):
    (tmp_path / "dev.csv").write_text("observation_id\n1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="needs --scores"):
        tune(conn, str(tmp_path / "dev.csv"), None, log=lambda *a: None)


def test_a_saved_comparison_gives_its_test_records_as_the_validation_set(conn, tmp_path):
    from mycomap_vision import evaluate
    from mycomap_vision.embed import embed_photos, photos_to_embed
    from test_models_and_scoreboard import Const, seed_two_species
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    bb = Const("m1")
    embed_photos(conn, store, bb, photos_to_embed(conn, bb.name, "large", store.location,
                                                  "local"), root / bb.name, log=lambda s: None)
    cid = evaluate.compare(conn, ["m1"], ["nearest"], test_days=28, embeddings_root=root,
                           log=lambda s: None)["comparison_id"]
    s, ref = occtune.comparison_set(conn, cid, embeddings_root=root)
    assert sorted(s.observation_ids) == ["106", "107"]        # validated after the cutoff
    assert len(ref) == 6 and not set(s.observation_ids) & {r.observation_id for r in ref}
    assert sorted(s.truth) == ["A x", "B y"] and sorted(s.uuids) == ["uuid-106", "uuid-107"]
    assert s.scores.shape == (2, 2) and s.group_is_species == [True, True]


def test_results_are_also_reported_without_a_list_of_records(store):
    # e.g. repeat finds: the same taxon at the same spot as another record.
    s = validation_set()
    repeat = set(s.observation_ids[:40])            # the 40 eastern records
    res = grid_search(prior_for(store, s), s, SMALL_GRID, log=lambda *a: None, without=repeat)
    w = res["without"]
    assert w["left_out"] == 40 and w["photo_only"]["n"] == 60
    assert w["photo_only"]["species_top1"] == w["chosen"]["species_top1"] == 0.75
    assert res["photo_only"]["species_top1"] == 0.45 and res["best"]["species_top1"] == 0.85


def test_a_set_is_scored_with_the_saved_values_and_nothing_is_saved(tmp_path, conn, store):
    s = validation_set()
    save_scored_set(tmp_path / "scores.npz", s)
    (tmp_path / "test.csv").write_text(
        "observation_id\n" + "\n".join(s.observation_ids) + "\n", encoding="utf-8")
    out = tmp_path / "params.json"
    from mycomap_vision.occprior import save_params
    save_params(OccParams(out_of_range_penalty=0.0, density_weight=0.0, season_weight=0.0), out)
    before = out.read_text(encoding="utf-8")
    res = tune(conn, str(tmp_path / "test.csv"), tmp_path / "scores.npz", store_path=store.path,
               out=out, evaluate_only=True, log=lambda *a: None)
    assert res["best"]["species_top1"] == 0.45 == res["photo_only"]["species_top1"]
    assert out.read_text(encoding="utf-8") == before


def test_tuning_takes_each_record_s_own_observation_back_out(tmp_path):
    # The only Westia pacifica within 3,000 km of the east is the record's own find.
    store = build_store(tmp_path, standard_observations()
                        + [obs("val-lone", 132, *EAST, "2026-05-01")])[0]
    scores = np.full((1, len(SPECIES)), 0.4, dtype=np.float32)
    scores[0, SPECIES.index("Westia pacifica")] = 0.80
    scores[0, SPECIES.index("Amanita orientalis")] = 0.79

    def lone(uuid):
        return ScoredSet(["lone"], SPECIES, scores, truth=["Westia pacifica"],
                         truth_genus=["Westia"], latitude=[EAST[0]], longitude=[EAST[1]],
                         observed_on=["2026-05-01"], uuids=[uuid],
                         group_genus=[x.split()[0] for x in SPECIES],
                         group_is_species=[len(x.split()) > 1 for x in SPECIES])
    grid = {**SMALL_GRID, "out_of_range_penalty": [6.0], "density_weight": [0.0],
            "season_weight": [0.0]}
    taken_out = grid_search(prior_for(store, lone("val-lone")), lone("val-lone"), grid,
                            log=lambda *a: None)
    assert taken_out["best"]["species_right"] == 0       # out of range once its own find is gone
    counted = grid_search(prior_for(store, lone(None)), lone(None), grid, log=lambda *a: None)
    assert counted["best"]["species_right"] == 1         # what the leak would have shown


def test_the_report_shows_dev_with_and_without_the_genus_rule(store):
    s = validation_set()
    res = grid_search(prior_for(store, s), s, SMALL_GRID, log=lambda *a: None)
    assert res["genus_rule"]["with"]["genus_rule"] is True
    assert res["genus_rule"]["without"]["genus_rule"] is False
    assert res["best"]["genus_rule"] is True                 # kept unless told otherwise
    dropped = grid_search(prior_for(store, s), s, SMALL_GRID, log=lambda *a: None,
                          choose_genus_rule=False)
    assert dropped["best"]["genus_rule"] is False
    assert {row["genus_rule"] for row in res["table"]} == {True, False}


def test_the_genus_rule_counts_where_only_the_genus_is_far(store):
    # Westia inventa: not on iNat, no DNA records, the whole genus 3,400 km west.
    groups = SPECIES + ["Westia inventa"]
    scores = np.full((1, len(groups)), 0.4, dtype=np.float32)
    scores[0, groups.index("Westia inventa")] = 0.80
    scores[0, groups.index("Amanita orientalis")] = 0.79
    one = ScoredSet(["x"], groups, scores, truth=["Amanita orientalis"], truth_genus=["Amanita"],
                    latitude=[EAST[0]], longitude=[EAST[1]], observed_on=["2026-10-01"],
                    uuids=["val-x"], group_genus=[x.split()[0] for x in groups],
                    group_is_species=[True] * len(groups))
    p = OccurrencePrior(store, OccParams())
    p.fit([], groups)
    grid = {**SMALL_GRID, "out_of_range_penalty": [6.0], "density_weight": [0.0],
            "season_weight": [0.0]}
    res = grid_search(p, one, grid, log=lambda *a: None)
    assert res["genus_rule"]["with"]["species_right"] == 1
    assert res["genus_rule"]["without"]["species_right"] == 0


def test_dev_can_choose_outright_exclusion_over_soft_penalties(store):
    from mycomap_vision.occprior import EXCLUDE
    from mycomap_vision.occtune import DEFAULT_GRID
    assert EXCLUDE in DEFAULT_GRID["out_of_range_penalty"]
    assert OccParams().out_of_range_penalty == 6.0            # the untuned default
    # The far lookalike now looks much better (0.15 in cosine: 7.5 nats at 0.02).
    s = validation_set()
    east = [i for i, t in enumerate(s.truth) if t == "Amanita orientalis"]
    s.scores[east, SPECIES.index("Amanita orientalis")] = 0.65
    grid = {**SMALL_GRID, "out_of_range_penalty": [0.0, 2.0, 6.0, EXCLUDE],
            "density_weight": [0.0], "season_weight": [0.0]}
    res = grid_search(prior_for(store, s), s, grid, log=lambda *a: None)
    assert res["best"]["out_of_range_penalty"] == EXCLUDE and res["best"]["exclusion"]
    assert res["best"]["species_top1"] == 0.85



def write_benchmark(tmp_path, s):
    """dev.csv, test.csv and split.json as the held-out benchmark lays them out."""
    import hashlib
    dev_ids, test_ids = s.observation_ids[:60], s.observation_ids[60:]
    for name, ids in (("dev", dev_ids), ("test", test_ids)):
        (tmp_path / f"{name}.csv").write_text("observation_id\n" + "\n".join(ids) + "\n",
                                              encoding="utf-8")
    split = {name: {"records": len(ids), "sha256_ids": hashlib.sha256(
        "\n".join(sorted(ids)).encode()).hexdigest()}
        for name, ids in (("dev", dev_ids), ("test", test_ids))}
    (tmp_path / "split.json").write_text(json.dumps(split), encoding="utf-8")


def test_no_search_on_the_test_split_without_allow_test(tmp_path, conn, store):
    s = validation_set()
    save_scored_set(tmp_path / "scores.npz", s)
    write_benchmark(tmp_path, s)
    kw = dict(store_path=store.path, out=tmp_path / "params.json", grid=SMALL_GRID,
              log=lambda *a: None)
    with pytest.raises(TuningRefused, match="test split"):
        tune(conn, str(tmp_path / "test.csv"), tmp_path / "scores.npz", **kw)
    # a renamed copy of the test ids is still the test split, and so is part of it
    (tmp_path / "mine.csv").write_text((tmp_path / "test.csv").read_text(), encoding="utf-8")
    with pytest.raises(TuningRefused, match="test split"):
        tune(conn, str(tmp_path / "mine.csv"), tmp_path / "scores.npz", **kw)
    part = s.observation_ids[:5] + s.observation_ids[60:65]
    (tmp_path / "part.csv").write_text("observation_id\n" + "\n".join(part) + "\n",
                                       encoding="utf-8")
    with pytest.raises(TuningRefused, match="test split"):
        tune(conn, str(tmp_path / "part.csv"), tmp_path / "scores.npz", **kw)
    assert tune(conn, str(tmp_path / "dev.csv"), tmp_path / "scores.npz", **kw)["split"] == "dev"
    assert tune(conn, str(tmp_path / "test.csv"), tmp_path / "scores.npz", evaluate_only=True,
                **kw)["split"] == "test"
    assert tune(conn, str(tmp_path / "test.csv"), tmp_path / "scores.npz", allow_test=True,
                save=False, **kw)["split"] == "test"


def test_evaluate_only_reports_the_saved_temperature_not_one_fitted_on_the_set(tmp_path, conn,
                                                                              store):
    from mycomap_vision.occprior import save_params
    s = validation_set()
    save_scored_set(tmp_path / "scores.npz", s)
    (tmp_path / "set.csv").write_text("observation_id\n" + "\n".join(s.observation_ids) + "\n",
                                      encoding="utf-8")
    out = tmp_path / "params.json"
    save_params(OccParams(confidence_temperature={"species": 7.0, "genus": 9.0}), out)
    res = tune(conn, str(tmp_path / "set.csv"), tmp_path / "scores.npz", store_path=store.path,
               out=out, evaluate_only=True, log=lambda *a: None)
    cal = res["calibration"]
    assert cal["species"]["temperature"] == 7.0 and cal["genus"]["temperature"] == 9.0
    assert cal["species"]["fitted_here"] is False
    assert res["params"]["confidence_temperature"] == {"species": 7.0, "genus": 9.0}


def test_uuids_come_from_the_manifest_by_observation_id(tmp_path, conn):
    from mycomap_vision.occtune import fill_from_csv
    s = validation_set()
    s.uuids = []
    conn.execute("insert into inat_observations (observation_id, status, uuid, fetched_at) "
                 "values ('1000', 'ok', ' uu-1000 ', 'now')")
    (tmp_path / "set.csv").write_text("observation_id,uuid\n1000,\n1001,   \n", encoding="utf-8")
    filled = fill_from_csv(s, tmp_path / "set.csv", conn)
    assert filled.uuids == ["uu-1000", None]


def test_the_names_list_with_its_method_is_written_beside_the_params(tmp_path, conn, store):
    s = validation_set()
    save_scored_set(tmp_path / "scores.npz", s)
    (tmp_path / "set.csv").write_text("observation_id\n" + "\n".join(s.observation_ids) + "\n",
                                      encoding="utf-8")
    out = tmp_path / "params.json"
    tune(conn, str(tmp_path / "set.csv"), tmp_path / "scores.npz", store_path=store.path,
         out=out, grid=SMALL_GRID, log=lambda *a: None)
    import csv as _csv
    rows = list(_csv.DictReader(open(tmp_path / "params-names.csv", encoding="utf-8")))
    assert {r["label"]: r["how"] for r in rows}["Amanita"] == "genus-label"
    assert {r["label"]: r["how"] for r in rows}["Westia pacifica"] == "species"


def test_dev_tunes_the_dna_effort_threshold(store):
    from mycomap_vision.evaluate import Record
    from mycomap_vision.occtune import DEFAULT_GRID
    assert 500 in DEFAULT_GRID["min_dna_effort"]
    # A provisional lookalike with 20 DNA records, all in the west, and no DNA sampling
    # of anything in the east: only a threshold of 0 lets its absence there count.
    groups = SPECIES + ["Amanita sp. 'IN01'"]
    scores = np.full((1, len(groups)), 0.4, dtype=np.float32)
    scores[0, groups.index("Amanita sp. 'IN01'")] = 0.80
    scores[0, groups.index("Amanita orientalis")] = 0.79
    one = ScoredSet(["x"], groups, scores, truth=["Amanita orientalis"], truth_genus=["Amanita"],
                    latitude=[EAST[0]], longitude=[EAST[1]], observed_on=["2026-10-01"],
                    uuids=["val-x"], group_genus=[g.split()[0] for g in groups],
                    group_is_species=[len(g.split()) > 1 for g in groups])
    dna = [Record(f"d{i}", "Amanita sp. 'IN01'", "Amanita", "F", None, "u", [],
                  latitude=WEST[0], longitude=WEST[1]) for i in range(20)]
    p = OccurrencePrior(store, OccParams())
    p.fit(dna, groups)
    grid = {**SMALL_GRID, "out_of_range_penalty": [6.0], "density_weight": [0.0],
            "season_weight": [0.0], "genus_rule": [False], "min_dna_effort": [0, 500]}
    res = grid_search(p, one, grid, log=lambda *a: None, choose_genus_rule=None)
    right = {row["min_dna_effort"]: row["species_right"] for row in res["table"]}
    assert right == {0: 1, 500: 0}
