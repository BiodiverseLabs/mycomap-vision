"""Likely sets: a list of names that holds the right one about as often as it says."""
import json
import math

import numpy as np
import pytest

from test_api import app_with_model, post_photos
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import evaluate, likely
from mycomap_vision.embed import embed_photos, photos_to_embed


def test_a_records_score_is_one_minus_the_probability_of_its_true_name():
    p = np.array([0.5, 0.3, 0.2])
    assert likely.score(p, 1) == pytest.approx(0.7)
    assert likely.score(p, None) == math.inf             # a name the reference set lacks


def test_a_true_name_past_the_cut_counts_as_a_miss():
    p = np.array([0.4, 0.3, 0.2, 0.1])
    assert likely.score(p, 3, cap=3) == math.inf
    assert likely.score(p, 2, cap=3) == pytest.approx(0.8)


def test_the_set_is_every_name_above_the_floor_and_always_the_top_one():
    p = np.array([0.1, 0.6, 0.25, 0.05])
    assert likely.likely_set(p, 0.2) == ([1, 2], False)
    assert likely.likely_set(p, 0.9) == ([1], False)       # nothing reaches it: the top name
    assert likely.likely_set(p, 0.01, cap=2) == ([1, 2], True)


def test_the_floor_comes_from_the_finite_sample_quantile():
    scores = [i / 20 for i in range(0, 20)]                # 0.00 .. 0.95
    assert likely.fit_floor(scores, 0.9) == pytest.approx(1 - 0.9)   # ceil(21*0.9)=19th
    assert likely.fit_floor([0.5] * 8 + [math.inf] * 2, 0.9) is None  # 20% unlistable


def simulate(n, rng, names=40, novel=0.0, concentration=0.1):
    """A calibrated model: the truth is drawn from the stated probabilities; a share of
    records has a truth the model has no name for."""
    rows = []
    for _ in range(n):
        p = rng.dirichlet(np.full(names, concentration))
        t = None if rng.random() < novel else int(rng.choice(names, p=p))
        rows.append((p, t))
    return rows


@pytest.mark.parametrize("novel", [0.0, 0.05])
def test_on_fresh_records_the_set_holds_the_truth_about_as_often_as_it_says(novel):
    rng = np.random.default_rng(7)
    fit = likely.fit_and_check(simulate(2000, rng, novel=novel), 0.9)
    fresh = likely.set_metrics(simulate(5000, rng, novel=novel), fit["floor"])
    # Unknown names count as misses, so they lower what is reachable, never what is stated.
    assert fit["coverage"] >= 0.6
    assert fit["coverage"] - 0.015 <= fresh["coverage"] <= fit["coverage"] + 0.04
    assert abs(fit["crosscheck"]["coverage"] - fresh["coverage"]) < 0.03


def test_a_target_whose_sets_are_too_long_steps_down_to_a_useful_one():
    rng = np.random.default_rng(3)
    vague = simulate(2000, rng, names=60, concentration=1.0)   # probabilities spread thin
    fit = likely.fit_and_check(vague, 0.9, max_mean_size=5)
    assert fit is None or (fit["coverage"] < 0.9 and fit["crosscheck"]["mean_size"] <= 5)
    roomy = likely.fit_and_check(vague, 0.9, cap=60, max_mean_size=60)
    assert roomy["coverage"] == 0.9


def compared(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    return evaluate.compare(conn, ["m1"], ["nearest"], embeddings_root=tmp_path / "emb",
                            log=lambda s: None)


def sets_of(conn, cid):
    row = conn.execute("select report_json from eval_runs where comparison_id = ?",
                       (cid,)).fetchone()
    return {r: c.get("sets") for r, c in json.loads(row[0])["all_photos"]["calibration"].items()}


def test_a_comparison_fits_a_likely_set_for_every_rank_with_a_held_back_check(conn, tmp_path):
    cid = compared(conn, tmp_path)["comparison_id"]
    sets = sets_of(conn, cid)
    for rank in ("species", "genus", "family"):
        assert sets[rank]["coverage"] <= 0.9 and 0 <= sets[rank]["floor"] <= 1
        assert "crosscheck" in sets[rank] and "in_sample" in sets[rank]
    assert evaluate.latest_calibration(conn, "m1", "nearest")["sets"]["species"] == sets["species"]


def test_calibrate_sets_adds_them_to_an_older_comparison_and_changes_nothing_else(conn, tmp_path):
    cid = compared(conn, tmp_path)["comparison_id"]
    fitted = sets_of(conn, cid)
    row = conn.execute("select id, species_top1, report_json from eval_runs").fetchone()
    old = json.loads(row[2])
    for c in old["all_photos"]["calibration"].values():
        c.pop("sets")                                       # as saved before sets existed
    conn.execute("update eval_runs set report_json = ?", (json.dumps(old),))
    conn.commit()
    evaluate.calibrate_sets(conn, cid, embeddings_root=tmp_path / "emb", log=lambda s: None)
    assert sets_of(conn, cid) == fitted
    after = conn.execute("select id, species_top1 from eval_runs").fetchone()
    assert tuple(after) == (row[0], row[1])                 # same row, same published score


def test_calibrate_sets_refuses_a_comparison_whose_records_changed(conn, tmp_path):
    cid = compared(conn, tmp_path)["comparison_id"]
    conn.execute("update eval_runs set record_set = 'other'")
    conn.commit()
    with pytest.raises(RuntimeError, match="have changed"):
        evaluate.calibrate_sets(conn, cid, embeddings_root=tmp_path / "emb", log=lambda s: None)


def test_an_identification_lists_the_likely_names_with_their_coverage(conn, tmp_path):
    client = app_with_model(conn, tmp_path)
    evaluate.compare(conn, ["m1"], ["nearest"], embeddings_root=tmp_path / "emb",
                     log=lambda s: None)
    conn.commit()
    [r] = post_photos(client, [247], "m1/nearest").json()["results"]
    sp = r["likely"]["species"]
    assert sp["names"] and sp["names"][0]["name"] == r["ranks"]["species"][0]["name"]
    assert 0.5 <= sp["coverage"] <= 0.9 and sp["capped"] is False
    assert all(0 <= n["confidence"] <= 1 for n in sp["names"])


def test_without_a_fitted_set_no_list_is_claimed(conn, tmp_path):
    client = app_with_model(conn, tmp_path)                 # never compared: not calibrated
    [r] = post_photos(client, [247], "m1/nearest").json()["results"]
    assert r["likely"] == {}
