"""The record-scoring experiment's report (scripts/record_scoring/run_v1.py): the decision
rule as pre-declared, and the tallies it rests on."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "record_scoring" / "run_v1.py"


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("record_scoring_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def scope(run, lb, cal, top1):
    """A primary scope with these one-sided lower bounds per arm (photos and +place alike),
    calibration {arm: (ece, nll)} and photos-only top-1 per arm."""
    a = run.ARMS[0]
    s = {"paired": {}, "calibration": {a: {"ece": 0.04, "nll": 2.0}}}
    for m in run.ARMS[1:]:
        for fam in ("photos", "place"):
            x, y = (m, a) if fam == "photos" else (run.PLACE[m], run.PLACE[a])
            s["paired"][f"{x} vs {y}"] = {"noninferior": lb[m] > -run.MARGIN, "a": top1[m]}
        e, n = cal[m]
        s["calibration"][m] = {"ece": e, "nll": n}
    return s


def test_an_arm_within_the_margin_is_adopted_and_the_tie_goes_to_the_simplest(run):
    b, c, d = run.ARMS[1:]
    s = scope(run, {b: -0.002, c: -0.001, d: -0.001},
              {b: (0.04, 2.0), c: (0.04, 2.0), d: (0.04, 2.0)},
              {b: 0.5170, c: 0.5175, d: 0.5179})
    assert run.decide(s, {})["adopt"] == b          # all within 0.1 pt of the best: simplest


def test_a_clearly_better_arm_beats_a_simpler_one(run):
    b, c, d = run.ARMS[1:]
    s = scope(run, {b: -0.002, c: 0.001, d: -0.001},
              {b: (0.04, 2.0), c: (0.04, 2.0), d: (0.04, 2.0)},
              {b: 0.517, c: 0.525, d: 0.519})
    assert run.decide(s, {})["adopt"] == c


@pytest.mark.parametrize("why", ["margin", "ece", "nll"])
def test_an_arm_outside_the_margin_or_worse_calibrated_is_not_adopted(run, why):
    b, c, d = run.ARMS[1:]
    lb = {b: -0.004 if why == "margin" else 0.0, c: -0.01, d: -0.01}
    cal = {b: (0.04 + (0.006 if why == "ece" else 0), 2.0 + (0.03 if why == "nll" else 0)),
           c: (0.04, 2.0), d: (0.04, 2.0)}
    assert run.decide(scope(run, lb, cal, {b: 0.52, c: 0.52, d: 0.52}), {})["adopt"] is None


def test_the_place_twin_must_also_be_non_inferior(run):
    b, c, d = run.ARMS[1:]
    s = scope(run, {b: 0.0, c: -0.01, d: -0.01}, {m: (0.04, 2.0) for m in (b, c, d)},
              {b: 0.52, c: 0.52, d: 0.52})
    s["paired"][f"{run.PLACE[b]} vs {run.PLACE[run.ARMS[0]]}"]["noninferior"] = False
    assert run.decide(s, {})["adopt"] is None


def test_outcomes_count_a_hit_at_k_and_the_looser_levels(run):
    finds = {"1": {"species": "Amanita muscaria", "genus": "Amanita"}}
    recs = {"1": {"top": {"species": ["Amanita pantherina", "Amanita muscaria"],
                          "genus": ["Amanita"]},
                  "cal": {"species": {"conf": 0.6, "p_true": 0.3}}}}
    o = run.outcomes(recs, finds)["1"]
    assert not o["species_strict@1"] and o["species_strict@3"]
    assert o["genus_strict@1"] and o["conf"] == 0.6


def test_ece_of_perfectly_calibrated_bins_is_zero_and_of_overconfidence_is_the_gap(run):
    conf = np.r_[np.full(50, 0.5), np.full(50, 1.0)]
    right = np.r_[np.tile([0, 1], 25), np.ones(50)]
    assert run.ece(conf, right, bins=2) == pytest.approx(0.0)
    assert run.ece(np.full(10, 0.9), np.r_[np.ones(5), np.zeros(5)], bins=1) == \
        pytest.approx(0.4)


def test_the_bootstrap_resamples_whole_observers(run):
    # one observer with every miss: resampling observers widens the interval far beyond
    # resampling finds
    values = np.r_[np.ones(90), np.zeros(10)]
    who = [f"u{i}" for i in range(90)] + ["miss"] * 10
    est = run.boot(values, who, reps=2000, seed=1)
    assert est.std() > 0.05


def test_twins_group_only_the_record_arms_and_their_place_twins(run):
    assert not run.grouped("nearest+mean") and not run.grouped("nearest+mean+place")
    for a in run.ARMS[1:]:
        assert run.grouped(a) and run.grouped(run.PLACE[a])


def test_a_twins_file_maps_records_to_their_specimen(run, tmp_path):
    p = tmp_path / "twins.tsv"
    p.write_text("observation_id\tspecimen\nmo:1\tt1\ninat:2\tt1\n", encoding="utf-8")
    assert run.read_twins(p) == {"mo:1": "t1", "inat:2": "t1"}
    bad = tmp_path / "bad.tsv"
    bad.write_text("record\tgroup\nx\ty\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="observation_id and specimen"):
        run.read_twins(bad)
