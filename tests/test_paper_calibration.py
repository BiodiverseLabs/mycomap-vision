"""Calibration and confidence in `mv paper-score` (paper_calibration.py, paper E13 and E11):
equal-mass ECE, Brier and NLL from the true name's probability, the confidence rule with
its lower bound, risk-coverage, likely sets by band, and telling unknown species apart."""

import math

import numpy as np
import pytest

from mycomap_vision import paper_calibration as pc
from mycomap_vision import paper_outcomes as po


def test_ece_is_zero_when_confidence_matches_accuracy_bin_by_bin():
    conf = [0.2] * 50 + [0.8] * 50
    right = [1] * 10 + [0] * 40 + [1] * 40 + [0] * 10
    r = pc.ece(conf, right, bins=2)
    assert r["ece"] == pytest.approx(0.0) and [b["n"] for b in r["bins"]] == [50, 50]
    over = pc.ece([0.9] * 100, [1, 0] * 50, bins=15)
    assert over["ece"] == pytest.approx(0.4, abs=0.02)       # 90% stated, ~50% right
    assert sum(b["n"] for b in over["bins"]) == 100 and len(over["bins"]) == 15


def test_brier_and_nll_follow_from_the_true_names_probability():
    # Two classes, p = (0.8, 0.2), truth the first: Brier (0.2^2 + 0.2^2) = 0.08.
    r = pc.brier_nll([0.8], [0.8 ** 2 + 0.2 ** 2])
    assert r["brier"] == pytest.approx(0.08) and r["nll"] == pytest.approx(-math.log(0.8))
    assert pc.brier_nll([0.0], [1.0])["nll"] == pytest.approx(-math.log(1e-12))  # no reference


def test_the_confidence_rule_reports_precision_coverage_and_a_lower_bound():
    rng = np.random.default_rng(0)
    conf = rng.random(1000)
    right = (rng.random(1000) < conf).astype(float)
    who = np.arange(1000) // 5
    r = pc.confidence_rule(conf, right, who, 0.7, 500, 0)
    assert r["confident"] == int((conf >= 0.7).sum())
    assert r["coverage"] == pytest.approx((conf >= 0.7).mean())
    assert r["lower_bound_one_sided_95"] < r["precision"] < r["ci95"][1]
    assert r["ci95"][0] < r["lower_bound_one_sided_95"]


def test_risk_coverage_puts_the_confident_finds_first():
    r = pc.risk_coverage([0.9, 0.8, 0.1, 0.2], [1, 1, 0, 0], steps=2)
    assert r["points"] == [{"coverage": 0.5, "accuracy": 1.0}, {"coverage": 1.0, "accuracy": 0.5}]
    assert r["aurc"] == pytest.approx(np.mean([0, 0, 1 / 3, 0.5]))


def test_auroc_counts_pairs_and_halves_ties():
    assert pc.auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert pc.auroc([0.5], [0.5]) == 0.5
    assert pc.auroc([0.9, 0.1], [0.5]) == 0.5
    assert pc.auroc([], [0.5]) is None


def test_the_summary_by_band_and_for_unknown_species():
    finds = [po.Find(str(i), f"o{i}", "A x", "A", "F", "formal", depth, 1)
             for i, depth in enumerate([0, 0, 5, 50, 50, 200])]
    cal = {f.key: {"species": {"conf": c, "p_true": p, "sumsq": 0.5,
                               "set": ["A x"] if f.depth else ["B y"]}}
           for f, c, p in zip(finds, [0.1, 0.2, 0.6, 0.9, 0.95, 0.99],
                              [0, 0, 0.6, 0.9, 0.95, 0.99])}
    outcome = {f.key: {"species_strict@1": f.depth > 0, "genus_strict@1": True} for f in finds}
    s = pc.summarise(finds, cal, outcome, "species", {"rules": [0.7], "not_sure_below": 0.3},
                     200, 0, lambda f: "0" if f.depth == 0 else "1+")
    assert s["n"] == 6 and s["unknown_species"]["auroc_known_vs_unknown"] == 1.0
    assert s["unknown_species"]["not_sure_share"] == {"known": 0.0, "unknown": 1.0,
                                                      "below": 0.3}
    assert s["unknown_species"]["genus_top1_without_species_reference"] == {"n": 2, "rate": 1.0}
    assert s["likely_sets"]["by_band"]["0"]["coverage"] == 0.0
    assert s["likely_sets"]["by_band"]["1+"]["coverage"] == 1.0
    assert s["rules"][0]["confident"] == 3 and s["rules"][0]["precision"] == 1.0


def test_likely_sets_report_the_coverage_achieved_with_its_interval_beside_the_target():
    finds = [po.Find(str(i), f"o{i % 3}", "A x", "A", "F", "formal", 5, 1) for i in range(10)]
    cal = {f.key: {"species": {"conf": 0.5, "p_true": 0.5, "sumsq": 0.5,
                               "set": ["A x"] if int(f.key) < 8 else ["B y"]}} for f in finds}
    outcome = {f.key: {"species_strict@1": True} for f in finds}
    s = pc.summarise(finds, cal, outcome, "species", {"list_target": {"species": 0.9}},
                     200, 0, lambda f: "5-19")
    ls = s["likely_sets"]
    assert ls["coverage"] == 0.8 and ls["target"] == 0.9          # achieved, not requested
    assert ls["ci95"][0] <= 0.8 <= ls["ci95"][1]
    assert pc.summarise(finds, cal, outcome, "species", {"list_target": 0.85}, 200, 0,
                        lambda f: "5-19")["likely_sets"]["target"] == 0.85
