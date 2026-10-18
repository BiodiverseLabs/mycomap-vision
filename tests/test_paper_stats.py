"""The paper's pre-declared statistics (paper_stats.py), checked against hand calculations
and closed forms: exact McNemar tails, Holm's adjustment, cluster-bootstrap intervals that
widen when finds cluster, non-inferiority and the intersection-union test, and the GEE
depth trend against the saturated logistic model's closed form."""

import math

import numpy as np
import pytest

from mycomap_vision import paper_stats as ps


# --- McNemar -----------------------------------------------------------------------------------

def test_exact_mcnemar_tails_are_binomial_halves():
    # 8 discordant finds, 7 for the first method: P(X >= 7) = (8 + 1) / 256.
    assert ps.mcnemar_exact(7, 1, "greater") == pytest.approx(9 / 256)
    assert ps.mcnemar_exact(7, 1, "less") == pytest.approx(255 / 256)
    assert ps.mcnemar_exact(7, 1) == pytest.approx(18 / 256)
    assert ps.mcnemar_exact(1, 7) == ps.mcnemar_exact(7, 1)
    assert ps.mcnemar_exact(0, 0, "greater") == 1.0
    assert ps.mcnemar_exact(5, 5) == 1.0                         # capped at 1
    with pytest.raises(ValueError):
        ps.mcnemar_exact(1, 2, "bigger")


def test_exact_mcnemar_holds_at_thousands_of_discordant_finds():
    p = ps.mcnemar_exact(1100, 1000, "greater")
    z = (1100 - 1050) / math.sqrt(2100 / 4)              # normal approximation, continuity
    assert p == pytest.approx(ps.normal_sf(z - 0.5 / math.sqrt(2100 / 4)), rel=0.05)


# --- Holm --------------------------------------------------------------------------------------

def test_holm_steps_down_and_stays_monotone():
    out = ps.holm({"H1a": 0.010, "H1b": 0.040, "H2": 0.030, "H3": 0.001, "H4": 0.20})
    assert out["H3"]["p_holm"] == pytest.approx(0.005)
    assert out["H1a"]["p_holm"] == pytest.approx(0.040)
    assert out["H2"]["p_holm"] == pytest.approx(0.090)
    assert out["H1b"]["p_holm"] == pytest.approx(0.090)       # monotone: not 0.08
    assert out["H4"]["p_holm"] == pytest.approx(0.20)
    assert [k for k, v in out.items() if v["reject"]] == ["H1a", "H3"]
    assert list(out) == ["H1a", "H1b", "H2", "H3", "H4"]     # input order kept


def test_holm_stops_at_the_first_failure():
    out = ps.holm({"a": 0.001, "b": 0.03, "c": 0.04})
    # 0.001 < 0.05/3; 0.03 > 0.05/2 stops: c is not rejected though 0.04 < 0.05.
    assert [k for k, v in out.items() if v["reject"]] == ["a"]


# --- bootstrap ---------------------------------------------------------------------------------

def test_the_cluster_bootstrap_widens_when_finds_cluster_by_observer():
    rng = np.random.default_rng(0)
    # 100 observers, 20 finds each; each observer is all right or all wrong.
    clustered = np.repeat(rng.random(100) < 0.5, 20).astype(float)
    who = np.repeat(np.arange(100), 20)
    lo, hi = ps.cluster_ci(clustered, who, reps=2000)
    lo1, hi1 = ps.cluster_ci(clustered, np.arange(2000), reps=2000)   # as if independent
    assert (hi - lo) > 3 * (hi1 - lo1)
    assert lo < clustered.mean() < hi


def test_a_rate_with_one_value_has_a_point_interval_and_none_without_finds():
    assert ps.cluster_ci([1, 1, 1], ["a", "b", "c"], reps=200) == (1.0, 1.0)
    assert ps.cluster_ci([], [], reps=200) is None


def test_the_bootstrap_is_seeded():
    x, who = [1, 0, 1, 1, 0, 0, 1], list("aabbccd")
    assert ps.cluster_ci(x, who, reps=500, seed=3) == ps.cluster_ci(x, who, reps=500, seed=3)


def test_the_paired_interval_is_of_the_difference_on_the_same_finds():
    rng = np.random.default_rng(4)
    b = (rng.random(400) < 0.5).astype(float)
    a = np.where(rng.random(400) < 0.3, 1.0, b)                 # a = b, plus some rescues
    who = np.arange(400) // 4
    lo, hi = ps.paired_ci(a, b, who, reps=2000)
    assert lo < a.mean() - b.mean() < hi and lo > 0
    # The same rates on unpaired finds would not tell them apart this tightly.
    lo_a, hi_a = ps.cluster_ci(a, who, reps=2000)
    assert hi - lo < hi_a - lo_a


# --- non-inferiority and the intersection-union test --------------------------------------------

def test_non_inferiority_holds_when_the_lower_bound_clears_the_margin():
    rng = np.random.default_rng(1)
    who = np.arange(2000) // 2
    b = (rng.random(2000) < 0.5).astype(float)
    same = ps.noninferiority(b, b, who, margin=0.02, reps=2000)
    assert same["holds"] and same["lower_bound"] == 0.0 and same["p"] < 0.001
    worse = b.copy()
    worse[np.flatnonzero(b == 1)[:150]] = 0                  # 7.5 points worse
    res = ps.noninferiority(worse, b, who, margin=0.02, reps=2000)
    assert not res["holds"] and res["p"] > 0.5


def test_the_intersection_union_test_takes_the_larger_p():
    assert ps.intersection_union(0.01, 0.04) == 0.04
    assert ps.intersection_union(0.2, 0.001) == 0.2


# --- GEE ---------------------------------------------------------------------------------------

def saturated(a, b, c, d):
    """y and x for a 2x2 table: x=0 has a hits of a+b, x=1 has c hits of c+d."""
    y = [1] * a + [0] * b + [1] * c + [0] * d
    x = [0] * (a + b) + [1] * (c + d)
    return np.array(y), np.array(x)


def test_gee_on_a_two_by_two_table_is_the_log_odds_ratio_with_the_closed_form_variance():
    a, b, c, d = 30, 70, 60, 40
    y, x = saturated(a, b, c, d)
    fit = ps.gee_logistic(y, x, np.arange(len(y)), ["x"], small_sample=False)
    slope = fit["coefficients"]["x"]
    assert slope["estimate"] == pytest.approx(math.log((c / d) / (a / b)), abs=1e-8)
    assert fit["coefficients"]["intercept"]["estimate"] == pytest.approx(math.log(a / b))
    # Saturated model, one find per cluster: robust = model-based = sqrt(sum of 1/cells).
    want = math.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    assert slope["se_robust"] == pytest.approx(want, rel=1e-6)
    assert slope["se_model"] == pytest.approx(want, rel=1e-6)


def test_gee_robust_errors_see_through_duplicated_finds_in_one_cluster():
    y, x = saturated(30, 70, 60, 40)
    one = ps.gee_logistic(y, x, np.arange(len(y)), ["x"], small_sample=False)
    # Every find twice, the copies in the same cluster: the model-based SE shrinks by
    # sqrt(2) as if there were twice the evidence; the clustered one does not.
    twice = ps.gee_logistic(np.repeat(y, 2), np.repeat(x, 2), np.repeat(np.arange(len(y)), 2),
                            ["x"], small_sample=False)
    s1, s2 = one["coefficients"]["x"], twice["coefficients"]["x"]
    assert s2["estimate"] == pytest.approx(s1["estimate"])
    assert s2["se_model"] == pytest.approx(s1["se_model"] / math.sqrt(2), rel=1e-6)
    assert s2["se_robust"] == pytest.approx(s1["se_robust"], rel=1e-6)


def test_the_depth_trend_test_is_one_sided_on_the_slope():
    rng = np.random.default_rng(2)
    depth = rng.integers(0, 200, 3000)
    p = 1 / (1 + np.exp(-(-2 + 0.4 * np.log2(1 + depth))))
    y = (rng.random(3000) < p).astype(int)
    fit = ps.gee_logistic(y, np.log2(1 + depth), np.arange(3000) // 3, ["log2_refs"])
    slope = fit["coefficients"]["log2_refs"]
    assert slope["estimate"] == pytest.approx(0.4, abs=0.08) and fit["converged"]
    assert ps.one_sided_wald(slope) < 1e-6
    assert ps.one_sided_wald(slope, "less") > 0.99
