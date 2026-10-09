"""exp/prior-tuning: the declared grid, the observer folds, nested cross-validation and
the way a prior setting combines with the photo scores."""

import math

import numpy as np
import pytest

from mycomap_vision import priortune as pt


def comps(S, SP=None, truth_at=None, GG=None, truth_g=None, **extra):
    S = np.asarray(S, dtype=np.float64)
    n, k = S.shape
    c = pt.Components(
        S=S, cand=np.tile(np.arange(k), (n, 1)),
        SP=np.ones((n, k), bool) if SP is None else np.asarray(SP, bool),
        truth_at=np.zeros(n, np.int64) if truth_at is None else np.asarray(truth_at),
        has_sp=np.ones(n, bool),
        GG=np.zeros((n, k), np.int64) if GG is None else np.asarray(GG),
        truth_g=np.zeros(n, np.int64) if truth_g is None else np.asarray(truth_g))
    for name, value in extra.items():
        setattr(c, name, value)
    return c


def test_every_declared_setting_is_scored_and_counted():
    grid = pt.settings()
    assert len(grid) == 288 + 1152 + 384
    assert {s["family"] for s in grid} == {"dna", "occ", "dna+occ"}
    # Steve's wide berth stays in the combined family.
    assert all(s["radius_km"] == 1500.0 for s in grid if s["family"] == "dna+occ")
    # The order is fixed: the same call gives the same list.
    assert grid == pt.settings()


def test_an_observer_never_spans_two_folds():
    observers = ["a"] * 7 + ["b"] * 3 + ["c"] * 3 + [None, None] + list("defghij")
    folds = pt.observer_folds(observers, k=3, seed=1)
    for o in set(observers) - {None}:
        assert len({folds[i] for i, x in enumerate(observers) if x == o}) == 1
    assert set(folds.tolist()) == {0, 1, 2}
    sizes = np.bincount(folds)
    assert sizes.max() - sizes.min() <= 7          # balanced up to the biggest observer


def test_each_fold_is_scored_with_a_setting_chosen_on_the_others():
    # Setting 1 is right only on fold 0's records, setting 0 on the rest: fold 0 must be
    # scored with setting 0 (chosen elsewhere), so its records count as wrong.
    folds = np.array([0, 0, 1, 1, 2, 2])
    sp = np.array([[0, 0, 1, 1, 1, 1], [1, 1, 0, 0, 0, 0]], bool)
    ge = np.zeros_like(sp)
    grid = [{"family": "dna", "place_weight": 0.25}, {"family": "dna", "place_weight": 0.5}]
    cv = pt.nested_cv(sp, ge, folds, grid)
    assert cv["chosen"][0] == 0
    assert not cv["species_right"][:2].any()
    assert cv["species_right"][2:].all()


def test_ties_go_to_the_gentler_setting():
    sp = np.array([[1, 0], [1, 0]], bool)
    ge = np.zeros_like(sp)
    grid = [{"family": "occ", "penalty": 50.0, "density_weight": 1.0, "season_weight": 0.0},
            {"family": "occ", "penalty": 2.0, "density_weight": 0.0, "season_weight": 0.0}]
    assert pt.choose(sp, ge, np.arange(2), grid) == 1


def test_wide_berth_penalises_only_species_out_of_range():
    S = [[0.50, 0.49]]
    out = np.array([[True, False]])
    none = np.zeros((1, 2), bool)
    c = comps(S, occ_out={1500.0: out}, occ_genus_out={1500.0: none},
              occ_dna_out={1500.0: none}, occ_dna_effort={1500.0: np.zeros(1)},
              occ_density={150.0: np.zeros((1, 2))}, occ_season=np.zeros((1, 2)))
    s = {"family": "occ", "radius_km": 1500.0, "penalty": 2.0, "genus_rule": False,
         "density_km": 150.0, "density_weight": 0.0, "season_weight": 0.0}
    z = pt.combine(c, s)
    assert z[0, 0] == pytest.approx(0.50 / 0.02 - 2.0)
    assert z[0, 1] == pytest.approx(0.49 / 0.02)


def test_dna_only_absence_counts_only_where_dna_sampling_is_dense():
    none = np.zeros((2, 1), bool)
    dna_out = np.ones((2, 1), bool)
    c = comps([[0.5], [0.5]], occ_out={1500.0: none}, occ_genus_out={1500.0: dna_out},
              occ_dna_out={1500.0: dna_out}, occ_dna_effort={1500.0: np.array([100, 900])})
    assert pt.out_of_range(c, 1500.0, genus_rule=False).ravel().tolist() == [False, True]
    assert pt.out_of_range(c, 1500.0, genus_rule=True).ravel().tolist() == [True, True]


def test_the_as_shipped_dna_prior_caps_place_and_season_together():
    place = np.array([[4.0, -1.0]])
    season = np.array([[2.0, -1.0]])
    c = comps([[0.0, 0.0]], dna_place={150.0: place}, dna_season={20.0: season})
    z = pt.combine(c, pt.AS_SHIPPED)
    assert z[0, 0] == pytest.approx(math.log(20.0))      # 6 capped at log 20
    assert z[0, 1] == pytest.approx(-2.0)


def test_a_species_answer_is_never_a_one_word_group():
    # Group 0 ("Russula", one word) scores best, but the species answer is group 1.
    c = comps([[0.9, 0.8]], SP=[[False, True]], truth_at=[1], GG=[[3, 3]], truth_g=[3])
    sp, ge = pt.top1(c, pt.combine(c, {"family": "none"}))
    assert sp.tolist() == [True] and ge.tolist() == [True]


def test_stated_confidence_is_a_probability_at_the_fitted_temperature():
    c = comps([[0.60, 0.50, 0.40]] * 4, truth_at=[0, 0, 0, 1])
    z = pt.combine(c, {"family": "none"})
    T = pt.fit_temperature(c, z, np.arange(4))
    conf = pt.top_confidence(c, z, T)
    assert 0.33 < conf[0] < 1.0                      # not saturated at 1.00
    total, used = pt.nll_curve(c, z, np.arange(4))
    assert used == 4 and np.isfinite(total).all()


def test_ranks_give_species_genera_and_families_with_confidence():
    z = np.array([5.0, 4.0, 3.0])
    cand = np.array([0, 1, 2])
    out = pt.ranks(z, cand, np.array([True, True, False]), ["A b", "A c", "A"],
                   ["A", "A", "A"], ["F", "F", "F"], T=1.0)
    assert [r["name"] for r in out["species"]] == ["A b", "A c"]
    assert [r["name"] for r in out["genus"]] == ["A"]
    assert out["genus"][0]["confidence"] == 1.0
    assert out["top_k"] == 10


def test_range_edges_are_reported_as_bands_never_places():
    assert pt.distance_band(50.0) == "< 100 km"
    assert pt.distance_band(1499.0) == "1,000-1,500 km"
    assert pt.distance_band(2400.0) == ">= 1,500 km"
    assert pt.distance_band(float("nan")) == "no known find"
    assert set(pt.DISTANCE_BANDS) >= {pt.distance_band(x) for x in (1, 200, 500, 1200, 1e4)}
