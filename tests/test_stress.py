"""Rules the leakage / TTA / stress scoring relies on (mycomap_vision.stress): masking a
reference record must score exactly as if it had never been in the reference."""

import numpy as np
import pytest

from mycomap_vision import methods, stress
from mycomap_vision.evaluate import Record, build_index
from mycomap_vision.prior import Context, RangeSeasonPrior


def _index(n_species=6, per=(1, 2, 3, 5, 4, 2), dim=16, seed=0):
    rng = np.random.default_rng(seed)
    recs, row = [], 0
    for s in range(n_species):
        for k in range(per[s]):
            recs.append(Record(f"{s}-{k}", f"Genus{s % 3} sp{s}", f"Genus{s % 3}", "Fam",
                               None, f"user{k % 2}", photo_rows=[row, row + 1],
                               latitude=40.0 + s + k * 0.01, longitude=-100.0 - s,
                               observed_on=f"2025-0{1 + (s + k) % 9}-15"))
            row += 2
    vecs = rng.normal(size=(row, dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    index = build_index(recs)
    return recs, index, vecs


def _brute_best_two(sims, starts, keep):
    ends = np.append(starts[1:], sims.shape[1])
    m1 = np.full((len(sims), len(starts)), -np.inf, np.float32)
    m2 = m1.copy()
    for g, (a, b) in enumerate(zip(starts, ends)):
        for q in range(len(sims)):
            v = sorted((sims[q, c] for c in range(a, b) if keep[c]), reverse=True)
            if v:
                m1[q, g] = v[0]
            if len(v) > 1:
                m2[q, g] = v[1]
    return m1, m2


def test_masked_best_two_equals_scoring_without_the_masked_photos():
    rng = np.random.default_rng(1)
    starts = np.array([0, 1, 3, 6, 11])
    sims = rng.normal(size=(3, 14)).astype(np.float32)
    sims[0, 3] = sims[0, 4]                          # a tie for best counts twice
    keep = rng.random(14) > 0.3
    keep[11:14] = False                              # a whole group masked
    m1, m2 = stress.best_two(sims, starts, keep)
    b1, b2 = _brute_best_two(sims, starts, keep)
    np.testing.assert_array_equal(m1, b1)
    np.testing.assert_array_equal(m2, b2)
    assert np.isneginf(m1[:, 4]).all()


def test_masking_only_touched_groups_gives_the_full_recompute():
    rng = np.random.default_rng(2)
    starts = np.array([0, 2, 5, 9])
    col_group = np.repeat(np.arange(4), np.diff(np.append(starts, 12)))
    sims = rng.normal(size=(2, 12)).astype(np.float32)
    masked = np.array([1, 6, 7])
    keep = np.ones(12, bool)
    keep[masked] = False
    fast = stress.best_two_masked(sims, starts, col_group, stress.best_two(sims, starts), masked)
    full = stress.best_two(sims, starts, keep)
    np.testing.assert_array_equal(fast[0], full[0])
    np.testing.assert_array_equal(fast[1], full[1])


def test_unmasked_scores_match_the_served_methods():
    _recs, index, vecs = _index()
    ref = vecs[index.cols]
    query = vecs[[0, 5, 9]] + 0.1
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    sims = query @ ref.T
    m1, m2 = stress.best_two(sims, index.starts)
    np.testing.assert_allclose(stress.nearest_scores(m1), methods.species_scores(sims, index),
                               rtol=1e-6)
    sums = stress.species_sums(ref, index.starts)
    sizes = np.diff(np.append(index.starts, len(index.cols)))
    col_group = np.repeat(np.arange(len(index.starts)), sizes)
    ms = stress.mean_sims(query, sums, sizes, ref, col_group, np.zeros(0, np.int64))
    want = (0.6 * methods.top_k_species_scores(sims, index, 2)
            + 0.4 * (query @ methods.species_means(vecs, index).T).mean(axis=0))
    np.testing.assert_allclose(stress.nearest_mean_scores(m1, m2, ms), want, rtol=1e-5)


def test_masked_records_leave_their_species_average_and_an_emptied_species_has_no_score():
    _recs, index, vecs = _index()
    ref = vecs[index.cols]
    sizes = np.diff(np.append(index.starts, len(index.cols)))
    col_group = np.repeat(np.arange(len(index.starts)), sizes)
    sums = stress.species_sums(ref, index.starts)
    query = vecs[[3]]
    masked = np.concatenate([np.arange(index.starts[0], index.starts[1]),   # all of group 0
                             [index.starts[3], index.starts[3] + 1]])        # part of group 3
    got = stress.mean_sims(query, sums, sizes, ref, col_group, masked)
    assert np.isneginf(got[0, 0])
    keep = np.setdiff1d(np.arange(index.starts[3], index.starts[4]), masked)
    mean3 = ref[keep].sum(axis=0)
    np.testing.assert_allclose(got[0, 3], query[0] @ (mean3 / np.linalg.norm(mean3)), rtol=1e-5)
    untouched = stress.mean_sims(query, sums, sizes, ref, col_group, np.zeros(0, np.int64))
    np.testing.assert_allclose(got[0, 1], untouched[0, 1])


def test_an_emptied_species_is_never_a_candidate():
    scores = np.array([0.9, -np.inf, 0.5, 0.7])
    assert stress.top_k(scores, 10).tolist() == [0, 3, 2]


def test_prior_without_some_records_equals_a_prior_fitted_without_them():
    recs, index, _vecs = _index()
    prior = RangeSeasonPrior()
    prior.fit(recs, index.species)
    ctx = Context(43.0, -103.0, "2025-04-10")
    np.testing.assert_allclose(stress.log_prior_excluding(prior, ctx), prior.log_prior(ctx))
    drop = [3, 7, 11]                                # each species keeps other records
    other = RangeSeasonPrior()
    other.fit([r for i, r in enumerate(recs) if i not in drop], index.species)
    np.testing.assert_allclose(stress.log_prior_excluding(prior, ctx, np.array(drop)),
                               other.log_prior(ctx), rtol=1e-9, atol=1e-12)


def test_same_collection_is_same_observer_near_in_place_and_time_only():
    user = np.array([1, 1, 1, 2, 1, 1])
    lat = np.array([45.0, 45.005, 45.5, 45.0, 45.0, np.nan])
    lon = np.array([-120.0, -120.0, -120.0, -120.0, -120.0, -120.0])
    day = np.array([100, 101, 100, 100, 110, 100])
    got = stress.same_collection(1, 45.0, -120.0, 100, user, lat, lon, day, km=1.0, days=1)
    # near + same day, near + next day; not far, not another observer, not 10 days
    # later, not without a place
    assert got.tolist() == [True, True, False, False, False, False]
    assert not stress.same_collection(None, 45.0, -120.0, 100, user, lat, lon, day).any()
    assert not stress.same_collection(1, 45.0, -120.0, None, user, lat, lon, day).any()


@pytest.mark.parametrize("x,band", [(0, "0"), (1, "1-9"), (9, "1-9"), (10, "10-99"),
                                    (100, "100+")])
def test_observer_bands(x, band):
    assert stress.OBSERVER_REFS.of(x) == band


def test_season_share_wraps_the_year_and_needs_dates():
    doys = np.array([5, 360, 180, -1])
    assert stress.season_share(1, doys) == pytest.approx(2 / 3)
    assert stress.season_share(None, doys) is None
    assert stress.season_share(10, np.array([-1])) is None
    assert stress.season_band(0.0).startswith("off season")
    assert stress.season_band(0.6).startswith("peak")


def test_observer_control_takes_the_same_depth_from_each_species_but_other_records():
    group = np.array([0, 0, 0, 0, 1, 1, 1, 2, 2])
    same = np.array([1, 1, 0, 0, 1, 0, 0, 0, 0], dtype=bool)
    ctrl = stress.matched_control(same, group, seed=3)
    assert not (ctrl & same).any()
    assert np.bincount(group[ctrl], minlength=3).tolist() == [2, 1, 0]
    # a species with too few other records loses all of them, no more
    small = np.array([1, 1, 0], dtype=bool)
    assert stress.matched_control(small, np.array([0, 0, 0]), seed=1).tolist() == \
        [False, False, True]
