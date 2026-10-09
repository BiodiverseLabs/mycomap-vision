"""The learning-curve re-scoring must give the served methods' scores on the full
reference, and its subsets must be whole records, nested, and never empty a species."""

import numpy as np

from mycomap_vision import learning_curve as lc
from mycomap_vision.evaluate import Index
from mycomap_vision.methods import species_means, top_k_species_scores


def _toy(seed=0, n_units=7, dim=16):
    rng = np.random.default_rng(seed)
    recs_per_unit = rng.integers(1, 6, n_units)
    rec_unit = np.repeat(np.arange(n_units), recs_per_unit)
    photos = rng.integers(1, 5, len(rec_unit))
    photos[0] = 1                                         # a single-photo record
    rec_col_starts = np.concatenate([[0], np.cumsum(photos)[:-1]])
    vecs = rng.normal(size=(photos.sum(), dim)).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    unit_rec_starts = np.concatenate([[0], np.cumsum(recs_per_unit)[:-1]])
    unit_col_starts = rec_col_starts[unit_rec_starts]
    query = rng.normal(size=(3, dim)).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    return vecs, query, rec_unit, rec_col_starts, unit_rec_starts, unit_col_starts


def _index(unit_col_starts, n_cols):
    return Index([f"s{i}" for i in range(len(unit_col_starts))], np.arange(n_cols),
                 unit_col_starts, None)


def test_full_reference_matches_nearest_and_nearest_mean():
    for seed in range(5):
        vecs, q, rec_unit, rcs, urs, ucs = _toy(seed)
        sims = q @ vecs.T
        best, second = lc.record_best_two(sims, rcs)
        kept = np.ones(len(rec_unit), bool)
        m1, top2 = lc.species_top(best, second, kept, urs)
        idx = _index(ucs, len(vecs))
        np.testing.assert_allclose(m1, np.maximum.reduceat(sims, ucs, axis=1), rtol=1e-6)
        want_top2 = np.stack([top_k_species_scores(sims[i:i + 1], idx, 2) for i in range(len(q))])
        np.testing.assert_allclose(top2, want_top2, rtol=1e-5, atol=1e-6)
        rec_sums = np.add.reduceat(vecs, rcs, axis=0)
        protos = lc.kept_means(rec_sums, kept, urs)
        np.testing.assert_allclose(protos, species_means(vecs, idx), rtol=1e-5, atol=1e-6)


def test_subset_scores_equal_scoring_only_the_kept_photos():
    vecs, q, rec_unit, rcs, urs, ucs = _toy(3, n_units=5)
    sims = q @ vecs.T
    best, second = lc.record_best_two(sims, rcs)
    keys = lc.record_order(len(rec_unit), 7)
    kept = lc.subset_mask(keys, np.ones(len(rec_unit), bool), rec_unit, 5, fraction=0.5)
    m1, top2 = lc.species_top(best, second, kept, urs)
    lengths = np.diff(np.append(rcs, len(vecs)))
    cols = np.concatenate([np.arange(rcs[r], rcs[r] + lengths[r]) for r in np.flatnonzero(kept)])
    unit_of_col = np.repeat(rec_unit, lengths)[cols]
    for u in range(5):
        s = sims[:, cols[unit_of_col == u]]
        np.testing.assert_allclose(m1[:, u], s.max(axis=1), rtol=1e-6)
        srt = np.sort(s, axis=1)
        want = srt[:, -2:].mean(axis=1) if s.shape[1] > 1 else srt[:, -1]
        np.testing.assert_allclose(top2[:, u], want, rtol=1e-5)


def test_tied_best_photos_in_one_record_count_twice():
    sims = np.array([[0.5, 0.9, 0.9, 0.1]], np.float32)
    best, second = lc.record_best_two(sims, np.array([0, 1]))
    assert np.allclose(best, [[0.5, 0.9]])
    assert np.isneginf(second[0, 0]) and np.isclose(second[0, 1], 0.9)


def test_a_species_with_no_kept_record_cannot_be_answered():
    sims = np.array([[0.2, 0.3, 0.8]], np.float32)
    best, second = lc.record_best_two(sims, np.array([0, 1, 2]))
    m1, top2 = lc.species_top(best, second, np.array([True, True, False]), np.array([0, 2]))
    assert np.isneginf(m1[0, 1]) and np.isneginf(top2[0, 1])
    assert np.isclose(top2[0, 0], 0.25)


def test_fractions_keep_whole_records_nested_and_at_least_one():
    rng = np.random.default_rng(1)
    rec_unit = np.sort(rng.integers(0, 30, 400))
    rec_unit = np.unique(rec_unit, return_inverse=True)[1]
    n = rec_unit.max() + 1
    base = rng.random(400) > 0.1
    keys = lc.record_order(400, 5)
    prev = None
    for f in (0.25, 0.5, 0.75, 1.0):
        m = lc.subset_mask(keys, base, rec_unit, n, fraction=f)
        assert not (m & ~base).any()
        have = np.bincount(rec_unit[base], minlength=n)
        kept = np.bincount(rec_unit[m], minlength=n)
        assert ((have == 0) | (kept >= 1)).all()
        assert (kept == np.where(have > 0, np.maximum(np.floor(f * have + 0.5), 1), 0)).all()
        if prev is not None:
            assert not (prev & ~m).any()          # nested in the larger fraction
        prev = m


def test_caps_keep_at_most_n_and_seeds_differ():
    rec_unit = np.repeat(np.arange(3), [10, 2, 5])
    base = np.ones(17, bool)
    a = lc.subset_mask(lc.record_order(17, 1), base, rec_unit, 3, cap=3)
    b = lc.subset_mask(lc.record_order(17, 2), base, rec_unit, 3, cap=3)
    assert np.bincount(rec_unit[a], minlength=3).tolist() == [3, 2, 3]
    assert not np.array_equal(a, b)


def test_bad_reference_ids_leave_out_ids_that_are_also_inat(tmp_path):
    p = tmp_path / "s.tsv"
    p.write_text("observation_id\tsource\tcoalesce\n1\tiNaturalist\tNA\n2\tMO Observations\tNA\n"
                 "3\tMycoPortal\tNA\n3\tiNaturalist\tNA\n", encoding="utf-8")
    bad, both = lc.bad_reference_ids(p)
    assert bad == {"2"} and both == 1


def test_saturating_fit_recovers_ceiling_and_half_point():
    ns = np.array([1, 2, 3, 5, 10, 20, 50, 100])
    acc = 0.8 * ns / (ns + 6)
    fit = lc.fit_saturating(ns, acc)
    assert abs(fit["ceiling"] - 0.8) < 0.01 and abs(fit["half_at"] - 6) < 0.2
    # Knee: one more record adds under one point beyond it.
    k = fit["knee"]
    assert lc.saturating(k + 1, fit) - lc.saturating(k, fit) < 0.0105
    assert lc.saturating(k, fit) - lc.saturating(k - 1, fit) > 0.0095


# --- report helpers --------------------------------------------------------------------

from mycomap_vision import learning_curve_report as lr  # noqa: E402


def _truth(units, genera, species="Amanita muscaria", genus="Amanita"):
    return {"species": species, "genus": genus, "units": units, "genera": genera}


def test_a_capped_truth_slots_in_among_the_other_species_scores():
    units = ["Amanita muscaria", "Amanita pantherina", "Russula emetica", "Amanita flavoconia"]
    genus_names = ["Amanita", "Russula"]
    genus_of_unit = np.array([0, 0, 1, 0])
    others = {"species": [(1, 0.9), (2, 0.7), (3, 0.5)], "genus": [(0, 0.9), (1, 0.7)]}
    t = _truth([0], [0])
    hi = lr.capped_hits("x", 0.95, others, t, 0, units, genus_of_unit, genus_names)
    mid = lr.capped_hits("x", 0.8, others, t, 0, units, genus_of_unit, genus_names)
    lo = lr.capped_hits("x", 0.1, others, t, 0, units, genus_of_unit, genus_names)
    assert hi[("species", "strict")] == 0
    assert mid[("species", "strict")] == 1
    assert lo[("species", "strict")] == 3
    assert lo[("genus", "strict")] == 0          # congeners keep the genus on top


def test_a_capped_truth_raises_its_genus_when_it_beats_the_congeners():
    units = ["Amanita muscaria", "Russula emetica"]
    others = {"species": [(1, 0.8)], "genus": [(1, 0.8)]}
    h = lr.capped_hits("x", 0.9, others, _truth([0], [0]), 0, units, np.array([0, 1]),
                       ["Amanita", "Russula"])
    assert h[("genus", "strict")] == 0


def test_a_strict_hit_counts_at_every_looser_level():
    h = lr.first_hits([0], [0], _truth([0], [0]), ["Amanita muscaria"], ["Amanita"])
    assert {h[k] for k in h} == {0}


def test_ladder_reports_n_and_top_k_in_percent():
    hits = {"a": {("species", "strict"): 0}, "b": {("species", "strict"): 4},
            "c": {("species", "strict"): None}}
    lad = lr.ladder(hits)["species strict"]
    assert lad["n"] == 3 and round(lad["top1"], 1) == 33.3 and round(lad["top5"], 1) == 66.7
