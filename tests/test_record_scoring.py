"""Record-grouped scoring (experiment record-scoring): a species scores by its best
reference RECORDS (specimens), not by its best photos, which can all be one record."""

import numpy as np
import pytest

from mycomap_vision.evaluate import METHODS, Record, build_index, evaluate
from mycomap_vision.methods import (NearestAndMean, NearestAndMeanRecords,
                                    NearestAndMeanRecordsBalanced, NearestAndMeanViews,
                                    method_ready, record_balanced_species_means,
                                    species_means, top_k_record_scores_per_photo)

RECORD_ARMS = ("nearest+mean-records", "nearest+mean-records-balanced", "nearest+mean-views")


def unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def random_reference(seed, layout, dim=12):
    """layout: per species, photos per record. Records are listed out of species order so
    build_index has to group them."""
    rng = np.random.default_rng(seed)
    recs, row = [], 0
    for s, photos in enumerate(layout):
        for j, n in enumerate(photos):
            recs.append(Record(f"{s}-{j}", f"G{s % 2} sp{s}", f"G{s % 2}", "F", "2026-01-01",
                               "u", list(range(row, row + n))))
            row += n
    rng.shuffle(recs)
    vecs = unit(rng.normal(size=(row, dim))).astype(np.float16)
    return vecs, recs


def brute(vecs, recs, query, k=2, weight=0.6, balanced=False, views=False):
    """The arms written out record by record, for comparison."""
    by_sp = {}
    for r in recs:
        by_sp.setdefault(r.unit, []).append(r.photo_rows)
    q = np.asarray(query, dtype=np.float32)
    v = vecs.astype(np.float32)
    out = []
    for sp in sorted(by_sp):
        records = by_sp[sp]
        rec_best = np.stack([(q @ v[rows].T).max(axis=1) for rows in records], axis=1)
        if views:
            top = np.sort(rec_best.mean(axis=0))[::-1][:k]
            nearest = np.full(len(q), top.mean())
        else:
            nearest = np.sort(rec_best, axis=1)[:, ::-1][:, :k].mean(axis=1)
        if balanced:
            mean = unit(sum(unit(v[rows].sum(axis=0)) for rows in records))
        else:
            mean = unit(v[[r for rows in records for r in rows]].sum(axis=0))
        out.append(weight * nearest + (1 - weight) * (q @ mean))
    return np.stack(out, axis=1).mean(axis=0)


def test_the_index_keeps_each_records_photos_together_inside_its_species():
    vecs, recs = random_reference(1, [[2, 1], [3], [1, 1, 4]])
    index = build_index(recs)
    assert len(index.rec_starts) == len(recs)
    bounds = np.append(index.rec_starts, len(index.cols))
    groups = [sorted(index.cols[a:b].tolist()) for a, b in zip(bounds[:-1], bounds[1:])]
    assert sorted(groups) == sorted(sorted(r.photo_rows) for r in recs)
    assert set(index.starts.tolist()) <= set(index.rec_starts.tolist())


def test_two_photos_of_one_specimen_no_longer_fill_both_places():
    # Species A: one record photographed twice, both photos very close to the query, and
    # one far record. Species B: two records, each fairly close. By photo, A's one specimen
    # takes both places and A wins; by record, A has one close specimen and B two.
    dim = 8
    q = np.zeros(dim)
    q[0] = 1.0
    far = np.zeros(dim)
    far[1] = 1.0
    vecs = unit([q + 0.05 * far, q + 0.06 * far, far,
                 q + 0.5 * far, q + 0.5 * far]).astype(np.float16)
    recs = [Record("a1", "G a", "G", "F", "2026-01-01", "u", [0, 1]),
            Record("a2", "G a", "G", "F", "2026-01-01", "u", [2]),
            Record("b1", "G b", "G", "F", "2026-01-01", "u", [3]),
            Record("b2", "G b", "G", "F", "2026-01-01", "u", [4])]
    index = build_index(recs)
    query = q[None, :].astype(np.float16)
    photo, record = NearestAndMean(), NearestAndMeanRecords()
    photo.fit(vecs, index)
    record.fit(vecs, index)
    # the specimen half alone, where the two rules differ
    photo.weight = record.weight = 1.0
    assert index.species[int(np.argmax(photo.species_scores(query)))] == "G a"
    assert index.species[int(np.argmax(record.species_scores(query)))] == "G b"


def test_record_top_k_is_the_mean_of_the_k_best_records_best_photos():
    vecs, recs = random_reference(2, [[1], [2, 2], [3, 1, 1], [1, 1, 1, 1]])
    index = build_index(recs)
    rng = np.random.default_rng(3)
    sims = rng.uniform(-1, 1, size=(3, len(index.cols))).astype(np.float32)
    bounds = np.append(index.rec_starts, len(index.cols))
    rec_species = np.searchsorted(index.starts, index.rec_starts, side="right") - 1
    for k in (1, 2, 3):
        got = top_k_record_scores_per_photo(sims, index, k)
        for s in range(len(index.species)):
            ours = [i for i in range(len(recs)) if rec_species[i] == s]
            best = np.stack([sims[:, bounds[i]:bounds[i + 1]].max(axis=1) for i in ours], 1)
            want = np.sort(best, axis=1)[:, ::-1][:, :k].mean(axis=1)
            assert np.allclose(got[:, s], want)


def test_the_balanced_average_counts_every_record_once():
    vecs, recs = random_reference(4, [[5, 1], [1], [2, 2, 3]])
    index = build_index(recs)
    by_sp = {}
    for r in recs:
        by_sp.setdefault(r.unit, []).append(r.photo_rows)
    v = vecs.astype(np.float32)
    want = np.stack([unit(sum(unit(v[rows].sum(axis=0)) for rows in by_sp[s]))
                     for s in index.species])
    for chunk in (1, 2, 4096):
        assert np.allclose(record_balanced_species_means(vecs, index, chunk=chunk), want,
                           atol=1e-5)
    # the photo-weighted average differs where records have unequal photo counts
    assert not np.allclose(species_means(vecs, index)[0], want[0], atol=1e-3)


@pytest.mark.parametrize("name,balanced,views", [
    ("nearest+mean-records", False, False),
    ("nearest+mean-records-balanced", True, False),
    ("nearest+mean-views", True, True)])
def test_each_arm_scores_as_written_record_by_record(name, balanced, views):
    vecs, recs = random_reference(5, [[2, 1], [3], [1, 1, 4], [2, 2]])
    index = build_index(recs)
    model = METHODS[name]()
    model.fit(vecs, index)
    query = unit(np.random.default_rng(6).normal(size=(3, vecs.shape[1]))).astype(np.float16)
    want = brute(vecs, recs, query, balanced=balanced, views=views)
    assert np.allclose(model.species_scores(query), want, atol=2e-3)


def test_view_matching_of_one_photo_is_the_balanced_record_arm():
    vecs, recs = random_reference(7, [[2, 1], [3], [1, 1, 4]])
    index = build_index(recs)
    views, balanced = NearestAndMeanViews(), NearestAndMeanRecordsBalanced()
    views.fit(vecs, index)
    balanced.fit(vecs, index)
    one = unit(np.random.default_rng(8).normal(size=(1, vecs.shape[1]))).astype(np.float16)
    assert np.allclose(views.species_scores(one), balanced.species_scores(one), atol=1e-6)


@pytest.mark.parametrize("cls", [NearestAndMeanRecords, NearestAndMeanRecordsBalanced])
@pytest.mark.parametrize("hide", ["whole record", "one photo of a record", "whole species"])
def test_leaving_columns_out_equals_fitting_without_them(cls, hide):
    # Records of several photos each, so a species' two best photos often share a record.
    vecs, recs = random_reference(9, [[4, 1, 4], [3, 3, 2], [1]])
    index = build_index(recs)
    bounds = np.append(index.rec_starts, len(index.cols))
    if hide == "whole record":
        hidden = np.arange(bounds[0], bounds[1])               # species 0's first record
    elif hide == "one photo of a record":
        target = int(np.argmax(np.diff(bounds)))                # a record with 2+ photos
        hidden = np.array([bounds[target]])
    else:
        hidden = np.arange(index.starts[2], len(index.cols))    # the last species, all of it
    hidden_rows = set(index.cols[hidden].tolist())
    model = cls()
    model.fit(vecs, index)
    query = unit(np.random.default_rng(10).normal(size=(6, vecs.shape[1]))).astype(np.float16)
    sims = model.photo_sims(query)
    got = model.leave_out(model.per_photo_scores(sims, query), sims, query, hidden)
    kept = [Record(r.observation_id, r.species, r.genus, r.family, r.validated_on, r.observer,
                   [p for p in r.photo_rows if p not in hidden_rows]) for r in recs]
    kept = [r for r in kept if r.photo_rows]
    refit_index = build_index(kept)
    refit = cls()
    refit.fit(vecs, refit_index)
    want = refit.per_photo_scores(refit.photo_sims(query), query)
    for s, name in enumerate(index.species):
        if name in refit_index.species:
            assert np.allclose(got[:, s], want[:, refit_index.species.index(name)], atol=1e-3)
        else:
            assert np.all(got[:, s] == -np.inf)


def test_the_record_arms_and_their_place_twins_are_research_only():
    for name in RECORD_ARMS:
        assert name in METHODS and f"{name}+place" in METHODS
        assert not method_ready(name) and not method_ready(f"{name}+place")
        twin = METHODS[f"{name}+place"]
        assert twin.calibration_from == name and twin.base_method == name
    assert METHODS["nearest+mean+place"].base_method == "nearest+mean"


def test_a_place_twin_without_place_or_date_answers_as_its_arm():
    vecs, recs = random_reference(11, [[2, 1], [3], [1, 1, 4]])
    index = build_index(recs)
    query = unit(np.random.default_rng(12).normal(size=(2, vecs.shape[1]))).astype(np.float16)
    for name in RECORD_ARMS:
        twin = METHODS[f"{name}+place"](store=None)
        assert type(twin.base) is METHODS[name]
        twin.fit(vecs, index, records=recs)
        arm = METHODS[name]()
        arm.fit(vecs, index)
        assert np.allclose(twin.species_scores(query, None), arm.species_scores(query))


@pytest.mark.parametrize("name", RECORD_ARMS)
def test_each_arm_runs_through_the_evaluation_and_calibrates(name):
    rng = np.random.default_rng(13)
    centres = unit(rng.normal(size=(5, 16)))
    vecs, recs = [], []
    for s in range(5):
        for j in range(6):
            rows = []
            for _ in range(1 + j % 3):
                vecs.append(unit(centres[s] + 0.15 * rng.normal(size=16)))
                rows.append(len(vecs) - 1)
            recs.append(Record(f"{s}-{j}", f"G sp{s}", "G", "F", "2026-01-01", "u", rows))
    vecs = np.asarray(vecs, dtype=np.float16)
    ref = [r for r in recs if not r.observation_id.endswith("-5")]
    test = [r for r in recs if r.observation_id.endswith("-5")]
    out = evaluate(vecs, ref, test, method=name)
    assert out["species"]["all"]["top1"] == 1.0
    assert out["calibration"]["species"]["n"] == 5


def test_per_record_answers_agree_with_the_summary_and_the_fitted_temperature():
    from mycomap_vision import likely
    from mycomap_vision.evaluate import fit_method, rank_scores
    rng = np.random.default_rng(14)
    centres = unit(rng.normal(size=(6, 16)))
    vecs, recs = [], []
    for s in range(6):
        for j in range(5):
            vecs.append(unit(centres[s] + 0.6 * rng.normal(size=16)))
            recs.append(Record(f"{s}-{j}", f"G{s % 2} sp{s}", f"G{s % 2}", "F", "2026-01-01",
                               "u", [len(vecs) - 1]))
    vecs = np.asarray(vecs, dtype=np.float16)
    ref = [r for r in recs if not r.observation_id.endswith(("-3", "-4"))]
    test = [r for r in recs if r.observation_id.endswith(("-3", "-4"))]
    test.append(Record("new", "G0 novum", "G0", "F", "2026-01-01", "u", [0]))
    fitted = fit_method(vecs, ref, "nearest+mean-records")
    out = evaluate(vecs, ref, test, method="nearest+mean-records", fitted=fitted,
                   per_record=True)
    rows = out["per_record"]
    assert [r["observation_id"] for r in rows] == [r.observation_id for r in test]
    hits = sum(a["top"]["species"][:1] == [r.species] for a, r in zip(rows, test))
    assert round(hits / len(test), 4) == out["species"]["all"]["top1"]
    index, model = fitted
    t = out["calibration"]["species"]["temperature"]
    for a, r in zip(rows, test):
        p = likely.probabilities(rank_scores(model.species_scores(vecs[r.photo_rows]),
                                             index, "species"), t)
        assert np.isclose(a["cal"]["species"]["conf"], p.max())
        if r.species in index.labels["species"]:
            want = p[index.labels["species"].index(r.species)]
            assert np.isclose(a["cal"]["species"]["p_true"], want)
        else:
            assert a["cal"]["species"]["p_true"] is None
    assert "per_record" not in evaluate(vecs, ref, test, method="nearest+mean-records",
                                        fitted=fitted)
