import numpy as np

from mycomap_vision.evaluate import METHODS, Record, build_index, evaluate
from mycomap_vision.methods import Hybrid, LinearHead


def clusters(sizes, dim=16, spread=0.15, seed=0):
    """Unit vectors around one random centre per species; sizes = photos per species."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(len(sizes), dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    vecs, recs = [], []
    for s, n in enumerate(sizes):
        for j in range(n):
            v = centres[s] + spread * rng.normal(size=dim)
            vecs.append(v / np.linalg.norm(v))
            recs.append(Record(f"{s}-{j}", f"G{s % 3} sp{s}", f"G{s % 3}", "F", "2026-01-01",
                               "u", [len(vecs) - 1]))
    return np.asarray(vecs, dtype=np.float16), recs, centres


def test_the_trained_methods_are_registered():
    assert {"nearest", "species-mean", "linear", "hybrid"} <= set(METHODS)


def test_the_linear_head_learns_separable_species():
    vecs, recs, centres = clusters([20] * 6)
    head = LinearHead()
    head.fit(vecs, build_index(recs))
    index = build_index(recs)
    hits = sum(int(np.argmax(head.species_scores(c[None, :].astype(np.float16))) ==
                   index.species.index(f"G{s % 3} sp{s}")) for s, c in enumerate(centres))
    assert hits == 6


def test_the_frequency_correction_takes_most_of_a_common_species_lead_away():
    # 300 photos of one species, 3 of another; a photo exactly between them.
    vecs, recs, centres = clusters([300, 3], spread=0.05, seed=1)
    index = build_index(recs)
    mid = centres[0] + centres[1]
    mid = (mid / np.linalg.norm(mid))[None, :].astype(np.float16)
    common, rare = index.species.index("G0 sp0"), index.species.index("G1 sp1")

    def lead(balanced):
        head = LinearHead()
        head.balanced = balanced
        head.fit(vecs, index)
        s = head.species_scores(mid)
        return s[common] - s[rare]

    # The frequency correction must take most of the common species' lead away
    # (measured: 11 with plain softmax, 2 with balanced).
    assert lead(balanced=True) < lead(balanced=False) / 3


def test_the_hybrid_scores_every_species_and_ranks_the_right_one_first():
    vecs, recs, centres = clusters([12, 12, 1, 12], seed=2)
    index = build_index(recs)
    hybrid = Hybrid()
    hybrid.fit(vecs, index)
    q = centres[2][None, :].astype(np.float16)            # the single-record species
    scores = hybrid.species_scores(q)
    assert scores.shape == (len(index.species),)
    assert index.species[int(np.argmax(scores))] == "G2 sp2"


def test_trained_methods_run_through_the_evaluation_and_calibrate():
    vecs, recs, _ = clusters([6] * 5, seed=3)
    ref = [r for r in recs if not r.observation_id.endswith("-5")]
    test = [Record(r.observation_id, r.species, r.genus, r.family, "2026-09-20", "u",
                   r.photo_rows) for r in recs if r.observation_id.endswith("-5")]
    for method in ("linear", "hybrid"):
        out = evaluate(vecs, ref, test, method=method)
        assert out["species"]["all"]["top1"] == 1.0
        assert out["calibration"]["species"]["n"] == 5


def test_a_comparison_fits_each_trained_method_once_for_both_scores(monkeypatch):
    from mycomap_vision import evaluate as ev
    calls = []
    real = LinearHead.fit

    def counting(self, vectors, index):
        calls.append(1)
        return real(self, vectors, index)

    monkeypatch.setattr(LinearHead, "fit", counting)
    vecs, recs, _ = clusters([6] * 3, seed=4)
    ref = [r for r in recs if not r.observation_id.endswith("-5")]
    test = [r for r in recs if r.observation_id.endswith("-5")]
    fitted = ev.fit_method(vecs, ref, "linear")
    ev.evaluate(vecs, ref, test, method="linear", fitted=fitted)
    ev.evaluate(vecs, ref, test, method="linear", fitted=fitted, first_photo_only=True)
    assert len(calls) == 1


def test_trained_weights_are_saved_and_reused_for_the_same_reference_set(tmp_path, monkeypatch):
    from mycomap_vision.identify import cache_key, fit_cached
    vecs, recs, centres = clusters([8] * 4, seed=5)
    index = build_index(recs)
    trained = []
    real = LinearHead.fit

    def counting(self, vectors, index, state=None):
        if state is None:
            trained.append(1)
        return real(self, vectors, index, state=state)

    monkeypatch.setattr(LinearHead, "fit", counting)
    key = cache_key("bb", "hybrid", np.arange(len(index.cols))[index.cols], index.species)
    first, second = Hybrid(), Hybrid()
    assert fit_cached(first, vecs, index, recs, tmp_path, key) is False
    assert fit_cached(second, vecs, index, recs, tmp_path, key) is True
    assert trained == [1]
    q = centres[1][None, :].astype(np.float16)
    assert np.allclose(first.species_scores(q), second.species_scores(q))


def test_a_changed_reference_set_gets_a_new_cache_key():
    from mycomap_vision.identify import cache_key
    a = cache_key("bb", "linear", np.array([1, 2, 3]), ["A x", "B y"])
    assert a != cache_key("bb", "linear", np.array([1, 2, 4]), ["A x", "B y"])
    assert a != cache_key("bb", "linear", np.array([1, 2, 3]), ["A x", "B z"])
    assert a != cache_key("other", "linear", np.array([1, 2, 3]), ["A x", "B y"])
