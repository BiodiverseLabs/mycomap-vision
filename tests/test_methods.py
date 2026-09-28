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
