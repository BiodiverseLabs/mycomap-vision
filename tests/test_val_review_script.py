"""The v1 val review script (scripts/val_review/run_v1.py): its per-find cells and its
calibration figures, served against own and cross-fitted."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "val_review" / "run_v1.py"


@pytest.fixture(scope="module")
def run():
    spec = importlib.util.spec_from_file_location("val_review_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_hit_at_k_counts_at_every_deeper_k_and_strict_is_the_exact_name(run):
    row = {"top": {"species": ["Russula a", "Russula b", "Russula c"], "genus": ["Lactarius",
                                                                              "Russula"]}}
    o = run.hits(row, {"species": "Russula b", "genus": "Russula"})
    assert [o[f"species_strict@{k}"] for k in (1, 3, 5, 10)] == [False, True, True, True]
    assert o["genus_strict@1"] is False and o["genus_strict@3"] is True


def test_a_wrongly_confident_answer_shows_in_the_served_ece(run):
    keys = [str(i) for i in range(100)]
    rows = {k: {"served": {"species": {"conf": 0.95, "p_true": 0.5, "in_set": i < 80,
                                       "set_size": 2}}}
            for i, k in enumerate(keys)}
    o = {k: {"species_strict@1": i % 2 == 0} for i, k in enumerate(keys)}
    b = run.calibration_block(keys, rows, o, "served")
    assert b["ece"] == pytest.approx(0.45, abs=0.01) and b["accuracy"] == 0.5
    assert b["likely_coverage"] == 0.8 and b["likely_mean_size"] == 2
    assert b["nll"] == pytest.approx(-np.log(0.5))


def test_the_crossfit_temperature_comes_from_the_other_half_of_the_observers(run):
    from mycomap_vision.evaluate import T_GRID
    rng = np.random.default_rng(0)
    keys, rows, o, finds = [], {}, {}, {}
    for i in range(400):
        k = str(i)
        scores = rng.normal(size=20) * 0.05
        true = int(rng.integers(20))
        scores[true] += 0.06                                # a model with some skill
        lse = [float(np.log(np.exp(scores / t).sum())) for t in T_GRID]
        rows[k] = {"grid": {"species": {"best": float(scores.max()),
                                        "true": float(scores[true]), "lse": lse}}}
        o[k] = {"species_strict@1": bool(np.argmax(scores) == true)}
        finds[k] = {"observer": f"u{i % 40}"}
        keys.append(k)
    cf = run.crossfit(keys, rows, o, finds)
    assert cf["n"] == 400 and len(cf["temperatures"]) == 2
    assert all(t in T_GRID for t in cf["temperatures"])
    assert cf["ece"] < 0.1
