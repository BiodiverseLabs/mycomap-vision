"""The review's val measurements (val_review.py): confident-wrong answers, selective
accuracy, genus back-off for unseen species, and the generalisation splits."""

import numpy as np
import pytest

from mycomap_vision import val_review as vr
from mycomap_vision.apiv1 import CONFIDENT_AT


def test_a_confident_answer_for_a_species_with_no_reference_is_counted_wrong():
    conf = [0.95, 0.80, 0.40, 0.75, 0.10]
    right = [True, False, False, False, False]
    depth = [12, 3, 0, 0, 0]
    out = vr.confident_answers(conf, right, depth)
    assert out["threshold"] == CONFIDENT_AT == 0.70
    assert out["all"] == {"n": 5, "confident": 3, "share": 0.6, "wrong": 2,
                          "wrong_share": 0.4, "precision": pytest.approx(1 / 3)}
    assert out["unknown"] == {"n": 3, "confident": 1, "share": pytest.approx(1 / 3),
                              "wrong": 1, "wrong_share": pytest.approx(1 / 3),
                              "precision": 0.0}


def test_the_confidence_rule_is_the_callers_threshold_when_given():
    out = vr.confident_answers([0.75], [False], [0], threshold=0.8)
    assert out["threshold"] == 0.8 and out["all"]["confident"] == 0


def test_a_threshold_is_reached_at_exactly_its_value():
    assert vr.confident_answers([0.70], [True], [5])["all"]["confident"] == 1


def test_selective_accuracy_rises_as_only_confident_answers_are_kept():
    conf = [0.9, 0.8, 0.6, 0.4, 0.2]
    right = [1, 1, 0, 1, 0]
    rows = vr.selective_accuracy(conf, right, thresholds=(0.0, 0.5, 0.85, 0.99))
    assert [r["n"] for r in rows] == [5, 3, 1, 0]
    assert [r["coverage"] for r in rows] == [1.0, 0.6, 0.2, 0.0]
    assert rows[0]["accuracy"] == 0.6 and rows[1]["accuracy"] == pytest.approx(2 / 3)
    assert rows[2]["accuracy"] == 1.0 and rows[3]["accuracy"] is None


def test_genus_back_off_counts_only_finds_whose_species_has_no_reference():
    assert vr.genus_backoff([True, False, True, True], [0, 0, 5, 0]) == \
        {"n": 3, "rate": pytest.approx(2 / 3)}
    assert vr.genus_backoff([True], [3]) == {"n": 0, "rate": None}


def test_an_observer_with_no_reference_record_is_unseen_and_no_observer_is_not():
    got = vr.unseen_observer(["ann", "bob", None, ""], ["ann", "cy", None])
    assert got.tolist() == [False, True, False, False]


@pytest.mark.parametrize("months,want", [
    (None, [False, False, True, False, False]),         # observed after the cutoff only
    (3, [False, True, True, False, False]),             # after 2026-05-17
    (12, [True, True, True, False, False])])
def test_the_recent_cohort_is_observed_within_months_of_the_cutoff(months, want):
    days = ["2025-12-01", "2026-06-30", "2026-09-01", None, "not a date"]
    assert vr.recent_cohort(days, "2026-08-17", months).tolist() == want


def test_a_months_window_ending_in_a_short_month_stays_a_real_date():
    # 31 May less 3 months: 28 February (2026), not 31 February
    assert vr.recent_cohort(["2026-02-28", "2026-03-01"], "2026-05-31", 3).tolist() == \
        [False, True]


def test_a_val_photo_within_four_bits_of_a_training_photo_is_flagged():
    train = np.array([0b0, 0xFFFF_0000_0000_0000], dtype=np.uint64)
    val = {"near": np.uint64(0b1111),            # 4 bits from train[0]
           "far": np.uint64(0b11111),            # 5 bits
           "copy": np.uint64(0xFFFF_0000_0000_0000)}
    got = vr.near_train_flags(val, train, chunk=1)
    assert got == {"near": True, "far": False, "copy": True}
    assert vr.near_train_flags(val, train, max_distance=5)["far"] is True
