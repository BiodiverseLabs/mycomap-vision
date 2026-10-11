"""The fungus gate's tuning rule (fungusgate_tune.py): declared before looking, DEV only."""

import numpy as np
import pytest

from mycomap_vision import fungusgate_tune as tune


def recs(*maxes_and_mins):
    return [np.array(v, dtype=float) for v in maxes_and_mins]


def test_not_a_fungus_is_set_where_no_dev_record_loses_its_answer():
    # One record's best photo is 0.32: "not a fungus" must stay at or below it.
    c = tune.choose_thresholds(recs([0.9, 0.2], [0.32], [0.8]), grid=(0.1, 0.2, 0.3, 0.4, 0.5))
    assert c.not_fungus == 0.3 and c.no_answer == 0
    assert c.photo_left_out == 1                    # the 0.2 photo of the first record


def test_not_sure_stays_within_the_share_of_dev_records_shown_without_a_chance():
    scores = recs(*([[0.9]] * 198 + [[0.42], [0.47]]))
    c = tune.choose_thresholds(scores, grid=(0.1, 0.2, 0.3, 0.4, 0.45, 0.5), max_unsure_share=0.005)
    assert c.not_fungus == 0.4                      # 0.42 is the lowest best photo
    assert c.unsure == 0.45 and c.unsure_records == 1   # 0.5 would put 2 of 200 (1%) unsure


def test_a_score_that_loses_a_dev_record_at_every_grid_value_is_not_chosen():
    c = tune.choose_thresholds(recs([0.01], [0.9]), grid=(0.05, 0.1))
    assert c.not_fungus is None and c.unsure is None


@pytest.mark.parametrize("ancestors,expected", [
    ("[48460, 47170, 48250]", "fungi"), ("[48460, 1, 2, 355675, 3]", "animal"),
    ("48460/47126/211194", "plant"), ("[48460, 47686]", "other"), (None, "other"),
])
def test_kingdom_comes_from_the_ancestor_ids(ancestors, expected):
    assert tune.kingdom(ancestors) == expected


def test_the_not_a_fungus_manifest_keeps_licence_and_a_checksum_for_every_photo(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"photo-a")
    out = tmp_path / "m.csv"
    tune.photo_manifest([{"file": "a.jpg", "license": "cc0", "owner": "x", "url": "u"}],
                        tmp_path, out)
    row = out.read_text(encoding="utf-8").splitlines()[1].split(",")
    assert "cc0" in row and row[-1] == tune.sha256_file(tmp_path / "a.jpg")
