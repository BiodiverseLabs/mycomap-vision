"""The label-audit re-scorer (tools/label_audit_rescore.py): edits to reference labels change
the score the way the audit relies on."""

import importlib.util
from collections import defaultdict
from pathlib import Path

import numpy as np

from mycomap_vision.heldout import Truth

spec = importlib.util.spec_from_file_location(
    "label_audit_rescore", Path(__file__).parents[1] / "tools" / "label_audit_rescore.py")
lar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lar)


def scorer(truth_species="Russula variata"):
    """Two query photos of one record. Reference records: 0 'Russula sp. IN67' (its photo is the
    nearest), 1 'Russula variata', 2 'Lactarius fallax'."""
    s = lar.Scorer.__new__(lar.Scorer)
    s.sp = np.array(["Russula sp. 'IN67'", truth_species, "Lactarius fallax"], dtype=object)
    s.ge = np.array(["Russula", "Russula", "Lactarius"], dtype=object)
    s.tx = np.array(["", "", ""], dtype=object)
    s.ref_oid = ["10", "11", "12"]
    s.rec_of = np.array([0, 1, 2])
    s.R = np.eye(3, 4, dtype=np.float16)
    s.Q = np.array([[0.9, 0.5, 0.1, 0], [0.8, 0.6, 0.2, 0]], dtype=np.float32)
    s.TI = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.int32)
    s.TS = np.array([[0.9, 0.5, 0.1], [0.8, 0.6, 0.2]], dtype=np.float32)
    s.photos = defaultdict(list, {"1": [0, 1]})
    s.truth = {"1": Truth(truth_species, "Russula", "Russulaceae", truth_species, False, None)}
    return s


def test_a_species_split_over_two_labels_is_a_miss_until_the_labels_are_merged():
    s = scorer()
    ix = s.index()
    assert not s.evaluate(ix, "nearest")["species strict"][0, 0]
    merged = s.evaluate(ix, "nearest", merge={"Russula sp. 'IN67'": "Russula variata"})
    assert merged["species strict"][0, 0]


def test_dropping_a_wrong_reference_record_lets_the_right_species_win():
    s = scorer()
    assert s.evaluate(s.index(exclude=frozenset({0})), "nearest")["species strict"][0, 0]


def test_relabelling_a_reference_record_to_its_current_name_counts():
    s = scorer()
    ix = s.index(relabel={0: ("Russula variata", "Russula", "")})
    assert s.evaluate(ix, "nearest")["species strict"][0, 0]
    assert ix["count"]["Russula variata"] == 2


def test_depth_bands_come_from_the_baseline_counts_when_given():
    s = scorer()
    out = s.evaluate(s.index(exclude=frozenset({1})), "nearest",
                     depth_from={"Russula variata": 7})
    assert list(out["depth"]) == ["5-19"]


def test_bands_match_the_standard_report():
    assert [lar.band(n) for n in (0, 1, 4, 5, 19, 20, 99, 100)] == [
        "0", "1-4", "1-4", "5-19", "5-19", "20-99", "20-99", "100+"]
