"""Cross-post twins as one specimen (twins.py): the groups file every record-grouped scoring
reads, and the commands that refuse to score such a method without it."""

import numpy as np
import pytest

from mycomap_vision import evaluate, twins
from mycomap_vision.evaluate import Record


def test_groups_are_the_connected_components_of_the_pairs_in_the_kept_records():
    pairs = [("mo:1", "10"), ("mo:2", "10"), ("mo:3", "30"), ("mo:4", "40")]
    groups = twins.groups_from_pairs(pairs, keep={"mo:1", "mo:2", "10", "mo:3", "30"})
    assert groups == {"mo:1": "twin-10", "mo:2": "twin-10", "10": "twin-10",
                      "mo:3": "twin-30", "30": "twin-30"}
    # a pair with a record outside `keep` (val, an erratum's) makes no group
    assert "mo:4" not in groups and "40" not in groups


def test_the_file_round_trips_and_the_same_groups_give_the_same_bytes(tmp_path):
    groups = {"mo:2": "twin-10", "10": "twin-10", "mo:1": "twin-10"}
    a = twins.write(groups, tmp_path / "a.tsv")
    b = twins.write(dict(reversed(list(groups.items()))), tmp_path / "b.tsv")
    assert a["sha256"] == b["sha256"] and a["records"] == 3 and a["groups"] == 1
    assert twins.read(tmp_path / "a.tsv") == groups
    assert twins.cite(tmp_path / "a.tsv")["sha256"] == a["sha256"]
    with pytest.raises(FileExistsError):
        twins.write(groups, tmp_path / "a.tsv")


def test_a_file_without_both_columns_is_refused(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("record\tgroup\nx\ty\n", encoding="utf-8")
    with pytest.raises(ValueError, match="observation_id and specimen"):
        twins.read(p)


def rec(oid):
    return Record(oid, "G a", "G", "F", "2026-01-01", "u", [0])


def test_a_twin_whose_other_side_is_not_in_the_reference_is_an_ordinary_record():
    groups = {"mo:1": "twin-10", "10": "twin-10", "mo:5": "twin-50", "50": "twin-50"}
    assert twins.specimen_of(groups, [rec("mo:1"), rec("10"), rec("mo:5"), rec("7")]) == \
        {"mo:1": "twin-10", "10": "twin-10"}


def test_only_the_record_grouped_methods_and_their_place_twins_are_grouped():
    for m in ("nearest+mean-records-balanced", "nearest+mean-records-balanced+place",
              "nearest+mean-records", "nearest+mean-views+place"):
        assert twins.is_grouped(m)
    for m in ("nearest", "nearest+mean", "nearest+mean+place", "species-mean"):
        assert not twins.is_grouped(m)


def test_compare_refuses_a_grouped_method_without_twin_groups(monkeypatch):
    monkeypatch.setattr(evaluate, "shared_records", lambda *a, **k: pytest.fail("ran"))
    with pytest.raises(ValueError, match="--twins FILE"):
        evaluate.compare(None, ["m1"], ["nearest+mean", "nearest+mean-records-balanced"])


def test_compare_groups_twins_for_the_grouped_method_only_and_cites_the_file(monkeypatch):
    # A species' twin pair (one find, two records) close to the test photo, and two
    # distinct finds of another species a little further: grouped, the twin counts once.
    dim = 8
    q = np.zeros(dim)
    q[0] = 1.0
    far = np.zeros(dim)
    far[1] = 1.0
    unit = lambda v: v / np.linalg.norm(v)  # noqa: E731
    vecs = np.asarray([unit(q + 0.05 * far), unit(q + 0.06 * far), unit(far),
                       unit(q + 0.5 * far), unit(q + 0.5 * far), q], dtype=np.float16)
    ref = [Record("mo:1", "G a", "G", "F", "2026-01-01", "u", [0]),
           Record("1", "G a", "G", "F", "2026-01-01", "u", [1]),
           Record("a2", "G a", "G", "F", "2026-01-01", "u", [2]),
           Record("b1", "G b", "G", "F", "2026-01-01", "u", [3]),
           Record("b2", "G b", "G", "F", "2026-01-01", "u", [4])]
    test = [Record("t", "G b", "G", "F", "2026-09-01", "u", [5])]
    shared = evaluate.SharedSet({"m1": (np.arange(6), vecs)}, ref, test, "2026-08-01",
                                "rs", 6)
    saved = []
    monkeypatch.setattr(evaluate, "check_not_trained_on_test", lambda *a: None)
    monkeypatch.setattr(evaluate, "save_run", lambda conn, cid, b, m, sh, td, ap, fp, extra:
                        saved.append((m, ap, extra)) or {})
    fits = []
    real_fit = evaluate.fit_method

    def spy(*a, **k):
        fits.append((a[2], k.get("specimen_of")))
        out = real_fit(*a, **k)
        out[1].weight = 1.0                       # the specimen half alone
        return out
    monkeypatch.setattr(evaluate, "fit_method", spy)
    groups = {"mo:1": "twin-1", "1": "twin-1"}
    evaluate.compare(None, ["m1"], ["nearest+mean-records-balanced", "nearest+mean"],
                     shared=shared, twin_groups=groups,
                     twins_cite={"file": "t.tsv", "sha256": "abc"}, sets=False,
                     log=lambda s: None)
    assert fits[0] == ("nearest+mean-records-balanced", groups)
    assert fits[1] == ("nearest+mean", None)
    by = {m: (ap, extra) for m, ap, extra in saved}
    assert by["nearest+mean-records-balanced"][1]["twins"] == \
        {"file": "t.tsv", "sha256": "abc", "records_grouped": 2}
    assert "twins" not in (by["nearest+mean"][1] or {})
    assert by["nearest+mean-records-balanced"][0]["species"]["all"]["top1"] == 1.0
