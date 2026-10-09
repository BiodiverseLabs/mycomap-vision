"""The standard summary of a benchmark report (heldout_summary.py, Steve 2026-10-09):
top 1 / 3 / 5 / 10 by strict, s.l. and complex rows, species by the true species'
reference records, iNat on the same records or "not run", printed first."""

import json
import sys
import types

import pytest
from test_heldout import (NAME, FakeInat, fetched, frozen, predicted, reference, reported,
                          scored_world)

import mycomap_vision
from mycomap_vision import cli, config, heldout, heldout_summary

ROWS = ["species strict", "species s.l.", "species complex (beta)", "genus strict", "genus s.l."]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def with_name_equiv(monkeypatch):
    """A stand-in for feat/name-equivalence's module: any Russula is Russula s.l."""
    fake = types.ModuleType("mycomap_vision.name_equiv")
    fake.species_match = lambda a, t: {"strict": a == t,
                                       "sl": a.split()[:1] == t.split()[:1] == ["Russula"],
                                       "complex": a.split()[:1] == t.split()[:1]}
    fake.genus_match = lambda a, t: {"strict": a == t, "sl": a[:3] == t[:3]}
    monkeypatch.setitem(sys.modules, "mycomap_vision.name_equiv", fake)
    # The real module is merged: the package attribute is found before sys.modules.
    monkeypatch.setattr(mycomap_vision, "name_equiv", fake, raising=False)
    return fake


def vision_only(conn, tmp_path, monkeypatch):
    root = reference(conn, tmp_path / "emb")
    frozen(conn, tmp_path)
    fetched(conn, tmp_path, monkeypatch)
    predicted(conn, tmp_path, root)


def test_answers_are_stored_ten_deep(conn, tmp_path, monkeypatch):
    scored_world(conn, tmp_path, monkeypatch)
    depths = {(b, json.loads(r)["top_k"]) for b, r in conn.execute(
        "select backbone, result_json from heldout_predictions")}
    assert depths == {("toy", 10), ("external:inat-cv", 10)}


def test_each_model_gets_top_1_3_5_10_by_strict_s_l_and_complex_rows(conn, tmp_path,
                                                                     monkeypatch,
                                                                     with_name_equiv):
    vision_only(conn, tmp_path, monkeypatch)
    table = reported(conn, tmp_path)["summary"]["models"]["toy/nearest"]
    assert list(table["rows"]) == ROWS and table["stored_depth"] == 10
    rows = table["rows"]
    assert set(rows["species strict"]) == {"n", "top1", "top3", "top5", "top10"}
    # dev: 101 Russula emetica (right), 105 Tubariua hiemalis (never right); 103 is one word.
    assert (rows["species strict"]["n"], rows["genus strict"]["n"]) == (2, 3)
    assert rows["species strict"]["top1"]["right"] == 1
    # A looser level never scores below a stricter one.
    for k in ("top1", "top3", "top5", "top10"):
        assert (rows["species complex (beta)"][k]["right"] >= rows["species s.l."][k]["right"]
                >= rows["species strict"][k]["right"])
        assert rows["genus s.l."][k]["right"] >= rows["genus strict"][k]["right"]


def test_without_name_equiv_the_s_l_rows_say_it_is_not_merged(conn, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "mycomap_vision.name_equiv", None)
    monkeypatch.delattr(mycomap_vision, "name_equiv", raising=False)
    vision_only(conn, tmp_path, monkeypatch)
    rows = reported(conn, tmp_path)["summary"]["models"]["toy/nearest"]["rows"]
    assert rows["species s.l."] == {"n": 2, "note": heldout_summary.EQUIV_MISSING}
    assert rows["species strict"]["top1"]["n"] == 2


class FakeLabeller:
    def same(self, a, b):
        return a == b


def truth(species, genus):
    return heldout.Truth(species, genus, "F", species or genus, False, None)


def test_answers_stored_five_deep_say_not_stored_at_top_10_unless_the_full_answer_goes_deeper():
    wrong = [{"name": f"A w{i}"} for i in range(5)]
    old = {"1": {"species": wrong, "genus": [{"name": "A"}]}}         # before top_k was kept
    truths = {"1": truth("A right", "A")}
    rows = heldout_summary.ladder(old, {"1"}, truths, FakeLabeller(), False, None)["rows"]
    assert rows["species strict"]["top10"] is None and rows["species strict"]["top5"]["n"] == 1
    m = ("toy", "nearest", "-", "large")
    judged = {m: {"1": {"species": (False, False, "A w0", 0.1), "genus": (True, True, "A", 1)}}}
    feats = {"1": {"species reference records": "1-4"}}
    deeper = {m: {"1": {"ranks": {"species": wrong + [{"name": "A right"}],
                                  "genus": [{"name": "A"}, {"name": "B"}]}}}}
    head = {"benchmark": "b", "split": "dev", "sealed": False, "released_at": None,
            "records": 1, "scored_records": 1}
    s = heldout_summary.standard_summary(head, {m: old}, judged, truths, feats, FakeLabeller(),
                                         deeper)
    top = s["models"]["toy/nearest"]["rows"]["species strict"]
    assert (top["top5"]["right"], top["top10"]["right"]) == (0, 1)


def test_a_right_answer_counts_from_its_own_place_on_and_never_before():
    def answer(right_at):
        cands = [{"name": f"A w{i}"} for i in range(10)]
        cands[right_at] = {"name": "A right"}
        return {"species": cands, "genus": [{"name": "A"}], "top_k": 10}
    results = {str(i): answer(at) for i, at in enumerate((0, 1, 2, 3, 4, 5, 9))}
    truths = {oid: truth("A right", "A") for oid in results}
    row = heldout_summary.ladder(results, set(results), truths, FakeLabeller(), False,
                                 None)["rows"]["species strict"]
    assert [row[f"top{k}"]["right"] for k in (1, 3, 5, 10)] == [1, 3, 5, 7]


def test_species_top_1_and_5_by_the_true_species_reference_records(conn, tmp_path, monkeypatch):
    vision_only(conn, tmp_path, monkeypatch)
    depth = reported(conn, tmp_path)["summary"]["models"]["toy/nearest"][
        "species_by_true_species_reference_records"]
    assert list(depth) == ["0", "1-4", "5-19", "20-99", "100+"]
    # Russula emetica has 2 reference records, Tubariua hiemalis none.
    assert (depth["1-4"]["n"], depth["1-4"]["top1"]["right"], depth["1-4"]["top5"]["right"]) == (
        1, 1, 1)
    assert (depth["0"]["n"], depth["0"]["top5"]["right"]) == (1, 0)
    assert depth["100+"]["n"] == 0


def test_inat_is_shown_on_the_same_records_as_vision_or_said_not_run(conn, tmp_path,
                                                                    monkeypatch):
    vision_only(conn, tmp_path, monkeypatch)
    s = reported(conn, tmp_path)["summary"]
    assert s["inat_same_records"] == {"status": "not run"}
    assert "iNat CV: not run" in heldout_summary.format_summary(s)
    heldout.inat_cv(conn, NAME, ["101", "105"], FakeInat(), log=lambda s: None)
    same = reported(conn, tmp_path, stamp="inat")["summary"]["inat_same_records"]
    assert same["records"] == 2
    assert set(same["models"]) == {"toy/nearest", "external:inat-cv/vision-max",
                                   "external:inat-cv/combined-max"}
    assert all(t["records"] == 2 for t in same["models"].values())


def test_mv_heldout_report_prints_the_summary_first_with_every_tables_n_then_the_json(
        conn, tmp_path, monkeypatch, capsys):
    scored_world(conn, tmp_path, monkeypatch)
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "manifest.sqlite")
    cli.main(["heldout", "report", "--name", NAME, "--split", "dev"])
    text = capsys.readouterr().out
    head, _, rest = text.partition("\n{")
    assert head.startswith(f"{NAME} / dev (development, not sealed)")
    assert "toy/nearest  (n = 3 records)" in head
    assert "species, by the true species' reference records" in head
    assert "iNat CV and Vision on the same records  (n = 3 records)" in head
    for label in ROWS:
        assert label in head
    summary = json.loads("{" + rest.split("\n-> ")[0])
    assert summary["models"]["toy/nearest"]["records"] == 3
