"""The first official numbers of a model trained on a dataset release: Steve's standard
tables for a comparison (standard_report.py), with s.l. and complex at top 1/3/5/10, and the
pre-declared sanity gate a model passes before it is served (sanity.py)."""

import json

import numpy as np
import pytest
from test_finetune_heads import write_model
from test_trainer_release import cut, embed  # noqa: F401  (the fixture)

from mycomap_vision import evaluate, finetune, sanity, standard_report
from mycomap_vision.evaluate import Record


def unit(a):
    return [np.cos(a), np.sin(a)]


def test_s_l_and_complex_count_a_near_name_anywhere_in_the_top_k():
    # References: Amanita closest to the query, then Russula xerampelina, then Boletus.
    vectors = np.asarray([unit(0.3), unit(0.0), unit(2.0), unit(0.25)], dtype=np.float32)
    ref = [Record("1", "Amanita muscaria", "Amanita", "Am", "2026-01-01", "a", [0]),
           Record("2", "Russula xerampelina", "Russula", "Ru", "2026-01-01", "a", [1]),
           Record("3", "Boletus edulis", "Boletus", "Bo", "2026-01-01", "a", [2])]
    # The truth is spelled with the other gender ending: never strict, s.l. at rank 2.
    test = [Record("9", "Russula xerampelinus", "Russula", "Ru", "2026-09-01", "b", [3])]
    out = evaluate.evaluate(vectors, ref, test, method="nearest", name_scores=True, sets=False)
    eq = out["standard_equivalence"]
    assert eq["species"]["n"] == 1
    assert eq["species"]["strict"] == {"top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0}
    assert eq["species"]["sl"] == {"top1": 0.0, "top3": 1.0, "top5": 1.0, "top10": 1.0}
    assert eq["species"]["complex"]["top3"] == 1.0          # looser levels count stricter hits
    assert eq["genus"]["strict"]["top1"] == 0.0 and eq["genus"]["strict"]["top3"] == 1.0
    # The strict row is the standard table's exact match, as evaluate's own standard block.
    assert out["standard"]["species"]["all"]["top3"] == eq["species"]["strict"]["top3"]


def compared(conn, store, tmp_path, monkeypatch, rel, methods=("nearest", "nearest+mean")):
    from mycomap_vision import config
    root = tmp_path / "emb"
    embed(conn, store, root, "base")
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    meta = write_model(tmp_path / "data" / "models", "base")
    meta.update(finetune.release_fields(rel))
    (tmp_path / "data" / "models" / "base.json").write_text(json.dumps(meta))
    out = evaluate.compare(conn, ["base"], list(methods), embeddings_root=root, release=rel,
                           name_scores=True, log=lambda s: None)
    return out, root


def test_the_standard_summary_names_the_release_and_gives_every_table_its_n(cut, tmp_path,
                                                                           monkeypatch):
    conn, _, store, rel, _ = cut
    out, _ = compared(conn, store, tmp_path, monkeypatch, rel)
    s = standard_report.summary(conn, out["comparison_id"])
    assert s["dataset_release"]["release_hash"] == rel.release_hash
    assert s["test_records"] == 2 and s["reference_records"] == 6
    table = s["photos_only"]["base / nearest+mean"]
    assert set(table["rows"]) == {"species strict", "species s.l.", "species complex (beta)",
                                  "genus strict", "genus s.l."}
    assert all(r["n"] == 2 for r in table["rows"].values())
    assert table["species_by_true_species_reference_records"]["1-4"]["n"] == 2
    assert s["inat_same_records"] == {"status": "not run"}
    text = standard_report.format_summary(s)
    assert "DRAFT" in text and "top 10" in text and "iNat CV on the same records: not run" in text


def test_without_name_scores_the_s_l_rows_say_so_instead_of_showing_nothing(cut, tmp_path,
                                                                            monkeypatch):
    conn, _, store, rel, _ = cut
    embed(conn, store, tmp_path / "emb", "base")
    out = evaluate.compare(conn, ["base"], ["nearest"], embeddings_root=tmp_path / "emb",
                           release=rel, log=lambda s: None)
    rows = standard_report.summary(conn, out["comparison_id"])["photos_only"]["base / nearest"]
    assert rows["rows"]["species s.l."]["note"] == standard_report.NEEDS_NAME_SCORES


def test_a_place_method_goes_in_the_photos_and_place_table():
    assert standard_report.uses_place("nearest+mean+place")
    assert standard_report.uses_place("nearest+prior")
    assert not standard_report.uses_place("nearest+mean")
    assert not standard_report.uses_place("ft-head")


def test_a_comparison_off_any_release_says_it_is_exploratory(conn, tmp_path):
    from test_finetune import embedded
    store, root = embedded(conn, tmp_path)
    out = evaluate.compare(conn, ["base"], ["nearest"], embeddings_root=root, log=lambda s: None)
    text = standard_report.format_summary(standard_report.summary(conn, out["comparison_id"]))
    assert "NOT on a dataset release" in text


# --- the sanity gate -------------------------------------------------------------------

def gate(conn, rel, cid, root, models_root, **kw):
    return sanity.run(conn, "base", rel, cid, embeddings_root=root, models_root=models_root,
                      **kw)


def test_a_whole_working_model_passes_the_sanity_gate(cut, tmp_path, monkeypatch):
    conn, _, store, rel, _ = cut
    out, root = compared(conn, store, tmp_path, monkeypatch, rel)
    monkeypatch.setattr(sanity, "PLAUSIBLE_SPECIES", (0.0, 1.0))   # two val records
    monkeypatch.setattr(sanity, "MAX_TOP1_SHARE", 0.5)
    r = gate(conn, rel, out["comparison_id"], root, tmp_path / "data" / "models")
    failed = [c for c in r["checks"] if not c["ok"]]
    assert r["passed"], failed
    assert {c["check"] for c in r["checks"]} >= {"files", "release", "version", "heads",
                                                 "embeddings", "species top-1", "genus top-1",
                                                 "family top-1", "calibration", "classes"}
    assert "PASS" in sanity.format_result(r)


def test_a_model_trained_on_another_release_fails(cut, tmp_path, monkeypatch):
    conn, _, store, rel, _ = cut
    out, root = compared(conn, store, tmp_path, monkeypatch, rel)
    path = tmp_path / "data" / "models" / "base.json"
    meta = json.loads(path.read_text())
    meta["dataset_release"]["release_hash"] = "0" * 64
    path.write_text(json.dumps(meta))
    checks = {c["check"]: c for c in sanity.check_files("base", rel, tmp_path / "data" / "models")}
    assert not checks["release"]["ok"]


def test_a_model_without_heads_or_with_a_dead_class_fails(cut, tmp_path):
    rel = cut[3]
    write_model(tmp_path / "old", "base", heads=False)
    assert not sanity.check_heads("base", rel, tmp_path / "old")["ok"]
    write_model(tmp_path / "new", "base")
    assert sanity.check_heads("base", rel, tmp_path / "new")["ok"]
    finetune.save_heads(tmp_path / "new" / "base.heads.npz",
                        {"species": np.asarray([unit(1.0), [0.0, 0.0]]),
                         "genus": np.asarray([unit(1.0), unit(2.0)]),
                         "family": np.asarray([unit(1.0)])},
                        {"species": ["A x", "B y"], "genus": ["A", "B"], "family": ["F"]}, 20.0)
    bad = sanity.check_heads("base", rel, tmp_path / "new")
    assert not bad["ok"] and "all-zero" in bad["detail"]


def test_missing_embeddings_fail_the_gate(cut, tmp_path):
    from mycomap_vision.embed import embed_photos, photos_to_embed
    from test_models_and_scoreboard import Const
    conn, _, store, rel, _ = cut
    root = tmp_path / "emb"
    embed(conn, store, root, "base")
    assert sanity.check_embeddings(conn, "base", rel, root)["ok"]
    todo = photos_to_embed(conn, "short", "large", store.location, "local")
    embed_photos(conn, store, Const("short"), todo[:-1], root / "short", log=lambda s: None)
    r = sanity.check_embeddings(conn, "short", rel, root)
    assert not r["ok"] and "1 missing" in r["detail"]


def test_implausible_numbers_fail_the_gate():
    def summary(sp1, ge1, deep, thin):
        rows = {"species strict": {"n": 500, "top1": sp1, "top3": sp1 + .1, "top5": sp1 + .15,
                                   "top10": sp1 + .2},
                "genus strict": {"n": 500, "top1": ge1}}
        return {"comparison_id": "c", "photos_only": {"m / nearest+mean": {
            "rows": rows, "species_by_true_species_reference_records": {
                "1-4": {"top1": thin}, "100+": {"top1": deep}}}}}
    good = sanity.check_numbers(summary(0.55, 0.82, 0.7, 0.2), "m")
    assert all(c["ok"] for c in good)
    broken = {c["check"]: c["ok"] for c in sanity.check_numbers(summary(0.02, 0.10, 0.1, 0.3),
                                                                 "m")}
    assert broken == {"species top-1": False, "genus top-1": False, "top-k rises": True,
                      "depth order": False}


def test_a_collapsed_model_fails_the_classes_check():
    collapsed = {"n": 1000, "top_share": 0.4, "truths_answered": 0.05, "distinct_truths": 600}
    assert not sanity.check_classes(collapsed)["ok"]
    healthy = {"n": 1000, "top_share": 0.01, "truths_answered": 0.6, "distinct_truths": 600}
    assert sanity.check_classes(healthy)["ok"]


def test_a_temperature_at_the_edge_of_the_grid_fails_calibration():
    edge = float(evaluate.T_GRID[-1])
    report = {"all_photos": {"calibration": {"species": {"temperature": edge, "n": 900},
                                             "genus": {"temperature": 0.05, "n": 900}}}}
    assert not sanity.check_calibration(report)["ok"]
    report["all_photos"]["calibration"]["species"]["temperature"] = 0.03
    assert sanity.check_calibration(report)["ok"]
    # Two perfectly separated records fit the sharpest temperature: not judged.
    report["all_photos"]["calibration"]["species"] = {"temperature": edge, "n": 2}
    assert sanity.check_calibration(report)["ok"]


def test_the_cpu_check_refuses_to_run_with_a_gpu(monkeypatch, cut):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        sanity.cpu_check(cut[0], "base", cut[3], [])
