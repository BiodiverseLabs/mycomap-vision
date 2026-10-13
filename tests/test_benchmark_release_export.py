"""A comparison on a frozen dataset release goes to the site's Results page the same way a
held-out report does (mv benchmark-export --comparison, then mv benchmark-import on the box),
marked reproduced-on-<release>; the scoreboard says which release each run was measured on,
so pre-release comparisons can be shown as development history."""

import json

import pytest
from test_trainer_release import cut  # noqa: F401  (the fixture)
from test_v1_evaluation import compared

from mycomap_vision import benchmark_io, evaluate


def frozen(conn, cid):
    """Mark the comparison's release as frozen (the fixture can only cut drafts)."""
    conn.execute("update eval_runs set report_json = json_set(report_json, "
                 "'$.dataset_release.draft', json('false')) where comparison_id = ?", (cid,))
    conn.commit()


def test_a_release_comparison_exports_as_a_reproducible_benchmark(cut, tmp_path, monkeypatch):
    conn, _, store, rel, _ = cut
    out, _ = compared(conn, store, tmp_path, monkeypatch, rel)
    cid = out["comparison_id"]
    frozen(conn, cid)
    p = benchmark_io.export_comparison(conn, cid, "val")
    assert p["kind"] == benchmark_io.KIND and p["benchmark"] == rel.id and p["split"] == "val"
    assert p["reproducibility"] == f"reproduced-on-{rel.id}" and p["sealed"] is False
    assert p["dataset_release"]["release_hash"] == rel.release_hash
    assert set(p["models"]) == {"base/nearest", "base/nearest+mean"}
    sp = p["models"]["base/nearest+mean"]["species"]["top1"]
    assert sp["n"] == 2 and 0 <= sp["rate"] <= 1 and sp["ci95"][0] <= sp["rate"] <= sp["ci95"][1]
    rows = p["summary"]["models"]["base/nearest+mean"]["rows"]
    assert {"species strict", "species s.l.", "species complex (beta)", "genus strict",
            "genus s.l."} <= set(rows)
    assert set(rows["species strict"]) == {"top1", "top3", "top5", "top10"}
    assert p["species_by_reference_records"]["base/nearest+mean"]["1-4"]["species"]["n"] == 2


def test_the_export_carries_no_record_or_name(cut, tmp_path, monkeypatch):
    conn, _, store, rel, _ = cut
    out, _ = compared(conn, store, tmp_path, monkeypatch, rel)
    frozen(conn, out["comparison_id"])
    text = json.dumps(benchmark_io.export_comparison(conn, out["comparison_id"]))
    labels = {r["label"] for r in rel.records(("train", "val"))}
    for secret in set(rel.record_keys()) | labels:
        assert json.dumps(secret) not in text, secret


def test_the_box_imports_it_as_a_val_split_and_lists_it(cut, tmp_path, monkeypatch):
    conn, _, store, rel, _ = cut
    out, _ = compared(conn, store, tmp_path, monkeypatch, rel)
    frozen(conn, out["comparison_id"])
    got = benchmark_io.import_summary(conn, benchmark_io.export_comparison(conn, out["comparison_id"]))
    assert got == {"benchmark": rel.id, "split": "val", "models": 2, "replaced": False}
    (listed,) = benchmark_io.published(conn)
    assert listed["reproducibility"] == f"reproduced-on-{rel.id}"


def test_a_draft_release_or_the_live_manifest_is_never_published_as_reproducible(cut, tmp_path,
                                                                                monkeypatch):
    conn, _, store, rel, _ = cut
    out, _ = compared(conn, store, tmp_path, monkeypatch, rel)
    with pytest.raises(ValueError, match="draft"):
        benchmark_io.export_comparison(conn, out["comparison_id"])
    conn.execute("update eval_runs set report_json = json_remove(report_json, '$.dataset_release')")
    with pytest.raises(ValueError, match="not on a dataset release"):
        benchmark_io.export_comparison(conn, out["comparison_id"])
    with pytest.raises(ValueError, match="val, or on a test release"):
        benchmark_io.export_comparison(conn, out["comparison_id"], "dev")


def test_scoreboard_rows_name_the_frozen_release_they_were_measured_on(cut, tmp_path,
                                                                      monkeypatch):
    conn, _, store, rel, _ = cut
    out, _ = compared(conn, store, tmp_path, monkeypatch, rel)
    cid = out["comparison_id"]
    assert {r["dataset_release"] for r in evaluate.scoreboard(conn, cid)} == {None}   # a draft
    frozen(conn, cid)
    assert {r["dataset_release"] for r in evaluate.scoreboard(conn, cid)} == {rel.id}


def test_the_command_takes_a_report_or_a_comparison_never_both(conn):
    from mycomap_vision import cli
    args = type("A", (), {"report": "r.json", "comparison": "c", "out": None})()
    with pytest.raises(SystemExit, match="--report .* or --comparison"):
        cli.cmd_benchmark_export(conn, args)
    with pytest.raises(SystemExit, match="not exported: no comparison"):
        cli.cmd_benchmark_export(conn, type("A", (), {"report": None, "comparison": "nope",
                                                      "split": "val", "sealed": False,
                                                      "out": None})())
