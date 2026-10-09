"""Held-out benchmark results travel to the server box as aggregates only, and the Models
page reads them from /api/benchmarks (Steve, 2026-10-09)."""

import json

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import ISSUER, PUBLIC_KEY, SECRET, sign_in

from mycomap_vision import benchmark_io, cli
from mycomap_vision.api import create_app
from mycomap_vision.signin import SigninConfig

MODEL = "bioclip-2-ft-x/nearest"


def report(split="dev", rate=0.48):
    """Shaped like `mv heldout report` JSON, with the record-level parts it also carries."""
    return {
        "benchmark": "heldout-x", "split": split, "sealed": False, "records": 3000,
        "scored_records": 2990, "code_version": "abc1234", "released_at": None,
        "models": {MODEL: {"records": 2980, "species": {"top1": {"n": 2900, "right": 1392,
                                                                   "rate": rate}}}},
        "breakdowns": {MODEL: {"species reference records": {"1-4": {"species": {"n": 300,
                                                                                    "rate": 0.14}}},
                               "same observer and day in reference": {"yes": {}}}},
        "paired": [{"a": MODEL, "b": "m2", "species": {"mcnemar_p": 0.01}}],
        "calibration": {MODEL: {"species": {"bins": []}}},
        "likely_sets": {MODEL: {"species": {"coverage": 0.7}}},
        "summary": {"benchmark": "heldout-x", "models": {}},
        # never to leave the laptop:
        "label_audit": {"rows": [{"observation_id": "123", "name": "Russula x"}]},
        "label_hygiene": {"examples": ["123"]},
        "files": {"json": "C:\\Users\\someone\\report.json"},
        "references": {MODEL: "5dbfdb1d24a5"},
    }


def test_the_export_carries_aggregates_only(tmp_path):
    out = benchmark_io.export_summary(report(), "report-dev-1.json")
    text = json.dumps(out)
    for private in ("label_audit", "label_hygiene", "files", "observation_id", "C:\\\\Users",
                    "same observer and day"):
        assert private not in text
    assert out["species_by_reference_records"] == {
        MODEL: {"1-4": {"species": {"n": 300, "rate": 0.14}}}}
    assert out["models"][MODEL]["species"]["top1"]["rate"] == 0.48
    assert out["split"] == "dev" and out["report"] == "report-dev-1.json"


def test_something_that_is_not_a_heldout_report_is_not_exported():
    with pytest.raises(ValueError, match="not an mv heldout report"):
        benchmark_io.export_summary({"rows": []}, "x.json")


def test_an_import_replaces_the_same_benchmark_split_and_keeps_others(conn):
    benchmark_io.import_summary(conn, benchmark_io.export_summary(report("dev", 0.4), "a"))
    benchmark_io.import_summary(conn, benchmark_io.export_summary(report("test"), "b"))
    again = benchmark_io.import_summary(conn, benchmark_io.export_summary(report("dev", 0.5), "c"))
    assert again["replaced"] is True
    got = {(b["split"], b["models"][MODEL]["species"]["top1"]["rate"])
           for b in benchmark_io.published(conn)}
    assert got == {("dev", 0.5), ("test", 0.48)}


@pytest.mark.parametrize("payload, why", [
    ({"kind": "something else"}, "not a benchmark export"),
    ({**benchmark_io.export_summary(report(), "a"), "split": "train"}, "unknown split"),
    ({**benchmark_io.export_summary(report(), "a"), "models": {}}, "no models"),
    ({**benchmark_io.export_summary(report(), "a"), "benchmark": ""}, "names no benchmark"),
])
def test_a_damaged_or_foreign_export_is_refused_and_changes_nothing(conn, payload, why):
    with pytest.raises(benchmark_io.ImportRefused, match=why):
        benchmark_io.import_summary(conn, payload)
    assert benchmark_io.published(conn) == []


def test_export_then_import_through_the_command_line(conn, tmp_path, capsys, monkeypatch):
    src = tmp_path / "report-dev-1.json"
    src.write_text(json.dumps(report()), encoding="utf-8")
    out = tmp_path / "dev.json"
    cli.cmd_benchmark_export(conn, type("A", (), {"report": str(src), "out": str(out)})())
    cli.cmd_benchmark_import(conn, type("A", (), {"file": str(out)})())
    assert json.loads(capsys.readouterr().out)["benchmark"] == "heldout-x"
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(SystemExit, match="not imported"):
        cli.cmd_benchmark_import(conn, type("A", (), {"file": str(bad)})())


def site(conn, tmp_path, mode):
    seed_two_species(conn, tmp_path)
    conn.commit()
    signin = SigninConfig(mode=mode, origin="http://testserver", issuer=ISSUER,
                          public_key=PUBLIC_KEY, secret=SECRET)
    app = create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                     backbone_loader=lambda n: Const(n), limits=open_limits(), signin=signin,
                     background=False)
    return TestClient(app, follow_redirects=False)


def test_the_models_page_reads_imported_benchmarks_behind_the_sites_sign_in(conn, tmp_path):
    benchmark_io.import_summary(conn, benchmark_io.export_summary(report(), "a"))
    conn.commit()
    c = site(conn, tmp_path, "all")
    assert c.get("/api/benchmarks").status_code == 401
    sign_in(c)
    got = c.get("/api/benchmarks").json()["benchmarks"]
    assert [b["split"] for b in got] == ["dev"] and "label_audit" not in json.dumps(got)
