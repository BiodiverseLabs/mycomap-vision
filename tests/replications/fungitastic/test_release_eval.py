"""Scoring on a dataset release: the published checkpoints in one re-runnable command
(`mv external run --release`), and the Picek vs Vision comparison (`mv compare --release`)
on the release's own split: val by default, test only when asked, sealed only once the
release has it; the reference is every split before the scored one."""

import json

import numpy as np
import pytest
from release_fixture import make_release
from test_models_and_scoreboard import Const, seed_two_species
from test_published import bright_is_a, install_two

from mycomap_vision import cli, config, evaluate
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.replications.fungitastic import (dataset, published_release,
                                                     release_eval)
from mycomap_vision.replications.fungitastic import crosswalk as gbif

RID = "dry-20261009-1830"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def world(conn, tmp_path, monkeypatch):
    store = seed_two_species(conn, tmp_path)
    root = install_two(tmp_path, monkeypatch)          # sets DATA_DIR to <tmp>/data
    make_release(tmp_path / "data", RID)
    return {"store": store, "root": root, "rel": dataset.load(RID, allow_draft=True)}


# --- which split ------------------------------------------------------------------------------

def test_val_is_the_default_test_needs_a_flag_and_sealed_needs_the_split(world, tmp_path,
                                                                         monkeypatch):
    rel = world["rel"]
    release_eval.check_eval_split(rel, "val")
    with pytest.raises(ValueError, match="--allow-test"):
        release_eval.check_eval_split(rel, "test")
    release_eval.check_eval_split(rel, "test", allow_test=True)
    with pytest.raises(ValueError, match="no sealed split"):
        release_eval.check_eval_split(rel, "sealed")
    d = tmp_path / "d2"
    monkeypatch.setattr(config, "DATA_DIR", d)
    make_release(d, RID, splits={"107": "sealed"})
    release_eval.check_eval_split(dataset.load(RID, True), "sealed")


def test_the_reference_is_every_split_before_the_scored_one(conn, world):
    rel = world["rel"]
    keys = lambda splits: sorted(r.observation_id  # noqa: E731
                                 for r in release_eval.records_of(conn, rel, splits))
    assert keys(release_eval.REFERENCE_FOR["val"]) == ["100", "101", "103", "104"]
    assert keys(release_eval.REFERENCE_FOR["test"]) == ["100", "101", "102", "103", "104",
                                                        "105"]
    assert keys(("test",)) == ["106", "107"]


# --- the published checkpoints, one command ---------------------------------------------------

class NoGbif:
    """A crosswalk.Matcher stand-in session: every name unmatched, every call counted."""
    asked = 0
    headers: dict = {}

    def get(self, url, params=None, headers=None, timeout=None):
        NoGbif.asked += 1

        class R:
            status_code, text = 200, ""

            def json(self):
                return {"diagnostics": {"matchType": "NONE"}, "matchType": "NONE"}
        return R()


def run_once(conn, world, tmp_path, calls, split="val", allow_test=False):
    def classify_for(model):
        def classify(images):
            calls.append(len(images))
            return bright_is_a(images)
        return classify
    matcher = gbif.Matcher(tmp_path / "gbif.sqlite", session=NoGbif(), interval=0,
                           sleep=lambda s: None)
    try:
        return published_release.run(conn, world["rel"], split, allow_test, ["two-b384"],
                                     classify_for=classify_for, matcher=matcher,
                                     log=lambda s: None)
    finally:
        matcher.close()


def test_the_published_run_does_every_step_into_a_release_keyed_folder(conn, world, tmp_path):
    calls = []
    out = run_once(conn, world, tmp_path, calls)
    folder = tmp_path / "data" / "research-results" / RID / "external" / "val"
    assert out["folder"] == str(folder)
    state = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert list(state["steps"]) == ["labels", "predict", "crosswalk", "coverage", "report"]
    assert state["dataset_release"]["manifest_sha256"] == world["rel"].manifest_sha256
    assert state["reference_splits"] == ["train"]
    lines = (folder / "two-b384.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(x)["observation_id"] for x in lines) == ["102", "105"]
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    model = report["models"]["external:two-b384"]
    assert model["answered"] == 2 and report["eval_split"] == "val"
    rows = model["exact names"]["formal_species"]["rows"]["species strict"]
    assert rows["top1"]["rate"] == 1.0
    assert (folder / "coverage.json").is_file() and sum(calls) == 2


def test_a_rerun_classifies_nothing_again_and_reproduces_the_answers(conn, world, tmp_path):
    calls = []
    run_once(conn, world, tmp_path, calls)
    folder = tmp_path / "data" / "research-results" / RID / "external" / "val"
    first = (folder / "two-b384.jsonl").read_bytes()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    calls.clear()
    run_once(conn, world, tmp_path, calls)
    assert calls == []                                   # resumable: every logit cached
    assert (folder / "two-b384.jsonl").read_bytes() == first
    assert json.loads((folder / "report.json").read_text(encoding="utf-8")) == report


def test_the_published_run_scores_test_only_when_asked(conn, world, tmp_path):
    with pytest.raises(ValueError, match="--allow-test"):
        run_once(conn, world, tmp_path, [], split="test")
    out = run_once(conn, world, tmp_path, [], split="test", allow_test=True)
    assert out["records"] == 2 and out["folder"].endswith("test")


def test_results_of_another_release_under_the_same_id_are_refused(conn, world, tmp_path):
    run_once(conn, world, tmp_path, [])
    state = tmp_path / "data" / "research-results" / RID / "external" / "val" / "run.json"
    doc = json.loads(state.read_text(encoding="utf-8"))
    doc["dataset_release"]["manifest_sha256"] = "0" * 64
    state.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="MANIFEST differs"):
        run_once(conn, world, tmp_path, [])


def test_mv_external_run_reads_the_manifest_read_only(tmp_path, monkeypatch):
    import sqlite3

    from mycomap_vision import manifest
    path = tmp_path / "manifest.sqlite"
    manifest.connect(path).close()
    monkeypatch.setattr(config, "MANIFEST_PATH", path)
    seen = {}

    def spy(conn, args):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("create table x (y)")
        seen["release"] = args.release
    monkeypatch.setattr(cli, "cmd_external", spy)
    cli.main(["external", "run", "--release", RID, "--allow-draft"])
    assert seen == {"release": RID}


# --- Picek vs Vision on the release ------------------------------------------------------------

def embedded(conn, world, tmp_path, names=("m1", "m2")):
    root = tmp_path / "emb"
    for name in names:
        bb = Const(name, twist=0.3 if name == "m2" else 0.0)
        todo = photos_to_embed(conn, bb.name, "large", world["store"].location, "local")
        embed_photos(conn, world["store"], bb, todo, root / bb.name, log=lambda s: None)
    return root


def test_a_release_comparison_scores_the_releases_split_against_the_splits_before_it(
        conn, world, tmp_path):
    root = embedded(conn, world, tmp_path)
    out = release_eval.compare(conn, world["rel"], ["m1", "m2"], ["nearest"], "val",
                               embeddings_root=root, log=lambda s: None)
    assert out["test_records"] == 2 and out["reference_records"] == 4
    assert out["eval_split"] == "val" and out["reference_splits"] == ["train"]
    board = evaluate.scoreboard(conn, out["comparison_id"])
    assert {r["backbone"] for r in board} == {"m1", "m2"}
    report = evaluate.run_report(conn, board[0]["id"])
    assert report["dataset_release"]["dataset_release"] == RID
    assert report["eval_split"] == "val"


def test_a_model_trained_on_the_val_split_is_not_scored_on_it(conn, world, tmp_path):
    root = embedded(conn, world, tmp_path, ("m1",))
    conn.executescript(evaluate.FINETUNE_SCHEMA)
    from mycomap_vision import finetune
    finetune.register(conn, {"name": "m1", "base": "x", "trained_through": "2026-01-03",
                             "test_days": 28, "created_at": "t"})
    with pytest.raises(ValueError, match="inflated"):
        release_eval.compare(conn, world["rel"], ["m1"], ["nearest"], "val",
                             embeddings_root=root, log=lambda s: None)
    out = release_eval.compare(conn, world["rel"], ["m1"], ["nearest"], "test",
                               allow_test=True, embeddings_root=root, log=lambda s: None)
    assert out["reference_records"] == 6


def test_a_model_trained_on_another_release_is_refused(conn, world, tmp_path, monkeypatch):
    root = embedded(conn, world, tmp_path, ("m1",))
    monkeypatch.setattr(release_eval, "model_release",
                        lambda name: {"dataset_release": "dry-other"})
    with pytest.raises(ValueError, match="trained on release dry-other"):
        release_eval.compare(conn, world["rel"], ["m1"], ["nearest"], "val",
                             embeddings_root=root, log=lambda s: None)
    assert np is not None
