"""Scoring on dataset releases (releases made by the builder, release_fixture): a release
is scored on its val split against its train split; its test set is a separate test
release, scored against the base release's train + val. The published checkpoints run as
one re-runnable command (`mv external run --release`), and the Picek vs Vision report is
`mv compare --release <base> --test-release <base>-test`."""

import json

import pytest
from release_fixture import BASE, TEST, release_world
from test_models_and_scoreboard import Const
from test_published import TWO, bright_is_a, write_metadata

from mycomap_vision import cli, config, evaluate, finetune
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.replications.fungitastic import crosswalk as gbif
from mycomap_vision.replications.fungitastic import (dataset, published, published_release,
                                                     release_eval)
from mycomap_vision.storage import LocalStore


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)


@pytest.fixture
def world(tmp_path, monkeypatch):
    w = release_world(tmp_path, monkeypatch)
    root = w["data"] / "external"                 # the fake checkpoint TWO ("A x", "B y")
    monkeypatch.setitem(published.MODELS, TWO.short, TWO)
    TWO.folder(root).mkdir(parents=True)
    (TWO.folder(root) / "pytorch_model.bin").write_bytes(b"w")
    write_metadata(root / "metadata" / TWO.metadata,
                   classes=[("A x L.", "A x", "F"), ("B y L.", "B y", "F")])
    published.write_labels(TWO, root)
    return w


# --- which split ------------------------------------------------------------------------------

def test_a_release_is_scored_on_val_and_its_test_release_on_test(world):
    base, test = dataset.load(BASE, True), dataset.load(TEST, True)
    assert release_eval.scored_split(base) == ("val", ("train",))
    assert release_eval.scored_split(test) == ("test", ("train", "val"))
    with pytest.raises(ValueError, match="separate release"):
        release_eval.scored_split(base, "test")
    with pytest.raises(ValueError, match="scored on its test split"):
        release_eval.scored_split(test, "val")
    assert release_eval.base_release(test).id == BASE


# --- the published checkpoints, one command ---------------------------------------------------

class NoGbif:
    """A crosswalk.Matcher stand-in session: every name unmatched."""
    headers: dict = {}

    def get(self, url, params=None, headers=None, timeout=None):
        class R:
            status_code, text = 200, ""

            def json(self):
                return {"diagnostics": {"matchType": "NONE"}, "matchType": "NONE"}
        return R()


def run_once(world, tmp_path, calls, rid=BASE, eval_split=None):
    def classify_for(model):
        def classify(images):
            calls.append(len(images))
            return bright_is_a(images)
        return classify
    matcher = gbif.Matcher(tmp_path / "gbif.sqlite", session=NoGbif(), interval=0,
                           sleep=lambda s: None)
    try:
        return published_release.run(world["conn"], dataset.load(rid, True), eval_split,
                                     ["two-b384"], classify_for=classify_for,
                                     matcher=matcher, log=lambda s: None)
    finally:
        matcher.close()


def folder_of(tmp_path, rid, split):
    return tmp_path / "data" / "research-results" / rid / "external" / split


def test_the_published_run_does_every_step_into_a_release_keyed_folder(world, tmp_path):
    calls = []
    out = run_once(world, tmp_path, calls)
    folder = folder_of(tmp_path, BASE, "val")
    assert out["folder"] == str(folder)
    state = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert list(state["steps"]) == ["labels", "predict", "crosswalk", "coverage", "report"]
    rel = dataset.load(BASE, True)
    assert state["dataset_release"] == dataset.cite(rel)
    assert state["reference"] == {"release": BASE, "splits": ["train"]}
    lines = [json.loads(x) for x in
             (folder / "two-b384.jsonl").read_text(encoding="utf-8").splitlines()]
    assert sorted(x["observation_id"] for x in lines) == ["5", "6"]
    assert {x["release_hash"] for x in lines} == {rel.release_hash}
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    assert report["dataset_release"]["labels_hash"] == rel.labels_hash
    model = report["models"]["external:two-b384"]
    assert model["answered"] == 2
    assert model["exact names"]["formal_species"]["rows"]["species strict"]["top1"]["rate"] == 1.0
    assert sum(calls) == 2


def test_the_test_release_is_scored_against_the_base_releases_train_and_val(world, tmp_path):
    out = run_once(world, tmp_path, [], rid=TEST)
    folder = folder_of(tmp_path, TEST, "test")
    assert out["folder"] == str(folder) and out["records"] == 2
    state = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    assert state["reference"] == {"release": BASE, "splits": ["train", "val"]}
    with pytest.raises(ValueError, match="separate release"):
        run_once(world, tmp_path, [], rid=BASE, eval_split="test")


def test_a_rerun_classifies_nothing_again_and_reproduces_the_outputs(world, tmp_path):
    calls = []
    run_once(world, tmp_path, calls)
    folder = folder_of(tmp_path, BASE, "val")
    first = (folder / "two-b384.jsonl").read_bytes()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    calls.clear()
    run_once(world, tmp_path, calls)
    assert calls == []                                   # resumable: every logit cached
    assert (folder / "two-b384.jsonl").read_bytes() == first
    assert json.loads((folder / "report.json").read_text(encoding="utf-8")) == report


def test_results_of_another_release_under_the_same_id_are_refused(world, tmp_path):
    run_once(world, tmp_path, [])
    state = folder_of(tmp_path, BASE, "val") / "run.json"
    doc = json.loads(state.read_text(encoding="utf-8"))
    doc["dataset_release"]["manifest_sha256"] = "0" * 64
    state.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="MANIFEST differs"):
        run_once(world, tmp_path, [])


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
    cli.main(["external", "run", "--release", BASE, "--allow-draft"])
    assert seen == {"release": BASE}


# --- Picek vs Vision on the test release -------------------------------------------------------

def embedded(world, tmp_path, names=("m1", "m2")):
    root = tmp_path / "emb"
    store = LocalStore(world["store"].root)
    for name in names:
        bb = Const(name, twist=0.3 if name == "m2" else 0.0)
        todo = photos_to_embed(world["conn"], bb.name, "large", store.location, "local")
        embed_photos(world["conn"], store, bb, todo, root / bb.name, log=lambda s: None)
    return root


def test_the_report_scores_the_test_release_against_the_base_train_and_val(world, tmp_path):
    root = embedded(world, tmp_path)
    base, test = dataset.load(BASE, True), dataset.load(TEST, True)
    out = release_eval.compare_on_test(world["conn"], base, test, ["m1", "m2"], ["nearest"],
                                       embeddings_root=root, log=lambda s: None)
    assert out["test_records"] == 2 and out["reference_records"] == 7
    assert out["test_release"]["dataset_release"] == TEST
    board = evaluate.scoreboard(world["conn"], out["comparison_id"])
    assert {r["backbone"] for r in board} == {"m1", "m2"}
    report = evaluate.run_report(world["conn"], board[0]["id"])
    assert report["dataset_release"]["dataset_release"] == BASE
    assert report["test_release"]["dataset_release"] == TEST


def test_a_test_release_of_another_release_is_refused(world, tmp_path):
    root = embedded(world, tmp_path, ("m1",))
    base = dataset.load(BASE, True)
    with pytest.raises(ValueError, match="not a test release of"):
        release_eval.compare_on_test(world["conn"], base, base, ["m1"], ["nearest"],
                                     embeddings_root=root, log=lambda s: None)


def test_a_model_that_saw_the_test_records_or_another_release_is_refused(world, tmp_path,
                                                                        monkeypatch):
    root = embedded(world, tmp_path, ("m1",))
    base, test = dataset.load(BASE, True), dataset.load(TEST, True)
    finetune.register(world["conn"], {"name": "m1", "base": "x", "trained_through":
                                      "2026-11-30", "test_days": 28, "created_at": "t"})
    with pytest.raises(ValueError, match="inflated"):
        release_eval.compare_on_test(world["conn"], base, test, ["m1"], ["nearest"],
                                     embeddings_root=root, log=lambda s: None)
    monkeypatch.setattr(release_eval, "model_release",
                        lambda name: {"dataset_release": "dry-20260101-000000"})
    with pytest.raises(ValueError, match="trained on release dry-20260101-000000"):
        release_eval.compare_on_test(world["conn"], base, test, ["m1"], ["nearest"],
                                     embeddings_root=root, log=lambda s: None)

