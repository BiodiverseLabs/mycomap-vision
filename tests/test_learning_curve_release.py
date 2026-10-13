"""The learning curve on a dataset release (`mv learning-curve release`, paper E7): val
queries against the train reference, errata records out of the reference, each query's
former-benchmark membership kept, and the report for val excluding the former held-out
beside full val."""

import argparse
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
from release_fixture import BASE, release_world  # noqa: E402
from test_models_and_scoreboard import Const  # noqa: E402

from mycomap_vision import learning_curve as lc  # noqa: E402
from mycomap_vision import learning_curve_report as lcr  # noqa: E402
from mycomap_vision.embed import embed_photos  # noqa: E402
from mycomap_vision.replications.fungitastic import dataset  # noqa: E402


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    w = release_world(tmp_path, monkeypatch, test_release=False)
    rel = dataset.load(BASE, True)
    items = [(int(p), path) for p, path in w["conn"].execute(
        "select photo_id, path from photo_copies")]
    from mycomap_vision import config
    embed_photos(w["conn"], w["store"], Const("m1"), items, config.DATA_DIR / "embeddings" / "m1",
                 log=lambda s: None)
    return {**w, "rel": rel}


def args_for(tmp_path, world, **kw):
    a = argparse.Namespace(release=BASE, allow_draft=True, backbone="m1",
                           manifest=str(world["path"]), errata=None, out=str(tmp_path / "scores"),
                           cache=None, seeds=2, fractions=[0.5], global_caps=[1], caps=[1, 2],
                           top=5, batch_photos=10, limit=0, shards=1, shard=0, prep_only=False)
    for k, v in kw.items():
        setattr(a, k, v)
    return a


def test_val_is_scored_against_train_and_keeps_its_benchmark_membership(world, tmp_path):
    lc.score_release(args_for(tmp_path, world))
    R = lcr.load_results(tmp_path / "scores")
    assert set(R["truths"]) == {"5", "6"}                 # the val split
    assert R["records"] == 5                              # train: 1, 2, 3, 4, mo:9
    assert set(R["benchmark_of"]) == {"5", "6"}
    assert R["dataset_release"]["dataset_release"] == BASE
    top = R["results"]["full-clean"]["nearest"]["5"]["species"][0][0]
    assert R["units"][top] == "A x"


def test_errata_records_leave_the_reference(world, tmp_path):
    errata = tmp_path / "errata.tsv"
    errata.write_text("kind\tkey\treason\tnote\nrecord\t1\tmo_twin_as_inat\t\n", encoding="utf-8")
    lc.score_release(args_for(tmp_path, world, errata=str(errata)))
    R = lcr.load_results(tmp_path / "scores")
    assert R["records"] == 4 and R["errata_sha256"]


def test_an_errata_file_without_a_key_column_is_refused(tmp_path):
    bad = tmp_path / "bad.tsv"
    bad.write_text("kind\tid\nrecord\t1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not an errata list"):
        lc.read_errata(bad)


def test_the_report_can_leave_the_former_held_out_queries_out(world, tmp_path):
    lc.score_release(args_for(tmp_path, world))
    full = lcr.report(tmp_path / "scores", tmp_path / "all")
    only6 = lcr.report(tmp_path / "scores", tmp_path / "excl", keep=lambda o: o != "5")
    assert full["scored_records"] == 2 and only6["scored_records"] == 1
