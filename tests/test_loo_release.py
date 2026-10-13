"""The leave-one-out scan prepared on a dataset release (scripts/loo_scan.py prep --release,
paper E17): the reference is the release's train + val, an erratum leaves its records out,
the temperature is the one given, and the release is cited."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
from release_fixture import BASE, release_world  # noqa: E402
from test_models_and_scoreboard import Const  # noqa: E402

from mycomap_vision import config  # noqa: E402
from mycomap_vision.embed import embed_photos  # noqa: E402

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "loo_scan.py"


@pytest.fixture
def scan_module(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    spec = importlib.util.spec_from_file_location("loo_scan_under_test", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    w = release_world(tmp_path, monkeypatch, test_release=False)
    monkeypatch.setattr(config, "MANIFEST_PATH", w["path"])
    items = [(int(p), path) for p, path in w["conn"].execute(
        "select photo_id, path from photo_copies")]
    embed_photos(w["conn"], w["store"], Const("m1"), items, config.DATA_DIR / "embeddings" / "m1",
                 log=lambda s: None)
    w["conn"].commit()
    return w


def test_prep_on_a_release_takes_train_and_val_and_cites_it(world, scan_module, tmp_path):
    out = tmp_path / "scan"
    out.mkdir()
    meta = scan_module.prep_release(out, BASE, "m1", 0.03, allow_draft=True)
    assert meta["records"] == 7                       # train 1-4, mo:9 + val 5, 6
    assert meta["temperature"] == {"species": 0.03}
    assert meta["dataset_release"]["dataset_release"] == BASE
    assert json.loads((out / "meta.json").read_text())["reproducibility"] == f"reproduced-on-{BASE}"
    assert (out / "ref32.npy").is_file() and (out / "sums.npy").is_file()


def test_an_erratum_leaves_its_records_out_of_the_scan(world, scan_module, tmp_path):
    errata = tmp_path / "e.tsv"
    errata.write_text("kind\tkey\treason\tnote\nrecord\t1\tmo_twin_as_inat\t\n", encoding="utf-8")
    out = tmp_path / "scan"
    out.mkdir()
    meta = scan_module.prep_release(out, BASE, "m1", 0.03, errata=errata, allow_draft=True)
    assert meta["records"] == 6
    assert meta["dataset_release"]["errata"]["records"] == 1
