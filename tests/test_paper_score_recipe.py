"""The recipe (the replicated FungiTastic classifier, retrain.py; the paper's E4) through
`mv paper-score` end to end: its three comparators (classifier, classifier+month,
classifier+month+place) score from the trained model's own head and training counts, and
the place one reads each find's public point only, never the release's true one."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
from test_models_and_scoreboard import Const  # noqa: E402
from test_paper_score import KEY, run, world  # noqa: E402,F401  (world: the fixture)

from mycomap_vision import backbone_screen, paper_score  # noqa: E402
from mycomap_vision.finetune import models_dir  # noqa: E402
from mycomap_vision.replications.fungitastic import retrain  # noqa: E402

NAME = "picek-fake-20261027"
SCALE = 20.0
TRUE_POINT = (47.0, -122.0)                 # release_fixture's point for every record
PUBLIC = {"5": (10.0, 10.0), "6": (None, None)}
SEPTEMBER = 8                               # every fixture record is observed 2025-09-01


class Features:
    """The recipe model's stored vectors: Const's colour feature (red A x points to -x,
    dark B y to +x) in encode_features' length-keeping form, as ClassifierBackbone stores."""

    def __init__(self, name):
        self.name, self.dim, self.const = name, 3, Const(name)

    def encode(self, images):
        return retrain.encode_features(self.const.encode(images), SCALE)


def fake_recipe(world):
    """A trained-classifier model in data/models: a weak head that reads the colour, month
    counts that put B y alone in September, and place counts with A x alone in the public
    point's cell and B y alone in the true point's cell."""
    rows = world["base"].db.execute("select manifest_photo_id from photos")
    backbone_screen.embed(world["conn"], {"photos": [int(r[0]) for r in rows]}, [NAME],
                          loader=lambda s: Features(s), log=lambda s: None)
    months = np.zeros((2, 12), dtype=np.int32)
    months[0, 0], months[1, SEPTEMBER] = 100, 1000
    cells = {retrain.cell_of(*PUBLIC["5"], 4.0): {0: 1000},
             retrain.cell_of(*TRUE_POINT, 4.0): {1: 1000}}
    data = SimpleNamespace(classes=["A x", "B y"], genus_of=["A", "B"],
                           class_photos=np.array([100, 1000]), month_counts=months, cells=cells)
    root = models_dir()
    root.mkdir(parents=True, exist_ok=True)
    weight = np.array([[-0.2, 0.0], [0.2, 0.0]], dtype=np.float32)
    retrain.save_head(root / f"{NAME}.classifier.npz", data, weight, np.zeros(2, np.float32),
                      1.0, SCALE, 4.0)
    (root / f"{NAME}.json").write_text(json.dumps({"name": NAME, "kind": retrain.KIND}),
                                       encoding="utf-8")


def recipe_config(**place_comparator):
    return {"format": 1, "answer_key": KEY, "bootstrap": {"reps": 200, "seed": 1},
            "comparators": [
                {"id": "recipe", "kind": "vision", "backbone": NAME, "method": "classifier"},
                {"id": "recipe-month", "kind": "vision", "backbone": NAME,
                 "method": "classifier+month"},
                {"id": "recipe-month-place", "kind": "vision", "backbone": NAME,
                 "method": "classifier+month+place", "place": True, **place_comparator}],
            "hypotheses": {}}


def first_answers(tmp_path, comparator):
    lines = (tmp_path / "out" / "predictions" / f"{comparator}.jsonl").read_text().splitlines()
    return {json.loads(x)["record_key"]: json.loads(x)["species"][0] for x in lines}


def test_the_recipes_three_comparators_score_from_its_own_head_and_counts(world, tmp_path,
                                                                          monkeypatch):
    fake_recipe(world)
    monkeypatch.setattr(paper_score, "public_points", lambda rel: PUBLIC)
    seen = []
    cell_of = retrain.cell_of

    def spy(lat, lon, degrees):
        seen.append((lat, lon))
        return cell_of(lat, lon, degrees)
    monkeypatch.setattr(retrain, "cell_of", spy)
    rep = run(world, tmp_path, "val", recipe_config())
    # The head alone reads the colour: both right.
    assert first_answers(tmp_path, "recipe") == {"5": "A x", "6": "B y"}
    # The month counts (September is B y's alone) outweigh the weak head for the red find.
    assert first_answers(tmp_path, "recipe-month") == {"5": "B y", "6": "B y"}
    # The public point's cell is A x's alone: it brings the red find back. Read at the true
    # point, B y's cell, it would have stayed B y.
    assert first_answers(tmp_path, "recipe-month-place") == {"5": "A x", "6": "B y"}
    assert TRUE_POINT not in seen and set(seen) == {PUBLIC["5"], PUBLIC["6"]}
    assert rep["inputs"]["recipe-month-place"]["place"] is True
    assert rep["inputs"]["recipe-month"]["place"] is False
    table = rep["standard"]["comparators"]["recipe"]["primary"]["all"]["table"]
    assert table["species strict"]["top1"]["rate"] == 1.0


def test_a_place_method_without_place_true_is_refused_before_anything_is_scored():
    cfg = recipe_config()
    del cfg["comparators"][2]["place"]
    with pytest.raises(ValueError, match="takes \"place\": true"):
        paper_score.check_config(cfg)
    cfg["comparators"][2]["method"] = "nearest+mean+place"
    with pytest.raises(ValueError, match="takes \"place\": true"):
        paper_score.check_config(cfg)


def test_the_recipes_place_comparator_takes_no_place_setting():
    paper_score.check_config(recipe_config())
    cfg = recipe_config(place_setting="penalty-500-s30-2026-10-19")
    with pytest.raises(ValueError, match="own counts"):
        paper_score.check_config(cfg)
