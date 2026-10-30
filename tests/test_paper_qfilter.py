"""The query photo filters of paper E34 (b) and E35 (a3-spec §1, paper_qfilter.py): a filtered
comparator leaves out the scored photos its pinned rules flag, its queries only; a find left
with no photo is a miss in the main tables and is left out of a second row; every pinned
file is checked."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_models_and_scoreboard import Const  # noqa: E402
from test_paper_freeze import frozen_spec, sealed_world  # noqa: E402
from test_paper_score import config, run, world  # noqa: E402,F401

from mycomap_vision import backbone_screen, paper_qfilter, paper_score  # noqa: E402
from mycomap_vision.embed import load_embeddings  # noqa: E402

VIEWS = ["fresh_fungus", "microscopy", "label", "scale", "dried", "habitat", "hand",
         "spore_print"]
CLASSES = ["specimen", "micrograph", "slip", "habitat", "uv", "chemical", "spore_print",
           "dried", "drawing", "tree", "people", "not_fungus"]
SHA = "c" * 64
# The stand-in vectors are two-dimensional and the toy weights large, so exp overflows to inf
# for photos far from a rule (p = 0, as intended), as it would in the pinned code.
pytestmark = pytest.mark.filterwarnings("ignore:overflow encountered in exp:RuntimeWarning")


def toy_bundle(path, micro=((10.0, 0.0), -5.0)):
    """A two-dimensional E34 bundle whose rules are known by hand (see the flags test)."""
    view_text = np.full((8, 2), -0.7, np.float32)
    view_text[VIEWS.index("habitat")] = [1, 0]
    view_text[VIEWS.index("dried")] = [0, 1]
    np.savez(path, micro_w=np.float32(micro[0]), micro_b=np.float64(micro[1]),
             slip_w=np.float32([0, 10]), slip_b=np.float64(-5),
             view_names=np.array(VIEWS), view_text=view_text, view_scale=np.float64(50),
             label_text=np.float32([0, 1]),
             uv_w=np.float32([0, 0]), uv_b=np.float64(5),
             phylo_tree_w=np.float32([0, 0]), phylo_tree_b=np.float64(-5),
             drawing_w=np.float32([-10, 0]), drawing_b=np.float64(0))
    return path


def toy_tagger(path):
    W = np.zeros((2, 12), np.float32)
    W[:, CLASSES.index("spore_print")] = [0, 20]
    W[:, CLASSES.index("chemical")] = [20, 0]
    np.savez(path, W=W, b=np.zeros(12, np.float32), classes=np.array(CLASSES))
    return path


class Gate:
    class thresholds:
        not_fungus = 0.5

    def p_fungus(self, v):
        return np.where(np.asarray(v)[:, 0] > 0.9, 0.1, 0.9)       # [1, 0] is "not a fungus"


def bundle_arrays(path):
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def test_each_rule_flags_exactly_the_photos_its_pinned_rule_does(tmp_path):
    v = np.array([[1, 0], [0, 1], [0.6, 0.8], [0.52, 0.854]], np.float32)
    tag = bundle_arrays(toy_tagger(tmp_path / "t.npz"))
    tag = {"W": tag["W"], "b": tag["b"], "classes": tag["classes"]}
    got = paper_qfilter.flags(v, bundle_arrays(toy_bundle(tmp_path / "b.npz")), Gate(), tag)
    want = {"R6_micrograph": [True, False, True, True],   # p = sig(10 x - 5) >= 0.5 (0.55)
            "R6_view": [True, True, True, True],             # habitat, dried, dried, dried
            "R6_view_habitat": [True, False, False, False],  # the habitat column alone
            "R6_gate": [True, False, False, False],
            "R6b_slip": [False, True, False, False],         # micrograph p >= 0.5 blocks it
            "R7_uv": [True] * 4, "R7_phylo_tree": [False] * 4, "R7_drawing": [False] * 4,
            "V21_spore_print": [False, True, True, True],
            "V21_chemical": [True, False, False, False]}
    assert {r: got[r].tolist() for r in got} == want
    one = paper_qfilter.flags(v, bundle_arrays(tmp_path / "b.npz"), rules=["R6_view_habitat"])
    assert list(one) == ["R6_view_habitat"]
    with pytest.raises(paper_qfilter.PinError, match="R6_gate needs"):
        paper_qfilter.flags(v, bundle_arrays(tmp_path / "b.npz"), rules=["R6_gate"])


def spec(bundle, sha=None, rules=("R6_micrograph",), **kw):
    return {"rules": list(rules), "bundle": {"file": str(bundle),
                                             "sha256": sha or paper_qfilter.sha256_file(bundle)},
            "backbone": "bioclip-2", "backbone_sha256": SHA, "empty": "miss_and_drop_row", **kw}


def test_a_filter_file_that_is_not_the_pinned_one_is_refused(tmp_path):
    b = toy_bundle(tmp_path / "b.npz")
    paper_qfilter.load(spec(b), Path)
    with pytest.raises(paper_qfilter.PinError, match="bundle"):
        paper_qfilter.load(spec(b, sha="0" * 64), Path)
    t = toy_tagger(tmp_path / "t.npz")
    s = spec(b, rules=["V21_chemical"],
             tagger_v21={"file": str(t), "sha256": "0" * 64, "threshold": 0.5})
    with pytest.raises(paper_qfilter.PinError, match="tagger"):
        paper_qfilter.load(s, Path)
    s["tagger_v21"]["sha256"] = paper_qfilter.sha256_file(t)
    assert paper_qfilter.load(s, Path)["tagger"] is not None
    gdir = tmp_path / "gate"
    gdir.mkdir()
    for n in ("fungus-gate.npz", "fungus-gate.json"):
        (gdir / n).write_bytes(n.encode())
    s = spec(b, rules=["R6_gate"], gate={"dir": str(gdir), "files": {
        "fungus-gate.npz": paper_qfilter.sha256_file(gdir / "fungus-gate.npz"),
        "fungus-gate.json": "0" * 64}})
    with pytest.raises(paper_qfilter.PinError, match="gate"):
        paper_qfilter.load(s, Path)


@pytest.mark.parametrize("change, refused", [
    (lambda s: s.update(rules=["R6_micrograph", "R8_people"]), "unknown query_photo_filter rule"),
    (lambda s: s.update(rules=[]), "list of distinct rules"),
    (lambda s: s.update(bundle={"file": "b.npz", "sha256": "TBD"}), "bundle is"),
    (lambda s: s.update(rules=["R6_gate"]), "R6_gate pins the gate"),
    (lambda s: s.update(rules=["V21_chemical"]), "a V21 rule pins tagger_v21"),
    (lambda s: s.update(rules=["V21_chemical"], tagger_v21={"file": "t.npz", "sha256": SHA,
                                                            "threshold": 1.5}), "threshold"),
    (lambda s: s.update(backbone="bioclip-2-336"), "backbone is"),
    (lambda s: s.update(empty="drop"), "miss_and_drop_row"),
])
def test_a_bad_query_photo_filter_is_refused_by_check_config(tmp_path, change, refused):
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"][1]["query_photo_filter"] = spec(toy_bundle(tmp_path / "b.npz"))
    paper_score.check_config(cfg)
    change(cfg["comparators"][1]["query_photo_filter"])
    with pytest.raises(ValueError, match=refused):
        paper_score.check_config(cfg)


def embed_filter_vectors(world):
    """The world's photos embedded under the filter's backbone name (a stand-in model)."""
    photos = [int(r[0]) for rel in (world["base"], world["test"])
              for r in rel.db.execute("select manifest_photo_id from photos")]
    backbone_screen.embed(world["conn"], {"photos": photos}, [paper_qfilter.BACKBONE],
                          loader=lambda s: Const(s), log=lambda s: None)


def bundle_flagging(world, rel, record_key, path):
    """A bundle whose micrograph rule flags the photos that look like `record_key`'s."""
    pid = int(rel.db.execute("select manifest_photo_id from photos where record_key = ?",
                             (record_key,)).fetchone()[0])
    ids, vecs = load_embeddings(world["conn"], paper_qfilter.BACKBONE)
    v = np.asarray(vecs[list(ids).index(pid)], np.float32)
    v = v / np.linalg.norm(v)
    return toy_bundle(path, micro=(tuple((v * 200).tolist()), -199.0))


def test_a_filtered_comparator_scores_a_find_left_with_no_photo_as_a_miss_and_drops_it_in_a_second_row(
        world, tmp_path):
    embed_filter_vectors(world)
    b = bundle_flagging(world, world["base"], "6", tmp_path / "b.npz")
    plain = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"][1]["query_photo_filter"] = spec(b)
    paper_score.check_config(cfg)
    rep = run(world, tmp_path, "val", cfg)
    arm = rep["standard"]["comparators"]["vision-mean"]["primary"]["all"]
    assert arm["table"]["species strict"]["top1"]["rate"] == 0.5        # 6 has no photo left
    q = rep["query_photo_filters"]["vision-mean"]
    assert q["finds_left_with_no_photo"] == 1 and q["photos_removed"] >= 1
    assert q["removed_by_rule"]["R6_micrograph"] == q["photos_removed"]
    assert (q["dropped_row"]["species_strict@1"]["n"],
            q["dropped_row"]["species_strict@1"]["rate"]) == (1, 1.0)
    assert q["vectors"]["fresh"] is False and q["vectors"]["photos_without_vectors"] == 0
    # the comparator without a filter scores exactly as before
    assert json.dumps(rep["standard"]["comparators"]["vision-photos"], sort_keys=True) == \
        json.dumps(plain["standard"]["comparators"]["vision-photos"], sort_keys=True)
    assert "query_photo_filters" not in plain
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Query photo filters (E34 (b), E35)" in md


def test_each_comparator_removes_only_its_own_rules(world, tmp_path):
    embed_filter_vectors(world)
    b = bundle_flagging(world, world["base"], "6", tmp_path / "b.npz")
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"][0]["query_photo_filter"] = spec(b)                           # flags 6
    cfg["comparators"][1]["query_photo_filter"] = spec(b, rules=["R7_phylo_tree"])  # flags none
    rep = run(world, tmp_path, "val", cfg)
    q = rep["query_photo_filters"]
    assert q["vision-photos"]["finds_left_with_no_photo"] == 1
    assert (q["vision-mean"]["photos_removed"], q["vision-mean"]["finds_left_with_no_photo"]) == (0, 0)
    std = rep["standard"]["comparators"]
    assert std["vision-photos"]["primary"]["all"]["table"]["species strict"]["top1"]["rate"] == 0.5
    assert std["vision-mean"]["primary"]["all"]["table"]["species strict"]["top1"]["rate"] == 1.0


def test_the_sealed_test_embeds_the_filter_s_vectors_fresh_and_pins_its_checkpoint(
        world, tmp_path, monkeypatch):
    embed_filter_vectors(world)
    b = bundle_flagging(world, world["test"], "8", tmp_path / "b.npz")
    base_cfg = config(tmp_path, ["7", "8"], code_commit="abcdef1234567")
    base_cfg["comparators"][1]["query_photo_filter"] = spec(b)
    frozen = frozen_spec(world, base_cfg)
    cfg, body, repo = sealed_world(world, tmp_path, monkeypatch, frozen)
    cfg["comparators"][1]["query_photo_filter"] = spec(b)
    body = json.dumps(cfg).encode()
    (repo / "paper" / "analysis-v1.json").write_bytes(body)
    from test_paper_score import git
    git(repo, "commit", "-qam", "the plan with its filter")
    git(repo, "tag", "-f", paper_score.TAG)
    monkeypatch.setattr(paper_score, "query_checkpoint_sha", lambda models_root=None: SHA)
    rep = paper_score.run(world["conn"], world["base"], world["test"], cfg, body,
                          tmp_path / "out", research_dir=repo, log=lambda s: None)
    q = rep["query_photo_filters"]["vision-mean"]
    assert q["vectors"]["fresh"] is True and q["finds_left_with_no_photo"] == 1
    arm = rep["standard"]["comparators"]["vision-mean"]["primary"]["all"]
    assert arm["table"]["species strict"]["top1"]["rate"] == 0.5      # 8 has no photo left
    monkeypatch.setattr(paper_score, "query_checkpoint_sha", lambda models_root=None: "d" * 64)
    with pytest.raises(paper_score.FrozenError, match="checkpoint"):
        paper_score.run(world["conn"], world["base"], world["test"], cfg, body,
                        tmp_path / "out2", research_dir=repo, log=lambda s: None)


def test_the_checkpoint_sha_is_the_alias_s_base_file(monkeypatch, tmp_path):
    from mycomap_vision import paper_freeze
    monkeypatch.setattr(paper_freeze, "model_files",
                        lambda name, root: {"x.json": "1", "base:open_clip_model.safetensors": "2"}
                        if name == "bioclip-2" else {})
    assert paper_score.query_checkpoint_sha(tmp_path) == "2"
