"""The freeze spec (paper PLAN §8): the sealed test is scored only by the predictor the
config freezes, part by part by hash; val and draft runs report what differs and go on."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
from test_paper_score import config, git, run, world  # noqa: E402,F401

from mycomap_vision import config as cfgmod  # noqa: E402
from mycomap_vision import embed, models, paper_freeze, paper_score  # noqa: E402


def test_vectors_hash_covers_ids_and_rows_and_counts_the_missing():
    ids = np.array([3, 1, 2])
    vecs = np.eye(3, dtype=np.float16)
    a = paper_freeze.vectors_sha(ids, vecs, [1, 2, 3, 9])
    assert (a["photos"], a["missing"]) == (3, 1)
    assert paper_freeze.vectors_sha(ids[::-1], vecs[::-1], [1, 2, 3, 9]) == a   # order-free
    moved = vecs.copy()
    moved[0, 1] = 0.5
    assert paper_freeze.vectors_sha(ids, moved, [1, 2, 3])["sha256"] != a["sha256"]
    assert paper_freeze.vectors_sha(ids, vecs, [1, 2])["sha256"] != a["sha256"]


def test_model_files_hash_the_models_own_files_and_its_base_checkpoint(tmp_path, monkeypatch):
    (tmp_path / "ft-a.json").write_text(json.dumps({"base_spec": "open_clip:hf-hub:x/y"}))
    (tmp_path / "ft-a.pt").write_bytes(b"w")
    (tmp_path / "ft-a.heads.npz").write_bytes(b"h")
    (tmp_path / "ft-ab.pt").write_bytes(b"other model")          # never mixed in
    base = tmp_path / "hub" / "open_clip_model.safetensors"
    base.parent.mkdir()
    base.write_bytes(b"base")
    monkeypatch.setattr(models, "open_clip_weights_file",
                        lambda name: str(base) if name == "hf-hub:x/y" else None)
    got = paper_freeze.model_files("ft-a", tmp_path)
    assert set(got) == {"ft-a.json", "ft-a.pt", "ft-a.heads.npz",
                        "base:open_clip_model.safetensors"}
    assert got["ft-a.pt"] == paper_freeze.sha256_file(tmp_path / "ft-a.pt")


def test_check_names_every_part_that_differs_or_is_left_out():
    now = {"release": {"release_hash": "a"}, "vectors": {"m1": {"sha256": "v"}},
           "code_commit": "c"}
    assert paper_freeze.check(json.loads(json.dumps(now)), now) == []
    assert paper_freeze.check(None, now) == ["no freeze block in the config"]
    moved = {**now, "vectors": {"m1": {"sha256": "w"}}}
    assert paper_freeze.check(moved, now) == ["vectors.m1.sha256: frozen w, now v"]
    short = {k: v for k, v in now.items() if k != "release"}
    assert paper_freeze.check(short, now) == ["release: not in the freeze block"]
    extra = {**now, "gate": {"fingerprint": "f"}}
    assert paper_freeze.check(extra, now) == [
        "gate: frozen but not part of this config's predictor"]
    assert paper_freeze.check({**now, "_note": "x"}, now) == []


def test_a_val_run_reports_the_predictor_and_what_differs_and_goes_on(world, tmp_path):
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))
    pred = rep["freeze"]["predictor"]
    assert pred["release"]["release_hash"] == rep["dataset_release"]["release_hash"]
    assert pred["vectors"]["m1"]["photos"] > 0 and pred["vectors"]["m1"]["missing"] == 0
    assert rep["freeze"]["differences"] == ["no freeze block in the config"]
    assert rep["finds"]["scored"] == 2


def sealed_world(world, tmp_path, monkeypatch, freeze):
    """The test release made sealed, a research repo tagged with `config(..., freeze=)`,
    and the running code = the frozen commit; returns (cfg, body, repo)."""
    world["test"].manifest["draft"] = False
    cfg = config(tmp_path, ["7", "8"], code_commit="abcdef1234567",
                 **({"freeze": freeze} if freeze is not None else {}))
    body = json.dumps(cfg).encode()
    repo = tmp_path / "research"
    (repo / "paper").mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.org")
    git(repo, "config", "user.name", "t")
    (repo / "paper" / "analysis-v1.json").write_bytes(body)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "plan")
    git(repo, "tag", paper_score.TAG)
    monkeypatch.setattr(cfgmod, "code_version", lambda: "abcdef1234567")
    return cfg, body, repo


def frozen_spec(world, cfg):
    return paper_freeze.compute(cfg, world["base"], world["conn"], paper_score.resolve)


def test_the_sealed_test_is_scored_by_the_frozen_predictor(world, tmp_path, monkeypatch):
    spec = frozen_spec(world, config(tmp_path, ["7", "8"], code_commit="abcdef1234567"))
    cfg, body, repo = sealed_world(world, tmp_path, monkeypatch, spec)
    rep = paper_score.run(world["conn"], world["base"], world["test"], cfg, body,
                          tmp_path / "out", research_dir=repo, log=lambda s: None)
    assert rep["guard"]["sealed"] and rep["freeze"]["differences"] == []
    assert rep["freeze"]["predictor"]["vectors"]["m1"] == spec["vectors"]["m1"]


def test_the_sealed_test_refuses_a_predictor_that_is_not_the_frozen_one(world, tmp_path,
                                                                        monkeypatch):
    spec = frozen_spec(world, config(tmp_path, ["7", "8"], code_commit="abcdef1234567"))
    spec["vectors"]["m1"]["sha256"] = "0" * 64              # the index was re-embedded since
    cfg, body, repo = sealed_world(world, tmp_path, monkeypatch, spec)
    with pytest.raises(paper_score.FrozenError, match="vectors.m1.sha256"):
        paper_score.run(world["conn"], world["base"], world["test"], cfg, body,
                        tmp_path / "out", research_dir=repo, log=lambda s: None)
    assert not (tmp_path / "out").exists()


def test_the_sealed_test_refuses_a_config_without_a_freeze_block(world, tmp_path, monkeypatch):
    cfg, body, repo = sealed_world(world, tmp_path, monkeypatch, None)
    with pytest.raises(paper_score.FrozenError, match="no freeze block"):
        paper_score.run(world["conn"], world["base"], world["test"], cfg, body,
                        tmp_path / "out", research_dir=repo, log=lambda s: None)


def test_a_changed_vector_after_the_freeze_is_seen(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"])
    before = frozen_spec(world, cfg)
    ids, vecs = embed.load_embeddings(world["conn"], "m1")
    vecs = vecs.copy()
    vecs[0] = -vecs[0]
    after = paper_freeze.compute(cfg, world["base"], world["conn"], paper_score.resolve,
                                 load_vectors=lambda b: (ids, vecs))
    assert paper_freeze.check(before, after) == [
        f"vectors.m1.sha256: frozen {before['vectors']['m1']['sha256'][:16]}, "
        f"now {after['vectors']['m1']['sha256'][:16]}"]



def test_a_sealed_run_takes_a_test_output_filled_after_the_tag_and_records_its_sha256(
        world, tmp_path, monkeypatch):
    """PLAN §8: the E14 file is made at scoring, after the tag. The tagged config holds a
    TBD placeholder named in post_freeze; the sealed run scores the filled config and the
    report records the file's own sha256."""
    filled = config(tmp_path, ["7", "8"], code_commit="abcdef1234567")
    inat = next(c for c in filled["comparators"] if c["id"] == "inat-vision")
    filled["post_freeze"] = [{"path": "comparators/inat-vision/file",
                              "rule": "E14: iNat computer vision on v1-test at scoring",
                              "kind": "test_output"}]
    filled["freeze"] = frozen_spec(world, filled)
    tagged = json.loads(json.dumps(filled))
    next(c for c in tagged["comparators"] if c["id"] == "inat-vision")["file"] = "TBD: e14"
    tagged_body = json.dumps(tagged).encode()
    _, _, repo = sealed_world(world, tmp_path, monkeypatch, None)
    (repo / "paper" / "analysis-v1.json").write_bytes(tagged_body)
    git(repo, "commit", "-q", "-am", "the plan with post_freeze")
    git(repo, "tag", "-f", paper_score.TAG)
    body = json.dumps(filled).encode()
    rep = paper_score.run(world["conn"], world["base"], world["test"], filled, body,
                          tmp_path / "out", research_dir=repo, log=lambda s: None)
    g = rep["guard"]
    assert g["sealed"] and rep["freeze"]["differences"] == []
    assert g["tagged_config_sha256"] == paper_score.sha256_bytes(tagged_body)
    (pf,) = g["post_freeze"]
    assert pf["filled"] and pf["file_sha256"] == paper_freeze.sha256_file(
        paper_score.resolve(inat["file"]))
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Filled after the tag (post_freeze): 1 of 1" in md


def test_a_calibration_left_for_after_the_tag_is_not_frozen_so_filling_it_adds_a_part(
        world, tmp_path):
    cfg = config(tmp_path, ["7", "8"])
    vis = next(c for c in cfg["comparators"] if c["kind"] == "vision")
    vis["calibration"] = "TBD: refit on val"
    assert vis["id"] not in (frozen_spec(world, cfg).get("calibration") or {})
    vis["calibration"] = {"species": {"temperature": 0.05}}
    assert frozen_spec(world, cfg)["calibration"][vis["id"]] == vis["calibration"]
