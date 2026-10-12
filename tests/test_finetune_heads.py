"""A fine-tune keeps its classifier heads, and they can be scored as a method (ft-head).

Before 2026-10-11 finetune.py saved only the trained tail blocks, so "how good is the
fine-tune's own classifier?" could not be asked of any model. Old models must keep
loading; a method that needs heads must say plainly when a model has none."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from test_finetune import embedded, ft_loader
from test_models_and_scoreboard import seed_two_species
from test_trainer import laptop_with_sample

from mycomap_vision import config, evaluate, finetune, methods, trainer

NAMES = {"species": ["A x", "B y"], "genus": ["A", "B"], "family": ["F"]}


def angle(a):
    return [np.cos(a), np.sin(a)]


def write_model(root, name, heads=True, base="base"):
    """A fine-tuned model's files as finetune.finetune writes them (no torch needed)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{name}.pt").write_bytes(b"weights")
    meta = {"name": name, "base": base, "weights": f"{name}.pt", "trained_through": "2026-08-24",
            "test_days": 28, "created_at": "t"}
    if heads:
        # Const's vectors: red photos sit near angle 2.9, blue ones near 0.1.
        finetune.save_heads(root / f"{name}{finetune.HEADS_SUFFIX}",
                            {"species": np.asarray([angle(2.9), angle(0.1)]) * 3.0,
                             "genus": np.asarray([angle(2.9), angle(0.1)]),
                             "family": np.asarray([angle(1.5)])}, NAMES, 20.0)
        meta["heads"] = f"{name}{finetune.HEADS_SUFFIX}"
    (root / f"{name}.json").write_text(json.dumps(meta))
    return meta


def index_of(species_labels):
    """An evaluate.Index stand-in: one group per entry; None = a one-word group."""
    names = sorted({s for s in species_labels if s})
    pos = {n: i for i, n in enumerate(names)}
    return SimpleNamespace(species=[s or "Genus" for s in species_labels],
                           labels={"species": names},
                           label_of={"species": np.asarray([pos.get(s, -1) for s in species_labels])})


def test_a_fine_tune_keeps_its_classifier_heads_with_their_class_names(tmp_path):
    write_model(tmp_path, "m-ft")
    heads = finetune.read_heads("m-ft", tmp_path)
    vecs, names = heads["species"]
    assert names == ["A x", "B y"] and heads["genus"][1] == ["A", "B"]
    assert heads["scale"] == 20.0
    np.testing.assert_allclose(np.linalg.norm(vecs.astype(np.float32), axis=1), 1, atol=1e-3)
    with np.load(tmp_path / "m-ft.heads.npz", allow_pickle=False) as z:   # no pickled objects
        assert set(z.files) >= {"species", "species_names", "scale"}


def test_heads_and_their_names_must_line_up(tmp_path):
    with pytest.raises(ValueError, match="2 class vectors for 3 names"):
        finetune.save_heads(tmp_path / "x.heads.npz", {"species": np.ones((2, 2)),
                                                       "genus": np.ones((2, 2)),
                                                       "family": np.ones((1, 2))},
                            {**NAMES, "species": ["a", "b", "c"]}, 20.0)


def test_a_model_trained_before_heads_were_kept_still_loads_and_says_it_has_none(tmp_path):
    write_model(tmp_path, "old-ft", heads=False)
    assert finetune.read_meta("old-ft", tmp_path)["weights"] == "old-ft.pt"
    with pytest.raises(FileNotFoundError, match="kept no classifier heads"):
        finetune.read_heads("old-ft", tmp_path)
    m = methods.FinetuneHead()
    m.backbone, m.models_root = "old-ft", tmp_path
    with pytest.raises(FileNotFoundError, match="kept no classifier heads"):
        m.fit(np.zeros((0, 2)), index_of(["A x"]))


def test_ft_head_scores_with_the_fine_tunes_own_classifier(tmp_path):
    write_model(tmp_path, "m-ft")
    m = methods.FinetuneHead()
    m.backbone, m.models_root = "m-ft", tmp_path
    idx = index_of(["A x", "B y"])
    m.fit(np.zeros((0, 2)), idx)
    red, blue = np.asarray([angle(2.8)]), np.asarray([angle(0.2), angle(0.0)])
    assert np.argmax(m.species_scores(red)) == 0
    assert np.argmax(m.species_scores(blue)) == 1
    # The record's score is the mean of its photos' own (identify.py's per-photo path).
    per = m.per_photo_scores(None, blue)
    np.testing.assert_allclose(per.mean(axis=0), m.species_scores(blue), rtol=1e-6)


def test_a_species_the_head_never_saw_is_never_a_top_answer(tmp_path):
    write_model(tmp_path, "m-ft")
    m = methods.FinetuneHead()
    m.backbone, m.models_root = "m-ft", tmp_path
    # "C z" arrived after the fine-tune; the one-word group has no species label.
    idx = index_of(["A x", "B y", "C z", None])
    m.fit(np.zeros((0, 2)), idx)
    s = m.species_scores(np.asarray([angle(1.5)]))     # halfway: no clear winner
    assert s[2] == s[3] == methods.FinetuneHead.FLOOR
    assert s[2] < s[:2].min()
    assert m.coverage() == {"groups": 4, "groups_with_head": 2}


def test_ft_head_refuses_to_fit_without_knowing_its_backbone():
    with pytest.raises(ValueError, match="bind it with for_backbone"):
        methods.FinetuneHead().fit(np.zeros((0, 2)), index_of(["A x"]))


def test_a_comparison_scores_ft_head_with_the_heads_of_the_backbone_compared(conn, tmp_path,
                                                                             monkeypatch):
    store, root = embedded(conn, tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    write_model(tmp_path / "data" / "models", "base")
    out = evaluate.compare(conn, ["base"], ["nearest", "ft-head"], test_days=28,
                           embeddings_root=root, log=lambda s: None)
    board = {r["method"]: r for r in evaluate.scoreboard(conn, out["comparison_id"])}
    assert board["ft-head"]["species_top1"] == 1.0
    report = json.loads(conn.execute("select report_json from eval_runs where comparison_id = ? "
                                     "and method = 'ft-head'", (out["comparison_id"],)
                                     ).fetchone()[0])
    assert report["ft_head_coverage"] == {"groups": 2, "groups_with_head": 2}


def heads_finetuner(conn, base, name, embeddings_root, models_dir):
    meta = write_model(models_dir, name, base=base)
    finetune.register(conn, meta)
    return meta


def test_a_trainer_run_uploads_the_heads_and_a_pull_brings_them_home(tmp_path):
    from mycomap_vision.manifest import connect
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    store = seed_two_species(inst, tmp_path)
    uploads = {}
    out = trainer.run_job(inst, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=heads_finetuner, log=lambda s: None)
    inst.close()
    assert "runs/r1/models/timm_a-ft-r1.heads.npz" in uploads
    for key, body in uploads.items():
        dest = tmp_path / "bucket" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    run_dir = tmp_path / "bucket" / "runs" / "r1"
    conn, home = laptop_with_sample(tmp_path)
    trainer.merge_results(conn, run_dir / "manifest-out.sqlite", run_dir / "embeddings", out,
                          data_dir=home)
    assert finetune.read_heads("timm_a-ft-r1", home / "models")["species"][1] == ["A x", "B y"]


def test_a_run_from_before_heads_were_kept_still_pulls(tmp_path):
    from mycomap_vision.manifest import connect
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    store = seed_two_species(inst, tmp_path)
    uploads = {}

    def old_finetuner(conn, base, name, embeddings_root, models_dir):
        meta = write_model(models_dir, name, heads=False, base=base)
        finetune.register(conn, meta)
        return meta
    out = trainer.run_job(inst, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=old_finetuner, log=lambda s: None)
    inst.close()
    assert not any(k.endswith(".heads.npz") for k in uploads)
    for key, body in uploads.items():
        dest = tmp_path / "bucket" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    run_dir = tmp_path / "bucket" / "runs" / "r1"
    conn, home = laptop_with_sample(tmp_path)
    merged = trainer.merge_results(conn, run_dir / "manifest-out.sqlite", run_dir / "embeddings",
                                   out, data_dir=home)
    assert "timm_a-ft-r1" in merged["replaced"] and not merged["refused"]


def test_ft_head_runs_only_on_a_fine_tune_that_kept_its_heads(conn, tmp_path, monkeypatch):
    store, root = embedded(conn, tmp_path)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    write_model(tmp_path / "data" / "models", "old-ft", heads=False)
    assert not evaluate.method_applies("ft-head", "base")          # frozen: no model files
    assert not evaluate.method_applies("ft-head", "old-ft")        # trained before heads
    write_model(tmp_path / "data" / "models", "base")
    assert evaluate.method_applies("ft-head", "base")
    assert evaluate.method_applies("nearest", "old-ft")            # other methods: always
    m = evaluate.make_method("ft-head", "base")
    assert m.backbone == "base"


def test_ft_head_is_research_only_and_never_offered_on_the_site():
    assert "ft-head" in evaluate.METHODS
    assert not methods.method_ready("ft-head")
