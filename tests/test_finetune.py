import json

import numpy as np
import pytest
from test_models_and_scoreboard import Const, seed_two_species
from test_trainer import laptop_with_sample

from mycomap_vision import aws, config, evaluate, finetune, models, trainer
from mycomap_vision.embed import embed_photos, photos_to_embed

# seed_two_species: six old records (Jan 2026) and two new ones (Sept 20/21, photos
# 1006 and 1007). With 28 test days the cutoff is 2026-08-24 and the new two are the test.
TEST_PHOTOS = {1006, 1007}


def embedded(conn, tmp_path, name="base", backbone=None):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, name, "large", store.location, "local")
    embed_photos(conn, store, backbone or Const(name), todo, root / name, log=lambda s: None)
    return store, root


# --- what it trains on -------------------------------------------------------

def test_fine_tuning_trains_only_on_reference_records_never_the_test_weeks(conn, tmp_path):
    store, root = embedded(conn, tmp_path)
    ts = finetune.build_trainset(conn, "base", store.location, "large", 28, root)
    assert {pid for pid, _ in ts.items}.isdisjoint(TEST_PHOTOS)
    assert len(ts.items) == 6
    assert ts.trained_through == evaluate.shared_records(conn, ["base"], 28, embeddings_root=root
                                                         ).cutoff
    assert ts.names["species"] == ["A x", "B y"] and ts.names["genus"] == ["A", "B"]
    assert set(ts.labels[:, 0]) == {0, 1}


def test_a_fine_tuned_model_is_never_scored_on_records_it_trained_on(conn, tmp_path):
    store, root = embedded(conn, tmp_path)
    todo = photos_to_embed(conn, "base-ft", "large", store.location, "local")
    embed_photos(conn, store, Const("base-ft", twist=0.2), todo, root / "base-ft",
                 log=lambda s: None)
    meta = {"name": "base-ft", "base": "base", "test_days": 28, "created_at": "t",
            "trained_through": "2026-09-21"}          # it saw the test weeks
    finetune.register(conn, meta)
    with pytest.raises(ValueError, match="inflated"):
        evaluate.compare(conn, ["base", "base-ft"], test_days=28, embeddings_root=root,
                         log=lambda s: None)
    # Trained through the cutoff itself: allowed, and scored.
    finetune.register(conn, {**meta, "trained_through": "2026-08-24"})
    out = evaluate.compare(conn, ["base", "base-ft"], test_days=28, embeddings_root=root,
                           log=lambda s: None)
    assert out["test_records"] == 2


def test_only_the_last_blocks_and_what_follows_them_are_trained():
    clip = ["class_embedding", "positional_embedding", "proj", "conv1.weight", "ln_pre.weight",
            "transformer.resblocks.0.attn.in_proj_weight", "transformer.resblocks.22.ln_1.bias",
            "transformer.resblocks.23.mlp.c_fc.weight", "ln_post.weight"]
    assert finetune.block_count(clip) == 24
    trained = {n for n in clip if finetune.is_trainable(n, 24, 2)}
    assert trained == {"transformer.resblocks.22.ln_1.bias",
                       "transformer.resblocks.23.mlp.c_fc.weight", "ln_post.weight", "proj"}
    timm = ["cls_token", "pos_embed", "patch_embed.proj.weight", "blocks.10.attn.qkv.weight",
            "blocks.11.mlp.fc1.weight", "norm.weight", "fc_norm.bias", "head.weight"]
    trained = {n for n in timm if finetune.is_trainable(n, 12, 1)}
    assert trained == {"blocks.11.mlp.fc1.weight", "norm.weight", "fc_norm.bias", "head.weight"}


def test_sampling_sits_between_the_natural_mix_and_fully_balanced():
    species = np.array([0] * 90 + [1] * 10)
    share = lambda w: w[species == 1].sum()  # noqa: E731
    assert share(finetune.sampling_weights(species, 0.0)) == pytest.approx(0.10)
    assert share(finetune.sampling_weights(species, 1.0)) == pytest.approx(0.50)
    assert 0.10 < share(finetune.sampling_weights(species, 0.5)) < 0.50


def test_classifiers_start_from_each_class_average():
    v = np.array([[1, 0], [0.8, 0.6], [0, 1]], dtype=np.float16)
    p = finetune.class_prototypes(v, np.array([0, 0, -1]), 2)
    assert np.allclose(np.linalg.norm(p[0]), 1) and p[0][0] > p[0][1] > 0
    assert not p[1].any()                          # no photos: no direction


# --- in the AWS run ----------------------------------------------------------

def fake_finetuner(conn, base, name, embeddings_root, models_dir):
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{name}.pt").write_bytes(b"weights")
    meta = {"name": name, "base": base, "trained_through": "2026-08-24", "test_days": 28,
            "created_at": "t"}
    (models_dir / f"{name}.json").write_text(json.dumps(meta))
    finetune.register(conn, meta)
    return meta


def ft_loader(spec):
    name = models.storage_name(spec)
    return Const(name, twist=0.3 if "-ft-" in name else 0.0)


def test_the_run_fine_tunes_embeds_uploads_and_compares_before_and_after(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    uploads = {}
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=fake_finetuner, log=lambda s: None)
    assert list(out["embedded"]) == ["timm_a", "timm_a-ft-r1"]
    assert out["finetuned"]["timm_a-ft-r1"]["trained_through"] == "2026-08-24"
    assert "runs/r1/models/timm_a-ft-r1.pt" in uploads
    assert "runs/r1/embeddings/timm_a-ft-r1/shard-00000.npy" in uploads
    board = evaluate.scoreboard(conn, out["comparison_id"])
    assert {r["backbone"] for r in board} == {"timm_a", "timm_a-ft-r1"}


def test_a_failed_fine_tune_leaves_the_rest_of_the_run_intact(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)

    def broken(*a):
        raise RuntimeError("CUDA out of memory")
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"], lambda p, k: None, "r2",
                          loader=ft_loader, data_dir=tmp_path / "inst2", finetune=["timm:a"],
                          finetuner=broken, log=lambda s: None)
    assert "out of memory" in out["failed"]["timm_a-ft-r2"]
    assert list(out["embedded"]) == ["timm_a"] and out["comparison_id"]


def test_fine_tuning_is_refused_unless_its_base_is_in_the_run(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    conn.execute("insert into photo_copies values (1000, 's3://b/', 'large', 'p', 1, 'h', 't')")
    with pytest.raises(ValueError, match="needs it in --backbones"):
        aws.check_trainer_request(conn, ["dinov2-l14"], ["nearest"], "large", "s3://b/",
                                  finetune=["bioclip-2"])
    ud = aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                      finetune=["bioclip-2"])
    assert '--source "s3://$BUCKET" --finetune bioclip-2' in ud
    assert "--finetune" not in aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10,
                                                            "bkt")


def test_pulling_a_run_brings_the_fine_tuned_weights_and_their_training_date(tmp_path,
                                                                            monkeypatch):
    inst_conn_path = tmp_path / "inst-manifest" / "manifest.sqlite"
    from mycomap_vision.manifest import connect
    inst = connect(inst_conn_path)
    store = seed_two_species(inst, tmp_path)
    uploads = {}
    out = trainer.run_job(inst, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=fake_finetuner, log=lambda s: None)
    inst.close()
    run_dir = tmp_path / "bucket" / "runs" / "r1"
    for key, body in uploads.items():
        dest = tmp_path / "bucket" / key
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(body)
    conn, home = laptop_with_sample(tmp_path)
    merged = trainer.merge_results(conn, run_dir / "manifest-out.sqlite", run_dir / "embeddings",
                                   out, data_dir=home)
    assert "timm_a-ft-r1" in merged["replaced"]
    assert (home / "models" / "timm_a-ft-r1.pt").read_bytes() == b"weights"
    row = conn.execute("select trained_through from finetunes where name = 'timm_a-ft-r1'"
                       ).fetchone()
    assert row[0] == "2026-08-24"
    # The laptop now finds it by name.
    monkeypatch.setattr(config, "DATA_DIR", home)
    assert models.resolve_spec("timm_a-ft-r1") == "finetuned:timm_a-ft-r1"
    assert models.storage_name("finetuned:timm_a-ft-r1") == "timm_a-ft-r1"
    with pytest.raises(ValueError, match="unknown backbone"):
        models.resolve_spec("not-a-model")
