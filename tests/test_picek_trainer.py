"""The Picek replication on the AWS trainer: planned, run, uploaded and scored with the rest
of a run (no torch: a stand-in trainer writes the model's files)."""

import json

import numpy as np
import pytest
from test_finetune import ft_loader
from test_models_and_scoreboard import seed_two_species
from test_picek import write_head

from mycomap_vision import aws, config, evaluate, finetune, picek, trainer


def test_a_replication_trains_from_photos_then_is_embedded_before_other_backbones():
    stages = trainer.plan_stages(["timm:a", "timm:b"], ["timm:a"], "r1",
                                 picek=["fungitastic-beit-b384@15"])
    assert [(s.kind, s.name) for s in stages] == [
        ("embed", "timm_a"), ("finetune", "timm_a-ft-r1"), ("embed", "timm_a-ft-r1"),
        ("picek", "picek-fungitastic-beit-b384-r1"), ("embed", "picek-fungitastic-beit-b384-r1"),
        ("embed", "timm_b")]
    est = trainer.estimate(stages[3:5], 580_000)
    row = est["stages"][0]
    assert "15 epochs" in row["stage"] and row["measured"] is False
    assert row["hours"] == pytest.approx(
        580_000 * 15 / trainer.picek_rate("fungitastic-beit-b384", False) / 3600, abs=0.1)


def test_preset_specs_are_checked_before_anything_is_paid_for(conn, tmp_path):
    assert picek.parse_spec("df20-vit-l384") == ("df20-vit-l384", 100)
    assert picek.parse_spec("fungitastic-beit-b384@15") == ("fungitastic-beit-b384", 15)
    for bad in ("nope", "fungitastic-beit-b384@0", "fungitastic-beit-b384@x"):
        with pytest.raises(ValueError):
            picek.parse_spec(bad)
    seed_two_species(conn, tmp_path)
    conn.execute("insert into photo_copies values (1000, 's3://b/', 'large', 'p', 1, 'h', 't')")
    with pytest.raises(ValueError, match="unknown Picek preset"):
        aws.check_trainer_request(conn, ["bioclip-2"], ["nearest"], "large", "s3://b/",
                                  picek=["beit-huge"])
    # A run may train only the replication (its own model is then the only backbone).
    assert aws.check_trainer_request(conn, [], ["classifier"], "large", "s3://b/",
                                     picek=["fungitastic-beit-b384"]) == 1
    with pytest.raises(ValueError, match="at least one backbone"):
        aws.check_trainer_request(conn, [], ["nearest"], "large", "s3://b/")
    ud = aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest", "classifier"], 10, "bkt",
                                      picek=["fungitastic-beit-b384@15"])
    assert "--picek fungitastic-beit-b384@15" in ud
    alone = aws.render_trainer_user_data("r", [], ["classifier"], 10, "bkt",
                                         picek=["fungitastic-beit-b384"])
    assert "--backbones none --methods classifier" in alone
    assert "--picek" not in aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10,
                                                         "bkt")


def fake_picek_trainer(conn, spec, name, models_dir):
    write_head(models_dir, name=name, classes=("A x", "B y"), genus=("A", "B"),
               weight=np.array([[3.0], [-3.0]], np.float32), temperature=1.0, scale=2.0)
    (models_dir / f"{name}.pt").write_bytes(b"weights")
    (models_dir / f"{name}.records.csv").write_text("observation_id,split\n100,train\n")
    meta = {"name": name, "kind": picek.KIND, "base": "timm:x", "preset": spec,
            "trained_through": "2026-08-24", "test_days": 28, "created_at": "t"}
    (models_dir / f"{name}.json").write_text(json.dumps(meta))
    finetune.register(conn, meta)
    return meta


def test_the_run_trains_uploads_and_scores_the_replication_with_its_own_methods(
        conn, tmp_path, monkeypatch):
    store = seed_two_species(conn, tmp_path)
    data_dir = tmp_path / "inst"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)      # as on the instance
    uploads = {}
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest", "classifier"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=data_dir, picek=["fungitastic-beit-b384@2"],
                          picek_trainer=fake_picek_trainer, log=lambda s: None)
    name = "picek-fungitastic-beit-b384-r1"
    assert list(out["embedded"]) == [name, "timm_a"]
    for suffix in trainer.PICEK_FILES:
        assert f"runs/r1/models/{name}{suffix}" in uploads, suffix
    board = evaluate.scoreboard(conn, out["comparison_id"])
    # The classifier method scores the replication only; nearest scores both.
    assert {(r["backbone"], r["method"]) for r in board} == {
        (name, "nearest"), (name, "classifier"), ("timm_a", "nearest")}
    assert trainer.resume_request(out)["picek"] == ["fungitastic-beit-b384@2"]


def test_a_photo_cache_shortens_the_estimate_and_counts_its_own_pass():
    plain = trainer.estimate(trainer.plan_stages([], None, "r", ["fungitastic-beit-b384@15"])[:1],
                             580_000)["stages"][0]["hours"]
    cached = trainer.estimate(trainer.plan_stages([], None, "r",
                                                  ["fungitastic-beit-b384@15@440"])[:1],
                              580_000)["stages"][0]["hours"]
    build = 580_000 / trainer.PICEK_CACHE_BUILD_RATE / 3600
    rate = trainer.picek_rate("fungitastic-beit-b384", True)
    assert cached == pytest.approx(580_000 * 15 / rate / 3600 + build, abs=0.15)


def test_the_training_rate_is_the_slower_of_loader_and_gpu(monkeypatch):
    monkeypatch.setitem(trainer.PICEK_LOADER_RATES, "x", 50.0)
    monkeypatch.setitem(trainer.PICEK_GPU_RATES, "x", 60.0)
    assert trainer.picek_rate("x", False) == 50.0                 # loader-bound
    assert trainer.picek_rate("x", True) == 60.0                  # cache: now GPU-bound
    assert trainer.picek_rate("unknown", False) is None


def test_a_photo_cache_gets_its_own_disk_on_the_instance():
    assert aws.picek_cache_gb(["fungitastic-beit-b384@15"], 590_000) == 0
    gb = aws.picek_cache_gb(["fungitastic-beit-b384@15@440"], 590_000)
    assert 50 <= gb <= 55                                   # 590k x ~80 KB, +10%
    args = aws.trainer_instance_args("r", "ami-1", "#!", "g6.xlarge", "/dev/xvda", 75,
                                     extra_gb=gb)
    assert args["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 75 + 60 + gb
