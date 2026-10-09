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


# --- launch readiness ---------------------------------------------------------------------

def head_sha():
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def test_a_dry_run_builds_and_checks_everything_and_sends_nothing(conn, tmp_path, monkeypatch):
    seed_two_species(conn, tmp_path)
    with conn:
        conn.executemany("insert into photo_copies values (?, 's3://bkt/', 'large', ?, 1, 'h', "
                         "'t')", [(1000 + i, f"p/{1000 + i}.png") for i in range(8)])
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setattr(aws, "release_commit", lambda **k: head_sha())

    def no_aws(*a, **k):
        raise AssertionError("a dry run must not touch AWS")
    monkeypatch.setattr(aws, "s3_client", no_aws)
    monkeypatch.setattr(aws, "session", no_aws)
    monkeypatch.setattr("shutil.which", lambda name: None)      # bash -n: tested elsewhere
    out = aws.launch_trainer(conn, [], ["classifier", "classifier+month"], max_hours=48,
                             instance_type="g6.xlarge", picek=["fungitastic-beit-b384@15"],
                             dry_run=True, dry_run_dir=tmp_path / "dry", log=lambda s: None)
    assert out["dry_run"] and out["sent"].startswith("nothing")
    for f in ("code.tar.gz", "user-data.sh", "manifest-in.sqlite", "picek-labels.json",
              "ec2-request.json"):
        assert (tmp_path / "dry" / f).is_file(), f
    assert out["code_archive"]["pins"]["timm"] == "1.0.30"
    labels = out["picek_labels"]
    assert labels["records_train"] + labels["records_validation"] == 6
    script = (tmp_path / "dry" / "user-data.sh").read_text(encoding="utf-8")
    assert f"--picek-labels-hash {labels['labels_hash']}" in script
    assert "--picek-exclude-benchmarks match" in script and "picek-exclude-ids" not in script
    request = json.loads((tmp_path / "dry" / "ec2-request.json").read_text(encoding="utf-8"))
    assert request["InstanceType"] == "g6.xlarge"


def test_the_code_archive_must_carry_what_a_picek_run_needs():
    import io
    import tarfile

    def archive(files):
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w:gz") as t:
            for name, text in files.items():
                data = text.encode()
                info = tarfile.TarInfo(name)
                info.size = len(data)
                t.addfile(info, io.BytesIO(data))
        return buf.getvalue()
    base = {n: "" for n in aws.TRAINER_NEEDS}
    base["requirements/trainer.txt"] = "timm==1.0.30 \\n    --hash=sha256:x\ntorch==2.14.0 \\n"
    assert aws.check_code_archive(archive(base))["pins"] == {"timm": "1.0.30", "torch": "2.14.0"}
    with pytest.raises(RuntimeError, match="picek.py"):
        aws.check_code_archive(archive(base), picek=["fungitastic-beit-b384"])
    assert aws.check_code_archive(archive({**base, "src/mycomap_vision/picek.py": ""}),
                                  picek=["fungitastic-beit-b384"])


def test_excluding_every_benchmark_ships_the_ids_to_the_instance():
    ud = aws.render_trainer_user_data("r", [], ["classifier"], 10, "bkt",
                                      picek=["fungitastic-beit-b384@15"], picek_exclude="all",
                                      picek_labels_hash="abc")
    assert 'aws s3 cp "s3://$BUCKET/$RUN/picek-exclude-ids.txt" data/picek-exclude-ids.txt' in ud
    assert "--picek-exclude-benchmarks all --picek-exclude-ids data/picek-exclude-ids.txt" in ud
    assert "--picek-labels-hash abc" in ud


def test_a_resumed_run_keeps_its_benchmark_exclusion():
    doc = {"run_id": "r", "picek_exclude_benchmarks": "all",
           "stages": [{"kind": "picek", "name": "picek-x-r", "spec": "fungitastic-beit-b384@15",
                       "status": "done"}]}
    assert trainer.resume_request(doc)["picek_exclude_benchmarks"] == "all"
