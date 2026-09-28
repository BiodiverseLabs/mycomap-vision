import json

import numpy as np
import pytest
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import aws, evaluate, models, trainer
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.manifest import connect


def const_loader(spec):
    return Const(models.storage_name(spec))


def run(conn, store, tmp_path, backbones=("timm:a",), methods=("nearest",), loader=None):
    uploads = {}

    def upload(path, key):
        uploads[key] = path.read_bytes()
    out = trainer.run_job(conn, store, list(backbones), list(methods), upload, "r1",
                          loader=loader or const_loader, data_dir=tmp_path / "inst",
                          log=lambda s: None)
    return out, uploads


# --- on the instance -------------------------------------------------------

def test_embedding_rows_from_the_laptop_are_cleared_so_every_photo_is_embedded_again(conn,
                                                                                    tmp_path):
    store = seed_two_species(conn, tmp_path)
    # The laptop embedded some photos; its shard files never reach the instance.
    todo = photos_to_embed(conn, "timm_a", "large", store.location, "local")[:3]
    embed_photos(conn, store, Const("timm_a"), todo, tmp_path / "laptop" / "timm_a",
                 log=lambda s: None)
    out, _ = run(conn, store, tmp_path)
    assert out["failed"] == {}
    assert conn.execute("select count(*) from embeddings where backbone = 'timm_a'"
                        ).fetchone()[0] == 8
    assert out["comparison_id"]


def test_the_run_uploads_shards_reports_manifest_and_result_last(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    out, uploads = run(conn, store, tmp_path, methods=("nearest", "species-mean"))
    keys = list(uploads)
    assert "runs/r1/embeddings/timm_a/shard-00000.npy" in keys
    assert "runs/r1/embeddings/timm_a/shard-00000.ids.npy" in keys
    assert f"runs/r1/reports/compare-{out['comparison_id']}.json" in keys
    assert "runs/r1/manifest-out.sqlite" in keys
    assert keys[-1] == "runs/r1/result.json"          # its presence means the rest is there
    assert json.loads(uploads["runs/r1/result.json"])["comparison_id"] == out["comparison_id"]
    # Nothing is written outside the run's own folder (the downloader's manifest included).
    assert all(k.startswith("runs/r1/") for k in keys)


def test_a_failing_backbone_is_reported_and_the_others_still_compared(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)

    def loader(spec):
        if spec == "timm:bad":
            raise RuntimeError("CUDA out of memory")
        return const_loader(spec)
    out, uploads = run(conn, store, tmp_path, backbones=("timm:bad", "timm:a"), loader=loader)
    assert "out of memory" in out["failed"]["timm_bad"]
    assert list(out["embedded"]) == ["timm_a"]
    assert not any("timm_bad" in k for k in uploads)
    board = evaluate.scoreboard(conn, out["comparison_id"])
    assert {r["backbone"] for r in board} == {"timm_a"}


def test_a_run_where_nothing_embedded_still_uploads_a_result(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)

    def loader(spec):
        raise OSError("gated")
    out, uploads = run(conn, store, tmp_path, loader=loader)
    assert out["comparison_id"] is None
    assert "runs/r1/result.json" in uploads


# --- launching ---------------------------------------------------------------

def test_the_trainer_always_shuts_itself_down_and_uses_its_own_manifest_copy():
    ud = aws.render_trainer_user_data("20260929-010000", ["bioclip-2", "dinov3-l16"],
                                      ["nearest", "linear"], 10, "bkt")
    assert "shutdown -h +630" in ud                   # backstop: max hours + 30 minutes
    assert "timeout 36000 " in ud                     # the job itself stops at max hours
    assert "trap finish EXIT" in ud and "shutdown -h now" in ud
    assert '"s3://$BUCKET/$RUN/manifest-in.sqlite"' in ud
    assert aws.MANIFEST_KEY not in ud                 # never the downloader's shared manifest
    assert "--backbones bioclip-2,dinov3-l16 --methods nearest,linear" in ud
    assert "nvidia-smi ||" in ud                      # no GPU: stop instead of paying for CPU


def test_trainer_instances_are_tagged_terminate_on_shutdown_and_fit_the_image(monkeypatch):
    monkeypatch.delenv("MV_INSTANCE_ROLE", raising=False)
    args = aws.trainer_instance_args("r9", "ami-1", "#!", "g6.2xlarge", "/dev/xvda", 75)
    assert args["InstanceType"] == "g6.2xlarge"
    assert args["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert args["MetadataOptions"] == {"HttpTokens": "required"}
    disk = args["BlockDeviceMappings"][0]
    assert disk["DeviceName"] == "/dev/xvda" and disk["Ebs"]["VolumeSize"] >= 75 + 50
    assert disk["Ebs"]["DeleteOnTermination"] is True
    for spec in args["TagSpecifications"]:
        assert {"Key": "Project", "Value": "mycomap-vision"} in spec["Tags"]
        assert {"Key": "Name", "Value": "mv-trainer-r9"} in spec["Tags"]


def test_a_trainer_run_is_refused_before_paying_when_it_cannot_work(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    conn.execute("insert into photo_copies values (1000, 's3://b/', 'large', 'p', 1, 'h', 't')")
    with pytest.raises(ValueError, match="unknown backbone"):
        aws.check_trainer_request(conn, ["nope"], ["nearest"], "large", "s3://b/")
    with pytest.raises(ValueError, match="unknown method"):
        aws.check_trainer_request(conn, ["bioclip-2"], ["magic"], "large", "s3://b/")
    with pytest.raises(ValueError, match="no large photos"):
        aws.check_trainer_request(conn, ["bioclip-2"], ["nearest"], "large", "s3://other/")
    assert aws.check_trainer_request(conn, ["bioclip-2"], ["nearest"], "large", "s3://b/") == 1


def test_the_ops_policy_can_find_the_deep_learning_image(monkeypatch):
    monkeypatch.setenv("MV_S3_BUCKET", "bkt")
    monkeypatch.setenv("MV_AWS_REGION", "us-east-2")
    policy = json.loads(aws.render_policy("ops-policy.template.json"))
    ssm = [s for s in policy["Statement"] if s["Action"] == "ssm:GetParameter"][0]
    param = aws.TRAINER_AMI_PARAMETER
    assert any(param.startswith(r.split("parameter", 1)[1].rstrip("*"))
               for r in ssm["Resource"])


# --- bringing it home --------------------------------------------------------

def finished_run(tmp_path):
    """A completed run on an 'instance' manifest, its uploads written to a fake bucket."""
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    store = seed_two_species(inst, tmp_path)
    out, uploads = run(inst, store, tmp_path, methods=("nearest", "species-mean"))
    inst.close()
    bucket = tmp_path / "bucket"
    for key, body in uploads.items():
        (bucket / key).parent.mkdir(parents=True, exist_ok=True)
        (bucket / key).write_bytes(body)
    return out, bucket / "runs" / "r1"


def laptop_with_sample(tmp_path, names=("timm_a", "other")):
    home = tmp_path / "home"
    conn = connect(home / "manifest.sqlite")
    store = seed_two_species(conn, tmp_path / "laptop-store")
    for name in names:
        todo = photos_to_embed(conn, name, "large", store.location, "local")[:2]
        embed_photos(conn, store, Const(name, twist=1.0), todo, home / "embeddings" / name,
                     log=lambda s: None)
    return conn, home


def test_pulled_embeddings_replace_the_sample_and_the_sample_is_archived(tmp_path):
    out, run_dir = finished_run(tmp_path)
    conn, home = laptop_with_sample(tmp_path)
    merged = trainer.merge_results(conn, run_dir / "manifest-out.sqlite", run_dir / "embeddings",
                                   out, data_dir=home)
    assert merged["replaced"] == ["timm_a"]
    count = lambda b: conn.execute("select count(*) from embeddings where backbone = ?",  # noqa
                                   (b,)).fetchone()[0]
    assert count("timm_a") == 8                        # the run's full set
    assert count("other") == 2                         # a backbone the run didn't touch
    archive = home / "embeddings-archive" / "timm_a-before-r1"
    assert (archive / "shard-00000.npy").exists()
    ids, vecs = __import__("mycomap_vision.embed", fromlist=["x"]).load_embeddings(
        conn, "timm_a", home / "embeddings" / "timm_a")
    assert len(ids) == 8 and vecs.shape == (8, 2)
    assert np.isfinite(vecs.astype(np.float32)).all()
    board = evaluate.scoreboard(conn, out["comparison_id"])
    assert {(r["backbone"], r["method"]) for r in board} == {("timm_a", "nearest"),
                                                              ("timm_a", "species-mean")}


def test_pulling_the_same_run_twice_changes_nothing(tmp_path):
    out, run_dir = finished_run(tmp_path)
    conn, home = laptop_with_sample(tmp_path)
    args = (conn, run_dir / "manifest-out.sqlite", run_dir / "embeddings", out)
    trainer.merge_results(*args, data_dir=home)
    again = trainer.merge_results(*args, data_dir=home)
    assert again["replaced"] == [] and again["scoreboard_rows_added"] == 0
    assert conn.execute("select count(*) from eval_runs").fetchone()[0] == 2
    assert conn.execute("select count(*) from embeddings where backbone = 'timm_a'"
                        ).fetchone()[0] == 8
    assert len(list((home / "embeddings-archive").iterdir())) == 1
    assert conn.execute("select count(*) from embed_runs where backbone = 'timm_a'"
                        ).fetchone()[0] == 2           # the laptop's run and the instance's


def test_only_the_runs_own_comparison_comes_home(tmp_path):
    out, run_dir = finished_run(tmp_path)
    conn, home = laptop_with_sample(tmp_path)
    # A scoreboard row the instance carried over from the laptop's snapshot.
    remote = connect(run_dir / "manifest-out.sqlite")
    remote.execute("insert into eval_runs (comparison_id, backbone, method, cutoff, test_days, "
                   "n_reference, n_test, record_set, report_json, created_at) values "
                   "('old-laptop-run', 'x', 'nearest', 'c', 28, 1, 1, 'h', '{}', 't')")
    remote.commit()
    remote.close()
    merged = trainer.merge_results(conn, run_dir / "manifest-out.sqlite",
                                   run_dir / "embeddings", out, data_dir=home)
    assert merged["scoreboard_rows_added"] == 2
    ids = {r[0] for r in conn.execute("select distinct comparison_id from eval_runs")}
    assert ids == {out["comparison_id"]}
