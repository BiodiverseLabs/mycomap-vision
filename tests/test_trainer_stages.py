"""The trainer before its first real run: stage order, the time estimate, a run that
stops at its time limit, partial pulls, the commit it runs and how it installs."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from test_models_and_scoreboard import Const, seed_two_species
from test_trainer import laptop_with_sample

from mycomap_vision import aws, config, models, trainer
from mycomap_vision.manifest import connect

REPO = Path(__file__).resolve().parents[1]


class Recorder:
    """upload(path, key) into a dict, keeping every version of each key in order."""

    def __init__(self):
        self.files, self.log = {}, []

    def __call__(self, path, key):
        self.files[key] = Path(path).read_bytes()
        self.log.append(key)

    def json(self, key):
        return json.loads(self.files[key])

    def write_bucket(self, root: Path, files=None):
        for key, body in (files or self.files).items():
            (root / key).parent.mkdir(parents=True, exist_ok=True)
            (root / key).write_bytes(body)
        return root


class Counting(Const):
    def __init__(self, name):
        super().__init__(name)
        self.batches = 0

    def encode(self, images):
        self.batches += 1
        return super().encode(images)


def fake_finetuner(conn, base, name, embeddings_root, models_dir):
    from mycomap_vision import finetune
    models_dir.mkdir(parents=True, exist_ok=True)
    (models_dir / f"{name}.pt").write_bytes(b"weights")
    meta = {"name": name, "base": base, "trained_through": "2026-08-24", "test_days": 28,
            "created_at": "t"}
    (models_dir / f"{name}.json").write_text(json.dumps(meta))
    finetune.register(conn, meta)
    return meta


# --- stage order -----------------------------------------------------------------

def test_the_fine_tune_follows_its_base_directly_and_other_backbones_come_after():
    stages = trainer.plan_stages(["dinov3-l16-512", "bioclip-2"], ["bioclip-2"], "r1")
    assert [(s.kind, s.name) for s in stages] == [
        ("embed", "bioclip-2"), ("finetune", "bioclip-2-ft-r1"), ("embed", "bioclip-2-ft-r1"),
        ("embed", "dinov3-l16-512")]
    assert [(s.kind, s.name) for s in trainer.plan_stages(["bioclip-2"], [], "r1")] == [
        ("embed", "bioclip-2")]


def test_the_run_embeds_fine_tunes_and_re_embeds_before_touching_another_backbone(conn,
                                                                                tmp_path):
    store = seed_two_species(conn, tmp_path)
    order = []

    def loader(spec):
        order.append(models.storage_name(spec))
        return Const(models.storage_name(spec))

    def finetuner(*a):
        order.append(f"finetune {a[2]}")
        return fake_finetuner(*a)
    out = trainer.run_job(conn, store, ["timm:b", "timm:a"], ["nearest"], Recorder(), "r1",
                          loader=loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=finetuner, log=lambda s: None)
    assert order == ["timm_a", "finetune timm_a-ft-r1", "timm_a-ft-r1", "timm_b"]
    assert out["state"] == "finished"


# --- the time estimate -------------------------------------------------------------

def test_the_recommended_first_run_fits_its_24_hours_and_the_old_plan_does_not_fit_14():
    first = trainer.estimate(trainer.plan_stages(["bioclip-2"], ["bioclip-2"], "r"), 593_214)
    assert first["total_hours"] < 24 - trainer.STOP_MARGIN_HOURS
    aws.check_run_time(first, 24, log=lambda s: None)
    old = trainer.estimate(trainer.plan_stages(["bioclip-2", "dinov3-l16-512"], ["bioclip-2"],
                                               "r"), 593_214)
    assert old["total_hours"] > 14
    with pytest.raises(ValueError, match="--allow-over-time"):
        aws.check_run_time(old, 14, log=lambda s: None)
    said = []
    aws.check_run_time(old, 14, allow_over_time=True, log=said.append)
    assert said and said[0].startswith("WARNING")


def test_an_unmeasured_backbone_is_estimated_slow_and_flagged():
    est = trainer.estimate(trainer.plan_stages(["timm:new"], [], "r"), 100_000)
    assert est["stages"][0]["measured"] is False
    assert est["stages"][0]["photos_per_second"] == trainer.UNMEASURED_EMBED_RATE
    assert "NOT MEASURED" in trainer.format_estimate(est)


def seed_s3_photos(conn, n=10):
    conn.executemany("insert into photo_copies values (?, 's3://bkt/', 'large', 'p', 1, 'h', 't')",
                     [(5000 + i,) for i in range(n)])
    conn.commit()


def test_a_launch_over_the_time_limit_is_refused_before_any_aws_call(conn, tmp_path,
                                                                    monkeypatch):
    seed_two_species(conn, tmp_path)
    seed_s3_photos(conn)
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setattr(trainer, "EMBED_RATES", {"bioclip-2": 1e-4})   # 10 photos: days

    def no_aws(*a, **k):
        raise AssertionError("AWS was called")
    monkeypatch.setattr(aws, "session", no_aws)
    monkeypatch.setattr(aws, "s3_client", no_aws)
    monkeypatch.setattr(aws, "release_commit", lambda **k: "a" * 40)
    with pytest.raises(ValueError, match="needs about"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           log=lambda s: None)


# --- a launch, with AWS mocked ---------------------------------------------------------

class FakeEC2:
    def __init__(self):
        self.launched = None

    def describe_images(self, ImageIds):
        return {"Images": [{"RootDeviceName": "/dev/xvda", "BlockDeviceMappings": [
            {"DeviceName": "/dev/xvda", "Ebs": {"VolumeSize": 75}}]}]}

    def run_instances(self, **kw):
        self.launched = kw
        return {"Instances": [{"InstanceId": "i-1"}]}


class FakeSSM:
    def get_parameter(self, Name):
        return {"Parameter": {"Value": "ami-1"}}


class FakeLaunchS3:
    def __init__(self):
        self.objects = {}

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def upload_file(self, path, bucket, key):
        self.objects[key] = Path(path).read_bytes()


def test_a_launch_ships_the_exact_commit_and_the_instance_records_it(conn, tmp_path,
                                                                     monkeypatch):
    seed_two_species(conn, tmp_path)
    seed_s3_photos(conn)
    sha = "0123456789abcdef0123456789abcdef01234567"
    ec2, s3, archived = FakeEC2(), FakeLaunchS3(), []
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setenv("MV_AWS_REGION", "us-east-2")
    monkeypatch.setattr(aws, "s3_client", lambda: s3)
    monkeypatch.setattr(aws, "session", lambda: type("S", (), {
        "client": lambda self, n: {"ec2": ec2, "ssm": FakeSSM()}[n]})())
    monkeypatch.setattr(aws, "release_commit", lambda **k: sha)
    monkeypatch.setattr(aws, "code_tarball", lambda ref: archived.append(ref) or b"tgz")
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             log=lambda s: None)
    assert archived == [sha]                           # the archive is that commit, not HEAD
    assert out["code_version"] == sha
    assert f"export MV_CODE_VERSION={sha}" in ec2.launched["UserData"]
    assert out["max_hours"] == 24


def test_uncommitted_or_unpushed_code_is_refused_unless_allowed(tmp_path):
    def git(*a, cwd=tmp_path / "repo"):
        return subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    (tmp_path / "repo").mkdir()
    git("init", "-q")
    git("config", "user.email", "t@example.org")
    git("config", "user.name", "t")
    (tmp_path / "repo" / "f.txt").write_text("1")
    git("add", "f.txt")
    git("commit", "-qm", "one")
    repo = tmp_path / "repo"
    with pytest.raises(ValueError, match="--allow-unpushed"):
        aws.release_commit(repo, log=lambda s: None)
    git("init", "-q", "--bare", str(tmp_path / "remote.git"), cwd=tmp_path)
    git("remote", "add", "origin", str(tmp_path / "remote.git"))
    git("push", "-q", "origin", "HEAD:refs/heads/main")
    git("fetch", "-q", "origin")
    sha = git("rev-parse", "HEAD")
    assert aws.release_commit(repo, log=lambda s: None) == sha
    (repo / "f.txt").write_text("2")                   # an edit that would not be sent
    with pytest.raises(ValueError, match="--allow-dirty"):
        aws.release_commit(repo, log=lambda s: None)
    said = []
    assert aws.release_commit(repo, allow_dirty=True, log=said.append) == sha
    assert "WITHOUT them" in said[0]


def test_the_code_version_on_the_instance_is_the_commit_it_was_given(monkeypatch):
    monkeypatch.setenv("MV_CODE_VERSION", "abc1234")
    assert config.code_version() == "abc1234"
    with pytest.raises(ValueError, match="not a commit"):
        aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 1, "bkt",
                                     code_version="x; rm -rf /")


# --- how the instance installs -------------------------------------------------------

def test_the_trainer_installs_only_pinned_hashed_packages_and_checks_the_gpu():
    ud = aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 24, "bkt",
                                      finetune=["bioclip-2"], code_version="abc1234")
    assert ".venv/bin/pip install -q --require-hashes -r requirements/trainer.txt" in ud
    assert '".[' not in ud and "--upgrade pip" not in ud      # nothing unpinned
    assert "torch.cuda.is_available()" in ud
    assert "PYTHONPATH=/opt/mv/src" in ud and "-m mycomap_vision.cli aws-train-job" in ud
    assert "--stop-after-hours 23.25" in ud and "timeout 86400 " in ud
    down = aws.render_user_data("r", "large", 1, "bkt")
    assert ".venv/bin/pip install -q --require-hashes -r requirements/downloader.txt" in down
    assert '".[' not in down


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash here")
def test_the_instance_scripts_are_valid_bash():
    for script in (aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 2, "bkt",
                                                finetune=["bioclip-2"], code_version="abc1234"),
                   aws.render_user_data("r", "large", 1, "bkt")):
        r = subprocess.run(["bash", "-n"], input=script.encode(), capture_output=True)
        assert r.returncode == 0, r.stderr.decode(errors="replace")


def lock(name):
    """{package: [hashes]} of a lockfile."""
    out, cur = {}, None
    for line in (REPO / "requirements" / name).read_text(encoding="utf-8").splitlines():
        s = line.strip().rstrip("\\").strip()
        if not s or s.startswith("#"):
            continue
        if s.startswith("--hash=sha256:"):
            out[cur].append(s.split(":", 1)[1])
        else:
            assert "==" in s, f"unpinned: {s}"
            cur = s.split("==")[0]
            out[cur] = []
    return out


def test_the_instance_lockfiles_pin_everything_with_hashes_for_linux():
    trainer_lock, download_lock = lock("trainer.txt"), lock("downloader.txt")
    for pinned in (trainer_lock, download_lock):
        assert pinned and all(len(h) >= 1 and all(len(x) == 64 for x in h)
                              for h in pinned.values())
    for pkg in ("torch", "torchvision", "timm", "open-clip-torch", "pillow", "numpy",
                "boto3", "requests"):
        assert pkg in trainer_lock, pkg
    # Linux-only: torch's CUDA runtime is there, Windows-only packages are not.
    assert "nvidia-cudnn-cu13" in trainer_lock and "colorama" not in trainer_lock
    assert {"boto3", "requests"} <= set(download_lock) and "torch" not in download_lock


# --- stopping at the time limit, and what is uploaded as it goes -----------------------

def stoppable_run(conn, store, tmp_path, rec, stop_in_b=True, finetune=None, **kw):
    """Backbones a then b; once b has encoded one batch, the time limit is reached."""
    made = {}

    def loader(spec):
        made[models.storage_name(spec)] = Counting(models.storage_name(spec))
        return made[models.storage_name(spec)]

    def should_stop():
        return stop_in_b and "timm_b" in made and made["timm_b"].batches >= 1
    out = trainer.run_job(conn, store, ["timm:a", "timm:b"], ["nearest"], rec, "r1",
                          loader=kw.pop("loader", loader), data_dir=tmp_path / "inst",
                          batch_size=2, should_stop=should_stop, finetune=finetune,
                          finetuner=fake_finetuner, log=lambda s: None, **kw)
    return out


def test_each_finished_stage_is_uploaded_with_its_progress_before_the_next_starts(conn,
                                                                               tmp_path):
    store = seed_two_species(conn, tmp_path)
    rec, seen = Recorder(), {}

    def loader(spec):
        name = models.storage_name(spec)
        if name == "timm_b":           # timm_a is finished: its files must be in the bucket
            seen["keys"] = set(rec.files)
            seen["progress"] = rec.json("runs/r1/progress.json")
        return Const(name)
    stoppable_run(conn, store, tmp_path, rec, stop_in_b=False, loader=loader)
    assert {"runs/r1/embeddings/timm_a/shard-00000.npy", "runs/r1/index.sqlite"} <= seen["keys"]
    stages = {s["name"]: s for s in seen["progress"]["stages"]}
    assert stages["timm_a"]["status"] == "done" and stages["timm_a"]["photos"] == 8
    assert stages["timm_b"]["status"] == "running"
    assert seen["progress"]["state"] == "running"
    # The index goes up before the progress that announces the stage.
    first_done = rec.log.index("runs/r1/embeddings/timm_a/shard-00000.npy")
    assert rec.log.index("runs/r1/index.sqlite") > first_done


def test_the_time_limit_stops_the_run_cleanly_and_progress_says_what_finished(conn, tmp_path,
                                                                             monkeypatch):
    monkeypatch.setenv("MV_CODE_VERSION", "abc1234")
    store = seed_two_species(conn, tmp_path)
    rec = Recorder()
    out = stoppable_run(conn, store, tmp_path, rec)
    assert out["state"] == "stopped" and list(out["embedded"]) == ["timm_a"]
    assert "stopped at 2 of 8" in out["stopped"]["timm_b"]
    assert out["comparison_id"] is None                # left to the laptop
    progress = rec.json("runs/r1/progress.json")
    assert progress["state"] == "stopped" and progress["code_version"] == "abc1234"
    assert [s["status"] for s in progress["stages"]] == ["done", "stopped"]
    assert not any("timm_b" in k for k in rec.files)   # half a backbone is never uploaded
    assert rec.log[-1] == "runs/r1/result.json"
    assert rec.json("runs/r1/result.json")["code_version"] == "abc1234"


def test_a_fine_tune_cut_short_by_the_time_limit_is_not_saved_and_later_stages_stop(conn,
                                                                                    tmp_path):
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    rec, late = Recorder(), []

    def stopped_finetuner(*a):
        late.append(True)              # the limit is reached during the fine-tune
        raise Stopped("stopped at step 3 of 10 (time limit); not saved")
    out = trainer.run_job(conn, store, ["timm:a", "timm:b"], ["nearest"], rec, "r1",
                          loader=lambda s: Const(models.storage_name(s)),
                          data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=stopped_finetuner, should_stop=lambda: bool(late),
                          log=lambda s: None)
    assert out["state"] == "stopped" and list(out["embedded"]) == ["timm_a"]
    statuses = {(s["kind"], s["name"]): s["status"] for s in out["stages"]}
    assert statuses == {("embed", "timm_a"): "done", ("finetune", "timm_a-ft-r1"): "stopped",
                        ("embed", "timm_a-ft-r1"): "stopped", ("embed", "timm_b"): "stopped"}
    assert "not saved" in out["stopped"]["timm_a-ft-r1"]
    assert not any("models/" in k for k in rec.files)


def test_the_deadline_says_stop_once_its_hours_are_up():
    now = [0.0]
    stop = trainer.deadline(2, clock=lambda: now[0])
    assert not stop()
    now[0] = 2 * 3600 - 1
    assert not stop()
    now[0] = 2 * 3600
    assert stop()
    assert trainer.deadline(None) is None


# --- bringing a partial run home -------------------------------------------------------

class FakeS3:
    """download_file and list_objects_v2 over a folder standing in for the bucket."""

    def __init__(self, root: Path):
        self.root = root

    def download_file(self, bucket, key, dest):
        src = self.root / key
        if not src.is_file():
            raise ClientError({"Error": {"Code": "404", "Message": "Not Found"}}, "HeadObject")
        Path(dest).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dest)

    def get_paginator(self, name):
        root = self.root

        class Pages:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [
                    {"Key": p.relative_to(root).as_posix(), "Size": p.stat().st_size}
                    for p in sorted(root.rglob("*"))
                    if p.is_file() and p.relative_to(root).as_posix().startswith(Prefix)]}
        return Pages()


def pull(tmp_path, monkeypatch, bucket_root, home):
    monkeypatch.setattr(aws, "s3_client", lambda: FakeS3(bucket_root))
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setattr(config, "DATA_DIR", home)
    monkeypatch.setattr(config, "REPORTS_DIR", home / "reports")
    conn = connect(home / "manifest.sqlite")
    return conn, aws.pull_trainer(conn, "r1", log=lambda s: None)


def rows(conn, backbone):
    return conn.execute("select count(*) from embeddings where backbone = ?",
                        (backbone,)).fetchone()[0]


def test_a_stopped_run_brings_home_its_finished_backbones_and_leaves_the_rest(tmp_path,
                                                                             monkeypatch):
    monkeypatch.setenv("MV_CODE_VERSION", "abc1234")
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    rec = Recorder()
    stoppable_run(inst, seed_two_species(inst, tmp_path), tmp_path, rec)
    inst.close()
    monkeypatch.delenv("MV_CODE_VERSION")
    bucket_root = rec.write_bucket(tmp_path / "bucket")
    laptop, home = laptop_with_sample(tmp_path, names=("timm_a", "timm_b"))
    laptop.close()
    conn, merged = pull(tmp_path, monkeypatch, bucket_root, home)
    assert merged["replaced"] == ["timm_a"] and merged["complete"] is False
    assert [m["stage"] for m in merged["missing"]] == ["embed timm_b"]
    assert rows(conn, "timm_a") == 8
    assert rows(conn, "timm_b") == 2                           # the laptop's, untouched
    assert not (home / "embeddings-archive" / "timm_b-before-r1").exists()
    pulled = json.loads((home / "aws" / "run-r1" / "pulled.json").read_text())
    assert pulled["code_version"] == "abc1234" and pulled["from"] == "result.json"
    marker = json.loads((home / "embeddings" / "timm_a" / ".run-r1").read_text())
    assert marker["code_version"] == "abc1234"


def test_a_killed_run_without_a_result_is_pulled_from_its_progress(conn, tmp_path, monkeypatch):
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    rec, at_kill = Recorder(), {}

    def loader(spec):
        name = models.storage_name(spec)
        if name == "timm_b":           # the hard kill: the bucket as it is at this moment
            at_kill.update(rec.files)
        return Const(name)
    stoppable_run(inst, seed_two_species(inst, tmp_path), tmp_path, rec, stop_in_b=False,
                  loader=loader)
    inst.close()
    assert "runs/r1/result.json" not in at_kill
    bucket_root = rec.write_bucket(tmp_path / "bucket", at_kill)
    _, home = laptop_with_sample(tmp_path, names=("timm_a", "timm_b"))
    conn2, merged = pull(tmp_path, monkeypatch, bucket_root, home)
    assert merged["replaced"] == ["timm_a"] and merged["state"] == "running"
    assert merged["missing"][0]["status"] == "running"
    assert rows(conn2, "timm_b") == 2
    assert json.loads((home / "aws" / "run-r1" / "pulled.json").read_text())["from"] == \
        "progress.json"


def test_a_backbone_whose_copy_is_incomplete_never_replaces_the_local_one(tmp_path,
                                                                         monkeypatch):
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    rec = Recorder()
    stoppable_run(inst, seed_two_species(inst, tmp_path), tmp_path, rec)
    inst.close()
    bucket_root = rec.write_bucket(tmp_path / "bucket")
    # Say timm_b finished although only part of it exists: a stray shard, no index rows.
    shutil.copytree(tmp_path / "inst" / "embeddings" / "timm_b",
                    bucket_root / "runs" / "r1" / "embeddings" / "timm_b")
    result = json.loads((bucket_root / "runs/r1/result.json").read_text())
    for s in result["stages"]:
        if s["name"] == "timm_b":
            s.update(status="done", photos=8)
    (bucket_root / "runs/r1/result.json").write_text(json.dumps(result))
    # And one of timm_a's shards is lost on the way.
    (bucket_root / "runs/r1/embeddings/timm_a/shard-00000.ids.npy").unlink()
    _, home = laptop_with_sample(tmp_path, names=("timm_a", "timm_b"))
    conn, merged = pull(tmp_path, monkeypatch, bucket_root, home)
    assert merged["replaced"] == []
    assert "missing" in merged["refused"]["timm_a"]
    assert "no rows" in merged["refused"]["timm_b"]
    assert rows(conn, "timm_a") == 2 and rows(conn, "timm_b") == 2
    assert not (home / "embeddings-archive").exists()


def test_a_run_with_nothing_uploaded_yet_is_refused_with_where_to_look(tmp_path, monkeypatch):
    (tmp_path / "bucket").mkdir()
    with pytest.raises(RuntimeError, match="neither result.json nor progress.json"):
        pull(tmp_path, monkeypatch, tmp_path / "bucket", tmp_path / "home")
