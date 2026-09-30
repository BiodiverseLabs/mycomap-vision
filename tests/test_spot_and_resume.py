"""The trainer on Spot: the launch asks for a one-time Spot instance, an interruption
notice ends the run "interrupted" with nothing half uploaded, the pull takes what
finished, and a resumed run does only what is left. EC2, S3 and the instance
metadata service are all stand-ins."""

import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from test_models_and_scoreboard import Const, seed_two_species
from test_trainer import laptop_with_sample
from test_trainer_stages import (Counting, FakeEC2, FakeLaunchS3, FakeS3, FakeSSM, Recorder,
                                 fake_finetuner, pull, rows, seed_s3_photos)

from mycomap_vision import aws, config, models, spot, trainer
from mycomap_vision.manifest import connect


# --- the launch -----------------------------------------------------------------------

def args_for(**kw):
    return aws.trainer_instance_args("r1", "ami-1", "#!/bin/bash", "g6.2xlarge", "/dev/xvda",
                                     75, **kw)


def test_a_spot_launch_asks_for_a_one_time_spot_instance_that_terminates_when_taken_back():
    a = args_for(spot=True)
    assert a["InstanceMarketOptions"] == {
        "MarketType": "spot",
        "SpotOptions": {"SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate"}}   # no MaxPrice: On-Demand cap
    assert a["InstanceInitiatedShutdownBehavior"] == "terminate"
    specs = {s["ResourceType"]: s["Tags"] for s in a["TagSpecifications"]}
    assert aws.PROJECT_TAG in specs["spot-instances-request"]           # the ops policy needs it
    capped = args_for(spot=True, spot_max_price=0.45)
    assert capped["InstanceMarketOptions"]["SpotOptions"]["MaxPrice"] == "0.4500"


def test_without_spot_the_launch_is_on_demand_with_no_market_options():
    a = args_for()
    assert "InstanceMarketOptions" not in a
    assert "spot-instances-request" not in {s["ResourceType"] for s in a["TagSpecifications"]}
    with pytest.raises(ValueError, match="needs --spot"):
        args_for(spot_max_price=0.5)
    with pytest.raises(ValueError, match="above 0"):
        args_for(spot=True, spot_max_price=0)


def launch_env(conn, tmp_path, monkeypatch, ec2=None, s3=None):
    seed_two_species(conn, tmp_path)
    seed_s3_photos(conn)
    ec2, s3 = ec2 or FakeEC2(), s3 or FakeLaunchS3()
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setenv("MV_AWS_REGION", "us-east-2")
    monkeypatch.setattr(aws, "s3_client", lambda: s3)
    monkeypatch.setattr(aws, "session", lambda: type("S", (), {
        "client": lambda self, n: {"ec2": ec2, "ssm": FakeSSM()}[n]})())
    monkeypatch.setattr(aws, "release_commit", lambda **k: "a" * 40)
    monkeypatch.setattr(aws, "code_tarball", lambda ref: b"tgz")
    return ec2, s3


def test_the_launcher_passes_spot_to_ec2_and_the_job_watches_for_the_notice(conn, tmp_path,
                                                                            monkeypatch):
    ec2, _ = launch_env(conn, tmp_path, monkeypatch)
    said = []
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             spot=True, log=said.append)
    assert ec2.launched["InstanceMarketOptions"]["MarketType"] == "spot"
    assert " --spot" in ec2.launched["UserData"] and out["market"] == "spot"
    assert any("2 minutes' notice" in s and f"--resume {out['run_id']}" in s for s in said)
    ec2.launched = None
    aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                       log=lambda s: None)
    assert "InstanceMarketOptions" not in ec2.launched
    assert "--spot" not in ec2.launched["UserData"]


class RefusingEC2(FakeEC2):
    def __init__(self, code):
        super().__init__()
        self.code = code

    def run_instances(self, **kw):
        raise ClientError({"Error": {"Code": self.code, "Message": "no"}}, "RunInstances")


@pytest.mark.parametrize("code, spot, words", [
    ("InsufficientInstanceCapacity", True, "no g6.2xlarge free right now"),
    ("MaxSpotInstanceCountExceeded", True, "All G and VT Spot Instance Requests"),
    ("SpotMaxPriceTooLow", True, "above --spot-max-price"),
    ("VcpuLimitExceeded", False, "Running On-Demand G and VT instances"),
])
def test_capacity_quota_and_price_refusals_say_what_happened_and_what_to_do(
        conn, tmp_path, monkeypatch, code, spot, words):
    launch_env(conn, tmp_path, monkeypatch, ec2=RefusingEC2(code))
    with pytest.raises(aws.LaunchRefused, match=code) as e:
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"], spot=spot,
                           log=lambda s: None)
    assert words in str(e.value) and "Nothing is running" in str(e.value)


def test_other_launch_errors_are_not_disguised_as_capacity(conn, tmp_path, monkeypatch):
    launch_env(conn, tmp_path, monkeypatch, ec2=RefusingEC2("UnauthorizedOperation"))
    with pytest.raises(ClientError):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], spot=True, finetune=[],
                           log=lambda s: None)


def test_the_ops_policy_lets_the_spot_launch_through_and_nothing_more(monkeypatch):
    monkeypatch.setenv("MV_S3_BUCKET", "bkt")
    monkeypatch.setenv("MV_AWS_REGION", "us-east-2")
    policy = json.loads(aws.render_policy("ops-policy.template.json"))
    by_sid = {s["Sid"]: s for s in policy["Statement"]}
    tagged = by_sid["LaunchOnlyTaggedInstances"]
    assert "arn:aws:ec2:us-east-2:*:spot-instances-request/*" in tagged["Resource"]
    assert tagged["Condition"] == {"StringEquals": {"aws:RequestTag/Project": "mycomap-vision"}}
    slr = by_sid["SpotServiceLinkedRoleOnce"]
    assert slr["Action"] == "iam:CreateServiceLinkedRole"
    assert slr["Resource"].endswith("/spot.amazonaws.com/AWSServiceRoleForEC2Spot")
    assert slr["Condition"] == {"StringEquals": {"iam:AWSServiceName": "spot.amazonaws.com"}}


@pytest.mark.skipif(shutil.which("bash") is None, reason="no bash here")
def test_the_spot_and_resume_instance_script_is_valid_bash():
    ud = aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 2, "bkt",
                                      finetune=["bioclip-2"], code_version="abc1234", spot=True,
                                      resume=True, code_key="code-resume-20261001-000000.tar.gz")
    assert "--spot --resume" in ud and '$RUN/code-resume-20261001-000000.tar.gz"' in ud
    r = subprocess.run(["bash", "-n"], input=ud.encode(), capture_output=True)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    with pytest.raises(ValueError, match="archive"):
        aws.render_trainer_user_data("r", ["b"], ["nearest"], 2, "bkt", code_key="x; rm -rf /")


# --- the notice -------------------------------------------------------------------------

class Answer:
    def __init__(self, status, body=None):
        self.status_code, self._body = status, body
        self.text = body if isinstance(body, str) else json.dumps(body)

    def json(self):
        return self._body


class FakeIMDS:
    """PUT /latest/api/token and GET .../spot/instance-action, as the metadata service."""

    def __init__(self, answers):
        self.answers, self.calls = list(answers), []

    def put(self, url, headers=None, timeout=None):
        self.calls.append(("PUT", url, dict(headers or {})))
        return Answer(200, "tok")

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, dict(headers or {})))
        a = self.answers.pop(0) if self.answers else Answer(404, "")
        if isinstance(a, Exception):
            raise a
        return a


def test_the_watcher_says_stop_only_once_aws_gives_notice():
    imds = FakeIMDS([Answer(404, "Not Found"), OSError("timed out"), Answer(500, "x"),
                     Answer(200, {"unexpected": 1}),
                     Answer(200, {"action": "terminate", "time": "2026-10-01T10:00:00Z"})])
    w = spot.SpotWatcher(session=imds, log=lambda s: None)
    for _ in range(4):          # no notice, a timeout, an error, an answer that is no notice
        assert w.check() is None and not w()
    assert w.check()["action"] == "terminate" and w()
    puts = [c for c in imds.calls if c[0] == "PUT"]
    assert len(puts) == 1 and puts[0][1] == spot.IMDS + spot.TOKEN_PATH       # IMDSv2 token
    gets = [c for c in imds.calls if c[0] == "GET"]
    assert all(c[1] == spot.IMDS + spot.ACTION_PATH and c[2]["X-aws-ec2-metadata-token"] == "tok"
               for c in gets)


def test_the_watcher_gets_a_new_token_when_the_old_one_is_refused():
    imds = FakeIMDS([Answer(401, ""), Answer(404, "")])
    w = spot.SpotWatcher(session=imds, log=lambda s: None)
    w.check()
    w.check()
    assert len([c for c in imds.calls if c[0] == "PUT"]) == 2


def test_the_watcher_polls_in_the_background():
    imds = FakeIMDS([Answer(404, ""), Answer(200, {"action": "terminate", "time": "t"})])
    w = spot.SpotWatcher(session=imds, interval=0.01, log=lambda s: None).start()
    w._thread.join(timeout=5)
    assert w() and not w._thread.is_alive()


# --- an interrupted run -----------------------------------------------------------------

def interrupted_run(conn, store, tmp_path, rec, during="timm_b", finetune=None,
                    data="inst", loaded=None):
    """Backbones a then b (a fine-tuned when asked); the Spot notice comes once the stage
    `during` has encoded one batch."""
    made = {}

    def loader(spec):
        name = models.storage_name(spec)
        made[name] = Counting(name)
        if loaded is not None:
            loaded.append(name)
        return made[name]

    def notice():
        return during in made and made[during].batches >= 1
    return trainer.run_job(conn, store, ["timm:a", "timm:b"], ["nearest"], rec, "r1",
                           loader=loader, data_dir=tmp_path / data, batch_size=2,
                           interrupted=notice, finetune=finetune, finetuner=fake_finetuner,
                           log=lambda s: None)


def test_a_spot_notice_stops_the_stage_and_the_run_ends_interrupted_with_nothing_half_uploaded(
        conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    rec = Recorder()
    out = interrupted_run(conn, store, tmp_path, rec)
    assert out["state"] == "interrupted" and list(out["embedded"]) == ["timm_a"]
    assert "Spot interruption" in out["stopped"]["timm_b"]
    progress = rec.json("runs/r1/progress.json")
    assert progress["state"] == "interrupted"                  # not "stopped": AWS took it
    assert [s["status"] for s in progress["stages"]] == ["done", "stopped"]
    assert "runs/r1/embeddings/timm_a/shard-00000.npy" in rec.files
    assert not any("timm_b" in k for k in rec.files)            # half a backbone never goes up
    assert "runs/r1/manifest-out.sqlite" not in rec.files       # no time for the big copy
    assert out["comparison_id"] is None
    assert rec.log[-2:] == ["runs/r1/progress.json", "runs/r1/result.json"]


def test_the_time_limit_still_ends_stopped_not_interrupted(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    rec = Recorder()
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"], rec, "r1",
                          loader=lambda s: Const(models.storage_name(s)),
                          data_dir=tmp_path / "inst", should_stop=lambda: True,
                          interrupted=lambda: False, log=lambda s: None)
    assert out["state"] == "stopped" and "time limit" in out["stopped"]["timm_a"]
    assert "runs/r1/manifest-out.sqlite" in rec.files


def test_the_pull_takes_an_interrupted_runs_finished_stages_and_says_how_to_resume(tmp_path,
                                                                                   monkeypatch):
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    rec = Recorder()
    interrupted_run(inst, seed_two_species(inst, tmp_path), tmp_path, rec)
    inst.close()
    bucket_root = rec.write_bucket(tmp_path / "bucket")
    _, home = laptop_with_sample(tmp_path, names=("timm_a", "timm_b"))
    said = []
    conn, merged = pull(tmp_path, monkeypatch, bucket_root, home, said)
    assert merged["state"] == "interrupted" and merged["complete"] is False
    assert merged["replaced"] == ["timm_a"]
    assert [m["stage"] for m in merged["missing"]] == ["embed timm_b"]
    assert rows(conn, "timm_a") == 8 and rows(conn, "timm_b") == 2   # b: the laptop's own
    assert any("mv aws-launch-trainer --resume r1" in s for s in said)


# --- resuming ---------------------------------------------------------------------------

def test_a_resumed_run_restores_its_finished_stages_and_runs_only_the_rest(tmp_path,
                                                                          monkeypatch):
    monkeypatch.setenv("MV_CODE_VERSION", "abc1234")
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    rec = Recorder()
    first = interrupted_run(inst, seed_two_species(inst, tmp_path), tmp_path, rec,
                            during="timm_a-ft-r1", finetune=["timm:a"])
    inst.close()
    assert first["state"] == "interrupted" and set(first["embedded"]) == {"timm_a"}
    assert set(first["finetuned"]) == {"timm_a-ft-r1"}
    # A new instance: a fresh copy of the run's manifest, nothing on its disk.
    monkeypatch.setenv("MV_CODE_VERSION", "def5678")
    again = connect(tmp_path / "inst2-manifest" / "manifest.sqlite")
    store = seed_two_species(again, tmp_path)
    files = trainer.RunFiles(FakeS3(rec.write_bucket(tmp_path / "bucket")), "bkt")
    loaded, tuned = [], []

    def finetuner(*a):
        tuned.append(a[2])
        return fake_finetuner(*a)
    out = trainer.run_job(again, store, ["timm:a", "timm:b"], ["nearest"], rec, "r1",
                          loader=lambda s: loaded.append(models.storage_name(s))
                          or Const(models.storage_name(s)),
                          data_dir=tmp_path / "inst2", batch_size=2, finetune=["timm:a"],
                          finetuner=finetuner, resume=files, log=lambda s: None)
    assert loaded == ["timm_a-ft-r1", "timm_b"]         # timm_a is not embedded again
    assert tuned == []                                  # and the fine-tune is not redone
    assert out["state"] == "finished" and out["comparison_id"]
    assert set(out["embedded"]) == {"timm_a", "timm_a-ft-r1", "timm_b"}
    assert rows(again, "timm_a") == 8                   # restored rows, for the comparison
    progress = rec.json("runs/r1/progress.json")
    assert progress["state"] == "finished" and progress["code_version"] == "def5678"
    restored = [s["name"] for s in progress["stages"] if s.get("restored")]
    assert restored == ["timm_a", "timm_a-ft-r1"]
    assert progress["attempts"] == [{"started_at": progress["attempts"][0]["started_at"],
                                     "updated_at": progress["attempts"][0]["updated_at"],
                                     "state": "interrupted", "code_version": "abc1234"}]
    # The whole run now comes home as complete.
    again.close()
    _, home = laptop_with_sample(tmp_path, names=("timm_a", "timm_b"))
    conn, merged = pull(tmp_path, monkeypatch, rec.write_bucket(tmp_path / "bucket2"), home)
    assert merged["complete"] is True
    assert sorted(merged["replaced"]) == ["timm_a", "timm_a-ft-r1", "timm_b"]


def test_a_finished_stage_whose_files_are_incomplete_is_run_again_not_trusted(tmp_path):
    inst = connect(tmp_path / "inst-manifest" / "manifest.sqlite")
    rec = Recorder()
    interrupted_run(inst, seed_two_species(inst, tmp_path), tmp_path, rec)
    inst.close()
    bucket_root = rec.write_bucket(tmp_path / "bucket")
    (bucket_root / "runs/r1/embeddings/timm_a/shard-00000.ids.npy").unlink()
    again = connect(tmp_path / "inst2-manifest" / "manifest.sqlite")
    loaded = []
    out = trainer.run_job(again, seed_two_species(again, tmp_path), ["timm:a", "timm:b"],
                          ["nearest"], rec, "r1",
                          loader=lambda s: loaded.append(models.storage_name(s))
                          or Const(models.storage_name(s)),
                          data_dir=tmp_path / "inst2", resume=trainer.RunFiles(
                              FakeS3(bucket_root), "bkt"), log=lambda s: None)
    assert loaded == ["timm_a", "timm_b"] and out["state"] == "finished"
    assert not any(s.get("restored") for s in rec.json("runs/r1/progress.json")["stages"])


class ResumeS3(FakeLaunchS3):
    def __init__(self, docs):
        super().__init__()
        self.docs = docs

    def get_object(self, Bucket, Key):
        if Key not in self.docs:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")
        import io
        return {"Body": io.BytesIO(json.dumps(self.docs[Key]).encode())}

    def head_object(self, Bucket, Key):
        if Key != "runs/r1/manifest-in.sqlite":
            raise ClientError({"Error": {"Code": "404", "Message": "x"}}, "HeadObject")
        return {}


class LiveEC2(FakeEC2):
    def __init__(self, live=()):
        super().__init__()
        self.live, self.filters = list(live), None

    def describe_instances(self, Filters):
        self.filters = Filters
        return {"Reservations": [{"Instances": [{"InstanceId": i} for i in self.live]}]}


def stopped_progress(state="interrupted"):
    stages = [{**asdict(s), "status": st} for s, st in zip(
        trainer.plan_stages(["bioclip-2", "dinov3-l16-512"], ["bioclip-2"], "r1"),
        ["done", "done", "stopped", "stopped"])]
    return {"run_id": "r1", "state": state, "code_version": "b" * 40, "size": "large",
            "methods": ["nearest", "linear"], "test_days": 21, "sample_records": None,
            "stages": stages, "updated_at": "t"}


def test_resuming_relaunches_the_same_run_with_its_own_settings_and_only_what_is_left(
        conn, tmp_path, monkeypatch):
    s3 = ResumeS3({"runs/r1/progress.json": stopped_progress()})
    ec2, _ = launch_env(conn, tmp_path, monkeypatch, ec2=LiveEC2(), s3=s3)
    said = []
    out = aws.launch_trainer(conn, ["timm:ignored"], ["nearest"], finetune=[], resume="r1",
                             spot=True, log=said.append)
    assert out["run_id"] == "r1" and out["resumed"] is True
    assert out["backbones"] == ["bioclip-2", "dinov3-l16-512"] and out["finetune"] == ["bioclip-2"]
    ud = ec2.launched["UserData"]
    assert "--run-id r1 " in ud and "--resume" in ud and "--test-days 21 " in ud
    assert "--methods nearest,linear " in ud
    [code_key] = [k for k in s3.objects if k.startswith("runs/r1/code-resume-")]
    assert f'$RUN/{code_key.split("/")[-1]}"' in ud
    assert "runs/r1/manifest-in.sqlite" not in s3.objects       # the run keeps its manifest
    estimate = next(s for s in said if s.startswith("Estimate"))
    assert "embed bioclip-2-ft-r1" in estimate and "embed dinov3-l16-512" in estimate
    assert "finetune" not in estimate and "embed bioclip-2 " not in estimate   # done: not counted
    assert {"Name": "tag:Name", "Values": ["mv-trainer-r1"]} in ec2.filters


def test_a_run_is_not_resumed_while_its_instance_lives_or_when_it_is_complete(conn, tmp_path,
                                                                             monkeypatch):
    s3 = ResumeS3({"runs/r1/progress.json": stopped_progress()})
    ec2, _ = launch_env(conn, tmp_path, monkeypatch, ec2=LiveEC2(["i-still"]), s3=s3)
    with pytest.raises(ValueError, match="still has a live instance"):
        aws.launch_trainer(conn, [], [], resume="r1", log=lambda s: None)
    assert ec2.launched is None and not s3.objects
    done = stopped_progress("finished")
    for s in done["stages"]:
        s["status"] = "done"
    s3.docs["runs/r1/progress.json"] = done
    ec2.live = []
    with pytest.raises(ValueError, match="complete"):
        aws.launch_trainer(conn, [], [], resume="r1", log=lambda s: None)
    del s3.docs["runs/r1/progress.json"]
    with pytest.raises(ValueError, match="nothing to resume"):
        aws.launch_trainer(conn, [], [], resume="r1", log=lambda s: None)
    assert ec2.launched is None
