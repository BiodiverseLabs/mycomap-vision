"""Cutting a dataset release in-region (aws_dataset.py): refused before any spend when it
could not be a valid release, the freeze trigger never passes through a shell, the instance
cleans up after itself, and its result lands in S3 last, ready for `mv dataset pull`."""

import json
from pathlib import Path

import pytest
from botocore.exceptions import ClientError
from test_trainer_release import cut  # noqa: F401  (the fixture)
from test_trainer_stages import FakeEC2, FakeSSM

from mycomap_vision import aws, aws_dataset, config
from mycomap_vision import dataset_release as dr
from mycomap_vision import dataset_s3

SHA = "0123456789abcdef0123456789abcdef01234567"


class BucketS3:
    """A dict standing in for the bucket: what upload_release and fetch_release use."""

    def __init__(self):
        self.objects, self.meta = {}, {}

    def upload_file(self, path, bucket, key, ExtraArgs=None):
        self.objects[key] = Path(path).read_bytes()
        self.meta[key] = (ExtraArgs or {}).get("Metadata", {})

    def put_object(self, Bucket, Key, Body):
        self.objects[Key] = Body

    def head_object(self, Bucket, Key):
        return {"ContentLength": len(self.objects[Key]), "Metadata": self.meta.get(Key, {})}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey", "Message": "x"}}, "GetObject")
        body = self.objects[Key]
        return {"Body": type("B", (), {"read": lambda self: body})()}

    def download_file(self, bucket, key, dest):
        if key not in self.objects:
            raise ClientError({"Error": {"Code": "404", "Message": "x"}}, "HeadObject")
        Path(dest).write_bytes(self.objects[key])

    def get_paginator(self, name):
        objects = self.objects

        class Pages:
            def paginate(self, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v)} for k, v in sorted(objects.items())
                                    if k.startswith(Prefix)]}
        return Pages()


def test_the_kept_copy_holds_everything_and_a_copy_without_the_private_part_loads(cut, tmp_path):
    rel, root = cut[3], cut[4]
    s3 = BucketS3()
    up = dataset_s3.upload_release(rel.id, root, s3, "bkt", log=lambda s: None)
    prefix = dataset_s3.release_prefix(rel.id)
    assert prefix + dr.PRIVATE_DB in s3.objects and prefix + dr.MANIFEST in s3.objects
    assert dataset_s3.remote_release_hash(rel.id, s3, "bkt") == rel.release_hash
    again = dataset_s3.upload_release(rel.id, root, s3, "bkt", log=lambda s: None)
    assert again["uploaded"] == 0 and again["already_there"] == up["uploaded"]
    dest = tmp_path / "instance" / "research-releases"
    folder = dataset_s3.fetch_release(rel.id, dest, s3, "bkt", include_private=False,
                                      log=lambda s: None)
    assert not (folder / dr.PRIVATE_DB).exists()
    copy = dr.load_release(rel.id, dest, allow_draft=True, without_private=True)
    assert copy.release_hash == rel.release_hash and not copy.has_private
    copy.close()


def test_a_kept_copy_with_other_bytes_is_never_overwritten(cut):
    rel, root = cut[3], cut[4]
    s3 = BucketS3()
    dataset_s3.upload_release(rel.id, root, s3, "bkt", log=lambda s: None)
    key = dataset_s3.release_prefix(rel.id) + dr.RELEASE_DB
    s3.objects[key] = b"tampered"
    with pytest.raises(dr.ReleaseError, match="never edited"):
        dataset_s3.upload_release(rel.id, root, s3, "bkt", log=lambda s: None)


def test_a_corrupted_download_from_the_kept_copy_is_refused(cut, tmp_path):
    rel, root = cut[3], cut[4]
    s3 = BucketS3()
    dataset_s3.upload_release(rel.id, root, s3, "bkt", log=lambda s: None)
    s3.objects[dataset_s3.release_prefix(rel.id) + dr.RELEASE_DB] += b"x"
    with pytest.raises(dr.ReleaseError, match="does not match"):
        dataset_s3.fetch_release(rel.id, tmp_path / "inst", s3, "bkt",
                                 include_private=False, log=lambda s: None)


def test_a_frozen_release_needs_the_freeze_trigger_and_every_derived_hash():
    with pytest.raises(dr.ReleaseError, match="freeze-approved"):
        aws_dataset.make_request("v1", None)
    with pytest.raises(dr.ReleaseError, match="freeze-approved"):
        aws_dataset.make_request("v1", "   ")
    with pytest.raises(dr.ReleaseError, match="every derived hash"):
        aws_dataset.make_request("v1", "Steve, 2026-10-12", derive="none")
    with pytest.raises(dr.ReleaseError, match="waits for the LOO exclusion list"):
        aws_dataset.make_request("v1", "Steve, 2026-10-12")
    req = aws_dataset.make_request("v1", "Steve, 2026-10-12", no_loo_exclusions="Steve, 10-12")
    assert req["derive"] == "all" and req["val_weeks"] == 8
    assert req["loo"] == {"none": "Steve, 10-12"}
    assert aws_dataset.make_request(dr.draft_id(), None, derive="none")["name"].startswith("dry-")
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        aws_dataset.make_request("v1", "Steve", val_cutoff="Aug 10")
    with pytest.raises(dr.ReleaseError):
        aws_dataset.make_request("../v1", "Steve")


def test_the_freeze_trigger_travels_in_a_file_never_through_the_shell():
    ud = aws_dataset.render_user_data("20261012-010203", "bkt", 6, SHA)
    assert "shutdown -h +390" in ud and "trap finish EXIT" in ud
    assert "aws-dataset-build-job --build-id 20261012-010203" in ud
    assert "--request request.json" in ud and "freeze" not in ud.lower()
    assert "requirements/trainer.txt" in ud and "timeout 21600" in ud
    with pytest.raises(ValueError, match="commit sha"):
        aws_dataset.render_user_data("20261012-010203", "bkt", 6, "main; rm -rf /")
    with pytest.raises(ValueError, match="build id"):
        aws_dataset.render_user_data("x/../y", "bkt", 6, SHA)


def test_the_estimate_counts_every_original_on_the_instances_cores():
    h = aws_dataset.estimate_hours(650_000, "c7i.2xlarge")
    assert 2.0 <= h <= 2.5                         # ~1.6 h of deriving + setup and upload
    assert aws_dataset.estimate_hours(650_000, "c7i.4xlarge") < h


def launch_env(monkeypatch, tmp_path, s3, ec2):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setenv("MV_AWS_REGION", "us-east-2")
    monkeypatch.setattr(aws, "s3_client", lambda: s3)
    monkeypatch.setattr(aws, "session", lambda: type("S", (), {
        "client": lambda self, n: {"ec2": ec2, "ssm": FakeSSM()}[n]})())
    monkeypatch.setattr(aws, "release_commit", lambda **k: SHA)
    monkeypatch.setattr(aws, "code_tarball", lambda ref: b"tgz")


def test_a_launch_ships_code_manifest_and_request_and_the_instance_ends_itself(cut, tmp_path,
                                                                              monkeypatch):
    conn = cut[0]
    s3, ec2 = BucketS3(), FakeEC2()
    launch_env(monkeypatch, tmp_path, s3, ec2)
    review = tmp_path / "loo.tsv"
    review.write_text("kind\tkey\treason\tnote\nrecord\tinat:101\tloo\tchecked\n")
    loo = tmp_path / "loo-mislabel-2026-10-09.tsv"
    loo.write_text("kind\tkey\treason\tnote\nrecord\tinat:102\tLOO mislabel\tSteve\n")
    out = aws_dataset.launch(conn, "v1", "Steve, 2026-10-12 (relayed by the coordinator)",
                             review_files=[review], loo_exclusions=loo, root=tmp_path / "rel",
                             log=lambda s: None)
    prefix = aws_dataset.build_prefix(out["build_id"])
    for key in ("code.tar.gz", "manifest.sqlite", "request.json", "inputs/loo.tsv",
                "inputs/" + loo.name):
        assert prefix + key in s3.objects
    req = json.loads(s3.objects[prefix + "request.json"])
    assert req["freeze_approved"].startswith("Steve, 2026-10-12") and req["name"] == "v1"
    assert req["review_files"] == ["loo.tsv"] and req["code_version"] == SHA
    assert req["loo_exclusions"] == loo.name
    assert req["loo"] == {"file": loo.name, "rows": 1, "sha256": dr.sha256_file(loo)}
    args = ec2.launched
    assert args["InstanceType"] == "c7i.2xlarge"
    assert args["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert args["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == 40
    tags = {t["Key"]: t["Value"] for t in args["TagSpecifications"][0]["Tags"]}
    assert tags["Project"] == "mycomap-vision" and tags["Name"].startswith("mv-dataset-build-")
    assert out["then"] == "mv dataset pull v1"


def test_a_release_already_in_s3_or_here_is_refused_before_any_spend(cut, tmp_path,
                                                                      monkeypatch):
    conn, _, _, rel, root = cut
    s3, ec2 = BucketS3(), FakeEC2()
    launch_env(monkeypatch, tmp_path, s3, ec2)
    with pytest.raises(dr.ReleaseError, match="exists here"):
        aws_dataset.launch(conn, rel.id, None, root=root, log=lambda s: None)
    s3.objects[dataset_s3.release_prefix("v1") + dr.MANIFEST] = json.dumps(
        {"release_hash": "a" * 64}).encode()
    with pytest.raises(dr.ReleaseError, match="never reused"):
        aws_dataset.launch(conn, "v1", "Steve", no_loo_exclusions="Steve (test)",
                           root=tmp_path / "rel", log=lambda s: None)
    with pytest.raises(dr.ReleaseError, match="waits for the LOO exclusion list"):
        aws_dataset.launch(conn, "v2", "Steve", root=tmp_path / "rel", log=lambda s: None)
    assert ec2.launched is None and not any(k.startswith("dataset-builds/") for k in s3.objects)


def test_the_instance_builds_verifies_uploads_and_writes_its_result_last(cut, tmp_path):
    _, path, _, _, _ = cut
    s3 = BucketS3()
    name = "dry-20261012-000001"
    request = aws_dataset.make_request(name, None, val_cutoff="2026-08-24", derive="none")
    out = aws_dataset.run_build_job("20261012-000001", request, s3, "bkt", manifest_path=path,
                                    root=tmp_path / "inst", log=lambda s: None)
    assert out["state"] == "finished" and out["verified"]
    keys = list(s3.objects)
    result_key = aws_dataset.build_prefix("20261012-000001") + aws_dataset.RESULT_FILE
    assert keys[-1] == result_key                            # written last
    assert json.loads(s3.objects[result_key])["release_hash"] == out["release_hash"]
    # Home: the full release, private part included, verified.
    home = tmp_path / "home"
    dataset_s3.fetch_release(name, home, s3, "bkt", include_private=True, log=lambda s: None)
    assert (home / name / dr.PRIVATE_DB).is_file()
    assert dr.verify(name, home)["release_hash"] == out["release_hash"]
    with pytest.raises(dr.ReleaseError, match="never overwritten"):
        dataset_s3.fetch_release(name, home, s3, "bkt", include_private=True, log=lambda s: None)


def test_a_failed_build_says_so_in_its_result(cut, tmp_path):
    _, path, _, _, _ = cut
    s3 = BucketS3()
    request = aws_dataset.make_request("dry-20261012-000002", None, derive="none")
    request["recipes"] = ["no-such-recipe"]
    with pytest.raises(Exception):
        aws_dataset.run_build_job("20261012-000002", request, s3, "bkt", manifest_path=path,
                                  root=tmp_path / "inst", log=lambda s: None)
    doc = json.loads(s3.objects[aws_dataset.build_prefix("20261012-000002") + "result.json"])
    assert doc["state"] == "failed" and "no-such-recipe" in doc["error"]


def test_the_instance_refuses_a_loo_list_other_than_the_one_the_launch_checked(cut, tmp_path):
    _, path, _, _, _ = cut
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    loo = inputs / "loo.tsv"
    loo.write_text("kind\tkey\treason\tnote\n")
    request = aws_dataset.make_request("dry-20261012-000003", None, derive="none",
                                       loo_exclusions=loo)
    loo.write_text("kind\tkey\treason\tnote\nrecord\tinat:101\tswapped\tx\n")
    s3 = BucketS3()
    with pytest.raises(dr.ReleaseError, match="not the LOO list the launch checked"):
        aws_dataset.run_build_job("20261012-000003", request, s3, "bkt", manifest_path=path,
                                  root=tmp_path / "inst", inputs=inputs, log=lambda s: None)
    doc = json.loads(s3.objects[aws_dataset.build_prefix("20261012-000003") + "result.json"])
    assert doc["state"] == "failed"


def test_the_build_ships_the_frozen_manifest_the_loo_list_was_written_against(cut, tmp_path,
                                                                             monkeypatch):
    from mycomap_vision.manifest import snapshot
    conn, path, _, _, _ = cut
    s3, ec2 = BucketS3(), FakeEC2()
    launch_env(monkeypatch, tmp_path, s3, ec2)
    frozen = snapshot(conn, tmp_path / "manifest-v1-freeze.sqlite")     # mv manifest-freeze
    out = aws_dataset.launch(conn, "v1", "Steve", no_loo_exclusions="Steve (test)",
                             manifest_path=frozen, root=tmp_path / "rel", log=lambda s: None)
    prefix = aws_dataset.build_prefix(out["build_id"])
    assert s3.objects[prefix + "manifest.sqlite"] == frozen.read_bytes()
    req = json.loads(s3.objects[prefix + "request.json"])
    assert req["manifest_sha256"] == dr.sha256_file(frozen) == out["manifest_sha256"]
    assert req["manifest_file"] == frozen.name


def test_the_instance_refuses_a_manifest_other_than_the_one_shipped(cut, tmp_path):
    _, path, _, _, _ = cut
    request = aws_dataset.make_request("dry-20261012-000004", None, derive="none")
    request["manifest_sha256"] = "0" * 64
    s3 = BucketS3()
    with pytest.raises(dr.ReleaseError, match="not the one the launch shipped"):
        aws_dataset.run_build_job("20261012-000004", request, s3, "bkt", manifest_path=path,
                                  root=tmp_path / "inst", log=lambda s: None)
