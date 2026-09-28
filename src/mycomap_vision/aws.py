"""Run the photo download on a small EC2 instance that writes straight to S3.

The instance fetches the code and the manifest from the bucket, downloads photos
into the bucket, copies the manifest back every 10 minutes, uploads its log, and
shuts itself down (which terminates it). A backstop shutdown is scheduled at boot
so a stuck run cannot keep billing.

Settings (see .env.example and deploy/aws/README.md): MV_S3_BUCKET, MV_AWS_REGION,
MV_AWS_PROFILE and MV_INSTANCE_ROLE.
"""

from __future__ import annotations

import io
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from . import config
from .manifest import snapshot

PROJECT_TAG = {"Key": "Project", "Value": "mycomap-vision"}


def bucket() -> str:
    return config.required("MV_S3_BUCKET", "the private S3 bucket for photos and manifests")


def region() -> str:
    return config.required("MV_AWS_REGION", "the AWS region of the bucket and instances")


def instance_role() -> str:
    return config.setting("MV_INSTANCE_ROLE", "mycomap-vision-instance")
MANIFEST_KEY = "manifest/manifest.sqlite"
AMI_PARAMETER = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"

USER_DATA = """#!/bin/bash
set -uo pipefail
BUCKET={bucket}
RUN=runs/{run_id}
LOG=/var/log/mv-download.log
exec > >(tee -a "$LOG") 2>&1
# Backstop: power off after {backstop_minutes} minutes whatever happens.
shutdown -h +{backstop_minutes}
finish() {{
  aws s3 cp "$LOG" "s3://$BUCKET/$RUN/download.log" || true
  shutdown -h now
}}
trap finish EXIT
( while sleep 900; do aws s3 cp "$LOG" "s3://$BUCKET/$RUN/download.log" >/dev/null 2>&1; done ) &
dnf install -y -q python3.11 python3.11-pip
mkdir -p /opt/mv/data && cd /opt/mv
aws s3 cp "s3://$BUCKET/$RUN/code.tar.gz" code.tar.gz
tar xzf code.tar.gz
python3.11 -m venv .venv
.venv/bin/pip install -q ".[aws]"
aws s3 cp "s3://$BUCKET/{manifest_key}" data/manifest.sqlite
MV_DATA_DIR=/opt/mv/data PYTHONUNBUFFERED=1 .venv/bin/mv download-photos \\
  --size {size} --dest "s3://$BUCKET" --checkpoint-to "s3://$BUCKET/{manifest_key}" \\
  --max-hours {max_hours}
"""


def render_user_data(run_id: str, size: str, max_hours: float, bucket_name: str) -> str:
    # The backstop leaves an hour for setup and the final manifest copy.
    backstop = int(max_hours * 60) + 60
    return USER_DATA.format(bucket=bucket_name, run_id=run_id, size=size, max_hours=max_hours,
                            backstop_minutes=backstop, manifest_key=MANIFEST_KEY)


def session():
    import boto3
    return boto3.Session(profile_name=config.setting("MV_AWS_PROFILE"), region_name=region())


def ensure_bucket(s3, bucket: str) -> bool:
    """Create the bucket private and encrypted if it does not exist. Returns True if created."""
    try:
        s3.head_bucket(Bucket=bucket)
        return False
    except s3.exceptions.ClientError as e:
        if e.response.get("Error", {}).get("Code") not in ("404", "NoSuchBucket"):
            raise
    s3.create_bucket(Bucket=bucket,
                     CreateBucketConfiguration={"LocationConstraint": region()})
    s3.put_public_access_block(Bucket=bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True,
        "BlockPublicPolicy": True, "RestrictPublicBuckets": True})
    s3.put_bucket_encryption(Bucket=bucket, ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}]})
    s3.put_bucket_tagging(Bucket=bucket, Tagging={"TagSet": [PROJECT_TAG]})
    return True


def code_tarball() -> bytes:
    """The committed code (HEAD), never the data folder or uncommitted edits."""
    out = subprocess.run(["git", "archive", "--format=tar.gz", "HEAD"], cwd=config.REPO_ROOT,
                         capture_output=True, check=True)
    # Sanity check: it must be a readable tar with pyproject.toml at the top.
    with tarfile.open(fileobj=io.BytesIO(out.stdout), mode="r:gz") as t:
        if "pyproject.toml" not in t.getnames():
            raise RuntimeError("git archive produced no pyproject.toml")
    return out.stdout


def run_instance_args(run_id: str, ami: str, user_data: str,
                      instance_type: str = "t3.small") -> dict:
    tags = [PROJECT_TAG, {"Key": "Name", "Value": f"mv-downloader-{run_id}"}]
    return {
        "ImageId": ami,
        "InstanceType": instance_type,
        "MinCount": 1, "MaxCount": 1,
        "IamInstanceProfile": {"Name": instance_role()},
        "InstanceInitiatedShutdownBehavior": "terminate",
        "UserData": user_data,
        "MetadataOptions": {"HttpTokens": "required"},
        "BlockDeviceMappings": [{"DeviceName": "/dev/xvda",
                                 "Ebs": {"VolumeSize": 16, "VolumeType": "gp3",
                                         "DeleteOnTermination": True}}],
        "TagSpecifications": [{"ResourceType": "instance", "Tags": tags},
                              {"ResourceType": "volume", "Tags": tags}],
    }


def launch_downloader(conn, size: str = "large", max_hours: float = 120,
                      instance_type: str = "t3.small", log=print) -> dict:
    sess = session()
    s3, ec2, ssm = sess.client("s3"), sess.client("ec2"), sess.client("ssm")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    b = bucket()
    if ensure_bucket(s3, b):
        log(f"Created private bucket s3://{b} in {region()}")
    s3.put_object(Bucket=b, Key=f"runs/{run_id}/code.tar.gz", Body=code_tarball())
    snap = snapshot(conn, config.DATA_DIR / "manifest-upload.sqlite")
    s3.upload_file(str(snap), b, MANIFEST_KEY)
    log(f"Uploaded code and manifest for run {run_id}")
    ami = ssm.get_parameter(Name=AMI_PARAMETER)["Parameter"]["Value"]
    resp = ec2.run_instances(**run_instance_args(
        run_id, ami, render_user_data(run_id, size, max_hours, b), instance_type))
    instance_id = resp["Instances"][0]["InstanceId"]
    return {"run_id": run_id, "instance_id": instance_id, "region": region(),
            "log": f"s3://{b}/runs/{run_id}/download.log",
            "manifest": f"s3://{b}/{MANIFEST_KEY}"}


def pull_manifest(dest: Path) -> Path:
    """Copy the instance's manifest back. Replaces the local one only after a full download."""
    wal = dest.with_name(dest.name + "-wal")
    if wal.exists():
        raise RuntimeError(f"{wal} exists: the manifest is still open (a fetch or download "
                           "running here?). Stop it first.")
    s3 = session().client("s3")
    tmp = dest.with_suffix(".download")
    s3.download_file(bucket(), MANIFEST_KEY, str(tmp))
    tmp.replace(dest)
    return dest


def s3_checkpoint(conn, url: str):
    """A callable that snapshots the manifest and uploads it to `url` (s3://bucket/key)."""
    import boto3
    from .storage import parse_s3_url
    bucket, key = parse_s3_url(url)
    key = key.rstrip("/")
    client = boto3.client("s3")
    snap_path = config.DATA_DIR / "manifest-checkpoint.sqlite"

    def checkpoint() -> None:
        snapshot(conn, snap_path)
        client.upload_file(str(snap_path), bucket, key)
    return checkpoint


POLICY_DIR = config.REPO_ROOT / "deploy" / "aws"


def render_policy(template: str) -> str:
    """An IAM policy template with this deployment's bucket, region and role filled in."""
    text = (POLICY_DIR / template).read_text(encoding="utf-8")
    return (text.replace("{{BUCKET}}", bucket()).replace("{{REGION}}", region())
                .replace("{{INSTANCE_ROLE}}", instance_role()))
