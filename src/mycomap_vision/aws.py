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
import json
import re
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
export AWS_DEFAULT_REGION={region}
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
# Pinned, hash-checked packages; the project itself runs from src/, not installed.
.venv/bin/pip install -q --require-hashes -r requirements/downloader.txt || exit 1
aws s3 cp "s3://$BUCKET/{manifest_key}" data/manifest.sqlite
MV_DATA_DIR=/opt/mv/data PYTHONPATH=/opt/mv/src PYTHONUNBUFFERED=1 \\
  .venv/bin/python -m mycomap_vision.cli download-photos \\
  --size {size} --dest "s3://$BUCKET" --checkpoint-to "s3://$BUCKET/{manifest_key}" \\
  --max-hours {max_hours} --static-day-gb {static_day_gb}
"""


def render_user_data(run_id: str, size: str, max_hours: float, bucket_name: str,
                     static_day_gb: float = 20) -> str:
    # The backstop leaves an hour for setup and the final manifest copy.
    backstop = int(max_hours * 60) + 60
    return USER_DATA.format(bucket=bucket_name, run_id=run_id, size=size, max_hours=max_hours,
                            static_day_gb=static_day_gb,
                            region=config.setting("MV_AWS_REGION", "us-east-2"),
                            backstop_minutes=backstop, manifest_key=MANIFEST_KEY)


def session():
    """This machine's AWS session: the MV_AWS_PROFILE profile when set (a laptop), else
    the default chain (on the instance, its role)."""
    import boto3
    return boto3.Session(profile_name=config.setting("MV_AWS_PROFILE"),
                         region_name=config.setting("MV_AWS_REGION"))


def s3_client():
    """One S3 client, shared across download threads, with adaptive retries."""
    from botocore.config import Config
    return session().client("s3", config=Config(retries={"mode": "adaptive",
                                                         "total_max_attempts": 6},
                                                max_pool_connections=32))


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


def code_tarball(ref: str = "HEAD") -> bytes:
    """One commit's code (HEAD by default), never the data folder or uncommitted edits."""
    out = subprocess.run(["git", "archive", "--format=tar.gz", ref], cwd=config.REPO_ROOT,
                         capture_output=True, check=True)
    # Sanity check: it must be a readable tar with pyproject.toml at the top.
    with tarfile.open(fileobj=io.BytesIO(out.stdout), mode="r:gz") as t:
        if "pyproject.toml" not in t.getnames():
            raise RuntimeError("git archive produced no pyproject.toml")
    return out.stdout


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True).stdout.strip()


def release_commit(repo: Path | None = None, allow_dirty: bool = False,
                   allow_unpushed: bool = False, log=print) -> str:
    """The full sha of the commit an instance will run (HEAD, shipped as `git archive`).

    Refused, unless allowed: uncommitted changes to tracked files (they are not sent, so
    the run would not be the code in front of you), and a commit no remote branch holds
    (the run's code_version would name a commit nobody else can look up)."""
    repo = repo or config.REPO_ROOT
    sha = _git(repo, "rev-parse", "HEAD")
    if _git(repo, "status", "--porcelain", "--untracked-files=no"):
        msg = (f"uncommitted changes in {repo}: the instance runs commit {sha[:10]} "
               "WITHOUT them. Commit (and push) first")
        if not allow_dirty:
            raise ValueError(msg + ", or pass --allow-dirty.")
        log(f"WARNING: {msg}.")
    if not _git(repo, "branch", "-r", "--contains", sha):
        msg = f"commit {sha[:10]} is on no remote branch (git push it)"
        if not allow_unpushed:
            raise ValueError(msg + ", or pass --allow-unpushed.")
        log(f"WARNING: {msg}; the run will name a commit only this machine has.")
    return sha


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
                      instance_type: str = "t3.small", static_day_gb: float = 20,
                      log=print) -> dict:
    sess = session()
    s3, ec2, ssm = s3_client(), sess.client("ec2"), sess.client("ssm")
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
        run_id, ami, render_user_data(run_id, size, max_hours, b, static_day_gb), instance_type))
    instance_id = resp["Instances"][0]["InstanceId"]
    return {"run_id": run_id, "instance_id": instance_id, "region": region(),
            "log": f"s3://{b}/runs/{run_id}/download.log",
            "manifest": f"s3://{b}/{MANIFEST_KEY}"}


def merge_manifest(conn, remote_path: Path) -> dict:
    """Bring an instance's results into this manifest without replacing it: its S3
    copies are added, and photos it found gone from iNat are marked missing. Anything
    done here meanwhile (local copies, embeddings, scoreboard) is left as it is."""
    conn.execute("attach database ? as remote", (str(remote_path),))
    try:
        with conn:
            before = conn.execute("select count(*) from photo_copies").fetchone()[0]
            conn.execute(
                "insert or ignore into photo_copies select * from remote.photo_copies "
                "where store like 's3://%'")
            added = conn.execute("select count(*) from photo_copies").fetchone()[0] - before
            missing = conn.execute(
                "update photos set status = 'missing', error = (select r.error from "
                "remote.photos r where r.photo_id = photos.photo_id) "
                "where status != 'done' and photo_id in "
                "(select photo_id from remote.photos where status = 'missing')").rowcount
    finally:
        conn.execute("detach database remote")
    return {"s3_copies_added": added, "marked_missing": missing}


def pull_manifest(conn, work_dir: Path) -> dict:
    """Download the instance's manifest and merge it in (see merge_manifest)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    local = work_dir / "manifest-from-s3.sqlite"
    s3_client().download_file(bucket(), MANIFEST_KEY, str(local))
    return merge_manifest(conn, local)


def backup(conn, log=print) -> dict:
    """Copy what can't be re-derived cheaply to s3://<bucket>/backup/<date>/: a manifest
    snapshot, the embeddings and the reports."""
    from datetime import date
    s3, b = s3_client(), bucket()
    prefix = f"backup/{date.today().isoformat()}/"
    snap = snapshot(conn, config.DATA_DIR / "manifest-backup.sqlite")
    files = [(snap, prefix + "manifest.sqlite")]
    for folder in ("embeddings", "reports"):
        root = config.DATA_DIR / folder
        if root.is_dir():
            files += [(p, prefix + p.relative_to(config.DATA_DIR).as_posix())
                      for p in root.rglob("*") if p.is_file()]
    total = 0
    for path, key in files:
        s3.upload_file(str(path), b, key)
        total += path.stat().st_size
    log(f"backed up {len(files)} files, {total / 2**20:.0f} MB, to s3://{b}/{prefix}")
    return {"files": len(files), "bytes": total, "prefix": f"s3://{b}/{prefix}"}


def s3_checkpoint(conn, url: str):
    """A callable that snapshots the manifest and uploads it to `url` (s3://bucket/key)."""
    from .storage import parse_s3_url
    bucket, key = parse_s3_url(url)
    key = key.rstrip("/")
    client = s3_client()
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


# ---------------------------------------------------------------------------
# The GPU trainer: embeds the large photos in S3, compares, uploads, shuts down.

TRAINER_AMI_PARAMETER = ("/aws/service/deeplearning/ami/x86_64/"
                         "base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id")
TRAINER_INSTANCE_TYPE = "g6.2xlarge"     # 1 NVIDIA L4 (24 GB), 8 vCPUs for JPEG decoding

TRAINER_USER_DATA = """#!/bin/bash
set -uo pipefail
BUCKET={bucket}
RUN=runs/{run_id}
LOG=/var/log/mv-train.log
exec > >(tee -a "$LOG") 2>&1
export AWS_DEFAULT_REGION={region}
# Backstop: power off after {backstop_minutes} minutes whatever happens.
shutdown -h +{backstop_minutes}
finish() {{
  aws s3 cp "$LOG" "s3://$BUCKET/$RUN/train.log" || true
  shutdown -h now
}}
trap finish EXIT
( while sleep 900; do aws s3 cp "$LOG" "s3://$BUCKET/$RUN/train.log" >/dev/null 2>&1; done ) &
nvidia-smi || {{ echo "no GPU visible"; exit 1; }}
dnf install -y -q python3.11 python3.11-pip python3.11-devel
mkdir -p /opt/mv/data && cd /opt/mv
aws s3 cp "s3://$BUCKET/$RUN/code.tar.gz" code.tar.gz
tar xzf code.tar.gz
python3.11 -m venv .venv
# Pinned, hash-checked packages (requirements/trainer.txt: torch from PyPI with its CUDA
# runtime; the Base AMI has the driver and no torch). The project runs from src/.
.venv/bin/pip install -q --require-hashes -r requirements/trainer.txt || exit 1
.venv/bin/python -c "import sys, torch; ok = torch.cuda.is_available(); \\
print('torch', torch.__version__, 'CUDA', torch.version.cuda, 'GPU', ok and torch.cuda.get_device_name(0)); \\
sys.exit(0 if ok else 1)" || {{ echo "torch cannot use the GPU (driver older than its CUDA?)"; exit 1; }}
# The run's own copy of the manifest; the downloader's shared one is never touched.
aws s3 cp "s3://$BUCKET/$RUN/manifest-in.sqlite" data/manifest.sqlite
export HF_HOME=/opt/mv/hf
# The commit this code was archived from (there is no .git here to ask).
export MV_CODE_VERSION={code_version}
# The job stops itself after {stop_hours} h, between batches, and uploads what it
# finished; `timeout` is only the hard backstop at the time limit.
MV_DATA_DIR=/opt/mv/data PYTHONPATH=/opt/mv/src PYTHONUNBUFFERED=1 timeout {job_seconds} \\
  .venv/bin/python -m mycomap_vision.cli aws-train-job \\
  --run-id {run_id} --backbones {backbones} --methods {methods} --size {size} \\
  --test-days {test_days} --stop-after-hours {stop_hours} \\
  --source "s3://$BUCKET"{finetune_arg}{sample_arg}
"""


def render_trainer_user_data(run_id: str, backbones: list[str], methods: list[str],
                             max_hours: float, bucket_name: str, size: str = "large",
                             test_days: int = 28, finetune: list[str] | None = None,
                             sample_records: int | None = None,
                             code_version: str = "unknown") -> str:
    # The job is killed at max_hours, and stops itself STOP_MARGIN_HOURS before that so
    # it can upload what it finished; the backstop leaves 30 minutes on top for setup
    # and the log upload.
    from .trainer import STOP_MARGIN_HOURS
    if not re.fullmatch(r"[0-9a-f]{7,40}(-dirty)?|unknown", code_version):
        raise ValueError(f"not a commit sha: {code_version!r}")
    stop = max(max_hours - STOP_MARGIN_HOURS, max_hours / 2)
    return TRAINER_USER_DATA.format(
        bucket=bucket_name, run_id=run_id, region=config.setting("MV_AWS_REGION", "us-east-2"),
        backstop_minutes=int(max_hours * 60) + 30, job_seconds=int(max_hours * 3600),
        stop_hours=round(stop, 2), code_version=code_version,
        backbones=",".join(backbones), methods=",".join(methods), size=size,
        test_days=test_days,
        finetune_arg=f" --finetune {','.join(finetune)}" if finetune else "",
        sample_arg=f" --sample-records {int(sample_records)}" if sample_records else "")


def trainer_instance_args(run_id: str, ami: str, user_data: str, instance_type: str,
                          root_device: str, snapshot_gb: int) -> dict:
    """Like the downloader's, with a disk big enough for the image plus model weights and
    embeddings (~1.2 GB per 1024-wide backbone)."""
    args = run_instance_args(run_id, ami, user_data, instance_type)
    args["TagSpecifications"] = [
        {"ResourceType": spec["ResourceType"],
         "Tags": [t if t["Key"] != "Name" else {"Key": "Name", "Value": f"mv-trainer-{run_id}"}
                  for t in spec["Tags"]]}
        for spec in args["TagSpecifications"]]
    args["BlockDeviceMappings"] = [{"DeviceName": root_device,
                                    "Ebs": {"VolumeSize": max(snapshot_gb, 1) + 60,
                                            "VolumeType": "gp3", "DeleteOnTermination": True}}]
    return args


def check_trainer_request(conn, backbones: list[str], methods: list[str], size: str,
                          store_location: str, finetune: list[str] | None = None) -> int:
    """Refuse a run that can't do anything useful, before anything is paid for. Returns
    how many photos there are to embed."""
    from . import models
    from .evaluate import METHODS
    if not backbones:
        raise ValueError("name at least one backbone")
    for b in backbones:
        models.resolve_spec(b)
    if not methods:
        raise ValueError("name at least one method")
    names = {models.storage_name(b) for b in backbones}
    for f in finetune or []:
        if models.storage_name(f) not in names:
            raise ValueError(f"fine-tuning {f} needs it in --backbones too: its embeddings "
                             "seed the classifier and are the before-and-after baseline")
    for m in methods:
        if m not in METHODS:
            raise ValueError(f"unknown method {m!r}: {', '.join(METHODS)}")
    n = conn.execute("select count(*) from photo_copies where store = ? and size = ?",
                     (store_location, size)).fetchone()[0]
    if not n:
        raise ValueError(f"the manifest lists no {size} photos in {store_location}; "
                         "run mv aws-pull-manifest first")
    return n


def photos_for_estimate(conn, photos: int, size: str, store_location: str,
                        sample_records: int | None) -> int:
    """How many photos a run will handle: all of them, or a rehearsal's share."""
    if not sample_records:
        return photos
    records = conn.execute(
        "select count(distinct op.observation_id) from observation_photos op "
        "join photo_copies c on c.photo_id = op.photo_id where c.store = ? and c.size = ?",
        (store_location, size)).fetchone()[0]
    return round(photos * min(1.0, sample_records / max(records, 1)))


def check_run_time(est: dict, max_hours: float, allow_over_time: bool = False,
                   log=print) -> None:
    """Refuse a run the time limit would cut short, unless allowed; warn when it's tight."""
    from .trainer import STOP_MARGIN_HOURS
    total, usable = est["total_hours"], max_hours - STOP_MARGIN_HOURS
    if total > usable:
        msg = (f"the run needs about {total} h, and --max-hours {max_hours} leaves {usable:.2f} h "
               "before it stops itself")
        if not allow_over_time:
            raise ValueError(f"{msg}. Raise --max-hours, drop backbones, or pass "
                             "--allow-over-time to run it anyway (it stops at the limit and "
                             "keeps the stages it finished).")
        log(f"WARNING: {msg}; the last stages will not finish (--allow-over-time).")
    elif total > 0.8 * usable:
        log(f"Note: the estimate ({total} h) is close to the limit ({usable:.2f} h usable).")


def launch_trainer(conn, backbones: list[str], methods: list[str], size: str = "large",
                   max_hours: float = 24, instance_type: str = TRAINER_INSTANCE_TYPE,
                   test_days: int = 28, finetune: list[str] | None = None,
                   sample_records: int | None = None, allow_over_time: bool = False,
                   allow_dirty: bool = False, allow_unpushed: bool = False,
                   log=print) -> dict:
    """Check the request, the time it needs and the commit it ships, all before anything
    is paid for; then upload the code and the manifest and start the instance."""
    from . import trainer
    from .models import storage_name
    b = bucket()
    photos = check_trainer_request(conn, backbones, methods, size, f"s3://{b}/", finetune)
    names = [storage_name(x) for x in backbones]
    ft = [storage_name(x) for x in finetune or []]
    est = trainer.estimate(trainer.plan_stages(names, ft, "<run>"),
                           photos_for_estimate(conn, photos, size, f"s3://{b}/",
                                               sample_records))
    log(trainer.format_estimate(est))
    check_run_time(est, max_hours, allow_over_time, log)
    sha = release_commit(allow_dirty=allow_dirty, allow_unpushed=allow_unpushed, log=log)
    sess = session()
    s3, ec2, ssm = s3_client(), sess.client("ec2"), sess.client("ssm")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    s3.put_object(Bucket=b, Key=f"runs/{run_id}/code.tar.gz", Body=code_tarball(sha))
    snap = snapshot(conn, config.DATA_DIR / "manifest-upload.sqlite")
    s3.upload_file(str(snap), b, f"runs/{run_id}/manifest-in.sqlite")
    log(f"Uploaded commit {sha[:10]} and the manifest for run {run_id} "
        f"({photos:,} {size} photos in S3)")
    ami = ssm.get_parameter(Name=TRAINER_AMI_PARAMETER)["Parameter"]["Value"]
    image = ec2.describe_images(ImageIds=[ami])["Images"][0]
    root = image["RootDeviceName"]
    snapshot_gb = next((m["Ebs"]["VolumeSize"] for m in image["BlockDeviceMappings"]
                        if m.get("DeviceName") == root and "Ebs" in m), 100)
    user_data = render_trainer_user_data(run_id, names, methods, max_hours, b, size, test_days,
                                         ft, sample_records, code_version=sha)
    resp = ec2.run_instances(**trainer_instance_args(run_id, ami, user_data, instance_type,
                                                     root, snapshot_gb))
    return {"run_id": run_id, "instance_id": resp["Instances"][0]["InstanceId"],
            "instance_type": instance_type, "region": region(), "backbones": names,
            "finetune": ft, "methods": methods, "sample_records": sample_records,
            "code_version": sha, "max_hours": max_hours, "estimate_hours": est["total_hours"],
            "log": f"s3://{b}/runs/{run_id}/train.log",
            "progress": f"s3://{b}/runs/{run_id}/progress.json"}


def pull_trainer(conn, run_id: str, log=print) -> dict:
    """Download a trainer run and merge it (see trainer.merge_results): the whole run when
    result.json is there, otherwise the stages progress.json says are done (a run stopped
    by its time limit, killed, or still going). What is missing is reported, and a
    backbone the run didn't finish never replaces the local embeddings."""
    from botocore.exceptions import ClientError

    from . import trainer
    s3, b = s3_client(), bucket()
    prefix = trainer.run_prefix(run_id)
    work = config.DATA_DIR / "aws" / f"run-{run_id}"
    work.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> Path | None:
        try:
            s3.download_file(b, prefix + name, str(work / name))
            return work / name
        except ClientError:
            return None

    doc = source = None
    for name in (trainer.RESULT_FILE, trainer.PROGRESS_FILE):
        path = fetch(name)
        if path is not None:
            doc, source = json.loads(path.read_text(encoding="utf-8")), name
            break
    if doc is None:
        raise RuntimeError(f"run {run_id} has neither {trainer.RESULT_FILE} nor "
                           f"{trainer.PROGRESS_FILE}: not started yet, or it failed during "
                           f"setup (see s3://{b}/{prefix}train.log)")
    summary = trainer.run_summary(doc)
    if not summary["complete"]:
        log(f"run {run_id} is NOT complete (state: {summary.get('state')}, last update "
            f"{doc.get('updated_at')}); bringing home only the stages it finished: "
            f"{', '.join(list(summary['embedded']) + list(summary['finetuned'])) or 'none'}")
        for m in summary["missing"]:
            log(f"  missing: {m['stage']} ({m['status']}{': ' + m['why'] if m['why'] else ''})")
    index = fetch(trainer.INDEX_FILE) or fetch("manifest-out.sqlite")
    if index is None:
        raise RuntimeError(f"run {run_id} has no {trainer.INDEX_FILE} yet: no stage finished")
    staged = work / "embeddings"
    wanted = [(f"embeddings/{n}/", staged / n) for n in summary["embedded"]]
    wanted += [(f"models/{n}.", work / "models") for n in summary["finetuned"]]
    wanted += [("reports/", config.REPORTS_DIR), ("skipped/", work / "skipped")]
    for sub, dest_root in wanted:
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=b, Prefix=prefix + sub):
            for obj in page.get("Contents", []):
                dest = dest_root / obj["Key"].rsplit("/", 1)[-1]
                if dest.exists() and dest.stat().st_size == obj["Size"]:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                s3.download_file(b, obj["Key"], str(dest))
    log(f"downloaded run {run_id} (commit {summary.get('code_version')}); merging")
    merged = trainer.merge_results(conn, index, staged, doc)
    for name, why in merged["refused"].items():
        log(f"NOT merged: {name}: {why} (local embeddings left as they were)")
    (work / "pulled.json").write_text(json.dumps(
        {**merged, "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "from": source},
        indent=2), encoding="utf-8")
    return merged
