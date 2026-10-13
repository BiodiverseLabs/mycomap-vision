"""Cutting a dataset release in-region: `mv aws-launch-dataset-build`.

A frozen release stores the derived hash of every included photo (docs/dataset-release.md
§7): every original (~650k photos, ~280 GB in S3) is read, resized and hashed. On the laptop
that means downloading all of it (~$25 of S3 transfer and many hours); on an instance in the
bucket's region the reads cost nothing and 8 vCPUs do it in about two hours for about a
dollar. So the build runs there:

- the laptop checks the request before anything is paid for (id, freeze trigger, review
  lists, no release of that id here or in S3, the commit is pushed), then uploads the code,
  a full snapshot of the manifest (the build needs the held-out membership tables, so not
  the trainer's stripped copy; the bucket is private), the iNat taxonomy cache, the review
  lists and request.json (the build's settings; the freeze-trigger text travels in this
  file, never through a shell) to `dataset-builds/<build>/`, and starts a self-terminating
  instance (c7i.2xlarge by default) with a backstop shutdown;
- the instance (`aws_dataset.run_build_job`) runs `dataset_release.build(..., derive all)`
  reading originals straight from S3, verifies the result, uploads it as the release's kept
  copy (`research-releases/<id>/`, dataset_s3), and writes `result.json` last;
- `mv dataset pull <id>` brings the release home (with its private part) and verifies it.

It installs the trainer's lockfile (requirements/trainer.txt): the same Pillow as the laptop
(12.3.0), so a derived image made later on the laptop matches the stored hash byte for byte.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path

from . import config
from . import dataset_release as dr

PREFIX = "dataset-builds/"
REQUEST_FILE = "request.json"
RESULT_FILE = "result.json"
BUILD_INSTANCE_TYPE = "c7i.2xlarge"
BUILD_VCPUS = {"c7i.xlarge": 4, "c7i.2xlarge": 8, "c7i.4xlarge": 16, "c6i.2xlarge": 8}
# Seconds per photo on one laptop core to read, hash and derive long500-q90 (measured
# 2026-10-11 on 300 large photos: 53.5 ms); an instance core is taken as no faster, and
# threads reach ~75% of the cores (PIL releases the GIL while decoding and encoding).
DERIVE_SECONDS_PER_PHOTO = 0.0535
THREAD_EFFICIENCY = 0.75
SETUP_HOURS = 0.4                     # boot, packages (torch comes with the lockfile), copies
VERIFY_UPLOAD_HOURS = 0.2


def build_prefix(build_id: str) -> str:
    if not re.fullmatch(r"[0-9]{8}-[0-9]{6}", build_id):
        raise ValueError(f"not a build id: {build_id!r}")
    return f"{PREFIX}{build_id}/"


def estimate_hours(photos: int, instance_type: str = BUILD_INSTANCE_TYPE) -> float:
    cores = BUILD_VCPUS.get(instance_type, 4) * THREAD_EFFICIENCY
    return round(SETUP_HOURS + VERIFY_UPLOAD_HOURS
                 + photos * DERIVE_SECONDS_PER_PHOTO / cores / 3600, 1)


USER_DATA = """#!/bin/bash
set -uo pipefail
BUCKET={bucket}
RUN={prefix}
LOG=/var/log/mv-dataset-build.log
exec > >(tee -a "$LOG") 2>&1
export AWS_DEFAULT_REGION={region}
# Backstop: power off after {backstop_minutes} minutes whatever happens.
shutdown -h +{backstop_minutes}
finish() {{
  aws s3 cp "$LOG" "s3://$BUCKET/${{RUN}}build.log" || true
  shutdown -h now
}}
trap finish EXIT
( while sleep 600; do aws s3 cp "$LOG" "s3://$BUCKET/${{RUN}}build.log" >/dev/null 2>&1; done ) &
dnf install -y -q python3.11 python3.11-pip
mkdir -p /opt/mv/data/taxonomy /opt/mv/inputs && cd /opt/mv
aws s3 cp "s3://$BUCKET/${{RUN}}code.tar.gz" code.tar.gz
tar xzf code.tar.gz
python3.11 -m venv .venv
.venv/bin/pip install -q --require-hashes -r requirements/trainer.txt || exit 1
aws s3 cp "s3://$BUCKET/${{RUN}}manifest.sqlite" data/manifest.sqlite
aws s3 cp "s3://$BUCKET/${{RUN}}taxonomy.sqlite" data/taxonomy/inat_genera.sqlite \\
  || echo "no iNat taxonomy shipped"
aws s3 cp --recursive "s3://$BUCKET/${{RUN}}inputs/" inputs/ || true
aws s3 cp "s3://$BUCKET/${{RUN}}{request}" {request}
export MV_CODE_VERSION={code_version}
MV_DATA_DIR=/opt/mv/data PYTHONPATH=/opt/mv/src PYTHONUNBUFFERED=1 timeout {job_seconds} \\
  .venv/bin/python -m mycomap_vision.cli aws-dataset-build-job --build-id {build_id} \\
  --request {request} --bucket "$BUCKET"
"""


def render_user_data(build_id: str, bucket_name: str, max_hours: float,
                     code_version: str) -> str:
    if not re.fullmatch(r"[0-9a-f]{7,40}", code_version):
        raise ValueError(f"not a commit sha: {code_version!r}")
    return USER_DATA.format(bucket=bucket_name, prefix=build_prefix(build_id),
                            region=config.setting("MV_AWS_REGION", "us-east-2"),
                            backstop_minutes=int(max_hours * 60) + 30,
                            job_seconds=int(max_hours * 3600), code_version=code_version,
                            build_id=build_id, request=REQUEST_FILE)


def make_request(name: str, freeze_approved: str | None, *, val_weeks: int = dr.DEFAULT_VAL_WEEKS,
                 val_cutoff: str | None = None, recipes=("original", "long500-q90"),
                 review_files: list[Path] = (), derive: str = "all",
                 loo_exclusions: Path | None = None, no_loo_exclusions: str | None = None
                 ) -> dict:
    """The build's settings, checked: what the instance reads from request.json. The LOO
    gate (dataset_release.loo_gate) is checked here, before any spend: a frozen release
    needs the LOO exclusion list or Steve's word that there is none."""
    dr.check_id(name)
    if not dr.is_draft(name):
        if not freeze_approved or not freeze_approved.strip():
            raise dr.ReleaseError(f"{name} is a frozen release id: cut it only with "
                                  "--freeze-approved '<who, when>' after Steve's freeze trigger")
        if derive != "all":
            raise dr.ReleaseError("a frozen release stores every derived hash (derive all)")
    if val_cutoff and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", val_cutoff):
        raise ValueError(f"--val-cutoff is YYYY-MM-DD, not {val_cutoff!r}")
    loo = dr.loo_gate(name, None, loo_exclusions, no_loo_exclusions)
    dr.read_review_lists([Path(p) for p in review_files])        # fails on a malformed list
    for r in recipes:
        dr.recipe_with_library(r)                                 # fails on an unknown recipe
    return {"name": name, "freeze_approved": freeze_approved, "val_weeks": int(val_weeks),
            "val_cutoff": val_cutoff, "recipes": list(recipes), "derive": derive,
            "review_files": [Path(p).name for p in review_files],
            "loo_exclusions": Path(loo_exclusions).name if loo_exclusions else None,
            "no_loo_exclusions": no_loo_exclusions, "loo": loo}


def launch(conn, name: str, freeze_approved: str | None, *, val_weeks: int = dr.DEFAULT_VAL_WEEKS,
           val_cutoff: str | None = None, review_files: list[Path] = (),
           loo_exclusions: Path | None = None, no_loo_exclusions: str | None = None,
           manifest_path: Path | None = None,
           instance_type: str = BUILD_INSTANCE_TYPE, max_hours: float = 6,
           allow_dirty: bool = False, allow_unpushed: bool = False, root: Path | None = None,
           log=print) -> dict:
    from . import aws, dataset_s3, sources
    from .manifest import snapshot
    from .taxonomy import cache_path
    request = make_request(name, freeze_approved, val_weeks=val_weeks, val_cutoff=val_cutoff,
                           review_files=review_files, loo_exclusions=loo_exclusions,
                           no_loo_exclusions=no_loo_exclusions)
    uploads = list(review_files) + ([Path(loo_exclusions)] if loo_exclusions else [])
    if len({Path(p).name for p in uploads}) != len(uploads):
        raise ValueError("two review lists share a file name")
    root = Path(root or dr.ROOT)
    if (root / name).exists():
        raise dr.ReleaseError(f"release {name} exists here ({root}); a release is never rebuilt")
    # The manifest the release freezes: a fixed copy (`manifest_path`, the file the LOO
    # exclusion list was written against) or, without one, a snapshot of this manifest now.
    frozen = None
    if manifest_path is not None:
        import sqlite3
        manifest_path = Path(manifest_path)
        if not manifest_path.is_file():
            raise ValueError(f"no manifest at {manifest_path}")
        frozen = sqlite3.connect(f"file:{manifest_path.resolve().as_posix()}?mode=ro", uri=True)
        conn = frozen
    sources.require_migrated(conn, "cutting a dataset release")
    photos = conn.execute("select count(*) from photo_copies where size = 'large' and "
                          "store like 's3://%'").fetchone()[0]
    hours = estimate_hours(photos, instance_type)
    log(f"Estimate: ~{photos:,} originals in S3, {hours} h on a {instance_type} "
        f"(--max-hours {max_hours})")
    if hours > max_hours - 0.25:
        raise ValueError(f"the build needs about {hours} h; raise --max-hours")
    sha = aws.release_commit(allow_dirty=allow_dirty, allow_unpushed=allow_unpushed, log=log)
    b = aws.bucket()
    s3 = aws.s3_client()
    if dataset_s3.remote_release_hash(name, s3, b) is not None:
        raise dr.ReleaseError(f"s3 already holds {name}: a release id is never reused")
    build_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = build_prefix(build_id)
    s3.put_object(Bucket=b, Key=prefix + "code.tar.gz", Body=aws.code_tarball(sha))
    if frozen is not None:
        frozen.close()
        snap = manifest_path                 # shipped byte for byte: the LOO list's manifest
    else:
        snap = snapshot(conn, config.DATA_DIR / "manifest-dataset-build.sqlite")
    manifest_sha = dr.sha256_file(snap)
    s3.upload_file(str(snap), b, prefix + "manifest.sqlite")
    if cache_path().is_file():
        s3.upload_file(str(cache_path()), b, prefix + "taxonomy.sqlite")
    else:
        log("WARNING: no iNat taxonomy cache: the release's families would be .org's")
    for p in uploads:
        s3.upload_file(str(p), b, prefix + "inputs/" + Path(p).name)
    s3.put_object(Bucket=b, Key=prefix + REQUEST_FILE,
                  Body=json.dumps({**request, "build_id": build_id, "code_version": sha,
                                   "manifest_sha256": manifest_sha,
                                   "manifest_file": Path(snap).name}, indent=2).encode())
    log(f"Uploaded commit {sha[:10]}, the manifest and the request for build {build_id}")
    sess = aws.session()
    ec2, ssm = sess.client("ec2"), sess.client("ssm")
    ami = ssm.get_parameter(Name=aws.AMI_PARAMETER)["Parameter"]["Value"]
    args = instance_args(build_id, ami, render_user_data(build_id, b, max_hours, sha),
                         instance_type)
    resp = ec2.run_instances(**args)
    return {"build_id": build_id, "release": name, "instance_id":
            resp["Instances"][0]["InstanceId"], "instance_type": instance_type,
            "estimate_hours": hours, "max_hours": max_hours, "code_version": sha,
            "manifest_sha256": manifest_sha,
            "log": f"s3://{b}/{prefix}build.log", "result": f"s3://{b}/{prefix}{RESULT_FILE}",
            "then": f"mv dataset pull {name}"}


def instance_args(build_id: str, ami: str, user_data: str, instance_type: str) -> dict:
    """A tagged, self-terminating instance with room for the packages, the manifest and
    the release (40 GB)."""
    from . import aws
    args = aws.run_instance_args(build_id, ami, user_data, instance_type)
    tags = [aws.PROJECT_TAG, {"Key": "Name", "Value": f"mv-dataset-build-{build_id}"}]
    args["TagSpecifications"] = [{"ResourceType": "instance", "Tags": tags},
                                 {"ResourceType": "volume", "Tags": tags}]
    args["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] = 40
    return args


# --- on the instance ------------------------------------------------------------------

def run_build_job(build_id: str, request: dict, s3, bucket_name: str, *,
                  manifest_path: Path | None = None, root: Path | None = None,
                  inputs: Path = Path("inputs"), store_reader=None, log=print) -> dict:
    """Build, verify, upload the kept copy, then write result.json (last: its presence
    means the release is in S3). A `dhash-index` request runs run_dhash_job instead."""
    from . import dataset_s3
    if request.get("job") == DHASH_JOB:
        return run_dhash_job(build_id, request, s3, bucket_name, root=root,
                             store_reader=store_reader, log=log)
    root = Path(root or dr.ROOT)
    prefix = build_prefix(build_id)
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        want = request.get("manifest_sha256")
        got = dr.sha256_file(Path(manifest_path or config.MANIFEST_PATH))
        if want and got != want:
            raise dr.ReleaseError("the manifest on the instance is not the one the launch "
                                  "shipped (sha256 differs)")
        loo_file = inputs / request["loo_exclusions"] if request.get("loo_exclusions") else None
        if loo_file is not None and dr.sha256_file(loo_file) != request["loo"]["sha256"]:
            raise dr.ReleaseError(f"{loo_file.name} on the instance is not the LOO list the "
                                  "launch checked (sha256 differs)")
        manifest = dr.build(name=request["name"], manifest_path=manifest_path, root=root,
                            loo_exclusions=loo_file,
                            no_loo_exclusions=request.get("no_loo_exclusions"),
                            review_files=[inputs / f for f in request["review_files"]],
                            recipes=request["recipes"], derive_photos=request["derive"],
                            val_cutoff=request["val_cutoff"], val_weeks=request["val_weeks"],
                            freeze_approved=request["freeze_approved"],
                            store_reader=store_reader or dr.default_store_reader(), log=log)
        checked = dr.verify(request["name"], root, allow_draft=True)
        up = dataset_s3.upload_release(request["name"], root, s3, bucket_name, log=log)
        out = {"state": "finished", "release": request["name"],
               "release_hash": manifest["release_hash"], "counts": manifest["counts"],
               "splits": manifest["splits"], "verified": checked["ok"], "uploaded": up}
    except Exception as e:  # noqa: BLE001 - reported in result.json, then re-raised
        out = {"state": "failed", "release": request["name"],
               "error": f"{e.__class__.__name__}: {e}"}
        _put_result(s3, bucket_name, prefix, {**out, "build_id": build_id, "started_at": started})
        raise
    out = {**out, "build_id": build_id, "started_at": started,
           "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _put_result(s3, bucket_name, prefix, out)
    return out


def _put_result(s3, bucket_name: str, prefix: str, doc: dict) -> None:
    s3.put_object(Bucket=bucket_name, Key=prefix + RESULT_FILE,
                  Body=json.dumps(doc, indent=2).encode())


# --- a base release's dHash index (dhash.py; paper plan §4, 2026-10-13) ----------------------
#
# A frozen test release flags near-copies of its base's photos by dHash. The base side reads
# every included original once (v1: 680k photos, ~290 GB): in the bucket's region that costs
# an instance hour, on the laptop ~$25 of transfer. The index is written beside the release's
# kept copy, never into it (a release is never edited): research-release-extras/<id>/.

DHASH_JOB = "dhash-index"
EXTRAS_PREFIX = "research-release-extras/"
# Wall seconds per photo on the instance with 16 threads: an S3 read of a ~0.4 MB original
# plus a 2.4 ms dHash (laptop, 2026-10-13). The v1 build read, resized and re-encoded every
# original at ~150/s on a c7i.2xlarge; a read and a dHash are taken as no faster.
DHASH_SECONDS_PER_PHOTO = 1 / 150


def dhash_index_key(rid: str) -> str:
    from .dhash import INDEX_FILE
    return f"{EXTRAS_PREFIX}{dr.check_id(rid)}/{INDEX_FILE}"


def estimate_dhash_hours(photos: int) -> float:
    return round(SETUP_HOURS + 0.1 + photos * DHASH_SECONDS_PER_PHOTO / 3600, 1)


def launch_dhash_index(rid: str, *, instance_type: str = BUILD_INSTANCE_TYPE,
                       max_hours: float = 4, allow_dirty: bool = False,
                       allow_unpushed: bool = False, log=print) -> dict:
    """Start a self-terminating instance that writes <rid>'s dHash index to S3. Refused
    before any spend when the release has no kept copy or its index is there already."""
    from . import aws, dataset_s3
    dr.check_id(rid)
    b, s3 = aws.bucket(), aws.s3_client()
    remote = dataset_s3.remote_release_hash(rid, s3, b)
    if remote is None:
        raise dr.ReleaseError(f"s3 holds no kept copy of {rid} (research-releases/{rid}/)")
    try:
        s3.head_object(Bucket=b, Key=dhash_index_key(rid))
        raise dr.ReleaseError(f"s3 already holds {dhash_index_key(rid)}: "
                              f"`mv dataset pull-dhash-index {rid}`")
    except dr.ReleaseError:
        raise
    except Exception:  # noqa: BLE001 - not there: go on
        pass
    manifest = json.loads(s3.get_object(Bucket=b, Key=dataset_s3.release_prefix(rid)
                                        + dr.MANIFEST)["Body"].read())
    photos = int(manifest["counts"]["photos_included"])
    hours = estimate_dhash_hours(photos)
    log(f"Estimate: {photos:,} included photos of {rid}, {hours} h on a {instance_type} "
        f"(--max-hours {max_hours})")
    if hours > max_hours - 0.25:
        raise ValueError(f"the index needs about {hours} h; raise --max-hours")
    sha = aws.release_commit(allow_dirty=allow_dirty, allow_unpushed=allow_unpushed, log=log)
    build_id = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    prefix = build_prefix(build_id)
    s3.put_object(Bucket=b, Key=prefix + "code.tar.gz", Body=aws.code_tarball(sha))
    request = {"job": DHASH_JOB, "name": rid, "release_hash": remote, "build_id": build_id,
               "code_version": sha}
    s3.put_object(Bucket=b, Key=prefix + REQUEST_FILE,
                  Body=json.dumps(request, indent=2).encode())
    sess = aws.session()
    ec2, ssm = sess.client("ec2"), sess.client("ssm")
    ami = ssm.get_parameter(Name=aws.AMI_PARAMETER)["Parameter"]["Value"]
    resp = ec2.run_instances(**instance_args(
        build_id, ami, render_user_data(build_id, b, max_hours, sha), instance_type))
    return {"build_id": build_id, "job": DHASH_JOB, "release": rid, "release_hash": remote,
            "instance_id": resp["Instances"][0]["InstanceId"], "instance_type": instance_type,
            "estimate_hours": hours, "max_hours": max_hours, "code_version": sha,
            "log": f"s3://{b}/{prefix}build.log", "result": f"s3://{b}/{prefix}{RESULT_FILE}",
            "then": f"mv dataset pull-dhash-index {rid}"}


def run_dhash_job(build_id: str, request: dict, s3, bucket_name: str, *,
                  root: Path | None = None, store_reader=None, log=print) -> dict:
    """On the instance: fetch the release's kept copy (private part: the stores), check it
    is the release the launch saw, dHash every included original, upload the index, then
    write result.json (last)."""
    from . import dataset_s3, dhash
    root = Path(root or dr.ROOT)
    prefix = build_prefix(build_id)
    rid = request["name"]
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    try:
        dataset_s3.fetch_release(rid, root, s3, bucket_name, include_private=True, log=log)
        rel = dr.load_release(rid, root, allow_draft=True)
        try:
            if rel.release_hash != request["release_hash"]:
                raise dr.ReleaseError(f"{rid} in s3 is not the release the launch saw "
                                      "(release_hash differs)")
            index = dhash.build_index(rel, store_reader or dr.default_store_reader(), log=log)
        finally:
            rel.close()
        path = dhash.save_index(index, root.parent / "extras" / rid / dhash.INDEX_FILE)
        sha = dr.sha256_file(path)
        s3.upload_file(str(path), bucket_name, dhash_index_key(rid), ExtraArgs={
            "Metadata": {"sha256": sha, "release_hash": index["release_hash"]}})
        out = {"state": "finished", "job": DHASH_JOB, "release": rid,
               "release_hash": index["release_hash"], "photos": index["photos"],
               "algorithm": index["algorithm"], "pillow": index["pillow"],
               "index_key": dhash_index_key(rid), "index_sha256": sha}
    except Exception as e:  # noqa: BLE001 - reported in result.json, then re-raised
        _put_result(s3, bucket_name, prefix, {"state": "failed", "job": DHASH_JOB,
                                              "release": rid, "build_id": build_id,
                                              "started_at": started,
                                              "error": f"{e.__class__.__name__}: {e}"})
        raise
    out = {**out, "build_id": build_id, "started_at": started,
           "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    _put_result(s3, bucket_name, prefix, out)
    return out


def pull_dhash_index(rid: str, s3, bucket_name: str, dest: Path | None = None,
                     log=print) -> Path:
    """Download <rid>'s dHash index (default dhash.index_path) and check it against the
    sha256 the instance stored with it."""
    from . import dhash
    key = dhash_index_key(rid)
    dest = Path(dest or dhash.index_path(rid))
    want = s3.head_object(Bucket=bucket_name, Key=key)["Metadata"].get("sha256")
    if dest.exists():
        if dr.sha256_file(dest) == want:
            return dest
        raise dr.ReleaseError(f"{dest} exists with other bytes than s3 {key}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".partial")
    s3.download_file(bucket_name, key, str(tmp))
    if not want or dr.sha256_file(tmp) != want:
        tmp.unlink()
        raise dr.ReleaseError(f"downloaded {key} does not match its stored sha256")
    tmp.rename(dest)
    log(f"dHash index of {rid}: {dest}")
    return dest
