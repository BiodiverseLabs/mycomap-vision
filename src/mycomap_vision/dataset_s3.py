"""A dataset release's kept copy in S3 (docs/dataset-release.md §2): the whole release
folder, private part included, under `research-releases/<id>/` in the project bucket, beside
the photo store and never under `releases/` (what the serving box reads). The in-region build
(aws_dataset.py) uploads it; `mv dataset pull <id>` brings it home; `mv dataset upload <id>`
sends a release built on the laptop. (A trainer gets its copy, without private.sqlite,
shipped with its run: aws.launch_trainer.)

Nothing here edits a release: uploading copies bytes, and both directions check each file
against the sha256 MANIFEST.json lists.
"""

from __future__ import annotations

import json
from pathlib import Path

from .dataset_release import MANIFEST, PRIVATE_DB, SUMS, ReleaseError, check_id, sha256_file

PREFIX = "research-releases/"


def release_prefix(rid: str) -> str:
    return f"{PREFIX}{check_id(rid)}/"


def _manifest(folder: Path) -> dict:
    path = folder / MANIFEST
    if not path.is_file():
        raise ReleaseError(f"{folder} has no {MANIFEST}: not a finished release")
    return json.loads(path.read_text(encoding="utf-8"))


def trainer_files(manifest: dict) -> list[dict]:
    """The files of a trainer copy: every listed file but private.sqlite."""
    return [f for f in manifest["files"] if f["path"] != PRIVATE_DB]


def _remote_sizes(s3, bucket: str, prefix: str) -> dict[str, int]:
    out = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"][len(prefix):]] = o["Size"]
    return out


def upload_release(rid: str, root: Path, s3, bucket: str, log=print) -> dict:
    """Upload the release folder as its kept copy (private.sqlite included: the bucket is
    private, and the kept copy is the whole release). Files already there with the right
    size and sha256 tag are skipped; a file there with other bytes is refused (a release is
    never edited). MANIFEST.json goes last, so a half-finished upload is never taken for a
    release."""
    folder = Path(root) / check_id(rid)
    manifest = _manifest(folder)
    prefix = release_prefix(rid)
    remote = _remote_sizes(s3, bucket, prefix)
    sent = skipped = 0
    for f in manifest["files"]:
        local = folder / f["path"]
        if sha256_file(local) != f["sha256"]:
            raise ReleaseError(f"{rid}: local {f['path']} does not match {MANIFEST}")
        key = prefix + f["path"]
        if f["path"] in remote:
            tag = s3.head_object(Bucket=bucket, Key=key).get("Metadata", {}).get("sha256")
            if remote[f["path"]] == f["bytes"] and tag == f["sha256"]:
                skipped += 1
                continue
            raise ReleaseError(f"s3://{bucket}/{key} exists with other bytes: a release is "
                               "never edited (cut the next one)")
        s3.upload_file(str(local), bucket, key, ExtraArgs={"Metadata": {"sha256": f["sha256"]}})
        sent += 1
    # SHA256SUMS is not in MANIFEST.json's file list (it lists them); MANIFEST pins its hash.
    if sha256_file(folder / SUMS) != manifest["sums_sha256"]:
        raise ReleaseError(f"{rid}: local {SUMS} does not match {MANIFEST}")
    s3.upload_file(str(folder / SUMS), bucket, prefix + SUMS,
                   ExtraArgs={"Metadata": {"sha256": manifest["sums_sha256"]}})
    s3.upload_file(str(folder / MANIFEST), bucket, prefix + MANIFEST)
    log(f"release {rid}: {sent} files uploaded, {skipped} already there -> s3://{bucket}/{prefix}")
    return {"id": rid, "uploaded": sent, "already_there": skipped,
            "release_hash": manifest["release_hash"], "prefix": prefix}


def remote_release_hash(rid: str, s3, bucket: str) -> str | None:
    """The release_hash of the kept copy in S3, or None when it is not there (yet)."""
    from botocore.exceptions import ClientError
    try:
        body = s3.get_object(Bucket=bucket, Key=release_prefix(rid) + MANIFEST)["Body"].read()
    except ClientError:
        return None
    return json.loads(body).get("release_hash")


def fetch_release(rid: str, dest_root: Path, s3, bucket: str, *, include_private: bool,
                  log=print) -> Path:
    """Download a release from its kept copy to <dest_root>/<id>/ and check every file
    against MANIFEST.json; returns the folder. Without the private part (a trainer copy)
    open it with load_release(id, dest_root, without_private=True)."""
    prefix = release_prefix(rid)
    folder = Path(dest_root) / rid
    if include_private and folder.exists():
        raise ReleaseError(f"{folder} exists: a release is never overwritten")
    folder.mkdir(parents=True, exist_ok=True)
    s3.download_file(bucket, prefix + MANIFEST, str(folder / MANIFEST))
    manifest = _manifest(folder)
    if manifest.get("id") != rid:
        raise ReleaseError(f"s3 {MANIFEST} of {rid} names {manifest.get('id')!r}")
    files = manifest["files"] if include_private else trainer_files(manifest)
    for f in files:
        dest = folder / f["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, prefix + f["path"], str(dest))
        if dest.stat().st_size != f["bytes"] or sha256_file(dest) != f["sha256"]:
            raise ReleaseError(f"{rid}: downloaded {f['path']} does not match {MANIFEST}")
    s3.download_file(bucket, prefix + SUMS, str(folder / SUMS))
    if sha256_file(folder / SUMS) != manifest["sums_sha256"]:
        raise ReleaseError(f"{rid}: downloaded {SUMS} does not match {MANIFEST}")
    if not include_private and (folder / PRIVATE_DB).exists():
        raise ReleaseError(f"{rid}: a trainer copy must not hold {PRIVATE_DB}")
    log(f"release {rid}: {'full release' if include_private else 'trainer copy'} fetched "
        f"({len(files)} files, release_hash {manifest['release_hash'][:12]})")
    return folder
