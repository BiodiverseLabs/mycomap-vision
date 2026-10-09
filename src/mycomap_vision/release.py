"""Releases: what the public site serves, published to S3 and pulled by the server box.

A release is the folder s3://<bucket>/releases/<id>/:

    manifest.sqlite             a snapshot of the manifest
    embeddings/<backbone>/...   the served backbones' vectors
    models/<name>.json, .pt     fine-tuned weights, when a served backbone is one
    taxonomy/inat_genera.sqlite iNat's families per genus (taxonomy.py), when fetched
    release.json                id, backbones, code version, and every file with its
                                size and sha256; written LAST, so a release without
                                it is unfinished and is never pulled

`releases/current.json` names the release the site should serve. Only
`mv release --make-current` moves it.

The box runs `mv pull-release`: it downloads into <root>/releases/<id>/, checks every
file's size and sha256, and only then rewrites <root>/current.txt (atomically). The
server reads its data from the release current.txt names (MV_RELEASE_ROOT, see
config.py); restart it to switch. The previous release stays on disk for rollback.

The box's AWS user may only read releases/ (deploy/aws/box-policy.template.json).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from . import config, holdouts
from .manifest import shippable_snapshot
from .taxonomy import CACHE as TAXONOMY_CACHE

PREFIX = "releases/"
CURRENT_KEY = PREFIX + "current.json"
POINTER = "current.txt"
RELEASE_ID = re.compile(r"^[0-9]{8}-[0-9]{6}(-[A-Za-z0-9_-]{1,40})?$")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def new_release_id(label: str | None = None, now: datetime | None = None) -> str:
    stamp = (now or datetime.now(timezone.utc)).strftime("%Y%m%d-%H%M%S")
    rid = f"{stamp}-{label}" if label else stamp
    if not RELEASE_ID.match(rid):
        raise ValueError(f"release label must be letters, digits, - or _: {label!r}")
    return rid


def release_files(conn, backbones: list[str], data_dir: Path) -> list[tuple[Path, str]]:
    """(local path, path inside the release) for everything the served backbones need.
    Refuses a backbone with no embeddings: the site could not answer with it."""
    from .embed import SCHEMA
    conn.executescript(SCHEMA)
    files: list[tuple[Path, str]] = []
    for b in backbones:
        n = conn.execute("select count(*) from embeddings where backbone = ?", (b,)).fetchone()[0]
        root = data_dir / "embeddings" / b
        if not n or not root.is_dir():
            raise ValueError(f"{b!r} has no embeddings to release")
        files += [(p, p.relative_to(data_dir).as_posix())
                  for p in sorted(root.rglob("*")) if p.is_file()]
        meta_path = data_dir / "models" / f"{b}.json"
        if meta_path.is_file():                       # a fine-tuned model: ship its weights
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            weights = data_dir / "models" / meta["weights"]
            if not weights.is_file():
                raise ValueError(f"{b!r}: weights file {meta['weights']} is missing")
            files += [(meta_path, f"models/{b}.json"), (weights, f"models/{meta['weights']}")]
    # iNat's families per genus (taxonomy.py): the site labels records as comparisons do.
    if (data_dir / TAXONOMY_CACHE).is_file():
        files.append((data_dir / TAXONOMY_CACHE, TAXONOMY_CACHE.as_posix()))
    return files


def publish(conn, backbones: list[str], *, label: str | None = None, make_current: bool = False,
            s3=None, bucket: str | None = None, data_dir: Path | None = None,
            work_dir: Path | None = None, log=print) -> dict:
    """Upload a release; with make_current, point releases/current.json at it.
    Refused while the records table holds a benchmark's held-out record (holdouts.py);
    the manifest it ships carries the held-out list to the box's nightly update."""
    if not backbones:
        raise ValueError("name at least one backbone to serve")
    holdouts.check_clean(conn, "the release")
    from . import aws
    s3 = s3 or aws.s3_client()
    bucket = bucket or aws.bucket()
    data_dir = data_dir or config.DATA_DIR
    rid = new_release_id(label)
    files = release_files(conn, backbones, data_dir)
    # The benchmark tables stay home; benchmark_holdouts ships (the nightly update needs it).
    snap = shippable_snapshot(conn, (work_dir or data_dir) / "manifest-release.sqlite")
    files = [(snap, "manifest.sqlite")] + files
    listed = []
    for path, rel in files:
        s3.upload_file(str(path), bucket, f"{PREFIX}{rid}/{rel}")
        listed.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256_of(path)})
    info = {"id": rid, "backbones": backbones, "code_version": config.code_version(),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "files": listed}
    s3.put_object(Bucket=bucket, Key=f"{PREFIX}{rid}/release.json",
                  Body=json.dumps(info, indent=2).encode(), ContentType="application/json")
    if make_current:
        set_current(rid, s3=s3, bucket=bucket)
    total = sum(f["bytes"] for f in listed)
    log(f"release {rid}: {len(listed)} files, {total / 2**20:.0f} MB"
        + (" (now current)" if make_current else ""))
    return {"id": rid, "files": len(listed), "bytes": total, "current": make_current}


def set_current(rid: str, *, s3, bucket: str) -> None:
    """Point releases/current.json at a finished release (one that has release.json)."""
    if not RELEASE_ID.match(rid):
        raise ValueError(f"not a release id: {rid!r}")
    read_json(s3, bucket, f"{PREFIX}{rid}/release.json")      # refuses an unfinished one
    s3.put_object(Bucket=bucket, Key=CURRENT_KEY, Body=json.dumps({"id": rid}).encode(),
                  ContentType="application/json")


def read_json(s3, bucket: str, key: str) -> dict:
    return json.loads(s3.get_object(Bucket=bucket, Key=key)["Body"].read())


def current_release(root: Path) -> str | None:
    """The release id <root>/current.txt names, or None."""
    p = root / POINTER
    rid = p.read_text(encoding="utf-8").strip() if p.is_file() else ""
    return rid if RELEASE_ID.match(rid) else None


def safe_relpath(rel: str) -> str:
    """A path from release.json may only point inside the release folder."""
    parts = rel.split("/")
    if not rel or rel.startswith("/") or "\\" in rel or any(p in ("", ".", "..") for p in parts) \
            or ":" in parts[0]:
        raise ValueError(f"release lists an unsafe path: {rel!r}")
    return rel


def pull(root: Path, release_id: str | None = None, *, s3=None, bucket: str | None = None,
         keep: int = 2, log=print) -> dict:
    """Download a release (default: the one releases/current.json names), verify it,
    then switch <root>/current.txt to it. Keeps the newest `keep` releases on disk."""
    from . import aws
    s3 = s3 or aws.s3_client()
    bucket = bucket or aws.bucket()
    rid = release_id or read_json(s3, bucket, CURRENT_KEY)["id"]
    if not RELEASE_ID.match(rid):
        raise ValueError(f"not a release id: {rid!r}")
    releases = root / "releases"
    final = releases / rid
    if current_release(root) == rid and final.is_dir():
        log(f"release {rid} is already current")
        return {"id": rid, "changed": False}
    info = read_json(s3, bucket, f"{PREFIX}{rid}/release.json")
    if info.get("id") != rid:
        raise ValueError(f"release.json of {rid} names {info.get('id')!r}")
    partial = releases / f"{rid}.partial"
    shutil.rmtree(partial, ignore_errors=True)
    total = 0
    for f in info["files"]:
        rel = safe_relpath(f["path"])
        dest = partial / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        s3.download_file(bucket, f"{PREFIX}{rid}/{rel}", str(dest))
        if dest.stat().st_size != f["bytes"] or sha256_of(dest) != f["sha256"]:
            raise ValueError(f"{rel} of release {rid} does not match release.json; "
                             "nothing was switched")
        total += f["bytes"]
    (partial / "release.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    if final.exists():
        shutil.rmtree(final)
    partial.rename(final)
    tmp = root / (POINTER + ".tmp")
    tmp.write_text(rid + "\n", encoding="utf-8")
    os.replace(tmp, root / POINTER)
    removed = prune(root, keep)
    log(f"release {rid} pulled ({total / 2**20:.0f} MB) and made current; "
        "restart the server to serve it")
    return {"id": rid, "changed": True, "bytes": total, "removed": removed}


def prune(root: Path, keep: int) -> list[str]:
    """Delete all but the newest `keep` releases; never the current one."""
    current = current_release(root)
    ids = sorted(p.name for p in (root / "releases").iterdir()
                 if p.is_dir() and RELEASE_ID.match(p.name))
    doomed = [r for r in ids[:-keep] if r != current] if keep > 0 else []
    for r in doomed:
        shutil.rmtree(root / "releases" / r)
    return doomed
