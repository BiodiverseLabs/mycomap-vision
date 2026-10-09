"""Opening a dataset release for the replication, on the builder's reader
(`mycomap_vision.dataset_release`, docs/dataset-release.md).

Only what the reader doesn't do yet lives here:

- `load(..., files=SHIPPED)`: a trainer instance gets release.sqlite, MANIFEST.json and
  SHA256SUMS only (never private.sqlite: coordinates and owners stay on the laptop), so it
  checks those files against MANIFEST.json and opens the release without the full file
  check (load_release's check_files needs every file).
- `manifest_sha256` / `cite`: the reader's cite() plus the hash of the MANIFEST.json read,
  which a launch records and the instance must find again.
- `allowed_sources`: the sources of the release's included records (the build excludes
  every other source, with its reason).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ... import dataset_release as dr

ReleaseError = dr.ReleaseError
SealedLeak = dr.SealedLeak
TRAINABLE = dr.TRAINABLE
RELEASES_DIR = "research-releases"                 # under the data folder (dr.ROOT)
SHIPPED = (dr.RELEASE_DB, dr.MANIFEST, dr.SUMS)       # what a trainer needs; not private.sqlite


def root_dir(root: Path | None = None) -> Path:
    """Where releases live, read when called (dataset_release.ROOT is fixed at import)."""
    from ... import config
    return Path(root) if root else config.DATA_DIR / RELEASES_DIR


def manifest_sha256(rel) -> str:
    return hashlib.sha256((rel.folder / dr.MANIFEST).read_bytes()).hexdigest()


def cite(rel) -> dict:
    """The reader's cite(), plus the MANIFEST.json hash a launch pins."""
    return {**rel.cite(), "manifest_sha256": manifest_sha256(rel)}


def is_test_release(rel) -> bool:
    return bool(rel.manifest.get("test_for"))


def verify_shipped(folder: Path, files=SHIPPED) -> None:
    """The shipped files against MANIFEST.json: each data file by its listed sha256,
    SHA256SUMS by `sums_sha256`."""
    manifest = json.loads((folder / dr.MANIFEST).read_text(encoding="utf-8"))
    listed = {f["path"]: f for f in manifest.get("files") or []}
    for name in files:
        if name == dr.MANIFEST:
            continue
        path = folder / name
        if not path.is_file():
            raise ReleaseError(f"{folder.name}: {name} is missing")
        want = (manifest.get("sums_sha256") if name == dr.SUMS
                else (listed.get(name) or {}).get("sha256"))
        if want is None:
            raise ReleaseError(f"{folder.name}: {name} is not listed in {dr.MANIFEST}")
        if dr.sha256_file(path) != want:
            raise ReleaseError(f"{folder.name}: {name} does not match {dr.MANIFEST} (altered)")


def load(rid: str, allow_draft: bool = False, root: Path | None = None,
         files: tuple[str, ...] | None = None, expect_manifest_sha256: str | None = None):
    """dataset_release.load_release with every file checked; or, with `files` (a trainer's
    shipped copy), only those. With `expect_manifest_sha256`, refuse any other release
    than the one the launch recorded."""
    base = root_dir(root)
    if files is None:
        rel = dr.load_release(rid, base, allow_draft=allow_draft)
    else:
        if not (base / rid / dr.MANIFEST).is_file():
            raise ReleaseError(f"release {rid} is missing or unfinished in {base}")
        verify_shipped(base / rid, files)
        rel = dr.load_release(rid, base, allow_draft=allow_draft, check_files=False)
    if expect_manifest_sha256 and manifest_sha256(rel) != expect_manifest_sha256:
        rel.close()
        raise ReleaseError(f"release {rid}'s {dr.MANIFEST} is not the one the launch "
                           "recorded")
    return rel


def allowed_sources(rel) -> tuple[str, ...]:
    """The sources of the release's included records."""
    got = tuple(sorted(r[0] for r in rel.db.execute(
        "select distinct source from records where included = 1")))
    if not got:
        raise ReleaseError(f"release {rel.id} includes no records")
    return got


def benchmark_records(rel, splits=TRAINABLE) -> int:
    """Included records of `splits` that belong to a held-out benchmark."""
    return rel.db.execute(
        f"select count(*) from records where included = 1 and benchmark is not null and "
        f"split in ({','.join('?' * len(splits))})", tuple(splits)).fetchone()[0]
