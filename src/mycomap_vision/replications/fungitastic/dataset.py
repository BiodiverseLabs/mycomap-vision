"""Opening a dataset release for the replication, on the builder's reader
(`mycomap_vision.dataset_release`, docs/dataset-release.md).

Only what the reader doesn't do yet lives here:

- `shipped_files` / `load(..., without_private=True)`: a trainer instance gets every file
  MANIFEST.json lists except private.sqlite (release.sqlite, DATASET_CARD.md), plus
  MANIFEST.json and SHA256SUMS; coordinates and owners stay on the laptop. The builder's
  trainer mode (load_release(id, without_private=True)) checks every one of them.
- `manifest_sha256` / `cite`: the reader's cite() plus the hash of the MANIFEST.json read,
  which a launch records and the instance must find again.
- `allowed_sources`: the sources of the release's included records (the build excludes
  every other source, with its reason).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ... import dataset_release as dr

ReleaseError = dr.ReleaseError
SealedLeak = dr.SealedLeak
TRAINABLE = dr.TRAINABLE
RELEASES_DIR = "research-releases"                 # under the data folder (dr.ROOT)
# A shipped file's path as the instance script names it (no spaces, quotes or '..').
SAFE_PATH = re.compile(r"[A-Za-z0-9_-][A-Za-z0-9._-]*(/[A-Za-z0-9_-][A-Za-z0-9._-]*)*")


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


def shipped_files(rel) -> tuple[str, ...]:
    """What a trainer copy of `rel` ships: MANIFEST.json, SHA256SUMS and every file
    MANIFEST.json lists except private.sqlite (release.sqlite, DATASET_CARD.md, any review
    inputs), which is what load_release(without_private=True) checks. Never private.sqlite.
    SHA256SUMS is shipped as part of the release but read by nothing: MANIFEST.json (pinned
    by its sha256 at launch) lists every file's hash."""
    listed = sorted(f["path"] for f in rel.manifest["files"] if f["path"] != dr.PRIVATE_DB)
    out = (dr.MANIFEST, dr.SUMS, *(p for p in listed if p not in (dr.MANIFEST, dr.SUMS)))
    bad = [p for p in out if not SAFE_PATH.fullmatch(p) or ".." in p.split("/")]
    if bad:
        raise ReleaseError(f"{rel.id}: file names an instance script can't carry: {bad}")
    return out


def load(rid: str, allow_draft: bool = False, root: Path | None = None,
         without_private: bool = False, expect_manifest_sha256: str | None = None):
    """dataset_release.load_release with every file checked; `without_private`: a
    trainer's shipped copy (shipped_files), the builder's trainer mode, which checks every
    listed file but private.sqlite. With `expect_manifest_sha256`, refuse any other
    release than the one the launch recorded."""
    rel = dr.load_release(rid, root_dir(root), allow_draft=allow_draft,
                          without_private=without_private)
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
