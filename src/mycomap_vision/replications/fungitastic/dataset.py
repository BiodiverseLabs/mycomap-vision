"""A dataset release, read for the replication (docs/dataset-release.md on feat/dataset-release).

A minimal local adapter until the release builder's reader merges.
TODO(feat/dataset-release): replace with `mycomap_vision.dataset_release.load_release`
(and its photo_ids(), training_records(), cite()) once that branch is on main; the names
here follow its spec (§2 layout, §3 tables, §6 hashes, §8 reader) so the switch is local.

What the replication takes from a release, and nothing else: its records, labels, splits,
photos (by key, with each original's sha256 and licence class), its allowed sources and its
hashes. Photo bytes come from the store, checked against `original_sha256`. The release is
opened read-only (`mode=ro&immutable=1`), and its files are checked against SHA256SUMS
and MANIFEST.json before anything reads them. `private.sqlite` (coordinates, owners) is
never read here and never shipped to a trainer.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

RELEASES_DIR = "research-releases"          # under the data folder (spec §2)
DB = "release.sqlite"
MANIFEST = "MANIFEST.json"
SUMS = "SHA256SUMS"
PRIVATE = "private.sqlite"
# What a trainer instance needs of a release: never private.sqlite.
SHIPPED = (DB, MANIFEST, SUMS)
TRAINING_SPLITS = ("train", "val")
NEVER_TRAINED = ("test", "sealed")
EVAL_SPLITS = ("val", "test", "sealed")
# TODO(feat/record-sources-mo): mo.MO_PHOTO_ID_BASE; the reader's photo_ids() will map keys.
MO_PHOTO_ID_BASE = 10 ** 12


class ReleaseError(ValueError):
    """A release that is missing, unfinished, altered, a draft without --allow-draft, or
    lacking what the replication needs."""


class SealedLeak(ReleaseError):
    """A test or sealed record asked for as training data (spec §5 guard)."""


def release_dir(release_id: str, root: Path | None = None) -> Path:
    from ... import config
    return Path(root) if root else config.DATA_DIR / RELEASES_DIR / release_id


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sums(path: Path) -> dict[str, str]:
    """`sha256sum -c` lines: '<hex>  <name>' (or '<hex> *<name>')."""
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, _, name = line.partition(" ")
        out[name.lstrip(" *").strip()] = digest.strip().lower()
    return out


def verify(folder: Path, files: tuple[str, ...] | None = None) -> dict:
    """`mv dataset verify` semantics for the files on disk: every file SHA256SUMS lists is
    re-hashed and must match it and MANIFEST.json's `files`. `files` None: every listed
    file must be present (a full release); else only those must be, and only those are
    checked (a trainer's shipped copy). Returns {file: sha256}."""
    if not (folder / MANIFEST).is_file():
        raise ReleaseError(f"{folder} has no {MANIFEST}: not a finished release")
    if not (folder / SUMS).is_file():
        raise ReleaseError(f"{folder} has no {SUMS}")
    manifest = json.loads((folder / MANIFEST).read_text(encoding="utf-8"))
    sums = read_sums(folder / SUMS)
    listed = manifest.get("files") or {}
    wanted = list(sums) if files is None else [f for f in files if f not in (SUMS,)]
    checked = {}
    for name in wanted:
        if name == MANIFEST and name not in sums:
            continue                       # MANIFEST.json is written last; its own hash is ours
        path = folder / name
        if not path.is_file():
            raise ReleaseError(f"{folder.name}: {name} is missing")
        if name not in sums:
            raise ReleaseError(f"{folder.name}: {name} is not in {SUMS}")
        got = file_sha256(path)
        if got != sums[name]:
            raise ReleaseError(f"{folder.name}: {name} does not match {SUMS} (altered)")
        want = (listed.get(name) or {}).get("sha256") if name != MANIFEST else got
        if want is not None and want != got:
            raise ReleaseError(f"{folder.name}: {name} does not match {MANIFEST}")
        checked[name] = got
    return checked


@dataclass
class Release:
    id: str
    path: Path
    manifest: dict
    manifest_sha256: str
    conn: sqlite3.Connection = field(repr=False)

    # --- identity ----------------------------------------------------------------------
    @property
    def draft(self) -> bool:
        return bool(self.manifest.get("draft"))

    def hash(self, name: str) -> str | None:
        hashes = self.manifest.get("hashes") or {}
        return hashes.get(name) or self.manifest.get(name)

    @property
    def labels_hash(self) -> str:
        h = self.hash("labels_hash")
        if not h:
            raise ReleaseError(f"release {self.id} has no labels_hash in {MANIFEST}")
        return h

    @property
    def allowed_sources(self) -> tuple[str, ...]:
        got = self.manifest.get("allowed_sources")
        if not got:
            raise ReleaseError(f"release {self.id} names no allowed_sources in {MANIFEST}: "
                               "the replication can't tell which record sources it may "
                               "train on")
        return tuple(got)

    def cite(self) -> dict:
        """What outputs carry (the reader's rel.cite())."""
        return {"dataset_release": self.id, "draft": self.draft,
                "manifest_sha256": self.manifest_sha256,
                "release_hash": self.hash("release_hash"), "labels_hash": self.hash("labels_hash"),
                "reference_hash": self.hash("reference_hash"),
                "code_commit": self.manifest.get("code_commit")}

    # --- contents ----------------------------------------------------------------------
    def splits(self) -> set[str]:
        return {r[0] for r in self.conn.execute(
            "select distinct split from records where included = 1 and split is not null")}

    def records(self, split: str) -> list[dict]:
        cur = self.conn.execute("select * from records where included = 1 and split = ? "
                                "order by record_key", (split,))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, row)) for row in cur]

    def training_records(self, include_val: bool = True) -> list[dict]:
        """train (and val): never test or sealed (spec §5)."""
        out = self.records("train") + (self.records("val") if include_val else [])
        never = {r["record_key"] for s in NEVER_TRAINED for r in self.records(s)}
        if any(r["record_key"] in never for r in out):
            raise SealedLeak(f"release {self.id}: a test or sealed record is in train/val")
        return out

    def photos(self, record_keys) -> dict[str, list[dict]]:
        keys = list(record_keys)
        out: dict[str, list[dict]] = {k: [] for k in keys}
        for at in range(0, len(keys), 500):
            part = keys[at:at + 500]
            cur = self.conn.execute(
                "select * from photos where included = 1 and record_key in "
                f"({','.join('?' * len(part))}) order by record_key, position", part)
            cols = [d[0] for d in cur.description]
            for row in cur:
                p = dict(zip(cols, row))
                out[p["record_key"]].append(p)
        return out

    def benchmark_records(self, splits=TRAINING_SPLITS) -> int:
        """Included records of `splits` that belong to a held-out benchmark."""
        return self.conn.execute(
            f"select count(*) from records where included = 1 and benchmark is not null and "
            f"split in ({','.join('?' * len(splits))})", tuple(splits)).fetchone()[0]

    def close(self) -> None:
        self.conn.close()


def photo_id(photo_key: str) -> int:
    """The manifest's photo id for a release photo key (the reader's photo_ids())."""
    source, _, n = photo_key.partition(":")
    if not n.isdigit():
        raise ReleaseError(f"not a photo key: {photo_key!r}")
    if source == "inat":
        return int(n)
    if source == "mo":
        return MO_PHOTO_ID_BASE + int(n)
    raise ReleaseError(f"photo {photo_key!r}: no photos are taken from source {source!r}")


def load(release_id: str, allow_draft: bool = False, root: Path | None = None,
         files: tuple[str, ...] | None = None,
         expect_manifest_sha256: str | None = None) -> Release:
    """Verify then open a release (spec §8 load_release). A draft needs `allow_draft`.
    `files`: verify only these (a trainer's shipped copy; see verify). With
    `expect_manifest_sha256`, refuse any other release than the one a launch recorded."""
    folder = release_dir(release_id, root)
    if not folder.is_dir():
        raise ReleaseError(f"no release {release_id!r} at {folder}")
    verify(folder, files)
    raw = (folder / MANIFEST).read_bytes()
    manifest = json.loads(raw)
    rid = manifest.get("id")
    if rid != release_id:
        raise ReleaseError(f"{folder} holds release {rid!r}, not {release_id!r}")
    if manifest.get("draft") and not allow_draft:
        raise ReleaseError(f"release {release_id} is a draft: pass --allow-draft to use it "
                           "(its results are not v1's)")
    sha = hashlib.sha256(raw).hexdigest()
    if expect_manifest_sha256 and sha != expect_manifest_sha256:
        raise ReleaseError(f"release {release_id}'s {MANIFEST} is not the one the launch "
                           f"recorded ({sha[:12]} vs {expect_manifest_sha256[:12]})")
    uri = (folder / DB).resolve().as_uri() + "?mode=ro&immutable=1"
    conn = sqlite3.connect(uri, uri=True)
    return Release(release_id, folder, manifest, sha, conn)
