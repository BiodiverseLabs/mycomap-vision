"""A tiny hand-built dataset release, laid out as docs/dataset-release.md (feat/dataset-release)
specifies: release.sqlite (records, photos, splits, info), private.sqlite, MANIFEST.json
(written last) and SHA256SUMS. It mirrors seed_two_species: records 100-107, photos
inat:1000-1007 (their store copies' sha256 is 'h'). Not the builder's code: the builder
isn't merged, so the replication's tests make releases by hand."""

import hashlib
import json
import sqlite3
from pathlib import Path

RECORDS_SQL = """create table records (
  record_key text primary key, source text, source_id text, org_source text, org_name text,
  mycobank_number text, label text, label_rank text, species text, genus text, family text,
  name_kind text, label_rule text, label_exported_at text, validated_on text,
  green_projects text, observed_on text, country text, state text, north_america integer,
  included integer, reason text, split text, benchmark text, benchmark_split text)"""
PHOTOS_SQL = """create table photos (
  photo_key text primary key, record_key text, position integer, source text,
  source_photo_id text, source_url text, original_sha256 text, original_bytes integer,
  original_ext text, license_code text, license_class text, license_checked_at text,
  owner_login text, owner_name text, attribution text, included integer, reason text)"""

# record key -> (species, validated_on, split); seed_two_species' records 100-107
SEEDED = {"100": ("A x", "2026-01-01", "train"), "101": ("A x", "2026-01-02", "train"),
          "102": ("A x", "2026-01-03", "val"), "103": ("B y", "2026-01-01", "train"),
          "104": ("B y", "2026-01-02", "train"), "105": ("B y", "2026-01-03", "val"),
          "106": ("A x", "2026-09-20", "test"), "107": ("B y", "2026-09-21", "test")}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def make_release(data_dir: Path, release_id: str = "dry-20261009-1830", draft: bool = True,
                 allowed_sources=("inat", "mo"), splits: dict | None = None,
                 sources: dict | None = None, benchmark: dict | None = None,
                 extra_records: dict | None = None, labels_hash: str = "lab" + "0" * 61,
                 omit=()) -> Path:
    """Write the release under <data_dir>/research-releases/<id>/ and return its folder.
    `splits` / `sources` / `benchmark`: per record key overrides; `extra_records`: more
    records {key: (species, validated_on, split, source)} with no photos; `omit`: MANIFEST
    keys to leave out (e.g. 'allowed_sources')."""
    folder = data_dir / "research-releases" / release_id
    folder.mkdir(parents=True)
    db = sqlite3.connect(folder / "release.sqlite")
    db.execute(RECORDS_SQL)
    db.execute(PHOTOS_SQL)
    db.execute("create table splits (split text primary key, cutoff text, records integer)")
    db.execute("create table info (key text primary key, value text)")
    rows = {k: (sp, d, s, "inat") for k, (sp, d, s) in SEEDED.items()}
    rows.update(extra_records or {})
    for key, (sp, day, split, src) in rows.items():
        split = (splits or {}).get(key, split)
        src = (sources or {}).get(key, src)
        bench = (benchmark or {}).get(key)
        db.execute("insert into records values (" + ",".join("?" * 25) + ")", (
            key, src, key.split(":")[-1], src, sp, None, sp, "species" if sp else "genus",
            sp, (sp or "A").split()[0], "F", "formal" if sp else "one-word", "rule", "t", day,
            "[]", day, "US", "WA", 1, 1, "ok", split, bench, "dev" if bench else None))
    for i in range(8):
        key = str(100 + i)
        db.execute("insert into photos values (" + ",".join("?" * 17) + ")", (
            f"inat:{1000 + i}", key, 0, "inat", str(1000 + i), "u", "h", 1, "png", "cc0",
            "open" if i % 2 else "arr", "t", "o", "O", "a", 1, "ok"))
    db.commit()
    db.close()
    sqlite3.connect(folder / "private.sqlite").close()
    files = {}
    for f in ("release.sqlite", "private.sqlite"):
        b = (folder / f).read_bytes()
        files[f] = {"bytes": len(b), "sha256": sha(b)}
    manifest = {"id": release_id, "draft": draft, "created": "t", "code_commit": "abc",
                "allowed_sources": list(allowed_sources),
                "hashes": {"labels_hash": labels_hash, "release_hash": "rel" + "0" * 61,
                           "reference_hash": "ref" + "0" * 61},
                "files": files}
    for k in omit:
        manifest.pop(k, None)
    (folder / "MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (folder / "SHA256SUMS").write_text(
        "".join(f"{v['sha256']}  {k}\n" for k, v in files.items()), encoding="utf-8")
    return folder
