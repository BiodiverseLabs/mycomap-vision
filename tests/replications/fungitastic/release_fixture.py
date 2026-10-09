"""Small dataset releases for the replication's tests, made by the builder itself
(dataset_release.build, feat/dataset-release) from a tiny migrated manifest:

- base release (a draft `dry-...`): train = records 1-4 and mo:9 (validated Jan 2026),
  val = records 5-6 (after the 2026-02-01 cutoff); species "A x" (red photos) and "B y"
  (dark photos), as the published-model tests' fake classifier reads them;
- its test release (`dry-...-test`): records 7-8, fresh, disjoint by record and bytes.
"""

import io
from pathlib import Path

from PIL import Image
from test_dataset_release import add_record, migrate

from mycomap_vision import config, manifest
from mycomap_vision import dataset_release as dr
from mycomap_vision.storage import LocalStore

BASE = "dry-20261009-120000"
TEST = "dry-20261009-130000-test"
VAL_CUTOFF = "2026-02-01"
# key -> (name, validated on, red level); A x is red, B y dark (bright_is_a reads red)
TRAIN = {"1": ("A x", "2026-01-01", 250), "2": ("A x", "2026-01-02", 240),
         "3": ("B y", "2026-01-01", 10), "4": ("B y", "2026-01-02", 20),
         "mo:9": ("A x", "2026-01-03", 230)}
VAL = {"5": ("A x", "2026-03-01", 245), "6": ("B y", "2026-03-02", 15)}
FRESH = {"7": ("A x", "2026-11-01", 248), "8": ("B y", "2026-11-02", 12)}


def photo_id(key: str) -> int:
    return 9_000_000_000 + int(key.split(":")[1]) if key.startswith("mo:") else 1000 + int(key)


def add_photo(conn, store: LocalStore, key: str, red: int) -> int:
    """One solid-colour JPEG for the record, kept as its large original."""
    pid = photo_id(key)
    source = "mo" if key.startswith("mo:") else "inat"
    buf = io.BytesIO()
    Image.new("RGB", (16, 16), (red, pid % 7, 0)).save(buf, "JPEG", quality=95)
    body = buf.getvalue()
    rel = f"photos/large/{pid % 1000:03d}/{pid}.jpg"
    store.put(rel, body)
    lclass = "arr" if pid % 2 else "open"
    conn.execute(
        "insert into photos (photo_id, source, source_photo_id, owner_user_id, owner_login, "
        "owner_name, license_code, license_class, attribution, source_url, first_seen_at, "
        "license_checked_at, status) values (?,?,?,?,?,?,?,?,?,?, 'now', 'now', 'done')",
        (pid, source, pid if source == "mo" else None, 7, "alice", "Alice",
         "" if lclass == "arr" else "cc-by", lclass, "(c) Alice", f"https://example.org/{pid}"))
    conn.execute("insert into observation_photos values (?, ?, 1)", (key, pid))
    conn.execute("insert into photo_copies values (?, ?, 'large', ?, ?, ?, 'now')",
                 (pid, store.location, rel, len(body), dr.sha256_bytes(body)))
    return pid


def add(conn, store, records: dict) -> None:
    for key, (name, day, red) in records.items():
        add_record(conn, key, name, source="mo" if key.startswith("mo:") else "inat",
                   vdate=day, lat=47.0, lon=-122.0)
        add_photo(conn, store, key, red)
    conn.commit()
    conn.execute("pragma wal_checkpoint(truncate)")


def release_world(tmp_path: Path, monkeypatch, test_release: bool = True) -> dict:
    """The manifest (config.DATA_DIR = <tmp>/data), its store, and the releases built from
    it under <tmp>/data/research-releases."""
    data = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", data)
    path = data / "manifest.sqlite"
    conn = manifest.connect(path)
    migrate(conn)
    store = LocalStore(tmp_path / "store")
    add(conn, store, {**TRAIN, **VAL})
    root = data / "research-releases"
    quiet = {"log": lambda *a: None, "derive_photos": "none", "recipes": ("original",)}
    dr.build(name=BASE, manifest_path=path, root=root, val_cutoff=VAL_CUTOFF, **quiet)
    if test_release:
        add(conn, store, FRESH)
        ids = tmp_path / "test-ids.csv"
        ids.write_text("observation_id\n" + "\n".join(FRESH) + "\n", encoding="utf-8")
        dr.build(name=TEST, manifest_path=path, root=root, test_for=BASE, test_ids=ids, **quiet)
    return {"conn": conn, "store": store, "root": root, "data": data, "path": path}
