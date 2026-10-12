"""What the public API says about the model answering (docs/api.md §4, `model`).

Steve, 2026-10-11: the public version reflects the TRAINING DATA, never the backbone. It is
the date the DNA-validated label set behind the served model was exported from
mycomap.org (the records' `exported_at` in the release's manifest: the reference set the
model compares with, and the labels any fine-tune in the release learned from), with
".2", ".3" for a later release built on the same export. `mv release` records that
version in release.json; a release made before it did falls back to the bare date.

`data` counts that snapshot (the release's own manifest, never what the nightly update
added since): records, species and genera, North America and no conflicting label, as
Vision uses them.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

from . import names

DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})")
VERSION = re.compile(r"^[A-Za-z0-9._:-]{1,40}$")
USED = "north_america = 1 and label_conflict = 0"


def snapshot_of(conn: sqlite3.Connection) -> str | None:
    """The date (YYYY-MM-DD) the manifest's records were exported, or None."""
    try:
        (at,) = conn.execute(f"select max(exported_at) from records where {USED}").fetchone()
    except sqlite3.OperationalError:
        return None
    m = DATE.match(at or "")
    return m[1] if m else None


def data_of(conn: sqlite3.Connection) -> dict:
    """{snapshot, records, species, genera} of a manifest, spellings of a name counted once
    (names.py), as the reference set uses them."""
    snapshot = snapshot_of(conn)
    try:
        records = conn.execute(f"select count(*) from records where {USED}").fetchone()[0]
        labels = names.manifest_labels(conn)
        species, genera = set(), set()
        for name, genus in conn.execute(f"select scientific_name, genus from records "
                                        f"where {USED} and coalesce(scientific_name, '') <> ''"):
            label = labels.get(name, name).strip()
            if len(label.split()) >= 2:
                species.add(label)
            if genus:
                genera.add(genus.strip())
    except sqlite3.OperationalError:
        return {"snapshot": snapshot, "records": 0, "species": 0, "genera": 0}
    return {"snapshot": snapshot, "records": int(records), "species": len(species),
            "genera": len(genera)}


def version_for(snapshot: str | None, earlier: list[str]) -> str | None:
    """The version of a new release on `snapshot`, given the versions of earlier releases:
    the date, then date.2, date.3 … for later builds on the same export."""
    if not snapshot:
        return None
    builds = [v for v in earlier if v == snapshot or v.startswith(snapshot + ".")]
    if not builds:
        return snapshot
    n = max(int(v.rpartition(".")[2]) if v != snapshot and v.rpartition(".")[2].isdigit()
            else 1 for v in builds)
    return f"{snapshot}.{n + 1}"


def card(release_dir: Path | None, manifest: Path | None) -> dict:
    """{version, data} for the served model: the release's own manifest and release.json
    when serving a release, else the given manifest (a laptop or test server)."""
    source = (release_dir / "manifest.sqlite") if release_dir else manifest
    data = {"snapshot": None, "records": 0, "species": 0, "genera": 0}
    if source and Path(source).is_file():
        conn = sqlite3.connect(f"file:{Path(source).as_posix()}?mode=ro", uri=True)
        try:
            data = data_of(conn)
        finally:
            conn.close()
    version = None
    if release_dir:
        try:
            version = json.loads((release_dir / "release.json").read_text(encoding="utf-8")
                                 ).get("version")
        except (OSError, ValueError):
            version = None
    if not (isinstance(version, str) and VERSION.match(version)):
        version = data["snapshot"]
    return {"version": version, "data": data}
