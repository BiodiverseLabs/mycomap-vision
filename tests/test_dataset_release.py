"""Dataset releases (dataset_release.py, docs/dataset-release.md): a frozen snapshot that
cannot change, whose hashes say exactly what is in it, whose derived images can be made
again byte for byte, and whose public variant holds no all-rights-reserved photo and no
coordinates. Held-out and excluded records never reach training."""

import io
import json
import os
import sqlite3
import subprocess
import sys

import pytest
from PIL import Image

from mycomap_vision import dataset_release as dr
from mycomap_vision import evaluate, holdouts, manifest

LAT, LON = 45.123456, -122.654321          # record 1's true place: never public


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)


def jpeg(seed: int, size=(640, 480)) -> bytes:
    import numpy as np
    y, x = np.mgrid[0:size[1], 0:size[0]]
    img = Image.fromarray(np.stack([(x * 7 + seed * 31) % 256, (y * 5 + seed) % 256,
                                    (x + y + seed) % 256], axis=-1).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=95)
    return buf.getvalue()


def migrate(conn):
    """The record-sources-v1 schema (feat/record-sources-mo), added when this checkout's
    manifest.py predates it."""
    for table, cols in {"records": {"org_source": "text", "source_id": "text"},
                        "photos": {"source": "text not null default 'inat'",
                                   "source_photo_id": "integer", "source_owner_id": "integer"}
                        }.items():
        have = {r[1] for r in conn.execute(f"pragma table_info({table})")}
        for c, decl in cols.items():
            if c not in have:
                conn.execute(f"alter table {table} add column {c} {decl}")
    conn.executescript("""
      create table if not exists manifest_migrations (name text primary key, applied_at text
        not null, code_version text, detail text);
      create table if not exists mo_observations (observation_id text primary key,
        mo_id integer not null, status text not null, owner_id integer, owner_login text,
        owner_name text, image_count integer, fetched_at text not null);""")
    conn.execute("insert or ignore into manifest_migrations values "
                 "('record-sources-v1', 'now', 't', '{}')")


class Store:
    def __init__(self, root):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def put(self, pid, body):
        rel = f"photos/large/{pid % 1000:03d}/{pid}.jpg"
        (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
        (self.root / rel).write_bytes(body)
        return rel


def add_record(conn, key, name, *, source="inat", vdate="2026-01-01", na=1, conflict=0,
               status="ok", lat=None, lon=None):
    genus = name.split()[0] if name and name != "Unknown" else ""
    conn.execute(
        "insert into records (observation_id, source, org_source, source_id, scientific_name, "
        "genus, family, latitude, longitude, observed_on, country, north_america, "
        "green_projects, validated_on, label_conflict, exported_at) values "
        "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (key, source, source, key.split(":")[-1], name, genus, "Fam" if genus else "", lat, lon, "2025-09-01",
         "US" if na else "FR", na, '["p1"]', vdate, conflict, "2026-10-09T12:00:00"))
    if status is None:
        return
    if source == "inat":
        conn.execute("insert into inat_observations (observation_id, status, uuid, user_login, "
                     "fetched_at) values (?,?,?,?, 'now')", (key, status, f"uuid-{key}", "alice"))
    elif source == "mo":
        conn.execute("insert into mo_observations values (?, ?, ?, 9, 'bob', 'Bob', 1, 'now')",
                     (key, int(key.split(":")[1]), status))


def add_photo(conn, store, key, pid, lic="cc-by", lclass="open", *, owner=7, copy=True,
              source="inat", seed=None):
    conn.execute(
        "insert into photos (photo_id, source, source_photo_id, owner_user_id, owner_login, "
        "owner_name, license_code, license_class, attribution, source_url, first_seen_at, "
        "license_checked_at, status) values (?,?,?,?,?,?,?,?,?,?, 'now', 'now', 'done')",
        (pid, source, pid if source == "mo" else None, owner, "alice", "Alice", lic, lclass,
         "(c) Alice", f"https://example.org/photos/{pid}/square.jpg"))
    conn.execute("insert into observation_photos values (?, ?, 1)", (key, pid))
    if copy:
        body = jpeg(seed if seed is not None else pid)
        rel = store.put(pid, body)
        conn.execute("insert into photo_copies values (?, ?, 'large', ?, ?, ?, 'now')",
                     (pid, str(store.root), rel, len(body), dr.sha256_bytes(body)))


@pytest.fixture
def world(tmp_path):
    """A manifest with one record of every kind, its photos in a local store."""
    path = tmp_path / "data" / "manifest.sqlite"
    conn = manifest.connect(path)
    migrate(conn)
    store = Store(tmp_path / "store")
    add_record(conn, "1", "Russula emetica", lat=LAT, lon=LON)
    add_photo(conn, store, "1", 11)                                   # CC
    add_photo(conn, store, "1", 12, lic="", lclass="arr")             # ARR, no answer: used
    add_record(conn, "2", "Amanita muscaria", vdate="2026-09-30")
    add_photo(conn, store, "2", 21)
    add_record(conn, "3", "Boletus edulis", vdate="2026-10-01")
    add_photo(conn, store, "3", 31, lic="", lclass="arr")             # ARR only
    add_record(conn, "mo:4", "Cantharellus cibarius", source="mo", vdate="2026-03-01")
    add_photo(conn, store, "mo:4", 9_000_000_004, lic="cc-by-sa", source="mo")
    add_record(conn, "mycoportal:5", "Russula emetica", source="mycoportal")
    add_record(conn, "com_sequence:6", "Russula emetica", source="com_sequence")
    add_record(conn, "7", "Russula emetica", na=0)
    add_photo(conn, store, "7", 71)
    add_record(conn, "8", "Russula emetica", conflict=1)
    add_photo(conn, store, "8", 81)
    add_record(conn, "9", "Unknown")
    add_photo(conn, store, "9", 91)
    add_record(conn, "10", "Russula emetica")
    add_photo(conn, store, "10", 101, copy=False)                     # no kept original
    add_record(conn, "11", "Russula emetica")
    add_photo(conn, store, "11", 111, lic="", lclass="arr", owner=66) # owner withdrew
    conn.execute("insert into photo_permissions (inat_user_id, status) values (66, 'withdrawn')")
    add_record(conn, "12", "Russula emetica")
    add_photo(conn, store, "12", 121)
    holdouts.add(conn, "paper", ["12"])                               # sealed benchmark
    add_record(conn, "13", "Russula emetica", status="missing")
    add_photo(conn, store, "13", 131)
    add_record(conn, "mo:15", "Russula emetica", source="mo", status=None)   # not fetched yet
    add_record(conn, "mo:16", "Russula emetica", source="mo", na=0, status=None)  # outside NA
    conn.commit()
    conn.execute("pragma wal_checkpoint(truncate)")
    yield {"conn": conn, "path": path, "root": tmp_path / "releases", "store": store,
           "tmp": tmp_path}
    conn.close()


def build(world, name=None, **kw):
    kw.setdefault("store_reader", dr.default_store_reader())
    kw.setdefault("val_weeks", 8)
    return dr.build(name=name or dr.draft_id(), manifest_path=world["path"], root=world["root"],
                    log=lambda *a: None, **kw)


def reasons(world, rid):
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        return {r["record_key"]: (r["reason"], r["split"]) for r in rel.db.execute(
            "select record_key, reason, split from records")}
    finally:
        rel.close()


# --- inclusion -------------------------------------------------------------------------------

def test_every_candidate_record_gets_exactly_one_reason(world):
    m = build(world)
    got = reasons(world, m["id"])
    assert got == {
        "1": ("ok", "train"), "2": ("ok", "val"), "3": ("ok", "val"), "mo:4": ("ok", "train"),
        "mycoportal:5": ("source_mycoportal", None),
        "com_sequence:6": ("source_com_sequence", None),
        "7": ("not_north_america", None), "8": ("label_conflict", None),
        "9": ("no_label", None), "10": ("no_photos", None), "11": ("no_photos", None),
        "12": ("held_out", None), "13": ("missing_at_source", None),
        "mo:15": ("not_fetched", None), "mo:16": ("not_north_america", None)}
    assert m["counts"]["records"] == 15 and m["counts"]["records_included"] == 4


def test_photo_reasons_follow_the_permission_and_original_rules(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    got = dict(rel.db.execute("select photo_key, reason from photos"))
    rel.close()
    assert got["inat:12"] == "ok"                       # ARR with no answer: used for training
    assert got["inat:101"] == "no_original"
    assert got["inat:111"] == "permission_withdrawn"
    assert got["inat:71"] == "record_excluded"
    assert got["mo:9000000004"] == "ok"


def test_excluded_records_never_appear_in_train_val_or_test(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    try:
        out = {r["record_key"] for r in rel.db.execute("select * from records where included = 0")}
        assert all(r["split"] is None for r in rel.db.execute(
            "select split from records where included = 0"))
        in_splits = {r["record_key"] for r in rel.records(dr.SPLITS)}
        assert in_splits == {"1", "2", "3", "mo:4"} and not (in_splits & out)
        recs = {r.observation_id for r in rel.as_records(dr.SPLITS)}
        assert not (recs & out)
    finally:
        rel.close()


def test_a_review_list_excludes_with_its_name_and_a_stale_list_fails(world, tmp_path):
    lst = tmp_path / "loo-scan.tsv"
    lst.write_text("kind\tkey\treason\tnote\nrecord\t2\tmislabel\tchecked by Steve\n"
                   "photo\tinat:11\tnot a fungus\t\n", encoding="utf-8")
    m = build(world, review_files=[lst])
    got = reasons(world, m["id"])
    assert got["2"] == ("review:loo-scan", None)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    assert rel.db.execute("select reason from photos where photo_key = 'inat:11'").fetchone()[0] \
        == "review:loo-scan"
    rel.close()
    assert (world["root"] / m["id"] / "inputs" / "loo-scan.tsv").is_file()
    stale = tmp_path / "stale.tsv"
    stale.write_text("kind\tkey\treason\tnote\nrecord\t999\tgone\t\n", encoding="utf-8")
    with pytest.raises(dr.ReleaseError, match="not in the manifest"):
        build(world, review_files=[stale])


# --- immutability ----------------------------------------------------------------------------

def test_a_release_is_immutable(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    with pytest.raises(sqlite3.OperationalError):
        rel.db.execute("update records set label = 'Amanita phalloides'")
    with pytest.raises(sqlite3.OperationalError):
        rel.private.execute("delete from record_places")
    rel.close()
    for p in (world["root"] / m["id"]).rglob("*"):
        if p.is_file():
            assert not os.access(p, os.W_OK), p
    with pytest.raises(dr.ReleaseError, match="never rebuilt"):
        build(world, name=m["id"])


def test_verify_catches_an_altered_release(world):
    m = build(world)
    folder = world["root"] / m["id"]
    assert dr.verify(m["id"], world["root"], photos="all",
                     store_reader=dr.default_store_reader())["ok"]
    dr._writable(folder)
    c = sqlite3.connect(folder / dr.RELEASE_DB)
    c.execute("update records set label = 'Amanita phalloides' where record_key = '2'")
    c.commit()
    c.close()
    with pytest.raises(dr.ReleaseError, match="does not match"):
        dr.verify(m["id"], world["root"])


def test_verify_catches_an_original_that_changed_in_the_store(world):
    m = build(world)
    world["store"].put(11, jpeg(999))
    with pytest.raises(dr.ReleaseError, match="no longer matches"):
        dr.verify(m["id"], world["root"], photos="all", store_reader=dr.default_store_reader())


def test_an_unfinished_release_is_never_read(world):
    m = build(world)
    folder = world["root"] / m["id"]
    dr._writable(folder)
    (folder / dr.MANIFEST).unlink()
    with pytest.raises(dr.ReleaseError, match="unfinished"):
        dr.load_release(m["id"], world["root"], allow_draft=True)


def test_building_never_writes_to_the_manifest(world):
    before = dr.sha256_file(world["path"])
    build(world)
    assert dr.sha256_file(world["path"]) == before


def test_a_frozen_release_needs_the_freeze_trigger(world):
    with pytest.raises(dr.ReleaseError, match="freeze"):
        build(world, name="v1")
    with pytest.raises(dr.ReleaseError, match="every derived hash"):
        build(world, name="v1", freeze_approved="Steve 2026-10-20", derive_photos="1")
    m = build(world, name="v1", freeze_approved="Steve 2026-10-20 (test)")
    assert m["draft"] is False and m["freeze_approved"].startswith("Steve")


def test_a_draft_is_refused_for_research_unless_allowed(world):
    m = build(world)
    with pytest.raises(dr.ReleaseError, match="draft"):
        dr.load_release(m["id"], world["root"])


def test_a_manifest_without_explicit_sources_is_refused(world):
    world["conn"].execute("delete from manifest_migrations")
    world["conn"].commit()
    with pytest.raises(dr.ReleaseError, match="record-sources-v1"):
        build(world)


# --- hashes ----------------------------------------------------------------------------------

def test_the_same_inputs_give_the_same_release_hash(world):
    a = build(world, name="dry-20261009-120000")
    b = build(world, name="dry-20261009-120001")
    assert a["hashes"] == b["hashes"]


def _change_label(c):
    c.execute("update records set scientific_name = 'Amanita phalloides' where observation_id = '2'")


def _change_photo(c, world):
    body = jpeg(4242)
    rel = world["store"].put(21, body)
    c.execute("update photo_copies set sha256 = ?, bytes = ? where photo_id = 21",
              (dr.sha256_bytes(body), len(body)))


def _change_licence(c):
    c.execute("update photos set license_code = 'cc-by-nc', license_class = 'nc' where photo_id = 11")


def _add_record(c, world):
    add_record(c, "14", "Russula emetica")
    add_photo(c, world["store"], "14", 141)


@pytest.mark.parametrize("change, parts", [
    ("label", {"records", "labels", "reference"}),
    ("photo", {"photos", "derived", "reference"}),
    ("licence", {"photos"}),
    ("record", {"records", "labels", "photos", "splits", "derived", "private", "reference"}),
])
def test_the_release_hash_changes_when_anything_in_it_changes(world, change, parts):
    a = build(world, name="dry-20261009-120000")
    c = world["conn"]
    {"label": lambda: _change_label(c), "photo": lambda: _change_photo(c, world),
     "licence": lambda: _change_licence(c), "record": lambda: _add_record(c, world)}[change]()
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    b = build(world, name="dry-20261009-120001")
    assert a["hashes"]["release"] != b["hashes"]["release"]
    changed = {k for k in a["hashes"] if a["hashes"][k] != b["hashes"][k]} - {"release"}
    assert parts <= changed


def test_the_release_hash_changes_with_the_recipes_and_the_split(world):
    base = build(world, name="dry-20261009-120000")["hashes"]
    only_original = build(world, name="dry-20261009-120001", recipes=["original"])["hashes"]
    assert only_original["recipes"] != base["recipes"] and only_original["release"] != base["release"]
    other_cut = build(world, name="dry-20261009-120002", val_cutoff="2026-01-15")["hashes"]
    assert other_cut["splits"] != base["splits"] and other_cut["release"] != base["release"]


def test_a_float_in_a_hashed_column_is_refused():
    with pytest.raises(dr.ReleaseError, match="float"):
        dr.canonical_line({"lat": 45.1})


# --- derived images --------------------------------------------------------------------------

def test_a_derived_image_is_byte_identical_across_runs(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    try:
        reader = dr.default_store_reader()
        one = rel.derive("inat:11", "long500-q90", reader)
        two = rel.derive("inat:11", "long500-q90", reader)
        assert one.body == two.body and (one.width, one.height) == (500, 375)
        stored = rel.db.execute("select derived_sha256, pixels_sha256 from derived where "
                                "photo_key = 'inat:11' and recipe = 'long500-q90'").fetchone()
        assert (one.sha256, one.pixels_sha256) == tuple(stored)
        assert rel.derive("inat:11", "original", reader).sha256 == \
            rel.db.execute("select original_sha256 from photos where photo_key = 'inat:11'"
                           ).fetchone()[0]
    finally:
        rel.close()
    # ... and in a fresh process.
    code = ("import sys; from mycomap_vision import dataset_release as dr; "
            "from pathlib import Path; "
            f"r = dr.load_release({m['id']!r}, Path({str(world['root'])!r}), allow_draft=True); "
            "print(r.derive('inat:11', 'long500-q90').sha256)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)})
    assert out.stdout.strip() == one.sha256, out.stderr


def test_derive_refuses_an_image_that_does_not_match_the_release(world, monkeypatch):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    try:
        monkeypatch.setitem(dr.RECIPES, "long500-q90", dr._recipe("long500-q90", 500, 80))
        real = dr.derive_bytes
        monkeypatch.setattr(dr, "derive_bytes", lambda body, recipe: real(
            body, {**recipe, "quality": 80}))
        with pytest.raises(dr.ReleaseError, match="derived bytes differ"):
            rel.derive("inat:11", "long500-q90", dr.default_store_reader())
    finally:
        rel.close()


# --- the public variant ----------------------------------------------------------------------

def _all_bytes(folder):
    return b"".join(p.read_bytes() for p in folder.rglob("*") if p.is_file())


def test_arr_photos_never_reach_the_public_variant(world):
    m = build(world)
    pub = dr.public_variant(m["id"], world["root"], log=lambda *a: None)
    rel = dr.load_release(pub["id"], world["root"], allow_draft=True)
    try:
        assert {r[0] for r in rel.db.execute("select license_class from photos")} <= {"open", "nc"}
        keys = {r[0] for r in rel.db.execute("select photo_key from photos")}
        assert keys == {"inat:11", "inat:21", "mo:9000000004", "inat:71", "inat:81", "inat:91",
                        "inat:101", "inat:121", "inat:131"}
        assert {r[0] for r in rel.db.execute("select photo_key from derived")} <= keys
        assert dict(rel.db.execute("select record_key, reason from records"))["3"] == "no_cc_photos"
    finally:
        rel.close()
    base = dr.load_release(m["id"], world["root"], allow_draft=True)
    arr_shas = [r[0] for r in base.db.execute(
        "select original_sha256 from photos where license_class = 'arr' and original_sha256 "
        "is not null")] + [r[0] for r in base.db.execute(
            "select d.derived_sha256 from derived d join photos p using (photo_key) "
            "where p.license_class = 'arr'")]
    base.close()
    blob = _all_bytes(world["root"] / pub["id"])
    assert arr_shas and not any(s.encode() in blob for s in arr_shas)


def test_no_coordinates_in_any_public_output(world):
    m = build(world)
    pub = dr.public_variant(m["id"], world["root"], log=lambda *a: None)
    needles = [str(v).encode() for v in (LAT, LON, round(LAT * 1e6), round(LON * 1e6), "45.12", "-122.65")]
    # The release's own public-facing files, and every file of the public variant.
    own = b"".join((world["root"] / m["id"] / f).read_bytes()
                   for f in (dr.RELEASE_DB, dr.MANIFEST, dr.SUMS, dr.CARD))
    for blob in (own, _all_bytes(world["root"] / pub["id"])):
        assert not any(n in blob for n in needles)
    # ... while the private companion keeps them for the location priors.
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    assert tuple(rel.private.execute("select lat_micro, lon_micro from record_places where "
                                     "record_key = '1'").fetchone()) == (45123456, -122654321)
    assert next(r for r in rel.as_records() if r.observation_id == "1").latitude == LAT
    rel.close()


def test_check_public_refuses_a_variant_holding_an_arr_photo(world):
    m = build(world)
    pub = dr.public_variant(m["id"], world["root"], log=lambda *a: None)
    folder = world["root"] / pub["id"]
    dr._writable(folder)
    c = sqlite3.connect(folder / dr.RELEASE_DB)
    c.execute("update photos set license_class = 'arr' where photo_key = 'inat:11'")
    c.commit()
    c.close()
    with pytest.raises(dr.ReleaseError, match="non-CC"):
        dr.check_public(folder)


# --- training never sees held-out or sealed records -----------------------------------------

def test_training_records_are_train_and_the_reference_is_train_plus_val(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    try:
        assert {r["record_key"] for r in rel.training_records()} == {"1", "mo:4"}
        assert {r["record_key"] for r in rel.reference_records()} == {"1", "2", "3", "mo:4"}
    finally:
        rel.close()


def test_a_sealed_benchmark_record_never_enters_a_release_split(world):
    m = build(world)
    assert reasons(world, m["id"])["12"] == ("held_out", None)


def test_a_test_release_is_disjoint_from_its_base_and_never_trained_on(world, tmp_path):
    base = build(world, name="dry-20261009-120000")
    c = world["conn"]
    add_record(c, "20", "Russula emetica", vdate="2026-11-01")
    add_photo(c, world["store"], "20", 201)
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    ids = tmp_path / "test.csv"
    ids.write_text("observation_id\n20\n", encoding="utf-8")
    t = build(world, name="dry-20261009-130000-test", test_for=base["id"], test_ids=ids)
    assert t["test_for"] == base["id"]
    rel = dr.load_release(t["id"], world["root"], allow_draft=True)
    try:
        assert [r["record_key"] for r in rel.records("test")] == ["20"]
        with pytest.raises(dr.SealedLeak):
            rel.training_records()
    finally:
        rel.close()
    overlap = tmp_path / "overlap.csv"
    overlap.write_text("observation_id\n1\n", encoding="utf-8")
    with pytest.raises(dr.SealedLeak, match="in dry-20261009-120000 already"):
        build(world, name="dry-20261009-140000-test", test_for=base["id"], test_ids=overlap)
    add_record(c, "21", "Russula emetica", vdate="2026-11-02")
    add_photo(c, world["store"], "21", 211, seed=11)          # the same image as photo 11
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    dup = tmp_path / "dup.csv"
    dup.write_text("observation_id\n21\n", encoding="utf-8")
    with pytest.raises(dr.SealedLeak, match="byte-identical"):
        build(world, name="dry-20261009-150000-test", test_for=base["id"], test_ids=dup)


# --- experiments read the release, not the live manifest ------------------------------------

def test_load_records_with_a_release_ignores_later_manifest_changes(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    try:
        c = world["conn"]
        _change_label(c)
        c.commit()
        photo_row = {p: p for p in (11, 12, 21, 31, 9_000_000_004)}
        got = {r.observation_id: r.species for r in
               evaluate.load_records(c, photo_row, release=rel)}
        assert got == {"1": "Russula emetica", "2": "Amanita muscaria", "3": "Boletus edulis",
                       "mo:4": "Cantharellus cibarius"}
        live = {r.observation_id: r.species for r in evaluate.load_records(c, photo_row)}
        assert live.get("2") == "Amanita phalloides"
    finally:
        rel.close()


def test_a_release_refuses_vectors_of_a_photo_whose_original_changed(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    try:
        c = world["conn"]
        _change_photo(c, world)
        c.commit()
        with pytest.raises(dr.ReleaseError, match="another original"):
            evaluate.load_records(c, {11: 11, 21: 21}, release=rel)
    finally:
        rel.close()


def test_cite_names_the_release_and_its_hashes(world):
    m = build(world)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    cite = rel.cite()
    rel.close()
    assert cite["dataset_release"] == m["id"] and cite["draft"] is True
    assert cite["reference_hash"] == m["hashes"]["reference"]


def test_diff_lists_relabels_and_new_records(world):
    a = build(world, name="dry-20261009-120000")
    c = world["conn"]
    _change_label(c)
    _add_record(c, world)
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    b = build(world, name="dry-20261009-120001")
    d = dr.diff(a["id"], b["id"], world["root"])
    assert d["relabelled"]["examples"] == [("2", "Amanita muscaria", "Amanita phalloides")]
    assert d["records_added"]["examples"] == ["14"]
    assert "release" in d["hashes_changed"]


def test_the_cli_builds_a_dry_run_without_a_name(world, capsys):
    from mycomap_vision import cli
    cli.main(["dataset", "--root", str(world["root"]), "build", "--dry-run",
              "--manifest", str(world["path"]), "--derive", "none"])
    out = capsys.readouterr().out
    assert out.startswith("DRY RUN, not v1")
    assert json.loads(out[out.index("{"):])["draft"] is True


def test_rebuilding_an_existing_release_is_refused_before_any_work(world, monkeypatch):
    m = build(world)

    def no_work(*a, **k):
        raise AssertionError("the manifest was copied for a release that exists")
    monkeypatch.setattr(dr, "snapshot_manifest", no_work)
    with pytest.raises(dr.ReleaseError, match="never rebuilt"):
        build(world, name=m["id"])


def test_verify_recomputes_the_content_not_only_the_file_hashes(world):
    """An edit whose file hashes were rewritten to match is still caught."""
    m = build(world)
    folder = world["root"] / m["id"]
    dr._writable(folder)
    c = sqlite3.connect(folder / dr.RELEASE_DB)
    c.execute("update records set label = 'Amanita phalloides' where record_key = '2'")
    c.commit()
    c.close()
    man = json.loads((folder / dr.MANIFEST).read_text(encoding="utf-8"))
    for f in man["files"]:
        p = folder / f["path"]
        f["bytes"], f["sha256"] = p.stat().st_size, dr.sha256_file(p)
    (folder / dr.MANIFEST).write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(dr.ReleaseError, match="content hashes differ"):
        dr.verify(m["id"], world["root"])


def test_the_taxonomy_snapshot_is_an_input_of_the_release(world):
    """iNat's families per genus shape labels: the build reads the cache beside the
    manifest and records its hash, so a changed taxonomy shows in the release."""
    from mycomap_vision import taxonomy
    m = build(world)
    assert m["inputs"]["manifest"]["taxonomy_sha256"] is None
    cache = world["path"].parent / taxonomy.CACHE
    taxonomy.open_cache(cache).close()
    m2 = build(world)
    assert m2["inputs"]["manifest"]["taxonomy_sha256"] == dr.sha256_file(cache)


def test_a_review_list_may_name_an_inat_record_as_inat_id(world, tmp_path):
    """Review lanes write record keys as <source>:<id>; an iNat record's key is bare."""
    lst = tmp_path / "label-audit.tsv"
    lst.write_text("kind\tkey\treason\tnote\nrecord\tinat:2\twrong photos\t\n", encoding="utf-8")
    m = build(world, review_files=[lst], recipes=["original"])
    assert reasons(world, m["id"])["2"] == ("review:label-audit", None)
