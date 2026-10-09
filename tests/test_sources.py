"""Record sources are explicit (sources.py): iNat, Mushroom Observer and the sources
Vision leaves out, and the migration of a manifest whose sources were guessed from ids."""

import json
import threading

import pytest
from conftest import inat_obs

from mycomap_vision import mo, photos, sources
from mycomap_vision.embed import SCHEMA as EMBED_SCHEMA
from mycomap_vision.embed import photos_to_embed
from mycomap_vision.evaluate import load_records
from mycomap_vision.identify import Identifier, PhotoInfo, Specimen
from mycomap_vision.inat import pending_ids, save_batch
from mycomap_vision.licenses import OPEN_DATA_HOST, photo_url
from mycomap_vision.permissions import PermissionView
from mycomap_vision.photos import HostGate, HostPolicy, Result, download_one, save_result
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

JPEG = b"\xff\xd8\xff\xe0" + b"x" * 100
MO_THUMB = "https://mushroomobserver.org/images/thumb/610775.jpg?1460360907"


def org_row(oid, source, name="Hydnum repandum", **kw):
    base = {"observation_id": str(oid), "source": source, "scientific_name": name,
            "genus": name.split()[0], "family": "Hydnaceae", "continent": "North America",
            "validation_status_1": "yes", "validation_project_1": "Indiana",
            "validation_date_1": "9/2/2026"}
    base.update(kw)
    return base


def mo_image(image_id, obs_ids, license="Creative Commons Wikipedia Compatible v3.0",
             ok_for_export=True, owner_id=123, login="mo_alice"):
    files = [f"https://mushroomobserver.org/images/{s}/{image_id}.jpg?1460360907"
             for s in ("thumb", "320", "640", "960", "1280", "orig")]
    return {"id": image_id, "type": "image", "copyright_holder": "Alice A", "license": license,
            "ok_for_export": ok_for_export, "files": files,
            "original_url": files[-1], "owner": {"id": owner_id, "login_name": login,
                                                 "legal_name": "Alice A"},
            "observation_ids": list(obs_ids)}


def export(conn, rows):
    return save_records(conn, build_records(rows, "t"))


def hold(conn, photo_id, size="large", store="local"):
    save_result(conn, Result(photo_id, "done", f"p/{photo_id}.jpg", 1, "h"), size, "now", store)


# ---------------------------------------------------------------------------
# The source comes from .org, and keys carry it

def test_the_source_is_orgs_column_never_the_look_of_the_id():
    recs = {r["observation_id"]: r for r in build_records([
        org_row("236438", "MO Observations"), org_row("1037122", "MycoPortal"),
        org_row("1037123", "MyCoPortal"), org_row("5001", "Sequences"),
        org_row("MK123", "GenBank Accessions"), org_row("104958644", "iNaturalist"),
        org_row("77", "Mushroom Observer")], "t")}
    assert {k: (r["source"], r["org_source"], r["source_id"]) for k, r in recs.items()} == {
        "mo:236438": ("mo", "MO Observations", "236438"),
        "mycoportal:1037122": ("mycoportal", "MycoPortal", "1037122"),
        "mycoportal:1037123": ("mycoportal", "MyCoPortal", "1037123"),
        "com_sequence:5001": ("com_sequence", "Sequences", "5001"),
        "genbank:MK123": ("genbank", "GenBank Accessions", "MK123"),
        "104958644": ("inat", "iNaturalist", "104958644"),
        "mo:77": ("mo", "Mushroom Observer", "77"),
    }


def test_a_record_without_a_source_org_names_is_unknown():
    [rec] = build_records([org_row("12", None)], "t")
    assert rec["source"] == "unknown" and rec["observation_id"] == "unknown:12"
    [rec] = build_records([org_row("12", "Somewhere new")], "t")
    assert rec["source"] == "unknown" and rec["org_source"] == "Somewhere new"


def test_an_inat_record_and_an_mo_record_with_the_same_number_stay_two_records(conn):
    export(conn, [org_row("500", "iNaturalist", "Russula emetica"),
                  org_row("500", "MO Observations", "Hydnum repandum")])
    got = dict(conn.execute("select observation_id, scientific_name from records"))
    assert got == {"500": "Russula emetica", "mo:500": "Hydnum repandum"}
    assert conn.execute("select sum(label_conflict) from records").fetchone()[0] == 0


# ---------------------------------------------------------------------------
# Fetching dispatches on the source

def test_a_numeric_mo_id_is_never_fetched_from_inat(conn):
    export(conn, [org_row("236438", "MO Observations"), org_row("104958644", "iNaturalist")])
    # Even a row stored under a bare number (as by hand or older code) is asked of iNat
    # only when its source says iNat.
    conn.execute("insert into records (observation_id, source, source_id, north_america, "
                 "exported_at) values ('999', 'mo', '999', 1, 't')")
    assert pending_ids(conn, refresh=True, north_america_only=True) == ["104958644"]


def test_mycoportal_and_sequences_records_never_enter_vision(conn, tmp_path):
    export(conn, [org_row("1037122", "MycoPortal"), org_row("5001", "Sequences"),
                  org_row("7", "GenBank Accessions"), org_row("104958644", "iNaturalist")])
    for key in ("mycoportal:1037122", "com_sequence:5001", "genbank:7"):
        # Photos linked to them anyway (an old fetch, a bug) are neither fetched,
        # downloaded, embedded nor used.
        conn.execute("insert into observation_photos values (?, 11, 0)", (key,))
    save_batch(conn, ["104958644"], [inat_obs(104958644, photos=[(0, 12, "cc0", OPEN_DATA_HOST)])],
               "t")
    conn.execute("insert into photos (photo_id, license_code, license_class, source_url, host, "
                 "first_seen_at, license_checked_at) values (11, 'cc0', 'open', "
                 "'https://inaturalist-open-data.s3.amazonaws.com/photos/11/square.jpg', ?, 't', 't')",
                 (OPEN_DATA_HOST,))
    assert pending_ids(conn, refresh=True, north_america_only=False) == ["104958644"]
    assert mo.pending(conn, refresh=True, north_america_only=False) == {}
    assert [r["photo_id"] for r in photos.pending_photos(conn, False, None, "large")] == [12]
    hold(conn, 11)
    hold(conn, 12)
    conn.executescript(EMBED_SCHEMA)
    assert [p for p, _ in photos_to_embed(conn, "b", "large", "local", "local", False)] == [12]
    assert [r.observation_id for r in load_records(conn, {11: 0, 12: 1}, False)] == ["104958644"]


def test_a_record_without_a_known_source_gets_no_photos(conn):
    export(conn, [org_row("12", None)])
    conn.execute("insert into observation_photos values ('unknown:12', 11, 0)")
    conn.execute("insert into photos (photo_id, license_code, license_class, source_url, "
                 "first_seen_at, license_checked_at) values (11, 'cc0', 'open', "
                 "'https://inaturalist-open-data.s3.amazonaws.com/photos/11/square.jpg', 't', 't')")
    assert photos.pending_photos(conn, False, None, "large") == []
    hold(conn, 11)
    conn.executescript(EMBED_SCHEMA)
    assert photos_to_embed(conn, "b", "large", "local", "local", False) == []
    assert load_records(conn, {11: 0}, False) == []


def test_an_inat_photo_linked_to_an_mo_record_is_never_downloaded_or_used(conn):
    export(conn, [org_row("5", "MO Observations")])
    save_batch(conn, [], [inat_obs(5, photos=[(0, 12, "cc0", OPEN_DATA_HOST)])], "t")
    conn.execute("insert into observation_photos values ('mo:5', 12, 0)")
    conn.execute("insert into mo_observations (observation_id, mo_id, status, fetched_at) "
                 "values ('mo:5', 5, 'ok', 't')")
    assert photos.pending_photos(conn, False, None, "large") == []
    assert load_records(conn, {12: 0}, False) == []


def test_a_new_manifest_starts_migrated_and_an_older_one_does_not(tmp_path):
    import sqlite3

    from mycomap_vision import manifest
    assert sources.migrated(manifest.connect(tmp_path / "new.sqlite"))
    old = tmp_path / "old.sqlite"
    sqlite3.connect(old).executescript(
        "create table records (observation_id text primary key, source text not null, "
        "north_america integer not null, exported_at text not null); "
        "insert into records values ('123', 'inat', 1, 'old');")
    assert not sources.migrated(manifest.connect(old))


def test_nothing_is_fetched_downloaded_or_embedded_before_the_manifest_is_migrated(conn):
    conn.executescript(EMBED_SCHEMA)
    conn.execute("delete from manifest_migrations")          # as a manifest older code made
    for call in (lambda: pending_ids(conn, False, True),
                 lambda: mo.pending(conn),
                 lambda: photos.pending_photos(conn, True, None),
                 lambda: photos_to_embed(conn, "b", "large", "local", "local")):
        with pytest.raises(sources.NotMigrated):
            call()
    export(conn, [])                       # an export (even an empty one) migrates
    assert sources.migrated(conn)
    assert pending_ids(conn, False, True) == []


# ---------------------------------------------------------------------------
# Migrating a manifest whose sources were guessed

def legacy_manifest(conn):
    """As older code left it: every numeric id stored as an iNat record, iNat details
    and photos fetched for each, embedded."""
    conn.executescript(EMBED_SCHEMA)
    for oid in ("123", "456", "789"):
        conn.execute("insert into records (observation_id, source, north_america, exported_at, "
                     "scientific_name) values (?, 'inat', 1, 'old', 'Old name')", (oid,))
    save_batch(conn, ["123", "456", "789"],
               [inat_obs(123, photos=[(0, 31, "cc0", OPEN_DATA_HOST), (1, 32, None, OPEN_DATA_HOST)]),
                inat_obs(456, photos=[(0, 41, "cc0", OPEN_DATA_HOST)]),
                inat_obs(789, photos=[(0, 51, "cc0", OPEN_DATA_HOST)])], "t")
    for pid in (31, 32, 41, 51):
        hold(conn, pid)
        conn.execute("insert into embeddings values ('ft', ?, 0, ?, 't')", (pid, pid))
    conn.execute("delete from manifest_migrations")


def test_the_first_export_takes_wrong_inat_photos_off_non_inat_records_with_an_audit_trail(conn):
    legacy_manifest(conn)
    assert not sources.migrated(conn)
    out = export(conn, [org_row("123", "MO Observations"), org_row("456", "iNaturalist"),
                        org_row("789", "iNaturalist"), org_row("789", "MO Observations")])
    assert out["source_migration"]["wrong_records"] == 1
    assert out["source_migration"]["photos_unlinked"] == 2
    assert out["source_migration"]["embeddings_removed"] == 2
    assert sources.migrated(conn)
    assert dict(conn.execute("select observation_id, source from records")) == {
        "mo:123": "mo", "456": "inat", "789": "inat", "mo:789": "mo"}
    # The MO record's iNat photos (iNat observation 123's) are off it, with their vectors.
    assert conn.execute("select count(*) from observation_photos where observation_id = '123'"
                        ).fetchone()[0] == 0
    assert conn.execute("select count(*) from inat_observations where observation_id = '123'"
                        ).fetchone()[0] == 0
    assert {r[0] for r in conn.execute("select photo_id from embeddings")} == {41, 51}
    # ...and each is written down: what .org says the record is, the photo as it was.
    audit = conn.execute("select observation_id, photo_id, org_source, new_key, photo_json, "
                         "copies_json, embeddings_json from source_removals order by photo_id"
                         ).fetchall()
    assert [(a[0], a[1], a[2], a[3]) for a in audit] == [
        ("123", 31, "MO Observations", "mo:123"), ("123", 32, "MO Observations", "mo:123")]
    assert json.loads(audit[0][4])["license_code"] == "cc0"
    assert json.loads(audit[0][5])[0]["path"] == "p/31.jpg"
    assert json.loads(audit[0][6])[0]["backbone"] == "ft"
    # A record green as both iNat and MO keeps its iNat photos.
    assert {r[0] for r in conn.execute(
        "select photo_id from observation_photos where observation_id = '789'")} == {51}


def test_the_migration_runs_once(conn):
    legacy_manifest(conn)
    rows = [org_row("123", "MO Observations"), org_row("456", "iNaturalist")]
    assert "source_migration" in export(conn, rows)
    assert "source_migration" not in export(conn, rows)
    assert conn.execute("select count(*) from source_removals").fetchone()[0] == 2
    assert conn.execute("select count(*) from manifest_migrations").fetchone()[0] == 1


# ---------------------------------------------------------------------------
# Mushroom Observer photos

def mo_manifest(conn, images, oid="236438"):
    export(conn, [org_row(oid, "MO Observations")])
    return mo.save_batch(conn, {int(oid): f"mo:{oid}"}, images, "t")


def test_mo_images_are_stored_as_mo_photos_with_their_owner_and_licence(conn):
    stats = mo_manifest(conn, [mo_image(610775, [236438]), mo_image(610776, [236438, 1]),
                               mo_image(7, [99])])
    assert stats == {"ok": 1, "missing": 0, "photos_new": 2, "license_changes": 0, "photos": 2}
    rows = conn.execute("select photo_id, source, source_photo_id, owner_user_id, "
                        "source_owner_id, owner_login, license_code, license_class, license_text, "
                        "ok_for_export from photos order by photo_id").fetchall()
    assert [tuple(r) for r in rows] == [
        (10**12 + 610775, "mo", 610775, None, 123, "mo_alice", "cc-by-sa", "open",
         "Creative Commons Wikipedia Compatible v3.0", 1),
        (10**12 + 610776, "mo", 610776, None, 123, "mo_alice", "cc-by-sa", "open",
         "Creative Commons Wikipedia Compatible v3.0", 1)]
    links = conn.execute("select observation_id, photo_id, position from observation_photos "
                         "order by position").fetchall()
    assert [tuple(r) for r in links] == [("mo:236438", 10**12 + 610775, 0),
                                         ("mo:236438", 10**12 + 610776, 1)]
    obs = conn.execute("select * from mo_observations").fetchone()
    assert (obs["status"], obs["owner_login"], obs["image_count"]) == ("ok", "mo_alice", 2)
    # No MO location is read or kept.
    cols = {r[1] for r in conn.execute("pragma table_info(mo_observations)")}
    assert not cols & {"latitude", "longitude", "lat", "lng", "location", "location_name"}


def test_an_mo_observation_with_no_images_is_missing_and_gets_none(conn):
    mo_manifest(conn, [])
    assert conn.execute("select status from mo_observations").fetchone()[0] == "missing"
    assert conn.execute("select count(*) from observation_photos").fetchone()[0] == 0


def test_an_mo_licence_we_cannot_read_counts_as_all_rights_reserved():
    assert mo.license_of("Creative Commons Non-commercial v3.0") == ("cc-by-nc-sa", "nc")
    assert mo.license_of("Public Domain") == ("pd", "open")
    assert mo.license_of("Copyright Reserved") == ("", "arr")
    assert mo.license_of("Some new licence") == ("", "arr")
    assert mo.license_of(None) == ("", "arr")


def test_mo_records_are_used_with_their_own_photos_and_orgs_name(conn):
    mo_manifest(conn, [mo_image(610775, [236438])])
    pid = 10**12 + 610775
    [rec] = load_records(conn, {pid: 0})
    assert (rec.observation_id, rec.source, rec.source_id) == ("mo:236438", "mo", "236438")
    assert rec.species == "Hydnum repandum"           # .org's name, never MO's consensus
    assert rec.observer == "mo:mo_alice"              # never the same-named iNat account
    assert rec.uuid == ""


@pytest.mark.parametrize("image", [
    mo_image(610775, [236438], license="Copyright Reserved"),
    mo_image(610775, [236438], ok_for_export=False),
], ids=["all-rights-reserved", "not-ok-for-export"])
def test_mo_photos_mo_marks_not_for_export_or_all_rights_reserved_are_not_used(conn, image):
    mo_manifest(conn, [image])
    assert load_records(conn, {10**12 + 610775: 0}) == []


def test_an_mo_owner_never_matches_an_inat_users_permission(conn):
    mo_manifest(conn, [mo_image(610775, [236438], owner_id=42)])
    # iNat user 42 withdrew; MO user 42 is someone else.
    conn.execute("insert into photo_permissions (inat_user_id, inat_login, status) "
                 "values (42, 'inat_42', 'withdrawn')")
    assert len(load_records(conn, {10**12 + 610775: 0})) == 1
    # ...and an iNat grant never shows an all-rights-reserved MO photo.
    view = PermissionView(granted=frozenset({42}), synced_at=None)
    owner = conn.execute("select owner_user_id from photos").fetchone()[0]
    assert owner is None and not view.may_show("arr", owner)


def test_mo_photo_urls_come_from_the_photos_source():
    assert photo_url("mo", MO_THUMB, "large") == \
        "https://mushroomobserver.org/images/960/610775.jpg"
    assert photo_url("mo", MO_THUMB, "medium") == \
        "https://mushroomobserver.org/images/640/610775.jpg"
    with pytest.raises(ValueError):
        photo_url("inat", MO_THUMB, "large")         # an MO URL is not an iNat photo
    with pytest.raises(ValueError):
        photo_url("mycoportal", MO_THUMB, "large")   # no photos from MyCoPortal


class Resp:
    def __init__(self, status, content=b"", location=None):
        self.status_code = status
        self.content = content
        self.headers = {"location": location} if location else {}
        self.is_redirect = status in (301, 302, 303, 307, 308) and bool(location)


class Session:
    def __init__(self, *resps):
        self.resps = list(resps)
        self.calls = []

    def get(self, url, timeout, allow_redirects=True):
        self.calls.append((url, allow_redirects))
        return self.resps.pop(0)


def gate():
    return HostGate(HostPolicy(concurrency=1, min_interval=0))


def test_an_mo_image_follows_one_redirect_to_an_mo_host_only(tmp_path):
    ok = Session(Resp(302, location="https://images.mushroomobserver.org/960/610775.jpg"),
                 Resp(200, JPEG))
    r = download_one(ok, gate(), 1, MO_THUMB, "large", LocalStore(tmp_path), threading.Event(), "mo")
    assert r.status == "done"
    assert ok.calls == [("https://mushroomobserver.org/images/960/610775.jpg", False),
                        ("https://images.mushroomobserver.org/960/610775.jpg", False)]
    for bad in ("https://evil.example/x.jpg", "http://images.mushroomobserver.org/x.jpg",
                "https://mushroomobserver.org.evil.example/x.jpg"):
        r = download_one(Session(Resp(302, location=bad)), gate(), 1, MO_THUMB, "large",
                         LocalStore(tmp_path), threading.Event(), "mo")
        assert r.status == "error" and r.error.startswith("redirect refused"), bad
    chain = Session(Resp(302, location="https://mushroomobserver.org/a.jpg"),
                    Resp(302, location="https://mushroomobserver.org/b.jpg"))
    r = download_one(chain, gate(), 1, MO_THUMB, "large", LocalStore(tmp_path),
                     threading.Event(), "mo")
    assert r.error == "redirect chain refused"


def test_mo_is_asked_no_more_than_once_every_five_seconds():
    assert mo.MIN_INTERVAL >= 5
    policy = photos.default_policies()["mushroomobserver.org"]
    assert policy.concurrency == 1 and policy.min_interval >= 5
    p = mo.Pacer()
    p.ran(8.0)                                   # a slow answer pushes the next one out
    assert p.limiter._next >= 8.0


def test_mo_fetch_sends_batches_and_keeps_each_answer(conn, tmp_path):
    export(conn, [org_row(n, "MO Observations") for n in (3, 1, 2)])
    seen = []

    def fake(session, ids, pacer):
        seen.append(list(ids))
        return [mo_image(100 + i, [i]) for i in ids]

    out = mo.fetch_all(conn, raw_dir=tmp_path, session=object(), pacer=mo.Pacer(0), fetch=fake,
                       log=lambda s: None)
    assert seen == [[1, 2, 3]]
    assert (out["ok"], out["photos"]) == (3, 3)
    assert len(list(tmp_path.glob("images-1-*.json.gz"))) == 1
    assert mo.pending(conn) == {}                 # nothing is asked twice


def test_a_shown_mo_match_links_to_mo_and_shows_mos_photo():
    rec = Specimen("mo:236438", "Hydnum repandum", "Hydnum", "Hydnaceae", "mo", "236438")
    info = PhotoInfo(MO_THUMB, "open", "mo_alice", None, "mo")
    [s] = Identifier._specimens([(0.9, 0, rec, [5], [0.9])], {5: info}, lambda i: True)
    assert s["record_url"] == "https://mushroomobserver.org/obs/236438"
    assert s["inat_url"] is None and s["source"] == "mo" and s["source_id"] == "236438"
    assert s["photo_url"] == "https://mushroomobserver.org/images/640/610775.jpg"
    inat = Specimen("104958644", "Russula emetica", "Russula", "Russulaceae", "inat", "104958644")
    [s] = Identifier._specimens([(0.9, 0, inat, [], [])], {}, lambda i: True)
    assert s["inat_url"] == s["record_url"] == "https://www.inaturalist.org/observations/104958644"


def test_a_record_of_a_source_vision_leaves_out_is_never_used_even_with_its_own_photos(conn):
    # Belt and braces: should a MyCoPortal photo ever be stored (it never is), its record
    # still stays out, because the record's source is not one Vision takes photos from.
    export(conn, [org_row("1037122", "MycoPortal")])
    conn.execute("insert into photos (photo_id, source, license_code, license_class, source_url, "
                 "first_seen_at, license_checked_at) values (11, 'mycoportal', 'cc0', 'open', "
                 "'https://example.org/11.jpg', 't', 't')")
    conn.execute("insert into observation_photos values ('mycoportal:1037122', 11, 0)")
    assert load_records(conn, {11: 0}, False) == []


def test_an_mo_record_mo_no_longer_answers_for_is_not_used(conn):
    mo_manifest(conn, [mo_image(610775, [236438])])
    conn.execute("update mo_observations set status = 'missing'")
    assert load_records(conn, {10**12 + 610775: 0}) == []


def test_a_photo_another_record_still_uses_keeps_its_vectors(conn):
    legacy_manifest(conn)
    conn.execute("insert into observation_photos values ('456', 31, 1)")   # 31 is also 456's
    out = export(conn, [org_row("123", "MO Observations"), org_row("456", "iNaturalist")])
    assert out["source_migration"]["embeddings_removed"] == 1               # only 32's
    assert {r[0] for r in conn.execute("select photo_id from embeddings")} == {31, 41, 51}
    kept = conn.execute("select embeddings_json from source_removals where photo_id = 31"
                        ).fetchone()[0]
    assert json.loads(kept) == []


def test_an_observation_mo_no_longer_has_gets_no_photos_and_the_rest_are_read(conn, tmp_path):
    export(conn, [org_row(n, "MO Observations") for n in (1, 2, 3)])
    asked = []

    def fake(session, ids, pacer):
        asked.append(list(ids))
        if 2 in ids:
            raise mo.MissingObservation(2)
        return [mo_image(100 + i, [i]) for i in ids]

    out = mo.fetch_all(conn, raw_dir=tmp_path, session=object(), pacer=mo.Pacer(0), fetch=fake,
                       log=lambda s: None)
    assert asked == [[1, 2, 3], [1, 3]]
    assert (out["ok"], out["missing"], out["gone_on_mo"]) == (2, 1, 1)
    assert conn.execute("select status from mo_observations where mo_id = 2").fetchone()[0] == "missing"


def test_mos_not_found_answer_is_read_as_a_missing_observation():
    class R:
        status_code = 200

        def json(self):
            return {"errors": [{"code": "API2::ObjectNotFoundByID", "fatal": "true",
                                "details": "Observation #122178 does not exist, or someone "
                                           "has deleted it."}]}

    class S:
        def get(self, url, params, timeout):
            return R()

    with pytest.raises(mo.MissingObservation) as e:
        mo.fetch_batch(S(), [122178, 5], mo.Pacer(0))
    assert e.value.mo_id == 122178
