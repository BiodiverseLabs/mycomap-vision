"""A linked observation leaves out every record of the same find, not just its own (twins.py,
2026-10-18): iNat 84404214 came back as its Mushroom Observer cross-post 444349, "100% alike"
on every photo, because only the iNat record was left out. Twins are found as the paper
finds them (errata.match_twins: MO <-> iNat, same label, a photo within TWIN_MAX_BITS dHash
bits, template hashes left out), by the sites' own link fields, and by the fetched photos'
dHashes."""

import io
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_api_v1_by_observation import app, by_number, inat_obs, mo_obs
from test_identify_from_link import by_link

from mycomap_vision import byobservation as bo, dhash, errata, twins
from mycomap_vision.identify import Identifier
from mycomap_vision.mo import photo_id_of


def write_table(path, groups=None, hashes=None, templates=()):
    """A twin table as `mv twins` writes it: finds as {record: group}, photo dHashes."""
    groups, hashes = groups or {}, hashes or {}
    ids = sorted(hashes)
    return twins.save({
        "meta": {"rule": twins.RULE},
        "photo_id": np.array(ids, dtype=np.int64),
        "dhash": np.array([hashes[p] for p in ids], dtype=np.uint64),
        "templates": np.array(sorted(templates), dtype=np.uint64),
        "group_record": np.array(list(groups), dtype=str),
        "group_id": np.array(list(groups.values()), dtype=np.int64)}, path)


def noisy_png(seed: int, red_mean: int = 20) -> bytes:
    """A photo whose dHash is not trivial (a flat colour hashes to 0), reddish as `red_mean`
    so the stand-in backbone puts it beside that species."""
    rng = np.random.default_rng(seed)
    red = rng.integers(red_mean - 8, red_mean + 9, size=(64, 72)).astype(np.uint8)
    px = np.stack([red, np.zeros_like(red), np.zeros_like(red)], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(px, "RGB").save(buf, "PNG")
    return buf.getvalue()


def specimens(body) -> set[str]:
    return {s["observation_id"] for s in body["specimens"]}


# --- the rule, as the paper applies it --------------------------------------------------

def manifest_of(records, photos):
    conn = sqlite3.connect(":memory:")
    conn.execute("create table records (observation_id text, scientific_name text, source text)")
    conn.execute("create table observation_photos (observation_id text, photo_id integer)")
    conn.executemany("insert into records values (?, ?, ?)", records)
    conn.executemany("insert into observation_photos values (?, ?)", photos)
    return conn


def test_the_twin_table_pairs_records_by_the_papers_rule():
    a, b, c = 0x0123456789ABCDEF, 0xFEDCBA9876543210, 0x00FF00FF00FF00FF
    conn = manifest_of(
        [("mo:1", "A x", "mo"), ("10", "A x", "inat"),       # one find: 2 bits apart
         ("mo:2", "A x", "mo"), ("20", "B y", "inat"),       # same picture, another label
         ("mo:3", "A x", "mo"), ("30", "A x", "inat")],      # 4 bits: another picture
        [("mo:1", photo_id_of(1)), ("10", 11), ("mo:2", photo_id_of(2)), ("20", 21),
         ("mo:3", photo_id_of(3)), ("30", 31)])
    keys = ["mo:1", "inat:11", "mo:2", "inat:21", "mo:3", "inat:31"]
    hashes = [a, a ^ 0b11, b, b, c, c ^ (0b1111 << 50)]
    table = twins.build(conn, keys, np.array(hashes, dtype=np.uint64))
    assert table["pairs"] == [("mo:1", "10")]
    # The same pairs errata.match_twins finds (one copy of the rule, not two).
    photos = [(k, r, lab, src, hh) for k, (r, lab, src), hh in zip(
        keys, [("mo:1", "A x", "mo"), ("10", "A x", "inat"), ("mo:2", "A x", "mo"),
               ("20", "B y", "inat"), ("mo:3", "A x", "mo"), ("30", "A x", "inat")], hashes)]
    assert sorted(errata.match_twins(photos)) == table["pairs"]
    assert dict(zip(table["group_record"].tolist(), table["group_id"].tolist())) == {
        "10": 0, "mo:1": 0}


def test_a_template_hash_never_makes_a_twin():
    h = 0x0F0F0F0F0F0F0F0F                  # a slip form on five records' photos
    conn = manifest_of([(f"mo:{i}", "A x", "mo") for i in range(3)]
                       + [(str(10 + i), "A x", "inat") for i in range(2)],
                       [(f"mo:{i}", photo_id_of(i)) for i in range(3)]
                       + [(str(10 + i), 10 + i) for i in range(2)])
    keys = [f"mo:{i}" for i in range(3)] + ["inat:10", "inat:11"]
    table = twins.build(conn, keys, np.full(5, h, dtype=np.uint64))
    assert table["pairs"] == [] and table["templates"].tolist() == [h]


def test_a_find_posted_three_times_is_one_group():
    h = 0x1122334455667788
    conn = manifest_of([("mo:1", "A x", "mo"), ("10", "A x", "inat"), ("11", "A x", "inat")],
                       [("mo:1", photo_id_of(1)), ("10", 10), ("11", 11)])
    table = twins.build(conn, ["mo:1", "inat:10", "inat:11"], np.full(3, h, dtype=np.uint64))
    assert len(set(table["group_id"].tolist())) == 1 and len(table["group_id"]) == 3


# --- leaving a find out -----------------------------------------------------------------

def fake_identifier(tmp_path, groups=None, hashes=None, templates=()):
    """Columns: iNat 100 (photos 1, 2) | MO 900 (photos 3, 4) | iNat 200 (photo 5)."""
    ident = Identifier.__new__(Identifier)
    ident.col_record = SimpleNamespace(source=np.array(["inat", "mo", "inat"]),
                                       source_id=np.array(["100", "900", "200"]),
                                       obs=np.array(["100", "mo:900", "200"]),
                                       rec=np.array([0, 0, 1, 1, 2]))
    ident.col_photo = np.array([1, 2, 3, 4, 5])
    ident.twins = (twins.ReferenceTwins(write_table(tmp_path / twins.FILE, groups, hashes,
                                                    templates),
                                        ident.col_photo, ident.col_record)
                   if groups is not None or hashes is not None else None)
    return ident


def test_a_reference_records_cross_post_twin_is_left_out_with_it(tmp_path):
    ident = fake_identifier(tmp_path, groups={"100": 0, "mo:900": 0})
    hidden, own = ident.left_out("inat", "100")
    assert own.tolist() == [0, 1] and hidden.tolist() == [0, 1, 2, 3]
    hidden, own = ident.left_out("mo", "900")
    assert own.tolist() == [2, 3] and hidden.tolist() == [0, 1, 2, 3]
    hidden, own = ident.left_out("inat", "200")                    # no twin: itself only
    assert hidden.tolist() == [4]


def test_without_a_twin_table_the_twin_stays(tmp_path):
    ident = fake_identifier(tmp_path)
    hidden, _ = ident.left_out("inat", "100")
    assert hidden.tolist() == [0, 1]


def test_the_record_a_link_field_names_is_left_out(tmp_path):
    ident = fake_identifier(tmp_path)
    hidden, own = ident.left_out("inat", "555", linked=[("mo", "900")])
    assert own.tolist() == [] and hidden.tolist() == [2, 3]


def test_a_reference_photo_of_the_same_picture_leaves_its_record_out(tmp_path):
    h = 0x00FF00FF00FF00FF
    ident = fake_identifier(tmp_path, hashes={3: h, 5: h ^ (0xFF << 40)})
    hidden, _ = ident.left_out("inat", "555", photo_hashes=[h ^ 0b111])     # 3 bits: same
    assert hidden.tolist() == [2, 3]                 # the whole record, not just photo 3
    hidden, _ = ident.left_out("inat", "555", photo_hashes=[h ^ (0b1111 << 30)])
    assert hidden.tolist() == []                     # 4 and 12 bits off: other pictures


def test_a_reference_records_own_photos_find_a_twin_of_another_label(tmp_path):
    h = 0x7777000077770000
    ident = fake_identifier(tmp_path, hashes={1: h, 3: h ^ 1})
    hidden, _ = ident.left_out("inat", "100")
    assert hidden.tolist() == [0, 1, 2, 3]


def test_a_template_photo_leaves_nothing_else_out(tmp_path):
    h = 0x0F0F0F0F0F0F0F0F
    ident = fake_identifier(tmp_path, hashes={1: h, 3: h, 5: h}, templates=[h])
    assert ident.left_out("inat", "100")[0].tolist() == [0, 1]
    assert ident.left_out("inat", "555", photo_hashes=[h])[0].tolist() == []


# --- the sites' link fields -------------------------------------------------------------

@pytest.mark.parametrize("value,linked", [
    ("http://mushroomobserver.org/observer/show_observation/444349", [("mo", "444349")]),
    ("https://mushroomobserver.org/444349", [("mo", "444349")]),
    ("https://mushroomobserver.org/observations/444349/", [("mo", "444349")]),
    ("https://example.org/444349", []), ("444349", []),
])
def test_inats_mushroom_observer_url_field_links_the_mo_record(value, linked):
    obs = inat_obs(84404214, [5])
    obs["ofvs"] = [{"name": "Mushroom Observer URL", "value": value},
                   {"name": "DNA Barcode ITS", "value": "ACGT"}]
    assert bo.parse_inat(obs).linked == linked


def test_mos_external_link_to_inat_links_the_inat_record():
    obs = mo_obs(444349, 1297828)
    obs["external_links"] = [
        {"id": 1, "url": "https://mycoportal.org/portal/collections/individual/index.php?occid=1"},
        {"id": 2, "url": "https://www.inaturalist.org/observations/84404214"}]
    assert bo.parse_mo(obs).linked == [("inat", "84404214")]


def test_inat_is_asked_for_its_observation_fields():
    assert "ofvs:(name:!t,value:!t)" in bo.INAT_FIELDS


# --- through the API --------------------------------------------------------------------

def test_a_reference_record_is_answered_without_its_cross_post_twin(conn, tmp_path):
    write_table(tmp_path / twins.FILE, groups={"100": 0, "106": 0})    # one "A x" find
    client, _ = app(conn, tmp_path)
    body = by_number(client, "inat", 100).json()
    assert body["excluded_reference"] == {"source": "inat", "source_id": "100"}
    assert body["left_out_photos"] == 2
    assert not {"100", "106"} & specimens(body)
    assert body["ranks"]["species"][0]["name"] == "A x"     # the other A records still match


def test_a_link_leaves_out_the_record_its_mo_link_names(conn, tmp_path):
    client, site = app(conn, tmp_path)
    obs = mo_obs(778, 5003)
    obs["external_links"] = [{"id": 2, "url": "https://www.inaturalist.org/observations/103"}]
    site.mo[778] = obs
    site.photos[5003] = site.photos[5002]
    for body in (by_number(client, "mo", 778).json(),
                 by_link(client, "mo", 778).json()["results"][0]):
        assert "103" not in specimens(body)
    about = by_link(client, "mo", 778).json()["observation"]
    assert about["left_out_photos"] == 1


def test_a_link_whose_photo_is_a_reference_records_picture_leaves_that_record_out(
        conn, tmp_path):
    photo = noisy_png(1, red_mean=20)                  # beside "B y" record 104 (red 20)
    write_table(tmp_path / twins.FILE, hashes={1004: dhash.dhash64(photo) ^ 0b101})
    client, site = app(conn, tmp_path)
    site.inat[560] = inat_obs(560, [9101])
    site.photos[9101] = photo
    v1 = by_number(client, "inat", 560).json()
    assert "104" not in specimens(v1) and v1["left_out_photos"] == 1
    assert v1["context_used"]["place_from"] is None        # 104 is part of the place prior
    linked = by_link(client, "inat", 560).json()
    assert "104" not in specimens(linked["results"][0])
    assert linked["observation"]["left_out_photos"] == 1


def test_a_different_picture_leaves_nothing_out(conn, tmp_path):
    photo = noisy_png(1, red_mean=20)
    write_table(tmp_path / twins.FILE, hashes={1004: dhash.dhash64(photo) ^ 0b1111})
    client, site = app(conn, tmp_path)
    site.inat[560] = inat_obs(560, [9101])
    site.photos[9101] = photo
    body = by_number(client, "inat", 560).json()
    assert body["left_out_photos"] == 0 and "104" in specimens(body)


def test_answers_kept_before_a_twin_table_are_not_served_after_it(conn, tmp_path):
    client, _ = app(conn, tmp_path)
    assert "106" in specimens(by_number(client, "inat", 100).json())
    assert by_number(client, "inat", 100).json()["cached"] is True
    write_table(tmp_path / twins.FILE, groups={"100": 0, "106": 0})
    after = by_number(client, "inat", 100).json()
    assert after["cached"] is False and "106" not in specimens(after)


# --- making and shipping the table ------------------------------------------------------

def test_a_release_ships_the_twin_table_when_there_is_one(conn, tmp_path):
    from test_serving import big_reference

    from mycomap_vision import release
    big_reference(conn, tmp_path / "embeddings", records=10, photos_each=1, shard_rows=20)
    shipped = {rel for _p, rel in release.release_files(conn, ["big"], tmp_path)}
    assert twins.FILE not in shipped
    write_table(tmp_path / twins.FILE, groups={"1": 0, "mo:1": 0})
    shipped = {rel for _p, rel in release.release_files(conn, ["big"], tmp_path)}
    assert twins.FILE in shipped


def test_mv_twins_writes_the_table_from_a_dhash_index(conn, tmp_path, capsys):
    import json

    from mycomap_vision import cli
    conn.executemany("insert into records (observation_id, source, source_id, scientific_name, "
                     "north_america, exported_at) values (?, ?, ?, 'A x', 1, 'now')",
                     [("10", "inat", "10"), ("mo:1", "mo", "1")])
    conn.executemany("insert into observation_photos (observation_id, photo_id) values (?, ?)",
                     [("10", 11), ("mo:1", photo_id_of(1))])
    index = tmp_path / "dhash-index.npz"
    dhash.save_index({"release": "t", "release_hash": "x", "algorithm": dhash.ALGORITHM,
                      "pillow": "p", "photos": 2, "photo_key": ["inat:11", "mo:1"],
                      "sha256": ["a", "b"], "dhash": np.array([7, 7], dtype=np.uint64)}, index)
    out = tmp_path / "out" / twins.FILE
    cli.cmd_twins(conn, SimpleNamespace(dhash_index=str(index), out=str(out)))
    said = json.loads(capsys.readouterr().out)
    assert (said["pairs"], said["records"], said["finds"]) == (1, 2, 1)
    assert twins.table_counts(out)["pairs"] == 1
