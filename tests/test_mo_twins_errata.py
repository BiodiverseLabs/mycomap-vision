"""Mushroom Observer finds held by mycomap.org as iNat records (2026-10-13): 71 iNaturalist
rows share a number AND a validation project with an 'MO Observations' row; their name and
photos are the unrelated iNat observation with that number (a duck, a panda...).

- The feed guard (sources.mo_twins, records.save_records, nightly.changes) never stores
  them, removes one stored before so the next nightly drops it from the served index, and
  keeps an iNat and an MO record numbered alike in different projects (two real finds).
- v1 was cut with 41 of them included: an erratum (errata.py) leaves them out of every
  split a command reads (`--errata`), refuses keys that are not the release's, and puts
  its sha256 into the release's cite(), so into every report made with it."""

import argparse
import json

import pytest
from test_dataset_release import add_photo, add_record, build, world  # noqa: F401
from test_nightly import box, night, row, seed_rows  # noqa: F401

from mycomap_vision import cli, errata, manifest, nightly, sources
from mycomap_vision import dataset_release as dr
from mycomap_vision.records import build_records, save_records


def org(oid, source, project, name="Hydnum repandum"):
    return {"observation_id": str(oid), "source": source, "scientific_name": name,
            "genus": name.split()[0], "continent": "North America",
            "validation_status_1": "yes", "validation_project_1": project,
            "validation_date_1": "10/12/2023"}


def keys(conn):
    return {r[0] for r in conn.execute("select observation_id from records")}


# --- the feed guard ------------------------------------------------------------------------

def test_an_inat_record_with_an_mo_twin_in_the_same_project_is_a_disguised_mo_find():
    recs = build_records([org(457150, "iNaturalist", "OMDL2", "Spatula cyanoptera"),
                          org(457150, "MO Observations", "OMDL2", "Amanita sp."),
                          org(500, "iNaturalist", "P1"), org(500, "MO Observations", "P2"),
                          org(600, "iNaturalist", "OMDL2")], "t")
    assert sources.mo_twins(recs) == {"457150"}
    kept, dropped = sources.drop_mo_twins(recs)
    assert dropped == ["457150"]
    assert {r["observation_id"] for r in kept} == {"mo:457150", "500", "mo:500", "600"}


def test_the_export_never_stores_one_and_removes_one_stored_before(conn):
    duck = org(457150, "iNaturalist", "OMDL2", "Spatula cyanoptera")
    save_records(conn, build_records([duck], "t1"))              # the twin not exported yet
    assert "457150" in keys(conn)
    out = save_records(conn, build_records([duck, org(457150, "MO Observations", "OMDL2"),
                                            org(500, "iNaturalist", "P1"),
                                            org(500, "MO Observations", "P2")], "t2"))
    assert out["mo_twins_dropped"] == 1 and out["removed"] == 1
    assert keys(conn) == {"mo:457150", "500", "mo:500"}           # different projects: both


def test_the_next_nightly_drops_a_stored_twin_and_rebuilds_the_served_index(box):
    layer = nightly.prepare(box, log=lambda s: None)
    rows = seed_rows()
    rows[-1] = {**rows[-1], "validation_project_1": "OMDL2"}      # record 107, embedded
    assert night(layer, rows)["ok"]
    conn = manifest.connect(layer.manifest)
    try:
        before = nightly.layer_version(conn)
    finally:
        conn.close()
    twin = {**row(107, "Pluteus chrysaegis"), "source": "MO Observations",
            "validation_project_1": "OMDL2"}
    report = night(layer, rows + [twin])
    assert report["ok"] and report["changed"]
    assert report["records"]["mo_twins_dropped"] == 1 and report["records"]["removed"] == 1
    conn = manifest.connect(layer.manifest)
    try:
        assert "107" not in keys(conn) and "mo:107" in keys(conn)
        assert nightly.layer_version(conn) > before               # the index is rebuilt
    finally:
        conn.close()


# --- the v1 erratum --------------------------------------------------------------------------

@pytest.fixture
def twinned(world):
    """A release whose train split holds two MO finds twice: as mo:77 and as iNat 77 (a
    duck), and as mo:78 and iNat 78, whose name a genuine record (1) also has."""
    c = world["conn"]
    for key, source, name in (("77", "inat", "Anas platyrhynchos"),
                              ("mo:77", "mo", "Cantharellus cibarius"),
                              ("78", "inat", "Russula emetica"),
                              ("mo:78", "mo", "Russula emetica")):
        add_record(c, key, name, source=source, vdate="2026-02-01")
        c.execute("update records set green_projects = '[\"OMDL2\"]' where observation_id = ?",
                  (key,))
    add_photo(c, world["store"], "77", 771)
    add_photo(c, world["store"], "mo:77", 9_000_000_077, source="mo")
    add_photo(c, world["store"], "78", 781)
    add_photo(c, world["store"], "mo:78", 9_000_000_078, source="mo")
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    m = build(world, name="dry-20261013-150000")
    return world, m["id"]


def erratum(world, rid, name="errata.tsv"):
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        rows = errata.mo_twin_rows(rel)
    finally:
        rel.close()
    return errata.write(rows, world["tmp"] / name)


def test_the_erratum_lists_the_disguised_find_by_the_rule_with_its_twin(twinned):
    world, rid = twinned
    path = erratum(world, rid)
    text = path.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "kind\tkey\treason\tnote"
    assert "record\t77\tmo_twin_as_inat\ttwin mo:77; projects OMDL2" in text
    e = errata.read(path)
    assert e.keys == {"77", "78"} and e.sha256 == dr.sha256_file(path)
    with pytest.raises(dr.ReleaseError, match="never rewritten"):
        erratum(world, rid)
    again = erratum(world, rid, name="again.tsv")                 # the same rows, same bytes
    assert dr.sha256_file(again) == e.sha256


def test_an_erratum_leaves_its_records_out_of_every_split_and_is_cited(twinned):
    world, rid = twinned
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        assert "77" in {r["record_key"] for r in rel.training_records()}
        photos_before = set(rel.photo_ids())
        rel.use_errata(errata.read(erratum(world, rid)))
        assert "77" not in {r["record_key"] for r in rel.training_records(include_val=True)}
        assert "77" not in {r.observation_id for r in rel.as_records()}
        assert "mo:77" in {r["record_key"] for r in rel.training_records()}   # the real find
        assert photos_before - set(rel.photo_ids()) == {771, 781}
        cite = rel.cite()
        assert cite["errata"]["sha256"] == dr.sha256_file(world["tmp"] / "errata.tsv")
        assert cite["errata"]["records"] == 2 and cite["release_hash"] == rel.release_hash
        # A label only erratum records taught the fine-tune can no longer be answered; one a
        # genuine record also has stays.
        assert errata.masked_head_labels(rel, rel.errata, [
            "Anas platyrhynchos", "Cantharellus cibarius", "Russula emetica"]) == \
            ["Anas platyrhynchos"]
    finally:
        rel.close()


def test_an_erratum_for_another_release_or_in_another_format_is_refused(twinned, tmp_path):
    world, rid = twinned
    stranger = tmp_path / "other.tsv"
    stranger.write_text("kind\tkey\treason\tnote\nrecord\t999999\tmo_twin_as_inat\tx\n",
                        encoding="utf-8")
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        with pytest.raises(dr.ReleaseError, match="not in dry-20261013-150000"):
            rel.use_errata(errata.read(stranger))
        assert rel.errata is None
    finally:
        rel.close()
    for body, match in (("kind\tkey\treason\tnote\nphoto\tinat:1\tmo_twin_as_inat\tx\n",
                         "lists records"),
                        ("kind\tkey\treason\tnote\nrecord\t77\tlooks odd\tx\n", "unknown reason"),
                        ("key\treason\n77\tmo_twin_as_inat\n", "columns"),
                        ("kind\tkey\treason\tnote\n", "lists no records")):
        bad = tmp_path / "bad.tsv"
        bad.write_text(body, encoding="utf-8")
        with pytest.raises(dr.ReleaseError, match=match):
            errata.read(bad)


def test_the_cli_applies_the_erratum_and_the_aws_launchers_do_not_take_it(twinned,
                                                                          monkeypatch, capsys):
    world, rid = twinned
    path = erratum(world, rid)
    monkeypatch.setattr(dr, "ROOT", world["root"])
    rel = cli._dataset_release(argparse.Namespace(release=rid, allow_draft=True,
                                                  errata=str(path)))
    try:
        assert rel.cite()["errata"]["file"] == "errata.tsv"
        assert "2 records left out" in capsys.readouterr().out
    finally:
        rel.close()
    for argv in (["aws-launch-trainer", "--release", "v1", "--errata", str(path)],
                 ["picek-train", "--release", "v1", "--errata", str(path)]):
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert "unrecognized arguments: --errata" in capsys.readouterr().err


def test_a_release_without_an_erratum_cites_as_before(twinned):
    world, rid = twinned
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        assert "errata" not in rel.cite()
        assert json.loads(json.dumps(rel.cite()))["dataset_release"] == rid
    finally:
        rel.close()
