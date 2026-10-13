"""Labels that are no fungus (Steve, 2026-10-13): a record whose label's genus has no fungal
genus on iNat, while every taxon of that name iNat has (any rank) is in Animalia, Plantae or
Bacteria, names a bee, a host plant or a bacterium, not the fungus in the photo. It leaves
references, training and the served index (evaluate.load_records), the dataset builder
lists it as `nonfungal_label`, a frozen release gets it through its erratum, and the
nightly that first learns such kingdoms rebuilds the index. Slime moulds (Protozoa) and a
fungal subgenus written as a genus (Telamonia, also a jumping spider genus) stay in."""

import json

import pytest
import test_taxonomy as tt
from test_dataset_release import add_photo, add_record, build, reasons, world  # noqa: F401
from test_nightly import FakeOutside, box, night, seed_rows  # noqa: F401
from test_taxonomy import LIFE, FUNGI, PLANTAE, evaluate_records, fetch, seed

from mycomap_vision import errata, nightly, taxonomy
from mycomap_vision import dataset_release as dr

ANIMALIA, PROTOZOA, BACTERIA = 1, 47686, 67333
EXTRA = {
    ANIMALIA: ("Animalia", "kingdom", 70, [LIFE], True, []),
    PROTOZOA: ("Protozoa", "kingdom", 70, [LIFE], True, []),
    BACTERIA: ("Bacteria", "kingdom", 70, [LIFE], True, []),
    70001: ("Bombus", "genus", 20, [LIFE, ANIMALIA], True, []),
    70002: ("Physarum", "genus", 20, [LIFE, PROTOZOA], True, []),
    70003: ("Telamonia", "genus", 20, [LIFE, ANIMALIA], True, []),           # a spider
    70004: ("Telamonia", "subgenus", 15, [LIFE, FUNGI, 47169, 50814, 47167, 50003], True, []),
    70005: ("Serratia", "genus", 20, [LIFE, BACTERIA], True, []),
}


@pytest.fixture(autouse=True)
def more_taxa(monkeypatch):
    for tid, row in EXTRA.items():
        monkeypatch.setitem(tt.TAXA, tid, row)


NAMES = [("Bombus impatiens", "Bombus", ""), ("Plantonly nigra", "Plantonly", ""),
         ("Serratia marcescens", "Serratia", ""), ("Physarum polycephalum", "Physarum", ""),
         ("Telamonia armillata", "Telamonia", ""), ("Russula emetica", "Russula", "")]


def test_a_genus_only_animals_plants_or_bacteria_have_names_no_fungus(conn, tmp_path):
    seed(conn, NAMES)
    fetch(conn, tmp_path)
    tax = taxonomy.load(tmp_path / taxonomy.CACHE)
    assert tax.nonfungal("Bombus") == "Animalia"
    assert tax.nonfungal("Plantonly") == "Plantae"
    assert tax.nonfungal("Serratia") == "Bacteria"
    assert tax.nonfungal("Physarum") is None             # a slime mould: in scope
    assert tax.nonfungal("Telamonia") is None            # a Cortinarius subgenus too
    assert tax.nonfungal("Russula") is None and tax.nonfungal("Nevertold") is None
    assert tax.outside["Telamonia"] == ("Animalia", "Fungi")


def test_such_records_leave_the_references_and_the_index(conn, tmp_path):
    seed(conn, NAMES)
    fetch(conn, tmp_path)
    kept = {r["genus"] for r in evaluate_records(conn)}
    assert kept == {"Physarum", "Telamonia", "Russula"}


def test_kingdoms_are_asked_once_per_name_and_only_for_names_with_no_fungal_genus(conn,
                                                                                tmp_path):
    seed(conn, NAMES)
    fake, out = fetch(conn, tmp_path)
    assert out["outside_asked"] == 5                     # not Russula: a fungal genus
    outside = [p for u, p in fake.calls if u.endswith("/taxa") and "rank" not in p]
    assert sorted(p["q"] for p in outside) == ["Bombus", "Physarum", "Plantonly", "Serratia",
                                               "Telamonia"]
    fake2, again = fetch(conn, tmp_path)
    assert again["outside_asked"] == 0 and fake2.calls == []


def test_a_cache_from_before_kingdoms_leaves_every_record_in(conn, tmp_path):
    seed(conn, NAMES)
    fetch(conn, tmp_path)
    db = taxonomy.open_cache(tmp_path / taxonomy.CACHE)
    db.execute("drop table outside_fungi")
    db.commit()
    db.close()
    tax = taxonomy.load(tmp_path / taxonomy.CACHE)
    assert tax.outside == {} and tax.nonfungal("Bombus") is None
    assert taxonomy.nonfungal(None, "Bombus") is None


def cache_beside(world, outside):
    """The taxonomy cache beside the world's manifest: these names have no fungal genus
    and these kingdoms."""
    db = taxonomy.open_cache(world["path"].parent / taxonomy.CACHE)
    for name, kingdoms in outside.items():
        db.execute("insert into answers values (?, 'genus', 'doubt', ?, null, '[]', '[]', 1, "
                   "'t')", (name, taxonomy.NOT_FOUND))
        db.execute("insert into outside_fungi values (?, ?, '[]', 't')",
                   (name, json.dumps(kingdoms)))
    db.commit()
    db.close()


def test_the_builder_lists_such_a_record_as_nonfungal_label(world):
    c = world["conn"]
    add_record(c, "18", "Bombus impatiens")
    add_photo(c, world["store"], "18", 181)
    add_record(c, "19", "Physarum polycephalum")
    add_photo(c, world["store"], "19", 191)
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    cache_beside(world, {"Bombus": ["Animalia"], "Physarum": ["Protozoa"]})
    m = build(world)
    got = reasons(world, m["id"])
    assert got["18"] == ("nonfungal_label", None)
    assert got["19"][0] == "ok" and got["1"][0] == "ok"


def test_a_frozen_release_gets_them_through_its_erratum_after_any_twin(world, tmp_path):
    c = world["conn"]
    add_record(c, "18", "Bombus impatiens")
    add_photo(c, world["store"], "18", 181)
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    m = build(world)                                     # cut before the kingdoms were known
    assert reasons(world, m["id"])["18"][0] == "ok"
    cache_beside(world, {"Bombus": ["Animalia"]})
    tax = taxonomy.load(world["path"].parent / taxonomy.CACHE)
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    try:
        rows = errata.nonfungal_rows(rel, tax)
        assert [(r["key"], r["reason"]) for r in rows] == [("18", "nonfungal_label")]
        assert "genus Bombus is Animalia on iNat" in rows[0]["note"]
        assert errata.nonfungal_rows(rel, tax, skip={"18"}) == []      # listed as a twin
        path = errata.write(rows, tmp_path / "e.tsv")
        rel.use_errata(errata.read(path))
        assert "18" not in {r["record_key"] for r in rel.training_records(include_val=True)}
    finally:
        rel.close()


class KingdomsLearnt(FakeOutside):
    def taxonomy(self, conn, max_minutes=None, cache=None, log=print):
        return {"asked": 0, "outside_asked": 3}


def test_the_nightly_that_learns_such_kingdoms_rebuilds_the_index(box):
    layer = nightly.prepare(box, log=lambda s: None)
    assert night(layer, seed_rows())["ok"]
    assert not night(layer, seed_rows())["changed"]     # nothing new: no rebuild
    report = night(layer, seed_rows(), outside=KingdomsLearnt({}))
    assert report["ok"] and report["changed"]


def test_the_same_reason_is_never_given_twice_and_both_reasons_are_known():
    assert set(errata.REASONS) == {"mo_twin_as_inat", "nonfungal_label"}
    assert "nonfungal_label" in dr.RECORD_REASONS
    assert taxonomy.NOT_FUNGI == {"Animalia", "Plantae", "Bacteria"}
    assert PLANTAE in taxonomy.KINGDOMS and taxonomy.KINGDOMS[PROTOZOA] == "Protozoa"
