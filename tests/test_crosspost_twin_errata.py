"""Erratum rev 4 (Steve, 2026-10-18): one find posted to both Mushroom Observer and iNaturalist
must not sit in train and val at once. The v1 miss review found 480 val finds whose MO or iNat
twin (same label, the same photo re-encoded) was in train; 99% of them were answered right.

The rule (errata.crosspost_twin_pairs / crosspost_twin_rows): an included MO record and an
included iNat record with the same label and a photo within 3 dHash bits pair, unless the hash
is a template (on the photos of 4+ records) or the photo is on the exclusion list
(micrographs). Only a pair straddling train and val is listed, by its val side."""

import json

import numpy as np
import pytest
from test_dataset_release import add_photo, add_record, build, world  # noqa: F401

from mycomap_vision import cli, dhash, errata
from mycomap_vision import dataset_release as dr

TRAIN, VAL = "2026-02-01", "2026-09-20"          # the world's val cutoff falls in August
TEMPLATE = 0x8E0723454523078E                     # a microscope field's hash, say


def h(n: int) -> int:
    """Distinct hashes, far apart (a seeded random 64-bit value per n)."""
    return int(np.random.default_rng(1000 + n).integers(0, 2**62, dtype=np.int64))


def flip(x: int, bits: int) -> int:
    """x with its lowest `bits` bits flipped: exactly `bits` bits away."""
    return x ^ ((1 << bits) - 1)


# (inat key, mo key, inat label, mo label, inat validated, mo validated, inat hash, mo hash)
PAIRS = {
    "straddle_mo_val": ("20", "mo:20", "Pluteus cervinus", "Pluteus cervinus", TRAIN, VAL,
                        h(20), flip(h(20), 3)),
    "straddle_inat_val": ("21", "mo:21", "Gymnopus dryophilus", "Gymnopus dryophilus", VAL,
                          TRAIN, h(21), h(21)),
    "train_train": ("22", "mo:22", "Mycena galericulata", "Mycena galericulata", TRAIN, TRAIN,
                    h(22), h(22)),
    "val_val": ("23", "mo:23", "Hydnum repandum", "Hydnum repandum", VAL, VAL, h(23), h(23)),
    "other_label": ("24", "mo:24", "Mycena", "Mycena galericulata", TRAIN, VAL, h(24), h(24)),
    "template": ("25", "mo:25", "Laccaria laccata", "Laccaria laccata", TRAIN, VAL, TEMPLATE,
                 TEMPLATE),
    "excluded_photo": ("26", "mo:26", "Lepiota cristata", "Lepiota cristata", TRAIN, VAL,
                       h(26), h(26)),
    "four_bits": ("27", "mo:27", "Amanita muscaria", "Amanita muscaria", TRAIN, VAL, h(27),
                  flip(h(27), 4)),
}


@pytest.fixture
def twins(world):
    c = world["conn"]
    for n, (ik, mk, il, ml, iv, mv, _, _) in enumerate(PAIRS.values()):
        add_record(c, ik, il, vdate=iv)
        add_record(c, mk, ml, source="mo", vdate=mv)
        # distinct pictures (the test JPEG repeats every 256 seeds): no byte copies to drop
        add_photo(c, world["store"], ik, 9000 + n, seed=140 + 2 * n)
        add_photo(c, world["store"], mk, 9_000_009_000 + n, source="mo", seed=141 + 2 * n)
    # two more records on the template hash: four records share it
    for k in ("28", "29"):
        add_record(c, k, "Russula emetica", vdate=TRAIN)
        add_photo(c, world["store"], k, 9100 + int(k), seed=130 + int(k) - 28)
    c.commit()
    c.execute("pragma wal_checkpoint(truncate)")
    m = build(world, name="dry-20261018-120000")
    rel = dr.load_release(m["id"], world["root"], allow_draft=True)
    want = {}
    for key, (ik, mk, *_rest, ih, mh) in PAIRS.items():
        want[ik], want[mk] = ih, mh
    want["28"] = want["29"] = TEMPLATE
    rows = rel.db.execute("select photo_key, record_key, original_sha256 from photos "
                          "where included = 1 order by photo_key").fetchall()
    index = {"release": rel.id, "release_hash": rel.release_hash, "algorithm": dhash.ALGORITHM,
             "pillow": dhash.pillow_version(), "photos": len(rows),
             "photo_key": [r[0] for r in rows], "sha256": [r[2] for r in rows],
             "dhash": np.array([want.get(r[1], h(5000 + i)) for i, r in enumerate(rows)],
                               dtype=np.uint64)}
    photo_of = {r[1]: r[0] for r in rows}
    yield world, rel, index, photo_of
    rel.close()


def splits(rel):
    return dict(rel.db.execute("select record_key, split from records"))


def test_the_world_puts_each_pair_in_the_splits_the_test_means(twins):
    _, rel, _, _ = twins
    s = splits(rel)
    assert (s["20"], s["mo:20"]) == ("train", "val")
    assert (s["21"], s["mo:21"]) == ("val", "train")
    assert (s["22"], s["mo:22"]) == ("train", "train")
    assert (s["23"], s["mo:23"]) == ("val", "val")


def test_a_same_label_twin_straddling_train_and_val_loses_its_val_side(twins):
    _, rel, index, photo_of = twins
    excl = {photo_of["mo:26"]}
    rows = errata.crosspost_twin_rows(rel, index, excl)
    assert {r["key"] for r in rows} == {"mo:20", "21"}         # MO val side, iNat val side
    note = next(r["note"] for r in rows if r["key"] == "mo:20")
    assert note.startswith("twin 20 (train), Pluteus cervinus;") and note.endswith("3 bits")
    assert all(r["reason"] == "crosspost_twin" and r["kind"] == "record" for r in rows)


def test_twins_within_one_split_stay_and_are_still_reported_as_pairs(twins):
    _, rel, index, photo_of = twins
    pairs = {(p.mo_record, p.inat_record): p
             for p in errata.crosspost_twin_pairs(rel, index, {photo_of["mo:26"]})}
    assert ("mo:22", "22") in pairs and ("mo:23", "23") in pairs   # E32 can group them
    keys = {r["key"] for r in errata.crosspost_twin_rows(rel, index, {photo_of["mo:26"]})}
    assert not keys & {"22", "mo:22", "23", "mo:23"}


def test_a_different_label_a_template_an_excluded_photo_or_four_bits_never_pair(twins):
    _, rel, index, photo_of = twins
    pairs = {(p.mo_record, p.inat_record) for p in
             errata.crosspost_twin_pairs(rel, index, {photo_of["mo:26"]})}
    for k in ("24", "25", "26", "27"):
        assert (f"mo:{k}", k) not in pairs
    # without the exclusion list the excluded pair is a twin: the list is what keeps it
    keys = {r["key"] for r in errata.crosspost_twin_rows(rel, index)}
    assert "mo:26" in keys


def test_records_already_in_the_erratum_neither_pair_nor_are_listed(twins):
    _, rel, index, photo_of = twins
    rows = errata.crosspost_twin_rows(rel, index, {photo_of["mo:26"]}, skip={"20"})
    assert {r["key"] for r in rows} == {"21"}                  # mo:20's twin is gone


def test_an_index_for_another_release_is_refused(twins):
    _, rel, index, _ = twins
    with pytest.raises(dhash.DhashError, match="not dry-20261018-120000"):
        errata.crosspost_twin_rows(rel, {**index, "release_hash": "0" * 64})


def test_the_cli_writes_a_new_revision_carrying_the_base_unchanged(twins, monkeypatch,
                                                                    capsys):
    world, rel, index, photo_of = twins
    ipath = dhash.save_index(index, world["tmp"] / "dhash-index.npz")
    excl = world["tmp"] / "micrographs.tsv"
    excl.write_text(f"photo_key\tp\n{photo_of['mo:26']}\t0.9\n", encoding="utf-8")
    base = errata.write([{"kind": "record", "key": "21", "reason": "mo_twin_as_inat",
                          "note": "an earlier row"}], world["tmp"] / "r3.tsv")
    monkeypatch.setattr(dr, "ROOT", world["root"])
    out = world["tmp"] / "r4.tsv"
    cli.main(["dataset", "errata", rel.id, "--reasons", "crosspost_twin", "--base", str(base),
              "--dhash-index", str(ipath), "--exclude-photos", str(excl), "--out", str(out)])
    report = json.loads(capsys.readouterr().out)
    e = errata.read(out)
    assert e.reasons == {"21": "mo_twin_as_inat", "mo:20": "crosspost_twin"}
    assert "record\t21\tmo_twin_as_inat\tan earlier row" in out.read_text(encoding="utf-8")
    assert report["base"]["sha256"] == dr.sha256_file(base)
    assert report["crosspost_twin"]["exclude_photos"]["photos"] == 1
    assert report["crosspost_twin"]["max_bits"] == 3
    assert report["included_by_reason_split"] == {"mo_twin_as_inat/val": 1,
                                                  "crosspost_twin/val": 1}
