"""Per-find attributes for paper-score (paper_attrs.py): the distance band to the nearest
reference record of the true species, and whether the find is in that species' season,
both from public points and dates only."""

import math

import numpy as np
import pytest

from mycomap_vision import paper_attrs as pa


def test_haversine_matches_a_known_distance():
    # One degree of latitude is ~111.2 km anywhere.
    assert pa.km_to_nearest(45.0, -100.0, np.array([46.0, 50.0]), np.array([-100.0, -90.0])) \
        == pytest.approx(111.2, abs=0.2)


def test_bands_follow_the_place_test():
    assert [pa.band_of_km(k) for k in (5, 100, 299.9, 300, 1499, 1500, 4000)] == [
        "<100", "100-300", "100-300", "300-1500", "300-1500", ">1500", ">1500"]


def test_season_is_circular_and_within_thirty_days():
    assert pa.circular_days(5, np.array([360]))[0] == 11       # 26 Dec to 5 Jan
    refs = [{"record_key": "r1", "species": "A x", "observed_on": "2020-12-20"},
            {"record_key": "r2", "species": "B y", "observed_on": "2021-06-01"},
            {"record_key": "r3", "species": "D w", "observed_on": "2021-06-01"},
            {"record_key": "r4", "species": "D w", "observed_on": "2021-09-01"},
            {"record_key": "r5", "species": "D w", "observed_on": "2021-10-01"}]
    scored = [{"record_key": "f1", "species": "A x", "observed_on": "2026-01-10"},
              {"record_key": "f2", "species": "B y", "observed_on": "2026-08-20"},
              {"record_key": "f3", "species": "C z", "observed_on": "2026-08-20"},
              {"record_key": "f4", "species": "A x", "observed_on": None},
              {"record_key": "f5", "species": "D w", "observed_on": "2026-06-10"}]
    a = pa.attributes(scored, refs, {})
    # f5: 1 of D w's 3 dated references within 30 days: some, under half.
    assert [a[k]["in_season"] for k in ("f1", "f2", "f3", "f4", "f5")] == [
        "core", "off", "no reference", "no date", "edge"]


def test_distance_uses_public_points_and_says_why_when_it_cannot():
    refs = [{"record_key": "r1", "species": "A x", "observed_on": None},
            {"record_key": "r2", "species": "B y", "observed_on": None}]
    public = {"r1": (45.0, -100.0), "r2": (None, None), "f1": (45.0, -99.0),
              "f2": (None, None), "f3": (40.0, -100.0)}
    scored = [{"record_key": "f1", "species": "A x"}, {"record_key": "f2", "species": "A x"},
              {"record_key": "f3", "species": "B y"}, {"record_key": "f4", "species": "Q q"}]
    a = pa.attributes(scored, refs, public)
    assert a["f1"]["distance_band"] == "<100" and a["f1"]["distance_km"] == pytest.approx(78.7, abs=0.5)
    assert a["f2"]["distance_band"] == "no point"
    assert a["f3"]["distance_band"] == "no reference point"     # its reference has no point
    assert a["f4"]["distance_band"] == "no reference"


def test_the_tsv_is_what_paper_score_reads(tmp_path):
    from mycomap_vision import paper_score
    path = pa.write(tmp_path / "attrs.tsv", {"k1": {"distance_band": "<100", "distance_km": 5.0,
                                                    "in_season": "in"}})
    assert paper_score.read_attrs(path) == {"k1": {"distance_band": "<100", "distance_km": "5.0",
                                                   "in_season": "in",
                                                   "ecoregion": "not computed"}}


def test_attributes_of_a_release_split(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    monkeypatch.setitem(sys.modules, "torch", None)
    sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
    from release_fixture import BASE, TEST, release_world

    from mycomap_vision.replications.fungitastic import dataset
    release_world(tmp_path, monkeypatch)
    base, test = dataset.load(BASE, True), dataset.load(TEST, True)
    monkeypatch.setattr(pa, "public_points", lambda rel: {k: (47.0, -122.0) for k in
                                                          ("1", "2", "3", "4", "5", "6", "7", "8",
                                                           "mo:9")})
    val = pa.for_release(base, "val")
    assert set(val) == {"5", "6"} and val["5"]["distance_band"] == "<100"
    t = pa.for_release(base, test)
    assert set(t) == {"7", "8"}


def test_vocabulary_membership_by_name_or_a_credited_crosswalk_pair():
    from mycomap_vision.heldout import name_key
    classes = {name_key("Amanita muscaria"), name_key("Trametes cinnabarina")}
    credited = {("Hydnum repandum", "Hydnum washingtonianum")}
    assert pa.in_vocabulary("Amanita muscaria var. guessowii", classes, set())   # species rank
    assert pa.in_vocabulary("Trametes cinnabarina", classes, set())
    assert not pa.in_vocabulary("Hydnum washingtonianum", classes, credited)      # Hydnum repandum is no class
    classes.add(name_key("Hydnum repandum"))
    assert pa.in_vocabulary("Hydnum washingtonianum", classes, credited)
    assert not pa.in_vocabulary("Hydnum washingtonianum", classes, set())
    assert not pa.in_vocabulary("", classes, credited)


def test_extra_columns_are_written_and_read_back(tmp_path):
    from mycomap_vision import paper_score
    path = pa.write(tmp_path / "a.tsv", {"k": {"distance_band": "<100", "distance_km": 1.0,
                                              "in_season": "core", "ecoregion": "TAIGA",
                                              "beit_vocab": "yes"}})
    assert paper_score.read_attrs(path)["k"]["beit_vocab"] == "yes"


def test_a_species_known_only_under_its_variety_has_a_distance_and_a_season():
    from mycomap_vision import paper_attrs as pa
    scored = [{"record_key": "1", "species": "Pseudohydnum gelatinosum", "observed_on": "2026-09-01"}]
    refs = [{"record_key": "r1", "species": "Pseudohydnum gelatinosum ssp. pusillum",
             "observed_on": "2025-09-03"}]
    a = pa.attributes(scored, refs, {"1": (45.0, -122.0), "r1": (45.1, -122.0)})["1"]
    assert a["distance_band"] != "no reference" and a["in_season"] == "core"



# --- E14 secondary rows (Steve, 27 Oct): created_at, inat_observation, RG as of a date ---------

def ident(uid, path, when, level=10, iid=None):
    return {"id": iid or hash((uid, when)) % 10_000, "user_id": uid, "taxon_path": path,
            "rank_level": level, "created_at": when}


SP_A, SP_B, GENUS = [1, 50, 700], [1, 50, 701], [1, 50]


@pytest.mark.parametrize("ids, rg", [
    ([ident(1, SP_A, "2024-01-01"), ident(2, SP_A, "2024-02-01")], True),
    ([ident(1, SP_A, "2024-01-01")], False),                               # one ID only
    ([ident(1, SP_A, "2024-01-01"), ident(2, SP_A, "2024-01-02"),
      ident(3, SP_B, "2024-01-03"), ident(4, SP_B, "2024-01-04")], False),  # 2/4 is not > 2/3
    ([ident(1, SP_A, "2024-01-01"), ident(2, SP_A, "2024-01-02"),
      ident(3, SP_A, "2024-01-03"), ident(4, SP_B, "2024-01-04")], True),   # 3/4
    ([ident(1, SP_A, "2024-01-01"), ident(2, SP_A, "2024-01-02"),
      ident(3, GENUS, "2024-01-03", level=20)], True),                      # an ancestor ID: neutral
    ([ident(1, SP_A, "2024-01-01"), ident(2, SP_A, "2025-06-01")], False),  # second after the date
    ([ident(1, SP_A, "2024-01-01"), ident(2, SP_B, "2024-01-02"),
      ident(2, SP_A, "2024-03-01")], True),                                 # user 2's latest counts
    ([ident(1, GENUS, "2024-01-01", level=20), ident(2, GENUS, "2024-01-02", level=20)], False),
])
def test_research_grade_by_identifications_as_it_stood_on_a_date(ids, rg):
    assert pa.rg_by_identifications(ids, "2025-01-01") is rg


def test_the_secondary_attributes_of_inat_mo_crossposted_and_unknown_finds():
    scored = [{"record_key": "i1", "source": "inat", "source_id": "11", "inat_uuid": "u1"},
              {"record_key": "m1", "source": "mo", "source_id": "501", "inat_uuid": None},
              {"record_key": "x1", "source": "mo", "source_id": "502", "inat_uuid": "u2"},
              {"record_key": "i2", "source": "inat", "source_id": "12", "inat_uuid": "u3"},
              {"record_key": "s1", "source": "com_sequence", "source_id": "9", "inat_uuid": None}]
    inat_created = {"i1": "2025-03-01T00:00:00Z", "x1": "2024-09-01T00:00:00Z", "i2": ""}
    mo_created = {501: "2023-05-05T10:00:00.000Z", 502: "2024-02-02T00:00:00.000Z"}
    snapshot = {"i1": {"identifications": [ident(1, SP_A, "2024-01-01"),
                                           ident(2, SP_A, "2024-02-01")]},
                "x1": {"identifications": [ident(1, SP_A, "2024-01-01")]}}
    a = pa.secondary_attrs(scored, inat_created, mo_created, snapshot, "2025-01-01")
    assert a["i1"] == {"inat_observation": "yes", "created_at": "2025-03-01T00:00:00Z",
                       "rg_at_inat_export": "yes"}
    assert a["m1"] == {"inat_observation": "none", "created_at": "2023-05-05T10:00:00.000Z",
                       "rg_at_inat_export": "no"}                       # MO-only: never on iNat
    assert a["x1"]["inat_observation"] == "yes"
    assert a["x1"]["created_at"] == "2024-02-02T00:00:00.000Z"          # the earlier of the two
    assert a["x1"]["rg_at_inat_export"] == "no"
    assert a["i2"] == {"inat_observation": "yes", "created_at": "", "rg_at_inat_export": "unknown"}
    assert a["s1"] == {"inat_observation": "none", "created_at": "", "rg_at_inat_export": "no"}
    assert "rg_at_inat_export" not in pa.secondary_attrs(scored, inat_created, mo_created)["i1"]


def test_mo_created_at_reads_the_cache_and_fetches_only_what_is_missing(tmp_path, monkeypatch):
    import gzip
    import json
    from mycomap_vision import mo
    with gzip.open(tmp_path / "observations-1-2026-10-12.json.gz", "wt", encoding="utf-8") as f:
        json.dump([{"id": 1, "created_at": "2010-01-01T00:00:00.000Z"}], f)
    asked = []

    def fake_fetch(session, ids, pacer):
        asked.append(list(ids))
        if 3 in ids:
            raise mo.MissingObservation(3)
        return [{"id": i, "created_at": f"2020-01-0{i}T00:00:00.000Z"} for i in ids]
    monkeypatch.setattr(mo.Pacer, "wait", lambda self: None)
    got, stats = pa.mo_created_at([1, 2, 3], tmp_path, fetch=fake_fetch, log=lambda s: None)
    assert got == {1: "2010-01-01T00:00:00.000Z", 2: "2020-01-02T00:00:00.000Z", 3: ""}
    assert asked == [[2, 3], [2]]                  # MO said 3 is missing: asked again without it
    assert stats["cached"] == 1 and stats["missing"] == 1
    again, stats2 = pa.mo_created_at([1, 2], tmp_path, offline=True)      # now from the cache
    assert again == {1: "2010-01-01T00:00:00.000Z", 2: "2020-01-02T00:00:00.000Z"}
    assert stats2["fetched"] == 0 and len(asked) == 2


def test_created_at_and_rg_reach_paper_score_through_the_attrs_file(tmp_path):
    from mycomap_vision import paper_score
    attrs = {"k1": {"distance_band": "<100", "distance_km": 1.0, "in_season": "core",
                    "ecoregion": "x", "created_at": "2025-03-01T00:00:00Z",
                    "inat_observation": "yes", "rg_at_inat_export": "no"}}
    path = pa.write(tmp_path / "attrs.tsv", attrs)
    back = paper_score.read_attrs(path)["k1"]
    assert back["created_at"] == "2025-03-01T00:00:00Z"
    assert back["inat_observation"] == "yes" and back["rg_at_inat_export"] == "no"
