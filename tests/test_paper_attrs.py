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
