"""nearest+mean+place (Steve, 2026-10-11): the boost prior only when a place or date is
given, the far-find cue beside every species and genus candidate, the release that carries
its data, and the fallback when a server can't run it."""

import json

import numpy as np
import pytest
from occurrence_fixtures import EAST, WEST, build_store
from test_api import app_with_model, png
from test_models_and_scoreboard import Const, seed_two_species
from test_prior import rec
from test_release import MemoryS3, publish, with_embeddings

from mycomap_vision import placeprior as pp
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.evaluate import build_index
from mycomap_vision.guards import Limits
from mycomap_vision.prior import Context

ORI, OCC = "Amanita orientalis", "Amanita occidentalis"


def records_east_west():
    """Our DNA records: orientalis near EAST in autumn, occidentalis near WEST in spring."""
    east = [rec(f"e{i}", ORI, EAST[0] + 0.01 * i, EAST[1], "2025-10-0" + str(1 + i % 5),
                rows=[i]) for i in range(6)]
    west = [rec(f"w{i}", OCC, WEST[0] + 0.01 * i, WEST[1], "2025-04-1" + str(i % 5),
                rows=[6 + i]) for i in range(6)]
    return east + west


def fitted_prior(tmp_path, store=True):
    recs = records_east_west()
    index = build_index(recs)
    prior = pp.PlaceBoost(build_store(tmp_path / "occ")[0] if store else None)
    if not store:
        prior.ready = lambda: False          # this machine's own store must not be read
    prior.fit(recs, index.species)
    return prior, index


# --- the prior ------------------------------------------------------------------------

def test_without_a_place_or_a_date_there_is_no_prior_at_all(tmp_path):
    prior, _ = fitted_prior(tmp_path, store=False)
    assert prior.log_prior(None) is None
    assert prior.log_prior(Context()) is None
    assert prior.log_prior(Context(None, None, "1970-01-01")) is None     # placeholder date


def test_nearby_dna_records_rank_a_species_up_and_absence_costs_nothing(tmp_path):
    prior, index = fitted_prior(tmp_path, store=False)
    lp = prior.log_prior(Context(*EAST))
    ori, occ = index.species.index(ORI), index.species.index(OCC)
    assert 0 < lp[ori] <= pp.LOG5 + 1e-9          # boosted, capped at log 5
    assert lp[occ] == 0.0                          # far from its records: no penalty


def test_beyond_the_wide_berth_a_species_known_elsewhere_loses_two(tmp_path):
    prior, index = fitted_prior(tmp_path, store=True)
    lp = prior.log_prior(Context(*EAST))
    occ = index.species.index(OCC)
    # occidentalis: many iNat finds, all ~3,400 km west, none within 1,500 km of EAST.
    assert lp[occ] == pytest.approx(-pp.BOOST["far_penalty"])


def test_a_find_never_puts_its_own_species_in_range_through_its_own_inat_observation(tmp_path):
    """The out-of-range term reads iNaturalist's counts, which may hold the scored find's
    own observation: given its uuid, the store takes it back out, exactly as if it had
    never been counted (paper: the own observation is left out, on val and on the test)."""
    from occurrence_fixtures import obs, standard_observations
    recs = records_east_west()
    index = build_index(recs)
    occ = index.species.index(OCC)
    # The only eastern occidentalis on iNaturalist is the find being scored.
    store = build_store(tmp_path / "with", standard_observations()
                        + [obs("self-1", 105, *EAST, date="2025-10-01")])[0]
    prior = pp.PlaceBoost(store)
    prior.fit(recs, index.species)
    someone_elses = prior.log_prior(Context(*EAST, "2025-10-01"))
    its_own = prior.log_prior(Context(*EAST, "2025-10-01", uuid="self-1"))
    assert someone_elses[occ] > -pp.BOOST["far_penalty"]          # in range: no far penalty
    assert its_own[occ] == pytest.approx(someone_elses[occ] - pp.BOOST["far_penalty"])
    never = pp.PlaceBoost(build_store(tmp_path / "never")[0])
    never.fit(recs, index.species)
    assert np.allclose(its_own, never.log_prior(Context(*EAST, "2025-10-01")))


def test_the_date_alone_moves_the_season_term_both_ways(tmp_path):
    prior, index = fitted_prior(tmp_path, store=False)
    lp = prior.log_prior(Context(None, None, "2026-10-03"))
    ori, occ = index.species.index(ORI), index.species.index(OCC)
    assert lp[ori] > 0 > lp[occ]                   # autumn favours orientalis
    assert abs(lp).max() <= pp.BOOST["season_weight"] * pp.LOG5 + 1e-9


def test_the_method_adds_the_shift_only_with_a_place_or_date(tmp_path):
    m = pp.NearestMeanPlace()
    m.prior, _ = fitted_prior(tmp_path, store=False)
    assert m.place_shift(None) is None and m.place_shift(Context()) is None
    shift = m.place_shift(Context(*EAST))
    assert shift is not None and np.allclose(shift, pp.PHOTO_TEMPERATURE
                                             * m.prior.log_prior(Context(*EAST)))


# --- the identifier ------------------------------------------------------------------

def identifiers(conn, tmp_path):
    from mycomap_vision.identify import Identifier, map_embeddings
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    conn.commit()
    root = tmp_path / "emb" / "m1"
    plain = Identifier(conn, "m1", "nearest+mean", embeddings_root=root, photo_info=False)
    place = Identifier(conn, "m1", "nearest+mean+place", embeddings_root=root,
                       photo_info=False)
    _, vecs = map_embeddings(conn, "m1", root)
    return plain, place, np.asarray(vecs[[0, 3]], dtype=np.float32)


def without_method(result):
    out = json.loads(json.dumps(result))
    out.pop("model")
    out.pop("prior")
    return out


def test_without_place_or_date_the_answer_is_nearest_plus_mean_exactly(conn, tmp_path):
    plain, place, query = identifiers(conn, tmp_path)
    for ctx in (None, Context(), Context(None, None, None)):
        a, b = plain.identify_vectors(query, context=ctx), place.identify_vectors(query, context=ctx)
        assert without_method(a) == without_method(b)
        assert b["prior"] == {"applied": False, "method": "nearest+mean+place"}
        assert b["photo_only_top"] == {}
        assert all(c["place"] is None for r in ("species", "genus", "family")
                   for c in b["ranks"][r])


def test_with_a_place_the_prior_is_applied_and_reported(conn, tmp_path):
    plain, place, query = identifiers(conn, tmp_path)
    shift = np.zeros(len(place.index.species))
    shift[place.index.species.index("B y")] = 5.0          # a place strong enough to win
    place.model.place_shift = lambda ctx: None if ctx is None else shift
    r = place.identify_vectors(query[:1], context=Context(*EAST))       # photo of A
    assert r["prior"]["applied"] is True
    assert r["ranks"]["species"][0]["name"] == "B y"
    assert r["photo_only_top"] == {"species": "A x", "genus": "A"}
    alone = plain.identify_vectors(query[:1], context=Context(*EAST))
    assert alone["ranks"]["species"][0]["name"] == "A x"


def test_a_shared_calibration_keeps_the_confidences_of_nearest_plus_mean():
    from mycomap_vision.evaluate import calibration_method
    assert calibration_method("nearest+mean+place") == "nearest+mean"
    assert calibration_method("nearest") == "nearest"


# --- the cue -------------------------------------------------------------------------

def cue_for(records, ctx):
    index = build_index(records)
    c = pp.PlaceCue()
    c.fit(records, index)
    t = c.tables(ctx)
    return {name: c.cue(t, "species", name) for name in index.labels["species"]}, c, t


def test_the_cue_says_how_far_each_species_nearest_record_is_rounded_to_10_km():
    recs = records_east_west()
    cues, c, t = cue_for(recs, Context(*EAST, None))
    near, far = cues[ORI], cues[OCC]
    assert near["nearest_record_km"] <= 20 and near["far_from_known"] is False
    assert near["note"] is None
    assert 3_000 < far["nearest_record_km"] < 3_700 and far["nearest_record_km"] % 10 == 0
    assert far["far_from_known"] is True
    assert far["note"] == pp.FAR_NOTE
    assert near["in_season"] is None                      # no date given
    assert c.cue(t, "genus", "Amanita")["nearest_record_km"] == near["nearest_record_km"]
    assert c.cue(t, "family", "F") is None                # species and genus only


def test_the_far_flag_starts_just_past_300_km():
    recs = [rec("a", ORI, 40.0, -80.0, "2025-10-01", rows=[0])]
    # 0.1 degree is half a cell: query from cell centres 2.7 and 2.8 degrees north.
    lat0 = pp.snap(40.0, -80.0)[0]
    under, _, _ = cue_for(recs, Context(lat0 + 2.6, pp.snap(40.0, -80.0)[1]))
    over, _, _ = cue_for(recs, Context(lat0 + 2.8, pp.snap(40.0, -80.0)[1]))
    assert under[ORI]["nearest_record_km"] <= 300 and under[ORI]["far_from_known"] is False
    assert over[ORI]["nearest_record_km"] > 300 and over[ORI]["far_from_known"] is True


def test_the_cue_never_places_a_record_finer_than_its_obscuring_cell():
    # Two records at different spots of one 0.2-degree cell give the same distance from
    # anywhere: the cue measures to the cell's centre.
    q = Context(41.37, -79.21)
    a, _, _ = cue_for([rec("a", ORI, 40.01, -80.19, None, rows=[0])], q)
    b, _, _ = cue_for([rec("b", ORI, 40.19, -80.01, None, rows=[0])], q)
    assert a[ORI]["nearest_record_km"] == b[ORI]["nearest_record_km"]


def test_off_season_is_flagged_only_for_a_species_with_a_known_season():
    recs = records_east_west()                            # six dated records each
    cues, _, _ = cue_for(recs, Context(None, None, "2026-04-15"))
    assert cues[OCC]["in_season"] is True and cues[OCC]["note"] is None
    assert cues[ORI]["in_season"] is False and cues[ORI]["note"] == pp.SEASON_NOTE
    assert cues[ORI]["nearest_record_km"] is None and cues[ORI]["far_from_known"] is None
    few, _, _ = cue_for(recs[:pp.MIN_DATED_FOR_SEASON - 1], Context(None, None, "2026-04-15"))
    assert few[ORI]["in_season"] is None                  # too few dated records to say


def test_far_and_off_season_together_give_both_notes():
    cues, _, _ = cue_for(records_east_west(), Context(*WEST, "2026-04-15"))
    assert cues[ORI]["note"] == f"{pp.FAR_NOTE}; {pp.SEASON_NOTE}"


# --- serving --------------------------------------------------------------------------

def test_the_default_falls_back_to_nearest_plus_mean_where_place_cannot_run():
    base = dict(rate=(1000, 60.0), max_in_flight=4, allowed_backbones={"m1"},
                default_method="nearest+mean+place")
    assert Limits(**base).served_default() == "nearest+mean+place"
    assert Limits(**base).served_default(lambda m: m != "nearest+mean+place") == "nearest+mean"
    assert Limits(**base, allowed_methods={"nearest", "nearest+mean"}).served_default() \
        == "nearest+mean"
    assert Limits(**base, allowed_methods={"nearest"}).served_default() == "nearest"


def test_a_release_ships_the_occurrence_store_but_never_its_leave_one_out_index(conn, tmp_path):
    data = tmp_path / "data"
    with_embeddings(conn, data, "m1")
    (data / "occurrence").mkdir()
    (data / "occurrence" / "inat-fungi-na.npz").write_bytes(b"store")
    (data / "occurrence" / "inat-fungi-na.loo.npz").write_bytes(b"loo")
    s3 = MemoryS3()
    rid = publish(conn, data, s3)["id"]
    paths = {f["path"] for f in json.loads(s3.objects[f"releases/{rid}/release.json"])["files"]}
    assert "occurrence/inat-fungi-na.npz" in paths
    assert not any(p.endswith(".loo.npz") for p in paths)


def test_the_identify_response_carries_the_prior_and_the_cue(conn, tmp_path):
    c = app_with_model(conn, tmp_path)
    files = [("photos", ("p0.png", png(247), "image/png"))]
    res = c.post("/api/identify", files=files,
                 data={"models": "m1/nearest+mean", "lat": "40.0", "lng": "-80.0",
                       "observed_on": "2026-10-01"})
    assert res.status_code == 200
    [r] = res.json()["results"]
    assert r["prior"] == {"applied": False, "method": "nearest+mean"}
    assert r["photo_only_top"] == {}
    cand = r["ranks"]["species"][0]
    assert set(cand["place"]) == {"nearest_record_km", "in_season", "far_from_known", "note"}
    assert r["ranks"]["family"][0]["place"] is None


def test_the_served_prior_builds_no_season_table(tmp_path):
    # Out of range is all the served prior takes from iNat; the season table is the
    # biggest one, and the box has 4 GB.
    prior, _ = fitted_prior(tmp_path, store=True)
    assert prior.far.weeks is None
    assert prior.far.parts(Context(*EAST, "2026-10-01"), (1500.0,)).season is None


def test_names_in_the_likely_set_and_each_photos_answer_carry_the_cue_too(conn, tmp_path):
    _, place, query = identifiers(conn, tmp_path)
    place._likely = lambda scores: {"species": {"names": [{"name": "A x", "confidence": 0.9}]}}
    r = place.identify_vectors(query, context=Context(*EAST, "2026-10-01"))
    [named] = r["likely"]["species"]["names"]
    assert named["place"] is not None and set(named["place"]) == {
        "nearest_record_km", "in_season", "far_from_known", "note"}
    for photo in r["per_photo"]:
        assert all("place" in c for c in photo["ranks"]["species"])
