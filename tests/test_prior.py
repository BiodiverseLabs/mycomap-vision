import numpy as np

from mycomap_vision.evaluate import METHODS, Record, build_index, evaluate
from mycomap_vision.prior import Context, RangeSeasonPrior, WithPrior, haversine_km


def rec(oid, species, lat, lon, observed, rows=(), vdate="2026-01-01"):
    return Record(oid, species, species.split()[0], "F", vdate, "u", list(rows),
                  latitude=lat, longitude=lon, observed_on=observed)


# Two look-alike species: one in the east in autumn, one in the west in spring.
EAST = [rec(f"e{i}", "Lookus orientalis", 40 + i * 0.1, -80, "2025-10-01") for i in range(8)]
WEST = [rec(f"w{i}", "Lookus occidentalis", 40 + i * 0.1, -120, "2025-04-15") for i in range(8)]
SPECIES = ["Lookus occidentalis", "Lookus orientalis"]


def fitted():
    p = RangeSeasonPrior()
    p.fit(EAST + WEST, SPECIES)
    return p


def test_distances_are_great_circle_kilometres():
    d = haversine_km(40.0, -80.0, np.array([40.0, 41.0]), np.array([-80.0, -80.0]))
    assert d[0] == 0 and 110 < d[1] < 112


def test_place_favours_the_species_found_there():
    lp = fitted().log_prior(Context(40.3, -80.2, None))
    assert lp[SPECIES.index("Lookus orientalis")] > 0 > lp[SPECIES.index("Lookus occidentalis")]


def test_season_favours_the_species_found_then():
    lp = fitted().log_prior(Context(None, None, "2026-04-20"))
    assert lp[SPECIES.index("Lookus occidentalis")] > lp[SPECIES.index("Lookus orientalis")]


def test_the_score_is_capped_so_it_cannot_overrule_clear_photos():
    here_and_now = Context(40.3, -80.2, "2025-10-01")
    east = SPECIES.index("Lookus orientalis")
    p = fitted()
    p.cap = np.inf
    uncapped = p.log_prior(here_and_now)[east]
    p.cap = 0.5
    assert uncapped > 0.5
    assert p.log_prior(here_and_now)[east] == 0.5


def test_no_place_or_date_means_no_effect():
    assert np.all(fitted().log_prior(Context()) == 0)
    assert np.all(fitted().log_prior(None) == 0)


def test_heavy_sampling_in_one_area_does_not_favour_every_species_there():
    # 200 records of a common species in the east make the east heavily sampled; a
    # species recorded evenly east and west must not look like an eastern species.
    common = [rec(f"c{i}", "Commonus vulgaris", 40, -80, None) for i in range(200)]
    even = ([rec(f"x{i}", "Evenus planus", 40, -80, None) for i in range(5)] +
            [rec(f"y{i}", "Evenus planus", 40, -120, None) for i in range(5)])
    p = RangeSeasonPrior()
    species = ["Commonus vulgaris", "Evenus planus"]
    p.fit(common + even, species)
    east = p.log_prior(Context(40, -80, None))[1]
    west = p.log_prior(Context(40, -120, None))[1]
    assert west > east          # it is relatively rarer among eastern records


def test_a_single_record_species_borrows_its_genus_range():
    one = [rec("s1", "Lookus solus", 40, -120, None)]
    species = sorted(SPECIES + ["Lookus solus"])
    solo = species.index("Lookus solus")

    def at_its_spot(shrink):
        p = RangeSeasonPrior()
        p.shrink = shrink
        p.fit(EAST + WEST + one, species)
        return p.log_prior(Context(40, -120, None))[solo]

    # One record right here would otherwise read as a strong local species.
    assert 0 < at_its_spot(3.0) < at_its_spot(1e-9)


def test_the_prior_methods_are_registered_and_break_a_visual_tie_by_place():
    assert {"linear+prior", "hybrid+prior"} <= set(METHODS)
    # Both species look identical: every photo vector is the same.
    ref = [rec(r.observation_id, r.species, r.latitude, r.longitude, r.observed_on,
               rows=[i]) for i, r in enumerate(EAST + WEST)]
    vecs = np.tile(np.array([[1.0, 0.0]], dtype=np.float16), (len(ref) + 2, 1))
    test = [rec("t-e", "Lookus orientalis", 40.2, -80.1, "2026-10-03", rows=[len(ref)],
                vdate="2026-09-20"),
            rec("t-w", "Lookus occidentalis", 40.2, -120.1, "2026-04-18", rows=[len(ref) + 1],
                vdate="2026-09-20")]
    with_prior = evaluate(vecs, ref, test, method="hybrid+prior")
    assert with_prior["species"]["all"]["top1"] == 1.0


def test_with_prior_needs_context_and_names_itself_after_its_base():
    from mycomap_vision.methods import LinearHead
    m = WithPrior(LinearHead)
    assert m.needs_context and m.name == "linear+prior"


def test_similarity_methods_also_come_with_the_prior_on_a_log_probability_scale():
    assert {"nearest+prior", "species-mean+prior"} <= set(METHODS)
    m = METHODS["nearest+prior"]()
    assert m.name == "nearest+prior" and m.needs_context
    ref = [rec(r.observation_id, r.species, r.latitude, r.longitude, r.observed_on,
               rows=[i]) for i, r in enumerate(EAST + WEST)]
    vecs = np.tile(np.array([[1.0, 0.0]], dtype=np.float16), (len(ref) + 1, 1))
    test = [rec("t-w", "Lookus occidentalis", 40.2, -120.1, "2026-04-18", rows=[len(ref)],
                vdate="2026-09-20")]
    assert evaluate(vecs, ref, test, method="nearest+prior")["species"]["all"]["top1"] == 1.0
