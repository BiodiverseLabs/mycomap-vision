import numpy as np
import pytest

from mycomap_vision import occprior
from mycomap_vision.evaluate import METHODS, Record, evaluate
from mycomap_vision.identify import CONFIDENCE_TEMPERATURE, Identifier, softmax_confidence
from mycomap_vision.methods import NearestSpecimen
from mycomap_vision.occprior import OccParams, OccurrencePrior, WithOccurrence
from mycomap_vision.prior import LOGPROB_CONFIDENCE_TEMPERATURE, Context, WithPrior
from occurrence_fixtures import EAST, WEST, build_store

GROUPS = sorted(["Amanita occidentalis", "Amanita orientalis", "Amanita rara",
                 "Amanita sp. 'IN01'", "Westia pacifica", "Westia inventa", "Amanita inventa",
                 "Nullus nihil"])


def rec(oid, species, lat, lon, observed=None, rows=(), vdate="2026-01-01"):
    return Record(oid, species, species.split()[0], "F", vdate, "u", list(rows),
                  latitude=lat, longitude=lon, observed_on=observed)


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    return build_store(tmp_path_factory.mktemp("occ"))[0]


def fitted(store, records=(), **params):
    p = OccurrencePrior(store, OccParams(**params))
    p.fit(list(records), GROUPS)
    return p


def g(name):
    return GROUPS.index(name)


def out_of_range(prior, lat, lon):
    return prior.parts(Context(lat, lon, None)).out(prior.params.radius_km)


# --- the wide berth ---------------------------------------------------------------------------

def test_a_species_is_out_of_range_only_beyond_the_radius(store):
    p = fitted(store)
    assert out_of_range(p, *EAST)[g("Westia pacifica")]           # ~3,400 km from all 30
    assert not out_of_range(p, 40.0, -108.0)[g("Westia pacifica")]  # ~1,000 km away
    assert not out_of_range(p, *WEST)[g("Westia pacifica")]
    assert not fitted(store, radius_km=4000.0).parts(Context(*EAST)).out(4000.0)[
        g("Westia pacifica")]
    lp = p.log_prior(Context(*EAST))
    assert lp[g("Westia pacifica")] <= -p.params.out_of_range_penalty + 2 * np.log(5)


def test_absence_counts_only_for_a_species_with_enough_occurrences(store):
    # Amanita rara: 5 observations, all in the west. Too few for its absence in the
    # east to mean anything.
    assert not out_of_range(fitted(store), *EAST)[g("Amanita rara")]
    assert out_of_range(fitted(store, min_occurrences=5), *EAST)[g("Amanita rara")]


def test_a_dna_record_nearby_vetoes_the_penalty(store):
    dna = [rec("d1", "Westia pacifica", EAST[0] + 0.5, EAST[1])]
    assert not out_of_range(fitted(store, dna), *EAST)[g("Westia pacifica")]


def test_a_provisional_species_uses_its_dna_records_under_the_same_rule(store):
    many = [rec(f"d{i}", "Amanita sp. 'IN01'", WEST[0], WEST[1]) for i in range(20)]
    few = many[:3]
    assert out_of_range(fitted(store, many), *EAST)[g("Amanita sp. 'IN01'")]
    assert not out_of_range(fitted(store, few), *EAST)[g("Amanita sp. 'IN01'")]
    assert not out_of_range(fitted(store, many), *WEST)[g("Amanita sp. 'IN01'")]


def test_a_species_whose_whole_genus_is_far_away_is_out_of_range(store):
    # Not on iNat, no DNA records, but no Westia at all within the radius.
    p = fitted(store)
    assert out_of_range(p, *EAST)[g("Westia inventa")]
    assert not out_of_range(p, *WEST)[g("Westia inventa")]


def test_no_penalty_where_fungi_go_unobserved_or_off_the_map(store):
    p = fitted(store)
    assert not out_of_range(p, 80.0, -60.0).any()          # inside the map, nothing near
    assert np.all(p.log_prior(Context(48.0, 2.0, "2025-10-01")) == 0)    # Europe: off the map
    assert np.all(p.log_prior(None) == 0) and np.all(p.log_prior(Context()) == 0)


# --- gentle inside the berth ------------------------------------------------------------------

def test_inside_the_berth_place_only_reweights_gently(store):
    p = fitted(store, radius_km=6000.0, season_weight=0.0)
    lp = p.log_prior(Context(*EAST))
    assert np.abs(lp).max() <= p.params.density_weight * p.params.density_cap + 1e-9
    assert lp[g("Amanita orientalis")] > 0 > lp[g("Amanita occidentalis")]


def test_season_favours_the_species_observed_that_week(store):
    p = fitted(store, density_weight=0.0)
    autumn, spring = p.log_prior(Context(None, None, "2026-10-03")), \
        p.log_prior(Context(None, None, "2026-04-18"))
    assert autumn[g("Amanita orientalis")] > autumn[g("Amanita occidentalis")]
    assert spring[g("Amanita occidentalis")] > spring[g("Amanita orientalis")]
    assert np.abs(autumn).max() <= p.params.season_weight * p.params.season_cap + 1e-9


def test_a_species_unknown_to_inat_borrows_its_genus_else_stays_neutral(store):
    p = fitted(store, radius_km=6000.0)
    parts = p.parts(Context(*EAST, "2026-10-03"))
    amanita_genus_alone = parts.density[g("Amanita inventa")]
    assert parts.density[g("Amanita sp. 'IN01'")] == amanita_genus_alone
    assert parts.density[g("Nullus nihil")] == 0 and parts.season[g("Nullus nihil")] == 0
    assert p.log_prior(Context(*EAST, "2026-10-03"))[g("Nullus nihil")] == 0


def test_names_are_mapped_once_and_summarised(store):
    p = fitted(store)
    assert p.mapping_summary() == {"species": 4, "provisional": 1, "not-on-inat": 2, "none": 1}


# --- the methods ------------------------------------------------------------------------------

@pytest.fixture
def default_store(store, monkeypatch, tmp_path):
    monkeypatch.setattr(occprior.OccurrenceStore, "load", classmethod(lambda cls, path=None: store))
    monkeypatch.setattr(occprior, "default_params_path", lambda: tmp_path / "none.json")
    return store


def test_occ_methods_are_registered_alongside_prior_and_break_a_visual_tie_by_place(default_store):
    assert {"nearest+occ", "species-mean+occ", "linear+occ", "hybrid+occ"} <= set(METHODS)
    assert isinstance(METHODS["nearest+prior"](), WithPrior)        # unchanged
    m = METHODS["nearest+occ"]()
    assert m.name == "nearest+occ" and m.needs_context
    # Two identical-looking species, one on each coast.
    ref = ([rec(f"e{i}", "Amanita orientalis", *EAST, rows=[i]) for i in range(3)]
           + [rec(f"w{i}", "Amanita occidentalis", *WEST, rows=[3 + i]) for i in range(3)])
    vecs = np.tile(np.array([[1.0, 0.0]], dtype=np.float16), (8, 1))
    test = [rec("t-e", "Amanita orientalis", EAST[0] + 0.2, EAST[1], "2026-10-03", rows=[6],
                vdate="2026-09-20"),
            rec("t-w", "Amanita occidentalis", WEST[0] + 0.2, WEST[1], "2026-04-18", rows=[7],
                vdate="2026-09-20")]
    assert evaluate(vecs, ref, test, method="nearest+occ")["species"]["all"]["top1"] == 1.0
    assert evaluate(vecs, ref, test, method="nearest")["species"]["all"]["top1"] < 1.0


def test_the_occ_method_reads_its_tuned_values(store, tmp_path):
    path = tmp_path / "params.json"
    occprior.save_params(OccParams(radius_km=900.0, photo_temperature=0.03,
                                   confidence_temperature={"species": 2.5}), path)
    m = WithOccurrence(NearestSpecimen, True, store=store, params=occprior.load_params(path))
    assert m.params.radius_km == 900.0 and m.base.temperature == 0.03
    assert m.confidence_temperature == {"species": 2.5}


# --- confidence -------------------------------------------------------------------------------

def identifier_with(model, calibration=None):
    ident = Identifier.__new__(Identifier)
    ident.model, ident.calibration = model, calibration
    return ident


def test_prior_methods_state_graded_confidence_instead_of_100_percent(store):
    # Two candidates 0.03 apart in cosine similarity: 1.5 nats apart as log-probabilities.
    logp = np.log(softmax_confidence(np.array([0.80, 0.77, 0.60]), 0.02))
    for model in (WithOccurrence(NearestSpecimen, True, store=store, params=OccParams()),
                  WithPrior(NearestSpecimen)):
        t = identifier_with(model).temperature("species")
        assert t == LOGPROB_CONFIDENCE_TEMPERATURE
        assert softmax_confidence(logp, t).max() < 0.75
    # The cosine default on the same log-probabilities is the bug: ~100% every time.
    assert softmax_confidence(logp, CONFIDENCE_TEMPERATURE).max() > 0.99


def test_plain_methods_keep_the_cosine_temperature_and_a_calibration_still_wins(store):
    assert identifier_with(NearestSpecimen()).temperature("species") == CONFIDENCE_TEMPERATURE
    m = WithOccurrence(NearestSpecimen, True, store=store,
                       params=OccParams(confidence_temperature={"species": 2.5, "genus": 3.0}))
    assert identifier_with(m).temperature("genus") == 3.0
    assert identifier_with(m).temperature("family") == 2.5      # falls back to species
    cal = {"temperatures": {"species": 1.1}, "n": {"species": 10}, "comparison_id": "c"}
    assert identifier_with(m, cal).temperature("species") == 1.1


# --- a record never finds itself --------------------------------------------------------------

def test_a_scored_record_never_finds_its_own_observation(tmp_path):
    from occurrence_fixtures import obs, standard_observations
    # One Westia pacifica far east of all the others: the record being scored.
    store = build_store(tmp_path, standard_observations()
                        + [obs("self-1", 132, *EAST, date="2025-10-01")])[0]
    p = fitted(store)
    westia = g("Westia pacifica")
    seen_by_others = p.parts(Context(*EAST, "2025-10-01"))
    itself = p.parts(Context(*EAST, "2025-10-01", uuid="self-1"))
    assert not seen_by_others.out(1500.0)[westia]       # someone else's find
    assert itself.out(1500.0)[westia]                    # its own find: taken out
    assert itself.density[westia] < seen_by_others.density[westia]
    assert itself.season[westia] < seen_by_others.season[westia]
    # Taking it out matches never having counted it.
    never = fitted(build_store(tmp_path / "x", standard_observations())[0])
    ref = never.parts(Context(*EAST, "2025-10-01"))
    assert np.allclose(itself.density, ref.density) and np.allclose(itself.season, ref.season)
    assert np.array_equal(itself.out(1500.0), ref.out(1500.0))


def test_the_build_records_what_each_observation_added(tmp_path):
    from occurrence_fixtures import obs
    store = build_store(tmp_path, [obs("v", 103, *EAST, date="2025-10-01"),
                                   obs("g", 101, *WEST, date="")])[0]
    v = store.own_contribution("v")
    assert sorted(store.names[u] for u in v.units) == ["Amanita", "Amanita muscaria",
                                                       "Amanita muscaria guessowii"]
    assert v.cell == store.grid.cell(*EAST) and v.week == 39
    gen = store.own_contribution("G")
    assert [store.names[u] for u in gen.units] == ["Amanita"] and gen.week == -1
    assert store.own_contribution("not-counted") is None and store.own_contribution(None) is None


def test_comparisons_score_each_record_without_its_own_observation(conn, tmp_path):
    from mycomap_vision.evaluate import context_of, load_records
    from test_models_and_scoreboard import seed_two_species
    seed_two_species(conn, tmp_path)
    photos = {p: p for (p,) in conn.execute("select photo_id from observation_photos")}
    recs = load_records(conn, photos)
    assert recs and all(r.uuid == f"uuid-{r.observation_id}" for r in recs)
    assert context_of(recs[0]).uuid == recs[0].uuid


def test_taking_a_record_out_matches_never_counting_it_under_observer_days(tmp_path):
    from occurrence_fixtures import obs, standard_observations
    extra = {"same-day-1": obs("same-day-1", 132, *EAST, "2025-10-01", observer=5),
             "same-day-2": obs("same-day-2", 132, EAST[0] + 0.1, EAST[1], "2025-10-01",
                               observer=5),
             "other-taxon": obs("other-taxon", 104, *EAST, "2025-10-01", observer=5),
             "alone": obs("alone", 132, EAST[0] + 3, EAST[1], "2025-10-08", observer=6)}
    base = standard_observations()
    full = fitted(build_store(tmp_path / "full", base + list(extra.values()))[0])
    for i, uuid in enumerate(extra):
        without = fitted(build_store(tmp_path / f"w{i}",
                                     base + [r for k, r in extra.items() if k != uuid])[0])
        for lat, lon, when in ((*EAST, "2025-10-01"), (EAST[0] + 3, EAST[1], "2025-10-08")):
            got = full.parts(Context(lat, lon, when, uuid=uuid))
            want = without.parts(Context(lat, lon, when))
            assert np.array_equal(got.out(1500.0), want.out(1500.0)), uuid
            assert np.allclose(got.density, want.density), uuid
            assert np.allclose(got.season, want.season), uuid


def test_a_find_shared_with_the_same_observer_day_still_counts(tmp_path):
    from occurrence_fixtures import obs, standard_observations
    # The only eastern Westia: two observations by one person on one day.
    rows = standard_observations() + [
        obs("twin-1", 132, *EAST, "2025-10-01", observer=5),
        obs("twin-2", 132, EAST[0] + 0.1, EAST[1], "2025-10-01", observer=5)]
    p = fitted(build_store(tmp_path, rows)[0])
    assert not p.parts(Context(*EAST, uuid="twin-1")).out(1500.0)[g("Westia pacifica")]
