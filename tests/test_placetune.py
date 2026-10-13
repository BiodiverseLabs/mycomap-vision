"""Stronger place penalties (Steve, 2026-10-13): the named settings placeprior serves, the
penalty math, and the tuning harness (placetune) scoring exactly what would be served."""

import numpy as np
import pytest
from occurrence_fixtures import EAST, WEST, build_store
from test_prior import rec

from mycomap_vision import placeprior as pp
from mycomap_vision import placetune as pt
from mycomap_vision.evaluate import build_index
from mycomap_vision.prior import Context

ORI, OCC, MUS = "Amanita orientalis", "Amanita occidentalis", "Amanita muscaria"
STEEP = {"pen_slope": 2.0, "pen_start_km": 300.0, "pen_cap": 4.0, "pen_known": "dna"}


def records():
    """DNA records: orientalis near EAST in autumn, occidentalis near WEST in spring, and
    muscaria near WEST only (iNat's open data also has it in the EAST)."""
    east = [rec(f"e{i}", ORI, EAST[0] + 0.01 * i, EAST[1], f"2025-10-0{1 + i % 5}", rows=[i])
            for i in range(6)]
    west = [rec(f"w{i}", OCC, WEST[0] + 0.01 * i, WEST[1], f"2025-04-1{i % 5}", rows=[6 + i])
            for i in range(6)]
    mus = [rec(f"m{i}", MUS, WEST[0], WEST[1] + 0.01 * i, "2025-08-01", rows=[12 + i])
           for i in range(3)]
    return east + west + mus


@pytest.fixture
def named(monkeypatch):
    """Register settings for a test; the served table itself is never edited."""
    def add(name, **extras):
        monkeypatch.setitem(pp.PLACE_SETTINGS, name, extras)
        return name
    return add


def prior(setting=None, store=None):
    recs = records()
    index = build_index(recs)
    p = pp.PlaceBoost(store, setting=setting)
    if store is None:
        p.ready = lambda: False            # never this machine's own store
    p.fit(recs, index.species)
    return p, index


# --- the settings ------------------------------------------------------------------------

def test_unset_the_served_setting_of_2026_10_11_is_used_and_adds_nothing(monkeypatch):
    monkeypatch.delenv(pp.SETTING_ENV, raising=False)
    p, _ = prior()
    assert p.setting_name == pp.DEFAULT_SETTING == "boost-2026-10-11"
    assert not p.has_extras() and p.extras(Context(*EAST, "2026-04-15")) is None
    assert not hasattr(p, "cell_group")    # the default holds no extra tables


def test_the_environment_picks_a_named_setting_and_a_typo_is_refused(monkeypatch, named):
    named("steep", **STEEP)
    monkeypatch.setenv(pp.SETTING_ENV, "steep")
    assert pp.PlaceBoost().setting_name == "steep"
    monkeypatch.setenv(pp.SETTING_ENV, "steeep")
    with pytest.raises(ValueError, match="not a place setting"):
        pp.PlaceBoost()


# --- the penalty math --------------------------------------------------------------------

def test_the_distance_penalty_starts_at_its_start_grows_per_1000_km_and_is_capped():
    km = np.array([0.0, 300.0, 800.0, 1300.0, 5000.0, np.inf, np.nan])
    got = pp.distance_penalty(km, 300.0, 2.0, 4.0)
    assert got.tolist() == [0.0, 0.0, 1.0, 2.0, 4.0, 0.0, 0.0]   # no known record: none


def test_damping_spares_a_taxon_known_from_few_records():
    assert pp.count_damping(np.array([3.0, 30.0]), 0.0).tolist() == [1.0, 1.0]
    d = pp.count_damping(np.array([3.0, 30.0]), 5.0)
    assert d[0] == pytest.approx(3 / 8) and d[1] == pytest.approx(30 / 35)


def test_far_species_are_pushed_down_and_near_ones_are_not(named):
    plain, index = prior()
    steep, _ = prior(named("steep", **STEEP))
    ctx = Context(*EAST)
    delta = steep.log_prior(ctx) - plain.log_prior(ctx)
    ori, occ = index.species.index(ORI), index.species.index(OCC)
    assert delta[ori] == 0.0
    assert delta[occ] == pytest.approx(-4.0)           # ~3,400 km: capped at 4


def test_damping_scales_the_penalty_by_the_taxons_records_with_a_place(named):
    plain, index = prior()
    damped, _ = prior(named("damped", **STEEP, damp_k=5.0))
    occ = index.species.index(OCC)
    delta = damped.log_prior(Context(*EAST)) - plain.log_prior(Context(*EAST))
    assert delta[occ] == pytest.approx(-4.0 * 6 / 11)


def test_a_far_candidate_pushed_down_still_carries_the_far_find_cue():
    recs = records()
    index = build_index(recs)
    cue = pp.PlaceCue()
    cue.fit(recs, index)
    c = cue.cue(cue.tables(Context(*EAST)), "species", OCC)
    assert c["far_from_known"] is True and c["note"] == pp.FAR_NOTE


def test_inat_finds_count_as_known_records_when_the_setting_says_so(tmp_path, named):
    store = build_store(tmp_path / "occ")[0]
    plain, index = prior(store=store)
    dna, _ = prior(named("dna", **STEEP), store=store)
    both, _ = prior(named("both", **{**STEEP, "pen_known": "dna+inat"}), store=store)
    mus = index.species.index(MUS)
    ctx = Context(*EAST)
    # muscaria: DNA records only in the west, but iNat finds at EAST itself.
    assert dna.log_prior(ctx)[mus] - plain.log_prior(ctx)[mus] == pytest.approx(-4.0)
    assert both.log_prior(ctx)[mus] == pytest.approx(plain.log_prior(ctx)[mus])
    assert pp.nearest_inat_km(both.far, *EAST)[mus] < 40     # its 0.5-degree cell centre


def test_off_season_costs_only_a_species_with_a_known_season(named):
    plain, index = prior()
    p, _ = prior(named("season", season_penalty=1.5))
    ctx = Context(None, None, "2026-04-15")             # spring
    delta = p.log_prior(ctx) - plain.log_prior(ctx)
    assert delta[index.species.index(ORI)] == pytest.approx(-1.5)    # autumn species
    assert delta[index.species.index(OCC)] == 0.0
    assert delta[index.species.index(MUS)] == 0.0       # 3 dated records: no known season


def test_more_shrinkage_damps_the_boost_two_nearby_records_give_a_sparse_species(named):
    # The Coprinus find (2026-10-12): a 3-record species, 2 of them ~50 km away, was
    # boosted over a 50-record look-alike by those two records alone.
    recs = [rec(f"s{i}", "Coprinus rarus", EAST[0] + 0.4 * (i < 2), EAST[1] + 30 * (i == 2),
                "2025-10-01", rows=[i]) for i in range(3)]
    recs += [rec(f"d{i}", "Coprinus densus", EAST[0] + 3 * (i % 10) - 15, EAST[1] - 2 * (i % 7),
                 "2025-10-01", rows=[3 + i]) for i in range(50)]
    index = build_index(recs)
    lp = {}
    for shrink in (3.0, 30.0):
        p = pp.PlaceBoost(None, setting=named(f"s{shrink:g}", shrink=shrink))
        p.ready = lambda: False
        p.fit(recs, index.species)
        lp[shrink] = p.log_prior(Context(*EAST))[index.species.index("Coprinus rarus")]
    assert 0 < lp[30.0] < lp[3.0]


# --- the harness scores what is served ---------------------------------------------------

def test_the_grid_is_declared_with_the_served_setting_first():
    grid = pt.settings()
    assert len(grid) == 9 * 97
    assert grid[0] == pt.SERVED and pt.SERVED["shrink"] == 3.0
    assert all(set(pp.NO_EXTRAS) <= set(s) for s in grid)
    assert len({pt.describe(s) for s in grid}) == len(grid)


@pytest.mark.parametrize("extras", [
    {}, STEEP, {**STEEP, "pen_known": "dna+inat", "damp_k": 5.0},
    {"shrink": 30.0, "season_penalty": 2.0}, {**STEEP, "shrink": 10.0, "season_penalty": 1.0}])
def test_a_setting_tuned_is_exactly_the_setting_served(tmp_path, named, extras):
    store = build_store(tmp_path / "occ")[0]
    recs = records()
    index = build_index(recs)
    served, _ = prior(named("x", **extras), store=store)
    tb = pt.TermBuilder(recs, index.species, store)
    G = len(index.species)
    for ctx in (Context(*EAST, "2026-04-15"), Context(*WEST, "2026-10-02"),
                Context(None, None, "2026-04-15")):
        t = tb.terms(ctx)
        c = pt.Components(S=np.zeros((1, G)), cand=np.arange(G)[None], SP=np.ones((1, G), bool),
                          GG=np.zeros((1, G), int), truth_at=np.array([0]),
                          has_sp=np.array([True]), truth_g=np.array([0]),
                          boost={sh: t[f"boost_{sh:g}"][None] for sh in pt.SHRINKS},
                          season={sh: t[f"season_{sh:g}"][None] for sh in pt.SHRINKS},
                          berth=t["berth"][None], km_dna=t["km_dna"][None],
                          km_inat=t["km_inat"][None], n_place=t["n_place"][None],
                          off_season=t["off_season"][None])
        tuned = pt.combine(c, {**pt.SERVED, **extras})[0]
        assert np.allclose(tuned, served.log_prior(ctx)), (extras, ctx)


# --- selection ---------------------------------------------------------------------------

def test_a_setting_that_breaks_the_band_on_the_training_folds_is_never_chosen():
    grid = [pt.SERVED, {**pt.SERVED, **STEEP}, {**pt.SERVED, "shrink": 10.0}]
    n = 100
    band = np.zeros(n, bool)
    band[:20] = True
    base = np.zeros(n, bool)
    base[:20] = True                                    # photos only: the band all right
    sp = np.zeros((3, n), bool)
    sp[0, :20] = True                                   # served: band kept
    sp[1, 20:60] = True                                 # steep: best overall, band lost
    sp[2, :20] = True
    sp[2, 20:25] = True                                 # shrink: band kept, a little better
    ge = sp.copy()
    folds = np.arange(n) % 5
    free = pt.nested_cv(sp, ge, folds, grid)
    held = pt.nested_cv(sp, ge, folds, grid, band=band, base=base)
    assert set(free["chosen"]) == {1}
    assert set(held["chosen"]) == {2}
    assert held["species_right"][:20].all()


def test_the_band_bound_is_a_paired_bootstrap_lower_bound():
    rng = np.random.default_rng(0)
    a = rng.random(700) < 0.3
    b = a.copy()
    b[:40] = False                                       # 40 broken in a 700-record band
    out = pt.band_lower_bound(a, b, np.ones(700, bool), np.arange(700) % 5, n_boot=2000)
    assert out["delta"] < 0 and out["lower95"] < out["delta"]
    assert out["passes_margin"] is (out["lower95"] > -pt.MARGIN)


def test_the_papers_bound_resamples_whole_observers():
    # 30 broken records: all from one observer (30 records), or one each from 30 observers.
    # Resampling observers whole, the first is far less certain.
    n = 700
    a = np.ones(n, bool)
    observers = ["big"] * 30 + [f"o{i}" for i in range(670)]
    lumpy = a.copy()
    lumpy[:30] = False
    spread = a.copy()
    spread[[30 + i * 20 for i in range(30)]] = False
    lo = pt.band_lower_bound_clustered(a, lumpy, np.ones(n, bool), observers, 4000)
    sp = pt.band_lower_bound_clustered(a, spread, np.ones(n, bool), observers, 4000)
    assert lo["delta"] == sp["delta"] == pytest.approx(-30 / 700)
    assert lo["observers"] == 671
    assert lo["lower95"] < sp["lower95"]


def test_bands_can_be_the_papers_labels():
    a = np.array([True, True, False, True])
    b = np.array([False, True, True, True])
    labels = np.array(["<100", "300-1500", "300-1500", "no point"])
    rows = {r["band"]: r for r in pt.label_table(a, b, labels, np.ones(4, bool), pt.PAPER_BANDS)}
    assert rows["<100"]["broken"] == 1 and rows["300-1500"]["fixed"] == 1
    assert rows["no point"]["n"] == 1 and rows[">1500"]["n"] == 0


def test_the_v1_candidates_are_served_by_name_and_the_default_is_unchanged(monkeypatch):
    monkeypatch.delenv(pp.SETTING_ENV, raising=False)
    assert pp.place_setting()[0] == "boost-2026-10-11"
    assert pp.place_setting()[1] == pp.NO_EXTRAS
    banded = pp.place_setting("penalty-500-2026-10-13")[1]
    assert banded == {**pp.NO_EXTRAS, "shrink": 10.0, "pen_slope": 2.0, "pen_start_km": 500.0,
                      "pen_cap": 8.0, "season_penalty": 1.0}
    # Each candidate is a setting of the declared grid, so it was scored as tuned.
    grid = [{k: s[k] for k in pp.NO_EXTRAS} for s in pt.settings()]
    for name in ("penalty-500-2026-10-13", "penalty-300-2026-10-13", "shrink-2026-10-13"):
        assert pp.place_setting(name)[1] in grid, name
