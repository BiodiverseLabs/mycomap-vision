"""Atlas habitat maps for Vision's place scoring (atlasprior.py), against synthetic exports
in the shapes Atlas writes (R/prior.R, R/continent.R): 1-based 20 km Albers cells, taxa by
name only, uint8 ranks and area-of-applicability ratios."""

import gzip
import json

import numpy as np
import pytest

from mycomap_vision import atlasprior as ap

# Atlas's grid.json for the 20 km grid (R/grid.R extent).
GRID = {"extent": {"xmin": -7445000, "xmax": 4735000, "ymin": -2955000, "ymax": 4740000},
        "cell_m": 20000, "ncol": 609, "nrow": 385}

# Atlas's own numpy port of its cell lookup, as grid.json carries it (R/prior.R).
ATLAS_PORT = """
import numpy as np
def atlas_cell(lat, lng):
    lat = np.asarray(lat, dtype=float); lng = np.asarray(lng, dtype=float)
    a = 6378137.0; f = 1 / 298.257222101; e2 = 2 * f - f * f; e = np.sqrt(e2); rad = np.pi / 180
    def q(phi):
        s = np.sin(phi)
        return (1 - e2) * (s / (1 - e2 * s * s) - (1 / (2 * e)) * np.log((1 - e * s) / (1 + e * s)))
    def m(phi):
        return np.cos(phi) / np.sqrt(1 - e2 * np.sin(phi) ** 2)
    phi0, phi1, phi2, lam0 = 40 * rad, 20 * rad, 60 * rad, -96 * rad
    n = (m(phi1) ** 2 - m(phi2) ** 2) / (q(phi2) - q(phi1))
    C = m(phi1) ** 2 + n * q(phi1)
    rho0 = a * np.sqrt(C - n * q(phi0)) / n
    rho = a * np.sqrt(C - n * q(lat * rad)) / n
    theta = n * (lng * rad - lam0)
    x = rho * np.sin(theta); y = rho0 - rho * np.cos(theta)
    xmin, xmax, ymin, ymax, size = -7445000, 4735000, -2955000, 4740000, 20000
    ncol = int(np.ceil((xmax - xmin) / size))
    col = np.floor((x - xmin) / size); row = np.floor((ymax - y) / size)
    inside = (x >= xmin) & (x < xmax) & (y > ymin) & (y <= ymax)
    return np.where(inside, row * ncol + col + 1, 0).astype(np.int64)
"""

IN01, CA01 = "Coprinus sp. 'comatus-IN01'", "Coprinus sp. 'comatus-CA01'"
MICHIGAN, CALIFORNIA = (42.46, -83.79), (37.8, -122.3)


def test_cells_match_atlas_own_lookup():
    ns = {}
    exec(ATLAS_PORT, ns)                                     # noqa: S102 - the export's text
    rng = np.random.default_rng(1)
    lat, lng = rng.uniform(15, 75, 500), rng.uniform(-170, -50, 500)
    assert (ap.AtlasGrid.from_json(GRID).cell(lat, lng) == ns["atlas_cell"](lat, lng)).all()
    assert ap.AtlasGrid.from_json(GRID).cell(0.0, 0.0) == 0     # off the grid


def tsv(path, header, rows, gz=False):
    text = "\t".join(header) + "\n" + "".join("\t".join(str(x) for x in r) + "\n" for r in rows)
    if gz:
        with gzip.open(path, "wt", encoding="utf-8") as f:
            f.write(text)
    else:
        path.write_text(text, encoding="utf-8")


def exports(tmp_path, continent_spread=False):
    """IN01 mapped (strong, map strength 0.8); CA01 mapped without a map strength in the
    continent taxa (the prior export has it). Michigan: IN01 in reach, rank 90, inside
    its area of applicability; CA01 rank 30, outside (ratio 3). California: the reverse."""
    g = ap.AtlasGrid.from_json(GRID)
    mi, ca = int(g.cell(*MICHIGAN)), int(g.cell(*CALIFORNIA))
    cont, prior = tmp_path / "continent", tmp_path / "prior"
    cont.mkdir()
    prior.mkdir()
    for d in (cont, prior):
        (d / "grid.json").write_text(json.dumps(GRID), encoding="utf-8")
    tsv(cont / "taxa.tsv", ["taxon_id", "name", "grade", "models", "algorithms"],
        [(1, CA01, "strong", 1, "esm"), (2, IN01, "strong", 2, "maxnet,rf")])
    extra = ["models", "rank_sd"] if continent_spread else []
    rows = [(2, mi, 90, 60), (1, mi, 30, 255), (2, ca, 20, 240), (1, ca, 85, 70)]
    tsv(cont / "continent.tsv.gz", ["taxon_id", "cell_id", "rank", "aoa"] + extra,
        [r + ((2, 10) if continent_spread else ()) for r in rows], gz=True)
    # The prior export numbers taxa its own way (ids differ: join by name).
    tsv(prior / "taxa.tsv", ["taxon_id", "name", "grade", "map_strength"],
        [(1, IN01, "strong", 0.8), (2, CA01, "strong", 0.5)])
    tsv(prior / "ranks.tsv.gz", ["taxon_id", "cell_id", "rank", "models", "rank_sd"],
        [(1, mi, 90, 2, 20), (2, ca, 85, 1, "")], gz=True)
    return cont, prior


def test_import_joins_the_two_exports_by_name_and_marks_reach(tmp_path):
    cont, prior = exports(tmp_path)
    summary = ap.import_exports(cont, prior, tmp_path / "table", "R_test")
    assert summary["taxa"] == 2 and summary["cells"] == 2
    assert summary["spread_from"] == "prior (in reach only)"
    t = ap.AtlasTable(tmp_path / "table")
    i01, c01 = t.columns_for([IN01, CA01]).tolist()
    assert i01 >= 0 and c01 >= 0
    assert t.columns_for(["Coprinus sp. 'COMATUS-in01'"])[0] == i01   # same key
    assert t.columns_for(["Coprinus nonesuch"])[0] == -1             # no strong map: none
    assert t.map_strength[i01] == pytest.approx(0.8) and t.map_strength[c01] == pytest.approx(0.5)
    mi = t.at(*MICHIGAN)
    assert mi.rank[i01] == 90 and mi.aoa[i01] == 60 and mi.reach[i01]
    assert mi.models[i01] == 2 and mi.rank_sd[i01] == 20               # from the prior, in reach
    assert mi.rank[c01] == 30 and not mi.reach[c01]
    assert mi.models[c01] == ap.NO_DATA                                # beyond reach: unknown
    assert t.at(0.0, 0.0) is None and t.at(60.0, -100.0) is None       # off grid / no cell


def test_the_continent_spread_is_used_when_atlas_adds_it(tmp_path):
    cont, prior = exports(tmp_path, continent_spread=True)
    assert ap.import_exports(cont, prior, tmp_path / "t", "R")["spread_from"] == "continent"
    t = ap.AtlasTable(tmp_path / "t")
    c01 = t.columns_for([CA01])[0]
    assert t.at(*MICHIGAN).models[c01] == 2 and t.at(*MICHIGAN).rank_sd[c01] == 10


def test_trust_is_strength_times_agreement_times_applicability():
    a, assumed = ap.agreement(np.array([2, 1, 3, ap.NO_DATA]), np.array([20, ap.NO_DATA, 80,
                                                                        ap.NO_DATA]))
    assert a.tolist() == pytest.approx([0.6, 1.0, 0.0, 1.0])
    assert assumed.tolist() == [False, False, False, True]
    aoa = np.array([60, 100, 200, ap.NO_DATA])
    assert ap.applicability(aoa, "cut").tolist() == [1.0, 1.0, 0.0, 0.0]
    assert ap.applicability(aoa, "graded") == pytest.approx([1.0, 1.0, np.exp(-1.0), 0.0])
    t = ap.trust(np.array([0.8, 0.8, 0.8, 0.8]), np.array([2, 2, 2, 2]),
                 np.array([20, 20, 20, 20]), aoa, "graded")
    assert t == pytest.approx([0.48, 0.48, 0.48 * np.exp(-1.0), 0.0])
    assert ap.trust(np.full(4, 0.8), np.full(4, 2), np.full(4, 20), aoa, "cut",
                    use_trust=False).tolist() == [1.0, 1.0, 0.0, 0.0]


def test_the_rank_factor_is_learned_from_where_truths_fall_against_look_alikes():
    truths = np.array([95, 92, 88, 85, 91, 97, 60])
    looks = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 5, 15, 25])
    lf = ap.rank_factor(truths, looks)
    assert lf[9] > 0 > lf[1]


def test_the_atlas_term_is_a_boost_only_and_silent_without_a_rank():
    lf = np.linspace(-1, 1, ap.RANK_BINS)
    got = ap.atlas_boost(np.array([100, 5, ap.NO_DATA]), np.array([0.5, 0.5, 0.5]), lf)
    assert got[0] == pytest.approx(0.5 * lf[10])
    assert got[1] == 0.0 and got[2] == 0.0                 # f < 1 is never a penalty


def test_boyce_index_rewards_truths_in_high_ranks():
    rng = np.random.default_rng(3)
    bg = rng.integers(0, 101, 5000)
    good = rng.integers(60, 101, 200)
    bad = rng.integers(0, 101, 200)
    assert ap.boyce_index(good, bg) > 0.8
    assert abs(ap.boyce_index(bad, bg)) < 0.8
    assert np.isnan(ap.boyce_index(good[:3], bg))


# --- the arms (placetune) ------------------------------------------------------------------

def arm_components():
    """One record, three candidates: 0 mapped (rank 95, inside AOA, in reach), 1 mapped
    (rank 95 but outside its reach and its AOA, ratio 3), 2 unmapped. DNA boost 1.0 each,
    candidate 1 also flagged by the served out-of-range rule."""
    from mycomap_vision import placetune as pt
    K = 3
    one = lambda v, dt=float: np.array([v], dtype=dt)                      # noqa: E731
    c = pt.Components(S=np.zeros((1, K)), cand=np.arange(K)[None], SP=np.ones((1, K), bool),
                      GG=np.zeros((1, K), int), truth_at=one(0, int), has_sp=one(True, bool),
                      truth_g=one(0, int), boost={sh: np.ones((1, K)) for sh in pt.SHRINKS},
                      season={sh: np.zeros((1, K)) for sh in pt.SHRINKS},
                      berth=np.array([[False, True, False]]), km_dna=np.zeros((1, K)),
                      km_inat=np.zeros((1, K)), n_place=np.ones((1, K)),
                      off_season=np.zeros((1, K), bool))
    a = pt.AtlasComponents(mapped=np.array([[True, True, False]]),
                           rank=np.array([[95, 95, ap.NO_DATA]], np.uint8),
                           aoa=np.array([[50, 200, ap.NO_DATA]], np.uint8),
                           models=np.array([[1, 1, ap.NO_DATA]], np.uint8),
                           rank_sd=np.full((1, K), ap.NO_DATA, np.uint8),
                           reach=np.array([[True, False, False]]),
                           strength=np.array([[0.8, 0.8, 0.0]]))
    return c, a


def test_p2c_replaces_the_dna_boost_for_mapped_taxa_only():
    from mycomap_vision import placetune as pt
    c, a = arm_components()
    log_f = np.full(ap.RANK_BINS, 1.0)
    z = pt.combine_atlas(c, a, {"arm": "P2c", "weight": 0.5, "applicability": "cut",
                                "trust": True}, log_f)[0]
    assert z[0] == pytest.approx(0.5 * 0.8 * 1.0)          # Atlas term, not the DNA boost
    assert z[1] == pytest.approx(0.0 - 2.0)                # outside AOA (cut): 0; berth kept
    assert z[2] == pytest.approx(1.0)                      # unmapped: the served DNA boost
    graded = pt.combine_atlas(c, a, {"arm": "P2c", "weight": 0.5, "applicability": "graded",
                                     "trust": True}, log_f)[0]
    assert graded[1] == pytest.approx(0.5 * 0.8 * np.exp(-1.0) - 2.0)
    untrusted = pt.combine_atlas(c, a, pt.p2c_u({"arm": "P2c", "weight": 0.5,
                                                 "applicability": "cut", "trust": True}),
                                 log_f)[0]
    assert untrusted[0] == pytest.approx(0.5)              # trust = 1


def test_p2d_uses_the_reach_as_the_range_with_a_mild_penalty_outside():
    from mycomap_vision import placetune as pt
    c, a = arm_components()
    z = pt.combine_atlas(c, a, {"arm": "P2d", "weight": 1.0, "applicability": "cut",
                                "trust": True, "outside_penalty": 0.5},
                         np.full(ap.RANK_BINS, 1.0))[0]
    assert z[0] == pytest.approx(0.8)
    assert z[1] == pytest.approx(-0.5)                     # outside reach; no 1,500 km berth
    assert z[2] == pytest.approx(1.0)


def test_the_declared_atlas_grid():
    from mycomap_vision import placetune as pt
    arms = [s["arm"] for s in pt.atlas_settings()]
    assert arms.count("P2c") == 6 and arms.count("P2d") == 6 and arms.count("P3") == 3
    assert arms.count("P2a") == 3 and arms.count("P2b") == 3


def test_the_primary_test_needs_both_overall_gain_and_the_band():
    from mycomap_vision import placetune as pt
    n = 2000
    has = np.ones(n, bool)
    band = np.zeros(n, bool)
    band[:700] = True
    folds = np.arange(n) % 5
    p0 = np.zeros(n, bool)
    p0[:300] = True
    gains = p0.copy()
    gains[1000:1100] = True                                 # +100 outside the band
    assert pt.primary_test(p0, gains, has, band, folds, 2000)["passes"] is True
    loses = gains.copy()
    loses[:60] = False                                      # -60 of 700 in the band (8.6%)
    r = pt.primary_test(p0, loses, has, band, folds, 2000)
    assert r["overall"]["passes"] is True and r["passes"] is False


def test_a_null_aoa_is_unknown_never_inside(tmp_path):
    # Final export (Atlas finish version 5): aoa is null where no model has a usable
    # threshold; grid.json counts those models.
    cont, prior = exports(tmp_path, continent_spread=True)
    meta = json.loads((cont / "grid.json").read_text(encoding="utf-8"))
    (cont / "grid.json").write_text(json.dumps({**meta, "models_without_threshold": 3}),
                                    encoding="utf-8")
    g = ap.AtlasGrid.from_json(GRID)
    tsv(cont / "continent.tsv.gz", ["taxon_id", "cell_id", "rank", "models", "rank_sd", "aoa"],
        [(2, int(g.cell(*MICHIGAN)), 90, 1, "", "")], gz=True)
    summary = ap.import_exports(cont, prior, tmp_path / "t", "R")
    assert summary["models_without_threshold"] == 3
    t = ap.AtlasTable(tmp_path / "t")
    i01 = t.columns_for([IN01])[0]
    view = t.at(*MICHIGAN)
    assert view.aoa[i01] == ap.NO_DATA and view.rank_sd[i01] == ap.NO_DATA
    assert ap.applicability(view.aoa[[i01]], "graded")[0] == 0.0


def test_a_capped_aoa_is_far_outside_and_only_a_null_is_unknown(tmp_path):
    # Atlas caps aoa at 255 (far outside); a null is "no threshold". They must differ.
    cont, prior = exports(tmp_path, continent_spread=True)
    g = ap.AtlasGrid.from_json(GRID)
    tsv(cont / "continent.tsv.gz", ["taxon_id", "cell_id", "rank", "models", "rank_sd", "aoa"],
        [(2, int(g.cell(*MICHIGAN)), 90, 1, "", 255), (1, int(g.cell(*MICHIGAN)), 30, 1, "", "")],
        gz=True)
    ap.import_exports(cont, prior, tmp_path / "t", "R")
    t = ap.AtlasTable(tmp_path / "t")
    i01, c01 = t.columns_for([IN01, CA01]).tolist()
    view = t.at(*MICHIGAN)
    assert view.aoa[i01] == 254 and view.aoa[c01] == ap.NO_DATA
    graded = ap.applicability(view.aoa[[i01, c01]], "graded")
    assert graded[0] == pytest.approx(np.exp(-1.54)) and graded[1] == 0.0
