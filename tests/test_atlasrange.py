import gzip
import json

import numpy as np
import pytest

from mycomap_vision import atlasrange
from mycomap_vision.atlasrange import (OUTSIDE, AlbersGrid, AtlasExport, AtlasRangeSource,
                                       LayeredSource, atlas_name_key, estimate_rank_factor)
from mycomap_vision.evaluate import Record
from mycomap_vision.occprior import RANGE_SOURCES, OccParams, OccurrencePrior, TuningRefused
from mycomap_vision.occtune import ScoredSet
from mycomap_vision.prior import Context, haversine_km
from occurrence_fixtures import EAST, WEST, build_store

GROUPS = sorted(["Amanita occidentalis", "Amanita orientalis", "Amanita rara",
                 "Amanita sp. 'IN01'", "Westia pacifica", "Nullus nihil"])
# North America Albers on GRS80 (the usual parameters; Atlas supplies its own).
NA_ALBERS = dict(lat_1=20.0, lat_2=60.0, lat_0=40.0, lon_0=-96.0, cell_m=20_000.0,
                 x_origin=-6_500_000.0, y_origin=5_000_000.0, ncol=600, nrow=450)


def g(name):
    return GROUPS.index(name)


# --- the grid ---------------------------------------------------------------------------------

def test_albers_matches_snyders_worked_example():
    # Snyder, Map Projections: A Working Manual (1987), Albers on the Clarke 1866
    # ellipsoid: 35 N, 75 W with parallels 29.5 and 45.5, origin 23 N, 96 W.
    grid = AlbersGrid(29.5, 45.5, 23.0, -96.0, 20_000.0, 0.0, 2_000_000.0, 500, 300,
                      a=6_378_206.4, es=0.00676866)
    x, y = grid.xy(35.0, -75.0)
    assert x == pytest.approx(1_885_472.7, abs=0.1) and y == pytest.approx(1_535_925.0, abs=0.1)
    lat, lon = grid.latlon(x, y)
    assert lat == pytest.approx(35.0, abs=1e-9) and lon == pytest.approx(-75.0, abs=1e-9)


def test_cell_ids_by_hand():
    grid = AlbersGrid(29.5, 45.5, 23.0, -96.0, 20_000.0, 0.0, 2_000_000.0, 500, 300,
                      a=6_378_206.4, es=0.00676866)
    # x 1,885,472.7 -> col 94; y 1,535,925.0 -> row floor(464,075.0 / 20,000) = 23.
    assert grid.cell(35.0, -75.0) == 23 * 500 + 94
    # The projection's origin: x 0 -> col 0; y 0 -> row 100.
    assert grid.cell(23.0, -96.0) == 100 * 500
    assert grid.cell(23.0, -150.0) == -1                     # west of the grid's first column


def test_a_cell_centre_lies_within_the_cell():
    grid = AlbersGrid(**NA_ALBERS)
    for lat, lon in (EAST, WEST, (19.6, -155.5), (64.0, -147.0)):
        c = grid.cell(lat, lon)
        clat, clon = grid.centres(np.array([c]))
        assert haversine_km(lat, lon, clat, clon)[0] < 15          # half a 20 km diagonal
        assert grid.cell(float(clat[0]), float(clon[0])) == c
    assert grid.cell(48.0, 2.0) == -1


# --- an export in the proposed shape --------------------------------------------------------

TAXA = [(11, "amanita orientalis"), (12, "amanita occidentalis"), (13, "westia pacifica"),
        (14, "amanita sp. 'in01'"), (15, "amanita rara"), (16, "coverus omnis")]
HOME = {11: EAST, 12: WEST, 13: WEST, 14: WEST, 15: WEST}


def cells_near(grid, lat, lon, km):
    out = set()
    for dlat in np.arange(-km / 111, km / 111, 0.1):
        for dlon in np.arange(-km / 80, km / 80, 0.1):
            c = grid.cell(lat + dlat, lon + dlon)
            if c >= 0:
                out.add(c)
    cells = np.array(sorted(out))
    clat, clon = grid.centres(cells)
    return cells[haversine_km(lat, lon, clat, clon) <= km]


def write_export(path, grid_params=NA_ALBERS, rara_listed=False):
    """Each taxon accessible within 300 km of home (rank 90 in its home cell, lower
    further out); Coverus omnis, in no index group, makes three more cells exist."""
    grid = AlbersGrid(**grid_params)
    path.mkdir(parents=True, exist_ok=True)
    (path / "grid.json").write_text(json.dumps({**grid_params, "release": "test"}),
                                    encoding="utf-8")
    taxa = [t for t in TAXA if rara_listed or t[0] != 15]
    with gzip.open(path / "taxa.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("taxon_id\tname\tgrade\tmodels\tsites\taccessible_cells\n")
        for tid, name in taxa:
            f.write(f"{tid}\t{name}\tstrong\t3\t40\t700\n")
    rows = []
    for tid, _ in taxa:
        if tid == 16:
            continue
        lat, lon = HOME[tid]
        home = grid.cell(lat, lon)
        cells = cells_near(grid, lat, lon, 300)
        clat, clon = grid.centres(cells)
        d = haversine_km(lat, lon, clat, clon)
        rows += [f"{tid}\t{c}\t{90 if c == home else int(85 - dd / 5)}"
                 for c, dd in zip(cells, d)]
    for lat, lon in (EAST, WEST, (40.0, -108.0)):          # cells the export covers
        rows += [f"16\t{grid.cell(lat, lon)}\t50"]
    with gzip.open(path / "ranks.tsv.gz", "wt", encoding="utf-8") as f:
        f.write("taxon_id\tcell_id\trank\n" + "\n".join(rows) + "\n")
    return path


@pytest.fixture(scope="module")
def export(tmp_path_factory):
    return AtlasExport.load(write_export(tmp_path_factory.mktemp("atlas")))


def source(export, records=(), **params):
    s = AtlasRangeSource(export, OccParams(**params))
    s.fit(list(records), GROUPS)
    return s


def test_the_export_is_held_dense_one_column_per_cell(export):
    assert export.ranks.dtype == np.uint8 and export.ranks.shape == (5, len(export.cells))
    col = export.column_of(export.grid.cell(*EAST))
    assert export.ranks[export.row_of_key["amanita orientalis"], col] == 90
    assert export.ranks[export.row_of_key["westia pacifica"], col] == OUTSIDE
    assert export.column_of(-5) == -1


def test_names_match_through_one_canonical_key():
    assert atlas_name_key("Amanita sp-IN01") == atlas_name_key("Amanita sp. 'IN01'") \
        == atlas_name_key("Amanita sp. IN01")
    assert atlas_name_key("Amanita muscaria (L.) Lam.") == atlas_name_key("amanita  Muscaria")
    assert atlas_name_key("Amanita muscaria var. guessowii Veselý") == \
        atlas_name_key("Amanita muscaria guessowii")
    assert atlas_name_key("Amanita occidentalis Peck") == "amanita occidentalis"


# --- the source ----------------------------------------------------------------------------

def test_out_of_range_only_when_the_accessible_area_is_beyond_the_radius(export):
    s = source(export)
    east = s.parts(Context(*EAST), (0.0, 1500.0)).out_of_range
    assert east[1500.0][g("Westia pacifica")] and east[1500.0][g("Amanita sp. 'IN01'")]
    assert not east[1500.0][g("Amanita orientalis")]
    # 1,000 km from Westia's area: outside its own cell's area, inside the wide berth.
    near = s.parts(Context(40.0, -108.0), (0.0, 1500.0)).out_of_range
    assert near[0.0][g("Westia pacifica")] and not near[1500.0][g("Westia pacifica")]


def test_a_taxon_atlas_does_not_list_is_no_information(export):
    s = source(export, rank_factor=[0.1 * i for i in range(11)])
    parts = s.parts(Context(*EAST), (0.0, 1500.0))
    for name in ("Amanita rara", "Nullus nihil"):            # unlisted: rows or not
        assert not parts.out_of_range[0.0][g(name)] and parts.density[g(name)] == 0
    assert s.mapping_summary() == {"atlas_listed": 4, "not_in_atlas": 2}


def test_a_cell_the_export_does_not_cover_is_no_information(export):
    s = source(export, rank_factor=[1.0] * 11)
    parts = s.parts(Context(30.0, -90.0), (0.0, 1500.0))       # on the grid, no rows there
    assert parts.density is None and not parts.out_of_range[1500.0].any()
    assert s.parts(Context(48.0, 2.0)).density is None          # off the grid


def test_the_place_term_is_the_learned_factor_never_a_raw_rank(export):
    assert source(export).parts(Context(*EAST)).density is None       # nothing learned yet
    factor = [-1.0] * 9 + [0.5, 0.5]
    parts = source(export, rank_factor=factor).parts(Context(*EAST))
    assert parts.density[g("Amanita orientalis")] == 0.5               # rank 90 at home
    assert parts.density[g("Westia pacifica")] == 0                     # outside: the penalty's job
    assert parts.season is None


def test_a_dna_record_nearby_vetoes_atlas_out_of_range(export):
    dna = [Record("d1", "Westia pacifica", "Westia", "F", None, "u", [],
                  latitude=EAST[0] + 0.3, longitude=EAST[1])]
    assert not source(export, dna).parts(Context(*EAST)).out_of_range[1500.0][
        g("Westia pacifica")]


def test_the_rank_factor_is_learned_as_a_likelihood_ratio():
    # One cell: the true species ranks it 95 in its own range, a lookalike 5 in its own.
    grid = AlbersGrid(**NA_ALBERS)
    cell = grid.cell(*EAST)
    ex = AtlasExport(grid, [1, 2], ["amanita orientalis", "westia pacifica"],
                     np.array([cell]), np.array([[95], [5]], dtype=np.uint8))
    species = ["Amanita orientalis", "Westia pacifica"]
    src = AtlasRangeSource(ex)
    src.fit([], species)
    sset = ScoredSet([str(i) for i in range(30)], species,
                     np.tile(np.array([[0.8, 0.9]], dtype=np.float32), (30, 1)),
                     truth=["Amanita orientalis"] * 30,
                     latitude=[EAST[0]] * 30, longitude=[EAST[1]] * 30)
    f = estimate_rank_factor(src, sset, top_k=2)
    assert len(f) == 11 and f[9] > 0 > f[0]


def test_atlas_first_inat_elsewhere_and_the_season_stays_vision_s_own(export, tmp_path):
    store = build_store(tmp_path)[0]
    params = OccParams(min_occurrences=5, rank_factor=[0.3] * 11)
    inat = OccurrencePrior(store, params)
    layered = LayeredSource(AtlasRangeSource(export, params), inat)
    layered.fit([], GROUPS)
    ctx = Context(*EAST, "2026-10-01")
    both, alone, atlas = layered.parts(ctx), inat.parts(ctx), layered.primary.parts(ctx)
    rara, westia = g("Amanita rara"), g("Westia pacifica")
    assert both.out(1500.0)[rara] == alone.out(1500.0)[rara]                     # iNat's
    assert both.density[rara] == alone.density[rara]
    assert both.density[westia] == atlas.density[westia]                          # Atlas's
    assert np.array_equal(both.season, alone.season)


def test_atlas_cannot_be_tuned_until_it_says_what_it_was_trained_on(export, tmp_path):
    with pytest.raises(TuningRefused, match="Atlas"):
        source(export).leak_check(["u1"])
    store = build_store(tmp_path)[0]
    with pytest.raises(TuningRefused, match="Atlas"):
        LayeredSource(source(export), OccurrencePrior(store)).leak_check(["u1"])


def test_atlas_sources_are_registered_but_not_ready_without_an_export(monkeypatch, tmp_path):
    monkeypatch.setattr(atlasrange, "default_export_dir", lambda: tmp_path / "none")
    assert {"atlas", "atlas+inat-occurrence"} <= set(RANGE_SOURCES)
    assert not RANGE_SOURCES["atlas"].ready()
    with pytest.raises(FileNotFoundError, match="no Atlas export"):
        AtlasExport.load(tmp_path / "none")



def test_where_atlas_says_nothing_every_component_is_inat_s(export, tmp_path):
    store = build_store(tmp_path)[0]
    inat = OccurrencePrior(store, OccParams(rank_factor=[0.3] * 11))
    layered = LayeredSource(AtlasRangeSource(export, inat.params), inat)
    layered.fit([], GROUPS)
    for ctx in (Context(30.0, -90.0, "2026-10-01"),      # a cell Atlas has no rows for
                Context(19.6, -155.5, "2026-10-01")):    # Hawaii: no Atlas rows either
        both, alone = layered.parts(ctx), inat.parts(ctx)
        assert np.array_equal(both.out(1500.0), alone.out(1500.0))
        assert np.array_equal(both.density, alone.density)


def test_until_the_rank_factor_is_learned_the_place_term_is_inat_s(export, tmp_path):
    store = build_store(tmp_path)[0]
    inat = OccurrencePrior(store, OccParams())
    layered = LayeredSource(AtlasRangeSource(export, inat.params), inat)
    layered.fit([], GROUPS)
    ctx = Context(*EAST, "2026-10-01")
    assert np.array_equal(layered.parts(ctx).density, inat.parts(ctx).density)
