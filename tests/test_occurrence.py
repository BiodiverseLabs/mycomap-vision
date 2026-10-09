import gzip

import numpy as np
import pytest

from mycomap_vision.occurrence import (BuildOptions, Grid, OccurrenceStore, build, name_key,
                                       read_taxa, week_of)
from occurrence_fixtures import (EAST, FUNGI, HEADER, MYCETOZOA, WEST, build_store, obs,
                                 write_observations, write_taxa)


def unit(store, name):
    return store.names.index(name)


def total(store, name):
    return int(store.unit_total[unit(store, name)])


def test_fungi_and_lichens_are_kept_animals_and_slime_molds_are_not(tmp_path):
    rows = [obs("f", 104, *EAST), obs("l", 122, *EAST), obs("a", 3, *EAST), obs("s", 901, *EAST)]
    store, meta = build_store(tmp_path, rows)
    assert total(store, "Amanita orientalis") == 1
    assert total(store, "Parmelia sulcata") == 1          # lichens are Fungi on iNat
    assert "Animalus vulgaris" not in store.names
    assert "Physarum polycephalum" not in store.names
    assert meta["stats"]["kept"] == 2 and meta["stats"]["not_kept_taxon"] == 2
    assert int(store.effort_cell.sum()) == 2


def test_slime_molds_count_only_with_the_flag(tmp_path):
    rows = [obs("s", 901, *EAST)]
    store, _ = build_store(tmp_path, rows, with_slime_molds=True)
    assert total(store, "Physarum polycephalum") == 1


def test_only_research_and_needs_id_observations_with_coordinates_count(tmp_path):
    rows = [obs("r", 104, *EAST, grade="research"), obs("n", 104, *EAST, grade="needs_id"),
            obs("c", 104, *EAST, grade="casual"), obs("x", 104, "", "")]
    store, meta = build_store(tmp_path, rows)
    assert total(store, "Amanita orientalis") == 2
    assert meta["stats"]["grade"] == 1 and meta["stats"]["no_coordinates"] == 1


def test_only_north_america_and_its_margin_count(tmp_path):
    rows = [obs("na", 104, *EAST), obs("margin", 104, 3.5, -80.0),      # 3.5 N: inside 7 - 5
            obs("europe", 104, 48.0, 2.0), obs("south", 104, -20.0, -60.0)]
    store, meta = build_store(tmp_path, rows)
    assert total(store, "Amanita orientalis") == 2
    assert meta["stats"]["outside_box"] == 2
    store, _ = build_store(tmp_path, rows, margin_deg=0.0)
    assert total(store, "Amanita orientalis") == 1


def test_imprecise_locations_are_left_out_but_a_blank_accuracy_is_kept(tmp_path):
    rows = [obs("ok", 104, *EAST, acc="100"), obs("far", 104, *EAST, acc="50000"),
            obs("blank", 104, *EAST, acc="")]
    store, meta = build_store(tmp_path, rows)
    assert total(store, "Amanita orientalis") == 2 and meta["stats"]["accuracy"] == 1


def test_excluded_uuids_are_never_counted(tmp_path):
    rows = [obs("keep-1", 104, *EAST), obs("HELD-OUT-2", 104, *EAST), obs("held-out-3", 105, *WEST)]
    store, meta = build_store(tmp_path, rows, exclude=["held-out-2", "HELD-OUT-3", "never-seen"])
    assert total(store, "Amanita orientalis") == 1
    assert total(store, "Amanita occidentalis") == 0
    assert int(store.effort_cell.sum()) == 1        # nor as sampling effort
    assert meta["stats"]["excluded"] == 2 and meta["excluded_uuids"] == 3
    assert store.excludes(["held-out-2", "Held-Out-3", "keep-1"]) == [True, True, False]


def test_an_exclusion_list_needs_a_uuid_column_to_apply(tmp_path):
    taxa = write_taxa(tmp_path / "taxa.csv.gz")
    header = [h for h in HEADER if h != "observation_uuid"]
    observations = write_observations(tmp_path / "o.csv.gz", [obs("u", 104, *EAST)[1:]], header)
    (tmp_path / "x.txt").write_text("u\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no observation_uuid column"):
        build(observations, taxa, tmp_path / "s.npz", BuildOptions(), [tmp_path / "x.txt"],
              log=lambda *a: None)


def test_a_variety_counts_for_its_species_and_everything_counts_for_its_genus(tmp_path):
    rows = [obs("v", 103, *EAST), obs("s", 102, *EAST), obs("c", 107, *EAST),
            obs("g", 101, *EAST), obs("f", 100, *EAST)]
    store, _ = build_store(tmp_path, rows)
    assert total(store, "Amanita muscaria guessowii") == 1
    assert total(store, "Amanita muscaria") == 2          # its own and its variety's
    assert total(store, "Amanita") == 4                   # all but the family-level one
    assert int(store.effort_cell.sum()) == 5              # effort: every fungus


def test_header_names_and_order_and_commas_are_tolerated(tmp_path):
    taxa = write_taxa(tmp_path / "taxa.csv.gz")
    path = tmp_path / "obs.csv"
    path.write_text('"Quality_Grade","taxon_id","Latitude","longitude","observed_on",'
                    '"observation_uuid"\n"research","104","40.0","-80.0","2025-10-01","a"\n'
                    '"research","104","40.1","-80.0","","b"\n', encoding="utf-8")
    meta = build(path, taxa, tmp_path / "s.npz", BuildOptions(), log=lambda *a: None)
    assert meta["stats"]["kept"] == 2


def test_season_counts_by_week_and_the_epoch_placeholder_is_no_date(tmp_path):
    assert week_of("2025-01-01") == 0 and week_of("2025-12-31") == 52
    assert week_of("1970-01-01") == -1 and week_of("") == -1 and week_of("2025-13-40") == -1
    rows = [obs("a", 104, *EAST, date="2025-10-01"), obs("b", 104, *EAST, date="1970-01-01")]
    store, _ = build_store(tmp_path, rows)
    u = unit(store, "Amanita orientalis")
    mine = store.week_unit == u
    assert store.week_count[mine].sum() == 1
    assert store.week_week[mine][0] == week_of("2025-10-01")


def test_observations_after_a_cutoff_date_can_be_left_out(tmp_path):
    rows = [obs("a", 104, *EAST, date="2025-10-01"), obs("b", 104, *EAST, date="2026-09-08")]
    store, meta = build_store(tmp_path, rows, observed_before="2026-09-08")
    assert total(store, "Amanita orientalis") == 1 and meta["stats"]["after_date"] == 1


def test_the_grid_covers_the_box_and_margin():
    g = Grid.for_options(BuildOptions())
    assert g.contains(*EAST) and g.contains(*WEST) and g.contains(19.6, -155.5)    # Hawaii
    assert not g.contains(48.0, 2.0)
    lat, lon = g.centres(np.array([g.cell(*EAST)]))
    assert abs(lat[0] - EAST[0]) <= 0.25 and abs(lon[0] - EAST[1]) <= 0.25


def test_taxa_parsing_finds_the_kingdoms_it_was_asked_for(tmp_path):
    t = read_taxa(write_taxa(tmp_path / "taxa.csv.gz"), (FUNGI, MYCETOZOA))
    assert t.roots == {FUNGI: "Fungi", MYCETOZOA: "Mycetozoa"}
    p = tmp_path / "bad.csv.gz"
    with gzip.open(p, "wt") as f:
        f.write("taxon_id\tancestry\trank_level\trank\tname\tactive\n1\t\t70\tkingdom\tAnimalia\ttrue\n")
    with pytest.raises(ValueError, match="no taxon"):
        build(p, p, tmp_path / "s.npz", BuildOptions(), log=lambda *a: None)


# --- Vision's names -> iNat taxa ------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return build_store(tmp_path)[0]


def test_an_active_name_maps_to_its_taxon_and_rank_words_are_ignored(store):
    m = store.resolve("Amanita muscaria")
    assert store.names[m.species_unit] == "Amanita muscaria" and m.how == "species"
    m = store.resolve("Amanita muscaria var. guessowii")
    assert store.names[m.species_unit] == "Amanita muscaria guessowii"
    assert store.names[m.genus_unit] == "Amanita"
    assert name_key("Amanita  muscaria var. Guessowii") == "amanita muscaria guessowii"


def test_an_inactive_name_maps_to_the_one_active_species_of_its_epithet_and_family(store):
    m = store.resolve("Lepista nuda", "Lepista")
    assert m.how == "synonym" and store.names[m.species_unit] == "Collybia nuda"
    assert store.names[m.genus_unit] == "Collybia"      # the taxon's own genus on iNat


def test_a_provisional_name_has_no_species_taxon_only_its_genus(store):
    m = store.resolve("Amanita sp. 'IN01'", "Amanita")
    assert m.species_unit == -1 and store.names[m.genus_unit] == "Amanita"
    assert m.how == "provisional"


def test_a_name_inat_does_not_know_is_never_given_a_guessed_species(store):
    m = store.resolve("Amanita inventa", "Amanita")
    assert m.species_unit == -1 and m.how == "not-on-inat"
    assert store.names[m.genus_unit] == "Amanita"
    m = store.resolve("Nullus nihil", "Nullus")
    assert (m.species_unit, m.genus_unit, m.how) == (-1, -1, "none")


def test_a_one_word_label_maps_to_its_genus(store):
    m = store.resolve("Westia")
    assert m.species_unit == -1 and store.names[m.genus_unit] == "Westia"
    assert m.how == "genus-label"


def test_a_missing_store_says_how_to_build_one(tmp_path):
    with pytest.raises(FileNotFoundError, match="mv build-occurrence"):
        OccurrenceStore.load(tmp_path / "none.npz")
