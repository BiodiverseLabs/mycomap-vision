"""What the mistakes are (paper_errors.py, paper E29 and E21): confused pairs, how far a
wrong answer lands in the taxonomy, and deadly taxa called by other names."""

from mycomap_vision import paper_errors as pe
from mycomap_vision import paper_outcomes as po


def find(key, species):
    genus = species.split(" ")[0]
    return po.Find(key, "o", species, genus, "", "formal", 5, 1)


RANKS = {"Amanita": {"family": "Amanitaceae", "order": "Agaricales"},
         "Limacella": {"family": "Amanitaceae", "order": "Agaricales"},
         "Agaricus": {"family": "Agaricaceae", "order": "Agaricales"},
         "Boletus": {"family": "Boletaceae", "order": "Boletales"}}


def test_confused_pairs_count_wrong_answers_by_truth_and_answer():
    finds = [find(str(i), "Amanita bisporigera") for i in range(4)] + [find("9", "Boletus edulis")]
    top1 = {"0": "Amanita bisporigera", "1": "Amanita virosa", "2": "Amanita virosa",
            "3": "Agaricus campestris", "9": "Boletus edulis"}
    r = pe.confused_pairs(finds, top1)
    assert r["wrong_answers"] == 3 and r["distinct_pairs"] == 2
    assert r["top"][0] == {"truth": "Amanita bisporigera", "answer": "Amanita virosa",
                           "finds": 2, "share_of_truth": 0.5, "same_genus": True}


def test_a_wrong_answer_is_placed_by_genus_family_order():
    assert pe.distance("Amanita", "Amanita", RANKS) == "same genus"
    assert pe.distance("Amanita", "Limacella", RANKS) == "same family"
    assert pe.distance("Amanita", "Agaricus", RANKS) == "same order"
    assert pe.distance("Amanita", "Boletus", RANKS) == "further"
    assert pe.distance("Amanita", "Nogenus", RANKS) == "unknown"
    finds = [find("1", "Amanita muscaria"), find("2", "Amanita muscaria")]
    d = pe.mistake_distance(finds, {"1": "Boletus edulis", "2": "Amanita muscaria"}, RANKS)
    assert d["wrong_answers"] == 1 and d["further"] == {"finds": 1, "share": 1.0}


DEADLY = {"genera": ["Galerina"], "species": ["Amanita bisporigera", "Amanita virosa"]}


def test_deadly_finds_called_harmless_names_are_counted_as_the_dangerous_direction():
    finds = [find("1", "Amanita bisporigera"), find("2", "Amanita bisporigera"),
             find("3", "Amanita bisporigera"), find("4", "Galerina marginata"),
             find("5", "Agaricus campestris"), find("6", "Agaricus campestris")]
    top1 = {"1": "Amanita bisporigera", "2": "Amanita virosa", "3": "Agaricus campestris",
            "4": "Galerina sp-CA01", "5": "Amanita virosa"}
    r = pe.deadly_taxa(finds, top1, DEADLY)
    bis = r["predicted_from"]["Amanita bisporigera"]
    assert (bis["finds"], bis["same_taxon"], bis["other_deadly"], bis["harmless"]) == (3, 1, 1, 1)
    assert r["predicted_from"]["Galerina"]["same_taxon"] == 1       # a code in a listed genus
    assert r["predicted_as"] == {"harmless_finds": 2, "called_deadly": 1,
                                 "by_taxon": {"Amanita virosa": 1}}
    assert "not for deciding what is safe to eat" in r["note"]


def test_a_listed_species_matches_its_gender_variant_spelling():
    # s.l. folds gender endings, so a respelled epithet stays on the list.
    assert pe.deadly_match("Cortinarius orellanus", {"species": ["Cortinarius orellanus"]})
    assert pe.deadly_match("Cortinarius orellana", {"species": ["Cortinarius orellanus"]})
    assert pe.deadly_match("Cortinarius violaceus", {"species": ["Cortinarius orellanus"]}) is None


def test_the_deadly_list_is_read_from_its_pinned_file(tmp_path):
    import hashlib
    import pytest
    path = tmp_path / "deadly.csv"
    path.write_text("scientific_name\nAmanita bisporigera\nGalerina\n", encoding="utf-8")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    d = pe.read_deadly({"file": str(path), "sha256": sha, "match": ["strict"]})
    assert d["species"] == ["Amanita bisporigera"] and d["genera"] == ["Galerina"]
    assert pe.deadly_match("Galerina marginata", d) == "Galerina"
    with pytest.raises(ValueError, match="not this list"):
        pe.read_deadly({"file": str(path), "sha256": "0" * 64})


def test_strict_only_matching_drops_the_gender_variant():
    strict = {"species": ["Cortinarius orellanus"], "match": ["strict"]}
    assert pe.deadly_match("Cortinarius orellanus", strict)
    assert pe.deadly_match("Cortinarius orellana", strict) is None
