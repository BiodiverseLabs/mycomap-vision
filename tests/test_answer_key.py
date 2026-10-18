"""The species answer key (Steve, 2026-10-18; V2-REQUIREMENTS 3a): a variety or subspecies
counts as its species, a name's writings are one name, and a hybrid code is answered right by
either parent. One function (name_equiv.species_credit / first_credited) for every comparator,
so a model, iNat's computer vision and an expert get the same credit for the same names. The
rule "exact" keeps v1's published numbers reproducible."""

import numpy as np
import pytest

from mycomap_vision import evaluate, name_equiv as ne
from mycomap_vision.evaluate import Record


@pytest.mark.parametrize("answer,truth", [
    ("Amanita muscaria subsp. flavivolvata", "Amanita muscaria var. flavivolvata"),
    ("Amanita muscaria ssp. flavivolvata", "Amanita muscaria"),
    ("Lactarius quietus var. incanus", "Lactarius quietus"),            # Steve's examples
    ("Pseudohydnum gelatinosum ssp. pusillum", "Pseudohydnum gelatinosum"),
    ("Chromosera lilacifolia var. lilacifolia", "Chromosera lilacifolia var. totililacicolor"),
    ("Mycena epipterygia lignicola", "Mycena epipterygia var. lignicola"),  # bare trinomial
    ("Amanita muscaria flavivolvata", "Amanita muscaria var. flavivolvata"),  # iNat's writing
])
def test_an_infraspecific_name_counts_as_its_species(answer, truth):
    assert ne.species_credit(answer, truth) and ne.species_credit(truth, answer)
    assert not ne.species_credit(answer, truth, "exact")


@pytest.mark.parametrize("answer,truth", [
    ("Mycena sp. 'IN7'", "Mycena sp. 'IN07'"),
    ("Physisporinus sp. 'IN01 x VT01'", "Physisporinus sp. 'IN01-x-VT01'"),
    ("Neoboletus sp. 'AZ brown 01'", "Neoboletus sp. 'AZ-brown-01'"),
    ("Callistosporium luteo-olivaceum", "Callistosporium luteoolivaceum"),
])
def test_writings_of_one_name_are_one_name(answer, truth):
    assert ne.species_credit(answer, truth)


def test_either_parent_answers_a_hybrid_code_but_not_the_other_way():
    hybrid = "Physisporinus sp. 'IN01-x-VT01'"
    assert ne.hybrid_parents(hybrid) == ("Physisporinus sp. 'IN01'", "Physisporinus sp. 'VT01'")
    assert ne.species_credit("Physisporinus sp. 'VT01'", hybrid)
    assert ne.species_credit("Physisporinus sp. 'IN1'", hybrid)          # padding folded too
    assert not ne.species_credit("Physisporinus sp. 'PR01'", hybrid)
    assert not ne.species_credit(hybrid, "Physisporinus sp. 'VT01'")      # a truth's parent
    assert not ne.species_credit("Physisporinus sp. 'VT01'", hybrid, "exact")


@pytest.mark.parametrize("answer,truth", [
    ("Laccaria nobilis", "Laccaria laccata group"),     # a group is fixed at source,
    ("Laccaria laccata", "Laccaria laccata group"),     # never cut to a species in scoring
    ("Russula emetica", "Russula nobilis"),
    ("Amanita muscaria", "Amanita pantherina"),
    ("Hygrophorus sp. 'chrysodon-CA01'", "Hygrophorus chrysodon"),   # a crosswalk, at source
])
def test_other_names_and_groups_get_no_credit(answer, truth):
    assert not ne.species_credit(answer, truth)


def test_answers_sharing_a_key_are_one_answer_in_a_ranked_list():
    ranked = ["Amanita muscaria subsp. flavivolvata", "Amanita muscaria var. guessowii",
              "Amanita muscaria", "Amanita pantherina"]
    assert ne.collapse(ranked) == ["Amanita muscaria subsp. flavivolvata", "Amanita pantherina"]
    assert ne.first_credited(ranked, "Amanita pantherina") == 1          # second, not fourth
    assert ne.first_credited(ranked, "Amanita pantherina", "exact") == 3
    assert ne.first_credited(ranked, "Boletus edulis") is None


def test_a_comparator_answer_gets_the_same_credit_as_a_vision_answer():
    """The paper's condition: every comparator is scored by the one function. iNat writes
    a subspecies without its rank word and an expert may write 'ssp.'; for the same names
    they get exactly Vision's credit and place."""
    truth = "Amanita muscaria var. flavivolvata"
    vision = ["Amanita muscaria subsp. flavivolvata", "Amanita pantherina"]
    inat_cv = ["Amanita muscaria flavivolvata", "Amanita pantherina"]
    expert = ["Amanita muscaria ssp. flavivolvata"]
    assert ne.first_credited(vision, truth) == ne.first_credited(inat_cv, truth) \
        == ne.first_credited(expert, truth) == 0
    for a, b in zip(vision, inat_cv):
        assert ne.species_credit(a, truth) == ne.species_credit(b, truth)
    assert ne.species_match(inat_cv[0], truth)["strict"]


def test_the_rule_must_be_known():
    with pytest.raises(ValueError, match="unknown answer key"):
        ne.answer_key("Amanita muscaria", "lenient")


def unit(a):
    return [np.cos(a), np.sin(a)]


def scored(answer_key=None):
    # The reference holds the species under its subspecies; the truth is written 'var.'.
    vectors = np.asarray([unit(0.0), unit(2.0), unit(0.05)], dtype=np.float32)
    ref = [Record("1", "Amanita muscaria subsp. flavivolvata", "Amanita", "Am", "2026-01-01",
                  "a", [0]),
           Record("2", "Boletus edulis", "Boletus", "Bo", "2026-01-01", "a", [1])]
    test = [Record("9", "Amanita muscaria var. flavivolvata", "Amanita", "Am", "2026-09-01",
                   "b", [2])]
    return evaluate.evaluate(vectors, ref, test, method="nearest", name_scores=True,
                             sets=False, answer_key=answer_key)


def test_evaluate_scores_species_by_the_answer_key_and_exact_reproduces_v1():
    out = scored()
    assert out["answer_key"] == ne.ANSWER_KEY
    assert out["species"]["all"]["top1"] == 1.0
    assert out["standard"]["species"]["all"]["top1"] == 1.0
    assert out["standard_equivalence"]["species"]["strict"]["top1"] == 1.0
    assert out["groups"]["observer"][0]["species_top1"] == 1.0
    old = scored("exact")
    assert old["answer_key"] == "exact"
    assert old["species"]["all"]["top1"] == 0.0 and old["standard"]["species"]["all"]["top1"] == 0.0
    assert old["standard_equivalence"]["species"]["strict"]["top1"] == 0.0
    assert old["genus"]["all"]["top1"] == out["genus"]["all"]["top1"] == 1.0   # genus as before


# --- the optional frozen-iNat-taxonomy step (the paper coordinator, 2026-10-18) ------------

def t(name, level, active=True, anc=(), succ=None):
    d = {"name": name, "rank_level": level, "is_active": active, "ancestor_ids": list(anc)}
    if succ is not None:
        d["current_synonymous_taxon_ids"] = succ
    return d


TAXA = {
    1: t("Fungi", 70),
    10: t("Psathyrella", 20, anc=[1]), 11: t("Candolleomyces", 20, anc=[1]),
    12: t("Coprinellus", 20, anc=[1]),
    100: t("Psathyrella candolleana", 10, active=False, anc=[1, 10], succ=[101]),
    101: t("Candolleomyces candolleanus", 10, anc=[1, 11]),
    200: t("Amanita", 20, anc=[1]), 201: t("Amanita muscaria", 10, anc=[1, 200]),
    202: t("Amanita muscaria flavivolvata", 5, anc=[1, 200, 201]),
    300: t("Boletus edulis", 10, anc=[1]),
    301: t("Boletus grandedulis", 10, active=False, anc=[1], succ=[300]),
    400: t("Russula emetica", 10, active=False, anc=[1], succ=[401, 402]),   # a split
    401: t("Russula one", 10, anc=[1]), 402: t("Russula two", 10, anc=[1]),
}
LABELS = ["Psathyrella candolleana", "Psathyrella candolleana group", "Amanita muscaria",
          "Amanita muscaria var. flavivolvata",
          "Boletus edulis", "Boletus grandedulis", "Russula emetica", "Amanita sp. 'NY03'"]


@pytest.fixture
def tax():
    return ne.Taxonomy(TAXA, LABELS, cite={"file": "taxa.csv.gz", "sha256": "x"})


def test_a_comparator_in_inats_newer_names_is_not_wrong_for_the_taxonomys_version(tax):
    truth = "Psathyrella candolleana"
    for answer in (101, "101", 100, "Candolleomyces candolleanus", "Psathyrella candolleana"):
        assert ne.species_credit(answer, truth, taxonomy=tax), answer
    assert ne.canonical(truth, tax) == ne.canonical(101, tax) == "inat:101"
    vision = ["Psathyrella candolleana", "Amanita muscaria"]
    inat = [101, 201]
    assert ne.first_credited(vision, truth, taxonomy=tax) == ne.first_credited(inat, truth, taxonomy=tax) == 0
    assert ne.species_match(101, truth, tax) == {"strict": True, "sl": True, "complex": True}


def test_an_infraspecific_taxon_id_counts_as_its_species(tax):
    assert ne.species_credit(202, "Amanita muscaria", taxonomy=tax)
    assert ne.species_credit("Amanita muscaria var. flavivolvata", 201, taxonomy=tax)


def test_labels_sharing_an_inat_taxon_stay_apart_one_to_one_only(tax):
    assert tax.shared_by == {300: {ne.answer_key("Boletus edulis"),
                                   ne.answer_key("Boletus grandedulis")}}
    assert not ne.species_credit("Boletus edulis", "Boletus grandedulis", taxonomy=tax)
    assert not ne.species_credit(300, "Boletus grandedulis", taxonomy=tax)
    assert ne.species_credit("Boletus edulis", "Boletus edulis", taxonomy=tax)   # as names


def test_a_group_label_is_no_species_and_does_not_take_its_species_resolution(tax):
    assert tax.species_id("Psathyrella candolleana group") is None
    assert tax.species_id("Psathyrella candolleana") == 101        # not made ambiguous
    assert not ne.species_credit(101, "Psathyrella candolleana group", taxonomy=tax)


def test_a_split_or_an_unknown_name_falls_back_to_the_string(tax):
    assert tax.species_id("Russula emetica") is None             # two successors: no guess
    assert ne.canonical("Russula emetica", tax) == ne.answer_key("Russula emetica")
    assert ne.species_credit("Amanita sp. 'NY03'", "Amanita sp. 'NY3'", taxonomy=tax)
    assert ne.canonical(999, tax) == "taxon:999" == ne.canonical(999)


def test_the_truth_genus_accepts_its_label_genus_and_its_resolved_species_genus(tax):
    """PLAN rule 4, both ways: Vision's 'Psathyrella' and an identifier's Candolleomyces
    (as a taxon id) are both right for Psathyrella candolleana; Coprinellus is not."""
    kw = {"taxonomy": tax, "truth_species": "Psathyrella candolleana"}
    assert ne.genus_match("Psathyrella", "Psathyrella", **kw)["strict"]
    assert ne.genus_match(11, "Psathyrella", **kw)["strict"]
    assert ne.genus_match("Candolleomyces", "Psathyrella", **kw)["strict"]
    assert not ne.genus_match("Coprinellus", "Psathyrella", **kw)["strict"]
    assert not ne.genus_match(12, "Psathyrella", **kw)["strict"]
    assert not ne.genus_match("Candolleomyces", "Psathyrella")["strict"]     # no taxonomy


def test_without_a_taxonomy_nothing_changes():
    assert ne.species_credit("Candolleomyces candolleanus", "Psathyrella candolleana") is False
    assert ne.species_credit(101, "Psathyrella candolleana") is False


def test_the_snapshot_loads_with_its_successors_and_is_cited(tmp_path):
    import gzip
    taxa = tmp_path / "taxa.csv.gz"
    with gzip.open(taxa, "wt", encoding="utf-8") as f:
        f.write("taxon_id\tancestry\trank_level\trank\tname\tactive\n"
                "47170\t48460\t70\tkingdom\tFungi\ttrue\n"
                "11\t48460/47170\t20\tgenus\tCandolleomyces\ttrue\n"
                "100\t48460/47170\t10\tspecies\tPsathyrella candolleana\tfalse\n"
                "101\t48460/47170/11\t10\tspecies\tCandolleomyces candolleanus\ttrue\n")
    succ = tmp_path / "successors.tsv"
    succ.write_text("taxon_id\tcurrent_ids\n100\t101\n", encoding="utf-8")
    tax = ne.Taxonomy.load(taxa, ["Psathyrella candolleana"], succ)
    assert ne.species_credit(101, "Psathyrella candolleana", taxonomy=tax)
    assert tax.cite["taxa"] == "taxa.csv.gz" and len(tax.cite["successors_sha256"]) == 64
    without = ne.Taxonomy.load(taxa, ["Psathyrella candolleana"])
    assert not ne.species_credit(101, "Psathyrella candolleana", taxonomy=without)


# --- rule 4 gaps found by the comparisons lane (E31b), each a test --------------------------

GAP_TAXA = {**TAXA,
            13: t("Gyromitra", 20, anc=[1]), 14: t("Neogyromitra", 20, anc=[1]),
            500: t("Gyromitra caroliniana", 10, active=False, anc=[1, 13], succ=[501]),
            501: t("Neogyromitra caroliniana", 10, anc=[1, 14]),
            600: t("Clavaria", 20, anc=[1]),
            601: t("Clavaria zollingeri", 10, active=False, anc=[1, 600], succ=[602, 603]),
            602: t("Clavaria one", 10, anc=[1, 600]), 603: t("Clavaria two", 10, anc=[1, 600])}
GAP_LABELS = ["Psathyrella candolleana", "Candolleomyces candolleanus",
              "Gyromitra caroliniana", "Neogyromitra caroliniana", "Clavaria zollingeri"]


@pytest.fixture
def gap():
    return ne.Taxonomy(GAP_TAXA, GAP_LABELS)


def test_gap1_the_truth_accepts_its_labels_own_genus_word(gap):
    # 'Candolleomyces candolleanus' filed under genus Psathyrella (release genus)
    kw = {"taxonomy": gap, "truth_species": "Candolleomyces candolleanus"}
    assert ne.genus_match(11, "Psathyrella", **kw)["strict"]          # the Candolleomyces id
    assert ne.genus_match(101, "Psathyrella", **kw)["strict"]         # a species id
    assert ne.genus_match("Psathyrella", "Psathyrella", **kw)["strict"]
    assert not ne.genus_match("Coprinellus", "Psathyrella", **kw)["strict"]
    # only the label's own word can say it: a code iNat doesn't know, filed under Psathyrella
    code = {"taxonomy": gap, "truth_species": "Candolleomyces sp. 'X01'"}
    assert ne.genus_match("Candolleomyces", "Psathyrella", **code)["strict"]


def test_gap2_a_kept_apart_label_keeps_its_inat_genus(gap):
    assert gap.species_id("Gyromitra caroliniana") is None              # kept apart for species
    kw = {"taxonomy": gap, "truth_species": "Gyromitra caroliniana"}
    assert ne.genus_match(501, "Gyromitra", **kw)["strict"]           # resolves to Neogyromitra
    assert ne.genus_match(14, "Gyromitra", **kw)["strict"]
    assert ne.genus_match("Gyromitra", "Gyromitra", **kw)["strict"]


def test_gap3_a_split_taxon_keeps_its_own_genus(gap):
    assert gap.species_id(601) is None                                  # no species credit
    assert ne.genus_match(601, "Clavaria", gap, "Clavaria zollingeri")["strict"]
    assert not ne.species_credit(601, "Clavaria zollingeri", taxonomy=gap)


def test_a_kept_apart_truth_does_not_resolve_so_it_leaves_the_paired_tests(gap):
    """Rule 7: paired tests use formal truths that resolve to an iNat species. 'Psathyrella
    candolleana' is kept apart from 'Candolleomyces candolleanus' (one iNat species): it
    reports resolves False, so iNat's Candolleomyces answer is not scored a miss on it."""
    assert gap.resolves("Psathyrella candolleana") is False
    assert gap.resolves("Candolleomyces candolleanus") is False
    assert not ne.species_credit(101, "Psathyrella candolleana", taxonomy=gap)
    one = ne.Taxonomy(GAP_TAXA, ["Psathyrella candolleana"])
    assert one.resolves("Psathyrella candolleana") is True
    assert one.resolves("Amanita sp. 'NY03'") is False                 # codes never resolve


def test_answer_side_extras_never_change_how_a_label_resolves():
    base = {**TAXA, 801: t("Inocybe olda", 10, active=False, anc=[1], succ=[802])}
    plain2 = ne.Taxonomy(base, LABELS + ["Inocybe olda"])
    more2 = ne.Taxonomy(base, LABELS + ["Inocybe olda"],
                        extras={802: t("Inocybe nova", 10, anc=[1])})   # its successor, newer
    assert plain2.species_id("Inocybe olda") is None is more2.species_id("Inocybe olda")
    assert more2.species_id(801) == 802                                 # as an answer it reads
    extras = {700: t("Psathyrella candolleana", 10, anc=[1, 10]),     # a newer homonym
              701: t("Russula three", 10, anc=[1]),
              400: t("changed", 10)}                                   # can't override the snapshot
    plain = ne.Taxonomy(TAXA, LABELS)
    more = ne.Taxonomy(TAXA, LABELS, extras=extras)
    for lab in LABELS:
        assert plain.species_id(lab) == more.species_id(lab), lab
    assert more.species_id(701) == 701 and plain.species_id(701) is None   # read as an answer
    assert more.taxa[400]["name"] == "Russula emetica"


class FakeRelease:
    def __init__(self, labels):
        self.rows = [{"label": lab} for lab in labels]

    def records(self, splits):
        return self.rows

    def cite(self):
        return {"dataset_release": "v1", "errata": {"file": "v1-errata-r4.tsv"}}


def test_the_label_vocabulary_is_written_once_sorted_and_pinned(tmp_path):
    rel = FakeRelease(["Russula emetica", "Amanita", "Amanita sp. 'NY03'", "Russula emetica",
                       "Boletus edulis var. grandedulis", ""])
    out = ne.write_label_vocabulary(rel, tmp_path / "labels.txt")
    assert out["labels"] == 4 and out["errata"] == "v1-errata-r4.tsv" and len(out["sha256"]) == 64
    assert ne.read_label_vocabulary(tmp_path / "labels.txt") == sorted(
        ["Russula emetica", "Amanita", "Amanita sp. 'NY03'", "Boletus edulis var. grandedulis"])
    with pytest.raises(FileExistsError):
        ne.write_label_vocabulary(rel, tmp_path / "labels.txt")
