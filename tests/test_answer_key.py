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
