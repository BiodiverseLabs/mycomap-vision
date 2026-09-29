"""The spelling rule, held to the fixture it shares with mycomap.org
(services/nameVariants.ts and its test-support/name-variants.json)."""

import json
from pathlib import Path

import pytest

from mycomap_vision.names import (MECHANICAL, NameCount, fold_name, group_name_variants,
                                  is_name_like, label_map, parse_name, reasons_for, sort_key,
                                  summary, variant_key)

SHARED = json.loads((Path(__file__).parent / "data" / "name-variants.json")
                    .read_text(encoding="utf-8"))


def rows(pairs, source="iNaturalist"):
    return [NameCount(name, source, count) for name, count in pairs]


def test_the_fixture_copy_has_every_part_of_the_shared_file():
    assert set(SHARED) == {"about", "names", "notNames", "groups", "noGroup"}
    assert len(SHARED["names"]) >= 35 and len(SHARED["notNames"]) >= 8
    assert len(SHARED["groups"]) >= 8 and len(SHARED["noGroup"]) >= 5
    for c in SHARED["names"]:
        assert set(c) == {"name", "key", "houseStyle", "isCode"}
    for g in SHARED["groups"]:
        assert set(g) == {"rule", "rows", "expect"}
        assert set(g["expect"]) == {"proposedName", "proposedIsNew", "confidence",
                                    "recordsToFix"}
    assert "names.py" in SHARED["about"] and "nameVariants.ts" in SHARED["about"]


@pytest.mark.parametrize("case", SHARED["names"], ids=lambda c: ascii(c["name"]))
def test_a_name_is_keyed_and_written_as_the_shared_rules_say(case):
    parts = parse_name(case["name"])
    assert parts.key == case["key"]
    assert parts.house_style == case["houseStyle"]
    assert (parts.code is not None) == case["isCode"]


@pytest.mark.parametrize("name", SHARED["notNames"], ids=ascii)
def test_what_is_not_a_name_is_never_a_name_to_fix(name):
    assert is_name_like(name) is False
    assert group_name_variants(rows([(name, 5)])) == []


@pytest.mark.parametrize("case", SHARED["groups"], ids=lambda g: g["rule"])
def test_spellings_group_as_the_shared_rules_say(case):
    groups = group_name_variants(rows(case["rows"]))
    assert len(groups) == 1
    g = groups[0]
    assert {"proposedName": g.proposed_name, "proposedIsNew": g.proposed_is_new,
            "confidence": g.confidence, "recordsToFix": g.records_to_fix} == case["expect"]


@pytest.mark.parametrize("case", SHARED["noGroup"], ids=lambda c: " / ".join(r[0] for r in c))
def test_different_names_are_never_grouped(case):
    assert group_name_variants(rows(case)) == []


def test_the_spelling_to_keep_is_never_one_to_fix_and_every_other_says_why_it_differs():
    [group] = group_name_variants(rows([("Mycena sp. 'IN22'", 175), ('Mycena "sp-IN22"', 11),
                                        ("Mycena \u201csp-IN22\u201d", 8)]))
    assert group.proposed_name == "Mycena sp. 'IN22'"
    assert [(s.name, s.keep) for s in group.spellings] == [
        ("Mycena sp. 'IN22'", True), ('Mycena "sp-IN22"', False),
        ("Mycena \u201csp-IN22\u201d", False)]
    assert group.spellings[0].reasons == []
    assert "sp_prefix" in group.spellings[1].reasons
    assert "quotes" in group.spellings[2].reasons
    assert group.records_to_fix == 19
    assert group.records == 194


def test_records_are_counted_per_source_and_one_spelling_from_two_sources_is_one_spelling():
    [group] = group_name_variants([
        NameCount("Tubaria sp. 'IN01'", "iNaturalist", 250),
        NameCount('Tubaria "sp-IN01"', "iNaturalist", 20),
        NameCount('Tubaria "sp-IN01"', "Sequences", 6)])
    old = next(s for s in group.spellings if s.name == 'Tubaria "sp-IN01"')
    assert old.by_source == {"iNaturalist": 20, "Sequences": 6}
    assert old.records == 26
    assert len(group.spellings) == 2


def test_a_name_or_code_group_lists_every_spelling_of_both_and_keeps_none():
    [group] = group_name_variants(rows([("Amanita sp. 'ostendemihi'", 43),
                                        ('Amanita "ostendemihi"', 3),
                                        ("Amanita ostendemihi", 1)]))
    assert group.reasons == ["name_or_code"]
    assert len(group.spellings) == 3
    assert not any(s.keep for s in group.spellings)


def test_groups_with_the_most_records_to_fix_come_first():
    groups = group_name_variants(rows([('Zythia "sp-CA02"', 1), ("Zythia sp. 'CA02'", 1),
                                       ("Tubaria sp. 'IN01'", 250), ('Tubaria "sp-IN01"', 26)]))
    assert [g.proposed_name for g in groups] == ["Tubaria sp. 'IN01'", "Zythia sp. 'CA02'"]


def test_rows_with_no_records_or_no_name_are_ignored():
    assert group_name_variants([NameCount('Tubaria "sp-IN01"', "iNaturalist", 0),
                                NameCount(None, "iNaturalist", 4)]) == []


def test_a_lower_case_genus_is_a_capitals_difference():
    assert reasons_for("galerina laticeps", "Galerina laticeps") == ["capitals"]
    [group] = group_name_variants(rows([("Galerina laticeps", 30), ("galerina laticeps", 1)]))
    assert group.proposed_name == "Galerina laticeps"
    assert group.confidence == "same"


# --- where Python and JavaScript could part ways --------------------------------------

def test_white_space_is_what_javascript_calls_white_space():
    # A non-breaking space, a thin space, a tab and a byte-order mark are spaces...
    assert fold_name("\ufeffAmanita\xa0 muscaria\u2009var.\tguessowii ") == \
        "Amanita muscaria var. guessowii"
    # ...but a control character Python alone calls a space is not folded away.
    assert fold_name("Amanita\x1fmuscaria") == "Amanita\x1fmuscaria"
    assert not is_name_like("Amanita\x1fmuscaria")


def test_only_plain_digits_make_a_code_and_a_trailing_line_break_is_not_house_style():
    assert parse_name("Russula IN\u0663").code is None            # an Arabic-Indic digit
    assert parse_name("Russula sp. 'IN01'").is_house_style
    assert not parse_name("Russula sp. 'IN01'\n").is_house_style
    assert variant_key("Russula sp. 'IN01'\n") == variant_key("Russula sp. 'IN01'")


def test_names_sort_the_same_on_every_machine_lower_case_first():
    names = ["Ab", "aB", "ab", "a-b", "a b", "a1", "\u00e9", "e", "E", "f"]
    assert sorted(names, key=sort_key) == ["a b", "a-b", "a1", "ab", "aB", "Ab",
                                           "e", "E", "\u00e9", "f"]
    # Equal counts: the spelling that sorts first is listed first, whatever the input order.
    a, b = "Russula sp. 'green madrone'", "Russula sp. 'Green Madrone'"
    for pairs in ([(a, 3), (b, 3)], [(b, 3), (a, 3)]):
        [group] = group_name_variants(rows(pairs))
        assert [s.name for s in group.spellings] == [a, b]
        assert group.proposed_name == a


# --- labels ---------------------------------------------------------------------------

COUNTS = {
    "Inocybe sp. 'PNW18'": 20, "Inocybe PNW18": 3,                     # same
    'Galerina "sp-CA02"': 24,                                          # same, new spelling
    "Tricholoma lutescentifolium\xa0": 7,                              # same, folded
    "Mycena sp. 'IN7'": 88, "Mycena sp. 'IN07'": 16,                   # check: zero padding
    "Craterellus sp. 'neotubaeformis'": 55, "Craterellus neotubaeformis": 19,   # check
    "Neoboletus sp. 'AZ brown 01'": 9, "Neoboletus sp. 'AZ-brown-01'": 1,       # check
    "Amanita muscaria": 500, "Amanita": 4, "sample mixup": 1,
}


def test_spellings_that_differ_only_in_writing_share_one_label():
    labels = label_map(COUNTS)
    assert labels["Inocybe PNW18"] == labels["Inocybe sp. 'PNW18'"] == "Inocybe sp. 'PNW18'"
    assert labels['Galerina "sp-CA02"'] == "Galerina sp. 'CA02'"
    assert labels["Tricholoma lutescentifolium\xa0"] == "Tricholoma lutescentifolium"


def test_names_a_person_has_to_decide_keep_their_own_labels():
    labels = label_map(COUNTS)
    for name in ["Mycena sp. 'IN7'", "Mycena sp. 'IN07'", "Craterellus sp. 'neotubaeformis'",
                 "Craterellus neotubaeformis", "Neoboletus sp. 'AZ brown 01'",
                 "Neoboletus sp. 'AZ-brown-01'"]:
        assert labels[name] == name


def test_every_counted_name_has_a_label_and_the_rest_are_themselves():
    labels = label_map(COUNTS)
    assert set(labels) == set(COUNTS)
    for name in ["Amanita muscaria", "Amanita", "sample mixup"]:
        assert labels[name] == name


def test_only_differences_in_writing_are_merged():
    # The guard behind label_map: a group is "same" only when every reason is mechanical.
    for g in group_name_variants((n, "", c) for n, c in COUNTS.items()):
        assert (g.confidence == "same") == all(r in MECHANICAL for r in g.reasons)
    assert "zero_padding" not in MECHANICAL and "name_or_code" not in MECHANICAL
    assert "descriptive_code" not in MECHANICAL


def test_the_summary_counts_what_is_merged_and_what_waits_for_a_person():
    counted_rows = [(n, "inat", c) for n, c in COUNTS.items()]
    s = summary(counted_rows, group_name_variants(counted_rows))
    assert s["names"] == 13 and s["labels"] == 12
    assert s["groups"] == 6
    assert s["merged_in_vision"] == {"groups": 3, "spellings": 4, "records": 54,
                                     "records_relabelled": 34}
    assert s["left_for_a_person"] == {"groups": 3, "spellings": 6, "records": 188}
    assert s["records_to_fix"] == 34 + 88 + 74 + 9
