"""One name, several writings in two lists (writing_split.py; Steve, 2026-11-18): "format" is
what mycomap.org's Names to fix calls "Old format" (its nameVariants.ts / nameFixRecommend.ts
rules, through names.py), everything else needs a look; the proposal is always MycoMap's
current format, never only the most used writing; the stored item is never rewritten."""

import pytest

from mycomap_vision import feedback as fb
from mycomap_vision import name_context as nc
from mycomap_vision import writing_split as ws


def split(current, proposed):
    return ws.classify(current, proposed)


# --- format only ---------------------------------------------------------------------------

@pytest.mark.parametrize("current, proposed, keep", [
    # the old provisional-code writings beside today's
    ("Mycena sp. 'IN10' (456) | Mycena \"sp-IN10\" (1)", "Mycena sp. 'IN10'", "Mycena sp. 'IN10'"),
    ("Lepiota sp. 'castaneidisca-PNW13' (17) | Lepiota 'castaneidisca PNW13' (1)",
     "Lepiota sp. 'castaneidisca-PNW13'", "Lepiota sp. 'castaneidisca-PNW13'"),
    ("Russula sp. 'IN26' (26) | Russula \"sp-IN26\" (3)", "Russula sp. 'IN26'", "Russula sp. 'IN26'"),
    ("Inocybe sp. ‘PNW12’ (2) | Inocybe sp. 'PNW12' (9)", "Inocybe sp. 'PNW12'",
     "Inocybe sp. 'PNW12'"),                                      # curly quotes
    ("Hebeloma  sp. 'KMF124' (1) | Hebeloma \"sp-KMF124\" (1) | Hebeloma sp. 'KMF124' (1)",
     "Hebeloma  sp. 'KMF124'", "Hebeloma sp. 'KMF124'"),
    # the seed proposed the most used writing, which was the old format: now today's
    ("Cortinarius Harrower58 (30) | Cortinarius sp. 'Harrower58' (1)", "Cortinarius Harrower58",
     "Cortinarius sp. 'Harrower58'"),
    ("Calonarius PNW58 (1) | Calonarius sp. 'PNW58' (1)", "Calonarius PNW58",
     "Calonarius sp. 'PNW58'"),
    ('Suillellus "sp-DT17" (2) | Suillellus sp. \'DT17\' (2)', 'Suillellus "sp-DT17"',
     "Suillellus sp. 'DT17'"),
    ('Amanita "longicuneus" (2) | Amanita sp. \'longicuneus\' (1)', 'Amanita "longicuneus"',
     "Amanita sp. 'longicuneus'"),
    ("Inocybe 'lanatopurpurea-PNW73' (1) | Inocybe sp. 'lanatopurpurea-PNW73' (1)",
     "Inocybe 'lanatopurpurea-PNW73'", "Inocybe sp. 'lanatopurpurea-PNW73'"),
    ('Crepidotus "sp-IN7" (1) | Crepidotus sp. \'IN7\' (1)', 'Crepidotus "sp-IN7"',
     "Crepidotus sp. 'IN7'"),
])
def test_an_old_way_of_writing_a_code_is_format_only_and_proposes_todays_writing(
        current, proposed, keep):
    s = split(current, proposed)
    assert (s.kind, s.reason, s.proposal) == ("format", "format", keep)


def test_every_format_proposal_is_a_code_in_house_style_with_no_space_inside():
    import re
    for current in ("Mycena \"sp-IN10\" (5) | Mycena sp. 'IN10' (1)",
                    "Cortinarius Harrower58 (30) | Cortinarius sp. 'Harrower58' (1)"):
        assert re.fullmatch(r"[A-Z][a-z]+ sp\. '[A-Za-z0-9-]+'", split(current, "").proposal)


# --- needs a look ----------------------------------------------------------------------------

@pytest.mark.parametrize("current, proposed, reason, keep", [
    # two current-format numbers are two names on .org (splitPaddedNames): nothing is proposed
    # over the item's own proposal
    ("Clitocybe sp. 'IN5' (50) | Clitocybe sp. 'IN05' (5)", "Clitocybe sp. 'IN5'", "number",
     "Clitocybe sp. 'IN5'"),
    ("Amanita rubescens alba (77) | Amanita rubescens var. alba (4)", "Amanita rubescens alba",
     "rank", "Amanita rubescens alba"),
    ("Lentaria pinicola var. robusta (35) | Lentaria pinicola robusta (5)",
     "Lentaria pinicola var. robusta", "rank", "Lentaria pinicola var. robusta"),
    # only a space, hyphen or capital: .org's "Spacing", not its "Old format" filter
    ("Cuphophyllus sp. 'borealis-NY01' (70) | Cuphophyllus sp. 'borealis NY01' (1)",
     "Cuphophyllus sp. 'borealis-NY01'", "spacing", "Cuphophyllus sp. 'borealis-NY01'"),
    ("Russula sp. 'Woo18' (30) | Russula sp. 'woo18' (1)", "Russula sp. 'Woo18'", "spacing",
     "Russula sp. 'Woo18'"),
    ("Pluteus rimosellus (20) | pluteus rimosellus (1)", "Pluteus rimosellus", "spacing",
     "Pluteus rimosellus"),
    # a described code: its space may be meant (.org: "Described code"); today's writing is
    # still the one proposed, not the most used old one
    ("Lactarius “acrid orange” (32) | Lactarius sp. 'acrid-orange' (13)",
     "Lactarius “acrid orange”", "described", "Lactarius sp. 'acrid-orange'"),
])
def test_anything_more_than_the_format_needs_a_look(current, proposed, reason, keep):
    s = split(current, proposed)
    assert (s.kind, s.reason, s.proposal) == ("look", reason, keep)


def test_a_code_beside_a_plain_name_and_a_lone_writing_need_a_look():
    assert split("Craterellus neotubaeformis (3) | Craterellus sp. 'neotubaeformis' (2)",
                 "Craterellus neotubaeformis").reason == "name_or_code"
    assert split("Mycena sp. 'IN10' (3)", "Mycena sp. 'IN10'").kind == "look"


def test_writings_are_read_with_their_counts():
    assert ws.writings("A sp. 'X1' (1,204) | A \"sp-X1\" (1) | A \"sp-X1\" (1)") == [
        ("A sp. 'X1'", 1204), ('A "sp-X1"', 1)]
    assert ws.writings("A b | A c") == [("A b", None), ("A c", None)]


# --- the store: two lists, the proposal at read time --------------------------------------

def seeded(tmp_path):
    store = fb.Store(tmp_path / "f.sqlite")
    store.seed([
        {"item_type": "name_writing_split", "subject": "harrower", "rule": "R1",
         "current": "Cortinarius Harrower58 (30) | Cortinarius sp. 'Harrower58' (1)",
         "proposed": "Cortinarius Harrower58", "records": 31},
        {"item_type": "name_writing_split", "subject": "mycena", "rule": "R1",
         "current": "Mycena sp. 'IN10' (456) | Mycena \"sp-IN10\" (1)",
         "proposed": "Mycena sp. 'IN10'", "records": 457},
        {"item_type": "name_writing_split", "subject": "alba", "rule": "R1",
         "current": "Amanita rubescens alba (77) | Amanita rubescens var. alba (4)",
         "proposed": "Amanita rubescens alba", "records": 81},
    ], "test")
    return store


def test_the_queue_is_listed_as_format_only_and_needs_a_look(tmp_path):
    store = seeded(tmp_path)
    fmt = store.items("name_writing_split", list_="format")
    assert fmt["total"] == 2 and {i["subject"] for i in fmt["items"]} == {"harrower", "mycena"}
    look = store.items("name_writing_split", list_="look")
    assert [i["subject"] for i in look["items"]] == ["alba"]
    assert store.items("name_writing_split")["total"] == 3
    t = next(t for t in store.summary() if t["type"] == "name_writing_split")
    assert t["lists"] == {"format": {"open": 2, "decided": 0}, "look": {"open": 1, "decided": 0}}
    with pytest.raises(fb.FeedbackError):
        store.items("name_writing_split", list_="other")
    with pytest.raises(fb.FeedbackError):
        store.items("name_retired", list_="format")


def test_the_proposal_is_worked_out_on_read_and_the_stored_item_is_not_rewritten(tmp_path):
    store = seeded(tmp_path)
    i = next(i for i in store.items("name_writing_split")["items"] if i["subject"] == "harrower")
    assert i["proposed"] == "Cortinarius sp. 'Harrower58'"
    assert i["proposed_seeded"] == "Cortinarius Harrower58"
    assert (i["list"], i["list_reason"]) == ("format", "format")
    assert store.conn.execute("select proposed from items where id = ?",
                              (i["id"],)).fetchone()[0] == "Cortinarius Harrower58"
    m = next(i for i in store.items("name_writing_split")["items"] if i["subject"] == "mycena")
    assert "proposed_seeded" not in m                       # already today's writing


def test_a_writing_split_card_asks_inat_about_each_writing():
    item = {"item_type": "name_writing_split", "proposed": "Cortinarius sp. 'Harrower58'",
            "current": "Cortinarius Harrower58 (30) | Cortinarius sp. 'Harrower58' (1)"}
    assert nc.item_names(item) == [("Cortinarius Harrower58", "records"),
                                   ("Cortinarius sp. 'Harrower58'", "records")]
    assert "name_writing_split" in nc.NAME_TYPES
