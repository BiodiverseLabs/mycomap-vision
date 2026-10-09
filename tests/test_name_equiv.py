"""Which names count as the same for scoring: strict, s.l. (genus groups), species complex (beta)."""
import json

import pytest

from test_likely import compared

from mycomap_vision import evaluate, name_equiv
from mycomap_vision.name_equiv import genus_match, species_match


def m(answer, truth):
    r = species_match(answer, truth)
    return r["strict"], r["sl"], r["complex"]


@pytest.mark.parametrize("answer, truth", [
    ("Panus conchatus", "Panus sp. 'conchatus-OH01'"),
    ("Exsudoporus frostii", "Exsudoporus sp. 'frostii-IN02'"),
    ("Sebacina sp. 'schweinitzii-IN01'", "Sebacina sp. 'schweinitzii-IN02'"),
    ("Lactarius fallax", "Lactarius sp. 'fallax-PNW03'"),
    ("Amanita muscaria", "Amanita muscaria subsp. flavivolvata"),
])
def test_a_described_name_and_the_provisional_names_on_its_stem_are_one_complex(answer, truth):
    assert m(answer, truth) == (False, False, True)
    assert m(truth, answer) == (False, False, True)


def test_bare_codes_are_never_a_complex_only_themselves():
    assert m("Cortinarius sp. 'CA04'", "Cortinarius sp. 'MI01'") == (False, False, False)
    assert m("Cortinarius sp. 'CA04'", "Cortinarius sp. 'CA07'") == (False, False, False)
    # Code letters are capitals however many there are; a code written small is short.
    assert m("Russula sp. 'TENN01'", "Russula sp. 'TENN04'") == (False, False, False)
    assert m("Cortinarius sp. 'ca04'", "Cortinarius sp. 'ca07'") == (False, False, False)
    assert m("Hydnum sp. 'PNW01'", "Hydnum sp. 'PNW01'") == (True, True, True)


def test_the_same_epithet_across_a_genus_group_is_the_same_species_sl():
    assert m("Calonarius pacificus", "Thaxterogaster pacificus") == (False, True, True)
    assert m("Cortinarius pacificus", "Calonarius pacificus") == (False, True, True)
    assert m("Inocybe unicolor", "Mallocybe unicolor") == (False, True, True)


def test_a_provisional_code_in_two_genera_of_a_group_is_not_one_species():
    # Codes are numbered within a genus: the same code under two genera is usually two taxa.
    assert m("Calonarius sp. 'IN06'", "Cortinarius sp. 'IN06'") == (False, False, False)
    assert m("Inocybe sp. 'CA01'", "Mallocybe sp. 'CA01'") == (False, False, False)


@pytest.mark.parametrize("answer, truth", [
    ("Inocybe rimosa", "Inosperma rimosum"),             # -a / -um
    ("Cortinarius albus", "Calonarius alba"),            # -us / -a
    ("Inocybe viridis", "Pseudosperma viride"),          # -is / -e
    ("Cortinarius ruber", "Phlegmacium rubrum"),         # -er / -rum
    ("Cortinarius niger", "Calonarius nigra"),           # -er / -ra
    ("Inocybe rimosa", "Inocybe rimosus"),               # one genus, gender misspelt
    ("Amanita muscaria subsp. flavivolvata", "Amanita muscarius subsp. flavivolvatus"),
])
def test_names_that_differ_only_by_a_gender_ending_are_one_species_sl(answer, truth):
    assert m(answer, truth) == (False, True, True)
    assert m(truth, answer) == (False, True, True)


def test_gender_endings_of_different_declensions_stay_apart():
    assert m("Lactarius acris", "Lactarius acra") == (False, False, False)


def test_a_gender_ending_reaches_the_complex_stem_but_never_the_code():
    assert m("Inosperma rimosum", "Inocybe sp. 'rimosa-CA01'") == (False, False, True)
    assert m("Russula sp. 'rosea-IN01'", "Russula sp. 'roseus-IN01'") == (False, False, True)
    # Codes are numbered within a genus: a different code is a different species.
    assert m("Russula sp. 'rosea-IN01'", "Russula sp. 'rosea-IN02'")[:2] == (False, False)


def test_a_temporary_code_is_scored_like_a_named_species():
    # About half the records are temporary codes: the same code in the same genus is a
    # strict match, however it is written; any other code is wrong.
    assert m("Cortinarius sp. 'CA04'", "Cortinarius sp. 'CA04'") == (True, True, True)
    assert m("Cortinarius CA4", "Cortinarius sp. 'CA04'") == (True, True, True)
    assert m("Cortinarius sp. 'CA05'", "Cortinarius sp. 'CA04'") == (False, False, False)
    assert m("Cortinarius pacificus", "Cortinarius sp. 'CA04'") == (False, False, False)


def test_different_epithets_in_a_group_share_only_the_genus_sl():
    assert m("Cortinarius glaucopus", "Phlegmacium pacificus") == (False, False, False)
    assert genus_match("Phlegmacium", "Cortinarius") == {"strict": False, "sl": True}
    assert genus_match("Inosperma", "Pseudosperma") == {"strict": False, "sl": True}


def test_outside_a_group_a_shared_epithet_or_stem_is_nothing():
    assert m("Lactarius fallax", "Russula fallax") == (False, False, False)
    assert genus_match("Russula", "Lactarius") == {"strict": False, "sl": False}


def test_spellings_of_one_name_are_strict_matches():
    assert m("Lactarius sp. 'fallax-PNW3'", "Lactarius sp. 'fallax-PNW03'") == (True, True, True)
    assert m("Russula emetica", "russula  emetica") == (True, True, True)
    assert genus_match("Amanita", "amanita") == {"strict": True, "sl": True}


def test_qualifiers_and_missing_answers():
    assert m("Russula cf. emetica", "Russula emetica")[2] is True
    assert m("", "Russula emetica") == (False, False, False)
    assert m("Russula emetica", None) == (False, False, False)


def test_each_reading_implies_the_next():
    names = ["Panus conchatus", "Panus sp. 'conchatus-OH01'", "Calonarius pacificus",
             "Thaxterogaster pacificus", "Cortinarius sp. 'CA04'", "Lactarius fallax",
             "Russula fallax", "Amanita muscaria subsp. flavivolvata", "Amanita muscaria",
             "Inocybe rimosa", "Inosperma rimosum", "Inocybe sp. 'rimosa-CA01'"]
    for a in names:
        for t in names:
            strict, sl, complex_ = m(a, t)
            assert (not strict or sl) and (not sl or complex_), (a, t)


def test_the_groups_file_is_valid_and_easy_to_extend(tmp_path, monkeypatch):
    data = json.loads(name_equiv.GROUPS_FILE.read_text(encoding="utf-8"))
    genera = [g.lower() for gs in data["groups"].values() for g in gs]
    assert len(genera) == len(set(genera))                     # no genus in two groups
    data["groups"]["Russula s.l."] = ["Russula", "Lactarius"]
    extended = tmp_path / "genus_groups.json"
    extended.write_text(json.dumps(data), encoding="utf-8")
    monkeypatch.setattr(name_equiv, "GROUPS_FILE", extended)
    name_equiv._group_of.cache_clear()
    try:
        assert m("Lactarius fallax", "Russula fallax") == (False, True, True)
    finally:
        name_equiv._group_of.cache_clear()


def test_a_comparison_adds_the_beta_columns_only_when_asked(conn, tmp_path):
    cid = compared(conn, tmp_path)["comparison_id"]
    plain = json.loads(conn.execute("select report_json from eval_runs where comparison_id = ?",
                                    (cid,)).fetchone()[0])["all_photos"]
    assert "name_equivalence" not in plain
    cid2 = evaluate.compare(conn, ["m1"], ["nearest"], embeddings_root=tmp_path / "emb",
                            name_scores=True, log=lambda s: None)["comparison_id"]
    # Two comparisons of the same records in one second share an id: read the newest row.
    rep = json.loads(conn.execute("select report_json from eval_runs where comparison_id = ? "
                                  "order by id desc limit 1", (cid2,)).fetchone()[0])["all_photos"]
    eq = rep["name_equivalence"]
    assert eq["beta"] is True and set(eq["species"]) == {"n", "strict", "sl", "complex"}
    assert set(eq["genus"]) == {"n", "strict", "sl"}
    # Strict is the scoreboard's own top-1: the beta columns sit beside it, never replace it.
    assert eq["species"]["strict"] == rep["species"]["all"]["top1"]
    assert eq["genus"]["strict"] == rep["genus"]["all"]["top1"]


def test_mv_compare_name_scores(conn, tmp_path, monkeypatch):
    from mycomap_vision import cli, config
    seen = {}
    monkeypatch.setattr(evaluate, "compare",
                        lambda *a, **kw: seen.update(kw) or {"comparison_id": "c"})
    monkeypatch.setattr(evaluate, "format_report", lambda r: "{}")
    monkeypatch.setattr(evaluate, "scoreboard", lambda conn, cid: [])
    monkeypatch.setattr(config, "MANIFEST_PATH", tmp_path / "m.sqlite")
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    cli.main(["compare", "--backbones", "bioclip-2", "--name-scores"])
    assert seen["name_scores"] is True
    cli.main(["compare", "--backbones", "bioclip-2"])
    assert seen["name_scores"] is False
