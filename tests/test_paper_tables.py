"""`mv paper-score`'s outcomes and tables on made-up finds: how each comparator's answer
becomes a per-find outcome (Vision's ranked names, iNat's taxon-id hits, the experts'
before-cutoff IDs, a published model's answer), what counts in each denominator, that
near-duplicates leave the primary numbers, and the families' tests and Holm."""

import math

import numpy as np
import pytest

from mycomap_vision import paper_outcomes as po
from mycomap_vision import paper_tables as pt

REPS = 300


def find(key, species="Amanita muscaria", genus="Amanita", depth=10, kind="formal",
         observer=None, **kw):
    return po.Find(key, observer or f"obs{int(key) % 7}", species, genus, "Amanitaceae", kind,
                   depth, 2, **kw)


# --- outcomes ----------------------------------------------------------------------------------

def test_ranked_names_hit_at_the_first_position_that_matches_each_level():
    f = find("1", species="Cortinarius violaceus", genus="Cortinarius")
    o = po.from_ranked(["Cortinarius alboviolaceus", "Cortinarius violaceus"],
                       ["Cortinarius"], f)
    assert o["species_strict@1"] is False and o["species_strict@3"] is True
    assert o["genus_strict@1"] is True and o["answered"] is True
    # A looser level counts every stricter hit.
    for k in po.K:
        assert o[f"species_sl@{k}"] >= o[f"species_strict@{k}"]
        assert o[f"species_complex@{k}"] >= o[f"species_sl@{k}"]


def test_a_one_word_truth_has_no_species_cells_and_no_answer_is_wrong():
    f = find("2", species="", genus="Russula", kind="one-word")
    o = po.from_ranked([], [], f)
    assert all(o[c] is None for c in po.CELLS if c.startswith("species"))
    assert all(o[c] is False for c in po.CELLS if c.startswith("genus"))
    assert o["answered"] is False


def e14_line(key, formal=True, vhit1=True, ghit1=True, names=("Amanita muscaria",)):
    hits = lambda h: {f"top{k}": h for k in po.K}  # noqa: E731
    return {"record_key": key, "photos": 2, "photo_ids": [1, 2], "px": 1024,
            "truth": {"species": 48715, "genus": 47392, "formal_species": formal,
                      "has_genus": True},
            "vision": {"species": [[48715, n, 90] for n in names], "genus": [[47392, "Amanita", 95]]},
            "combined": {"species": [], "genus": []},
            "answered": {"vision": True, "combined": False},
            "vision_hits": {"species": hits(vhit1) if formal else None, "genus": hits(ghit1)},
            "combined_hits": {"species": hits(False) if formal else None, "genus": hits(False)}}


def test_inat_strict_cells_come_from_taxon_id_hits_and_need_a_formal_truth():
    f = find("3")
    o = po.inat_outcome(e14_line("3", names=("Amanita muscaria var. guessowii",)), f, "vision")
    assert o["species_strict@1"] is True                 # by taxon id, not spelling
    assert o["genus_strict@1"] is True and o["answered"] is True
    c = po.inat_outcome(e14_line("3"), f, "combined")
    assert c["answered"] is False and c["species_strict@10"] is False
    prov = po.inat_outcome(e14_line("4", formal=False), find("4", species="Amanita sp-CA01",
                                                            kind="provisional"), "vision")
    assert all(prov[c] is None for c in po.CELLS if c.startswith("species"))
    assert prov["genus_strict@1"] is True


def taxa(species_right=None, genus_right=None, truth="Amanita muscaria"):
    """An identification's taxa (E31 format 2) that the answer key judges as given against
    `truth` (the default find's species)."""
    g = truth.split()[0]
    sp = None if species_right is None else (
        {"id": 1, "name": truth} if species_right else {"id": 2, "name": f"{g} phalloides"})
    if genus_right is None:
        genus_right = species_right
    ge = None if genus_right is None else (
        {"id": 3, "name": g} if genus_right else {"id": 4, "name": "Boletus"})
    return {"species_taxon": sp, "genus_taxon": ge, "species_right": species_right,
            "genus_right": genus_right}


def e31_line(key, ids, status="ok", observer_named=False, community=None):
    return {"record_key": key, "status": status, "ids": ids, "format": 2,
            "observer_named_species_before_cutoff": observer_named,
            "community": community or {"present": True, "at_species": True,
                                       **taxa(species_right=False, genus_right=True)}}


def ident(tier="1", before=True, species_right=None, genus_right=None, role="identifier",
          made_at="2025-01-01T00:00:00Z"):
    return {"role": role, "tier": tier, "before_cutoff": before, "made_at": made_at,
            **taxa(species_right, genus_right)}


def xo(line, **kw):
    """The experts' outcome on the default find (Amanita muscaria) under the answer key."""
    return po.expert_outcome(line, find=find(line["record_key"]), **kw)


def test_experts_score_only_identifiers_of_the_tier_before_the_dna_was_visible():
    line = e31_line("5", [ident(species_right=False, genus_right=True),
                          ident(species_right=True, genus_right=True, before=False),
                          ident(species_right=True, role="validator"),
                          ident(tier="2", species_right=True)])
    o = xo(line, view="best")
    assert o["species_strict@1"] is False                # the right ones came after or weren't tier 1
    assert o["genus_strict@1"] is True
    assert o["species_strict@3"] is None                 # one name: only top-1 cells
    best = xo(e31_line("6", [ident(species_right=False),
                                            ident(species_right=True)]), view="best")
    assert best["species_strict@1"] is True              # "best tier-1": any of them right
    genus_only = xo(e31_line("7", [ident(genus_right=True)]), view="best")
    assert genus_only["species_strict@1"] is None        # stopping at genus is coverage
    assert xo(e31_line("8", [], status="no_cutoff")) is None
    assert xo(e31_line("9", [ident(species_right=True)], observer_named=True),
                             strict=True) is None
    com = xo(line, view="community")
    assert com["species_strict@1"] is False and com["genus_strict@1"] is True


def test_a_file_in_another_format_is_refused():
    with pytest.raises(ValueError, match="not format 1"):
        po.check_format([{"record_key": "1", "truth": {}}], "e14")


# --- tables ------------------------------------------------------------------------------------

def world(n=60, seed=0):
    rng = np.random.default_rng(seed)
    finds, a, b = [], {}, {}
    for i in range(n):
        f = find(str(i), depth=int(rng.choice([0, 3, 10, 50, 200])),
                 near_duplicate={"cosine": i < 6})
        finds.append(f)
        right_a = bool(rng.random() < 0.7)
        right_b = right_a and bool(rng.random() < 0.6)
        a[f.key] = po.from_ranked(["Amanita muscaria" if right_a else "Amanita flavoconia"],
                                  ["Amanita"], f)
        b[f.key] = po.from_ranked(["Amanita muscaria" if right_b else "Russula emetica"],
                                  ["Amanita" if right_b else "Russula"], f)
    return finds, {"a": a, "b": b}


def test_near_duplicates_leave_the_primary_numbers_and_are_counted():
    finds, out = world()
    primary = pt.in_scope(finds)
    assert len(primary) == 54 and not any(f.is_near_duplicate for f in primary)
    assert len(pt.in_scope(finds, "with_near_duplicates")) == 60
    res = pt.all_standard(finds, out, REPS, 0)
    assert res["near_duplicates"] == {"finds": 60, "near_duplicate_finds": 6,
                                      "by_kind": {"cosine": 6}}
    assert res["comparators"]["a"]["primary"]["all"]["finds"] == 54


def test_both_denominators_and_no_reference_counts_as_wrong():
    finds, out = world()
    t = pt.all_standard(finds, out, REPS, 0)["comparators"]["a"]["primary"]
    no_ref = [f for f in pt.in_scope(finds) if f.depth == 0]
    assert t["all"]["finds"] - t["has_reference"]["finds"] == len(no_ref)
    cell = t["all"]["table"]["species strict"]["top1"]
    assert cell["n"] == t["all"]["finds"] and cell["ci95"] is not None
    assert t["all"]["species_by_reference_band"]["0"]["top1"]["n"] == len(no_ref)


def test_a_comparator_with_no_line_for_a_find_counts_it_as_unanswered():
    finds = [find("1"), find("2")]
    o = {"1": po.from_ranked(["Amanita muscaria"], ["Amanita"], finds[0])}
    r = pt.rate(finds, o, "species_strict@1", REPS, 0)
    assert (r["n"], r["right"]) == (2, 1)


def test_a_paired_test_uses_only_the_discordant_finds():
    finds, out = world(200, seed=3)
    r = pt.paired(pt.in_scope(finds), out["a"], out["b"], "species_strict@1", "greater",
                  REPS, 0)
    assert r["only_b"] == 0 and r["only_a"] > 0                # b is right only where a is
    assert r["p"] == pytest.approx(0.5 ** r["only_a"])
    assert r["ci95"][0] > 0


def test_families_are_holm_adjusted_and_a_missing_input_is_reported_not_tested():
    finds, out = world(200, seed=3)
    spec = {"A": [{"id": "H1a", "kind": "mcnemar", "a": "a", "b": "b",
                   "cell": "species_strict@1", "name_kind": "formal"},
                  {"id": "H1b", "kind": "mcnemar", "a": "a", "b": "b",
                   "cell": "genus_strict@1"},
                  {"id": "H3", "kind": "gee_depth", "comparator": "a",
                   "cell": "species_strict@1", "covariates": ["name_kind", "photos"]},
                  {"id": "H4", "kind": "mcnemar", "a": "a", "b": "experts",
                   "cell": "species_strict@1", "alternative": "two-sided"}]}
    res = pt.families(spec, finds, out, REPS, 0)["A"]
    assert "not_tested" in res["H4"] and res["H4"]["reject"] is False
    m = 4
    ranked = sorted(res, key=lambda h: res[h]["p_test"] if "p_test" in res[h] else 1.0)
    assert res[ranked[0]]["p_holm"] == pytest.approx(min(1, m * res[ranked[0]]["p_test"]))
    assert res["H1a"]["reject"] and res["H1a"]["n"] == 194   # 200 - 6 near-duplicates
    assert "slope" in res["H3"]
    # Every made-up find is formal with 2 photos: those covariates are dropped, and said so.
    assert res["H3"]["dropped_covariates"] == ["provisional_name", "photos"]


def test_a_test_can_be_limited_to_one_reference_band_and_a_bad_band_is_refused():
    finds = [find(str(i), depth=d) for i, d in enumerate([0, 2, 3, 10, 50, 150])]
    assert [f.depth for f in pt.select(finds, {"depth_band": "1-4"}, {})] == [2, 3]
    assert [f.depth for f in pt.select(finds, {"depth_band": "100+"}, {})] == [150]
    with pytest.raises(ValueError, match="depth_band"):
        pt.select(finds, {"depth_band": "1-5"}, {})


def test_the_place_band_is_one_intersection_union_test():
    finds, out = world(400, seed=5)
    for i, f in enumerate(finds):
        f.attrs["distance_band"] = "300-1500" if i % 2 else "<100"
    spec = {"id": "E8", "kind": "iut", "a": "a", "b": "b", "cell": "species_strict@1",
            "band_attr": "distance_band", "band": "300-1500", "margin": 0.02}
    r = pt.test_one(spec, finds, out, REPS, 0)
    assert r["p_test"] == max(r["superiority"]["p"], r["noninferiority"]["p"])
    assert r["noninferiority"]["n"] == sum(1 for f in pt.in_scope(finds)
                                           if f.attrs["distance_band"] == "300-1500")


def test_slices_give_intervals_only_and_pool_small_groups():
    finds, out = world()
    for i, f in enumerate(finds):
        f.attrs["source"] = "inat" if i % 10 else "mo"
    s = pt.slices(finds, out, ["source"], ["species_strict@1"], ["a"], REPS, 0, min_finds=10)
    assert set(s["source"]) == {"inat", "(small)"}
    cell = s["source"]["inat"]["a"]["species_strict@1"]
    assert set(cell) == {"n", "right", "rate", "ci95"}          # no p-values in slices

def test_the_primary_expert_view_is_the_first_tier_one_species_id_before_the_cutoff():
    line = e31_line("10", [
        ident(species_right=True, made_at="2025-03-02T10:00:00Z"),
        ident(species_right=False, genus_right=True, made_at="2025-03-01T09:00:00Z"),
        ident(genus_right=False, made_at="2025-02-01T00:00:00Z"),          # genus-only, earlier
        ident(species_right=True, before=False, made_at="2025-01-01T00:00:00Z")])
    first = xo(line)                      # view="first" is the default
    assert first["species_strict@1"] is False            # the earliest species-level ID
    assert first["genus_strict@1"] is False              # the earliest genus-scorable ID
    assert xo(line, view="best")["species_strict@1"] is True
    assert xo(e31_line("11", [ident(genus_right=True)]))["species_strict@1"] is None


def test_collapsing_to_species_rank_scores_a_variety_label_by_its_species():
    exact = po.AnswerKey("exact")                      # v1's published rule
    f = find("20", species="Amanita muscaria var. guessowii")
    plain = po.from_ranked(["Amanita muscaria"], ["Amanita"], f, key=exact)
    both = po.from_ranked(["Amanita muscaria"], ["Amanita"], f, collapse=True, key=exact)
    # Steve's key takes species at species rank for every comparator, no flag needed.
    assert po.from_ranked(["Amanita muscaria"], ["Amanita"], f)["species_strict@1"] is True
    assert plain["species_strict@1"] is False and both["species_strict@1"] is True
    # Answers that collapse to one species count once in the ranking.
    two = po.from_ranked(["Amanita muscaria var. alba", "Amanita muscaria", "Amanita flavoconia"],
                         ["Amanita"], find("21", species="Amanita flavoconia"), collapse=True,
                         key=exact)
    assert two["species_strict@1"] is False and two["species_strict@3"] is True
    assert po.species_rank("Collybiopsis confluens ssp. campanulata") == "Collybiopsis confluens"


def test_generalisation_rows_compare_each_named_row_with_the_other_finds():
    finds = [find(str(i), observer=f"o{i}",
                  attrs={"observer_new": "new" if i < 4 else "known"}) for i in range(10)]
    finds.append(find("99", near_duplicate={"dhash": True}, attrs={"observer_new": "new"}))
    outcome = {f.key: {"species_strict@1": int(f.key) >= 4} for f in finds}
    g = pt.generalisation(finds, {"v": outcome}, {"unseen observers": {"observer_new": "new"}},
                          ["species_strict@1"], ["v", "absent"], REPS, 0)
    row = g["rows"]["unseen observers"]
    assert (row["finds"], row["others"]) == (4, 6)            # the near-duplicate stays out
    assert row["v"]["species_strict@1"]["in"]["rate"] == 0.0
    assert row["v"]["species_strict@1"]["out"]["rate"] == 1.0
    assert "absent" not in row
    nd = g["near_duplicates"]
    assert nd["near_duplicate_finds"] == 1
    assert nd["v"]["species_strict@1"]["primary"]["n"] == 10
    assert nd["v"]["species_strict@1"]["with_near_duplicates"]["n"] == 11


def test_pooled_experts_take_the_first_id_by_any_identifier_with_a_thousand_ids_or_more():
    line = e31_line("30", [
        ident(species_right=True, tier="1", made_at="2025-03-02T00:00:00Z"),
        ident(species_right=False, tier="2", made_at="2025-03-01T00:00:00Z"),   # earlier
        ident(species_right=True, tier="3", made_at="2025-02-01T00:00:00Z")])   # under 1,000
    assert xo(line, tier="1")["species_strict@1"] is True
    assert xo(line, tier="2")["species_strict@1"] is False
    assert xo(line, tier="1+2")["species_strict@1"] is False   # tier 2 was first
    assert xo(line, tier="1+2", view="best")["species_strict@1"] is True



def test_each_comparator_is_tabled_tested_and_sliced_on_its_own_finds():
    """O1: a comparator with its own reference (finds_of) gets its own depth bands, its own
    finds in the hypotheses it leads, and its own attribute values in slices."""
    finds = [po.Find(str(i), f"o{i}", "A x", "A", "F", "formal", 50, 1,
                  {"observer_new": "known"}) for i in range(6)]
    own = [po.Find(f.key, f.observer, f.species, f.genus, f.family, f.name_kind, 2, 1,
                {"observer_new": "new"}) for f in finds]
    outcomes = {"full": {f.key: {"species_strict@1": True} for f in finds},
                "public": {f.key: {"species_strict@1": i % 2 == 0} for i, f in enumerate(finds)}}
    t = pt.all_standard(finds, outcomes, 50, 0, finds_of={"public": own})["comparators"]
    assert t["full"]["primary"]["all"]["species_by_reference_band"]["20-99"]["top1"]["n"] == 6
    assert t["public"]["primary"]["all"]["species_by_reference_band"]["1-4"]["top1"]["n"] == 6
    sl = pt.slices(finds, outcomes, ["observer_new"], ["species_strict@1"], ["full", "public"],
                   50, 0, finds_of={"public": own})
    assert sl["observer_new"]["known"]["full"]["species_strict@1"]["n"] == 6
    assert sl["observer_new"]["known"]["public"]["species_strict@1"]["n"] == 0
    seen = []
    orig = pt.test_one
    try:
        pt.test_one = lambda spec, fs, *a, **k: seen.append((spec["id"], fs[0].depth)) or \
            {"p_test": 0.5}
        fam = pt.families({"B": [{"id": "pub-first", "kind": "mcnemar", "a": "public",
                                  "b": "full", "cell": "species_strict@1"},
                                 {"id": "full-first", "kind": "mcnemar", "a": "full",
                                  "b": "public", "cell": "species_strict@1"}],
                           "side": [{"id": "s", "kind": "mcnemar", "a": "full", "b": "public",
                                     "cell": "species_strict@1"}]},
                          finds, outcomes, 50, 0, finds_of={"public": own})
    finally:
        pt.test_one = orig
    assert seen == [("pub-first", 2), ("full-first", 50), ("s", 50)]
    assert fam["B"]["pub-first"]["p_holm"] == 1.0                       # Holm on 2 p of 0.5
    assert fam["side"]["s"] == {"p_test": 0.5, "spec": fam["side"]["s"]["spec"],
                                "p_holm": None, "reject": None, "outside_holm": True}


# --- the floor kind (Steve's A1s / A1g, 28 Oct): a comparator's own rate against a floor ------

S1 = "species_strict@1"


def test_a_floor_test_is_the_cluster_robust_wald_of_the_rate_worked_by_hand():
    # observers g1: right, right, wrong; g2: right, wrong; g3: right, and one find with no
    # outcome line (the cell applies: counts wrong). Two one-word truths have no species
    # cell (one with a line, one without): both leave the test.
    finds = [find(str(i), observer=g) for i, g in
             enumerate(["g1", "g1", "g1", "g2", "g2", "g3", "g3"], 1)]
    finds += [find("8", species="", kind="one-word", observer="g1"),
              find("9", species="", kind="one-word", observer="g2")]
    line = {"1": True, "2": True, "3": False, "4": True, "5": False, "6": True}
    out = {k: {S1: v} for k, v in line.items()}
    out["8"] = {S1: None}
    r = pt.floor_test(finds, out, S1, 0.5, "greater", REPS, 0)
    # y = [1,1,0 | 1,0 | 1,0], n 7, p 4/7; cluster residual sums 2/7, -1/7, -1/7;
    # SE = sqrt(3/2 * 6/49) / 7 = 3/49; z = (4/7 - 1/2) / (3/49) = 7/6.
    assert (r["n"], r["right"], r["clusters"]) == (7, 4, 3)
    assert r["rate"] == pytest.approx(4 / 7) and r["difference"] == pytest.approx(4 / 7 - 0.5)
    assert r["se_robust"] == pytest.approx(3 / 49)
    assert r["z"] == pytest.approx(7 / 6)
    assert r["p"] == pytest.approx(0.5 * math.erfc((7 / 6) / math.sqrt(2)))
    assert r["n_by_name_kind"] == {"formal": 7}
    lo, hi = r["ci95"]
    assert lo <= 4 / 7 <= hi
    less = pt.floor_test(finds, out, S1, 0.5, "less", REPS, 0)
    assert less["p"] == pytest.approx(1 - r["p"])


def test_a_floor_test_with_one_observer_or_no_finds_has_no_test():
    finds = [find(str(i), observer="g1") for i in range(4)]
    out = {f.key: {S1: i % 2 == 0} for i, f in enumerate(finds)}
    one = pt.floor_test(finds, out, S1, 0.5, "greater", REPS, 0)
    assert one["n"] == 4 and one["z"] is None and one["p"] == 1.0
    none = pt.floor_test([], {}, S1, 0.5, "greater", REPS, 0)
    assert none["n"] == 0 and none["rate"] is None and none["p"] == 1.0


def test_family_a_of_two_floors_and_the_depth_trend_is_holm_adjusted_over_three():
    finds, out = world(300, seed=4)
    spec = {"A": [{"id": "A1s", "kind": "floor", "comparator": "a", "cell": S1, "floor": 0.5},
                  {"id": "A1g", "kind": "floor", "comparator": "a", "cell": "genus_strict@1",
                   "floor": 0.6},
                  {"id": "H3", "kind": "gee_depth", "comparator": "a", "cell": S1}],
            "C": [{"id": "H1a", "kind": "mcnemar", "a": "a", "b": "b", "cell": S1}]}
    res = pt.families(spec, finds, out, REPS, 0)
    a = res["A"]
    assert set(a) == {"A1s", "A1g", "H3"} and set(res["C"]) == {"H1a"}
    assert a["A1s"]["n"] == 294 and a["A1s"]["p_test"] == a["A1s"]["p"] < 0.05
    ps_ = sorted(a[h]["p_test"] for h in a)
    first = min(a, key=lambda h: a[h]["p_test"])
    assert a[first]["p_holm"] == pytest.approx(min(1, 3 * ps_[0]))
    assert a["A1s"]["reject"] is True
    # C is its own family: Holm over one, so its p is unadjusted
    assert res["C"]["H1a"]["p_holm"] == pytest.approx(res["C"]["H1a"]["p_test"])


def test_a_floor_whose_comparator_has_no_outcomes_is_not_tested_and_counts_p_1_in_holm():
    finds, out = world(300, seed=4)
    spec = {"A": [{"id": "A1s", "kind": "floor", "comparator": "a", "cell": S1, "floor": 0.5},
                  {"id": "A1g", "kind": "floor", "comparator": "missing", "cell": S1,
                   "floor": 0.5},
                  {"id": "H3", "kind": "gee_depth", "comparator": "a", "cell": S1}]}
    a = pt.families(spec, finds, out, REPS, 0)["A"]
    assert "not_tested" in a["A1g"] and a["A1g"]["reject"] is False
    assert a["A1g"]["p_holm"] == 1.0
    p = sorted([a["A1s"]["p_test"], a["H3"]["p_test"], 1.0])
    first = min(("A1s", "H3"), key=lambda h: a[h]["p_test"])
    assert a[first]["p_holm"] == pytest.approx(min(1, 3 * p[0]))     # still over three


def test_secondary_rows_run_a_floor_and_give_an_inat_paired_test_n_0_where_inat_cannot_name():
    finds, out = world(60, seed=2)
    for i, f in enumerate(finds):
        if i != 59:                              # the last find has no taxonomy: unknown
            f.attrs["truth_resolves"] = "no" if i % 3 == 0 else "yes"
            f.attrs["truth_genus_known"] = "yes"
    hyps = {"A": [{"id": "A1s", "kind": "floor", "comparator": "a", "cell": S1, "floor": 0.5}],
            "C": [{"id": "H1a", "kind": "mcnemar", "a": "a", "b": "b", "cell": S1,
                   "truth_resolves": True}]}
    block = {"for": ["A1s", "H1a"], "rows": [
        {"id": "inat-cannot-name-species", "where": {"truth_resolves": "no"}},
        {"id": "inat-can-name-species", "where": {"truth_resolves": "yes"}}]}
    rows = pt.secondary_rows(block, hyps, finds, out, REPS, 0)
    no, yes = rows["inat-cannot-name-species"], rows["inat-can-name-species"]
    n_no = sum(1 for i, f in enumerate(finds) if i >= 6 and i != 59 and i % 3 == 0)
    assert no["finds"] == n_no and no["unknown"] == 1 and yes["unknown"] == 1
    assert no["hypotheses"]["A1s"]["n"] == n_no and no["hypotheses"]["A1s"]["outside_holm"]
    h = no["hypotheses"]["H1a"]
    assert h["n"] == 0 and h["p_test"] == 1.0 and h["difference"] is None
    assert yes["hypotheses"]["H1a"]["n"] == yes["finds"] > 0


# --- the test scope (Steve 28 Oct): the finds outside it, described, never tested -------------

def test_the_finds_outside_the_test_scope_are_counted_by_country_and_their_rate_given():
    finds = [find("1", observer="g1"), find("2", observer="g1"), find("3", observer="g2"),
             find("4", observer="g3"), find("5", observer="g3",
                                         near_duplicate={"cosine": True})]
    for f, c in zip(finds, ["MX", "MX", "unknown", "GB", "MX"]):
        f.attrs["country"] = c
    out = {"p": {"1": {S1: True, "genus_strict@1": True}, "2": {S1: False, "genus_strict@1": True},
                 "3": {S1: True, "genus_strict@1": True}}}           # "4": no line, counts wrong
    r = pt.outside_test_scope(finds, out, ["p"], [S1, "genus_strict@1"], REPS, 0)
    assert (r["finds"], r["primary"]) == (5, 4)                 # the near-duplicate leaves
    assert r["by_country"] == {"GB": 1, "MX": 3} and r["unknown"] == 1
    sp = r["comparators"]["p"][S1]
    assert (sp["n"], sp["right"], sp["rate"]) == (4, 2, 0.5) and sp["ci95"] is not None
    assert r["comparators"]["p"]["genus_strict@1"]["rate"] == 0.75
    none = pt.outside_test_scope([], {}, ["p"], [S1], REPS, 0)
    assert none["finds"] == 0 and none["comparators"]["p"][S1]["n"] == 0
    assert none["comparators"]["p"][S1]["rate"] is None


def test_vision_holds_are_split_by_whether_the_final_label_took_vision_s_genus():
    finds = [po.Find(k, "o", f"{g} x", g, "", "formal", 3, 1)
             for k, g in (("1", "Russula"), ("2", "Lactarius"), ("3", "Amanita"), ("4", "A"))]
    holds = {"1": "russula", "2": "Russula", "3": None, "9": "Boletus"}   # 9 is not scored
    assert pt.hold_outcomes(finds, holds) == {"holds": 3, "kept_dna_genus": 1,
                                              "took_vision_genus": 1, "vision_genus_unknown": 1}
