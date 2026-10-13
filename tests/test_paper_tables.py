"""`mv paper-score`'s outcomes and tables on made-up finds: how each comparator's answer
becomes a per-find outcome (Vision's ranked names, iNat's taxon-id hits, the experts'
before-cutoff IDs, a published model's answer), what counts in each denominator, that
near-duplicates leave the primary numbers, and the families' tests and Holm."""

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


def e31_line(key, ids, status="ok", observer_named=False, community=None):
    return {"record_key": key, "status": status, "ids": ids,
            "observer_named_species_before_cutoff": observer_named,
            "community": community or {"present": True, "at_species": True,
                                       "species_right": False, "genus_right": True}}


def ident(tier="1", before=True, species_right=None, genus_right=None, role="identifier",
          made_at="2025-01-01T00:00:00Z"):
    return {"role": role, "tier": tier, "before_cutoff": before, "made_at": made_at,
            "species_right": species_right, "genus_right": genus_right}


def test_experts_score_only_identifiers_of_the_tier_before_the_dna_was_visible():
    line = e31_line("5", [ident(species_right=False, genus_right=True),
                          ident(species_right=True, genus_right=True, before=False),
                          ident(species_right=True, role="validator"),
                          ident(tier="2", species_right=True)])
    o = po.expert_outcome(line, view="best")
    assert o["species_strict@1"] is False                # the right ones came after or weren't tier 1
    assert o["genus_strict@1"] is True
    assert o["species_strict@3"] is None                 # one name: only top-1 cells
    best = po.expert_outcome(e31_line("6", [ident(species_right=False),
                                            ident(species_right=True)]), view="best")
    assert best["species_strict@1"] is True              # "best tier-1": any of them right
    genus_only = po.expert_outcome(e31_line("7", [ident(genus_right=True)]), view="best")
    assert genus_only["species_strict@1"] is None        # stopping at genus is coverage
    assert po.expert_outcome(e31_line("8", [], status="no_cutoff")) is None
    assert po.expert_outcome(e31_line("9", [ident(species_right=True)], observer_named=True),
                             strict=True) is None
    com = po.expert_outcome(line, view="community")
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
    first = po.expert_outcome(line)                      # view="first" is the default
    assert first["species_strict@1"] is False            # the earliest species-level ID
    assert first["genus_strict@1"] is False              # the earliest genus-scorable ID
    assert po.expert_outcome(line, view="best")["species_strict@1"] is True
    assert po.expert_outcome(e31_line("11", [ident(genus_right=True)]))["species_strict@1"] is None
