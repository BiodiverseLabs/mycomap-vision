"""E31 expert identifiers: the DNA-visible cutoff, IDs as they stood, the community taxon,
tiers and exclusions, scoring by taxon id, and the fetcher's manners."""

import gzip
import json

import pytest

from mycomap_vision import expert_ids as E

# A tiny taxonomy: Fungi 1 > family 5 > genus 10 (Russula) > species 100, 101; genus 11
# (Lactarius) > species 110; species 99 is retired, its successor 100.
TAXA = {
    1: {"id": 1, "name": "Fungi", "rank": "kingdom", "rank_level": 70, "is_active": True, "ancestor_ids": []},
    5: {"id": 5, "name": "Russulaceae", "rank": "family", "rank_level": 30, "is_active": True, "ancestor_ids": [1]},
    10: {"id": 10, "name": "Russula", "rank": "genus", "rank_level": 20, "is_active": True, "ancestor_ids": [1, 5]},
    11: {"id": 11, "name": "Lactarius", "rank": "genus", "rank_level": 20, "is_active": True, "ancestor_ids": [1, 5]},
    100: {"id": 100, "name": "Russula emetica", "rank": "species", "rank_level": 10, "is_active": True, "ancestor_ids": [1, 5, 10]},
    101: {"id": 101, "name": "Russula silvicola", "rank": "species", "rank_level": 10, "is_active": True, "ancestor_ids": [1, 5, 10]},
    110: {"id": 110, "name": "Lactarius indigo", "rank": "species", "rank_level": 10, "is_active": True, "ancestor_ids": [1, 5, 11]},
    99: {"id": 99, "name": "Russula oldname", "rank": "species", "rank_level": 10, "is_active": False,
         "ancestor_ids": [1, 5, 10], "current_synonymous_taxon_ids": [100]},
}
PATH = {t: TAXA[t]["ancestor_ids"] + [t] for t in TAXA}
LEVEL = {t: TAXA[t]["rank_level"] for t in TAXA}


def ident(uid, taxon, created, current=True, updated=None, disagreement=False, remark=False, iid=None):
    ident.n = getattr(ident, "n", 0) + 1
    return {"id": iid or ident.n, "user_id": uid, "taxon_id": taxon, "taxon_path": PATH[taxon],
            "rank_level": LEVEL[taxon], "taxon_active": TAXA[taxon]["is_active"],
            "created_at": E.utc(created), "updated_at": E.utc(updated or created),
            "current": current, "category": None, "disagreement": disagreement,
            "taxon_change": False, "remark_mentions_dna": remark}


def obs(idents, ofvs=(), comments=(), counts=None, observer=1):
    return {"observation_id": "42", "observer_id": observer, "identifications": list(idents),
            "ofvs": list(ofvs), "comments": list(comments),
            "identifier_counts": counts or {}}


CUT = "2024-07-02T00:00:00Z"
TRUTH = E.TruthTaxa(species=100, genus=10, formal_species=True)


def key(uid):
    return f"h{uid}"


# --- the cutoff -------------------------------------------------------------------

def test_cutoff_is_the_earliest_of_every_dna_source():
    o = obs([ident(2, 100, "2024-01-01", remark=True)],
            ofvs=[{"field_id": 2330, "name": "DNA Barcode ITS", "created_at": "2024-07-02T19:48:21Z"},
                  {"field_id": 1162, "name": "Voucher Specimen Taken", "created_at": "2021-11-01T00:00:00Z"},
                  {"field_id": 10675, "name": "Provisional Species Name", "created_at": "2024-06-01T00:00:00Z"}],
            comments=[{"created_at": "2023-05-05T00:00:00Z", "mentions_dna": False}])
    c = E.dna_cutoff(o, [("sequence_created", "2024-03-01 10:00:00")])
    assert c.source == "comment" and c.at.startswith("2024-01-01")   # the ID remark mentioned DNA
    assert set(c.sources) == {"comment", "dna_field", "name_push", "sequence_created"}
    o["identifications"][0]["remark_mentions_dna"] = False
    c = E.dna_cutoff(o, [("sequence_created", "2024-03-01 10:00:00")])
    assert c.source == "sequence_created" and c.at == "2024-03-01T10:00:00.000000Z"


def test_a_voucher_field_does_not_end_the_window():
    o = obs([], ofvs=[{"field_id": 1162, "name": "Voucher Specimen Taken", "created_at": "2021-11-01T00:00:00Z"},
                      {"field_id": 2863, "name": "Voucher Number(s)", "created_at": "2021-12-01T00:00:00Z"}])
    assert E.dna_cutoff(o, validated_on="2024-07-02").source == "validated_on"
    assert E.dna_cutoff(o).source == "none"


def test_mention_rule_catches_sequence_words_but_not_its():
    assert E.MENTION.search("Sequenced, ITS matches R. emetica")
    assert E.MENTION.search("per DNA")
    assert E.MENTION.search("ITS2 says otherwise")
    assert not E.MENTION.search("its cap is red; it's a Russula")


# --- IDs as they stood ----------------------------------------------------------------

def test_an_id_replaced_after_the_cutoff_still_stands_and_ids_after_it_do_not():
    o = obs([ident(2, 101, "2024-01-01", current=False, updated="2024-08-01"),
             ident(2, 100, "2024-08-01"),            # changed after the DNA: not counted
             ident(3, 100, "2024-09-01")])           # first ID after the DNA: not counted
    st = E.standing_ids(o, CUT)
    assert set(st) == {2} and st[2]["taxon_id"] == 101


def test_an_id_withdrawn_before_the_cutoff_does_not_stand():
    o = obs([ident(2, 100, "2023-01-01", current=False, updated="2023-06-01")])
    assert E.standing_ids(o, CUT) == {}


def test_an_id_withdrawn_at_an_unknown_time_after_the_cutoff_stands_flagged():
    o = obs([ident(2, 100, "2023-01-01", current=False, updated="2025-01-01")])
    st = E.standing_ids(o, CUT)
    assert st[2]["withdrawal_ambiguous"] is True


# --- the community taxon --------------------------------------------------------------

def test_community_taxon_needs_two_ids_and_more_than_two_thirds():
    assert E.community_taxon([ident(2, 100, "2023-01-01")]) is None
    assert E.community_taxon([ident(2, 100, "2023-01-01"), ident(3, 100, "2023-01-02")]) == 100
    # 100, 100, 101: species 100 has 2/3 (not more), so the genus wins with 3/3
    three = [ident(2, 100, "2023-01-01"), ident(3, 100, "2023-01-02"), ident(4, 101, "2023-01-03")]
    assert E.community_taxon(three) == 10
    # a fourth agreeing ID tips species 100 over 2/3 (3/4)
    assert E.community_taxon(three + [ident(5, 100, "2023-01-04")]) == 100


def test_an_ancestor_id_counts_against_only_when_it_disagrees():
    two = [ident(2, 100, "2023-01-01"), ident(3, 100, "2023-01-02")]
    assert E.community_taxon(two + [ident(4, 10, "2023-01-03")]) == 100
    assert E.community_taxon(two + [ident(4, 10, "2023-01-03", disagreement=True)]) == 10


# --- scoring ---------------------------------------------------------------------------

def test_tiers_follow_the_predeclared_lines():
    assert [E.tier_of(c) for c in (None, 999, 1000, 9999, 10000)] == \
        ["unknown", "below", "2", "2", "1"]


def test_observer_and_validators_are_never_counted_as_identifiers():
    o = obs([ident(1, 100, "2023-01-01"), ident(2, 100, "2023-01-02"), ident(3, 100, "2023-01-03")],
            counts={1: 50000, 2: 50000, 3: 50000})
    line = E.score_find(o, E.Cutoff(CUT, "dna_field"), TRUTH, E.TaxonResolver(TAXA), {3}, key)
    roles = {i["identifier"]: i["role"] for i in line["ids"]}
    assert roles == {"h1": "observer", "h2": "identifier", "h3": "validator"}
    pushed = E.score_find(o, E.Cutoff(CUT, "dna_field"), TRUTH, E.TaxonResolver(TAXA), set(), key, {2})
    assert {i["identifier"]: i["role"] for i in pushed["ids"]}["h2"] == "push_account"
    assert E.summarise([pushed])["all"]["each_species"]["n"] == 1      # h3 only, h2 is the push account
    s = E.summarise([line])
    assert s["all"]["each_species"]["n"] == 1      # only the one eligible tier-1 ID


def test_a_genus_level_id_counts_at_genus_only_and_a_retired_name_follows_its_successor():
    o = obs([ident(2, 10, "2023-01-01"), ident(3, 99, "2023-01-02"), ident(4, 110, "2023-01-03")],
            counts={2: 20000, 3: 20000, 4: 20000})
    line = E.score_find(o, E.Cutoff(CUT, "dna_field"), TRUTH, E.TaxonResolver(TAXA), set(), key)
    by = {i["identifier"]: i for i in line["ids"]}
    assert by["h2"]["species_right"] is None and by["h2"]["genus_right"] is True
    assert by["h3"]["species_right"] is True               # 99 retired -> 100
    assert by["h4"]["species_right"] is False and by["h4"]["genus_right"] is False
    s = E.summarise([line])["all"]
    assert s["best_species"] == {"n": 1, "right": 1, "accuracy": 1.0}
    assert s["each_species"]["n"] == 2 and s["each_genus"] == {"n": 3, "right": 2, "accuracy": 0.6667}


def test_species_is_not_scored_on_a_provisional_truth():
    o = obs([ident(2, 100, "2023-01-01")], counts={2: 20000})
    truth = E.TruthTaxa(species=None, genus=10, formal_species=False)
    line = E.score_find(o, E.Cutoff(CUT, "dna_field"), truth, E.TaxonResolver(TAXA), set(), key)
    assert line["ids"][0]["species_right"] is None and line["ids"][0]["genus_right"] is True


def test_strict_variant_drops_finds_the_observer_had_named_at_species():
    named = obs([ident(1, 100, "2023-01-01"), ident(2, 100, "2023-01-02")], counts={2: 20000})
    line = E.score_find(named, E.Cutoff(CUT, "dna_field"), TRUTH, E.TaxonResolver(TAXA), set(), key)
    assert line["observer_named_species_before_cutoff"] is True
    s = E.summarise([line])
    assert s["all"]["best_species"]["n"] == 1 and s["strict"]["best_species"]["n"] == 0


def test_community_taxon_is_rebuilt_as_it_stood_at_the_cutoff():
    o = obs([ident(2, 101, "2023-01-01"), ident(3, 101, "2023-01-02"),
             ident(4, 100, "2024-08-01"), ident(5, 100, "2024-08-02"), ident(6, 100, "2024-08-03")])
    line = E.score_find(o, E.Cutoff(CUT, "dna_field"), TRUTH, E.TaxonResolver(TAXA), set(), key)
    assert line["community"]["species_right"] is False      # then: 101, not today's 100
    assert line["community"]["genus_right"] is True


def test_scoring_lines_carry_hashed_identifiers_never_ids():
    k = E.hasher(b"secret")
    assert k(123456) != k(123457) and "123456" not in k(123456) and len(k(1)) == 16
    assert E.hasher(b"other")(123456) != k(123456)


def test_truth_taxa_species_only_for_formal_names():
    by = E.name_index(TAXA)
    t = E.truth_taxa({"name_kind": "formal", "label_rank": "species", "species": "Russula emetica",
                      "genus": "Russula"}, by, TAXA)
    assert (t.species, t.genus, t.formal_species) == (100, 10, True)
    t = E.truth_taxa({"name_kind": "provisional", "label_rank": "species",
                      "species": "Russula sp. 'IN01'", "genus": "Russula"}, by, TAXA)
    assert (t.species, t.genus, t.formal_species) == (None, 10, False)


def test_name_index_prefers_the_active_taxon():
    taxa = {**TAXA, 98: {**TAXA[99], "id": 98, "name": "Russula emetica", "is_active": False}}
    assert E.name_index(taxa)[("russula emetica", "species")] == 100


# --- the snapshot and the sample -------------------------------------------------------

RAILS = {
    "user_id": 1, "user_login": "someone", "created_at": "2021-11-01T15:14:24-04:00",
    "quality_grade": "research", "latitude": "39.1", "longitude": "-86.5", "private_latitude": None,
    "identifications": [{"id": 7, "user_id": 2, "user": {"id": 2, "login": "expert", "name": "E X"},
                         "taxon": {"id": 100, "rank": "species", "rank_level": 10, "ancestry": "1/5/10",
                                   "is_active": True, "name": "Russula emetica"},
                         "created_at": "2021-11-01T19:14:24.099Z", "updated_at": "2021-11-01T19:14:24.099Z",
                         "current": True, "body": "nice", "taxon_change_id": None}],
    "observation_field_values": [{"observation_field_id": 2330, "observation_field": {"name": "DNA Barcode ITS"},
                                  "value": "ACGT", "created_at": "2024-07-02T19:48:21.539Z",
                                  "updated_at": "2024-07-02T19:48:21.539Z", "user_id": 9}],
    "comments": [{"user_id": 3, "user": {"login": "c"}, "body": "BLAST says R. emetica",
                  "created_at": "2024-07-01T00:00:00Z"}],
}


def test_snapshot_rows_hold_no_logins_names_coordinates_or_values():
    row = E.slim_observation("42", RAILS, {"community_taxon_id": 100,
                                          "identifications": [{"user": {"id": 2, "identifications_count": 25117}}]})
    text = json.dumps(row)
    for leak in ("someone", "expert", "E X", "39.1", "-86.5", "ACGT", "nice", "BLAST says"):
        assert leak not in text
    assert row["identifications"][0]["taxon_path"] == [1, 5, 10, 100]
    assert row["identifier_counts"] == {2: 25117}
    assert row["comments"][0]["mentions_dna"] is True
    assert E.dna_cutoff(row).source == "comment"


def test_sample_depends_only_on_the_set_and_the_seed():
    ids = [str(i) for i in range(1000)]
    a = E.draw_sample(ids, 50, seed=7)
    assert a == E.draw_sample(list(reversed(ids)), 50, seed=7)
    assert a != E.draw_sample(ids, 50, seed=8) and len(set(a)) == 50


# --- the fetcher's manners -----------------------------------------------------------

class FakeResponse:
    def __init__(self, code, body=None, headers=None):
        self.status_code, self._body, self.headers, self.text = code, body, headers or {}, ""

    def json(self):
        return self._body


class FakeSession:
    def __init__(self, answers):
        self.answers, self.urls, self.headers = list(answers), [], {}

    def get(self, url, params=None, timeout=None):
        self.urls.append(url)
        return self.answers.pop(0)


def test_fetcher_backs_off_on_429_then_caches_so_a_rerun_costs_nothing(tmp_path):
    sleeps = []
    s = FakeSession([FakeResponse(429, headers={"Retry-After": "120"}), FakeResponse(200, RAILS)])
    c = E.HistoryClient(tmp_path, session=s, sleep=sleeps.append)
    got = c.observation("42")
    assert got["user_id"] == 1 and c.calls == 2
    assert max(sleeps) >= 120                       # honoured Retry-After
    c2 = E.HistoryClient(tmp_path, session=FakeSession([]), sleep=sleeps.append)
    assert c2.observation("42")["user_id"] == 1 and c2.calls == 0
    assert s.headers["User-Agent"].startswith("MycoMap-Vision")


def test_fetcher_paces_every_request_to_at_most_one_a_second(tmp_path):
    t = [0.0]
    sleeps = []

    def sleep(d):
        sleeps.append(d)
        t[0] += d
    s = FakeSession([FakeResponse(200, RAILS) for _ in range(3)])
    c = E.HistoryClient(tmp_path, session=s, sleep=sleep)
    c.pace._clock = lambda: t[0]
    for oid in ("1", "2", "3"):
        c.observation(oid)
    assert sum(sleeps) >= 2 * 1.0                   # three calls span at least two seconds


def test_fetcher_records_a_deleted_observation_and_stops_at_the_call_budget(tmp_path):
    c = E.HistoryClient(tmp_path, session=FakeSession([FakeResponse(404)]), sleep=lambda d: None,
                        max_calls=1)
    assert c.observation("5")["missing"] == 404
    with pytest.raises(E.CallBudgetSpent):
        c.observation("6")


def test_snapshot_is_hashed_and_refuses_unfetched_observations(tmp_path):
    s = FakeSession([FakeResponse(200, {"results": [{"id": 42, "community_taxon_id": 100,
                                                     "identifications": []}]}),
                     FakeResponse(200, RAILS), FakeResponse(200, {"results": [{"identifications_count": 3}]})])
    c = E.HistoryClient(tmp_path / "cache", session=s, sleep=lambda d: None)
    E.fetch(c, ["42"], log=lambda *a: None)
    out = tmp_path / "snap.jsonl.gz"
    m = E.build_snapshot(c, ["42"], out, {"set": "test"})
    assert m["snapshot_sha256"] == E.sha256_file(out) and m["ids_sha256"] == E.ids_sha256(["42"])
    rows = E.read_snapshot(out)
    assert rows[0]["identifier_counts"] == {2: 3}   # filled from /users when v2 lacked it
    with pytest.raises(RuntimeError, match="not fetched"):
        E.build_snapshot(c, ["42", "43"], tmp_path / "x.jsonl.gz", {})


def test_mushroom_observer_consensus_name_is_not_a_dna_field():
    o = obs([], ofvs=[{"field_id": 5006, "name": "Mushroom Observer Consensus Name",
                       "created_at": "2016-06-29T00:00:00Z"},
                      {"field_id": 16718, "name": "Reads in Consensus (RiC)",
                       "created_at": "2024-01-01T00:00:00Z"}])
    c = E.dna_cutoff(o)
    assert c.source == "dna_field" and c.at.startswith("2024-01-01")


def test_an_observation_that_is_not_the_validated_find_is_flagged_not_scored():
    fungus = obs([ident(2, 100, "2023-01-01")])
    fungus["identifications"][0]["taxon_path"] = [E.FUNGI, 5, 10, 100]
    bird = obs([ident(2, 100, "2023-01-01")])
    bird["identifications"][0]["taxon_path"] = [3, 7251, 100]     # Aves, not Fungi
    assert E.not_this_observation(fungus, {"source_id": "42"}, set()) is None
    assert E.not_this_observation(bird, {"source_id": "42"}, set()) == "not_a_fungus_on_inat"
    assert E.not_this_observation(fungus, {"source_id": "42"}, {"42"}) == "mo_id_collision"


def test_ids_after_the_cutoff_are_listed_but_never_scored():
    o = obs([ident(2, 101, "2023-01-01", current=False, updated="2024-08-01"),
             ident(2, 100, "2024-08-01"), ident(3, 100, "2024-09-01")],
            counts={2: 20000, 3: 20000})
    line = E.score_find(o, E.Cutoff(CUT, "dna_field"), TRUTH, E.TaxonResolver(TAXA), set(), key)
    flags = sorted((i["identifier"], i["before_cutoff"], i["species_right"]) for i in line["ids"])
    assert flags == [("h2", False, True), ("h2", True, False), ("h3", False, True)]
    s = E.summarise([line])["all"]
    assert s["each_species"] == {"n": 1, "right": 0, "accuracy": 0.0}


def test_a_trashed_sequence_dated_at_the_observations_creation_is_set_aside():
    ours = [("com_seq_created_trashed", "2021-11-01 19:30:00"),
            ("com_seq_created_trashed", "2023-05-01 00:00:00"),
            ("com_seq_created", "2021-11-01 20:00:00")]
    keep, dropped = E.drop_backdated(ours, "2021-11-01T15:14:24-04:00")
    assert dropped == 1 and ("com_seq_created", "2021-11-01 20:00:00") in keep
    assert E.kind_of("org_name_update_identification") == "name_push"


def test_sql_runner_strips_comments_batches_ids_and_parses_tabs():
    seen = []

    def runner(host, sql):
        seen.append(sql)
        ids = sql.split("IN (")[1].split(")")[0].split(",")
        return "inat_observation_id\tsource\tearliest_timestamp_utc\n" + \
            "".join(f"{i}\tcom_seq_created\t2024-01-01 00:00:00\n" for i in ids)
    rows = E.run_sql("h", "-- note\nSELECT x FROM t WHERE id IN ({ids}) -- tail\n",
                     [str(i) for i in range(1, 8)] + ["1; drop"], batch=3, runner=runner)
    assert len(seen) == 3 and not any("--" in q or "drop" in q for q in seen)
    assert [r["inat_observation_id"] for r in rows] == [str(i) for i in range(1, 8)]
