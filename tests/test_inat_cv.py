from collections import Counter

import pytest

from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import evaluate, inat_cv
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.evaluate import Record
from mycomap_vision.inat_cv import (Truth, best_per_rank, parse_aggregated, plain_binomial,
                                    resolve_truth, score_records)


def result(tid, name, rank, vision, combined):
    return {"taxon": {"id": tid, "name": name, "rank": rank},
            "vision_score": vision, "normalized_combined_score": combined}


def test_aggregated_answers_are_split_by_rank_with_both_scores():
    body = {"results": [result(1, "Russula emetica", "species", 60, 80),
                        result(10, "Russula", "genus", 90, 95),
                        result(100, "Russulaceae", "family", 99, 99),
                        result(999, "Fungi", "kingdom", 100, 100)]}
    got = parse_aggregated(body)
    assert got["species"][1] == {"name": "Russula emetica", "vision": 60.0, "combined": 80.0}
    assert list(got["genus"]) == [10] and list(got["family"]) == [100]


def test_the_strongest_photo_wins_for_each_taxon():
    cap = parse_aggregated({"results": [result(1, "A a", "species", 30, 30),
                                        result(2, "B b", "species", 70, 70)]})
    gills = parse_aggregated({"results": [result(1, "A a", "species", 90, 90)]})
    ranked = best_per_rank([cap, gills], "vision")["species"]
    assert ranked == [(1, 90.0), (2, 70.0)]


def test_provisional_and_open_names_have_no_plain_binomial():
    assert plain_binomial("Russula emetica") == "Russula emetica"
    assert plain_binomial("Amanita muscaria var. guessowii") == "Amanita muscaria"
    for name in ["Clitocybe sp. 'IN13'", "Cortinarius 'fulvo-IN01'", "Russula", "Russula sp."]:
        assert plain_binomial(name) is None


class FakeTaxa:
    """Just enough of iNat's taxa API: exact-name search and full taxon reads."""

    def __init__(self, taxa):
        self.taxa = {t["id"]: t for t in taxa}
        self.requests = []

    def get(self, path, params=None):
        self.requests.append((path, params))
        if path == "/taxa":
            active = params["is_active"] == "true"
            return {"results": [t for t in self.taxa.values()
                                if t["name"].lower() == params["q"].lower()
                                and t["rank"] == params["rank"] and t.get("is_active", True) == active]}
        return {"results": [self.taxa[int(path.rsplit("/", 1)[1])]]}


FAM = {"id": 100, "name": "Tricholomataceae", "rank": "family"}
GEN = {"id": 10, "name": "Clitocybe", "rank": "genus", "ancestors": [FAM]}


def test_a_provisional_name_iNat_carries_is_matched_as_is():
    taxa = FakeTaxa([FAM, GEN, {"id": 5, "name": "Clitocybe sp. 'IN13'", "rank": "species",
                                "ancestors": [FAM, GEN]}])
    rec = Record("1", "Clitocybe sp. 'IN13'", "Clitocybe", "X", None, None, [])
    assert resolve_truth(taxa, rec) == Truth(species=5, genus=10, family=100, species_known=True)


def test_a_name_iNat_lacks_still_gets_genus_and_family_in_iNats_taxonomy():
    taxa = FakeTaxa([FAM, GEN])
    rec = Record("1", "Clitocybe sp. 'IN99'", "Clitocybe", "Our family spelling", None, None, [])
    assert resolve_truth(taxa, rec) == Truth(species=None, genus=10, family=100,
                                             species_known=False)


def test_a_retired_name_follows_to_its_current_taxon():
    taxa = FakeTaxa([FAM, GEN,
                     {"id": 6, "name": "Clitocybe old", "rank": "species", "is_active": False,
                      "current_synonymous_taxon_ids": [7]},
                     {"id": 7, "name": "Clitocybe new", "rank": "species", "ancestors": [FAM, GEN]}])
    rec = Record("1", "Clitocybe old", "Clitocybe", "", None, None, [])
    assert resolve_truth(taxa, rec).species == 7


def test_species_accuracy_is_also_given_on_names_iNat_knows():
    recs = [Record("a", "A a", "A", "F", None, None, []),
            Record("b", "A sp. 'X1'", "A", "F", None, None, [])]
    scores = {"a": [parse_aggregated({"results": [result(1, "A a", "species", 90, 90),
                                                  result(10, "A", "genus", 95, 95)]})],
              "b": [parse_aggregated({"results": [result(1, "A a", "species", 80, 80),
                                                  result(10, "A", "genus", 95, 95)]})]}
    truths = {"a": Truth(1, 10, 100, True), "b": Truth(None, 10, 100, False)}
    out = score_records(recs, scores, truths, Counter({"A a": 3}), "vision")
    assert out["species"]["all"]["top1"] == 0.5
    assert out["species"]["names iNat knows"] == {"n": 1, "top1": 1.0, "top5": 1.0}
    assert out["genus"]["all"]["top1"] == 1.0


class FakeInat(FakeTaxa):
    def __init__(self, taxa, answer):
        super().__init__(taxa)
        self.answer, self.scored = answer, []

    @property
    def calls(self):
        return len(self.scored)

    def score_image(self, photo_id, image, lat, lng):
        self.scored.append((photo_id, lat, lng))
        return self.answer


def test_the_baseline_joins_a_comparison_on_exactly_its_records(conn, tmp_path, monkeypatch):
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    monkeypatch.setattr(inat_cv, "photo_inputs",
                        lambda conn, rec: [(p, store.root / f"p/{p}.png") for p in rec.photo_rows])
    cmp = evaluate.compare(conn, ["m1"], embeddings_root=tmp_path / "emb", log=lambda s: None)
    taxa = [{"id": 100, "name": "F", "rank": "family"},
            {"id": 10, "name": "A", "rank": "genus", "ancestors": [{"id": 100, "rank": "family"}]},
            {"id": 1, "name": "A x", "rank": "species",
             "ancestors": [{"id": 100, "rank": "family"}, {"id": 10, "rank": "genus"}]}]
    fake = FakeInat(taxa, {"results": [result(1, "A x", "species", 90, 95),
                                       result(10, "A", "genus", 95, 97)]})
    out = inat_cv.run(conn, cmp["comparison_id"], fake, tmp_path / "emb", log=lambda s: None)
    board = evaluate.scoreboard(conn, cmp["comparison_id"])
    rows = {(r["backbone"], r["method"]) for r in board}
    assert ("external:inat-cv", "vision-max") in rows and ("external:inat-cv", "combined-max") in rows
    assert len({r["record_set"] for r in board}) == 1
    assert out["test_records"] == cmp["test_records"]
    # iNat's public coordinates are what's sent (the fixture's inat observation location).
    assert all((lat, lng) == (39.1, -86.5) for _, lat, lng in fake.scored)


def test_the_baseline_refuses_a_comparison_whose_records_have_changed(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    cmp = evaluate.compare(conn, ["m1"], embeddings_root=tmp_path / "emb", log=lambda s: None)
    conn.execute("update eval_runs set record_set = 'stale'")
    conn.commit()
    with pytest.raises(RuntimeError, match="changed since it ran"):
        inat_cv.run(conn, cmp["comparison_id"], FakeInat([], {}), tmp_path / "emb",
                    log=lambda s: None)


def test_a_dropped_connection_is_retried_not_fatal(tmp_path):
    import requests as rq

    class Flaky:
        headers = {}

        def __init__(self):
            self.n = 0

        def request(self, method, url, timeout, **kw):
            self.n += 1
            if self.n == 1:
                raise rq.exceptions.SSLError("EOF occurred in violation of protocol")

            class R:
                status_code = 200

                def json(self):
                    return {"results": []}
            return R()

    client = inat_cv.InatClient("jwt", tmp_path, session=Flaky(), interval=0,
                                sleep=lambda s: None)
    assert client.get("/taxa/1") == {"results": []}
    assert client.calls == 2
