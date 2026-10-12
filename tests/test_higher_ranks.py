"""Order, class and phylum in the public API's answers (higher.py, docs/api.md §4)."""

from collections import Counter

import numpy as np
from test_api import post_photos
from test_api_v1_identify import app_token, bearer, identify, v1_app

from mycomap_vision import identify as identify_mod
from mycomap_vision.higher import HigherRanks

LINEAGE = {"A": {"order": "Agaricales", "class": "Agaricomycetes", "phylum": "Basidiomycota"},
           "B": {"order": "Boletales", "class": "Agaricomycetes", "phylum": "Basidiomycota"}}


def test_a_higher_name_scores_as_its_best_genus():
    h = HigherRanks(["A", "B", "C"], LINEAGE, Counter({"A": 4, "B": 3, "C": 9}))
    out = h.candidates(np.array([0.2, 0.7, 0.9], np.float32), top_k=5)
    assert [c["name"] for c in out["order"]] == ["Boletales", "Agaricales"]
    assert out["order"][0]["score"] == 0.7           # C has no order: it adds to none
    assert [c["name"] for c in out["class"]] == ["Agaricomycetes"]
    assert out["class"][0]["score"] == 0.7
    assert out["class"][0]["reference_records"] == 7   # A's 4 + B's 3; C's are not placed
    assert all(c["confidence"] is None and c["place"] is None
               for cands in out.values() for c in cands)


def test_without_a_taxonomy_there_are_no_higher_ranks():
    assert HigherRanks.from_taxonomy(None, ["A"], Counter()) is None


def with_lineage(monkeypatch):
    monkeypatch.setattr(identify_mod.HigherRanks, "from_taxonomy",
                        classmethod(lambda cls, tax, genera, counts: cls(genera, LINEAGE, counts)))


def test_v1_answers_carry_order_class_and_phylum_never_confident(conn, tmp_path, monkeypatch):
    with_lineage(monkeypatch)
    client = v1_app(conn, tmp_path)
    for reds, genus in (((247, 245), "A"), ((12, 15), "B")):
        body = identify(client, bearer(app_token()), reds=reds).json()
        assert body["ranks"]["genus"][0]["name"] == genus
        # The best order is the best genus's own (the best of its genera).
        for rank in ("order", "class", "phylum"):
            assert body["ranks"][rank][0]["name"] == LINEAGE[genus][rank]
    for rank in ("order", "class", "phylum"):
        assert all(c["confidence"] is None and c["confident"] is False
                   for c in body["ranks"][rank])
        assert body["per_photo"][0]["ranks"][rank]
    assert set(body["per_photo"][0]["overall_top"]) == {"species", "genus", "family"}
    assert body["answer"] is None or body["answer"]["rank"] in ("species", "genus", "family")


def test_v1_lists_the_higher_ranks_empty_without_a_taxonomy(conn, tmp_path):
    body = identify(v1_app(conn, tmp_path), bearer(app_token())).json()
    assert body["ranks"]["order"] == body["ranks"]["class"] == body["ranks"]["phylum"] == []


def test_the_websites_answer_is_unchanged(conn, tmp_path, monkeypatch):
    with_lineage(monkeypatch)
    client = v1_app(conn, tmp_path)
    [r] = post_photos(client, [247], "m1/nearest").json()["results"]
    assert set(r["ranks"]) == {"species", "genus", "family"}
    assert set(r["per_photo"][0]["ranks"]) == {"species", "genus", "family"}
