"""E14, iNat's computer vision on a release's finds: what is sent, how a find is scored,
and the seal."""

import io
import json
import sqlite3

import pytest
from PIL import Image

from mycomap_vision import expert_ids as E
from mycomap_vision import paper_inat_cv as P


def jpeg(w, h):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 80, 40)).save(buf, "JPEG")
    return buf.getvalue()


def answer(*taxa):
    """score_image body: (rank, taxon id, vision, combined) rows."""
    return {"results": [{"taxon": {"id": t, "rank": r, "name": f"T{t}"}, "vision_score": v,
                         "combined_score": c} for r, t, v, c in taxa]}


class FakeClient:
    def __init__(self, answers):
        self.answers, self.calls, self.sent = answers, 0, []

    def score_image(self, pk, image, lat, lng, observed_on=None):
        body = image() if callable(image) else image
        self.calls += 1
        self.sent.append({"photo": pk, "lat": lat, "lng": lng, "date": observed_on,
                          "size": Image.open(io.BytesIO(body)).size})
        return {**self.answers[pk], "_scored_at": "2026-11-01T00:00:00Z"}


class FakeRelease:
    def __init__(self, test_for=None):
        self.id = "v1-test" if test_for else "v1"
        self.manifest = {"test_for": test_for}
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row         # as dataset_release._ro
        self.db.execute("create table records (record_key text, included integer, genus text, "
                        "observed_on text)")
        self.db.execute("create table photos (photo_key text, record_key text, included integer, "
                        "position integer, manifest_photo_id integer)")
        self.db.executemany("insert into records values (?,?,?,?)",
                            [("a", 1, "Russula", "2025-09-01"), ("b", 1, "Russula", None)])
        self.db.executemany("insert into photos values (?,?,?,?,?)",
                            [("a1", "a", 1, 0, 11), ("a2", "a", 1, 1, 12), ("a3", "a", 0, 2, 13),
                             ("b1", "b", 1, 0, 21)])

    def public_places(self):
        return {"a": {"public_lat_micro": 39_100_000, "public_lon_micro": -86_500_000}}

    def cite(self):
        return {"dataset_release": self.id}


TRUTH = {"a": E.TruthTaxa(100, 10, True), "b": E.TruthTaxa(None, 10, False)}
ANSWERS = {
    "a1": answer(("species", 101, 60, 40), ("genus", 10, 90, 90)),
    "a2": answer(("species", 100, 50, 70), ("genus", 10, 80, 85)),
    "b1": {"results": []},                      # iNat suggested nothing
}


@pytest.fixture
def run(tmp_path, monkeypatch):
    monkeypatch.setattr(P.config, "DATA_DIR", tmp_path)

    def go(rel=None, **kw):
        client = FakeClient(ANSWERS)
        meta = P.run(rel or FakeRelease(), "t", ["a", "b"], client,
                     lambda pk: jpeg(2048, 1536) if pk == "a1" else jpeg(1024, 768), TRUTH,
                     log=lambda *a: None, **kw)
        lines = [json.loads(l) for l in (tmp_path / "paper-comparisons/e14/t/finds.jsonl").open()]
        return client, meta, {l["record_key"]: l for l in lines}
    return go


def test_only_included_photos_go_at_1024_px_with_the_public_point_and_date(run):
    client, meta, _ = run()
    sent = {s["photo"]: s for s in client.sent}
    assert set(sent) == {"a1", "a2", "b1"}                  # a3 is not included
    assert max(sent["a1"]["size"]) == 1024 and sent["a2"]["size"] == (1024, 768)
    assert (sent["a1"]["lat"], sent["a1"]["lng"], sent["a1"]["date"]) == (39.1, -86.5, "2025-09-01")
    assert sent["b1"]["lat"] is None and sent["b1"]["date"] is None   # none frozen: none sent
    assert meta["photos"] == 3 and meta["first_scored_at"] == "2026-11-01T00:00:00Z"
    assert meta["place_rule"].startswith("the release's frozen public point")
    _, _, lines = run()
    assert lines["a"]["photo_ids"] == [11, 12]               # what iNat saw, in order


def test_best_photo_wins_and_photo_only_and_combined_stay_apart(run):
    _, _, lines = run()
    a = lines["a"]
    assert a["vision"]["species"][0] == [101, "T101", 60.0]  # photo only: the wrong one leads
    assert a["combined"]["species"][0] == [100, "T100", 70.0]  # with place: the right one
    assert a["answered"] == {"vision": True, "combined": True} and a["px"] == 1024
    assert a["vision_hits"]["species"]["top1"] is False and a["vision_hits"]["species"]["top3"]
    assert a["combined_hits"]["species"]["top1"] is True
    assert a["first_photo"]["combined"]["species"][0][0] == 101


def test_no_suggestion_is_wrong_and_species_waits_for_a_formal_name(run):
    _, meta, lines = run()
    b = lines["b"]
    assert b["vision_hits"]["species"] is None               # provisional truth: genus only
    assert b["vision_hits"]["genus"]["top10"] is False       # nothing suggested = a miss
    assert b["answered"] == {"vision": False, "combined": False}
    assert meta["finds_with_no_suggestion"] == 1
    s = P.summarise(list(lines.values()))
    assert s["vision_genus"]["n"] == 2 and s["vision_genus"]["top1"] == 0.5
    assert s["vision_species"]["n"] == 1


def test_a_truth_inat_does_not_know_stays_in_as_a_miss():
    line = {"vision_hits": {"species": P.hits([[5, "X y", 90.0]], None), "genus": None},
            "combined_hits": {"species": None, "genus": None}}
    assert P.summarise([line])["vision_species"] == {"n": 1, "top1": 0.0, "top3": 0.0,
                                                     "top5": 0.0, "top10": 0.0}


def test_a_sealed_test_release_sends_nothing_before_the_freeze_tag(run, monkeypatch):
    monkeypatch.setattr(E, "freeze_tagged", lambda *a: False)
    with pytest.raises(RuntimeError, match="sealed"):
        run(FakeRelease(test_for="v1"))
    monkeypatch.setattr(E, "freeze_tagged", lambda *a: True)
    client, _, _ = run(FakeRelease(test_for="v1"))
    assert client.calls == 3


def test_the_fallback_sample_is_fixed_by_the_ids_and_seed():
    ids = [f"r{i}" for i in range(5000)]
    a = P.draw_fallback(ids, 2000, 20261030)
    assert a == P.draw_fallback(list(reversed(ids)), 2000, 20261030) and len(a) == 2000


def test_photos_at_or_under_1024_px_are_sent_untouched():
    small = jpeg(1024, 700)
    assert P.at_most_1024(small)[0] == small
    big, w, h = P.at_most_1024(jpeg(3000, 1000))
    assert (w, h) == (1024, 341) and big != small
