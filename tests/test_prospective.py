import io
import json

from PIL import Image

from conftest import inat_obs

from mycomap_vision import prospective
from mycomap_vision.records import build_records, save_records


def jpeg():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 0, 0)).save(buf, "JPEG")
    return buf.getvalue()


class FakeFetcher:
    def __init__(self, obs):
        self.obs = obs

    def observations(self, ids):
        return {i: self.obs[i] for i in ids if i in self.obs}

    def photo(self, url):
        return jpeg()


class FakeIdentifier:
    backbone, method, records = "m1", "hybrid", 100

    def __init__(self):
        self.contexts = []

    def identify(self, backbone_model, images, context=None):
        self.contexts.append(context)
        top = lambda name: [{"name": name, "confidence": 0.7}]  # noqa: E731
        return {"ranks": {"family": top("F"), "genus": top("A"), "species": top("A x")}}


def candidate(oid, name="A sp. pending", continent="North America"):
    return {"observation_id": oid, "scientific_name": name, "continent": continent}


def test_only_unassessed_north_american_candidates_without_a_prediction_are_queued(conn):
    prospective.save_candidates(conn, [candidate("10"), candidate("11"),
                                       candidate("12", continent="Europe")], "t0")
    save_records(conn, build_records([{"observation_id": "11", "scientific_name": "A x",
                                       "continent": "North America",
                                       "validation_status_1": "yes"}], "t1"))
    assert prospective.unpredicted(conn, "m1", "hybrid") == ["10"]


def test_predictions_use_iNats_public_location_and_are_saved_with_their_time(conn):
    prospective.save_candidates(conn, [candidate("10")], "t0")
    ident = FakeIdentifier()
    fetcher = FakeFetcher({"10": inat_obs(10, photos=[(0, 1, "cc0", "x")])})
    stats = prospective.predict_pending(conn, ident, object(), ["10"], fetcher, log=lambda s: None)
    assert stats == {"predicted": 1}
    assert (ident.contexts[0].latitude, ident.contexts[0].longitude) == (39.1, -86.5)
    row = conn.execute("select backbone, method, photos, result_json from predictions").fetchone()
    assert (row[0], row[1], row[2]) == ("m1", "hybrid", 1)
    assert json.loads(row[3])["species"][0]["name"] == "A x"


def test_a_candidate_gone_from_iNat_is_counted_not_fatal(conn):
    stats = prospective.predict_pending(conn, FakeIdentifier(), object(), ["99"], FakeFetcher({}),
                                        log=lambda s: None)
    assert stats == {"gone from iNat": 1}


def test_only_predictions_made_before_the_record_turned_green_count(conn):
    prospective.save_candidates(conn, [candidate("10"), candidate("20")], "t0")
    ident = FakeIdentifier()
    fetcher = FakeFetcher({"10": inat_obs(10, photos=[(0, 1, "cc0", "x")]),
                           "20": inat_obs(20, photos=[(0, 2, "cc0", "x")])})
    prospective.predict_pending(conn, ident, object(), ["10", "20"], fetcher, log=lambda s: None)
    conn.execute("update predictions set predicted_at = '2026-09-01T00:00:00+00:00'")
    green = lambda oid, name: {"observation_id": oid, "scientific_name": name,  # noqa: E731
                               "genus": name.split()[0], "family": "F",
                               "continent": "North America", "validation_status_1": "yes"}
    save_records(conn, build_records([green("10", "A x"), green("20", "A y")],
                                     "2026-09-10T00:00:00+00:00"))
    # A third record turned green BEFORE it was predicted: it must not count.
    conn.execute("insert into predictions values ('30', 'm1', 'hybrid', "
                 "'2026-09-20T00:00:00+00:00', 'v', 1, 1, ?)",
                 (json.dumps({"species": [{"name": "A z", "confidence": 1.0}]}),))
    save_records(conn, build_records([green("10", "A x"), green("20", "A y"), green("30", "A z")],
                                     "2026-09-15T00:00:00+00:00"))
    [model] = prospective.report(conn)
    assert model["predicted"] == 3 and model["resolved"] == 2
    assert model["species_top1"] == 0.5 and model["genus_top1"] == 1.0
    assert model["mean_species_confidence"] == 0.7


def predicted_then_validated(conn, pairs):
    """Records predicted on 9/1 (the answer given) that turned green on 9/10 (the DNA name)."""
    green = []
    for i, (answer, dna_name) in enumerate(pairs):
        oid = str(50 + i)
        conn.execute("insert into predictions values (?, 'm1', 'hybrid', "
                     "'2026-09-01T00:00:00+00:00', 'v', 1, 1, ?)",
                     (oid, json.dumps({"species": [{"name": answer, "confidence": 0.6}],
                                       "genus": [{"name": answer.split()[0], "confidence": 0.9}]})))
        green.append({"observation_id": oid, "scientific_name": dna_name,
                      "genus": dna_name.split()[0], "family": "F",
                      "continent": "North America", "validation_status_1": "yes"})
    save_records(conn, build_records(green, "2026-09-10T00:00:00+00:00"))
    [model] = prospective.report(conn)
    return model


def test_an_answer_spelled_another_way_than_the_dna_name_is_right(conn):
    conn.executescript(prospective.SCHEMA)
    model = predicted_then_validated(conn, [
        ("Inocybe PNW18", "Inocybe sp. 'PNW18'"),          # an old spelling, gone from .org
        ("Clitocybe sp. 'PNW01'", 'Clitocybe "sp-PNW01"'),  # the label of a name not yet fixed
        ("Inocybe sp. 'PNW18'", "Inocybe sp. 'PNW19'")])    # another code: wrong
    assert model["resolved"] == 3
    assert model["species_top1"] == round(2 / 3, 4)
    assert model["genus_top1"] == 1.0


def test_an_answer_that_differs_by_more_than_spelling_is_still_wrong(conn):
    conn.executescript(prospective.SCHEMA)
    model = predicted_then_validated(conn, [
        ("Mycena sp. 'IN7'", "Mycena sp. 'IN07'"),                           # a person decides
        ("Craterellus neotubaeformis", "Craterellus sp. 'neotubaeformis'")])
    assert model["resolved"] == 2 and model["species_top1"] == 0.0



def test_an_occurrence_prior_that_cannot_leave_the_candidates_find_out_skips_it(conn):
    from mycomap_vision.occprior import TuningRefused

    class Model:
        def __init__(self):
            self.asked = []

        def leak_check(self, uuids, allow_missing=False):
            self.asked.append(uuids)
            if uuids == ["uuid-20"]:
                raise TuningRefused("counted, and no leave-one-out index")
            return {}

    prospective.save_candidates(conn, [candidate("10"), candidate("20")], "t0")
    ident = FakeIdentifier()
    ident.model = Model()
    fetcher = FakeFetcher({"10": inat_obs(10, photos=[(0, 1, "cc0", "x")]),
                           "20": inat_obs(20, photos=[(0, 2, "cc0", "x")])})
    stats = prospective.predict_pending(conn, ident, object(), ["10", "20"], fetcher,
                                        log=lambda s: None)
    assert stats == {"predicted": 1,
                     "skipped: occurrence prior can't leave its own find out": 1}
    assert ident.model.asked == [["uuid-10"], ["uuid-20"]]
