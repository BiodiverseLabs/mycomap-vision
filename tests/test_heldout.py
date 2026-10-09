"""Held-out benchmarks (heldout.py, heldout_report.py): the answer key, the freeze (sealed
or for development, and again on a new snapshot), photos kept apart, Vision's and iNat's
answers, the dev/test split and the report."""

import csv
import io
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import inat_obs
from mycomap_vision import heldout, heldout_report, holdouts, photos
from mycomap_vision.embed import SCHEMA as EMBED_SCHEMA
from mycomap_vision.heldout import (SOURCE_FORMAT, SOURCE_INDEX, SOURCE_INDEX_ONE_WORD,
                                    SOURCE_OTHER_GENUS, SOURCE_OTHER_SPECIES,
                                    SOURCE_TITLE_ONE_WORD, TRUTH_NONE, TRUTH_ONE_WORD,
                                    TRUTH_SPECIES, answer_key, name_key)
from mycomap_vision.inat import save_batch
from mycomap_vision.licenses import STATIC_HOST
from mycomap_vision.photos import HostPolicy
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

OPEN = "inaturalist-open-data.s3.amazonaws.com"
RED, GREEN = (220, 20, 20), (20, 220, 20)
NAME = "bench-1"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)


# --- the answer key ----------------------------------------------------------------------

def test_the_answer_is_the_title_where_it_differs_from_the_index_name():
    key = answer_key("Polyporales", "373882777 - Trametes gibbosa")
    assert key == heldout.AnswerKey("Trametes gibbosa", TRUTH_SPECIES, SOURCE_INDEX_ONE_WORD)


def test_without_a_differing_title_the_index_name_is_the_answer():
    assert answer_key("Russula emetica", None) == heldout.AnswerKey(
        "Russula emetica", TRUTH_SPECIES, SOURCE_INDEX)
    assert answer_key("", None).status == TRUTH_NONE


@pytest.mark.parametrize("index, title, source", [
    ("Russula rosea", "Russula emetica", SOURCE_OTHER_SPECIES),
    ("Tubariua hiemalis", "Tubaria hiemalis", SOURCE_OTHER_GENUS),
    ('Tricholoma "sp-IN13"', "Tricholoma sp. 'IN13'", SOURCE_FORMAT),
])
def test_how_the_index_name_differs_from_the_title_is_audited(index, title, source):
    assert answer_key(index, title) == heldout.AnswerKey(title, TRUTH_SPECIES, source)


def test_a_one_word_title_is_flagged_stale_and_answers_above_species_only(conn):
    key = answer_key("Russula emetica", "Russula")
    assert (key.truth, key.status, key.source) == ("Russula", TRUTH_ONE_WORD,
                                                   SOURCE_TITLE_ONE_WORD)
    save_records(conn, build_records([green(1, "Russula emetica")], "t"))
    truth = heldout.Labeller(conn).truth("Russula")
    assert (truth.species, truth.genus, truth.family) == ("", "Russula", "Russulaceae")


@pytest.mark.parametrize("written, known", [
    ('Coprinopsis "uliginicola-CA01"', "Coprinopsis sp. 'uliginicola-CA01'"),
    ("Inocybe sp-PNW01", "Inocybe sp. 'PNW01'"),
    ("Cortinarius 'X PNW01'", "Cortinarius sp. 'X-PNW01'"),
    ("Hygrocybe glutinipes rubra", "Hygrocybe glutinipes var. rubra"),
    ("russula  Emetica", "Russula emetica"),
])
def test_names_are_compared_with_writing_differences_set_aside(conn, written, known):
    assert name_key(written) == name_key(known)
    save_records(conn, build_records([green(1, known)], "t"))
    labeller = heldout.Labeller(conn)
    assert labeller.truth(written).label == known
    assert labeller.same(written, known)
    answer = {"species": [{"name": written, "confidence": 0.9}]}
    assert heldout_report.judge(answer, labeller.truth(known), labeller,
                                by_taxon_id=False)["species"][0] is True


def test_titles_come_from_the_export_through_the_links(tmp_path):
    links = write_csv(tmp_path / "links.csv", ["index_id", "record_id", "external_id"],
                      [(1, "c1", "11"), (2, "c2", "12"), (3, "c3", "12"), (4, "c4", "13")])
    titles = tmp_path / "titles.tsv"
    titles.write_text("record_id\tindex_name\ttitle\nc1\tPolyporales\t11 - Trametes gibbosa\n"
                      "c2\tA\tRussula emetica\nc3\tB\tRussula rosea\n", encoding="utf-8")
    # An observation linked to two records with different titles gets none.
    assert heldout.titles_by_observation(titles, links, ["11", "12", "13"]) == {
        "11": "Trametes gibbosa"}


# --- the toy world: a reference, and a frozen set -----------------------------------------

def green(oid, name, vdate="1/1/2026", family="Russulaceae"):
    return {"observation_id": str(oid), "scientific_name": name, "genus": name.split()[0],
            "family": family, "continent": "North America", "latitude": 39.1,
            "longitude": -86.5, "validation_status_1": "yes", "validation_date_1": vdate}


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v)


class Toy:
    """A backbone that sees colour: a red photo is a red vector."""
    name, dim = "toy", 4

    def __init__(self):
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        return np.stack([unit([*np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0), 1])
                         for im in images])


def jpeg(colour, size=(900, 600)):
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG")
    return buf.getvalue()


def reference(conn, root):
    """Two red Russula emetica and one green Amanita muscaria, embedded with 'toy'."""
    save_records(conn, build_records([green(1, "Russula emetica"), green(2, "Russula emetica"),
                                      green(3, "Amanita muscaria", family="Amanitaceae")], "t"))
    save_batch(conn, ["1", "2", "3"], [inat_obs(i, photos=[(0, i * 10, "cc0", OPEN)])
                                       for i in (1, 2, 3)], "t")
    conn.executescript(EMBED_SCHEMA)
    out = root / "toy"
    out.mkdir(parents=True)
    vecs = [unit([*RED, 1]), unit([*RED, 1]), unit([*GREEN, 1])]
    np.save(out / "shard-00000.npy", np.stack(vecs).astype(np.float16))
    np.save(out / "shard-00000.ids.npy", np.asarray([10, 20, 30], dtype=np.int64))
    conn.executemany("insert into embeddings values ('toy', ?, 0, ?, 'now')",
                     [(10, 0), (20, 1), (30, 2)])
    conn.commit()
    return out


POOL = [  # id, .com index name (com_name), title where it differs, longitude, split
    ("101", "Polyporales", "Russula emetica", -80.0, "dev"),
    ("102", "Amanita muscaria", None, -120.0, "test"),
    ("103", "Russula rosea", "Russula", -80.0, "dev"),
    ("104", "", None, -80.0, "test"),
    ("105", "Tubariua hiemalis", None, -80.0, "dev"),
    ("106", "Tubariua hiemalis", "Tubaria hiemalis", -80.0, "test"),
]


def read(path):
    return Path(path).read_text(encoding="utf-8")


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path


def inputs(tmp_path, pool=POOL):
    d = tmp_path / "in"
    d.mkdir(exist_ok=True)
    titles = d / "titles.tsv"
    titles.write_text("record_id\tindex_name\ttitle\n" + "".join(
        f"c{i}\t{index}\t{i} - {title}\n" for i, index, title, *_r in pool if title),
        encoding="utf-8")
    files = {
        "pool_csv": write_csv(d / "pool.csv", ["observation_id", "com_name", "lat", "lng",
                                               "country", "org_name"],
                              [(i, n, 40.0, lng, "US", "") for i, n, _t, lng, _sp in pool]),
        "titles_tsv": titles,
        "links_csv": write_csv(d / "links.csv", ["index_id", "record_id", "external_id"],
                               [(k, f"c{i}", i) for k, (i, *_r) in enumerate(pool)]),
    }
    splits = {sp: write_csv(d / f"{sp}.csv", ["observation_id"],
                            [(i,) for i, *_r, s in pool if s == sp]) for sp in ("dev", "test")}
    return files, splits


def frozen(conn, tmp_path, **kw):
    files, splits = inputs(tmp_path)
    return heldout.freeze(conn, NAME, splits=splits, **files, **kw)


class FakeHTTP:
    """iNat's photo hosts, serving every photo red."""
    urls: list = []

    def __init__(self):
        self.headers = {}

    def mount(self, *a):
        pass

    def get(self, url, timeout):
        self.urls.append(url)

        class R:
            status_code = 200
            content = jpeg(RED)
        return R()


def observations(ids):
    shots = {"101": [(0, 1011)], "102": [(0, 1021), (1, 1022)], "103": [(0, 1031)],
             "104": [(0, 1041)], "105": [(0, 1051)]}
    return [inat_obs(int(i), photos=[(pos, pid, "cc0", OPEN) for pos, pid in shots[i]])
            for i in ids if i in shots]


def fetched(conn, tmp_path, monkeypatch, ids=None):
    ids = ids or heldout.benchmark_ids(conn, NAME)
    heldout.fetch_details(conn, NAME, ids, session=object(),
                          fetch=lambda s, batch, lim: observations(batch), log=lambda s: None)
    monkeypatch.setattr(photos.requests, "Session", FakeHTTP)
    FakeHTTP.urls = []
    store = LocalStore(heldout.bench_dir(conn, NAME))
    return heldout.fetch_photos(conn, NAME, ids, store, policies={OPEN: HostPolicy(4, 0)},
                                poll=0.05, log=lambda s: None)


def predicted(conn, tmp_path, root, methods=("nearest",), **kw):
    from mycomap_vision.identify import Identifier
    toy = Toy()
    out = heldout.predict(conn, NAME, "toy", list(methods), heldout.benchmark_ids(conn, NAME),
                          lambda: toy, make_identifier=lambda m: Identifier(
                              conn, "toy", m, embeddings_root=root, photo_info=False),
                          log=lambda s: None, **kw)
    return out, toy


# --- freeze -------------------------------------------------------------------------------------

def test_freeze_stores_the_answer_key_split_and_input_hashes(conn, tmp_path):
    out = frozen(conn, tmp_path)
    assert out["records"] == 6 and out["splits"] == {"dev": 3, "test": 3}
    assert out["answer_key"] == {TRUTH_SPECIES: 4, TRUTH_ONE_WORD: 1, TRUTH_NONE: 1}
    assert out["answer_source"] == {SOURCE_INDEX: 3, SOURCE_INDEX_ONE_WORD: 1,
                                    SOURCE_OTHER_GENUS: 1, SOURCE_TITLE_ONE_WORD: 1}
    row = conn.execute("select truth_name, index_name, name_source, split from "
                       "heldout_records where observation_id = '101'").fetchone()
    assert tuple(row) == ("Russula emetica", "Polyporales", SOURCE_INDEX_ONE_WORD, "dev")
    inputs_json = json.loads(conn.execute("select inputs_json from heldout_sets").fetchone()[0])
    assert {"pool", "titles", "links", "split_dev", "split_test"} <= set(inputs_json)
    assert len(inputs_json["pool"]["sha256"]) == 64


def test_a_development_benchmark_holds_nothing_out(conn, tmp_path):
    save_records(conn, build_records([green(102, "Amanita muscaria")], "t"))
    out = frozen(conn, tmp_path)
    assert (out["held_out"], out["held_out_added"], out["already_reference_records"]) == (
        False, 0, 1)
    assert holdouts.held_out_ids(conn) == set() and not heldout.sealed(conn, NAME)


def test_a_sealed_benchmark_refuses_records_vision_holds_and_holds_every_id_out(conn,
                                                                               tmp_path):
    save_records(conn, build_records([green(102, "Amanita muscaria")], "t"))
    with pytest.raises(ValueError, match="already reference or training records"):
        frozen(conn, tmp_path, holdout=True)
    assert holdouts.held_out_ids(conn) == set()
    conn.execute("delete from records")
    conn.commit()
    assert frozen(conn, tmp_path, holdout=True)["held_out_added"] == 6
    assert holdouts.held_out_ids(conn) == {p[0] for p in POOL} and heldout.sealed(conn, NAME)


def test_freezing_a_new_snapshot_takes_the_new_names_and_logs_it(conn, tmp_path):
    frozen(conn, tmp_path)
    assert frozen(conn, tmp_path)["already_frozen"]
    refreshed = [p if p[0] != "103" else ("103", "Russula rosea", None, -80.0, "dev")
                 for p in POOL]
    files, splits = inputs(tmp_path, refreshed)
    out = heldout.freeze(conn, NAME, splits=splits, **files)
    assert (out["refrozen"], out["names_changed"], out["names_changed_first"]) == (
        True, 1, ["103"])
    assert tuple(conn.execute("select truth_name, truth_status from heldout_records where "
                              "observation_id = '103'").fetchone()) == ("Russula rosea",
                                                                        TRUTH_SPECIES)
    assert conn.execute("select count(*) from heldout_freezes").fetchone()[0] == 2
    files, _ = inputs(tmp_path, POOL[:3])
    with pytest.raises(ValueError, match="already frozen with other ids"):
        heldout.freeze(conn, NAME, **files)


def test_freeze_refuses_ids_that_are_not_the_expected_set(conn, tmp_path):
    files, _ = inputs(tmp_path)
    with pytest.raises(ValueError, match="not the frozen set"):
        heldout.freeze(conn, NAME, expect_sha="174bfabab103fe74", **files)
    ids = [p[0] for p in POOL]
    assert heldout.freeze(conn, NAME, expect_sha=heldout.ids_sha256(ids)[:16],
                          **files)["records"] == 6


def test_the_splits_must_cover_the_pool_exactly(conn, tmp_path):
    files, splits = inputs(tmp_path)
    write_csv(splits["test"], ["observation_id"], [("102",), ("104",), ("106",), ("101",)])
    with pytest.raises(ValueError, match="in both"):
        heldout.freeze(conn, NAME, splits=splits, **files)
    write_csv(splits["test"], ["observation_id"], [("102",)])
    with pytest.raises(ValueError, match="cover the pool exactly"):
        heldout.freeze(conn, NAME, splits=splits, **files)


def test_the_splits_must_match_split_json(conn, tmp_path):
    files, splits = inputs(tmp_path)
    info = {sp: {"records": len(ids), "sha256_ids": heldout.split_sha256(ids)}
            for sp, ids in (("dev", ["101", "103", "105"]), ("test", ["102", "104", "106"]))}
    (tmp_path / "split.json").write_text(json.dumps({**info, "test": {"records": 3,
                                                                      "sha256_ids": "x"}}))
    with pytest.raises(ValueError, match="does not match split.json"):
        heldout.freeze(conn, NAME, splits=splits, split_json=tmp_path / "split.json", **files)
    (tmp_path / "split.json").write_text(json.dumps(info))
    out = heldout.freeze(conn, NAME, splits=splits, split_json=tmp_path / "split.json", **files)
    assert out["splits"] == {"dev": 3, "test": 3}


# --- fetch --------------------------------------------------------------------------------------

def test_benchmark_photos_are_kept_apart_from_reference_photos(conn, tmp_path, monkeypatch):
    frozen(conn, tmp_path)
    got = fetched(conn, tmp_path, monkeypatch)
    assert got["done"] == 6
    # No copy in a reference store, and nothing queued for one.
    assert conn.execute("select count(*) from photo_copies").fetchone()[0] == 0
    assert photos.pending_photos(conn, False, None) == []
    path = conn.execute("select store, path, size from heldout_photos where photo_id = 1011"
                        ).fetchone()
    assert path[0] == str(heldout.bench_dir(conn, NAME)) and path[2] == "large"
    assert (heldout.bench_dir(conn, NAME) / path[1]).is_file()
    # Large photos, and a rerun fetches nothing again.
    assert FakeHTTP.urls and all(u.endswith("/large.jpg") for u in FakeHTTP.urls)
    assert heldout.fetch_photos(conn, NAME, heldout.benchmark_ids(conn, NAME),
                                LocalStore(heldout.bench_dir(conn, NAME)),
                                log=lambda s: None)["queued"] == 0
    assert conn.execute("select status from heldout_observations where observation_id = '106'"
                        ).fetchone()[0] == "missing"


def test_a_withdrawn_photographers_all_rights_reserved_photos_are_not_used(conn, tmp_path):
    from mycomap_vision import permissions
    frozen(conn, tmp_path)
    arr = [inat_obs(101, photos=[(0, 1011, None, STATIC_HOST)])]
    heldout.save_details(conn, NAME, ["101"], arr, "t")
    pending = heldout.pending_photos(conn, NAME, ["101"], "large", "s")
    assert [r["photo_id"] for r in pending] == [1011]
    permissions.ensure_schema(conn)
    conn.execute("insert into photo_permissions (inat_user_id, status) values (7, 'withdrawn')")
    assert heldout.pending_photos(conn, NAME, ["101"], "large", "s") == []


def test_benchmark_downloads_count_against_the_hosts_budget(conn, tmp_path):
    from datetime import datetime, timezone

    from mycomap_vision.ratelimit import ByteBudget
    frozen(conn, tmp_path)
    arr = [inat_obs(101, photos=[(0, 1011, None, STATIC_HOST)])]
    heldout.save_details(conn, NAME, ["101"], arr, "t")
    conn.execute("update heldout_photos set status = 'done', bytes = 700, "
                 "downloaded_at = '2026-10-08T11:30:00+00:00'")
    hour = ByteBudget(1000, 3600)
    now = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)
    assert photos.seed_budgets(conn, {STATIC_HOST: HostPolicy(2, 0, [hour])}, now=now) == {
        STATIC_HOST: 700}


def test_fetch_keeps_each_observations_uuid_where_every_records_inat_details_go(
        conn, tmp_path, monkeypatch):
    from mycomap_vision import evaluate
    frozen(conn, tmp_path)
    fetched(conn, tmp_path, monkeypatch)
    row = conn.execute("select uuid, inat_latitude, inat_longitude, observed_on from "
                       "inat_observations where observation_id = '101'").fetchone()
    assert tuple(row) == ("uuid-101", 39.1, -86.5, "2025-09-01")
    assert conn.execute("select uuid from heldout_observations where observation_id = '101'"
                        ).fetchone()[0] == "uuid-101"
    # Inert without a records row: no reference set loads them.
    assert evaluate.load_records(conn, {1011: 0, 1021: 1}) == []


# --- predict ------------------------------------------------------------------------------------

SCORED_SET_KEYS = {"observation_id", "species", "scores", "kind", "truth", "truth_genus",
                   "truth_family", "latitude", "longitude", "observed_on", "uuid",
                   "group_genus", "group_family", "group_is_species"}


def test_predict_writes_photo_scores_in_the_occurrence_tuning_format(conn, tmp_path,
                                                                    monkeypatch):
    root = reference(conn, tmp_path / "emb")
    frozen(conn, tmp_path)
    fetched(conn, tmp_path, monkeypatch)
    predicted(conn, tmp_path, root)                     # answered already: scores still written
    out, _ = predicted(conn, tmp_path, root, scores_out=tmp_path / "dev.npz")
    assert out["scores_out"]["records"] == 5 and out["nearest"].get("predicted", 0) == 0
    with np.load(tmp_path / "dev.npz", allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    assert set(d) == SCORED_SET_KEYS and str(d["kind"]) == "similarity"
    assert d["species"].tolist() == ["Amanita muscaria", "Russula emetica"]
    assert d["scores"].shape == (5, 2)
    i = d["observation_id"].tolist().index("101")
    assert d["scores"][i, 1] > d["scores"][i, 0]                    # red: Russula
    assert (d["truth"][i], d["truth_genus"][i], d["uuid"][i]) == ("Russula emetica", "Russula",
                                                                  "uuid-101")
    j = d["observation_id"].tolist().index("103")                   # one word: no species
    assert (d["truth"][j], d["truth_genus"][j]) == ("", "Russula")
    assert (d["latitude"][i], d["longitude"][i]) == (39.1, -86.5)     # iNat's public place
    assert d["group_genus"].tolist() == ["Amanita", "Russula"]
    assert d["group_is_species"].tolist() == [True, True]



def test_predictions_keep_the_full_answer_and_the_reference_they_were_made_against(
        conn, tmp_path, monkeypatch):
    root = reference(conn, tmp_path / "emb")
    frozen(conn, tmp_path)
    fetched(conn, tmp_path, monkeypatch)
    out, toy = predicted(conn, tmp_path, root, methods=("nearest", "nearest+prior"))
    assert out["nearest"]["predicted"] == 5 and out["embedding"]["embedded"] == 6
    ref = out["nearest"]["reference_hash"]
    rows = {r[0]: (json.loads(r[1]), r[2], r[3]) for r in conn.execute(
        "select observation_id, result_json, reference_records, code_version from "
        "heldout_predictions where method = 'nearest'")}
    assert rows["101"][0]["species"][0]["name"] == "Russula emetica"
    assert rows["101"][1] == 3 and rows["101"][2]
    full = heldout.full_answer(conn, NAME, "101", "toy", "nearest", ref)
    assert {"ranks", "specimens", "per_photo"} <= set(full)
    run = json.loads(conn.execute("select reference_json from heldout_runs where method = "
                                  "'nearest'").fetchone()[0])
    assert run["records"] == 3 and len(run["records_hash"]) == 12
    # Vectors go to the benchmark's own index, never the manifest's embeddings table.
    assert conn.execute("select count(*) from embeddings where photo_id > 1000").fetchone()[0] == 0
    # A rerun answers nothing again and embeds nothing again.
    again, toy2 = predicted(conn, tmp_path, root)
    assert again["nearest"].get("predicted", 0) == 0 and toy2.calls == 0
    # A relabel is another reference: its answers are kept beside the first run's.
    save_records(conn, build_records([green(1, "Russula emetica"), green(2, "Russula rosea"),
                                      green(3, "Amanita muscaria", family="Amanitaceae")], "t2"))
    after, _ = predicted(conn, tmp_path, root)
    assert after["nearest"]["reference_hash"] != ref and after["nearest"]["predicted"] == 5
    assert conn.execute("select count(distinct reference_hash) from heldout_predictions "
                        "where observation_id = '101' and method = 'nearest'").fetchone()[0] == 2


class LeakyIdentifier:
    """An index that holds benchmark record 101 and photo 1021."""
    backbone, records, embedded, calibration = "toy", 2, 2, None

    def __init__(self, obs=("1", "101"), photos_=(10, 1021)):
        self.col_record = type("C", (), {"obs": np.asarray(obs), "names": np.asarray(["A b"]),
                                         "taxa": np.zeros((len(obs), 3), dtype=np.int32)})()
        self.col_photo = np.asarray(photos_)
        self.index = type("I", (), {"labels": {"species": ["A b"]}})()
        self.rank_counts = {r: {} for r in heldout.RANKS}

    def identify_vectors(self, q, top_k=5, context=None):
        top = [{"name": "A b", "confidence": 1.0, "score": 1.0, "reference_records": 1}]
        return {"ranks": {r: top for r in heldout.RANKS}}


def predict_with(conn, ident):
    return heldout.predict(conn, NAME, "toy", ["nearest"], heldout.benchmark_ids(conn, NAME),
                           Toy, make_identifier=lambda m: ident, log=lambda s: None)


def test_predict_refuses_an_index_that_holds_a_sealed_benchmarks_record(conn, tmp_path,
                                                                       monkeypatch):
    frozen(conn, tmp_path, holdout=True)
    fetched(conn, tmp_path, monkeypatch)
    with pytest.raises(holdouts.HeldOutLeak, match="holds 1 records of the sealed"):
        predict_with(conn, LeakyIdentifier())
    assert conn.execute("select count(*) from heldout_predictions").fetchone()[0] == 0


def test_predict_skips_records_and_photos_already_in_the_reference(conn, tmp_path, monkeypatch):
    frozen(conn, tmp_path)                       # a development benchmark
    fetched(conn, tmp_path, monkeypatch)
    out = predict_with(conn, LeakyIdentifier())
    assert out["nearest"]["already a reference record (skipped)"] == 1
    assert out["nearest"]["photo also a reference photo (skipped)"] == 1
    answered = {r[0] for r in conn.execute("select observation_id from heldout_predictions")}
    assert answered == {"103", "104", "105"}


def test_the_place_given_is_inats_public_one_unless_asked(conn, tmp_path):
    rec = heldout.HeldOutRecord("1", "A b", TRUTH_SPECIES, None, SOURCE_INDEX, 45.0, -120.0,
                                "2025-09-01", 39.1, -86.5, 7, [])
    assert (heldout.context_for(rec, "inat").latitude, heldout.context_for(rec, "org").latitude,
            heldout.context_for(rec, "none")) == (39.1, 45.0, None)


# --- iNat's computer vision -------------------------------------------------------------------

TAXA = {"Russula emetica": (11, "species", [(12, "genus"), (13, "family")]),
        "Amanita muscaria": (21, "species", [(22, "genus"), (23, "family")])}


class FakeInat:
    def __init__(self):
        self.calls = 0
        self.sent = []

    def score_image(self, photo_id, image, lat, lng):
        self.calls += 1
        body = image() if callable(image) else image
        self.sent.append((photo_id, Image.open(io.BytesIO(body)).size, lat, lng))
        t = lambda i, n, r, v, c: {"taxon": {"id": i, "name": n, "rank": r},  # noqa: E731
                                   "vision_score": v, "combined_score": c}
        return {"results": [t(11, "Russula emetica", "species", 80, 60),
                            t(21, "Amanita muscaria", "species", 10, 70),
                            t(12, "Russula", "genus", 85, 50),
                            t(13, "Russulaceae", "family", 90, 90)]}

    def get(self, path, params=None):
        self.calls += 1
        if path == "/taxa":
            hit = TAXA.get(params["q"])
            return {"results": [{"id": hit[0], "name": params["q"]}] if hit and
                    hit[1] == params["rank"] else []}
        tid = int(path.rsplit("/", 1)[1])
        name, (i, rank, anc) = next((n, v) for n, v in TAXA.items() if v[0] == tid)
        return {"results": [{"id": i, "name": name, "rank": rank, "is_active": True,
                             "ancestors": [{"id": a, "rank": r} for a, r in anc]}]}


def test_inat_answers_best_per_rank_with_the_answer_looked_up_on_inat(conn, tmp_path,
                                                                     monkeypatch):
    reference(conn, tmp_path / "emb")
    frozen(conn, tmp_path)
    fetched(conn, tmp_path, monkeypatch)
    client = FakeInat()
    out = heldout.inat_cv(conn, NAME, ["101", "102", "104"], client, log=lambda s: None)
    assert out["scored"] == 2 and out["no answer key (not sent)"] == 1
    # iNat's public place, and photos cut down to 500 px before they are sent.
    assert {(s[2], s[3]) for s in client.sent} == {(39.1, -86.5)}
    assert all(max(s[1]) == 500 for s in client.sent)
    res = {(r[0], r[1]): json.loads(r[2]) for r in conn.execute(
        "select observation_id, method, result_json from heldout_predictions")}
    assert res[("101", "vision-max")]["species"][0]["taxon_id"] == 11
    assert res[("101", "combined-max")]["species"][0]["taxon_id"] == 21
    assert res[("102", "vision-max")]["truth_taxa"] == {"species": 21, "genus": 22, "family": 23,
                                                        "species_known": True}


# --- the report ---------------------------------------------------------------------------------

def test_wilson_intervals_and_mcnemar():
    assert heldout_report.wilson(0, 0) is None
    lo, hi = heldout_report.wilson(8, 10)
    assert (round(lo, 3), round(hi, 3)) == (0.490, 0.943)
    assert heldout_report.mcnemar(0, 10) == pytest.approx(2 / 1024)
    assert heldout_report.mcnemar(5, 5) == 1.0
    assert heldout_report.mcnemar(600, 500) == pytest.approx(0.0029, abs=2e-4)


def test_calibration_bins_compare_stated_confidence_with_accuracy():
    cal = heldout_report.calibration([(0.95, True), (0.92, False), (0.15, False), (1.0, True)])
    assert [b["confidence"] for b in cal["bins"]] == ["0.1-0.2", "0.9-1.0"]
    assert cal["bins"][1]["n"] == 3 and cal["bins"][1]["accuracy"] == pytest.approx(0.6667, 1e-3)


def test_inat_is_judged_by_taxon_id_where_the_answer_was_found_on_inat(conn):
    truth = heldout.Truth("Russula emetica", "Russula", "Russulaceae", "Russula emetica", False,
                          None)
    labeller = heldout.Labeller(conn)
    by_id = {"species": [{"name": "Russula emetica s.l.", "taxon_id": 11, "confidence": .9}],
             "truth_taxa": {"species": 11, "genus": None}}
    judged = heldout_report.judge(by_id, truth, labeller, by_taxon_id=True)
    assert judged["species"][0] is True
    by_name = {"genus": [{"name": "Russula", "taxon_id": 99}], "truth_taxa": {"genus": None}}
    assert heldout_report.judge(by_name, truth, labeller, by_taxon_id=True)["genus"][0] is True


def test_likely_sets_are_scored_when_the_answers_carry_them(conn):
    labeller = heldout.Labeller(conn)
    truth = heldout.Truth("Russula emetica", "Russula", "Russulaceae", "Russula emetica", False,
                          None)
    answers = {"1": {"likely": {"species": {"names": [{"name": "Russula emetica"},
                                                      {"name": "Russula rosea"}]}}},
               "2": {"likely": {"species": {"names": [{"name": "Russula rosea"}]}}},
               "3": {"ranks": {}}}
    got = heldout_report.likely_metrics(answers, {k: truth for k in answers}, labeller)
    assert got == {"species": {"n": 2, "coverage": 0.5, "mean_size": 1.5}}
    assert heldout_report.likely_metrics({"3": {}}, {"3": truth}, labeller) is None


def reported(conn, tmp_path, split="dev", **kw):
    return heldout_report.report(conn, NAME, split=split, out_dir=tmp_path / "reports",
                                 log=kw.pop("log", lambda s: None), **kw)


def scored_world(conn, tmp_path, monkeypatch, **freeze):
    root = reference(conn, tmp_path / "emb")
    frozen(conn, tmp_path, **freeze)
    fetched(conn, tmp_path, monkeypatch)
    predicted(conn, tmp_path, root)
    heldout.inat_cv(conn, NAME, heldout.benchmark_ids(conn, NAME), FakeInat(), log=lambda s: None)
    return root


def test_a_report_on_dev_never_scores_a_test_record(conn, tmp_path, monkeypatch):
    scored_world(conn, tmp_path, monkeypatch)
    out = reported(conn, tmp_path)
    assert out["split"] == "dev" and out["records"] == 3
    rows = list(csv.DictReader(io.StringIO(read(out["files"]["csv"]))))
    assert {r["observation_id"] for r in rows} <= {"101", "103", "105"}
    with pytest.raises(ValueError, match="name the split"):
        reported(conn, tmp_path, split=None)


def test_every_report_on_a_sealed_test_split_is_recorded_and_says_so(conn, tmp_path,
                                                                    monkeypatch):
    scored_world(conn, tmp_path, monkeypatch, holdout=True)
    said = []
    first = reported(conn, tmp_path, split="test", log=said.append)
    second = reported(conn, tmp_path, split="test", stamp="later", log=said.append)
    assert (first["test_look"], second["test_look"]) == (1, 2)
    assert "SEALED TEST SPLIT" in said[0] and "look number 2" in said[1]
    looks = conn.execute("select benchmark, models_json, records from heldout_test_looks"
                         ).fetchall()
    assert len(looks) == 2 and "toy/nearest" in looks[0][1]
    reported(conn, tmp_path, split="dev", stamp="dev")
    assert conn.execute("select count(*) from heldout_test_looks").fetchone()[0] == 2


def test_a_development_benchmarks_test_split_is_not_sealed(conn, tmp_path, monkeypatch):
    scored_world(conn, tmp_path, monkeypatch)
    said = []
    out = reported(conn, tmp_path, split="test", log=said.append)
    assert not out["sealed"] and "test_look" not in out and said == []
    assert conn.execute("select count(*) from heldout_test_looks").fetchone()[0] == 0


def test_the_label_audit_flags_stale_names_and_index_names_in_another_genus(conn, tmp_path,
                                                                            monkeypatch):
    scored_world(conn, tmp_path, monkeypatch)
    dev = reported(conn, tmp_path)
    assert dev["label_audit"]["name_may_be_stale_needs_com_refresh"] == [
        {"observation_id": "103", "name": "Russula"}]
    assert dev["models"]["toy/nearest"]["records"] == 3       # 103 at genus and family only
    assert "species" not in heldout_report.judge(
        {"species": [], "genus": []}, heldout.Labeller(conn).truth("Russula"),
        heldout.Labeller(conn), by_taxon_id=False)
    test = reported(conn, tmp_path, split="test", stamp="t")
    assert test["label_audit"]["index_name_in_another_genus"] == [
        {"observation_id": "106", "index_name": "Tubariua hiemalis", "title": "Tubaria hiemalis"}]
    assert test["label_audit"]["without_an_answer"] == 1
    assert test["models"]["toy/nearest"]["species"]["top1"] == {
        "n": 1, "right": 0, "rate": 0.0, "ci95": heldout_report.wilson(0, 1)}
    assert test["models"]["external:inat-cv/combined-max"]["species"]["top1"]["right"] == 1
    pair = next(p for p in test["paired"] if p["b"] == "toy/nearest")
    assert pair["species"]["n"] == 1


def test_label_hygiene_normalises_writing_and_lists_what_it_cannot_place(conn):
    save_records(conn, build_records([green(1, "Hygrocybe glutinipes var. rubra"),
                                      green(2, "Tubaria hiemalis"),
                                      green(3, "Russula emetica"),
                                      green(4, "Candolleomyces sp. 'FL03'")], "t"))
    labeller = heldout.Labeller(conn)
    t = labeller.truth("Hygrocybe glutinipes rubra")
    assert (t.label, t.species, t.normalised) == ("Hygrocybe glutinipes var. rubra",
                                                  "Hygrocybe glutinipes var. rubra", True)
    recs = [heldout.HeldOutRecord(str(i), n, TRUTH_SPECIES, None, SOURCE_INDEX, None, None, None,
                                  None, None, None, [])
            for i, n in enumerate(["Hygrocybe glutinipes rubra", "Tubariua hiemalis",
                                   "Russula emeticaa", "Candolleomyces sp. 'FL05'",
                                   "Russula emetica pallida"])]
    truths = {r.observation_id: labeller.truth(r.truth_name) for r in recs}
    h = heldout_report.label_hygiene(recs, truths, labeller)
    assert h["normalised"][0]["label"] == "Hygrocybe glutinipes var. rubra"
    assert h["unknown_genus"][0]["genus"] == "Tubariua"
    assert h["unknown_genus"][0]["closest_known_genus"] == "Tubaria"
    # A temporary code close to another code is a different taxon, not a typo.
    assert h["close_to_a_known_name"] == [{"name": "Russula emeticaa", "records": 1,
                                           "closest_known_name": "Russula emetica"}]
    assert h["rank_word_missing_no_known_form"] == [{"name": "Russula emetica pallida",
                                                     "records": 1}]
    # Reported, never fixed: the typo stays the answer.
    assert truths["2"].label == "Russula emeticaa"


def test_breakdowns_place_a_record_by_reference_depth_observer_day_and_side():
    ref = {"rank_counts": {"species": {"A b": 7}, "genus": {"A": 120}},
           "observer_day_set": {"7|2025-09-01"}}
    t = heldout.Truth("A b", "A", "F", "A b", False, None)
    rec = heldout.HeldOutRecord("1", "A b", TRUTH_SPECIES, None, SOURCE_INDEX, 45.0, -120.0,
                                "2025-09-01", 39.1, -86.5, 7, [(1, "s", "p"), (2, "s", "p")])
    f = heldout_report.features(rec, t, ref)
    assert f == {"species reference records": "5-19", "genus reference records": "100+",
                 "true species in reference": "seen", "photos": "2",
                 "east or west of -100": "west", "same observer and day in reference": "yes",
                 "name kind": "formal"}
    other = heldout.HeldOutRecord("2", "Inocybe sp. 'X01'", TRUTH_SPECIES, None, SOURCE_INDEX,
                                  None, None, "2025-09-02", None, -80.0, 7, [(1, "s", "p")])
    t2 = heldout.Truth("Inocybe sp. 'X01'", "Inocybe", "F", "Inocybe sp. 'X01'", False, None)
    f2 = heldout_report.features(other, t2, ref)
    assert (f2["species reference records"], f2["true species in reference"],
            f2["east or west of -100"], f2["same observer and day in reference"],
            f2["name kind"]) == ("0", "unseen", "east", "no", "provisional")


def test_each_model_is_scored_on_one_reference_and_an_older_one_can_be_named(conn, tmp_path,
                                                                             monkeypatch):
    root = scored_world(conn, tmp_path, monkeypatch)
    before = conn.execute("select reference_hash from heldout_runs").fetchone()[0]
    save_records(conn, build_records([green(1, "Russula rosea"), green(2, "Russula rosea"),
                                      green(3, "Amanita muscaria", family="Amanitaceae")], "t2"))
    after, _ = predicted(conn, tmp_path, root)
    now = reported(conn, tmp_path, stamp="after")
    assert now["references"]["toy/nearest"] == after["nearest"]["reference_hash"]
    assert now["other_reference_answers_left_out"] == {"toy/nearest": 3}
    assert now["models"]["toy/nearest"]["species"]["top1"]["right"] == 0
    old = reported(conn, tmp_path, stamp="before", reference_hash=before)
    assert old["references"]["toy/nearest"] == before
    assert old["models"]["toy/nearest"]["species"]["top1"]["right"] == 1
    with pytest.raises(ValueError, match="no run"):
        reported(conn, tmp_path, reference_hash="nope")


def numbers(value):
    if isinstance(value, dict):
        for v in value.values():
            yield from numbers(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from numbers(v)
    elif isinstance(value, (int, float)):
        yield value


def test_the_report_carries_no_coordinates(conn, tmp_path, monkeypatch):
    scored_world(conn, tmp_path, monkeypatch)
    coords = {39.1, -86.5, -80.0, -120.0, 40.0}
    out = reported(conn, tmp_path)
    report = json.loads(read(out["files"]["json"]))
    assert not coords & set(numbers(report))
    rows = list(csv.reader(io.StringIO(read(out["files"]["csv"]))))
    assert not {c for c in rows[0] if re.search("lat|lng|lon", c)}
    assert not {str(c) for c in coords} & {v for row in rows for v in row}
