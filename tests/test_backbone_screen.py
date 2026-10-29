"""The frozen-backbone screen on a release's seeded subset (backbone_screen.py, paper E5):
the references are a sample of species taken whole (not topped up for the queries), the
queries come in two halves by former-benchmark membership, the subset is drawn once, and
every backbone is scored on exactly the same records and photos."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
from release_fixture import BASE, release_world  # noqa: E402
from test_models_and_scoreboard import Const  # noqa: E402

from mycomap_vision import backbone_screen as bs  # noqa: E402
from mycomap_vision import evaluate  # noqa: E402
from mycomap_vision.replications.fungitastic import dataset  # noqa: E402


def world_records(n_species=60, seed=1):
    """Train records of species with 1..120 records each; val records of every species,
    every third one from the former benchmark."""
    import random
    rng = random.Random(seed)
    rows, k = [], 0
    for s in range(n_species):
        depth = rng.choice([1, 2, 3, 6, 10, 25, 60, 120])
        for _ in range(depth):
            k += 1
            rows.append({"record_key": str(k), "split": "train", "species": f"G{s % 7} s{s}",
                         "benchmark": None})
        for j in range(3):
            k += 1
            rows.append({"record_key": str(k), "split": "val", "species": f"G{s % 7} s{s}",
                         "benchmark": "heldout-2026-10-08" if j == 0 else None})
    return rows


def test_references_are_whole_species_with_one_cap_for_all_not_topped_up_for_queries():
    rows = world_records()
    doc = bs.draw(rows, reference_records=300, cap=50, queries=20)
    train = {}
    for r in rows:
        if r["split"] == "train":
            train.setdefault(r["species"], []).append(r["record_key"])
    refs = set(doc["references"])
    for s in doc["species"]:
        got = [k for k in train[s] if k in refs]
        assert len(got) == min(len(train[s]), 50)          # whole species, the same cap
    for s in set(train) - set(doc["species"]):
        assert not refs & set(train[s])                    # unsampled species: none at all
    assert len(refs) >= 300 and len(refs) - 50 < 300       # stops once the target is met
    sampled = set(doc["species"])
    by_key = {r["record_key"]: r for r in rows}
    for half in doc["queries"].values():
        assert all(by_key[k]["species"] in sampled for k in half["records"])


def test_queries_come_in_two_halves_by_former_benchmark_and_by_depth_band():
    rows = world_records()
    doc = bs.draw(rows, reference_records=400, cap=50, queries=30)
    by_key = {r["record_key"]: r for r in rows}
    excl = doc["queries"]["val-excl-former-heldout"]
    held = doc["queries"]["val-former-heldout"]
    assert all(not by_key[k]["benchmark"] for k in excl["records"])
    assert all(by_key[k]["benchmark"] for k in held["records"])
    assert all(by_key[k]["split"] == "val" for k in excl["records"] + held["records"])
    for half in (excl, held):
        assert len(half["records"]) == min(30, sum(half["pool"].values()))
        assert sum(half["by_band"].values()) == len(half["records"])
        assert set(half["by_band"]) <= {"1-4", "5-19", "20-99"}     # capped at 50


def test_proportional_allocation_by_largest_remainder():
    assert bs.allocate({"a": 10, "b": 30, "c": 60}, 10) == {"a": 1, "b": 3, "c": 6}
    assert bs.allocate({"a": 1, "b": 1, "c": 1}, 2) == {"a": 1, "b": 1, "c": 0}
    assert bs.allocate({"a": 3, "b": 2}, 10) == {"a": 3, "b": 2}     # fewer than asked


def test_the_draw_is_seeded_and_the_species_list_hashed():
    rows = world_records()
    a = bs.draw(rows, reference_records=300, queries=20)
    assert bs.draw(rows, reference_records=300, queries=20) == a
    b = bs.draw(rows, seed=7, reference_records=300, queries=20)
    assert b["species_sha256"] != a["species_sha256"]
    import hashlib
    assert a["species_sha256"] == hashlib.sha256("\n".join(a["species"]).encode()).hexdigest()


# --- on a real release (release_fixture) -------------------------------------------------------

@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    return release_world(tmp_path, monkeypatch, test_release=False)


def test_the_subset_is_drawn_once_embedded_once_and_scored_on_the_same_photos(world, tmp_path):
    conn, rel = world["conn"], dataset.load(BASE, True)
    out = bs.results_dir(rel.id)
    doc = bs.make_subset(rel, out, reference_records=100, queries=10)
    assert doc["dataset_release"]["release_hash"] == rel.release_hash
    assert set(doc["references"]) == {"1", "2", "3", "4", "mo:9"}
    assert doc["queries"]["val-excl-former-heldout"]["records"] == ["5", "6"]
    assert doc["queries"]["val-former-heldout"]["records"] == []
    assert len(doc["photos"]) == 7
    assert bs.make_subset(rel, out, reference_records=100, queries=10) == doc   # same draw
    with pytest.raises(ValueError, match="drawn once"):
        bs.make_subset(rel, out, reference_records=50, queries=10)

    def loader(spec):
        return Const(spec, twist=0.3 if spec == "m2" else 0.0)
    first = bs.embed(conn, doc, ["m1", "m2"], loader=loader, log=lambda s: None)
    assert first == {"m1": {"embedded": 7, "skipped": 0, "seconds": pytest.approx(
        first["m1"]["seconds"])}, "m2": first["m2"]}
    again = bs.embed(conn, doc, ["m1"], loader=loader, log=lambda s: None)
    assert again["m1"]["embedded"] == 0                       # nothing embedded twice
    res = bs.score(conn, rel, doc, ["m1", "m2"], ["nearest", "nearest+mean"],
                   log=lambda s: None)
    assert res["species_sha256"] == doc["species_sha256"] and res["photos_scored"] == 7
    cid = res["halves"]["val-excl-former-heldout"]["comparison_id"]
    rows = evaluate.scoreboard(conn, cid)
    assert {(r["backbone"], r["method"]) for r in rows} == {
        (b, m) for b in ("m1", "m2") for m in ("nearest", "nearest+mean")}
    report = json.loads(conn.execute("select report_json from eval_runs where comparison_id = ? "
                                     "limit 1", (cid,)).fetchone()[0])
    assert report["backbone_screen"]["half"] == "val-excl-former-heldout"
    assert report["dataset_release"]["release_hash"] == rel.release_hash
    assert "standard_equivalence" in report["all_photos"]        # Steve's tables can be made


def test_a_subset_photo_with_no_local_copy_is_refused_not_fetched(world):
    conn, rel = world["conn"], dataset.load(BASE, True)
    doc = bs.make_subset(rel, bs.results_dir(rel.id), reference_records=100, queries=10)
    conn.execute("update photo_copies set store = 's3://bucket/' where photo_id = ?",
                 (doc["photos"][0],))
    with pytest.raises(ValueError, match="no large copy on this machine"):
        bs.photos_to_embed(conn, "m1", doc["photos"])


def test_a_subset_from_another_release_is_refused(world):
    conn, rel = world["conn"], dataset.load(BASE, True)
    doc = bs.make_subset(rel, bs.results_dir(rel.id), reference_records=100, queries=10)
    doc["dataset_release"]["release_hash"] = "0" * 64
    with pytest.raises(ValueError, match="another release"):
        bs.score(conn, rel, doc, ["m1"], ["nearest"], log=lambda s: None)


def test_an_erratum_given_to_the_release_leaves_the_subset_and_is_recorded(world, tmp_path):
    from mycomap_vision import errata
    rel = dataset.load(BASE, True)
    path = tmp_path / "e.tsv"
    path.write_text("kind\tkey\treason\tnote\nrecord\t1\tmo_twin_as_inat\t\n", encoding="utf-8")
    rel.use_errata(errata.read(path))
    doc = bs.make_subset(rel, bs.results_dir(rel.id), reference_records=100, queries=10)
    assert "1" not in doc["references"] and set(doc["references"]) == {"2", "3", "4", "mo:9"}
    assert doc["dataset_release"]["errata"]["sha256"] == errata.read(path).sha256
    plain = dataset.load(BASE, True)                       # the same draw without it: refused
    with pytest.raises(ValueError, match="drawn once"):
        bs.make_subset(plain, bs.results_dir(rel.id), reference_records=100, queries=10)


def test_a_reused_vector_not_made_from_a_large_copy_is_embedded_again_and_never_read(world):
    """The v1 screen reused vectors of the 28 Sep sample made from 500 px medium copies
    beside the 1024 px large ones it embedded (audit 2026-10-29): the screen now reuses only
    vectors recorded as made from a large copy, and scores the one the index points at."""
    from mycomap_vision import config
    from mycomap_vision.embed import embed_photos, load_embeddings
    from mycomap_vision.storage import open_store
    conn, rel = world["conn"], dataset.load(BASE, True)
    doc = bs.make_subset(rel, bs.results_dir(rel.id), reference_records=100, queries=10)
    # earlier work: two subset photos in one shard under the same name, made with other
    # vectors (twist) to tell them apart; one recorded as from a large copy, one of unknown
    # size (made before sizes were recorded, like the 28 Sep medium vectors)
    first = bs.photos_to_embed(conn, "m1", doc["photos"])
    where, items = next(iter(first.items()))
    embed_photos(conn, open_store(where, config.DATA_DIR), Const("m1", twist=1.0), items[:2],
                 config.DATA_DIR / "embeddings" / "m1", log=lambda s: None)
    kept, stale = items[0][0], items[1][0]
    conn.execute("insert into embedding_sources values ('m1', ?, 'large')", (kept,))
    assert sum(map(len, bs.photos_to_embed(conn, "m1", doc["photos"]).values())) == 5
    assert sum(map(len, bs.photos_to_embed(conn, "m1", doc["photos"],
                                           source_size="large").values())) == 6
    done = bs.embed(conn, doc, ["m1"], loader=lambda s: Const(s), log=lambda s: None)
    assert done["m1"]["embedded"] == 6                     # the unknown one again too
    assert bs.embed(conn, doc, ["m1"], loader=lambda s: Const(s),
                    log=lambda s: None)["m1"]["embedded"] == 0
    ids, vecs = load_embeddings(conn, "m1")
    assert ids.tolist().count(stale) == 2                  # its old vector is still in shard 0
    keep = bs.indexed_rows(conn, "m1", ids, set(doc["photos"]))
    assert sorted(ids[keep].tolist()) == sorted(doc["photos"])      # each photo once
    fresh = bs.embed(conn, doc, ["m2"], loader=lambda s: Const(s), log=lambda s: None)
    assert fresh["m2"]["embedded"] == 7
    i2, v2 = load_embeddings(conn, "m2")
    want = dict(zip(i2.tolist(), v2.tolist()))
    got = dict(zip(ids[keep].tolist(), vecs[keep].tolist()))
    assert got[stale] == want[stale]                       # the new vector, never the old
    assert got[kept] != want[kept]                         # a recorded large one is reused
    res = bs.score(conn, rel, doc, ["m1"], ["nearest"], log=lambda s: None)
    assert res["vectors_by_source_size"] == {"m1": {"large": 7}}


def test_the_screen_reports_vectors_of_unknown_source_size(world):
    conn, rel = world["conn"], dataset.load(BASE, True)
    doc = bs.make_subset(rel, bs.results_dir(rel.id), reference_records=100, queries=10)
    from mycomap_vision import config
    from mycomap_vision.embed import embed_photos
    from mycomap_vision.storage import open_store
    for where, items in bs.photos_to_embed(conn, "m1", doc["photos"]).items():
        embed_photos(conn, open_store(where, config.DATA_DIR), Const("m1"), items,
                     config.DATA_DIR / "embeddings" / "m1", log=lambda s: None)
    said = []
    res = bs.score(conn, rel, doc, ["m1"], ["nearest"], log=said.append)
    assert res["vectors_by_source_size"] == {"m1": {"unknown": 7}}
    assert any("WARNING m1" in s for s in said)
