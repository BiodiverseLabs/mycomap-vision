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


def test_errata_records_are_left_out_of_references_and_queries(tmp_path):
    rows = world_records()
    first = bs.draw(rows, reference_records=300, queries=20)
    bad = set(first["references"][:5]) | set(first["queries"]["val-excl-former-heldout"]
                                             ["records"][:3])
    path = tmp_path / "v1-errata.tsv"
    path.write_text("record_key\treason\n" + "".join(f"{k}\tnot this record\n" for k in bad),
                    encoding="utf-8")
    keys, sha = bs.read_errata(path)
    assert keys == bad and len(sha) == 64
    doc = bs.draw(rows, reference_records=300, queries=20, errata=keys)
    used = set(doc["references"]) | {k for h in doc["queries"].values() for k in h["records"]}
    assert not used & bad and doc["errata_records_left_out"] == len(bad)


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
