"""The fungus gate's tuning rule (fungusgate_tune.py): declared before looking, DEV only."""

import numpy as np
import pytest

from mycomap_vision import fungusgate_tune as tune


def recs(*maxes_and_mins):
    return [np.array(v, dtype=float) for v in maxes_and_mins]


def test_not_a_fungus_is_set_where_no_dev_record_loses_its_answer():
    # One record's best photo is 0.32: "not a fungus" must stay at or below it.
    c = tune.choose_thresholds(recs([0.9, 0.2], [0.32], [0.8]), grid=(0.1, 0.2, 0.3, 0.4, 0.5))
    assert c.not_fungus == 0.3 and c.no_answer == 0
    assert c.photo_left_out == 1                    # the 0.2 photo of the first record


def test_not_sure_stays_within_the_share_of_dev_records_shown_without_a_chance():
    scores = recs(*([[0.9]] * 198 + [[0.42], [0.47]]))
    c = tune.choose_thresholds(scores, grid=(0.1, 0.2, 0.3, 0.4, 0.45, 0.5), max_unsure_share=0.005)
    assert c.not_fungus == 0.4                      # 0.42 is the lowest best photo
    assert c.unsure == 0.45 and c.unsure_records == 1   # 0.5 would put 2 of 200 (1%) unsure


def test_a_score_that_loses_a_dev_record_at_every_grid_value_is_not_chosen():
    c = tune.choose_thresholds(recs([0.01], [0.9]), grid=(0.05, 0.1))
    assert c.not_fungus is None and c.unsure is None


@pytest.mark.parametrize("ancestors,expected", [
    ("[48460, 47170, 48250]", "fungi"), ("[48460, 1, 2, 355675, 3]", "animal"),
    ("48460/47126/211194", "plant"), ("[48460, 47686]", "other"), (None, "other"),
])
def test_kingdom_comes_from_the_ancestor_ids(ancestors, expected):
    assert tune.kingdom(ancestors) == expected


def test_the_not_a_fungus_manifest_keeps_licence_and_a_checksum_for_every_photo(tmp_path):
    (tmp_path / "a.jpg").write_bytes(b"photo-a")
    out = tmp_path / "m.csv"
    tune.photo_manifest([{"file": "a.jpg", "license": "cc0", "owner": "x", "url": "u"}],
                        tmp_path, out)
    row = out.read_text(encoding="utf-8").splitlines()[1].split(",")
    assert "cc0" in row and row[-1] == tune.sha256_file(tmp_path / "a.jpg")


# --- re-tuned on a dataset release (paper E10) ---------------------------------------------

def test_a_lost_share_above_zero_allows_that_share_of_dev_records_to_lose_their_answer():
    scores = [np.array([0.5])] * 199 + [np.array([0.12])]
    assert tune.choose_thresholds(scores).not_fungus == 0.1          # none may lose it
    assert tune.choose_thresholds(scores, max_lost_share=0.005).not_fungus == 0.5


@pytest.fixture
def world(tmp_path, monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
    from release_fixture import BASE, release_world

    from mycomap_vision.replications.fungitastic import dataset
    w = release_world(tmp_path, monkeypatch, test_release=False)
    return {**w, "rel": dataset.load(BASE, True)}


def test_the_release_probe_takes_train_positives_and_only_the_given_negatives(world, monkeypatch):
    rel = world["rel"]
    ids = np.array([int(p) for (p,) in rel.db.execute("select manifest_photo_id from photos")])
    vecs = np.eye(len(ids), 4, dtype=np.float32) + 0.1
    neg = np.ones((3, 4), np.float32)
    monkeypatch.setattr(tune, "POSITIVE_RECORDS", 100)
    X, y, prov = tune.release_probe_set(rel, ids, vecs, neg)
    train = {r["record_key"] for r in rel.records("train")}
    assert prov["positive_records"] == len(train) == 5               # never val
    assert int(y.sum()) == prov["positive_photos"] and int((1 - y).sum()) == 3
    assert prov["negative_records"] == 0                             # no record is a negative


def test_the_release_probe_leaves_an_erratum_out(world, monkeypatch, tmp_path):
    from mycomap_vision import errata
    rel = world["rel"]
    path = tmp_path / "e.tsv"
    path.write_text("kind\tkey\treason\tnote\nrecord\t1\tmo_twin_as_inat\t\n", encoding="utf-8")
    rel.use_errata(errata.read(path))
    ids = np.array([int(p) for (p,) in rel.db.execute("select manifest_photo_id from photos")])
    monkeypatch.setattr(tune, "POSITIVE_RECORDS", 100)
    _X, _y, prov = tune.release_probe_set(rel, ids, np.ones((len(ids), 4), np.float32),
                                          np.ones((1, 4), np.float32))
    assert prov["positive_records"] == 4


def test_dev_photos_must_be_val_and_the_former_held_out_does_not_decide(world):
    rel = world["rel"]
    val = [str(p) for (p,) in rel.db.execute(
        "select p.manifest_photo_id from photos p join records r using (record_key) "
        "where r.split = 'val'")]
    recs, decide = tune.release_dev_records(rel, val)
    assert set(recs) == {"5", "6"} and decide == {"5", "6"}          # no benchmark records here
    train = str(rel.db.execute("select p.manifest_photo_id from photos p join records r "
                               "using (record_key) where r.split = 'train'").fetchone()[0])
    with pytest.raises(ValueError, match="only val tunes the gate"):
        tune.release_dev_records(rel, val + [train])


def test_only_val_records_outside_the_former_held_out_decide():
    class Db:
        def execute(self, *_):
            return [(11, "a", "val", None), (12, "a", "val", None),
                    (13, "b", "val", "heldout-2026-10-08"), (14, "c", "train", None)]

    class Rel:
        id, db = "r", Db()
    recs, decide = tune.release_dev_records(Rel(), ["11", "12", "13"])
    assert recs == ["a", "a", "b"] and decide == {"a"}


def test_the_cli_passes_the_declared_lost_share_to_the_release_tune(monkeypatch, tmp_path):
    from mycomap_vision import cli, embed
    seen = {}

    def fake_tune(*args, max_lost_share=0.0, **kw):
        seen["max_lost_share"] = max_lost_share
        return {"chosen": {}}
    monkeypatch.setattr(tune, "tune_release", fake_tune)
    monkeypatch.setattr(cli, "_dataset_release", lambda args: object())
    monkeypatch.setattr(embed, "load_embeddings", lambda conn, name: (None, None))
    files = ["--dev", "d.npz", "--ood", "o.npz", "--ood-manifest", "m.csv",
             "--negatives", "n.npz", "--out", str(tmp_path)]
    cli.main(["fungus-gate", "tune", "--release", "v1", *files, "--max-lost-share", "0.001"])
    assert seen["max_lost_share"] == 0.001
    cli.main(["fungus-gate", "tune", "--release", "v1", *files])
    assert seen["max_lost_share"] == 0.0                  # the default: none may lose it
    with pytest.raises(SystemExit, match="max-lost-share"):
        cli.main(["fungus-gate", "tune", "--release", "v1", *files, "--max-lost-share", "1.5"])
