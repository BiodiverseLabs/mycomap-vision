"""A benchmark's held-out records never reach training, the reference index, a release,
a trainer or the nightly update (holdouts.py), whichever path tries to bring them in."""

import json

import numpy as np
import pytest

from conftest import inat_obs
from mycomap_vision import aws, cli, config, evaluate, holdouts, nightly, trainer
from mycomap_vision.embed import SCHEMA as EMBED_SCHEMA
from mycomap_vision.embed import photos_to_embed
from mycomap_vision.inat import save_batch
from mycomap_vision.photos import pending_photos
from mycomap_vision.records import build_records, save_records

OPEN = "inaturalist-open-data.s3.amazonaws.com"


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)


def green(oid, name="Russula emetica", vdate="1/1/2026"):
    return {"source": "iNaturalist", "observation_id": str(oid), "scientific_name": name, "genus": name.split()[0],
            "family": "Russulaceae", "continent": "North America",
            "validation_status_1": "yes", "validation_date_1": vdate}


def stored_by_old_code(conn, ids):
    """Records stored before benchmark_holdouts existed (or by an older checkout), with
    iNat details and one photo each: what the read paths must still leave out."""
    save_records(conn, build_records([green(i) for i in ids], "t"))
    save_batch(conn, [str(i) for i in ids],
               [inat_obs(i, photos=[(0, i * 10, "cc0", OPEN)]) for i in ids], "t")


def with_embeddings(conn, root, backbone, ids):
    conn.executescript(EMBED_SCHEMA)
    out = root / backbone
    out.mkdir(parents=True, exist_ok=True)
    pids = np.asarray([i * 10 for i in ids], dtype=np.int64)
    rng = np.random.default_rng(0)
    v = rng.standard_normal((len(pids), 4)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    np.save(out / "shard-00000.npy", v.astype(np.float16))
    np.save(out / "shard-00000.ids.npy", pids)
    conn.executemany("insert into embeddings values (?, ?, 0, ?, 'now')",
                     [(backbone, int(p), r) for r, p in enumerate(pids.tolist())])
    conn.commit()


def test_an_export_never_stores_a_held_out_record(conn):
    holdouts.add(conn, "bench", ["2"])
    changes = save_records(conn, build_records([green(1), green(2)], "t1"))
    assert [r[0] for r in conn.execute("select observation_id from records")] == ["1"]
    assert changes == {"new": 1, "removed": 0, "renamed": 0, "held_out": 1}


def test_an_export_removes_a_held_out_record_older_code_stored(conn):
    stored_by_old_code(conn, [1, 2])
    holdouts.add(conn, "bench", ["2"])
    save_records(conn, build_records([green(1), green(2)], "t2"))
    assert [r[0] for r in conn.execute("select observation_id from records")] == ["1"]


def test_the_nightly_update_never_counts_a_held_out_record_as_new(conn):
    save_records(conn, build_records([green(1)], "t1"))
    holdouts.add(conn, "bench", ["2"])
    plan = nightly.changes(conn, build_records([green(1), green(2), green(3)], "t2"))
    assert (plan["new"], plan["examples"]["new"], plan["held_out"]) == (1, ["3"], 1)


def test_reference_sets_never_load_a_held_out_record(conn):
    stored_by_old_code(conn, [1, 2, 3])
    holdouts.add(conn, "bench", ["2"])
    recs = evaluate.load_records(conn, {10: 0, 20: 1, 30: 2})
    assert sorted(r.observation_id for r in recs) == ["1", "3"]


def test_no_reference_photo_is_downloaded_for_a_held_out_record(conn):
    stored_by_old_code(conn, [1, 2])
    holdouts.add(conn, "bench", ["2"])
    assert [r["photo_id"] for r in pending_photos(conn, True, None)] == [10]


def test_no_reference_photo_is_embedded_for_a_held_out_record(conn):
    stored_by_old_code(conn, [1, 2])
    conn.executemany("insert into photo_copies values (?, 'store', 'large', ?, 1, 'h', 'now')",
                     [(10, "p10"), (20, "p20")])
    holdouts.add(conn, "bench", ["2"])
    assert photos_to_embed(conn, "m", "large", "store", "store") == [(10, "p10")]


def test_fine_tuning_never_trains_on_a_held_out_record(conn, tmp_path):
    from mycomap_vision.finetune import build_trainset
    stored_by_old_code(conn, [1, 2, 3])
    conn.executemany("insert into photo_copies values (?, 's', 'large', ?, 1, 'h', 'now')",
                     [(i * 10, f"p{i}") for i in (1, 2, 3)])
    # One record validated much later is the comparison's test set; the rest train.
    save_records(conn, build_records([green(1), green(2), green(3), green(4, vdate="9/1/2026")],
                                     "t2"))
    save_batch(conn, ["4"], [inat_obs(4, photos=[(0, 40, "cc0", OPEN)])], "t2")
    with_embeddings(conn, tmp_path, "m", [1, 2, 3, 4])
    holdouts.add(conn, "bench", ["2"])
    ts = build_trainset(conn, "m", "s", "large", embeddings_root=tmp_path)
    assert sorted(pid for pid, _ in ts.items) == [10, 30]


def test_the_served_index_never_holds_a_held_out_record(conn, tmp_path):
    from mycomap_vision.identify import Identifier
    stored_by_old_code(conn, [1, 2, 3])
    with_embeddings(conn, tmp_path, "m", [1, 2, 3])
    holdouts.add(conn, "bench", ["2"])
    ident = Identifier(conn, "m", "nearest", embeddings_root=tmp_path / "m", photo_info=False)
    assert sorted(ident.col_record.obs.tolist()) == ["1", "3"]


def test_a_release_is_refused_while_the_records_hold_a_held_out_record(conn, tmp_path):
    from test_release import MemoryS3, publish
    from test_release import with_embeddings as release_embeddings
    stored_by_old_code(conn, [1, 2])
    release_embeddings(conn, tmp_path)
    holdouts.add(conn, "bench", ["2"])
    s3 = MemoryS3()
    with pytest.raises(holdouts.HeldOutLeak, match="release refused"):
        publish(conn, tmp_path, s3)
    assert s3.objects == {}
    with conn:
        conn.execute("delete from records where observation_id = '2'")
    assert publish(conn, tmp_path, s3)["files"]


def test_a_trainer_run_is_refused_while_the_records_hold_a_held_out_record(conn, tmp_path):
    stored_by_old_code(conn, [1, 2])
    holdouts.add(conn, "bench", ["2"])
    with pytest.raises(holdouts.HeldOutLeak, match="trainer run refused"):
        aws.check_trainer_request(conn, ["bioclip-2"], ["nearest"], "large", "s3://b/")
    uploads = []
    with pytest.raises(holdouts.HeldOutLeak):
        trainer.run_job(conn, object(), ["bioclip-2"], ["nearest"],
                        lambda p, k: uploads.append(k), "run1", data_dir=tmp_path,
                        log=lambda s: None)
    assert uploads == []


def test_holding_out_again_changes_nothing_and_the_list_shows_leaks(conn):
    stored_by_old_code(conn, [5])
    first = holdouts.add(conn, "bench", ["5", "6", " 6 ", ""])
    again = holdouts.add(conn, "bench", ["5"])
    assert (first["added"], first["in_records"], again["added"]) == (2, 1, 0)
    assert holdouts.summary(conn)[0] | {"first_added": None, "last_added": None} == {
        "benchmark": "bench", "records": 2, "first_added": None, "last_added": None,
        "in_records": 1}


def test_releasing_a_benchmark_lets_its_records_in_with_the_next_export_and_is_logged(conn):
    holdouts.add(conn, "bench", ["2"])
    holdouts.add(conn, "paper", ["3"])
    save_records(conn, build_records([green(1), green(2), green(3)], "t1"))
    out = holdouts.release(conn, "bench")
    assert (out["released"], holdouts.is_held_out(conn, "bench"),
            holdouts.is_held_out(conn, "paper")) == (1, False, True)
    assert [tuple(r) for r in conn.execute("select benchmark, records from "
                                           "benchmark_holdout_releases")] == [("bench", 1)]
    save_records(conn, build_records([green(1), green(2), green(3)], "t2"))
    assert sorted(r[0] for r in conn.execute("select observation_id from records")) == ["1", "2"]


def test_benchmark_names_are_plain_words(conn):
    with pytest.raises(ValueError):
        holdouts.add(conn, "../x", ["1"])


def test_mv_holdout_add_and_list(conn, tmp_path, monkeypatch, capsys):
    path = tmp_path / "manifest.sqlite"
    monkeypatch.setattr(config, "MANIFEST_PATH", path)
    ids = tmp_path / "ids.csv"
    ids.write_text("observation_id,x\n7,a\n8,b\n", encoding="utf-8")
    cli.main(["holdout", "add", "--benchmark", "b1", "--csv", str(ids)])
    assert json.loads(capsys.readouterr().out)["added"] == 2
    cli.main(["holdout", "list"])
    assert json.loads(capsys.readouterr().out)[0]["records"] == 2
    cli.main(["holdout", "release", "--benchmark", "b1"])
    assert json.loads(capsys.readouterr().out)["released"] == 2
    cli.main(["holdout", "list"])
    assert json.loads(capsys.readouterr().out) == []
