"""What the server holds to identify, and swapping it: the reference vectors are read
from disk (memory-mapped), never copied into memory, so a rebuilt index never needs a
second copy; a rebuild is swapped in whole, a failed one leaves the old index serving,
and identifications during a rebuild wait for it or are told to try again."""

import gc
import threading
import time
import tracemalloc
import weakref
from datetime import datetime, timezone

import numpy as np
import pytest
from test_api import app_with_model, post_photos
from test_models_and_scoreboard import OPEN
from test_permissions import answer, set_photo

from conftest import inat_obs
from mycomap_vision import api, identify, permissions
from mycomap_vision.embed import load_embeddings
from mycomap_vision.identify import Identifier
from mycomap_vision.inat import save_batch
from mycomap_vision.methods import Scorer
from mycomap_vision.records import build_records, save_records
from mycomap_vision.serving import MappedVectors, ServedIndexes, Updating, map_embeddings

DIM = 2048                 # wide vectors, so they dwarf the records' own memory


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """As on the server box: no GPU (a GPU copy is the GPU's memory, not the box's)."""
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)


def big_reference(conn, root, records=400, photos_each=8, shard_rows=1000, seed=0):
    """`records` DNA-verified records of 40 species, `photos_each` photos each, with
    random unit float16 vectors written as embedding shards of backbone "big"."""
    rows, obs, ids = [], [], []
    for i in range(records):
        oid = 5000 + i
        sp = f"G{i % 8} s{i % 40}"
        rows.append({"observation_id": str(oid), "scientific_name": sp, "genus": sp.split()[0],
                     "family": "F", "continent": "North America",
                     "validation_status_1": "yes", "validation_date_1": "1/1/2026"})
        pids = [oid * 100 + k for k in range(photos_each)]
        obs.append(inat_obs(oid, photos=[(k, p, "cc0", OPEN) for k, p in enumerate(pids)]))
        ids.extend(pids)
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, [r["observation_id"] for r in rows], obs, "t")
    rng = np.random.default_rng(seed)
    out = root / "big"
    out.mkdir(parents=True, exist_ok=True)
    conn.executescript("create table if not exists embeddings (backbone text not null, "
                       "photo_id integer not null, shard integer not null, row integer not null, "
                       "created_at text not null, primary key (backbone, photo_id))")
    for shard, s in enumerate(range(0, len(ids), shard_rows)):
        chunk = np.asarray(ids[s:s + shard_rows], dtype=np.int64)
        v = rng.standard_normal((len(chunk), DIM)).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        np.save(out / f"shard-{shard:05d}.npy", v.astype(np.float16))
        np.save(out / f"shard-{shard:05d}.ids.npy", chunk)
        conn.executemany("insert into embeddings values ('big', ?, ?, ?, 'now')",
                         [(int(p), shard, i) for i, p in enumerate(chunk.tolist())])
    conn.commit()
    return len(ids) * DIM * 2            # bytes of the reference vectors


def traced_peak(fn):
    """(result, peak bytes Python and numpy allocated while fn ran)."""
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        return result, tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


# --- the reference vectors, memory-mapped --------------------------------------------

def test_building_an_index_never_reads_the_reference_vectors_into_memory(conn, tmp_path):
    vec_bytes = big_reference(conn, tmp_path / "emb")
    ident, peak = traced_peak(lambda: Identifier(conn, "big", "nearest", tmp_path / "emb" / "big",
                                                 photo_info=False))
    # Reading them in (and copying them into species order) took twice their size.
    assert peak < vec_bytes / 4, f"building allocated {peak / 2**20:.1f} MB"
    ref = ident.nearest.scorer.ref
    assert all(isinstance(s, np.memmap) for s in ref.vectors.shards)
    # Identifying scores straight from the files, one shard (widened to float32) at a time.
    query = ident.nearest.scorer.ref.vectors[[0, 1]]
    _, peak = traced_peak(lambda: ident.identify_vectors(query))
    assert peak < vec_bytes, f"identifying allocated {peak / 2**20:.1f} MB"


def test_a_built_index_keeps_no_record_objects(conn, tmp_path):
    """The records a build reads are ~0.5 GB of small objects at the full set; the index
    keeps arrays instead (identify.Specimens), so that memory goes back to the OS."""
    from mycomap_vision.evaluate import Record
    big_reference(conn, tmp_path / "emb", records=60, photos_each=2, shard_rows=50)
    ident = Identifier(conn, "big", "nearest", tmp_path / "emb" / "big", photo_info=False)
    gc.collect()
    ours = {str(5000 + i) for i in range(60)}
    assert not [o for o in gc.get_objects()
                if type(o) is Record and o.observation_id in ours]
    first = ident.col_record[0]
    assert first.observation_id.isdigit() and first.species.startswith("G")
    assert len(ident.rec_starts) == 60


def test_scores_read_from_disk_are_the_scores_of_the_vectors_in_memory(conn, tmp_path,
                                                                      monkeypatch):
    big_reference(conn, tmp_path / "emb", records=120, photos_each=3, shard_rows=100)
    mapped = Identifier(conn, "big", "nearest", tmp_path / "emb" / "big")
    monkeypatch.setattr(identify, "map_embeddings",                       # the old way
                        lambda c, b, root, layer_root=None: load_embeddings(c, b, root))
    in_memory = Identifier(conn, "big", "nearest", tmp_path / "emb" / "big")
    assert isinstance(in_memory.nearest.scorer.ref, np.ndarray)
    ids, vecs = load_embeddings(conn, "big", tmp_path / "emb" / "big")
    query = vecs[[5, 200, 301]]
    np.testing.assert_allclose(mapped.nearest.photo_sims(query),
                               in_memory.nearest.photo_sims(query), rtol=0, atol=1e-6)
    a, b = mapped.identify_vectors(query), in_memory.identify_vectors(query)
    assert [s["observation_id"] for s in a["specimens"]] == \
        [s["observation_id"] for s in b["specimens"]]
    for rank in ("family", "genus", "species"):
        assert [c["name"] for c in a["ranks"][rank]] == [c["name"] for c in b["ranks"][rank]]
        np.testing.assert_allclose([c["score"] for c in a["ranks"][rank]],
                                   [c["score"] for c in b["ranks"][rank]], atol=1e-4)


def test_mapped_vectors_give_the_rows_of_the_shards_in_order(tmp_path):
    rng = np.random.default_rng(1)
    parts = [rng.standard_normal((n, 4)).astype(np.float16) for n in (3, 5, 2)]
    for i, p in enumerate(parts):
        np.save(tmp_path / f"{i}.npy", p)
    mv = MappedVectors([np.load(tmp_path / f"{i}.npy", mmap_mode="r") for i in range(3)])
    whole = np.concatenate(parts)
    assert mv.shape == (10, 4) and len(mv) == 10
    rows = np.array([9, 0, 4, 3, 7])
    np.testing.assert_array_equal(mv[rows], whole[rows])
    np.testing.assert_array_equal(np.asarray(mv.select(rows)), whole[rows])
    q = rng.standard_normal((2, 4)).astype(np.float32)
    np.testing.assert_allclose(Scorer(mv.select(rows)).sims(q),
                               q @ whole[rows].astype(np.float32).T, rtol=0, atol=1e-5)


def test_embedding_files_that_disagree_are_refused_before_serving(conn, tmp_path):
    big_reference(conn, tmp_path / "emb", records=10, photos_each=2, shard_rows=8)
    np.save(tmp_path / "emb" / "big" / "shard-00001.ids.npy", np.arange(3, dtype=np.int64))
    with pytest.raises(ValueError, match="shard 1: 3 photo ids"):
        map_embeddings(conn, "big", tmp_path / "emb" / "big")


# --- swapping ---------------------------------------------------------------------

class Built:
    def __init__(self, version):
        self.version = version


def test_an_index_swap_never_holds_a_second_copy_of_the_reference_vectors(conn, tmp_path):
    vec_bytes = big_reference(conn, tmp_path / "emb")
    manifest_path = tmp_path / "manifest.sqlite"

    def build(key, version):
        import sqlite3
        own = sqlite3.connect(manifest_path)
        try:
            return Identifier(own, "big", "nearest", tmp_path / "emb" / "big", photo_info=False)
        finally:
            own.close()
    served = ServedIndexes(build, note=lambda s: None)
    old = weakref.ref(served.get(("big", "nearest"), 1))
    new, peak = traced_peak(lambda: served.get(("big", "nearest"), 2))
    assert peak < vec_bytes / 4, f"the swap allocated {peak / 2**20:.1f} MB"
    gc.collect()
    assert old() is None, "the old index is released once the new one is in"
    assert served.current(("big", "nearest"))[1] is new


def test_a_rebuild_is_swapped_in_whole_and_callers_meanwhile_hear_updating():
    go, calls = threading.Event(), []

    def build(key, version):
        calls.append(version)
        if version == 2:
            assert go.wait(5)
        return Built(version)
    served = ServedIndexes(build, wait=0.1, note=lambda s: None)
    v1 = served.get("k", 1)
    with pytest.raises(Updating, match="being updated"):
        served.get("k", 2)                       # not the old one: v2 may drop photos
    with pytest.raises(Updating):
        served.get("k", 2)
    assert calls == [1, 2], "one build of v2, however many callers ask"
    assert served.current("k")[1] is v1           # still in place until v2 is complete
    go.set()
    assert served.get("k", 2, wait=5).version == 2
    assert served.current("k")[0] == 2


def test_a_failed_build_leaves_the_old_index_serving_and_is_retried_later():
    now = [0.0]
    fail = [True]
    notes = []

    def build(key, version):
        if version == 2 and fail[0]:
            raise OSError("release file unreadable")
        return Built(version)
    served = ServedIndexes(build, wait=5, retry_seconds=60, note=notes.append,
                           clock=lambda: now[0])
    v1 = served.get("k", 1)
    assert served.get("k", 2) is v1
    assert any("failed" in n and "keeps serving" in n for n in notes)
    assert served.get("k", 2) is v1, "not rebuilt on every request"
    fail[0] = False
    now[0] = 61
    assert served.get("k", 2).version == 2


def test_with_nothing_to_serve_yet_a_failed_build_is_reported():
    def build(key, version):
        raise identify.NotTrainedHere("no saved training")
    served = ServedIndexes(build, note=lambda s: None)
    with pytest.raises(identify.NotTrainedHere):
        served.get("k", 1)


def test_builds_run_one_at_a_time():
    running, most = [0], [0]
    lock = threading.Lock()

    def build(key, version):
        with lock:
            running[0] += 1
            most[0] = max(most[0], running[0])
        time.sleep(0.05)
        with lock:
            running[0] -= 1
        return Built(version)
    served = ServedIndexes(build, note=lambda s: None)
    threads = [threading.Thread(target=served.get, args=(k, 1)) for k in "abcd"]
    for t in threads:
        t.start()
    for t in threads:
        t.join(5)
    assert most[0] == 1
    assert all(served.current(k)[0] == 1 for k in "abcd")


# --- through the API ----------------------------------------------------------------

def specimen_ids(res):
    assert res.status_code == 200, res.text
    return {s["observation_id"] for s in res.json()["results"][0]["specimens"]}


def test_identifications_during_a_rebuild_are_told_to_try_again_then_use_the_new_index(
        conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_INDEX_WAIT_SECONDS", "0.2")
    client = app_with_model(conn, tmp_path)
    set_photo(conn, 1002, "arr", 43)                     # obs 102
    assert "102" in specimen_ids(post_photos(client, [247], "m1/nearest"))
    real, go = api.Identifier, threading.Event()

    def slow(*a, **kw):
        assert go.wait(10)
        return real(*a, **kw)
    monkeypatch.setattr(api, "Identifier", slow)
    permissions.save_snapshot(conn, answer((43, "withdrawn")), datetime.now(timezone.utc))
    res = post_photos(client, [247], "m1/nearest")
    assert res.status_code == 503 and res.headers["retry-after"]
    assert "being updated" in res.json()["detail"]
    assert client.get("/api/stats").status_code == 200      # the rest of the site answers
    go.set()
    end = time.monotonic() + 10
    while (res := post_photos(client, [247], "m1/nearest")).status_code == 503:
        assert time.monotonic() < end
        time.sleep(0.05)
    assert "102" not in specimen_ids(res), "the new index leaves the withdrawn photo out"


def test_a_rebuild_that_fails_leaves_the_old_index_answering(conn, tmp_path, monkeypatch):
    client = app_with_model(conn, tmp_path)
    before = specimen_ids(post_photos(client, [247], "m1/nearest"))

    def broken(*a, **kw):
        raise OSError("the embedding files are unreadable")
    monkeypatch.setattr(api, "Identifier", broken)
    set_photo(conn, 1002, "arr", 43)
    permissions.save_snapshot(conn, answer((43, "withdrawn")), datetime.now(timezone.utc))
    assert specimen_ids(post_photos(client, [247], "m1/nearest")) == before


def test_the_server_builds_its_index_memory_mapped_without_every_photos_licence(
        conn, tmp_path, monkeypatch):
    seen = []
    real = api.Identifier

    def spy(*a, **kw):
        ident = real(*a, **kw)
        seen.append(ident)
        return ident
    monkeypatch.setattr(api, "Identifier", spy)
    client = app_with_model(conn, tmp_path)
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    [ident] = seen
    assert ident.photos == {}, "licences are read per identification instead"
    assert all(isinstance(s, np.memmap) for s in ident.nearest.scorer.ref.vectors.shards)

