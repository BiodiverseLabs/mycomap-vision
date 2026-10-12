"""A development benchmark's own downloaded photos become ordinary copies, so the v1 cut puts
them into the main store with copy-photos instead of fetching them from iNat a second time;
never a sealed benchmark's, never a file that isn't the one downloaded."""

import hashlib

import pytest
from conftest import inat_obs
from test_models_and_scoreboard import OPEN

from mycomap_vision import heldout, photos
from mycomap_vision.inat import save_batch
from mycomap_vision.storage import LocalStore

BENCH = "heldout-2026-10-08"


def setup(conn, tmp_path, sealed=0):
    heldout.ensure_schema(conn)
    folder = tmp_path / "benchmarks" / BENCH
    conn.execute("insert into heldout_sets (name, records, ids_sha256, inputs_json, truth_rule, "
                 "frozen_at, code_version, sealed) values (?, 3, 'x', '{}', 'r', 't', 'c', ?)",
                 (BENCH, sealed))
    save_batch(conn, ["7001", "7002"], [inat_obs(7001, photos=[(0, 70011, "cc0", OPEN)]),
                                         inat_obs(7002, photos=[(0, 70021, "cc0", OPEN)])], "t")
    rows = []
    for pid, body in ((70011, b"photo one"), (70021, b"photo two"), (99999, b"not ours")):
        rel = f"photos/large/{pid % 1000:03d}/{pid}.jpg"
        (folder / rel).parent.mkdir(parents=True, exist_ok=True)
        (folder / rel).write_bytes(body)
        rows.append((BENCH, str(pid)[:4], pid, 0, "cc0", "open", "a", 7, "alice", "u", "h",
                     "done", str(folder), "large", rel, len(body),
                     hashlib.sha256(body).hexdigest(), 1, None, "t"))
    conn.executemany(f"insert into heldout_photos values ({','.join('?' * 20)})", rows)
    conn.commit()
    return folder


def test_downloaded_benchmark_photos_become_copies_in_their_folder(conn, tmp_path):
    folder = setup(conn, tmp_path)
    out = photos.adopt_benchmark_photos(conn, BENCH, log=lambda s: None)
    assert out["candidates"] == 2 and out["adopted"] == 2      # 99999: no photos row
    assert out["stores"] == [str(folder)]
    got = dict(conn.execute("select photo_id, path from photo_copies where store = ?",
                            (str(folder),)))
    assert set(got) == {70011, 70021}
    again = photos.adopt_benchmark_photos(conn, BENCH, log=lambda s: None)
    assert again["candidates"] == 0                            # nothing twice


def test_a_photo_listed_on_two_benchmark_observations_is_adopted_once(conn, tmp_path):
    folder = setup(conn, tmp_path)
    row = conn.execute("select * from heldout_photos where photo_id = 70011").fetchone()
    conn.execute(f"insert into heldout_photos values ({','.join('?' * 20)})",
                 (row[0], "7999", *row[2:]))
    conn.commit()
    out = photos.adopt_benchmark_photos(conn, BENCH, log=lambda s: None)
    assert out["adopted"] == 2
    assert conn.execute("select count(*) from photo_copies where store = ? and photo_id = 70011",
                        (str(folder),)).fetchone()[0] == 1


def test_a_missing_or_resized_file_is_not_adopted(conn, tmp_path):
    folder = setup(conn, tmp_path)
    (folder / "photos/large/011/70011.jpg").unlink()
    (folder / "photos/large/021/70021.jpg").write_bytes(b"something else entirely")
    out = photos.adopt_benchmark_photos(conn, BENCH, log=lambda s: None)
    assert out["adopted"] == 0 and out["missing_file"] == 1 and out["size_differs"] == 1


def test_a_sealed_or_unknown_benchmark_is_refused(conn, tmp_path):
    setup(conn, tmp_path, sealed=1)
    with pytest.raises(ValueError, match="sealed"):
        photos.adopt_benchmark_photos(conn, BENCH, log=lambda s: None)
    with pytest.raises(ValueError, match="no benchmark"):
        photos.adopt_benchmark_photos(conn, "nope", log=lambda s: None)


def test_copy_photos_then_moves_them_hash_checked(conn, tmp_path):
    folder = setup(conn, tmp_path)
    (folder / "photos/large/021/70021.jpg").write_bytes(b"photo TWO")   # same size, new bytes
    photos.adopt_benchmark_photos(conn, BENCH, log=lambda s: None)
    dest = LocalStore(tmp_path / "main-store")
    out = photos.copy_photos(conn, LocalStore(folder), dest, "large", log=lambda s: None)
    assert out["copied"] == 1 and out["mismatch"] == 1          # the changed file stays home
    assert (tmp_path / "main-store/photos/large/011/70011.jpg").read_bytes() == b"photo one"
    assert not (tmp_path / "main-store/photos/large/021/70021.jpg").exists()
