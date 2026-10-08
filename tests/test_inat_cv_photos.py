"""Which photos the iNat baseline sends, and what a run that can't send them does."""
import pytest

from test_inat_cv import FakeInat, result
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import evaluate, inat_cv
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.evaluate import Record

S3 = "s3://bucket/photos/"


class FakeS3:
    def __init__(self):
        self.read = []

    def get(self, path):
        self.read.append(path)
        return b"jpeg from s3"


def copy(conn, pid, store, size, path):
    conn.execute("insert into photo_copies (photo_id, store, size, path) values (?, ?, ?, ?)",
                 (pid, store, size, path))


def test_a_photo_held_only_in_s3_is_read_from_s3_and_a_local_copy_wins(conn, tmp_path):
    copy(conn, 1, S3, "large", "l/1.jpg")
    copy(conn, 1, S3, "medium", "m/1.jpg")
    copy(conn, 2, S3, "large", "l/2.jpg")
    copy(conn, 2, str(tmp_path), "large", "local/2.jpg")
    s3 = FakeS3()
    got = dict(inat_cv.photo_inputs(conn, Record("10", "A x", "A", "F", None, "u",
                                                 photo_rows=[1, 2]), stores={S3: s3}))
    assert got[2] == tmp_path / "local/2.jpg"
    assert s3.read == []                                   # nothing read until it is needed
    assert inat_cv.read_source(got[1]) == b"jpeg from s3" and s3.read == ["m/1.jpg"]


def comparison(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    return evaluate.compare(conn, ["m1"], embeddings_root=tmp_path / "emb", log=lambda s: None)


TAXA = [{"id": 100, "name": "F", "rank": "family"},
        {"id": 10, "name": "A", "rank": "genus", "ancestors": [{"id": 100, "rank": "family"}]},
        {"id": 1, "name": "A x", "rank": "species",
         "ancestors": [{"id": 100, "rank": "family"}, {"id": 10, "rank": "genus"}]}]


def test_a_run_whose_test_records_have_no_photo_to_send_refuses_before_asking_inat(conn,
                                                                                  tmp_path):
    cmp = comparison(conn, tmp_path)
    conn.execute("delete from photo_copies")              # the photos are nowhere to be read
    conn.commit()
    fake = FakeInat(TAXA, {"results": []})
    with pytest.raises(RuntimeError, match="no photo on this machine or in S3"):
        inat_cv.run(conn, cmp["comparison_id"], fake, tmp_path / "emb", log=lambda s: None)
    assert fake.scored == []
    assert not [r for r in evaluate.scoreboard(conn, cmp["comparison_id"])
                if r["backbone"] == inat_cv.BACKBONE]


def test_a_rerun_replaces_the_comparisons_earlier_inat_rows(conn, tmp_path):
    cmp = comparison(conn, tmp_path)
    fake = FakeInat(TAXA, {"results": [result(1, "A x", "species", 90, 95)]})
    for _ in range(2):
        out = inat_cv.run(conn, cmp["comparison_id"], fake, tmp_path / "emb", log=lambda s: None)
    rows = [r for r in evaluate.scoreboard(conn, cmp["comparison_id"])
            if r["backbone"] == inat_cv.BACKBONE]
    assert sorted(r["method"] for r in rows) == ["combined-max", "vision-max"]
    assert out["without_photos"] == 0


def test_a_cached_answer_never_reads_the_photo(tmp_path):
    client = inat_cv.InatClient("jwt", tmp_path, sleep=lambda s: None)
    client._cached("score:7:none", lambda: {"results": []})          # answered before
    read = []
    client.score_image(7, lambda: read.append(1) or b"x", None, None)
    assert read == [] and client.calls == 0
