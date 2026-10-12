import io
import json

import numpy as np
import pytest

from mycomap_vision import config, release
from mycomap_vision.embed import SCHEMA as EMBED_SCHEMA


class MemoryS3:
    """Just enough of an S3 client, backed by a dict of key -> bytes."""

    def __init__(self):
        self.objects: dict[str, bytes] = {}

    def upload_file(self, filename, bucket, key):
        with open(filename, "rb") as f:
            self.objects[key] = f.read()

    def put_object(self, Bucket, Key, Body, **kw):  # noqa: N803 (boto3 names)
        self.objects[Key] = Body

    def get_object(self, Bucket, Key):  # noqa: N803
        if Key not in self.objects:
            raise KeyError(Key)
        return {"Body": io.BytesIO(self.objects[Key])}

    def download_file(self, bucket, key, filename):
        with open(filename, "wb") as f:
            f.write(self.objects[key])

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        objects = self.objects

        class Pages:
            def paginate(self, Bucket, Prefix, Delimiter):  # noqa: N803
                below = sorted({k[len(Prefix):].split(Delimiter)[0] + Delimiter
                                for k in objects if k.startswith(Prefix)
                                and Delimiter in k[len(Prefix):]})
                return [{"CommonPrefixes": [{"Prefix": Prefix + p} for p in below]}]
        return Pages()


def with_embeddings(conn, data_dir, backbone="m1", finetuned=False):
    conn.executescript(EMBED_SCHEMA)
    root = data_dir / "embeddings" / backbone
    root.mkdir(parents=True)
    np.save(root / "shard-00000.npy", np.ones((2, 3), dtype=np.float16))
    np.save(root / "shard-00000.ids.npy", np.array([1, 2]))
    conn.executemany("insert into embeddings (backbone, photo_id, shard, row, created_at) "
                     "values (?, ?, 0, ?, '2026-09-29')", [(backbone, 1, 0), (backbone, 2, 1)])
    conn.commit()
    if finetuned:
        models = data_dir / "models"
        models.mkdir()
        (models / f"{backbone}.json").write_text(json.dumps({"weights": f"{backbone}.pt",
                                                             "base_spec": "bioclip-2"}))
        (models / f"{backbone}.pt").write_bytes(b"weights")


def publish(conn, data_dir, s3, backbones=("m1",), **kw):
    return release.publish(conn, list(backbones), s3=s3, bucket="b", data_dir=data_dir,
                           work_dir=data_dir, log=lambda s: None, **kw)


def test_a_release_holds_the_manifest_the_served_embeddings_and_fine_tuned_weights(conn, tmp_path):
    data = tmp_path / "data"
    with_embeddings(conn, data, "m1", finetuned=True)
    s3 = MemoryS3()
    out = publish(conn, data, s3)
    rid = out["id"]
    info = json.loads(s3.objects[f"releases/{rid}/release.json"])
    paths = {f["path"] for f in info["files"]}
    assert paths == {"manifest.sqlite", "embeddings/m1/shard-00000.npy",
                     "embeddings/m1/shard-00000.ids.npy", "models/m1.json", "models/m1.pt"}
    assert info["backbones"] == ["m1"]
    assert "releases/current.json" not in s3.objects          # not live until asked


def test_a_backbone_without_embeddings_cannot_be_released(conn, tmp_path):
    with pytest.raises(ValueError, match="no embeddings"):
        publish(conn, tmp_path, MemoryS3(), backbones=["nothing"])


def test_only_a_finished_release_can_be_made_current(conn, tmp_path):
    s3 = MemoryS3()
    with pytest.raises(KeyError):
        release.set_current("20260101-000000", s3=s3, bucket="b")
    assert "releases/current.json" not in s3.objects


def test_the_box_pulls_the_current_release_verifies_it_and_switches_to_it(conn, tmp_path):
    data = tmp_path / "data"
    with_embeddings(conn, data)
    s3 = MemoryS3()
    rid = publish(conn, data, s3, make_current=True)["id"]
    box = tmp_path / "box"
    out = release.pull(box, s3=s3, bucket="b", log=lambda s: None)
    assert out == {"id": rid, "changed": True, "bytes": out["bytes"], "removed": []}
    assert release.current_release(box) == rid
    assert config.release_data_dir(box) == box / "releases" / rid
    assert (box / "releases" / rid / "embeddings/m1/shard-00000.npy").is_file()
    assert (box / "releases" / rid / "manifest.sqlite").is_file()
    assert release.pull(box, s3=s3, bucket="b", log=lambda s: None)["changed"] is False


def test_a_corrupted_download_never_becomes_current(conn, tmp_path):
    data = tmp_path / "data"
    with_embeddings(conn, data)
    s3 = MemoryS3()
    first = publish(conn, data, s3, make_current=True)["id"]
    box = tmp_path / "box"
    release.pull(box, s3=s3, bucket="b", log=lambda s: None)
    second = publish(conn, data, s3, label="next")["id"]
    s3.objects[f"releases/{second}/embeddings/m1/shard-00000.npy"] = b"tampered"
    with pytest.raises(ValueError, match="does not match"):
        release.pull(box, second, s3=s3, bucket="b", log=lambda s: None)
    assert release.current_release(box) == first
    assert not (box / "releases" / second).exists()


@pytest.mark.parametrize("bad", ["../escape", "/etc/passwd", "a/../../b", "C:/x", "a\\b", ""])
def test_a_release_cannot_write_outside_its_folder(bad):
    with pytest.raises(ValueError, match="unsafe"):
        release.safe_relpath(bad)


def test_the_box_keeps_the_previous_release_for_rollback_and_drops_older_ones(tmp_path):
    box = tmp_path / "box"
    ids = ["20260101-000000", "20260102-000000", "20260103-000000"]
    for rid in ids:
        (box / "releases" / rid).mkdir(parents=True)
    (box / "current.txt").write_text(ids[-1])
    assert release.prune(box, keep=2) == [ids[0]]
    assert sorted(p.name for p in (box / "releases").iterdir()) == ids[1:]


def test_before_the_first_pull_the_server_has_no_data_folder(tmp_path):
    assert config.release_data_dir(tmp_path) == tmp_path / "no-release"
    (tmp_path / "current.txt").write_text("../../etc")
    assert config.release_data_dir(tmp_path) == tmp_path / "no-release"


def test_release_labels_are_plain_words():
    with pytest.raises(ValueError):
        release.new_release_id("../x")
    assert release.RELEASE_ID.match(release.new_release_id("bioclip2-full"))


def exported(conn, at):
    """Records exported from mycomap.org at `at`."""
    from test_models_and_scoreboard import seed_two_species  # noqa: F401 (schema)
    from mycomap_vision.records import build_records, save_records
    rows = [{"source": "iNaturalist", "observation_id": str(100 + i), "scientific_name": sp,
             "genus": sp.split()[0], "family": "F", "continent": "North America",
             "validation_status_1": "yes", "validation_date_1": "1/1/2026"}
            for i, sp in enumerate(["A x", "A x", "B y"])]
    save_records(conn, build_records(rows, at))
    conn.commit()


def test_a_release_records_its_data_snapshot_and_numbers_later_builds_on_it(conn, tmp_path):
    data = tmp_path / "data"
    with_embeddings(conn, data)
    exported(conn, "2026-10-06T18:01:55+00:00")
    s3 = MemoryS3()

    def version(**kw):
        rid = publish(conn, data, s3, **kw)["id"]
        info = json.loads(s3.objects[f"releases/{rid}/release.json"])
        return info["snapshot"], info["version"]
    assert version() == ("2026-10-06", "2026-10-06")
    assert version(label="rebuild") == ("2026-10-06", "2026-10-06.2")
    assert version(label="again") == ("2026-10-06", "2026-10-06.3")
    conn.execute("update records set exported_at = '2026-10-20T09:00:00+00:00'")
    conn.commit()
    assert version(label="new-export") == ("2026-10-20", "2026-10-20")
