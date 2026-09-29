import io

import numpy as np
from PIL import Image

from conftest import inat_obs

from mycomap_vision.embed import (archive_embeddings, decode, embed_photos, load_embeddings,
                                  normalise, photos_to_embed)
from mycomap_vision.inat import save_batch
from mycomap_vision.photos import Result, save_result
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

OPEN = "inaturalist-open-data.s3.amazonaws.com"


class MeanColour:
    """A stand-in backbone: the image's mean RGB, so vectors are predictable."""
    name = "mean-colour"
    dim = 3

    def __init__(self):
        self.calls = 0

    def encode(self, images):
        self.calls += 1
        return np.stack([np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0) + 1
                         for im in images])


def jpeg(colour):
    buf = io.BytesIO()
    Image.new("RGB", (16, 12), colour).save(buf, "JPEG")
    return buf.getvalue()


def seed(conn, tmp_path, n=5, broken=()):
    rows = [{"observation_id": "5", "scientific_name": "X", "continent": "North America",
             "validation_status_1": "yes"}]
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, ["5"], [inat_obs(5, photos=[(i, 100 + i, "cc0", OPEN) for i in range(n)])],
               "t")
    store = LocalStore(tmp_path / "store")
    for i in range(n):
        rel = f"photos/large/{100 + i}.jpg"
        store.put(rel, b"not a jpeg" if 100 + i in broken else jpeg((40 * i, 10, 200)))
        save_result(conn, Result(100 + i, "done", rel, 1, "h"), "large", "now", store.location)
    return store


def test_vectors_are_unit_length_float16():
    v = normalise(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert v.dtype == np.float16
    assert np.allclose(v[0].astype(np.float32), [0.6, 0.8], atol=1e-3)
    assert np.all(np.isfinite(v))


def test_decode_applies_exif_rotation_and_rgb():
    img = decode(jpeg((1, 2, 3)))
    assert img.mode == "RGB" and img.size == (16, 12)


def test_every_held_photo_is_embedded_once_across_shards_and_runs(conn, tmp_path):
    store = seed(conn, tmp_path, n=5)
    out = tmp_path / "emb"
    todo = photos_to_embed(conn, "mean-colour", "large", store.location, "local")
    assert [pid for pid, _ in todo] == [100, 101, 102, 103, 104]
    stats = embed_photos(conn, store, MeanColour(), todo[:3], out, batch_size=2, shard_rows=2,
                         log=lambda s: None)
    assert (stats.embedded, stats.shards) == (3, 2)
    # A second run picks up only what is left, in a new shard.
    todo = photos_to_embed(conn, "mean-colour", "large", store.location, "local")
    assert [pid for pid, _ in todo] == [103, 104]
    embed_photos(conn, store, MeanColour(), todo, out, batch_size=2, log=lambda s: None)
    ids, vecs = load_embeddings(conn, "mean-colour", out)
    assert sorted(ids.tolist()) == [100, 101, 102, 103, 104]
    assert vecs.shape == (5, 3)
    assert photos_to_embed(conn, "mean-colour", "large", store.location, "local") == []


def test_another_backbone_embeds_the_same_photos_again(conn, tmp_path):
    store = seed(conn, tmp_path, n=2)
    todo = photos_to_embed(conn, "mean-colour", "large", store.location, "local")
    embed_photos(conn, store, MeanColour(), todo, tmp_path / "emb", log=lambda s: None)
    assert len(photos_to_embed(conn, "other", "large", store.location, "local")) == 2


def test_unreadable_photos_are_skipped_not_fatal(conn, tmp_path):
    store = seed(conn, tmp_path, n=3, broken={101})
    todo = photos_to_embed(conn, "mean-colour", "large", store.location, "local")
    stats = embed_photos(conn, store, MeanColour(), todo, tmp_path / "emb", log=lambda s: None)
    assert (stats.embedded, stats.unreadable) == (2, 1)
    ids, _ = load_embeddings(conn, "mean-colour", tmp_path / "emb")
    assert sorted(ids.tolist()) == [100, 102]


def test_photos_downloaded_before_stores_were_recorded_count_as_local(conn, tmp_path):
    store = seed(conn, tmp_path, n=1)
    conn.execute("update photos set store = null")
    assert len(photos_to_embed(conn, "m", "large", store.location, store.location)) == 1
    assert photos_to_embed(conn, "m", "large", "s3://b/", store.location) == []


def test_archived_embeddings_are_kept_aside_and_every_photo_is_embedded_again(conn, tmp_path):
    import pytest
    store = seed(conn, tmp_path, n=3)
    data = tmp_path / "data"
    todo = photos_to_embed(conn, "mean-colour", "large", store.location, "local")
    embed_photos(conn, store, MeanColour(), todo, data / "embeddings" / "mean-colour",
                 log=lambda s: None)
    out = archive_embeddings(conn, "mean-colour", "medium", data_dir=data)
    assert out["rows"] == 3
    kept = data / "embeddings-archive" / "mean-colour-medium"
    assert np.load(kept / "shard-00000.ids.npy").tolist() == [100, 101, 102]
    assert not (data / "embeddings" / "mean-colour").exists()
    assert len(photos_to_embed(conn, "mean-colour", "large", store.location, "local")) == 3
    # An earlier archive is never overwritten, and there is nothing left to archive now.
    (data / "embeddings" / "mean-colour").mkdir(parents=True)
    with pytest.raises(FileExistsError):
        archive_embeddings(conn, "mean-colour", "medium", data_dir=data)
    with pytest.raises(ValueError, match="no embeddings"):
        archive_embeddings(conn, "other", "medium", data_dir=data)


def test_photos_at_other_sizes_are_not_embedded(conn, tmp_path):
    store = seed(conn, tmp_path, n=2)
    assert photos_to_embed(conn, "m", "medium", store.location, "local") == []


def test_each_run_records_its_speed_and_small_runs_do_not_count(conn, tmp_path):
    from mycomap_vision.embed import photos_per_second
    store = seed(conn, tmp_path, n=3)
    todo = photos_to_embed(conn, "mean-colour", "large", store.location, "local")
    embed_photos(conn, store, MeanColour(), todo, tmp_path / "emb", log=lambda s: None)
    assert [tuple(r) for r in conn.execute("select backbone, photos from embed_runs")] == [("mean-colour", 3)]
    assert photos_per_second(conn) == {}                      # 3 photos: loading dominates
    conn.execute("insert into embed_runs values ('big', 2000, 10.0, 'cuda', 't')")
    assert photos_per_second(conn) == {"big": 200.0}
