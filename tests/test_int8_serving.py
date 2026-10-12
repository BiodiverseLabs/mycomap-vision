"""int8 reference vectors on the box (Steve, 2026-10-12: v1 stays on the 4 GB box): a
release made with --int8-vectors ships int8 codes + a float32 scale per vector, the server
maps them, scores them as the dequantized vectors, mixes them with the nightly update's
float16 shards, and answers as it does from float16 (accuracy: mv int8-check)."""

import numpy as np
import pytest
from test_serving import big_reference

from mycomap_vision import quantize, release
from mycomap_vision.embed import load_embeddings
from mycomap_vision.identify import Identifier
from mycomap_vision.serving import SCALE_SUFFIX, MappedVectors, ServedIndexes, map_embeddings


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)


def unit(rng, n, dim=64):
    v = rng.standard_normal((n, dim)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def test_int8_rows_come_out_and_score_as_their_dequantized_vectors():
    rng = np.random.default_rng(0)
    a, b = unit(rng, 300), unit(rng, 200)
    qa, sa = quantize.quantize_int8(a)
    mv = MappedVectors([qa, b], [sa, None])            # an int8 release + a float16 night
    assert mv.dtype == np.float16 and mv.shape == (500, 64)
    assert mv.storage() == {"int8_rows": 300, "float_rows": 200}
    want = np.vstack([quantize.dequantize(qa, sa), b.astype(np.float32)])
    rows = np.array([0, 299, 300, 499, 17])
    np.testing.assert_allclose(mv[rows].astype(np.float32), want[rows], atol=1e-3)
    cols = rng.permutation(500)
    q = unit(rng, 3).astype(np.float32)
    np.testing.assert_allclose(mv.select(cols).sims(q), q @ want[cols].T, atol=1e-5)


def test_int8_codes_without_their_scales_are_refused():
    rng = np.random.default_rng(1)
    q, s = quantize.quantize_int8(unit(rng, 10))
    with pytest.raises(ValueError, match="one scale per row"):
        MappedVectors([q], [s[:5]])
    with pytest.raises(ValueError, match="float16 or float32"):
        MappedVectors([q], [None])


def test_a_float_folder_converts_to_int8_shards_and_keeps_its_ids(conn, tmp_path):
    big_reference(conn, tmp_path / "emb", records=50, photos_each=3, shard_rows=40)
    got = quantize.write_int8_shards(tmp_path / "emb" / "big", tmp_path / "int8" / "big")
    assert got["shards"] == 4 and got["vectors"] == 150 and got["worst_cosine"] >= 0.999
    out = tmp_path / "int8" / "big"
    assert np.load(out / "shard-00000.npy").dtype == np.int8
    assert np.load(out / f"shard-00000{SCALE_SUFFIX}").shape == (40,)
    np.testing.assert_array_equal(np.load(out / "shard-00003.ids.npy"),
                                  np.load(tmp_path / "emb" / "big" / "shard-00003.ids.npy"))
    assert np.load(tmp_path / "emb" / "big" / "shard-00000.npy").dtype == np.float16  # untouched
    with pytest.raises(ValueError, match="exists"):
        quantize.write_int8_shards(tmp_path / "emb" / "big", out)


def test_an_index_on_int8_shards_answers_as_the_float16_one(conn, tmp_path):
    big_reference(conn, tmp_path / "emb", records=120, photos_each=3, shard_rows=100)
    quantize.write_int8_shards(tmp_path / "emb" / "big", tmp_path / "int8" / "big")
    _, vecs = load_embeddings(conn, "big", tmp_path / "emb" / "big")
    query = vecs[[5, 200, 301]]
    for method in ("nearest", "nearest+mean"):
        f16 = Identifier(conn, "big", method, tmp_path / "emb" / "big", photo_info=False)
        i8 = Identifier(conn, "big", method, tmp_path / "int8" / "big", photo_info=False)
        assert i8.vector_storage() == {"int8_rows": 360, "float_rows": 0}
        assert f16.vector_storage() == {"int8_rows": 0, "float_rows": 360}
        assert all(isinstance(s, np.memmap) and s.dtype == np.int8
                   for s in i8.nearest.scorer.ref.vectors.shards)
        np.testing.assert_allclose(i8.nearest.photo_sims(query), f16.nearest.photo_sims(query),
                                   atol=2e-2)
        a, b = i8.identify_vectors(query), f16.identify_vectors(query)
        for rank in ("family", "genus", "species"):
            assert a["ranks"][rank][0]["name"] == b["ranks"][rank][0]["name"]


def test_int8_codes_without_a_scale_file_are_refused_before_serving(conn, tmp_path):
    big_reference(conn, tmp_path / "emb", records=20, photos_each=2, shard_rows=40)
    out = tmp_path / "int8" / "big"
    quantize.write_int8_shards(tmp_path / "emb" / "big", out)
    (out / f"shard-00000{SCALE_SUFFIX}").unlink()
    with pytest.raises(ValueError, match="without their"):
        map_embeddings(conn, "big", out)


def test_a_release_with_int8_vectors_ships_codes_and_scales_not_the_floats(conn, tmp_path):
    big_reference(conn, tmp_path / "embeddings", records=30, photos_each=2, shard_rows=40)
    files = release.release_files(conn, ["big"], tmp_path, int8_dir=tmp_path / "work")
    shipped = {rel: path for path, rel in files}
    assert f"embeddings/big/shard-00000{SCALE_SUFFIX}" in shipped
    assert np.load(shipped["embeddings/big/shard-00000.npy"]).dtype == np.int8
    assert str(shipped["embeddings/big/shard-00000.npy"]).startswith(str(tmp_path / "work"))
    plain = {rel for _p, rel in release.release_files(conn, ["big"], tmp_path)}
    assert f"embeddings/big/shard-00000{SCALE_SUFFIX}" not in plain


def test_served_indexes_list_what_they_serve():
    s = ServedIndexes(lambda key, version: f"index {key}", note=lambda m: None)
    assert s.served() == {}
    s.get(("big", "nearest"), 1)
    assert s.served() == {("big", "nearest"): "index ('big', 'nearest')"}


def test_health_says_how_each_served_index_holds_its_vectors(conn, tmp_path):
    from test_api import app_with_model, post_photos
    client = app_with_model(conn, tmp_path)
    assert client.get("/api/health").json()["vectors"] == {}        # nothing built yet
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    vectors = client.get("/api/health").json()["vectors"]
    assert vectors == {"m1/nearest": {"int8_rows": 0, "float_rows": 8}}
