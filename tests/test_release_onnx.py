"""A serving release ships the ONNX image tower (models/<b>.onnx and
models/<b>.onnx.json, `mv export-onnx`) when it is there, and never one exported from
other weights than the fine-tune it ships."""

import json

import pytest
from test_serving import big_reference

from mycomap_vision import release


def model(tmp_path, name="big"):
    (tmp_path / "models").mkdir(exist_ok=True)
    (tmp_path / "models" / f"{name}.pt").write_bytes(b"tail weights")
    (tmp_path / "models" / f"{name}.json").write_text(json.dumps({"weights": f"{name}.pt"}))
    return release.sha256_of(tmp_path / "models" / f"{name}.pt")


def onnx(tmp_path, sha, name="big"):
    (tmp_path / "models" / f"{name}.onnx").write_bytes(b"onnx")
    (tmp_path / "models" / f"{name}.onnx.json").write_text(json.dumps(
        {"source_weights": {"file": f"{name}.pt", "sha256": sha}, "quantization": "dynamic-int8"}))


def shipped(conn, tmp_path):
    return {rel for _p, rel in release.release_files(conn, ["big"], tmp_path)}


def test_no_onnx_export_ships_nothing_extra(conn, tmp_path):
    big_reference(conn, tmp_path / "embeddings", records=10, photos_each=1, shard_rows=20)
    model(tmp_path)
    assert not any(r.endswith(".onnx") for r in shipped(conn, tmp_path))


def test_an_onnx_export_of_the_shipped_weights_rides_along(conn, tmp_path):
    big_reference(conn, tmp_path / "embeddings", records=10, photos_each=1, shard_rows=20)
    onnx(tmp_path, model(tmp_path))
    assert {"models/big.onnx", "models/big.onnx.json"} <= shipped(conn, tmp_path)


def test_a_stale_or_half_onnx_export_is_refused(conn, tmp_path):
    big_reference(conn, tmp_path / "embeddings", records=10, photos_each=1, shard_rows=20)
    model(tmp_path)
    onnx(tmp_path, "0" * 64)
    with pytest.raises(ValueError, match="other weights"):
        shipped(conn, tmp_path)
    (tmp_path / "models" / "big.onnx").unlink()
    with pytest.raises(ValueError, match="go together"):
        shipped(conn, tmp_path)


def test_the_export_may_name_its_source_as_a_plain_sha():
    assert release._onnx_source_sha({"source_weights": "a" * 64}) == "a" * 64
    assert release._onnx_source_sha({}) is None
