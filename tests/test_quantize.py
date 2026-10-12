"""The int8 reference-vector check (quantize.py): the round trip is faithful, the comparison
scores both on the same records with the same queries, and the verdict follows the
pre-declared rule."""

import numpy as np
import pytest
from test_trainer_release import cut, embed  # noqa: F401  (the fixture)

from mycomap_vision import quantize


def test_int8_round_trip_stays_within_half_a_step_of_each_vector():
    rng = np.random.default_rng(0)
    v = rng.normal(size=(200, 768)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    q, scale = quantize.quantize_int8(v)
    assert q.dtype == np.int8 and scale.dtype == np.float32
    back = quantize.dequantize(q, scale)
    assert np.all(np.abs(back - v) <= scale[:, None] / 2 + 1e-7)
    cos = (back * v).sum(1) / np.linalg.norm(back, axis=1)
    assert cos.min() > 0.999
    assert quantize.bytes_per_vector(768) == {"float16": 1536, "int8": 772}


def test_a_zero_vector_survives_the_round_trip():
    q, scale = quantize.quantize_int8(np.zeros((1, 4), dtype=np.float32))
    assert scale[0] == 0 and not quantize.dequantize(q, scale).any()


def test_the_sign_test_reads_discordant_pairs():
    assert quantize.sign_test_p(0, 0) == 1.0
    assert quantize.sign_test_p(5, 5) == 1.0
    assert quantize.sign_test_p(20, 2) < 0.001
    assert quantize.sign_test_p(3, 1) > 0.5


def test_the_check_scores_both_on_the_same_records_and_names_the_release(cut, tmp_path):
    conn, _, store, rel, _ = cut
    embed(conn, store, tmp_path / "emb", "base")
    r = quantize.check(conn, "base", rel, ["nearest", "nearest+mean"],
                       embeddings_root=tmp_path / "emb", log=lambda s: None)
    assert r["dataset_release"]["release_hash"] == rel.release_hash
    assert r["test_records"] == 2 and r["reference_photos"] == 8
    sp = r["methods"]["nearest+mean"]["species_top1"]
    assert sp["n"] == 2 and sp["float16"] == sp["int8"] == 1.0
    assert sp["only_float16"] == sp["only_int8"] == 0
    assert r["verdict_nearest_mean"] == "no loss"
    assert "nearest+mean" in quantize.format_check(r)


def test_a_real_loss_is_not_called_no_loss():
    a = {str(i): (True, True, True, True) for i in range(100)}
    b = {str(i): (i >= 30, True, True, True) for i in range(100)}        # 30 lost at species
    s = quantize.summarise(a, b)["species_top1"]
    assert s["points"] == -30.0 and s["only_float16"] == 30 and s["sign_test_p"] < 0.001


def test_queries_stay_float_only_the_references_are_quantized(monkeypatch, cut, tmp_path):
    conn, _, store, rel, _ = cut
    embed(conn, store, tmp_path / "emb", "base")
    seen = []
    real = quantize.hits

    def spy(ref_vectors, query_vectors, *a, **k):
        seen.append((ref_vectors, query_vectors))
        return real(ref_vectors, query_vectors, *a, **k)
    monkeypatch.setattr(quantize, "hits", spy)
    quantize.check(conn, "base", rel, ["nearest"], embeddings_root=tmp_path / "emb",
                   log=lambda s: None)
    (f_ref, f_q), (i_ref, i_q) = seen
    assert f_q is f_ref and i_q is f_q            # the same float queries both times
    assert i_ref is not f_ref                      # int8 round trip for the references
