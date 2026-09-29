"""What keeps the server inside the 4 GB box: the reference vectors are never copied
to float32 on the CPU, and the model is loaded without its text tower."""

import numpy as np
import pytest

from mycomap_vision import methods


def reference(n=1000, dim=16, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, dim)).astype(np.float32)
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float16)


def cpu_scorer(ref, monkeypatch, chunk=None):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)    # as on the box: no GPU path
    if chunk:
        monkeypatch.setattr(methods.Scorer, "CHUNK", chunk)
    return methods.Scorer(ref)


def test_on_the_cpu_the_reference_vectors_stay_float16(monkeypatch):
    ref = reference()
    s = cpu_scorer(ref, monkeypatch)
    assert s.ref.dtype == np.float16
    assert s.ref.nbytes == ref.nbytes


def test_scoring_in_blocks_gives_the_scores_of_a_float32_copy(monkeypatch):
    ref = reference(n=1003)
    query = reference(n=3, seed=1)
    s = cpu_scorer(ref, monkeypatch, chunk=100)          # 11 blocks, the last one short
    expected = query.astype(np.float32) @ ref.astype(np.float32).T
    got = s.sims(query)
    assert got.dtype == np.float32 and got.shape == (3, 1003)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-6)


def test_float32_reference_vectors_are_not_rounded_to_float16(monkeypatch):
    ref = reference().astype(np.float32) + np.float32(1e-4)
    s = cpu_scorer(ref, monkeypatch)
    assert s.ref.dtype == np.float32


def test_nearest_specimen_scores_are_unchanged_by_the_blocks(monkeypatch):
    from types import SimpleNamespace
    ref = reference(n=500)
    index = SimpleNamespace(cols=np.arange(500), starts=np.arange(0, 500, 5))
    query = reference(n=2, seed=3)
    whole = cpu_scorer(ref, monkeypatch, chunk=10**9)
    whole_scores = methods.species_scores(whole.sims(query), index)
    blocked = cpu_scorer(ref, monkeypatch, chunk=64)
    np.testing.assert_allclose(methods.species_scores(blocked.sims(query), index),
                               whole_scores, rtol=0, atol=1e-6)


# --- the model (needs torch, open_clip and BioCLIP 2 in the local cache) -------------

def bioclip_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    hit = try_to_load_from_cache("imageomics/bioclip-2", "open_clip_model.safetensors")
    return isinstance(hit, str)


@pytest.mark.skipif(not bioclip_cached(), reason="BioCLIP 2 is not in the local model cache")
def test_the_image_tower_alone_gives_the_same_vectors_as_the_whole_model():
    torch = pytest.importorskip("torch")
    open_clip = pytest.importorskip("open_clip")
    from PIL import Image

    from mycomap_vision.models import load_open_clip_image_tower

    name = "hf-hub:imageomics/bioclip-2"
    img = Image.fromarray(np.random.default_rng(0).integers(0, 255, (300, 400, 3),
                                                            dtype=np.uint8))
    whole, _, pre = open_clip.create_model_and_transforms(name)
    with torch.inference_mode():
        expected = whole.eval().encode_image(pre(img).unsqueeze(0)).numpy()
    del whole
    tower, pre2 = load_open_clip_image_tower(name, None)
    assert not any(t.device.type == "meta" for t in tower.parameters())
    with torch.inference_mode():
        got = tower.eval()(pre2(img).unsqueeze(0)).numpy()
    np.testing.assert_array_equal(got, expected)
    # Only the image tower is held: BioCLIP 2's is 303 M of the model's 427 M parameters.
    assert sum(p.numel() for p in tower.parameters()) < 310_000_000
