"""Uploaded photos are decoded once and kept only at the size the models need, with
their EXIF place and date read before they are shrunk, and what a model sees of them
stays (almost exactly) what it saw of the full-size photo."""

import io
from fractions import Fraction

import numpy as np
import pytest
from PIL import Image
from test_api import app_with_model, open_limits

from mycomap_vision import api, uploads
from mycomap_vision.embed import embed_photos, photos_to_embed

pytest.importorskip("torchvision")
from torchvision import transforms  # noqa: E402


def textured(w, h, seed=0):
    """A photo-like image: smooth colour patches with fine grain, not flat colour."""
    rng = np.random.default_rng(seed)
    coarse = Image.fromarray(rng.integers(0, 255, (h // 64 + 2, w // 64 + 2, 3), dtype=np.uint8))
    img = np.asarray(coarse.resize((w, h), Image.BICUBIC), dtype=np.int16)
    img = img + rng.integers(-12, 12, (h, w, 3))
    return Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))


def jpeg(img, orientation=None, lat=None, lon=None, when=None, quality=92):
    exif = img.getexif()
    if orientation:
        exif[0x0112] = orientation
    if lat is not None:
        def dms(v):
            v = abs(v)
            d, m = int(v), int((v - int(v)) * 60)
            return (Fraction(d), Fraction(m), Fraction(round(((v - d) * 60 - m) * 60 * 100), 100))
        gps = exif.get_ifd(0x8825)
        gps[1], gps[2] = ("N" if lat >= 0 else "S"), dms(lat)
        gps[3], gps[4] = ("E" if lon >= 0 else "W"), dms(lon)
    if when:
        exif.get_ifd(0x8769)[36867] = when
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality, exif=exif)
    return buf.getvalue()


def clip_preprocess(side=224):
    """What BioCLIP 2 does to a photo before the model sees it (open_clip's eval
    transform): shorter side to 224, bicubic with antialiasing, centre crop."""
    return transforms.Compose([
        transforms.Resize(side, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(side), transforms.ToTensor()])


class Model:
    """A stand-in backbone with a real preprocessing pipeline."""
    def __init__(self, side):
        self.transform = clip_preprocess(side)


def test_uploads_are_kept_at_twice_the_largest_input_of_the_loaded_models():
    assert uploads.kept_side([Model(224)]) == 448
    assert uploads.kept_side([Model(224), Model(518)]) == 1036
    # Before a model is loaded, or one that doesn't say: the largest we offer.
    assert uploads.kept_side([]) == 2 * uploads.FALLBACK_MODEL_SIDE
    assert uploads.kept_side([Model(224), object()]) == 2 * uploads.FALLBACK_MODEL_SIDE


def test_an_upload_is_decoded_once_and_kept_only_at_the_size_the_models_need():
    img, _ = uploads.decode_upload(jpeg(textured(4032, 3024)), 448)
    assert img.size == (597, 448) and img.mode == "RGB"
    small, _ = uploads.decode_upload(jpeg(textured(300, 200)), 448)
    assert small.size == (300, 200), "a small photo is never enlarged"


def test_place_and_date_are_read_from_the_exif_before_the_photo_is_shrunk():
    body = jpeg(textured(4032, 3024), orientation=6, lat=39.1653, lon=-86.5264,
                when="2026:09:12 10:03:00")
    img, (lat, lon, when) = uploads.decode_upload(body, 448)
    assert round(lat, 3) == 39.165 and round(lon, 3) == -86.526 and when == "2026-09-12"
    assert img.size == (448, 597), "turned upright (EXIF orientation 6) and shrunk"


def test_an_unset_camera_clock_still_counts_as_no_date():
    _, (_, _, when) = uploads.decode_upload(
        jpeg(textured(2000, 1500), when="1970:01:01 00:00:05"), 448)
    assert when is None


def test_the_upright_photo_matches_the_full_size_one_turned_the_same_way():
    base = textured(1600, 1200, seed=3)
    img, _ = uploads.decode_upload(jpeg(base, orientation=6), 448)
    full = Image.open(io.BytesIO(jpeg(base, orientation=6)))
    from PIL import ImageOps
    expected = uploads.shrink(ImageOps.exif_transpose(full).convert("RGB"), 448)
    assert img.size == expected.size
    diff = np.abs(np.asarray(img, np.int16) - np.asarray(expected, np.int16))
    assert diff.mean() < 1.0


@pytest.mark.parametrize("w, h", [(4032, 3024), (3024, 4032), (6000, 4000), (8000, 5000),
                                  (1600, 1200)])
def test_what_the_model_sees_is_unchanged_by_shrinking_first(w, h):
    pre = clip_preprocess(224)
    body = jpeg(textured(w, h, seed=w))
    full = pre(Image.open(io.BytesIO(body)).convert("RGB"))
    small, _ = uploads.decode_upload(body, uploads.kept_side([Model(224)]))
    kept = pre(small)
    assert kept.shape == full.shape
    diff = (kept - full).abs()
    # Measured 0.0007-0.0008 on average (pixel values 0-1). Decoding a big JPEG at a
    # scale nearer the kept size (draft mode down to 448 px) gave 0.0010-0.0014.
    assert float(diff.mean()) < 0.001 and float(diff.max()) < 0.01, \
        f"mean {float(diff.mean()):.5f} max {float(diff.max()):.4f}"


def bioclip_cached() -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    return isinstance(try_to_load_from_cache("imageomics/bioclip-2",
                                             "open_clip_model.safetensors"), str)


@pytest.mark.skipif(not bioclip_cached(), reason="BioCLIP 2 is not in the local model cache")
def test_bioclip_says_the_same_of_a_shrunk_upload_as_of_the_full_photo():
    from mycomap_vision.models import load_backbone
    bb = load_backbone("bioclip-2")
    side = uploads.kept_side([bb])
    assert side == 448
    for w, h in [(4032, 3024), (3024, 4032)]:
        body = jpeg(textured(w, h, seed=h))
        full = Image.open(io.BytesIO(body)).convert("RGB")
        small, _ = uploads.decode_upload(body, side)
        a, b = bb.encode([bb.transform(full)])[0], bb.encode([bb.transform(small)])[0]
        cos = float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))
        assert cos > 0.999, cos


# --- through the API ----------------------------------------------------------------

class Sizes:
    """A backbone that records the size of every photo it is given."""

    def __init__(self, name, side=224):
        self.name, self.dim, self.seen = name, 2, []
        self.transform = clip_preprocess(side)

    def encode(self, images):
        self.seen.extend(im.size for im in images)
        return np.asarray([[np.asarray(im, np.float32)[..., 0].mean() / 255, 0.5] for im in images])


def test_identify_hands_the_models_photos_no_bigger_than_they_need(conn, tmp_path):
    from fastapi.testclient import TestClient
    from test_models_and_scoreboard import Const, seed_two_species
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    conn.commit()
    model = Sizes("m1")
    client = TestClient(api.create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                                       backbone_loader=lambda n: model, limits=open_limits(),
                                       background=False, web_dist=None, note=lambda s: None))
    body = jpeg(textured(4032, 3024), orientation=6, lat=39.1653, lon=-86.5264,
                when="2026:09:12 10:03:00")
    photos = [("photos", (f"p{i}.jpg", body, "image/jpeg")) for i in range(2)]
    first = client.post("/api/identify", files=photos, data={"models": "m1/nearest"})
    assert first.status_code == 200, first.text
    assert first.json()["context_used"]["place_from"] == "photo"
    assert first.json()["context_used"]["observed_on"] == "2026-09-12"
    # Before the model was loaded: the largest size any offered model needs.
    assert all(min(s) == uploads.kept_side([]) for s in model.seen)
    model.seen.clear()
    again = client.post("/api/identify", files=photos, data={"models": "m1/nearest"})
    assert again.status_code == 200
    assert model.seen == [(448, 597)] * 2, "upright, at twice the model's 224 px"


def test_photos_over_the_request_limit_together_are_refused_even_without_a_length(
        conn, tmp_path, monkeypatch):
    client = app_with_model(conn, tmp_path)
    part = jpeg(textured(400, 300), quality=100)
    monkeypatch.setattr(api, "MAX_REQUEST_BYTES", len(part) * 3 // 2)     # one fits, two don't
    boundary = "vision-test-boundary"
    body = b"".join(
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"photos\"; filename=\"p{i}.jpg\"\r\n"
        f"Content-Type: image/jpeg\r\n\r\n".encode() + part + b"\r\n" for i in range(2))
    body += f"--{boundary}--\r\n".encode()

    def chunks():                      # a streamed body: no content-length header
        for i in range(0, len(body), 8192):
            yield body[i:i + 8192]
    res = client.post("/api/identify", content=chunks(),
                      headers={"content-type": f"multipart/form-data; boundary={boundary}"})
    assert res.status_code == 413 and "together" in res.json()["detail"]
