"""The image tower as ONNX (onnxtower.py, MV_EMBED_ONNX): exported and checked against torch,
used only for the weights it was made from, torch otherwise."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("onnxruntime")
pytest.importorskip("onnx")

from fastapi.testclient import TestClient  # noqa: E402
from test_api import open_limits, post_photos  # noqa: E402
from test_models_and_scoreboard import Const, seed_two_species  # noqa: E402
from torchvision import transforms as T  # noqa: E402

from mycomap_vision import finetune, onnxtower  # noqa: E402
from mycomap_vision.api import create_app  # noqa: E402
from mycomap_vision.embed import embed_photos, photos_to_embed  # noqa: E402


class Tiny:
    """A small image model shaped like a served backbone (transform, image_module, encode)."""

    def __init__(self, dim=8, seed=0):
        torch.manual_seed(seed)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 3, stride=2), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(16, dim)).eval()
        self.transform = T.Compose([T.Resize(32, interpolation=T.InterpolationMode.BICUBIC),
                                    T.CenterCrop(32), lambda im: im.convert("RGB"), T.ToTensor(),
                                    T.Normalize([0.48, 0.46, 0.41], [0.27, 0.26, 0.28])])
        self.dim, self.name = dim, "tiny"

    def image_module(self):
        return self.model

    def prepare(self, image):
        return self.transform(image)

    def encode(self, items):
        x = torch.stack([i if isinstance(i, torch.Tensor) else self.transform(i) for i in items])
        with torch.no_grad():
            return self.model(x).numpy()


def exported(tmp_path, name="m1", dim=8, weights=b"trained weights", precision="fp32"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / f"{name}.pt").write_bytes(weights)
    (tmp_path / f"{name}.json").write_text(json.dumps({"weights": f"{name}.pt"}))
    tiny = Tiny(dim)
    meta = onnxtower.export(tiny, name, tmp_path, tmp_path / f"{name}.pt",
                            check=onnxtower.check_images(12), precision=precision,
                            note=lambda s: None)
    return tiny, meta


@pytest.mark.parametrize("precision", ["fp32", "int8"])
def test_the_onnx_tower_embeds_as_torch_does(tmp_path, precision):
    tiny, meta = exported(tmp_path, precision=precision)
    assert meta["precision"] == precision
    assert (tmp_path / "m1.onnx").is_file() and (tmp_path / "m1.onnx.json").is_file()
    assert meta["check"]["min_cosine"] >= onnxtower.MIN_COSINE
    assert meta["source_weights"]["sha256"] == onnxtower.sha256_of(tmp_path / "m1.pt")
    assert meta["input"]["resize"] == 32 and meta["output_dim"] == 8
    onnx = onnxtower.OnnxBackbone("m1", tmp_path / "m1.onnx", meta)
    photos = onnxtower.check_images(6, seed=3)
    a, b = onnx.encode(photos), tiny.encode(photos)
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert cos.min() >= 0.99
    # Rebuilt from the .json alone, the preprocessing equals the model's own.
    assert torch.allclose(onnx.prepare(photos[0]), tiny.prepare(photos[0]), atol=1e-5)


def test_an_export_that_doesnt_match_torch_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(onnxtower, "MIN_COSINE", 1.01)
    with pytest.raises(ValueError, match="min cosine"):
        exported(tmp_path)
    assert not (tmp_path / "m1.onnx").exists() and not (tmp_path / "m1.onnx.json").exists()


def test_the_loader_uses_onnx_only_for_the_weights_it_was_made_from(tmp_path):
    exported(tmp_path)
    torch_loaded = []
    load = onnxtower.loader(lambda n: torch_loaded.append(n) or "torch model", tmp_path,
                            note=lambda s: None)
    assert isinstance(load("m1"), onnxtower.OnnxBackbone)
    assert onnxtower.STATE.state()["m1"] == {"mode": "onnx-fp32", "reason": None}
    (tmp_path / "m1.pt").write_bytes(b"retrained weights")            # a new .pt
    assert load("m1") == "torch model"
    assert "other weights" in onnxtower.STATE.state()["m1"]["reason"]
    assert load("never-exported") == "torch model"
    assert "no never-exported.onnx" in onnxtower.STATE.state()["never-exported"]["reason"]
    assert torch_loaded == ["m1", "never-exported"]


def test_a_tampered_onnx_file_is_not_used(tmp_path):
    exported(tmp_path)
    with open(tmp_path / "m1.onnx", "ab") as f:
        f.write(b"\0")
    load = onnxtower.loader(lambda n: "torch model", tmp_path, note=lambda s: None)
    assert load("m1") == "torch model"


def test_the_server_embeds_with_onnx_when_turned_on(conn, tmp_path, monkeypatch):
    models = tmp_path / "models"
    exported(models, name="m1", dim=2)            # the test reference set is 2-dimensional
    monkeypatch.setattr(finetune, "models_dir", lambda: models)
    monkeypatch.setenv("MV_EMBED_ONNX", "1")
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    client = TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                   backbone_loader=lambda n: Const(n), limits=open_limits(),
                                   background=False, fungus_gate=None, note=lambda s: None))
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    assert client.get("/api/health").json()["embed"]["m1"]["mode"] == "onnx-fp32"


def test_the_fp32_tower_gives_torchs_vectors_and_an_unknown_precision_is_refused(tmp_path):
    tiny, meta = exported(tmp_path)
    assert meta["check"]["min_cosine"] >= 0.99999 and meta["quantization"] is None
    with pytest.raises(ValueError, match="precision"):
        onnxtower.export(tiny, "m1", tmp_path, None, precision="fp8")
