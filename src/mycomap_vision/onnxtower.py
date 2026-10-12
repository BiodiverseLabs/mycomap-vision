"""The image tower as an ONNX model (docs/capacity.md option D; MV_EMBED_ONNX=1).

Embedding a photo costs ~1.6 s on the box's CPU in torch (float32 BioCLIP 2, ViT-L/14).
`mv export-onnx --backbone <b>` (laptop, at release time) exports the served model's image
tower to ONNX and writes, beside the model's own files (release paths agreed with the
retrain lane):

  models/<b>.onnx        the tower: a preprocessed 224x224 float tensor in, the
                         backbone's vector (before normalising) out, as encode() gives
  models/<b>.onnx.json   where it came from and how it was checked: source_weights
                         (the .pt's sha256), the preprocessing, and the check figures

Precision (measured 2026-10-12 on the served fine-tuned BioCLIP 2, 24 held-out photos, two
laptop threads): float32 ONNX gives torch's vectors exactly (cosine 1.0000) in 1.65 s a
photo against torch's 3.60 s on the same CPU, so it is the default. int8 (onnxruntime dynamic
quantization) takes 0.66 s but moves vectors too far (cosine down to 0.89, mean 0.95; keeping
attention in float32 doesn't help, the MLPs drift), so `--precision int8` exists only to be
measured again on a later model; the check below refuses it today.

It refuses to write unless every check image's vector matches torch's to cosine >= MIN_COSINE.
On the server, MV_EMBED_ONNX=1 loads the .onnx instead of the torch model: the preprocessing
is rebuilt from the .json (no torch weights are loaded), when the .json's source_weights
match the served .pt. Anything missing or mismatched falls back to torch, and /api/health says
why (STATE).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path

import numpy as np
from PIL import Image

MIN_COSINE = 0.99
CHECK_IMAGES = 64
SUFFIX, META_SUFFIX = ".onnx", ".onnx.json"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def check_images(n: int = CHECK_IMAGES, seed: int = 0) -> list[Image.Image]:
    """Fixed, varied test photos (colour fields, gradients, noise, shapes, odd sizes)."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        h, w = int(rng.integers(160, 640)), int(rng.integers(160, 640))
        kind = i % 4
        if kind == 0:
            a = np.broadcast_to(rng.integers(0, 256, 3, dtype=np.uint8), (h, w, 3))
        elif kind == 1:
            y = np.linspace(0, 1, h)[:, None, None]
            x = np.linspace(0, 1, w)[None, :, None]
            a = (255 * (y * rng.random(3) + x * rng.random(3)) / 2).astype(np.uint8)
            a = np.broadcast_to(a, (h, w, 3))
        elif kind == 2:
            a = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        else:
            a = np.full((h, w, 3), rng.integers(0, 256, 3), dtype=np.uint8)
            cy, cx, r = h // 2, w // 2, min(h, w) // 3
            yy, xx = np.ogrid[:h, :w]
            a[(yy - cy) ** 2 + (xx - cx) ** 2 < r * r] = rng.integers(0, 256, 3)
        out.append(Image.fromarray(np.ascontiguousarray(a), "RGB"))
    return out


def preprocessing_of(transform) -> dict:
    """The resize/crop/normalise a torchvision transform does, as data for the .json."""
    steps = {type(t).__name__: t for t in getattr(transform, "transforms", [])}
    resize, norm = steps.get("Resize"), steps.get("Normalize")
    crop = steps.get("CenterCrop")
    if resize is None or norm is None:
        raise ValueError("the model's preprocessing has no Resize or Normalize step")
    size = resize.size if isinstance(resize.size, int) else max(resize.size)
    crop_size = (crop.size[0] if isinstance(crop.size, (tuple, list)) else crop.size) if crop else size
    return {"resize": int(size), "crop": int(crop_size),
            "interpolation": str(getattr(resize, "interpolation", "bicubic")).split(".")[-1].lower(),
            "mean": [float(m) for m in norm.mean], "std": [float(s) for s in norm.std]}


def transform_from(pre: dict):
    """The preprocessing back from the .json, without loading the model."""
    from torchvision import transforms as T
    interp = getattr(T.InterpolationMode, pre.get("interpolation", "bicubic").upper(),
                     T.InterpolationMode.BICUBIC)
    return T.Compose([T.Resize(pre["resize"], interpolation=interp), T.CenterCrop(pre["crop"]),
                      lambda im: im.convert("RGB"), T.ToTensor(),
                      T.Normalize(pre["mean"], pre["std"])])


PRECISIONS = ("fp32", "int8")


def export(backbone, name: str, out_dir: Path, source_weights: Path | None,
           check: list | None = None, opset: int = 17, precision: str = "fp32",
           note=print) -> dict:
    """Write models/<name>.onnx (+ .onnx.json) for a loaded torch backbone (encode() is
    the reference). Refuses (ValueError) unless every check image matches."""
    if precision not in PRECISIONS:
        raise ValueError(f"precision is one of {PRECISIONS}, not {precision!r}")
    import torch
    from onnxruntime.quantization import QuantType, quantize_dynamic
    module = backbone.image_module().eval().float().to("cpu")
    pre = preprocessing_of(backbone.transform)
    rebuilt = transform_from(pre)
    images = check if check is not None else check_images()
    batch = torch.stack([backbone.transform(im) for im in images])
    if float((torch.stack([rebuilt(im) for im in images]) - batch).abs().max()) > 1e-4:
        raise ValueError("the preprocessing rebuilt from the .json differs from the model's")
    with torch.no_grad():
        reference = module(batch).float().numpy()
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fp32 = Path(tmp) / "fp32.onnx"
        torch.onnx.export(module, batch[:1], str(fp32), input_names=["pixels"],
                          output_names=["vector"], opset_version=opset, do_constant_folding=True,
                          dynamic_axes={"pixels": {0: "n"}, "vector": {0: "n"}}, dynamo=False)
        made = fp32
        if precision == "int8":
            made = Path(tmp) / "int8.onnx"
            quantize_dynamic(str(fp32), str(made), weight_type=QuantType.QInt8, per_channel=True)
        got = OnnxSession(made).run(batch.numpy())
        cos = np.sum(got * reference, axis=1) / (np.linalg.norm(got, axis=1)
                                                 * np.linalg.norm(reference, axis=1))
        meta = {"name": name, "format": "onnx", "precision": precision,
                "quantization": "dynamic-int8 (per-channel)" if precision == "int8" else None,
                "opset": opset, "input": {"shape": [None, *batch.shape[1:]], **pre},
                "output_dim": int(reference.shape[1]),
                "source_weights": ({"file": source_weights.name, "sha256": sha256_of(source_weights)}
                                   if source_weights else None),
                "check": {"images": len(images), "min_cosine": round(float(cos.min()), 5),
                          "mean_cosine": round(float(cos.mean()), 5), "required": MIN_COSINE}}
        if cos.min() < MIN_COSINE:
            raise ValueError(f"{precision} tower differs from torch: min cosine "
                             f"{cos.min():.4f} < {MIN_COSINE} (nothing written)")
        target = out_dir / f"{name}{SUFFIX}"
        os.replace(made, target)
    meta["sha256"] = sha256_of(target)
    (out_dir / f"{name}{META_SUFFIX}").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    note(f"[onnx] {target} ({target.stat().st_size / 2**20:.0f} MB), check min cosine "
         f"{meta['check']['min_cosine']}")
    return meta


class OnnxSession:
    def __init__(self, path: Path, threads: int | None = None):
        import onnxruntime as rt
        opts = rt.SessionOptions()
        opts.intra_op_num_threads = threads or int(os.environ.get("OMP_NUM_THREADS") or 0)
        self.session = rt.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def run(self, pixels: np.ndarray) -> np.ndarray:
        return self.session.run(None, {self.input: np.ascontiguousarray(pixels, np.float32)})[0]


class OnnxBackbone:
    """A served backbone answered by its ONNX tower: prepare() and encode() as the
    torch backbone's, `transform` for uploads.kept_side."""

    def __init__(self, name: str, onnx_path: Path, meta: dict):
        self.name, self.meta = name, meta
        self.transform = transform_from(meta["input"])
        self.session = OnnxSession(onnx_path)
        self.dim = int(meta["output_dim"])
        self.device = "cpu"

    def prepare(self, image):
        return self.transform(image)

    def encode(self, items):
        import torch
        x = torch.stack([i if isinstance(i, torch.Tensor) else self.transform(i) for i in items])
        return self.session.run(x.numpy())


class State:
    """For /api/health: how photos are embedded, and why ONNX isn't used when it isn't."""

    def __init__(self):
        self.lock = threading.Lock()
        self.by_backbone: dict[str, dict] = {}

    def set(self, backbone: str, mode: str, reason: str | None = None) -> None:
        with self.lock:
            self.by_backbone[backbone] = {"mode": mode, "reason": reason}

    def state(self) -> dict:
        with self.lock:
            return dict(self.by_backbone)


STATE = State()


def source_weights_of(name: str, models_dir: Path) -> Path | None:
    """The .pt a fine-tuned backbone's weights come from (None for a frozen one)."""
    meta_file = models_dir / f"{name}.json"
    if not meta_file.is_file():
        return None
    weights = json.loads(meta_file.read_text(encoding="utf-8")).get("weights")
    return models_dir / weights if weights else None


def loader(torch_loader, models_dir: Path, note=print):
    """A backbone loader that answers with the ONNX tower when its files are there and
    match the served weights, else with `torch_loader`."""
    def load(name: str):
        onnx_path = models_dir / f"{name}{SUFFIX}"
        meta_path = models_dir / f"{name}{META_SUFFIX}"
        reason = None
        try:
            if not (onnx_path.is_file() and meta_path.is_file()):
                reason = f"no {onnx_path.name} and {meta_path.name} in the release"
            else:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                weights = source_weights_of(name, models_dir)
                recorded = (meta.get("source_weights") or {}).get("sha256")
                if weights is not None and (not weights.is_file()
                                            or sha256_of(weights) != recorded):
                    reason = f"{meta_path.name} was exported from other weights than {weights.name}"
                elif meta.get("sha256") and sha256_of(onnx_path) != meta["sha256"]:
                    reason = f"{onnx_path.name} doesn't match its {meta_path.name}"
                else:
                    backbone = OnnxBackbone(name, onnx_path, meta)
                    precision = meta.get("precision", "int8")
                    STATE.set(name, f"onnx-{precision}")
                    note(f"[onnx] {name}: embedding with the ONNX tower ({precision})")
                    return backbone
        except Exception as e:  # noqa: BLE001 - torch answers instead
            reason = f"{type(e).__name__}: {e}"
        STATE.set(name, "torch", reason)
        note(f"[onnx] {name}: torch ({reason})")
        return torch_loader(name)
    return load
