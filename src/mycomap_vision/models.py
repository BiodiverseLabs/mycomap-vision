"""Backbones: the frozen image models that turn a photo into a vector.

Any timm or open_clip model can be used by name, with no code change:

    timm:<model name>               e.g. timm:vit_large_patch14_dinov2.lvd142m
    open_clip:<hub or model name>   e.g. open_clip:hf-hub:imageomics/bioclip-2

A timm spec can carry model options after "?", e.g.
timm:vit_large_patch16_dinov3.lvd1689m?img_size=512&global_pool=token
(input size and pooling); the photos are then resized to that input size too.

Short aliases below name the ones we use often. The storage name of a backbone
(its embeddings folder and scoreboard key) is the alias, or a filesystem-safe
form of the spec.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from PIL import Image


@dataclass(frozen=True)
class Alias:
    spec: str
    note: str


ALIASES: dict[str, Alias] = {
    "bioclip-2": Alias("open_clip:hf-hub:imageomics/bioclip-2",
                       "Tree-of-life CLIP (includes fungi), 224 px"),
    "dinov2-b14": Alias("timm:vit_base_patch14_dinov2.lvd142m",
                        "Self-supervised, fine detail, 518 px, base size"),
    "dinov2-l14": Alias("timm:vit_large_patch14_dinov2.lvd142m",
                        "Self-supervised, fine detail, 518 px, large size"),
    # Screening candidates (docs/PLAN.md). Names checked against timm/open_clip 2026-09-28.
    "dinov3-b16": Alias("timm:vit_base_patch16_dinov3.lvd1689m",
                        "DINOv3 self-supervised, base size (Meta DINOv3 licence)"),
    "dinov3-l16": Alias("timm:vit_large_patch16_dinov3.lvd1689m",
                        "DINOv3 self-supervised, large size (Meta DINOv3 licence)"),
    # timm's DINOv3 default is 256 px with averaged patch tokens, which lost badly to
    # DINOv2 at 518 px on its class token (docs/PLAN.md); these match DINOv2's setup.
    "dinov3-b16-512": Alias("timm:vit_base_patch16_dinov3.lvd1689m?img_size=512&global_pool=token",
                            "DINOv3 base at 512 px on the class token"),
    "dinov3-l16-512": Alias("timm:vit_large_patch16_dinov3.lvd1689m?img_size=512&global_pool=token",
                            "DINOv3 large at 512 px on the class token"),
    "siglip2-l16-384": Alias("open_clip:ViT-L-16-SigLIP2-384@webli",
                             "SigLIP 2 image-text model, large, 384 px"),
    "eva02-l14-448": Alias("timm:eva02_large_patch14_448.mim_m38m_ft_in22k",
                           "EVA-02 large, ImageNet-22k fine-tune, 448 px"),
    "convnextv2-l": Alias("timm:convnextv2_large.fcmae_ft_in22k_in1k_384",
                          "ConvNeXt V2 large (a convolutional net, for contrast), 384 px"),
}


def storage_name(spec: str) -> str:
    """Filesystem- and URL-safe key for a backbone: its alias, or a slug of its spec."""
    if spec in ALIASES:
        return spec
    if spec.startswith("finetuned:"):
        return spec.partition(":")[2]
    for name, alias in ALIASES.items():
        if alias.spec == spec:
            return name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec).strip("_")


def resolve_spec(name_or_spec: str) -> str:
    if name_or_spec in ALIASES:
        return ALIASES[name_or_spec].spec
    if name_or_spec.startswith(("timm:", "open_clip:", "finetuned:")):
        return name_or_spec
    from .finetune import models_dir
    if re.fullmatch(r"[A-Za-z0-9._-]+", name_or_spec) and             (models_dir() / f"{name_or_spec}.json").is_file():
        return f"finetuned:{name_or_spec}"
    raise ValueError(f"unknown backbone {name_or_spec!r}: use an alias "
                     f"({', '.join(ALIASES)}), a fine-tuned model in data/models, "
                     "or timm:<name> / open_clip:<name>")


def split_options(target: str) -> tuple[str, dict]:
    """'model?img_size=512&global_pool=token' -> ('model', {'img_size': 512, ...})."""
    name, _, query = target.partition("?")
    options = {}
    for part in filter(None, query.split("&")):
        key, sep, value = part.partition("=")
        if not sep or not key:
            raise ValueError(f"bad model option {part!r} in {target!r}: use key=value")
        options[key] = int(value) if value.isdigit() else value
    return name, options


def data_config(cfg: dict, options: dict) -> dict:
    """timm's preprocessing config follows the pretrained size, not an img_size option;
    without this the photo would be shrunk to the old size before the model sees it."""
    size = options.get("img_size")
    if size is None:
        return cfg
    h, w = (size, size) if isinstance(size, int) else size
    return {**cfg, "input_size": (cfg["input_size"][0], h, w)}


class TimmBackbone:
    """Any timm model: its pooled image feature (classifier removed)."""

    def __init__(self, name: str, timm_name: str):
        import timm
        import torch
        self.torch = torch
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        timm_name, options = split_options(timm_name)
        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0,
                                       **options).eval().to(self.device)
        cfg = data_config(timm.data.resolve_data_config({}, model=self.model), options)
        self.transform = timm.data.create_transform(**cfg)
        self.dim = self.model.num_features

    def prepare(self, image: Image.Image):
        return self.transform(image)

    def _forward(self, x):
        return self.model(x)

    def image_module(self):
        """The image network, whose parameters fine-tuning trains."""
        return self.model

    def forward_features(self, x):
        """Embeddings with gradients, for training (encode() is inference only)."""
        return self._forward(x)

    def encode(self, items):
        torch = self.torch
        x = torch.stack([i if isinstance(i, torch.Tensor) else self.transform(i)
                         for i in items]).to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16,
                                                    enabled=self.device == "cuda"):
            return self._forward(x).float().cpu().numpy()


class OpenClipBackbone(TimmBackbone):
    """Any open_clip model: the image tower's embedding."""

    def __init__(self, name: str, clip_name: str):
        import open_clip
        import torch
        self.torch = torch
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name, _, pretrained = clip_name.partition("@")
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained or None)
        self.model = model.eval().to(self.device)
        self.transform = preprocess
        with torch.inference_mode():
            probe = self.model.encode_image(
                self.transform(Image.new("RGB", (64, 64))).unsqueeze(0).to(self.device))
        self.dim = probe.shape[1]

    def _forward(self, x):
        return self.model.encode_image(x)

    def image_module(self):
        return self.model.visual


class FinetunedBackbone:
    """A base backbone with the weights fine-tuning trained put back in (finetune.py)."""

    def __init__(self, name: str, model_name: str, root=None):
        import torch

        from .finetune import read_meta, models_dir
        root = root or models_dir()
        meta = read_meta(model_name, root)
        base = load_backbone(meta["base_spec"])
        trained = torch.load(root / meta["weights"], map_location="cpu", weights_only=True)
        module = base.image_module()
        own = dict(module.named_parameters())
        unknown = sorted(set(trained) - set(own))
        if unknown:
            raise ValueError(f"{model_name}: weights for parameters {base.name} doesn't have: "
                             f"{unknown[:3]}")
        with torch.no_grad():
            for n, value in trained.items():
                own[n].copy_(value.to(own[n].device, own[n].dtype))
        self.base, self.meta, self.name = base, meta, name
        self.dim, self.device, self.transform = base.dim, base.device, base.transform

    def prepare(self, image):
        return self.base.prepare(image)

    def encode(self, items):
        return self.base.encode(items)

    def image_module(self):
        return self.base.image_module()

    def forward_features(self, x):
        return self.base.forward_features(x)


LOADERS: dict[str, Callable[[str, str], object]] = {
    "timm": TimmBackbone,
    "open_clip": OpenClipBackbone,
    "finetuned": FinetunedBackbone,
}


def load_backbone(name_or_spec: str):
    spec = resolve_spec(name_or_spec)
    kind, _, target = spec.partition(":")
    return LOADERS[kind](storage_name(name_or_spec), target)
