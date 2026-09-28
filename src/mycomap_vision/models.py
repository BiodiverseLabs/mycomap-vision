"""Backbones: the frozen image models that turn a photo into a vector.

Any timm or open_clip model can be used by name, with no code change:

    timm:<model name>               e.g. timm:vit_large_patch14_dinov2.lvd142m
    open_clip:<hub or model name>   e.g. open_clip:hf-hub:imageomics/bioclip-2

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
}


def storage_name(spec: str) -> str:
    """Filesystem- and URL-safe key for a backbone: its alias, or a slug of its spec."""
    if spec in ALIASES:
        return spec
    for name, alias in ALIASES.items():
        if alias.spec == spec:
            return name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec).strip("_")


def resolve_spec(name_or_spec: str) -> str:
    if name_or_spec in ALIASES:
        return ALIASES[name_or_spec].spec
    if name_or_spec.startswith(("timm:", "open_clip:")):
        return name_or_spec
    raise ValueError(f"unknown backbone {name_or_spec!r}: use an alias "
                     f"({', '.join(ALIASES)}) or timm:<name> / open_clip:<name>")


class TimmBackbone:
    """Any timm model: its pooled image feature (classifier removed)."""

    def __init__(self, name: str, timm_name: str):
        import timm
        import torch
        self.torch = torch
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = timm.create_model(timm_name, pretrained=True, num_classes=0).eval().to(
            self.device)
        cfg = timm.data.resolve_data_config({}, model=self.model)
        self.transform = timm.data.create_transform(**cfg)
        self.dim = self.model.num_features

    def prepare(self, image: Image.Image):
        return self.transform(image)

    def _forward(self, x):
        return self.model(x)

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


LOADERS: dict[str, Callable[[str, str], object]] = {
    "timm": TimmBackbone,
    "open_clip": OpenClipBackbone,
}


def load_backbone(name_or_spec: str):
    spec = resolve_spec(name_or_spec)
    kind, _, target = spec.partition(":")
    return LOADERS[kind](storage_name(name_or_spec), target)
