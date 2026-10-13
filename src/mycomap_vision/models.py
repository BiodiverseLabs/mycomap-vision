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

import gc
import re
import sys
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
    # BioCLIP 2's predecessor (ViT-B/16, 224 px): the ablation beside it in the screen.
    "bioclip": Alias("open_clip:hf-hub:imageomics/bioclip",
                     "BioCLIP (v1), tree-of-life CLIP, ViT-B/16, 224 px"),
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
    if spec.startswith(("finetuned:", "classifier:")):
        return spec.partition(":")[2]
    for name, alias in ALIASES.items():
        if alias.spec == spec:
            return name
    return re.sub(r"[^A-Za-z0-9._-]+", "_", spec).strip("_")


def resolve_spec(name_or_spec: str) -> str:
    if name_or_spec in ALIASES:
        return ALIASES[name_or_spec].spec
    if name_or_spec.startswith(("timm:", "open_clip:", "finetuned:", "classifier:")):
        return name_or_spec
    from .finetune import models_dir
    if re.fullmatch(r"[A-Za-z0-9._-]+", name_or_spec) and             (models_dir() / f"{name_or_spec}.json").is_file():
        from .replications.fungitastic.retrain import is_classifier
        kind = "classifier" if is_classifier(name_or_spec) else "finetuned"
        return f"{kind}:{name_or_spec}"
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


def release_memory() -> None:
    """Hand memory freed after loading back to the OS (glibc keeps it otherwise), so
    the server's footprint is what it holds, not what loading once needed."""
    gc.collect()
    if sys.platform.startswith("linux"):
        import ctypes
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


def open_clip_weights_file(model_name: str):
    """The local safetensors checkpoint of an open_clip `hf-hub:` model, or None (no such
    file, or not cached while offline)."""
    if not model_name.startswith("hf-hub:"):
        return None
    try:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(model_name[len("hf-hub:"):], "open_clip_model.safetensors")
    except Exception:
        return None


def load_open_clip_image_tower(model_name: str, pretrained: str | None):
    """(image tower, preprocess) of an open_clip model. Only the image tower is ever
    used (encode_image is the tower's output), so the text tower is never kept.

    For a Hugging Face model with a safetensors checkpoint, the model is laid out
    without memory ("meta") and only the image tower's weights are read into it:
    BioCLIP 2 then peaks at 1.55 GB instead of 3.2 GB (whole model plus a second
    copy while loading), measured on the 4 GB server box, with identical vectors."""
    import open_clip
    if model_name.startswith("hf-hub:") and not pretrained:
        path = open_clip_weights_file(model_name)
        if path:
            from safetensors import safe_open
            model, _, preprocess = open_clip.create_model_and_transforms(
                model_name, load_weights=False, pretrained_text=False, device="meta")
            tower = model.visual
            del model
            with safe_open(path, "pt") as f:
                weights = {k[len("visual."):]: f.get_tensor(k)
                           for k in f.keys() if k.startswith("visual.")}
            tower.load_state_dict(weights, strict=True, assign=True)
            del weights
            left = [n for n, t in list(tower.named_parameters()) + list(tower.named_buffers())
                    if t.device.type == "meta"]
            if left:
                raise RuntimeError(f"{model_name}: no weights for {left[:3]}")
            return tower, preprocess
    model, _, preprocess = open_clip.create_model_and_transforms(model_name,
                                                                 pretrained=pretrained)
    tower = model.visual
    del model
    return tower, preprocess


class OpenClipBackbone(TimmBackbone):
    """Any open_clip model: the image tower's embedding."""

    def __init__(self, name: str, clip_name: str):
        import torch
        self.torch = torch
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model_name, _, pretrained = clip_name.partition("@")
        tower, self.transform = load_open_clip_image_tower(model_name, pretrained or None)
        self.model = tower.eval().to(self.device)
        release_memory()
        with torch.inference_mode():
            probe = self.model(
                self.transform(Image.new("RGB", (64, 64))).unsqueeze(0).to(self.device))
        self.dim = probe.shape[1]

    def _forward(self, x):
        return self.model(x)          # = CLIP.encode_image (the tower, not normalised)

    def image_module(self):
        return self.model


def fork_point(base, meta: dict) -> int | None:
    """The first block fine-tuning trained, when `base` is an open_clip ViT whose earlier
    blocks are untouched (so its vectors and the base's share them); else None."""
    tower = getattr(base, "model", None)
    blocks = getattr(getattr(tower, "transformer", None), "resblocks", None)
    if blocks is None or not all(hasattr(tower, a) for a in ("_embeds", "_pool", "ln_post")):
        return None
    n_trained = int(meta.get("blocks_trained") or 0)
    if not 0 < n_trained <= len(blocks):
        return None
    return len(blocks) - n_trained


def tail_names(tower, start: int) -> list[str]:
    """The tower's parameters after the fork point: its last blocks, final norm, projection."""
    inside = (tuple(f"transformer.resblocks.{i}." for i in range(start, len(tower.transformer.resblocks)))
              + ("ln_post.",))
    return [n for n, _ in tower.named_parameters() if n.startswith(inside) or n == "proj"]


def _tail_module():
    import torch

    class Tail(torch.nn.Module):
        """The tower's blocks after the fork, then its pooling and projection."""

        def __init__(self, tower, start: int):
            super().__init__()
            self.tower, self.start = tower, start

        def forward(self, h):
            tower = self.tower
            for block in tower.transformer.resblocks[self.start:]:
                h = block(h)
            if not tower.transformer.batch_first:
                h = h.transpose(0, 1)
            pooled, _ = tower._pool(h)
            return pooled @ tower.proj if tower.proj is not None else pooled
    return Tail


class BaseTail:
    """The base model's own values of the parameters after the fork point, kept so the
    base vector (fungusgate.py) comes from the fine-tuned model's own pass: the shared
    blocks run once, then the tail runs twice, with trained and with base values.

    The base values are read from the base model's safetensors file when there is one:
    memory-mapped, file-backed pages the OS can drop under pressure (the server box runs
    close to its memory cap), not ~200 MB of private copies. Otherwise they are cloned
    from the tower before the fine-tuned weights go in."""

    def __init__(self, tower, start: int, trained: dict, weights_file=None, prefix: str = "visual."):
        names = tail_names(tower, start)
        outside = sorted(set(trained) - set(names))
        if outside:
            raise ValueError(f"trained parameters before the fork point: {outside[:3]}")
        self.start, self.file_backed = start, False
        own = dict(tower.named_parameters())
        if weights_file:
            from safetensors import safe_open
            with safe_open(str(weights_file), "pt") as f:
                keys = set(f.keys())
                if all(prefix + n in keys for n in names):
                    self.params = {n: f.get_tensor(prefix + n) for n in names}
                    self.file_backed = True
        if not self.file_backed:
            self.params = {n: own[n].detach().clone() for n in names}
        for n in names:
            if self.params[n].shape != own[n].shape:
                raise ValueError(f"base weight {n} has shape {tuple(self.params[n].shape)}, "
                                 f"the model {tuple(own[n].shape)}")
        self.module = _tail_module()(tower, start)

    def __call__(self, tower, h):
        from torch.func import functional_call
        device = next(tower.parameters()).device
        params = {f"tower.{n}": v.to(device) for n, v in self.params.items()}
        return functional_call(self.module, params, (h,), strict=False)


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
        # The base model's own values of the last blocks, pooling norm and projection, so
        # the base vector (fungusgate.py) comes from the same pass: the shared blocks run
        # once, then the last blocks run twice, trained and as they were. Read from the
        # base checkpoint on first use only, so a server without the gate pays nothing;
        # with no checkpoint file they must be cloned now, before the trained ones go in.
        start = fork_point(base, meta)
        try:
            spec = resolve_spec(meta["base_spec"])
        except ValueError:                             # a base known only to its loader
            spec = ""
        weights_file = (open_clip_weights_file(spec.partition(":")[2].partition("@")[0])
                        if spec.startswith("open_clip:") else None)
        self._base_tail = self._make_tail = None
        if start is not None:
            outside = sorted(set(trained) - set(tail_names(base.model, start)))
            if outside:
                raise ValueError(f"{model_name}: trained parameters before the fork point: "
                                 f"{outside[:3]}")
        if start is not None and weights_file:
            names = dict.fromkeys(trained)
            self._make_tail = lambda: BaseTail(base.model, start, names, weights_file)
        elif start is not None:
            self._base_tail = BaseTail(base.model, start, trained)
        with torch.no_grad():
            for n, value in trained.items():
                own[n].copy_(value.to(own[n].device, own[n].dtype))
        self.base, self.meta, self.name = base, meta, name
        self.dim, self.device, self.transform = base.dim, base.device, base.transform
        self.base_name = meta.get("base")

    def prepare(self, image):
        return self.base.prepare(image)

    def encode(self, items):
        return self.base.encode(items)

    def encode_with_base(self, items):
        """(fine-tuned vectors, the base model's vectors) for the same photos in one pass,
        or None when the base isn't an open_clip ViT this can fork."""
        if self._base_tail is None and self._make_tail is not None:
            self._base_tail, self._make_tail = self._make_tail(), None
        if self._base_tail is None:
            return None
        torch = self.base.torch
        tower, start = self.base.model, self._base_tail.start
        x = torch.stack([i if isinstance(i, torch.Tensor) else self.transform(i)
                         for i in items]).to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16,
                                                    enabled=self.device == "cuda"):
            h = tower._embeds(x)
            if not tower.transformer.batch_first:
                h = h.transpose(0, 1).contiguous()
            for block in tower.transformer.resblocks[:start]:
                h = block(h)
            tuned = self._base_tail.module(h)          # the tail as trained
            base = self._base_tail(tower, h)           # the same tail, base weights
        return tuned.float().cpu().numpy(), base.float().cpu().numpy()

    def image_module(self):
        return self.base.image_module()

    def forward_features(self, x):
        return self.base.forward_features(x)


def _classifier_backbone(name: str, model_name: str):
    """A classifier trained by mv picek-train (picek.ClassifierBackbone)."""
    from .replications.fungitastic.retrain import ClassifierBackbone
    return ClassifierBackbone(name, model_name)


LOADERS: dict[str, Callable[[str, str], object]] = {
    "timm": TimmBackbone,
    "open_clip": OpenClipBackbone,
    "finetuned": FinetunedBackbone,
    "classifier": lambda name, model_name: _classifier_backbone(name, model_name),
}


def load_backbone(name_or_spec: str):
    spec = resolve_spec(name_or_spec)
    kind, _, target = spec.partition(":")
    return LOADERS[kind](storage_name(name_or_spec), target)
