"""Uploaded photos, kept no bigger than the models need.

A phone photo decoded at full size is 36 MB (12 MP) to 120 MB (40 MP, the most
guards.py lets in), and an identification used to hold every photo of the request
that way (up to MAX_PHOTOS) while each model ran. Now each photo is decoded once as
it is read: its EXIF place and date are read first (fill_context and the Identify
page use them, and shrinking drops the EXIF), it is turned upright, and it is shrunk
so its shorter side is SIDE_MARGIN times the largest input side of the loaded models
(448 px for BioCLIP 2's 224). Only that copy is kept.

The models resize a photo themselves (shorter side to their input size, then a
centre crop), so shrinking it first to twice that changes little: on 12 phone-sized
photos BioCLIP 2's vectors kept a cosine similarity of at least 0.9997 with those
from the full-size photo. A very large JPEG is decoded at a reduced scale (Pillow's
draft mode) only while that still leaves DRAFT_MARGIN times the kept side, which
left those vectors unchanged; decoding nearer the kept size moved them more.
"""

from __future__ import annotations

import io
import warnings

from PIL import Image, ImageOps

from .exif import place_and_date
from .guards import check_image_size

SIDE_MARGIN = 2          # kept shorter side = 2 x the model's input side
DRAFT_MARGIN = 4         # JPEG draft decoding only down to 4 x the kept side
# Before any model is loaded: the largest input side among the offered backbones
# (DINOv2 at 518 px), so no model ever gets less than it needs.
FALLBACK_MODEL_SIDE = 518


def model_side(backbone) -> int | None:
    """The side a backbone resizes photos to (its preprocessing's first Resize), or None
    when it doesn't say."""
    for t in getattr(getattr(backbone, "transform", None), "transforms", None) or []:
        if type(t).__name__ == "Resize":
            size = t.size
            return int(max(size) if isinstance(size, (tuple, list)) else size)
    return None


def kept_side(backbones) -> int:
    """The shorter side uploads are shrunk to, for these loaded backbones (one that
    doesn't say its size, or none loaded yet, counts as the largest we offer)."""
    sides = [model_side(b) or FALLBACK_MODEL_SIDE for b in backbones]
    return SIDE_MARGIN * max(sides, default=FALLBACK_MODEL_SIDE)


def shrink(img: Image.Image, side: int) -> Image.Image:
    """The image with its shorter side at most `side`, aspect kept (LANCZOS)."""
    w, h = img.size
    short = min(w, h)
    if short <= side:
        return img
    return img.resize((max(1, round(w * side / short)), max(1, round(h * side / short))),
                      Image.LANCZOS)


def decode_upload(body: bytes, side: int) -> tuple[Image.Image, tuple]:
    """(RGB image, upright, shorter side at most `side`; its EXIF (latitude, longitude,
    date)). Refuses an image too large in pixels before decoding it (TooLarge)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        img = Image.open(io.BytesIO(body))
    check_image_size(img)
    found = place_and_date(img)              # first: shrinking drops the EXIF
    if img.format == "JPEG":
        draft = side * DRAFT_MARGIN
        img.draft("RGB", (draft, draft))
    ImageOps.exif_transpose(img, in_place=True)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return shrink(img, side), found
