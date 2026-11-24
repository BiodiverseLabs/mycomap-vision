"""Limits for the public API: request size, image size, request rate and GPU queue.

All are settings (see .env.example) so a deployment can tune them without code.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Callable

from PIL import Image

from . import config, models

MAX_PHOTOS = 10
MAX_FILE_BYTES = 25 * 1024 * 1024
MAX_REQUEST_BYTES = 80 * 1024 * 1024
# Pixel limits, checked from the declared size before anything is decoded. A JPEG is
# decoded at a reduced scale (uploads.decode_upload, dhash.dhash64: JPEG draft), so a
# 48-200 MP phone photo costs little memory; any other format is decoded whole. The old
# single 40 MP limit refused 48 MP phone photos (three 413s on 2026-10-23).
MAX_PIXELS = 60_000_000           # PNG, WebP, ...: decoded whole
MAX_JPEG_PIXELS = 210_000_000     # JPEG: the 200 MP phones, decoded in draft

# Pillow refuses anything past twice this on its own; keep that backstop in line.
Image.MAX_IMAGE_PIXELS = MAX_JPEG_PIXELS

# The only decoders a photo from outside (an upload, iNat, Mushroom Observer) ever reaches:
# Image.open(..., formats=PHOTO_FORMATS). Pillow has dozens more (TIFF, FPX, MIC, PDF, ...),
# where most of its advisories live (41 against 12.3.0, October 2026); none is needed for a
# photo. MPO is how Pillow names many phone JPEGs; HEIF comes from pillow-heif.
PHOTO_FORMATS = ("JPEG", "MPO", "PNG", "WEBP", "HEIF", "AVIF")
JPEG_FAMILY = ("JPEG", "MPO")      # decoded in draft (reduced scale)


def photo_formats() -> tuple[str, ...]:
    """PHOTO_FORMATS this Pillow can open, for Image.open(formats=...): every plugin
    registered first (Pillow raises KeyError on a format it hasn't loaded yet), and one
    that isn't installed (HEIF without pillow-heif, AVIF on an old Pillow) left out."""
    Image.init()
    return tuple(f for f in PHOTO_FORMATS if f in Image.OPEN)


class TooLarge(ValueError):
    pass


def check_image_size(img: Image.Image, max_pixels: int | None = None) -> None:
    """Refuse an image by its declared size, before its pixels are decoded: past
    MAX_JPEG_PIXELS for a JPEG (decoded in draft), past MAX_PIXELS for anything else."""
    if max_pixels is None:
        max_pixels = MAX_JPEG_PIXELS if img.format in JPEG_FAMILY else MAX_PIXELS
    w, h = img.size
    if w * h > max_pixels:
        raise TooLarge(f"{w}x{h} pixels is over the {max_pixels // 1_000_000} MP limit")


def parse_rate(spec: str | None, default: tuple[int, float] = (30, 600.0)) -> tuple[int, float]:
    """'30/600' -> (30 requests, 600 seconds)."""
    if not spec:
        return default
    n, _, window = spec.partition("/")
    return int(n), float(window or 60)


class RateLimiter:
    """At most `limit` requests per key in any sliding `window` seconds."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque[float]] = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key: str) -> float:
        """0 if allowed (and counted), else seconds until the next request is allowed."""
        with self.lock:
            now = self.clock()
            q = self.hits[key]
            while q and q[0] <= now - self.window:
                q.popleft()
            if len(q) >= self.limit:
                return q[0] + self.window - now
            q.append(now)
            if len(self.hits) > 10_000:            # forget idle clients
                for k in [k for k, v in self.hits.items() if not v]:
                    del self.hits[k]
            return 0.0


class Gate:
    """Counts identify requests in flight; refuses when more than `limit` would be."""

    def __init__(self, limit: int):
        self.limit, self.active = limit, 0
        self.lock = threading.Lock()

    def enter(self) -> bool:
        with self.lock:
            if self.active >= self.limit:
                return False
            self.active += 1
            return True

    def leave(self) -> None:
        with self.lock:
            self.active -= 1


def _names(value: str | None) -> set[str] | None:
    """A comma list of names as a set; None when unset or empty."""
    names = {v.strip() for v in (value or "").split(",") if v.strip()}
    return names or None


# A default method that can't answer here falls back to this one before nearest.
# Each served method's fallback when this server can't answer it (not allowed, or not ready:
# the place store or the twin groups missing). Steve, 2026-10-18 (E32): nearest+mean-records-
# balanced(+place) serves, so it falls back to it, then to the photo-level methods it replaced.
DEFAULT_FALLBACK = {"nearest+mean-records-balanced+place": "nearest+mean-records-balanced",
                    "nearest+mean-records-balanced": "nearest+mean+place",
                    "nearest+mean+place": "nearest+mean"}


# The callers that get the site's default model (MV_SITE_DEFAULT_BACKBONE) when they name
# none: the website itself, and API v1 with an app token (apiv1.Credential.kind "app": the
# mobile app). mycomap.org's server key ("server") and developer keys ("key") keep
# MV_DEFAULT_BACKBONE.
SITE_KINDS = ("site", "app")


@dataclass
class Limits:
    rate: tuple[int, float]
    max_in_flight: int
    allowed_backbones: set[str]
    # None = every method. The server box lists only what it can answer in seconds.
    allowed_methods: set[str] | None = None
    # False = never train a method's classifier on this machine (minutes of CPU on the
    # box; hours at the full photo set). A trained method then needs its saved training.
    fit_on_demand: bool = True
    # The method an identification uses when none is named (MV_DEFAULT_METHOD). Steve,
    # 2026-10-09: nearest + species average. Falls back to nearest where not offered.
    default_method: str = "nearest+mean"
    # Most models one identification may run (None = no limit). The box: 2, so a
    # side-by-side comparison can't outlast the proxy's timeout.
    max_models: int | None = None
    # The backbone an identification uses when none is named (MV_DEFAULT_BACKBONE), on every
    # path: the site, API v1, the app's sessions, by-observation and its kept answers, the
    # nightly advance predictions, taxon counts, Human Feedback's re-asks and excluded maps;
    # the site and the app only until site_default_backbone is set.
    # When it is named but not served here, those are refused, never answered by another
    # model (default_of). None: the first ready backbone by name, as before (deprecated on a
    # server with more than one model: names decide the default).
    default_backbone: str | None = None
    # The backbones API v1 (/api/v1/...: mycomap.org, the app, developer keys) answers with
    # (MV_API_V1_BACKBONES). None: the default backbone only. A request naming another is
    # refused (400 model_not_offered), so the v2 beta stays on the site.
    api_v1_backbones: set[str] | None = None
    # The model the site's visitors and the app get when they name none
    # (MV_SITE_DEFAULT_BACKBONE; Steve, 2026-11-24: "lets go with v2" for site visitors, and
    # "mobile app should move with the site", while mycomap.org stays on v1 until its own
    # switch): the Identify page's first choice and the site's identifications that name no
    # model (uploads and "from a link"), and API v1 calls made with an app token (the app's
    # sessions and identify; SITE_KINDS). An app token may also use it through API v1 when
    # MV_API_V1_BACKBONES doesn't list it. Every other path keeps default_backbone. Refused,
    # never replaced by another model, while it isn't served. None: default_backbone.
    site_default_backbone: str | None = None

    @classmethod
    def from_settings(cls) -> "Limits":
        allowed = config.setting("MV_ALLOWED_BACKBONES")
        methods = config.setting("MV_METHODS")
        return cls(
            rate=parse_rate(config.setting("MV_RATE_LIMIT")),
            max_in_flight=int(config.setting("MV_MAX_QUEUE", "4")),
            allowed_backbones=({b.strip() for b in allowed.split(",") if b.strip()}
                               if allowed else set(models.ALIASES)),
            allowed_methods=({m.strip() for m in methods.split(",") if m.strip()}
                             if methods else None),
            fit_on_demand=(config.setting("MV_FIT_ON_DEMAND", "1").lower()
                           not in ("0", "false", "no", "off")),
            max_models=(int(config.setting("MV_MAX_MODELS")) if (config.setting("MV_MAX_MODELS") or "").isdigit()
                        else None),
            default_method=(config.setting("MV_DEFAULT_METHOD") or "nearest+mean").strip(),
            default_backbone=(config.setting("MV_DEFAULT_BACKBONE") or "").strip() or None,
            api_v1_backbones=_names(config.setting("MV_API_V1_BACKBONES")),
            site_default_backbone=(config.setting("MV_SITE_DEFAULT_BACKBONE") or "").strip()
            or None,
        )

    def default_of(self, ready: "list[str]") -> str | None:
        """The default backbone among `ready` (the backbones offered now, in name order):
        MV_DEFAULT_BACKBONE when it is one of them, else None (refused: never another model
        in its place); unset, the first ready one, as before."""
        if self.default_backbone:
            return self.default_backbone if self.default_backbone in ready else None
        return ready[0] if ready else None

    def named_default_for(self, kind: str) -> str | None:
        """The model a caller of `kind` gets when it names none, as set: for the site and an
        app token (SITE_KINDS) MV_SITE_DEFAULT_BACKBONE when set; otherwise, and for
        mycomap.org's server key and developer keys always, MV_DEFAULT_BACKBONE (None: unset,
        name order decides)."""
        if kind in SITE_KINDS and self.site_default_backbone:
            return self.site_default_backbone
        return self.default_backbone

    def setting_for(self, kind: str) -> str:
        """The setting that names `kind`'s default (for refusals and the log)."""
        return ("MV_SITE_DEFAULT_BACKBONE" if kind in SITE_KINDS and self.site_default_backbone
                else "MV_DEFAULT_BACKBONE")

    def default_for(self, kind: str, ready: "list[str]") -> str | None:
        """default_of for a caller of `kind`: its named default (named_default_for) when it
        is ready, else None (refused: never another model); with neither setting, the first
        ready backbone, as before."""
        if kind in SITE_KINDS and self.site_default_backbone:
            return self.site_default_backbone if self.site_default_backbone in ready else None
        return self.default_of(ready)

    def default_conflicts(self) -> list[str]:
        """Settings under which some caller's default model is one that caller may not use
        (DEFAULT-SWITCH-PLAN trap 1: MV_DEFAULT_BACKBONE=v2 with MV_API_V1_BACKBONES=v1
        refused every call of mycomap.org's that names no model). Each line says which
        callers are refused. Said at startup ("[default] REFUSED") and in the server-key
        /api/health (`default_backbone.conflicts`); those callers' no-model calls are
        refused (503), never answered by another model. Empty: none."""
        out = []
        d, s = self.default_backbone, self.site_default_backbone
        if d and d not in self.allowed_backbones:
            out.append(f"MV_DEFAULT_BACKBONE={d} is not in MV_ALLOWED_BACKBONES: "
                       + ("mycomap.org's server key, developer keys and the nightly's paths "
                          "that name no model are refused" if s
                          else "every path that names no model is refused"))
        if d and self.api_v1_backbones is not None and d not in self.api_v1_backbones:
            out.append(f"MV_DEFAULT_BACKBONE={d} is not in MV_API_V1_BACKBONES: mycomap.org's "
                       "server key and developer keys are refused (503 not_served) on every "
                       "API v1 call that names no model")
        if s and s not in self.allowed_backbones:
            out.append(f"MV_SITE_DEFAULT_BACKBONE={s} is not in MV_ALLOWED_BACKBONES: the "
                       "site's and the app's identifications that name no model are refused")
        if s and not d:
            out.append("MV_SITE_DEFAULT_BACKBONE is set but MV_DEFAULT_BACKBONE is not: "
                       "mycomap.org's default would be decided by name order; set both")
        return out

    def method_allowed(self, method: str) -> bool:
        return self.allowed_methods is None or method in self.allowed_methods

    def served_default(self, ready: "Callable[[str], bool]" = lambda m: True) -> str:
        """The default method, if this server offers it and can answer it; else its
        fallback (DEFAULT_FALLBACK: nearest+mean+place without its occurrence store falls
        back to nearest+mean, its own answer without place and date); else nearest."""
        m = self.default_method
        while m:
            if self.method_allowed(m) and ready(m):
                return m
            m = DEFAULT_FALLBACK.get(m)
        return "nearest"
