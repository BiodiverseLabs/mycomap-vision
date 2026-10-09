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
MAX_PIXELS = 40_000_000           # 40 MP: bigger than any phone photo, far below a bomb

# Pillow refuses anything past twice this on its own; keep that backstop in line.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class TooLarge(ValueError):
    pass


def check_image_size(img: Image.Image, max_pixels: int = MAX_PIXELS) -> None:
    """Refuse an image by its declared size, before its pixels are decoded."""
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
        )

    def method_allowed(self, method: str) -> bool:
        return self.allowed_methods is None or method in self.allowed_methods

    def served_default(self, ready: "Callable[[str], bool]" = lambda m: True) -> str:
        """The default method, if this server offers it and can answer it; else nearest."""
        m = self.default_method
        return m if self.method_allowed(m) and ready(m) else "nearest"
