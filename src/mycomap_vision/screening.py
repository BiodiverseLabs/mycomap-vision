"""Screen several backbones in one go: embed each, then compare them all fairly.

A candidate that fails (weights not downloadable, out of GPU memory, ...) is
reported and skipped; the others still run. Only candidates that embedded every
photo join the comparison, so the shared photo set isn't shrunk by a partial run.
"""

from __future__ import annotations

import sqlite3
import time
import traceback
from dataclasses import dataclass, field
from typing import Callable

from . import config, evaluate, models
from .embed import embed_photos, photos_to_embed
from .storage import PhotoStore


@dataclass
class ScreenResult:
    embedded: dict[str, dict] = field(default_factory=dict)    # name -> {photos, per_second}
    failed: dict[str, str] = field(default_factory=dict)       # name -> reason
    compared: list[str] = field(default_factory=list)
    comparison: dict | None = None


def screen(conn: sqlite3.Connection, store: PhotoStore, candidates: list[str],
           baselines: list[str], size: str = "medium", methods: list[str] | None = None,
           batch_size: int = 16, loader: Callable[[str], object] = models.load_backbone,
           embeddings_root=None, compare: bool = True, readers: int = 8,
           after_embed: Callable[[str], None] | None = None, log=print) -> ScreenResult:
    """`after_embed(name)` runs once a candidate has embedded every photo (the AWS trainer
    uploads its shards there, so a later failure doesn't lose the hours spent)."""
    result = ScreenResult()
    for spec in candidates:
        name = models.storage_name(spec)
        root = (embeddings_root or config.DATA_DIR / "embeddings") / name
        try:
            todo = photos_to_embed(conn, name, size, store.location, str(config.DATA_DIR))
            log(f"[{name}] {len(todo):,} photos to embed")
            per_second = None
            if todo:
                started = time.monotonic()
                backbone = loader(spec)
                if backbone.name != name:
                    raise RuntimeError(f"loader named the backbone {backbone.name!r}, "
                                       f"expected {name!r}")
                stats = embed_photos(conn, store, backbone, todo, root, batch_size=batch_size,
                                     readers=readers,
                                     log=lambda s: log(f"[{name}] {s.strip()}"))
                per_second = round(stats.embedded / max(time.monotonic() - started, 1e-6), 1)
                del backbone
                _free_gpu()
            left = photos_to_embed(conn, name, size, store.location, str(config.DATA_DIR))
            if left:
                raise RuntimeError(f"{len(left):,} photos still not embedded (unreadable?)")
            result.embedded[name] = {"per_second": per_second}
            if after_embed is not None:
                after_embed(name)
            log(f"[{name}] done" + (f", {per_second}/s" if per_second else " (already embedded)"))
        except Exception as e:  # keep screening the rest
            result.failed[name] = f"{e.__class__.__name__}: {e}"
            log(f"[{name}] FAILED: {result.failed[name]}")
            log(traceback.format_exc(limit=3))
            _free_gpu()
    result.compared = [models.storage_name(b) for b in baselines] + [
        n for n in result.embedded if n not in {models.storage_name(b) for b in baselines}]
    if compare and len(result.compared) >= 1:
        result.comparison = evaluate.compare(conn, result.compared, methods or ["nearest"],
                                             embeddings_root=embeddings_root, log=log)
    return result


def _free_gpu() -> None:
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
