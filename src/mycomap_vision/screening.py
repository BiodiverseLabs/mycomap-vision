"""Screen several backbones in one go: embed each, then compare them all fairly.

A candidate that fails (weights not downloadable, out of GPU memory, ...) is
reported and skipped; the others still run. Only candidates that embedded every
readable photo join the comparison, so the shared photo set isn't shrunk by a
partial run. A few unreadable photos are skipped and listed; too many fail the
candidate (SKIP_THRESHOLD).
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


# A few unreadable photos (corrupt, truncated, gone from the store) are skipped; more
# than this share of a candidate's photos points at something systemic (a broken
# store, a wrong size, a decoder problem), and the candidate fails instead.
SKIP_THRESHOLD = 0.01


class Stopped(Exception):
    """should_stop() ended a candidate's embedding before every photo was done."""


@dataclass
class ScreenResult:
    # name -> {per_second, photos, skipped, seconds}
    embedded: dict[str, dict] = field(default_factory=dict)
    failed: dict[str, str] = field(default_factory=dict)       # name -> reason
    stopped: dict[str, str] = field(default_factory=dict)      # name -> how far it got
    skipped: dict[str, list] = field(default_factory=dict)     # name -> [(photo_id, why)]
    compared: list[str] = field(default_factory=list)
    comparison: dict | None = None


def screen(conn: sqlite3.Connection, store: PhotoStore, candidates: list[str],
           baselines: list[str], size: str = "medium", methods: list[str] | None = None,
           batch_size: int = 16, loader: Callable[[str], object] = models.load_backbone,
           embeddings_root=None, compare: bool = True, readers: int = 8,
           after_embed: Callable[[str], None] | None = None,
           should_stop: Callable[[], bool] | None = None,
           skip_threshold: float | None = None, log=print) -> ScreenResult:
    """`after_embed(name)` runs once a candidate has embedded every readable photo (the
    AWS trainer uploads its shards there, so a later failure doesn't lose the hours
    spent). Unreadable photos are skipped and listed in result.skipped; a candidate
    fails when they are more than `skip_threshold` of its photos. `should_stop()` is
    asked between batches: a candidate it stops lands in result.stopped, never in
    result.embedded, and no further candidate starts."""
    result = ScreenResult()
    if skip_threshold is None:
        skip_threshold = SKIP_THRESHOLD
    for spec in candidates:
        name = models.storage_name(spec)
        root = (embeddings_root or config.DATA_DIR / "embeddings") / name
        if should_stop is not None and should_stop():
            result.stopped[name] = "not started (time limit)"
            continue
        try:
            todo = photos_to_embed(conn, name, size, store.location, str(config.DATA_DIR))
            log(f"[{name}] {len(todo):,} photos to embed")
            per_second, skipped, seconds = None, [], 0.0
            if todo:
                started = time.monotonic()
                backbone = loader(spec)
                if backbone.name != name:
                    raise RuntimeError(f"loader named the backbone {backbone.name!r}, "
                                       f"expected {name!r}")
                stats = embed_photos(conn, store, backbone, todo, root, batch_size=batch_size,
                                     readers=readers, should_stop=should_stop,
                                     log=lambda s: log(f"[{name}] {s.strip()}"))
                seconds = time.monotonic() - started
                per_second = round(stats.embedded / max(seconds, 1e-6), 1)
                skipped = stats.skipped
                result.skipped[name] = skipped
                del backbone
                _free_gpu()
                if stats.stopped:
                    raise Stopped(f"stopped at {stats.embedded:,} of {len(todo):,} photos")
                if len(skipped) > skip_threshold * len(todo):
                    raise RuntimeError(
                        f"{len(skipped):,} of {len(todo):,} photos unreadable "
                        f"({len(skipped) / len(todo):.1%}, more than {skip_threshold:.0%}): "
                        f"a systemic problem, not a few bad files; first: {skipped[0][1]}")
                for pid, why in skipped[:20]:
                    log(f"[{name}] skipped photo {pid}: {why}")
            left = {pid for pid, _ in photos_to_embed(conn, name, size, store.location,
                                                      str(config.DATA_DIR))}
            left -= {pid for pid, _ in skipped}
            if left:
                raise RuntimeError(f"{len(left):,} photos still not embedded")
            done = conn.execute("select count(*) from embeddings where backbone = ?",
                                (name,)).fetchone()[0]
            result.embedded[name] = {"per_second": per_second, "photos": done,
                                     "skipped": len(skipped), "seconds": round(seconds, 1)}
            if after_embed is not None:
                after_embed(name)
            log(f"[{name}] done" + (f", {per_second}/s" if per_second else " (already embedded)")
                + (f", {len(skipped):,} unreadable photos skipped" if skipped else ""))
        except Stopped as e:
            result.stopped[name] = str(e)
            log(f"[{name}] STOPPED: {e}")
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
