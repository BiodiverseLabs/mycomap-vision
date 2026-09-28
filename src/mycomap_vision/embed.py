"""Turn every downloaded photo into one vector per backbone.

The backbone stays frozen: swapping it means re-running this, not retraining
anything large. Vectors are L2-normalised float16, written in shards under
`data/embeddings/<backbone>/`, and indexed in the manifest so runs resume.
"""

from __future__ import annotations

import io
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from PIL import Image, ImageOps

from . import config
from .storage import PhotoStore

SCHEMA = """
create table if not exists embeddings (
  backbone   text not null,
  photo_id   integer not null,
  shard      integer not null,
  row        integer not null,
  created_at text not null,
  primary key (backbone, photo_id)
);
"""


class Backbone(Protocol):
    """`prepare` (optional) turns one image into model input; it runs in the reader
    threads so image resizing overlaps the GPU. `encode` takes a list of prepared items."""
    name: str
    dim: int

    def encode(self, images: list) -> np.ndarray: ...


def decode(body: bytes) -> Image.Image:
    img = Image.open(io.BytesIO(body))
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def normalise(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(n, 1e-12)).astype(np.float16)


def photos_to_embed(conn: sqlite3.Connection, backbone: str, size: str, store_location: str,
                    local_location: str, north_america_only: bool = True,
                    limit: int | None = None) -> list[tuple[int, str]]:
    """(photo_id, relpath) of held photos at `size` in the store, not yet embedded.

    Photos downloaded before the manifest recorded a store count as local.
    """
    conn.executescript(SCHEMA)
    na = "and r.north_america = 1" if north_america_only else ""
    sql = f"""
      select distinct p.photo_id, p.local_path
      from photos p
      join observation_photos op on op.photo_id = p.photo_id
      join records r on r.observation_id = op.observation_id {na}
      left join embeddings e on e.backbone = ? and e.photo_id = p.photo_id
      where p.status = 'done' and p.size = ? and coalesce(p.store, ?) = ?
        and e.photo_id is null
      order by p.photo_id
    """
    if limit is not None:
        sql += f" limit {int(limit)}"
    return [(r[0], r[1]) for r in conn.execute(
        sql, (backbone, size, local_location, store_location))]


def next_shard(conn: sqlite3.Connection, backbone: str) -> int:
    row = conn.execute("select max(shard) from embeddings where backbone = ?",
                       (backbone,)).fetchone()
    return 0 if row[0] is None else row[0] + 1


@dataclass
class EmbedStats:
    embedded: int = 0
    unreadable: int = 0
    shards: int = 0
    seconds: float = 0.0


def _batches(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def embed_photos(conn: sqlite3.Connection, store: PhotoStore, backbone: Backbone,
                 todo: list[tuple[int, str]], out_dir: Path, batch_size: int = 32,
                 shard_rows: int = 8192, readers: int = 8,
                 log: Callable[[str], None] = print) -> EmbedStats:
    conn.executescript(SCHEMA)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = EmbedStats()
    started = time.monotonic()
    shard = next_shard(conn, backbone.name)
    buf_vecs: list[np.ndarray] = []
    buf_ids: list[int] = []

    def flush() -> None:
        nonlocal shard, buf_vecs, buf_ids
        if not buf_ids:
            return
        vecs = np.concatenate(buf_vecs)
        np.save(out_dir / f"shard-{shard:05d}.npy", vecs)
        np.save(out_dir / f"shard-{shard:05d}.ids.npy", np.asarray(buf_ids, dtype=np.int64))
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with conn:
            conn.executemany(
                "insert or replace into embeddings (backbone, photo_id, shard, row, created_at) "
                "values (?, ?, ?, ?, ?)",
                [(backbone.name, pid, shard, i, now) for i, pid in enumerate(buf_ids)])
        stats.shards += 1
        shard += 1
        buf_vecs, buf_ids = [], []

    prepare = getattr(backbone, "prepare", None) or (lambda img: img)

    def load(item: tuple[int, str]):
        pid, rel = item
        try:
            return pid, prepare(decode(store.get(rel)))
        except Exception:
            return pid, None

    with ThreadPoolExecutor(max_workers=readers) as pool:
        # Submit the next batch's reads before encoding this one, so they overlap.
        batches = list(_batches(todo, batch_size))
        ahead = [pool.submit(load, it) for it in batches[0]] if batches else []
        for bi in range(len(batches)):
            loaded = [f.result() for f in ahead]
            ahead = ([pool.submit(load, it) for it in batches[bi + 1]]
                     if bi + 1 < len(batches) else [])
            good = [(pid, img) for pid, img in loaded if img is not None]
            stats.unreadable += len(loaded) - len(good)
            if not good:
                continue
            vecs = normalise(backbone.encode([img for _, img in good]))
            if vecs.shape != (len(good), backbone.dim):
                raise RuntimeError(f"{backbone.name} returned {vecs.shape}, "
                                   f"expected ({len(good)}, {backbone.dim})")
            buf_vecs.append(vecs)
            buf_ids.extend(pid for pid, _ in good)
            stats.embedded += len(good)
            if len(buf_ids) >= shard_rows:
                flush()
                log(f"  {stats.embedded:,}/{len(todo):,} photos, "
                    f"{stats.embedded / max(time.monotonic() - started, 1e-6):.1f}/s")
    flush()
    stats.seconds = time.monotonic() - started
    return stats


def load_embeddings(conn: sqlite3.Connection, backbone: str,
                    root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """All vectors of a backbone as (photo_ids, vectors), in shard order."""
    root = root or config.DATA_DIR / "embeddings" / backbone
    shards = sorted({r[0] for r in conn.execute(
        "select distinct shard from embeddings where backbone = ?", (backbone,))})
    ids, vecs = [], []
    for s in shards:
        ids.append(np.load(root / f"shard-{s:05d}.ids.npy"))
        vecs.append(np.load(root / f"shard-{s:05d}.npy"))
    if not ids:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float16)
    return np.concatenate(ids), np.concatenate(vecs)


# --- Real backbones (need torch; loaded only when asked for) -------------------

class TimmBackbone:
    """DINOv2 and other timm models: the pooled image feature."""

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

    def prepare(self, image):
        return self.transform(image)

    def encode(self, items):
        torch = self.torch
        x = torch.stack([self._as_tensor(i) for i in items]).to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16,
                                                    enabled=self.device == "cuda"):
            return self.model(x).float().cpu().numpy()

    def _as_tensor(self, item):
        return item if isinstance(item, self.torch.Tensor) else self.transform(item)


class OpenClipBackbone:
    """BioCLIP and other open_clip models: the image tower's embedding."""

    def __init__(self, name: str, hub_name: str):
        import open_clip
        import torch
        self.torch = torch
        self.name = name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _, preprocess = open_clip.create_model_and_transforms(hub_name)
        self.model = model.eval().to(self.device)
        self.transform = preprocess
        with torch.inference_mode():
            probe = self.model.encode_image(
                self.transform(Image.new("RGB", (64, 64))).unsqueeze(0).to(self.device))
        self.dim = probe.shape[1]

    def prepare(self, image):
        return self.transform(image)

    def encode(self, items):
        torch = self.torch
        x = torch.stack([self._as_tensor(i) for i in items]).to(self.device, non_blocking=True)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.float16,
                                                    enabled=self.device == "cuda"):
            return self.model.encode_image(x).float().cpu().numpy()

    def _as_tensor(self, item):
        return item if isinstance(item, self.torch.Tensor) else self.transform(item)


BACKBONES: dict[str, Callable[[], Backbone]] = {
    # Self-supervised, strong on fine-grained detail; 518 px input.
    "dinov2-l14": lambda: TimmBackbone("dinov2-l14", "vit_large_patch14_dinov2.lvd142m"),
    "dinov2-b14": lambda: TimmBackbone("dinov2-b14", "vit_base_patch14_dinov2.lvd142m"),
    # Trained on the Tree of Life (includes fungi) with taxonomic text; 224 px input.
    "bioclip-2": lambda: OpenClipBackbone("bioclip-2", "hf-hub:imageomics/bioclip-2"),
}
