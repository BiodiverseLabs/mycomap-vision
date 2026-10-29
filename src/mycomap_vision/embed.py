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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from PIL import Image, ImageOps

from . import config, holdouts, sources
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

-- One row per embedding run, for the speed shown next to accuracy.
create table if not exists embed_runs (
  backbone   text not null,
  photos     integer not null,
  seconds    real not null,
  device     text,
  created_at text not null
);
"""

# The photo copy size a vector was made from, when the caller said (embed_photos'
# source_size). A side table, so the embeddings rows keep their shape (other manifests and
# the trainer copy them whole), created only where vectors are written: SCHEMA runs on
# read-only manifests too. No row: made before sizes were recorded, size unknown.
SOURCES_SCHEMA = """
create table if not exists embedding_sources (
  backbone    text not null,
  photo_id    integer not null,
  source_size text not null,
  primary key (backbone, photo_id)
);
"""


def recorded_sizes(conn: sqlite3.Connection, backbone: str) -> dict[int, str]:
    """{photo id: copy size} of the backbone's vectors whose size was recorded."""
    if not conn.execute("select 1 from sqlite_master where name = 'embedding_sources'").fetchone():
        return {}
    return dict(conn.execute("select photo_id, source_size from embedding_sources where "
                             "backbone = ?", (backbone,)).fetchall())


class Backbone(Protocol):
    """`prepare` (optional) turns one image into model input; it runs in the reader
    threads so image resizing overlaps the GPU. `encode` takes a list of prepared items."""
    name: str
    dim: int

    def encode(self, images: list) -> np.ndarray: ...


def decode(body: bytes) -> Image.Image:
    from .guards import photo_formats
    img = Image.open(io.BytesIO(body), formats=photo_formats())
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")


def normalise(v: np.ndarray) -> np.ndarray:
    v = v.astype(np.float32)
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return (v / np.maximum(n, 1e-12)).astype(np.float16)


def photos_to_embed(conn: sqlite3.Connection, backbone: str, size: str, store_location: str,
                    local_location: str, north_america_only: bool = True,
                    limit: int | None = None) -> list[tuple[int, str]]:
    """(photo_id, relpath) of photos with a copy at `size` in the store, not yet embedded.

    `local_location` is kept for callers; copies downloaded before stores were recorded
    were filed under the manifest's own data folder when photo_copies was created.
    Photos of a benchmark's held-out records (holdouts.py) are never embedded here, nor
    photos of a record whose source Vision takes no photos from, nor a photo whose
    source is not its record's (sources.py). Refused on a manifest whose sources are
    still guessed from ids.
    """
    sources.require_migrated(conn, "embedding photos")
    conn.executescript(SCHEMA)
    holdouts.ensure_schema(conn)
    na = "and r.north_america = 1" if north_america_only else ""
    sql = f"""
      select distinct c.photo_id, c.path
      from photo_copies c
      join observation_photos op on op.photo_id = c.photo_id
      join photos p on p.photo_id = c.photo_id
      join records r on r.observation_id = op.observation_id {na}
        and r.source in {sources.PHOTO_SOURCES_SQL} and p.source = r.source
      left join embeddings e on e.backbone = ? and e.photo_id = c.photo_id
      where c.size = ? and c.store = ? and e.photo_id is null
        and {holdouts.not_held_out('r.observation_id')}
      order by c.photo_id
    """
    if limit is not None:
        sql += f" limit {int(limit)}"
    return [(r[0], r[1]) for r in conn.execute(sql, (backbone, size, store_location))]


# --- specimen crops of excluded slip photos (feedback.sync_crops, recovered_labels.py) -----------
# A crop is embedded as a vector of its own, under a negative id that no real photo has. Its
# box is applied to the decoded photo in memory: no cropped file is ever written.
CROP_SLOTS = 16                 # crop numbers per photo (slipcrop.MAX_BOXES is far fewer)
CROP_VECTOR_SQL = f"-(c.photo_id * {CROP_SLOTS} + c.n)"


def crop_vector_id(photo_id: int, n: int) -> int:
    """The embeddings id of crop `n` of a photo: negative, so never a photo's own."""
    if not 0 < n < CROP_SLOTS:
        raise ValueError(f"crop number {n} is outside 1..{CROP_SLOTS - 1}")
    return -(int(photo_id) * CROP_SLOTS + int(n))


def apply_box(img: Image.Image, box) -> Image.Image:
    """The part of `img` inside `box` (l, t, r, b as fractions of the photo)."""
    w, h = img.size
    l, t, r, b = (min(max(float(v), 0.0), 1.0) for v in box)
    if r <= l or b <= t:
        raise ValueError(f"empty crop box {box}")
    return img.crop((int(round(l * w)), int(round(t * h)),
                     max(int(round(r * w)), int(round(l * w)) + 1),
                     max(int(round(b * h)), int(round(t * h)) + 1)))


def _crops_sql(north_america_only: bool, copy_join: str = "") -> str:
    """The kept crops not yet embedded by the backbone (the first parameter), on records
    whose photos Vision may use; `copy_join` joins the photo's copy (alias pc)."""
    na = "and r.north_america = 1" if north_america_only else ""
    return f"""
      from feedback_photo_crops c {copy_join}
      join observation_photos op on op.photo_id = c.photo_id
      join photos p on p.photo_id = c.photo_id
      join records r on r.observation_id = op.observation_id {na}
        and r.source in {sources.PHOTO_SOURCES_SQL} and p.source = r.source
      left join embeddings e on e.backbone = ? and e.photo_id = {CROP_VECTOR_SQL}
      where e.photo_id is null and {holdouts.not_held_out('r.observation_id')}
    """


def crops_to_embed(conn: sqlite3.Connection, backbone: str, size: str, store_location: str,
                   north_america_only: bool = True, limit: int | None = None
                   ) -> list[tuple[int, str, tuple]]:
    """(crop vector id, relpath of the photo's copy, box) of each kept crop
    (feedback_photo_crops) not yet embedded whose photo has a copy at `size` in the store.
    The same rules as photos_to_embed: no held-out record, only a source Vision uses."""
    from .feedback import ensure_crops_schema
    conn.executescript(SCHEMA)
    holdouts.ensure_schema(conn)
    ensure_crops_schema(conn)
    sql = (f"select distinct {CROP_VECTOR_SQL}, pc.path, c.l, c.t, c.r, c.b"
           + _crops_sql(north_america_only, "join photo_copies pc on pc.photo_id = c.photo_id "
                                            "and pc.size = ? and pc.store = ?")
           + " order by c.photo_id, c.n")
    if limit is not None:
        sql += f" limit {int(limit)}"
    rows = conn.execute(sql, (size, store_location, backbone))
    return [(int(r[0]), r[1], (r[2], r[3], r[4], r[5])) for r in rows]


def crop_photos_to_fetch(conn: sqlite3.Connection, backbone: str, size: str,
                         store_location: str, north_america_only: bool = True,
                         limit: int | None = None):
    """Photos with a kept crop not yet embedded by `backbone` and no copy at `size` in the
    store, as photos.run_downloads takes them (photo_id, source_url, host, attempts, source):
    a release's photos are embedded but not held on the box."""
    from .feedback import ensure_crops_schema
    from .photos import MAX_ATTEMPTS
    conn.executescript(SCHEMA)
    holdouts.ensure_schema(conn)
    ensure_crops_schema(conn)
    sql = ("select distinct p.photo_id, p.source_url, p.host, p.attempts, p.source"
           + _crops_sql(north_america_only)
           + f" and p.status != 'missing' and p.attempts < {MAX_ATTEMPTS}"
             " and not exists (select 1 from photo_copies x where x.photo_id = c.photo_id"
             " and x.size = ? and x.store = ?) order by p.photo_id")
    if limit is not None:
        sql += f" limit {int(limit)}"
    return conn.execute(sql, (backbone, size, store_location)).fetchall()


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
    # (photo_id, why) of each photo that could not be read or decoded; they are left
    # unembedded, and screening decides whether there are too many (skip_threshold).
    skipped: list = field(default_factory=list)
    stopped: bool = False          # should_stop() said stop before every batch was done


def _batches(items: list, n: int) -> Iterable[list]:
    for i in range(0, len(items), n):
        yield items[i:i + n]


def embed_photos(conn: sqlite3.Connection, store: PhotoStore, backbone: Backbone,
                 todo: list[tuple[int, str]], out_dir: Path, batch_size: int = 32,
                 shard_rows: int = 8192, readers: int = 8,
                 log: Callable[[str], None] = print,
                 should_stop: Callable[[], bool] | None = None,
                 source_size: str | None = None) -> EmbedStats:
    """Embed `todo`: (photo_id, relpath), or (crop vector id, relpath, box) for a specimen
    crop (crops_to_embed). A photo that can't be read or decoded is skipped and listed in
    stats.skipped with the reason. `should_stop()` is asked before each batch; when it
    says stop, what is embedded so far is saved and stats.stopped is set. `source_size`:
    the copy size every photo in `todo` was read at, recorded in embedding_sources (None:
    any earlier record for these vectors is dropped, their size being unknown now)."""
    conn.executescript(SCHEMA + SOURCES_SCHEMA)
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
            if source_size is None:
                conn.executemany("delete from embedding_sources where backbone = ? and "
                                 "photo_id = ?", [(backbone.name, pid) for pid in buf_ids])
            else:
                conn.executemany("insert or replace into embedding_sources values (?, ?, ?)",
                                 [(backbone.name, pid, source_size) for pid in buf_ids])
        stats.shards += 1
        shard += 1
        buf_vecs, buf_ids = [], []

    prepare = getattr(backbone, "prepare", None) or (lambda img: img)

    def load(item: tuple):
        pid, rel = item[0], item[1]
        try:
            # decode converts to RGB, which reads every pixel: a truncated file fails here.
            img = decode(store.get(rel))
            if len(item) > 2:                # a specimen crop (crops_to_embed): its box only
                img = apply_box(img, item[2])
            return pid, prepare(img), None
        except Exception as e:           # unreadable, corrupt, truncated, gone
            return pid, None, f"{e.__class__.__name__}: {e}"[:300]

    with ThreadPoolExecutor(max_workers=readers) as pool:
        # Submit the next batch's reads before encoding this one, so they overlap.
        batches = list(_batches(todo, batch_size))
        ahead = [pool.submit(load, it) for it in batches[0]] if batches else []
        for bi in range(len(batches)):
            if should_stop is not None and should_stop():
                for f in ahead:
                    f.cancel()
                stats.stopped = True
                log(f"  stopped at {stats.embedded:,}/{len(todo):,} photos (told to stop)")
                break
            loaded = [f.result() for f in ahead]
            ahead = ([pool.submit(load, it) for it in batches[bi + 1]]
                     if bi + 1 < len(batches) else [])
            good = [(pid, img) for pid, img, _ in loaded if img is not None]
            bad = [(pid, why) for pid, img, why in loaded if img is None]
            stats.unreadable += len(bad)
            stats.skipped.extend(bad)
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
    if stats.embedded:
        with conn:
            conn.execute("insert into embed_runs values (?, ?, ?, ?, ?)",
                         (backbone.name, stats.embedded, stats.seconds,
                          getattr(backbone, "device", None),
                          datetime.now(timezone.utc).isoformat(timespec="seconds")))
    return stats


def archive_embeddings(conn: sqlite3.Connection, backbone: str, label: str,
                       data_dir: Path | None = None) -> dict:
    """Set a backbone's embeddings aside (e.g. before embedding the same photos at another
    size): the folder moves to embeddings-archive/<backbone>-<label>, and its rows leave
    the index, so every photo counts as not embedded. Nothing is deleted; each shard's
    .ids.npy file says which photos it holds. Refuses to overwrite an earlier archive."""
    import shutil
    data_dir = data_dir or config.DATA_DIR
    conn.executescript(SCHEMA + SOURCES_SCHEMA)
    src = data_dir / "embeddings" / backbone
    dest = data_dir / "embeddings-archive" / f"{backbone}-{label}"
    if dest.exists():
        raise FileExistsError(f"{dest} already exists; choose another label")
    rows = conn.execute("select count(*) from embeddings where backbone = ?",
                        (backbone,)).fetchone()[0]
    if not rows and not src.exists():
        raise ValueError(f"{backbone} has no embeddings to archive")
    if src.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    with conn:
        conn.execute("delete from embeddings where backbone = ?", (backbone,))
        conn.execute("delete from embedding_sources where backbone = ?", (backbone,))
    return {"backbone": backbone, "rows": rows, "archive": str(dest)}


def photos_per_second(conn: sqlite3.Connection) -> dict[str, float]:
    """Each backbone's throughput over its runs of 1,000 photos or more (small runs are
    dominated by model loading)."""
    conn.executescript(SCHEMA)
    rows = conn.execute("select backbone, sum(photos) / sum(seconds) from embed_runs "
                        "where photos >= 1000 group by backbone").fetchall()
    return {b: round(v, 1) for b, v in rows}


def load_embeddings(conn: sqlite3.Connection, backbone: str,
                    root: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """All vectors of a backbone as (photo_ids, vectors), in shard order."""
    root = root or config.DATA_DIR / "embeddings" / backbone
    conn.executescript(SCHEMA)
    shards = sorted({r[0] for r in conn.execute(
        "select distinct shard from embeddings where backbone = ?", (backbone,))})
    ids, vecs = [], []
    for s in shards:
        ids.append(np.load(root / f"shard-{s:05d}.ids.npy"))
        vecs.append(np.load(root / f"shard-{s:05d}.npy"))
    if not ids:
        return np.zeros(0, dtype=np.int64), np.zeros((0, 0), dtype=np.float16)
    return np.concatenate(ids), np.concatenate(vecs)
