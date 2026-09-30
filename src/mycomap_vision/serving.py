"""What the server holds to identify, and how it replaces it without holding two.

Reference vectors are read from the embedding shards memory-mapped (np.load with
mmap_mode="r"), never copied into memory. Only the pages a query touches are
resident, as file pages the OS can drop when memory is short, and a rebuilt index
maps the same files, so a swap never needs a second copy. Before, the shards were
concatenated in memory and then copied again in species order: at the full photo
set 0.9 GB + 0.9 GB while building, and a rebuild did that while the old index still
held its own 0.9 GB (docs/deploy.md, "Memory on the 4 GB box").

On the server box the shards are the release's files, which `mv pull-release` has
already checked against their sha256; a build checks their shapes (MappedVectors)
and reads every page once before it is swapped in (Identifier.warm).

ServedIndexes swaps indexes: a new version (someone withdrew permission, a new
calibration) is built on its own thread and swapped in only when it is complete; the
old one is released right after. Builds run one at a time. A failed build leaves the
old index serving and is retried after a pause.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path
from typing import Callable, Hashable

import numpy as np

from . import config


# ---------------------------------------------------------------------------
# Reference vectors, memory-mapped

class MappedVectors:
    """A backbone's embedding shards, memory-mapped read-only, as one (rows, dim) matrix
    in shard order (the order embed.load_embeddings gives)."""

    def __init__(self, shards: list[np.ndarray]):
        if not shards:
            raise ValueError("no embedding shards")
        dims = {s.shape[1] if s.ndim == 2 else None for s in shards}
        dtypes = {s.dtype for s in shards}
        if len(dims) != 1 or None in dims:
            raise ValueError(f"embedding shards disagree on their shape: {sorted(map(str, dims))}")
        if len(dtypes) != 1 or next(iter(dtypes)) not in (np.float16, np.float32):
            raise ValueError(f"embedding shards must all be float16 or float32: {dtypes}")
        self.shards = shards
        self.offsets = np.cumsum([0] + [len(s) for s in shards]).astype(np.int64)
        self.dtype = shards[0].dtype
        self.shape = (int(self.offsets[-1]), dims.pop())
        self.ndim = 2

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows) -> np.ndarray:
        """Those rows as an ordinary array in memory (for methods that need one, such as
        training a classifier; the nearest-specimen lookup uses select() instead)."""
        rows = np.arange(len(self))[rows] if isinstance(rows, slice) else np.asarray(rows)
        out = np.empty((len(rows), self.shape[1]), dtype=self.dtype)
        for shard, lo, hi in zip(self.shards, self.offsets[:-1], self.offsets[1:]):
            here = (rows >= lo) & (rows < hi)
            if here.any():
                out[here] = shard[rows[here] - lo]
        return out

    def select(self, cols: np.ndarray) -> "MappedSelection":
        return MappedSelection(self, cols)


class MappedSelection:
    """Rows `cols` of a MappedVectors, in that order, scored straight from disk. Each
    block is widened to float32 only while it is scored (as methods.Scorer does)."""

    CHUNK = 65_536

    def __init__(self, vectors: MappedVectors, cols: np.ndarray):
        self.vectors = vectors
        self.cols = np.asarray(cols, dtype=np.int64)
        self.shape = (len(self.cols), vectors.shape[1])
        self.dtype = vectors.dtype
        self.ndim = 2
        # Per shard: which of its rows are wanted (in file order, for sequential reads)
        # and where each goes in the output. None = every row, in order.
        self.plan = []
        for shard, lo, hi in zip(vectors.shards, vectors.offsets[:-1], vectors.offsets[1:]):
            pos = np.flatnonzero((self.cols >= lo) & (self.cols < hi))
            if not len(pos):
                continue
            local = self.cols[pos] - lo
            order = np.argsort(local, kind="stable")
            pos, local = pos[order], local[order]
            whole = len(local) == len(shard) and bool((local == np.arange(len(shard))).all())
            self.plan.append((shard, None if whole else local, pos))

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = self.vectors[self.cols]
        return out if dtype is None else out.astype(dtype, copy=False)

    def sims(self, query: np.ndarray) -> np.ndarray:
        q = np.asarray(query, dtype=np.float32)
        out = np.empty((len(q), len(self.cols)), dtype=np.float32)
        for shard, local, pos in self.plan:
            for s in range(0, len(pos), self.CHUNK):
                block = shard[s:s + self.CHUNK] if local is None else shard[local[s:s + self.CHUNK]]
                out[:, pos[s:s + self.CHUNK]] = q @ block.astype(np.float32).T
        return out


def map_embeddings(conn: sqlite3.Connection, backbone: str,
                   root: Path | None = None) -> tuple[np.ndarray, MappedVectors]:
    """embed.load_embeddings, with the vectors memory-mapped instead of read in: the same
    photo ids and rows, in the same order. Refuses shards whose ids and vectors differ in
    length (a truncated file fails to map at all)."""
    from .embed import SCHEMA
    root = root or config.DATA_DIR / "embeddings" / backbone
    conn.executescript(SCHEMA)
    shards = sorted({r[0] for r in conn.execute(
        "select distinct shard from embeddings where backbone = ?", (backbone,))})
    ids, vecs = [], []
    for s in shards:
        i = np.load(root / f"shard-{s:05d}.ids.npy")
        v = np.load(root / f"shard-{s:05d}.npy", mmap_mode="r")
        if v.ndim != 2 or len(v) != len(i):
            raise ValueError(f"{backbone} shard {s}: {len(i)} photo ids but vectors of shape "
                             f"{v.shape}")
        ids.append(i)
        vecs.append(v)
    if not ids:
        return np.zeros(0, dtype=np.int64), None
    return np.concatenate(ids), MappedVectors(vecs)


# ---------------------------------------------------------------------------
# Swapping indexes

_DEFAULT_WAIT = object()


class Updating(RuntimeError):
    """The index is being rebuilt, and the old one may not be used meanwhile."""


class ServedIndexes:
    """One served index per key (backbone, method), each with a version.

    get(key, version) returns the index for that version. When the version has
    changed, a build of the new one starts on its own thread; the caller waits for it
    (`wait` seconds when an older index exists, as long as it takes when there is
    none) and gets it once it is swapped in, or Updating if it is still being built.
    The old index is never used for the new version: a new version can mean someone
    withdrew their photos. But if a build fails, the old index keeps serving (and the
    failure is noted); the new version is tried again after `retry_seconds`.

    Builds run one at a time, so two builds' working memory never add up, and the old
    index is released as soon as the new one is in (`release` runs after that)."""

    def __init__(self, build: Callable[[Hashable, Hashable], object], wait: float = 20.0,
                 retry_seconds: float = 60.0, note: Callable[[str], None] = print,
                 release: Callable[[], None] = lambda: None,
                 clock: Callable[[], float] = time.monotonic):
        self._build, self.wait, self.retry_seconds = build, wait, retry_seconds
        self.note, self.release, self.clock = note, release, clock
        self._lock = threading.Condition()
        self._one_build = threading.Lock()
        self._served: dict = {}         # key -> (version, index)
        self._building: dict = {}       # key -> version
        self._failed: dict = {}         # key -> (version, when, exception)

    def current(self, key) -> tuple | None:
        """(version, index) being served for `key`, or None."""
        with self._lock:
            return self._served.get(key)

    def get(self, key, version, wait=_DEFAULT_WAIT):
        """The index for `version`. `wait`: seconds to wait for a rebuild when an older
        index exists (default: the instance's); None waits as long as it takes."""
        wait = self.wait if wait is _DEFAULT_WAIT else wait
        deadline = None
        with self._lock:
            while True:
                served = self._served.get(key)
                if served and served[0] == version:
                    return served[1]
                failed = self._failed.get(key)
                if failed and failed[0] == version and self.clock() - failed[1] < self.retry_seconds:
                    if served:
                        return served[1]           # a failed load leaves the old one serving
                    raise failed[2]
                if self._building.get(key) is None:
                    self._start(key, version)
                if served is None or wait is None:
                    self._lock.wait()
                    continue
                if deadline is None:
                    deadline = self.clock() + wait
                left = deadline - self.clock()
                if left <= 0:
                    raise Updating("the reference set is being updated; try again in a minute")
                self._lock.wait(left)

    def _start(self, key, version) -> None:
        """Start building `version` for `key` (call with the lock held)."""
        self._building[key] = version
        threading.Thread(target=self._run, args=(key, version), daemon=True,
                         name=f"build {key}").start()

    def _run(self, key, version) -> None:
        with self._one_build:
            try:
                built = self._build(key, version)
            except BaseException as e:  # noqa: BLE001 - noted; the old index keeps serving
                with self._lock:
                    self._failed[key] = (version, self.clock(), e)
                    self._building.pop(key, None)
                    old = self._served.get(key)
                    self._lock.notify_all()
                self.note(f"[index] building {key} failed ({type(e).__name__}: {e}); "
                          + ("the previous index keeps serving" if old else "nothing to serve yet")
                          + f", retrying after {self.retry_seconds:.0f} s")
                return
            with self._lock:
                old = self._served.get(key)
                self._served[key] = (version, built)
                self._building.pop(key, None)
                self._failed.pop(key, None)
                self._lock.notify_all()
            del built
            if old is not None:
                del old                                  # the last reference: freed now
                self.note(f"[index] {key} swapped in; the previous index is released")
            self.release()
