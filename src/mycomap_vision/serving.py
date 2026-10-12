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

A release made with `mv release --int8-vectors` (Steve, 2026-10-12: v1 stays on the 4 GB
box) ships each reference vector as int8 codes, `shard-NNNNN.npy`, with one float32 scale
per vector beside it, `shard-NNNNN.scale.npy` (quantize.py): 772 bytes a vector instead
of 1,536. The codes are memory-mapped like float16 shards; the scales (4 bytes a vector)
are read in. A score is (query . codes) x scale, block by block, which is the query
against the dequantized vector. The nightly update's own shards stay float16, so one
backbone may mix both; every row comes out (or is scored) as the same float vector.

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

SCALE_SUFFIX = ".scale.npy"


class MappedVectors:
    """A backbone's embedding shards, memory-mapped read-only, as one (rows, dim) matrix
    in shard order (the order embed.load_embeddings gives). `scales`: per shard, None for
    a float shard, or the float32 scale of each row of an int8 shard."""

    def __init__(self, shards: list[np.ndarray], scales: list | None = None):
        if not shards:
            raise ValueError("no embedding shards")
        scales = list(scales) if scales is not None else [None] * len(shards)
        if len(scales) != len(shards):
            raise ValueError("one scale array (or None) per shard")
        dims = {s.shape[1] if s.ndim == 2 else None for s in shards}
        if len(dims) != 1 or None in dims:
            raise ValueError(f"embedding shards disagree on their shape: {sorted(map(str, dims))}")
        floats = set()
        for s, sc in zip(shards, scales):
            if sc is None:
                if s.dtype not in (np.float16, np.float32):
                    raise ValueError(f"a float embedding shard must be float16 or float32, "
                                     f"not {s.dtype}")
                floats.add(s.dtype)
            elif s.dtype != np.int8 or sc.ndim != 1 or len(sc) != len(s):
                raise ValueError(f"an int8 shard needs int8 codes and one scale per row "
                                 f"({s.dtype}, {len(s)} rows, {sc.shape} scales)")
        if len(floats) > 1:
            raise ValueError(f"float embedding shards must share one dtype: {floats}")
        self.shards, self.scales = shards, [None if sc is None else
                                            np.asarray(sc, dtype=np.float32) for sc in scales]
        self.offsets = np.cumsum([0] + [len(s) for s in shards]).astype(np.int64)
        # What rows come out as: the float shards' dtype; int8 rows are dequantized to it
        # (float16 when every shard is int8).
        self.dtype = floats.pop() if floats else np.dtype(np.float16)
        self.shape = (int(self.offsets[-1]), dims.pop())
        self.ndim = 2

    def storage(self) -> dict:
        """Rows held as int8 and as float (for /api/health and the box's memory sums)."""
        n8 = sum(len(s) for s, sc in zip(self.shards, self.scales) if sc is not None)
        return {"int8_rows": int(n8), "float_rows": int(len(self) - n8)}

    def __len__(self) -> int:
        return self.shape[0]

    def __getitem__(self, rows) -> np.ndarray:
        """Those rows as an ordinary array in memory (for methods that need one, such as
        training a classifier; the nearest-specimen lookup uses select() instead)."""
        rows = np.arange(len(self))[rows] if isinstance(rows, slice) else np.asarray(rows)
        out = np.empty((len(rows), self.shape[1]), dtype=self.dtype)
        for shard, scale, lo, hi in zip(self.shards, self.scales, self.offsets[:-1],
                                        self.offsets[1:]):
            here = (rows >= lo) & (rows < hi)
            if here.any():
                local = rows[here] - lo
                if scale is None:
                    out[here] = shard[local]
                else:
                    out[here] = shard[local].astype(np.float32) * scale[local][:, None]
        return out

    def int8_rows(self, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Those rows as int8 codes and a float32 scale each (quantize.py's rule): an int8
        shard's codes as stored, float rows quantized. For the fast search (fastsearch.py)."""
        from .quantize import quantize_int8
        rows = np.asarray(rows)
        codes = np.empty((len(rows), self.shape[1]), dtype=np.int8)
        scales = np.empty(len(rows), dtype=np.float32)
        for shard, scale, lo, hi in zip(self.shards, self.scales, self.offsets[:-1],
                                        self.offsets[1:]):
            here = (rows >= lo) & (rows < hi)
            if here.any():
                local = rows[here] - lo
                if scale is None:
                    codes[here], scales[here] = quantize_int8(shard[local])
                else:
                    codes[here], scales[here] = shard[local], scale[local]
        return codes, scales

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
        # The fast int8 search (fastsearch.py, MV_SEARCH_INT8), when packed; else the
        # mapped search below.
        self.packed = None
        self.refine = 0
        # Per shard: which of its rows are wanted (in file order, for sequential reads)
        # and where each goes in the output. None = every row, in order.
        self.plan = []
        for shard, scale, lo, hi in zip(vectors.shards, vectors.scales, vectors.offsets[:-1],
                                        vectors.offsets[1:]):
            pos = np.flatnonzero((self.cols >= lo) & (self.cols < hi))
            if not len(pos):
                continue
            local = self.cols[pos] - lo
            order = np.argsort(local, kind="stable")
            pos, local = pos[order], local[order]
            whole = len(local) == len(shard) and bool((local == np.arange(len(shard))).all())
            # An int8 shard's scales, in the same order as the rows scored (in memory).
            sc = None if scale is None else (scale if whole else scale[local])
            self.plan.append((shard, None if whole else local, pos, sc))

    def __len__(self) -> int:
        return self.shape[0]

    def __array__(self, dtype=None, copy=None):
        out = self.vectors[self.cols]
        return out if dtype is None else out.astype(dtype, copy=False)

    def use_int8(self, refine: int, slot=None) -> None:
        """Pack these rows for the fast int8 search, after taking the process's one packed
        slot (the index serving now drops its copy and searches mapped until the swap)."""
        from . import fastsearch
        slot = slot or fastsearch.SLOT
        slot.take(self)
        self.drop_packed()
        packed = fastsearch.PackedInt8.build(
            lambda lo, hi: self.vectors.int8_rows(self.cols[lo:hi]), len(self.cols))
        self.packed, self.refine = packed, refine
        slot.hold(self)

    def drop_packed(self) -> None:
        self.packed = None

    def exact(self, query: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """Scores of `query` against the rows at these positions, computed exactly as
        sims() does without the fast search ((q . codes) x scale for int8 rows), so a
        refined column scores the same either way."""
        q = np.asarray(query, dtype=np.float32)
        positions = np.asarray(positions, dtype=np.int64)
        rows = self.cols[positions]
        vec = self.vectors
        which = np.searchsorted(vec.offsets, rows, side="right") - 1
        out = np.empty((len(q), len(positions)), dtype=np.float32)
        for i in np.unique(which):
            here = which == i
            local = rows[here] - vec.offsets[i]
            part = q @ vec.shards[i][local].astype(np.float32).T
            if vec.scales[i] is not None:
                part *= vec.scales[i][local][None, :]
            out[:, here] = part
        return out

    def sims(self, query: np.ndarray) -> np.ndarray:
        packed = self.packed                    # one read: a concurrent drop can't split it
        if packed is not None:
            from .fastsearch import search
            return search(packed, query, self.refine, exact=self.exact)
        q = np.asarray(query, dtype=np.float32)
        out = np.empty((len(q), len(self.cols)), dtype=np.float32)
        for shard, local, pos, scale in self.plan:
            for s in range(0, len(pos), self.CHUNK):
                block = shard[s:s + self.CHUNK] if local is None else shard[local[s:s + self.CHUNK]]
                part = q @ block.astype(np.float32).T
                if scale is not None:            # int8 codes: (q . codes) x scale
                    part *= scale[s:s + self.CHUNK][None, :]
                out[:, pos[s:s + self.CHUNK]] = part
        return out


def layer_first_shard(conn: sqlite3.Connection, backbone: str) -> int | None:
    """The first shard number the nightly update wrote for `backbone`, or None (no layer)."""
    has = conn.execute("select 1 from sqlite_master where type = 'table' and "
                       "name = 'nightly_layer'").fetchone()
    if not has:
        return None
    row = conn.execute("select first_shard from nightly_layer where backbone = ?",
                       (backbone,)).fetchone()
    return None if row is None else int(row[0])


def map_embeddings(conn: sqlite3.Connection, backbone: str, root: Path | None = None,
                   layer_root: Path | None = None) -> tuple[np.ndarray, MappedVectors]:
    """embed.load_embeddings, with the vectors memory-mapped instead of read in: the same
    photo ids and rows, in the same order. Refuses shards whose ids and vectors differ in
    length (a truncated file fails to map at all).

    `layer_root`: where the nightly update (nightly.py) writes the shards it adds after
    the release's. Shards from the backbone's first nightly shard on are read from
    <layer_root>/<backbone>/, the rest from `root`."""
    from .embed import SCHEMA
    root = root or config.DATA_DIR / "embeddings" / backbone
    conn.executescript(SCHEMA)
    shards = sorted({r[0] for r in conn.execute(
        "select distinct shard from embeddings where backbone = ?", (backbone,))})
    first_layer = layer_first_shard(conn, backbone) if layer_root else None
    ids, vecs, scales = [], [], []
    for s in shards:
        here = layer_root / backbone if first_layer is not None and s >= first_layer else root
        i = np.load(here / f"shard-{s:05d}.ids.npy")
        v = np.load(here / f"shard-{s:05d}.npy", mmap_mode="r")
        if v.ndim != 2 or len(v) != len(i):
            raise ValueError(f"{backbone} shard {s}: {len(i)} photo ids but vectors of shape "
                             f"{v.shape}")
        scale_file = here / f"shard-{s:05d}{SCALE_SUFFIX}"
        sc = np.load(scale_file) if scale_file.is_file() else None
        if v.dtype == np.int8 and sc is None:
            raise ValueError(f"{backbone} shard {s}: int8 codes without their {SCALE_SUFFIX}")
        ids.append(i)
        vecs.append(v)
        scales.append(sc)
    if not ids:
        return np.zeros(0, dtype=np.int64), None
    return np.concatenate(ids), MappedVectors(vecs, scales)


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

    def served(self) -> dict:
        """{key: index} of everything being served now (a snapshot)."""
        with self._lock:
            return {k: idx for k, (_v, idx) in self._served.items()}

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
