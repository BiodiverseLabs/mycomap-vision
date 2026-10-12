"""Fast int8 search of the reference vectors (docs/capacity.md option C; MV_SEARCH_INT8=1).

Scoring a query against every reference photo costs ~1.9 s a call on the box: each
float16 (or int8, quantize.py) block is widened to float32 on every call. Here the
reference rows are packed ONCE per index build into fbgemm's int8 matrix format (torch's
x86/fbgemm quantized engine; the box's CPU has AVX-512 but no VNNI, which fbgemm handles),
and a query is scored with one int8 matrix product per block: ~0.01-0.03 s on two threads.

- Accuracy: the query is quantized to 8 bits too, so a score moves by up to ~0.003. Each
  query's best `refine` columns are then scored again exactly from the memory-mapped
  vectors (as before), so the names and specimens that decide an answer score as they did;
  only columns far down the list keep the int8 approximation.
- Memory (Steve, 2026-10-12: keep it low, no resize): the packed matrix lives in memory
  (~1 byte per dimension per photo, ~465 MB at 605k photos), unlike the mapped files the
  kernel can drop. It is packed a block at a time, and the process holds at most ONE
  packed matrix (`SLOT`): an index being built first takes the slot from the one serving,
  which goes back to the mapped search until the swap, so a rebuild never holds two.

Rows come from serving.MappedSelection in index order: an int8 release's codes as they
are, float16 rows (the nightly update's) quantized by quantize.py's rule.
"""

from __future__ import annotations

import threading

import numpy as np

BLOCK = 32_768        # rows packed at a time: the build needs ~2 extra copies of one block
REFINE = 4096


def engine() -> str:
    """fbgemm (or x86, or onednn) when this torch has it; otherwise RuntimeError (the float search
    stays)."""
    import torch
    engines = torch.backends.quantized.supported_engines
    for name in ("fbgemm", "x86", "onednn"):
        if name in engines:
            torch.backends.quantized.engine = name
            return name
    raise RuntimeError(f"no fbgemm/x86/onednn quantized engine in this torch ({engines})")


class PackedInt8:
    """Reference rows packed for int8 matrix products, in blocks."""

    def __init__(self, blocks: list, rows: int, dim: int, engine_name: str):
        self.blocks, self.rows, self.dim, self.engine = blocks, rows, dim, engine_name

    @property
    def nbytes(self) -> int:
        return self.rows * (self.dim + 4)

    @classmethod
    def build(cls, rows_int8, total: int, block: int = BLOCK) -> "PackedInt8":
        """`rows_int8(lo, hi)` -> (int8 codes, float32 scale per row) for rows lo..hi."""
        import torch
        name = engine()
        blocks, dim = [], 0
        for lo in range(0, total, block):
            codes, scales = rows_int8(lo, min(lo + block, total))
            dim = codes.shape[1]
            safe = np.where(scales > 0, scales, 1e-12).astype(np.float64)
            w = torch._make_per_channel_quantized_tensor(
                torch.from_numpy(np.ascontiguousarray(codes)), torch.from_numpy(safe),
                torch.zeros(len(codes), dtype=torch.long), 0)
            blocks.append(torch.ops.quantized.linear_prepack(w, None))
            del w, codes, scales
        return cls(blocks, total, dim, name)

    def approx(self, query: np.ndarray) -> np.ndarray:
        """(queries, rows) int8 scores of unit queries against every row."""
        import torch
        x = torch.from_numpy(np.ascontiguousarray(query, dtype=np.float32))
        with torch.no_grad():
            parts = [torch.ops.quantized.linear_dynamic(x, b, True) for b in self.blocks]
        return torch.cat(parts, dim=1).numpy()


def search(packed: PackedInt8, query: np.ndarray, refine: int = REFINE, *, exact) -> np.ndarray:
    """Every column scored in int8, then each query's best `refine` columns scored again
    exactly: `exact(query, cols)` scores those columns as the plain search would."""
    q = np.asarray(query, dtype=np.float32)
    out = packed.approx(q)
    k = min(refine, out.shape[1])
    if k <= 0:
        return out
    best = np.unique(np.concatenate([np.argpartition(-row, k - 1)[:k] for row in out]))
    out[:, best] = exact(q, best)
    return out


class Slot:
    """The one packed matrix this process may hold, and who holds it. `take(owner)` drops
    the current holder's matrix before a new one is packed (the holder falls back to the
    mapped search), so two never coexist; `peak_holders` records the most ever held."""

    def __init__(self):
        self._lock = threading.Lock()
        self.owner = None
        self.peak_holders = 0
        self.last_error: str | None = None

    def take(self, owner) -> None:
        with self._lock:
            if self.owner is not None and self.owner is not owner:
                self.owner.drop_packed()
            self.owner = None

    def hold(self, owner) -> None:
        with self._lock:
            held = sum(1 for o in (self.owner, owner) if o is not None and o.packed is not None)
            self.peak_holders = max(self.peak_holders, held if self.owner is not owner else 1)
            self.owner = owner

    def state(self) -> dict:
        """For /api/health: how the reference set is searched right now."""
        with self._lock:
            p = self.owner.packed if self.owner is not None else None
        return {"mode": p.engine if p is not None else "float",
                "packed_rows": p.rows if p is not None else 0,
                "packed_bytes": p.nbytes if p is not None else 0,
                "last_error": self.last_error}


SLOT = Slot()
