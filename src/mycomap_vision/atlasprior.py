"""Atlas habitat maps in Vision's place scoring (docs/atlas-environment.md section 1).

Atlas maps where a taxon's habitat is, from climate, soil and host trees, not only where
its records are. Two Atlas exports, both on a 20 km Albers grid (cell ids 1-based,
row-major from the north-west corner; `AtlasGrid`):

- the prior export (`/api/prior`; files ranks, taxa, models, aoa, grid.json): per strong
  taxon and cell WITHIN its reach (500 km of its sites) a rank 0-100, the number of
  strong models judging it and the spread of their ranks; per taxon its map strength;
- the continent export (private; continent rows, taxa, grid.json): per strong taxon and
  cell anywhere in North America a rank on the same scale, and `aoa`: the smallest
  dissimilarity / threshold x 100 over its models (<= 100 inside an area of
  applicability; 255 unknown).

Both list taxa by name only (taxon ids are per build), so they are joined by
`atlas_name_key`. `import_exports` turns them, once and offline, into a cell-major table
(one row of uint8 per cell, a column per taxon) that the server memory-maps: a place
costs one cell lookup and one row read.

The arms (pre-declared, section 1): an Atlas term `weight x trust x log f(rank)` replaces
the DNA boost for mapped taxa. f is learned on training folds (`rank_factor`), boost only
(f >= 1). trust = map_strength x agreement x applicability, where
agreement = clip(1 - rank_sd / 50, 0, 1) (1 with one model, or where the export gives no
spread: flagged `agreement_assumed`), and applicability = 1 inside an area of
applicability, else 0 (cut) or exp(-(aoa / 100 - 1)) (graded).

Nothing here returns a coordinate or a cell id to a caller outside the server.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .atlasrange import AlbersGrid, atlas_name_key
from .occurrence import column, read_table

NO_DATA = 255               # uint8: no rank / unknown applicability / beyond reach
RANK_BINS = 11              # 0-9, 10-19, ..., 90-99, 100
AGREEMENT_SD = 50.0
TABLE_FILES = ("meta.json", "names.txt", "cells.npy", "rank.npy", "aoa.npy", "models.npy",
               "rank_sd.npy", "reach.npy", "map_strength.npy")
# Atlas's projection (R/grid.R ATLAS_CRS): NAD83 Albers, lat_0 40, lon_0 -96, 20 and 60.
ATLAS_ALBERS = dict(lat_1=20.0, lat_2=60.0, lat_0=40.0, lon_0=-96.0)


@dataclass(frozen=True)
class AtlasGrid:
    xmin: float
    xmax: float
    ymin: float
    ymax: float
    cell_m: float
    ncol: int
    nrow: int

    @classmethod
    def from_json(cls, meta: dict) -> "AtlasGrid":
        e = meta["extent"]
        return cls(float(e["xmin"]), float(e["xmax"]), float(e["ymin"]), float(e["ymax"]),
                   float(meta["cell_m"]), int(meta["ncol"]), int(meta["nrow"]))

    def cell(self, lat, lon) -> np.ndarray:
        """1-based cell ids (0 off the grid), as Atlas's own atlas_cell."""
        alb = AlbersGrid(**ATLAS_ALBERS, cell_m=self.cell_m, x_origin=self.xmin,
                         y_origin=self.ymax, ncol=self.ncol, nrow=self.nrow)
        x, y = alb.xy(np.asarray(lat, dtype=float), np.asarray(lon, dtype=float))
        col = np.floor((x - self.xmin) / self.cell_m)
        row = np.floor((self.ymax - y) / self.cell_m)
        inside = (x >= self.xmin) & (x < self.xmax) & (y > self.ymin) & (y <= self.ymax)
        return np.where(inside, row * self.ncol + col + 1, 0).astype(np.int64)


# --- import (offline, on the laptop) --------------------------------------------------------

def _first(folder: Path, *names: str) -> Path:
    for n in names:
        if (folder / n).is_file():
            return folder / n
    raise FileNotFoundError(f"{folder} has none of {', '.join(names)}")


def _u8(s: str) -> int:
    s = s.strip()
    return NO_DATA if s in ("", "NA") else min(int(round(float(s))), NO_DATA)


def _taxa(path: Path) -> tuple[dict[int, str], dict[str, float]]:
    """{taxon_id: name key}, {name key: map_strength} (absent: no column)."""
    header, rows = read_table(path)
    i_id, i_name = column(header, "taxon_id"), column(header, "name")
    i_ms = column(header, "map_strength", required=False)
    ids, strength = {}, {}
    for r in rows:
        if len(r) <= max(i_id, i_name) or not r[i_id].strip():
            continue
        k = atlas_name_key(r[i_name])
        ids[int(r[i_id])] = k
        if i_ms >= 0 and len(r) > i_ms and r[i_ms].strip() not in ("", "NA"):
            strength[k] = float(r[i_ms])
    return ids, strength


def import_exports(continent_dir: Path, prior_dir: Path | None, out_dir: Path,
                   release: str) -> dict:
    """The continent export (and, when given, the same release's prior export) -> the
    served table in `out_dir`. Returns its summary (written to meta.json)."""
    continent_dir, out_dir = Path(continent_dir), Path(out_dir)
    grid_meta = json.loads((continent_dir / "grid.json").read_text(encoding="utf-8"))
    grid = AtlasGrid.from_json(grid_meta)
    c_ids, c_strength = _taxa(_first(continent_dir, "taxa.tsv", "taxa.tsv.gz"))
    strength = dict(c_strength)
    reach_rows: dict[str, list[tuple[int, int, int, int]]] = {}
    if prior_dir is not None:
        prior_dir = Path(prior_dir)
        p_ids, p_strength = _taxa(_first(prior_dir, "taxa.tsv", "taxa.tsv.gz"))
        for k, v in p_strength.items():
            strength.setdefault(k, v)
        header, rows = read_table(_first(prior_dir, "ranks.tsv.gz", "ranks.tsv"))
        i_t, i_c, i_r = (column(header, "taxon_id"), column(header, "cell_id"),
                         column(header, "rank"))
        i_m, i_s = column(header, "models", required=False), column(header, "rank_sd",
                                                                     required=False)
        for r in rows:
            k = p_ids.get(int(r[i_t]))
            if k is None:
                continue
            reach_rows.setdefault(k, []).append(
                (int(r[i_c]), _u8(r[i_r]), _u8(r[i_m]) if i_m >= 0 else NO_DATA,
                 _u8(r[i_s]) if i_s >= 0 else NO_DATA))
    keys = sorted(set(c_ids.values()))
    col_of = {k: j for j, k in enumerate(keys)}
    header, rows = read_table(_first(continent_dir, "continent.tsv.gz", "continent.tsv"))
    i_t, i_c = column(header, "taxon_id"), column(header, "cell_id")
    i_r, i_a = column(header, "rank"), column(header, "aoa")
    i_m, i_s = column(header, "models", required=False), column(header, "rank_sd",
                                                                 required=False)
    entries = []
    for r in rows:
        k = c_ids.get(int(r[i_t]))
        if k is None:
            continue
        entries.append((int(r[i_c]), col_of[k], _u8(r[i_r]), _u8(r[i_a]),
                        _u8(r[i_m]) if i_m >= 0 else NO_DATA,
                        _u8(r[i_s]) if i_s >= 0 else NO_DATA))
    cells = np.unique(np.array([e[0] for e in entries], dtype=np.int32))
    pos = {int(c): i for i, c in enumerate(cells.tolist())}
    shape = (len(cells), len(keys))
    arrays = {name: np.full(shape, NO_DATA, np.uint8) for name in ("rank", "aoa", "models",
                                                                   "rank_sd")}
    reach = np.zeros(shape, bool)
    for cell, j, rank, aoa, models, sd in entries:
        i = pos[cell]
        arrays["rank"][i, j], arrays["aoa"][i, j] = rank, aoa
        arrays["models"][i, j], arrays["rank_sd"][i, j] = models, sd
    for k, rows_ in reach_rows.items():
        j = col_of.get(k)
        if j is None:
            continue
        for cell, _rank, models, sd in rows_:
            i = pos.get(cell)
            if i is None:
                continue
            reach[i, j] = True
            # In reach, the prior export's model count and spread (the continent export
            # carries none until Atlas adds them).
            if arrays["models"][i, j] == NO_DATA:
                arrays["models"][i, j], arrays["rank_sd"][i, j] = models, sd
    ms = np.array([strength.get(k, np.nan) for k in keys], dtype=np.float32)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "cells.npy", cells)
    for name, a in arrays.items():
        np.save(out_dir / f"{name}.npy", a)
    np.save(out_dir / "reach.npy", reach)
    np.save(out_dir / "map_strength.npy", ms)
    (out_dir / "names.txt").write_text("\n".join(keys), encoding="utf-8")
    summary = {"release": release, "taxa": len(keys), "cells": int(len(cells)),
               "grid": {k: v for k, v in grid.__dict__.items()},
               "map_strength_missing": int(np.isnan(ms).sum()),
               "spread_from": "continent" if i_s >= 0 else (
                   "prior (in reach only)" if prior_dir is not None else "none"),
               "bytes": int(sum(a.nbytes for a in arrays.values()) + reach.nbytes)}
    (out_dir / "meta.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


# --- the served table -----------------------------------------------------------------------

@dataclass
class CellView:
    """One cell's row for every Atlas taxon (uint8, NO_DATA where none)."""
    rank: np.ndarray
    aoa: np.ndarray
    models: np.ndarray
    rank_sd: np.ndarray
    reach: np.ndarray


class AtlasTable:
    def __init__(self, folder: Path, mmap: bool = True):
        folder = Path(folder)
        mode = "r" if mmap else None
        self.meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        self.grid = AtlasGrid(**self.meta["grid"])
        self.keys = (folder / "names.txt").read_text(encoding="utf-8").split("\n")
        self.cells = np.load(folder / "cells.npy")
        self.rank = np.load(folder / "rank.npy", mmap_mode=mode)
        self.aoa = np.load(folder / "aoa.npy", mmap_mode=mode)
        self.models = np.load(folder / "models.npy", mmap_mode=mode)
        self.rank_sd = np.load(folder / "rank_sd.npy", mmap_mode=mode)
        self.reach = np.load(folder / "reach.npy", mmap_mode=mode)
        self.map_strength = np.load(folder / "map_strength.npy")
        self.col_of = {k: j for j, k in enumerate(self.keys)}

    def columns_for(self, names: list[str]) -> np.ndarray:
        """(len(names),) the Atlas column of each Vision name (-1: no strong map)."""
        return np.array([self.col_of.get(atlas_name_key(n), -1) for n in names], np.int64)

    def at(self, lat: float, lon: float) -> CellView | None:
        """The cell's row, or None off the grid / where no taxon has a value."""
        cell = int(self.grid.cell(lat, lon))
        i = int(np.searchsorted(self.cells, cell))
        if cell <= 0 or i >= len(self.cells) or self.cells[i] != cell:
            return None
        return CellView(np.asarray(self.rank[i]), np.asarray(self.aoa[i]),
                        np.asarray(self.models[i]), np.asarray(self.rank_sd[i]),
                        np.asarray(self.reach[i]))


# --- the terms --------------------------------------------------------------------------------

def rank_bin(rank) -> np.ndarray:
    r = np.asarray(rank, dtype=np.int64)
    return np.minimum(r // 10, RANK_BINS - 1)


def agreement(models: np.ndarray, rank_sd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(agreement, assumed): clip(1 - sd/50, 0, 1) with two or more models; 1 with one;
    1 and flagged `assumed` where the export gives no model count or spread."""
    models = np.asarray(models).astype(np.int64)
    sd = np.asarray(rank_sd).astype(np.float64)
    known = (models != NO_DATA) & ((models < 2) | (sd != NO_DATA))
    a = np.where((models >= 2) & (models != NO_DATA) & (sd != NO_DATA),
                 np.clip(1.0 - sd / AGREEMENT_SD, 0.0, 1.0), 1.0)
    return a, ~known


def applicability(aoa: np.ndarray, mode: str) -> np.ndarray:
    """1 inside an area of applicability (aoa <= 100); outside, 0 ("cut") or
    exp(-(ratio - 1)) ("graded"); unknown (255): 0."""
    aoa = np.asarray(aoa).astype(np.float64)
    ratio = aoa / 100.0
    if mode == "cut":
        out = (ratio <= 1.0).astype(np.float64)
    elif mode == "graded":
        out = np.where(ratio <= 1.0, 1.0, np.exp(-(ratio - 1.0)))
    else:
        raise ValueError(f"applicability mode {mode!r}")
    return np.where(aoa == NO_DATA, 0.0, out)


def trust(map_strength: np.ndarray, models, rank_sd, aoa, mode: str,
          use_trust: bool = True) -> np.ndarray:
    """map_strength x agreement x applicability (P2c); `use_trust` False: applicability
    only (P2c-u keeps the area-of-applicability rule, drops the weighting)."""
    app = applicability(aoa, mode)
    if not use_trust:
        return app
    ms = np.nan_to_num(np.asarray(map_strength, dtype=np.float64), nan=0.0)
    return ms * agreement(models, rank_sd)[0] * app


def rank_factor(truth_ranks: np.ndarray, candidate_ranks: np.ndarray,
                smoothing: float = 1.0) -> np.ndarray:
    """log f per rank bin, learned on training records only: how much more often the true
    taxon's rank at its record's cell falls in a bin than its look-alikes' (the photo's
    top candidates with a map) do. Within each taxon's own ranks, never across taxa."""
    t = np.bincount(rank_bin(truth_ranks), minlength=RANK_BINS).astype(float)
    c = np.bincount(rank_bin(candidate_ranks), minlength=RANK_BINS).astype(float)
    p_t = (t + smoothing) / (t.sum() + smoothing * RANK_BINS)
    p_c = (c + smoothing) / (c.sum() + smoothing * RANK_BINS)
    return np.log(p_t / p_c)


def atlas_boost(rank: np.ndarray, weight_trust: np.ndarray, log_f: np.ndarray) -> np.ndarray:
    """weight x trust x log f(rank), boost only (f >= 1); 0 where there is no rank."""
    rank = np.asarray(rank)
    has = rank != NO_DATA
    lf = np.maximum(np.asarray(log_f)[rank_bin(np.where(has, rank, 0))], 0.0)
    return np.where(has, weight_trust * lf, 0.0)


def boyce_index(truth_ranks: np.ndarray, background_ranks: np.ndarray,
                bins: int = 10) -> float:
    """Continuous-Boyce-style index: Spearman correlation between rank class and the
    predicted-to-expected ratio of truths in it (fix 5, H). NaN with too few truths."""
    truth_ranks = np.asarray(truth_ranks, dtype=float)
    background_ranks = np.asarray(background_ranks, dtype=float)
    if len(truth_ranks) < 5 or not len(background_ranks):
        return math.nan
    edges = np.linspace(0, 101, bins + 1)
    p = np.histogram(truth_ranks, edges)[0] / len(truth_ranks)
    e = np.histogram(background_ranks, edges)[0] / len(background_ranks)
    ok = e > 0
    pe = p[ok] / e[ok]
    if len(pe) < 3:
        return math.nan
    a = np.argsort(np.argsort(np.arange(len(pe)))).astype(float)
    b = np.argsort(np.argsort(pe)).astype(float)
    return float(np.corrcoef(a, b)[0, 1])
