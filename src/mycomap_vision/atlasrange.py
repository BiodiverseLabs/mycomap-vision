"""MycoMap Atlas's modelled ranges as a range source (a stub until Atlas's export exists).

Built against the export the Atlas lane proposed on 2026-10-08 (awaiting Steve):

- ranks: taxon_id, cell_id, rank (0-100, a uint8): the mean over the taxon's strong
  models, over its whole accessible area. A listed taxon with no row for a cell:
  that cell is outside its accessible area (strongly unlikely there).
- taxa: taxon_id, name (already canonical, atlas_name_key), grade, models, sites,
  accessible_cells. A taxon not listed is no information: neutral, never out of range.
- grid: Albers equal-area conic on GRS80 (NAD83), its parameters, the top-left
  origin, 20 km cells, ncol and nrow; cell_id = row * ncol + col (`AlbersGrid`).
- Parquet plus a gzipped TSV beside it; this reads the TSVs (`ranks.tsv.gz`,
  `taxa.tsv.gz` or `taxa.tsv`, `grid.json`). About 400 taxa: held as a dense uint8
  [taxa x used cells] array (~18 MB), so a photo costs one cell id and one column.

A rank is a percentile within the taxon's own range, not an occurrence
probability: raw ranks are never compared across taxa. The place term is a
factor f(rank) learned on the dev set (`estimate_rank_factor`, stored as
`OccParams.rank_factor`; neutral until learned), and "outside the accessible
area" is the out-of-range test, with its radius and penalty tuned like iNat's
(radius 0 = the query's own cell). Atlas sets no threshold; Vision derives its own.

No phenology in Atlas: `LayeredSource` keeps the iNat source's season, and iNat's
answer wherever Atlas has no map. One way only: Vision reads Atlas and never
writes to or feeds it. Not to be tuned or evaluated until Atlas's rebuilt release
exists: `leak_check` refuses until Atlas states which records its maps were
trained on.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config, names
from .occprior import OccParams, Parts, RangeSource, TuningRefused
from .occurrence import column, name_key, read_table
from .prior import Context, haversine_km

EXPORT_DIR = Path("atlas") / "export"
OUTSIDE = 255                       # dense array: no row = outside the accessible area
RANK_BINS = 11                      # f(rank) per 0-9, 10-19, ..., 90-99, 100
_SP_DASH = re.compile(r"\bsp-(?=\S)")
_RANK_WORD = {"var.", "var", "subsp.", "subsp", "ssp.", "ssp", "f.", "forma", "fo."}


def default_export_dir() -> Path:
    return config.DATA_DIR / EXPORT_DIR


def atlas_name_key(name: str) -> str:
    """One key for every way of writing a name: a provisional code in house style
    ("Amanita sp. 'IN01'", 'Amanita sp-IN01'), else genus and epithets without
    authors or rank words, lower case. Must agree with Atlas's atlas_name_key."""
    s = _SP_DASH.sub("sp. ", names.fold_name(name))
    parts = names.parse_name(s)
    if parts.code is not None:
        return parts.key
    words = s.split()
    kept = words[:1]
    epithet_next = True                 # after the genus, or after 'var.': any case
    for w in words[1:]:
        if w.lower() in _RANK_WORD:
            epithet_next = True
            continue
        if re.fullmatch(r"[a-z][a-z-]*", w) or (epithet_next
                                                and re.fullmatch(r"[A-Za-z][a-z-]*", w)):
            kept.append(w)
            epithet_next = False
        else:
            break                       # an author ("(L.)", "Fr.", "Peck") ends the name
    return name_key(" ".join(kept))


@dataclass(frozen=True)
class AlbersGrid:
    """Albers equal-area conic on an ellipsoid (Snyder 1987, ch. 14), and square cells
    numbered row * ncol + col from the top-left corner (x_origin, y_origin)."""
    lat_1: float
    lat_2: float
    lat_0: float
    lon_0: float
    cell_m: float
    x_origin: float
    y_origin: float
    ncol: int
    nrow: int
    x_0: float = 0.0                    # false easting / northing
    y_0: float = 0.0
    a: float = 6_378_137.0              # GRS80
    rf: float = 298.257222101
    es: float | None = None             # e squared, if given instead of rf

    @property
    def e2(self) -> float:
        if self.es is not None:
            return self.es
        f = 1.0 / self.rf
        return 2 * f - f * f

    def _q(self, phi):
        e2 = self.e2
        e = np.sqrt(e2)
        s = np.sin(phi)
        return (1 - e2) * (s / (1 - e2 * s * s)
                           - np.log((1 - e * s) / (1 + e * s)) / (2 * e))

    def _consts(self):
        e2 = self.e2
        p1, p2, p0 = np.radians([self.lat_1, self.lat_2, self.lat_0])
        m = lambda p: np.cos(p) / np.sqrt(1 - e2 * np.sin(p) ** 2)  # noqa: E731
        m1, m2 = m(p1), m(p2)
        q1, q2, q0 = self._q(p1), self._q(p2), self._q(p0)
        n = (m1 ** 2 - m2 ** 2) / (q2 - q1) if self.lat_1 != self.lat_2 else np.sin(p1)
        c = m1 ** 2 + n * q1
        rho0 = self.a * np.sqrt(c - n * q0) / n
        return n, c, rho0

    def xy(self, lat, lon):
        n, c, rho0 = self._consts()
        rho = self.a * np.sqrt(c - n * self._q(np.radians(lat))) / n
        theta = n * np.radians(np.asarray(lon, dtype=float) - self.lon_0)
        return rho * np.sin(theta) + self.x_0, rho0 - rho * np.cos(theta) + self.y_0

    def latlon(self, x, y):
        n, c, rho0 = self._consts()
        x = np.asarray(x, dtype=float) - self.x_0
        y = rho0 - (np.asarray(y, dtype=float) - self.y_0)
        rho = np.hypot(x, y)
        theta = np.arctan2(x, y) if n > 0 else np.arctan2(-x, -y)
        q = (c - (rho * n / self.a) ** 2) / n
        e2 = self.e2
        e = np.sqrt(e2)
        phi = np.arcsin(np.clip(q / 2, -1, 1))
        for _ in range(12):
            s = np.sin(phi)
            phi = phi + (1 - e2 * s * s) ** 2 / (2 * np.cos(phi)) * (
                q / (1 - e2) - s / (1 - e2 * s * s)
                + np.log((1 - e * s) / (1 + e * s)) / (2 * e))
        return np.degrees(phi), self.lon_0 + np.degrees(theta / n)

    def cell(self, lat: float, lon: float) -> int:
        """The cell id of a point, or -1 off the grid."""
        x, y = self.xy(lat, lon)
        col = int(np.floor((float(x) - self.x_origin) / self.cell_m))
        row = int(np.floor((self.y_origin - float(y)) / self.cell_m))
        if not (0 <= col < self.ncol and 0 <= row < self.nrow):
            return -1
        return row * self.ncol + col

    def centres(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cells = np.asarray(cells, dtype=np.int64)
        row, col = cells // self.ncol, cells % self.ncol
        return self.latlon(self.x_origin + (col + 0.5) * self.cell_m,
                           self.y_origin - (row + 0.5) * self.cell_m)


def _first(path: Path, *names_: str) -> Path:
    for n in names_:
        if (path / n).is_file():
            return path / n
    raise FileNotFoundError(f"{path} has none of {', '.join(names_)}")


class AtlasExport:
    """One Atlas release's export: the grid, the listed taxa, and their ranks as a dense
    uint8 [taxa x used cells] array (OUTSIDE where a taxon has no row)."""

    def __init__(self, grid: AlbersGrid, taxon_ids: list[int], keys: list[str],
                 cells: np.ndarray, ranks: np.ndarray, meta: dict | None = None,
                 path: Path | None = None):
        self.grid, self.taxon_ids, self.keys = grid, taxon_ids, keys
        self.cells, self.ranks = cells, ranks          # sorted cell ids; (taxa, cells)
        self.meta, self.path = meta or {}, path
        self.row_of_key = {k: i for i, k in enumerate(keys)}

    @classmethod
    def load(cls, path: Path | None = None) -> "AtlasExport":
        path = Path(path or default_export_dir())
        if not (path / "grid.json").is_file():
            raise FileNotFoundError(f"no Atlas export in {path} (grid.json, taxa.tsv.gz, "
                                    "ranks.tsv.gz from an Atlas release)")
        meta = json.loads((path / "grid.json").read_text(encoding="utf-8"))
        grid = AlbersGrid(**{k: meta[k] for k in AlbersGrid.__dataclass_fields__ if k in meta})
        header, rows = read_table(_first(path, "taxa.tsv.gz", "taxa.tsv"))
        i_id, i_name = column(header, "taxon_id", "id"), column(header, "name", "taxon")
        taxa = [(int(r[i_id]), atlas_name_key(r[i_name])) for r in rows
                if len(r) > max(i_id, i_name) and r[i_id].strip()]
        row_of = {tid: i for i, (tid, _) in enumerate(taxa)}
        header, rows = read_table(_first(path, "ranks.tsv.gz", "ranks.tsv"))
        i_t, i_c = column(header, "taxon_id", "id"), column(header, "cell_id", "cell")
        i_r = column(header, "rank", "percentile")
        t, c, r = [], [], []
        for row in rows:
            if len(row) <= max(i_t, i_c, i_r):
                continue
            ti = row_of.get(int(row[i_t]))
            if ti is not None:                         # ranks of an unlisted taxon: ignored
                t.append(ti)
                c.append(int(row[i_c]))
                r.append(int(float(row[i_r])))
        cells = np.unique(np.array(c, dtype=np.int64))
        ranks = np.full((len(taxa), len(cells)), OUTSIDE, dtype=np.uint8)
        if c:
            ranks[np.array(t), np.searchsorted(cells, np.array(c))] = np.clip(r, 0, 100)
        return cls(grid, [x[0] for x in taxa], [x[1] for x in taxa], cells, ranks, meta, path)

    def column_of(self, cell: int) -> int:
        """The dense column of a cell id, or -1 for a cell no listed taxon has a row for."""
        j = int(np.searchsorted(self.cells, cell))
        return j if j < len(self.cells) and self.cells[j] == cell else -1


def rank_bin(rank: np.ndarray) -> np.ndarray:
    return np.minimum(np.asarray(rank, dtype=np.int64) // 10, RANK_BINS - 1)


class AtlasRangeSource(RangeSource):
    """Out of range: a listed taxon whose accessible area has no cell within the
    radius (radius 0: the query's own cell is outside it); a DNA record nearby still
    vetoes it. Place term: f(rank) at the query's cell, learned on dev. Unlisted
    taxa, and a cell the export doesn't cover: no information."""
    name = "atlas"

    def __init__(self, export: AtlasExport, params: OccParams | None = None):
        self.export = export
        self.params = params or OccParams()

    def fit(self, records, species: list[str]) -> None:
        ex = self.export
        self.n_groups = len(species)
        self.row = np.array([ex.row_of_key.get(atlas_name_key(s), -1) for s in species],
                            dtype=np.int64)                        # once per name
        self.has_map = self.row >= 0
        self.cell_lat, self.cell_lon = ex.grid.centres(ex.cells)
        gpos = {s: i for i, s in enumerate(species)}
        dna = [(gpos[r.unit], r.latitude, r.longitude) for r in records
               if r.unit in gpos and r.latitude is not None and r.longitude is not None]
        self.dna_group = np.array([d[0] for d in dna], dtype=np.int64)
        self.dna_lat = np.array([d[1] for d in dna], dtype=np.float64)
        self.dna_lon = np.array([d[2] for d in dna], dtype=np.float64)

    def query_column(self, ctx: Context | None) -> int:
        if ctx is None or ctx.latitude is None or ctx.longitude is None:
            return -1
        cell = self.export.grid.cell(ctx.latitude, ctx.longitude)
        return self.export.column_of(cell) if cell >= 0 else -1

    def ranks_here(self, col: int) -> np.ndarray:
        """(n_groups,) rank at this column; OUTSIDE for no row or no map (see has_map)."""
        out = np.full(self.n_groups, OUTSIDE, dtype=np.int64)
        out[self.has_map] = self.export.ranks[self.row[self.has_map], col]
        return out

    def parts(self, ctx: Context | None, radii: tuple[float, ...] | None = None) -> Parts:
        radii = radii or (self.params.radius_km,)
        none = Parts({r: np.zeros(self.n_groups, dtype=bool) for r in radii}, None, None)
        col = self.query_column(ctx)
        if col < 0:
            return none                       # off the grid, or a cell nobody has rows for
        ex = self.export
        listed = self.row[self.has_map]
        d = haversine_km(ctx.latitude, ctx.longitude, self.cell_lat, self.cell_lon)
        d_dna = haversine_km(ctx.latitude, ctx.longitude, self.dna_lat, self.dna_lon)
        slack = ex.grid.cell_m / 1000 * 0.71
        accessible = ex.ranks[listed] != OUTSIDE                  # (mapped groups, cells)
        out = {}
        for r in radii:
            within = (np.arange(len(ex.cells)) == col) if r <= 0 else d <= r + slack
            near = np.zeros(self.n_groups)
            near[self.has_map] = accessible[:, within].any(axis=1)
            near += np.bincount(self.dna_group, weights=(d_dna <= max(r, 0.0)).astype(float),
                                minlength=self.n_groups)
            out[r] = self.has_map & (near == 0)
        factor = self.params.rank_factor
        if not factor:
            return Parts(out, None, None)        # nothing learned yet: no information
        place = np.zeros(self.n_groups)
        here = self.ranks_here(col)
        inside = self.has_map & (here != OUTSIDE)
        place[inside] = np.asarray(factor, dtype=float)[rank_bin(here[inside])]
        return Parts(out, place, None)

    def mapping_summary(self) -> dict:
        return {"atlas_listed": int(self.has_map.sum()),
                "not_in_atlas": int((~self.has_map).sum())}

    def leak_check(self, uuids, allow_missing: bool = False) -> dict:
        raise TuningRefused("the Atlas source can't be tuned yet: Atlas hasn't said which "
                            "records its maps were trained on (and the rebuilt release isn't "
                            "out), so the validation records may be inside its maps")


def estimate_rank_factor(source: AtlasRangeSource, s, top_k: int = 20,
                         smoothing: float = 1.0) -> list[float]:
    """f(rank) from a validation set: per rank bin, log of how much more often the
    true species sits in that bin at its record's cell than the photo's top candidates
    do. A likelihood ratio within each taxon's own ranks, never across taxa."""
    truth_n = np.zeros(RANK_BINS)
    cand_n = np.zeros(RANK_BINS)
    pos = {name: i for i, name in enumerate(s.species)}
    for i in range(len(s.observation_ids)):
        ctx = Context(s.latitude[i], s.longitude[i])
        col = source.query_column(ctx)
        if col < 0:
            continue
        here = source.ranks_here(col)
        t = pos.get(s.truth[i], -1)
        if t >= 0 and source.has_map[t] and here[t] != OUTSIDE:
            truth_n[rank_bin(here[t])] += 1
        top = np.argsort(-s.scores[i])[:top_k]
        top = top[source.has_map[top] & (here[top] != OUTSIDE)]
        np.add.at(cand_n, rank_bin(here[top]), 1)
    p_true = (truth_n + smoothing) / (truth_n.sum() + smoothing * RANK_BINS)
    p_cand = (cand_n + smoothing) / (cand_n.sum() + smoothing * RANK_BINS)
    return [round(float(x), 4) for x in np.log(p_true / p_cand)]


class LayeredSource(RangeSource):
    """`primary` (Atlas) for the taxa it lists, `fallback` (iNat occurrences) for the
    rest; the season always from the fallback (Atlas has none)."""

    def __init__(self, primary: AtlasRangeSource, fallback: RangeSource):
        self.primary, self.fallback = primary, fallback
        self.params = fallback.params
        self.name = f"{primary.name}+{fallback.name}"

    def fit(self, records, species: list[str]) -> None:
        self.primary.fit(records, species)
        self.fallback.fit(records, species)
        self.n_groups = len(species)

    def parts(self, ctx: Context | None, radii: tuple[float, ...] | None = None) -> Parts:
        """Per component: Atlas's answer for the taxa it lists where it has one, iNat's
        everywhere else. A place Atlas says nothing about (off its grid, a cell no taxon
        has rows for) is iNat's for every taxon; so is the place term until Atlas's rank
        factor is learned."""
        radii = radii or (self.params.radius_km,)
        f = self.fallback.parts(ctx, radii)
        if self.primary.query_column(ctx) < 0:
            return f
        p = self.primary.parts(ctx, radii)
        has = self.primary.has_map
        out = {r: np.where(has, p.out_of_range[r], f.out_of_range[r]) for r in radii}
        genus = ({r: ~has & f.genus_out_of_range[r] for r in radii}
                 if f.genus_out_of_range else None)
        dna = ({r: ~has & f.dna_out_of_range[r] for r in radii}
               if f.dna_out_of_range else None)
        if p.density is None:
            density = f.density
        else:
            fd = f.density if f.density is not None else np.zeros(self.n_groups)
            density = np.where(has, p.density, fd)
        return Parts(out, density, f.season, genus, dna, f.dna_effort)

    def mapping_summary(self) -> dict:
        return {**self.primary.mapping_summary(), **self.fallback.mapping_summary()}

    def leak_check(self, uuids, allow_missing: bool = False) -> dict:
        return {self.primary.name: self.primary.leak_check(uuids, allow_missing),
                self.fallback.name: self.fallback.leak_check(uuids, allow_missing)}
