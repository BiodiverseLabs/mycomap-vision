"""Where fungi are found, from iNaturalist's open data: a compact grid of counts.

`mv build-occurrence` streams iNat's open-data export (observations.csv.gz and
taxa.csv.gz, tab-separated despite the name) from local files and keeps:

- Kingdom Fungi by ancestry, lichens included (they are Fungi on iNat); slime
  molds (Mycetozoa, under Protozoa) only with `with_slime_molds`.
- Research-grade and needs-ID observations with coordinates, inside North
  America's box plus a margin, with a positional accuracy no worse than a cap.
- Never an observation whose uuid is in an exclusion file. Test, validation and
  benchmark records are themselves iNat observations: counting them would put
  the answer at the very spot being tested.

For every species-or-lower taxon and every genus it stores the number of
observations in each grid cell (0.5 degrees by default) and per latitude band
and week of the year; for all fungi together, the same per cell and per band and
week (sampling effort). An observation of a variety also counts for its species,
and every observation counts for its genus.

The store holds counts per cell of public iNat data only, never a Vision record's
coordinates (CLAUDE.md: never expose coordinates). Reading it is `OccurrenceStore`.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import re
from array import array
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from . import config, names
from .dates import PLACEHOLDER

FUNGI = 47170              # iNat's Kingdom Fungi (lichens are inside it)
MYCETOZOA = 47685          # iNat's slime molds (Phylum Mycetozoa, under Protozoa 47686)
SPECIES_LEVEL = 10         # iNat rank_level: species 10, variety/subspecies/form 5
GENUS_LEVEL = 20
FAMILY_LEVEL = 30
WEEKS = 53                 # week of the year: (day of year - 1) // 7, 0..52
KEEP_GRADES = ("research", "needs_id")
# North America as records.is_north_america draws it when only coordinates are known.
NA_BOX = (7.0, 84.0, -170.0, -50.0)        # south, north, west, east

STORE_NAME = Path("occurrence") / "inat-fungi-na.npz"


def default_store_path() -> Path:
    return config.DATA_DIR / STORE_NAME


@dataclass
class BuildOptions:
    box: tuple[float, float, float, float] = NA_BOX
    margin_deg: float = 5.0          # added on every side of the box
    cell_deg: float = 0.5
    band_deg: float = 10.0           # latitude bands for the season counts
    max_accuracy_m: float | None = 25_000.0   # None = no cap; a blank accuracy is kept
    grades: tuple[str, ...] = KEEP_GRADES
    with_slime_molds: bool = False
    observed_before: str | None = None        # ISO date: leave out anything observed on/after


@dataclass(frozen=True)
class Grid:
    south: float
    west: float
    cell_deg: float
    n_lat: int
    n_lon: int
    band_deg: float

    @classmethod
    def for_options(cls, o: BuildOptions) -> "Grid":
        s, n, w, e = o.box
        s, n = max(-90.0, s - o.margin_deg), min(90.0, n + o.margin_deg)
        w, e = max(-180.0, w - o.margin_deg), min(180.0, e + o.margin_deg)
        return cls(s, w, o.cell_deg, int(np.ceil((n - s) / o.cell_deg)),
                   int(np.ceil((e - w) / o.cell_deg)), o.band_deg)

    @property
    def n_cells(self) -> int:
        return self.n_lat * self.n_lon

    @property
    def n_bands(self) -> int:
        return int(np.ceil(self.n_lat * self.cell_deg / self.band_deg))

    @property
    def north(self) -> float:
        return self.south + self.n_lat * self.cell_deg

    @property
    def east(self) -> float:
        return self.west + self.n_lon * self.cell_deg

    def contains(self, lat: float, lon: float) -> bool:
        return self.south <= lat < self.north and self.west <= lon < self.east

    def cell(self, lat: float, lon: float) -> int:
        """Cell index, or -1 outside the grid."""
        if not self.contains(lat, lon):
            return -1
        r = int((lat - self.south) / self.cell_deg)
        c = int((lon - self.west) / self.cell_deg)
        return min(r, self.n_lat - 1) * self.n_lon + min(c, self.n_lon - 1)

    def band(self, lat: float) -> int:
        b = int((lat - self.south) / self.band_deg)
        return min(max(b, 0), self.n_bands - 1)

    def centres(self, cells: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cells = np.asarray(cells, dtype=np.int64)
        r, c = cells // self.n_lon, cells % self.n_lon
        return (self.south + (r + 0.5) * self.cell_deg, self.west + (c + 0.5) * self.cell_deg)

    def band_of_cells(self, cells: np.ndarray) -> np.ndarray:
        lat, _ = self.centres(cells)
        return np.clip(((lat - self.south) / self.band_deg).astype(np.int64), 0,
                       self.n_bands - 1)


def week_of(iso: str | None) -> int:
    """0..52, or -1 for no usable date."""
    if not iso or len(iso) < 10:
        return -1
    try:
        d = date(int(iso[0:4]), int(iso[5:7]), int(iso[8:10]))
    except ValueError:
        return -1
    if d == PLACEHOLDER:                 # 1970-01-01 is no date (dates.py)
        return -1
    return min((d.timetuple().tm_yday - 1) // 7, WEEKS - 1)


# --- reading iNat's files -----------------------------------------------------------------

def open_text(path: Path) -> io.TextIOBase:
    """A gzip file or a plain one, by its first bytes, as UTF-8 text."""
    with open(path, "rb") as f:
        gz = f.read(2) == b"\x1f\x8b"
    if gz:
        return gzip.open(path, "rt", encoding="utf-8", newline="", errors="replace")
    return open(path, "rt", encoding="utf-8", newline="", errors="replace")


def _norm_header(h: str) -> str:
    return h.strip().strip('"').lstrip("﻿").strip().lower()


def read_table(path: Path) -> tuple[list[str], Iterator[list[str]]]:
    """(header, rows) of a tab- or comma-separated file. iNat's export is tab-separated
    without quoting: those lines are split directly, which is several times faster
    than the csv module on 200 million rows."""
    f = open_text(path)
    first = f.readline().rstrip("\r\n")
    tab = "\t" in first
    header = [_norm_header(h) for h in (first.split("\t") if tab
                                        else next(csv.reader([first])))]

    def rows():
        try:
            if tab:
                for line in f:
                    yield line.rstrip("\r\n").split("\t")
            else:
                yield from csv.reader(f)
        finally:
            f.close()
    return header, rows()


def column(header: list[str], *aliases: str, required: bool = True) -> int:
    for a in aliases:
        if a in header:
            return header.index(a)
    if required:
        raise ValueError(f"no {aliases[0]!r} column (looked for {', '.join(aliases)}); "
                         f"the file has: {', '.join(header)}")
    return -1


def _float(s: str) -> float | None:
    try:
        v = float(s)
    except (TypeError, ValueError):
        return None
    return v if v == v else None          # NaN is no value


def _truthy(s: str) -> bool:
    return (s or "").strip().strip('"').lower() in ("true", "t", "1", "yes")


def read_exclusions(paths: Iterable[Path]) -> set[str]:
    """uuids, one per line (blank lines and # comments ignored), lower-cased."""
    out: set[str] = set()
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            s = line.split("#", 1)[0].strip().strip('"').lower()
            if s:
                out.add(s)
    return out


def uuid_hash(uuid: str) -> int:
    return int.from_bytes(hashlib.blake2b(uuid.strip().lower().encode(), digest_size=8).digest(),
                          "little")


@dataclass
class TaxaTable:
    """Fungal taxa (and slime molds when asked) from taxa.csv: what the store keeps."""
    ids: list[int] = field(default_factory=list)          # unit order
    names: list[str] = field(default_factory=list)
    rank_level: list[float] = field(default_factory=list)
    active: list[bool] = field(default_factory=list)
    family_id: list[int] = field(default_factory=list)    # 0 = none
    genus_unit: list[int] = field(default_factory=list)   # -1 = none
    species_unit: list[int] = field(default_factory=list)  # infraspecific: its species; else -1
    # taxon id (as written in observations.csv) -> (own unit, species unit, genus unit)
    lookup: dict[str, tuple[int, int, int]] = field(default_factory=dict)
    roots: dict[int, str] = field(default_factory=dict)   # kept root id -> name found


def read_taxa(path: Path, roots: tuple[int, ...]) -> TaxaTable:
    header, rows = read_table(path)
    i_id = column(header, "taxon_id", "id", "taxonid")
    i_anc = column(header, "ancestry")
    i_rl = column(header, "rank_level")
    i_rank = column(header, "rank", required=False)
    i_name = column(header, "name", "scientific_name", "scientificname")
    i_act = column(header, "active", required=False)
    root_set = set(roots)
    raw: dict[int, tuple[str, float, bool, list[int]]] = {}
    root_names: dict[int, str] = {}
    width = max(i_id, i_anc, i_rl, i_name, i_act, i_rank) + 1
    for row in rows:
        if len(row) < width:
            continue
        try:
            tid = int(float(row[i_id]))
        except ValueError:
            continue
        anc = [int(a) for a in row[i_anc].replace(",", "/").split("/") if a.strip().isdigit()]
        if tid in root_set:
            root_names[tid] = row[i_name].strip()
        if tid not in root_set and not root_set.intersection(anc):
            continue
        rl = _float(row[i_rl])
        raw[tid] = (row[i_name].strip(), 99.0 if rl is None else rl,
                    _truthy(row[i_act]) if i_act >= 0 else True, anc)
    t = TaxaTable(roots=root_names)
    unit_of: dict[int, int] = {}
    for tid, (name, rl, act, _anc) in sorted(raw.items()):
        if rl <= SPECIES_LEVEL or rl == GENUS_LEVEL:
            unit_of[tid] = len(t.ids)
            t.ids.append(tid)
            t.names.append(name)
            t.rank_level.append(rl)
            t.active.append(act)
    t.family_id = [0] * len(t.ids)
    t.genus_unit = [-1] * len(t.ids)
    t.species_unit = [-1] * len(t.ids)
    for tid, (name, rl, act, anc) in raw.items():
        chain = [a for a in anc if a in raw] + [tid]
        genus = next((a for a in reversed(chain) if raw[a][1] == GENUS_LEVEL), None)
        species = next((a for a in reversed(chain)
                        if raw[a][1] == SPECIES_LEVEL and a != tid), None)
        family = next((a for a in reversed(chain) if raw[a][1] == FAMILY_LEVEL), 0)
        g = unit_of.get(genus, -1) if genus is not None else -1
        own = unit_of.get(tid, -1)
        sp = unit_of.get(species, -1) if (species is not None and rl < SPECIES_LEVEL) else -1
        if own >= 0:
            t.family_id[own] = family
            t.genus_unit[own] = g
            t.species_unit[own] = sp
        # An observation counts for its own taxon (species or lower), its species (when
        # it is a variety), and its genus. A genus observation counts once, as its genus;
        # a complex or section is no unit and counts for its genus only.
        t.lookup[str(tid)] = (own if rl != GENUS_LEVEL else -1, sp,
                              g if rl <= GENUS_LEVEL else -1)
    return t


@dataclass
class BuildStats:
    rows: int = 0
    not_kept_taxon: int = 0
    grade: int = 0
    no_coordinates: int = 0
    outside_box: int = 0
    accuracy: int = 0
    excluded: int = 0
    after_date: int = 0
    kept: int = 0


def build(observations: Path, taxa: Path, out: Path, options: BuildOptions | None = None,
          exclude_files: Iterable[Path] = (), log=print) -> dict:
    """Stream the two files into a store at `out` (an .npz). Returns the build summary."""
    o = options or BuildOptions()
    grid = Grid.for_options(o)
    roots = (FUNGI, MYCETOZOA) if o.with_slime_molds else (FUNGI,)
    log(f"Reading taxa from {taxa}...")
    tax = read_taxa(Path(taxa), roots)
    missing = [r for r in roots if r not in tax.roots]
    if missing:
        raise ValueError(f"taxa file has no taxon {missing}: is it iNat's taxa.csv?")
    log(f"  {len(tax.lookup):,} taxa under {', '.join(f'{v} ({k})' for k, v in tax.roots.items())}")
    excluded = read_exclusions(exclude_files)
    header, rows = read_table(Path(observations))
    i_uuid = column(header, "observation_uuid", "uuid", required=False)
    if excluded and i_uuid < 0:
        raise ValueError("an exclusion list was given but the observations file has no "
                         "observation_uuid column, so it could not be applied")
    i_lat = column(header, "latitude", "lat", "decimallatitude")
    i_lon = column(header, "longitude", "lng", "lon", "decimallongitude")
    i_taxon = column(header, "taxon_id", "taxonid")
    i_grade = column(header, "quality_grade", required=False)
    i_acc = column(header, "positional_accuracy", "coordinateuncertaintyinmeters",
                   required=False)
    i_date = column(header, "observed_on", "eventdate", required=False)
    width = max(i_uuid, i_lat, i_lon, i_taxon, i_grade, i_acc, i_date) + 1
    grades = set(o.grades)
    lookup = tax.lookup
    st = BuildStats()
    seen_excluded = 0
    a_own, a_sp, a_ge, a_cell, a_week, a_band = (array("i") for _ in range(6))
    a_hash = array("Q")             # each kept observation's uuid hash: the leave-one-out index
    log(f"Reading observations from {observations}...")
    for row in rows:
        st.rows += 1
        if st.rows % 10_000_000 == 0:
            log(f"  {st.rows:,} rows, {st.kept:,} kept")
        if len(row) < width:
            st.not_kept_taxon += 1
            continue
        tid = row[i_taxon]
        units = lookup.get(tid)
        if units is None and "." in tid:
            units = lookup.get(tid.split(".")[0])
        if units is None:
            st.not_kept_taxon += 1
            continue
        if i_grade >= 0 and row[i_grade].strip().lower() not in grades:
            st.grade += 1
            continue
        lat, lon = _float(row[i_lat]), _float(row[i_lon])
        if lat is None or lon is None:
            st.no_coordinates += 1
            continue
        cell = grid.cell(lat, lon)
        if cell < 0:
            st.outside_box += 1
            continue
        if o.max_accuracy_m is not None and i_acc >= 0:
            acc = _float(row[i_acc])
            if acc is not None and acc > o.max_accuracy_m:
                st.accuracy += 1
                continue
        if excluded and row[i_uuid].strip().lower() in excluded:
            st.excluded += 1
            seen_excluded += 1
            continue
        when = row[i_date].strip() if i_date >= 0 else ""
        if o.observed_before and when and when[:10] >= o.observed_before:
            st.after_date += 1
            continue
        st.kept += 1
        own, sp, ge = units
        a_own.append(own)
        a_sp.append(sp)
        a_ge.append(ge)
        a_cell.append(cell)
        a_week.append(week_of(when))
        a_band.append(grid.band(lat))
        a_hash.append(uuid_hash(row[i_uuid]) if i_uuid >= 0 else 0)
    store = _aggregate(grid, tax, a_own, a_sp, a_ge, a_cell, a_week, a_band)
    meta = {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "code_version": config.code_version(),
        "options": asdict(o),
        "grid": asdict(grid),
        "roots": {str(k): v for k, v in tax.roots.items()},
        "sources": {k: {"name": Path(p).name, "bytes": Path(p).stat().st_size}
                    for k, p in (("observations", observations), ("taxa", taxa))},
        "stats": asdict(st),
        "excluded_uuids": len(excluded),
        "excluded_uuids_seen": seen_excluded,
        "units": int(len(tax.ids)),
        "units_with_observations": int((store["unit_total"] > 0).sum()),
        "pairs": int(len(store["pair_cell"])),
    }
    excl = np.array(sorted(uuid_hash(u) for u in excluded), dtype=np.uint64)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".tmp.npz")
    np.savez_compressed(tmp, meta=np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8),
                        names=np.frombuffer("\n".join(tax.names).encode(), dtype=np.uint8),
                        excluded_hashes=excl, **store)
    tmp.replace(out)
    meta["file_bytes"] = out.stat().st_size
    if i_uuid >= 0:
        meta["leave_one_out_bytes"] = _save_loo(loo_path(out), a_hash, a_own, a_ge, a_cell,
                                                a_week, a_band)
    log(json.dumps(meta, indent=2))
    return meta


def loo_path(store: Path) -> Path:
    return Path(store).with_name(Path(store).stem + ".loo.npz")


def _save_loo(path: Path, a_hash, a_own, a_ge, a_cell, a_week, a_band) -> int:
    """Beside the store: every counted observation's uuid hash and what it added (its
    taxon's unit or genus, cell, week, band), so a scored record can take its own
    observation back out (OccurrenceStore.own_contribution). Evaluation and tuning
    need it; serving photos with no iNat observation doesn't."""
    h = np.frombuffer(a_hash, dtype=np.uint64)
    order = np.argsort(h, kind="stable")
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        tmp, hash=h[order],
        own=np.frombuffer(a_own, dtype=np.int32)[order],
        genus=np.frombuffer(a_ge, dtype=np.int32)[order],
        cell=np.frombuffer(a_cell, dtype=np.int32)[order],
        week=np.frombuffer(a_week, dtype=np.int32)[order].astype(np.int8),
        band=np.frombuffer(a_band, dtype=np.int32)[order].astype(np.int8))
    tmp.replace(path)
    return path.stat().st_size


def _aggregate(grid: Grid, tax: TaxaTable, a_own, a_sp, a_ge, a_cell, a_week, a_band) -> dict:
    n_units, C, B = len(tax.ids), grid.n_cells, grid.n_bands
    cell = np.frombuffer(a_cell, dtype=np.int32).astype(np.int64)
    week = np.frombuffer(a_week, dtype=np.int32).astype(np.int64)
    band = np.frombuffer(a_band, dtype=np.int32).astype(np.int64)
    dated = week >= 0
    effort_cell = np.bincount(cell, minlength=C).astype(np.int64)
    effort_bw = np.bincount(band[dated] * WEEKS + week[dated],
                            minlength=B * WEEKS).reshape(B, WEEKS).astype(np.int64)
    units, cells, weeks, bands = [], [], [], []
    for a in (a_own, a_sp, a_ge):
        u = np.frombuffer(a, dtype=np.int32).astype(np.int64)
        keep = u >= 0
        units.append(u[keep])
        cells.append(cell[keep])
        weeks.append(week[keep])
        bands.append(band[keep])
    u, c, w, b = (np.concatenate(x) if x else np.zeros(0, np.int64)
                  for x in (units, cells, weeks, bands))
    unit_total = np.bincount(u, minlength=n_units).astype(np.int64)
    keys, counts = np.unique(u * C + c, return_counts=True)
    pair_unit = keys // C
    ptr = np.zeros(n_units + 1, dtype=np.int64)
    np.cumsum(np.bincount(pair_unit, minlength=n_units), out=ptr[1:])
    d = w >= 0
    wkeys, wcounts = np.unique((u[d] * B + b[d]) * WEEKS + w[d], return_counts=True)
    return {
        "unit_taxon_id": np.asarray(tax.ids, dtype=np.int64),
        "unit_rank_level": np.asarray(tax.rank_level, dtype=np.float32),
        "unit_active": np.asarray(tax.active, dtype=bool),
        "unit_family_id": np.asarray(tax.family_id, dtype=np.int64),
        "unit_genus": np.asarray(tax.genus_unit, dtype=np.int32),
        "unit_species": np.asarray(tax.species_unit, dtype=np.int32),
        "unit_total": unit_total,
        "pair_ptr": ptr,
        "pair_cell": (keys % C).astype(np.int32),
        "pair_count": counts.astype(np.int32),
        "week_unit": (wkeys // (B * WEEKS)).astype(np.int32),
        "week_band": ((wkeys // WEEKS) % B).astype(np.int16),
        "week_week": (wkeys % WEEKS).astype(np.int16),
        "week_count": wcounts.astype(np.int32),
        "effort_cell": effort_cell,
        "effort_band_week": effort_bw,
    }


# --- reading the store --------------------------------------------------------------------

_RANK_WORDS = re.compile(r"\s+(?:var|subsp|ssp|f|forma|fo|subvar)\.?\s+", re.IGNORECASE)


def name_key(name: str) -> str:
    """Spellings of one name agree: folded quotes and spaces, case, and rank words
    ('var.', 'subsp.', 'f.'): iNat writes a variety without one."""
    s = names.fold_name(name)
    s = _RANK_WORDS.sub(" ", s)
    return " ".join(s.split()).lower()


@dataclass(frozen=True)
class Contribution:
    """One counted observation: the units it counted for, its cell, week (-1: none), band."""
    units: tuple[int, ...]
    cell: int
    week: int
    band: int


@dataclass(frozen=True)
class Mapping:
    species_unit: int        # -1: no species-level taxon on iNat to use
    genus_unit: int          # -1: no genus on iNat either
    how: str                 # species | synonym | genus-label | provisional | not-on-inat | none


class OccurrenceStore:
    def __init__(self, arrays: dict, path: Path | None = None):
        self.path = path
        self.meta = json.loads(bytes(arrays["meta"]).decode())
        self.grid = Grid(**self.meta["grid"])
        raw = bytes(arrays["names"]).decode()
        self.names = raw.split("\n") if raw else []
        for k in ("unit_taxon_id", "unit_rank_level", "unit_active", "unit_family_id",
                  "unit_genus", "unit_species", "unit_total", "pair_ptr", "pair_cell",
                  "pair_count", "week_unit", "week_band", "week_week", "week_count",
                  "effort_cell", "effort_band_week", "excluded_hashes"):
            setattr(self, k, arrays[k])
        self._by_key: dict[str, list[int]] | None = None
        self._by_epithet: dict[tuple, list[int]] | None = None
        self._loo: dict | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "OccurrenceStore":
        path = Path(path or default_store_path())
        if not path.is_file():
            raise FileNotFoundError(
                f"no occurrence store at {path}: build one with `mv build-occurrence` "
                "(needs iNat's open-data observations and taxa files)")
        with np.load(path) as z:
            return cls({k: z[k] for k in z.files}, path)

    @property
    def n_units(self) -> int:
        return len(self.unit_taxon_id)

    @property
    def has_loo(self) -> bool:
        return self.path is not None and loo_path(self.path).is_file()

    def own_contribution(self, uuid: str | None) -> "Contribution | None":
        """What this observation added to the counts, or None when it wasn't counted
        (excluded, filtered out, or not in the export)."""
        if not uuid or not self.has_loo:
            return None
        if self._loo is None:
            with np.load(loo_path(self.path)) as z:
                self._loo = {k: z[k] for k in z.files}
        lo = self._loo
        x = np.uint64(uuid_hash(uuid))
        i = int(np.searchsorted(lo["hash"], x))
        if i >= len(lo["hash"]) or lo["hash"][i] != x:
            return None
        own, ge = int(lo["own"][i]), int(lo["genus"][i])
        sp = int(self.unit_species[own]) if own >= 0 else -1
        units = tuple(u for u in (own, sp, ge) if u >= 0)
        return Contribution(units, int(lo["cell"][i]), int(lo["week"][i]), int(lo["band"][i]))

    def excludes(self, uuids: Iterable[str]) -> list[bool]:
        """Whether each uuid was in the build's exclusion list."""
        h = self.excluded_hashes
        out = []
        for u in uuids:
            x = np.uint64(uuid_hash(u))
            i = int(np.searchsorted(h, x))
            out.append(i < len(h) and h[i] == x)
        return out

    def _index(self) -> dict[str, list[int]]:
        if self._by_key is None:
            self._by_key = {}
            for i, n in enumerate(self.names):
                self._by_key.setdefault(name_key(n), []).append(i)
        return self._by_key

    def _genus(self, word: str) -> int:
        hits = [i for i in self._index().get(name_key(word), [])
                if self.unit_rank_level[i] == GENUS_LEVEL]
        active = [i for i in hits if self.unit_active[i]]
        hits = active or hits
        if len(hits) == 1:
            return hits[0]
        with_obs = [i for i in hits if self.unit_total[i] > 0]
        return with_obs[0] if len(with_obs) == 1 else -1

    def resolve(self, label: str, genus: str = "") -> Mapping:
        """Vision's label -> the iNat taxa to read. Active names first; an inactive name
        maps to the one active species of the same epithet in the same family, if there
        is exactly one; a provisional name ("Amanita sp. 'IN01'") or a name iNat doesn't
        know has no species taxon: only its genus is used. Never a guess beyond that."""
        parts = names.parse_name(label)
        words = name_key(label).split()
        if parts.code is not None:
            return Mapping(-1, self._genus(parts.code.genus), "provisional")
        if len(words) == 1:
            g = self._genus(words[0])
            return Mapping(-1, g, "genus-label" if g >= 0 else "none")
        g_label = self._genus(genus or label.split()[0])
        hits = [i for i in self._index().get(name_key(label), [])
                if self.unit_rank_level[i] <= SPECIES_LEVEL]
        active = [i for i in hits if self.unit_active[i]]
        if len(active) == 1:
            s = active[0]
            return Mapping(s, self._genus_of(s, g_label), "species")
        if len(active) > 1:              # two active taxa of one name: no guess
            return Mapping(-1, g_label, "ambiguous")
        if hits:
            s = self._synonym(hits)
            if s >= 0:
                return Mapping(s, self._genus_of(s, g_label), "synonym")
        return Mapping(-1, g_label, "not-on-inat" if g_label >= 0 else "none")

    def _genus_of(self, s: int, fallback: int) -> int:
        g = int(self.unit_genus[s])
        return g if g >= 0 else fallback

    def _synonym(self, inactive: list[int]) -> int:
        if self._by_epithet is None:
            self._by_epithet = {}
            for j, n in enumerate(self.names):
                w = n.split()
                if self.unit_active[j] and len(w) > 1 and self.unit_rank_level[j] <= SPECIES_LEVEL:
                    key = (w[-1].lower(), int(self.unit_family_id[j]),
                           float(self.unit_rank_level[j]))
                    self._by_epithet.setdefault(key, []).append(j)
        found = set()
        for i in inactive:
            fam = int(self.unit_family_id[i])
            if fam:
                found.update(self._by_epithet.get(
                    (self.names[i].split()[-1].lower(), fam, float(self.unit_rank_level[i])), []))
        return found.pop() if len(found) == 1 else -1


def exclusion_uuids(conn) -> list[str]:
    """Every iNat observation uuid the manifest knows: Vision's own DNA records (the
    test and validation records among them) and, when that table exists, the frozen
    benchmark holdouts. An occurrence store built without them would count each test
    record at its own spot."""
    out = {r[0].strip().lower() for r in
           conn.execute("select uuid from inat_observations where uuid is not null and uuid != ''")}
    out |= set(holdout_uuids(conn))
    return sorted(out)


def holdout_uuids(conn) -> list[str]:
    if not _has_table(conn, "benchmark_holdouts"):
        return []
    cols = {r[1] for r in conn.execute("pragma table_info(benchmark_holdouts)")}
    if "uuid" in cols:
        rows = conn.execute("select uuid from benchmark_holdouts where uuid is not null")
    elif "observation_id" in cols:
        rows = conn.execute("select o.uuid from benchmark_holdouts b join inat_observations o "
                            "on o.observation_id = cast(b.observation_id as text) "
                            "where o.uuid is not null")
    else:
        return []
    return [r[0].strip().lower() for r in rows if r[0]]


def _has_table(conn, name: str) -> bool:
    return conn.execute("select 1 from sqlite_master where type = 'table' and name = ?",
                        (name,)).fetchone() is not None
