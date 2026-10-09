"""Location and season from iNat occurrence counts, with a wide berth for "out of range".

Steve's rule (2026-10-08): a species is pushed down for its location only when it
is essentially on the other side of the continent: no occurrence within
`radius_km` (default 1,500 km) of the query, and enough occurrences overall
(`min_occurrences`, default 20) for that absence to mean something. Inside that
berth, place and season only reweight gently, each capped.

Per species (an index group), for a place and date:

- out of range (strong, `out_of_range_penalty` nats): enough evidence and none of
  it within the radius. Evidence is the species' iNat occurrences plus its own
  DNA-verified reference records, so a DNA record nearby always vetoes the
  penalty. A species iNat doesn't know (a provisional name) has only its DNA
  records, under the same rule. The same rule at genus level: no occurrence of
  its genus within the radius. It never applies where fungi go unobserved
  (fewer than `min_local_effort` observations of any fungus within the radius),
  or outside the store's map.
- density (gentle): log of how much more often the species is observed near here
  than fungi in general are (a Gaussian kernel over grid cells), pulled towards
  its genus by `shrink` observations' worth, capped, times `density_weight`.
- season (gentle): the same for the week of the year, within latitude bands near
  the query, times `season_weight`.

A species with no iNat taxon falls back to its genus for density and season,
else neutral. The score is computed from counts per grid cell and from the
records' coordinates, and only ever returns scores, never a location.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import numpy as np

from . import config
from .occurrence import WEEKS, OccurrenceStore
from .prior import LOGPROB_CONFIDENCE_TEMPERATURE, Context, haversine_km

PARAMS_NAME = Path("occurrence") / "params.json"


@dataclass
class OccParams:
    radius_km: float = 1500.0
    min_occurrences: int = 20
    min_local_effort: int = 50
    out_of_range_penalty: float = 6.0     # nats; 50 is in effect exclusion
    density_bandwidth_km: float = 150.0
    density_weight: float = 0.5
    density_cap: float = float(np.log(5.0))
    season_bandwidth_weeks: float = 2.0
    season_weight: float = 0.5
    season_cap: float = float(np.log(5.0))
    band_window: int = 1                  # latitude bands either side used for season
    shrink: float = 20.0                  # observations' worth of weight given to the genus
    photo_temperature: float = 0.02       # similarity methods: softmax temperature (AsLogProb)
    # Atlas only: log-factor per rank bin (0-9 ... 90-99, 100), learned on dev
    # (atlasrange.estimate_rank_factor); None = no place term from Atlas.
    rank_factor: list | None = None
    # Confidence: softmax(score / T) per rank; a float for all ranks or {rank: T}.
    confidence_temperature: float | dict = LOGPROB_CONFIDENCE_TEMPERATURE
    provenance: dict = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict) -> "OccParams":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


def default_params_path() -> Path:
    return config.DATA_DIR / PARAMS_NAME


def load_params(path: Path | None = None) -> OccParams:
    """The tuned parameters (`mv tune-occurrence`), else the defaults."""
    path = Path(path or default_params_path())
    if path.is_file():
        return OccParams.from_dict(json.loads(path.read_text(encoding="utf-8")))
    return OccParams()


def save_params(params: OccParams, path: Path | None = None) -> Path:
    path = Path(path or default_params_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(params.to_dict(), indent=2), encoding="utf-8")
    return path


class TuningRefused(ValueError):
    pass


def check_store_excludes(store: OccurrenceStore, uuids: list[str | None],
                         allow_missing: bool = False) -> dict:
    """Refuse when a validation observation would count towards its own score: the
    store counts it and has no leave-one-out index to take it back out at scoring.
    Records with no known uuid can't be checked: refused unless allow_missing."""
    known = [u for u in uuids if u]
    missing = len(uuids) - len(known)
    counted = [u for u, ok in zip(known, store.excludes(known)) if not ok]
    if counted and not store.has_loo:
        raise TuningRefused(
            f"the occurrence store counts {len(counted)} of the validation observations "
            "(their uuids were not excluded when it was built) and has no leave-one-out "
            "index beside it, so each would be found at its own spot. Rebuild it (the "
            "build writes the index) or exclude them with --exclude-uuids.")
    if missing and not allow_missing:
        raise TuningRefused(f"{missing} validation records have no known iNat uuid, so the "
                            "store can't be shown to leave them out; give uuids (a uuid "
                            "column) or pass --allow-missing-uuids")
    return {"checked": len(known), "missing_uuid": missing,
            "excluded_at_build": len(known) - len(counted),
            "taken_out_at_scoring": len(counted)}


def gaussian(d: np.ndarray, bandwidth: float) -> np.ndarray:
    return np.exp(-0.5 * (d / bandwidth) ** 2)


@dataclass
class Parts:
    """A place and date's components, before weights and caps (for tuning)."""
    out_of_range: dict           # radius_km -> (n_groups,) bool
    density: np.ndarray | None   # (n_groups,) log-ratio, uncapped; None = no place
    season: np.ndarray | None    # (n_groups,) log-ratio, uncapped; None = no date


class RangeSource:
    """Where a species lives, as the +occ methods and the tuning harness use it.

    A source answers, per index group, for a place and date (`parts`): is it out of
    range here (one answer per radius asked), a gentle log-score for the place, and
    one for the week. Weights, caps and the penalty are applied here, the same for
    every source, so two sources (iNat occurrences; MycoMap Atlas's modelled ranges
    next) can be compared on the same validation set with the same harness.
    `None` for a component means the source has nothing to say (no place, no date,
    or a source without phenology)."""
    name = "range"
    params: OccParams
    n_groups: int

    def fit(self, records, species: list[str]) -> None:
        raise NotImplementedError

    def parts(self, ctx: Context | None, radii: tuple[float, ...] | None = None) -> Parts:
        raise NotImplementedError

    def mapping_summary(self) -> dict:
        return {}

    def leak_check(self, uuids: list[str | None], allow_missing: bool = False) -> dict:
        """Show the source was not built from these (validation) observations, or
        raise TuningRefused. A source that can't show it may not be tuned."""
        raise TuningRefused(f"the {self.name} source can't show it leaves the "
                            "validation records out")

    def combine(self, parts: Parts, params: OccParams | None = None) -> np.ndarray:
        p = params or self.params
        out = np.zeros(self.n_groups)
        if parts.density is not None and p.density_weight:
            out += p.density_weight * np.clip(parts.density, -p.density_cap, p.density_cap)
        if parts.season is not None and p.season_weight:
            out += p.season_weight * np.clip(parts.season, -p.season_cap, p.season_cap)
        oor = parts.out_of_range.get(p.radius_km)
        if oor is not None:
            out -= p.out_of_range_penalty * oor
        return out

    def log_prior(self, ctx: Context | None) -> np.ndarray:
        return self.combine(self.parts(ctx))


class OccurrencePrior(RangeSource):
    """The range source from iNat occurrence counts (an OccurrenceStore)."""
    name = "inat-occurrence"

    def __init__(self, store: OccurrenceStore, params: OccParams | None = None):
        self.store = store
        self.params = params or OccParams()

    def fit(self, records, species: list[str]) -> None:
        """records: the reference (DNA-verified) records; species: index group order."""
        st = self.store
        self.n_groups = len(species)
        genus_of = {}
        for r in records:
            genus_of.setdefault(r.unit, r.genus or "")
        maps = [st.resolve(s, genus_of.get(s, "")) for s in species]
        self.mapping = maps
        sp = np.array([m.species_unit for m in maps], dtype=np.int64)
        ge = np.array([m.genus_unit for m in maps], dtype=np.int64)
        needed = np.unique(np.concatenate([sp[sp >= 0], ge[ge >= 0]]))
        local = {int(u): i for i, u in enumerate(needed.tolist())}
        self.local = local
        self.g_sp = np.array([local.get(int(u), -1) for u in sp], dtype=np.int64)
        self.g_ge = np.array([local.get(int(u), -1) for u in ge], dtype=np.int64)
        n_local = len(needed)
        self.total = st.unit_total[needed].astype(np.float64)
        # Occupied cells and the needed units' (cell, count) pairs on them.
        occ = np.flatnonzero(st.effort_cell > 0)
        self.cell_lat, self.cell_lon = st.grid.centres(occ)
        self.cell_effort = st.effort_cell[occ].astype(np.float64)
        self.total_effort = max(float(self.cell_effort.sum()), 1.0)
        pos_of_cell = np.full(st.grid.n_cells, -1, dtype=np.int64)
        pos_of_cell[occ] = np.arange(len(occ))
        self.pos_of_cell = pos_of_cell
        starts, ends = st.pair_ptr[needed], st.pair_ptr[needed + 1]
        lengths = ends - starts
        idx = (np.repeat(starts - np.cumsum(lengths) + lengths, lengths)
               + np.arange(int(lengths.sum()))) if len(needed) else np.zeros(0, np.int64)
        self.pair_local = np.repeat(np.arange(n_local), lengths)
        self.pair_pos = pos_of_cell[st.pair_cell[idx]]
        self.pair_count = st.pair_count[idx].astype(np.float64)
        # Season: (local unit, band, week) counts, and all fungi per (band, week).
        B = st.grid.n_bands
        self.weeks = np.zeros((n_local, B, WEEKS), dtype=np.float32)
        lu = np.array([local.get(int(u), -1) for u in st.week_unit.tolist()], dtype=np.int64) \
            if n_local else np.zeros(0, np.int64)
        k = lu >= 0
        np.add.at(self.weeks, (lu[k], st.week_band[k].astype(np.int64),
                               st.week_week[k].astype(np.int64)), st.week_count[k])
        self.effort_weeks = st.effort_band_week.astype(np.float64)
        # DNA evidence: the reference records' coordinates per group.
        gpos = {s: i for i, s in enumerate(species)}
        rows = [(gpos[r.unit], r.latitude, r.longitude) for r in records
                if r.unit in gpos and r.latitude is not None and r.longitude is not None]
        self.dna_group = np.array([r[0] for r in rows], dtype=np.int64)
        self.dna_lat = np.array([r[1] for r in rows], dtype=np.float64)
        self.dna_lon = np.array([r[2] for r in rows], dtype=np.float64)
        self.dna_total = np.bincount(self.dna_group, minlength=self.n_groups).astype(np.float64)

    # --- components ---------------------------------------------------------------------

    def _unit_values(self, per_local: np.ndarray, which: np.ndarray, fill: float) -> np.ndarray:
        out = np.full(self.n_groups, fill, dtype=np.float64)
        ok = which >= 0
        out[ok] = per_local[which[ok]]
        return out

    def _shrunk_log_ratio(self, k_local: np.ndarray, n_local: np.ndarray,
                          expected_share: float) -> np.ndarray:
        """Per group: log((k/n) / expected_share), the species pulled to its genus and the
        genus to 1 by `shrink` observations' worth. No taxon at all: 0."""
        s = self.params.shrink
        ratio = k_local / max(expected_share, 1e-12)
        r_unit_alone = (ratio + s * 1.0) / (n_local + s)                 # towards 1
        r_ge = self._unit_values(r_unit_alone, self.g_ge, 1.0)
        k_sp = self._unit_values(ratio, self.g_sp, 0.0)
        n_sp = self._unit_values(n_local, self.g_sp, 0.0)
        r_sp = (k_sp + s * r_ge) / (n_sp + s)
        return np.log(np.clip(r_sp, 1e-6, None))

    def parts(self, ctx: Context | None, radii: tuple[float, ...] | None = None) -> Parts:
        p = self.params
        radii = radii or (p.radius_km,)
        none = Parts({r: np.zeros(self.n_groups, dtype=bool) for r in radii}, None, None)
        if ctx is None:
            return none
        has_place = ctx.latitude is not None and ctx.longitude is not None
        if has_place and not self.store.grid.contains(ctx.latitude, ctx.longitude):
            return none                 # off the map: no information, not "absent"
        out_of_range, density, season = none.out_of_range, None, None
        # The query's own iNat observation, when the store counted it, comes back out.
        own = self.store.own_contribution(ctx.uuid)
        mine = np.array([self.local[u] for u in own.units if u in self.local]
                        if own else [], dtype=np.int64)
        at = int(self.pos_of_cell[own.cell]) if own else -1
        total, total_effort = self.total, self.total_effort
        if own:
            total = total.copy()
            total[mine] -= 1
            total_effort -= 1
        if has_place:
            d = haversine_km(ctx.latitude, ctx.longitude, self.cell_lat, self.cell_lon)
            slack = self.store.grid.cell_deg * 111.2 * 0.71      # a cell's half diagonal
            d_dna = haversine_km(ctx.latitude, ctx.longitude, self.dna_lat, self.dna_lon)
            out_of_range = {}
            for r in radii:
                within = d <= r + slack
                self_near = float(within[at]) if at >= 0 else 0.0
                if self.cell_effort[within].sum() - self_near < p.min_local_effort:
                    out_of_range[r] = np.zeros(self.n_groups, dtype=bool)
                    continue
                near_local = np.bincount(self.pair_local, minlength=len(self.total),
                                         weights=self.pair_count * within[self.pair_pos])
                near_local[mine] -= self_near
                dna_near = np.bincount(self.dna_group, weights=(d_dna <= r).astype(float),
                                       minlength=self.n_groups)
                # Species: iNat occurrences plus its own DNA records.
                tot = self._unit_values(total, self.g_sp, 0.0) + self.dna_total
                near = self._unit_values(near_local, self.g_sp, 0.0) + dna_near
                out = (tot >= p.min_occurrences) & (near == 0)
                # Genus: none of the genus within the radius either (its DNA records too).
                g_tot = self._unit_values(total, self.g_ge, 0.0)
                g_near = self._unit_values(near_local, self.g_ge, 0.0) + dna_near
                out |= (self.g_ge >= 0) & (g_tot >= p.min_occurrences) & (g_near == 0)
                out_of_range[r] = out
            k = gaussian(d, p.density_bandwidth_km)
            self_k = float(k[at]) if at >= 0 else 0.0
            local_effort = float((k * self.cell_effort).sum()) - self_k
            if local_effort > 0:
                k_local = np.bincount(self.pair_local, minlength=len(self.total),
                                      weights=self.pair_count * k[self.pair_pos])
                k_local[mine] -= self_k
                density = self._shrunk_log_ratio(k_local, total, local_effort / total_effort)
        week = None
        doy = ctx.day_of_year
        if doy is not None:
            week = min((doy - 1) // 7, WEEKS - 1)
        if week is not None:
            B = self.store.grid.n_bands
            if has_place:
                b = self.store.grid.band(ctx.latitude)
                bands = slice(max(0, b - p.band_window), min(B, b + p.band_window + 1))
            else:
                bands = slice(0, B)
            sp_weeks = self.weeks[:, bands, :].sum(axis=1).astype(np.float64)   # (local, 53)
            all_weeks = self.effort_weeks[bands, :].sum(axis=0)
            if own and own.week >= 0 and bands.start <= own.band < bands.stop:
                sp_weeks[mine, own.week] -= 1
                all_weeks[own.week] -= 1
            dist = np.abs(np.arange(WEEKS) - week)
            kw = gaussian(np.minimum(dist, WEEKS - dist), p.season_bandwidth_weeks)
            e_total = all_weeks.sum()
            e_here = float(kw @ all_weeks)
            if e_total > 0 and e_here > 0:
                season = self._shrunk_log_ratio(sp_weeks @ kw, sp_weeks.sum(axis=1),
                                                e_here / e_total)
        return Parts(out_of_range, density, season)

    def mapping_summary(self) -> dict:
        from collections import Counter
        return dict(Counter(m.how for m in self.mapping))

    def leak_check(self, uuids: list[str | None], allow_missing: bool = False) -> dict:
        return check_store_excludes(self.store, uuids, allow_missing)


@dataclass(frozen=True)
class SourceKind:
    """How to make a range source, whether this machine has its data, and the method
    suffix it gives (nearest+occ). A MycoMap Atlas source registers here too."""
    make: object                 # (params, store_path | None) -> RangeSource (unfitted)
    ready: object                # () -> bool
    suffix: str


def _atlas(params, path=None):
    from .atlasrange import AtlasExport, AtlasRangeSource
    return AtlasRangeSource(AtlasExport.load(path), params)


def _atlas_then_inat(params, path=None):
    """Atlas where it has a strong map, iNat occurrences elsewhere (`path`: the store)."""
    from .atlasrange import AtlasExport, AtlasRangeSource, LayeredSource
    return LayeredSource(AtlasRangeSource(AtlasExport.load(), params),
                         OccurrencePrior(OccurrenceStore.load(path), params))


def _atlas_ready() -> bool:
    from .atlasrange import default_export_dir
    return (default_export_dir() / "grid.json").is_file()


RANGE_SOURCES: dict[str, SourceKind] = {
    "inat-occurrence": SourceKind(
        make=lambda params, path=None: OccurrencePrior(OccurrenceStore.load(path), params),
        ready=lambda: default_store_path().is_file(), suffix="occ"),
    # Stubs until Atlas's rebuilt release exports its "here" index (atlasrange.py): not
    # offered as methods yet, and tuning refuses them until Atlas states its training set.
    "atlas": SourceKind(make=_atlas, ready=_atlas_ready, suffix="atlas"),
    "atlas+inat-occurrence": SourceKind(
        make=_atlas_then_inat, ready=lambda: _atlas_ready() and default_store_path().is_file(),
        suffix="atlas+occ"),
}


def default_store_path() -> Path:
    from .occurrence import default_store_path as store_path
    return store_path()


class WithOccurrence:
    """A log-probability method plus a range source (`<base>+occ` for iNat occurrences)."""
    needs_context = True

    def __init__(self, base_cls, similarity: bool = False, store: OccurrenceStore | None = None,
                 params: OccParams | None = None, source: str = "inat-occurrence"):
        self.params = params or load_params()
        if similarity:
            from .methods import AsLogProb
            self.base = AsLogProb(base_cls, temperature=self.params.photo_temperature)
        else:
            self.base = base_cls()
        self.source = source
        self.name = f"{self.base.name}+{RANGE_SOURCES[source].suffix}"
        self._store = store

    @staticmethod
    def ready(source: str = "inat-occurrence") -> bool:
        """Whether this machine has the source's data (methods.method_ready)."""
        return bool(RANGE_SOURCES[source].ready())

    @property
    def confidence_temperature(self):
        return self.params.confidence_temperature

    def fit(self, vectors: np.ndarray, index, records=None, state: dict | None = None) -> None:
        if hasattr(self.base, "state"):
            self.base.fit(vectors, index, state=state)
        else:
            self.base.fit(vectors, index)
        if self._store is not None:
            self.prior = OccurrencePrior(self._store, self.params)
        else:
            self.prior = RANGE_SOURCES[self.source].make(self.params)
        self.prior.fit(records or [], index.species)

    @property
    def trainable(self) -> bool:
        return hasattr(self.base, "state")

    def state(self) -> dict:
        return self.base.state()

    def species_scores(self, query: np.ndarray, context: Context | None = None) -> np.ndarray:
        return self.base.species_scores(query) + self.prior.log_prior(context)

    def photo_sims(self, query: np.ndarray) -> np.ndarray:
        return self.base.photo_sims(query)
