"""Tune the occurrence prior on a validation set, and only on one.

`mv tune-occurrence` grid-searches the radius, the out-of-range penalty, the
density and season weights and the photo temperature for species top-1 (then
genus top-1, then the gentler setting) on a validation set, fits the confidence
temperature per rank on the chosen setting, and writes the chosen values with
where they came from (data/occurrence/params.json, which the +occ methods read).

The validation set is a parameter:

- a CSV of observation ids (the held-out benchmark's dev.csv), with the photo
  scores precomputed per record (`--scores`, an .npz written by
  `save_scored_set`: the held-out records are not in the manifest's index), or
- `comparison:<id>`: a saved time-split comparison's test records (validated
  in the `test_days` after its cutoff), scored here from the stored vectors.

It refuses to tune:

- on any record registered in a `benchmark_holdouts` table (when the manifest
  has one; rows marked split = 'dev' are the tuning set and allowed), or in a
  set named with `--sealed`;
- when a validation observation would count towards its own score: every record
  is scored with its own iNat observation taken back out of the counts (by uuid,
  through the store's leave-one-out index), so a record without a known uuid is
  refused (unless --allow-missing-uuids), and so is a store that counts them
  with no index beside it.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from . import config, names
from .occprior import (EXCLUDE, RANGE_SOURCES, OccParams, TuningRefused, load_params,
                       save_params)
from .occurrence import OccurrenceStore  # noqa: F401
from .prior import Context

DEFAULT_GRID = {
    "radius_km": [1000.0, 1500.0, 2000.0, 3000.0],
    # Soft penalties and outright exclusion (EXCLUDE): dev picks (Steve, 2026-10-08).
    "out_of_range_penalty": [0.0, 2.0, 4.0, 6.0, 10.0, EXCLUDE],
    "density_weight": [0.0, 0.25, 0.5, 1.0],
    "season_weight": [0.0, 0.25, 0.5, 1.0],
    "photo_temperature": [0.015, 0.02, 0.03],
    # Scored both ways so the report shows dev with and without the genus rule; the
    # choice keeps it unless the search is told it may drop it (choose_genus_rule).
    "genus_rule": [True, False],
    # DNA-only out of range only where this many DNA records of any species lie
    # within the radius (OccParams.min_dna_effort).
    "min_dna_effort": [0, 250, 500, 1000, 2000],
}
DEV_SPLITS = ("dev", "tune", "tuning", "validation")


@dataclass
class ScoredSet:
    """Validation records with their photo scores against one reference index."""
    observation_ids: list[str]
    species: list[str]                    # the index's groups, in order
    scores: np.ndarray                    # (n, groups): cosine scores, or log-probabilities
    kind: str = "similarity"              # "similarity" | "logprob"
    truth: list[str] = field(default_factory=list)          # true group at species, or ''
    truth_genus: list[str] = field(default_factory=list)
    truth_family: list[str] = field(default_factory=list)
    latitude: list[float | None] = field(default_factory=list)
    longitude: list[float | None] = field(default_factory=list)
    observed_on: list[str | None] = field(default_factory=list)
    uuids: list[str | None] = field(default_factory=list)
    group_genus: list[str] = field(default_factory=list)    # per group; '' = none
    group_family: list[str] = field(default_factory=list)
    group_is_species: list[bool] = field(default_factory=list)


def save_scored_set(path: Path, s: ScoredSet) -> None:
    n = len(s.observation_ids)

    def strs(v, k=n):
        v = list(v) if v else [""] * k
        return np.array(["" if x is None else str(x) for x in v], dtype=str)

    def nums(v):
        v = list(v) if v else [None] * n
        return np.array([np.nan if x is None else float(x) for x in v], dtype=np.float64)
    g = len(s.species)
    np.savez_compressed(
        path, observation_id=strs(s.observation_ids), species=strs(s.species, g),
        scores=np.asarray(s.scores, dtype=np.float32), kind=np.array(s.kind),
        truth=strs(s.truth), truth_genus=strs(s.truth_genus), truth_family=strs(s.truth_family),
        latitude=nums(s.latitude), longitude=nums(s.longitude),
        observed_on=strs(s.observed_on), uuid=strs(s.uuids),
        group_genus=strs(s.group_genus, g), group_family=strs(s.group_family, g),
        group_is_species=np.array(s.group_is_species or [len(x.split()) > 1 for x in s.species],
                                  dtype=bool))


def load_scored_set(path: Path) -> ScoredSet:
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}

    def strs(k):
        return [str(x) for x in d[k].tolist()] if k in d else []

    def opt(k):
        return [None if x == "" else x for x in strs(k)]

    def nums(k):
        return [None if np.isnan(x) else float(x) for x in d[k].tolist()] if k in d else []
    return ScoredSet(strs("observation_id"), strs("species"), d["scores"],
                     str(d["kind"]) if "kind" in d else "similarity",
                     strs("truth"), strs("truth_genus"), strs("truth_family"),
                     nums("latitude"), nums("longitude"), opt("observed_on"), opt("uuid"),
                     strs("group_genus"), strs("group_family"),
                     d["group_is_species"].tolist() if "group_is_species" in d else [])


# --- guards -------------------------------------------------------------------------------

def read_ids(path: Path) -> list[str]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = csv.DictReader(f)
        col = next((c for c in (rows.fieldnames or []) if c.strip().lower()
                    in ("observation_id", "id", "inat_id")), None)
        if col is None:
            raise ValueError(f"{path} has no observation_id column")
        return [r[col].strip() for r in rows if (r.get(col) or "").strip()]


def sealed_files(records: str, extra: list[Path]) -> list[Path]:
    """The sealed sets named with --sealed. (The 2026-10-08 held-out benchmark is no
    longer sealed, Steve 2026-10-08: its test.csv may be scored and checked. A future
    paper set is registered in benchmark_holdouts, which is always checked.)"""
    return [Path(p) for p in extra]


def check_split(records: str, ids: list[str], searching: bool, allow_test: bool = False) -> str | None:
    """Which split of a benchmark (split.json beside the records CSV) these records are,
    and a refusal to search (tune) on its test split without allow_test: the set is no
    longer sealed (Steve, 2026-10-08), but tuning happens on dev. Records overlapping a
    test.csv beside it count as the test split."""
    if records.startswith("comparison:"):
        return None
    folder = Path(records).parent
    split_json = folder / "split.json"
    if not split_json.is_file():
        return None
    split = json.loads(split_json.read_text(encoding="utf-8"))
    sha = ids_sha(ids)
    which = next((name for name, v in split.items()
                  if isinstance(v, dict) and v.get("sha256_ids") == sha), None)
    if which is None and Path(records).stem in split:
        which = Path(records).stem
    test_csv = folder / "test.csv"
    if which != "test" and test_csv.is_file() and Path(records).resolve() != test_csv.resolve():
        if set(ids) & set(read_ids(test_csv)):
            which = "test"
    if which == "test" and searching and not allow_test:
        raise TuningRefused(f"{records} is (or overlaps) the test split in {split_json}: tune "
                            "on dev, and score test with --evaluate-only (or pass --allow-test)")
    return which or "unknown"


def check_not_sealed(ids: list[str], sealed: list[Path], conn: sqlite3.Connection | None) -> None:
    mine = set(ids)
    for path in sealed:
        overlap = mine & set(read_ids(path))
        if overlap:
            raise TuningRefused(f"{len(overlap)} of the {len(mine)} tuning records are in the "
                                f"sealed set {path}; never tune on it (e.g. {sorted(overlap)[:3]})")
    if conn is None:
        return
    frozen = frozen_holdout_ids(conn)
    overlap = mine & frozen
    if overlap:
        raise TuningRefused(f"{len(overlap)} tuning records are registered as frozen benchmark "
                            f"holdouts (benchmark_holdouts); never tune on them "
                            f"(e.g. {sorted(overlap)[:3]})")


def frozen_holdout_ids(conn: sqlite3.Connection) -> set[str]:
    """Observation ids registered as frozen holdouts: every row of benchmark_holdouts,
    except those whose `split` says they are the tuning set. A table with no id column
    can't be checked, so nothing may be tuned until it can."""
    if conn.execute("select 1 from sqlite_master where type = 'table' and "
                    "name = 'benchmark_holdouts'").fetchone() is None:
        return set()
    cols = {r[1] for r in conn.execute("pragma table_info(benchmark_holdouts)")}
    if "observation_id" not in cols:
        raise TuningRefused("benchmark_holdouts has no observation_id column, so tuning "
                            "can't prove it leaves the frozen holdouts out")
    where = ""
    if "split" in cols:
        marks = ",".join("?" for _ in DEV_SPLITS)
        where = f" where coalesce(lower(split), '') not in ({marks})"
    return {str(r[0]).strip() for r in conn.execute(
        f"select observation_id from benchmark_holdouts{where}",
        DEV_SPLITS if where else ())}


# --- building the validation set ----------------------------------------------------------

def read_records_csv(path: Path) -> dict[str, dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return {r["observation_id"].strip(): r for r in csv.DictReader(f)
                if (r.get("observation_id") or "").strip()}


def fill_from_csv(s: ScoredSet, path: Path, conn: sqlite3.Connection | None = None) -> ScoredSet:
    """Keep the CSV's records only, and fill what the scores file leaves blank (place,
    date, uuid, truth) from its columns; a uuid still missing comes from the manifest's
    inat_observations by observation id (the held-out fetch writes them there)."""
    rows = read_records_csv(path)
    uuid_of = {}
    if conn is not None:
        uuid_of = {str(o).strip(): (u or "").strip() for o, u in conn.execute(
            "select observation_id, uuid from inat_observations where uuid is not null")}
    keep = [i for i, oid in enumerate(s.observation_ids) if oid in rows]
    absent = set(rows) - set(s.observation_ids)
    if absent:
        print(f"  {len(absent)} records of {path.name} have no photo scores; left out")

    def col(v, i, *names, cast=str):
        if v and v[i] is not None and str(v[i]).strip():
            return v[i]
        r = rows[s.observation_ids[i]]
        for n in names:
            x = (r.get(n) or "").strip()
            if x:
                try:
                    return cast(x)
                except ValueError:
                    return None
        return None if cast is float else ""
    out = ScoredSet([s.observation_ids[i] for i in keep], s.species, s.scores[keep], s.kind,
                    group_genus=s.group_genus, group_family=s.group_family,
                    group_is_species=s.group_is_species)
    for i in keep:
        out.truth.append(col(s.truth, i, "label", "dna_name", "org_name", "com_name"))
        out.truth_genus.append(col(s.truth_genus, i, "genus_true", "genus")
                               or (out.truth[-1].split()[0] if out.truth[-1] else ""))
        out.truth_family.append(col(s.truth_family, i, "family_true", "family"))
        out.latitude.append(col(s.latitude, i, "lat", "latitude", cast=float))
        out.longitude.append(col(s.longitude, i, "lng", "lon", "longitude", cast=float))
        out.observed_on.append(col(s.observed_on, i, "observed_on") or None)
        out.uuids.append(col(s.uuids, i, "uuid", "observation_uuid")
                         or uuid_of.get(s.observation_ids[i]) or None)
    return out


def comparison_set(conn: sqlite3.Connection, comparison_id: str, base: str = "nearest",
                   backbone: str | None = None, embeddings_root=None):
    """(ScoredSet, reference records) for a saved comparison's test records, re-split at
    its cutoff (newer records since then are left out of both sides)."""
    from datetime import date, timedelta

    from .embed import load_embeddings
    from .evaluate import (SCOREBOARD_SCHEMA, build_index, comparison_backbones, load_records,
                           truth, with_rows)
    from .methods import NearestSpecimen, SpeciesMean
    conn.executescript(SCOREBOARD_SCHEMA)
    row = conn.execute("select cutoff, test_days, n_test from eval_runs where comparison_id = ? "
                       "limit 1", (comparison_id,)).fetchone()
    if not row:
        raise ValueError(f"no comparison {comparison_id!r} on the scoreboard")
    cutoff, test_days, n_test = row
    backbone = backbone or comparison_backbones(conn, comparison_id)[0]
    ids, vecs = load_embeddings(conn, backbone, embeddings_root / backbone
                                if embeddings_root else None)
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    records = with_rows(load_records(conn, {int(p): int(p) for p in ids.tolist()}), row_of)
    end = (date.fromisoformat(cutoff) + timedelta(days=test_days)).isoformat()
    ref = [r for r in records if not (r.validated_on and r.validated_on > cutoff)]
    test = [r for r in records if r.validated_on and cutoff < r.validated_on <= end]
    print(f"  comparison {comparison_id}: {len(test)} test records now "
          f"(it scored {n_test}), {len(ref)} reference")
    index = build_index(ref)
    model = {"nearest": NearestSpecimen, "species-mean": SpeciesMean}[base]()
    model.fit(vecs, index)
    scores = np.stack([model.species_scores(vecs[r.photo_rows]) for r in test]).astype(np.float32)
    uuid_of = dict(conn.execute("select observation_id, uuid from inat_observations"))
    lab = {rank: (index.labels[rank], index.label_of[rank]) for rank in ("genus", "family")}

    def group_label(rank):
        names, of = lab[rank]
        return [names[i] if i >= 0 else "" for i in of.tolist()]
    s = ScoredSet([r.observation_id for r in test], list(index.species), scores, "similarity",
                  [truth(r, "species") for r in test], [r.genus for r in test],
                  [r.family for r in test], [r.latitude for r in test],
                  [r.longitude for r in test], [r.observed_on for r in test],
                  [uuid_of.get(r.observation_id) for r in test],
                  group_label("genus"), group_label("family"),
                  (index.label_of["species"] >= 0).tolist())
    return s, ref


def reference_records(conn: sqlite3.Connection, leave_out: set[str]):
    """Every usable DNA record in the manifest except the validation ones: the DNA
    evidence (and veto) for the prior."""
    from .evaluate import load_records
    photos = {int(p): int(p) for (p,) in conn.execute("select photo_id from observation_photos")}
    return [r for r in load_records(conn, photos) if r.observation_id not in leave_out]


# --- the search ---------------------------------------------------------------------------

def _cand(s: ScoredSet, i: int, pos: dict[str, int], top_k: int) -> tuple[np.ndarray, int]:
    row = s.scores[i]
    k = min(top_k, len(row))
    cand = np.argpartition(-row, k - 1)[:k]
    t = pos.get(s.truth[i], -1) if s.truth else -1
    if t >= 0 and t not in set(cand.tolist()):
        cand = np.append(cand, t)
    return cand, t


def grid_search(prior, s: ScoredSet, grid: dict | None = None,
                top_k: int = 500, log=print, without: set[str] | None = None,
                choose_genus_rule: bool | None = True, temperatures: dict | None = None) -> dict:
    """Score every combination of the grid on `s`; return the table and the choice.
    `prior` is any fitted occprior.RangeSource, so sources compare on the same records.
    choose_genus_rule: True keeps the genus rule in the choice, False leaves it out,
    None lets the search pick; the report shows the best with and without either way."""
    grid = {**DEFAULT_GRID, **(grid or {})}
    if s.kind != "similarity":
        grid["photo_temperature"] = [None]
    radii = tuple(float(r) for r in grid["radius_km"])
    p0 = prior.params
    n, G = len(s.observation_ids), len(s.species)
    pos = {name: i for i, name in enumerate(s.species)}
    is_sp = np.array(s.group_is_species or [len(x.split()) > 1 for x in s.species], dtype=bool)
    genus_names = sorted({g for g in s.group_genus if g} | {g for g in s.truth_genus if g})
    gid = {g: i for i, g in enumerate(genus_names)}
    group_gid = np.array([gid.get(g, -1) for g in (s.group_genus or
                          [x.split()[0] for x in s.species])], dtype=np.int64)
    K = min(top_k, G) + 1
    S = np.full((n, K), -np.inf, dtype=np.float64)
    D = np.zeros((n, K))
    Se = np.zeros((n, K))
    O = {r: np.zeros((n, K), dtype=bool) for r in radii}
    OG = {r: np.zeros((n, K), dtype=bool) for r in radii}       # the genus rule's own
    OD = {r: np.zeros((n, K), dtype=bool) for r in radii}       # out on DNA records only
    DE = {r: np.zeros(n) for r in radii}                        # DNA records within r
    SP = np.zeros((n, K), dtype=bool)
    GG = np.full((n, K), -2, dtype=np.int64)
    truth_at = np.full(n, -1, dtype=np.int64)
    # A one-word truth ("Russula") is no species: scored at genus only.
    has_sp = np.array([len(t.split()) > 1 for t in (s.truth or [""] * n)])
    truth_g = np.array([gid.get(g, -3) if g else -3 for g in (s.truth_genus or [""] * n)])
    places = dates = 0
    for i in range(n):
        cand, t = _cand(s, i, pos, top_k)
        m = len(cand)
        ctx = Context(s.latitude[i] if s.latitude else None,
                      s.longitude[i] if s.longitude else None,
                      s.observed_on[i] if s.observed_on else None,
                      s.uuids[i] if s.uuids else None)       # its own observation comes out
        parts = prior.parts(ctx, radii)
        places += parts.density is not None
        dates += parts.season is not None
        S[i, :m] = s.scores[i, cand]
        if parts.density is not None:
            D[i, :m] = np.clip(parts.density[cand], -p0.density_cap, p0.density_cap)
        if parts.season is not None:
            Se[i, :m] = np.clip(parts.season[cand], -p0.season_cap, p0.season_cap)
        for r in radii:
            O[r][i, :m] = parts.out_of_range[r][cand]
            if parts.genus_out_of_range:
                OG[r][i, :m] = parts.genus_out_of_range[r][cand]
            if parts.dna_out_of_range:
                OD[r][i, :m] = parts.dna_out_of_range[r][cand]
                DE[r][i] = parts.dna_effort[r]
        SP[i, :m] = is_sp[cand]
        GG[i, :m] = group_gid[cand]
        if t >= 0:
            truth_at[i] = int(np.flatnonzero(cand == t)[0])
    log(f"  {n} records: {places} with a usable place, {dates} with a date")
    rows_ = np.arange(n)

    def combined(T, r, pen, wd, ws, gr=True, nd=0):
        r = float(r)
        out = O[r] | (OD[r] & (DE[r] >= nd)[:, None])
        if gr:
            out = out | OG[r]
        return (S / T if T else S) + wd * D + ws * Se - pen * out

    def right(z):
        sp_best = np.argmax(np.where(SP, z, -np.inf), axis=1)
        return (has_sp & (truth_at >= 0) & (sp_best == truth_at),
                GG[rows_, np.argmax(z, axis=1)] == truth_g)

    def rates(sp_right, ge_right, keep):
        return {"n": int(keep.sum()),
                "species_top1": round(float(sp_right[keep].sum() / max(has_sp[keep].sum(), 1)), 4),
                "genus_top1": round(float(ge_right[keep].mean()) if keep.any() else 0.0, 4)}

    table = []
    keys = ("photo_temperature", "radius_km", "out_of_range_penalty", "density_weight",
            "season_weight", "genus_rule", "min_dna_effort")
    for T, r, pen, wd, ws, gr, nd in product(*(grid[k] for k in keys)):
        sp_right, ge_right = right(combined(T, r, pen, wd, ws, gr, nd))
        table.append({"photo_temperature": T, "radius_km": float(r),
                      "out_of_range_penalty": pen, "exclusion": pen >= EXCLUDE,
                      "density_weight": wd, "season_weight": ws, "genus_rule": bool(gr),
                      "min_dna_effort": nd,
                      "species_top1": round(float(sp_right.sum() / max(has_sp.sum(), 1)), 4),
                      "genus_top1": round(float(ge_right.mean()), 4),
                      "species_right": int(sp_right.sum()), "genus_right": int(ge_right.sum())})

    def order(row):
        return (-row["species_right"], -row["genus_right"], row["out_of_range_penalty"],
                row["density_weight"] + row["season_weight"], abs(row["radius_km"] - 1500.0),
                abs((row["photo_temperature"] or 0.02) - 0.02), not row["genus_rule"],
                abs(row["min_dna_effort"] - 500))
    table.sort(key=order)
    allowed = [t for t in table
               if choose_genus_rule is None or t["genus_rule"] == choose_genus_rule] or table
    best = allowed[0]
    genus_rule = {label: next((t for t in table if t["genus_rule"] == flag), None)
                  for label, flag in (("with", True), ("without", False))}
    z = combined(*(best[k] for k in keys))
    every = np.ones(n, dtype=bool)
    photo_right, chosen_right = right(S), right(z)        # photo-only: no T changes its order
    photo_only = rates(*photo_right, every)
    out = {"n": n, "with_place": places, "with_date": dates, "best": best,
           "photo_only": photo_only,
           "calibration": calibrate(z, SP, GG, truth_at, truth_g, has_sp, temperatures),
           "genus_rule": genus_rule, "table": table}
    if without:
        # e.g. repeat finds: the same taxon at the same spot as another record.
        keep = np.array([oid not in without for oid in s.observation_ids])
        out["without"] = {"left_out": int((~keep).sum()),
                          "photo_only": rates(*photo_right, keep),
                          "chosen": rates(*chosen_right, keep)}
    return out


def calibrate(z, SP, GG, truth_at, truth_g, has_sp, fixed: dict | None = None) -> dict:
    """Confidence temperature per rank (softmax(score / T)) by likelihood, on the chosen
    scores; plus how often the top answer was stated at 99% or more. With `fixed`
    ({rank: T}, the saved values) nothing is fitted: the NLL and the rest are those of
    the saved temperature on these records."""
    from .evaluate import T_GRID, nll_by_temperature
    if fixed:
        grids = {rank: np.array([float(fixed.get(rank, fixed.get("species")))])
                 for rank in ("species", "genus")}
    else:
        grids = {"species": T_GRID, "genus": T_GRID}

    def nll_at(scores, t, grid):
        if not fixed:
            return nll_by_temperature(scores, t)
        zz = scores[None, :].astype(np.float64) / grid[:, None]
        m = zz.max(axis=1, keepdims=True)
        return (m + np.log(np.exp(zz - m).sum(axis=1, keepdims=True)))[:, 0] - zz[:, t]
    out = {}
    T_GRID = grids["species"]
    nll = np.zeros(len(T_GRID))
    used = 0
    for i in range(len(z)):
        if has_sp[i] and truth_at[i] >= 0 and SP[i, truth_at[i]]:
            keep = SP[i] & np.isfinite(z[i])
            idx = np.flatnonzero(keep)
            nll += nll_at(z[i, idx], int(np.flatnonzero(idx == truth_at[i])[0]), T_GRID)
            used += 1
    if used:
        out["species"] = _fit(nll, used, T_GRID)
    T_GRID = grids["genus"]
    nll = np.zeros(len(T_GRID))
    used = 0
    for i in range(len(z)):
        ok = np.isfinite(z[i]) & (GG[i] >= 0)
        gs = np.unique(GG[i, ok])
        if truth_g[i] not in set(gs.tolist()):
            continue
        best = np.full(len(gs), -np.inf)
        np.maximum.at(best, np.searchsorted(gs, GG[i, ok]), z[i, ok])
        nll += nll_at(best, int(np.searchsorted(gs, truth_g[i])), T_GRID)
        used += 1
    if used:
        out["genus"] = _fit(nll, used, T_GRID)
    for rank, c in out.items():
        T = c["temperature"]
        tops = []
        for i in range(len(z)):
            row = z[i][(SP[i] if rank == "species" else np.ones(z.shape[1], bool))
                       & np.isfinite(z[i])]
            if len(row):
                e = np.exp((row - row.max()) / T)
                tops.append(float(1.0 / e.sum()))
        c["share_stated_99"] = round(float(np.mean(np.array(tops) >= 0.99)), 4) if tops else None
        c["mean_top_confidence"] = round(float(np.mean(tops)), 4) if tops else None
        c["fitted_here"] = not fixed
    return out


def _fit(nll, used, grid) -> dict:
    j = int(np.argmin(nll))
    return {"temperature": float(grid[j]), "n": used, "nll": round(float(nll[j] / used), 4)}


def ids_sha(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest()


def tune(conn: sqlite3.Connection, records: str, scores: Path | None = None,
         sealed: list[Path] = (), store_path: Path | None = None, out: Path | None = None,
         grid: dict | None = None, top_k: int = 500, allow_missing_uuids: bool = False,
         base: str = "nearest", backbone: str | None = None, save: bool = True,
         source: str = "inat-occurrence", report_without: Path | None = None,
         evaluate_only: bool = False, choose_genus_rule: bool | None = True,
         allow_test: bool = False, log=print) -> dict:
    """The whole command: build the validation set, check it, search, save."""
    if records.startswith("comparison:"):
        cid = records.split(":", 1)[1]
        s, ref = comparison_set(conn, cid, base, backbone)
        origin = {"comparison": cid}
    else:
        if scores is None:
            raise ValueError("a records CSV needs --scores (per-record photo scores, an .npz "
                             "from save_scored_set): its records are not in the index")
        s = fill_from_csv(load_scored_set(Path(scores)), Path(records), conn)
        labels = names.manifest_labels(conn)        # .org spellings -> Vision's labels
        s.truth = [labels.get(t, t) if t else t for t in s.truth]
        ref = reference_records(conn, set(s.observation_ids))
        origin = {"records_csv": str(records), "scores": str(scores)}
    split = check_split(records, s.observation_ids, searching=not evaluate_only,
                        allow_test=allow_test)
    origin["split"] = split
    sealed = sealed_files(records, list(sealed))
    check_not_sealed(s.observation_ids, sealed, conn)
    params = load_params(out)
    prior = RANGE_SOURCES[source].make(params, store_path)
    excl = prior.leak_check(s.uuids, allow_missing_uuids)
    prior.fit(ref, s.species)
    log(f"  names: {prior.mapping_summary()}")
    temperatures = None
    if evaluate_only:           # score with the saved values: one point, nothing saved
        grid = {k: [getattr(params, k)] for k in DEFAULT_GRID}
        choose_genus_rule = params.genus_rule
        save = False
        t = params.confidence_temperature
        temperatures = t if isinstance(t, dict) else {r: t for r in ("species", "genus")}
    without = set(read_ids(report_without)) if report_without else None
    res = grid_search(prior, s, grid, top_k, log, without, choose_genus_rule, temperatures)
    b = res["best"]
    chosen = OccParams.from_dict({**params.to_dict(), **{
        k: b[k] for k in ("radius_km", "out_of_range_penalty", "density_weight",
                          "season_weight", "genus_rule", "min_dna_effort")}})
    if b["photo_temperature"]:
        chosen.photo_temperature = b["photo_temperature"]
    temps = {rank: c["temperature"] for rank, c in res["calibration"].items()}
    if temps and not evaluate_only:
        temps.setdefault("family", temps.get("genus", temps.get("species")))
        chosen.confidence_temperature = temps
    chosen.provenance = {
        **origin, "n_records": res["n"], "records_sha256": ids_sha(s.observation_ids),
        "sealed_checked": [str(p) for p in sealed], "store_exclusion_check": excl,
        "source": source,
        "store": ({"path": str(prior.store.path), "built_at": prior.store.meta.get("built_at"),
                   "sources": prior.store.meta.get("sources")}
                  if getattr(prior, "store", None) is not None else None),
        "base": base, "kind": s.kind, "top_k": top_k, "names": prior.mapping_summary(),
        "selection": "species top-1, then genus top-1, then the gentler setting",
        "result": {k: b[k] for k in ("species_top1", "genus_top1")},
        "photo_only": {k: res["photo_only"][k] for k in ("species_top1", "genus_top1")},
        "calibration": res["calibration"],
        "without": res.get("without"),
        "genus_rule": res["genus_rule"],
        "grid": {**DEFAULT_GRID, **(grid or {})},
        "top": res["table"][:15],
        "code_version": config.code_version(),
        "tuned_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if save:
        path = save_params(chosen, out)
        log(f"-> {path}")
        if getattr(prior, "mapping", None) is not None:
            names_csv = path.with_name(path.stem + "-names.csv")
            write_names_csv(prior, s.species, names_csv)
            log(f"-> {names_csv}")
    return {"params": chosen.to_dict(), "split": split, **{k: res.get(k) for k in (
        "n", "best", "photo_only", "calibration", "without", "genus_rule")}}


def write_names_csv(prior, species: list[str], path: Path,
                    records: dict[str, int] | None = None) -> dict:
    """Every label's mapping to iNat and how it was made (the report CSV)."""
    from collections import Counter
    st = prior.store
    tally = Counter()
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["label", "records", "how", "inat_species", "inat_species_obs",
                    "inat_genus", "inat_genus_obs", "epithet_guess", "epithet_guess_used"])
        for label, m in zip(species, prior.mapping):
            tally[m.how] += 1

            def nm(u):
                return st.names[u] if u >= 0 else ""

            def n_obs(u):
                return int(st.unit_total[u]) if u >= 0 else ""
            w.writerow([label, (records or {}).get(label, ""), m.how,
                        nm(m.species_unit), n_obs(m.species_unit),
                        nm(m.genus_unit), n_obs(m.genus_unit), nm(m.guess_unit),
                        "place/season only" if (m.guess_unit >= 0
                                                and prior.params.epithet_guesses) else ""])
    return dict(tally)


# --- commands (wired into cli.py) ---------------------------------------------------------

def _floats(v: str) -> list[float]:
    return [float(x) for x in v.split(",") if x.strip()]


def cmd_build_occurrence(conn, args) -> None:
    from .occurrence import BuildOptions, build, default_store_path
    opts = BuildOptions(margin_deg=args.margin_deg, cell_deg=args.cell_deg,
                        max_accuracy_m=None if args.max_accuracy_m <= 0 else args.max_accuracy_m,
                        with_slime_molds=args.with_slime_molds,
                        observed_before=args.observed_before)
    build(Path(args.observations), Path(args.taxa), Path(args.out or default_store_path()),
          opts, [Path(p) for p in args.exclude_uuids or []])


def cmd_occurrence_exclusions(conn, args) -> None:
    from .occurrence import exclusion_uuids
    uuids = exclusion_uuids(conn)
    out = Path(args.out or config.DATA_DIR / "occurrence" / "exclude-uuids.txt")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(uuids) + "\n", encoding="utf-8")
    print(f"{len(uuids):,} uuids -> {out}")


def cmd_occurrence_names(conn, args) -> None:
    """How every label in the manifest maps to iNat's taxa, and how (a CSV in reports/)."""
    from collections import Counter

    from .evaluate import load_records
    from .occprior import OccurrencePrior
    store = OccurrenceStore.load(Path(args.store) if args.store else None)
    photos = {int(p): int(p) for (p,) in conn.execute("select photo_id from observation_photos")}
    recs = load_records(conn, photos)
    units = Counter(r.unit for r in recs)
    prior = OccurrencePrior(store, load_params())
    prior.fit(recs, sorted(units))
    config.ensure_dirs()
    path = config.REPORTS_DIR / "occurrence-names.csv"
    tally = write_names_csv(prior, sorted(units), path, units)
    print(json.dumps({"labels": tally}, indent=2))
    print(f"-> {path}")


def cmd_tune_occurrence(conn, args) -> None:
    grid = {}
    for key, attr in (("radius_km", "radii"), ("out_of_range_penalty", "penalties"),
                      ("density_weight", "density_weights"),
                      ("season_weight", "season_weights"),
                      ("photo_temperature", "photo_temperatures")):
        if getattr(args, attr):
            grid[key] = _floats(getattr(args, attr))
    res = tune(conn, args.records, Path(args.scores) if args.scores else None,
               [Path(p) for p in args.sealed or []], Path(args.store) if args.store else None,
               Path(args.out) if args.out else None, grid, args.top_k, args.allow_missing_uuids,
               args.base, args.backbone, save=not args.dry_run, source=args.source,
               report_without=Path(args.report_without) if args.report_without else None,
               evaluate_only=args.evaluate_only,
               choose_genus_rule={"keep": True, "drop": False, "tune": None}[args.genus_rule],
               allow_test=args.allow_test)
    print(json.dumps({k: res[k] for k in ("n", "best", "photo_only", "calibration", "without",
                                          "genus_rule")}, indent=2))


def add_commands(sub) -> None:
    p = sub.add_parser("build-occurrence", help="iNat open-data fungi -> a grid of occurrence "
                                                "counts for the +occ methods (local files only)")
    p.add_argument("--observations", required=True, help="observations.csv.gz from iNat open data")
    p.add_argument("--taxa", required=True, help="taxa.csv.gz from iNat open data")
    p.add_argument("--out", help="default: <data>/occurrence/inat-fungi-na.npz")
    p.add_argument("--exclude-uuids", action="append",
                   help="file of observation uuids never to count (repeatable); see "
                        "mv occurrence-exclusions")
    p.add_argument("--with-slime-molds", action="store_true",
                   help="also keep Mycetozoa (under Protozoa)")
    p.add_argument("--cell-deg", type=float, default=0.5)
    p.add_argument("--margin-deg", type=float, default=5.0,
                   help="degrees added around the North America box")
    p.add_argument("--max-accuracy-m", type=float, default=25_000.0,
                   help="positional accuracy cap in metres (0 = none)")
    p.add_argument("--observed-before", help="ISO date: leave out observations from then on")
    p.set_defaults(occ_handler=cmd_build_occurrence)

    p = sub.add_parser("occurrence-exclusions",
                       help="write the uuids an occurrence store must not count (the "
                            "manifest's records and any benchmark holdouts)")
    p.add_argument("--out", help="default: <data>/occurrence/exclude-uuids.txt")
    p.set_defaults(occ_handler=cmd_occurrence_exclusions)

    p = sub.add_parser("occurrence-names", help="how the manifest's labels map to iNat taxa")
    p.add_argument("--store")
    p.set_defaults(occ_handler=cmd_occurrence_names)

    p = sub.add_parser("tune-occurrence", help="grid-search the occurrence prior on a "
                                               "validation set and save the chosen values")
    p.add_argument("--records", required=True,
                   help="a CSV of observation ids (e.g. the benchmark's dev.csv), or "
                        "comparison:<id> for a saved comparison's test records")
    p.add_argument("--scores", help="per-record photo scores (.npz, occtune.save_scored_set); "
                                    "needed with a CSV")
    p.add_argument("--sealed", action="append",
                   help="CSV of ids never to tune on (repeatable); benchmark_holdouts "
                        "is always checked")
    p.add_argument("--store", help="occurrence store (default: <data>/occurrence/...)")
    p.add_argument("--out", help="params file (default: <data>/occurrence/params.json)")
    p.add_argument("--base", default="nearest", choices=["nearest", "species-mean"])
    p.add_argument("--source", default="inat-occurrence", choices=sorted(RANGE_SOURCES),
                   help="where ranges come from (an Atlas source can register next to iNat's)")
    p.add_argument("--backbone", help="with comparison:<id>: which of its backbones")
    p.add_argument("--top-k", type=int, default=500, help="candidates kept per record")
    p.add_argument("--radii")
    p.add_argument("--penalties")
    p.add_argument("--density-weights")
    p.add_argument("--season-weights")
    p.add_argument("--photo-temperatures")
    p.add_argument("--allow-missing-uuids", action="store_true",
                   help="tune even when some records' uuids are unknown (not leak-checked)")
    p.add_argument("--dry-run", action="store_true", help="search but don't save")
    p.add_argument("--evaluate-only", action="store_true",
                   help="score the set with the saved values (e.g. test.csv after tuning on "
                        "dev.csv); nothing is searched or saved")
    p.add_argument("--allow-test", action="store_true",
                   help="search on a benchmark's test split (split.json); normally tune on "
                        "dev and score test with --evaluate-only")
    p.add_argument("--genus-rule", default="keep", choices=["keep", "drop", "tune"],
                   help="the genus-level out-of-range rule in the chosen values (the report "
                        "shows dev with and without it either way)")
    p.add_argument("--report-without",
                   help="CSV of ids (e.g. repeat finds) to also report the results without")
    p.set_defaults(occ_handler=cmd_tune_occurrence)
