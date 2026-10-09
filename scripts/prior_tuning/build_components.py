"""Every prior term per record and candidate, for the declared grid (priortune.py).

usage: python build_components.py <snapshot.sqlite> <sets_dir> <split>
Reads <sets_dir>/<split>-scores.npz (score_sets.py), writes
<sets_dir>/<split>-components-<base>.npz for base in (nearest_mean, nearest), plus
<split>-meta.npz. CPU only, against the manifest SNAPSHOT.
"""
import os
import sqlite3
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"       # the GPU belongs to observation-sets
os.environ.setdefault("OPENBLAS_NUM_THREADS", "12")
sys.modules["torch"] = None   # CPU numpy only: torch + OpenBLAS threads crashed (access violation)
import numpy as np

from mycomap_vision import config, heldout
from mycomap_vision import priortune as pt
from mycomap_vision.evaluate import build_index, load_records
from mycomap_vision.occprior import OccParams, OccurrencePrior
from mycomap_vision.occurrence import OccurrenceStore
from mycomap_vision.prior import Context, RangeSeasonPrior, day_distance, haversine_km
from mycomap_vision.heldout_report import DEPTH_BUCKETS, bucket

BENCH = config.DATA_DIR / "benchmarks" / "heldout-2026-10-08"
heldout.bench_dir = lambda conn, name: BENCH


def main():
    snap, sets, split = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    part, nparts = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)
    t0 = time.time()
    z = dict(np.load(sets / f"{split}-scores.npz"))
    n_all = len(z["observation_id"])
    lo, hi = n_all * part // nparts, n_all * (part + 1) // nparts
    for k in ("observation_id", "nearest", "nearest_mean", "n_photos", "max_sim", "truth_name",
              "truth", "truth_genus", "truth_family", "truth_guest", "org_lat", "org_lng",
              "inat_lat", "inat_lng", "observed_on", "uuid", "user_id"):
        z[k] = z[k][lo:hi]
    tag = f".part{part}of{nparts}" if nparts > 1 else ""
    species = z["species"].tolist()
    conn = sqlite3.connect(snap)
    photos = {int(p): int(p) for (p,) in conn.execute(
        "select photo_id from embeddings where backbone = 'bioclip-2-ft-20261007-165400'")}
    records = load_records(conn, photos)
    index = build_index(records)
    assert index.species == species, "group order differs from the scored set"
    print(f"{len(records):,} reference records [{time.time() - t0:.0f}s]", flush=True)

    # Truth positions by the report's rule (Labeller.same: label, then name_key).
    lab = heldout.Labeller(conn, extra=z["truth_name"].tolist())
    key = lambda name: heldout.name_key(lab.label(name)) if name else ""     # noqa: E731
    is_sp = z["group_is_species"]
    sp_pos = {}
    for j, s in enumerate(species):
        if is_sp[j]:
            sp_pos.setdefault(key(s), j)
    genus_names = sorted({g for g in z["group_genus"].tolist() if g})
    gkey = {}
    for i, g in enumerate(genus_names):
        gkey.setdefault(key(g), i)
    group_gid = np.array([gkey.get(key(g), -1) if g else -1 for g in z["group_genus"].tolist()])
    n = len(z["observation_id"])
    truth = z["truth"].tolist()
    truth_sp = np.array([sp_pos.get(key(t), -1) if t else -1 for t in truth])
    truth_g = np.array([gkey.get(key(g), -3) if g else -3 for g in z["truth_genus"].tolist()])
    has_sp = np.array([bool(t) for t in truth])
    guest = z["truth_guest"]
    print(f"truth: {has_sp.sum()} species answers, {(truth_sp >= 0).sum()} found among the "
          f"groups; {guest.sum()} guests", flush=True)

    dna = RangeSeasonPrior()
    dna.fit(records, species)
    store = OccurrenceStore.load()
    occ = OccurrencePrior(store, OccParams())
    occ.fit(records, species)
    print("leak check:", occ.leak_check(z["uuid"].tolist(), allow_missing=True), flush=True)
    G = len(species)
    K = pt.TOP_K + 1

    def blank(dtype=np.float32):
        return np.zeros((n, K), dtype=dtype)
    bases = {"nearest_mean": z["nearest_mean"], "nearest": z["nearest"]}
    out = {b: {"S": np.full((n, K), -np.inf, np.float32), "cand": np.full((n, K), -1, np.int32),
               "SP": blank(bool), "GG": np.full((n, K), -2, np.int32),
               "truth_at": np.full(n, -1, np.int64),
               **{f"dna_place_{int(k)}": blank() for k in pt.PLACE_KM},
               **{f"dna_season_{int(d)}": blank() for d in pt.SEASON_DAYS},
               **{f"occ_out_{int(r)}": blank(bool) for r in pt.RADII},
               **{f"occ_genus_out_{int(r)}": blank(bool) for r in pt.RADII},
               **{f"occ_dna_out_{int(r)}": blank(bool) for r in pt.RADII},
               **{f"occ_density_{int(k)}": blank() for k in pt.DENSITY_KM},
               "occ_season": blank()} for b in bases}
    dna_effort = {r: np.zeros(n) for r in pt.RADII}
    has_place = np.zeros(n, bool)
    has_date = np.zeros(n, bool)
    truth_km = np.full(n, np.nan)
    truth_out = {r: np.zeros(n, bool) for r in pt.RADII}
    known_lat = ~np.isnan(dna.lat)
    known_doy = ~np.isnan(dna.doy)
    lat, lng = z["org_lat"], z["org_lng"]
    for i in range(n):
        ctx = Context(None if np.isnan(lat[i]) else float(lat[i]),
                      None if np.isnan(lng[i]) else float(lng[i]),
                      z["observed_on"][i] or None, z["uuid"][i] or None)
        place = ctx.latitude is not None and ctx.longitude is not None
        has_place[i] = place
        full = {}
        if place:
            d = haversine_km(ctx.latitude, ctx.longitude, np.nan_to_num(dna.lat),
                             np.nan_to_num(dna.lon))
            for k in pt.PLACE_KM:
                full[f"dna_place_{int(k)}"] = dna._log_ratio(np.exp(-0.5 * (d / k) ** 2),
                                                             known_lat)
        doy = ctx.day_of_year
        has_date[i] = doy is not None
        if doy is not None:
            dd = day_distance(doy, np.nan_to_num(dna.doy))
            for k in pt.SEASON_DAYS:
                full[f"dna_season_{int(k)}"] = dna._log_ratio(np.exp(-0.5 * (dd / k) ** 2),
                                                              known_doy)
        occ.params.density_bandwidth_km = 150.0
        parts = occ.parts(ctx, pt.RADII)
        for r in pt.RADII:
            full[f"occ_out_{int(r)}"] = parts.out_of_range[r]
            full[f"occ_genus_out_{int(r)}"] = parts.genus_out_of_range[r]
            full[f"occ_dna_out_{int(r)}"] = parts.dna_out_of_range[r]
            dna_effort[r][i] = parts.dna_effort[r]
            if truth_sp[i] >= 0:
                truth_out[r][i] = bool(parts.out(r, True, pt.MIN_DNA_EFFORT)[truth_sp[i]])
        if parts.density is not None:
            full["occ_density_150"] = parts.density
        if parts.season is not None:
            full["occ_season"] = parts.season
        if parts.density is not None:
            for k in (75.0, 300.0):
                full[f"occ_density_{int(k)}"] = density_at(occ, store, ctx, k)
            if i < 5:
                assert np.allclose(density_at(occ, store, ctx, 150.0), parts.density), "density"
        if place and truth_sp[i] >= 0:
            truth_km[i] = nearest_find_km(occ, store, ctx, int(truth_sp[i]))
        for b, scores in bases.items():
            o = out[b]
            c = pt.candidates(scores[i], int(truth_sp[i]))
            m = len(c)
            o["S"][i, :m] = scores[i, c]
            o["cand"][i, :m] = c
            o["SP"][i, :m] = is_sp[c]
            o["GG"][i, :m] = group_gid[c]
            if truth_sp[i] >= 0:
                o["truth_at"][i] = int(np.flatnonzero(c == truth_sp[i])[0])
            for name, vec in full.items():
                o[name][i, :m] = vec[c]
        if (i + 1) % 250 == 0:
            print(f"  {i + 1:,}/{n:,} [{time.time() - t0:.0f}s]", flush=True)
    for b, o in out.items():
        np.savez_compressed(sets / f"{split}-components-{b}{tag}.npz", **o,
                            **{f"occ_dna_effort_{int(r)}": dna_effort[r] for r in pt.RADII})
    np.savez_compressed(
        sets / f"{split}-meta{tag}.npz", observation_id=z["observation_id"], user_id=z["user_id"],
        truth_name=z["truth_name"], has_sp=has_sp, truth_g=truth_g, guest=guest,
        has_place=has_place, has_date=has_date, truth_km=truth_km,
        **{f"truth_out_{int(r)}": truth_out[r] for r in pt.RADII},
        depth=np.array([bucket(int(z["ref_count"][j]), DEPTH_BUCKETS) if j >= 0 else
                        ("0" if t else "n/a") for j, t in zip(truth_sp.tolist(), truth)]),
        group_names=np.array(species), group_genus=z["group_genus"],
        group_family=z["group_family"], genus_names=np.array(genus_names))
    print(f"done [{time.time() - t0:.0f}s]", flush=True)


def density_at(occ, store, ctx, bandwidth_km: float) -> np.ndarray:
    """occprior.OccurrencePrior.parts' density term at another kernel width (same code,
    own observation taken out), without recomputing the out-of-range parts."""
    from mycomap_vision.occprior import gaussian
    own = store.own_contribution(ctx.uuid)
    mine = np.array([occ.local[u] for u in own.units if u in occ.local] if own else [],
                    dtype=np.int64)
    at = int(occ.pos_of_cell[own.cell]) if own else -1
    total, total_effort = occ.total, occ.total_effort
    own_effort = 1.0 if own and own.effort else 0.0
    if own:
        total = total.copy()
        total[mine] -= 1
        total_effort -= own_effort
    d = haversine_km(ctx.latitude, ctx.longitude, occ.cell_lat, occ.cell_lon)
    k = gaussian(d, bandwidth_km)
    self_k = float(k[at]) if at >= 0 else 0.0
    local_effort = float((k * occ.cell_effort).sum()) - self_k * own_effort
    k_local = np.bincount(occ.pair_local, minlength=len(occ.total),
                          weights=occ.pair_count * k[occ.pair_pos]).astype(np.float64)
    k_local[mine] -= self_k
    return occ._shrunk_log_ratio(k_local, total, local_effort / total_effort)


def merge(sets: Path, split: str, nparts: int) -> None:
    for name in [f"components-{b}" for b in ("nearest_mean", "nearest")] + ["meta"]:
        parts = [dict(np.load(sets / f"{split}-{name}.part{p}of{nparts}.npz"))
                 for p in range(nparts)]
        n0 = len(parts[0]["observation_id"]) if name == "meta" else len(parts[0]["S"])
        out = {}
        for k in parts[0]:
            per = [x[k] for x in parts]
            same_len = per[0].ndim >= 1 and len(per[0]) == (
                len(parts[0]["observation_id"]) if name == "meta" else len(parts[0]["S"]))
            out[k] = np.concatenate(per) if same_len and k not in (
                "group_names", "group_genus", "group_family", "genus_names") else per[0]
        np.savez_compressed(sets / f"{split}-{name}.npz", **out)
        print(name, {k: v.shape for k, v in out.items() if k in ("S", "observation_id")})


def nearest_find_km(occ, store, ctx, group: int) -> float:
    """Distance to the true species' nearest known find, iNat (its own observation taken
    out) or DNA record. Only ever reported as a band."""
    best = np.inf
    u = int(occ.g_sp[group])
    if u >= 0:
        sel = occ.pair_local == u
        pos = occ.pair_pos[sel]
        cnt = occ.pair_count[sel].copy()
        own = store.own_contribution(ctx.uuid)
        if own:
            at = int(occ.pos_of_cell[own.cell])
            mine = {occ.local[x] for x in own.units if x in occ.local}
            if u in mine:
                cnt[pos == at] -= 1
        pos = pos[cnt > 0]
        if len(pos):
            best = float(haversine_km(ctx.latitude, ctx.longitude, occ.cell_lat[pos],
                                      occ.cell_lon[pos]).min())
    sel = occ.dna_group == group
    if sel.any():
        best = min(best, float(haversine_km(ctx.latitude, ctx.longitude, occ.dna_lat[sel],
                                            occ.dna_lon[sel]).min()))
    return best


if __name__ == "__main__":
    if sys.argv[1] == "merge":
        merge(Path(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
    else:
        main()
