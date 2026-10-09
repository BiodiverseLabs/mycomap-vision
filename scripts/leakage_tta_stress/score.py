"""Step 2: score the dev queries under each leakage condition (CPU, numpy, no torch).

    python scripts/leakage_tta_stress/score.py --data data/stress [--queries tta.npz --views ...]

For each query its photo similarities to every reference photo are computed once, then
each condition masks reference records and each method scores what is left
(mycomap_vision.stress). Conditions:

  base         the reference as served
  obs          minus every record by the query's own iNat observer
  coll         minus the same collection: same observer within 1 km and 1 day, or any
               reference record with a photo at cosine > 0.98 to a query photo
  coll_wide    the same at 5 km and 3 days, cosine > 0.95 (sensitivity)
  clean, clean+obs, clean+coll   the same, also minus the known non-iNat references
               (MO / MyCoPortal / .com Sequences ids read as iNat ids: wrong photos)

Methods: nearest, nearest+mean, nearest+prior@org; and for base / clean the first
photo alone (nearest, nearest+mean). Writes <data>/scores-<tag>.npz: per condition and
method the top 10 species and genus (label indices) and, per query, the counts and
distance bands the slices need (no coordinates).
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from mycomap_vision import stress
from mycomap_vision.prior import Context, haversine_km

TOP = 10
CONDITIONS = ("base", "obs", "coll", "coll_wide", "clean", "clean+obs", "clean+coll")
METHODS = ("nearest", "nearest+mean", "nearest+prior@org")
FIRST_PHOTO = {"base", "clean"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/stress"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--queries", type=Path, help="TTA vectors (tta_embed.py output)")
    ap.add_argument("--views", default="", help="comma list of view names to combine")
    ap.add_argument("--combine", choices=("embedding", "score"), default="embedding")
    ap.add_argument("--conditions", default=",".join(CONDITIONS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch-photos", type=int, default=96)
    ap.add_argument("--shard", default="0/1", help="k/n: score every n-th query from k")
    a = ap.parse_args()
    t0 = time.time()
    prep = pickle.load(open(a.data / "prep.pkl", "rb"))
    meta = np.load(a.data / "ref_meta.npz")
    # float16 and memory-mapped: shards running side by side share the pages.
    ref = np.load(a.data / "ref_vecs.npy", mmap_mode="r")
    print(f"loaded {ref.shape} in {time.time() - t0:.0f} s", flush=True)
    starts = meta["starts"]
    col_group, col_rec = meta["col_group"], meta["col_rec"]
    sizes = np.bincount(col_group, minlength=len(starts))
    sums = stress.species_sums(ref, starts)
    unit_means = stress.unit(sums)
    r_user, r_lat, r_lon, r_day, r_doy = (meta[k] for k in ("user", "lat", "lon", "day", "doy"))
    r_group = meta["group"]
    non_inat = np.isin(meta["source"], ["MO Observations", "MycoPortal", "MyCoPortal",
                                        "Sequences", "GenBank Accessions"])
    n_rec = len(r_user)
    clean_cols = np.flatnonzero(non_inat[col_rec])
    clean_sums = sums.copy()
    g = col_group[clean_cols]
    run = np.flatnonzero(np.diff(g, prepend=-1) != 0)
    clean_sums[g[run]] -= np.add.reduceat(np.asarray(ref[clean_cols], dtype=np.float32),
                                          run, axis=0)
    clean_sizes = sizes - np.bincount(g, minlength=len(sizes))
    clean_unit_means = stress.unit(clean_sums)
    clean_unit_means[clean_sizes == 0] = 0
    rec_cols = np.argsort(col_rec, kind="stable")
    rec_start = np.searchsorted(col_rec[rec_cols], np.arange(n_rec + 1))
    prior = prep["prior"]
    label_of = prep["label_of"]
    n_sp, n_ge = len(prep["labels"]["species"]), len(prep["labels"]["genus"])
    group_pos = {s: i for i, s in enumerate(prep["species"])}
    obs_refs = np.bincount(r_user[r_user >= 0])
    conditions = a.conditions.split(",")

    k_shard, n_shard = map(int, a.shard.split("/"))
    queries = prep["queries"][: a.limit or None][k_shard::n_shard]
    views = [v for v in a.views.split(",") if v]
    if a.queries:
        tta = np.load(a.queries, allow_pickle=True)
        tta_rows = {int(p): i for i, p in enumerate(tta["photo_ids"].tolist())}
        view_names = list(tta["views"])
        pick = [view_names.index(v) for v in views]
        tvec = tta["vectors"]                    # (photos, views, dim) float16

    def query_vectors(q):
        if not a.queries:
            return [q["vecs"].astype(np.float32)]
        rows = [tta_rows[p] for p in q["photo_ids"]]
        v = tvec[rows][:, pick, :].astype(np.float32)        # (photos, views, dim)
        if a.combine == "embedding":
            return [stress.unit(stress.unit(v).mean(axis=1))]
        return [v[:, j, :] for j in range(v.shape[1])]       # score each view, average

    n = len(queries)
    out = {f"{c}|{m}|{r}": np.full((n, TOP), -1, np.int32) for c in conditions
           for m in METHODS for r in ("species", "genus")}
    for c in FIRST_PHOTO & set(conditions):
        for m in ("nearest", "nearest+mean"):
            for r in ("species", "genus"):
                out[f"{c}|{m} first photo|{r}"] = np.full((n, TOP), -1, np.int32)
    f = {k: np.full(n, -1, np.int64) for k in (
        "obs_refs", "same_obs_refs_masked", "coll_records", "coll_wide_records",
        "coll_place_records", "dup98_records", "dup95_records", "dup99_records",
        "same_obs_day_records", "truth_refs", "truth_refs_obs", "refs_100km")}
    for k in ("max_sim", "nearest_same_species_km", "season_share"):
        f[k] = np.full(n, np.nan)
    for km, days in ((0.1, 0), (0.1, 1), (1, 0), (1, 1), (1, 3), (5, 1), (5, 3), (25, 1)):
        f[f"coll_{km}km_{days}d"] = np.full(n, -1, np.int64)

    def put(cond, method, scores, i):
        for rank, nl, sp in (("species", n_sp, True), ("genus", n_ge, False)):
            rs = stress.rank_scores(scores, label_of[rank], nl, sp)
            t = stress.top_k(rs, TOP)
            out[f"{cond}|{method}|{rank}"][i, :len(t)] = t

    def records_cols(recs):
        if not len(recs):
            return np.zeros(0, np.int64)
        return np.concatenate([rec_cols[rec_start[r]:rec_start[r + 1]] for r in recs])

    batch, done = [], 0
    t1 = time.time()

    def flush():
        nonlocal done
        if not batch:
            return
        stacked = np.concatenate([v for _i, vs in batch for v in vs])
        sims_all = np.empty((len(stacked), len(ref)), dtype=np.float32)
        for s0 in range(0, len(ref), 65536):
            sims_all[:, s0:s0 + 65536] = stacked @ np.asarray(ref[s0:s0 + 65536],
                                                              dtype=np.float32).T
        at = 0
        for i, vs in batch:
            k = sum(len(v) for v in vs)
            score_one(i, vs, sims_all[at:at + k])
            at += k
        done += len(batch)
        batch.clear()
        el = time.time() - t1
        print(f"  {done:,}/{n:,} queries, {el:.0f} s, ETA {el / done * (n - done):.0f} s",
              flush=True)

    def score_one(i, vs, sims_cat):
        q = queries[i]
        nv = len(vs)
        sims_views = np.split(sims_cat, nv) if nv > 1 else [sims_cat]
        qv_views = vs
        truth = q["truth"]
        # --- facts (from the plain query vectors / first view)
        sims0 = sims_views[0]
        f["max_sim"][i] = float(sims0.max())
        user = q["user"] if q["user"] is not None and q["user"] >= 0 else None
        f["obs_refs"][i] = int(obs_refs[user]) if user is not None and user < len(obs_refs) else 0
        same_obs = (r_user == user) if user is not None else np.zeros(n_rec, bool)
        lat, lon = q["org_lat"], q["org_lon"]
        if lat is None or lon is None:
            lat, lon = q["lat"], q["lon"]
        day = q["day"] if q["day"] >= 0 else None
        for km, days in ((0.1, 0), (0.1, 1), (1, 0), (1, 1), (1, 3), (5, 1), (5, 3), (25, 1)):
            f[f"coll_{km}km_{days}d"][i] = int(stress.same_collection(
                user, lat, lon, day, r_user, r_lat, r_lon, r_day, km, days).sum())
        coll_place = stress.same_collection(user, lat, lon, day, r_user, r_lat, r_lon, r_day,
                                            stress.SAME_COLLECTION_KM,
                                            stress.SAME_COLLECTION_DAYS)
        coll_place_w = stress.same_collection(user, lat, lon, day, r_user, r_lat, r_lon,
                                              r_day, 5.0, 3)
        best_per_col = np.max([s.max(axis=0) for s in sims_views], axis=0)
        dup = {t: np.zeros(n_rec, bool) for t in (0.95, 0.98, 0.99)}
        for t in dup:
            dup[t][np.unique(col_rec[best_per_col > t])] = True
        f["dup95_records"][i] = int(dup[0.95].sum())
        f["dup98_records"][i] = int(dup[0.98].sum())
        f["dup99_records"][i] = int(dup[0.99].sum())
        f["same_obs_day_records"][i] = int((same_obs & (r_day == day)).sum()) if day else 0
        f["coll_place_records"][i] = int(coll_place.sum())
        coll = coll_place | dup[0.98]
        coll_wide = coll_place_w | dup[0.95]
        f["coll_records"][i] = int(coll.sum())
        f["coll_wide_records"][i] = int(coll_wide.sum())
        f["same_obs_refs_masked"][i] = int(same_obs.sum())
        g = group_pos.get(truth.species) if truth is not None and truth.species else None
        if g is not None:
            mine = np.flatnonzero(r_group == g)
            f["truth_refs"][i] = len(mine)
            f["truth_refs_obs"][i] = int(same_obs[mine].sum())
            if lat is not None and lon is not None:
                ok = mine[~np.isnan(r_lat[mine])]
                if len(ok):
                    f["nearest_same_species_km"][i] = float(
                        haversine_km(lat, lon, r_lat[ok], r_lon[ok]).min())
            share = stress.season_share(q["doy"] if q["doy"] >= 0 else None, r_doy[mine])
            if share is not None:
                f["season_share"][i] = share
        elif truth is not None and truth.species:
            f["truth_refs"][i] = 0
        if lat is not None and lon is not None:
            have = ~np.isnan(r_lat)
            f["refs_100km"][i] = int((haversine_km(lat, lon, r_lat[have], r_lon[have])
                                      <= 100).sum())
        ctx = Context(q["org_lat"], q["org_lon"], q["observed_on"])
        base_means = [qv @ unit_means.T for qv in qv_views]
        masks = {"base": None, "obs": same_obs, "coll": coll, "coll_wide": coll_wide,
                 "clean": non_inat, "clean+obs": non_inat | same_obs,
                 "clean+coll": non_inat | coll}
        bases = [stress.best_two(sv, starts) for sv in sims_views]
        clean_bases = [stress.best_two_masked(sv, starts, col_group, b2, clean_cols)
                       for sv, b2 in zip(sims_views, bases)]
        clean_means = [np.where(clean_sizes > 0, qv @ clean_unit_means.T, stress.NEG)
                       for qv in qv_views]
        pparts = stress.prior_weights(prior, ctx)
        for cond in conditions:
            mrec = masks[cond]
            drop_recs = np.flatnonzero(mrec) if mrec is not None else np.zeros(0, np.int64)
            clean = cond.startswith("clean")
            extra = masks[cond.split("+", 1)[1]] if clean and "+" in cond else (
                None if clean else mrec)
            extra = extra & ~non_inat if (clean and extra is not None) else extra
            masked_cols = (np.flatnonzero(extra[col_rec]) if extra is not None and extra.any()
                           else np.zeros(0, np.int64))
            per_view = []
            for k, (sv, qv) in enumerate(zip(sims_views, qv_views)):
                b2 = clean_bases[k] if clean else bases[k]
                m1, m2 = stress.best_two_masked(sv, starts, col_group, b2, masked_cols)
                ms = stress.mean_sims(qv, clean_sums if clean else sums,
                                      clean_sizes if clean else sizes, ref, col_group,
                                      masked_cols, base=(clean_means if clean else base_means)[k])
                per_view.append((m1, m2, ms))
            if len(per_view) == 1:
                m1, m2, ms = per_view[0]
                near = stress.nearest_scores(m1)
                nm = stress.nearest_mean_scores(m1, m2, ms)
            else:            # TTA by scores: each view scored, scores averaged
                near = np.mean([stress.nearest_scores(p[0]) for p in per_view], axis=0)
                nm = np.mean([stress.nearest_mean_scores(*p) for p in per_view], axis=0)
            lp = stress.log_prior_from(prior, pparts, drop_recs)
            npri = stress.log_softmax(near.astype(np.float64) / stress.LOGPROB_TEMPERATURE) + lp
            put(cond, "nearest", near, i)
            put(cond, "nearest+mean", nm, i)
            put(cond, "nearest+prior@org", npri, i)
            if cond in FIRST_PHOTO and len(per_view) == 1:
                m1, m2, ms = per_view[0]
                put(cond, "nearest first photo", m1[0], i)
                put(cond, "nearest+mean first photo",
                    stress.nearest_mean_scores(m1[:1], m2[:1], ms[:1]), i)

    photos = 0
    for i, q in enumerate(queries):
        vs = query_vectors(q)
        batch.append((i, vs))
        photos += sum(len(v) for v in vs)
        if photos >= a.batch_photos:
            flush()
            photos = 0
    flush()
    name = a.tag if n_shard == 1 else f"{a.tag}.shard{k_shard}of{n_shard}"
    np.savez_compressed(a.data / f"scores-{name}.npz",
                        oid=np.array([q["oid"] for q in queries]),
                        **{f"top|{k}": v for k, v in out.items()},
                        **{f"fact|{k}": v for k, v in f.items()})
    print(f"wrote scores-{name}.npz in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
