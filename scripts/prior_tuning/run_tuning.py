"""The prior-tuning analysis (priortune.py) on the components build_components.py wrote.

usage: python run_tuning.py <snapshot.sqlite> <sets_dir> dev [boost]
       python run_tuning.py <snapshot.sqlite> <sets_dir> test [boost]   (confirmation only, once)
"boost": the follow-up grid (priortune.boost_settings), files suffixed -boost.
Writes <sets_dir>/tuning-<split>.json and prints the tables. No coordinates anywhere.
"""
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
sys.modules["torch"] = None
import numpy as np

from mycomap_vision import config, heldout, name_equiv
from mycomap_vision import priortune as pt
from mycomap_vision.heldout_report import mcnemar, score
from mycomap_vision.heldout_summary import KS, LEVELS, by_reference_depth, ladder

BENCH = config.DATA_DIR / "benchmarks" / "heldout-2026-10-08"
heldout.bench_dir = lambda conn, name: BENCH


def load(sets: Path, split: str, base: str):
    z = np.load(sets / f"{split}-components-{base}.npz")
    m = np.load(sets / f"{split}-meta.npz")
    c = pt.Components(
        S=z["S"].astype(np.float64), cand=z["cand"], SP=z["SP"], truth_at=z["truth_at"],
        has_sp=m["has_sp"], GG=z["GG"].astype(np.int64), truth_g=m["truth_g"].astype(np.int64),
        dna_place={k: z[f"dna_place_{int(k)}"].astype(np.float64) for k in pt.PLACE_KM},
        dna_season={d: z[f"dna_season_{int(d)}"].astype(np.float64) for d in pt.SEASON_DAYS},
        occ_out={r: z[f"occ_out_{int(r)}"] for r in pt.RADII},
        occ_genus_out={r: z[f"occ_genus_out_{int(r)}"] for r in pt.RADII},
        occ_dna_out={r: z[f"occ_dna_out_{int(r)}"] for r in pt.RADII},
        occ_dna_effort={r: z[f"occ_dna_effort_{int(r)}"] for r in pt.RADII},
        occ_density={k: z[f"occ_density_{int(k)}"].astype(np.float64) for k in pt.DENSITY_KM},
        occ_season=z["occ_season"].astype(np.float64))
    return c, {k: m[k] for k in m.files}


def subset(c: pt.Components, keep: np.ndarray) -> pt.Components:
    def cut(v):
        if isinstance(v, dict):
            return {k: cut(x) for k, x in v.items()}
        return v[keep] if isinstance(v, np.ndarray) else v
    return pt.Components(**{f: cut(getattr(c, f)) for f in c.__dataclass_fields__})


def results_for(c, z_all, T_per_row, meta):
    names, gg, gf = (meta["group_names"].tolist(), meta["group_genus"].tolist(),
                     meta["group_family"].tolist())
    out = {}
    for i, oid in enumerate(meta["observation_id"].tolist()):
        out[oid] = pt.ranks(z_all[i], c.cand[i], c.SP[i], names, gg, gf, float(T_per_row[i]))
    return out


def tables(results, truths, labeller, feats, ids):
    key = ("m",)
    judged = score({key: results}, truths, labeller)[key]
    lad = ladder(results, set(ids), truths, labeller, False, name_equiv)
    depth = by_reference_depth(judged, feats)
    return lad, depth, judged


def fmt_ladder(name, lad, depth) -> str:
    lines = [f"{name}  (n = {lad['records']:,} records)",
             f"    {'':<24}" + "".join(f"{'top ' + str(k):>8}" for k in KS) + f"{'n':>9}"]
    for label, row in lad["rows"].items():
        cells = "".join(f"{100 * row[f'top{k}']['rate']:7.1f}%" for k in KS)
        lines.append(f"    {label:<24}{cells}{row['n']:>9,}")
    lines.append(f"    {'species by true refs':<24}{'top 1':>8}{'top 5':>8}{'n':>9}")
    for b, row in depth.items():
        r1 = row["top1"]["rate"]
        r5 = row["top5"]["rate"]
        lines.append(f"    {b + ' records':<24}"
                     f"{'-' if r1 is None else f'{100 * r1:7.1f}%':>8}"
                     f"{'-' if r5 is None else f'{100 * r5:7.1f}%':>8}{row['n']:>9,}")
    return "\n".join(lines)


def pairs(judged_a, judged_b, rank):
    both = [o for o in judged_a if o in judged_b and rank in judged_a[o] and rank in judged_b[o]]
    a = np.array([judged_a[o][rank][0] for o in both])
    b = np.array([judged_b[o][rank][0] for o in both])
    fixed, broken = int((b & ~a).sum()), int((a & ~b).sum())
    return {"n": len(both), "fixed": fixed, "broken": broken,
            "mcnemar_p": round(mcnemar(broken, fixed), 6)}


DNA_BANDS = [(0, 100, "< 100 km"), (100, 300, "100-300 km"), (300, 1500, "300-1,500 km"),
             (1500, 1e9, "> 1,500 km")]


def dna_distances(conn, sets: Path, split: str, nm,
                  keep: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per record: km to the nearest DNA reference record of its true species (NaN: none,
    or no place), and whether that species has 6+ reference records. Bands only leave.
    `keep`: the records `nm` holds (guests left out), so rows line up with the scores file."""
    from mycomap_vision.evaluate import load_records
    from mycomap_vision.prior import haversine_km
    z = np.load(sets / f"{split}-scores.npz")
    species = z["species"].tolist()
    pos = {sp: i for i, sp in enumerate(species)}
    photos = {int(p): int(p) for (p,) in conn.execute(
        "select photo_id from embeddings where backbone = 'bioclip-2-ft-20261007-165400'")}
    by = {}
    for r in load_records(conn, photos):
        if r.unit in pos and r.latitude is not None and r.longitude is not None:
            by.setdefault(pos[r.unit], []).append((r.latitude, r.longitude))
    by = {k: np.array(v) for k, v in by.items()}
    lat, lng, refs = z["org_lat"][keep], z["org_lng"][keep], z["ref_count"]
    assert len(lat) == len(nm.S), "records and scores file out of line"
    d = np.full(len(nm.S), np.nan)
    deep = np.zeros(len(nm.S), bool)
    for i in range(len(nm.S)):
        t = int(nm.cand[i, nm.truth_at[i]]) if nm.truth_at[i] >= 0 else -1
        if t in by and not np.isnan(lat[i]):
            d[i] = haversine_km(lat[i], lng[i], by[t][:, 0], by[t][:, 1]).min()
            deep[i] = refs[t] >= 6
    return d, deep


def main():
    snap, sets, split = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    boost = len(sys.argv) > 4 and sys.argv[4] == "boost"
    sfx = "-boost" if boost else ""
    t0 = time.time()
    nm, meta = load(sets, split, "nearest_mean")
    ne, _ = load(sets, split, "nearest")
    keep = ~meta["guest"]
    nm, ne = subset(nm, keep), subset(ne, keep)
    meta = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
            for k, v in meta.items()}
    n = len(nm.S)
    print(f"{split}: {n:,} records (guests left out: {(~keep).sum()}), "
          f"{int(meta['has_place'].sum()):,} with a place, {int(meta['has_date'].sum()):,} "
          f"with a date [{time.time() - t0:.0f}s]", flush=True)
    grid = pt.boost_settings() if boost else pt.settings()
    fam = np.array([s["family"] for s in grid])
    folds = pt.observer_folds(meta["user_id"].tolist())
    conn = sqlite3.connect(snap)
    labeller = heldout.Labeller(conn, extra=meta["truth_name"].tolist()
                                + meta["group_names"].tolist())
    truths = {o: labeller.truth(t) for o, t in zip(meta["observation_id"].tolist(),
                                                   meta["truth_name"].tolist()) if t}
    truths = {o: t for o, t in truths.items() if not t.guest}
    ids = list(truths)
    feats = {o: {"species reference records": d} for o, d in
             zip(meta["observation_id"].tolist(), meta["depth"].tolist())}
    report = {"split": split, "records": n, "grid_size": len(grid), "grid": "boost" if boost
              else "main", "reproducibility": "exploratory-pre-freeze",
              "grid_declared": pt.BOOST_DECLARED if boost else pt.GRID_DECLARED, "fold_sizes": np.bincount(folds).tolist()}

    # The methods to report: fixed references, then the tuned prior families.
    refs = {"nearest": (ne, {"family": "none"}), "nearest+mean": (nm, {"family": "none"}),
            "nearest+prior@org (as shipped)": (ne, pt.AS_SHIPPED),
            "nearest+mean+prior@org (as shipped)": (nm, pt.AS_SHIPPED)}
    if boost:
        first = json.loads((sets / "chosen.json").read_text(encoding="utf-8"))["any"]
        refs.pop("nearest+mean+prior@org (as shipped)")
        # The first experiment's choice, fixed (on dev it is in-sample).
        refs["nearest+mean+prior (first choice)"] = (nm, first)
    if split == "dev":
        sp, ge = pt.score_all(nm, grid)
        print(f"scored {len(grid)} settings [{time.time() - t0:.0f}s]", flush=True)
        report["every_setting"] = [
            {**{k: v for k, v in s.items()}, "species_right": int(sp[j][nm.has_sp].sum()),
             "genus_right": int(ge[j].sum())} for j, s in enumerate(grid)]
        families = ({"boost": fam == "boost", "boost+far": fam == "boost+far",
                     "any": np.ones(len(grid), bool)} if boost else
                    {"dna": fam == "dna", "occ": fam == "occ", "dna+occ": fam == "dna+occ",
                     "any": np.ones(len(grid), bool)})
        cv = {}
        for f, allowed in families.items():
            res = pt.nested_cv(sp, ge, folds, grid, allowed)
            full = pt.choose(sp, ge, np.arange(n), grid, allowed)
            cv[f] = {"fold_choices": [pt.describe(grid[j]) for j in res["chosen"]],
                     "fold_choice_ids": res["chosen"],
                     "cv_species_top1": round(float(res["species_right"][nm.has_sp].mean()), 4),
                     "cv_genus_top1": round(float(res["genus_right"].mean()), 4),
                     "chosen_on_all_dev": pt.describe(grid[full]), "chosen_id": full,
                     "in_sample_species_top1": round(float(sp[full][nm.has_sp].mean()), 4),
                     "in_sample_genus_top1": round(float(ge[full].mean()), 4)}
            print(f"  {f}: CV {cv[f]['cv_species_top1']:.4f} / {cv[f]['cv_genus_top1']:.4f}; "
                  f"all-dev pick {cv[f]['chosen_on_all_dev']} "
                  f"({cv[f]['in_sample_species_top1']:.4f} in-sample)", flush=True)
        report["cv"] = cv
        (sets / f"chosen{sfx}.json").write_text(json.dumps(
            {f: grid[v["chosen_id"]] for f, v in cv.items()}, indent=2), encoding="utf-8")
        labels = ((("boost", "boost"), ("boost+far", "boost+far"), ("any", "boost-best"))
                  if boost else (("dna", "prior"), ("occ", "occ"), ("dna+occ", "prior+occ"),
                                 ("any", "best")))
        tuned = {f"nearest+mean+{label} (CV)": ("cv", f) for f, label in labels}
    else:
        chosen = json.loads((sets / f"chosen{sfx}.json").read_text(encoding="utf-8"))
        temps = json.loads((sets / f"temperatures-dev{sfx}.json").read_text(encoding="utf-8"))
        # Confirmation only: the ONE configuration the declared procedure chose on all of
        # dev (the whole grid, family included), next to the fixed references.
        name = "nearest+mean+boost-best (chosen on dev)" if boost else             "nearest+mean+best (chosen on dev)"
        tuned = {name: ("fixed", chosen["any"])}
        refs.pop("nearest+mean+prior@org (as shipped)", None)

    # Per method: combined scores per record (out of fold for CV), temperature per record.
    runs = {}
    dev_temps = {}
    for name, (c, s) in refs.items():
        zz = pt.combine(c, s)
        if split == "dev":
            T = np.zeros(n)
            for f in range(folds.max() + 1):
                T[folds == f] = pt.fit_temperature(c, zz, np.flatnonzero(folds != f))
            dev_temps[name] = pt.fit_temperature(c, zz, np.arange(n))
        else:
            T = np.full(n, temps[name])
        runs[name] = (c, zz, T, [s] * n)
    for name, (kind, what) in tuned.items():
        if kind == "cv":
            choice = cv[what]["fold_choice_ids"]
            zz = np.zeros_like(nm.S)
            T = np.zeros(n)
            used = [None] * n
            for f in range(folds.max() + 1):
                rows = folds == f
                zf = pt.combine(nm, grid[choice[f]])
                zz[rows] = zf[rows]
                T[rows] = pt.fit_temperature(nm, zf, np.flatnonzero(~rows))
                for i in np.flatnonzero(rows):
                    used[i] = grid[choice[f]]
            full = grid[cv[what]["chosen_id"]]
            dev_temps[name.replace(" (CV)", " (chosen on dev)")] = pt.fit_temperature(
                nm, pt.combine(nm, full), np.arange(n))
        else:
            zz = pt.combine(nm, what)
            T = np.full(n, temps[name])
            used = [what] * n
        runs[name] = (nm, zz, T, used)
    if split == "dev":
        (sets / f"temperatures-dev{sfx}.json").write_text(json.dumps(dev_temps, indent=2),
                                                    encoding="utf-8")
    print(f"combined [{time.time() - t0:.0f}s]", flush=True)

    out_tables, judged_all = {}, {}
    text = []
    for name, (c, zz, T, used) in runs.items():
        res = results_for(c, zz, T, meta)
        lad, depth, judged = tables(res, truths, labeller, feats, ids)
        judged_all[name] = judged
        out_tables[name] = {"ladder": lad, "by_depth": depth}
        text.append(fmt_ladder(name, lad, depth))
        # Calibration of the top species at the stated temperature.
        conf = pt.top_confidence(c, zz, T)
        sp_right, _ = pt.top1(c, zz)
        nll = None
        if split == "dev":
            tot = 0.0
            used_n = 0
            for f in range(folds.max() + 1):
                rows = np.flatnonzero(folds == f)
                tf = float(T[rows[0]]) if len(rows) else 1.0
                cur, k = pt.nll_curve(c, zz, rows, np.array([tf]))
                tot += float(cur[0])
                used_n += k
            nll = tot / max(used_n, 1)
        cal = pt.calibration_report(conf, sp_right, c.has_sp, nll)
        out_tables[name]["calibration_species"] = cal
        # What the method states at the old default (cosine 0.02 on a prior's log scale).
        out_tables[name]["share_stated_99_at_old_default"] = round(float(np.mean(
            pt.top_confidence(c, zz, 0.02)[c.has_sp] >= 0.99)), 4)
        print(f"  tables {name} [{time.time() - t0:.0f}s]", flush=True)

    base = judged_all["nearest+mean"]
    report["paired_vs_nearest_mean"] = {
        name: {rank: pairs(base, j, rank) for rank in ("species", "genus")}
        for name, j in judged_all.items() if name != "nearest+mean"}
    # Range edges: by the true species' nearest known find, and where the wide berth
    # flagged the truth. Counts only.
    bands = [pt.distance_band(None if np.isnan(x) else float(x)) for x in meta["truth_km"]]
    edge = {}
    for name, (c, zz, T, used) in runs.items():
        sp_right, _ = pt.top1(c, zz)
        tab = {}
        for b in pt.DISTANCE_BANDS:
            rows = np.array([x == b for x in bands]) & c.has_sp
            tab[b] = {"n": int(rows.sum()), "species_right": int(sp_right[rows].sum())}
        flagged = {}
        for r in pt.RADII:
            rows = meta[f"truth_out_{int(r)}"] & c.has_sp
            flagged[f"{int(r)} km"] = {"n": int(rows.sum()),
                                       "species_right": int(sp_right[rows].sum())}
        edge[name] = {"by_nearest_known_find": tab, "truth_flagged_out_of_range": flagged}
    report["range_edges"] = edge
    # By distance to the true species' nearest DNA reference record (search count).
    dist, deep = dna_distances(conn, sets, split, nm, keep)
    base_sp, _ = pt.top1(nm, runs["nearest+mean"][1])
    dna_bands = {}
    for name, (c, zz, T, used) in runs.items():
        sp_right, _ = pt.top1(c, zz)
        out = {}
        for label, sel in (("6+ refs", deep), ("all", ~np.isnan(dist))):
            for lo, hi, band in DNA_BANDS:
                m = sel & (dist >= lo) & (dist < hi) & c.has_sp
                out[f"{label} | {band}"] = {
                    "n": int(m.sum()), "right": int(sp_right[m].sum()),
                    "nearest_mean_right": int(base_sp[m].sum()),
                    "fixed": int((sp_right & ~base_sp)[m].sum()),
                    "broken": int((base_sp & ~sp_right)[m].sum())}
        dna_bands[name] = out
    report["dna_distance_bands"] = dna_bands
    report["tables"] = out_tables
    if split == "dev":
        report["selection_counts"] = {f: dict(Counter(v["fold_choices"])) for f, v in cv.items()}
    path = sets / f"tuning-{split}{sfx}.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("\n\n".join(text))
    print(json.dumps({"paired": report["paired_vs_nearest_mean"]}, indent=1))
    print(f"-> {path} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
