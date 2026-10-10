"""Follow-up 2: the prior only when the photos are unsure (priortune.GATE_*).

usage: python run_gate.py <manifest copy> <sets_dir> dev
       python run_gate.py <manifest copy> <sets_dir> test   (confirmation only, once)

Selection (dev, 5-fold CV by observer): priortune.GATE_RULE. Pass, judged on test: the
chosen setting beats nearest+mean on species top-1 overall AND loses no species right in the
300-1,500 km band (records whose true species' nearest DNA record lies 300-1,500 km away),
for every truth with a DNA reference and for truths with 6+ references. Counts only leave.
"""

import json
import sqlite3
import sys
import time
from pathlib import Path

sys.modules["torch"] = None
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tuning as rt  # noqa: E402

from mycomap_vision import priortune as pt  # noqa: E402


def bands(c, z, base_sp, dist, deep) -> dict:
    sp, _ = pt.top1(c, z)
    out = {}
    for label, sel in (("6+ refs", deep), ("all", ~np.isnan(dist))):
        for lo, hi, band in rt.DNA_BANDS:
            m = sel & (dist >= lo) & (dist < hi) & c.has_sp
            out[f"{label} | {band}"] = {
                "n": int(m.sum()), "nearest_mean_right": int(base_sp[m].sum()),
                "right": int(sp[m].sum()), "fixed": int((sp & ~base_sp)[m].sum()),
                "broken": int((base_sp & ~sp)[m].sum())}
    return out


def main():
    snap, sets, split = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    t0 = time.time()
    nm, meta = rt.load(sets, split, "nearest_mean")
    keep = ~meta["guest"]
    nm = rt.subset(nm, keep)
    meta = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
            for k, v in meta.items()}
    n = len(nm.S)
    conn = sqlite3.connect(snap)
    dist, deep = rt.dna_distances(conn, sets, split, nm, keep)
    band = (dist >= pt.GATE_BAND_KM[0]) & (dist < pt.GATE_BAND_KM[1]) & nm.has_sp
    margins = pt.photo_margins(nm)
    print(f"{split}: {n:,} records, {int(band.sum())} in the 300-1,500 km band "
          f"[{time.time() - t0:.0f}s]", flush=True)
    base_z = pt.combine(nm, {"family": "none"})
    base_sp, _ = pt.top1(nm, base_z)
    report = {"split": split, "records": n, "reproducibility": "exploratory-pre-freeze",
              "grid_declared": pt.GATE_DECLARED, "rule": pt.GATE_RULE}
    if split == "dev":
        finite = margins[np.isfinite(margins)]
        thresholds = {q: float(np.quantile(finite, q))
                      for q in sorted({v for k, v in pt.GATE_GRID["gate"] if k == "hard"})}
        grid = pt.gate_settings(thresholds)
        sp = np.zeros((len(grid), n), bool)
        ge = np.zeros_like(sp)
        for j, s in enumerate(grid):
            sp[j], ge[j] = pt.top1(nm, pt.combine_gated(nm, s, margins))
        folds = pt.observer_folds(meta["user_id"].tolist())
        cv = pt.nested_cv_within_band(sp, ge, folds, grid, band, base_sp)
        full = pt.choose_within_band(sp, ge, np.arange(n), grid, band, base_sp)
        chosen = grid[full]
        report["every_setting"] = [
            {**s, "species_right": int(sp[j][nm.has_sp].sum()), "genus_right": int(ge[j].sum()),
             "band_right": int(sp[j][band].sum()), "band_nearest_mean": int(base_sp[band].sum())}
            for j, s in enumerate(grid)]
        report["folds"] = [
            {"choice": pt.describe(grid[j]), "records": int((folds == f).sum()),
             "nearest_mean": round(float(base_sp[(folds == f) & nm.has_sp].mean()), 4),
             "chosen": round(float(sp[j][(folds == f) & nm.has_sp].mean()), 4),
             "band_n": int((band & (folds == f)).sum()),
             "band_nearest_mean": int(base_sp[band & (folds == f)].sum()),
             "band_chosen": int(sp[j][band & (folds == f)].sum())}
            for f, j in enumerate(cv["chosen"])]
        report["thresholds"] = thresholds
        report["chosen_on_all_dev"] = pt.describe(chosen)
        (sets / "chosen-gate.json").write_text(json.dumps(
            {"setting": chosen, "thresholds": thresholds}, indent=2), encoding="utf-8")
        for f in report["folds"]:
            print("  ", f, flush=True)
        print(f"  all-dev choice: {pt.describe(chosen)}", flush=True)
        gated_z = np.zeros_like(nm.S)
        T_gated = np.zeros(n)
        for f, j in enumerate(cv["chosen"]):
            rows = folds == f
            zf = pt.combine_gated(nm, grid[j], margins)
            gated_z[rows] = zf[rows]
            T_gated[rows] = pt.fit_temperature(nm, zf, np.flatnonzero(~rows))
        gated_name = "nearest+mean+gated (CV)"
        temps = {"nearest+mean": pt.fit_temperature(nm, base_z, np.arange(n)),
                 "gated": pt.fit_temperature(nm, pt.combine_gated(nm, chosen, margins),
                                             np.arange(n))}
        (sets / "temperatures-dev-gate.json").write_text(json.dumps(temps), encoding="utf-8")
        T_base = np.zeros(n)
        for f in range(pt.FOLDS):
            T_base[folds == f] = pt.fit_temperature(nm, base_z, np.flatnonzero(folds != f))
    else:
        saved = json.loads((sets / "chosen-gate.json").read_text(encoding="utf-8"))
        temps = json.loads((sets / "temperatures-dev-gate.json").read_text(encoding="utf-8"))
        chosen = saved["setting"]
        gated_z = pt.combine_gated(nm, chosen, margins)
        T_gated = np.full(n, temps["gated"])
        T_base = np.full(n, temps["nearest+mean"])
        gated_name = "nearest+mean+gated (chosen on dev)"
        report["chosen"] = pt.describe(chosen)
    runs = {"nearest+mean": (base_z, T_base),
            "first choice (fixed)": (pt.combine(nm, pt.GATE_PRIORS["dna"]), T_base),
            "boost (fixed)": (pt.combine(nm, pt.GATE_PRIORS["boost"]), T_base),
            gated_name: (gated_z, T_gated)}
    labeller = rt.heldout.Labeller(conn, extra=meta["truth_name"].tolist()
                                   + meta["group_names"].tolist())
    truths = {o: labeller.truth(t) for o, t in zip(meta["observation_id"].tolist(),
                                                   meta["truth_name"].tolist()) if t}
    truths = {o: t for o, t in truths.items() if not t.guest}
    feats = {o: {"species reference records": d} for o, d in
             zip(meta["observation_id"].tolist(), meta["depth"].tolist())}
    judged, text = {}, []
    report["tables"], report["dna_distance_bands"] = {}, {}
    for name, (z, T) in runs.items():
        res = rt.results_for(nm, z, T, meta)
        lad, depth, judged[name] = rt.tables(res, truths, labeller, feats, list(truths))
        sp_right, _ = pt.top1(nm, z)
        cal = pt.calibration_report(pt.top_confidence(nm, z, T), sp_right, nm.has_sp)
        report["tables"][name] = {"ladder": lad, "by_depth": depth, "calibration_species": cal}
        report["dna_distance_bands"][name] = bands(nm, z, base_sp, dist, deep)
        text.append(rt.fmt_ladder(name, lad, depth))
        print(f"  tables {name} [{time.time() - t0:.0f}s]", flush=True)
    report["paired_vs_nearest_mean"] = {
        name: {rank: rt.pairs(judged["nearest+mean"], judged[name], rank)
               for rank in ("species", "genus")}
        for name in runs if name != "nearest+mean"}
    g = report["dna_distance_bands"][gated_name]
    overall = report["paired_vs_nearest_mean"][gated_name]["species"]
    report["pass"] = {
        "overall_up": overall["fixed"] > overall["broken"],
        "band_held_all": g["all | 300-1,500 km"]["right"]
        >= g["all | 300-1,500 km"]["nearest_mean_right"],
        "band_held_6plus": g["6+ refs | 300-1,500 km"]["right"]
        >= g["6+ refs | 300-1,500 km"]["nearest_mean_right"]}
    report["pass"]["passes"] = all(report["pass"].values())
    path = sets / f"tuning-{split}-gate.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("\n\n".join(text))
    for name, b in report["dna_distance_bands"].items():
        print(name)
        for k, x in b.items():
            print(f"   {k:22s} n {x['n']:5d} nm {x['nearest_mean_right']:5d} -> {x['right']:5d} "
                  f"(+{x['fixed']}/-{x['broken']})")
    print(json.dumps({"paired": report["paired_vs_nearest_mean"], "pass": report["pass"]},
                     indent=1))
    print(f"-> {path} [{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
