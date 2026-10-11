"""The prior variants on the leakage/stress lane's weak slices (exp/leakage-tta-stress,
stress.py): the true species' nearest DNA reference record 200-1,000 km or more than
1,000 km away, and off season (none of the true species' dated reference records within
30 days of the query's day of the year). Development only, fixed settings, nothing chosen.

usage: python slices.py <manifest copy> <sets_dir> [dev]
Species top-1 by the search's count (about 0.5 point under the report's judge). No
coordinates leave: counts per slice only.
"""

import json
import sqlite3
import sys
from pathlib import Path

sys.modules["torch"] = None
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_tuning as rt  # noqa: E402

from mycomap_vision import priortune as pt  # noqa: E402
from mycomap_vision.evaluate import load_records  # noqa: E402
from mycomap_vision.prior import Context, day_distance  # noqa: E402

VARIANTS = {
    "nearest+mean": {"family": "none"},
    "DNA prior, tuned (first choice)": pt.GATE_PRIORS["dna"],
    "boost + far penalty (tested)": pt.GATE_PRIORS["boost"],
    "boost + far penalty, season boost-only": {**pt.GATE_PRIORS["boost"],
                                               "season_boost_only": True},
}


def season_off(conn, sets: Path, split: str, nm, keep, observed_on) -> np.ndarray:
    """stress.season_share == 0: the true species has dated reference records and none is
    within 30 days (circular) of the query's day. False where either has no date."""
    z = np.load(sets / f"{split}-scores.npz")
    pos = {sp: i for i, sp in enumerate(z["species"].tolist())}
    photos = {int(p): int(p) for (p,) in conn.execute(
        "select photo_id from embeddings where backbone = 'bioclip-2-ft-20261007-165400'")}
    doys = {}
    for r in load_records(conn, photos):
        d = Context(observed_on=r.observed_on).day_of_year
        if r.unit in pos and d is not None:
            doys.setdefault(pos[r.unit], []).append(d)
    doys = {k: np.array(v) for k, v in doys.items()}
    out = np.zeros(len(nm.S), dtype=bool)
    for i in range(len(nm.S)):
        t = int(nm.cand[i, nm.truth_at[i]]) if nm.truth_at[i] >= 0 else -1
        q = Context(observed_on=observed_on[i] or None).day_of_year
        if t in doys and q is not None:
            out[i] = not (day_distance(q, doys[t]) <= 30).any()
    return out


def main():
    snap, sets = Path(sys.argv[1]), Path(sys.argv[2])
    split = sys.argv[3] if len(sys.argv) > 3 else "dev"
    if split != "dev":
        raise SystemExit("development only: these slices choose nothing and look at no test")
    nm, meta = rt.load(sets, split, "nearest_mean")
    keep = ~meta["guest"]
    nm = rt.subset(nm, keep)
    observed = np.load(sets / f"{split}-scores.npz")["observed_on"][keep]
    conn = sqlite3.connect(snap)
    dist, _ = rt.dna_distances(conn, sets, split, nm, keep)
    off = season_off(conn, sets, split, nm, keep, observed) & nm.has_sp
    slices = {
        "all (species answer)": nm.has_sp,
        "nearest ref > 1,000 km": nm.has_sp & (dist > 1000),
        "nearest ref 200-1,000 km": nm.has_sp & (dist > 200) & (dist <= 1000),
        "nearest ref 300-1,500 km (Steve's band)": nm.has_sp & (dist >= 300) & (dist < 1500),
        "nearest ref 100-300 km": nm.has_sp & (dist >= 100) & (dist < 300),
        "off season (no true-species ref within 30 d)": off,
    }
    base, _ = pt.top1(nm, pt.combine(nm, VARIANTS["nearest+mean"]))
    out = {}
    for name, s in VARIANTS.items():
        sp, ge = pt.top1(nm, pt.combine(nm, s))
        out[name] = {"genus_top1": round(float(ge.mean()), 4)}
        for label, m in slices.items():
            out[name][label] = {"n": int(m.sum()), "right": int(sp[m].sum()),
                                "fixed": int((sp & ~base)[m].sum()),
                                "broken": int((base & ~sp)[m].sum())}
    path = sets / f"slices-{split}.json"
    path.write_text(json.dumps(out, indent=1), encoding="utf-8")
    for label in slices:
        print(label)
        for name, row in out.items():
            x = row[label]
            print(f"   {name:42s} n {x['n']:5d}  {100 * x['right'] / max(x['n'], 1):5.1f}%  "
                  f"(+{x['fixed']}/-{x['broken']})")
    for name, row in out.items():
        print(f"genus top-1 {name}: {100 * row['genus_top1']:.1f}%")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
