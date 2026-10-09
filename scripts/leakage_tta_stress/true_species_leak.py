"""How often the reference holds the TRUE species from the query's own observer or its own
collection trip, and what removing those does to exactly those queries (aggregates only).

    python scripts/leakage_tta_stress/true_species_leak.py --data data/stress --tag main
"""

from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
from pathlib import Path

import numpy as np

from mycomap_vision import heldout, stress
from mycomap_vision.heldout_report import judge


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/stress"))
    ap.add_argument("--tag", default="main")
    ap.add_argument("--manifest", type=Path)
    a = ap.parse_args()
    prep = pickle.load(open(a.data / "prep.pkl", "rb"))
    meta = np.load(a.data / "ref_meta.npz")
    z = np.load(a.data / f"scores-{a.tag}.npz")
    labels = prep["labels"]
    group_pos = {s: i for i, s in enumerate(prep["species"])}
    conn = sqlite3.connect(a.manifest or a.data / "manifest-snapshot.sqlite")
    labeller = heldout.Labeller(conn, extra=[q["truth_name"] for q in prep["queries"]
                                             if q["truth_name"]])
    r = {k: meta[k] for k in ("user", "lat", "lon", "day", "group")}
    row = {oid: i for i, oid in enumerate(z["oid"].tolist())}
    sets = {"own observer has the true species": [],
            "own trip (<=1 km, +-1 day) has the true species": [],
            "own same-day records (any place) have the true species": []}
    for q in prep["queries"]:
        t = q["truth"]
        g = group_pos.get(t.species) if t is not None and t.species else None
        if g is None or q["user"] is None or q["user"] < 0:
            continue
        mine = r["group"] == g
        same = r["user"] == q["user"]
        lat, lon = (q["org_lat"], q["org_lon"]) if q["org_lat"] is not None else (q["lat"], q["lon"])
        day = q["day"] if q["day"] >= 0 else None
        trip = stress.same_collection(q["user"], lat, lon, day, r["user"], r["lat"], r["lon"],
                                      r["day"])
        if (mine & same).any():
            sets["own observer has the true species"].append(q["oid"])
        if (mine & trip).any():
            sets["own trip (<=1 km, +-1 day) has the true species"].append(q["oid"])
        if day is not None and (mine & same & (r["day"] == day)).any():
            sets["own same-day records (any place) have the true species"].append(q["oid"])
    # near-duplicate photos: coll minus the place rule, only where the truth is the species
    dup = z["fact|dup98_records"]
    out = {"queries": len(prep["queries"]), "species_queries": int(sum(
        1 for q in prep["queries"] if q["truth"] is not None and q["truth"].species))}
    for name, ids in sets.items():
        res = {}
        for method in ("nearest", "nearest+mean", "nearest+prior@org"):
            for cond in ("base", "obs", "obs_ctrl", "coll"):
                key = f"top|{cond}|{method}|species"
                if key not in z.files:
                    continue
                hits = []
                for oid in ids:
                    i = row[oid]
                    cand = [{"name": labels["species"][j]} for j in z[key][i] if j >= 0]
                    hits.append(judge({"species": cand}, prep_truth(prep, oid), labeller,
                                      False)["species"][0])
                res[f"{method} {cond}"] = round(float(np.mean(hits)), 4) if hits else None
        out[name] = {"queries": len(ids), "species_top1": res}
    out["queries with any reference photo at cosine > 0.98"] = int((dup > 0).sum())
    print(json.dumps(out, indent=1))
    (a.data / f"true-species-leak-{a.tag}.json").write_text(json.dumps(out, indent=1))


_TRUTH = {}


def prep_truth(prep, oid):
    if not _TRUTH:
        _TRUTH.update({q["oid"]: q["truth"] for q in prep["queries"]})
    return _TRUTH[oid]


if __name__ == "__main__":
    main()
