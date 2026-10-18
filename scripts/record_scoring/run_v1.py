"""Record-grouped scoring on dataset release v1's val (experiment record-scoring, E32).

usage:
  python run_v1.py score  --manifest M --backbone B --errata TSV --out DIR [--arms a,b,...]
                          [--release-root R]
  python run_v1.py report --out DIR [--release-root R] [--errata TSV]
  python run_v1.py memory --manifest M --backbone B --errata TSV [--release-root R]

score: for each method, the same path as `mv compare --release v1 --errata TSV` (evaluate.
shared_records, fit_method, evaluate), against train, on val; the place methods use each
find's PUBLIC point (obscured where iNat obscures; never the true one) and the place setting
in MV_PLACE_SETTING. Writes DIR/<method>.summary.json (evaluate's report, temperature fitted
on val) and DIR/<method>.records.jsonl (each find's answer). Nothing is written to the
manifest: open a COPY anyway (the served calibration reads the shared one's eval_runs).

report: the pre-declared analysis (research experiments/2026-10-18-record-scoring.md): the
standard tables, depth by reference records and by reference photos, the 0-reference finds,
calibration (ECE over 15 equal-mass bins, NLL, likely sets) and the decision rule (paired
observer-cluster bootstrap, one-sided 95% lower bound against -0.3 points, photos-only AND
+place). Writes DIR/report.json and prints the tables.

memory: what the record-grouped methods add to a fitted reference index (bytes).

DIR holds per-record answers (record keys): keep it private, outside this repository.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARMS = ["nearest+mean", "nearest+mean-records", "nearest+mean-records-balanced",
        "nearest+mean-views"]
PLACE = {a: ("nearest+mean+place" if a == "nearest+mean" else f"{a}+place") for a in ARMS}
LETTER = dict(zip(ARMS, "abcd"))
PLACE_SETTING = "penalty-500-2026-10-13"
MARGIN = 0.003
REPS, SEED = 10_000, 20261018
ECE_MARGIN, NLL_MARGIN = 0.005, 0.02
TIE = 0.001
BINS = 15
K = (1, 3, 5, 10)
REC_BANDS = [(0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"), (20, 99, "20-99"), (100, 10**9, "100+")]
PHOTO_BANDS = [(0, 0, "0"), (1, 9, "1-9"), (10, 49, "10-49"), (50, 249, "50-249"),
               (250, 10**9, "250+")]


def band(n, bands):
    return next(label for lo, hi, label in bands if lo <= n <= hi)


def ro(path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def release_of(args):
    from mycomap_vision.dataset_release import load_release
    rel = load_release("v1", Path(args.release_root) if args.release_root else None)
    if args.errata:
        from mycomap_vision import errata
        rel.use_errata(errata.read(Path(args.errata)))
    return rel


UNGROUPED = "@ungrouped"         # a record arm scored without --twins (sensitivity row)


def read_twins(path) -> dict[str, str]:
    """{observation_id: specimen key} from a TSV with observation_id and specimen columns
    (records filed twice for one find; evaluate.build_index's specimen_of)."""
    import csv
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="	"))
    if rows and not {"observation_id", "specimen"} <= set(rows[0]):
        raise SystemExit(f"{path}: needs observation_id and specimen columns")
    return {r["observation_id"]: r["specimen"] for r in rows}


def grouped(method: str) -> bool:
    """The record-grouped arms (b-d, with or without place): those that --twins groups."""
    return any(method.startswith(a) for a in ARMS[1:])


def public_points(rel) -> dict[str, tuple]:
    """Each record's public point (paper rule: obscured where iNat obscures, MO's box
    centre), (None, None) when it has none."""
    return {str(k): ((lat / 1e6, lon / 1e6) if lat is not None and kind != "none"
                     else (None, None))
            for k, lat, lon, kind in rel.private.execute(
                "select record_key, public_lat_micro, public_lon_micro, public_kind "
                "from record_public_places")}


# --- score ----------------------------------------------------------------------------------

def cmd_score(args) -> None:
    from dataclasses import replace

    from mycomap_vision import evaluate
    from mycomap_vision.placeprior import SETTING_ENV
    os.environ[SETTING_ENV] = args.place_setting
    # The code that runs is the code checked out now: read once, so a commit made while the
    # arms score never relabels them.
    from mycomap_vision import config
    code_version = config.code_version()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = ro(args.manifest)
    rel = release_of(args)
    t0 = time.time()
    shared = evaluate.shared_records(conn, [args.backbone], release=rel)
    ids, vecs = shared.loaded[args.backbone]
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    ref = evaluate.with_rows(shared.ref, row_of)
    test = evaluate.with_rows(shared.test, row_of)
    public = public_points(rel)
    test_public = [replace(r, latitude=public.get(r.observation_id, (None, None))[0],
                           longitude=public.get(r.observation_id, (None, None))[1])
                   for r in test]
    print(f"{len(test):,} val finds against {len(ref):,} train records, record set "
          f"{shared.record_set} [{time.time() - t0:.0f}s]", flush=True)
    methods = [m.strip() for m in args.arms.split(",")] if args.arms else \
        [m for a in ARMS for m in (a, PLACE[a])]
    twins = read_twins(args.twins) if args.twins else None
    if twins and not args.arms:
        methods.append(ARMS[2] + UNGROUPED)          # arm c without grouping (declared)
    for name in methods:
        m = name.removesuffix(UNGROUPED)
        t1 = time.time()
        specimen_of = twins if twins and grouped(m) and not name.endswith(UNGROUPED) else None
        fitted = evaluate.fit_method(vecs, ref, m, backbone=args.backbone,
                                     specimen_of=specimen_of)
        place = getattr(fitted[1], "needs_context", False)
        res = evaluate.evaluate(vecs, ref, test_public if place else test, method=m,
                                fitted=fitted, name_scores=True, sets=True, per_record=True)
        records = res.pop("per_record")
        used = getattr(getattr(fitted[1], "prior", None), "setting_name", None)
        res["run"] = {"method": m, "backbone": args.backbone, "record_set": shared.record_set,
                      "dataset_release": rel.cite(), "place_setting": used,
                      "place_points": "public" if place else None,
                      "twins": ({"file": str(args.twins), "sha256": sha256_of(args.twins),
                                 "records_grouped": sum(1 for r in ref
                                                        if r.observation_id in specimen_of),
                                 "runs": len(fitted[0].rec_starts)}
                                if specimen_of else None),
                      "code_version": code_version,
                      "seconds": round(time.time() - t1)}
        (out / f"{name}.summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        with open(out / f"{name}.records.jsonl", "w", encoding="utf-8") as fh:
            for r in records:
                fh.write(json.dumps(r) + "\n")
        print(f"  {name}: species top-1 {res['species']['all']['top1']:.4f}, T "
              f"{res['calibration']['species']['temperature']:.4g} "
              f"[{time.time() - t1:.0f}s]", flush=True)


def sha256_of(path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --- report ---------------------------------------------------------------------------------

def finds_of(rel):
    """Per val find: truth, observer, former held-out, name kind, and the true species'
    reference depth in train (records and photos)."""
    train = rel.records("train")
    photos = Counter()
    for (key,) in rel.db.execute("select record_key from photos where included = 1"):
        photos[key] += 1
    rec_depth, photo_depth = Counter(), Counter()
    for r in train:
        if r["species"]:
            rec_depth[r["species"]] += 1
            photo_depth[r["species"]] += photos[r["record_key"]]
    out = {}
    for r in rel.records("val"):
        sp = r["species"] or ""
        out[str(r["record_key"])] = {
            "species": sp, "genus": r["genus"] or "", "observer": r["observer_login"]
            or f"(unknown):{r['record_key']}", "former_heldout": bool(r["benchmark"]),
            "name_kind": r["name_kind"], "rec_depth": rec_depth.get(sp, 0) if sp else None,
            "photo_depth": photo_depth.get(sp, 0) if sp else None}
    return out


def load_arm(out: Path, m: str) -> tuple[dict, dict]:
    summary = json.loads((out / f"{m}.summary.json").read_text(encoding="utf-8"))
    recs = {}
    with open(out / f"{m}.records.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            recs[r["observation_id"]] = r
    return summary, recs


def outcomes(recs: dict, finds: dict) -> dict:
    """Per find: hit flags for the standard table's cells, plus calibration inputs."""
    from mycomap_vision import name_equiv
    out = {}
    for key, a in recs.items():
        f = finds[key]
        o = {}
        if f["species"]:
            top = a["top"].get("species", [])
            for lv in ("strict", "sl", "complex"):
                first = None
                for i, ans in enumerate(top[:10]):
                    mm = name_equiv.species_match(ans, f["species"])
                    mm["strict"] = ans == f["species"]
                    levels = ("strict", "sl", "complex")
                    if any(mm[x] for x in levels[:levels.index(lv) + 1]):
                        first = i
                        break
                for k in K:
                    o[f"species_{lv}@{k}"] = first is not None and first < k
            cal = a.get("cal", {}).get("species")
            if cal:
                o["conf"], o["p_true"] = cal["conf"], cal["p_true"]
        if f["genus"]:
            top = a["top"].get("genus", [])
            for lv in ("strict", "sl"):
                first = None
                for i, ans in enumerate(top[:10]):
                    mm = name_equiv.genus_match(ans, f["genus"])
                    mm["strict"] = ans == f["genus"]
                    if mm["strict"] or (lv == "sl" and mm["sl"]):
                        first = i
                        break
                for k in K:
                    o[f"genus_{lv}@{k}"] = first is not None and first < k
        out[key] = o
    return out


def groups_sum(values, clusters):
    labels, inv = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    return (np.bincount(inv, weights=np.asarray(values, float), minlength=len(labels)),
            np.bincount(inv, minlength=len(labels)).astype(float))


def boot(values, clusters, reps=REPS, seed=SEED):
    """Observer-cluster bootstrap distribution of a mean (ratio of sums)."""
    sums, counts = groups_sum(values, clusters)
    rng = np.random.default_rng(seed)
    est = np.empty(reps)
    for at in range(0, reps, 500):
        n = min(500, reps - at)
        pick = rng.integers(0, len(sums), size=(n, len(sums)))
        est[at:at + n] = sums[pick].sum(axis=1) / np.maximum(counts[pick].sum(axis=1), 1)
    return est


def ece(conf, right, bins=BINS) -> float | None:
    conf, right = np.asarray(conf, float), np.asarray(right, float)
    if not len(conf):
        return None
    order = np.argsort(conf, kind="stable")
    return float(sum(len(p) / len(conf) * abs(right[p].mean() - conf[p].mean())
                     for p in np.array_split(order, min(bins, len(conf))) if len(p)))


def table(keys, o, finds, cells) -> dict:
    out = {}
    for c in cells:
        vals = [o[k][c] for k in keys if c in o.get(k, {})]
        out[c] = {"n": len(vals), "rate": float(np.mean(vals)) if vals else None}
    return out


def calibration(keys, o) -> dict:
    rows = [o[k] for k in keys if "conf" in o.get(k, {})]
    conf = [r["conf"] for r in rows]
    right = [r["species_strict@1"] for r in rows]
    known = [r["p_true"] for r in rows if r["p_true"] is not None]
    return {"n": len(rows), "ece": ece(conf, right),
            "nll_n": len(known),
            "nll": float(np.mean(-np.log(np.maximum(known, 1e-12)))) if known else None,
            "confident_0.70": {"n": int(np.sum(np.array(conf) >= 0.7)),
                               "accuracy": float(np.mean([r for c, r in zip(conf, right)
                                                          if c >= 0.7]))
                               if any(c >= 0.7 for c in conf) else None}}


def cmd_report(args) -> None:
    out = Path(args.out)
    rel = release_of(args)
    finds = finds_of(rel)
    arms = {}
    for a in ARMS:
        for m in (a, PLACE[a]):
            if (out / f"{m}.summary.json").exists():
                arms[m] = load_arm(out, m)
    if (out / f"{ARMS[2]}{UNGROUPED}.summary.json").exists():
        arms[ARMS[2] + UNGROUPED] = load_arm(out, ARMS[2] + UNGROUPED)
    o = {m: outcomes(recs, finds) for m, (_, recs) in arms.items()}
    keys_all = sorted(set.intersection(*(set(v) for v in o.values())))
    scopes = {"primary (val excl. former held-out)":
              [k for k in keys_all if not finds[k]["former_heldout"]],
              "full val": keys_all}
    cells = ([f"species_{lv}@{k}" for lv in ("strict", "sl", "complex") for k in K]
             + [f"genus_{lv}@{k}" for lv in ("strict", "sl") for k in K])
    report = {"finds": len(keys_all), "arms": list(arms), "scopes": {}}
    for scope, keys in scopes.items():
        sp_keys = [k for k in keys if finds[k]["species"]]
        s = report["scopes"][scope] = {"finds": len(keys), "species_finds": len(sp_keys),
                                       "standard": {}, "depth_records": {},
                                       "depth_photos": {}, "calibration": {},
                                       "zero_reference": {}, "paired": {}}
        for m in arms:
            s["standard"][m] = table(keys, o[m], finds, cells)
            for name, field_, bands in (("depth_records", "rec_depth", REC_BANDS),
                                        ("depth_photos", "photo_depth", PHOTO_BANDS)):
                s[name][m] = {}
                for _, _, label in bands:
                    kk = [k for k in sp_keys if band(finds[k][field_], bands) == label]
                    s[name][m][label] = table(kk, o[m], finds,
                                              ["species_strict@1", "species_strict@5"])
            s["calibration"][m] = calibration(sp_keys, o[m])
            zero = [k for k in sp_keys if finds[k]["rec_depth"] == 0]
            s["zero_reference"][m] = {**table(zero, o[m], finds, ["genus_strict@1"]),
                                      **calibration(zero, o[m])}
        who = [finds[k]["observer"] for k in sp_keys]
        for family in ("photos", "place"):
            base = ARMS[0] if family == "photos" else PLACE[ARMS[0]]
            if base not in arms:
                continue
            xb = np.array([o[base][k]["species_strict@1"] for k in sp_keys], float)
            extra = [ARMS[2] + UNGROUPED] if family == "photos" else []
            for a in ARMS[1:] + extra:
                m = a if family == "photos" else PLACE[a]
                if m not in arms:
                    continue
                xa = np.array([o[m][k]["species_strict@1"] for k in sp_keys], float)
                est = boot(xa - xb, who)
                s["paired"][f"{m} vs {base}"] = {
                    "n": len(sp_keys), "a": float(xa.mean()), "b": float(xb.mean()),
                    "difference": float((xa - xb).mean()),
                    "ci95": [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))],
                    "lower_bound_one_sided_95": float(np.percentile(est, 5)),
                    "noninferior": float(np.percentile(est, 5)) > -MARGIN,
                    "only_a": int(((xa == 1) & (xb == 0)).sum()),
                    "only_b": int(((xa == 0) & (xb == 1)).sum())}
    for m, (summary, _) in arms.items():
        report.setdefault("val_fit", {})[m] = {
            "temperature": summary["calibration"]["species"]["temperature"],
            "nll_in_evaluate": summary["calibration"]["species"]["nll"],
            "likely_sets_species": summary["calibration"]["species"].get("sets"),
            "likely_sets_genus": summary["calibration"].get("genus", {}).get("sets"),
            "run": summary.get("run")}
    report["decision"] = decide(report["scopes"]["primary (val excl. former held-out)"], arms)
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print_report(report)


def decide(s: dict, arms: dict) -> dict:
    """The pre-declared rule (entry record-scoring, confirmed by the Vision coordinator
    2026-10-18)."""
    a = ARMS[0]
    cal_a = s["calibration"].get(a)
    passing = {}
    for m in ARMS[1:]:
        photos = s["paired"].get(f"{m} vs {a}")
        place = s["paired"].get(f"{PLACE[m]} vs {PLACE[a]}")
        cal = s["calibration"].get(m)
        if not (photos and place and cal and cal_a):
            continue
        ok_cal = (cal["ece"] <= cal_a["ece"] + ECE_MARGIN
                  and cal["nll"] <= cal_a["nll"] + NLL_MARGIN)
        passing[m] = {"photos_noninferior": photos["noninferior"],
                      "place_noninferior": place["noninferior"], "calibration_ok": ok_cal,
                      "passes": photos["noninferior"] and place["noninferior"] and ok_cal,
                      "top1": photos["a"]}
    ok = [m for m in ARMS[1:] if passing.get(m, {}).get("passes")]
    if not ok:
        return {"arms": passing, "adopt": None}
    best = max(passing[m]["top1"] for m in ok)
    pick = next(m for m in ok if passing[m]["top1"] >= best - TIE)   # simplest within 0.1 pt
    return {"arms": passing, "adopt": pick}


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def print_report(r: dict) -> None:
    for scope, s in r["scopes"].items():
        print(f"\n== {scope}: {s['finds']:,} finds ({s['species_finds']:,} with a species)")
        print("standard (top 1/3/5/10):")
        for m, t in s["standard"].items():
            row = []
            for pre, lvs in (("species", ("strict", "sl", "complex")), ("genus", ("strict", "sl"))):
                for lv in lvs:
                    row.append(f"{pre[:2]} {lv} " + "/".join(pct(t[f'{pre}_{lv}@{k}']['rate'])
                                                             for k in K))
            n_sp, n_ge = t["species_strict@1"]["n"], t["genus_strict@1"]["n"]
            print(f"  {m:38s} n={n_sp:,}/{n_ge:,}  " + " | ".join(row))
        for name in ("depth_records", "depth_photos"):
            print(f"{name} (species top-1 / top-5, n):")
            for m, bands in s[name].items():
                print(f"  {m:38s} " + "  ".join(
                    f"{b}: {pct(v['species_strict@1']['rate'])}/{pct(v['species_strict@5']['rate'])}"
                    f" (n={v['species_strict@1']['n']:,})" for b, v in bands.items()))
        print("calibration (species, each at its own val temperature):")
        for m, c in s["calibration"].items():
            z = s["zero_reference"][m]
            print(f"  {m:38s} ECE {pct(c['ece'])}  NLL {c['nll']:.3f} (n={c['nll_n']:,})  "
                  f">=0.70: {c['confident_0.70']['n']:,} at {pct(c['confident_0.70']['accuracy'])}"
                  f"  | 0-ref n={z['genus_strict@1']['n']:,} genus {pct(z['genus_strict@1']['rate'])}"
                  f" confident {z['confident_0.70']['n']}")
        print("paired species top-1 (arm - a), observer-cluster bootstrap:")
        for k, p in s["paired"].items():
            print(f"  {k:62s} {100 * p['difference']:+.2f} pt (95% {100 * p['ci95'][0]:+.2f} to "
                  f"{100 * p['ci95'][1]:+.2f}; one-sided LB {100 * p['lower_bound_one_sided_95']:+.2f})"
                  f" {'NON-INFERIOR' if p['noninferior'] else 'not shown'}  "
                  f"[{p['only_a']} vs {p['only_b']} discordant]")
    print("\ndecision:", json.dumps(r["decision"], indent=1))


# --- memory ---------------------------------------------------------------------------------

def cmd_memory(args) -> None:
    from mycomap_vision import evaluate, methods
    conn = ro(args.manifest)
    rel = release_of(args)
    shared = evaluate.shared_records(conn, [args.backbone], release=rel)
    ids, vecs = shared.loaded[args.backbone]
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    index = evaluate.build_index(evaluate.with_rows(shared.ref, row_of))
    rec_species = methods.species_record_starts(index)
    q = np.asarray(vecs[:4], dtype=np.float32)
    sims = q @ np.asarray(vecs[index.cols[:1]], dtype=np.float32).T   # shape only
    n_cols, n_recs = len(index.cols), len(index.rec_starts)
    print(json.dumps({
        "reference_photos": n_cols, "reference_records": n_recs,
        "species": len(index.species),
        "persistent_bytes": {"index.rec_starts": int(index.rec_starts.nbytes),
                             "rec_species": int(rec_species.nbytes),
                             "species_means": "same shape as nearest+mean's (replaced, "
                                              "not added)"},
        "transient_bytes_per_query_photo": {
            "nearest+mean argsort over photos": n_cols * (8 + 4 + 8),
            "record arms: record maxima + argsort over records": n_recs * (4 + 8 + 4 + 8)},
    }, indent=1))
    del sims


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("score", "report", "memory"):
        s = sub.add_parser(name)
        s.add_argument("--release-root")
        s.add_argument("--errata")
        if name != "report":
            s.add_argument("--manifest", required=True)
            s.add_argument("--backbone", required=True)
        if name != "memory":
            s.add_argument("--out", required=True)
        if name == "score":
            s.add_argument("--arms")
            s.add_argument("--twins", help="TSV observation_id, specimen: records filed "
                                           "twice for one find, grouped in arms b-d")
            s.add_argument("--place-setting", default=PLACE_SETTING)
    args = p.parse_args(argv)
    {"score": cmd_score, "report": cmd_report, "memory": cmd_memory}[args.cmd](args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    main()
