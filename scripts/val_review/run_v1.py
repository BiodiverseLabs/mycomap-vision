"""The review's measurement fixes on dataset release v1's val (Task B, 2026-10-18).

usage:
  python run_v1.py score  --manifest M --backbone B --errata TSV --out DIR [--release-root R]
  python run_v1.py dhash  --errata TSV --out DIR [--release-root R]
  python run_v1.py report --errata TSV --out DIR [--release-root R] [--months 6]

score: nearest+mean (photos only) and nearest+mean+place (penalty-500-2026-10-13, each
find's PUBLIC point) fitted on train, every val find scored. Per find and rank (species,
genus): the top 10 names, and the confidence, the true name's probability and the likely
set's verdict (a) as SERVED: nearest+mean's stored calibration (temperature and floor from
the newest eval_runs row for the backbone, the one the live server reads) and (b) with the
method's OWN temperature and floor fitted on val (the min-NLL grid of evaluate, the floor of
likely.fit_and_check). Writes DIR/<method>.jsonl and DIR/<method>.fit.json.
dhash: each val photo flagged when a train photo lies within dHash distance 4 (the v1
index in data/research-release-extras/v1/; val_review.near_train_flags). DIR/dhash.json.
report: B2 place calibration (served vs own, own also cross-fitted on observer halves), B3
unknown-species behaviour, B4 generalisation splits, in the standard format. DIR/report.json.

DIR holds per-record answers: keep it private, outside this repository. The manifest is
opened read-only; use a COPY anyway (the served calibration reads the shared one).
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

PLACE_SETTING = "penalty-500-2026-10-13"
METHODS = ("nearest+mean", "nearest+mean+place")
RANKS = ("species", "genus")
THRESHOLDS = (0.3, 0.5, 0.7, 0.9)
K = (1, 3, 5, 10)
REPS, SEED = 10_000, 20261018
BINS = 15


def ro(path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def release_of(args):
    from mycomap_vision import errata
    from mycomap_vision.dataset_release import load_release
    rel = load_release("v1", Path(args.release_root) if args.release_root else None)
    rel.use_errata(errata.read(Path(args.errata)))
    return rel


def public_points(rel) -> dict[str, tuple]:
    return {str(k): ((lat / 1e6, lon / 1e6) if lat is not None and kind != "none"
                     else (None, None))
            for k, lat, lon, kind in rel.private.execute(
                "select record_key, public_lat_micro, public_lon_micro, public_kind "
                "from record_public_places")}


# --- score ----------------------------------------------------------------------------------

def cmd_score(args) -> None:
    from dataclasses import replace

    from mycomap_vision import evaluate, likely
    from mycomap_vision.placeprior import SETTING_ENV
    os.environ[SETTING_ENV] = PLACE_SETTING
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = ro(args.manifest)
    served = evaluate.latest_calibration(conn, args.backbone, "nearest+mean")
    if not served or not served.get("sets"):
        raise SystemExit("no stored nearest+mean calibration with likely sets for this model")
    rel = release_of(args)
    shared = evaluate.shared_records(conn, [args.backbone], release=rel)
    ids, vecs = shared.loaded[args.backbone]
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    ref = evaluate.with_rows(shared.ref, row_of)
    public = public_points(rel)
    test = [replace(r, latitude=public.get(r.observation_id, (None, None))[0],
                    longitude=public.get(r.observation_id, (None, None))[1])
            for r in evaluate.with_rows(shared.test, row_of)]
    for m in METHODS:
        t0 = time.time()
        index, model = evaluate.fit_method(vecs, ref, m, backbone=args.backbone)
        place = getattr(model, "needs_context", False)
        pos = {rank: {n: i for i, n in enumerate(index.labels[rank])} for rank in RANKS}
        nll = {rank: np.zeros(len(evaluate.T_GRID)) for rank in RANKS}
        rows, compact = [], {rank: [] for rank in RANKS}
        for rec in test:
            scores = (model.species_scores(vecs[rec.photo_rows], evaluate.context_of(rec))
                      if place else model.species_scores(vecs[rec.photo_rows]))
            row = {"key": rec.observation_id, "top": {}, "served": {}}
            for rank in RANKS:
                truth = evaluate.truth(rec, rank)
                if not truth:
                    continue
                rs = evaluate.rank_scores(scores, index, rank)
                ti = pos[rank].get(truth)
                lse = likely.logsumexp_by_temperature(rs, evaluate.T_GRID)
                if ti is not None:
                    nll[rank] += evaluate.nll_by_temperature(rs, ti, lse)
                compact[rank].append((len(rows), likely.compact(rs, ti, evaluate.T_GRID,
                                                                 lse=lse),
                                      float(rs.max()), None if ti is None else float(rs[ti])))
                row["top"][rank] = evaluate.top_labels(rs, index.labels[rank], 10)
                t = served["temperatures"][rank]
                p = likely.probabilities(rs, t)
                chosen, _ = likely.likely_set(p, served["sets"][rank]["floor"])
                row["served"][rank] = {"conf": float(p.max()),
                                       "p_true": None if ti is None else float(p[ti]),
                                       "in_set": ti is not None and ti in chosen,
                                       "set_size": len(chosen)}
            rows.append(row)
        fit = {"method": m, "served_from": {"comparison_id": served["comparison_id"],
                                            "temperatures": served["temperatures"],
                                            "floors": {r: served["sets"][r]["floor"]
                                                       for r in RANKS}},
               "place_setting": getattr(getattr(model, "prior", None), "setting_name", None),
               "own": {}}
        for rank in RANKS:
            ti = int(np.argmin(nll[rank]))
            t = float(evaluate.T_GRID[ti])
            expanded = [likely.expand(c, ti, t) for _, c, _, _ in compact[rank]]
            sets = likely.fit_and_check(expanded)
            fit["own"][rank] = {"temperature": t, "sets": sets}
            for (i, c, best, true), (probs, tpos) in zip(compact[rank], expanded):
                own = {"conf": float(np.exp(best / t - c[1][ti])),
                       "p_true": None if true is None else float(np.exp(true / t - c[1][ti]))}
                if sets:
                    chosen, _ = likely.likely_set(probs, sets["floor"])
                    own["in_set"] = tpos is not None and tpos >= 0 and tpos in chosen
                    own["set_size"] = len(chosen)
                rows[i].setdefault("own", {})[rank] = own
                # for the cross-fit: everything a temperature needs, at every grid T
                rows[i].setdefault("grid", {})[rank] = {"best": best, "true": true,
                                                        "lse": c[1].round(6).tolist()}
        fit["seconds"] = round(time.time() - t0)
        (out / f"{m}.fit.json").write_text(json.dumps(fit, indent=1), encoding="utf-8")
        with open(out / f"{m}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        print(f"{m}: {len(rows):,} finds [{fit['seconds']}s]", flush=True)


# --- dhash ----------------------------------------------------------------------------------

def cmd_dhash(args) -> None:
    from mycomap_vision import config, dhash, val_review
    rel = release_of(args)
    index = dhash.load_index(Path(args.index) if args.index else
                             config.DATA_DIR / "research-release-extras" / "v1" / dhash.INDEX_FILE)
    dhash.check_index(index, rel)
    of_key = dict(zip(index["photo_key"], index["dhash"].tolist()))
    split_of = {str(r["record_key"]): r["split"] for r in rel.records(("train", "val"))}
    photos = rel.db.execute("select photo_key, record_key from photos where included = 1")
    train, val, rec_of = [], {}, {}
    for pk, rk in photos:
        s = split_of.get(str(rk))
        if s == "train":
            train.append(of_key[pk])
        elif s == "val":
            val[pk] = np.uint64(of_key[pk])
            rec_of[pk] = str(rk)
    t0 = time.time()
    flags = val_review.near_train_flags(val, np.asarray(train, dtype=np.uint64), chunk=64)
    near = sorted({rec_of[pk] for pk, f in flags.items() if f})
    out = {"val_photos": len(val), "train_photos": len(train),
           "val_photos_flagged": int(sum(flags.values())), "finds_flagged": near,
           "max_distance": dhash.NEAR, "seconds": round(time.time() - t0)}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "dhash.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"{out['val_photos_flagged']:,} of {len(val):,} val photos within {dhash.NEAR} bits "
          f"of a train photo; {len(near):,} finds")


# --- report ---------------------------------------------------------------------------------

def load(out: Path, m: str) -> dict:
    with open(out / f"{m}.jsonl", encoding="utf-8") as fh:
        return {r["key"]: r for r in map(json.loads, fh)}


def finds_of(rel) -> dict:
    train = rel.records("train")
    depth = Counter(r["species"] for r in train if r["species"])
    seen = {r["observer_login"] for r in train if r["observer_login"]}
    return {str(r["record_key"]): {
        "species": r["species"] or "", "genus": r["genus"] or "",
        "observer": r["observer_login"] or f"(unknown):{r['record_key']}",
        "has_observer": bool(r["observer_login"]), "observed_on": r["observed_on"],
        "former_heldout": bool(r["benchmark"]),
        "depth": depth.get(r["species"], 0) if r["species"] else None,
        "observer_seen": r["observer_login"] in seen} for r in rel.records("val")}


def hits(row, f) -> dict:
    """The standard table's cells for one find (strict, s.l., complex; genus strict, s.l.)."""
    from mycomap_vision import name_equiv
    o = {}
    for rank, levels, match in (("species", ("strict", "sl", "complex"), name_equiv.species_match),
                                ("genus", ("strict", "sl"), name_equiv.genus_match)):
        truth = f[rank]
        if not truth or rank not in row["top"]:
            continue
        top = row["top"][rank]
        for j, lv in enumerate(levels):
            first = next((i for i, a in enumerate(top[:10])
                          if a == truth or any(match(a, truth)[x] for x in levels[1:j + 1])),
                         None)
            for k in K:
                o[f"{rank}_{lv}@{k}"] = first is not None and first < k
    return o


def boot_ci(values, clusters, reps=REPS, seed=SEED):
    labels, inv = np.unique(np.asarray(clusters, dtype=str), return_inverse=True)
    sums = np.bincount(inv, weights=np.asarray(values, float), minlength=len(labels))
    counts = np.bincount(inv, minlength=len(labels)).astype(float)
    rng = np.random.default_rng(seed)
    est = np.empty(reps)
    for at in range(0, reps, 500):
        n = min(500, reps - at)
        pick = rng.integers(0, len(sums), size=(n, len(sums)))
        est[at:at + n] = sums[pick].sum(axis=1) / np.maximum(counts[pick].sum(axis=1), 1)
    return [float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5))]


def ece(conf, right, bins=BINS):
    conf, right = np.asarray(conf, float), np.asarray(right, float)
    if not len(conf):
        return None
    order = np.argsort(conf, kind="stable")
    return float(sum(len(p) / len(conf) * abs(right[p].mean() - conf[p].mean())
                     for p in np.array_split(order, min(bins, len(conf))) if len(p)))


def standard(keys, o) -> dict:
    cells = ([f"species_{lv}@{k}" for lv in ("strict", "sl", "complex") for k in K]
             + [f"genus_{lv}@{k}" for lv in ("strict", "sl") for k in K])
    out = {}
    for c in cells:
        v = [o[k][c] for k in keys if c in o[k]]
        out[c] = {"n": len(v), "rate": float(np.mean(v)) if v else None}
    return out


def calibration_block(keys, rows, o, which, rank="species") -> dict:
    """ECE, NLL, Brier-free summary and likely-set coverage for `which` ('served' / 'own')."""
    cal = [(rows[k][which][rank], o[k][f"{rank}_strict@1"]) for k in keys
           if rank in rows[k].get(which, {}) and f"{rank}_strict@1" in o[k]]
    conf = [c["conf"] for c, _ in cal]
    right = [r for _, r in cal]
    known = [c["p_true"] for c, _ in cal if c["p_true"] is not None]
    sets = [c for c, _ in cal if "in_set" in c]
    return {"n": len(cal), "ece": ece(conf, right),
            "nll": float(np.mean(-np.log(np.maximum(known, 1e-12)))) if known else None,
            "nll_n": len(known),
            "mean_confidence": float(np.mean(conf)) if conf else None,
            "accuracy": float(np.mean(right)) if right else None,
            "likely_coverage": float(np.mean([c["in_set"] for c in sets])) if sets else None,
            "likely_mean_size": float(np.mean([c["set_size"] for c in sets])) if sets else None}


def crossfit(keys, rows, o, finds, rank="species") -> dict:
    """The method's own temperature fitted on one half of the observers and measured on the
    other (both ways): its calibration without fitting on the finds it is judged on."""
    from mycomap_vision.evaluate import T_GRID
    obs = sorted({finds[k]["observer"] for k in keys})
    rng = np.random.default_rng(SEED)
    half = set(rng.permutation(obs)[: len(obs) // 2].tolist())
    parts = ([k for k in keys if finds[k]["observer"] in half],
             [k for k in keys if finds[k]["observer"] not in half])
    conf, right, ptrue, temps = [], [], [], []
    for fit_on, judge in (parts, parts[::-1]):
        nll = np.zeros(len(T_GRID))
        for k in fit_on:
            g = rows[k]["grid"].get(rank)
            if g and g["true"] is not None:
                nll += np.asarray(g["lse"]) - g["true"] / T_GRID
        ti = int(np.argmin(nll))
        t = float(T_GRID[ti])
        temps.append(t)
        for k in judge:
            g = rows[k]["grid"].get(rank)
            if not g or f"{rank}_strict@1" not in o[k]:
                continue
            conf.append(float(np.exp(g["best"] / t - g["lse"][ti])))
            right.append(o[k][f"{rank}_strict@1"])
            if g["true"] is not None:
                ptrue.append(float(np.exp(g["true"] / t - g["lse"][ti])))
    return {"temperatures": temps, "n": len(conf), "ece": ece(conf, right),
            "nll": float(np.mean(-np.log(np.maximum(ptrue, 1e-12)))), "nll_n": len(ptrue)}


def cmd_report(args) -> None:
    from mycomap_vision import val_review as vr
    from mycomap_vision.apiv1 import CONFIDENT_AT
    out = Path(args.out)
    rel = release_of(args)
    finds = finds_of(rel)
    rows = {m: load(out, m) for m in METHODS}
    fits = {m: json.loads((out / f"{m}.fit.json").read_text(encoding="utf-8")) for m in METHODS}
    o = {m: {k: hits(r, finds[k]) for k, r in rows[m].items()} for m in METHODS}
    keys = sorted(set(rows[METHODS[0]]) & set(rows[METHODS[1]]))
    scopes = {"primary": [k for k in keys if not finds[k]["former_heldout"]], "full val": keys}
    near = set(json.loads((out / "dhash.json").read_text(encoding="utf-8"))["finds_flagged"]) \
        if (out / "dhash.json").exists() else None
    cutoff = rel.val_cutoff
    report = {"val_cutoff": cutoff, "months": args.months, "fits": fits, "scopes": {}}
    for scope, kk in scopes.items():
        sp = [k for k in kk if finds[k]["species"]]
        s = report["scopes"][scope] = {"finds": len(kk), "species_finds": len(sp)}
        # B2: place calibration, served vs own
        s["calibration"] = {m: {"served": calibration_block(sp, rows[m], o[m], "served"),
                                "own_in_sample": calibration_block(sp, rows[m], o[m], "own"),
                                "own_crossfit": crossfit(sp, rows[m], o[m], finds),
                                "genus_served": calibration_block(
                                    [k for k in kk if finds[k]["genus"]], rows[m], o[m],
                                    "served", "genus")}
                            for m in METHODS}
        # B3: unknown species
        s["unknown_species"] = {}
        for m in METHODS:
            conf = [rows[m][k]["served"]["species"]["conf"] for k in sp]
            right = [o[m][k]["species_strict@1"] for k in sp]
            depth = [finds[k]["depth"] for k in sp]
            zero = [k for k in sp if finds[k]["depth"] == 0]
            s["unknown_species"][m] = {
                "confident": vr.confident_answers(conf, right, depth, CONFIDENT_AT),
                "selective_all": vr.selective_accuracy(conf, right, THRESHOLDS),
                "selective_known": vr.selective_accuracy(
                    [c for c, d in zip(conf, depth) if d > 0],
                    [r for r, d in zip(right, depth) if d > 0], THRESHOLDS),
                "genus_backoff": vr.genus_backoff([o[m][k].get("genus_strict@1", False)
                                                   for k in sp], depth),
                "genus_backoff_ci95": boot_ci([o[m][k].get("genus_strict@1", False)
                                               for k in zero],
                                              [finds[k]["observer"] for k in zero])
                if zero else None,
                "genus_confident_unknown": vr.confident_answers(
                    [rows[m][k]["served"]["genus"]["conf"] for k in zero],
                    [o[m][k].get("genus_strict@1", False) for k in zero],
                    [0] * len(zero), CONFIDENT_AT)["all"] if zero else None}
        # B4: generalisation splits, standard tables with species/genus top-1 CIs
        unseen = vr.unseen_observer([finds[k]["observer"] if finds[k]["has_observer"] else None
                                     for k in kk],
                                    [r["observer_login"] for r in rel.records("train")])
        recent = vr.recent_cohort([finds[k]["observed_on"] for k in kk], cutoff, args.months)
        after = vr.recent_cohort([finds[k]["observed_on"] for k in kk], cutoff, None)
        dated = vr.recent_cohort([finds[k]["observed_on"] for k in kk], "1900-01-01", None)
        groups = {
            "all": list(kk),
            "observer unseen in train": [k for k, u in zip(kk, unseen) if u],
            "observer seen in train": [k for k, u, h in zip(kk, unseen, [finds[k]["has_observer"]
                                                                        for k in kk])
                                       if h and not u],
            f"observed within {args.months} months before the cutoff or after": [
                k for k, r in zip(kk, recent) if r],
            f"observed earlier than {args.months} months before the cutoff": [
                k for k, r, d in zip(kk, recent, dated) if d and not r],
            "observed after the cutoff": [k for k, r in zip(kk, after) if r],
            "no observation date": [k for k, d in zip(kk, dated) if not d]}
        if near is not None:
            groups["excluding dHash<=4 near copies of a train photo"] = [k for k in kk
                                                                         if k not in near]
            groups["dHash<=4 near copies only"] = [k for k in kk if k in near]
        s["splits"] = {}
        for g, gk in groups.items():
            s["splits"][g] = {}
            for m in METHODS:
                t = standard(gk, o[m])
                for c in ("species_strict@1", "genus_strict@1"):
                    pick = [k for k in gk if c in o[m][k]]
                    t[c]["ci95"] = boot_ci([o[m][k][c] for k in pick],
                                           [finds[k]["observer"] for k in pick]) if pick else None
                spk = [k for k in gk if finds[k]["species"]]
                t["depth"] = {}
                for lo, hi, label in ((0, 0, "0"), (1, 4, "1-4"), (5, 19, "5-19"),
                                      (20, 99, "20-99"), (100, 10**9, "100+")):
                    bk = [k for k in spk if lo <= finds[k]["depth"] <= hi]
                    t["depth"][label] = {
                        "n": len(bk),
                        "top1": float(np.mean([o[m][k]["species_strict@1"] for k in bk]))
                        if bk else None,
                        "top5": float(np.mean([o[m][k]["species_strict@5"] for k in bk]))
                        if bk else None}
                s["splits"][g][m] = t
    (out / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print_report(report)


def pct(x):
    return "-" if x is None else f"{100 * x:.1f}"


def print_report(r) -> None:
    for scope, s in r["scopes"].items():
        print(f"\n== {scope}: {s['finds']:,} finds ({s['species_finds']:,} with a species)")
        print("B2 calibration, species (served = nearest+mean's T and floor):")
        for m, c in s["calibration"].items():
            for which in ("served", "own_in_sample", "own_crossfit"):
                b = c[which]
                print(f"  {m:20s} {which:14s} n={b['n']:,} ECE {pct(b['ece'])} NLL "
                      f"{b['nll']:.3f}" + (f"  likely cov {pct(b.get('likely_coverage'))} size "
                                           f"{b.get('likely_mean_size') or 0:.2f}"
                                           if b.get("likely_coverage") is not None else "")
                      + (f"  T {b['temperatures']}" if "temperatures" in b else ""))
            g = c["genus_served"]
            print(f"  {m:20s} genus served   n={g['n']:,} ECE {pct(g['ece'])} likely cov "
                  f"{pct(g['likely_coverage'])} size {g['likely_mean_size'] or 0:.2f}")
        print("B3 unknown species (served confidence):")
        for m, u in s["unknown_species"].items():
            c = u["confident"]
            print(f"  {m:20s} all n={c['all']['n']:,} confident {pct(c['all']['share'])} "
                  f"wrong-confident {pct(c['all']['wrong_share'])} precision "
                  f"{pct(c['all']['precision'])} | unknown n={c['unknown']['n']:,} confident "
                  f"{c['unknown']['confident']} ({pct(c['unknown']['share'])}) | genus back-off "
                  f"{pct(u['genus_backoff']['rate'])} (n={u['genus_backoff']['n']})")
            print("     selective (all): " + "  ".join(
                f">={x['threshold']}: {pct(x['accuracy'])} at {pct(x['coverage'])}"
                for x in u["selective_all"]))
        print("B4 splits (species strict top 1/3/5/10 | genus strict top 1, n):")
        for g, by in s["splits"].items():
            for m, t in by.items():
                print(f"  {g[:52]:52s} {m:20s} n={t['species_strict@1']['n']:,} sp "
                      + "/".join(pct(t[f'species_strict@{k}']['rate']) for k in K)
                      + f" ({pct(t['species_strict@1']['ci95'][0]) if t['species_strict@1'].get('ci95') else '-'}-"
                      f"{pct(t['species_strict@1']['ci95'][1]) if t['species_strict@1'].get('ci95') else '-'})"
                      f" | ge {pct(t['genus_strict@1']['rate'])}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("score", "dhash", "report"):
        s = sub.add_parser(name)
        s.add_argument("--release-root")
        s.add_argument("--errata", required=True)
        s.add_argument("--out", required=True)
        if name == "score":
            s.add_argument("--manifest", required=True)
            s.add_argument("--backbone", required=True)
        if name == "dhash":
            s.add_argument("--index")
        if name == "report":
            s.add_argument("--months", type=int, default=6)
    args = p.parse_args(argv)
    {"score": cmd_score, "dhash": cmd_dhash, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    main()
