"""Step 4: the tables (CPU). Steve's standard summary for each condition, fixed / broken
against the same method on the base condition, leakage by depth band, and the stress
slices. No record ids, coordinates or photo paths are printed or written.

    python scripts/leakage_tta_stress/report.py --data data/stress --tags main[,tta-...] \
        --out data/stress/report-main
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from mycomap_vision import heldout, stress
from mycomap_vision.heldout_report import DEPTH_BUCKETS, bucket, judge, mcnemar
from mycomap_vision.heldout_summary import KS, LEVELS, by_reference_depth, ladder, name_equiv

NEWEST_FROM = "2026-09-11"         # the newest four weeks before the freeze
MIN_REGION = 40                    # smaller states are pooled as "other"


def pct(x):
    return "  -  " if x is None else f"{100 * x:5.1f}"


def load(data: Path, tag: str):
    z = np.load(data / f"scores-{tag}.npz")
    tops = defaultdict(dict)
    for k in z.files:
        if k.startswith("top|"):
            _t, cond, method, rank = k.split("|")
            tops[(cond, method)][rank] = z[k]
    facts = {k[5:]: z[k] for k in z.files if k.startswith("fact|")}
    return list(z["oid"]), tops, facts


def results_of(oids, top, labels):
    out = {}
    for i, oid in enumerate(oids):
        out[oid] = {rank: [{"name": labels[rank][j], "confidence": None}
                           for j in top[rank][i] if j >= 0] for rank in ("species", "genus")}
        out[oid]["top_k"] = 10
    return out


def hits(results, truths, labeller):
    """{oid: {rank: (top1, top5)}} strict."""
    out = {}
    for oid, res in results.items():
        t = truths.get(oid)
        if t is None:
            continue
        out[oid] = judge(res, t, labeller, False)
    return out


def rate_row(vals):
    n = len(vals)
    k = sum(vals)
    return {"n": n, "rate": round(k / n, 4) if n else None, "ci95": stress.wilson(k, n)}


def depth_adjusted(ids, judged, band_of, weights) -> float | None:
    """Species top-1 standardised to the whole set's depth mix: the slice's rate in each
    reference-depth band, weighted by that band's share of all records (bands the slice
    has no records in are left out and the weights renormalised). It says whether a
    slice is weak beyond holding rarer species."""
    by = defaultdict(list)
    for o in ids:
        if "species" in judged[o] and band_of.get(o) in weights:
            by[band_of[o]].append(judged[o]["species"][0])
    w = sum(weights[b] for b in by)
    return round(sum(weights[b] * np.mean(v) for b, v in by.items()) / w, 4) if w else None


def slice_table(groups: dict, judged: dict, order=None, band_of=None, weights=None
                ) -> list[dict]:
    rows = []
    keys = order or sorted(groups)
    for g in keys:
        ids = [o for o in groups.get(g, []) if o in judged]
        sp = [judged[o]["species"] for o in ids if "species" in judged[o]]
        ge = [judged[o]["genus"] for o in ids if "genus" in judged[o]]
        rows.append({"slice": g, "records": len(ids),
                     "species_n": len(sp),
                     "species_top1": rate_row([h[0] for h in sp])["rate"],
                     "species_top1_ci": rate_row([h[0] for h in sp])["ci95"],
                     "species_top5": rate_row([h[1] for h in sp])["rate"],
                     "genus_n": len(ge),
                     "genus_top1": rate_row([h[0] for h in ge])["rate"],
                     "genus_top5": rate_row([h[1] for h in ge])["rate"],
                     "species_top1_depth_adjusted": depth_adjusted(ids, judged, band_of,
                                                                   weights)
                     if band_of else None})
    return rows


def fmt_slice(title, rows):
    lines = [f"  {title}",
             f"    {'slice':40s} {'n sp':>6s} {'sp@1':>6s} {'95% CI':>13s} {'sp@5':>6s} "
             f"{'n ge':>6s} {'ge@1':>6s} {'ge@5':>6s} {'sp@1 depth-adj':>15s}"]
    for r in rows:
        ci = r["species_top1_ci"]
        cis = f"{100 * ci[0]:4.1f}-{100 * ci[1]:4.1f}" if ci else "-"
        lines.append(f"    {str(r['slice'])[:40]:40s} {r['species_n']:6,d} "
                     f"{pct(r['species_top1']):>6s} {cis:>13s} {pct(r['species_top5']):>6s} "
                     f"{r['genus_n']:6,d} {pct(r['genus_top1']):>6s} {pct(r['genus_top5']):>6s}"
                     f" {pct(r.get('species_top1_depth_adjusted')):>15s}")
    return lines


def fmt_ladder(name, lad, depth):
    lines = [f"{name}  (n = {lad['records']:,} records)",
             f"    {'':26s} {'top 1':>7s} {'top 3':>7s} {'top 5':>7s} {'top 10':>7s} {'n':>7s}"]
    for _r, _l, label in LEVELS:
        row = lad["rows"][label]
        cells = " ".join(f"{pct((row.get(f'top{k}') or {}).get('rate')):>7s}" for k in KS)
        lines.append(f"    {label:26s} {cells} {row['n']:7,d}")
    lines.append("    species, by the true species' reference records")
    lines.append(f"    {'':26s} {'top 1':>7s} {'top 5':>7s} {'n':>7s}")
    for _lo, _hi, b in DEPTH_BUCKETS:
        d = depth[b]
        lines.append(f"    {b + ' records':26s} {pct(d['top1']['rate']):>7s} "
                     f"{pct(d['top5']['rate']):>7s} {d['n']:7,d}")
    return lines


def paired(j_a, j_b, rank):
    """fixed / broken of b against a (top-1), on records both judged at `rank`."""
    fixed = broken = 0
    for oid in j_a:
        if oid in j_b and rank in j_a[oid] and rank in j_b[oid]:
            a, b = j_a[oid][rank][0], j_b[oid][rank][0]
            fixed += (not a) and b
            broken += a and not b
    return {"fixed": fixed, "broken": broken, "p": round(mcnemar(fixed, broken), 5)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/stress"))
    ap.add_argument("--tags", default="main")
    ap.add_argument("--models", default="",
                    help="tag:condition:method,... (default: every one in the tags)")
    ap.add_argument("--baseline-tag", default="main")
    ap.add_argument("--slices", action="store_true")
    ap.add_argument("--leakage", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", "--snapshot", dest="snapshot", type=Path,
                    help="default: <data>/manifest-snapshot.sqlite")
    a = ap.parse_args()
    prep = pickle.load(open(a.data / "prep.pkl", "rb"))
    labels = prep["labels"]
    queries = {q["oid"]: q for q in prep["queries"]}
    conn = sqlite3.connect(a.snapshot or a.data / "manifest-snapshot.sqlite")
    labeller = heldout.Labeller(conn, extra=[q["truth_name"] for q in prep["queries"]
                                             if q["truth_name"]])
    truths = {oid: q["truth"] for oid, q in queries.items() if q["truth"] is not None}
    equiv = name_equiv()
    tags = [a.baseline_tag] + [t for t in a.tags.split(",") if t != a.baseline_tag]
    loaded = {t: load(a.data, t) for t in tags}
    base_oids, _bt, base_facts = loaded[a.baseline_tag]
    fact = {k: dict(zip(base_oids, v.tolist())) for k, v in base_facts.items()}
    feats = {oid: {"species reference records":
                   bucket(int(fact["truth_refs"][oid]), DEPTH_BUCKETS)
                   if fact["truth_refs"][oid] >= 0 else "n/a"} for oid in base_oids}

    models = ([tuple(m.split(":")) for m in a.models.split(",") if m] or
              [(t, c, m) for t, (_o, tops, _f) in loaded.items() for (c, m) in tops])
    report = {"reference_hash": prep["ref_hash"], "split": prep["split"],
              "queries": len(queries), "models": {}}
    text = [f"heldout-2026-10-08 / {prep['split']} (development, not sealed); reference "
            f"{prep['ref_hash']} (manifest snapshot 2026-10-09 14:37 UTC)", ""]
    judged_all = {}
    for tag, cond, method in models:
        oids, tops, _f = loaded[tag]
        res = results_of(oids, tops[(cond, method)], labels)
        ids = set(res) & set(truths)
        lad = ladder(res, ids, truths, labeller, False, equiv)
        jd = hits(res, truths, labeller)
        judged_all[(tag, cond, method)] = jd
        depth = by_reference_depth(jd, feats)
        name = f"{tag}:{cond}:{method}"
        base_key = (a.baseline_tag, "base", method.replace(" first photo", ""))
        entry = {"ladder": lad, "depth": depth}
        if base_key != (tag, cond, method) and base_key in judged_all:
            entry["vs_base"] = {r: paired(judged_all[base_key], jd, r)
                                for r in ("species", "genus")}
        report["models"][name] = entry
        text += fmt_ladder(name, lad, depth)
        if "vs_base" in entry:
            vb = entry["vs_base"]
            text.append(f"    vs {':'.join(base_key)} top-1: species +{vb['species']['fixed']}"
                        f" / -{vb['species']['broken']} (p {vb['species']['p']}), genus "
                        f"+{vb['genus']['fixed']} / -{vb['genus']['broken']} "
                        f"(p {vb['genus']['p']})")
        text.append("")

    if a.leakage:
        text += leakage_tables(report, judged_all, fact, feats, base_oids, a.baseline_tag)
    if a.slices:
        text += slice_tables(report, judged_all, fact, queries, base_oids, a.baseline_tag,
                             feats)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=1, default=str))
    a.out.with_suffix(".txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    print("\n".join(text))


def leakage_tables(report, judged_all, fact, feats, oids, tag):
    text = ["== Leakage: how much each condition changes species top-1, by depth band "
            "(bands from the base reference) =="]
    out = {}
    for method in ("nearest", "nearest+mean", "nearest+prior@org"):
        text.append(f"  {method}")
        header = f"    {'condition':12s} {'all':>12s}" + "".join(
            f" {b + ' refs':>14s}" for _l, _h, b in DEPTH_BUCKETS)
        text.append(header)
        rows = {}
        for cond in ("base", "obs", "obs_ctrl", "coll", "coll_wide", "clean", "clean+obs",
                     "clean+coll"):
            jd = judged_all.get((tag, cond, method))
            if jd is None:
                continue
            row = {}
            cells = []
            for b in ["all"] + [x[2] for x in DEPTH_BUCKETS]:
                hits = [jd[o]["species"][0] for o in jd if "species" in jd[o] and
                        (b == "all" or feats[o]["species reference records"] == b)]
                row[b] = {"n": len(hits), "top1": round(sum(hits) / len(hits), 4)
                          if hits else None}
                cells.append(f"{pct(row[b]['top1'])} ({len(hits):4d})")
            rows[cond] = row
            text.append(f"    {cond:12s} " + " ".join(f"{c:>14s}" for c in cells))
        out[method] = rows
    report["leakage_by_depth"] = out

    # how much there is to remove
    n = len(oids)
    def share(key, cond=lambda v: v > 0):
        vals = [fact[key][o] for o in oids]
        k = sum(1 for v in vals if cond(v))
        return k, n
    counts = {}
    text.append("")
    text.append("  What the conditions remove (dev queries with at least one such "
                "reference record; median records removed among those)")
    for key, label in (("same_obs_refs_masked", "same observer (obs)"),
                       ("coll_0.1km_0d", "same observer <=100 m, same day"),
                       ("coll_0.1km_1d", "same observer <=100 m, +-1 day"),
                       ("coll_1km_0d", "same observer <=1 km, same day"),
                       ("coll_1km_1d", "same observer <=1 km, +-1 day (coll)"),
                       ("coll_1km_3d", "same observer <=1 km, +-3 days"),
                       ("coll_5km_1d", "same observer <=5 km, +-1 day"),
                       ("coll_5km_3d", "same observer <=5 km, +-3 days (coll_wide)"),
                       ("coll_25km_1d", "same observer <=25 km, +-1 day"),
                       ("same_obs_day_records", "same observer, same day (any place)"),
                       ("dup99_records", "photo cosine > 0.99"),
                       ("dup98_records", "photo cosine > 0.98 (coll)"),
                       ("dup95_records", "photo cosine > 0.95 (coll_wide)"),
                       ("coll_records", "coll = place rule OR cosine > 0.98"),
                       ("coll_wide_records", "coll_wide")):
        vals = np.array([fact[key][o] for o in oids])
        pos = vals[vals > 0]
        counts[label] = {"queries": int(len(pos)), "of": n,
                         "median_records": float(np.median(pos)) if len(pos) else 0}
        text.append(f"    {label:45s} {len(pos):5,d} / {n:,} ({100 * len(pos) / n:4.1f}%)  "
                    f"median {counts[label]['median_records']:.0f}")
    # same-collection refs that are the TRUE species (the leak that matters)
    report["removed"] = counts
    text.append("")
    return text


def slice_tables(report, judged_all, fact, queries, oids, tag, feats):
    text = ["== Stress slices (base reference; the newest-four-weeks cut is validation "
            f"date >= {NEWEST_FROM}) =="]
    groups = defaultdict(lambda: defaultdict(list))
    region_n = Counter()
    for o in oids:
        q = queries[o]
        region_n[q["region"][0] or "unknown"] += 1
    for o in oids:
        q = queries[o]
        groups["observer's reference records (A1)"][stress.OBSERVER_REFS.of(
            fact["obs_refs"][o])].append(o)
        km = fact["nearest_same_species_km"][o]
        groups["distance to nearest true-species reference"][
            "no true-species ref" if (fact["truth_refs"][o] <= 0)
            else "true species ref has no place" if math.isnan(km)
            else stress.NEAREST_SAME_SPECIES_KM.of(km)].append(o)
        r100 = fact["refs_100km"][o]
        groups["reference records (any species) within 100 km"][
            "no place" if r100 < 0 else stress.REFS_WITHIN_100KM.of(r100)].append(o)
        sh = fact["season_share"][o]
        groups["season: true-species refs within 30 days of the query's day"][
            stress.season_band(None if math.isnan(sh) else sh)].append(o)
        groups["photos per observation"][stress.PHOTOS.of(len(q["photo_ids"]))].append(o)
        fv = (q["first_validated"] or "")[:10]
        groups["validated"][
            "newest 4 weeks" if fv >= NEWEST_FROM
            else "4-8 weeks before" if fv >= "2026-08-14" else "older"].append(o)
        st = q["region"][0] or "unknown"
        groups["state / province (nearest reference record's)"][
            st if region_n[st] >= MIN_REGION else "other (smaller)"].append(o)
        groups["country"][q["region"][1] or "unknown"].append(o)
        groups["true species seen by this observer in the reference"][
            "n/a" if fact["truth_refs"][o] < 0 else
            "yes" if fact["truth_refs_obs"][o] > 0 else "no"].append(o)
    orders = {
        "observer's reference records (A1)": list(stress.OBSERVER_REFS.labels),
        "photos per observation": list(stress.PHOTOS.labels),
        "season: true-species refs within 30 days of the query's day":
            [stress.season_band(x) for x in (0.6, 0.3, 0.1, 0.0, None)],
        "validated": ["newest 4 weeks", "4-8 weeks before", "older"],
        "reference records (any species) within 100 km":
            list(stress.REFS_WITHIN_100KM.labels) + ["no place"],
        "distance to nearest true-species reference":
            list(stress.NEAREST_SAME_SPECIES_KM.labels) + ["true species ref has no place",
                                                          "no true-species ref"],
    }
    band_of = {o: feats[o]["species reference records"] for o in oids}
    counts = Counter(b for o, b in band_of.items() if b != "n/a")
    weights = {b: c / sum(counts.values()) for b, c in counts.items() if b != "0"}
    text.append("  (sp@1 depth-adj: species top-1 standardised to the whole dev set's mix of "
                "1-4 / 5-19 / 20-99 / 100+ reference bands; 0-ref truths left out)")
    out = {}
    for method in ("nearest+mean", "nearest", "nearest+prior@org"):
        jd = judged_all.get((tag, "base", method))
        if jd is None:
            continue
        text.append(f"-- {method}")
        out[method] = {}
        for title, g in groups.items():
            order = orders.get(title) or sorted(g, key=lambda k: -len(g[k]))
            rows = slice_table(g, jd, [k for k in order if k in g], band_of, weights)
            out[method][title] = rows
            text += fmt_slice(title, rows)
        for first in ("nearest+mean first photo", "nearest first photo"):
            if first.startswith(method + " ") and (tag, "base", first) in judged_all:
                fj = judged_all[(tag, "base", first)]
                allp = {o: jd[o] for o in jd}
                multi = [o for o in oids if len(queries[o]["photo_ids"]) > 1]
                rows = slice_table({"all photos": multi}, allp) + \
                    slice_table({"first photo only": multi}, fj)
                out[method]["first photo vs all (records with 2+ photos)"] = rows
                text += fmt_slice("first photo vs all photos (records with 2+ photos)", rows)
        text.append("")
    report["slices"] = out
    return text


if __name__ == "__main__":
    main()
