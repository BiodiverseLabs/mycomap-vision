"""Tables, curves, fits and the "sequence these next" list from a learning_curve run.

Reads the run's results.pkl (private: it holds observation ids) and writes only
aggregates: report.json, report.md and a chart (SVG) beside it. No record ids,
coordinates, people or photo paths leave here; species and genera appear by name.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import name_equiv
from .learning_curve import BANDS, METHODS, band_of, fit_saturating, saturating

KS = (1, 3, 5, 10)
LEVELS = (("species", "strict"), ("species", "sl"), ("species", "complex"),
          ("genus", "strict"), ("genus", "sl"))
CURVE_NS = (1, 2, 3, 5, 10, 20, 50, 100)
GAIN_DS = (1, 2, 3, 5, 10, 20, 50)
GAIN_KS = (1, 5, 10)


@lru_cache(maxsize=None)
def _sp(answer: str, truth: str) -> tuple[bool, bool]:
    m = name_equiv.species_match(answer, truth)
    return m["sl"], m["complex"]


@lru_cache(maxsize=None)
def _ge(answer: str, truth: str) -> bool:
    return name_equiv.genus_match(answer, truth)["sl"]


def first_hits(species: list[int], genus: list[int], truth: dict, units: list[str],
               genus_names: list[str]) -> dict:
    """{(rank, level): 0-based position of the first right answer or None}. Strict uses
    the run's label match (the truth's units / genera); s.l. and complex name_equiv. A
    stricter hit counts at every looser level."""
    out = {}
    if truth["species"]:
        tu = set(truth["units"])
        strict = [u in tu for u in species]
        sl, cx = [], []
        for u, s in zip(species, strict):
            a, b = _sp(units[u], truth["species"])
            sl.append(s or a)
            cx.append(s or a or b)
        for level, hits in (("strict", strict), ("sl", sl), ("complex", cx)):
            out[("species", level)] = next((i for i, h in enumerate(hits) if h), None)
    if truth["genus"]:
        tg = set(truth["genera"])
        strict = [g in tg for g in genus]
        sl = [s or _ge(genus_names[g], truth["genus"]) for g, s in zip(genus, strict)]
        out[("genus", "strict")] = next((i for i, h in enumerate(strict) if h), None)
        out[("genus", "sl")] = next((i for i, h in enumerate(sl) if h), None)
    return out


def ladder(hits: dict[str, dict]) -> dict:
    """{level: {"n", "top1", ...}} in percent."""
    out = {}
    for rank, level in LEVELS:
        pos = [h[(rank, level)] for h in hits.values() if (rank, level) in h]
        n = len(pos)
        out[f"{rank} {level}"] = {"n": n, **{f"top{k}": (100.0 * sum(p is not None and p < k
                                                                    for p in pos) / n if n else None)
                                             for k in KS}}
    return out


def depth_table(hits: dict[str, dict], depth: dict[str, int]) -> dict:
    out = {}
    for lo, hi, label in BANDS:
        pos = [h[("species", "strict")] for o, h in hits.items()
               if ("species", "strict") in h and lo <= depth[o] <= hi]
        n = len(pos)
        out[label] = {"n": n,
                      "top1": 100.0 * sum(p == 0 for p in pos) / n if n else None,
                      "top5": 100.0 * sum(p is not None and p < 5 for p in pos) / n if n else None}
    return out


def judge_condition(res: dict, truths: dict, units, genus_names) -> dict[str, dict]:
    return {oid: first_hits([u for u, _ in e["species"]], [g for g, _ in e["genus"]],
                            truths[oid], units, genus_names)
            for oid, e in res.items()}


def truth_depth(truths: dict, counts: np.ndarray) -> dict[str, int]:
    return {o: int(sum(counts[u] for u in t["units"])) for o, t in truths.items()}


def rate(hits: dict, ids, rank="species", level="strict", k=1) -> float | None:
    pos = [hits[o][(rank, level)] for o in ids if (rank, level) in hits[o]]
    return 100.0 * sum(p is not None and p < k for p in pos) / len(pos) if pos else None


def spread(vals: list[float]) -> dict:
    v = [x for x in vals if x is not None]
    if not v:
        return {"mean": None}
    return {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0,
            "min": float(min(v)), "max": float(max(v)), "seeds": len(v)}


# --- truth caps: only the true species cut to N --------------------------------------------

def capped_hits(oid: str, score: float, others: dict, truth: dict, unit: int, units,
                genus_of_unit, genus_names, top: int = 10) -> dict:
    """The record's answers when its true species scores `score` and every other species
    keeps its full-reference score: the truth slotted into the others' list."""
    sp = others["species"]
    p = sum(1 for _u, s in sp if s > score)
    species = [u for u, _ in sp[:p]] + [unit] + [u for u, _ in sp[p:]]
    species = species[:top]
    g_u = int(genus_of_unit[unit])
    gl = dict(others["genus"])
    if g_u >= 0:
        gl[g_u] = max(gl.get(g_u, -np.inf), score)
    genus = [g for g, _ in sorted(gl.items(), key=lambda kv: -kv[1])][:top]
    return first_hits(species, genus, truth, units, genus_names)


def bootstrap_ci(diffs: np.ndarray, reps: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(diffs) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = diffs[rng.integers(0, len(diffs), (reps, len(diffs)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


PER_RECORD = ("base_others", "truth_info", "truths", "n_photos")


def merge_results(parts: list[dict]) -> dict:
    """One run from its shard files: the same reference and conditions, disjoint records."""
    R = parts[0]
    for P in parts[1:]:
        if (P["reference_hash"], P["conditions"], P["caps"]) != (R["reference_hash"],
                                                                 R["conditions"], R["caps"]):
            raise ValueError("shards come from different runs")
        for c in R["conditions"]:
            for m in METHODS:
                R["results"][c][m].update(P["results"][c][m])
        for m in METHODS:
            R["capped"][m].update(P["capped"][m])
        for k in PER_RECORD:
            R[k].update(P[k])
        for k, v in P["skipped"].items():
            R["skipped"][k] = max(R["skipped"].get(k, 0), v)
    return R


def load_results(path: Path) -> dict:
    files = sorted(path.glob("results-*of*.pkl")) if path.is_dir() else [path]
    parts = []
    for f in files:
        with open(f, "rb") as fh:
            parts.append(pickle.load(fh))
    if not parts:
        raise FileNotFoundError(f"no results in {path}")
    return merge_results(parts)


def report(results_path: Path, out_dir: Path, pool_names: dict[str, int] | None = None,
           inat_counts: dict[str, int] | None = None) -> dict:
    R = load_results(results_path)
    units, genus_names = R["units"], R["genus_names"]
    genus_of_unit = R["genus_of_unit"]
    truths = R["truths"]
    seeds = R["seeds"]
    out: dict = {"reference_hash": R["reference_hash"], "backbone": R["backbone"],
                 "benchmark": R["benchmark"], "split": R["split"],
                 "reference_records": R["records"], "reference_photos": R["photos"],
                 "bad_in_reference": R["bad_in_reference"], "bad_ids_total": R["bad_ids_total"],
                 "bad_also_inat_left_in": R["bad_also_inat"], "skipped": R["skipped"],
                 "scored_records": len(truths)}
    judged = {c: {m: judge_condition(R["results"][c][m], truths, units, genus_names)
                  for m in METHODS} for c in R["conditions"]}
    full_depth = truth_depth(truths, R["unit_counts"]["full-clean"])
    full_band = {o: band_of(d) for o, d in full_depth.items()}
    species_ids = [o for o, t in truths.items() if t["species"]]

    # 1. Standard tables at 100% (clean and with the wrong-photo records).
    out["standard"] = {}
    for c in ("full-clean", "full-with-bad"):
        depth = truth_depth(truths, R["unit_counts"][c])
        out["standard"][c] = {m: {"ladder": ladder(judged[c][m]),
                                  "by_depth": depth_table(judged[c][m], depth)} for m in METHODS}

    # 2. Fractions: every species keeps 25 / 50 / 75 / 100% of its records.
    fr = {}
    for m in METHODS:
        fr[m] = {}
        for f in list(R["fractions"]) + [1.0]:
            conds = (["full-clean"] if f == 1.0 else [f"frac{int(f * 100)}-s{s}" for s in seeds])
            row = {}
            for _lo, _hi, label in BANDS:
                ids = [o for o in species_ids if full_band[o] == label]
                row[label] = {"n": len(ids),
                              "top1": spread([rate(judged[c][m], ids) for c in conds]),
                              "top5": spread([rate(judged[c][m], ids, k=5) for c in conds]),
                              "refs_kept_median": float(np.median(
                                  [truth_depth({o: truths[o]}, R["unit_counts"][conds[0]])[o]
                                   for o in ids])) if ids else None}
            row["all species"] = {"n": len(species_ids),
                                  "top1": spread([rate(judged[c][m], species_ids) for c in conds]),
                                  "top5": spread([rate(judged[c][m], species_ids, k=5)
                                                  for c in conds])}
            gids = [o for o, t in truths.items() if t["genus"]]
            row["genus"] = {"n": len(gids),
                            "top1": spread([rate(judged[c][m], gids, "genus") for c in conds]),
                            "top5": spread([rate(judged[c][m], gids, "genus", k=5) for c in conds])}
            for _lo, _hi, label in BANDS:
                ids = [o for o in gids if o in full_band and truths[o]["species"]
                       and full_band[o] == label]
                row[f"genus | species band {label}"] = {
                    "n": len(ids), "top1": spread([rate(judged[c][m], ids, "genus") for c in conds])}
            fr[m][f"{int(f * 100)}%"] = row
    out["fractions"] = fr

    # 3. Global caps: every species keeps at most N records.
    gc = {}
    for m in METHODS:
        gc[m] = {}
        for cap in R["global_caps"]:
            conds = [c for c in R["conditions"] if c.startswith(f"gcap{cap}-")]
            gc[m][str(cap)] = {label: spread([rate(judged[c][m], [o for o in species_ids
                                                                 if full_band[o] == label])
                                              for c in conds]) for _lo, _hi, label in BANDS[1:]}
            gc[m][str(cap)]["genus"] = spread([rate(judged[c][m], [o for o, t in truths.items()
                                                                   if t["genus"]], "genus")
                                               for c in conds])
    out["global_caps"] = gc

    # 4. Truth caps ("add N more"): only the true species cut to N.
    info = R["truth_info"]
    cap_ids = [o for o in species_ids if o in info and info[o]["n_clean"] >= 1]
    hits_at: dict = {m: defaultdict(dict) for m in METHODS}   # m -> (cap, seed) -> oid -> hits
    for m in METHODS:
        for o in cap_ids:
            u, n = info[o]["unit"], info[o]["n_clean"]
            others = R["base_others"][o][m]
            for (cap, s), score in R["capped"][m].get(o, {}).items():
                hits_at[m][(cap, s)][o] = capped_hits(o, score, others, truths[o], u, units,
                                                      genus_of_unit, genus_names)

    def acc_at(m, o_list, N, s, rank="species", k=1):
        """Accuracy over o_list when each keeps min(N, n) records of its true species."""
        vals = []
        for o in o_list:
            n = info[o]["n_clean"]
            h = hits_at[m].get((min(N, n), s), {}).get(o)
            if h is None or (rank, "strict") not in h:
                continue
            p = h[(rank, "strict")]
            vals.append(p is not None and p < k)
        return 100.0 * float(np.mean(vals)) if vals else None

    # Check: the truth-cap machinery at full depth gives the full-clean answers.
    agree = Counter()
    for o in cap_ids:
        n = info[o]["n_clean"]
        h = hits_at["nearest+mean"][(n, seeds[0])].get(o)
        agree[h[("species", "strict")] == judged["full-clean"]["nearest+mean"][o][("species", "strict")]
              if h else "missing"] += 1
    out["truth_cap_full_agrees_with_full_reference"] = dict((str(k), v) for k, v in agree.items())

    caps_view = {}
    fits = {}
    for m in METHODS:
        caps_view[m], fits[m] = {}, {}
        for _lo, _hi, label in BANDS[1:]:
            ids = [o for o in cap_ids if full_band[o] == label]
            row = {"n": len(ids)}
            for N in CURVE_NS:
                row[str(N)] = spread([acc_at(m, ids, N, s) for s in seeds])
                row[f"{N} top5"] = spread([acc_at(m, ids, N, s, k=5) for s in seeds])
                row[f"{N} genus"] = spread([acc_at(m, ids, N, s, "genus") for s in seeds])
            row["full"] = spread([acc_at(m, ids, 10**9, s) for s in seeds[:1]])
            caps_view[m][label] = row
            # Saturating fit on the band's curve (N = 0 means not in the index: 0%).
            ns = [0] + list(CURVE_NS)
            ys = [0.0] + [row[str(N)]["mean"] / 100 for N in CURVE_NS]
            fits[m][label] = fit_saturating(np.array(ns[1:]), np.array(ys[1:]))
        # Within-species curve of the deep species (100+): every N is a real cut.
        deep = [o for o in cap_ids if info[o]["n_clean"] >= 100]
        ys = [np.mean([acc_at(m, deep, N, s) for s in seeds]) / 100 for N in CURVE_NS]
        fits[m]["deep species, cut to N"] = fit_saturating(np.array(CURVE_NS), np.array(ys))
        fits[m]["deep species, cut to N"]["n"] = len(deep)
        caps_view[m]["deep species, cut to N"] = {"n": len(deep), **{
            str(N): {"mean": 100 * y} for N, y in zip(CURVE_NS, ys)}}
    out["truth_caps"] = caps_view
    out["fits"] = fits

    # 5. Paired marginal gains: records whose species has >= d + k records, cut to d
    # and to d + k (nested, same seed), the change in top-1.
    gains = {}
    for m in METHODS:
        gains[m] = {}
        for d in GAIN_DS:
            for k in GAIN_KS:
                ids = [o for o in cap_ids if info[o]["n_clean"] >= d + k]
                per_seed, diffs = [], []
                for s in seeds:
                    a = hits_at[m].get((d, s), {})
                    b = hits_at[m].get((d + k, s), {})
                    dd = [(b[o][("species", "strict")] == 0) - (a[o][("species", "strict")] == 0)
                          for o in ids if o in a and o in b]
                    if dd:
                        per_seed.append(100.0 * float(np.mean(dd)))
                        diffs.append(np.array(dd, float))
                if not per_seed:
                    continue
                avg = np.mean(np.stack(diffs), axis=0) * 100 if len({len(x) for x in diffs}) == 1 else None
                lo, hi = bootstrap_ci(avg) if avg is not None else (None, None)
                gains[m][f"{d}+{k}"] = {"n": len(ids), "gain": float(np.mean(per_seed)),
                                        "seed_sd": float(np.std(per_seed, ddof=1)) if len(per_seed) > 1 else 0.0,
                                        "ci95": [lo, hi]}
    out["marginal_gains"] = gains

    # 6. Sequence-these-next: expected gain of five more records x how often the species
    # arrives (held-out pool) / is observed (iNat).
    out["sequence_next"] = sequence_next(R, fits, gains, pool_names or {}, inat_counts or {})

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    (out_dir / "report.md").write_text(format_report(out), encoding="utf-8")
    (out_dir / "learning-curve.svg").write_text(chart_svg(out), encoding="utf-8")
    return out


def gain_of(depth: int, k: int, curve: dict) -> float:
    """Expected top-1 points from k more records for a species with `depth` records, from
    the band's fitted curve (0 records: not in the index, 0%)."""
    band = band_of(max(depth, 1))
    fit = curve.get(band) or curve["1-4"]
    now = 0.0 if depth == 0 else float(saturating(depth, fit))
    return 100.0 * (float(saturating(depth + k, fit)) - now)


def sequence_next(R: dict, fits: dict, gains: dict, pool: dict[str, int],
                  inat: dict[str, int], method: str = "nearest+mean", k: int = 5) -> dict:
    """Species and genera ranked by expected gain (top-1 points from k more records)
    times how often the species arrives in MycoMap's own stream of DNA records (the
    13,145-record held-out pool, both splits). iNat's North American observation count
    is shown beside it, and a second ranking uses it."""
    counts = R["unit_counts"]["full-clean"]
    units, species_of_unit = R["units"], R["species_of_unit"]
    genus_names, genus_of_unit = R["genus_names"], R["genus_of_unit"]
    from .heldout import name_key
    depth = {units[u]: int(counts[u]) for u in range(len(units)) if species_of_unit[u] >= 0}
    # Arrivals and iNat counts are keyed by label; fold writing differences onto the unit.
    unit_by_key = {name_key(n): n for n in depth}
    folded_pool, folded_inat = Counter(), {}
    for n, c in pool.items():
        folded_pool[unit_by_key.get(name_key(n), n)] += c
    for n, c in inat.items():
        folded_inat.setdefault(unit_by_key.get(name_key(n), n), c)
    pool, inat = folded_pool, folded_inat
    names = set(depth) | set(pool)
    total_pool = max(sum(pool.values()), 1)
    rows = []
    for name in names:
        d = depth.get(name, 0)
        g = gain_of(d, k, fits[method])
        arrivals = pool.get(name, 0)
        rows.append({"species": name, "refs": d, "band": band_of(d), "gain_pts": g,
                     "pool_arrivals": arrivals, "inat_na_obs": inat.get(name),
                     "expected": g * arrivals / total_pool * 100})
    by_pool = sorted([r for r in rows if r["pool_arrivals"] > 0],
                     key=lambda r: (-r["expected"], r["species"]))
    total_inat = max(sum(v for v in inat.values() if v), 1)
    for r in rows:
        r["expected_inat"] = (r["gain_pts"] * (r["inat_na_obs"] or 0) / total_inat * 100)
    by_inat = sorted([r for r in rows if r["inat_na_obs"]], key=lambda r: -r["expected_inat"])
    genus = defaultdict(lambda: {"expected": 0.0, "pool_arrivals": 0, "species": 0,
                                 "species_under_20": 0, "refs": 0})
    for r in rows:
        gname = r["species"].split()[0]
        G = genus[gname]
        G["expected"] += r["expected"]
        G["pool_arrivals"] += r["pool_arrivals"]
        G["species"] += 1
        G["species_under_20"] += r["refs"] < 20
        G["refs"] += r["refs"]
    genera = sorted(({"genus": g, **v} for g, v in genus.items() if v["pool_arrivals"]),
                    key=lambda r: -r["expected"])
    return {"method": method, "k": k, "pool_total": total_pool,
            "species_by_pool": by_pool[:60], "species_by_inat": by_inat[:40],
            "genera": genera[:30],
            "absent_species_arrivals": sum(r["pool_arrivals"] for r in rows if r["refs"] == 0),
            "share_of_expected_in_bands": {
                b: sum(r["expected"] for r in by_pool if r["band"] == b)
                / max(sum(r["expected"] for r in by_pool), 1e-9) for _l, _h, b in BANDS}}


# --- output ----------------------------------------------------------------------------

def _f(x, nd=1):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def _ms(s: dict) -> str:
    if not s or s.get("mean") is None:
        return "–"
    if s.get("seeds", 1) > 1:
        return f"{s['mean']:.1f} ±{s['sd']:.1f}"
    return f"{s['mean']:.1f}"


def format_report(o: dict) -> str:
    L = [f"# Learning curve — {o['benchmark']} {o['split']} (reference {o['reference_hash']})", "",
         f"Model {o['backbone']}; reference {o['reference_records']:,} records / "
         f"{o['reference_photos']:,} photos; {o['scored_records']:,} records scored; "
         f"wrong-photo non-iNat records in the reference: {o['bad_in_reference']:,} "
         f"(excluded except in 'full-with-bad'); {o['bad_also_inat_left_in']} ids green as both "
         f"iNat and non-iNat left in. Truth-cap check (full depth == full reference): "
         f"{o['truth_cap_full_agrees_with_full_reference']}.", ""]
    for c, by_m in o["standard"].items():
        for m, t in by_m.items():
            L += [f"## Standard table, {c}, {m}", "", "| row | n | top 1 | top 3 | top 5 | top 10 |",
                  "|---|---|---|---|---|---|"]
            for row, v in t["ladder"].items():
                L.append(f"| {row} | {v['n']:,} | " + " | ".join(_f(v[f'top{k}']) for k in KS) + " |")
            L += ["", "| true species refs | n | top 1 | top 5 |", "|---|---|---|---|"]
            for b, v in t["by_depth"].items():
                L.append(f"| {b} | {v['n']:,} | {_f(v['top1'])} | {_f(v['top5'])} |")
            L.append("")
    for m, by_f in o["fractions"].items():
        L += [f"## Fractions of every species' records, {m} (species top-1 by FULL band; mean ±sd "
              "over seeds)", "",
              "| kept | " + " | ".join(f"{b}" for _l, _h, b in BANDS) + " | all species | genus |",
              "|---|" + "---|" * (len(BANDS) + 2)]
        for f, row in by_f.items():
            L.append(f"| {f} | " + " | ".join(_ms(row[b]["top1"]) for _l, _h, b in BANDS)
                     + f" | {_ms(row['all species']['top1'])} | {_ms(row['genus']['top1'])} |")
        first = next(iter(by_f.values()))
        L.append("| n | " + " | ".join(f"{first[b]['n']:,}" for _l, _h, b in BANDS)
                 + f" | {first['all species']['n']:,} | {first['genus']['n']:,} |")
        L.append("")
    for m, by_c in o["global_caps"].items():
        L += [f"## Every species capped at N records, {m} (species top-1 by FULL band)", "",
              "| N | " + " | ".join(b for _l, _h, b in BANDS[1:]) + " | genus |",
              "|---|" + "---|" * len(BANDS)]
        for cap, row in by_c.items():
            L.append(f"| {cap} | " + " | ".join(_ms(row[b]) for _l, _h, b in BANDS[1:])
                     + f" | {_ms(row['genus'])} |")
        L.append("")
    for m, by_b in o["truth_caps"].items():
        L += [f"## Add N more: only the true species keeps min(N, its records), {m} (top-1)", "",
              "| full band | n | " + " | ".join(str(N) for N in CURVE_NS) + " | full |",
              "|---|---|" + "---|" * (len(CURVE_NS) + 1)]
        for b, row in by_b.items():
            L.append(f"| {b} | {row['n']:,} | " + " | ".join(_ms(row.get(str(N))) for N in CURVE_NS)
                     + f" | {_ms(row.get('full'))} |")
        L += ["", "Genus top-1 under the same cut:", "",
              "| full band | " + " | ".join(str(N) for N in CURVE_NS) + " |",
              "|---|" + "---|" * len(CURVE_NS)]
        for b, row in by_b.items():
            if f"1 genus" in row:
                L.append(f"| {b} | " + " | ".join(_ms(row.get(f'{N} genus')) for N in CURVE_NS) + " |")
        L += ["", "Fits acc(N) = ceiling x N / (N + half):", "",
              "| curve | ceiling | half at | knee (<1 pt / record) | 90% at | rmse |", "|---|---|---|---|---|---|"]
        for b, f in o["fits"][m].items():
            L.append(f"| {b} | {100 * f['ceiling']:.1f} | {f['half_at']:.1f} | {f['knee']:.0f} | "
                     f"{f['n90']:.0f} | {100 * f['rmse']:.1f} |")
        L += ["", f"Paired marginal gains, {m} (records whose species has >= d + k records; "
              "top-1 points; 95% bootstrap over records):", "",
              "| at d | +1 | +5 | +10 |", "|---|---|---|---|"]
        g = o["marginal_gains"][m]
        for d in GAIN_DS:
            cells = []
            for k in GAIN_KS:
                v = g.get(f"{d}+{k}")
                cells.append("–" if not v else
                             f"{v['gain']:+.1f} ({_f(v['ci95'][0])}..{_f(v['ci95'][1])}; n {v['n']:,})")
            L.append(f"| {d} | " + " | ".join(cells) + " |")
        L.append("")
    sn = o["sequence_next"]
    L += [f"## Sequence these next ({sn['method']}, +{sn['k']} records; expected = gain x share of "
          f"{sn['pool_total']:,} held-out arrivals)", "",
          "| # | species | refs | gain (pts) | arrivals | iNat NA obs | expected |",
          "|---|---|---|---|---|---|---|"]
    for i, r in enumerate(sn["species_by_pool"][:40], 1):
        L.append(f"| {i} | {r['species']} | {r['refs']} | {r['gain_pts']:.1f} | {r['pool_arrivals']} | "
                 f"{r['inat_na_obs'] if r['inat_na_obs'] is not None else '–'} | {r['expected']:.3f} |")
    L += ["", "| # | genus | expected | arrivals | species | of them < 20 refs |", "|---|---|---|---|---|---|"]
    for i, r in enumerate(sn["genera"][:25], 1):
        L.append(f"| {i} | {r['genus']} | {r['expected']:.2f} | {r['pool_arrivals']} | {r['species']} | "
                 f"{r['species_under_20']} |")
    L += ["", "Share of the expected gain by band: " + ", ".join(
        f"{b} {100 * v:.0f}%" for b, v in sn["share_of_expected_in_bands"].items()), ""]
    return "\n".join(L) + "\n"


def chart_svg(o: dict, method: str = "nearest+mean") -> str:
    """Species top-1 against the true species' records kept (truth caps), one line per
    full-depth band, seed spread as a band; log x."""
    W, H, ml, mr, mt, mb = 720, 420, 60, 150, 30, 50
    xs = [1, 2, 3, 5, 10, 20, 50, 100]
    def X(n):
        return ml + (math.log(n) - math.log(1)) / (math.log(100) - math.log(1)) * (W - ml - mr)
    def Y(a):
        return mt + (1 - a / 100) * (H - mt - mb)
    colors = {"1-4": "#d95f02", "5-19": "#7570b3", "20-99": "#1b9e77", "100+": "#e7298a",
              "deep species, cut to N": "#666666"}
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'font-family="sans-serif" font-size="12">',
             f'<rect width="{W}" height="{H}" fill="white"/>']
    for a in range(0, 101, 20):
        parts.append(f'<line x1="{ml}" x2="{W - mr}" y1="{Y(a)}" y2="{Y(a)}" stroke="#ddd"/>'
                     f'<text x="{ml - 8}" y="{Y(a) + 4}" text-anchor="end">{a}</text>')
    for n in xs:
        parts.append(f'<text x="{X(n)}" y="{H - mb + 18}" text-anchor="middle">{n}</text>')
    parts.append(f'<text x="{(ml + W - mr) / 2}" y="{H - 10}" text-anchor="middle">records of the '
                 f'true species kept (others at full depth)</text>')
    parts.append(f'<text x="15" y="{(mt + H - mb) / 2}" transform="rotate(-90 15 {(mt + H - mb) / 2})" '
                 f'text-anchor="middle">species top-1 %</text>')
    for i, (band, row) in enumerate(o["truth_caps"][method].items()):
        pts = [(n, row.get(str(n))) for n in xs if row.get(str(n)) and row[str(n)].get("mean") is not None]
        if not pts:
            continue
        c = colors.get(band, "#000")
        if all("sd" in s for _n, s in pts):
            up = " ".join(f"{X(n):.1f},{Y(s['max']):.1f}" for n, s in pts)
            dn = " ".join(f"{X(n):.1f},{Y(s['min']):.1f}" for n, s in reversed(pts))
            parts.append(f'<polygon points="{up} {dn}" fill="{c}" opacity="0.15"/>')
        line = " ".join(f"{X(n):.1f},{Y(s['mean']):.1f}" for n, s in pts)
        dash = ' stroke-dasharray="5,4"' if band.startswith("deep") else ""
        parts.append(f'<polyline points="{line}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        label = f"{band} (n {row['n']:,})"
        parts.append(f'<text x="{W - mr + 10}" y="{mt + 16 + 18 * i}" fill="{c}">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--results", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frequencies", help="JSON {pool: {name: n}, inat: {name: n}}")
    a = p.parse_args(argv)
    freq = json.loads(Path(a.frequencies).read_text(encoding="utf-8")) if a.frequencies else {}
    report(Path(a.results), Path(a.out), freq.get("pool"), freq.get("inat"))


if __name__ == "__main__":
    main()
