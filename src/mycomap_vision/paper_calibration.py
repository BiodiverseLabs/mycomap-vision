"""Calibration and confidence for `mv paper-score` (paper E13, E11): how far the stated
confidence can be trusted on the scored finds, with every setting taken from val and none
refitted here.

Per find and rank a comparator gives (paper_score.vision_answers, or iNat's own score):
`conf` (the top answer's probability), `p_true` (the true name's probability; 0 when the
reference lacks it), `sumsq` (the sum of squared probabilities, for the multiclass Brier
score) and `set` (the likely set's names, from the val floor). From these:

- ECE over `bins` equal-mass bins (default 15), with the reliability bins themselves;
- Brier = sumsq - 2 p_true + 1; NLL = -log max(p_true, 1e-12);
- a confidence rule (e.g. 70%): of the finds whose top answer reaches it, the share right
  (precision) with its observer-clustered interval and one-sided 95% lower bound, and the
  share of finds that reach it (coverage);
- risk-coverage: accuracy of the most confident q of finds, q = 10%, 20%, ... 100%, and the
  area under the risk-coverage curve;
- likely sets: coverage and mean size, overall and by reference band;
- E11: AUROC of the top answer's confidence for telling finds whose species has a
  reference from finds whose species has none, and the share of each under a "not sure"
  confidence.
"""

from __future__ import annotations

import math

import numpy as np

from . import paper_stats as ps

BINS = 15
FLOOR = 1e-12


def ece(conf, right, bins: int = BINS) -> dict:
    """Expected calibration error over equal-mass bins (sorted by confidence, split into
    `bins` groups of as equal size as possible), with each bin's mean confidence, accuracy
    and n."""
    conf, right = np.asarray(conf, float), np.asarray(right, float)
    n = len(conf)
    if not n:
        return {"n": 0, "ece": None, "bins": []}
    order = np.argsort(conf, kind="stable")
    parts = [p for p in np.array_split(order, min(bins, n)) if len(p)]
    rows, total = [], 0.0
    for p in parts:
        c, a = float(conf[p].mean()), float(right[p].mean())
        total += len(p) / n * abs(a - c)
        rows.append({"n": int(len(p)), "confidence": c, "accuracy": a})
    return {"n": n, "ece": total, "bins": rows}


def brier_nll(p_true, sumsq) -> dict:
    p_true, sumsq = np.asarray(p_true, float), np.asarray(sumsq, float)
    if not len(p_true):
        return {"n": 0, "brier": None, "nll": None}
    return {"n": int(len(p_true)), "brier": float(np.mean(sumsq - 2 * p_true + 1)),
            "nll": float(np.mean(-np.log(np.maximum(p_true, FLOOR))))}


def confidence_rule(conf, right, who, threshold: float, reps: int, seed: int) -> dict:
    """Precision and coverage of "the top answer when its confidence reaches `threshold`"."""
    conf, right = np.asarray(conf, float), np.asarray(right, float)
    keep = conf >= threshold
    n = int(keep.sum())
    out = {"threshold": threshold, "finds": int(len(conf)), "confident": n,
           "coverage": n / len(conf) if len(conf) else None,
           "precision": float(right[keep].mean()) if n else None, "ci95": None,
           "lower_bound_one_sided_95": None}
    if n:
        w = [x for x, k in zip(who, keep) if k]
        out["ci95"] = ps.cluster_ci(right[keep], w, reps, seed)
        lo = ps.cluster_ci(right[keep], w, reps, seed, level=0.90)
        out["lower_bound_one_sided_95"] = lo[0] if lo else None
    return out


def risk_coverage(conf, right, steps: int = 10) -> dict:
    """Accuracy of the most confident q of finds for q = 1/steps ... 1, and the area under
    the risk (1 - accuracy) curve over every coverage (lower is better)."""
    conf, right = np.asarray(conf, float), np.asarray(right, float)
    n = len(conf)
    if not n:
        return {"n": 0, "points": [], "aurc": None}
    order = np.argsort(-conf, kind="stable")
    acc = np.cumsum(right[order]) / np.arange(1, n + 1)
    points = [{"coverage": q / steps, "accuracy": float(acc[max(0, math.ceil(n * q / steps) - 1)])}
              for q in range(1, steps + 1)]
    return {"n": n, "points": points, "aurc": float(np.mean(1 - acc))}


def auroc(pos, neg) -> float | None:
    """P(a positive scores above a negative), ties half (Mann-Whitney)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    i = 0
    while i < len(sv):                      # average ranks over ties
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def summarise(finds, cal: dict, outcome: dict, rank: str, cfg: dict, reps: int, seed: int,
              band_of) -> dict:
    """Every calibration figure for one comparator at one rank, on `finds` (primary scope).
    `cal`: {record key: {rank: {conf, p_true, sumsq, set}}}; `outcome`: its per-find cells
    (the top-1 right/wrong); `band_of(find)`: the reference band."""
    cell = f"{rank}_strict@1"
    rows = []
    for f in finds:
        c = (cal.get(f.key) or {}).get(rank)
        right = (outcome.get(f.key) or {}).get(cell)
        if c is None or right is None:
            continue
        rows.append((f, c, bool(right)))
    conf = [c["conf"] for _, c, _ in rows]
    right = [r for _, _, r in rows]
    who = [f.observer for f, _, _ in rows]
    out = {"rank": rank, "n": len(rows), "reliability": ece(conf, right, cfg.get("bins", BINS)),
           "risk_coverage": risk_coverage(conf, right)}
    if rows and "p_true" in rows[0][1]:
        out["scores"] = brier_nll([c["p_true"] for _, c, _ in rows],
                                  [c["sumsq"] for _, c, _ in rows])
    out["rules"] = [confidence_rule(conf, right, who, t, reps, seed)
                    for t in cfg.get("rules", [0.7])]
    with_sets = [(f, c) for f, c, _ in rows if c.get("set") is not None]
    if with_sets:
        truth = (lambda f: f.species) if rank == "species" else (lambda f: f.genus)
        by_band: dict = {}
        for f, c in with_sets:
            b = by_band.setdefault(band_of(f), {"n": 0, "hits": 0, "size": 0})
            b["n"] += 1
            b["hits"] += truth(f) in c["set"]
            b["size"] += len(c["set"])
        tot = {k: sum(b[k] for b in by_band.values()) for k in ("n", "hits", "size")}
        hits = [truth(f) in c["set"] for f, c in with_sets]
        # The coverage achieved on these finds, never the target the val floor was fit
        # for (outside review, 18 Oct): the target is printed beside it, not instead.
        out["likely_sets"] = {
            "n": tot["n"], "coverage": tot["hits"] / tot["n"],
            "ci95": ps.cluster_ci(hits, [f.observer for f, _ in with_sets], reps, seed),
            "target": (cfg.get("list_target") or {}).get(rank)
            if isinstance(cfg.get("list_target"), dict) else cfg.get("list_target"),
            "mean_size": tot["size"] / tot["n"],
            "by_band": {b: {"n": v["n"], "coverage": v["hits"] / v["n"],
                            "mean_size": v["size"] / v["n"]} for b, v in sorted(by_band.items())}}
    if rank == "species":
        known = [c["conf"] for f, c, _ in rows if f.depth > 0]
        unknown = [c["conf"] for f, c, _ in rows if f.depth == 0]
        e11 = {"known": len(known), "unknown": len(unknown),
               "auroc_known_vs_unknown": auroc(known, unknown)}
        if cfg.get("not_sure_below") is not None:
            t = cfg["not_sure_below"]
            e11["not_sure_share"] = {
                "known": float(np.mean(np.array(known) < t)) if known else None,
                "unknown": float(np.mean(np.array(unknown) < t)) if unknown else None,
                "below": t}
        g = [(outcome.get(f.key) or {}).get("genus_strict@1") for f, _, _ in rows
             if f.depth == 0]
        g = [bool(x) for x in g if x is not None]
        e11["genus_top1_without_species_reference"] = {
            "n": len(g), "rate": float(np.mean(g)) if g else None}
        out["unknown_species"] = e11
    return out
