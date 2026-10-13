"""The tables and tests of `mv paper-score`, from finds and per-find outcomes
(paper_outcomes): pure functions, so they are tested on made-up finds and run unchanged on
val, a draft test release and the sealed test.

- `standard`: Steve's table for one comparator: top 1/3/5/10 x species strict / s.l. /
  complex and genus strict / s.l., each cell with n, rate and an observer-clustered
  interval; species top-1 and top-5 by the true species' reference band (0, 1-4, 5-19,
  20-99, 100+). Two denominators: every find, and finds whose species has a reference.
- `paired`: two comparators on the finds both can be scored on for one cell: the rates,
  the difference with its paired cluster-bootstrap interval, the discordant counts and
  the exact McNemar p.
- `families`: the pre-declared hypotheses (config), each family Holm-adjusted.
- `slices`: rates by a find attribute, intervals only (no tests).

Near-duplicates (paper analysis plan: byte-identical, dHash-flagged or cosine >= 0.99 to a
reference photo) are left out of every primary number; `scope="with_near_duplicates"`
gives the same tables with them, and the count is reported beside.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import paper_stats as ps
from .paper_outcomes import CELLS, K, LEVELS, Find

BANDS = ("0", "1-4", "5-19", "20-99", "100+")
DENOMINATORS = ("all", "has_reference")
SCOPES = ("primary", "with_near_duplicates")


def band(depth: int) -> str:
    from .evaluate import standard_band
    return standard_band(depth)


def in_scope(finds: list[Find], scope: str = "primary", denominator: str = "all"
             ) -> list[Find]:
    if scope not in SCOPES or denominator not in DENOMINATORS:
        raise ValueError(f"scope {scope!r} / denominator {denominator!r}")
    out = [f for f in finds if scope == "with_near_duplicates" or not f.is_near_duplicate]
    if denominator == "has_reference":
        out = [f for f in out if f.depth > 0]
    return out


def rate(finds: list[Find], outcome: dict, cell: str, reps: int, seed: int,
         ci: bool = True) -> dict:
    """n, right, rate and the observer-clustered interval of one cell over the finds the
    comparator can be scored on there (outcome not None); a find with no outcome line at
    all counts as unanswered (wrong) when the cell applies to it."""
    vals, who = [], []
    for f in finds:
        o = outcome.get(f.key)
        v = o.get(cell) if o is not None else (False if _applies(f, cell) else None)
        if v is None:
            continue
        vals.append(bool(v))
        who.append(f.observer)
    n, k = len(vals), int(sum(vals))
    return {"n": n, "right": k, "rate": k / n if n else None,
            "ci95": ps.cluster_ci(vals, who, reps, seed) if ci and n else None}


def _applies(f: Find, cell: str) -> bool:
    return bool(f.species) if cell.startswith("species") else bool(f.genus)


def standard(finds: list[Find], outcome: dict, reps: int, seed: int, ci_cells=None) -> dict:
    """Steve's table for one comparator on these finds (see the module docstring).
    `ci_cells`: the cells that get an interval (default: every top-1 cell)."""
    ci_cells = set(ci_cells or [c for c in CELLS if c.endswith("@1")])
    table = {}
    for rank, levels in LEVELS.items():
        for lv in levels:
            table[f"{rank} {lv}"] = {f"top{k}": rate(finds, outcome, f"{rank}_{lv}@{k}", reps,
                                                     seed, f"{rank}_{lv}@{k}" in ci_cells)
                                     for k in K}
    by_band = {}
    for b in BANDS:
        sub = [f for f in finds if f.species and band(f.depth) == b]
        by_band[b] = {"top1": rate(sub, outcome, "species_strict@1", reps, seed),
                      "top5": rate(sub, outcome, "species_strict@5", reps, seed, ci=False)}
    answered = sum(bool((outcome.get(f.key) or {}).get("answered")) for f in finds)
    return {"finds": len(finds), "answered": answered, "table": table,
            "species_by_reference_band": by_band}


def all_standard(finds: list[Find], outcomes: dict[str, dict], reps: int, seed: int) -> dict:
    """Every comparator's table, both denominators, both scopes; the near-duplicate count."""
    out = {"near_duplicates": near_duplicate_counts(finds), "comparators": {}}
    for cid, outcome in outcomes.items():
        out["comparators"][cid] = {
            scope: {den: standard(in_scope(finds, scope, den), outcome, reps, seed)
                    for den in DENOMINATORS} for scope in SCOPES}
    return out


def near_duplicate_counts(finds: list[Find]) -> dict:
    kinds = defaultdict(int)
    for f in finds:
        for k, v in f.near_duplicate.items():
            kinds[k] += bool(v)
    return {"finds": len(finds), "near_duplicate_finds": sum(f.is_near_duplicate for f in finds),
            "by_kind": dict(sorted(kinds.items()))}


# --- paired comparisons and hypotheses -----------------------------------------------------------

def paired(finds: list[Find], a: dict, b: dict, cell: str, alternative: str, reps: int,
           seed: int) -> dict:
    """a vs b on the finds both can be scored on for `cell`."""
    xa, xb, who = _pairs(finds, a, b, cell)
    only_a = int(((xa == 1) & (xb == 0)).sum())
    only_b = int(((xa == 0) & (xb == 1)).sum())
    n = len(xa)
    return {"cell": cell, "n": n, "a_rate": float(xa.mean()) if n else None,
            "b_rate": float(xb.mean()) if n else None,
            "difference": float((xa - xb).mean()) if n else None,
            "ci95": ps.paired_ci(xa, xb, who, reps, seed) if n else None,
            "only_a": only_a, "only_b": only_b, "alternative": alternative,
            "p": ps.mcnemar_exact(only_a, only_b, alternative)}


def select(finds: list[Find], spec: dict, outcomes: dict[str, dict]) -> list[Find]:
    """The finds a hypothesis is tested on: its scope and denominator, then its filters:
    `name_kind` (e.g. formal), `has` (comparators that must have a non-None outcome at the
    cell, e.g. experts who named a species), `attr` {name: value}, and `depth_band` (the
    true species' reference-record band, one of BANDS; "band" is the iut place band)."""
    out = in_scope(finds, spec.get("scope", "primary"), spec.get("denominator", "all"))
    if spec.get("name_kind"):
        out = [f for f in out if f.name_kind == spec["name_kind"]]
    if spec.get("depth_band"):
        if spec["depth_band"] not in BANDS:
            raise ValueError(f"depth_band {spec['depth_band']!r}: one of {BANDS}")
        out = [f for f in out if band(f.depth) == spec["depth_band"]]
    for cid in spec.get("has", []):
        cell = spec["cell"]
        out = [f for f in out if (outcomes[cid].get(f.key) or {}).get(cell) is not None]
    for k, v in (spec.get("attr") or {}).items():
        out = [f for f in out if f.attrs.get(k) == v]
    return out


def test_one(spec: dict, finds: list[Find], outcomes: dict[str, dict], reps: int,
             seed: int) -> dict:
    """One pre-declared hypothesis: kind mcnemar (a vs b on a cell), gee_depth (H3), or
    iut (superiority overall and non-inferiority in a band, one intersection-union test)."""
    kind = spec["kind"]
    if kind == "mcnemar":
        sub = select(finds, spec, outcomes)
        r = paired(sub, outcomes[spec["a"]], outcomes[spec["b"]], spec["cell"],
                   spec.get("alternative", "greater"), reps, seed)
        return {**r, "p_test": r["p"]}
    if kind == "gee_depth":
        sub = [f for f in select(finds, spec, outcomes) if f.species]
        o = outcomes[spec["comparator"]]
        y = [bool((o.get(f.key) or {}).get(spec["cell"])) for f in sub]
        cols, names = [[np.log2(1 + f.depth) for f in sub]], ["log2_refs"]
        if "name_kind" in spec.get("covariates", []):
            cols.append([1.0 if f.name_kind == "provisional" else 0.0 for f in sub])
            names.append("provisional_name")
        if "photos" in spec.get("covariates", []):
            cols.append([float(f.photos) for f in sub])
            names.append("photos")
        # A covariate with one value on these finds is collinear with the intercept: left
        # out and said so (the slope's own column is never dropped).
        dropped = [n for c, n in zip(cols[1:], names[1:]) if np.ptp(c) == 0]
        keep = [i for i, n in enumerate(names) if n not in dropped]
        fit = ps.gee_logistic(y, np.column_stack([cols[i] for i in keep]),
                              [f.observer for f in sub], [names[i] for i in keep])
        p = ps.one_sided_wald(fit["coefficients"]["log2_refs"], "greater")
        return {**fit, "p_test": p, "slope": fit["coefficients"]["log2_refs"],
                "dropped_covariates": dropped}
    if kind == "iut":
        sub = select(finds, spec, outcomes)
        sup = paired(sub, outcomes[spec["a"]], outcomes[spec["b"]], spec["cell"], "greater",
                     reps, seed)
        band_finds = [f for f in sub if f.attrs.get(spec["band_attr"]) == spec["band"]]
        xa, xb, who = _pairs(band_finds, outcomes[spec["a"]], outcomes[spec["b"]],
                             spec["cell"])
        ni = ps.noninferiority(xa, xb, who, spec["margin"], reps, seed)
        return {"superiority": sup, "noninferiority": {**ni, "band": spec["band"]},
                "p_test": ps.intersection_union(sup["p"], ni["p"])}
    raise ValueError(f"unknown hypothesis kind {kind!r}")


def _pairs(finds, a, b, cell):
    """Paired 0/1 values of a and b at `cell`, and the finds' observers: a find either
    cannot be scored on is left out; a comparator with no line for a find it applies to
    counts as not answering (wrong)."""
    xa, xb, who = [], [], []
    for f in finds:
        vals = []
        for o in (a, b):
            line = o.get(f.key)
            vals.append(line.get(cell) if line is not None
                        else (False if _applies(f, cell) else None))
        if None in vals:
            continue
        xa.append(float(vals[0]))
        xb.append(float(vals[1]))
        who.append(f.observer)
    return np.array(xa), np.array(xb), who


def families(spec: dict, finds: list[Find], outcomes: dict[str, dict], reps: int,
             seed: int, alpha: float = 0.05) -> dict:
    """Every family's hypotheses tested and Holm-adjusted within the family. A hypothesis
    whose comparator has no outcomes (its input not given) is reported, not tested, and
    leaves the family's Holm denominator unchanged (its p counts as 1)."""
    out = {}
    for fam, hyps in spec.items():
        results, pvals = {}, {}
        for h in hyps:
            needed = [h[k] for k in ("a", "b", "comparator") if k in h] + h.get("has", [])
            missing = [c for c in needed if not outcomes.get(c)]
            if missing:
                results[h["id"]] = {"not_tested": f"no outcomes for {missing}", "spec": h}
                pvals[h["id"]] = 1.0
                continue
            results[h["id"]] = {**test_one(h, finds, outcomes, reps, seed), "spec": h}
            pvals[h["id"]] = results[h["id"]]["p_test"]
        adj = ps.holm(pvals, alpha)
        for hid, r in results.items():
            r.update(p_holm=adj[hid]["p_holm"], reject=adj[hid]["reject"]
                     and "not_tested" not in r)
        out[fam] = results
    return out


def slices(finds: list[Find], outcomes: dict[str, dict], attrs: list[str], cells: list[str],
           comparators: list[str], reps: int, seed: int, min_finds: int = 1) -> dict:
    """Rates with intervals by each attribute's values (primary scope, every find); values
    with fewer than `min_finds` finds are pooled as "(small)". No tests."""
    base = in_scope(finds)
    out = {}
    for attr in attrs:
        groups = defaultdict(list)
        for f in base:
            groups[str(f.attrs.get(attr, "(unknown)"))].append(f)
        small = [v for v, fs in groups.items() if len(fs) < min_finds]
        if small:
            groups["(small)"] = [f for v in small for f in groups.pop(v)]
        out[attr] = {v: {"finds": len(fs), **{cid: {c: rate(fs, outcomes[cid], c, reps, seed)
                                                     for c in cells}
                                               for cid in comparators if cid in outcomes}}
                     for v, fs in sorted(groups.items())}
    return out
