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
- `per_species`: PLAN rule 5's per-species rows beside the per-find rate: macro accuracy
  (each true species weighted equally, interval by resampling species), the share of
  species with at least one find right, the number of species. Secondary: no test uses them.
- `recoverability`: PLAN C4's species recoverability, the expected share of species one
  find would name right: each species' reference band times that band's per-find accuracy,
  averaged over species, over every species name in the release (leading; no reference =
  band 0) and over the species in the reference; with every sparse species lifted to 20+
  references beside it. Model-based (accuracy within a band taken as the same for every
  species); bootstrap over finds. Descriptive: no test uses it.

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


PER_SPECIES_METRICS = ("macro_species_top1", "share_species_ge1_right", "n_species")


def per_species(finds: list[Find], outcome: dict, cell: str, reps: int, seed: int,
                species_key=None) -> dict:
    """PLAN rule 5's per-species rows for one cell (species strict top-1 in the config):
    the finds with a true species grouped by `species_key(species)` (the answer key, so a
    species pools with its varieties, as reference depth does); per species the share of
    its finds right; `macro_species_top1` their mean (interval: species resampled),
    `share_species_ge1_right` the share of species with at least one find right,
    `n_species` the species. A find with no outcome counts as wrong, as in `rate`."""
    key = species_key or (lambda name: name)
    right: dict[str, list[bool]] = defaultdict(list)
    for f in finds:
        if not f.species:
            continue
        o = outcome.get(f.key)
        v = o.get(cell) if o is not None else False
        if v is None:
            continue
        right[key(f.species)].append(bool(v))
    names = sorted(right)
    acc = np.array([np.mean(right[n]) for n in names], dtype=np.float64)
    n = len(names)
    return {"cell": cell, "n_species": n,
            "finds": int(sum(len(v) for v in right.values())),
            "macro_species_top1": float(acc.mean()) if n else None,
            "macro_ci95": ps.cluster_ci(acc, names, reps, seed) if n else None,
            "species_ge1_right": int((acc > 0).sum()),
            "share_species_ge1_right": float((acc > 0).mean()) if n else None}


RECOVERABILITY_KS = (1, 3, 10)
SPARSE_BANDS = ("0", "1-4", "5-19")    # lifted to 20-99 in the counterfactual
LIFTED_BAND = "20-99"
# PLAN C4's two denominators, (b) leading
KNOWN, SEEN = "species MycoMap knows", "species Vision has seen"


def recoverability(finds: list[Find], outcome: dict, denominators: dict[str, list],
                   reps: int, seed: int, ks=RECOVERABILITY_KS) -> dict:
    """PLAN C4. `denominators`: {name: units}, each unit a species (answer-key unit) as
    (reference records, formal name?): (b) every species name in the release, a name with no
    reference at 0; (a) the species in the scored reference.

    Per band, the per-find accuracy at species strict top-k over the scored finds whose true
    species falls in that band (a find with no outcome counts as wrong). Band 0 is 0 by
    construction: a species with no reference record sharing its answer key can never be
    named (PLAN rule 7). A species' estimate is its band's accuracy; per denominator the
    rows average over all species, formal names only, and all species with every 0 / 1-4 /
    5-19 species given the 20-99 band's accuracy (the counterfactual: sequencing fills the
    sparse end). Intervals: percentile, scored finds resampled with replacement (the band
    accuracies move, the species stay). A species in a band with no scored finds is left out
    and counted."""
    from . import paper_stats as stats
    nb, zero = len(BANDS), BANDS.index("0")
    fs = [f for f in finds if f.species]
    fband = np.array([BANDS.index(band(f.depth)) for f in fs], dtype=np.int64)
    hits = {k: np.array([bool((outcome.get(f.key) or {}).get(f"species_strict@{k}"))
                         for f in fs], dtype=np.float64) for k in ks}

    def band_acc(h, fb):
        """(..., bands) per-band accuracy from hits and bands (NaN: no finds); band 0 = 0."""
        sums = np.stack([(h * (fb == b)).sum(axis=-1) for b in range(nb)], axis=-1)
        cnts = np.stack([(fb == b).sum(axis=-1) for b in range(nb)], axis=-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            acc = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        acc[..., zero] = 0.0
        return acc, cnts

    acc, cnts = band_acc(np.stack([hits[k] for k in ks]), fband[None])     # (k, bands)
    rng = np.random.default_rng(seed)
    boot = {k: [] for k in ks}
    for at in range(0, reps if len(fs) else 0, stats.CHUNK):
        n = min(stats.CHUNK, reps - at)
        pick = rng.integers(0, len(fs), size=(n, len(fs)))
        for k in ks:
            boot[k].append(band_acc(hits[k][pick], fband[pick])[0])           # (n, bands)
    boot = {k: np.concatenate(v) for k, v in boot.items() if v}
    out = {"model": "a species' estimate is its reference band's per-find accuracy "
                    "(assumes accuracy within a band does not depend on the species)",
           "ci": "bootstrap over finds", "finds": len(fs),
           "band_accuracy": {BANDS[b]: {"finds": int(cnts[0][b]),
                                        **{f"top{k}": (None if np.isnan(acc[i][b])
                                                       else float(acc[i][b]))
                                           for i, k in enumerate(ks)}}
                             for b in range(nb)},
           "denominators": {}}
    for dname, units in denominators.items():
        ubands = np.array([BANDS.index(band(d)) for d, _ in units], dtype=np.int64)
        formal = np.array([bool(fm) for _, fm in units], dtype=bool)
        lifted = ubands.copy()
        lifted[np.isin(ubands, [BANDS.index(b) for b in SPARSE_BANDS])] = BANDS.index(LIFTED_BAND)
        every = np.ones(len(units), dtype=bool)
        rows = {}
        for name, (ub, keep) in {"all names": (ubands, every), "formal names": (ubands, formal),
                                 "every species at 20+ references": (lifted, every)}.items():
            known = keep & ~np.isnan(acc[0][ub])
            row = {"species": int(known.sum()), "left_out": int((keep & ~known).sum())}
            # the mean over species = the band accuracies weighted by each band's species
            w = np.bincount(ub[known], minlength=nb).astype(np.float64)
            for i, k in enumerate(ks):
                est = float(np.nansum(acc[i] * w) / w.sum()) if w.sum() else None
                ci = None
                if k in boot and w.sum():
                    ok = ~np.isnan(boot[k])
                    per = (np.where(ok, boot[k], 0.0) @ w) / np.maximum(ok @ w, 1e-12)
                    lo, hi = np.percentile(per, [100 * (1 - stats.LEVEL) / 2,
                                                 100 * (1 + stats.LEVEL) / 2])
                    ci = (float(lo), float(hi))
                row[f"top{k}"] = {"estimate": est, "ci95": ci}
            rows[name] = row
        out["denominators"][dname] = {
            "species_by_band": {BANDS[b]: int((ubands == b).sum()) for b in range(nb)},
            "rows": rows}
    return out


def standard(finds: list[Find], outcome: dict, reps: int, seed: int, ci_cells=None,
             per_species_rows: dict | None = None, species_key=None) -> dict:
    """Steve's table for one comparator on these finds (see the module docstring).
    `ci_cells`: the cells that get an interval (default: every top-1 cell).
    `per_species_rows` (the config block): adds `per_species` (PLAN rule 5)."""
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
    out = {"finds": len(finds), "answered": answered, "table": table,
           "species_by_reference_band": by_band}
    if per_species_rows:
        out["per_species"] = per_species(finds, outcome,
                                         per_species_rows.get("cell", "species_strict@1"),
                                         reps, seed, species_key)
    return out


def all_standard(finds: list[Find], outcomes: dict[str, dict], reps: int, seed: int,
                 per_species_rows: dict | None = None, species_key=None,
                 finds_of: dict[str, list[Find]] | None = None) -> dict:
    """Every comparator's table, both denominators, both scopes; the near-duplicate count.
    `finds_of`: a comparator's own finds (the same finds, depth counted in its reference),
    when its reference is not the full one."""
    out = {"near_duplicates": near_duplicate_counts(finds), "comparators": {}}
    for cid, outcome in outcomes.items():
        fs = (finds_of or {}).get(cid, finds)
        out["comparators"][cid] = {
            scope: {den: standard(in_scope(fs, scope, den), outcome, reps, seed,
                                  per_species_rows=per_species_rows, species_key=species_key)
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
    # PLAN rule 7 and Steve's genus rule (18 Oct): the iNat- and expert-paired tests keep
    # only truths iNat can name, at species (truth_resolves: resolves one-to-one) and at
    # genus (truth_genus_known: the label's genus or the resolved species' genus is an iNat
    # genus at the snapshot); refused when the run has no taxonomy.
    for flag in ("truth_resolves", "truth_genus_known"):
        if spec.get(flag):
            if any(flag not in f.attrs for f in out):
                raise ValueError(f"{spec.get('id')}: {flag} needs the config's "
                                 "answer_key_taxonomy")
            out = [f for f in out if f.attrs[flag] == "yes"]
    return out


TAXONOMY_FILTERS = ("truth_resolves", "truth_genus_known")


HYPOTHESIS_KINDS = ("mcnemar", "gee_depth", "iut", "floor")


def floor_test(finds: list[Find], outcome: dict, cell: str, floor: float, alternative: str,
               reps: int, seed: int) -> dict:
    """One comparator's own rate at `cell` against a fixed floor (Steve's A1s/A1g, 28 Oct):
    the finds the cell applies to (a find with no outcome line counts as wrong, as in
    `_pairs`), the rate with its observer-clustered interval, and a one-sided Wald test of
    the mean with an observer-clustered (CR1) standard error,
    SE = sqrt(G/(G-1) * sum_g (sum_{i in g} (y_i - p))^2) / n, z = (p - floor) / SE, the
    robust-Wald family of gee_depth's slope test. With fewer than two observers, or every
    answer the same (SE 0), there is no test: z None and p 1."""
    y, who, kinds = [], [], defaultdict(int)
    for f in finds:
        o = outcome.get(f.key)
        v = o.get(cell) if o is not None else (False if _applies(f, cell) else None)
        if v is None:
            continue
        y.append(float(bool(v)))
        who.append(f.observer)
        kinds[f.name_kind] += 1
    n = len(y)
    out = {"cell": cell, "n": n, "right": int(sum(y)), "floor": floor,
           "alternative": alternative, "rate": None, "difference": None, "ci95": None,
           "clusters": len(set(who)), "se_robust": None, "z": None, "p": 1.0,
           "n_by_name_kind": dict(sorted(kinds.items()))}
    if not n:
        return out
    yv = np.asarray(y)
    p_hat = float(yv.mean())
    out.update(rate=p_hat, difference=p_hat - floor, ci95=ps.cluster_ci(y, who, reps, seed))
    sums: dict = defaultdict(float)
    for v, g in zip(yv - p_hat, who):
        sums[g] += v
    G = len(sums)
    if G < 2:
        return out
    se = float(np.sqrt(G / (G - 1) * sum(s * s for s in sums.values())) / n)
    if se == 0:
        return out
    z = (p_hat - floor) / se
    out.update(se_robust=se, z=z,
               p=ps.normal_sf(z) if alternative == "greater" else ps.normal_sf(-z))
    return out


def test_one(spec: dict, finds: list[Find], outcomes: dict[str, dict], reps: int,
             seed: int) -> dict:
    """One pre-declared hypothesis: kind mcnemar (a vs b on a cell), gee_depth (H3), iut
    (superiority overall and non-inferiority in a band, one intersection-union test), or
    floor (one comparator's rate against a fixed floor)."""
    kind = spec["kind"]
    if kind == "floor":
        r = floor_test(select(finds, spec, outcomes), outcomes[spec["comparator"]],
                       spec["cell"], spec["floor"], spec.get("alternative", "greater"),
                       reps, seed)
        return {**r, "p_test": r["p"]}
    if kind == "mcnemar":
        sub = select(finds, spec, outcomes)
        r = paired(sub, outcomes[spec["a"]], outcomes[spec["b"]], spec["cell"],
                   spec.get("alternative", "greater"), reps, seed)
        return {**r, "p_test": r["p"],
                "n_by_name_kind": paired_name_kinds(sub, outcomes[spec["a"]],
                                                    outcomes[spec["b"]], spec["cell"])}
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


def paired_name_kinds(finds, a, b, cell) -> dict:
    """The paired finds' n by the truth's name kind (formal / provisional / one-word): the
    finds _pairs keeps (rule 4: a provisional truth that resolves enters the iNat-paired
    tests, so their n is reported split)."""
    out: dict = defaultdict(int)
    for f in finds:
        if len(_pairs([f], a, b, cell)[0]):
            out[f.name_kind] += 1
    return dict(sorted(out.items()))


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


OUTSIDE_HOLM = ("side",)        # disclosed side comparisons, e.g. internal - public (O1)


def finds_for(spec: dict, finds: list[Find], finds_of: dict | None) -> list[Find]:
    """The finds a hypothesis is tested on: those of its comparator (gee_depth) or of its
    "a" side (paired), so depth and reference attributes are counted in that comparator's
    reference; the pairing is by find key either way."""
    cid = spec.get("comparator") or spec.get("a")
    return (finds_of or {}).get(cid, finds)


def families(spec: dict, finds: list[Find], outcomes: dict[str, dict], reps: int,
             seed: int, alpha: float = 0.05, finds_of: dict | None = None) -> dict:
    """Every family's hypotheses tested and Holm-adjusted within the family. A hypothesis
    whose comparator has no outcomes (its input not given) is reported, not tested, and
    leaves the family's Holm denominator unchanged (its p counts as 1). A family in
    OUTSIDE_HOLM ("side") is tested and reported unadjusted, with no reject decision."""
    out = {}
    for fam, hyps in spec.items():
        results, pvals = {}, {}
        for h in hyps:
            fs = finds_for(h, finds, finds_of)
            needed = [h[k] for k in ("a", "b", "comparator") if k in h] + h.get("has", [])
            missing = [c for c in needed if not outcomes.get(c)]
            if missing:
                results[h["id"]] = {"not_tested": f"no outcomes for {missing}", "spec": h}
                pvals[h["id"]] = 1.0
                continue
            results[h["id"]] = {**test_one(h, fs, outcomes, reps, seed), "spec": h}
            pvals[h["id"]] = results[h["id"]]["p_test"]
            if any(h.get(f) for f in TAXONOMY_FILTERS):
                # sensitivity, not tested in the family: the same test with the truths
                # iNat cannot name kept in (they count as the iNat side's misses)
                kept = {k: v for k, v in h.items() if k not in TAXONOMY_FILTERS}
                results[h["id"]]["kept_in"] = test_one(kept, fs, outcomes, reps, seed)
        if fam in OUTSIDE_HOLM:
            for r in results.values():
                r.update(p_holm=None, reject=None, outside_holm=True)
            out[fam] = results
            continue
        adj = ps.holm(pvals, alpha)
        for hid, r in results.items():
            r.update(p_holm=adj[hid]["p_holm"], reject=adj[hid]["reject"]
                     and "not_tested" not in r)
        out[fam] = results
    return out


# --- the test scope (Steve 28 Oct): finds outside it, described, never tested --------------------

def outside_test_scope(finds: list[Find], outcomes: dict[str, dict], comparators: list[str],
                       cells: list[str], reps: int, seed: int) -> dict:
    """The finds left out by the config's test_scope: their number by country (no country:
    "unknown", on its own line) and, for each comparator named, the rate at each cell with
    its observer-clustered interval over their primary scope: the estimate a floor reports,
    with no test. No finds: n 0."""
    by_country: dict = defaultdict(int)
    for f in finds:
        by_country[f.attrs.get("country") or "unknown"] += 1
    unknown = by_country.pop("unknown", 0)
    primary = in_scope(finds, "primary", "all")
    return {"finds": len(finds), "primary": len(primary),
            "by_country": dict(sorted(by_country.items())), "unknown": unknown,
            "comparators": {cid: {cell: rate(primary, outcomes.get(cid) or {}, cell, reps, seed)
                                  for cell in cells} for cid in comparators}}


# --- secondary rows (E14, Steve 27 Oct): descriptive, outside Holm -----------------------------

WHERE_KEYS = ("name_kind", "created_after", "rg_at_inat_export", "inat_observation",
              "truth_resolves", "truth_genus_known", "any_of")
# Attributes compared as given ("yes" / "no"); truth_resolves and truth_genus_known come from
# the answer-key taxonomy (unknown, so the find is counted, when the run has none).
WHERE_ATTRS = ("rg_at_inat_export", "inat_observation", "truth_resolves", "truth_genus_known")
UNKNOWN = ("", None, "unknown")


def where_holds(f: Find, where: dict) -> bool | None:
    """Does find `f` meet a row's `where` (every condition; `any_of`: at least one of its
    conditions)? None when an attribute it needs is unknown and nothing else decides it.
    `created_after` is a resolved ISO date: the find's observation was created on a later
    day (created_at's date part)."""
    verdicts = []
    for k, v in where.items():
        if k == "any_of":
            parts = [where_holds(f, w) for w in v]
            verdicts.append(True if True in parts else None if None in parts else False)
        elif k == "name_kind":
            verdicts.append(f.name_kind == v)
        elif k == "created_after":
            c = f.attrs.get("created_at")
            verdicts.append(None if c in UNKNOWN else str(c)[:10] > str(v)[:10])
        elif k in WHERE_ATTRS:
            a = f.attrs.get(k)
            verdicts.append(None if a in UNKNOWN else a == v)
        else:
            raise ValueError(f"secondary row: unknown condition {k!r}")
    if False in verdicts:
        return False
    return None if None in verdicts else True


def secondary_rows(block: dict, hypotheses: dict, finds: list[Find], outcomes: dict[str, dict],
                   reps: int, seed: int, finds_of: dict | None = None) -> dict:
    """Each row's finds (`where`, resolved), and each hypothesis of block["for"] tested on
    them: n, estimate, interval and p, unadjusted, with no reject decision. A find whose
    attribute is unknown leaves the row and is counted; a row whose where still names a TBD
    config value is reported "not run: TBD"."""
    by_id = {h["id"]: h for hs in hypotheses.values() for h in hs}
    scored = in_scope(finds, "primary", "all")
    out = {}
    for row in block.get("rows") or []:
        if row.get("not_run"):
            out[row["id"]] = {"not_run": row["not_run"]}
            continue
        held = {f.key: where_holds(f, row.get("where") or {}) for f in finds}
        kept = [f for f in scored if held[f.key]]
        res = {"where": row.get("where") or {}, "finds": len(kept),
               "share_of_scored": len(kept) / len(scored) if scored else None,
               "unknown": sum(1 for f in scored if held[f.key] is None), "hypotheses": {}}
        for hid in block.get("for") or []:
            h = by_id[hid]
            needed = [h[k] for k in ("a", "b", "comparator") if k in h] + h.get("has", [])
            missing = [c for c in needed if not outcomes.get(c)]
            if missing:
                res["hypotheses"][hid] = {"not_tested": f"no outcomes for {missing}"}
                continue
            fs = [f for f in finds_for(h, finds, finds_of) if held.get(f.key)]
            res["hypotheses"][hid] = {**test_one(h, fs, outcomes, reps, seed),
                                      "outside_holm": True, "p_holm": None, "reject": None}
        out[row["id"]] = res
    return out


def slices(finds: list[Find], outcomes: dict[str, dict], attrs: list[str], cells: list[str],
           comparators: list[str], reps: int, seed: int, min_finds: int = 1,
           finds_of: dict | None = None) -> dict:
    """Rates with intervals by each attribute's values (primary scope, every find); values
    with fewer than `min_finds` finds are pooled as "(small)". No tests. A comparator with
    its own finds (`finds_of`) is sliced by its own attribute values (e.g. observer new to
    its reference)."""
    def grouped(fs_all, attr):
        groups = defaultdict(list)
        for f in in_scope(fs_all):
            groups[str(f.attrs.get(attr, "(unknown)"))].append(f)
        small = [v for v, fs in groups.items() if len(fs) < min_finds]
        if small:
            groups["(small)"] = [f for v in small for f in groups.pop(v)]
        return groups
    out = {}
    for attr in attrs:
        base = grouped(finds, attr)
        own = {cid: grouped(finds_of[cid], attr) for cid in comparators
               if cid in outcomes and cid in (finds_of or {})}
        out[attr] = {v: {"finds": len(fs), **{cid: {c: rate(own[cid].get(v, []) if cid in own
                                                             else fs, outcomes[cid], c, reps,
                                                             seed)
                                                     for c in cells}
                                               for cid in comparators if cid in outcomes}}
                     for v, fs in sorted(base.items())}
    return out


def generalisation(finds: list[Find], outcomes: dict[str, dict], rows: dict[str, dict],
                   cells: list[str], comparators: list[str], reps: int, seed: int) -> dict:
    """The outside-review rows (PLAN E20, E12, P7/P15): per named row {attr: value}, each
    comparator's rate inside the row and outside it (primary scope), with intervals; and
    the near-duplicate sensitivity: every find (primary) vs near-duplicates kept, with the
    count. Descriptive only: intervals, no tests."""
    base = in_scope(finds)
    out = {"rows": {}, "near_duplicates": {}}
    for name, flt in rows.items():
        inside = [f for f in base if all(str(f.attrs.get(k)) == str(v) for k, v in flt.items())]
        ids = {id(f) for f in inside}
        outside = [f for f in base if id(f) not in ids]
        out["rows"][name] = {
            "filter": flt, "finds": len(inside), "others": len(outside),
            **{cid: {c: {"in": rate(inside, outcomes[cid], c, reps, seed),
                         "out": rate(outside, outcomes[cid], c, reps, seed)} for c in cells}
               for cid in comparators if cid in outcomes}}
    kept = in_scope(finds, "with_near_duplicates")
    out["near_duplicates"] = {
        **near_duplicate_counts(finds),
        **{cid: {c: {"primary": rate(base, outcomes[cid], c, reps, seed),
                     "with_near_duplicates": rate(kept, outcomes[cid], c, reps, seed)}
                 for c in cells}
           for cid in comparators if cid in outcomes}}
    return out
