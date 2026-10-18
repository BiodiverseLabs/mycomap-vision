"""Atlas habitat arms on dataset release v1's val (docs/atlas-environment.md section 1),
against Atlas's leave-val-out hindcast (R_train). Atlas paper E21 (arms) and E22 (H).

usage:
  python run_atlas.py import --continent DIR --prior DIR --table OUT --release ID
  python run_atlas.py leak   --trained-ids TSV.gz --release-root R [--errata TSV]
  python run_atlas.py run    --components DIR --table DIR --out DIR --release-root R
                             --manifest M [--errata TSV] [--answer-key K]
                             [--analysis-config JSON] [--paper-attrs TSV]

import: the R_train continent + prior exports -> atlasprior's table (offline).
leak: R_train's trained ids against v1 val (must be 0) and train (positive control > 0).
run: reads a place run's components (run_v1.py score + components: its candidates and
served terms; read-only), adds each candidate's Atlas values at the record's place, and
runs the pre-declared arms with nested 5-fold CV by observer under Steve's band rule:
P2c (primary: one intersection-union test against photos only), P2c-u, P2a, P2b, P2d, P3
(secondary, Holm), and, EXPLORATORY and not declared, P2c on top of the place setting now
chosen (penalty-500-2026-10-13). Every arm's temperature is fitted on its own training
folds (its own calibration). Writes atlas.json; prints the summary.

Files in DIR hold per-record results; keep it private (never this repository).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_v1  # noqa: E402

CHOSEN = "penalty-500-2026-10-13"
SOURCE_ALIASES = {"inaturalist": "inat", "inat": "inat", "mo observations": "mo", "mo": "mo",
                  "mushroom observer": "mo"}


def cmd_import(args) -> None:
    from mycomap_vision import atlasprior as ap
    t0 = time.time()
    s = ap.import_exports(Path(args.continent), Path(args.prior), Path(args.table), args.release)
    print(json.dumps(s, indent=1), f"[{time.time() - t0:.0f}s]")


def source_key(source: str, source_id: str) -> tuple[str, str]:
    s = (source or "").strip().lower()
    return SOURCE_ALIASES.get(s, s), str(source_id).strip()


def leak_counts(trained: set[tuple[str, str]], rel) -> dict:
    """How many v1 val and train records Atlas trained on (by source and id)."""
    out = {}
    for split in ("val", "train"):
        keys = {source_key(r["source"], r["source_id"]) for r in rel.records(split)}
        out[split] = {"records": len(keys), "in_trained_ids": len(keys & trained)}
    return out


def cmd_leak(args) -> None:
    import csv
    import gzip
    rel = run_v1.release_of(args)
    with gzip.open(args.trained_ids, "rt", encoding="utf-8", newline="") as f:
        trained = {source_key(r["source"], r["source_id"])
                   for r in csv.DictReader(f, delimiter="\t")}
    counts = leak_counts(trained, rel)
    counts["trained_ids"] = len(trained)
    counts["ok"] = counts["val"]["in_trained_ids"] == 0 and counts["train"]["in_trained_ids"] > 0
    print(json.dumps(counts, indent=1))
    if not counts["ok"]:
        raise SystemExit("leak check FAILED: R_train must hold no val record and some train "
                         "records (positive control)")


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if not n:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return round((c - h) / d, 4), round((c + h) / d, 4)


def auc(pos, neg) -> float | None:
    """P(a right top answer's suitability > a wrong one's), ties half."""
    import numpy as np
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                         # average ranks over ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return round(float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                       / (len(pos) * len(neg))), 4)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m, run, out = len(items), 0.0, {}
    for i, (k, p) in enumerate(items):
        run = max(run, min(1.0, (m - i) * p))
        out[k] = round(run, 6)
    return out


def cmd_run(args) -> None:
    sys.modules["torch"] = None
    import numpy as np
    from mycomap_vision import atlasprior as ap
    from mycomap_vision import heldout, name_equiv
    from mycomap_vision import placeprior as pp
    from mycomap_vision import placetune as pt
    from mycomap_vision.heldout_report import mcnemar
    from mycomap_vision.prior import Context
    t0 = time.time()
    comp, out = Path(args.components), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    taxonomy, taxonomy_cite = None, None
    if args.analysis_config:
        from mycomap_vision import paper_score
        doc = json.loads(Path(args.analysis_config).read_text(encoding="utf-8"))
        m0 = np.load(comp / "val-meta.npz")
        taxonomy, taxonomy_cite = paper_score.answer_key_taxonomy(
            doc.get("answer_key_taxonomy"), {t for t in m0["truth_name"].tolist() if t})
    c, meta = run_v1.load_components(comp, args.answer_key, taxonomy)
    meta["depth"] = run_v1.key_depth_bands(comp, meta)
    z = np.load(comp / "val-scores.npz")
    assert z["record_key"].tolist() == meta["record_key"].tolist(), "scores and meta differ"
    lat_all, lng_all = z["lat"], z["lng"]
    keep = ~meta["guest"]
    # Leak rule: no val record Atlas trained on is scored. Those found are left out and
    # listed (their first validation dates disagree between v1 and Atlas's pull).
    leaked = []
    if args.trained_ids:
        import csv
        import gzip
        with gzip.open(args.trained_ids, "rt", encoding="utf-8", newline="") as f:
            trained = {source_key(r["source"], r["source_id"])
                       for r in csv.DictReader(f, delimiter="\t")}
        rel0 = run_v1.release_of(args)
        by_key = {r["record_key"]: source_key(r["source"], r["source_id"])
                  for r in rel0.records("val")}
        leaked = [k for k in meta["record_key"].tolist() if by_key.get(k) in trained]
        keep &= ~np.isin(meta["record_key"], leaked)
        print(f"leak rule: {len(leaked)} val records Atlas trained on are left out: {leaked}",
              flush=True)
    c = run_v1.subset(c, keep)
    meta = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
            for k, v in meta.items()}
    lat, lng = lat_all[keep], lng_all[keep]
    n, K = c.S.shape
    folds = pt.observer_folds(meta["observer"].tolist())
    observers = meta["observer"].tolist()
    km = meta["truth_km"]
    if args.paper_attrs:
        import csv
        with open(args.paper_attrs, encoding="utf-8", newline="") as f:
            band_of = {r["record_key"]: r["distance_band"]
                       for r in csv.DictReader(f, delimiter="\t")}
        paper = np.array([band_of[k] for k in meta["record_key"].tolist()], dtype=str)
        rule = (paper == pt.PAPER_RULE_BAND) & c.has_sp
    else:
        paper = None
        rule = pt.band_of(km, *pt.RULE_BAND) & c.has_sp

    # Atlas values for every record's candidates, from the R_train table.
    table = ap.AtlasTable(Path(args.table))
    group_col = table.columns_for(meta["group_names"].tolist())
    arr = {"mapped": np.zeros((n, K), bool), "rank": np.full((n, K), ap.NO_DATA, np.uint8),
           "aoa": np.full((n, K), ap.NO_DATA, np.uint8),
           "models": np.full((n, K), ap.NO_DATA, np.uint8),
           "rank_sd": np.full((n, K), ap.NO_DATA, np.uint8), "reach": np.zeros((n, K), bool),
           "strength": np.zeros((n, K))}
    for i in range(n):
        ctx = Context(None if np.isnan(lat[i]) else float(lat[i]),
                      None if np.isnan(lng[i]) else float(lng[i]))
        t = pt.atlas_terms(table, group_col, ctx, c.cand[i])
        for k_, v in t.items():
            arr[k_][i] = v
    a = pt.AtlasComponents(**arr)
    print(f"{n:,} records; Atlas values built [{time.time() - t0:.0f}s]", flush=True)

    # Coverage (R_train's own strong set): truths with a strong map.
    truth_col = np.where(c.truth_at >= 0,
                         group_col[np.maximum(c.cand[np.arange(n), np.maximum(c.truth_at, 0)],
                                              0)], -1)
    mapped_truth = c.has_sp & (c.truth_at >= 0) & (truth_col >= 0)
    rel = run_v1.release_of(args)
    train_labels = [r["label"] for r in rel.records("train")]
    tcols = table.columns_for(sorted(set(train_labels)))
    tmapped = dict(zip(sorted(set(train_labels)), (tcols >= 0).tolist()))
    coverage = {"atlas_release": table.meta.get("release"), "strong_taxa": len(table.keys),
                "val_records": int(c.has_sp.sum()), "val_truth_mapped": int(mapped_truth.sum()),
                "val_share": round(float(mapped_truth.sum() / max(c.has_sp.sum(), 1)), 4),
                "train_records": len(train_labels),
                "train_truth_mapped": int(sum(tmapped[x] for x in train_labels)),
                "val_mapped_in_rule_band": int((mapped_truth & rule).sum())}
    coverage["train_share"] = round(coverage["train_truth_mapped"] / max(len(train_labels), 1), 4)
    print("coverage:", json.dumps(coverage), flush=True)

    # References: photos only, the boost served when the arms were declared, the setting
    # chosen since.
    p0_sp, p0_ge = pt.top1(c, pt.combine(c, pt.NONE))
    p1_sp, p1_ge = pt.top1(c, pt.combine(c, pt.SERVED))
    chosen = {**pt.SERVED, **pp.PLACE_SETTINGS[CHOSEN]}
    pc_sp, pc_ge = pt.top1(c, pt.combine(c, chosen))
    sp_rows = np.flatnonzero(c.has_sp)

    def summary(sp_, ge_):
        o = {"species_top1": round(float(sp_[c.has_sp].mean()), 4),
             "genus_top1": round(float(ge_.mean()), 4),
             "vs_photos_only": pt.fixed_broken(p0_sp, sp_, sp_rows),
             "vs_served_boost": pt.fixed_broken(p1_sp, sp_, sp_rows),
             "vs_" + CHOSEN: pt.fixed_broken(pc_sp, sp_, sp_rows),
             "place_only_effect_pts": round(100 * float(sp_[c.has_sp].mean()
                                                        - p0_sp[c.has_sp].mean()), 2),
             "mapped_truths": pt.fixed_broken(p0_sp, sp_, np.flatnonzero(mapped_truth)),
             "unmapped_truths": pt.fixed_broken(
                 p0_sp, sp_, np.flatnonzero(c.has_sp & ~mapped_truth)),
             "rule_band_vs_photos_only": pt.band_lower_bound(p0_sp, sp_, rule, folds),
             "rule_band_vs_photos_only_cluster": pt.band_lower_bound_clustered(
                 p0_sp, sp_, rule, observers),
             "rule_band_vs_" + CHOSEN: pt.band_lower_bound_clustered(pc_sp, sp_, rule,
                                                                     observers)}
        if paper is not None:
            o["paper_bands_vs_photos_only"] = pt.label_table(p0_sp, sp_, paper, c.has_sp,
                                                             pt.PAPER_BANDS)
        # Fix 4: mapped against unmapped truths at matched reference depth.
        o["by_depth_mapped_vs_unmapped"] = {
            d: {"mapped": pt.fixed_broken(p0_sp, sp_, np.flatnonzero(
                    mapped_truth & (meta["depth"] == d))),
                "unmapped": pt.fixed_broken(p0_sp, sp_, np.flatnonzero(
                    c.has_sp & ~mapped_truth & (meta["depth"] == d)))}
            for d in ("20-99", "100+")}
        return o

    grid = pt.atlas_settings()
    arm_of = np.array([s["arm"] for s in grid])
    runs = {}

    def cv(name, idx, band_rule=True, **kw):
        g = [grid[j] for j in idx]
        res = pt.nested_cv_atlas(c, a, folds, g, band=rule if band_rule else None,
                                 base=p0_sp, base_ge=p0_ge, **kw)
        runs[name] = res
        o = summary(res["species_right"], res["genus_right"])
        o["fold_settings"] = res["settings"]
        o["fold_log_f"] = [None if x is None else np.round(x, 4).tolist() for x in res["log_f"]]
        print(f"  {name}: species {o['species_top1']:.4f} genus {o['genus_top1']:.4f}; "
              f"vs P0 +{o['vs_photos_only']['fixed']}/-{o['vs_photos_only']['broken']}; "
              f"band vs P0 {o['rule_band_vs_photos_only']['delta']:+.4f} "
              f"(lb obs {o['rule_band_vs_photos_only_cluster']['lower95']:+.4f}) "
              f"[{time.time() - t0:.0f}s]", flush=True)
        return o

    arms = {"P2c (primary)": cv("P2c (primary)", np.flatnonzero(arm_of == "P2c"))}
    p2c_u = [None if s is None else pt.p2c_u(s) for s in runs["P2c (primary)"]["settings"]]
    arms["P2c-u"] = cv("P2c-u", [], fixed=p2c_u)
    for arm in ("P2a", "P2b", "P2d", "P3"):
        arms[arm] = cv(arm, np.flatnonzero(arm_of == arm))
    arms[f"P2c on {CHOSEN} (exploratory)"] = cv(
        f"P2c on {CHOSEN} (exploratory)", np.flatnonzero(arm_of == "P2c"), on=chosen)
    # EXPLORATORY, not declared: the arms without the band rule. The declared rule judges
    # against photos only, and the DNA boost the arms sit on already loses that band, so
    # under it they fall back to photos only in most folds and the comparison that matters
    # (the Atlas term against the DNA boost it replaces, the doc's key secondary) is lost.
    for arm in ("P2c", "P2a", "P2d", "P3"):
        arms[f"{arm} without the band rule (exploratory)"] = cv(
            f"{arm} without the band rule (exploratory)", np.flatnonzero(arm_of == arm),
            band_rule=False)

    # The primary hypothesis: one intersection-union test (fix 1-2).
    prim = runs["P2c (primary)"]
    primary = pt.primary_test(p0_sp, prim["species_right"], c.has_sp, rule, folds)
    primary["band_300_1500_observer_cluster"] = pt.band_lower_bound_clustered(
        p0_sp, prim["species_right"], rule, observers)
    # Secondary arms, Holm-adjusted one-sided McNemar against photos only.
    pvals = {}
    for name, o in arms.items():
        if name == "P2c (primary)":
            continue
        fb = o["vs_photos_only"]
        p = mcnemar(fb["broken"], fb["fixed"])
        pvals[name] = p / 2 if fb["fixed"] > fb["broken"] else 1 - p / 2
    secondary_holm = holm(pvals)

    # Calibration on each method's own temperatures (fitted per fold on training rows).
    def scores_for(name):
        res = runs[name]
        zz = np.zeros_like(c.S)
        T = np.zeros(n)
        for f in range(5):
            rows = folds == f
            s = res["settings"][f]
            zf = (pt.combine(c, pt.NONE) if s is None else
                  pt.combine_atlas(c, a, s, res["log_f"][f],
                                   base=chosen if f"on {CHOSEN}" in name else None))
            zz[rows] = zf[rows]
            T[rows] = pt.fit_temperature(c, zf, np.flatnonzero(~rows))
        return zz, T

    def fixed_scores(setting):
        zf = pt.combine(c, setting)
        T = np.zeros(n)
        for f in range(5):
            rows = folds == f
            T[rows] = pt.fit_temperature(c, zf, np.flatnonzero(~rows))
        return zf, T

    conn = run_v1.ro(args.manifest)
    labeller = heldout.Labeller(conn, extra=meta["truth_name"].tolist()
                                + meta["group_names"].tolist())
    keys = meta["record_key"].tolist()
    truths = {k: labeller.truth(t) for k, t in zip(keys, meta["truth_name"].tolist()) if t}
    names, gg, gf = (meta["group_names"].tolist(), meta["group_genus"].tolist(),
                     meta["group_family"].tolist())
    from mycomap_vision import paper_outcomes as po
    key = po.AnswerKey(args.answer_key or name_equiv.ANSWER_KEY, taxonomy)
    key_truths = {k: (truths[k].species, truths[k].genus, meta["depth"][i])
                  for i, k in enumerate(keys) if k in truths}
    tables = {}
    for name, (zz, T) in {"P0 photos only": fixed_scores(pt.NONE),
                          "boost-2026-10-11 (served at declaration)": fixed_scores(pt.SERVED),
                          CHOSEN: fixed_scores(chosen),
                          "P2c (primary)": scores_for("P2c (primary)"),
                          "P2c without the band rule (exploratory)":
                              scores_for("P2c without the band rule (exploratory)"),
                          f"P2c on {CHOSEN} (exploratory)":
                              scores_for(f"P2c on {CHOSEN} (exploratory)")}.items():
        res = {k: pt.ranks(zz[i], c.cand[i], c.SP[i], names, gg, gf, float(T[i]))
               for i, k in enumerate(keys)}
        keyed = pt.key_ladder(res, key_truths, key)
        conf = pt.top_confidence(c, zz, T)
        sp_right, _ = pt.top1(c, zz)
        tables[name] = {"ladder": {k_: keyed[k_] for k_ in ("records", "stored_depth", "rows")},
                        "by_depth": keyed["by_depth"],
                        "calibration_species": pt.calibration_report(conf, sp_right, c.has_sp),
                        "temperature_per_fold": sorted(set(np.round(T, 4).tolist()))}
        print(run_v1.fmt_ladder(name, tables[name]["ladder"], tables[name]["by_depth"]),
              flush=True)

    # H (fix 5): the projection where it extrapolates: mapped truths 500+ km from their
    # train DNA records. Truth's rank at the record's cell against its rank everywhere.
    far = mapped_truth & np.isfinite(km) & (km >= 500)
    t_rank = a.rank[np.arange(n), np.maximum(c.truth_at, 0)]
    t_aoa = a.aoa[np.arange(n), np.maximum(c.truth_at, 0)]
    has_rank = far & (t_rank != ap.NO_DATA)
    inside = has_rank & (t_aoa <= 100)
    outside = has_rank & (t_aoa > 100)

    def projection(sel):
        ranks_ = t_rank[sel].astype(int)
        cols = truth_col[sel]
        bg = np.concatenate([np.asarray(table.rank[:, j])[np.asarray(table.rank[:, j])
                                                          != ap.NO_DATA]
                             for j in sorted(set(cols.tolist()))]) if len(cols) else np.zeros(0)
        k50 = int((ranks_ >= 50).sum())
        return {"n": int(sel.sum()), "boyce": None if len(ranks_) < 5 else
                round(ap.boyce_index(ranks_, bg), 4),
                "share_rank_50_plus": None if not len(ranks_) else round(k50 / len(ranks_), 4),
                "share_ci95": wilson(k50, len(ranks_)),
                "valid": bool(len(ranks_) and wilson(k50, len(ranks_))[0] > 0.5)}
    # Cue value: the served answer's own suitability at the place, right vs wrong.
    z1 = pt.combine(c, chosen)
    best = np.argmax(np.where(c.SP, z1, -np.inf), axis=1)
    b_rank = a.rank[np.arange(n), best]
    ok = c.has_sp & a.mapped[np.arange(n), best] & (b_rank != ap.NO_DATA)
    terciles = {}
    if ok.any():
        edges = np.quantile(b_rank[ok], [1 / 3, 2 / 3])
        tier = np.digitize(b_rank, edges)
        for t_, lab in enumerate(("low", "middle", "high")):
            sel = ok & (tier == t_)
            terciles[lab] = {"n": int(sel.sum()),
                             "right_share": None if not sel.any() else
                             round(float(pc_sp[sel].mean()), 4)}
    h = {"far_mapped_truths": int(far.sum()), "with_projected_rank": int(has_rank.sum()),
         "inside_aoa": projection(inside), "outside_aoa": projection(outside),
         "cue_auc_right_vs_wrong": auc(b_rank[ok & pc_sp], b_rank[ok & ~pc_sp]),
         "cue_n_answers": int(ok.sum()), "right_share_by_suitability_tercile": terciles}

    report = {"atlas_release": table.meta.get("release"), "records": n,
              "leaked_val_records_left_out": leaked,
              "declared": pt.ATLAS_DECLARED, "coverage": coverage,
              "references": {"P0 photos only": summary(p0_sp, p0_ge),
                             "boost-2026-10-11 (served at declaration)": summary(p1_sp, p1_ge),
                             CHOSEN: summary(pc_sp, pc_ge)},
              "arms": arms, "primary_IUT": primary, "secondary_holm_p": secondary_holm,
              "H_projection": h, "tables": tables,
              "answer_key": args.answer_key or name_equiv.ANSWER_KEY,
              "answer_key_taxonomy": taxonomy_cite, "code_commit": run_v1.code_commit(),
              "components": str(comp)}
    path = out / "atlas.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print("PRIMARY:", json.dumps(primary, default=str))
    print("H:", json.dumps(h, default=str))
    print(f"-> {path} [{time.time() - t0:.0f}s]")


def main(argv=None) -> None:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("step", choices=("import", "leak", "run"))
    ap_.add_argument("--continent")
    ap_.add_argument("--prior")
    ap_.add_argument("--table")
    ap_.add_argument("--release")
    ap_.add_argument("--trained-ids", help="R_train's trained ids (leak: val records in it are "
                     "left out of run, and must be none for leak)")
    ap_.add_argument("--components")
    ap_.add_argument("--out")
    ap_.add_argument("--release-root")
    ap_.add_argument("--manifest")
    ap_.add_argument("--errata")
    ap_.add_argument("--answer-key")
    ap_.add_argument("--analysis-config")
    ap_.add_argument("--paper-attrs")
    args = ap_.parse_args(argv)
    {"import": cmd_import, "leak": cmd_leak, "run": cmd_run}[args.step](args)


if __name__ == "__main__":
    main()
