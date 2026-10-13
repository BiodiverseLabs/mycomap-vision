"""Stronger place penalties on dataset release v1's val (placetune.py; Steve, 2026-10-13).

usage:
  python run_v1.py score      --release-root R --manifest M --backbone B --out DIR
                              [--validated-after YYYY-MM-DD] [--heldout NAME]
  python run_v1.py components --release-root R --manifest M --out DIR
  python run_v1.py tune       --release-root R --manifest M --out DIR [--named-case KEY ...]

score: reference = v1 `train` records with a vector for backbone B; queries = v1 `val`
records (validated after the date, for a model fine-tuned on records up to it) scored with
the served nearest + species average (methods.NearestAndMean itself). Query vectors come
from the manifest's embeddings, else the held-out benchmark's own store. Nothing is
embedded or fetched here.
components: every term of placetune's declared grid per record and candidate, made by
placeprior's own code (placetune.TermBuilder), from train records only, each record's own
iNat observation taken back out of the occurrence store.
tune: every setting, nested 5-fold CV by observer, the picks, the frontier, distance
bands, calibration and the standard tables. Writes tuning.json; prints the summary.

The files in DIR hold .org's true coordinates (inputs) and per-record results: keep DIR
private (never this repository). The manifest is opened read-only.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path


def ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{Path(path).as_posix()}?mode=ro", uri=True)


def release_of(args):
    """Release v1, with its erratum when --errata is given (Release.use_errata: its
    records leave every split; cite() names it and its sha256)."""
    from mycomap_vision.dataset_release import load_release
    rel = load_release("v1", Path(args.release_root))
    if args.errata:
        from mycomap_vision import errata
        rel.use_errata(errata.read(Path(args.errata)))
    return rel


# --- score --------------------------------------------------------------------------------

def cmd_score(args) -> None:
    import numpy as np
    from mycomap_vision import heldout
    from mycomap_vision.evaluate import build_index
    from mycomap_vision.identify import map_embeddings
    from mycomap_vision.methods import NearestAndMean
    t0 = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    conn = ro(args.manifest)
    rel = release_of(args)
    rel.check_photos_match(conn)
    ids, vecs = map_embeddings(conn, args.backbone, Path(args.embeddings) if args.embeddings
                               else None, None)
    row_of = {int(p): i for i, p in enumerate(ids.tolist())}
    train = rel.as_records("train", row_of)
    index = build_index(train)
    m = NearestAndMean()
    m.fit(vecs, index)
    print(f"reference: {len(train):,} train records, {len(index.cols):,} photos, "
          f"{len(index.species):,} groups [{time.time() - t0:.0f}s]", flush=True)
    held = {}
    if args.heldout:
        heldout.bench_dir = lambda c, name: Path(args.heldout_dir)
        held = heldout.load_vectors(conn, args.heldout, args.backbone, "large")
    val = rel.as_records("val", None)               # photo_rows = manifest photo ids
    labels = {r["record_key"]: (r["label"], r["validated_on"], r["observer_login"],
                                r["inat_uuid"]) for r in rel.records("val")}
    rows = {k: [] for k in ("key", "nm", "nphotos")}
    meta = []
    skipped = {}
    for i, r in enumerate(val):
        label, validated, observer, uuid = labels[r.observation_id]
        if args.validated_after and not (validated and validated > args.validated_after):
            skipped["validated on or before the model's training"] = \
                skipped.get("validated on or before the model's training", 0) + 1
            continue
        q = [np.asarray(vecs[[row_of[p]]], dtype=np.float32)[0] if p in row_of else held.get(p)
             for p in r.photo_rows]
        q = [v for v in q if v is not None]
        if not q:
            skipped["no vectors"] = skipped.get("no vectors", 0) + 1
            continue
        q = np.stack(q)
        q /= np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)
        rows["key"].append(r.observation_id)
        rows["nm"].append(m.species_scores(q).astype(np.float32))
        rows["nphotos"].append(len(q))
        meta.append((r, label, observer, uuid, validated))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,}/{len(val):,} [{time.time() - t0:.0f}s]", flush=True)
    S = lambda xs: np.array(["" if x is None else str(x) for x in xs], dtype=str)  # noqa: E731
    F = lambda xs: np.array([np.nan if x is None else float(x) for x in xs])       # noqa: E731
    g = lambda rank: S(index.labels[rank][j] if j >= 0 else ""                      # noqa: E731
                       for j in index.label_of[rank].tolist())
    np.savez_compressed(
        out / "val-scores.npz", record_key=S(rows["key"]), species=S(index.species),
        nearest_mean=np.stack(rows["nm"]), n_photos=np.array(rows["nphotos"]),
        truth_name=S(x[1] for x in meta), observer=S(x[2] for x in meta),
        uuid=S(x[3] for x in meta), validated_on=S(x[4] for x in meta),
        lat=F(x[0].latitude for x in meta), lng=F(x[0].longitude for x in meta),
        observed_on=S(x[0].observed_on for x in meta),
        group_genus=g("genus"), group_family=g("family"),
        group_is_species=(index.label_of["species"] >= 0),
        ref_count=np.array([index.ref_count[s] for s in index.species]))
    run = {"backbone": args.backbone, "release": rel.cite() if hasattr(rel, "cite") else "v1",
           "validated_after": args.validated_after, "scored": len(rows["key"]),
           "skipped": skipped, "reference_records": len(train),
           "reference_photos": int(len(index.cols)), "groups": len(index.species),
           "code_commit": code_commit()}
    (out / "run-score.json").write_text(json.dumps(run, indent=2, default=str), encoding="utf-8")
    print(json.dumps(run, indent=2, default=str), flush=True)


def code_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                              cwd=Path(__file__).parent, check=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --- components ---------------------------------------------------------------------------

def cmd_components(args) -> None:
    sys.modules["torch"] = None     # numpy only: torch + OpenBLAS crashed these runs before
    import numpy as np
    from mycomap_vision import heldout
    from mycomap_vision import placetune as pt
    from mycomap_vision.evaluate import build_index
    from mycomap_vision.heldout_report import DEPTH_BUCKETS, bucket
    from mycomap_vision.occurrence import OccurrenceStore
    from mycomap_vision.prior import Context, haversine_km
    t0 = time.time()
    out = Path(args.out)
    z = dict(np.load(out / "val-scores.npz"))
    conn = ro(args.manifest)
    rel = release_of(args)
    train = rel.as_records("train", None)
    species = z["species"].tolist()
    keep_units = set(species)
    train = [r for r in train if r.unit in keep_units]   # the scored reference's groups
    index = build_index(train)
    assert index.species == species, "group order differs from the scored reference"
    store = OccurrenceStore.load(Path(args.store) if args.store else None)
    tb = pt.TermBuilder(train, species, store)
    print(f"terms from {len(train):,} train records [{time.time() - t0:.0f}s]", flush=True)
    # Truth by the report's rule (Labeller: label, then name_key).
    lab = heldout.Labeller(conn, extra=z["truth_name"].tolist())
    key = lambda name: heldout.name_key(lab.label(name)) if name else ""   # noqa: E731
    is_sp = z["group_is_species"]
    sp_pos = {}
    for j, s in enumerate(species):
        if is_sp[j]:
            sp_pos.setdefault(key(s), j)
    genus_names = sorted({x for x in z["group_genus"].tolist() if x})
    gkey = {}
    for i, x in enumerate(genus_names):
        gkey.setdefault(key(x), i)
    group_gid = np.array([gkey.get(key(x), -1) if x else -1 for x in z["group_genus"].tolist()])
    truths = [lab.truth(t) if t else None for t in z["truth_name"].tolist()]
    truth_sp = np.array([sp_pos.get(key(t.species), -1) if t and t.species else -1
                         for t in truths])
    truth_g = np.array([gkey.get(key(t.genus), -3) if t and t.genus else -3 for t in truths])
    has_sp = np.array([bool(t and t.species) for t in truths])
    guest = np.array([bool(t and t.guest) for t in truths])
    # Bands: the true species' nearest TRAIN DNA record (atlas-environment.md: never val).
    by = {}
    for r in train:
        if r.latitude is not None and r.longitude is not None:
            by.setdefault(r.unit, []).append((r.latitude, r.longitude))
    by = {species.index(u): np.array(v) for u, v in by.items()}
    n, K = len(z["record_key"]), pt.TOP_K + 1
    S = z["nearest_mean"]
    arr = lambda dt, fill: np.full((n, K), fill, dtype=dt)                 # noqa: E731
    o = {"S": arr(np.float32, -np.inf), "cand": arr(np.int32, -1), "SP": arr(bool, False),
         "GG": arr(np.int32, -2), "truth_at": np.full(n, -1, np.int64),
         **{f"boost_{sh:g}": arr(np.float32, 0) for sh in pt.SHRINKS},
         **{f"season_{sh:g}": arr(np.float32, 0) for sh in pt.SHRINKS},
         "berth": arr(bool, False), "km_dna": arr(np.float32, np.inf),
         "km_inat": arr(np.float32, np.inf), "n_place": arr(np.float32, 0),
         "off_season": arr(bool, False)}
    truth_km = np.full(n, np.nan)
    has_place, has_date = np.zeros(n, bool), np.zeros(n, bool)
    lat, lng = z["lat"], z["lng"]
    for i in range(n):
        ctx = Context(None if np.isnan(lat[i]) else float(lat[i]),
                      None if np.isnan(lng[i]) else float(lng[i]),
                      z["observed_on"][i] or None, z["uuid"][i] or None)
        has_place[i] = ctx.latitude is not None and ctx.longitude is not None
        has_date[i] = ctx.day_of_year is not None
        t = tb.terms(ctx)
        c = pt.candidates(S[i], int(truth_sp[i]))
        m = len(c)
        o["S"][i, :m], o["cand"][i, :m] = S[i, c], c
        o["SP"][i, :m], o["GG"][i, :m] = is_sp[c], group_gid[c]
        if truth_sp[i] >= 0:
            o["truth_at"][i] = int(np.flatnonzero(c == truth_sp[i])[0])
            if has_place[i] and int(truth_sp[i]) in by:
                p = by[int(truth_sp[i])]
                truth_km[i] = haversine_km(ctx.latitude, ctx.longitude, p[:, 0], p[:, 1]).min()
        for name, vec in t.items():
            o[name][i, :m] = vec[c]
        if (i + 1) % 250 == 0:
            print(f"  {i + 1:,}/{n:,} [{time.time() - t0:.0f}s]", flush=True)
    np.savez_compressed(out / "val-components.npz", **o)
    np.savez_compressed(
        out / "val-meta.npz", record_key=z["record_key"], observer=z["observer"],
        truth_name=z["truth_name"], has_sp=has_sp, truth_g=truth_g, guest=guest,
        has_place=has_place, has_date=has_date, truth_km=truth_km,
        depth=np.array([bucket(int(z["ref_count"][j]), DEPTH_BUCKETS) if j >= 0 else
                        ("0" if t else "n/a") for j, t in zip(truth_sp.tolist(), has_sp)]),
        group_names=np.array(species), group_genus=z["group_genus"],
        group_family=z["group_family"], genus_names=np.array(genus_names))
    print(f"components: {n:,} records, {int(has_place.sum()):,} with a place, "
          f"{int(has_date.sum()):,} with a date [{time.time() - t0:.0f}s]", flush=True)


# --- tune ---------------------------------------------------------------------------------

def load_components(out: Path):
    import numpy as np
    from mycomap_vision import placetune as pt
    z = np.load(out / "val-components.npz")
    m = np.load(out / "val-meta.npz")
    c = pt.Components(
        S=z["S"].astype(np.float64), cand=z["cand"], SP=z["SP"], GG=z["GG"].astype(np.int64),
        truth_at=z["truth_at"], has_sp=m["has_sp"], truth_g=m["truth_g"].astype(np.int64),
        boost={sh: z[f"boost_{sh:g}"].astype(np.float64) for sh in pt.SHRINKS},
        season={sh: z[f"season_{sh:g}"].astype(np.float64) for sh in pt.SHRINKS},
        berth=z["berth"], km_dna=z["km_dna"].astype(np.float64),
        km_inat=z["km_inat"].astype(np.float64), n_place=z["n_place"].astype(np.float64),
        off_season=z["off_season"])
    return c, {k: m[k] for k in m.files}


def subset(c, keep):
    from mycomap_vision import placetune as pt

    def cut(v):
        if isinstance(v, dict):
            return {k: cut(x) for k, x in v.items()}
        return v[keep] if hasattr(v, "shape") and v.shape[:1] == keep.shape else v
    return pt.Components(**{f: cut(getattr(c, f)) for f in c.__dataclass_fields__})


def cmd_tune(args) -> None:
    sys.modules["torch"] = None
    import numpy as np
    from mycomap_vision import heldout, name_equiv
    from mycomap_vision import placetune as pt
    from mycomap_vision.heldout_report import mcnemar, score
    from mycomap_vision.heldout_summary import by_reference_depth, ladder
    t0 = time.time()
    out = Path(args.out)
    c, meta = load_components(out)
    keep = ~meta["guest"]
    c = subset(c, keep)
    meta = {k: (v[keep] if isinstance(v, np.ndarray) and v.shape[:1] == keep.shape else v)
            for k, v in meta.items()}
    n = len(c.S)
    grid = pt.settings()
    folds = pt.observer_folds(meta["observer"].tolist())
    sp, ge = pt.score_all(c, grid)
    p0_sp, p0_ge = pt.top1(c, pt.combine(c, pt.NONE))
    p1 = 0                                               # grid[0] is the served setting
    km = meta["truth_km"]                 # true .org points (the lane's own; extra detail)
    observers = meta["observer"].tolist()
    paper = None
    if args.paper_attrs:
        # The paper's distance_band (E8: public points, the release's own species), read
        # from `mv paper-attrs`, so a find gets the paper's band here too.
        import csv
        with open(args.paper_attrs, encoding="utf-8", newline="") as f:
            by_key = {r["record_key"]: r["distance_band"] for r in csv.DictReader(f, delimiter="\t")}
        keys = meta["record_key"].tolist()
        missing = [k for k in keys if k not in by_key]
        if missing:
            raise SystemExit(f"{len(missing)} scored records have no paper band "
                             f"(e.g. {missing[:3]}): wrong --paper-attrs file?")
        paper = np.array([by_key[k] for k in keys], dtype=str)
        rule = (paper == pt.PAPER_RULE_BAND) & c.has_sp
    else:
        rule = pt.band_of(km, *pt.RULE_BAND) & c.has_sp
    print(f"{n:,} records ({int(c.has_sp.sum()):,} with a species answer, "
          f"{int(rule.sum()):,} in the 300-1,500 km band, "
          f"{'paper bands' if paper is not None else 'true-point bands'}); "
          f"scored {len(grid)} settings [{time.time() - t0:.0f}s]", flush=True)

    def summary_of(cv_sp, cv_ge):
        out = {"species_top1": round(float(cv_sp[c.has_sp].mean()), 4),
               "genus_top1": round(float(cv_ge.mean()), 4),
               "vs_photos_only": pt.fixed_broken(p0_sp, cv_sp, np.flatnonzero(c.has_sp)),
               "vs_served": pt.fixed_broken(sp[p1], cv_sp, np.flatnonzero(c.has_sp)),
               "bands_vs_photos_only": pt.band_table(p0_sp, cv_sp, km, c.has_sp),
               "bands_vs_served": pt.band_table(sp[p1], cv_sp, km, c.has_sp),
               "rule_band_vs_photos_only": pt.band_lower_bound(p0_sp, cv_sp, rule, folds),
               "rule_band_vs_served": pt.band_lower_bound(sp[p1], cv_sp, rule, folds),
               "rule_band_vs_photos_only_cluster": pt.band_lower_bound_clustered(
                   p0_sp, cv_sp, rule, observers),
               "rule_band_vs_served_cluster": pt.band_lower_bound_clustered(
                   sp[p1], cv_sp, rule, observers)}
        if paper is not None:
            out["paper_bands_vs_photos_only"] = pt.label_table(p0_sp, cv_sp, paper, c.has_sp,
                                                               pt.PAPER_BANDS)
            out["paper_bands_vs_served"] = pt.label_table(sp[p1], cv_sp, paper, c.has_sp,
                                                          pt.PAPER_BANDS)
        return out

    picks = {}
    slope = np.array([s["pen_slope"] for s in grid])
    seasonp = np.array([s["season_penalty"] for s in grid])
    shrink = np.array([s["shrink"] for s in grid])
    families = {
        "(i) best overall": dict(),
        "(ii) best holding 300-1,500 km within 2% of photos only": dict(band=rule, base=p0_sp),
        "(iv) shrinkage only (no new penalty)": dict(allowed=(slope == 0) & (seasonp == 0)),
        "season penalty only": dict(allowed=(slope == 0) & (shrink == 3.0)),
        **{f"(iii) frontier: slope {s:g}/1,000 km": dict(allowed=slope == s) for s in pt.SLOPES},
    }
    for name, kw in families.items():
        res = pt.nested_cv(sp, ge, folds, grid, **kw)
        allowed = kw.get("allowed")
        # The setting the same procedure picks on all of val (what would ship): with the
        # band rule when the family has one, as each fold's pick did.
        ok_all = np.ones(len(grid), bool) if allowed is None else allowed.copy()
        if "band" in kw:
            ok_all &= pt.holds_band(sp, kw["base"], np.arange(n), kw["band"])
        full = (pt.choose(sp, ge, np.arange(n), grid, ok_all) if ok_all.any()
                else None)
        picks[name] = {
            "fold_picks": [None if j is None else pt.describe(grid[j]) for j in res["chosen"]],
            "fold_pick_ids": res["chosen"],
            "all_val_pick": "photos only (no setting holds the band)" if full is None
            else pt.describe(grid[full]),
            "all_val_pick_id": full,
            "all_val_pick_setting": None if full is None else grid[full],
            "all_val_pick_in_sample_species_top1": None if full is None
            else round(float(sp[full][c.has_sp].mean()), 4),
            "out_of_fold": summary_of(res["species_right"], res["genus_right"])}
        oof = picks[name]["out_of_fold"]
        print(f"  {name}: out of fold {oof['species_top1']:.4f} / {oof['genus_top1']:.4f}; "
              f"band vs P0 {oof['rule_band_vs_photos_only']['delta']:+.4f} "
              f"(lb {oof['rule_band_vs_photos_only']['lower95']:+.4f}); "
              f"pick {picks[name]['all_val_pick']}", flush=True)
    refs = {"P0 photos only": (p0_sp, p0_ge), "P1 served": (sp[p1], ge[p1])}
    for name, (a, b) in refs.items():
        v = summary_of(a, b)
        nb = v["rule_band_vs_photos_only"]
        print(f"  {name}: {v['species_top1']:.4f} / {v['genus_top1']:.4f}; band vs P0 "
              f"{nb['delta']:+.4f} (lb {nb['lower95']:+.4f})", flush=True)
    report = {"records": n, "grid_size": len(grid), "grid_declared": pt.GRID_DECLARED,
              "fold_sizes": np.bincount(folds).tolist(),
              "references": {k: summary_of(*v) for k, v in refs.items()},
              "picks": picks,
              "every_setting": [{"id": j, "setting": pt.describe(s),
                                 "species_right": int(sp[j][c.has_sp].sum()),
                                 "genus_right": int(ge[j].sum()),
                                 "rule_band_right": int(sp[j][rule].sum())}
                                for j, s in enumerate(grid)]}

    # Standard tables (Steve's format): P0, P1 and each pick, out of fold.
    conn = ro(args.manifest)
    labeller = heldout.Labeller(conn, extra=meta["truth_name"].tolist()
                                + meta["group_names"].tolist())
    keys = meta["record_key"].tolist()
    truths = {k: labeller.truth(t) for k, t in zip(keys, meta["truth_name"].tolist()) if t}
    feats = {k: {"species reference records": d} for k, d in zip(keys, meta["depth"].tolist())}
    names, gg, gf = (meta["group_names"].tolist(), meta["group_genus"].tolist(),
                     meta["group_family"].tolist())
    tables = {}
    runs = {"P0 photos only": [pt.NONE] * 5, "P1 served": [grid[p1]] * 5}
    for name in ("(i) best overall", "(ii) best holding 300-1,500 km within 2% of photos only",
                 "(iv) shrinkage only (no new penalty)"):
        runs[name] = [pt.NONE if j is None else grid[j] for j in picks[name]["fold_pick_ids"]]
    for name, per_fold in runs.items():
        zz = np.zeros_like(c.S)
        T = np.zeros(n)
        for f in range(5):
            rows = folds == f
            zf = pt.combine(c, per_fold[f])
            zz[rows] = zf[rows]
            T[rows] = pt.fit_temperature(c, zf, np.flatnonzero(~rows))
        res = {k: pt.ranks(zz[i], c.cand[i], c.SP[i], names, gg, gf, float(T[i]))
               for i, k in enumerate(keys)}
        judged = score({("m",): res}, truths, labeller)[("m",)]
        lad = ladder(res, set(truths), truths, labeller, False, name_equiv)
        conf = pt.top_confidence(c, zz, T)
        sp_right, _ = pt.top1(c, zz)
        tables[name] = {"ladder": lad, "by_depth": by_reference_depth(judged, feats),
                        "calibration_species": pt.calibration_report(conf, sp_right, c.has_sp),
                        "temperature_per_fold": sorted(set(np.round(T, 4).tolist()))}
        print(fmt_ladder(name, lad, tables[name]["by_depth"]), flush=True)
    report["tables"] = tables
    # Named cases: each record's top 5 species under P0, P1 and the picks (names only).
    cases = {}
    for k in args.named_case or []:
        if k not in keys:
            cases[k] = "not in this run"
            continue
        i = keys.index(k)
        cases[k] = {name: [x["name"] for x in pt.ranks(
            pt.combine(subset(c, np.arange(n) == i), per_fold[folds[i]])[0], c.cand[i], c.SP[i],
            names, gg, gf, 1.0, top=5)["species"]] for name, per_fold in runs.items()}
    report["named_cases"] = cases
    report["mcnemar_species_vs_photos_only"] = {
        name: mcnemar(v["out_of_fold"]["vs_photos_only"]["broken"],
                      v["out_of_fold"]["vs_photos_only"]["fixed"]) for name, v in picks.items()}
    path = out / "tuning.json"
    path.write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    print(f"-> {path} [{time.time() - t0:.0f}s]")


def fmt_ladder(name, lad, depth) -> str:
    from mycomap_vision.heldout_summary import KS
    lines = [f"{name}  (n = {lad['records']:,} records)",
             f"    {'':<24}" + "".join(f"{'top ' + str(k):>8}" for k in KS) + f"{'n':>9}"]
    for label, row in lad["rows"].items():
        if "note" in row:
            lines.append(f"    {label:<24}{row['note']}")
            continue
        cells = "".join("       -" if row[f"top{k}"] is None
                        else f"{100 * row[f'top{k}']['rate']:7.1f}%" for k in KS)
        lines.append(f"    {label:<24}{cells}{row['n']:>9,}")
    lines.append(f"    {'species by true refs':<24}{'top 1':>8}{'top 5':>8}{'n':>9}")
    for b, row in depth.items():
        r1, r5 = row["top1"]["rate"], row["top5"]["rate"]
        lines.append(f"    {b + ' records':<24}"
                     f"{'-' if r1 is None else f'{100 * r1:7.1f}%':>8}"
                     f"{'-' if r5 is None else f'{100 * r5:7.1f}%':>8}{row['n']:>9,}")
    return "\n".join(lines)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("score", "components", "tune"))
    ap.add_argument("--release-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--backbone", default="bioclip-2-ft-20261007-165400")
    ap.add_argument("--embeddings", help="the backbone's embeddings folder (default: data's)")
    ap.add_argument("--validated-after", help="only val records validated after this date")
    ap.add_argument("--heldout", help="also read query vectors from this held-out benchmark")
    ap.add_argument("--heldout-dir")
    ap.add_argument("--store", help="occurrence store (default: data/occurrence's)")
    ap.add_argument("--named-case", action="append", help="a record_key to report by name")
    ap.add_argument("--errata", help="the release's erratum TSV (errata.py)")
    ap.add_argument("--paper-attrs", help="`mv paper-attrs` TSV: use the paper's distance_band "
                    "(public points) for the bands and the 300-1,500 km rule")
    args = ap.parse_args(argv)
    {"score": cmd_score, "components": cmd_components, "tune": cmd_tune}[args.step](args)


if __name__ == "__main__":
    main()
