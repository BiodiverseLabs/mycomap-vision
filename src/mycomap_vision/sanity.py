"""The sanity gate a newly trained model passes before it may be served (Steve, 2026-10-11).

Not a comparison with the model it replaces: pre-declared checks that the model is whole and
working, read from its files, its embeddings and its own comparison on the dataset release
it was trained on. Thresholds are fixed here, before any result, and the runbook
(docs/retrain-v1.md) repeats them:

- files       its weights, heads and metadata are there; it names the release it was
              trained on (same release_hash), its version, and its val cutoff;
- heads       the classifier heads are finite, none all zero, and know every species of
              the release's train split;
- embeddings  every included photo of the release has a finite, unit-length vector
              (at most MAX_SKIPPED_SHARE missing: photos that would not decode);
- numbers     on the release's val split, photos only, with the default method: species
              top-1 within PLAUSIBLE_SPECIES, genus top-1 at least MIN_GENUS, family at
              least MIN_FAMILY, top-k rising with k, and the 100+ band above the 1-4 band;
- classes     no collapse: no species is the top answer for more than MAX_TOP1_SHARE of the
              val records, and at least MIN_DISTINCT_SHARE of the val records' true species
              appear somewhere as a top answer;
- calibration each rank's fitted temperature lies inside the searched grid (an edge means
              the scores are degenerate).

The CPU check (cpu_check) runs separately with no GPU visible, as the box has none.
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np

from . import evaluate, finetune, standard_report

DEFAULT_METHOD = "nearest+mean"
PLAUSIBLE_SPECIES = (0.35, 0.90)
MIN_GENUS = 0.65
MIN_FAMILY = 0.70
MAX_SKIPPED_SHARE = 0.002
MAX_TOP1_SHARE = 0.02
MIN_DISTINCT_SHARE = 0.30
NORM_TOLERANCE = 0.02
MIN_CALIBRATION_N = 50
# CPU, as on the box: seconds per photo to embed, per record to search, at CPU_THREADS.
CPU_THREADS = 2
MAX_EMBED_SECONDS = 3.0
MAX_SEARCH_SECONDS = 5.0


def check(name: str, ok: bool, detail: str) -> dict:
    return {"check": name, "ok": bool(ok), "detail": detail}


def check_files(backbone: str, release, models_root: Path | None = None) -> list[dict]:
    out = []
    try:
        meta = finetune.read_meta(backbone, models_root)
    except FileNotFoundError as e:
        return [check("files", False, str(e))]
    root = models_root or finetune.models_dir()
    weights = (root / meta.get("weights", "")).is_file() if meta.get("weights") else False
    cite = meta.get("dataset_release") or {}
    out.append(check("files", weights and bool(meta.get("heads")),
                     f"weights {'ok' if weights else 'MISSING'}, heads "
                     f"{meta.get('heads') or 'MISSING'}"))
    out.append(check("release", cite.get("release_hash") == release.release_hash
                     and meta.get("val_cutoff") == release.val_cutoff
                     and meta.get("split_trained") == "train",
                     f"trained on {cite.get('dataset_release')} "
                     f"({str(cite.get('release_hash'))[:12]}), split "
                     f"{meta.get('split_trained')}, val cutoff {meta.get('val_cutoff')}; "
                     f"this release {release.id} ({release.release_hash[:12]}), "
                     f"cutoff {release.val_cutoff}"))
    out.append(check("version", meta.get("version") == release.snapshot_date,
                     f"version {meta.get('version')}, release snapshot {release.snapshot_date}"))
    return out


def check_heads(backbone: str, release, models_root: Path | None = None) -> dict:
    try:
        heads = finetune.read_heads(backbone, models_root)
    except FileNotFoundError as e:
        return check("heads", False, str(e))
    problems = []
    for rank in finetune.RANKS:
        v = heads[rank][0].astype(np.float32)
        if not np.isfinite(v).all():
            problems.append(f"{rank}: non-finite values")
        zero = int((np.abs(v).sum(axis=1) == 0).sum())
        if zero:
            problems.append(f"{rank}: {zero} all-zero class vectors")
    trained = {r["species"] for r in release.training_records() if r["species"]}
    missing = trained - set(heads["species"][1])
    if missing:
        problems.append(f"{len(missing)} train species without a head "
                        f"(e.g. {sorted(missing)[:3]})")
    return check("heads", not problems,
                 "; ".join(problems) or f"{len(heads['species'][1]):,} species heads, finite, "
                 f"every train species covered")


def check_embeddings(conn: sqlite3.Connection, backbone: str, release,
                     embeddings_root: Path | None = None) -> dict:
    from .embed import load_embeddings
    root = embeddings_root / backbone if embeddings_root else None
    ids, vecs = load_embeddings(conn, backbone, root)
    want = set(release.photo_ids())
    have = set(ids.tolist())
    missing = len(want - have)
    share = missing / max(len(want), 1)
    v = vecs.astype(np.float32)
    finite = bool(np.isfinite(v).all())
    norms = np.linalg.norm(v, axis=1) if len(v) else np.zeros(0)
    off = int((np.abs(norms - 1) > NORM_TOLERANCE).sum())
    return check("embeddings", share <= MAX_SKIPPED_SHARE and finite and off == 0,
                 f"{len(want) - missing:,} of {len(want):,} release photos embedded "
                 f"({missing:,} missing, max {MAX_SKIPPED_SHARE:.1%}); "
                 f"{'finite' if finite else 'NON-FINITE values'}; "
                 f"{off:,} vectors off unit length")


def check_numbers(summary: dict, backbone: str, method: str = DEFAULT_METHOD) -> list[dict]:
    model = summary["photos_only"].get(f"{backbone} / {method}")
    if model is None:
        return [check("numbers", False, f"comparison {summary['comparison_id']} has no "
                                        f"{backbone} / {method} row")]
    rows = model["rows"]
    sp, ge = rows["species strict"], rows["genus strict"]
    out = []
    lo, hi = PLAUSIBLE_SPECIES
    out.append(check("species top-1", lo <= (sp.get("top1") or 0) <= hi,
                     f"{100 * (sp.get('top1') or 0):.1f}% on {sp['n']:,} val records "
                     f"(plausible {lo:.0%}-{hi:.0%})"))
    out.append(check("genus top-1", (ge.get("top1") or 0) >= MIN_GENUS,
                     f"{100 * (ge.get('top1') or 0):.1f}% on {ge['n']:,} (at least {MIN_GENUS:.0%})"))
    rising = all((sp.get(f"top{a}") or 0) <= (sp.get(f"top{b}") or 0)
                 for a, b in zip(evaluate.STANDARD_K, evaluate.STANDARD_K[1:]))
    out.append(check("top-k rises", rising, " / ".join(
        f"{100 * (sp.get(f'top{k}') or 0):.1f}" for k in evaluate.STANDARD_K)))
    bands = model["species_by_true_species_reference_records"]
    deep, thin = bands.get("100+", {}).get("top1"), bands.get("1-4", {}).get("top1")
    out.append(check("depth order", deep is None or thin is None or deep > thin,
                     f"100+ refs {deep}, 1-4 refs {thin}"))
    return out


def check_family(report: dict) -> dict:
    fam = (report["all_photos"].get("family") or {}).get("all", {}).get("top1")
    return check("family top-1", (fam or 0) >= MIN_FAMILY,
                 f"{100 * (fam or 0):.1f}% (at least {MIN_FAMILY:.0%})")


def check_calibration(report: dict) -> dict:
    """An edge temperature on a real val split means degenerate scores; on a handful of
    perfectly separated records it is simply the sharpest fit, so only ranks calibrated on
    at least MIN_CALIBRATION_N records are judged."""
    cal = report["all_photos"].get("calibration") or {}
    edges = (float(evaluate.T_GRID[0]), float(evaluate.T_GRID[-1]))
    bad = [r for r, c in cal.items() if c.get("temperature") in edges
           and c.get("n", MIN_CALIBRATION_N) >= MIN_CALIBRATION_N]
    return check("calibration", bool(cal) and not bad,
                 ", ".join(f"{r} T={c.get('temperature'):.4g}" for r, c in cal.items())
                 + (f"; AT THE GRID EDGE: {bad}" if bad else ""))


def top1_spread(vectors, ref, test, method: str = DEFAULT_METHOD, backbone: str | None = None
                ) -> dict:
    """How the val records' top-1 species answers spread: the largest single share, and how
    many of their true species appear somewhere as a top answer."""
    index, model = evaluate.fit_method(vectors, ref, method, backbone=backbone)
    names = index.labels["species"]
    answers = Counter()
    truths = set()
    for rec in test:
        if not rec.species:
            continue
        rs = evaluate.rank_scores(model.species_scores(vectors[rec.photo_rows]), index,
                                  "species")
        answers[names[int(np.argmax(rs))]] += 1
        truths.add(rec.species)
    n = sum(answers.values())
    top, count = answers.most_common(1)[0] if answers else ("", 0)
    return {"n": n, "top_species": top, "top_share": count / max(n, 1),
            "distinct_answers": len(answers), "distinct_truths": len(truths),
            "truths_answered": len(truths & set(answers)) / max(len(truths), 1)}


def check_classes(spread: dict) -> dict:
    ok = spread["top_share"] <= MAX_TOP1_SHARE and spread["truths_answered"] >= MIN_DISTINCT_SHARE
    return check("classes", ok,
                 f"most frequent top answer {spread['top_share']:.1%} of {spread['n']:,} val "
                 f"records (max {MAX_TOP1_SHARE:.0%}); {spread['truths_answered']:.0%} of "
                 f"{spread['distinct_truths']:,} true species given as a top answer somewhere "
                 f"(min {MIN_DISTINCT_SHARE:.0%})")


def run(conn: sqlite3.Connection, backbone: str, release, comparison_id: str,
        method: str = DEFAULT_METHOD, embeddings_root: Path | None = None,
        models_root: Path | None = None, spread: dict | None = None) -> dict:
    """Every check but the CPU one. `spread` (top1_spread) is computed when not given."""
    results = check_files(backbone, release, models_root)
    results.append(check_heads(backbone, release, models_root))
    results.append(check_embeddings(conn, backbone, release, embeddings_root))
    summary = standard_report.summary(conn, comparison_id)
    if (summary.get("dataset_release") or {}).get("release_hash") != release.release_hash:
        results.append(check("comparison", False, f"comparison {comparison_id} was not "
                                                  f"measured on {release.id}"))
    results += check_numbers(summary, backbone, method)
    row = conn.execute("select report_json from eval_runs where comparison_id = ? and "
                       "backbone = ? and method = ?", (comparison_id, backbone, method)).fetchone()
    if row:
        report = json.loads(row[0])
        results.append(check_family(report))
        results.append(check_calibration(report))
    if spread is None:
        shared = evaluate.shared_records(conn, [backbone], embeddings_root=embeddings_root,
                                         release=release)
        ids, vecs = shared.loaded[backbone]
        row_of = {int(p): i for i, p in enumerate(ids.tolist())}
        spread = top1_spread(vecs, evaluate.with_rows(shared.ref, row_of),
                             evaluate.with_rows(shared.test, row_of), method, backbone)
    results.append(check_classes(spread))
    return {"backbone": backbone, "dataset_release": release.cite(),
            "comparison_id": comparison_id, "method": method,
            "passed": all(r["ok"] for r in results), "checks": results}


def cpu_check(conn: sqlite3.Connection, backbone: str, release, images: list,
              method: str = DEFAULT_METHOD, threads: int = CPU_THREADS) -> dict:
    """Load the model with no GPU, as the box does, embed `images` (PIL images) and search
    the whole reference index for each one: seconds per photo and per search. Refused when
    a GPU is visible (set CUDA_VISIBLE_DEVICES=-1 before starting). With MV_SEARCH_INT8=1
    (the box's setting since v1) the index packs the fast int8 search as the server does
    (api.py, fastsearch.py) before the searches are timed, so the timing is the box's;
    without it the plain search is timed. The detail says which."""
    import time

    import torch

    from . import config, fastsearch, models
    from .identify import Identifier
    if torch.cuda.is_available():
        raise RuntimeError("a GPU is visible: start with CUDA_VISIBLE_DEVICES=-1 so the "
                           "check runs on the CPU, as on the box")
    torch.set_num_threads(threads)
    model = models.load_backbone(backbone)
    ident = Identifier(conn, backbone, method, photo_info=False, release=release, train=False)
    search = "plain search"
    if config.setting("MV_SEARCH_INT8") == "1":
        ref = getattr(getattr(getattr(ident, "nearest", None), "scorer", None), "ref", None)
        if not hasattr(ref, "use_int8"):
            raise RuntimeError("MV_SEARCH_INT8=1 but the reference vectors aren't memory-"
                               "mapped here: the box's int8 search can't be timed")
        ref.use_int8(fastsearch.REFINE)
        search = "int8 search"
    t0 = time.monotonic()
    vecs, _ = ident.embed(model, images)
    embed_s = (time.monotonic() - t0) / max(len(images), 1)
    t0 = time.monotonic()
    answered = 0
    for v in vecs:
        res = ident.identify_vectors(v[None, :], photo_lookup=lambda ids: {})
        answered += bool(res.get("ranks", {}).get("species"))
    search_s = (time.monotonic() - t0) / max(len(vecs), 1)
    ok = embed_s <= MAX_EMBED_SECONDS and search_s <= MAX_SEARCH_SECONDS and \
        answered == len(vecs)
    return check("cpu", ok, f"{embed_s:.2f} s/photo to embed (max {MAX_EMBED_SECONDS}), "
                            f"{search_s:.2f} s per search (max {MAX_SEARCH_SECONDS}, {search}), "
                            f"{answered}/{len(vecs)} answered, {threads} threads")


def format_result(r: dict) -> str:
    lines = [f"Sanity gate: {r['backbone']} on {r['dataset_release']['dataset_release']} "
             f"(comparison {r['comparison_id']}, {r['method']}): "
             f"{'PASS' if r['passed'] else 'FAIL'}"]
    for c in r["checks"]:
        lines.append(f"  [{'ok' if c['ok'] else 'FAIL'}] {c['check']:<14} {c['detail']}")
    return "\n".join(lines)
