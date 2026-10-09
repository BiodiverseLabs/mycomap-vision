"""The published checkpoints on a dataset release, in one re-runnable command:
`mv external run --release v1 [--allow-draft]` (its val split, against its train split)
and `mv external run --release v1-test` (the test release, against v1's train + val).

labels -> predict -> crosswalk -> coverage -> report (release_eval.scored_split), with the
release's labels as the answer key and the reference splits for the reference-depth
breakdown and the reference coverage. Everything goes under one release-keyed folder,
<data>/research-results/<release>/external/<split>/:

    run.json                the steps done, the release cited, the inputs' hashes
    coverage.json           (iv), per model, exact names (and with the crosswalk)
    <model>.logits.sqlite   each photo's logits (resumable: a re-run classifies only what
                            is missing)
    <model>.jsonl           one answer per record (rewritten from the logits each run)
    report.json             the protocol tables (published_report.tables), both scorers

Idempotent: a re-run with the same release, split and checkpoints reproduces the files;
nothing is written to the manifest. Photo only, temperature 1, as everywhere else.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from . import dataset, release_eval

RESULTS_DIR = "research-results"


def results_dir(rel, split: str, root: Path | None = None) -> Path:
    from ... import config
    return (root or config.DATA_DIR / RESULTS_DIR) / rel.id / "external" / split


def truth_of(row: dict):
    """The answer key: the release's own label (heldout.Truth)."""
    from ...heldout import Truth
    return Truth(species=row.get("species") or "", genus=row.get("genus") or "",
                 family=row.get("family") or "", label=row.get("label") or "",
                 normalised=False, guest=None)


def _write_json(path: Path, doc: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    tmp.replace(path)


def run(conn, rel, eval_split: str | None = None, models=None,
        classify_for: Callable | None = None, matcher=None, fetch_crosswalk: bool = True,
        batch_size: int = 16, root: Path | None = None, out_root: Path | None = None,
        vision_results=(), log=print) -> dict:
    """The whole protocol for each published checkpoint on a release split. `classify_for(
    model)` returns a logits function (default: the checkpoint on this machine's GPU/CPU);
    `matcher` a crosswalk.Matcher (default: the GBIF cache, asked only for names it lacks
    when `fetch_crosswalk`); `vision_results` JSONL files of Vision's answers on the same
    records ({"backbone", "method", "observation_id", "result"}) for the same-vocabulary
    tables."""
    from ... import heldout, heldout_report
    from . import crosswalk as gbif
    from . import published as external
    from . import published_report as pr
    split, ref_splits = release_eval.scored_split(rel, eval_split)
    base = release_eval.base_release(rel) if dataset.is_test_release(rel) else rel
    cited = dataset.cite(rel)
    out = results_dir(rel, split, out_root)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "run.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    if state.get("dataset_release", {}).get("manifest_sha256") not in (None,
                                                                       cited["manifest_sha256"]):
        raise ValueError(f"{out} holds results of another {rel.id} (MANIFEST differs): "
                         "a release never changes; remove the folder if it was a draft")
    state.update({"dataset_release": cited, "eval_split": split,
                  "reference": {"release": base.id, "splits": list(ref_splits)},
                  "protocol": "photo only, temperature 1, mean of the photos' logits"})
    steps = state.setdefault("steps", {})

    def done(step, **info):
        steps[step] = {"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), **info}
        _write_json(state_path, state)
    chosen = [external.model_for(m) for m in models] if models else list(external.MODELS.values())

    # 1. labels: each checkpoint's class map, rebuilt and checked once
    for m in chosen:
        if not external.labels_path(m, root).is_file():
            external.check_config(m, root)
            external.write_labels(m, root)
    done("labels", models={m.short: external.checkpoint_id(m, root) for m in chosen})

    rows = [dict(r) for r in rel.records(split)]
    ref_rows = [dict(r) for r in base.records(ref_splits)]
    truths = {str(r["record_key"]): truth_of(r) for r in rows}
    ref_count = Counter(r.get("species") or r.get("label") for r in ref_rows)
    ref_truths = [(truth_of(r), 1) for r in ref_rows]

    # 2. predict: logits per photo (cached, resumable), one answer per record
    vision: dict = {}
    for path in vision_results:
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if str(r["observation_id"]) in truths:
                    vision.setdefault((r["backbone"], r.get("method", "")), {})[
                        str(r["observation_id"])] = r["result"]
    extra = [t.label for t in truths.values()] + [t.label for t, _ in ref_truths]
    labeller = heldout.Labeller(conn, extra=extra)
    stores: dict = {}
    ext, predicted = {}, {}
    for m in chosen:
        cn = external.vision_names(external.read_labels(m, root), labeller)
        cache = external.LogitCache(out / f"{m.short}.logits.sqlite")
        try:
            inputs = {k: external.reference_photo_inputs(
                conn, [int(p["manifest_photo_id"]) for p in rel.photos(k)], stores)
                for k in truths}
            bare = [k for k, got in inputs.items() if not got]
            if len(bare) > external.MAX_WITHOUT_PHOTOS * max(1, len(truths)):
                raise RuntimeError(f"{len(bare):,} of {len(truths):,} {split} records have no "
                                   f"photo on this machine or in S3 (first: "
                                   f"{', '.join(bare[:5])}); nothing was scored")
            todo = [x for got in inputs.values() for x in got]
            missing = [p for p, _r in todo if p not in cache.get([p])]
            classify = None
            if missing:
                classify = (classify_for(m) if classify_for
                            else external.Classifier(m, root=root).logits)
            stats = external.classify_photos(todo, classify or (lambda ims: None), cache,
                                             batch_size, log)
            answers = {}
            with open(out / f"{m.short}.jsonl", "w", encoding="utf-8") as f:
                for k in sorted(truths):
                    pids = [p for p, _r in inputs[k]]
                    got = cache.get(pids)
                    rows_ = [got[p] for p in pids if p in got]
                    if not rows_:
                        continue
                    answers[k] = external.record_answer(np.stack(rows_), cn)
                    f.write(json.dumps({"dataset_release": rel.id, "split": split,
                                        "release_hash": cited["release_hash"],
                                        "observation_id": k, "backbone": m.backbone,
                                        "method": external.METHOD, "photos": len(rows_),
                                        "result": answers[k]}) + "\n")
        finally:
            cache.close()
        ext[m.backbone] = answers
        predicted[m.short] = {"records": len(answers), "without_photos": len(bare), **stats}
    done("predict", models=predicted)

    ctx = {"bench": rel.id, "split": split, "records": rows, "truths": truths,
           "vision": vision, "ext": ext, "want": [m.backbone for m in chosen],
           "labeller": labeller, "chosen": {}, "ref_truths": ref_truths}

    # 3. crosswalk: GBIF Backbone + Catalogue of Life, exact matches, cached
    wanted = pr.crosswalk_names(conn, ctx, with_reference=True)
    own = matcher is None
    cw = None
    if own and not fetch_crosswalk and not gbif.cache_path().is_file():
        log("  no GBIF cache and --no-fetch: tables by exact names only")
    else:
        matcher = matcher or gbif.Matcher()
        try:
            if fetch_crosswalk:
                matcher.fill(wanted, log=log)
            cw = gbif.Crosswalk.from_cache(matcher, wanted)
        finally:
            if own:
                matcher.close()
    done("crosswalk", names=len(wanted), used=cw is not None)

    # 4. coverage and the report
    coverage = {}
    for m in chosen:
        cn = pr.class_names(ctx, m.backbone)
        coverage[m.short] = {
            label: {"split": pr.coverage(((t, 1) for t in truths.values()), pr.Vocabulary(cn, x)),
                    "reference": pr.coverage(ref_truths, pr.Vocabulary(cn, x))}
            for label, x in ((pr.EXACT, None), (pr.CROSSWALK, cw)) if label == pr.EXACT or x}
    _write_json(out / "coverage.json", {"dataset_release": cited, "eval_split": split,
                                        "models": coverage})
    done("coverage")
    feats = {o: {"species reference records": heldout_report.bucket(
        ref_count.get(t.species, 0), pr.DEPTH_BUCKETS) if t.species else "n/a"}
        for o, t in truths.items()}
    scorers = {pr.EXACT: pr.Scorer(labeller)}
    if cw is not None:
        scorers[pr.CROSSWALK] = pr.Scorer(labeller, cw)
    report = {"dataset_release": cited, "eval_split": split,
              "reference": {"release": base.id, "splits": list(ref_splits)},
              "records": len(rows), "scored_records": len(truths),
              "formal_species_records": sum(pr.name_kind(t) == pr.FORMAL for t in truths.values()),
              "photo_only": state["protocol"], "scorers": list(scorers), "models": {},
              "vision_models": {}}
    for m in chosen:
        report["models"][m.backbone] = {
            "hf_id": m.hf_id, "classes": m.num_classes, "licence": external.LICENCE,
            "checkpoint": external.checkpoint_id(m, root),
            "answered": len(set(ext[m.backbone]) & set(truths)),
            **{label: pr.tables(ctx, sc, m.backbone, ref_truths, feats)
               for label, sc in scorers.items()}}
    for label, sc in scorers.items():
        for name, t in pr.tables(ctx, sc, None, ref_truths, feats).items():
            report["vision_models"].setdefault(name, {})[label] = t
    _write_json(out / "report.json", report)
    done("report")
    return {"folder": str(out), "dataset_release": rel.id, "eval_split": split,
            "records": len(rows), "models": predicted,
            "crosswalk": "used" if cw is not None else "not used"}
