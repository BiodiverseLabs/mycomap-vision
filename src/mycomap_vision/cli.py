"""`mv` command line."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import config, inat, manifest, photos, records
from .storage import open_store


def cmd_export_records(conn, args) -> None:
    print("Exporting green records from mycomap.org (read-only)...")
    print(json.dumps(records.export_records(conn), indent=2))


def cmd_fetch_inat(conn, args) -> None:
    print("Fetching iNat metadata...")
    stats = inat.fetch_all(conn, refresh=args.refresh, north_america_only=not args.all_regions,
                           limit=args.limit)
    print(json.dumps(stats, indent=2))


def cmd_record_sources(conn, args) -> None:
    """What the export would do to record sources (a dry run), or do it (--apply)."""
    from . import sources
    print("Reading green records and their sources from mycomap.org (read-only)...")
    text = records.fetch_export()
    exported_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    recs = records.build_records(records.parse_export(text), exported_at)
    by_source = {s: sum(r["source"] == s for r in recs) for s in sources.SOURCES}
    out = {"migrated_already": sources.migrated(conn), "records_by_source": by_source,
           "excluded_from_vision": {s: by_source[s] for s in sources.EXCLUDED}}
    if not out["migrated_already"]:
        plan = sources.plan_legacy(conn, recs)
        out["wrong_sourced_records"] = len(plan["wrong_records"])
        out["wrong_sourced_by_source"] = plan["by_source"]
        out["wrong_photos_to_unlink"] = plan["wrong_photos"]
    if args.apply:
        out["export"] = records.save_records(conn, recs)
    else:
        out["note"] = "dry run: nothing changed. --apply exports and migrates."
    print(json.dumps(out, indent=2))


def cmd_fetch_mo(conn, args) -> None:
    from . import mo
    print("Fetching Mushroom Observer image lists, licences and owners (read-only, 5 s apart)...")
    stats = mo.fetch_all(conn, refresh=args.refresh, north_america_only=not args.all_regions,
                         limit=args.limit)
    stats["relicensed"] = mo.relicense(conn)
    stats["licences"] = mo.licence_report(conn)
    print(json.dumps(stats, indent=2))


def cmd_mo_image_list(conn, args) -> None:
    """The MO images to ask for (a CSV for whoever makes the zip from MO's image store)."""
    from . import mo
    rows = mo.image_list(conn, args.store, args.size, north_america_only=not args.all_regions)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=mo.IMAGE_LIST_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(json.dumps({"images": len(rows), "out": args.out}, indent=2))


def cmd_import_mo_zip(conn, args) -> None:
    from . import mo
    store = open_store(args.dest, config.DATA_DIR)
    print(json.dumps(mo.import_zip(conn, Path(args.zip), store, size=args.size,
                                   max_side=args.max_side or None), indent=2))


def cmd_download_photos(conn, args) -> None:
    store = open_store(args.dest, config.DATA_DIR)
    checkpoint = None
    if args.checkpoint_to:
        from .aws import s3_checkpoint
        checkpoint = s3_checkpoint(conn, args.checkpoint_to)
    print(f"Downloading {args.size} photos to {store.location}...")
    stats = photos.download_all(conn, store, size=args.size,
                                north_america_only=not args.all_regions, limit=args.limit,
                                max_hours=args.max_hours, checkpoint=checkpoint,
                                random_order=args.random, record_sample=args.record_sample,
                                held_at=args.held_at,
                                hosts=(photos.OPEN_DATA_HOST,) if args.open_data_only else None,
                                policies=photos.default_policies(args.static_day_gb))
    stats["gb"] = round(stats["bytes"] / photos.GB, 2)
    print(json.dumps(stats, indent=2))


def cmd_adopt_benchmark_photos(conn, args) -> None:
    """A development benchmark's own downloaded photos as ordinary copies (photos.py)."""
    print(json.dumps(photos.adopt_benchmark_photos(conn, args.benchmark), indent=2))


def cmd_copy_photos(conn, args) -> None:
    source = open_store(args.source, config.DATA_DIR)
    dest = open_store(args.dest, config.DATA_DIR)
    print(f"Copying {args.size} photos from {source.location} to {dest.location}...")
    stats = photos.copy_photos(conn, source, dest, args.size, held_at=args.held_at)
    stats["gb"] = round(stats["bytes"] / photos.GB, 2)
    print(json.dumps(stats, indent=2))


def status_report(conn) -> dict:
    q = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    out = {
        "records": q("select count(*) from records"),
        "records_north_america": q("select count(*) from records where north_america = 1"),
        "records_label_conflicts": q("select count(*) from records where label_conflict = 1"),
        "inat_fetched": q("select count(*) from inat_observations where status = 'ok'"),
        "inat_missing": q("select count(*) from inat_observations where status = 'missing'"),
        "photos": q("select count(*) from photos"),
    }
    out["records_by_source"] = dict(conn.execute(
        "select source, count(*) from records group by 1").fetchall())
    from .sources import migrated
    out["record_sources_migrated"] = migrated(conn)
    out["photos_by_status"] = dict(conn.execute(
        "select status, count(*) from photos group by 1").fetchall())
    out["photos_by_license"] = dict(conn.execute(
        "select license_class, count(*) from photos group by 1").fetchall())
    out["downloaded_gb"] = round((q("select coalesce(sum(bytes), 0) from photos") or 0)
                                 / photos.GB, 2)
    return out


def cmd_aws_launch_downloader(conn, args) -> None:
    from . import aws
    print(json.dumps(aws.launch_downloader(conn, size=args.size, max_hours=args.max_hours,
                                           instance_type=args.instance_type,
                                           static_day_gb=args.static_day_gb), indent=2))


def cmd_aws_policies(conn, args) -> None:
    """Print the IAM policies with this deployment's bucket, region and role filled in."""
    from . import aws
    for name in ("instance-policy.template.json", "ops-policy.template.json",
                 "box-policy.template.json"):
        print(f"=== {name.replace('.template', '')}")
        print(aws.render_policy(name))


def cmd_aws_pull_manifest(conn, args) -> None:
    from . import aws
    print(json.dumps(aws.pull_manifest(conn, config.DATA_DIR / "aws"), indent=2))


def cmd_aws_backup(conn, args) -> None:
    from . import aws
    print(json.dumps(aws.backup(conn), indent=2))


def _split(v: str) -> list[str]:
    return [x.strip() for x in v.split(",") if x.strip()]


def cmd_aws_launch_trainer(conn, args) -> None:
    from . import aws
    finetune = [] if args.finetune.strip().lower() == "none" else _split(args.finetune)
    if args.resume:
        chosen = [f"--{k.replace('_', '-')}" for k in RUN_OWN_OPTIONS
                  if getattr(args, k) != args.parser_defaults[k]]
        if chosen:
            print(f"--resume keeps the run's own settings; ignoring {', '.join(chosen)}",
                  file=sys.stderr)
    backbones = [] if args.backbones.strip().lower() == "none" else _split(args.backbones)
    print(json.dumps(aws.launch_trainer(conn, backbones, _split(args.methods),
                                        size=args.size, max_hours=args.max_hours,
                                        instance_type=args.instance_type,
                                        test_days=args.test_days, finetune=finetune,
                                        sample_records=args.sample_records,
                                        allow_over_time=args.allow_over_time,
                                        allow_dirty=args.allow_dirty,
                                        allow_unpushed=args.allow_unpushed,
                                        spot=args.spot, spot_max_price=args.spot_max_price,
                                        resume=args.resume, picek=_split(args.picek),
                                        picek_exclude=args.picek_exclude_benchmarks,
                                        dry_run=args.dry_run, ec2_check=args.ec2_check,
                                        release=_dataset_release(args),
                                        allow_draft=args.allow_draft),
                     indent=2))


# What a resumed run takes from the run itself, not from the command line.
RUN_OWN_OPTIONS = ("backbones", "methods", "size", "test_days", "finetune", "sample_records",
                   "picek", "picek_exclude_benchmarks")


def trainer_release(args):
    """On the instance: the release shipped with the run (no private.sqlite), checked
    file by file and refused unless its MANIFEST.json is the one the launch recorded."""
    if not getattr(args, "release", None):
        return None
    from .replications.fungitastic import dataset
    return dataset.load(args.release, args.allow_draft, without_private=True,
                        expect_manifest_sha256=args.release_manifest)


def cmd_aws_train_job(conn, args) -> None:
    """Runs on the trainer instance (see aws.TRAINER_USER_DATA)."""
    from . import trainer
    from .storage import S3Store
    store = S3Store(args.source)
    should_stop = trainer.deadline(args.stop_after_hours)
    watcher = None
    if args.spot:
        from .spot import SpotWatcher
        watcher = SpotWatcher().start()
        print("Spot instance: watching for an interruption notice every "
              f"{watcher.interval:.0f} s")

    def upload(path, key):
        store.client.upload_file(str(path), store.bucket, key)
    backbones = [] if args.backbones.strip().lower() == "none" else _split(args.backbones)
    out = trainer.run_job(conn, store, backbones, _split(args.methods), upload,
                          args.run_id, size=args.size, test_days=args.test_days,
                          batch_size=args.batch_size, readers=args.readers,
                          finetune=_split(args.finetune), sample_records=args.sample_records,
                          picek=_split(args.picek), should_stop=should_stop,
                          picek_exclude=args.picek_exclude_benchmarks,
                          picek_ids_file=args.picek_exclude_ids,
                          picek_labels_hash=args.picek_labels_hash, interrupted=watcher,
                          picek_release=args.picek_release,
                          picek_release_manifest=args.picek_release_manifest,
                          picek_allow_draft=args.picek_allow_draft,
                          picek_place_cells=args.picek_place_cells,
                          picek_place_cells_sha256=args.picek_place_cells_sha256,
                          resume=trainer.RunFiles(store.client, store.bucket)
                          if args.resume else None, release=trainer_release(args))
    if watcher is not None:
        watcher.stop()
    print(json.dumps(out, indent=2))
    if out["comparison_id"]:
        from . import evaluate
        print_scoreboard(evaluate.scoreboard(conn, out["comparison_id"]))


def cmd_manifest_freeze(conn, args) -> None:
    """A consistent, fixed copy of the manifest for a release cut: the LOO exclusion list is
    written against it and `mv aws-launch-dataset-build --manifest` ships it unchanged."""
    from pathlib import Path

    from . import dataset_release as dr
    from .manifest import snapshot
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a freeze copy is never overwritten")
    snapshot(conn, out)
    print(json.dumps({"manifest": str(out), "bytes": out.stat().st_size,
                      "sha256": dr.sha256_file(out),
                      "records": conn.execute("select count(*) from records").fetchone()[0]},
                     indent=2))


def cmd_aws_launch_dataset_build(conn, args) -> None:
    from pathlib import Path

    from . import aws_dataset
    print(json.dumps(aws_dataset.launch(
        conn, args.name, args.freeze_approved, val_weeks=args.val_weeks,
        val_cutoff=args.val_cutoff, review_files=[Path(p) for p in args.exclusions],
        loo_exclusions=Path(args.loo_exclusions) if args.loo_exclusions else None,
        no_loo_exclusions=args.no_loo_exclusions,
        manifest_path=Path(args.manifest) if args.manifest else None,
        instance_type=args.instance_type, max_hours=args.max_hours,
        allow_dirty=args.allow_dirty, allow_unpushed=args.allow_unpushed), indent=2))


def cmd_aws_launch_dhash_index(args) -> None:
    from . import aws_dataset
    print(json.dumps(aws_dataset.launch_dhash_index(
        args.id, instance_type=args.instance_type, max_hours=args.max_hours,
        allow_dirty=args.allow_dirty, allow_unpushed=args.allow_unpushed), indent=2))


def cmd_aws_dataset_build_job(args) -> None:
    """Runs on the dataset-build instance (aws_dataset.USER_DATA)."""
    from pathlib import Path

    from . import aws, aws_dataset
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    out = aws_dataset.run_build_job(args.build_id, request, aws.s3_client(), args.bucket)
    print(json.dumps(out, indent=2))


def cmd_int8_check(conn, args) -> None:
    """Would int8 reference vectors change the answers? (quantize.py)"""
    from . import models, quantize
    rel = _dataset_release(args)
    r = quantize.check(conn, models.storage_name(args.backbone), rel, _split(args.methods))
    print(json.dumps(r, indent=2) if args.json else quantize.format_check(r))


def cmd_export_onnx(conn, args) -> None:
    """The served model's image tower as ONNX, for MV_EMBED_ONNX (onnxtower.py)."""
    from pathlib import Path

    from . import models, onnxtower
    from .finetune import models_dir
    name = models.storage_name(args.backbone)
    out = Path(args.out) if args.out else models_dir()
    backbone = models.load_backbone(args.backbone)
    meta = onnxtower.export(backbone, name, out, onnxtower.source_weights_of(name, models_dir()),
                            precision=args.precision)
    print(json.dumps(meta, indent=2))


def cmd_standard_summary(conn, args) -> None:
    """Steve's standard tables for a saved comparison (standard_report.py)."""
    from . import standard_report
    s = standard_report.summary(conn, args.comparison)
    print(json.dumps(s, indent=2) if args.json else standard_report.format_summary(s))


def cmd_model_sanity(conn, args) -> None:
    """The pre-declared sanity gate before a new model is served (sanity.py)."""
    import io

    from PIL import Image

    from . import sanity
    rel = _dataset_release(args)
    if rel is None:
        raise SystemExit("--release: the dataset release the model was trained on")
    r = sanity.run(conn, args.backbone, rel, args.comparison, method=args.method)
    if args.cpu_photos:
        store = open_store(args.source, config.DATA_DIR)
        val = set(rel.photo_ids(splits=("val",)))
        paths = conn.execute("select photo_id, path from photo_copies where store = ? and "
                             "size = 'large' order by photo_id", (store.location,)).fetchall()
        picked = [p for pid, p in paths if pid in val][:args.cpu_photos]
        images = [Image.open(io.BytesIO(store.get(p))).convert("RGB") for p in picked]
        r["checks"].append(sanity.cpu_check(conn, args.backbone, rel, images, args.method))
        r["passed"] = all(c["ok"] for c in r["checks"])
    print(sanity.format_result(r))
    if args.json:
        print(json.dumps(r, indent=2))
    if not r["passed"]:
        raise SystemExit(1)


def cmd_aws_pull_trainer(conn, args) -> None:
    from . import aws, evaluate
    out = aws.pull_trainer(conn, args.run)
    print(json.dumps(out, indent=2))
    if out["comparison_id"]:
        print_scoreboard(evaluate.scoreboard(conn, out["comparison_id"]))


def cmd_finetune(conn, args) -> None:
    """Fine-tune on this machine (a smoke test on the sample; full runs go to AWS)."""
    from datetime import datetime

    from . import finetune, models
    base = models.storage_name(args.base)
    name = args.name or f"{base}-ft-{datetime.now():%Y%m%d-%H%M%S}"
    cfg = finetune.FinetuneConfig(epochs=args.epochs, blocks=args.blocks,
                                  batch_size=args.batch_size, workers=args.workers,
                                  max_steps=args.max_steps)
    meta = finetune.finetune(conn, base, open_store(args.source, config.DATA_DIR), args.size,
                             name, cfg, test_days=args.test_days,
                             release=_dataset_release(args))
    print(json.dumps(meta, indent=2))
    print(f"next: mv embed --backbone {name} --size {args.size}, then mv compare "
          f"--backbones {base},{name}")


def cmd_embed(conn, args) -> None:
    from . import embed, models
    name = models.storage_name(args.backbone)
    models.resolve_spec(args.backbone)          # fail early on an unknown backbone
    store = open_store(args.source, config.DATA_DIR)
    todo = embed.photos_to_embed(conn, name, args.size, store.location,
                                 str(config.DATA_DIR), not args.all_regions, args.limit)
    print(f"{len(todo):,} {args.size} photos to embed with {name} from {store.location}")
    if not todo:
        return
    backbone = models.load_backbone(args.backbone)
    stats = embed.embed_photos(conn, store, backbone, todo,
                               config.DATA_DIR / "embeddings" / name,
                               batch_size=args.batch_size)
    print(json.dumps({**stats.__dict__, "skipped": stats.skipped[:20]}, indent=2))


def cmd_archive_embeddings(conn, args) -> None:
    from . import embed, models
    for b in _split(args.backbones):
        print(json.dumps(embed.archive_embeddings(conn, models.storage_name(b), args.label)))


def cmd_compare(conn, args) -> None:
    """Score several backbones and methods on the same test and reference photos."""
    from . import evaluate, models
    backbones = [models.storage_name(b.strip()) for b in args.backbones.split(",") if b.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rel = _dataset_release(args)
    if getattr(args, "test_release", None):
        # Reference = the release's train + val, scored = its separate test release
        # (replications/fungitastic/release_eval.py): the Picek vs Vision report.
        from .dataset_release import load_release
        from .replications.fungitastic import release_eval
        if rel is None or args.into or args.max_test:
            raise SystemExit("--test-release needs --release <its base>; no --into or "
                             "--max-test")
        test_rel = load_release(args.test_release, allow_draft=args.allow_draft)
        result = release_eval.compare_on_test(conn, rel, test_rel, backbones, methods,
                                              name_scores=args.name_scores,
                                              sets=not args.no_sets, per_image=args.per_image)
    else:
        result = evaluate.compare(conn, backbones, methods, test_days=args.test_days,
                                  max_test=args.max_test, name_scores=args.name_scores,
                                  sets=not args.no_sets, per_image=args.per_image,
                                  into=args.into, release=rel)
    config.ensure_dirs()
    path = config.REPORTS_DIR / f"compare-{result['comparison_id']}.json"
    path.write_text(evaluate.format_report(result), encoding="utf-8")
    print_scoreboard(evaluate.scoreboard(conn, result["comparison_id"]))
    print(f"-> {path}")


def _dataset_release(args):
    """The dataset release named by --release (dataset_release.py), or None."""
    if not getattr(args, "release", None):
        return None
    from .dataset_release import load_release
    rel = load_release(args.release, allow_draft=args.allow_draft)
    print(f"dataset release {rel.id} (release_hash {rel.release_hash[:12]}, "
          f"reference_hash {rel.reference_hash[:12]})"
          + (" DRAFT" if rel.manifest["draft"] else ""))
    if getattr(args, "errata", None):
        from pathlib import Path

        from . import errata
        e = errata.read(Path(args.errata))
        rel.use_errata(e)
        print(f"erratum {e.file} (sha256 {e.sha256[:12]}): {len(e.keys):,} records left out")
    return rel


def cmd_dataset(args) -> None:
    """Dataset releases for research (dataset_release.py, docs/dataset-release.md). Runs
    without opening the manifest for writing: build reads a private copy of it."""
    from pathlib import Path

    from . import dataset_release as dr
    root = Path(args.root) if args.root else dr.ROOT
    if args.action == "build":
        name = args.name or dr.draft_id(suffix="-test" if args.test_for else "")
        if args.dry_run and not dr.is_draft(name):
            raise SystemExit("--dry-run builds a draft: leave out --name (or give dry-<stamp>)")
        reader = dr.default_store_reader() if args.derive != "none" else None
        print(("DRY RUN, not v1: " if dr.is_draft(name) else "") + f"building {name} ...")
        m = dr.build(name=name, manifest_path=Path(args.manifest) if args.manifest else None,
                     root=root, review_files=[Path(p) for p in args.exclusions],
                     recipes=dr.parse_list(args.recipes), derive_photos=args.derive,
                     val_cutoff=args.val_cutoff, val_weeks=args.val_weeks,
                     north_america_only=not args.all_regions,
                     freeze_approved=args.freeze_approved, test_for=args.test_for,
                     test_ids=Path(args.ids) if args.ids else None, store_reader=reader,
                     loo_exclusions=Path(args.loo_exclusions) if args.loo_exclusions else None,
                     no_loo_exclusions=args.no_loo_exclusions,
                     base_dhash=Path(args.base_dhash) if args.base_dhash else None)
        print(json.dumps({k: m.get(k) for k in ("id", "draft", "hashes", "counts", "splits",
                                                "photos_identical_to_base_dropped",
                                                "near_base")}, indent=2))
    elif args.action == "dhash-index":
        from . import dhash
        rel = dr.load_release(args.id, root, allow_draft=True)
        try:
            index = dhash.build_index(rel, dr.default_store_reader())
        finally:
            rel.close()
        path = dhash.save_index(index, Path(args.out) if args.out else dhash.index_path(args.id))
        print(json.dumps({"release": args.id, "photos": index["photos"], "index": str(path),
                          "sha256": dr.sha256_file(path)}, indent=2))
    elif args.action == "errata":
        from . import errata
        rel = dr.load_release(args.id, root, allow_draft=True)
        try:
            rows = errata.mo_twin_rows(rel)
            included = {r[0] for r in rel.db.execute("select record_key from records "
                                                     "where included = 1")}
        finally:
            rel.close()
        path = errata.write(rows, Path(args.out))
        print(json.dumps({"release": args.id, "reason": "mo_twin_as_inat", "records": len(rows),
                          "included": sum(r["key"] in included for r in rows),
                          "file": str(path), "sha256": dr.sha256_file(path)}, indent=2))
    elif args.action == "pull-dhash-index":
        from . import aws, aws_dataset
        path = aws_dataset.pull_dhash_index(args.id, aws.s3_client(), aws.bucket())
        print(json.dumps({"release": args.id, "index": str(path),
                          "sha256": dr.sha256_file(path)}, indent=2))
    elif args.action == "verify":
        print(json.dumps(dr.verify(args.id, root, photos=args.photos), indent=2))
    elif args.action == "derive":
        rel = dr.load_release(args.id, root, allow_draft=True)
        try:
            d = rel.derive(args.photo, args.recipe)
        finally:
            rel.close()
        if args.out:
            Path(args.out).write_bytes(d.body)
        print(json.dumps({"photo": args.photo, "recipe": args.recipe, "sha256": d.sha256,
                          "pixels_sha256": d.pixels_sha256, "width": d.width,
                          "height": d.height, "matches_release": True, "out": args.out},
                         indent=2))
    elif args.action == "diff":
        print(json.dumps(dr.diff(args.a, args.b, root), indent=2))
    elif args.action == "pull":
        from . import aws, dataset_s3
        dataset_s3.fetch_release(args.id, root, aws.s3_client(), aws.bucket(),
                                 include_private=True)
        print(json.dumps(dr.verify(args.id, root), indent=2))
    elif args.action == "upload":
        from . import aws, dataset_s3
        dr.load_release(args.id, root, allow_draft=True).close()     # checks every file
        print(json.dumps(dataset_s3.upload_release(args.id, root, aws.s3_client(), aws.bucket()),
                         indent=2))
    elif args.action == "public":
        m = dr.public_variant(args.id, root)
        print(json.dumps({k: m[k] for k in ("id", "hashes", "counts")}, indent=2))
    else:
        print(json.dumps(dr.show(args.id, root), indent=2))


def print_scoreboard(rows: list[dict]) -> None:
    pct = lambda v: "   -  " if v is None else f"{100 * v:5.1f}%"  # noqa: E731
    print(f"{'comparison':<24} {'backbone':<28} {'method':<13} {'species':>7} {'genus':>7} "
          f"{'family':>7} {'1st photo':>9}  test/ref")
    for r in rows:
        print(f"{r['comparison_id']:<24} {r['backbone']:<28} {r['method']:<13} "
              f"{pct(r['species_top1']):>7} {pct(r['genus_top1']):>7} {pct(r['family_top1']):>7} "
              f"{pct(r['species_top1_first_photo']):>9}  {r['n_test']}/{r['n_reference']}")


def cmd_screen(conn, args) -> None:
    """Embed several candidate backbones, then compare them with the baselines."""
    from . import evaluate, screening
    store = open_store(args.source, config.DATA_DIR)
    split = lambda v: [x.strip() for x in v.split(",") if x.strip()]  # noqa: E731
    result = screening.screen(conn, store, split(args.candidates), split(args.baselines),
                              size=args.size, methods=split(args.methods),
                              batch_size=args.batch_size)
    config.ensure_dirs()
    print(json.dumps({"embedded": result.embedded, "failed": result.failed,
                      "compared": result.compared}, indent=2))
    if result.comparison:
        print_scoreboard(evaluate.scoreboard(conn, result.comparison["comparison_id"]))


def cmd_inat_baseline(conn, args) -> None:
    """Score a saved comparison's test records with iNat's computer vision."""
    from . import evaluate, inat_cv
    client = inat_cv.InatClient(inat_cv.read_jwt(), config.DATA_DIR / "inat_cv_cache")
    result = inat_cv.run(conn, args.comparison, client)
    print(json.dumps({k: v for k, v in result.items() if k != "runs"}, indent=2))
    print_scoreboard(evaluate.scoreboard(conn, args.comparison))


def cmd_refresh(conn, args) -> None:
    """Export, fetch, new genera's taxonomy, download, embed and compare: the weekly loop."""
    from . import refresh
    store = open_store(args.dest, config.DATA_DIR)
    backbones = [b.strip() for b in args.backbones.split(",") if b.strip()] or None
    report = refresh.refresh(conn, store, scope=args.scope, size=args.size, backbones=backbones,
                             compare=not args.no_compare, lookup_taxonomy=not args.no_taxonomy,
                             taxonomy_minutes=args.taxonomy_minutes)
    print(json.dumps({"export": report.export, "fetch": report.fetch,
                      "taxonomy": report.taxonomy, "download": report.download,
                      "embedded": report.embedded}, indent=2))
    if report.comparison:
        from . import evaluate
        print_scoreboard(evaluate.scoreboard(conn, report.comparison["comparison_id"]))


def cmd_candidates(conn, args) -> None:
    """Records on .org that have a sequence but haven't been assessed yet."""
    from . import prospective
    print(f"{prospective.export_candidates(conn):,} candidates awaiting validation on .org")


def cmd_predict_pending(conn, args) -> None:
    """Identify not-yet-validated records now; check the answers when they turn green."""
    from . import models, prospective
    from .identify import Identifier
    name = models.storage_name(args.backbone)
    ids = prospective.unpredicted(conn, name, args.method, args.limit)
    print(f"{len(ids):,} candidates to predict with {name} / {args.method}")
    if not ids:
        return
    identifier = Identifier(conn, name, args.method)
    stats = prospective.predict_pending(conn, identifier, models.load_backbone(args.backbone),
                                        ids, prospective.PhotoFetcher())
    print(json.dumps(stats, indent=2))


def cmd_nightly(conn, args) -> None:
    """The nightly update (nightly.py): what it did, what tonight would change, or run it now.
    --plan only reads, so it also runs before the update is turned on (and on a laptop)."""
    from . import nightly, permissions
    settings = nightly.Settings.from_env()
    if args.plan:
        base_url, key = config.setting("MV_ORG_BASE_URL"), permissions.org_key()
        if not (base_url and key):
            raise SystemExit("set MV_ORG_BASE_URL and MV_ORG_VISION_KEY(_FILE)")
        rows, generated = nightly.fetch_green(base_url, key)
        plan = nightly.changes(conn, records.build_records(rows, generated))
        limit = settings.removal_limit(plan["before"])
        plan["removals_refused"] = plan["removed"] > limit
        plan["removal_limit"] = limit
        print(json.dumps(plan, indent=2))
        return
    if not (config.RELEASE_ROOT and nightly.enabled()):
        raise SystemExit("the nightly update runs on the server box: set MV_RELEASE_ROOT and "
                         "MV_NIGHTLY=1")
    layer = nightly.layer_for(config.RELEASE_ROOT)
    if args.now:
        if not layer.manifest.is_file():
            raise SystemExit("the server has not made this release's layer yet: start it first")
        if args.accept_removals:
            layer.accept_removals.touch()
        layer.run_now.touch()
        print("asked the server to run the nightly update within a minute"
              + (", taking the removals" if args.accept_removals else "")
              + "; `mv nightly` shows the result")
        return
    zone = None
    try:
        zone = settings.zone()
    except Exception as e:  # noqa: BLE001 - the report is still useful without it
        print(f"(time zone {settings.tz!r}: {e})", file=sys.stderr)
    print(json.dumps(nightly.status(conn, settings, zone=zone), indent=2))


def cmd_prospective(conn, args) -> None:
    from . import prospective
    print(json.dumps(prospective.report(conn), indent=2))


def cmd_scoreboard_export(conn, args) -> None:
    """One model's rows of a comparison as JSON, for mv scoreboard-import on another machine."""
    from . import scoreboard_io
    text = json.dumps(scoreboard_io.export_runs(conn, args.comparison, args.backbone), indent=1)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


def cmd_benchmark_export(conn, args) -> None:
    """A held-out report's aggregates as JSON, for mv benchmark-import on the server box."""
    from pathlib import Path
    from . import benchmark_io
    comparison = getattr(args, "comparison", None)
    if bool(getattr(args, "report", None)) == bool(comparison):
        raise SystemExit("give --report (a held-out report) or --comparison (on a release)")
    if comparison:
        try:
            payload = benchmark_io.export_comparison(conn, comparison,
                                                     getattr(args, "split", "val"),
                                                     sealed=getattr(args, "sealed", False))
        except ValueError as e:
            raise SystemExit(f"not exported: {e}")
    else:
        path = Path(args.report)
        report = json.loads(path.read_text(encoding="utf-8"))
        payload = benchmark_io.export_summary(report, path.name, args.reproducibility)
    text = json.dumps(payload, indent=1)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    else:
        print(text)


def cmd_experiments_check(args) -> int:
    """Check every experiment entry against the registry's rules; exit 1 on any bad one.
    Run it in the private research checkout before committing an entry: the site skips a
    bad entry (and logs it) rather than failing the whole list."""
    from pathlib import Path
    from . import experiments
    directory = Path(args.dir) if args.dir else experiments.EXPERIMENTS_DIR
    if not directory.is_dir():
        print(f"no experiments folder at {directory}", file=sys.stderr)
        return 1
    found, problems = experiments.load_checked(directory)
    for name, why in problems:
        print(f"BAD {name}: {why}")
    print(f"{len(found)} entries ok, {len(problems)} bad ({directory})")
    return 1 if problems else 0


def cmd_benchmark_import(conn, args) -> None:
    """(server box) A benchmark export into the manifest the site serves; '-' reads stdin."""
    from . import benchmark_io
    text = sys.stdin.read() if args.file == "-" else open(args.file, encoding="utf-8").read()
    try:
        result = benchmark_io.import_summary(conn, json.loads(text))
    except (benchmark_io.ImportRefused, ValueError) as e:
        raise SystemExit(f"not imported: {e}")
    print(json.dumps(result, indent=2))


def cmd_scoreboard_import(conn, args) -> None:
    """Rows from mv scoreboard-export into this machine's scoreboard (on the server box: the
    manifest the site serves). `-` reads them from standard input."""
    from . import scoreboard_io
    if args.file == "-":
        text = sys.stdin.read()
    else:
        with open(args.file, encoding="utf-8") as f:
            text = f.read()
    try:
        result = scoreboard_io.import_runs(conn, json.loads(text))
    except (scoreboard_io.ImportRefused, ValueError) as e:
        raise SystemExit(f"not imported: {e}")
    print(json.dumps(result, indent=2))


def cmd_calibrate_sets(conn, args) -> None:
    """Fit likely sets (likely.py) into a comparison saved before they existed."""
    from . import evaluate
    done = evaluate.calibrate_sets(conn, args.comparison, args.backbone, args.method)
    for d in done:
        print(f"{d['backbone']} / {d['method']}")
        for rank, s in d["sets"].items():
            if not s:
                print(f"  {rank:8} no short enough set reaches 50%")
                continue
            check = s.get("crosscheck") or {}
            print(f"  {rank:8} target {s['coverage']:.0%} (asked {s['requested']:.0%}), "
                  f"floor {s['floor']:.4f}; on held-back halves: coverage "
                  f"{check.get('coverage')}, mean size {check.get('mean_size')}; "
                  f"true names no list could hold: {s['unlistable_share']:.1%} of {s['n']:,}")


def cmd_scoreboard(conn, args) -> None:
    from . import evaluate
    print_scoreboard(evaluate.scoreboard(conn, args.comparison))


def cmd_models(conn, args) -> None:
    from . import embed, evaluate, models
    conn.executescript(embed.SCHEMA)
    counts = dict(conn.execute("select backbone, count(*) from embeddings group by 1").fetchall())
    print("Backbones (use an alias, or any timm:<name> / open_clip:<name>):")
    for name, alias in models.ALIASES.items():
        print(f"  {name:<14} {counts.pop(name, 0):>9,} photos embedded  {alias.note}")
        print(f"  {'':<14} {alias.spec}")
    for name, n in counts.items():
        print(f"  {name:<14} {n:>9,} photos embedded")
    print("Methods: " + ", ".join(evaluate.METHODS))


def cmd_fungus_gate(conn, args) -> None:
    from datetime import datetime, timezone
    from pathlib import Path

    from . import fungusgate
    if args.action in ("embed", "tune"):
        from . import fungusgate_tune as tune
        if not args.out:
            raise SystemExit("--out is required")
        if args.action == "embed":
            if not args.list:
                raise SystemExit("embed needs --list (key<TAB>path per line)")
            n = tune.embed_list(args.backbone, Path(args.list), Path(args.out))
            print(f"{n} photos embedded into {args.out}")
            return
        missing = [o for o in ("dev", "ood", "ood_manifest", "exclude") if not getattr(args, o)]
        if missing:
            raise SystemExit("tune needs " + ", ".join("--" + m.replace("_", "-") for m in missing))
        report = tune.tune(conn, Path(args.dev), Path(args.ood), Path(args.ood_manifest),
                           Path(args.exclude), Path(args.out), args.benchmark,
                           Path(args.negatives) if args.negatives else None,
                           Path(args.embeddings_root) if args.embeddings_root else None,
                           Path(args.text_from) if args.text_from else None,
                           args.on_served)
        print(json.dumps(report["chosen"], indent=2))
        print(f"gate and report.json in {args.out}; copy the gate into data/models to serve it")
        return
    if args.action == "show":
        gate = fungusgate.FungusGate.load()
        if gate is None:
            print("No fungus gate saved: identifications run ungated.")
            return
        t = gate.thresholds
        print(f"Fungus gate ({gate.mode}) on {gate.base} vectors: not a fungus below p={t.not_fungus}, "
              f"not sure below p={t.unsure}; {len(gate.classes)} prompts in "
              f"{len(set(gate.classes.tolist()))} classes")
        print(json.dumps(gate.provenance, indent=2))
        return
    if not 0 <= args.not_fungus <= args.unsure <= 1:
        raise SystemExit("need 0 <= --not-fungus <= --unsure <= 1")
    probe = None
    if args.probe:
        import numpy as np
        with np.load(args.probe) as f:
            probe = (f["w"], float(f["b"]))
    gate = fungusgate.build(fungusgate.Thresholds(args.not_fungus, args.unsure),
                            provenance={"tuned_on": args.tuned_on, "note": args.note,
                                        "built_at": datetime.now(timezone.utc).isoformat()},
                            probe=probe, mode=args.mode)
    print(f"saved {gate.save()}")


def cmd_serve(conn, args) -> None:
    from .api import serve
    conn.close()
    serve(args.host, args.port)


def cmd_fetch_place_fallback(conn, args) -> None:
    """Placeless records' region from their own source (placefallback.py): iNat place_ids,
    MO location names (mopoints reads every MO record's point and name first)."""
    from . import mopoints, placefallback
    out = placefallback.fetch_inat(conn)
    out["mo_points"] = mopoints.fetch_all(conn)
    out.update(placefallback.decide_mo(conn))
    out["records_marked_north_america"] = placefallback.apply(conn)
    print(json.dumps(out, indent=2))


def cmd_fetch_mo_points(conn, args) -> None:
    """MO records' public points (mopoints.py), read anonymously at MO's pace."""
    from . import mopoints
    print(json.dumps(mopoints.fetch_all(conn, refresh=args.refresh, limit=args.limit),
                     indent=2))


def cmd_release(conn, args) -> None:
    from . import release
    print(json.dumps(release.publish(conn, _split(args.backbones), label=args.label,
                                     make_current=args.make_current,
                                     int8_vectors=args.int8_vectors), indent=2))


def cmd_pull_release(conn, args) -> None:
    from . import release
    if not config.RELEASE_ROOT:
        raise SystemExit("pull-release runs on the server box: set MV_RELEASE_ROOT")
    print(json.dumps(release.pull(config.RELEASE_ROOT, args.release, keep=args.keep), indent=2))


def shown(name: str) -> str:
    """A name with what can't be seen made visible (a non-breaking space, a character
    that was lost on the way), as Python would write it."""
    return "".join(c if c.isprintable() and ord(c) != 0xFFFD
                   else c.encode("unicode_escape").decode() for c in name)


def cmd_name_spellings(conn, args) -> None:
    """Names written more than one way: which Vision merges, which wait for a person."""
    from . import names
    if hasattr(sys.stdout, "reconfigure"):      # a console that can't print a curly quote
        sys.stdout.reconfigure(errors="backslashreplace")
    rows = names.name_counts(conn)
    groups = names.group_name_variants(rows)
    report = names.summary(rows, groups)
    if args.json:
        print(json.dumps({"summary": report, "groups": [g.as_dict() for g in groups]},
                         indent=2))
        return
    merged, left = report["merged_in_vision"], report["left_for_a_person"]
    print(f"{report['names']:,} names in the manifest are {report['labels']:,} labels in Vision")
    print(f"{report['groups']:,} names need a fix on mycomap.org "
          f"({report['records_to_fix']:,} records)")
    print(f"  merged in Vision (only the writing differs): {merged['groups']:,} names, "
          f"{merged['spellings']:,} spellings, {merged['records']:,} records "
          f"({merged['records_relabelled']:,} re-labelled)")
    print(f"  left for a person, kept as separate labels: {left['groups']:,} names, "
          f"{left['spellings']:,} spellings, {left['records']:,} records")
    for g in groups:
        if g.confidence != "check":
            continue
        proposed = f"proposed: {g.proposed_name}" if g.proposed_name else "a person chooses"
        print()
        print(f"{shown(g.key)}  [{', '.join(g.reasons)}]  {proposed}")
        for s in g.spellings:
            print(f"  {s.records:>7,}  {shown(s.name)}")


def cmd_fetch_taxonomy(conn, args) -> None:
    """Family, order, class and phylum per genus from iNat (1 request/s, resumable)."""
    from . import taxonomy
    print("Asking iNat about each genus the records use (read-only, 1 request/s)...")
    print(json.dumps(taxonomy.fetch(conn, refresh=args.refresh,
                                    older_than_days=args.older_than, limit=args.limit),
                     indent=2))
    print(json.dumps(taxonomy.report(conn), indent=2))


def cmd_taxonomy(conn, args) -> None:
    """What the iNat taxonomy changes, and the genera a person should look at."""
    from . import taxonomy
    print(json.dumps(taxonomy.report(conn), indent=2))


def cmd_guests(conn, args) -> None:
    """Records whose DNA name is a guest of the fungus in the photo (guests.py)."""
    from . import guests, names, taxonomy
    labels = names.manifest_labels(conn)
    tax = taxonomy.for_manifest(conn)
    counts: dict[tuple[str, str], int] = {}
    for name, genus, family in conn.execute("select scientific_name, genus, family from records"):
        label = (labels.get(name, name) or "").strip()
        lab = taxonomy.labels_for(label, genus, family, label != (name or "").strip(), tax)
        group = guests.group_of(lab.genus)
        if group:
            counts[(group, lab.genus)] = counts.get((group, lab.genus), 0) + 1
    out = {}
    for group in guests.GROUPS:
        rows = sorted(((g, n) for (gr, g), n in counts.items() if gr == group), key=lambda r: -r[1])
        out[group] = {"left out": bool(guests.excluded(next(iter(guests.GROUPS[group])))),
                      "records": sum(n for _, n in rows), "genera": dict(rows)}
    print(json.dumps(out, indent=2))


def cmd_status(conn, args) -> None:
    print(json.dumps(status_report(conn), indent=2))


def cmd_permissions(conn, args) -> None:
    """Photographers' answers from mycomap.org: pull them now, and/or show where they stand."""
    from . import permissions
    if args.sync:
        try:
            print(json.dumps(permissions.sync(conn, accept_shrink=args.accept_shrink), indent=2))
        except permissions.PermissionSyncError as e:
            print(f"sync failed: {e}", file=sys.stderr)
            sys.exit(1)
    print(json.dumps(permissions.status_report(conn), indent=2))


def cmd_refresh_licenses(conn, args) -> None:
    print(f"Re-reading iNat licences for records last checked over {args.older_than_hours} h ago "
          "(1 request/s)...")
    print(json.dumps(inat.refresh_licenses(conn, older_than_hours=args.older_than_hours,
                                           limit=args.limit), indent=2))


CONTRIBUTORS_SQL = """
select p.owner_login, max(p.owner_name) as owner_name, p.owner_user_id,
       count(*) as photos,
       sum(p.license_class = 'arr') as arr_photos,
       sum(p.license_class = 'nc') as nc_photos,
       sum(p.license_class = 'open') as open_photos,
       count(distinct op.observation_id) as records
from photos p
join observation_photos op on op.photo_id = p.photo_id
join records r on r.observation_id = op.observation_id
group by p.owner_login, p.owner_user_id
order by arr_photos desc, photos desc
"""


def cmd_contributors(conn, args) -> None:
    """Who took the training photos, with an all-rights-reserved count for permission requests."""
    config.ensure_dirs()
    rows = conn.execute(CONTRIBUTORS_SQL).fetchall()
    if args.arr_only:
        rows = [r for r in rows if r["arr_photos"]]
    path = config.REPORTS_DIR / ("contributors-arr.csv" if args.arr_only else "contributors.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(rows[0].keys() if rows else [])
        for r in rows:
            w.writerow(list(r))
    print(f"{len(rows):,} contributors -> {path}")


def cmd_holdout(conn, args) -> None:
    """Records no training, reference index, release or nightly update may take (holdouts.py)."""
    from pathlib import Path

    from . import holdouts
    if args.action == "add":
        print(json.dumps(holdouts.add(conn, args.benchmark, holdouts.read_ids_csv(Path(args.csv))),
                         indent=2))
    elif args.action == "release":
        print(json.dumps(holdouts.release(conn, args.benchmark), indent=2))
    else:
        print(json.dumps(holdouts.summary(conn), indent=2))


def _heldout_ids(conn, args) -> list[str]:
    from pathlib import Path

    from . import heldout, holdouts
    subset = holdouts.read_ids_csv(Path(args.subset)) if getattr(args, "subset", None) else None
    return heldout.benchmark_ids(conn, args.name, subset, getattr(args, "limit", None),
                                 split=getattr(args, "split", None))


def cmd_heldout(conn, args) -> None:
    """The frozen held-out benchmark (heldout.py, heldout_report.py)."""
    from pathlib import Path

    from . import heldout
    if args.action == "freeze":
        if bool(args.dev) != bool(args.test):
            raise SystemExit("give both --dev and --test, or neither")
        out = heldout.freeze(conn, args.name, Path(args.csv),
                             titles_tsv=Path(args.titles) if args.titles else None,
                             links_csv=Path(args.links) if args.links else None,
                             snapshot_csv=Path(args.snapshot) if args.snapshot else None,
                             expect_sha=args.expect_sha,
                             splits={"dev": Path(args.dev), "test": Path(args.test)}
                             if args.dev else None,
                             split_json=Path(args.split_json) if args.split_json else None,
                             holdout=args.holdout, force=args.force)
        print(json.dumps(out, indent=2))
    elif args.action == "fetch":
        ids = _heldout_ids(conn, args)
        print(f"iNat details for {len(ids):,} records of {args.name} (1 request/s)...")
        out = {"details": heldout.fetch_details(conn, args.name, ids, refresh=args.refresh)}
        if not args.details_only:
            store = open_store(args.dest, heldout.bench_dir(conn, args.name))
            out["photos"] = heldout.fetch_photos(
                conn, args.name, ids, store, size=args.size, max_hours=args.max_hours,
                policies=photos.default_policies(args.static_day_gb))
            out["photos"]["gb"] = round(out["photos"]["bytes"] / photos.GB, 2)
        print(json.dumps(out, indent=2))
    elif args.action == "predict":
        from . import models
        ids = _heldout_ids(conn, args)
        name = models.storage_name(args.backbone)
        models.resolve_spec(args.backbone)          # fail early on an unknown backbone
        out = heldout.predict(conn, args.name, name, _split(args.methods), ids,
                              lambda: models.load_backbone(args.backbone), place=args.place,
                              size=args.size, batch_size=args.batch_size, redo=args.redo,
                              scores_out=Path(args.scores_out) if args.scores_out else None,
                              release=_dataset_release(args))
        print(json.dumps(out, indent=2))
    elif args.action == "inat":
        from . import inat_cv
        ids = _heldout_ids(conn, args)
        cache = Path(args.cache) if args.cache else config.DATA_DIR / "inat_cv_cache"
        client = inat_cv.InatClient(inat_cv.read_jwt(), cache)
        print(f"iNat computer vision on {len(ids):,} records of {args.name} (1 request/s)...")
        print(json.dumps(heldout.inat_cv(conn, args.name, ids, client, size=args.size,
                                         redo=args.redo), indent=2))
    elif args.action == "import-external":
        print(json.dumps(heldout.import_external(conn, args.name, Path(args.results),
                                                 args.backbone, redo=args.redo,
                                                 release=_dataset_release(args)), indent=2))
    else:
        from . import heldout_report, holdouts
        subset = holdouts.read_ids_csv(Path(args.subset)) if args.subset else None
        out = heldout_report.report(conn, args.name, split=args.split, subset=subset,
                                    reference_backbone=args.reference_backbone,
                                    reference_hash=args.reference_hash,
                                    log=lambda s: print(s, file=sys.stderr))
        print_heldout_report(out)


def cmd_external(conn, args) -> None:
    """Published fungi classifiers as outside baselines (replications/fungitastic/published*.py)."""
    from pathlib import Path

    from . import heldout
    from .replications.fungitastic import published as external
    from .replications.fungitastic import published_report as external_report
    models = ([external.model_for(m) for m in _split(args.model)] if getattr(args, "model", None)
              else list(external.MODELS.values()))
    if args.action == "run":
        from .replications.fungitastic import published_release
        rel = _dataset_release(args)
        if rel is None:
            raise SystemExit("mv external run needs --release (v1 for val, v1-test for test)")
        print(json.dumps(published_release.run(
            conn, rel, args.eval, [m.short for m in models],
            fetch_crosswalk=not args.no_fetch, batch_size=args.batch_size,
            vision_results=[Path(p) for p in args.vision_results or []]), indent=2))
        return
    if args.action == "labels":
        for m in models:
            external.check_config(m)
            print(json.dumps(external.write_labels(m), indent=2))
    elif args.action == "coverage":
        out = external_report.coverage_report(conn, args.name, args.split,
                                              [m.short for m in models],
                                              crosswalk=not args.no_crosswalk)
        print(json.dumps(out, indent=2))
    elif args.action == "crosswalk":
        from .replications.fungitastic import crosswalk as gbif
        matcher = gbif.Matcher(interval=1 / args.per_second)
        try:
            print(json.dumps(external_report.fill_crosswalk(
                conn, args.name, args.split, [Path(p) for p in args.results or []],
                vision_backbone=args.vision, models=[m.short for m in models],
                matcher=matcher), indent=2))
        finally:
            matcher.close()
    elif args.action == "predict":
        ids = _heldout_ids(conn, args)
        labeller = heldout.Labeller(conn)
        for m in models:
            out_path = (Path(args.out) if args.out and len(models) == 1 else
                        heldout.bench_dir(conn, args.name) / "external"
                        / f"{m.short}-{args.split or 'all'}.jsonl")
            print(f"{m.backbone} on {len(ids):,} records of {args.name} -> {out_path}")
            print(json.dumps(external.predict_heldout(
                conn, args.name, m, ids, out_path, labeller=labeller, size=args.size,
                batch_size=args.batch_size), indent=2))
    elif args.action == "baseline":
        for m in models:
            out = external.run_comparison(conn, args.comparison, m, batch_size=args.batch_size)
            print(json.dumps({k: v for k, v in out.items() if k != "run"}, indent=2))
        from . import evaluate
        print_scoreboard(evaluate.scoreboard(conn, args.comparison))
    else:
        out = external_report.report(
            conn, args.name, split=args.split,
            results_files=[Path(p) for p in args.results or []],
            vision_backbone=args.vision, models=_split(args.model) if args.model else None,
            with_reference=not args.no_reference, crosswalk=not args.no_crosswalk,
            out_path=Path(args.out) if args.out else heldout.bench_dir(conn, args.name)
            / "reports" / f"external-{args.split}-{heldout.now_iso().replace(':', '')[:15]}.json")
        print(external_report.format_report(out))
        print(f"-> {out['file']}")


def print_heldout_report(out: dict) -> None:
    """The standard summary as text (every table with its n), then as JSON, then where the
    full report (CIs, paired tests, calibration, breakdowns, label audit) was written."""
    from .heldout_summary import format_summary
    print(format_summary(out["summary"]))
    print()
    print(format_f1_and_per_image(out.get("models") or {}))
    print()
    print(json.dumps(out["summary"], indent=2))
    print(f"-> {out['files']['json']}")
    print(f"-> {out['files']['csv']}")

def format_f1_and_per_image(models: dict) -> str:
    """Species macro-F1 and per-image vs per-record species top-1 for each model (the
    numbers the Picek group's papers report), from the report's models section."""
    pct = lambda v: "   -  " if v is None else f"{100 * v:5.1f}%"  # noqa: E731
    lines = ["Species macro-F1, and species top-1 per record vs per photo:",
             f"  {'model':<56} {'macro-F1':>8} {'top-1/record':>12} {'top-1/photo':>11} "
             f"{'F1/photo':>8}"]
    for name, m in models.items():
        rec = (m.get("species") or {}).get("top1") or {}
        img = m.get("per_image") or {}
        lines.append(f"  {name:<56} {pct((m.get('species_macro_f1') or {}).get('macro_f1')):>8} "
                     f"{pct(rec.get('rate')):>12} "
                     f"{pct((img.get('species') or {}).get('rate')):>11} "
                     f"{pct(img.get('species_macro_f1')):>8}")
    return "\n".join(lines)


def cmd_picek_train(conn, args) -> None:
    """Train the Picek group's classifier recipe on our records (retrain.py). Full runs belong
    on a GPU instance; here: a smoke test (--max-steps) or a short run."""
    from datetime import datetime

    from .replications.fungitastic import retrain as picek
    preset = args.preset
    name = args.name or f"picek-{preset}-{datetime.now():%Y%m%d-%H%M%S}"
    cfg = picek.PicekConfig(preset=preset, epochs=args.epochs, lr=args.lr,
                            effective_batch=args.effective_batch, micro_batch=args.micro_batch,
                            val_days=args.val_days, val_max_photos=args.val_max_photos,
                            workers=args.workers, max_steps=args.max_steps,
                            grad_checkpointing=False if args.no_grad_checkpointing else None,
                            seed=args.seed, cache_px=args.cache_px,
                            exclude_benchmarks=args.exclude_benchmarks,
                            release=args.release, allow_draft=args.allow_draft)
    meta = picek.train(conn, open_store(args.source, config.DATA_DIR), args.size, name, cfg,
                       test_days=args.test_days)
    print(json.dumps({k: v for k, v in meta.items() if k != "history"}, indent=2))
    print(f"next: mv embed --backbone {name} --size {args.size}, then mv compare --backbones "
          f"{name} --methods classifier,classifier+month,classifier+month+place --per-image, "
          f"and mv heldout predict --backbone {name} --methods classifier,classifier+month")


def cmd_picek_bench(conn, args) -> None:
    """Measure the replication's speed: the data loader alone (CPU) and the GPU alone."""
    from .replications.fungitastic import retrain as picek
    out = {}
    if not args.gpu_only:
        store = open_store(args.source, config.DATA_DIR)
        items = [(int(pid), path, 0) for pid, path in conn.execute(
            "select photo_id, path from photo_copies where store = ? and size = ? "
            "order by photo_id limit ?", (store.location, args.size, args.photos))]
        out["loader"] = [picek.bench_loader(store, items, args.preset, w, args.photos)
                         for w in _split_ints(args.workers)]
    if not args.loader_only:
        out["gpu"] = []
        for mb in _split_ints(args.micro_batch):
            for gc in ((True, False) if args.both_checkpointing
                       else (not args.no_grad_checkpointing,)):
                try:
                    out["gpu"].append(picek.bench_gpu(args.preset, args.classes, mb,
                                                      args.steps, gc))
                except RuntimeError as e:          # e.g. CUDA out of memory: say so, go on
                    out["gpu"].append({"micro_batch": mb, "grad_checkpointing": gc,
                                       "error": str(e).splitlines()[0][:200]})
                finally:
                    import gc as _gc

                    import torch
                    _gc.collect()
                    torch.cuda.empty_cache()
    print(json.dumps(out, indent=2))


def _split_ints(v: str) -> list[int]:
    return [int(x) for x in _split(v)]


def loo_options(p) -> None:
    p.add_argument("--loo-exclusions", metavar="TSV",
                   help="the person-checked LOO exclusion list (scripts/loo_exclusions.py, "
                        "against the manifest this build freezes); a frozen release needs it")
    p.add_argument("--no-loo-exclusions", metavar="WHO_WHEN",
                   help="instead of the list: Steve's explicit word that there are none")


def release_options(p, errata: bool = True) -> None:
    p.add_argument("--release", metavar="ID",
                   help="read records, labels, photos and splits from this dataset release "
                        "(docs/dataset-release.md), not the live manifest")
    p.add_argument("--allow-draft", action="store_true",
                   help="accept a draft (dry-run) release: exploratory only")
    if errata:                # not yet on the AWS launchers: the instance would not apply it
        p.add_argument("--errata", metavar="TSV",
                       help="the release's erratum (errata.py): leave its records out of every "
                            "split; its sha256 goes into the outputs")


def cmd_paper_inat_cv(args) -> None:
    """E14: iNat's computer vision on a release's finds (photos, public point, date)."""
    from . import dataset_release as dr
    from . import expert_ids as E
    from . import inat_cv
    from . import paper_inat_cv as P
    from datetime import datetime, timezone
    rel = dr.load_release(args.release, allow_draft=args.allow_draft)
    d = P.work_dir(args.name)
    ids_path = d / "ids.txt"
    if args.action == "sample":
        if ids_path.exists() and not args.redo:
            raise SystemExit(f"{ids_path} exists; a drawn sample is never redrawn (--redo)")
        pool = [r["record_key"] for r in rel.db.execute(
            "select record_key, split from records where included = 1 order by record_key")
            if not args.split or r["split"] == args.split]
        ids = P.draw_fallback(pool, args.n, args.seed) if args.n else pool
        print(json.dumps(E.write_ids(ids_path, ids, {"set": args.name, **rel.cite(),
                                                     "seed": args.seed, "n": args.n,
                                                     "split": args.split, "pool": len(pool)}),
                         indent=2))
        return
    ids = E.read_ids(ids_path)[:args.limit] if args.limit else E.read_ids(ids_path)
    taxa = E.load_open_taxa(config.DATA_DIR / E.OPEN_TAXA)
    by_name = E.name_index(taxa)
    recs = {r["record_key"]: dict(r) for r in rel.db.execute("select * from records")}
    truths = {k: E.truth_taxa(recs[k], by_name, taxa) for k in ids}
    client = inat_cv.InatClient(inat_cv.read_jwt(), config.DATA_DIR / "paper-comparisons" /
                                "inat-cv-cache", interval=args.interval)
    if rel.manifest.get("test_for") and not args.inat_model:
        raise SystemExit("record the iNat CV model version first: --inat-model '<version, date>'")
    model = {"version": args.inat_model, "noted_at": E.utc(datetime.now(timezone.utc).isoformat())}         if args.inat_model else None
    meta = P.run(rel, args.name, ids, client, lambda pk: rel.derive(pk, "original").body, truths,
                 model=model)
    lines = [json.loads(l) for l in (d / "finds.jsonl").open(encoding="utf-8")]
    print(json.dumps({**meta, "summary": P.summarise(lines)}, indent=2))


def cmd_expert_ids(args) -> None:
    """E31: iNat identifications made before the DNA was visible, scored against it."""
    from . import dataset_release as dr
    from . import expert_ids as E
    rel = dr.load_release(args.release, allow_draft=args.allow_draft)
    d = E.work_dir(args.name)
    ids_path = d / "ids.txt"
    if args.action == "sample":
        if ids_path.exists() and not args.redo:
            raise SystemExit(f"{ids_path} exists; a drawn sample is never redrawn (--redo)")
        pool = [r["source_id"] for r in E.inat_records(rel)
                if not args.split or r["split"] == args.split]
        ids = E.draw_sample(pool, args.n, args.seed) if args.n else sorted(pool, key=lambda s: (len(s), s))
        print(json.dumps(E.write_ids(ids_path, ids, {"set": args.name, **rel.cite(),
                                                     "seed": args.seed, "n": args.n,
                                                     "split": args.split, "pool": len(pool)}),
                         indent=2))
        return
    ids = E.read_ids(ids_path)
    client = E.HistoryClient(E.cache_dir(), max_calls=args.max_calls)
    if args.limit:
        ids = ids[:args.limit]
    d_our, d_val = d / "our_dates.csv", d / "validators.csv"
    if args.action == "dna-dates":
        print(json.dumps(E.dna_dates(
            args.name, ids, args.sql_dir or config.RESEARCH_DIR / "paper" / "e31" / "sql",
            config.required("MV_ORG_SQL_SSH_HOST", "the read-only .org SQL route"),
            args.com_host or config.required("MV_COM_SQL_SSH_HOST",
                                             "the read-only .com SQL route")), indent=2))
    elif args.action == "resolve-logins":
        print(json.dumps({"validator_rows_given_an_id": E.resolve_logins(client, args.name)}))
    elif args.action == "fetch":
        try:
            print(json.dumps(E.fetch(client, ids), indent=2))
        except E.CallBudgetSpent as e:
            print(f"{e}; rerun to continue (answers are cached)")
    elif args.action == "snapshot":
        print(json.dumps(E.build_snapshot(client, ids, d / "snapshot.jsonl.gz",
                                          {"set": args.name, **rel.cite()}), indent=2))
    elif args.action == "score":
        push = {int(x) for x in (args.push_accounts or "").split(",") if x.strip()}
        our = args.our_dates or (d_our if d_our.exists() else None)
        val = args.validators or (d_val if d_val.exists() else None)
        summary = E.score(rel, args.name, client, our, val, push,
                          config.DATA_DIR / "secrets" / "e31_identifier_salt.txt")
        print(json.dumps({k: summary[k] for k in ("finds", "tier1", "tier2")}, indent=2))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mv", description="MycoMap Vision data tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("export-records", help="pull green records from mycomap.org (read-only)")

    p = sub.add_parser("record-sources", help="record sources from mycomap.org: what an export "
                       "would change (dry run), or --apply (exports and migrates)")
    p.add_argument("--apply", action="store_true")

    p = sub.add_parser("fetch-mo", help="fetch Mushroom Observer image lists, licences and owners "
                       "for MO records (read-only, MO's pace)")
    p.add_argument("--refresh", action="store_true", help="re-fetch records already fetched")
    p.add_argument("--all-regions", action="store_true", help="not only North America")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("mo-image-list", help="CSV of the MO images wanted (for a zip made from "
                       "MO's image store; nothing is downloaded from MO)")
    p.add_argument("--out", required=True)
    p.add_argument("--size", default="large")
    p.add_argument("--store", help="leave out images this store already holds at --size")
    p.add_argument("--all-regions", action="store_true")

    p = sub.add_parser("import-mo-zip", help="store MO image files from a zip (named "
                       "<MO image id>.jpg) as copies of the listed MO photos")
    p.add_argument("--zip", required=True)
    p.add_argument("--dest", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--size", default="large")
    p.add_argument("--max-side", type=int, default=1280,
                   help="shrink larger images to this many px (0 = keep as given)")

    p = sub.add_parser("fetch-inat", help="fetch iNat photo lists, licenses and owners")
    p.add_argument("--refresh", action="store_true", help="re-fetch records already fetched")
    p.add_argument("--all-regions", action="store_true", help="not only North America")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("download-photos", help="download photos within iNat's limits")
    p.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--max-hours", type=float)
    p.add_argument("--dest", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--checkpoint-to", help="s3://bucket/key to copy the manifest to every 10 min")
    p.add_argument("--random", action="store_true",
                   help="random order, for a representative sample with --limit")
    p.add_argument("--record-sample", type=int,
                   help="only the photos of this many randomly chosen records")
    p.add_argument("--held-at", choices=["small", "medium", "large"],
                   help="only photos the destination already holds at this other size "
                        "(fetch a sample again at a new size)")
    p.add_argument("--static-day-gb", type=float, default=20,
                   help="day cap for all-rights-reserved photos from static.inaturalist.org "
                        "(default 20, under iNat's 24); raise only by a person's decision. "
                        "The 4 GB hourly cap stays")
    p.add_argument("--open-data-only", action="store_true",
                   help="only photos in iNat's open-data bucket; leaves the capped host's "
                        "daily budget to another downloader")

    p = sub.add_parser("adopt-benchmark-photos",
                       help="register a development benchmark's downloaded photos as copies, so "
                            "copy-photos can put them in S3 instead of fetching them again")
    p.add_argument("--benchmark", required=True, help="e.g. heldout-2026-10-08")

    p = sub.add_parser("copy-photos", help="copy photos we already hold from one store to "
                                           "another (e.g. S3 to this machine), hash-checked")
    p.add_argument("--source", required=True, help="folder or s3://bucket/prefix")
    p.add_argument("--dest", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--held-at", choices=["small", "medium", "large"],
                   help="only photos the destination already holds at this other size")

    p = sub.add_parser("aws-launch-downloader",
                       help="download photos on a self-terminating EC2 instance into S3")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--max-hours", type=float, default=120)
    p.add_argument("--instance-type", default="t3.small")
    p.add_argument("--static-day-gb", type=float, default=20,
                   help="day cap for the static host on the instance (see download-photos)")

    sub.add_parser("aws-policies",
                   help="print the IAM policies filled in for MV_S3_BUCKET / MV_AWS_REGION")

    sub.add_parser("aws-pull-manifest",
                   help="merge the instance's S3 copies into the local manifest")
    sub.add_parser("aws-backup", help="copy the manifest, embeddings and reports to S3")

    p = sub.add_parser("aws-launch-trainer",
                       help="embed the S3 photos and compare on a self-terminating GPU instance")
    p.add_argument("--backbones", default="bioclip-2",
                   help="comma-separated aliases or specs (default: bioclip-2; 'none' with "
                        "--picek to train only the replication)")
    p.add_argument("--methods", default="nearest,species-mean,linear,hybrid")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--max-hours", type=float, default=24,
                   help="time limit; the job stops itself 45 minutes before it (default 24)")
    p.add_argument("--instance-type", default="g6.xlarge",
                   help="default g6.xlarge: the only size the 4-vCPU On-Demand G quota fits")
    p.add_argument("--test-days", type=int, default=28)
    p.add_argument("--finetune", default="bioclip-2",
                   help="backbones (also in --backbones) to fine-tune on the reference records "
                        "(default: bioclip-2; 'none' for no fine-tuning)")
    p.add_argument("--sample-records", type=int,
                   help="rehearsal: run everything on this many random records only "
                        "(its results can't be merged home)")
    p.add_argument("--picek", default="",
                   help="also train the Picek group's classifier recipe (retrain.py): presets "
                        "as preset[@epochs[@cache_px]], e.g. fungitastic-beit-b384@15@440 "
                        "(15 epochs from a 440 px photo cache); add the methods "
                        "classifier,classifier+month to score it")
    p.add_argument("--picek-exclude-benchmarks", default="match", choices=["match", "all"],
                   help="benchmark records the replication leaves out: match (default) = "
                        "exactly what the Vision models leave out (sealed benchmarks), so "
                        "both train on the same records; all = every benchmark's records. "
                        "With --release the release's splits decide, and a mismatch is refused")
    release_options(p, errata=False)  # the Picek stage trains on it; shipped without private.sqlite
    p.add_argument("--dry-run", action="store_true",
                   help="check and build everything locally (code archive, instance script, "
                        "manifest copy, label snapshot, EC2 request) and send nothing")
    p.add_argument("--ec2-check", action="store_true",
                   help="with --dry-run: also ask EC2 with DryRun=True (needs aws login; no "
                        "instance, no cost)")
    p.add_argument("--allow-over-time", action="store_true",
                   help="launch even when the time estimate exceeds --max-hours (the run "
                        "stops at the limit and keeps the stages it finished)")
    p.add_argument("--allow-dirty", action="store_true",
                   help="launch with uncommitted changes (they are NOT sent)")
    p.add_argument("--allow-unpushed", action="store_true",
                   help="launch a commit that is on no remote branch")
    p.add_argument("--spot", action="store_true",
                   help="a one-time Spot instance instead of On-Demand (uses the \"All G and "
                        "VT Spot Instance Requests\" quota); on AWS's 2-minute notice the run "
                        "keeps its finished stages and ends 'interrupted'")
    p.add_argument("--spot-max-price", type=float, metavar="DOLLARS_PER_HOUR",
                   help="with --spot: the most to pay an hour (default: the On-Demand price)")
    p.add_argument("--resume", metavar="RUN",
                   help="continue that run (stopped, interrupted or killed) under its own run "
                        "id: its finished stages are restored, only the rest run; it keeps "
                        "its own backbones, fine-tunes, methods, size and manifest")
    p.set_defaults(parser_defaults={k: p.get_default(k) for k in RUN_OWN_OPTIONS})

    p = sub.add_parser("aws-train-job", help="(runs on the trainer instance) embed, compare, "
                                             "upload the results to runs/<run>/")
    p.add_argument("--run-id", required=True)
    p.add_argument("--backbones", required=True)
    p.add_argument("--methods", required=True)
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--source", required=True, help="s3://bucket holding the photos")
    p.add_argument("--test-days", type=int, default=28)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--readers", type=int, default=16, help="parallel photo reads from S3")
    p.add_argument("--finetune", default="")
    p.add_argument("--picek", default="")
    p.add_argument("--picek-exclude-benchmarks", default="match", choices=["match", "all"])
    p.add_argument("--picek-exclude-ids", help="the ids to leave out with 'all' (shipped)")
    p.add_argument("--picek-labels-hash", help="refuse to train on other labels than these")
    p.add_argument("--picek-release", help="the shipped dataset release the Picek stage uses")
    p.add_argument("--picek-release-manifest", help="sha256 of its MANIFEST.json at launch")
    p.add_argument("--picek-allow-draft", action="store_true")
    p.add_argument("--picek-place-cells",
                   help="the place prior's counts the launch made from the release (shipped)")
    p.add_argument("--picek-place-cells-sha256", help="their sha256 at launch")
    p.add_argument("--release", help="the shipped dataset release the fine-tune trains on")
    p.add_argument("--release-manifest", help="sha256 of its MANIFEST.json at launch")
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--sample-records", type=int)
    p.add_argument("--stop-after-hours", type=float,
                   help="stop cleanly (between batches) after this long and upload what "
                        "finished; set by the launcher to leave time before the hard limit")
    p.add_argument("--spot", action="store_true",
                   help="watch the instance metadata for a Spot interruption notice")
    p.add_argument("--resume", action="store_true",
                   help="restore the stages this run finished before, from its S3 folder")

    p = sub.add_parser("finetune", help="fine-tune a backbone's last blocks on the reference "
                                        "records (smoke test here; full runs on AWS)")
    p.add_argument("--base", required=True, help="an embedded backbone, e.g. bioclip-2")
    p.add_argument("--name", help="default: <base>-ft-<date-time>")
    p.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--max-steps", type=int)
    p.add_argument("--test-days", type=int, default=28)
    release_options(p)

    from .replications.fungitastic.retrain import PRESETS
    p = sub.add_parser("picek-train", help="train the Picek group's fungi classifier recipe "
                                           "(FungiTastic / DF20) on our records (retrain.py)")
    p.add_argument("--preset", default="fungitastic-beit-b384", choices=sorted(PRESETS))
    p.add_argument("--name", help="default: picek-<preset>-<date-time>")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--epochs", type=int, help="default: the preset's (50; DF20 100)")
    p.add_argument("--lr", type=float, help="default: the preset's")
    p.add_argument("--max-steps", type=int, help="stop after this many optimizer steps (smoke)")
    p.add_argument("--effective-batch", type=int, default=256)
    p.add_argument("--micro-batch", type=int, default=16)
    p.add_argument("--val-days", type=int, default=28,
                   help="validation slice: the last days before the comparison cutoff")
    p.add_argument("--val-max-photos", type=int, help="cap the validation pass (smoke)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--exclude-benchmarks", default="match", choices=["match", "all"],
                   help="match (default): leave out what the Vision models leave out "
                        "(sealed benchmarks); all: every benchmark's records")
    release_options(p, errata=False)      # reads the release by id (retrain.py), not here
    p.add_argument("--cache-px", type=int,
                   help="resize the training photos once to this shorter side on local disk "
                        "(e.g. 440) and train from those; off by default")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--test-days", type=int, default=28)

    p = sub.add_parser("picek-bench", help="the replication's speed: data loader (CPU) and "
                                           "GPU measured apart")
    p.add_argument("--preset", default="fungitastic-beit-b384", choices=sorted(PRESETS))
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--photos", type=int, default=512)
    p.add_argument("--workers", default="1,4", help="loader worker counts to try")
    p.add_argument("--micro-batch", default="16", help="micro-batch sizes to try on the GPU")
    p.add_argument("--classes", type=int, default=18000)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--no-grad-checkpointing", action="store_true")
    p.add_argument("--both-checkpointing", action="store_true",
                   help="measure with and without gradient checkpointing")
    p.add_argument("--loader-only", action="store_true")
    p.add_argument("--gpu-only", action="store_true")

    p = sub.add_parser("manifest-freeze", help="a fixed copy of the manifest for a release "
                                               "cut (the LOO list and the build both use it)")
    p.add_argument("--out", required=True)

    p = sub.add_parser("aws-launch-dataset-build",
                       help="cut a dataset release on a self-terminating instance in the "
                            "bucket's region (reads every original for the derived hashes)")
    p.add_argument("--name", required=True, help="v1, v2 ... (or a dry-<stamp> draft)")
    p.add_argument("--freeze-approved", help="who gave the freeze trigger, and when")
    p.add_argument("--val-weeks", type=int, default=8)
    p.add_argument("--val-cutoff", help="YYYY-MM-DD: validated after it = val")
    p.add_argument("--exclusions", action="append", default=[],
                   help="a person-checked review list (TSV); repeat")
    loo_options(p)
    p.add_argument("--manifest", help="ship this manifest file as the one the release freezes "
                                      "(the copy the LOO list was written against); default: a "
                                      "snapshot of the live manifest now")
    p.add_argument("--instance-type", default="c7i.2xlarge")
    p.add_argument("--max-hours", type=float, default=6)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--allow-unpushed", action="store_true")

    p = sub.add_parser("aws-launch-dhash-index",
                       help="write a release's dHash index (every included photo) to s3 on a "
                            "self-terminating instance in the bucket's region")
    p.add_argument("id", help="the release, e.g. v1 (its kept copy must be in s3)")
    p.add_argument("--instance-type", default="c7i.2xlarge")
    p.add_argument("--max-hours", type=float, default=4)
    p.add_argument("--allow-dirty", action="store_true")
    p.add_argument("--allow-unpushed", action="store_true")

    p = sub.add_parser("aws-dataset-build-job", help="(runs on the dataset-build instance)")
    p.add_argument("--build-id", required=True)
    p.add_argument("--request", required=True)
    p.add_argument("--bucket", required=True)

    p = sub.add_parser("export-onnx", help="write models/<backbone>.onnx (+ .onnx.json): the "
                                           "image tower as ONNX, checked against torch "
                                           "(MV_EMBED_ONNX on the server)")
    p.add_argument("--backbone", required=True)
    p.add_argument("--out", default=None, help="folder (default: data/models)")
    p.add_argument("--precision", choices=("fp32", "int8"), default="fp32",
                   help="fp32 (default: torch's vectors exactly, ~2x faster on CPU) or int8 "
                        "(refused unless it matches torch to cosine 0.99)")
    p = sub.add_parser("int8-check", help="score a release's val records with float16 and "
                                          "int8 reference vectors (box memory option)")
    p.add_argument("--backbone", required=True)
    p.add_argument("--methods", default="nearest,nearest+mean")
    p.add_argument("--json", action="store_true")
    release_options(p)

    p = sub.add_parser("standard-summary", help="Steve's standard tables (top 1/3/5/10, s.l., "
                                                "depth bands) for a saved comparison")
    p.add_argument("--comparison", required=True)
    p.add_argument("--json", action="store_true")

    p = sub.add_parser("model-sanity", help="the pre-declared sanity gate for a new model on "
                                            "its dataset release (docs/retrain-v1.md)")
    p.add_argument("--backbone", required=True, help="the fine-tuned model, e.g. "
                                                     "bioclip-2-ft-<run>")
    p.add_argument("--comparison", required=True, help="its comparison on the release")
    p.add_argument("--method", default="nearest+mean")
    p.add_argument("--cpu-photos", type=int, default=0,
                   help="also time N val photos on the CPU (start with CUDA_VISIBLE_DEVICES=-1)")
    p.add_argument("--source", help="photo store for --cpu-photos (default: the data folder)")
    p.add_argument("--json", action="store_true")
    release_options(p)

    p = sub.add_parser("aws-pull-trainer", help="bring a trainer run home (or the stages a "
                                                "stopped one finished): its complete "
                                                "embeddings replace the local ones (archived)")
    p.add_argument("--run", required=True, help="the run id aws-launch-trainer printed")

    p = sub.add_parser("embed", help="one vector per photo with a frozen backbone")
    p.add_argument("--backbone", required=True,
                   help="an alias (mv models) or timm:<name> / open_clip:<name>")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=32)

    p = sub.add_parser("archive-embeddings",
                       help="set backbones' embeddings aside (kept in data/embeddings-archive), "
                            "e.g. to embed the same photos again at another size")
    p.add_argument("--backbones", required=True, help="comma-separated aliases or names")
    p.add_argument("--label", required=True, help="e.g. medium; names the archive folder")

    p = sub.add_parser("compare", help="score backbones x methods on the same photos "
                                       "(newest weeks vs older records); saves to the scoreboard")
    p.add_argument("--backbones", required=True, help="comma-separated aliases or specs")
    p.add_argument("--methods", default="nearest", help="comma-separated: nearest, species-mean")
    p.add_argument("--test-days", type=int, default=28)
    p.add_argument("--max-test", type=int, help="sample this many test records")
    p.add_argument("--name-scores", action="store_true",
                   help="also score names s.l. and as species complexes (name_equiv.py, beta)")
    p.add_argument("--no-sets", action="store_true",
                   help="don't fit likely sets (likely.py) into the calibration")
    p.add_argument("--per-image", action="store_true",
                   help="also answer every test photo on its own (top-1 per rank and species "
                        "macro-F1 per photo, as the Picek group reports)")
    p.add_argument("--test-release", metavar="ID",
                   help="with --release: score this test release (e.g. v1-test) against the "
                        "release's train + val instead of its val split")
    p.add_argument("--into", metavar="COMPARISON",
                   help="add the runs to this saved comparison (same records only; replaces "
                        "a run of the same backbone and method there)")
    release_options(p)

    p = sub.add_parser("screen", help="embed candidate backbones one after another (skipping "
                                      "any that fail), then compare them with the baselines")
    p.add_argument("--candidates", required=True, help="comma-separated aliases or specs")
    p.add_argument("--baselines", default="", help="already-embedded backbones to include")
    p.add_argument("--methods", default="nearest,species-mean")
    p.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--batch-size", type=int, default=16)

    p = sub.add_parser("paper-inat-cv", help="(paper E14) iNat's computer vision on a release's "
                                             "finds: every photo at <= 1024 px, the public point "
                                             "and date (needs data/secrets/inat_jwt.txt)")
    p.add_argument("action", choices=["sample", "run"])
    p.add_argument("--release", default="v1")
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--name", required=True, help="the set's folder under "
                   "data/paper-comparisons/e14/, e.g. v1-val-pilot or v1-test")
    p.add_argument("--n", type=int, help="sample: how many (default: every included find)")
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--split", help="sample: only this split (val for the pilot)")
    p.add_argument("--redo", action="store_true")
    p.add_argument("--limit", type=int, help="run: only the first N finds (tests)")
    p.add_argument("--interval", type=float, default=7.5, help="run: seconds between iNat "
                   "requests (default 7.5: 8 a minute)")
    p.add_argument("--inat-model", help="run: iNat's CV model version and release date, as "
                   "iNat states it (required on a test release)")

    p = sub.add_parser("expert-ids", help="(paper E31) iNat identifications made before the DNA "
                                          "was visible, by high-volume identifiers, against "
                                          "the DNA name; public API, no token")
    p.add_argument("action", choices=["sample", "dna-dates", "fetch", "resolve-logins",
                                      "snapshot", "score"])
    p.add_argument("--release", default="v1")
    p.add_argument("--allow-draft", action="store_true")
    p.add_argument("--name", required=True, help="the set's folder under "
                   "data/paper-comparisons/e31/, e.g. v1-random10000 or v1-test")
    p.add_argument("--n", type=int, help="sample: how many (default: every iNat record)")
    p.add_argument("--seed", type=int, default=20261013)
    p.add_argument("--split", help="sample: only this split")
    p.add_argument("--redo", action="store_true", help="sample: draw again (never on a "
                   "sample already used)")
    p.add_argument("--limit", type=int, help="fetch/snapshot only the first N ids (tests)")
    p.add_argument("--max-calls", type=int, help="stop after this many iNat requests")
    p.add_argument("--our-dates", type=Path, help="score: CSV observation_id,source,at from "
                   ".org/.com (sequence created, name push)")
    p.add_argument("--validators", type=Path, help="score: CSV observation_id,inat_user_id")
    p.add_argument("--push-accounts", help="score: iNat user ids of MycoMap's push accounts")
    p.add_argument("--sql-dir", type=Path, help="dna-dates: the SQL templates (default: "
                   "paper/e31/sql in the research checkout)")
    p.add_argument("--com-host", help="dna-dates: ssh host of the read-only .com SQL route "
                   "(default MV_COM_SQL_SSH_HOST)")

    p = sub.add_parser("inat-baseline",
                       help="add iNat's computer vision to a comparison, on the same records "
                            "(needs a 24-hour token in data/secrets/inat_jwt.txt)")
    p.add_argument("--comparison", required=True, help="comparison id from mv scoreboard")

    p = sub.add_parser("refresh", help="the weekly loop: export, fetch, taxonomy of new genera, "
                                       "download, embed, compare")
    p.add_argument("--scope", default="new", choices=["new", "all"],
                   help="download photos of new records only (laptop) or every missing photo")
    p.add_argument("--dest", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--backbones", default="", help="default: every backbone already embedded")
    p.add_argument("--no-compare", action="store_true")
    p.add_argument("--no-taxonomy", action="store_true",
                   help="skip asking iNat about genera new since the last lookup")
    p.add_argument("--taxonomy-minutes", type=float, default=20,
                   help="stop asking iNat about new genera after this long; the next "
                        "refresh carries on (default 20)")

    sub.add_parser("candidates", help="export records awaiting validation on .org")
    p = sub.add_parser("predict-pending",
                       help="identify records awaiting validation, to check once they're green")
    p.add_argument("--backbone", required=True)
    p.add_argument("--method", default="hybrid")
    p.add_argument("--limit", type=int, help="newest candidates first")
    p = sub.add_parser("nightly", help="(server box) the nightly update: last runs and layer "
                       "size; --plan shows tonight's changes; --now runs it within a minute")
    p.add_argument("--plan", action="store_true",
                   help="read mycomap.org and print what tonight would change, changing nothing")
    p.add_argument("--now", action="store_true", help="ask the running server to update now")
    p.add_argument("--accept-removals", action="store_true",
                   help="with --now: take an answer that removes more records than the limit")
    sub.add_parser("prospective", help="how advance predictions fared once the DNA came in")

    p = sub.add_parser("scoreboard", help="saved comparison results")
    p.add_argument("--comparison", help="only this comparison id")

    p = sub.add_parser("calibrate-sets", help="fit likely sets (a list holding the right name "
                       "about 9 times in 10) into a saved comparison's own rows")
    p.add_argument("--comparison", required=True, help="comparison id from mv scoreboard")
    p.add_argument("--backbone", help="only this backbone's rows")
    p.add_argument("--method", help="only this method's rows")
    p = sub.add_parser("scoreboard-export", help="one model's rows of a comparison as JSON "
                       "(e.g. the iNat baseline, for the server box)")
    p.add_argument("--comparison", required=True, help="comparison id from mv scoreboard")
    p.add_argument("--backbone", default="external:inat-cv",
                   help="whose rows (default: the iNat baseline)")
    p.add_argument("--out", help="write to this file (default: standard output)")
    p = sub.add_parser("scoreboard-import", help="(server box) add rows from scoreboard-export "
                       "to a comparison this machine has; '-' reads standard input")
    p.add_argument("file", help="the export, or - for standard input")
    p = sub.add_parser("benchmark-export", help="a held-out report's, or a release "
                       "comparison's, aggregates (no record ids, names, observers or places) "
                       "for the server box's Results page")
    p.add_argument("--report", help="a report-*.json from mv heldout report")
    p.add_argument("--comparison", help="a comparison on a frozen dataset release (mv compare "
                   "--release); its reproducibility is reproduced-on-<release>")
    p.add_argument("--split", choices=("val", "test"), default="val",
                   help="with --comparison: val (the release's own) or test (a test release)")
    p.add_argument("--sealed", action="store_true",
                   help="with --comparison: a sealed test release, scored once")
    p.add_argument("--out", help="write to this file (default: standard output)")
    p.add_argument("--reproducibility", default="exploratory-pre-freeze",
                   help="exploratory-pre-freeze (default) or reproduced-on-<release>")
    p = sub.add_parser("benchmark-import", help="(server box) add a benchmark-export to the "
                       "site; '-' reads standard input")
    p.add_argument("file", help="the export, or - for standard input")
    p = sub.add_parser("experiments-check", help="check the experiment registry's entries "
                       "(the private research checkout) before committing; exit 1 on any bad one")
    p.add_argument("dir", nargs="?", help="the experiments folder (default: the research "
                                         "checkout's experiments/)")
    sub.add_parser("models", help="known backbones, photos embedded, methods")

    p = sub.add_parser("fungus-gate", help="the not-a-fungus gate in front of Identify: build "
                       "its BioCLIP 2 text vectors with thresholds (tune them on DEV only), or "
                       "show the saved one")
    p.add_argument("action", choices=["build", "show", "embed", "tune"],
                   help="build/show the saved gate; embed photos through the serving fork; "
                        "tune: fit the probe and pick thresholds on DEV (fungusgate_tune.py)")
    p.add_argument("--list", help="embed: key<TAB>path per line")
    p.add_argument("--backbone", default="bioclip-2-ft-20261007-165400",
                   help="embed: the fine-tuned model whose fork gives base vectors")
    p.add_argument("--out", help="embed: the .npz to write; tune: the folder for gate + report")
    p.add_argument("--dev", help="tune: DEV photo vectors (.npz from embed)")
    p.add_argument("--ood", help="tune: not-a-fungus photo vectors (.npz from embed)")
    p.add_argument("--ood-manifest", help="tune: the not-a-fungus set's manifest CSV")
    p.add_argument("--exclude", help="tune: TSV of reference records that are not iNat records "
                                     "(observation_id column); their animal/plant photos are "
                                     "the probe's negatives")
    p.add_argument("--negatives", help="tune: extra negative vectors (.npz from embed)")
    p.add_argument("--benchmark", default="heldout-2026-10-08")
    p.add_argument("--embeddings-root", help="tune: the frozen bioclip-2 reference embeddings")
    p.add_argument("--on-served", default=None, metavar="BACKBONE",
                   help="tune: the lighter gate, a probe on this served backbone's own vectors "
                        "(no second set of weights at serve time)")
    p.add_argument("--text-from", help="tune: a folder with a saved gate whose text vectors to "
                                       "reuse (skips loading BioCLIP 2's text tower)")
    p.add_argument("--not-fungus", type=float, default=0.0,
                   help="p(fungus) below this: not a fungus (no answer)")
    p.add_argument("--unsure", type=float, default=0.0,
                   help="p(fungus) below this: not sure (names, no chance)")
    p.add_argument("--probe", default=None,
                   help="an .npz with a linear probe's w and b on the base model's vectors "
                        "(fungusgate.fit_logistic)")
    p.add_argument("--mode", default="zero-shot", choices=["zero-shot", "probe", "both"],
                   help="which scores make p(fungus); probe and both need --probe")
    p.add_argument("--tuned-on", default="", help="where the thresholds came from (provenance)")
    p.add_argument("--note", default="")

    p = sub.add_parser("serve", help="run the API for the frontend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)

    sub.add_parser("fetch-place-fallback", help="records .org gives no place: ask iNat (place "
                                                "ids) and MO (location name) whether they are "
                                                "North American; read-only, polite pace")

    p = sub.add_parser("fetch-mo-points", help="each MO record's public point (shown GPS, else "
                                               "its location box) for dataset releases; "
                                               "anonymous, MO's pace (~5 s a request)")
    p.add_argument("--refresh", action="store_true", help="read every MO record again")
    p.add_argument("--limit", type=int)

    p = sub.add_parser("release", help="publish what the site serves (manifest, embeddings, "
                                       "fine-tuned weights) to s3://<bucket>/releases/<id>/")
    p.add_argument("--backbones", required=True, help="comma list of backbones to serve")
    p.add_argument("--label", help="appended to the release id, e.g. bioclip2-full")
    p.add_argument("--make-current", action="store_true",
                   help="also point releases/current.json at it (the box pulls that one)")
    p.add_argument("--int8-vectors", action="store_true",
                   help="ship the reference vectors as int8 + a scale each: half the box's "
                        "vector memory (Steve, 2026-10-12; accuracy: mv int8-check)")

    p = sub.add_parser("dataset", help="dataset releases for research: frozen, verifiable "
                                       "snapshots (docs/dataset-release.md)")
    p.add_argument("--root", help="where releases live (default data/research-releases)")
    dsub = p.add_subparsers(dest="action", required=True)
    q = dsub.add_parser("build", help="cut a release from a private copy of the manifest")
    q.add_argument("--name", help="v1, v2, v1-test ... (frozen; needs --freeze-approved); "
                                  "default: a draft dry-<stamp>")
    q.add_argument("--dry-run", action="store_true", help="a draft: counts and would-be hashes")
    q.add_argument("--freeze-approved", help="who gave the freeze trigger, and when")
    q.add_argument("--manifest", help="the manifest to copy (default: MV_MANIFEST_PATH)")
    q.add_argument("--exclusions", action="append", default=[],
                   help="a person-checked review list (TSV: kind, key, reason, note); repeat")
    q.add_argument("--recipes", default="original,long500-q90")
    q.add_argument("--derive", default="all",
                   help="derived hashes for 'all' included photos, 'none', or a number (drafts)")
    q.add_argument("--val-cutoff", help="YYYY-MM-DD: validated after it = val")
    q.add_argument("--val-weeks", type=int, default=8,
                   help="without --val-cutoff: the newest N weeks of validations are val")
    q.add_argument("--all-regions", action="store_true", help="not only North America")
    q.add_argument("--test-for", metavar="RELEASE",
                   help="a test release for this one: only --ids, all 'test', disjoint from it")
    q.add_argument("--ids", help="CSV with observation_id (or record_key): the test records")
    q.add_argument("--base-dhash", help="a test release: the --test-for release's dHash index "
                                        "(`mv dataset pull-dhash-index <id>`); flags near-copies")
    loo_options(q)
    q = dsub.add_parser("dhash-index", help="dHash every included photo of a release, here "
                                            "(a large release: mv aws-launch-dhash-index)")
    q.add_argument("id")
    q.add_argument("--out", help="default data/research-release-extras/<id>/dhash-index.npz")
    q = dsub.add_parser("pull-dhash-index", help="download a release's dHash index from s3 "
                                                 "and check its sha256")
    q.add_argument("id")
    q = dsub.add_parser("errata", help="write a release's erratum TSV for mo_twin_as_inat "
                                       "(errata.py): MO finds held as iNat records")
    q.add_argument("id")
    q.add_argument("--out", required=True, help="the TSV to write (never overwritten)")
    q = dsub.add_parser("verify", help="recompute every file and content hash")
    q.add_argument("id")
    q.add_argument("--photos", default="0", help="also re-read N originals (or 'all')")
    q = dsub.add_parser("derive", help="make one derived image from our original, hash-checked")
    q.add_argument("id")
    q.add_argument("--photo", required=True, help="photo key, e.g. inat:123456")
    q.add_argument("--recipe", required=True)
    q.add_argument("--out", help="write the image here")
    q = dsub.add_parser("diff", help="what changed between two releases")
    q.add_argument("a")
    q.add_argument("b")
    q = dsub.add_parser("public", help="write <id>-cc: CC photos only, no coordinates")
    q.add_argument("id")
    q = dsub.add_parser("show", help="counts and hashes")
    q.add_argument("id")
    q = dsub.add_parser("upload", help="copy the release to s3 research-releases/<id>/ (its "
                                       "kept copy, private part included)")
    q.add_argument("id")
    q = dsub.add_parser("pull", help="download a release's kept copy from s3 (with its "
                                     "private part) and verify it")
    q.add_argument("id")

    p = sub.add_parser("pull-release", help="(server box) download a release, verify it and "
                                            "make it current; then restart the server")
    p.add_argument("--release", help="a release id (default: releases/current.json)")
    p.add_argument("--keep", type=int, default=2, help="releases kept on disk (default 2)")

    sub.add_parser("status", help="counts of records, photos and licenses")

    p = sub.add_parser("name-spellings", help="names written more than one way: those Vision "
                                              "merges and those a person has to decide "
                                              "(read-only)")
    p.add_argument("--json", action="store_true", help="the full list, as JSON")

    p = sub.add_parser("fetch-taxonomy", help="family, order, class and phylum of every genus "
                                              "from iNat, kept in data/taxonomy/ (1 request/s; "
                                              "a re-run resumes)")
    p.add_argument("--refresh", action="store_true", help="ask about every genus again")
    p.add_argument("--older-than", type=float, metavar="DAYS",
                   help="ask again about genera answered more than DAYS ago")
    p.add_argument("--limit", type=int, help="at most this many genera, most records first")

    sub.add_parser("taxonomy", help="what iNat's taxonomy changes, and the genera in doubt "
                                    "(data/reports/taxonomy-doubts.csv; no network)")
    sub.add_parser("guests", help="records whose DNA name is a yeast or parasite of the "
                                  "fungus in the photo, and which are left out (guests.py)")

    p = sub.add_parser("permissions", help="photographers' answers from mycomap.org "
                                           "(needs MV_ORG_BASE_URL and MV_ORG_VISION_KEY)")
    p.add_argument("--sync", action="store_true", help="pull the answers now")
    p.add_argument("--accept-shrink", action="store_true",
                   help="with --sync: take an answer that leaves out people the last good "
                        "list has (refused otherwise: mycomap.org never deletes an answer)")

    p = sub.add_parser("refresh-licenses", help="re-read photo licences from iNat for records "
                                                "last checked long ago (mv serve can do this daily)")
    p.add_argument("--older-than-hours", type=float, default=24)
    p.add_argument("--limit", type=int, help="at most this many records, oldest first")

    p = sub.add_parser("contributors", help="write the contributor list as CSV")
    p.add_argument("--arr-only", action="store_true",
                   help="only people with all-rights-reserved photos")

    p = sub.add_parser("holdout", help="records held out for a benchmark: never trained on, "
                                       "indexed, released or added at night (holdouts.py)")
    hsub = p.add_subparsers(dest="action", required=True)
    q = hsub.add_parser("add", help="hold out the ids in a CSV's observation_id column")
    q.add_argument("--benchmark", required=True, help="the frozen set they belong to")
    q.add_argument("--csv", required=True)
    q = hsub.add_parser("release", help="lift a benchmark's exclusion: its records may join "
                                        "training and the reference index (logged)")
    q.add_argument("--benchmark", required=True)
    hsub.add_parser("list", help="held-out records per benchmark, and any the records "
                                 "table holds (should be 0)")

    p = sub.add_parser("heldout", help="held-out benchmarks: freeze, fetch, predict, inat, "
                                       "report (heldout.py)")
    hsub = p.add_subparsers(dest="action", required=True)

    def ids_options(q, subset_help="only the ids in this CSV's observation_id column"):
        q.add_argument("--name", required=True, help="the benchmark, e.g. heldout-2026-10-08")
        q.add_argument("--subset", help=subset_help)
        q.add_argument("--split", choices=["dev", "test"], help="only this split")
        q.add_argument("--limit", type=int, help="at most this many records (in id order)")

    q = hsub.add_parser("freeze", help="store a set, its answer key and split (again on a "
                                       "new snapshot of the same ids: takes the new names)")
    q.add_argument("--name", required=True)
    q.add_argument("--csv", required=True,
                   help="pool.csv (observation_id, com_name = .com's index name, lat, lng, ...)")
    q.add_argument("--titles", help="the .com record titles where they differ from the index "
                                    "name (TSV: record_id, index_name, title): the answer")
    q.add_argument("--links", help="with --titles: linked43.csv (record_id, external_id = the "
                                   "iNat id)")
    q.add_argument("--snapshot", help="the source snapshot CSV, to record its hash")
    q.add_argument("--dev", help="dev.csv: the split to tune and explore on")
    q.add_argument("--test", help="test.csv: the other split (sealed with --holdout)")
    q.add_argument("--split-json", help="split.json: counts and id hashes to check the splits")
    q.add_argument("--expect-sha", help="refuse unless the ids' sha256 starts with this")
    q.add_argument("--holdout", action="store_true",
                   help="a sealed set (the paper's): refuse records Vision has held and hold "
                        "every id out of training and the reference index")
    q.add_argument("--force", action="store_true",
                   help="on a sealed set whose test split was looked at: change test answers "
                        "anyway (logged in heldout_answer_history)")

    q = hsub.add_parser("fetch", help="iNat details and photos (resumable, iNat's limits)")
    ids_options(q)
    q.add_argument("--size", default="large", choices=["small", "medium", "large"])
    q.add_argument("--dest", help="folder or s3://bucket/prefix (default: the benchmark's "
                                  "folder beside the manifest)")
    q.add_argument("--details-only", action="store_true", help="no photos")
    q.add_argument("--refresh", action="store_true", help="read iNat's details again")
    q.add_argument("--max-hours", type=float)
    q.add_argument("--static-day-gb", type=float, default=20,
                   help="day cap for all-rights-reserved photos (see download-photos)")

    q = hsub.add_parser("predict", help="embed the photos and answer with Vision's index")
    ids_options(q)
    q.add_argument("--backbone", required=True, help="e.g. bioclip-2-ft-20261007-165400")
    q.add_argument("--methods", default="nearest,nearest+prior")
    q.add_argument("--place", default="inat", choices=["inat", "org", "none"],
                   help="place given to +prior methods: iNat's public one (default), "
                        ".org's true one, or none")
    q.add_argument("--size", default="large", choices=["small", "medium", "large"])
    q.add_argument("--batch-size", type=int, default=32)
    q.add_argument("--redo", action="store_true", help="answer records answered before")
    q.add_argument("--scores-out", help="also write each record's photo scores (.npz in "
                                        "occtune.save_scored_set's format), e.g. for "
                                        "mv tune-occurrence --scores")
    release_options(q)

    q = hsub.add_parser("inat", help="iNat's computer vision on a named subsample "
                                     "(needs a 24-hour token in data/secrets/inat_jwt.txt)")
    ids_options(q, subset_help="the subsample CSV, e.g. inat-subsample-2000.csv")
    q.add_argument("--size", default="large", choices=["small", "medium", "large"])
    q.add_argument("--cache", help="answer cache folder (default: data/inat_cv_cache)")
    q.add_argument("--redo", action="store_true")

    q = hsub.add_parser("report", help="scores with CIs, paired tests, calibration, "
                                       "breakdowns and the label audit (JSON + CSV)")
    q.add_argument("--name", required=True)
    q.add_argument("--split", default="dev", choices=["dev", "test"],
                   help="dev (default) to tune and explore; on a sealed benchmark test is the "
                        "paper number and every look at it is recorded")
    q.add_argument("--subset", help="only the ids in this CSV")
    q.add_argument("--reference-backbone",
                   help="whose reference index the breakdowns use (default: the newest)")
    q.add_argument("--reference-hash",
                   help="score the answers made against this reference (e.g. the run before "
                        "a relabel; default: each model's newest)")
    q = hsub.add_parser("import-external",
                        help="store an external model's answers (mv external predict's JSONL)")
    q.add_argument("--name", required=True)
    q.add_argument("--backbone", required=True, help="e.g. external:df20-vit-l384")
    q.add_argument("--results", required=True, help="the JSONL mv external predict wrote")
    q.add_argument("--redo", action="store_true", help="replace answers stored before")
    release_options(q)

    p = sub.add_parser("external", help="published fungi classifiers (DF20, FungiTastic) as "
                                        "outside baselines (replications/fungitastic/)")
    xsub = p.add_subparsers(dest="action", required=True)
    model_help = ("comma list of models (short name, external:<name> or BVRA/<repo>); "
                  "default: all")
    q = xsub.add_parser("run", help="the whole protocol on a dataset release's split, one "
                                    "re-runnable command: labels, coverage, predict, crosswalk, "
                                    "report (under research-results/<release>/external/<split>)")
    release_options(q)
    q.add_argument("--eval", choices=["val", "test"],
                   help="the scored split (default: val for a release, test for a test "
                        "release such as v1-test; anything else is refused)")
    q.add_argument("--model", help=model_help)
    q.add_argument("--batch-size", type=int, default=16)
    q.add_argument("--no-fetch", action="store_true",
                   help="crosswalk from the GBIF cache only (no requests)")
    q.add_argument("--vision-results", nargs="*",
                   help="JSONL of Vision's answers on the same records (same-vocabulary tables)")
    q = xsub.add_parser("labels", help="rebuild each class map from the dataset's metadata CSV")
    q.add_argument("--model", help=model_help)
    q = xsub.add_parser("coverage", help="(read-only) how much of a split and of Vision's North "
                                         "American records each model can name")
    q.add_argument("--name", required=True)
    q.add_argument("--split", default="dev", choices=["dev", "test"])
    q.add_argument("--model", help=model_help)
    q.add_argument("--no-crosswalk", action="store_true", help="exact names only")
    q = xsub.add_parser("crosswalk", help="match every formal name a report compares in GBIF "
                                          "(Backbone and Catalogue of Life; cached, paced; "
                                          "scoring only)")
    q.add_argument("--name", required=True)
    q.add_argument("--split", default="dev", choices=["dev", "test"])
    q.add_argument("--model", help=model_help)
    q.add_argument("--results", nargs="*", help="mv external predict JSONL files")
    q.add_argument("--vision", default="bioclip-2-ft-20261007-165400")
    q.add_argument("--per-second", type=float, default=4.0,
                   help="GBIF requests a second, both checklists together (default 4)")
    q = xsub.add_parser("predict", help="(reads the manifest only) answer a held-out split; "
                                        "writes JSONL for mv heldout import-external")
    ids_options(q)
    q.add_argument("--model", help=model_help)
    q.add_argument("--size", default="large", choices=["small", "medium", "large"])
    q.add_argument("--batch-size", type=int, default=16)
    q.add_argument("--out", help="the JSONL (one model only; default: the benchmark's "
                                 "external/<model>-<split>.jsonl)")
    q = xsub.add_parser("baseline", help="add a model to a saved scoreboard comparison, on the "
                                         "same test records (like inat-baseline)")
    q.add_argument("--comparison", required=True, help="comparison id from mv scoreboard")
    q.add_argument("--model", help=model_help)
    q.add_argument("--batch-size", type=int, default=16)
    q = xsub.add_parser("report", help="(read-only) the protocol tables: coverage, genus and "
                                       "family on all, formal species, same vocabulary")
    q.add_argument("--name", required=True)
    q.add_argument("--split", default="dev", choices=["dev", "test"])
    q.add_argument("--model", help=model_help)
    q.add_argument("--results", nargs="*", help="mv external predict JSONL files (default: "
                                                "only answers imported to the manifest)")
    q.add_argument("--vision", default="bioclip-2-ft-20261007-165400",
                   help="the Vision backbone whose stored answers are compared")
    q.add_argument("--no-reference", action="store_true",
                   help="skip coverage of Vision's North American records")
    q.add_argument("--no-crosswalk", action="store_true",
                   help="exact names only (default: also with the GBIF crosswalk, from its "
                        "cache)")
    q.add_argument("--out", help="where to write the JSON")
    from . import occtune
    occtune.add_commands(sub)       # build-occurrence, tune-occurrence, ...

    args = parser.parse_args(argv)
    if args.command == "dataset":             # never opens the manifest for writing
        cmd_dataset(args)
        return 0
    if args.command == "aws-dataset-build-job":   # the build reads a private copy only
        cmd_aws_dataset_build_job(args)
        return 0
    if args.command == "expert-ids":          # reads a release and iNat; no manifest
        cmd_expert_ids(args)
        return 0
    if args.command == "paper-inat-cv":
        cmd_paper_inat_cv(args)
        return 0
    if args.command == "aws-launch-dhash-index":  # reads the release's kept copy, no manifest
        cmd_aws_launch_dhash_index(args)
        return 0
    if args.command == "experiments-check":   # the private registry only: no manifest
        return cmd_experiments_check(args)
    if args.command == "pull-release":        # before any release exists: no manifest yet
        cmd_pull_release(None, args)
        return 0
    if config.RELEASE_ROOT and not config.DATA_DIR.is_dir():
        parser.error(f"no release in {config.RELEASE_ROOT} yet: run `mv pull-release` first")
    read_only = {"name-spellings": cmd_name_spellings, "fetch-taxonomy": cmd_fetch_taxonomy,
                 "taxonomy": cmd_taxonomy, "guests": cmd_guests}
    if args.command == "external" and args.action in ("coverage", "crosswalk", "predict", "run",
                                                       "report"):
        read_only["external"] = cmd_external  # these only read the manifest
    if args.command in read_only:             # the manifest is opened as it is, read-only
        conn = sqlite3.connect(config.MANIFEST_PATH.resolve().as_uri() + "?mode=ro", uri=True)
        read_only[args.command](conn, args)
        return 0
    from . import nightly
    # On the server box with the nightly update on, its layer copy is the live manifest.
    conn = manifest.connect(nightly.served_manifest())
    handler = getattr(args, "occ_handler", None) or {
        "export-records": cmd_export_records,
        "fetch-inat": cmd_fetch_inat,
        "record-sources": cmd_record_sources,
        "fetch-mo": cmd_fetch_mo,
        "mo-image-list": cmd_mo_image_list,
        "import-mo-zip": cmd_import_mo_zip,
        "download-photos": cmd_download_photos,
        "copy-photos": cmd_copy_photos,
        "adopt-benchmark-photos": cmd_adopt_benchmark_photos,
        "aws-launch-downloader": cmd_aws_launch_downloader,
        "aws-pull-manifest": cmd_aws_pull_manifest,
        "aws-policies": cmd_aws_policies,
        "aws-backup": cmd_aws_backup,
        "aws-launch-trainer": cmd_aws_launch_trainer,
        "aws-train-job": cmd_aws_train_job,
        "standard-summary": cmd_standard_summary,
        "fetch-mo-points": cmd_fetch_mo_points,
        "fetch-place-fallback": cmd_fetch_place_fallback,
        "int8-check": cmd_int8_check,
        "export-onnx": cmd_export_onnx,
        "aws-launch-dataset-build": cmd_aws_launch_dataset_build,
        "manifest-freeze": cmd_manifest_freeze,
        "model-sanity": cmd_model_sanity,
        "aws-pull-trainer": cmd_aws_pull_trainer,
        "finetune": cmd_finetune,
        "picek-train": cmd_picek_train,
        "picek-bench": cmd_picek_bench,
        "embed": cmd_embed,
        "archive-embeddings": cmd_archive_embeddings,
        "compare": cmd_compare,
        "screen": cmd_screen,
        "inat-baseline": cmd_inat_baseline,
        "refresh": cmd_refresh,
        "candidates": cmd_candidates,
        "predict-pending": cmd_predict_pending,
        "prospective": cmd_prospective,
        "nightly": cmd_nightly,
        "scoreboard": cmd_scoreboard,
        "scoreboard-export": cmd_scoreboard_export,
        "calibrate-sets": cmd_calibrate_sets,
        "scoreboard-import": cmd_scoreboard_import,
        "benchmark-export": cmd_benchmark_export,
        "benchmark-import": cmd_benchmark_import,
        "models": cmd_models,
        "fungus-gate": cmd_fungus_gate,
        "serve": cmd_serve,
        "status": cmd_status,
        "release": cmd_release,
        "permissions": cmd_permissions,
        "refresh-licenses": cmd_refresh_licenses,
        "contributors": cmd_contributors,
        "holdout": cmd_holdout,
        "heldout": cmd_heldout,
        "external": cmd_external,
    }[args.command]
    handler(conn, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
