"""`mv` command line."""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys

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
                                        dry_run=args.dry_run, ec2_check=args.ec2_check),
                     indent=2))


# What a resumed run takes from the run itself, not from the command line.
RUN_OWN_OPTIONS = ("backbones", "methods", "size", "test_days", "finetune", "sample_records",
                   "picek", "picek_exclude_benchmarks")


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
                          resume=trainer.RunFiles(store.client, store.bucket)
                          if args.resume else None)
    if watcher is not None:
        watcher.stop()
    print(json.dumps(out, indent=2))
    if out["comparison_id"]:
        from . import evaluate
        print_scoreboard(evaluate.scoreboard(conn, out["comparison_id"]))


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
                             name, cfg, test_days=args.test_days)
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
    result = evaluate.compare(conn, backbones, methods, test_days=args.test_days,
                              max_test=args.max_test, name_scores=args.name_scores,
                              sets=not args.no_sets, per_image=args.per_image)
    config.ensure_dirs()
    path = config.REPORTS_DIR / f"compare-{result['comparison_id']}.json"
    path.write_text(evaluate.format_report(result), encoding="utf-8")
    print_scoreboard(evaluate.scoreboard(conn, result["comparison_id"]))
    print(f"-> {path}")


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


def cmd_serve(conn, args) -> None:
    from .api import serve
    conn.close()
    serve(args.host, args.port)


def cmd_release(conn, args) -> None:
    from . import release
    print(json.dumps(release.publish(conn, _split(args.backbones), label=args.label,
                                     make_current=args.make_current), indent=2))


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
                              scores_out=Path(args.scores_out) if args.scores_out else None)
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
                                                 args.backbone, redo=args.redo), indent=2))
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
                            exclude_benchmarks=args.exclude_benchmarks)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mv", description="MycoMap Vision data tools")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("export-records", help="pull green records from mycomap.org (read-only)")

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
    p.add_argument("--instance-type", default="g6.2xlarge")
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
                        "both train on the same records; all = every benchmark's records")
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

    p = sub.add_parser("screen", help="embed candidate backbones one after another (skipping "
                                      "any that fail), then compare them with the baselines")
    p.add_argument("--candidates", required=True, help="comma-separated aliases or specs")
    p.add_argument("--baselines", default="", help="already-embedded backbones to include")
    p.add_argument("--methods", default="nearest,species-mean")
    p.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--batch-size", type=int, default=16)

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
    sub.add_parser("models", help="known backbones, photos embedded, methods")

    p = sub.add_parser("serve", help="run the API for the frontend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)

    p = sub.add_parser("release", help="publish what the site serves (manifest, embeddings, "
                                       "fine-tuned weights) to s3://<bucket>/releases/<id>/")
    p.add_argument("--backbones", required=True, help="comma list of backbones to serve")
    p.add_argument("--label", help="appended to the release id, e.g. bioclip2-full")
    p.add_argument("--make-current", action="store_true",
                   help="also point releases/current.json at it (the box pulls that one)")

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

    p = sub.add_parser("external", help="published fungi classifiers (DF20, FungiTastic) as "
                                        "outside baselines (replications/fungitastic/)")
    xsub = p.add_subparsers(dest="action", required=True)
    model_help = ("comma list of models (short name, external:<name> or BVRA/<repo>); "
                  "default: all")
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
    if args.command == "pull-release":        # before any release exists: no manifest yet
        cmd_pull_release(None, args)
        return 0
    if config.RELEASE_ROOT and not config.DATA_DIR.is_dir():
        parser.error(f"no release in {config.RELEASE_ROOT} yet: run `mv pull-release` first")
    read_only = {"name-spellings": cmd_name_spellings, "fetch-taxonomy": cmd_fetch_taxonomy,
                 "taxonomy": cmd_taxonomy, "guests": cmd_guests}
    if args.command == "external" and args.action in ("coverage", "crosswalk", "predict",
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
        "download-photos": cmd_download_photos,
        "copy-photos": cmd_copy_photos,
        "aws-launch-downloader": cmd_aws_launch_downloader,
        "aws-pull-manifest": cmd_aws_pull_manifest,
        "aws-policies": cmd_aws_policies,
        "aws-backup": cmd_aws_backup,
        "aws-launch-trainer": cmd_aws_launch_trainer,
        "aws-train-job": cmd_aws_train_job,
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
        "models": cmd_models,
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
