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
    print(json.dumps(aws.launch_trainer(conn, _split(args.backbones), _split(args.methods),
                                        size=args.size, max_hours=args.max_hours,
                                        instance_type=args.instance_type,
                                        test_days=args.test_days, finetune=finetune,
                                        sample_records=args.sample_records,
                                        allow_over_time=args.allow_over_time,
                                        allow_dirty=args.allow_dirty,
                                        allow_unpushed=args.allow_unpushed,
                                        spot=args.spot, spot_max_price=args.spot_max_price,
                                        resume=args.resume), indent=2))


# What a resumed run takes from the run itself, not from the command line.
RUN_OWN_OPTIONS = ("backbones", "methods", "size", "test_days", "finetune", "sample_records")


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
    out = trainer.run_job(conn, store, _split(args.backbones), _split(args.methods), upload,
                          args.run_id, size=args.size, test_days=args.test_days,
                          batch_size=args.batch_size, readers=args.readers,
                          finetune=_split(args.finetune), sample_records=args.sample_records,
                          should_stop=should_stop, interrupted=watcher,
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
                              max_test=args.max_test, sets=not args.no_sets)
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
                   help="comma-separated aliases or specs (default: bioclip-2)")
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
    p.add_argument("--no-sets", action="store_true",
                   help="don't fit likely sets (likely.py) into the calibration")

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

    args = parser.parse_args(argv)
    if args.command == "pull-release":        # before any release exists: no manifest yet
        cmd_pull_release(None, args)
        return 0
    if config.RELEASE_ROOT and not config.DATA_DIR.is_dir():
        parser.error(f"no release in {config.RELEASE_ROOT} yet: run `mv pull-release` first")
    read_only = {"name-spellings": cmd_name_spellings, "fetch-taxonomy": cmd_fetch_taxonomy,
                 "taxonomy": cmd_taxonomy, "guests": cmd_guests}
    if args.command in read_only:             # the manifest is opened as it is, read-only
        conn = sqlite3.connect(config.MANIFEST_PATH.resolve().as_uri() + "?mode=ro", uri=True)
        read_only[args.command](conn, args)
        return 0
    from . import nightly
    # On the server box with the nightly update on, its layer copy is the live manifest.
    conn = manifest.connect(nightly.served_manifest())
    handler = {
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
    }[args.command]
    handler(conn, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
