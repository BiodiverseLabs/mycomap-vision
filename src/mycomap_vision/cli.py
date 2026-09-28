"""`mv` command line."""

from __future__ import annotations

import argparse
import csv
import json
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
                                random_order=args.random, record_sample=args.record_sample)
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
                                           instance_type=args.instance_type), indent=2))


def cmd_aws_policies(conn, args) -> None:
    """Print the IAM policies with this deployment's bucket, region and role filled in."""
    from . import aws
    for name in ("instance-policy.template.json", "ops-policy.template.json"):
        print(f"=== {name.replace('.template', '')}")
        print(aws.render_policy(name))


def cmd_aws_pull_manifest(conn, args) -> None:
    from . import aws
    conn.close()
    print(f"Manifest replaced from S3: {aws.pull_manifest(config.MANIFEST_PATH)}")


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
    print(json.dumps(stats.__dict__, indent=2))


def cmd_compare(conn, args) -> None:
    """Score several backbones and methods on the same test and reference photos."""
    from . import evaluate, models
    backbones = [models.storage_name(b.strip()) for b in args.backbones.split(",") if b.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    result = evaluate.compare(conn, backbones, methods, test_days=args.test_days,
                              max_test=args.max_test)
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


def cmd_status(conn, args) -> None:
    print(json.dumps(status_report(conn), indent=2))


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

    p = sub.add_parser("aws-launch-downloader",
                       help="download photos on a self-terminating EC2 instance into S3")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--max-hours", type=float, default=120)
    p.add_argument("--instance-type", default="t3.small")

    sub.add_parser("aws-policies",
                   help="print the IAM policies filled in for MV_S3_BUCKET / MV_AWS_REGION")

    sub.add_parser("aws-pull-manifest",
                   help="replace the local manifest with the one the instance wrote")

    p = sub.add_parser("embed", help="one vector per photo with a frozen backbone")
    p.add_argument("--backbone", required=True,
                   help="an alias (mv models) or timm:<name> / open_clip:<name>")
    p.add_argument("--size", default="large", choices=["small", "medium", "large"])
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--all-regions", action="store_true")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=32)

    p = sub.add_parser("compare", help="score backbones x methods on the same photos "
                                       "(newest weeks vs older records); saves to the scoreboard")
    p.add_argument("--backbones", required=True, help="comma-separated aliases or specs")
    p.add_argument("--methods", default="nearest", help="comma-separated: nearest, species-mean")
    p.add_argument("--test-days", type=int, default=28)
    p.add_argument("--max-test", type=int, help="sample this many test records")

    p = sub.add_parser("screen", help="embed candidate backbones one after another (skipping "
                                      "any that fail), then compare them with the baselines")
    p.add_argument("--candidates", required=True, help="comma-separated aliases or specs")
    p.add_argument("--baselines", default="", help="already-embedded backbones to include")
    p.add_argument("--methods", default="nearest,species-mean")
    p.add_argument("--size", default="medium", choices=["small", "medium", "large"])
    p.add_argument("--source", help="folder or s3://bucket/prefix (default: the data folder)")
    p.add_argument("--batch-size", type=int, default=16)

    p = sub.add_parser("scoreboard", help="saved comparison results")
    p.add_argument("--comparison", help="only this comparison id")

    sub.add_parser("models", help="known backbones, photos embedded, methods")

    p = sub.add_parser("serve", help="run the API for the frontend")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8010)

    sub.add_parser("status", help="counts of records, photos and licenses")

    p = sub.add_parser("contributors", help="write the contributor list as CSV")
    p.add_argument("--arr-only", action="store_true",
                   help="only people with all-rights-reserved photos")

    args = parser.parse_args(argv)
    conn = manifest.connect(config.MANIFEST_PATH)
    handler = {
        "export-records": cmd_export_records,
        "fetch-inat": cmd_fetch_inat,
        "download-photos": cmd_download_photos,
        "aws-launch-downloader": cmd_aws_launch_downloader,
        "aws-pull-manifest": cmd_aws_pull_manifest,
        "aws-policies": cmd_aws_policies,
        "embed": cmd_embed,
        "compare": cmd_compare,
        "screen": cmd_screen,
        "scoreboard": cmd_scoreboard,
        "models": cmd_models,
        "serve": cmd_serve,
        "status": cmd_status,
        "contributors": cmd_contributors,
    }[args.command]
    handler(conn, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
