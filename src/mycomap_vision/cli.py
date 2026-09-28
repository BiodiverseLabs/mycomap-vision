"""`mv` command line."""

from __future__ import annotations

import argparse
import csv
import json
import sys

from . import config, inat, manifest, photos, records


def cmd_export_records(conn, args) -> None:
    print("Exporting green records from mycomap.org (read-only)...")
    print(json.dumps(records.export_records(conn), indent=2))


def cmd_fetch_inat(conn, args) -> None:
    print("Fetching iNat metadata...")
    stats = inat.fetch_all(conn, refresh=args.refresh, north_america_only=not args.all_regions,
                           limit=args.limit)
    print(json.dumps(stats, indent=2))


def cmd_download_photos(conn, args) -> None:
    print(f"Downloading {args.size} photos...")
    stats = photos.download_all(conn, size=args.size, north_america_only=not args.all_regions,
                                limit=args.limit, max_hours=args.max_hours)
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
        "status": cmd_status,
        "contributors": cmd_contributors,
    }[args.command]
    handler(conn, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
