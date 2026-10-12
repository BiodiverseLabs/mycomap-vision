"""A person-checked LOO review list -> the dataset release's exclusion TSV.

    python scripts/loo_exclusions.py --review REVIEWED.csv --manifest FROZEN.sqlite \
        --out loo-mislabel-2026-10-09.tsv

REVIEWED.csv: a review list (review-c-wrong-photos.csv etc.) with two columns a person
filled: decision (exclude-record | exclude-photo | keep | relabel-on-.com) and checked_by;
photo_id for exclude-photo; optional note. Writes kind, key, reason, note (UTF-8, tabs,
header), the format of docs/dataset-release.md section 4 (feat/dataset-release). Keys
are the manifest's own; any key the manifest doesn't hold stops the run (exit 1) and
nothing is written. Relabels are not exclusions: they go to .com via the coordinator.
"""

from __future__ import annotations

import argparse
import csv
import sqlite3
import sys
from pathlib import Path

from mycomap_vision import loo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    with open(a.review, encoding="utf-8", newline="") as f:
        reviewed = list(csv.DictReader(f))
    conn = sqlite3.connect(f"file:{a.manifest}?mode=ro", uri=True)
    keys = {r[0] for r in conn.execute("select observation_id from records")}
    photos = {r[0] for r in conn.execute("select photo_id from observation_photos")}
    rows, problems = loo.exclusion_rows(reviewed, keys, photos,
                                        reason_note=f"LOO mislabel scan, {a.review.name}")
    if problems:
        print("\n".join(problems), file=sys.stderr)
        raise SystemExit(f"{len(problems)} problem(s); nothing written")
    with open(a.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["kind", "key", "reason", "note"], delimiter="\t",
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} rows -> {a.out}")


if __name__ == "__main__":
    main()
