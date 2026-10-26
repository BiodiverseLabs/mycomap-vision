"""Check a base model's vector folder before it is copied into a release (excluded_looked.py).

    python scripts/check_base_vectors.py --manifest <release>/manifest.sqlite --folder DIR
        [--backbone bioclip-2-336] [--dim 768] [--served bioclip-2-336-ft-20261018-171420]

The Research > Excluded photos page offers "base" maps only when <release>/embeddings/<backbone>/
holds every shard the release's manifest names for that backbone. This checks a candidate
folder against that manifest, row by row, and loads it the way the server does
(serving.map_embeddings), so it is known to load before anything is copied. The manifest is
opened read-only; nothing is written. With --served, it also says how many of the served
model's photos the base lacks (records the base index then simply doesn't have).
Exit 0 when the folder will load, 1 when it won't (the problems are printed).
"""

from __future__ import annotations

import sys

sys.modules["torch"] = None      # noqa: E402

import argparse
import json
import sqlite3
from pathlib import Path

from mycomap_vision import excluded_looked as xl


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--manifest", required=True, help="the release's manifest.sqlite")
    ap.add_argument("--folder", required=True, help="the candidate embeddings/<backbone>/ folder")
    ap.add_argument("--backbone", default="bioclip-2-336")
    ap.add_argument("--dim", type=int, default=None, help="the model's vector width (768)")
    ap.add_argument("--served", default=None, help="the served backbone, to compare coverage")
    args = ap.parse_args(argv)
    conn = sqlite3.connect(f"file:{Path(args.manifest).as_posix()}?mode=ro", uri=True)
    report = xl.check_vectors(conn, args.backbone, Path(args.folder), dim=args.dim)
    if args.served:
        missing = conn.execute(
            "select count(*) from embeddings s where s.backbone = ? and not exists (select 1 "
            "from embeddings b where b.backbone = ? and b.photo_id = s.photo_id)",
            (args.served, args.backbone)).fetchone()[0]
        report["served_photos_without_base_vectors"] = missing
    print(json.dumps({"backbone": args.backbone, "folder": str(args.folder), **report}, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
