"""The whole leave-one-out mislabel scan as ONE re-runnable command (exp/loo-mislabel-scan).

    python scripts/loo_mislabel.py --release DIR  --out OUT [--audits DIR] [--workers 8]
    python scripts/loo_mislabel.py --manifest PATH --out OUT [--embeddings DIR] ...

--release DIR: a release folder (release.py): DIR/manifest.sqlite and
DIR/embeddings/<backbone>/. --manifest PATH: a manifest file; it is copied into OUT first
(the shared manifest changes during the day) and the embeddings come from --embeddings
(default data/embeddings/<backbone>). A release is never copied: it is immutable.

Steps (each skipped when its output is there, so a re-run resumes):
  1. prep    reference snapshot -> OUT/scan (vectors in column order, layout, meta.json
             with the reference hash, manifest sha256 and code commit)
  2. score   every reference record against all others, leave-one-out, CPU workers
  3. analyse categories (a)-(d), tags from the other lanes' lists (--audits), review
             lists and the removal lists -> OUT
  4. dev     held-out dev unchanged vs the removal lists (--bench; skip with --no-dev)

A re-run on a different manifest or release into an OUT made from another refuses: the
manifest's sha256 is checked against OUT/inputs.json. CPU only (torch never imported);
OPENBLAS_NUM_THREADS=1 is set for the workers. Reproducibility of everything run before
the Dataset release v1 freeze: exploratory-pre-freeze.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
BACKBONE = "bioclip-2-ft-20261007-165400"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(args: list[str], env: dict) -> None:
    print("+", " ".join(str(a) for a in args), flush=True)
    subprocess.run([sys.executable, "-u", *map(str, args)], env=env, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--release", type=Path, help="a release folder (immutable)")
    src.add_argument("--manifest", type=Path, help="a manifest file (copied into --out)")
    ap.add_argument("--embeddings", type=Path, help="with --manifest: the backbone's shards")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--audits", type=Path, help="the other lanes' lists (tags); default "
                    "<data>/audits")
    ap.add_argument("--bench", type=Path, help="held-out benchmark folder; default "
                    "<data>/benchmarks/heldout-2026-10-08")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--no-dev", action="store_true", help="skip the held-out dev step")
    a = ap.parse_args()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if a.release:
        manifest = (a.release / "manifest.sqlite").resolve()
        embeddings = (a.release / "embeddings" / BACKBONE).resolve()
    else:
        manifest = out / "manifest-copy.sqlite"
        if not manifest.exists():
            print(f"copying {a.manifest} -> {manifest}", flush=True)
            src_db = sqlite3.connect(f"file:{a.manifest}?mode=ro", uri=True)
            dst = sqlite3.connect(manifest)
            src_db.backup(dst)
            dst.execute("pragma journal_mode=delete")
            dst.close()
            src_db.close()
        embeddings = a.embeddings.resolve() if a.embeddings else None
    digest = sha256(manifest)
    inputs = out / "inputs.json"
    if inputs.exists():
        was = json.loads(inputs.read_text())
        if was["manifest_sha256"] != digest:
            raise SystemExit(f"{out} was made from another manifest ({was['manifest_sha256'][:12]}"
                             f" vs {digest[:12]}): use a new --out")
    else:
        inputs.write_text(json.dumps({
            "release": str(a.release) if a.release else None,
            "manifest_source": str(a.manifest) if a.manifest else None,
            "manifest": str(manifest), "manifest_sha256": digest,
            "embeddings": str(embeddings) if embeddings else None}, indent=2))
    env = {**os.environ, "OPENBLAS_NUM_THREADS": "1", "MV_MANIFEST_PATH": str(manifest),
           "PYTHONPATH": str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")}
    data_dir = Path(env.get("MV_DATA_DIR") or (HERE.parent / "data"))
    audits = a.audits or data_dir / "audits"
    scan = out / "scan"
    if not (scan / "meta.json").exists():
        run([HERE / "loo_scan.py", "prep", "--out", scan,
             *(["--embeddings", embeddings] if embeddings else [])], env)
    run([HERE / "loo_scan.py", "score", "--out", scan, "--workers", a.workers], env)
    run([HERE / "loo_analyse.py", "--scan", scan, "--audits", audits, "--out", out], env)
    if not a.no_dev:
        removes = sorted(out.glob("remove-*.txt"))
        run([HERE / "loo_dev_effect.py", "--scan", scan, "--out", out / "dev-effect",
             *(["--bench", a.bench] if a.bench else []),
             *[x for p in removes for x in ("--remove", f"{p.stem[7:]}={p}")]], env)
    meta = json.loads((scan / "meta.json").read_text())
    print(json.dumps({"out": str(out), "reference_hash": meta["reference_hash"],
                      "manifest_sha256": digest[:16],
                      "code_version_prep": meta.get("code_version_prep"),
                      "reproducibility": "exploratory-pre-freeze"}, indent=2))


if __name__ == "__main__":
    main()
