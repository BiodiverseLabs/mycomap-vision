"""The whole prior-tuning run in one command (exploratory before the Dataset release v1 freeze).

    python scripts/prior_tuning/run_all.py --manifest <manifest.sqlite> --out <private dir>
    python scripts/prior_tuning/run_all.py --release <release dir> --out <private dir>
        [--parts 3] [--confirm-test] [--limit N]

It never touches the manifest it is given: it copies it (sqlite backup) into --out and runs
on the copy. --out holds .org's true coordinates and values made from them: keep it private.
Steps: score dev and test (score_sets.py), the prior terms for dev (build_components.py, in
--parts processes, then merged), the declared grid with nested 5-fold CV by observer
(run_tuning.py dev). With --confirm-test, and only once per --out: the prior terms for test
and the one chosen setting on test. run.json records what was run on what: the manifest,
the reference hash, the code commit, the grid, and the reproducibility label.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
REPRODUCIBILITY = "exploratory-pre-freeze"


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True).stdout.strip()


def step(name: str, args: list[str], env: dict, log: Path) -> None:
    print(f"[{now()}] {name}", flush=True)
    with open(log, "w", encoding="utf-8") as f:
        rc = subprocess.run([sys.executable, "-X", "faulthandler", *args], env=env,
                            stdout=f, stderr=subprocess.STDOUT).returncode
    if rc != 0:
        raise SystemExit(f"{name} failed (exit {rc}); see {log}")


def components(split: str, copy: Path, out: Path, parts: int, env: dict) -> None:
    script = str(HERE / "build_components.py")
    procs = []
    for p in range(parts):
        log = open(out / f"components-{split}.p{p}.log", "w", encoding="utf-8")
        procs.append((subprocess.Popen([sys.executable, "-X", "faulthandler", script, str(copy),
                                        str(out), split, str(p), str(parts)], env=env,
                                       stdout=log, stderr=subprocess.STDOUT), log))
    bad = []
    for p, (proc, log) in enumerate(procs):
        if proc.wait() != 0:
            bad.append(p)
        log.close()
    if bad:
        raise SystemExit(f"components {split}: parts {bad} failed; see {out}")
    step(f"merge {split}", [script, "merge", str(out), split, str(parts)], env,
         out / f"components-{split}.merge.log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="a manifest.sqlite (copied, never written)")
    src.add_argument("--release", help="a release folder holding manifest.sqlite and embeddings/")
    ap.add_argument("--data-dir", help="data folder with embeddings/ and benchmarks/ "
                                       "(default: the manifest's folder)")
    ap.add_argument("--out", required=True, help="a PRIVATE folder for the run's files")
    ap.add_argument("--parts", type=int, default=3, help="processes for the prior terms")
    ap.add_argument("--confirm-test", action="store_true",
                    help="also score the one chosen setting on test (once per --out)")
    ap.add_argument("--limit", type=int, help="records per split (a smoke test)")
    a = ap.parse_args()
    manifest = Path(a.manifest) if a.manifest else Path(a.release) / "manifest.sqlite"
    data_dir = Path(a.data_dir) if a.data_dir else manifest.parent
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.confirm_test and (out / "tuning-test.json").exists():
        raise SystemExit(f"{out} already has a test confirmation: test is scored once")
    env = {**os.environ, "MV_DATA_DIR": str(data_dir), "CUDA_VISIBLE_DEVICES": "-1",
           "PYTHONPATH": str(REPO / "src"), "OPENBLAS_NUM_THREADS": "4"}
    copy = out / "manifest-copy.sqlite"
    run = {"reproducibility": REPRODUCIBILITY, "started_at": now(),
           "manifest": str(manifest), "manifest_bytes": manifest.stat().st_size,
           "data_dir": str(data_dir), "code_commit": git("rev-parse", "HEAD"),
           "code_dirty": bool(git("status", "--porcelain")), "limit": a.limit}
    if not copy.exists():
        print(f"[{now()}] copy {manifest} -> {copy}", flush=True)
        src_db = sqlite3.connect(f"file:{manifest.as_posix()}?mode=ro", uri=True)
        dst = sqlite3.connect(copy)
        src_db.backup(dst)
        dst.close()
        src_db.close()
    sys.path.insert(0, str(REPO / "src"))
    from mycomap_vision import priortune as pt
    run["grid"] = {"settings": len(pt.settings()), "declared": pt.GRID_DECLARED,
                   "folds": pt.FOLDS, "seed": pt.SEED}
    limit = ["--limit", str(a.limit)] if a.limit else []
    if not (out / "dev-scores.npz").exists():
        step("score dev and test", [str(HERE / "score_sets.py"), str(copy), str(out), *limit],
             env, out / "score_sets.log")
    import numpy as np
    with np.load(out / "dev-scores.npz") as z:
        run["reference_hash"] = str(z["ref_hash"])
    if not (out / "dev-components-nearest_mean.npz").exists():
        components("dev", copy, out, a.parts, env)
    step("tune on dev", [str(HERE / "run_tuning.py"), str(copy), str(out), "dev"], env,
         out / "tuning-dev.txt")
    run["dev"] = {"chosen": json.loads((out / "chosen.json").read_text(encoding="utf-8"))["any"],
                  "status": "provisional, confirm on Dataset release v1"}
    if a.confirm_test:
        components("test", copy, out, a.parts, env)
        step("confirm on test (once)", [str(HERE / "run_tuning.py"), str(copy), str(out), "test"],
             env, out / "tuning-test.txt")
        run["test_confirmed_at"] = now()
    run["finished_at"] = now()
    (out / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(f"-> {out / 'run.json'}")


if __name__ == "__main__":
    main()
