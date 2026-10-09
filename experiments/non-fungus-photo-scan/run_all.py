"""The whole scan and dev re-score as one re-runnable command (exploratory, pre-freeze).

    python run_all.py --manifest <manifest.sqlite> --org-sources <observations-source.tsv>
                      [--data-dir <data or release folder>] [--skip-rescore]

The manifest is copied first (sqlite backup from a read-only connection), and every step
reads that copy; nothing writes to the given manifest, the benchmark or any outside
service. --org-sources is a read-only export of .org `observations` (observation_id, source;
every row), e.g. from the read-only SQL route:
    ssh <ro-sql-host> "select observation_id, source from observations" > sources.tsv
Steps: zero-shot scores, reference frame, non-iNat list (by rule), review list, dev re-score.
Writes <data>/audits/non-fungus-scan/provenance.json: inputs, manifest copy and its sha256,
reference hashes, code commit, and the time of each step. CPU only.
"""
import argparse
import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--org-sources", required=True, type=Path)
    ap.add_argument("--data-dir", type=Path, help="data or release folder (default: MV_DATA_DIR "
                    "or the repo's data/)")
    ap.add_argument("--date", default="2026-10-09", help="suffix of the non-iNat list file")
    ap.add_argument("--skip-rescore", action="store_true")
    a = ap.parse_args()
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE.parents[1] / "src"), env.get("PYTHONPATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = "-1"      # an empty value doesn't reach a Windows child
    if a.data_dir:
        env["MV_DATA_DIR"] = str(a.data_dir)
    data = Path(env.get("MV_DATA_DIR") or HERE.parents[1] / "data")
    out = data / "audits" / "non-fungus-scan"
    out.mkdir(parents=True, exist_ok=True)
    copy = out / "manifest-copy.sqlite"
    env["MV_NONINAT_LIST"] = str(out / f"non-inat-reference-records-rule-{a.date}.tsv")
    src = sqlite3.connect(a.manifest.resolve().as_uri() + "?mode=ro", uri=True)
    dst = sqlite3.connect(copy)
    src.backup(dst)
    dst.close()
    src.close()
    commit = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True,
                            text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "."],
                                capture_output=True, text=True).stdout.strip())
    prov = {"started": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reproducibility": "exploratory-pre-freeze", "manifest": str(a.manifest),
            "manifest_copy": str(copy), "manifest_copy_sha256": sha256(copy),
            "org_sources": str(a.org_sources), "org_sources_sha256": sha256(a.org_sources),
            "non_inat_list": env["MV_NONINAT_LIST"], "code_commit": commit, "code_dirty": dirty, "steps": {}}
    steps = [("zero-shot", ["zs_score.py", str(copy), "bioclip-2"]),
             ("frame", ["frame.py", str(copy)]),
             ("non-iNat list", ["mislinked_list.py", str(copy), str(a.org_sources), a.date]),
             ("review list", ["review_list.py", str(copy), str(a.org_sources)])]
    if not a.skip_rescore:
        steps.append(("dev re-score", ["rescore.py", str(copy)]))
    for name, cmd in steps:
        t = time.time()
        print(f"== {name}", flush=True)
        subprocess.run([sys.executable, "-X", "faulthandler", *cmd], cwd=HERE, env=env, check=True)
        prov["steps"][name] = round(time.time() - t, 1)
    res = out / "rescore-dev.json"
    if res.exists() and not a.skip_rescore:
        prov["reference"] = json.loads(res.read_text(encoding="utf-8"))["index"]
    (out / "provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    print(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
