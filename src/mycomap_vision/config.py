"""Paths and identifiers shared by every command."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Everything downloaded or derived lives here and is never committed.
DATA_DIR = Path(os.environ.get("MV_DATA_DIR", REPO_ROOT / "data"))
MANIFEST_PATH = DATA_DIR / "manifest.sqlite"
RAW_DIR = DATA_DIR / "raw"
PHOTOS_DIR = DATA_DIR / "photos"
REPORTS_DIR = DATA_DIR / "reports"

# iNat asks API clients to identify themselves.
USER_AGENT = "MycoMap-Vision/0.1 (+https://mycomap.org)"

# Read-only production SQL for mycomap.org (DB-enforced, 60 s statement cap).
ORG_SQL_SSH_HOST = "SQL-ROUTE-HOST"


def ensure_dirs() -> None:
    for d in (DATA_DIR, RAW_DIR, PHOTOS_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def code_version() -> str:
    """Short commit of this checkout, with -dirty when there are uncommitted changes."""
    try:
        head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
