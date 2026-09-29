"""Paths, settings and identifiers shared by every command.

Deployment-specific names (where records come from, the S3 bucket, the AWS
profile and region) are settings, not code: they come from environment
variables, or from a git-ignored `.env` file at the repo root. See `.env.example`.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path) -> None:
    """KEY=value lines into the environment; real environment variables win."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(REPO_ROOT / ".env")


def setting(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name, "").strip()
    return value or default


def required(name: str, purpose: str) -> str:
    value = setting(name)
    if not value:
        raise RuntimeError(f"Set {name} ({purpose}) in the environment or in .env; "
                           "see .env.example.")
    return value


def release_data_dir(root: Path) -> Path:
    """On the server box: the release <root>/current.txt names (see release.py), or
    <root>/no-release before the first `mv pull-release`."""
    pointer = root / "current.txt"
    rid = pointer.read_text(encoding="utf-8").strip() if pointer.is_file() else ""
    if not rid or "/" in rid or "\\" in rid or rid.startswith("."):
        return root / "no-release"
    return root / "releases" / rid


# Everything downloaded or derived lives here and is never committed. A server box
# sets MV_RELEASE_ROOT instead and reads the current release.
RELEASE_ROOT = Path(setting("MV_RELEASE_ROOT")) if setting("MV_RELEASE_ROOT") else None
DATA_DIR = Path(setting("MV_DATA_DIR")
                or (release_data_dir(RELEASE_ROOT) if RELEASE_ROOT else REPO_ROOT / "data"))
MANIFEST_PATH = DATA_DIR / "manifest.sqlite"
RAW_DIR = DATA_DIR / "raw"
PHOTOS_DIR = DATA_DIR / "photos"
REPORTS_DIR = DATA_DIR / "reports"

# iNat asks API clients to identify themselves.
USER_AGENT = "MycoMap-Vision/0.1 (+https://mycomap.org)"


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
