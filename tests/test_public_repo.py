"""The repository is public: no tracked file may carry a secret or a private operating
detail (a key, a developer's home path, an internal ssh alias, the production config
file of mycomap.org, an AI session id). Settings belong in the git-ignored .env."""
import re
import subprocess

from mycomap_vision import config

FORBIDDEN = {
    "AWS access key": r"\b(AKIA|ASIA)[0-9A-Z]{16}\b",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY",
    "GitHub token": r"\b(ghp|gho|ghs|ghu)_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}",
    "Anthropic key": r"sk-ant-[A-Za-z0-9_-]{10,}",
    # a real person's home folder; placeholders (someone, you, user, name) are fine in fixtures
    "developer home path": r"(?:[A-Za-z]:[\\/]+|/c/|/)(?:Users|home)[\\/]+"
                           r"(?!(?:someone|you|user|name|runner)\b)[A-Za-z]",
    "mycomap.org production config": r"ecosystem\.[a-z]+\.config",
    "internal ssh alias": r"\bssh\s+mycomap-[a-z-]+",
    "AI session id": r"\blocal_[0-9a-f]{8}\b",
}
# Paths that may name a pattern: this file. Lockfiles hold hashes, not text worth scanning.
SKIP = {"tests/test_public_repo.py", "web/pnpm-lock.yaml"}


def tracked_text_files():
    out = subprocess.run(["git", "ls-files", "-z"], cwd=config.REPO_ROOT,
                         capture_output=True, check=True).stdout
    for rel in out.decode().split("\0"):
        if not rel or rel in SKIP or rel.startswith("requirements/"):
            continue
        path = config.REPO_ROOT / rel
        if not path.is_file():
            continue
        try:
            yield rel, path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue                      # images and other binaries


def test_no_tracked_file_carries_a_secret_or_private_operating_detail():
    found = []
    for rel, text in tracked_text_files():
        for what, pattern in FORBIDDEN.items():
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                found.append(f"{rel}:{line}: {what}: {m.group(0)!r}")
    assert not found, "\n".join(found)


def test_unpublished_results_stay_out_of_the_repo():
    """The experiment registry and the paper draft live in the private research checkout
    (config.RESEARCH_DIR); the site shows them to members (Steve, 2026-10-12)."""
    out = subprocess.run(["git", "ls-files", "docs/experiments", "docs/paper"],
                         cwd=config.REPO_ROOT, capture_output=True, check=True).stdout
    assert not out.decode().split(), "move these to the research checkout: " + out.decode()


def test_the_scan_sees_the_files_it_must_guard():
    files = {rel for rel, _ in tracked_text_files()}
    assert {"README.md", "CLAUDE.md", "docs/deploy.md", "docs/PLAN.md", ".env.example"} <= files
