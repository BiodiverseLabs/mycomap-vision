"""The experiment registry (Steve, 2026-10-09): every experiment written down so it can be
reviewed, in experiments/YYYY-MM-DD-<slug>.md with a YAML front matter block, in the private
research checkout (config.RESEARCH_DIR, MV_RESEARCH_DIR), never in this public repo.

The site lists them under Research > Experiments for signed-in members (signin.MEMBERS_ONLY):
they hold development numbers and unpublished results. experiments/README.md there has the
template and the rules; tests/test_experiments.py holds every entry to them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from . import config

EXPERIMENTS_DIR = config.RESEARCH_DIR / "experiments"
STATUSES = ("planned", "running", "done", "adopted", "dropped")
REQUIRED = ("title", "slug", "date", "status", "reproducibility", "question", "headline",
            "verdict", "decision")
# Reproducible on a frozen dataset release (docs/PLAN.md, "a reproducible dataset release"):
# an entry is either exploratory (before the freeze, read the live manifest) or reproduced on a
# named release, and then says exactly how to rebuild its numbers.
EXPLORATORY = "exploratory-pre-freeze"
REPRODUCED = re.compile(r"^reproduced-on-([a-z0-9][a-z0-9.-]*)$")
REPRO_FIELDS = ("dataset_release", "reference_hash", "code_commit", "reproduce_command")
# The day release v1 was cut (its MANIFEST `created`, 2026-10-12): entries dated on or after it
# must be reproduced on a release.
FREEZE_DATE: str | None = "2026-10-12"
# An exploratory entry that has been re-run on a release names its re-run (`reproduced_by`, a
# slug); load_all checks it and fills `reproduced_on` (the re-run's release), so the site shows
# the old entry as method history rather than as a provisional result.
LISTED = REQUIRED + ("branch", "commits", "benchmark", "split", "model", "methods", "related") \
    + REPRO_FIELDS + ("reproduced_by", "reproduced_on")
FILE_NAME = re.compile(r"^(\d{4}-\d{2}-\d{2})-([a-z0-9]+(?:-[a-z0-9]+)*)\.md$")
SLUG = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


class BadEntry(ValueError):
    """An experiment file that doesn't meet the registry's rules."""


@dataclass
class Experiment:
    meta: dict
    body: str
    path: Path

    @property
    def slug(self) -> str:
        return self.meta["slug"]


def parse(path: Path, freeze_date: str | None = None) -> Experiment:
    m = FILE_NAME.match(path.name)
    if not m:
        raise BadEntry(f"{path.name}: name it YYYY-MM-DD-<slug>.md")
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    if not text.startswith("---\n") or "\n---\n" not in text[4:]:
        raise BadEntry(f"{path.name}: starts with a '---' front matter block")
    head, body = text[4:].split("\n---\n", 1)
    try:
        meta = yaml.safe_load(head) or {}
    except yaml.YAMLError as e:
        raise BadEntry(f"{path.name}: front matter is not YAML ({e})") from e
    if not isinstance(meta, dict):
        raise BadEntry(f"{path.name}: front matter is not a mapping")
    missing = [k for k in REQUIRED if not str(meta.get(k) or "").strip()]
    if missing:
        raise BadEntry(f"{path.name}: missing {', '.join(missing)}")
    meta["date"] = str(meta["date"])
    if meta["date"] != m.group(1) or meta["slug"] != m.group(2):
        raise BadEntry(f"{path.name}: date and slug must match the file name")
    if meta["status"] not in STATUSES:
        raise BadEntry(f"{path.name}: status must be one of {', '.join(STATUSES)}")
    repro = str(meta["reproducibility"])
    released = REPRODUCED.match(repro)
    if repro != EXPLORATORY and not released:
        raise BadEntry(f"{path.name}: reproducibility is {EXPLORATORY} or reproduced-on-<release>")
    if released:
        missing = [k for k in REPRO_FIELDS if not str(meta.get(k) or "").strip()]
        if missing:
            raise BadEntry(f"{path.name}: reproduced on a release, so it needs "
                           f"{', '.join(missing)}")
        if str(meta["dataset_release"]) != released.group(1):
            raise BadEntry(f"{path.name}: dataset_release must be {released.group(1)}")
    freeze = freeze_date if freeze_date is not None else FREEZE_DATE
    if freeze and meta["date"] >= freeze and not released:
        raise BadEntry(f"{path.name}: dated on or after the freeze ({freeze}), so it must be "
                       "reproduced on a dataset release")
    by = meta.get("reproduced_by")
    if by is not None:
        if released:
            raise BadEntry(f"{path.name}: reproduced_by belongs on the exploratory entry that "
                           "was re-run, not on the re-run")
        if not isinstance(by, str) or not SLUG.match(by):
            raise BadEntry(f"{path.name}: reproduced_by is one slug")
    meta.pop("reproduced_on", None)               # filled by load_all, never written by hand
    related = meta.get("related") or []
    if not isinstance(related, list) or not all(isinstance(r, str) and SLUG.match(r)
                                                for r in related):
        raise BadEntry(f"{path.name}: related is a list of slugs")
    return Experiment(meta, body.strip() + "\n", path)


def load_checked(directory: Path = EXPERIMENTS_DIR, freeze_date: str | None = None
                 ) -> tuple[list[Experiment], list[tuple[str, str]]]:
    """Every entry that meets the rules, newest first, and (file name, reason) for each one
    that doesn't. Never raises on a bad entry, so one bad file can't take the site's list
    down (the server logs and counts it; `mv experiments-check` catches it before a commit)."""
    if not directory.is_dir():
        return [], []
    out: list[Experiment] = []
    problems: list[tuple[str, str]] = []
    for p in sorted(directory.glob("*.md")):
        if p.name.lower() == "readme.md":
            continue
        try:
            out.append(parse(p, freeze_date))
        except BadEntry as e:
            problems.append((p.name, str(e)))
    slugs = [e.slug for e in out]
    dup = {s for s in slugs if slugs.count(s) > 1}
    for e in [e for e in out if e.slug in dup]:
        problems.append((e.path.name, f"slug used twice: {e.slug}"))
    out = [e for e in out if e.slug not in dup]
    by_slug = {e.slug: e for e in out}
    kept = []
    for e in out:
        by = e.meta.get("reproduced_by")
        if by is not None:
            target = by_slug.get(by)
            if target is None or not REPRODUCED.match(str(target.meta["reproducibility"])):
                problems.append((e.path.name, f"{e.path.name}: reproduced_by names {by}, which "
                                              "is not an entry reproduced on a dataset release"))
                continue
            e.meta["reproduced_on"] = str(target.meta["dataset_release"])
        kept.append(e)
    return sorted(kept, key=lambda e: (e.meta["date"], e.slug), reverse=True), problems


def load_all(directory: Path = EXPERIMENTS_DIR,
             freeze_date: str | None = None) -> list[Experiment]:
    """Every entry, newest first. A bad entry raises: the tests keep the registry clean."""
    found, problems = load_checked(directory, freeze_date)
    if problems:
        raise BadEntry(problems[0][1])
    return found


def summary(e: Experiment) -> dict:
    return {k: e.meta.get(k) for k in LISTED if e.meta.get(k) is not None}
