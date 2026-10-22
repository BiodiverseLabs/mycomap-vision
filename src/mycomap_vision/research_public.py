"""What the Research section's data says to someone who is not signed in.

The Research pages and the data they load are open to anyone (Steve, 2026-10-22;
signin.research_path). Before they are, each answer is cut to what may be public:

- no file path from the machine a result was scored on (run reports name the files they
  read, e.g. the cross-post twin table, by their absolute path): every answer, everyone;
- no per-person result: a run report's breakdown by observer names people, so it is left
  out unless the caller is signed in (the breakdown by project stays);
- nothing from the paper's sealed test (v1-test: records first green after dataset
  release v1's label snapshot) before the paper is submitted: no imported sealed
  benchmark, no scoreboard run whose test records start after the snapshot, and not even
  how many advance predictions fall in it;
- no photographer permission counts (Steve, 2026-10-21: no grant or decline counts on any
  public page), for anyone: /api/stats keeps only whether the answers are fresh enough to
  show photos.

The paper draft and the experiment registry are not here: they stay members only
(signin.MEMBERS_ONLY).
"""

from __future__ import annotations

import re
from datetime import date, timedelta

# The sealed test's start (prospective.SEAL_DEFAULT; v1's label snapshot). Fixed here, not
# read from MV_PROSPECTIVE_SEAL: that setting turns off at the analysis freeze, and the
# public pages stay sealed until the paper is submitted.
SEAL = "2026-10-12T13:17:51+00:00"

# An absolute path on some machine, Windows (C:\..., C:/..., \\host\...) or POSIX (/home/...,
# not a URL's //host/...): the whole token, and its last part.
_PATH = re.compile(r"(?:(?<![\w:/.])(?:[A-Za-z]:[\\/]|\\\\)|(?<![\w:/.\\])/(?=[^\s/]))"
                   r"(?:[^\s\\/\"'<>|]+[\\/])+([^\s\\/\"'<>|]*)")


def strip_paths(value):
    """`value` with every absolute file path in a string cut to its file name."""
    if isinstance(value, str):
        return _PATH.sub(lambda m: m.group(1), value)
    if isinstance(value, dict):
        return {k: strip_paths(v) for k, v in value.items()}
    if isinstance(value, list):
        return [strip_paths(v) for v in value]
    return value


def run_report(report: dict, signed_in: bool) -> dict:
    """A scoreboard run's report (/api/scoreboard/{id}) as this caller may see it."""
    out = strip_paths(report)
    if not signed_in:
        for part in ("all_photos", "first_photo_only"):
            groups = (out.get(part) or {}).get("groups")
            if isinstance(groups, dict):
                groups.pop("observer", None)
    return out


def sealed_run(run: dict) -> bool:
    """A scoreboard run that may have test records first validated after the seal: its test
    starts after it, or (on the live data, not a frozen release) its test weeks reach it.
    A release's validation split ends at its label snapshot, so only its cutoff counts."""
    cutoff = str(run.get("cutoff") or "")[:10]
    if not cutoff or cutoff >= SEAL[:10]:
        return bool(cutoff)
    if run.get("dataset_release"):
        return False
    days = int(run.get("test_days") or 0)
    if days <= 0:
        return True                      # every record after the cutoff
    end = date.fromisoformat(cutoff) + timedelta(days=days)
    return end.isoformat() >= SEAL[:10]


def scoreboard(runs: list[dict], signed_in: bool) -> list[dict]:
    return strip_paths(runs if signed_in else [r for r in runs if not sealed_run(r)])


def benchmarks(found: list[dict], signed_in: bool) -> list[dict]:
    return strip_paths(found if signed_in else [b for b in found if not b.get("sealed")])


def prospective(models: list[dict], signed_in: bool) -> list[dict]:
    """Advance predictions per model; signed out, without the count held for the sealed test."""
    if signed_in:
        return models
    return [{k: v for k, v in m.items() if k != "sealed"} for m in models]


def permission_status(status: dict) -> dict:
    """permissions.status_report without the counts, or the last failure's own words (which
    can name hosts): whether photos may be shown, and when the answers were last read."""
    last = status.get("last_attempt")
    return {"last_good_sync": status.get("last_good_sync"),
            "fresh_for_showing": status.get("fresh_for_showing"),
            "last_attempt": ({"attempted_at": last.get("attempted_at"), "ok": last.get("ok")}
                             if last else None)}
