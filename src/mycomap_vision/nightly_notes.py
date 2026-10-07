"""What the nightly update's part of /api/health means, in a line or two for people.

verify.sh runs this on the health answer after every deploy:

    curl -s https://vision.mycomap.org/api/health | python -m mycomap_vision.nightly_notes

and prints "ok" or "WARN" lines; it never fails a deploy (a night can fail because
mycomap.org was briefly down, and the next one makes it good). Standard library
only, so it runs wherever verify.sh does.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

STALE_HOURS = 26          # a day and the time a night's run can take
LONG_RUN_HOURS = 3


def _when(value: str | None) -> datetime | None:
    if not value:
        return None
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return t if t.tzinfo else t.replace(tzinfo=timezone.utc)


def notes(block: dict | None, now: datetime | None = None,
          stale_hours: float = STALE_HOURS) -> list[tuple[str, str]]:
    """(level, text) lines for the health answer's "nightly" block; level is "ok" or "warn"."""
    now = now or datetime.now(timezone.utc)
    if not block:
        return [("warn", "the server does not report the nightly update (older code?)")]
    if not block.get("enabled"):
        return [("ok", "nightly update off")]
    out: list[tuple[str, str]] = []
    state, last_ok = block.get("state"), _when(block.get("last_ok_at"))
    nxt = block.get("next_run") or "unknown"
    if state == "never":
        out.append(("ok", f"nightly update on, no run yet; first at {nxt}"))
    elif state == "running":
        started = _when(block.get("last_run_at"))
        hours = (now - started).total_seconds() / 3600 if started else 0
        out.append(("warn" if hours > LONG_RUN_HOURS else "ok",
                    f"nightly update running for {hours:.1f} h"))
    elif state == "failed":
        why = "refused mycomap.org's answer" if block.get("refused") else "failed"
        out.append(("warn", f"last nightly run {why} at {block.get('last_run_at')}; "
                            "see `mv nightly` on the box"))
    else:
        c = block.get("changed") or {}
        out.append(("ok", f"last nightly run {block.get('last_run_at')}: "
                          f"{c.get('new', 0)} new, {c.get('removed', 0)} removed, "
                          f"{c.get('renamed', 0)} renamed, {c.get('embedded', 0)} photos embedded; "
                          f"next {nxt}"))
    if state != "never":
        if last_ok is None:
            out.append(("warn", "no nightly run has succeeded yet"))
        elif (now - last_ok).total_seconds() > stale_hours * 3600:
            hours = (now - last_ok).total_seconds() / 3600
            out.append(("warn", f"no successful nightly run for {hours:.0f} h"))
    for backbone, size in sorted((block.get("layer") or {}).items()):
        if size.get("new_release_due"):
            out.append(("warn", f"{backbone}: the nights added {size.get('nightly_photos', 0):,} "
                                f"photos, {size.get('share', 0):.0%} of the release's: time for "
                                "a new release"))
    return out


def main() -> int:
    try:
        health = json.load(sys.stdin)
        lines = notes(health.get("nightly") if isinstance(health, dict) else None)
    except Exception as e:  # noqa: BLE001 - a report, never a failure
        lines = [("warn", f"could not read the health answer ({type(e).__name__})")]
    for level, text in lines:
        print(f"{'ok' if level == 'ok' else 'WARN':<6}{text}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
