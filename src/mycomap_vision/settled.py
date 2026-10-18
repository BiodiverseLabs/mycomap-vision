"""When a rank's top name is settled: the one rule behind the website's answer headline,
the API's `confident` flags and `answer` (apiv1.mark_confident), and the "the genus looks
settled" hint (identify.improvement_hints). web/src/lib/answer.ts holds the same rule for
the page; tests/data/settled_rule_cases.json is the case table both sides are tested on.

- Genus and family: the top name's calibrated chance is at least SETTLED_AT (Steve's
  answer-headline rule, 2026-10-11).
- Species: the top name is at least SETTLED_AT outright, or at least SPECIES_SETTLED_MIN
  with the runner-up species strictly below SPECIES_RUNNER_UP_MAX (Steve, 2026-10-18: one
  rule with the app and .org collection importance; a species at 62% with the next at 7%
  is the answer, though under 70%). The mobile app's Vision card ("likely") and
  mycomap.org's collection importance ("sure") use the same rule.

A withheld chance (None) is never settled.
"""

from __future__ import annotations

from typing import Iterable

SETTLED_AT = 0.70             # genus and family (Steve, 2026-10-11)
# Species (Steve, 2026-10-18: one rule with the app and .org collection importance):
# SETTLED_AT outright, or SPECIES_SETTLED_MIN with the runner-up strictly below
# SPECIES_RUNNER_UP_MAX.
SPECIES_SETTLED_MIN = 0.50
SPECIES_RUNNER_UP_MAX = 0.20
SETTLED_RANKS = ("family", "genus", "species")    # coarse to fine


def settled(rank: str, top: float | None, runner_up: float | None = None) -> bool:
    """Whether a rank's top name is settled. `top`: its calibrated chance; `runner_up`: the
    best other name's at the same rank (None when no other name is known)."""
    if rank not in SETTLED_RANKS or top is None:
        return False
    if top >= SETTLED_AT:
        return True
    return (rank == "species" and top >= SPECIES_SETTLED_MIN
            and (runner_up is None or runner_up < SPECIES_RUNNER_UP_MAX))


def runner_up(top_name: str, *lists: Iterable[dict]) -> float | None:
    """The best chance of any name other than `top_name` in these candidate lists (a rank's
    `ranks` and `likely` names); None when there is none."""
    others = [c.get("confidence") for names in lists for c in (names or [])
              if c.get("name") != top_name and c.get("confidence") is not None]
    return max(others) if others else None


def rule() -> dict:
    """The rule as GET /api/v1/models states it (`confident_rule`)."""
    return {"species": {"outright": SETTLED_AT, "min": SPECIES_SETTLED_MIN,
                        "runner_up_below": SPECIES_RUNNER_UP_MAX},
            "genus": SETTLED_AT, "family": SETTLED_AT}
