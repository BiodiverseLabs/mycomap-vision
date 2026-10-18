"""The settled rule (settled.py): the API's `confident`/`answer` and the "genus looks
settled" hint follow it, on the same case table the website's headline is tested on
(tests/data/settled_rule_cases.json, read by web/src/lib/answer.test.ts too)."""

import json
import re
from pathlib import Path

import pytest

from mycomap_vision import apiv1, settled
from mycomap_vision.identify import improvement_hints

ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads((ROOT / "tests" / "data" / "settled_rule_cases.json")
                   .read_text("utf-8"))["cases"]


def names(pairs):
    return [{"name": n, "confidence": c} for n, c in pairs]


def as_result(case: dict) -> dict:
    out = {"ranks": {rank: names(pairs) for rank, pairs in case["ranks"].items()},
           "gate": {"verdict": "fungus"}}
    if case.get("likely"):
        out["likely"] = {rank: {"names": names(pairs)} for rank, pairs in case["likely"].items()}
    return out


@pytest.mark.parametrize("case", CASES, ids=[c["why"][:50] for c in CASES])
def test_the_api_answer_is_the_headline_the_website_shows(case):
    r = apiv1.mark_confident(as_result(case))
    assert (r["answer"] or {}).get("rank") == case["headline"]
    for rank, cands in r["ranks"].items():
        # Only a top name is ever confident, and the answer's rank's top name always is.
        assert not any(c["confident"] for c in cands[1:])
        if rank == case["headline"]:
            assert cands[0]["confident"]


def test_the_screenshot_case_headlines_the_species():
    r = apiv1.mark_confident(as_result(CASES[0]))
    assert r["answer"] == {"rank": "species", "name": "Deconica montana", "confidence": 0.62}
    assert [c["confident"] for c in r["ranks"]["species"]] == [True, False, False]


def test_the_species_rule_and_the_genus_rule():
    # Steve, 2026-10-18: one rule with the app and .org collection importance.
    assert settled.settled("species", 0.62, 0.07)
    assert not settled.settled("species", 0.55, 0.21)
    assert settled.settled("species", 0.55, 0.19)
    assert not settled.settled("species", 0.55, 0.20), "the runner-up must be below 0.20"
    assert settled.settled("species", 0.72, 0.25), "0.70 is settled outright"
    assert not settled.settled("species", 0.49, 0.05)
    assert settled.settled("species", 0.5, None), "no other name known"
    assert not settled.settled("genus", 0.69, 0.01), "genus still needs 0.70"
    assert not settled.settled("genus", 0.6, 0.01) and not settled.settled("family", 0.6, None)
    assert settled.settled("genus", 0.70, 0.29) and settled.settled("family", 0.70, None)
    assert not settled.settled("order", 0.99, None), "no confidence fitted above family"
    assert not settled.settled("species", None, None), "a withheld chance is never settled"


def test_the_hint_never_calls_a_settled_species_open():
    hint = "The genus looks settled"
    screenshot = {rank: names(pairs) for rank, pairs in CASES[0]["ranks"].items()}
    assert not any(hint in h for h in improvement_hints(3, screenshot, 10))
    close = {rank: names(pairs) for rank, pairs in CASES[1]["ranks"].items()}   # 0.55 / 0.21
    assert any("Deconica montana and Deconica inquilina are close" in h
               for h in improvement_hints(3, close, 10))
    loose = {"species": names([["A a", 0.4], ["A b", 0.35]]), "genus": names([["A", 0.69]])}
    assert not any(hint in h for h in improvement_hints(3, loose, 10)), "genus at 0.69"


def test_the_models_doc_states_the_rule_and_keeps_confident_at():
    [model] = apiv1.models_doc({"name": "MycoMap Vision"}, {}, {}, {})["models"]
    assert model["confident_at"] == 0.70
    assert model["confident_rule"] == {"species": {"outright": 0.7, "min": 0.5,
                                                   "runner_up_below": 0.2},
                                       "genus": 0.7, "family": 0.7}


def test_the_website_holds_the_same_numbers():
    src = (ROOT / "web" / "src" / "lib" / "answer.ts").read_text("utf-8")

    def const(name):
        return float(re.search(rf"export const {name} = ([0-9.]+);", src).group(1))
    assert const("SETTLED") == settled.SETTLED_AT
    assert const("SPECIES_SETTLED_MIN") == settled.SPECIES_SETTLED_MIN
    assert const("SPECIES_RUNNER_UP_MAX") == settled.SPECIES_RUNNER_UP_MAX
