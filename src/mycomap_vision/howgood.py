"""How good is it: the served model's newest measured results, for How it works (Steve,
2026-10-22: the redesigned page, every number live).

The page used to pick its run from the whole scoreboard and showed top-1 only. It now also
shows how often the right name is in the top 1, 3, 5 and 10 answers, species accuracy by
how many DNA-verified records the true species has, and the same records answered from
their first photo only. Each run's report already holds those (evaluate.evaluate's
"standard" summary); this cuts them out of the one run the page shows, so the page never
loads a whole report.

The run is the one publicView.headline picks: the newest comparison that scored a served
model with the served default method (scoreboard order), with iNaturalist's vision on the
same records when that comparison has it. Never a run from the paper's sealed test
(research_public.sealed_run), whoever asks: How it works shows no sealed-test numbers.
"""
from __future__ import annotations

import sqlite3
from collections.abc import Iterable

from . import evaluate, research_public

INAT = "external:inat-cv"
RANKS_SHOWN = ("species", "genus", "family")


def pick(runs: list[dict], served: Iterable[str], method: str) -> tuple[dict | None, dict | None]:
    """(our run, iNat's run on the same records) from scoreboard rows, newest first."""
    served = set(served)
    open_runs = [r for r in runs if not research_public.sealed_run(r)]
    ours = next((r for r in open_runs if r["backbone"] in served and r["method"] == method), None)
    if ours is None:
        return None, None
    same = [r for r in open_runs if r["comparison_id"] == ours["comparison_id"]
            and r["backbone"] == INAT]
    inat = next((r for r in same if r["method"] == "combined-max"), same[0] if same else None)
    return ours, inat


def _scores(row: dict | None, ks: list[int]) -> dict | None:
    if not row or not row.get("n"):
        return None
    out = {"n": row["n"]}
    for k in ks:
        v = row.get(f"top{k}")
        out[f"top{k}"] = None if v is None else float(v)
    return out


def shape(run: dict, report: dict | None, inat: dict | None = None) -> dict:
    """What the page shows of one run: its test, top-1 per rank, the top-k ladder, species by
    reference depth and the first-photo comparison. A part the report lacks (runs scored
    before the standard summary) is None, never a made-up number."""
    report = report or {}
    std = (report.get("all_photos") or {}).get("standard") or {}
    ks = [int(k) for k in std.get("k") or []]
    top_k = None
    if ks and any(std.get(rank, {}).get("all") for rank in RANKS_SHOWN):
        top_k = {"k": ks, **{rank: _scores(std.get(rank, {}).get("all"), ks)
                             for rank in RANKS_SHOWN}}
    by_depth = None
    bands = std.get("bands") or []
    if ks and bands and std.get("species"):
        by_depth = [{"band": b, **_scores(std["species"][b], ks)}
                    for b in bands if _scores(std["species"].get(b), ks)]
    first = ((report.get("first_photo_only") or {}).get("species") or {}).get("all")
    every = ((report.get("all_photos") or {}).get("species") or {}).get("all")
    first_photo = None
    if first and every and first.get("n") and first.get("top1") is not None:
        first_photo = {"n": first["n"], "first_photo_top1": float(first["top1"]),
                       "all_photos_n": every.get("n"), "all_photos_top1": every.get("top1")}
    return {
        "run": {k: run.get(k) for k in ("id", "backbone", "method", "cutoff", "test_days",
                                        "n_test", "n_reference")},
        "top1": {rank: run.get(f"{rank}_top1") for rank in RANKS_SHOWN},
        "top_k": top_k,
        "species_by_depth": by_depth or None,
        "first_photo": first_photo,
        "inat": None if inat is None else {
            "method": inat["method"], "n_test": inat.get("n_test"),
            **{f"{rank}_top1": inat.get(f"{rank}_top1") for rank in RANKS_SHOWN}},
    }


def how_good(conn: sqlite3.Connection, served: Iterable[str], method: str) -> dict:
    """The page's answer: {"run": None} when no open run scored a served model yet."""
    ours, inat = pick(evaluate.scoreboard(conn), served, method)
    if ours is None:
        return {"run": None}
    return shape(ours, evaluate.run_report(conn, ours["id"]), inat)
