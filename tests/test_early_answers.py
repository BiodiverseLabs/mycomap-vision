"""Answered before the DNA verdict (early_answers.py; Steve, 2026-10-23): every fresh
by-observation answer is frozen when given, the first per observation and model is scored
once the record turns green, and only if it came before the verdict."""

import json
import sqlite3
from datetime import datetime, timezone

import pytest

from test_api_v1_identify import SERVER_KEY, bearer
from test_run_links import by_link, person, site, visitor

from mycomap_vision import answercache, early_answers as ea, research_public
from mycomap_vision.identify import DEEP_RANKS
from mycomap_vision.records import (VISION_ASSISTED_FIELD, VISION_ASSISTED_METHOD, build_records,
                                    save_records)

SERVER = bearer(SERVER_KEY)
SEAL = research_public.SEAL


def ts(moment: str) -> float:
    return datetime.fromisoformat(moment.replace("Z", "+00:00")).timestamp()


def kept(app) -> list[dict]:
    store = app.state.early_store
    assert store.flush(10), "the writer did not catch up"
    with store.lock:
        cur = store.conn.execute("select * from early_answers order by id")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]


def answer(store: sqlite3.Connection, number: str, at: str, species=("A x", "B y"),
           genus=("A", "B"), model="m1/nearest", depth=10, **kw) -> int:
    lists = {"species": [{"name": n, "confidence": 0.5} for n in species],
             "genus": [{"name": n, "confidence": 0.5} for n in genus], "family": []}
    row = ea.make_row(source=kw.pop("source", "inat"), observation_id=number, model=model,
                      answered_at=ts(at), path=kw.pop("path", ea.V1_OBSERVATION), lists=lists,
                      depth=depth, asked_by=kw.pop("asked_by", "mycomap.org"),
                      place_used=kw.pop("place_used", True), **kw)
    with store:
        return store.execute(ea.INSERT, row).lastrowid


def green(conn, number: str, name: str, day: str, **extra):
    """A record green on .org since `day` (M/D/YYYY), as the feed gives it."""
    row = {"source": "iNaturalist", "observation_id": number, "scientific_name": name,
           "genus": name.split()[0], "family": "F", "continent": "North America",
           "validation_status_1": "yes", "validation_date_1": day, **extra}
    # Each export is the whole green list (save_records drops what it leaves out): the
    # test's records so far, plus this one. Vision's nightly first listed them at 03:00.
    db = next(r[2] for r in conn.execute("pragma database_list") if r[1] == "main")
    have = _GREEN.setdefault(db, {})
    have[number] = row
    save_records(conn, build_records(list(have.values()), "2026-10-22T03:00:00+00:00"))


_GREEN: dict = {}


def first_seen(conn, number: str, at: str | None):
    conn.execute("update records set first_seen_at = ? where observation_id = ?", (at, number))
    conn.commit()


def outcome(store, answer_id) -> dict:
    cur = store.execute("select * from early_outcomes where answer_id = ?", (answer_id,))
    row = cur.fetchone()
    return dict(zip([d[0] for d in cur.description], row)) if row else None


@pytest.fixture
def store():
    return ea.connect(None)


# --- freezing ------------------------------------------------------------------------------

def test_the_first_answer_is_frozen_ten_deep_and_a_rerun_is_kept_without_replacing_it(
        conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    c = visitor(app)
    first = c.post("/api/v1/identify/inat/555", json={"top_k": 1}, headers=SERVER)
    assert first.status_code == 200, first.text
    assert len(first.json()["ranks"]["species"]) == 1          # what the caller sees: top_k
    assert c.post("/api/v1/identify/inat/555", json={"top_k": 1},
                  headers=SERVER).json()["cached"] is True     # a kept answer: nothing new
    again = c.post("/api/v1/identify/inat/555", json={"top_k": 1, "rerun": True},
                   headers=SERVER)
    assert again.status_code == 200
    rows = kept(app)
    assert len(rows) == 2                                       # the rerun is kept too
    a, b = rows
    assert (a["source"], a["observation_id"], a["path"], a["asked_by"]) == (
        "inat", "555", ea.V1_OBSERVATION, "mycomap.org")
    assert a["depth"] == ea.DEPTH == DEEP_RANKS
    assert len(json.loads(a["ranks_json"])["species"]) == 2     # deeper than top_k: both names
    assert a["model"].startswith("m1/") and a["answers_version"] and a["provenance"] == ea.LIVE
    assert (a["excluded_reference"], a["left_out_photos"]) == (0, 0)
    store = app.state.early_store
    with store.lock:
        [only] = ea.first_answers(store.conn)
    assert only["id"] == a["id"] and only["answered_at"] <= b["answered_at"]
    with store.lock, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.conn.execute("update early_answers set answered_at = 0")
    with store.lock, pytest.raises(sqlite3.DatabaseError, match="append-only"):
        store.conn.execute("delete from early_answers")


def test_a_site_answer_from_a_link_is_frozen_as_the_sites_and_never_says_who(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    res = by_link(person(app, "user-7", name="Amanita Ann"), "inat", 555)
    assert res.status_code == 200, res.text
    [row] = kept(app)
    assert (row["path"], row["asked_by"], row["observation_id"]) == (
        ea.SITE_OBSERVATION, "site", "555")
    assert "user-7" not in json.dumps(row) and "Amanita Ann" not in json.dumps(row)


def test_an_observation_that_is_a_reference_record_is_frozen_marked_so(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    assert visitor(app).post("/api/v1/identify/inat/100", json={},
                             headers=SERVER).status_code == 200
    [row] = kept(app)
    assert row["excluded_reference"] == 1 and row["left_out_photos"] >= 1


def test_ranked_lists_take_the_deep_names_else_the_answer_and_say_how_deep():
    deep = [{"species": [{"name": f"A s{i}", "confidence": 0.1}] * 1 for i in range(1)}]
    assert ea.ranked_lists(deep, None, 5)[1] == ea.DEPTH
    shown = {"species": [{"name": "A x", "confidence": 0.9}], "genus": [], "family": []}
    assert ea.ranked_lists(None, shown, 5) == (
        {"species": [{"name": "A x", "confidence": 0.9}], "genus": [], "family": []}, 5)
    assert ea.ranked_lists([], {"species": []}, 5)[1] == ea.DEPTH   # no answer: nothing deeper


# --- when the record turned green -----------------------------------------------------------

def test_with_only_the_green_day_an_answer_must_come_before_that_day_began():
    at, rule = ea.green_moment("2026-10-20", None)
    assert at == datetime(2026, 10, 20, tzinfo=timezone.utc) and rule == "day"
    # Vision saw it green earlier than the day says: the earlier one.
    assert ea.green_moment("2026-10-20", "2026-10-19T07:00:00+00:00")[0] == datetime(
        2026, 10, 19, 7, tzinfo=timezone.utc)
    # .org's moment, when the feed gives it, is used as is.
    assert ea.green_moment("2026-10-20", "2026-10-21T07:00:00+00:00",
                           "2026-10-20T13:12:40Z") == (
        datetime(2026, 10, 20, 13, 12, 40, tzinfo=timezone.utc), "moment")
    assert ea.green_moment(None, None) == (None, None)


def test_a_record_is_sealed_unless_something_shows_it_green_before_the_seal():
    assert ea.sealed("2026-10-13", None, None, SEAL)
    assert ea.sealed("2026-10-12", None, None, SEAL)           # the seal's own day: held back
    assert ea.sealed(None, None, None, SEAL)                   # nothing known: held back
    assert not ea.sealed("2026-10-11", None, None, SEAL)
    assert not ea.sealed("2026-10-12", "2026-10-12T06:00:00+00:00", None, SEAL)
    assert not ea.sealed("2026-10-12", None, "2026-10-12T10:00:00Z", SEAL)
    assert ea.sealed("2026-10-12", None, "2026-10-12T14:00:00Z", SEAL)
    assert not ea.sealed("2026-10-20", None, None, None)        # the seal lifted


# --- scoring -------------------------------------------------------------------------------

def test_an_answer_given_before_the_green_day_is_scored(conn, store):
    green(conn, "900", "A x", "10/20/2026")
    aid = answer(store, "900", "2026-10-19T22:00:00Z")
    assert ea.score_pending(store, conn, seal=False) == {ea.SCORED: 1}
    o = outcome(store, aid)
    assert o["outcome"] == ea.SCORED and o["sp_strict"] == 0 and o["ge_strict"] == 0
    assert o["green_rule"] == "day" and o["truth_species"] == "A x"


def test_an_answer_given_after_the_verdict_is_never_scored(conn, store):
    green(conn, "901", "A x", "10/20/2026")
    later = answer(store, "901", "2026-10-21T10:00:00Z")
    same_day = (green(conn, "902", "A x", "10/20/2026"),
                answer(store, "902", "2026-10-20T09:00:00Z"))[1]
    assert ea.score_pending(store, conn, seal=False) == {ea.AFTER: 2}
    assert outcome(store, later)["outcome"] == ea.AFTER
    # Only the day is known: an answer that same day may have followed the verdict.
    assert outcome(store, same_day)["outcome"] == ea.AFTER


def test_with_dot_orgs_first_green_moment_an_answer_minutes_before_it_counts(conn, store):
    green(conn, "903", "A x", "10/20/2026", first_green_at="2026-10-20T13:12:40Z")
    aid = answer(store, "903", "2026-10-20T13:05:00Z")
    late = (green(conn, "904", "A x", "10/20/2026", first_green_at="2026-10-20T13:12:40Z"),
            answer(store, "904", "2026-10-20T13:12:40Z"))[1]
    ea.score_pending(store, conn, seal=False)
    assert outcome(store, aid)["outcome"] == ea.SCORED
    assert outcome(store, aid)["green_rule"] == "moment"
    assert outcome(store, late)["outcome"] == ea.AFTER


def test_only_the_first_answer_per_observation_and_model_is_scored(conn, store):
    green(conn, "905", "A x", "10/20/2026")
    first = answer(store, "905", "2026-10-18T00:00:00Z", species=("B y",), genus=("B",))
    rerun = answer(store, "905", "2026-10-19T00:00:00Z", species=("A x",))
    other_model = answer(store, "905", "2026-10-19T00:00:00Z", model="m2/nearest")
    ea.score_pending(store, conn, seal=False)
    assert outcome(store, first)["sp_strict"] is None            # its wrong answer stands
    assert outcome(store, rerun) is None
    assert outcome(store, other_model)["outcome"] == ea.SCORED


def test_an_outcome_is_decided_once_and_never_changed(conn, store):
    green(conn, "906", "A x", "10/20/2026")
    aid = answer(store, "906", "2026-10-19T00:00:00Z")
    ea.score_pending(store, conn, seal=False)
    conn.execute("update records set scientific_name = 'B y' where observation_id = '906'")
    conn.commit()
    assert ea.score_pending(store, conn, seal=False) == {}
    assert outcome(store, aid)["truth_name"] == "A x"
    with pytest.raises(sqlite3.DatabaseError, match="ledger"):
        store.execute("update early_outcomes set outcome = 'x'")


def test_a_record_still_awaiting_its_verdict_waits(conn, store):
    answer(store, "907", "2026-10-19T00:00:00Z")
    assert ea.score_pending(store, conn, seal=False) == {"awaiting_verdict": 1}
    assert store.execute("select count(*) from early_outcomes").fetchone()[0] == 0


def test_an_observation_held_in_the_reference_when_answered_is_left_out(conn, store):
    green(conn, "908", "A x", "10/20/2026")
    green(conn, "909", "A x", "10/20/2026")
    own = answer(store, "908", "2026-10-19T00:00:00Z", excluded_reference=True,
                 left_out_photos=2)
    twin = answer(store, "909", "2026-10-19T00:00:00Z", excluded_reference=False,
                  left_out_photos=1)
    assert ea.score_pending(store, conn, seal=False) == {ea.IN_REFERENCE: 1, ea.TWIN: 1}
    assert outcome(store, own)["outcome"] == ea.IN_REFERENCE
    assert outcome(store, twin)["outcome"] == ea.TWIN


def test_a_record_validated_by_dna_and_vision_is_not_scored_but_counted(conn, store):
    green(conn, "910", "A x", "10/20/2026", **{VISION_ASSISTED_FIELD: VISION_ASSISTED_METHOD})
    green(conn, "911", "A x", "10/20/2026")
    assert [tuple(r) for r in conn.execute(
        "select observation_id, vision_assisted from records where "
        "observation_id in ('910', '911') order by 1")] == [("910", 1), ("911", 0)]
    helped = answer(store, "910", "2026-10-19T00:00:00Z")
    answer(store, "911", "2026-10-19T00:00:00Z")
    assert ea.score_pending(store, conn, seal=False) == {ea.VISION_ASSISTED: 1, ea.SCORED: 1}
    assert outcome(store, helped)["outcome"] == ea.VISION_ASSISTED
    [g] = ea.report(store, conn, sealed_after=None)
    assert g["scored"] == 1 and g["left_out"][ea.VISION_ASSISTED] == 1


def test_the_vision_assisted_flag_is_read_leniently_from_the_feed():
    from mycomap_vision.records import vision_assisted
    assert vision_assisted(True) and vision_assisted("DNA + MycoMap Vision")
    assert vision_assisted(["manual", "dna + mycomap  vision"]) and vision_assisted(1)
    assert not vision_assisted(None) and not vision_assisted("manual")
    assert not vision_assisted(False) and not vision_assisted(0)


def test_names_are_matched_by_the_answer_key_and_its_sl_and_complex_readings(conn, store):
    green(conn, "912", "Cortinarius pacificus", "10/20/2026")
    green(conn, "913", "Amanita muscaria var. flavivolvata", "10/20/2026")
    green(conn, "914", "Inocybe sp. 'IN07'", "10/20/2026")
    sl = answer(store, "912", "2026-10-19T00:00:00Z",
                species=("Amanita muscaria", "Calonarius pacificus"),
                genus=("Amanita", "Calonarius"))
    variety = answer(store, "913", "2026-10-19T00:00:00Z", species=("Amanita muscaria",),
                     genus=("Amanita",))
    code = answer(store, "914", "2026-10-19T00:00:00Z", species=("Inocybe sp. 'IN7'",),
                  genus=("Inocybe",))
    ea.score_pending(store, conn, seal=False)
    o = outcome(store, sl)
    assert (o["sp_strict"], o["sp_sl"], o["sp_complex"]) == (None, 1, 1)
    assert (o["ge_strict"], o["ge_sl"]) == (None, 1)               # Cortinarius s.l.
    assert outcome(store, variety)["sp_strict"] == 0               # a variety is its species
    assert outcome(store, code)["sp_strict"] == 0                  # one code, two writings


def test_a_guest_or_non_fungal_label_is_left_out_as_advance_predictions_leave_it(conn, store,
                                                                                 monkeypatch):
    from mycomap_vision import guests
    monkeypatch.setattr(guests, "excluded", lambda genus: "hidden" if genus == "Hypomyces" else None)
    green(conn, "915", "Hypomyces lactifluorum", "10/20/2026")
    aid = answer(store, "915", "2026-10-19T00:00:00Z")
    ea.score_pending(store, conn, seal=False)
    assert outcome(store, aid)["outcome"] == ea.NOT_FUNGUS


# --- the sealed test -------------------------------------------------------------------------

def test_a_sealed_test_record_is_not_scored_while_the_seal_holds(conn, store):
    green(conn, "920", "A x", "10/20/2026")                 # first green after the seal
    aid = answer(store, "920", "2026-10-19T00:00:00Z")
    assert ea.score_pending(store, conn, seal=SEAL) == {"sealed": 1}
    assert outcome(store, aid) is None
    [g] = ea.report(store, conn, sealed_after=SEAL)
    assert g["sealed"] == 1 and g["scored"] == 0


def test_sealed_test_scores_are_never_in_the_public_response(conn, tmp_path, monkeypatch):
    # On the box the seal setting turns off at the analysis freeze; pages stay sealed.
    monkeypatch.setenv("MV_PROSPECTIVE_SEAL", "off")
    app, _, _ = site(conn, tmp_path)
    store = app.state.early_store
    green(conn, "930", "A x", "10/01/2026")                     # before the seal
    green(conn, "931", "B y", "10/20/2026")                     # v1-test
    with store.lock:
        answer(store.conn, "930", "2026-09-30T00:00:00Z")
        answer(store.conn, "931", "2026-10-19T00:00:00Z", species=("B y",), genus=("B",))
        # The seal lifted on the box (MV_PROSPECTIVE_SEAL=off, after the analysis freeze):
        # the sealed record is scored in the store...
        assert ea.score_pending(store.conn, conn, seal=False) == {ea.SCORED: 2}
    for who in ("signed out", "member"):
        client = visitor(app) if who == "signed out" else person(app, "user-1")
        body = client.get("/api/early-answers").json()
        assert body["sealed_after"] == SEAL
        [g] = body["groups"]
        # ...and still never on a page: one record scored, the sealed one held back.
        assert g["scored"] == 1, who
        assert g["top_k"]["species strict"]["n"] == 1
        assert ("sealed" in g) == (who == "member"), who
        if who == "member":
            assert g["sealed"] == 1
        text = json.dumps(body)
        assert "930" not in text and "931" not in text


def test_the_public_cut_refuses_anything_that_names_a_record():
    with pytest.raises(ValueError):
        research_public.early_answers([{"scored": 1, "observation_id": "930"}], False)
    assert research_public.early_answers([{"scored": 1, "sealed": 3}], False) == [{"scored": 1}]
    assert research_public.early_answers([{"scored": 1, "sealed": 3}], True) == [
        {"scored": 1, "sealed": 3}]


# --- the card's content ------------------------------------------------------------------------

def test_the_report_is_the_standard_summary_by_model_and_context_with_dates_and_ns(conn, store):
    green(conn, "940", "A x", "10/05/2026")
    green(conn, "941", "B y", "10/06/2026")
    green(conn, "942", "A x", "10/07/2026")
    answer(store, "940", "2026-10-01T00:00:00Z")                          # right at 1
    answer(store, "941", "2026-10-02T00:00:00Z")                          # right at 2
    answer(store, "942", "2026-10-03T00:00:00Z", depth=5, species=("B y",), genus=("B",),
           provenance=ea.FROM_ANSWERS)                                    # wrong, 5 deep
    answer(store, "943", "2026-10-04T00:00:00Z", place_used=False)        # not green yet
    ea.score_pending(store, conn, seal=False)
    groups = ea.report(store, conn, sealed_after=SEAL)
    by = {g["context"]: g for g in groups}
    g = by[ea.WITH_CONTEXT]
    assert by[ea.PHOTOS_ONLY]["awaiting_verdict"] == 1
    assert (g["backbone"], g["method"], g["answers"], g["scored"]) == ("m1", "nearest", 3, 3)
    assert g["answered"] == {"from": "2026-10-01", "through": "2026-10-03"}
    assert g["validated"] == {"from": "2026-10-05", "through": "2026-10-07"}
    sp = g["top_k"]["species strict"]
    assert sp["n"] == 3 and sp["top1"] == round(1 / 3, 4) and sp["top3"] == round(2 / 3, 4)
    assert sp["top10"] == 1.0 and sp["n_top10"] == 2           # the 5-deep answer: not in top 10
    assert set(g["top_k"]) == {"species strict", "species s.l.", "species complex (beta)",
                               "genus strict", "genus s.l."}
    assert set(g["species_by_true_species_reference_records"]) == {"0", "1-4", "5-19",
                                                                   "20-99", "100+"}
    assert g["inat_cv"] == {"status": "not run"}
    assert g["asked_by"] == {"mycomap.org": 3}
    assert set(g["left_out"]) == set(ea.OUTCOMES) - {ea.SCORED}
    research_public.early_answers(groups, False)               # nothing per record in it
    text = ea.format_report(groups)
    assert "species complex (beta)" in text and "iNat CV: not run" in text and "n top10" in text


# --- backfill ------------------------------------------------------------------------------------

def cache_with_answers(tmp_path):
    path = tmp_path / "observation-answers.sqlite"
    cache = answercache.AnswerCache(path, clock=lambda: ts("2026-10-19T12:00:00Z"))
    v1_body = {"ranks": {"species": [{"name": "A x", "confidence": 0.8}],
                         "genus": [{"name": "A", "confidence": 0.9}], "family": []},
               "observation": {"source": "inat", "id": "950"}, "excluded_reference": None,
               "left_out_photos": 0, "context_used": {"place_from": "observation"},
               "model": {"version": "r1", "answers_version": "av1"}}
    cache.put(answercache.answer_key("v1", "inat", "950", "m1", "nearest",
                                     {"top_k": 5, "place": True, "date": True}), "av1", v1_body,
              "mycomap.org")
    site_body = {"result": {"ranks": {"species": [{"name": "B y", "confidence": 0.6}]}},
                 "hidden": [3, 4], "used": {}, "observation": {"excluded_reference": False},
                 "release": "r1"}
    cache.put(answercache.answer_key("site", "mo", "951", "m1", "nearest",
                                     {"place": True, "date": True}), "av1", site_body, "site")
    return path


def test_a_backfill_is_a_dry_run_unless_applied_and_never_twice(conn, tmp_path, store):
    path = cache_with_answers(tmp_path)
    dry = ea.backfill(store, path, None)
    assert dry["new_rows"] == 2 and dry["answer_cache"] == {"rows": 2}
    assert store.execute("select count(*) from early_answers").fetchone()[0] == 0
    done = ea.backfill(store, path, None, apply=True)
    assert done["new_rows"] == 2
    rows = {r[0]: r for r in store.execute(
        "select observation_id, path, asked_by, depth, left_out_photos, provenance "
        "from early_answers")}
    assert rows["950"][1:] == (ea.V1_OBSERVATION, "mycomap.org", 5, 0, ea.FROM_ANSWERS)
    assert rows["951"][1:] == (ea.SITE_OBSERVATION, "site", 5, 2, ea.FROM_ANSWERS)
    with pytest.raises(ValueError, match="imported already"):
        ea.backfill(store, path, None, apply=True)


def test_dot_org_runs_backfill_by_their_answered_time_and_the_model_of_their_version(
        tmp_path, store):
    path = cache_with_answers(tmp_path)
    runs = [{"platform": "inat", "observation_id": "952", "answers_version": "av1",
             "result": {"answeredAt": "2026-10-20T13:00:00.000Z", "kept": False,
                        "combined": {"species": [{"name": "A x", "confidence": 0.7}]},
                        "placeUsed": True, "dateUsed": True, "excludedReference": False}},
            {"platform": "inat", "observation_id": "953", "answers_version": "unknown",
             "result": {"answeredAt": "2026-10-20T13:00:00Z", "combined": {}}},
            {"platform": "inat", "observation_id": "954", "answers_version": "av1",
             "result": {"combined": {}}},
            # Saved before .org kept answered.at: when .org had it back, no earlier.
            {"platform": "mo", "observation_id": "955", "answers_version": "av1",
             "finished_at": "2026-10-19 13:03:53.504", "result": {"combined": {}}},
            # The same answer the cache holds: kept once.
            {"platform": "inat", "observation_id": "950", "answers_version": "av1",
             "result": {"answeredAt": "2026-10-19T12:00:00Z", "kept": True,
                        "answeredBy": "mycomap.org", "combined": {}}}]
    out = ea.backfill(store, path, runs, apply=True)
    assert out["org_runs"] == {"runs": 5, "rows": 3, "model unknown": 1, "no answered time": 1,
                               "time from .org's finished_at (at or after the answer)": 1}
    assert out["new_rows"] == 4
    assert store.execute("select answered_utc from early_answers where observation_id = '955'"
                         ).fetchone() == ("2026-10-19T13:03:53Z",)
    row = store.execute("select model, answered_utc, left_out_photos, provenance from "
                        "early_answers where observation_id = '952'").fetchone()
    assert row == ("m1/nearest", "2026-10-20T13:00:00Z", None, ea.FROM_ORG_RUNS)


def test_mv_early_answers_backfill_writes_nothing_without_apply(conn, tmp_path, capsys):
    from mycomap_vision import cli
    path = cache_with_answers(tmp_path)
    conn.commit()
    target = tmp_path / "state" / "early-answers.sqlite"
    assert cli.main(["early-answers", "backfill", "--answers", str(path), "--store", str(target),
                     "--manifest", str(tmp_path / "manifest.sqlite")]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["new_rows"] == 2 and out["preview"]["first_answers"] == 2
    assert not target.exists()
