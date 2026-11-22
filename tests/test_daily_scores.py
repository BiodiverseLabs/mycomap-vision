"""Scores since training (daily_scores.py; Steve, 2026-11-22): a newer model scored day by day on
the records first green after its training cut, with a cumulative total, a 7-day rolling value
and the new photos, behind two settings that are off by default. v1 is never scored here, and
the held records are never scored or counted. Synthetic records and hold lists only."""

import hashlib
import json
import sqlite3

import pytest

from test_api import app_with_model
from test_signin import gated_site_with_pages

from mycomap_vision import daily_scores as ds, early_answers as ea, research_public
from mycomap_vision.heldout_report import wilson
from mycomap_vision.records import build_records, save_records

CUT = "2026-11-22T08:24:03+00:00"
V2 = ds.Declared("v2", "bb2", "nearest", CUT)
PLACE = "nearest+mean+place"
SETTINGS = ("MV_DAILY_SCORES", "MV_DAILY_SCORES_PUBLIC", "MV_DAILY_SCORES_MODELS",
            "MV_FEEDBACK_HOLD_KEYS", "MV_FEEDBACK_HOLD_SHA256")


@pytest.fixture(autouse=True)
def no_settings(monkeypatch):
    """Every test starts with the settings unset (a developer's .env must not turn them on)."""
    for name in SETTINGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def store():
    return ea.connect(None)


def add(conn, recs: list[dict]):
    """Green records: {"n": number, "name", "day": "YYYY-MM-DD" (first green day),
    "moment": .org's first green moment or None, "seen": Vision's first listing or None,
    "assisted", "conflict"}. Called once a test."""
    rows = []
    for r in recs:
        y, m, d = r["day"].split("-")
        rows.append({"source": "iNaturalist", "observation_id": r["n"], "scientific_name": r["name"],
                     "genus": r["name"].split()[0], "family": "F", "continent": "North America",
                     "validation_status_1": "yes", "validation_date_1": f"{int(m)}/{int(d)}/{y}"})
    # One export: the whole green list (save_records drops what it leaves out).
    save_records(conn, build_records(rows, "2026-12-01T03:00:00+00:00"))
    for r in recs:
        conn.execute("update records set first_seen_at = ?, first_green_at = ?, vision_assisted = ?, "
                     "label_conflict = ? where observation_id = ?",
                     (r.get("seen", "2026-12-01T03:00:00+00:00"), r.get("moment"),
                      int(r.get("assisted", 0)), int(r.get("conflict", 0)), r["n"]))
    conn.commit()


def says(right: dict[str, bool], photos: int = 2, asked: list | None = None):
    """An answer function for tonight's records: right (the record's own name first, family F)
    or wrong ("Z z" first, family G); a record not in `right` has no vectors yet."""
    def answer(model, recs):
        out = {}
        for rec in recs:
            key = rec["observation_id"]
            if asked is not None:
                asked.append(key)
            if key not in right:
                continue
            sp = rec["scientific_name"] if right[key] else "Z z"
            out[key] = {"lists": {"species": [{"name": sp, "confidence": 0.7}],
                                  "genus": [{"name": sp.split()[0], "confidence": 0.8}],
                                  "family": [{"name": "F" if right[key] else "G", "confidence": 0.9}]},
                        "photos": photos, "place_used": True, "date_used": True}
        return out
    return answer


def ranked(species=(), genus=(), family=()):
    """One answer, the names in order."""
    return {"lists": {"species": [{"name": n, "confidence": 0.1} for n in species],
                      "genus": [{"name": n, "confidence": 0.1} for n in genus],
                      "family": [{"name": n, "confidence": 0.1} for n in family]},
            "photos": 1, "place_used": True, "date_used": True}


def rows(store) -> list[dict]:
    ds.ensure_schema(store)
    cur = store.execute("select * from cut_answers order by record_key")
    return [dict(zip([d[0] for d in cur.description], r)) for r in cur.fetchall()]


def hold_file(tmp_path, keys) -> tuple[str, str]:
    path = tmp_path / "hold.txt"
    path.write_bytes(("".join(f"{k}\n" for k in keys)).encode())
    return str(path), hashlib.sha256(path.read_bytes()).hexdigest()


# --- which models --------------------------------------------------------------------------

def test_v1_is_never_declared_for_scoring_so_it_is_never_scored_on_post_seal_records():
    # The rule that protects the sealed v1-test: no path here scores v1 or a model of its era.
    with pytest.raises(ds.DeclarationRefused, match="v1 is never scored"):
        ds.parse_models(f"v1=bb1/nearest@{CUT}")
    with pytest.raises(ds.DeclarationRefused, match="before the paper's seal"):
        ds.parse_models("v2=bb1/nearest@2026-08-17T00:00:00Z")            # v1's era cut
    with pytest.raises(ds.DeclarationRefused, match="trained through 2026-08-17"):
        ds.parse_models(f"v2=bb1/nearest@{CUT}", trained_through={"bb1": "2026-08-17"})
    assert ds.parse_models(f"v2=bb2/nearest@{CUT}", trained_through={"bb2": "2026-11-15"})


def test_a_later_model_joins_by_config_alone_and_a_bad_declaration_is_refused_whole():
    got = ds.parse_models("v2=bb2/nearest+mean-records-balanced+place@2026-11-22T08:24:03Z, "
                          "v2-open=bb3/nearest@2026-12-01T00:00:00Z")
    assert [(m.id, m.backbone, m.method, m.cut) for m in got] == [
        ("v2", "bb2", "nearest+mean-records-balanced+place", CUT),
        ("v2-open", "bb3", "nearest", "2026-12-01T00:00:00+00:00")]
    for bad in ("v2=bb2@" + CUT, "v2=bb2/nearest", f"v2=bb2/nearest@{CUT} v2=bb3/nearest@{CUT}",
                "v2=bb2/nearest@yesterday", f"V 2=bb2/nearest@{CUT}"):
        with pytest.raises(ds.DeclarationRefused):
            ds.parse_models(bad)
    assert ds.parse_models("") == [] and ds.parse_models(None) == []


# --- which records -------------------------------------------------------------------------

def test_only_records_first_green_after_the_cut_are_scored(conn, store):
    add(conn, [
        {"n": "1", "name": "A x", "day": "2026-11-20"},                               # before
        {"n": "2", "name": "A x", "day": "2026-11-22"},                               # cut day, no moment
        {"n": "3", "name": "A x", "day": "2026-11-22", "moment": "2026-11-22T08:24:03Z"},  # at the cut
        {"n": "4", "name": "A x", "day": "2026-11-22", "moment": "2026-11-22T09:00:00Z"},  # after
        {"n": "5", "name": "A x", "day": "2026-11-23"},                               # after
        {"n": "6", "name": "A x", "day": "2026-11-24", "seen": "2026-11-21T03:00:00+00:00"},  # listed before
    ])
    out = ds.score_pending(store, conn, [V2], says({k: True for k in "123456"}), frozenset())
    assert [r["record_key"] for r in rows(store)] == ["4", "5"]
    assert out["v2"]["scored"] == 2
    assert {r["record_key"]: r["green_day"] for r in rows(store)} == {"4": "2026-11-22", "5": "2026-11-23"}


def test_a_record_with_no_vectors_waits_and_a_first_answer_is_never_asked_again(conn, store):
    add(conn, [{"n": "11", "name": "A x", "day": "2026-11-23"},
               {"n": "12", "name": "B y", "day": "2026-11-23"}])
    first = ds.score_pending(store, conn, [V2], says({"11": True}), frozenset(), now="2026-11-24T08:00:00+00:00")
    assert first["v2"] == {"scored": 1, "waiting: no vectors yet": 1}
    # The next night 11's answer would be wrong; it is never asked again. 12 has vectors now.
    ds.score_pending(store, conn, [V2], says({"11": False, "12": False}), frozenset(),
                     now="2026-11-25T08:00:00+00:00")
    got = {r["record_key"]: (r["sp_pos"], r["decided_at"]) for r in rows(store)}
    assert got == {"11": (0, "2026-11-24T08:00:00+00:00"), "12": (None, "2026-11-25T08:00:00+00:00")}
    with pytest.raises(sqlite3.DatabaseError, match="ledger"):
        store.execute("update cut_answers set sp_pos = 0")
    with pytest.raises(sqlite3.DatabaseError, match="ledger"):
        store.execute("delete from cut_answers")


def test_vision_assisted_label_conflicts_and_non_fungi_are_left_out_as_early_answers_leaves_them(
        conn, store):
    add(conn, [{"n": "21", "name": "A x", "day": "2026-11-23", "assisted": 1},
               {"n": "22", "name": "A x", "day": "2026-11-23", "conflict": 1},
               {"n": "23", "name": "Candida albicans", "day": "2026-11-23"},
               {"n": "24", "name": "A x", "day": "2026-11-23"}])
    out = ds.score_pending(store, conn, [V2], says({k: True for k in ("21", "22", "23", "24")}),
                           frozenset())
    assert out["v2"] == {"vision_assisted": 1, "label_conflict": 1, "not_a_fungus_label": 1, "scored": 1}
    rep = ds.report(store, V2, frozenset())
    assert rep["records"] == 1
    assert rep["left_out"] == {"vision_assisted": 1, "not_a_fungus_label": 1}


# --- the numbers ---------------------------------------------------------------------------

def test_daily_cumulative_rolling_and_new_photo_figures(conn, store):
    # Nine days after the cut; day 2 has none. Right: species and genus alike.
    plan = {"2026-11-23": [True, True, False], "2026-11-24": [], "2026-11-25": [True],
            "2026-11-26": [False, False], "2026-11-27": [True], "2026-11-28": [True, True],
            "2026-11-29": [False], "2026-11-30": [True, False], "2026-12-01": [True, True, True]}
    recs, right, n = [], {}, 100
    for day, marks in plan.items():
        for ok in marks:
            n += 1
            recs.append({"n": str(n), "name": "A x", "day": day})
            right[str(n)] = ok
    add(conn, recs)
    ds.score_pending(store, conn, [V2], says(right, photos=3), frozenset(),
                     now="2026-12-01T08:00:00+00:00")
    rep = ds.report(store, V2, frozenset())
    days = {d["day"]: d for d in rep["days"]}
    assert rep["days"][0]["day"] == "2026-11-22" and rep["days"][-1]["day"] == "2026-12-01"
    assert days["2026-11-22"]["records"] == 0 and days["2026-11-22"]["species"]["top1"] is None
    assert days["2026-11-23"]["species"] == {"right": 2, "n": 3, "top1": round(2 / 3, 4),
                                            "ci95": list(wilson(2, 3))}
    assert days["2026-11-23"]["genus"]["n"] == 3 and days["2026-11-23"]["photos"] == 9
    assert days["2026-11-24"]["records"] == 0
    # 7 days ending 1 December: 25 November .. 1 December.
    window = [ok for d, marks in plan.items() if d >= "2026-11-25" for ok in marks]
    roll = days["2026-12-01"]["rolling_7_days"]
    assert roll["records"] == len(window) == 12
    assert roll["species"]["right"] == sum(window) and roll["species"]["n"] == 12
    assert rep["rolling_7_days"]["through"] == "2026-12-01"
    assert rep["rolling_7_days"]["species"] == roll["species"]
    every = [ok for marks in plan.values() for ok in marks]
    cum = rep["cumulative"]
    assert cum["records"] == rep["records"] == len(every) == 15
    assert cum["species"]["right"] == sum(every) and cum["species"]["ci95"] == list(wilson(sum(every), 15))
    assert cum["genus"]["right"] == sum(every) and cum["family"]["right"] == sum(every)
    assert cum["family"]["ci95"] == list(wilson(sum(every), 15))
    assert days["2026-11-23"]["family"]["n"] == 3 and roll["family"]["n"] == 12
    assert rep["new_photos"] == cum["photos"] == 45
    assert (cum["from"], cum["through"]) == ("2026-11-22", "2026-12-01")
    assert rep["trained_through"] == CUT and rep["context"] == "photos + place and date"
    assert rep["with_place"] == rep["with_date"] == 15
    # The day the nightly is still on may get more records: it says so.
    assert days["2026-12-01"]["complete"] is False and days["2026-11-30"]["complete"] is True


def test_genus_can_be_right_when_species_is_wrong(conn, store):
    add(conn, [{"n": "31", "name": "A x", "day": "2026-11-23"}])

    def answer(model, recs):
        return {r["observation_id"]: ranked(["A z"], ["A"], ["F"]) for r in recs}
    ds.score_pending(store, conn, [V2], answer, frozenset())
    day = ds.report(store, V2, frozenset())["days"][-1]
    assert (day["species"]["right"], day["genus"]["right"]) == (0, 1)


# --- the held records ----------------------------------------------------------------------

def test_held_records_are_never_scored_and_never_counted(conn, store, tmp_path):
    add(conn, [{"n": "41", "name": "A x", "day": "2026-11-23"},
               {"n": "42", "name": "A x", "day": "2026-11-23"}])
    path, sha = hold_file(tmp_path, ["42"])
    held = ds.held_keys(path, sha)
    asked: list = []
    out = ds.score_pending(store, conn, [V2], says({"41": True, "42": True}, asked=asked), held)
    assert asked == ["41"], "a held record is never even answered"
    assert [r["record_key"] for r in rows(store)] == ["41"]
    assert out["v2"]["held"] == 1
    # A record held after it was scored leaves every report.
    path2, sha2 = hold_file(tmp_path, ["41", "42"])
    assert ds.report(store, V2, ds.held_keys(path2, sha2))["records"] == 0


def test_a_hold_list_that_cannot_be_trusted_refuses_scoring_and_showing(conn, store, tmp_path, monkeypatch):
    path, sha = hold_file(tmp_path, ["42"])
    for bad in ((path, "0" * 64), (path, None), (str(tmp_path / "missing.txt"), sha), (None, sha)):
        with pytest.raises(ds.HoldListRefused) as e:
            ds.held_keys(*[x if x is not None else "" for x in bad])
        assert "42" not in str(e.value)
    assert ds.held_keys("", "") == frozenset()                     # none named: none held
    add(conn, [{"n": "43", "name": "A x", "day": "2026-11-23"}])
    ds.score_pending(store, conn, [V2], says({"43": True}), frozenset())
    monkeypatch.setenv("MV_FEEDBACK_HOLD_KEYS", path)
    monkeypatch.setenv("MV_FEEDBACK_HOLD_SHA256", "0" * 64)
    shown = ds.public_payload(store, [V2], show=True)
    assert shown == {"public": True, "available": ["v1"], "default": "v1", "models": {}}


# --- the settings and the endpoint ---------------------------------------------------------

def scored_site(conn, tmp_path, monkeypatch, compute=True, public=True):
    """The test server (record 106 'A x', 107 'B y', each one photo embedded by m1) with both
    records moved to first green after a v2 cut, and m1 with the place method declared as v2."""
    client = app_with_model(conn, tmp_path)
    conn.execute("update records set validated_on = '2026-11-23', first_seen_at = "
                 "'2026-11-24T08:00:00+00:00' where observation_id in ('106', '107')")
    conn.commit()
    monkeypatch.setenv("MV_DAILY_SCORES_MODELS", f"v2=m1/{PLACE}@{CUT}")
    if compute:
        monkeypatch.setenv("MV_DAILY_SCORES", "1")
    if public:
        monkeypatch.setenv("MV_DAILY_SCORES_PUBLIC", "1")
    return client


def test_with_the_settings_off_nothing_is_computed_or_served(conn, tmp_path, monkeypatch):
    client = scored_site(conn, tmp_path, monkeypatch, compute=False, public=False)
    app = client.app
    assert app.state.score_daily_scores()["off"]
    store = app.state.early_store
    with store.lock:
        assert rows(store.conn) == []
    assert client.get("/api/daily-scores").json() == {
        "public": False, "available": ["v1"], "default": "v1", "models": {}}
    # Computed but not public: still only v1 on the pages.
    monkeypatch.setenv("MV_DAILY_SCORES", "1")
    assert app.state.score_daily_scores()["v2"]["scored"] == 2
    assert client.get("/api/daily-scores").json()["available"] == ["v1"]


def test_the_nightly_step_scores_from_the_served_index_and_the_endpoint_defaults_to_v2(
        conn, tmp_path, monkeypatch):
    client = scored_site(conn, tmp_path, monkeypatch)
    out = client.app.state.score_daily_scores()
    assert out["v2"]["scored"] == 2, out
    got = client.get("/api/daily-scores").json()
    assert got["available"] == ["v1", "v2"] and got["default"] == "v2"
    v2 = got["models"]["v2"]
    assert v2["records"] == 2 and v2["new_photos"] == 2
    # Each record answered with itself left out: the other records of its species name it.
    assert v2["cumulative"]["species"]["right"] == 2
    assert v2["days"][-1] == {**v2["days"][-1], "day": "2026-11-23", "records": 2}


def test_the_endpoint_serves_aggregates_only(conn, tmp_path, monkeypatch):
    client = scored_site(conn, tmp_path, monkeypatch)
    client.app.state.score_daily_scores()
    body = client.get("/api/daily-scores").text
    for leak in ('"106"', '"107"', "alice", "Alice", "http", "record_key", "observation_id",
                 "ranks_json", "truth_", "latitude", "longitude", "A x", "B y"):
        assert leak not in body, leak
    with pytest.raises(ValueError, match="per-record"):
        research_public.daily_scores({"models": {"v2": {"days": [{"record_key": "106"}]}}})
    with pytest.raises(ValueError, match="per-record"):
        research_public.daily_scores({"models": {"v2": {"photo_url": "x"}}})


def test_a_v1_declaration_scores_nothing_on_the_server(conn, tmp_path, monkeypatch):
    client = scored_site(conn, tmp_path, monkeypatch)
    monkeypatch.setenv("MV_DAILY_SCORES_MODELS", f"v1=m1/{PLACE}@{CUT}")
    assert "v1 is never scored" in client.app.state.score_daily_scores()["error"]
    monkeypatch.setenv("MV_DAILY_SCORES_MODELS", f"v2=m1/{PLACE}@2026-10-01T00:00:00Z")
    assert "before the paper's seal" in client.app.state.score_daily_scores()["error"]
    store = client.app.state.early_store
    with store.lock:
        assert rows(store.conn) == []
    assert client.get("/api/daily-scores").json()["available"] == ["v1"]


def test_a_sha_mismatch_refuses_the_nightly_step(conn, tmp_path, monkeypatch):
    client = scored_site(conn, tmp_path, monkeypatch)
    path, _ = hold_file(tmp_path, ["999"])
    monkeypatch.setenv("MV_FEEDBACK_HOLD_KEYS", path)
    monkeypatch.setenv("MV_FEEDBACK_HOLD_SHA256", "f" * 64)
    out = client.app.state.score_daily_scores()
    assert "sha256 does not match" in out["error"] and "999" not in json.dumps(out)
    store = client.app.state.early_store
    with store.lock:
        assert rows(store.conn) == []


def test_the_feed_answers_a_signed_out_visitor(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    assert c.get("/api/daily-scores").status_code == 200


# --- measured as v1 is: public place and date (Steve, 2026-11-22) ------------------------------

def test_v2_scores_with_the_place_and_date_v1s_runs_use_and_none_of_it_leaves(
        conn, tmp_path, monkeypatch):
    # Steve, 2026-11-22: like with like. v1's runs give a test record evaluate.context_of of
    # its evaluate.load_records record: .org's coordinates (never shown), .org's date else
    # iNat's. A record with no coordinates is answered with its date alone, as v1's are.
    from mycomap_vision import evaluate, placeprior
    client = scored_site(conn, tmp_path, monkeypatch)
    conn.execute("update records set latitude = 45.5, longitude = -70.25, observed_on = '2026-11-01' "
                 "where observation_id = '106'")
    conn.execute("update records set latitude = null, longitude = null where observation_id = '107'")
    conn.commit()
    seen: list = []
    real = placeprior.PlaceBoost.log_prior

    def spy(self, ctx):
        seen.append((ctx.latitude, ctx.longitude, ctx.observed_on, ctx.uuid))
        return real(self, ctx)
    monkeypatch.setattr(placeprior.PlaceBoost, "log_prior", spy)
    out = client.app.state.score_daily_scores()
    assert out["v2"]["scored"] == 2, out
    ids = [r[0] for r in conn.execute("select photo_id from embeddings where backbone = 'm1'")]
    v1_way = {r.observation_id: evaluate.context_of(r)
              for r in evaluate.load_records(conn, {p: p for p in ids})}
    for key in ("106", "107"):
        c = v1_way[key]
        assert (c.latitude, c.longitude, c.observed_on, c.uuid) in seen, key
    assert (45.5, -70.25, "2026-11-01", "uuid-106") in seen, ".org's point and date, not iNat's public ones"
    assert (None, None, "2025-09-01", "uuid-107") in seen, "no coordinates: the date alone"
    store = client.app.state.early_store
    with store.lock:
        got = {r["record_key"]: (r["outcome"], r["place_used"], r["date_used"]) for r in rows(store.conn)}
    assert got == {"106": ("scored", 1, 1), "107": ("scored", 0, 1)}
    body = client.get("/api/daily-scores").text
    for leak in ("45.5", "-70.25", "39.1", "-86.5", "latitude", "longitude", "uuid-10", "2026-11-01"):
        assert leak not in body, leak
    v2 = json.loads(body)["models"]["v2"]
    assert v2["context"] == "photos + place and date"
    assert (v2["with_place"], v2["with_date"]) == (1, 2)
    with pytest.raises(ValueError, match="per-record"):
        research_public.daily_scores({"models": {"v2": {"days": [{"latitude": 45.5}]}}})


def test_top_k_and_family_are_judged_at_write_time_from_the_ten_deep_answer(conn, store):
    add(conn, [{"n": "61", "name": "A x", "day": "2026-11-23"},
               {"n": "62", "name": "A x", "day": "2026-11-23"},
               {"n": "63", "name": "A x", "day": "2026-11-23"}])
    answers = {"61": ranked(["A x"], ["A"], ["F"]),
               "62": ranked(["B y", "C z", "A x"], ["B", "A"], ["G", "H", "I", "F"]),
               "63": ranked(["B y", "C z", "D w", "E v", "G u", "H t", "A x"], ["B"], ["G"])}
    ds.score_pending(store, conn, [V2], lambda m, recs: answers, frozenset())
    got = {r["record_key"]: (r["sp_pos"], r["ge_pos"], r["fa_pos"]) for r in rows(store)}
    assert got == {"61": (0, 0, 0), "62": (2, 1, 3), "63": (6, None, None)}
    tk = ds.report(store, V2, frozenset())["top_k"]
    assert tk["k"] == [1, 3, 5, 10]
    assert tk["species"] == {"n": 3, "top1": round(1 / 3, 4), "top3": round(2 / 3, 4),
                             "top5": round(2 / 3, 4), "top10": 1.0}
    assert tk["genus"] == {"n": 3, "top1": round(1 / 3, 4), "top3": round(2 / 3, 4),
                           "top5": round(2 / 3, 4), "top10": round(2 / 3, 4)}
    assert tk["family"]["top1"] == round(1 / 3, 4) and tk["family"]["top5"] == round(2 / 3, 4)


def test_species_by_depth_uses_the_reference_at_the_cut_and_says_too_few(conn, store):
    # 'A x' has 25 records first green before the cut (band 20-99) and 'B y' 2 (1-4); after
    # the cut, 30 new 'N n' (no record at the cut: band 0), 5 'A x', 1 'B y'.
    recs = [{"n": str(1000 + i), "name": "A x", "day": "2026-11-01"} for i in range(25)]
    recs += [{"n": str(2000 + i), "name": "B y", "day": "2026-11-02"} for i in range(2)]
    after = [{"n": str(3000 + i), "name": "N n", "day": "2026-11-24"} for i in range(30)]
    after += [{"n": str(4000 + i), "name": "A x", "day": "2026-11-24"} for i in range(5)]
    after += [{"n": "5000", "name": "B y", "day": "2026-11-24"}]
    # Green on the cut's own day, but after the cut: a test record, not the reference then.
    after += [{"n": "5001", "name": "A x", "day": "2026-11-22", "moment": "2026-11-22T09:00:00Z"}]
    add(conn, recs + after)
    right = {r["n"]: r["name"] != "N n" for r in after}
    ds.score_pending(store, conn, [V2], says(right), frozenset())
    depth = {r["record_key"]: r["species_depth_at_cut"] for r in rows(store)}
    assert depth["3000"] == 0 and depth["4000"] == depth["5001"] == 25 and depth["5000"] == 2, \
        "records after the cut are not the reference at the cut"
    bands = {b["band"]: b for b in ds.report(store, V2, frozenset())["species_by_depth"]}
    assert list(bands) == ["0", "1-4", "5-19", "20-99", "100+"]
    assert bands["0"] == {"band": "0", "n": 30, "top1": 0.0, "top3": 0.0, "top5": 0.0, "top10": 0.0,
                          "too_few": False}
    assert bands["20-99"] == {"band": "20-99", "n": 6, "top1": None, "top3": None, "top5": None,
                              "top10": None, "too_few": True}
    assert bands["1-4"]["n"] == 1 and bands["1-4"]["too_few"] and bands["1-4"]["top1"] is None
    assert bands["100+"]["n"] == 0 and bands["100+"]["too_few"]
    assert ds.MIN_BAND_RECORDS == 30


def test_a_band_with_too_few_records_never_leaves_with_a_rate():
    good = {"models": {"v2": {"min_band_records": 30, "species_by_depth": [
        {"band": "0", "n": 29, "top1": None, "top3": None, "top5": None, "top10": None, "too_few": True},
        {"band": "1-4", "n": 30, "top1": 0.5, "top3": 0.6, "top5": 0.7, "top10": 0.8, "too_few": False}]}}}
    assert research_public.daily_scores(good) == good
    for band in ({"band": "0", "n": 29, "top1": 0.9, "too_few": False},
                 {"band": "0", "n": 29, "top1": 0.9, "too_few": True},
                 {"band": "0", "n": 3, "top1": None, "too_few": False}):
        with pytest.raises(ValueError, match="too few"):
            research_public.daily_scores({"models": {"v2": {"min_band_records": 30,
                                                            "species_by_depth": [band]}}})
    for leak in ("sp_pos", "truth_family", "species_depth_at_cut", "place_used", "uuid"):
        with pytest.raises(ValueError, match="per-record"):
            research_public.daily_scores({"models": {"v2": {"days": [{leak: 1}]}}})


def test_a_method_without_place_and_date_is_refused_not_scored_on_photos_alone(conn, tmp_path, monkeypatch):
    client = scored_site(conn, tmp_path, monkeypatch)
    monkeypatch.setenv("MV_DAILY_SCORES_MODELS", f"v2=m1/nearest@{CUT}")
    out = client.app.state.score_daily_scores()
    assert "no place and date weighting" in out["v2"]["refused"], out
    store = client.app.state.early_store
    with store.lock:
        assert rows(store.conn) == []


def test_the_nightly_fits_the_place_prior_without_tonights_records(
        conn, tmp_path, monkeypatch):
    from mycomap_vision import placeprior
    client = scored_site(conn, tmp_path, monkeypatch)
    fitted: list[set] = []
    real_fit = placeprior.PlaceBoost.fit

    def spy(self, records, species):
        fitted.append({r.observation_id for r in records})
        return real_fit(self, records, species)
    monkeypatch.setattr(placeprior.PlaceBoost, "fit", spy)
    out = client.app.state.score_daily_scores()
    assert out["v2"]["scored"] == 2, out
    tonight = fitted[-1]
    assert tonight and not tonight & {"106", "107"}, "tonight's records are not in the prior"
    assert {"100", "103"} <= tonight
    store = client.app.state.early_store
    with store.lock:
        assert {r["date_used"] for r in rows(store.conn)} == {1}
    v2 = client.get("/api/daily-scores").json()["models"]["v2"]
    assert v2["context"] == "photos + place and date"
    assert v2["cumulative"]["family"]["n"] == 2 and v2["top_k"]["species"]["n"] == 2


def test_a_left_out_record_takes_the_given_place_shift_and_a_method_without_one_refuses_it(conn, tmp_path):
    import numpy as np
    from mycomap_vision.identify import Identifier
    app_with_model(conn, tmp_path)
    plain = Identifier(conn, "m1", "nearest", tmp_path / "emb" / "m1", photo_info=False)
    v = plain.reference_vectors(np.array([0]))
    with pytest.raises(ValueError, match="no place and date shift"):
        plain.identify_vectors(v, 5, place_shift=np.zeros(len(plain.index.species)))
    place = Identifier(conn, "m1", PLACE, tmp_path / "emb" / "m1", photo_info=False)
    hidden, own = place.left_out("inat", "106")                       # an 'A x' record
    vecs = place.reference_vectors(own)
    alone = place.identify_vectors(vecs, 5, hidden=hidden)
    assert alone["ranks"]["species"][0]["name"] == "A x"
    shift = np.zeros(len(place.index.species))
    shift[place.index.species.index("B y")] = 5.0                   # the place strongly says B y
    moved = place.identify_vectors(vecs, 5, hidden=hidden, place_shift=shift)
    assert moved["ranks"]["species"][0]["name"] == "B y" and moved["prior"]["applied"]
