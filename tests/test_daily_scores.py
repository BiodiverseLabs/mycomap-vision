"""Scores since training (daily_scores.py; Steve, 2026-11-22): a newer model scored day by day on
the records first green after its training cut, with a cumulative total, a 7-day rolling value
and the new photos, behind settings that are off by default; and, paired with it on the same
records, the served v1 (Steve, 2026-11-24; the Paper Coordinator's ruling, research f79c7d2),
internal only. The held records are never scored or counted. Synthetic records and hold lists
only."""

import hashlib
import json
import sqlite3
from collections import Counter

import pytest

from test_api import app_with_model
from test_signin import gated_site_with_pages

from mycomap_vision import daily_scores as ds, early_answers as ea, research_public
from mycomap_vision.heldout_report import wilson
from mycomap_vision.records import build_records, save_records

CUT = "2026-11-22T08:24:03+00:00"
V2 = ds.Declared("v2", "bb2", "nearest", CUT)
PLACE = "nearest+mean+place"
V1C = ds.Companion("bb1", "nearest")                 # v1, paired with each cohort
SETTINGS = ("MV_DAILY_SCORES", "MV_DAILY_SCORES_PUBLIC", "MV_DAILY_SCORES_MODELS",
            "MV_DAILY_SCORES_V1", "MV_DAILY_SCORES_PUBLIC_V1", "MV_TWINS",
            "MV_FEEDBACK_HOLD_KEYS", "MV_FEEDBACK_HOLD_SHA256")


def route(method="manual", roles=(), called=False, rules=None) -> str:
    """A validation route as records.validation_route stores it from .org's feed."""
    return json.dumps({"method": method, "rules_version": rules,
                       "vision_roles": None if roles is None else list(roles),
                       "vision_called": called}, sort_keys=True)


MANUAL = route()                                     # a person, Vision never called: primary


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
    "assisted", "held", "route" (validation_route JSON), "conflict"}. Called once a test."""
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
                     "held_by_vision = ?, validation_route = ?, label_conflict = ? "
                     "where observation_id = ?",
                     (r.get("seen", "2026-12-01T03:00:00+00:00"), r.get("moment"),
                      int(r.get("assisted", 0)), int(r.get("held", 0)), r.get("route"),
                      int(r.get("conflict", 0)), r["n"]))
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


def rows(store, model_id="v2") -> list[dict]:
    """The ledger's rows answered by `model_id` (None: every model's)."""
    ds.ensure_schema(store)
    cur = store.execute("select * from cut_answers where ? is null or model_id = ? "
                        "order by record_key, model_id", (model_id, model_id))
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


def test_a_model_cut_before_the_rulings_start_is_refused():
    # The ruling covers records first validated after 2026-11-22T08:24:03Z only.
    with pytest.raises(ds.DeclarationRefused, match="covers only records first validated after"):
        ds.parse_models("v2=bb2/nearest@2026-11-22T08:24:02Z")
    assert ds.PAIRED_FROM == CUT and ds.parse_models(f"v2=bb2/nearest@{CUT}")
    # v1 is only ever the companion, never its own declared model or a declared model's twin.
    assert ds.parse_companion("bb1/nearest+place") == ds.Companion("bb1", "nearest+place")
    assert ds.parse_companion("") is None and ds.parse_companion(None) is None
    with pytest.raises(ds.DeclarationRefused):
        ds.parse_companion("bb1")
    with pytest.raises(ds.DeclarationRefused, match="v2's own backbone"):
        ds.parse_companion("bb2/nearest", [V2])


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
    assert first["v2"] == {"scored": 1, "waiting: no vectors yet": 1, "groups": {"route_unknown": 1}}
    # The next night 11's answer would be wrong; it is never asked again. 12 has vectors now.
    ds.score_pending(store, conn, [V2], says({"11": False, "12": False}), frozenset(),
                     now="2026-11-25T08:00:00+00:00")
    got = {r["record_key"]: (r["sp_pos"], r["decided_at"]) for r in rows(store)}
    assert got == {"11": (0, "2026-11-24T08:00:00+00:00"), "12": (None, "2026-11-25T08:00:00+00:00")}
    with pytest.raises(sqlite3.DatabaseError, match="ledger"):
        store.execute("update cut_answers set sp_pos = 0")
    with pytest.raises(sqlite3.DatabaseError, match="ledger"):
        store.execute("delete from cut_answers")


def test_label_conflicts_wait_non_fungi_get_no_rank_and_vision_assisted_records_are_scored_apart(
        conn, store):
    # Steve, 2026-11-24: a Vision-assisted record is answered and scored too, in its own group.
    add(conn, [{"n": "21", "name": "A x", "day": "2026-11-23", "assisted": 1},
               {"n": "22", "name": "A x", "day": "2026-11-23", "conflict": 1},
               {"n": "23", "name": "Candida albicans", "day": "2026-11-23"},
               {"n": "24", "name": "A x", "day": "2026-11-23"}])
    asked: list = []
    out = ds.score_pending(store, conn, [V2], says({k: True for k in ("21", "22", "23", "24")},
                                                   asked=asked), frozenset())
    assert sorted(asked) == ["21", "23", "24"], "only the label conflict waits"
    assert out["v2"] == {"label_conflict": 1, "not_a_fungus_label": 1, "scored": 2,
                         "groups": {"vision_assisted": 1, "route_unknown": 2}}
    rep = ds.report(store, V2, frozenset())
    assert rep["records"] == 2
    assert rep["left_out"] == {"not_a_fungus_label": 1}


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

def twin_table(tmp_path, monkeypatch, groups=None):
    """The served release's twin pass (twins.npz, here by MV_TWINS): finds as {record: group}."""
    from test_link_twins import write_table
    path = write_table(tmp_path / "twins.npz", groups=groups or {})
    monkeypatch.setenv("MV_TWINS", str(path))
    return path


def scored_site(conn, tmp_path, monkeypatch, compute=True, public=True):
    """The test server (record 106 'A x', 107 'B y', each one photo embedded by m1) with both
    records moved to first green after a v2 cut, validated by a person with no Vision call
    (the primary), a twin table served, and m1 with the place method declared as v2."""
    twin_table(tmp_path, monkeypatch)
    client = app_with_model(conn, tmp_path)
    conn.execute("update records set validated_on = '2026-11-23', first_seen_at = "
                 "'2026-11-24T08:00:00+00:00', validation_route = ? "
                 "where observation_id in ('106', '107')", (MANUAL,))
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


# --- v1 and v2 paired (Steve, 2026-11-24; the Paper Coordinator's ruling, research f79c7d2) --------

V1P = V1C.paired(V2)


def every_kind_of_validation(conn):
    """One record of each way .org can have validated it, all first green after the cut."""
    add(conn, [
        {"n": "71", "name": "A x", "day": "2026-11-23"},                              # no route given
        {"n": "72", "name": "A x", "day": "2026-11-23", "assisted": 1,                 # DNA + Vision
         "route": route("auto", ["agreement_validated"], True, "1.4")},
        {"n": "73", "name": "A x", "day": "2026-11-23",                                # Vision gated
         "route": route("auto", ["penalty_lift"], True, "1.4")},
        {"n": "74", "name": "A x", "day": "2026-11-23",                                # on screen
         "route": route("manual", ["vision_shown_at_manual_validation"], True)},
        {"n": "75", "name": "A x", "day": "2026-11-23", "route": MANUAL},              # never called
        {"n": "76", "name": "A x", "day": "2026-11-23",                                # called, no part
         "route": route("auto", ["vision_checked_no_effect"], True, "1.4")},
        {"n": "77", "name": "A x", "day": "2026-11-23", "held": 1,                     # not known
         "route": route("manual", [], None)},
        {"n": "78", "name": "A x", "day": "2026-11-23",                                # roles not said
         "route": route("auto", None, True, "1.4")},
    ])
    return {"71": ds.ROUTE_UNKNOWN, "72": ds.VISION_ASSISTED, "73": ds.VISION_GATED,
            "74": ds.VISION_ASSISTED, "75": ds.PRIMARY, "76": ds.PRIMARY, "77": ds.PRIMARY,
            "78": ds.ROUTE_UNKNOWN}


def test_every_new_green_record_is_answered_by_both_models_whether_or_not_vision_was_called(
        conn, store):
    # Steve: "any record that gets green checked that has vision should be scored and any green
    # check that doesnt have vision should have it run". How .org validated it only decides
    # its group.
    groups = every_kind_of_validation(conn)
    asked: dict = {"v2": [], "v1": []}

    def answer(model, recs):
        return says({r["observation_id"]: True for r in recs}, asked=asked[model.id])(model, recs)
    out = ds.score_pending(store, conn, [V2], answer, frozenset(), companion=V1C)
    assert sorted(asked["v2"]) == sorted(asked["v1"]) == sorted(groups)
    assert out["v2"]["scored"] == len(groups)
    got = {(r["record_key"], r["model_id"]): r["route_group"] for r in rows(store, None)}
    assert got == {(k, m): g for k, g in groups.items() for m in ("v1", "v2")}
    assert out["v2"]["groups"] == dict(Counter(groups.values()))


def test_the_primary_leaves_out_vision_assisted_vision_gated_and_unknown_route_records_but_counts_them(
        conn, store):
    groups = every_kind_of_validation(conn)
    ds.score_pending(store, conn, [V2], says({k: True for k in groups}), frozenset(),
                     companion=V1C)
    primary = [k for k, g in groups.items() if g == ds.PRIMARY]
    rep = ds.report(store, V2, frozenset(), groups=(ds.PRIMARY,))
    assert rep["records"] == len(primary) == 3
    assert rep["left_out"] == {"not_a_fungus_label": 0, "vision_gated": 1, "vision_assisted": 2,
                               "route_unknown": 2}
    assert rep["groups"] == ["primary"]
    # The public figure is the primary's.
    shown = ds.public_payload(store, [V2], frozenset(), show=True, show_v1=False)
    assert shown["models"]["v2"]["records"] == 3
    # Inside: every group scored by both models, labelled, and the primary's sub-counts.
    inside = ds.internal_payload(store, [V2], V1C, frozenset())
    g = inside["cohorts"]["v2"]["groups"]
    assert {k: v["records"] for k, v in g.items()} == {"primary": 3, "vision_gated": 1,
                                                       "vision_assisted": 2, "route_unknown": 2}
    assert all(v["label"] for v in g.values())
    assert g["primary"]["no_vision_call"] == 1                          # 75
    assert g["primary"]["vision_call_not_known"] == 1                   # 77
    assert g["primary"]["held_by_vision"] == 1                          # 77
    for name, entry in g.items():
        assert entry["models"]["v2"]["records"] == entry["models"]["v1"]["records"] == entry["records"]
        assert entry["paired"]["records"] == entry["records"], name
    assert inside["cohorts"]["v2"]["primary"]["v1"]["records"] == 3


def test_v1_and_v2_answer_from_the_same_photos_or_the_record_waits(conn, store):
    add(conn, [{"n": "91", "name": "A x", "day": "2026-11-23", "route": MANUAL},
               {"n": "92", "name": "A x", "day": "2026-11-23", "route": MANUAL}])

    def answer(model, recs):
        out = says({r["observation_id"]: True for r in recs})(model, recs)
        for key, got in out.items():
            got["photos_sha"] = "same" if key == "91" or model.id == "v2" else "other"
            got["hidden_sha"] = "h"
        return out
    out = ds.score_pending(store, conn, [V2], answer, frozenset(), companion=V1C)
    assert out["v2"]["waiting: the models' photos or hidden records differ"] == 1
    assert {(r["record_key"], r["model_id"]) for r in rows(store, None)} == {("91", "v1"), ("91", "v2")}
    # v1 with no answer for 92 yet: neither model's row is written.

    def v2_only(model, recs):
        return answer(model, recs) if model.id == "v2" else {}
    out = ds.score_pending(store, conn, [V2], v2_only, frozenset(), companion=V1C)
    assert out["v2"]["waiting: no vectors yet"] == 1
    assert "92" not in {r["record_key"] for r in rows(store, None)}


def test_each_row_records_the_model_release_reference_time_first_green_route_and_commit(conn, store):
    add(conn, [{"n": "95", "name": "A x", "day": "2026-11-23", "moment": "2026-11-23T10:11:12Z",
                "route": route("auto", ["vision_checked_no_effect"], True, "1.4")}])

    def answer(model, recs):
        out = says({r["observation_id"]: True for r in recs})(model, recs)
        for got in out.values():
            got.update(release=f"rel-{model.id}", reference_state=f"state-{model.id}",
                       photos_sha="p", hidden_sha="h")
        return out
    ds.score_pending(store, conn, [V2], answer, frozenset(), companion=V1C,
                     now="2026-11-24T08:00:00+00:00", code="abc1234")
    got = {r["model_id"]: r for r in rows(store, None)}
    assert set(got) == {"v1", "v2"}
    for mid, r in got.items():
        assert (r["cohort"], r["backbone"]) == ("v2", "bb1" if mid == "v1" else "bb2")
        assert (r["release"], r["reference_state"]) == (f"rel-{mid}", f"state-{mid}")
        assert r["decided_at"] == "2026-11-24T08:00:00+00:00" and r["code_commit"] == "abc1234"
        assert r["first_green_at"] == "2026-11-23T10:11:12Z"
        assert (r["route_method"], r["route_rules_version"], r["vision_roles"], r["vision_called"]) == (
            "auto", "1.4", '["vision_checked_no_effect"]', 1)
        assert (r["route_group"], r["cut"], r["photos_sha"], r["hidden_sha"]) == ("primary", CUT, "p", "h")


def test_the_ledger_and_the_routes_it_learns_are_insert_only(conn, store):
    add(conn, [{"n": "96", "name": "A x", "day": "2026-11-23"}])
    ds.score_pending(store, conn, [V2], says({"96": True}), frozenset(), companion=V1C)
    conn.execute("update records set validation_route = ? where observation_id = '96'", (MANUAL,))
    conn.commit()
    out = ds.score_pending(store, conn, [V2], says({}), frozenset(), companion=V1C)
    assert out["v2"]["routes learned"] == 1
    for table in ("cut_answers", "cut_routes"):
        with pytest.raises(sqlite3.DatabaseError, match="ledger: no UPDATE"):
            store.execute(f"update {table} set route_group = 'primary'")
        with pytest.raises(sqlite3.DatabaseError, match="ledger: no DELETE"):
            store.execute(f"delete from {table}")


def test_a_route_learned_later_moves_the_record_into_its_group_and_never_changes_its_answer(conn, store):
    add(conn, [{"n": "97", "name": "A x", "day": "2026-11-23"}])
    ds.score_pending(store, conn, [V2], says({"97": True}), frozenset(), companion=V1C,
                     now="2026-11-24T08:00:00+00:00")
    assert ds.report(store, V2, frozenset(), groups=(ds.PRIMARY,))["records"] == 0
    before = rows(store, None)
    conn.execute("update records set validation_route = ? where observation_id = '97'", (MANUAL,))
    conn.commit()
    ds.score_pending(store, conn, [V2], says({}), frozenset(), companion=V1C,
                     now="2026-11-25T08:00:00+00:00")
    assert rows(store, None) == before, "the answers and their rows never change"
    assert ds.report(store, V2, frozenset(), groups=(ds.PRIMARY,))["records"] == 1
    assert ds.paired(store, V2, V1C, frozenset())["records"] == 1
    # Learned once: a later change in the feed adds nothing.
    conn.execute("update records set validation_route = ? where observation_id = '97'",
                 (route("auto", ["agreement_validated"], True),))
    conn.commit()
    out = ds.score_pending(store, conn, [V2], says({}), frozenset(), companion=V1C)
    assert "routes learned" not in out["v2"]
    assert ds.report(store, V2, frozenset(), groups=(ds.PRIMARY,))["records"] == 1


def test_the_ledger_refuses_records_first_validated_after_s2_fail_closed(conn, store):
    # The Paper Coordinator's ruling (d): nothing first validated after 2027-01-15T00:00:00Z
    # until the v2-test sealed run is done and the coordinator gives the word.
    assert ds.S2_STOP == "2027-01-15T00:00:00+00:00"
    add(conn, [
        {"n": "81", "name": "A x", "day": "2027-01-14", "seen": "2027-01-14T05:00:00+00:00"},
        {"n": "82", "name": "A x", "day": "2027-01-16", "moment": "2027-01-16T10:00:00Z",
         "seen": "2027-01-17T03:00:00+00:00"},
        # Its day is S2's own: nothing shows it was green by 00:00 UTC, so it is refused.
        {"n": "83", "name": "A x", "day": "2027-01-15", "seen": "2027-01-16T03:00:00+00:00"},
        # .org's moment shows it was green a second before S2.
        {"n": "84", "name": "A x", "day": "2027-01-14", "moment": "2027-01-14T23:59:59Z",
         "seen": "2027-01-15T03:00:00+00:00"},
    ])
    asked: list = []
    out = ds.score_pending(store, conn, [V2], says({k: True for k in ("81", "82", "83", "84")},
                                                   asked=asked), frozenset(), companion=V1C)
    assert sorted(asked) == ["81", "81", "84", "84"], "each model asked about 81 and 84 only"
    assert sorted({r["record_key"] for r in rows(store, None)}) == ["81", "84"]
    assert out["v2"]["refused: first green after S2 (v2-test seal)"] == 2
    assert not ds.s2_allows({}), "no sign of when: refused"
    # The ledger itself refuses such a row, whoever writes it.
    ds.ensure_schema(store)
    row = {c: None for c in ds.COLUMNS}
    row.update(cohort="v2", model_id="v2", backbone="bb2", method="nearest", cut=CUT,
               record_key="x", decided_at="2027-01-16T00:00:00+00:00", outcome="scored",
               green_at="2027-01-15T00:00:01+00:00", green_day="2027-01-15", code_commit="c",
               route_group="primary", vision_assisted=0, held_by_vision=0)
    with pytest.raises(sqlite3.DatabaseError, match="stops at S2"):
        store.execute(f"insert or ignore into cut_answers ({', '.join(ds.COLUMNS)}) values "
                      f"({', '.join('?' for _ in ds.COLUMNS)})", [row[c] for c in ds.COLUMNS])


def test_the_paired_difference_has_counts_a_paired_interval_and_mcnemar(conn, store):
    # Ten primary records: both right 4, only v2 right 4, only v1 right 1, both wrong 1.
    keys = [str(800 + i) for i in range(10)]
    add(conn, [{"n": k, "name": "A x", "day": "2026-11-23", "route": MANUAL} for k in keys])
    v2_right = dict(zip(keys, [True] * 8 + [False] * 2))
    v1_right = dict(zip(keys, [True] * 4 + [False] * 4 + [True, False]))

    def answer(model, recs):
        return says(v2_right if model.id == "v2" else v1_right)(model, recs)
    ds.score_pending(store, conn, [V2], answer, frozenset(), companion=V1C)
    got = ds.paired(store, V2, V1C, frozenset())
    sp = got["species"]
    assert got["records"] == 10 and got["against"] == "v1"
    assert (sp["n"], sp["both_right"], sp["only_cohort_right"], sp["only_v1_right"],
            sp["both_wrong"]) == (10, 4, 4, 1, 1)
    assert sp["top1"] == {"v2": 0.8, "v1": 0.5} and sp["difference"] == 0.3
    assert sp["mcnemar_p"] == 0.375                          # 2 * (1 + 5) / 2**5
    lo, hi = sp["ci95"]
    assert lo <= 0.3 <= hi and -1 <= lo < hi <= 1
    assert ds.paired_interval([True] * 5, [True] * 5) == [0.0, 0.0]
    assert ds.mcnemar_p(0, 0) is None


def test_v2_alone_is_scored_as_before_when_no_v1_is_declared(conn, store):
    add(conn, [{"n": "98", "name": "A x", "day": "2026-11-23", "route": MANUAL}])
    ds.score_pending(store, conn, [V2], says({"98": True}), frozenset())
    assert [(r["record_key"], r["model_id"]) for r in rows(store, None)] == [("98", "v2")]
    assert ds.internal_payload(store, [V2], None, frozenset())["cohorts"]["v2"]["models"] == {
        "v2": "bb2/nearest"}


# --- the feed's validation route (records.validation_route) ---------------------------------------

def test_the_feeds_validation_route_is_read_and_a_feed_without_it_leaves_the_route_unknown():
    def row(n, **extra):
        return {"source": "iNaturalist", "observation_id": n, "scientific_name": "A x",
                "genus": "A", "family": "F", "continent": "North America",
                "validation_status_1": "yes", "validation_date_1": "11/23/2026", **extra}
    recs = {r["observation_id"]: r for r in build_records([
        row("1"),
        row("2", first_green_method="manual", vision_roles=[], vision_called=False),
        row("3", first_green_method="auto", first_green_rules_version="1.4",
            vision_roles=["second_source_vision"], vision_called=True),
        row("4", first_green_method="auto", vision_called=True),       # roles not given
        row("5", first_green_method="by hand", vision_roles=[]),         # not a method
    ], "t")}
    assert recs["1"]["validation_route"] is None
    assert json.loads(recs["3"]["validation_route"]) == {
        "method": "auto", "rules_version": "1.4", "vision_roles": ["second_source_vision"],
        "vision_called": True}
    groups = {k: ds.route_of(r)["route_group"] for k, r in recs.items()}
    assert groups == {"1": "route_unknown", "2": "primary", "3": "vision_gated",
                      "4": "route_unknown", "5": "route_unknown"}
    assert ds.route_of(recs["2"])["vision_called"] == 0


# --- on the server: both models, the same records, twins hidden, v1 kept internal -----------------

def paired_site(conn, tmp_path, monkeypatch, groups=None, public=True, twins=True):
    """The test server with two models: m1 declared as v2 and m0 (another stand-in) as the v1
    paired with it, both with the place method; 106 and 107 first green after the cut,
    validated by a person with no Vision call; the release's twin table `groups`."""
    from fastapi.testclient import TestClient
    from test_api import open_limits
    from test_models_and_scoreboard import Const, seed_two_species

    from mycomap_vision.api import create_app
    from mycomap_vision.embed import embed_photos, photos_to_embed
    store_ = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    for bb in (Const("m1"), Const("m0", twist=0.4)):
        embed_photos(conn, store_, bb, photos_to_embed(conn, bb.name, "large", store_.location,
                                                       "local"), root / bb.name, log=lambda s: None)
    conn.execute("update records set validated_on = '2026-11-23', first_seen_at = "
                 "'2026-11-24T08:00:00+00:00', validation_route = ? "
                 "where observation_id in ('106', '107')", (MANUAL,))
    conn.commit()
    if twins:
        twin_table(tmp_path, monkeypatch, groups)
    monkeypatch.setenv("MV_DAILY_SCORES_MODELS", f"v2=m1/{PLACE}@{CUT}")
    monkeypatch.setenv("MV_DAILY_SCORES_V1", f"m0/{PLACE}")
    monkeypatch.setenv("MV_DAILY_SCORES", "1")
    if public:
        monkeypatch.setenv("MV_DAILY_SCORES_PUBLIC", "1")
    return TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                 backbone_loader=lambda name: Const(name),
                                 limits=open_limits(allowed_backbones={"m1", "m0"}),
                                 background=False))


def test_v1_and_v2_are_both_scored_on_the_same_records_on_the_server(conn, tmp_path, monkeypatch):
    client = paired_site(conn, tmp_path, monkeypatch)
    out = client.app.state.score_daily_scores()
    assert out["v2"]["scored"] == 2, out
    store = client.app.state.early_store
    with store.lock:
        got = rows(store.conn, None)
    by = {(r["record_key"], r["model_id"]): r for r in got}
    assert set(by) == {("106", "v1"), ("106", "v2"), ("107", "v1"), ("107", "v2")}
    for key in ("106", "107"):
        a, b = by[(key, "v2")], by[(key, "v1")]
        assert (a["backbone"], b["backbone"]) == ("m1", "m0")
        assert a["photos_sha"] == b["photos_sha"] and a["hidden_sha"] == b["hidden_sha"]
        assert a["photos"] == b["photos"] == 1 and a["green_at"] == b["green_at"]
        assert a["reference_state"] and b["reference_state"]
        assert a["code_commit"] and a["route_group"] == "primary"
    # Each answers right: the other records of its species name it, with itself left out.
    assert all(r["sp_pos"] == 0 for r in got)


def test_v1_is_never_in_the_public_payload_by_default(conn, tmp_path, monkeypatch):
    client = paired_site(conn, tmp_path, monkeypatch)
    client.app.state.score_daily_scores()
    res = client.get("/api/daily-scores")
    body = res.json()
    assert set(body) == {"public", "available", "default", "models"}
    assert list(body["models"]) == ["v2"] and body["available"] == ["v1", "v2"]
    assert "m0" not in res.text and "v1_live" not in res.text and '"paired"' not in res.text
    for sneaked in ({"models": {"v1": {"records": 2}}}, {"models": {"x": {"id": "v1"}}},
                    {"models": {}, "v1_live": {}}, {"models": {}, "paired": {}}):
        with pytest.raises(ValueError, match="v1's live scores"):
            research_public.daily_scores(sneaked)
    # Only a separate, explicit setting shows v1 publicly (off until the v1 paper is out).
    monkeypatch.setenv("MV_DAILY_SCORES_PUBLIC_V1", "1")
    body = client.get("/api/daily-scores").json()
    assert body["v1_live"]["v2"]["records"] == 2 and body["paired"]["v2"]["records"] == 2
    assert "v1" not in body["models"] and body["available"] == ["v1", "v2"]


def test_twins_are_hidden_from_both_references(conn, tmp_path, monkeypatch):
    # The release's twin pass puts 100 (an 'A x') in one find with 106: hidden from both
    # models' references while tonight's records are answered.
    from mycomap_vision.identify import Identifier
    client = paired_site(conn, tmp_path, monkeypatch, groups={"106": 0, "100": 0})
    seen: list = []
    real = Identifier.identify_vectors

    def spy(self, *a, hidden=None, **kw):
        obs = set(self.col_record.obs[self.col_record.rec[hidden]].tolist()) if hidden is not None else set()
        seen.append((self.backbone, obs))
        return real(self, *a, hidden=hidden, **kw)
    monkeypatch.setattr(Identifier, "identify_vectors", spy)
    out = client.app.state.score_daily_scores()
    assert out["v2"]["scored"] == 2, out
    assert {b for b, _ in seen} == {"m1", "m0"}
    for backbone, obs in seen:
        assert {"100", "106", "107"} <= obs, (backbone, obs)


def test_the_other_sites_record_of_the_same_name_and_day_is_hidden_as_a_twin_the_release_does_not_know():
    # A cross-post newer than the release's twin pass: same name, same day, the other site.
    from mycomap_vision.evaluate import Record
    tonight = [Record("106", "A x", "A", "F", None, None, observed_on="2026-11-20", source="inat")]
    loaded = tonight + [
        Record("mo:1", "A x", "A", "F", None, None, observed_on="2026-11-20", source="mo"),   # twin
        Record("mo:2", "A x", "A", "F", None, None, observed_on="2026-11-21", source="mo"),   # other day
        Record("mo:3", "B y", "B", "F", None, None, observed_on="2026-11-20", source="mo"),   # other name
        Record("101", "A x", "A", "F", None, None, observed_on="2026-11-20", source="inat"),  # same site
        Record("mo:4", "A x", "A", "F", None, None, observed_on=None, source="mo")]           # no day
    assert ds.same_find_records(tonight, [loaded]) == {"mo:1"}


def test_an_index_with_no_twin_table_is_refused_not_scored(conn, tmp_path, monkeypatch):
    client = paired_site(conn, tmp_path, monkeypatch, twins=False)
    out = client.app.state.score_daily_scores()
    assert "no twin table" in out["v2"]["refused"], out
    store = client.app.state.early_store
    with store.lock:
        assert rows(store.conn, None) == []


def test_a_pair_with_v1_not_served_waits_and_writes_nothing(conn, tmp_path, monkeypatch):
    client = paired_site(conn, tmp_path, monkeypatch)
    monkeypatch.setenv("MV_DAILY_SCORES_V1", f"m9/{PLACE}")
    out = client.app.state.score_daily_scores()
    assert "not served here: m9/" in out["v2"]["waiting"], out
    store = client.app.state.early_store
    with store.lock:
        assert rows(store.conn, None) == []


def test_the_internal_view_is_for_the_server_key_and_admins_and_names_v1(conn, tmp_path):
    from test_identify_log import SERVER
    from test_run_links import person, site, visitor
    app, _, _ = site(conn, tmp_path, by_observation=True)
    assert visitor(app).get("/api/daily-scores/internal").status_code == 401
    assert person(app, "user-5").get("/api/daily-scores/internal").status_code == 403
    assert person(app, "admin-1").get("/api/daily-scores/internal").status_code == 200
    res = visitor(app).get("/api/daily-scores/internal", headers=SERVER)
    assert res.status_code == 200 and res.headers["cache-control"] == "private, no-store"
    body = res.json()
    assert body["internal"] is True and body["s2_stop"] == ds.S2_STOP
    assert body["paired_from"] == ds.PAIRED_FROM and body["public_v1"] is False
