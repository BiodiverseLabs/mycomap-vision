"""The identification log (identifylog.py; Steve, 2026-10-20): one row for every answer
Vision gives or serves, on every path, insert-only, written off the request path so it
never fails or slows an identification; and the paper's read, per observation the first
time Vision answered it, for mycomap.org's server key and admins only."""

import re
import sqlite3
import threading
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from test_api import png
from test_api_v1_identify import DEV_KEY, SERVER_KEY, app_token, bearer
from test_api_v1_sessions import add, app as session_app, open_session
from test_feedback_api import as_user, item_id, make_site as feedback_site
from test_run_links import by_link, identify, person, site, visitor

from mycomap_vision import answercache, cli, identifylog as il, runs

SERVER = bearer(SERVER_KEY)
SRC = Path(il.__file__).parent


def log_of(app) -> il.IdentifyLog:
    return app.state.identify_log


def rows(app_or_log) -> list[dict]:
    """Every row written so far, oldest first (after the writer has caught up)."""
    log = app_or_log if isinstance(app_or_log, il.IdentifyLog) else log_of(app_or_log)
    assert log.flush(10), "the log's writer did not catch up"
    with log._db:
        cur = log.conn.execute("select * from identify_log order by id")
        names = [d[0] for d in cur.description]
        return [dict(zip(names, r)) for r in cur.fetchall()]


def app_of(client: TestClient):
    return client.app


# --- one row per answer, on every path ---------------------------------------------------

def test_a_site_upload_is_logged_with_who_ran_it_the_model_and_its_link(conn, tmp_path):
    app, _, _ = site(conn, tmp_path)
    res = identify(person(app, "user-1", name="Mycena Fan"))
    assert res.status_code == 200, res.text
    [row] = rows(app)
    assert row["path"] == il.SITE_UPLOAD and row["fresh"] == 1 and row["at_exact"] == 1
    assert row["provenance"] == il.LIVE
    assert (row["asked_by_kind"], row["asked_by_id"], row["asked_by_name"]) == (
        "site", "user-1", "Mycena Fan")
    assert row["run_id"] == res.json()["run"]["id"]
    assert row["model"].startswith("m1/") and row["answers_version"]
    assert row["source"] is None and row["observation_id"] is None   # photos carry no id
    assert row["at_utc"].endswith("Z") and abs(row["at"] - time.time()) < 60


def test_a_site_link_is_logged_fresh_then_kept_with_the_observation(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    who = person(app, "user-2", name="Amanita Ann")
    first = by_link(who, "inat", 555)
    again = by_link(who, "inat", 555)
    assert first.status_code == again.status_code == 200
    assert again.json()["cached"] is True
    a, b = rows(app)
    for row in (a, b):
        assert row["path"] == il.SITE_OBSERVATION
        assert (row["source"], row["observation_id"], row["observation_from"]) == (
            "inat", "555", "request")
        assert row["answer_key"].startswith("site:inat:555:")
        assert (row["asked_by_kind"], row["asked_by_id"]) == ("site", "user-2")
    assert (a["fresh"], b["fresh"]) == (1, 0)
    assert a["answers_version"] == b["answers_version"] == first.json()["answered"]["answers_version"]
    assert a["run_id"] == first.json()["run"]["id"] and b["run_id"] == again.json()["run"]["id"]


def test_v1_identify_is_logged_with_the_key_id_and_an_observation_named_by_client_ref(
        conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    client = visitor(app)
    files = [("photos", ("p.png", png(247), "image/png"))]
    assert client.post("/api/v1/identify", files=files, headers=bearer(DEV_KEY),
                       data={"client_ref": "mo:777"}).status_code == 200
    assert client.post("/api/v1/identify", files=files, headers=SERVER,
                       data={"client_ref": "obs-local-1"}).status_code == 200
    a, b = rows(app)
    assert a["path"] == b["path"] == il.V1_IDENTIFY and a["fresh"] == b["fresh"] == 1
    assert (a["asked_by_kind"], a["asked_by_id"]) == ("key", "42")
    assert (a["source"], a["observation_id"], a["observation_from"]) == ("mo", "777", "client_ref")
    assert (b["asked_by_kind"], b["asked_by_id"]) == ("server", None)
    assert b["observation_id"] is None and b["client_ref"] == "obs-local-1"
    assert a["answers_version"] and a["model"].startswith("m1/")


def test_v1_by_observation_and_the_kept_answer_read_are_logged(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    client = visitor(app)
    member = bearer(app_token(sub="u_member9"))
    assert client.post("/api/v1/identify/inat/555", json={}, headers=member).status_code == 200
    assert client.post("/api/v1/identify/inat/555", json={}, headers=SERVER).status_code == 200
    assert client.get("/api/v1/answers/inat/555", headers=SERVER).status_code == 200
    fresh, kept, read = rows(app)
    assert (fresh["path"], fresh["fresh"]) == (il.V1_OBSERVATION, 1)
    assert (fresh["asked_by_kind"], fresh["asked_by_id"]) == ("app", "u_member9")
    assert fresh["client_ref"] == "inat:555"
    assert (kept["path"], kept["fresh"], kept["asked_by_kind"]) == (il.V1_OBSERVATION, 0, "server")
    assert (read["path"], read["fresh"]) == (il.V1_ANSWERS, 0)
    for row in (fresh, kept, read):
        assert (row["source"], row["observation_id"]) == ("inat", "555")
        assert row["answer_key"].startswith("v1:inat:555:")
        assert row["answers_version"] == fresh["answers_version"]


def test_each_new_answer_in_an_app_session_is_logged(conn, tmp_path):
    client = session_app(conn, tmp_path)
    sid = open_session(client, client_ref="inat:12345")["session_id"]
    assert add(client, sid, 247, "a").status_code == 200
    assert add(client, sid, 245, "b").status_code == 200
    assert add(client, sid, 245, "b").status_code == 200        # a resend: nothing new
    got = rows(app_of(client))
    assert [r["path"] for r in got] == [il.APP_SESSION, il.APP_SESSION]
    assert all(r["fresh"] == 1 and r["asked_by_kind"] == "app" and r["asked_by_id"] == "u_member1"
               for r in got)
    assert (got[0]["source"], got[0]["observation_id"], got[0]["client_ref"]) == (
        "inat", "12345", "inat:12345")


def test_a_human_feedback_rescore_is_logged_with_the_reviewer(conn, tmp_path):
    c, fstore, _ = feedback_site(conn, tmp_path)
    iid = item_id(fstore, "photos_unlike_label")
    as_user(c, "admin-1")
    c.post(f"/api/feedback/items/{iid}/decision", json={"decision": "rename", "value": "A x"})
    assert c.post(f"/api/feedback/items/{iid}/refresh").json()["refreshed"][0]["vision"]["available"]
    [row] = rows(app_of(c))
    assert (row["path"], row["fresh"], row["source"], row["observation_id"]) == (
        il.FEEDBACK_RESCORE, 1, "inat", "100")
    assert (row["asked_by_kind"], row["asked_by_id"]) == ("reviewer", "admin-1")


def test_a_refused_identification_is_not_logged(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    client = visitor(app)
    assert client.post("/api/v1/identify/inat/999999", json={}, headers=SERVER).status_code == 404
    assert client.get("/api/v1/answers/inat/558", headers=SERVER).status_code == 404
    assert rows(app) == []


# --- insert only ------------------------------------------------------------------------

def test_the_database_itself_refuses_update_and_delete(tmp_path):
    log = il.IdentifyLog(tmp_path / il.FILE, start=False)
    log.record(il.V1_IDENTIFY, fresh=True)
    log.drain()
    raw = sqlite3.connect(tmp_path / il.FILE)
    for sql in ("update identify_log set fresh = 0", "delete from identify_log",
                "update identify_log set at = 0 where id = 1"):
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            raw.execute(sql)
    raw.rollback()
    assert raw.execute("select count(*), min(fresh) from identify_log").fetchone() == (1, 1)


def test_no_identification_path_ever_updates_or_deletes_a_row(conn, tmp_path):
    """Every statement the log's database runs while every path answers (fresh, kept,
    reruns, refusals) is an INSERT or a read."""
    app, _, _ = site(conn, tmp_path, by_observation=True)
    seen: list[str] = []
    log_of(app).conn.set_trace_callback(seen.append)
    who, client = person(app, "user-3"), visitor(app)
    identify(who)
    by_link(who, "inat", 555)
    by_link(who, "inat", 555)
    by_link(who, "inat", 555, rerun=True)
    client.post("/api/v1/identify", files=[("photos", ("p.png", png(247), "image/png"))],
                headers=SERVER)
    client.post("/api/v1/identify/mo/777", json={}, headers=SERVER)
    client.post("/api/v1/identify/mo/777", json={"rerun": True}, headers=SERVER)
    client.get("/api/v1/answers/mo/777", headers=SERVER)
    client.get("/api/identify-log/first", headers=SERVER)
    assert len(rows(app)) == 8
    writes = [s for s in seen if re.match(r"\s*(update|delete|replace|drop|alter)\b", s, re.I)
              or re.search(r"\binsert\s+or\s+replace\b", s, re.I)]
    assert writes == []
    assert any(s.lower().startswith("insert into identify_log") for s in seen)


def test_no_code_in_the_package_updates_or_deletes_log_rows():
    pattern = re.compile(r"(update\s+identify_log|delete\s+from\s+identify_log|replace\s+into\s+"
                         r"identify_log|drop\s+(table|trigger)\s+(if\s+exists\s+)?identify_log)",
                         re.I)
    offenders = [f"{p.name}: {m.group(0)}" for p in SRC.rglob("*.py")
                 for m in pattern.finditer(p.read_text(encoding="utf-8"))
                 if not (p.name == "identifylog.py" and "raise(abort" in
                         p.read_text(encoding="utf-8")[m.start():m.start() + 120])]
    assert offenders == []


def test_importing_the_same_store_twice_is_refused(tmp_path):
    answers = answercache.AnswerCache(tmp_path / "observation-answers.sqlite", clock=lambda: 1e9)
    answers.put("site:inat:555:m1/nearest:date=True,place=True", "v", {}, "site")
    log = il.IdentifyLog(tmp_path / il.FILE, start=False)
    assert il.import_answers(log, tmp_path / "observation-answers.sqlite") == 1
    with pytest.raises(ValueError, match="imported already"):
        il.import_answers(log, tmp_path / "observation-answers.sqlite")
    assert len(rows(log)) == 1


# --- never fails, never slows -------------------------------------------------------------

class StuckLog(il.IdentifyLog):
    """A log whose database hangs (a locked file, a full disk that never answers)."""

    def __init__(self, path):
        super().__init__(path, note=lambda s: None)
        self.release = threading.Event()
        self.tried = threading.Event()

    def insert(self, rows):
        self.tried.set()
        self.release.wait(30)
        super().insert(rows)


class FailingLog(il.IdentifyLog):
    def __init__(self, path):
        super().__init__(path, note=lambda s: None, retries=())

    def insert(self, rows):
        raise sqlite3.OperationalError("disk I/O error")


class BrokenLog:
    """Not even a queue: record itself raises."""
    def record(self, *a, **kw):
        raise RuntimeError("broken")

    def state(self):
        return {}


def site_with(folder: Path, log):
    """A site with v1 and identify-by-observation, its own manifest in `folder`."""
    from test_api import open_limits
    from test_api_v1_by_observation import fetcher
    from test_models_and_scoreboard import Const, seed_two_species
    from test_run_links import Roles

    from mycomap_vision import manifest
    from mycomap_vision.api import create_app
    from mycomap_vision.embed import embed_photos, photos_to_embed
    folder.mkdir(parents=True, exist_ok=True)
    conn = manifest.connect(folder / "manifest.sqlite")
    store = seed_two_species(conn, folder)
    root = folder / "emb"
    embed_photos(conn, store, Const("m1"), photos_to_embed(conn, "m1", "large", store.location,
                                                           "local"), root / "m1", log=lambda s: None)
    conn.commit()
    app = create_app(folder / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), background=False, fungus_gate=None, api_v1=True,
                     server_key=SERVER_KEY, by_observation=True, observation_fetcher=fetcher(),
                     feedback_org=Roles(), note=lambda s: None, identify_log=log)
    return TestClient(app, follow_redirects=False)


def upload(client):
    return client.post("/api/identify", files=[("photos", ("a.png", png(247), "image/png"))])


def test_a_hung_log_database_never_holds_up_an_answer(tmp_path):
    log = StuckLog(tmp_path / "log" / il.FILE)
    client = site_with(tmp_path / "site", log)
    try:
        start = time.monotonic()
        res = upload(client)
        took = time.monotonic() - start
        assert res.status_code == 200, res.text
        assert log.tried.wait(10)            # the writer is stuck inside the database now
        res2 = client.post("/api/v1/identify/inat/555", json={}, headers=SERVER)
        assert res2.status_code == 200
        assert time.monotonic() - start < 10 and took < 10
        assert log.state()["pending"] + log._busy >= 1
    finally:
        log.release.set()
    assert len(rows(log)) == 2


@pytest.mark.parametrize("make", [FailingLog, lambda path: BrokenLog()])
def test_a_failing_log_never_fails_an_answer_or_changes_it(tmp_path, make):
    good = site_with(tmp_path / "a", il.IdentifyLog(tmp_path / "a-log" / il.FILE))
    bad = site_with(tmp_path / "b", make(tmp_path / "b-log" / il.FILE))
    answers = {}
    for name, client in (("good", good), ("bad", bad)):
        res = upload(client)
        assert res.status_code == 200, res.text
        obs = client.post("/api/v1/identify/inat/555", json={}, headers=SERVER)
        assert obs.status_code == 200, obs.text
        answers[name] = ([r["ranks"] for r in res.json()["results"]], obs.json()["ranks"])
    assert answers["good"] == answers["bad"]
    log = bad.app.state.identify_log
    if isinstance(log, FailingLog):
        assert log.flush(10)
        assert log.state()["failed"] == 2 and log.state()["written"] == 0
        assert "disk I/O error" in log.state()["last_error"]


def test_a_full_queue_drops_and_counts_rows_without_waiting(tmp_path):
    log = il.IdentifyLog(None, start=False, max_pending=2, note=lambda s: None)
    start = time.monotonic()
    for _ in range(5):
        log.record(il.V1_IDENTIFY, fresh=True)
    assert time.monotonic() - start < 1
    assert log.state()["dropped"] == 3 and log.state()["pending"] == 2
    assert log.drain() == 2 and log.state()["written"] == 2


def test_a_bad_row_is_counted_not_raised():
    log = il.IdentifyLog(None, start=False, note=lambda s: None)
    log.record("no-such-path", fresh=True)
    log.record(il.V1_IDENTIFY, fresh=True, release=lambda: 1 / 0)
    log.record(il.V1_IDENTIFY, fresh=True)
    log.drain()
    assert log.state()["failed"] == 2 and log.state()["written"] == 1


def test_the_writer_thread_starts_with_the_first_row_and_stamps_the_time_it_was_asked():
    times = iter([100.0, 200.0])
    log = il.IdentifyLog(None, clock=lambda: next(times), note=lambda s: None)
    assert log._thread is None
    log.record(il.V1_IDENTIFY, fresh=True)
    log.record(il.V1_IDENTIFY, fresh=True)
    assert log._thread is not None
    assert [r["at"] for r in rows(log)] == [100.0, 200.0]


# --- the paper's read ------------------------------------------------------------------

def test_the_first_answer_per_observation_is_the_earliest_with_its_path():
    log = il.IdentifyLog(None, start=False)
    clock = iter([10.0, 20.0, 30.0, 40.0])
    log.clock = lambda: next(clock)
    log.record(il.V1_OBSERVATION, fresh=True, source="inat", observation_id="1")
    log.record(il.SITE_OBSERVATION, fresh=False, source="inat", observation_id="1")
    log.record(il.SITE_UPLOAD, fresh=True)
    log.record(il.SITE_OBSERVATION, fresh=True, source="mo", observation_id="1")
    log.drain()
    first = log.first_answers()
    assert [(f["source"], f["observation_id"], f["first_path"], f["rows"]) for f in first] == [
        ("inat", "1", il.V1_OBSERVATION, 2), ("mo", "1", il.SITE_OBSERVATION, 1)]
    assert first[0]["first_at"] == il.iso_utc(10.0) and first[0]["first_exact"] is True
    assert first[0]["first_live_at"] == il.iso_utc(10.0)
    assert [f["source"] for f in log.first_answers(source="mo")] == ["mo"]
    assert [f["source"] for f in log.first_answers(since=35.0)] == ["mo"]


def test_recovered_rows_say_at_or_before_and_never_pass_as_live(tmp_path):
    answers = answercache.AnswerCache(tmp_path / "observation-answers.sqlite",
                                      clock=lambda: 1_000.0)
    answers.put("v1:inat:7:m1/nearest:date=True,place=True,top_k=5", "ver1", {}, "mycomap.org")
    answers.put("site:mo:8:m1/nearest:date=True,place=True", "ver2", {}, None)
    log = il.IdentifyLog(tmp_path / il.FILE, start=False, clock=lambda: 2_000.0)
    assert il.import_answers(log, tmp_path / "observation-answers.sqlite") == 2
    log.record(il.V1_ANSWERS, fresh=False, source="inat", observation_id="7")
    log.drain()
    got = {r["observation_id"]: r for r in rows(log) if r["provenance"] == il.FROM_ANSWERS}
    assert got["7"]["path"] == il.V1_OBSERVATION and got["7"]["asked_by_kind"] == "server"
    assert got["8"]["path"] == il.SITE_OBSERVATION and got["8"]["asked_by_kind"] == "unknown"
    assert all(r["at_exact"] == 0 and r["at"] == 1_000.0 and r["observation_from"] == "answer_key"
               for r in got.values())
    assert got["7"]["answer_key"].startswith("v1:inat:7:") and got["7"]["answers_version"] == "ver1"
    first = {f["observation_id"]: f for f in log.first_answers()}
    assert first["7"]["first_exact"] is False and first["7"]["first_provenance"] == il.FROM_ANSWERS
    assert first["7"]["first_live_at"] == il.iso_utc(2_000.0)
    assert first["8"]["first_live_at"] is None


def test_run_links_are_imported_from_the_owner_index_labelled(conn, tmp_path):
    app, store, _ = site(conn, tmp_path, by_observation=True)
    who = person(app, "user-4", name="Boletus Bo")
    run = by_link(who, "inat", 555).json()["run"]
    identify(who)
    log = il.IdentifyLog(tmp_path / "import" / il.FILE, start=False)
    assert il.import_runs(log, tmp_path / "site-runs.sqlite", store.get) == 2
    got = rows(log)
    assert {r["path"] for r in got} == {il.SITE_OBSERVATION, il.SITE_UPLOAD}
    linked = next(r for r in got if r["path"] == il.SITE_OBSERVATION)
    assert (linked["source"], linked["observation_id"], linked["run_id"]) == ("inat", "555", run["id"])
    assert (linked["asked_by_id"], linked["asked_by_name"]) == ("user-4", "Boletus Bo")
    assert all(r["provenance"] == il.FROM_RUNS and r["at_exact"] == 0 for r in got)
    with pytest.raises(ValueError, match="imported already"):
        il.import_runs(log, tmp_path / "site-runs.sqlite", store.get)


def test_the_read_is_for_the_server_key_and_admins_private_and_never_says_who(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    visitor(app).post("/api/v1/identify/inat/555", json={}, headers=SERVER)
    res = visitor(app).get("/api/identify-log/first", headers=SERVER)
    assert res.status_code == 200
    assert res.headers["cache-control"] == "private, no-store"
    assert "noindex" in res.headers["x-robots-tag"]
    [obs] = res.json()["observations"]
    assert (obs["source"], obs["observation_id"], obs["first_path"]) == ("inat", "555",
                                                                         il.V1_OBSERVATION)
    assert not any("asked_by" in k or "client_ref" in k for k in obs)
    tsv = visitor(app).get("/api/identify-log/first", params={"format": "tsv"}, headers=SERVER)
    assert tsv.text.splitlines()[0].split("\t") == list(il.FIRST_COLUMNS)
    assert tsv.text.splitlines()[1].startswith("inat\t555\t")
    assert visitor(app).get("/api/identify-log/first").status_code == 401
    assert visitor(app).get("/api/identify-log/first",
                            headers=bearer("x" * 40)).status_code == 401
    assert person(app, "user-5").get("/api/identify-log/first").status_code == 403
    assert person(app, "admin-1").get("/api/identify-log/first").status_code == 200
    assert visitor(app).get("/api/identify-log/first", headers=SERVER,
                            params={"since": "2999-01-01T00:00:00Z"}).json()["count"] == 0
    assert visitor(app).get("/api/identify-log/first", headers=SERVER,
                            params={"since": "soon"}).status_code == 400


def test_the_logs_counters_are_in_health_for_the_server_key_only(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, by_observation=True)
    identify(person(app, "user-1"))
    rows(app)
    full = visitor(app).get("/api/health", headers=SERVER).json()
    assert full["identify_log"]["written"] == 1 and full["identify_log"]["dropped"] == 0
    assert "identify_log" not in visitor(app).get("/api/health").json()


def test_mv_identify_log_first_writes_the_tsv(tmp_path, capsys):
    path = tmp_path / il.FILE
    log = il.IdentifyLog(path, start=False, clock=lambda: 1_760_000_000.0)
    log.record(il.V1_OBSERVATION, fresh=True, source="inat", observation_id="42")
    log.drain()
    out = tmp_path / "first.tsv"
    assert cli.main(["identify-log", "first", "--log", str(path), "--out", str(out)]) == 0
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[1].split("\t")[:5] == ["inat", "42", il.iso_utc(1_760_000_000.0), "1",
                                        il.V1_OBSERVATION]
    assert cli.main(["identify-log", "--log", str(path)]) == 0
    assert '"v1_observation"' in capsys.readouterr().out


def test_the_log_sits_in_state_on_the_box_and_beside_the_manifest_elsewhere(tmp_path, monkeypatch):
    from mycomap_vision import config
    monkeypatch.setattr(config, "RELEASE_ROOT", tmp_path / "srv")
    assert il.default_path(tmp_path / "m.sqlite") == tmp_path / "srv" / "state" / il.FILE
    monkeypatch.setattr(config, "RELEASE_ROOT", None)
    assert il.default_path(tmp_path / "m.sqlite") == tmp_path / il.FILE
