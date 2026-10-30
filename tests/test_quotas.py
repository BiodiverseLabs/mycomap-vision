"""Quotas (quotas.py, docs/api.md §6): monthly identifications per member, counted and
enforced on Vision, reported to mycomap.org in the rows its /api/vision/usage takes."""

import re
from datetime import datetime, timezone

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, png
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer, identify
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, quotas as q
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

SUB = "u_member0001"                                   # .org's subject format
OCT = datetime(2026, 10, 12, 9, tzinfo=timezone.utc).timestamp()


def cred(kind="app", subject=SUB, key_id=None, limit=None, source="plan"):
    return apiv1.Credential(kind, subject=subject, key_id=key_id, plan="free", limit=limit,
                            limit_source=source)


def quotas(tmp_path, mode="enforce", now=OCT):
    clock = [now]
    qs = q.Quotas(tmp_path / "usage.sqlite", mode=mode, clock=lambda: clock[0])
    return qs, clock


# --- the rules ------------------------------------------------------------------------

def test_the_app_and_a_members_tokens_share_one_pool_and_an_override_counts_alone(tmp_path):
    qs, _ = quotas(tmp_path)
    app = cred(limit={"monthly": 10})
    pooled = cred("key", key_id="42", limit={"monthly": 10})
    own = cred("key", key_id="43", limit={"monthly": 500}, source="override")
    qs.count(app, 1, 3)
    qs.count(pooled, 1, 1)
    qs.count(own, 1, 2)
    assert qs.usage(app)["used"] == qs.usage(pooled)["used"] == 2
    assert qs.usage(own)["used"] == 1
    rows = {(r["kind"], r.get("sub") or r.get("keyId")): r for r in qs.report_rows("2026-10")}
    assert rows[("user", SUB)]["identifications"] == 2 and rows[("user", SUB)]["photos"] == 4
    assert rows[("key", "42")]["identifications"] == 1        # each key's own use, for .org
    assert rows[("key", "43")]["identifications"] == 1
    # mycomap.org's own server key is never counted.
    qs.count(apiv1.Credential("server"), 1, 1)
    assert qs.usage(apiv1.Credential("server")) is None and len(rows) == 3


def test_limits_null_is_unlimited_and_none_given_is_the_default(tmp_path):
    qs, _ = quotas(tmp_path)
    assert qs.limit_of(cred(limit={"monthly": None})) is None
    assert qs.limit_of(cred(limit=None)) == q.DEFAULT_MONTHLY
    staff = cred(limit={"monthly": None})
    for _ in range(3):
        qs.count(staff, 1, 1)
        qs.check(staff)                                   # never refused


def test_past_the_limit_is_quota_exceeded_until_the_month_renews(tmp_path):
    qs, clock = quotas(tmp_path)
    c = cred(limit={"monthly": 2})
    for _ in range(2):
        qs.check(c)
        qs.count(c, 1, 1)
    with pytest.raises(apiv1.V1Error) as e:
        qs.check(c)
    err = e.value
    assert (err.status, err.code) == (429, "quota_exceeded")
    assert err.extra == {"limit": 2, "used": 2, "resets_at": "2026-11-01T00:00:00Z"}
    assert err.retry_after == int(datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp() - OCT)
    assert qs.report_rows("2026-10")[0]["refused"] == {"quota_exceeded": 1}
    clock[0] = datetime(2026, 11, 1, 0, 5, tzinfo=timezone.utc).timestamp()
    qs.check(c)                                           # a new month
    assert qs.usage(c)["used"] == 0 and qs.usage(c)["period"] == "2026-11"


def test_count_mode_counts_and_never_refuses_and_off_counts_nothing(tmp_path):
    qs, _ = quotas(tmp_path, mode="count")
    c = cred(limit={"monthly": 1})
    for _ in range(3):
        qs.check(c)
        qs.count(c, 1, 1)
    assert qs.usage(c)["used"] == 3
    off = q.Quotas(tmp_path / "never.sqlite", mode="off")
    off.count(c, 1, 1)
    assert off.usage(c) is None and not (tmp_path / "never.sqlite").exists()
    with pytest.raises(ValueError):
        q.Quotas(tmp_path / "x.sqlite", mode="sometimes")


def test_reports_are_rows_mycomap_org_accepts_with_last_month_for_two_days(tmp_path):
    qs, clock = quotas(tmp_path, now=datetime(2026, 10, 31, 23, tzinfo=timezone.utc).timestamp())
    qs.count(cred(), 1, 2)
    qs.count(cred("key", key_id="42"), 1, 1)
    clock[0] = datetime(2026, 11, 1, 12, tzinfo=timezone.utc).timestamp()
    qs.count(cred(), 1, 1)
    assert qs.periods_to_report() == ["2026-10", "2026-11"]
    sent = []
    assert q.report(qs, sent.append, clock=lambda: clock[0]) == 3   # Oct: member + key; Nov: member
    assert [b["period"] for b in sent] == ["2026-10", "2026-11"]
    for body in sent:                                  # what .org's parseVisionUsageReport takes
        assert re.fullmatch(r"\d{4}-(0[1-9]|1[0-2])", body["period"])
        for r in body["rows"]:
            assert r["kind"] in ("user", "key")
            if r["kind"] == "user":
                assert re.fullmatch(r"u_[A-Za-z0-9_-]{8,64}", r["sub"])
            else:
                assert re.fullmatch(r"\d{1,9}", r["keyId"])
            assert isinstance(r["identifications"], int) and isinstance(r["photos"], int)
            assert isinstance(r["refused"], dict) and r["last_used_at"].endswith("Z")
    clock[0] = datetime(2026, 11, 4, tzinfo=timezone.utc).timestamp()
    assert qs.periods_to_report() == ["2026-11"]


def test_a_report_is_split_at_mycomap_orgs_row_limit(tmp_path, monkeypatch):
    assert q.MAX_ROWS == 5000                              # .org's VISION_USAGE_MAX_ROWS
    monkeypatch.setattr(q, "MAX_ROWS", 4)
    qs, _ = quotas(tmp_path)
    for i in range(q.MAX_ROWS + 3):
        qs.count(cred(subject=f"u_member{i:06d}"), 1, 1)
    sent = []
    q.report(qs, sent.append)
    assert [len(b["rows"]) for b in sent] == [q.MAX_ROWS, 3]


# --- 2026-10-29/30: reports lost to a kept-alive connection the far end had closed -------------

@pytest.fixture
def org_that_drops_idle_connections():
    """A stand-in for mycomap.org behind Cloudflare and nginx: it answers the first report on a
    connection and closes that connection, with no response, when a second report comes on it
    (what a kept-alive connection closed while idle looks like: RemoteDisconnected)."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    got: list[dict] = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.seen = getattr(self, "seen", 0) + 1
            if self.seen > 1:                         # the stale connection: closed, no answer
                self.close_connection = True
                return
            got.append({"auth": self.headers["Authorization"], "body": body,
                        "path": self.path})
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"{}")

        def log_message(self, *a):
            pass
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", got
    server.shutdown()
    server.server_close()


def test_a_report_on_a_connection_the_far_end_closed_is_sent_again_on_a_fresh_one(
        tmp_path, org_that_drops_idle_connections):
    base, got = org_that_drops_idle_connections
    qs, clock = quotas(tmp_path)
    post = q.org_poster(base, "k1")
    qs.count(cred(), 1, 2)
    assert q.report(qs, post, clock=lambda: clock[0]) == 1
    qs.count(cred(), 1, 3)                            # a later report: the running totals
    assert q.report(qs, post, clock=lambda: clock[0]) == 1
    assert [g["body"]["rows"][0]["photos"] for g in got] == [2, 5]
    assert {g["auth"] for g in got} == {"Bearer k1"}
    assert {g["path"] for g in got} == {"/api/vision/usage"}


class FakeSession:
    def __init__(self, outcomes, made):
        self.outcomes, self.closed, self.bodies = outcomes, False, []
        made.append(self)

    def post(self, url, json, timeout, headers):
        self.bodies.append(json)
        out = self.outcomes.pop(0)
        if isinstance(out, Exception):
            raise out
        return out

    def close(self):
        self.closed = True


class Answer:
    def __init__(self, status):
        self.status = status

    def raise_for_status(self):
        if self.status >= 400:
            import requests
            raise requests.HTTPError(f"{self.status}")


def poster_with(outcomes):
    made: list[FakeSession] = []
    post = q.org_poster("https://org.test", "k1",
                        session_factory=lambda: FakeSession(outcomes, made))
    return post, made


def test_a_reset_or_closed_connection_is_tried_once_more_and_the_stale_session_dropped():
    import http.client

    import requests
    for error in (requests.ConnectionError(http.client.RemoteDisconnected("closed")),
                  requests.ConnectionError(ConnectionResetError(104, "reset by peer"))):
        post, made = poster_with([error, Answer(200)])
        post({"period": "2026-10", "rows": [1]})
        assert len(made) == 2 and made[0].closed and not made[1].closed
        assert made[0].bodies == made[1].bodies == [{"period": "2026-10", "rows": [1]}]


def test_only_one_retry_and_never_for_an_http_error_or_a_read_timeout():
    import requests
    post, made = poster_with([requests.ConnectionError("down"), requests.ConnectionError("down")])
    with pytest.raises(requests.ConnectionError):     # reported; the next report sends it all
        post({"rows": []})
    assert len(made) == 2
    post, made = poster_with([Answer(503)])
    with pytest.raises(requests.HTTPError):
        post({"rows": []})
    assert len(made) == 1
    post, made = poster_with([requests.ReadTimeout("slow")])
    with pytest.raises(requests.ReadTimeout):
        post({"rows": []})
    assert len(made) == 1


# --- through the API --------------------------------------------------------------------

def app(conn, tmp_path, mode="enforce", posted=None):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(), background=False, server_key=SERVER_KEY, api_v1=True,
        sessions=True, app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
        fungus_gate=None, note=lambda s: None,
        quotas=q.Quotas(tmp_path / "usage.sqlite", mode=mode),
        usage_post=(posted.append if posted is not None else None)))


def member(limit=2):
    return bearer(app_token(sub=SUB, limit=limit))


def test_identify_shows_usage_and_refuses_past_the_monthly_limit(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_PHOTOS_PER_MINUTE", "4")
    client = app(conn, tmp_path)
    used = [identify(client, member()).json()["usage"]["used"] for _ in range(2)]
    assert used == [1, 2]
    res = identify(client, member())
    assert res.status_code == 429 and res.json()["code"] == "quota_exceeded"
    body = res.json()
    assert (body["limit"], body["used"]) == (2, 2) and body["resets_at"].endswith("Z")
    assert int(res.headers["retry-after"]) > 0
    # A refusal spends nothing: two photos of the 4-a-minute window are still free.
    other = bearer(app_token(sub="u_member0002", limit=5))
    assert identify(client, other, reds=(1, 2)).status_code == 200


def test_count_mode_never_refuses(conn, tmp_path):
    client = app(conn, tmp_path, mode="count")
    for _ in range(3):
        assert identify(client, member(limit=1)).status_code == 200
    assert identify(client, member(limit=1)).json()["usage"]["used"] == 4


def test_with_quotas_off_there_is_no_usage(conn, tmp_path):
    res = identify(app(conn, tmp_path, mode="off"), member())
    assert res.status_code == 200 and res.json()["usage"] is None


def test_a_failed_identification_is_not_counted(conn, tmp_path):
    client = app(conn, tmp_path)
    bad = client.post("/api/v1/identify", headers=member(),
                      files=[("photos", ("x.png", b"nope", "image/png"))])
    assert bad.status_code == 415
    assert identify(client, member()).json()["usage"]["used"] == 1


def test_a_session_counts_once_at_its_first_photo(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_PHOTOS_PER_MINUTE", "20")
    client = app(conn, tmp_path)
    sid = client.post("/api/v1/sessions", json={}, headers=member()).json()["session_id"]
    def add(red, ref):
        return client.post(f"/api/v1/sessions/{sid}/photos", headers=member(),
                           files={"photo": ("p.png", png(red), "image/png")},
                           data={"photo_ref": ref})
    first = add(247, "p1").json()
    assert first["usage"]["used"] == 1 and first["combined"]["usage"]["used"] == 1
    for i, red in enumerate((245, 240)):
        assert add(red, f"p{i + 2}").json()["usage"]["used"] == 1     # photos cost no more
    assert add(247, "p1").json()["usage"]["used"] == 1                # nor a resend
    [row] = [r for r in client.app.state.quotas.report_rows(q.period_of(__import__("time").time()))]
    assert (row["identifications"], row["photos"]) == (1, 3)


def test_a_session_cannot_start_past_the_limit_but_one_already_counted_goes_on(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = client.post("/api/v1/sessions", json={}, headers=member(limit=1)).json()["session_id"]
    def add(sid, red, ref):
        return client.post(f"/api/v1/sessions/{sid}/photos", headers=member(limit=1),
                           files={"photo": ("p.png", png(red), "image/png")},
                           data={"photo_ref": ref})
    assert add(sid, 247, "p1").status_code == 200                      # the month's one
    assert add(sid, 245, "p2").status_code == 200                      # same identification
    sid2 = client.post("/api/v1/sessions", json={}, headers=member(limit=1)).json()["session_id"]
    res = add(sid2, 247, "p1")
    assert res.status_code == 429 and res.json()["code"] == "quota_exceeded"


def test_the_server_reports_use_to_mycomap_org(conn, tmp_path):
    posted = []
    client = app(conn, tmp_path, posted=posted)
    identify(client, member())
    assert client.app.state.report_usage() == 1
    [body] = posted
    assert body["rows"][0]["sub"] == SUB and body["rows"][0]["identifications"] == 1
    assert client.get("/api/health").json()["quotas"] == {"mode": "enforce"}


def test_app_and_key_limits_read_null_as_unlimited_and_absence_as_none():
    assert apiv1.limit_of({}) is None
    assert apiv1.limit_of({"limit": None}) == {"monthly": None}
    assert apiv1.limit_of({"limit": 300}) == {"monthly": 300}
    assert apiv1.limit_of({"limit": {"monthly": None, "period": "2026-10"}})["monthly"] is None
    assert apiv1.limit_of({"limit": {"monthly": "lots"}})["monthly"] == 0      # fails closed


# --- a find re-opened within a day is free (Steve, 2026-10-12) --------------------------

def test_a_find_counted_in_the_last_day_is_free_and_later_counts_again(tmp_path):
    qs, clock = quotas(tmp_path)
    member_cred = cred(limit={"monthly": 5})
    assert not qs.counted_before(member_cred, "obs-1")
    qs.remember_counted(member_cred, "obs-1")
    clock[0] += 23 * 3600
    assert qs.counted_before(member_cred, "obs-1")
    assert not qs.counted_before(member_cred, "obs-2")                 # another find
    assert not qs.counted_before(cred(subject="u_member0009"), "obs-1")  # another member
    assert not qs.counted_before(member_cred, None)                    # no find id: counts
    clock[0] += 2 * 3600                                               # 25 h after
    assert not qs.counted_before(member_cred, "obs-1")
    # The member's own keys share the pool, so the same find is free through them too.
    assert qs.counted_before(cred("key", key_id="42"), "obs-1") is False
    qs.remember_counted(member_cred, "obs-1")
    assert qs.counted_before(cred("key", key_id="42"), "obs-1")


def test_a_reopened_session_for_the_same_find_is_not_counted_again(conn, tmp_path):
    client = app(conn, tmp_path)
    headers = member(limit=1)

    def session(client_ref):
        return client.post("/api/v1/sessions", json={"client_ref": client_ref},
                           headers=headers).json()["session_id"]

    def add(sid, red, ref):
        return client.post(f"/api/v1/sessions/{sid}/photos", headers=headers,
                           files={"photo": ("p.png", png(red), "image/png")},
                           data={"photo_ref": ref})
    first = add(session("obs-1"), 247, "p1")
    assert first.status_code == 200 and first.json()["usage"]["used"] == 1
    # The session is lost (restart, idle expiry); the app re-opens it for the same find
    # and resends: free, even though the month's one identification is used.
    again = add(session("obs-1"), 247, "p1")
    assert again.status_code == 200 and again.json()["usage"]["used"] == 1
    # A new find is a new identification: refused at the limit.
    res = add(session("obs-2"), 245, "p1")
    assert res.status_code == 429 and res.json()["code"] == "quota_exceeded"
    # A session without a find id always counts.
    assert add(session(None), 245, "p1").status_code == 429


def test_identify_with_the_finds_client_ref_is_free_after_its_session_was_counted(conn, tmp_path):
    # Signal drops mid-find: the session counted the first photo; after saving, the app
    # sends all the photos to /identify with the same client_ref (integration testing).
    client = app(conn, tmp_path)
    headers = member(limit=1)
    sid = client.post("/api/v1/sessions", json={"client_ref": "app-1760000000"},
                      headers=headers).json()["session_id"]
    assert client.post(f"/api/v1/sessions/{sid}/photos", headers=headers,
                       files={"photo": ("p.png", png(247), "image/png")},
                       data={"photo_ref": "p1"}).json()["usage"]["used"] == 1
    finish = identify(client, headers, reds=(247, 245), client_ref="app-1760000000")
    assert finish.status_code == 200 and finish.json()["usage"]["used"] == 1
    # Another find, or none named, is a new identification.
    assert identify(client, headers, client_ref="app-1760009999").status_code == 429
    assert identify(client, headers).status_code == 429
    # And an /identify counted first makes the find's later session free.
    other = bearer(app_token(sub="u_member0003", limit=1))
    assert identify(client, other, client_ref="obs-9").json()["usage"]["used"] == 1
    sid = client.post("/api/v1/sessions", json={"client_ref": "obs-9"},
                      headers=other).json()["session_id"]
    assert client.post(f"/api/v1/sessions/{sid}/photos", headers=other,
                       files={"photo": ("p.png", png(247), "image/png")},
                       data={"photo_ref": "p1"}).status_code == 200
    assert identify(client, headers, client_ref="x" * 65).json()["code"] == "bad_request"
