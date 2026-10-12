"""The app's observation sessions (docs/api.md §3.3): photos sent one at a time, each answer
the combined answer over all of them, exactly as identifying them together would give,
from stored vectors (never re-embedded) and with no photo kept."""

from collections.abc import Mapping

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, png
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer, identify
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, sessions as session_store
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

class Fresh(Mapping):
    """Authorization headers whose app token is minted when read. A token minted once at
    import lives 60 s (+30 s skew), and on a slow, loaded run the last tests of this file
    started later than that and got 401 token_expired."""

    def __init__(self, **claims):
        self.claims = claims

    def _headers(self) -> dict:
        return bearer(app_token(**self.claims))

    def __getitem__(self, key):
        return self._headers()[key]

    def __iter__(self):
        return iter(self._headers())

    def __len__(self):
        return 1


APP = Fresh()
OTHER = Fresh(sub="u_someone_else")


class FakeGate:
    """A gate on the served vectors that calls everything a fungus and counts its calls."""
    base = "m1"

    def __init__(self):
        self.fp, self.judged = "gate-1", 0

    def fingerprint(self):
        return self.fp

    def judge(self, base_vectors):
        self.judged += len(base_vectors)
        return [{"verdict": "fungus", "p_fungus": 0.97, "message": None}
                for _ in range(len(base_vectors))]

    def summary(self):
        return {"fingerprint": self.fp}


def app(conn, tmp_path, sessions=True, gate=None, api_v1=True):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(), background=False, server_key=SERVER_KEY, api_v1=api_v1,
        sessions=sessions, app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
        fungus_gate=gate, note=lambda s: None))


def open_session(client, headers=APP, **body):
    res = client.post("/api/v1/sessions", json=body, headers=headers)
    assert res.status_code == 201, res.text
    return res.json()


def add(client, sid, red, ref, headers=APP):
    return client.post(f"/api/v1/sessions/{sid}/photos", headers=headers,
                       files={"photo": ("p.png", png(red), "image/png")},
                       data={"photo_ref": ref})


def same_answer(a: dict, b: dict) -> None:
    """Two v1 answers agree on everything but their request id."""
    for key in ("ranks", "likely", "specimens", "answer", "photos", "gate", "hints"):
        assert a[key] == b[key], key
    assert [p["ranks"] for p in a["per_photo"]] == [p["ranks"] for p in b["per_photo"]]


def test_opening_a_session(conn, tmp_path):
    body = open_session(app(conn, tmp_path), latitude=40.12, longitude=-82.98,
                        observed_on="2026-10-11", client_ref="obs-1")
    assert body["session_id"].startswith("s_") and body["max_photos"] == 10
    assert body["model"]["name"] == "MycoMap Vision" and body["ttl_seconds"] == 1800
    assert body["context_used"]["latitude"] == 40.1 and body["context_used"]["place_from"] == "entered"
    assert body["expires_at"].endswith("Z")


def test_each_photo_gets_its_own_answer_and_the_combined_one_equals_identify(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    first = add(client, sid, 247, "cap")
    assert first.status_code == 200, first.text
    one = first.json()
    assert one["photo"]["photo_ref"] == "cap" and one["photo"]["index"] == 0
    assert one["photos_scored"] == 1 and one["combined"]["ranks"]["species"][0]["name"] == "A x"
    assert (one["timing"]["photos_embedded"], one["timing"]["photos_reused"]) == (1, 0)
    # The combined answer carries the call's timing, like any v1 answer (the app reads it).
    assert one["combined"]["timing"] == one["timing"] and one["session"]["session_id"] == sid
    assert "usage" in one["combined"]
    two = add(client, sid, 12, "gills").json()
    assert two["photos_scored"] == 2 and two["photo"]["photo_ref"] == "gills"
    # The stored photo is scored again from its vector, never embedded again.
    assert (two["timing"]["photos_embedded"], two["timing"]["photos_reused"]) == (1, 1)
    assert [p["photo_ref"] for p in two["combined"]["per_photo"]] == ["cap", "gills"]
    together = identify(client, APP, reds=(247, 12)).json()
    same_answer(two["combined"], together)


def test_a_resent_photo_ref_returns_the_answer_without_embedding_or_counting(conn, tmp_path,
                                                                            monkeypatch):
    monkeypatch.setenv("MV_PHOTOS_PER_MINUTE", "2")
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    first = add(client, sid, 247, "p1").json()
    again = add(client, sid, 247, "p1")
    assert again.status_code == 200
    assert again.json()["timing"]["photos_embedded"] == 0
    assert again.json()["photos_scored"] == 1
    same_answer(again.json()["combined"], first["combined"])
    # Two resends cost nothing: a second new photo still fits the 2-a-minute window.
    assert add(client, sid, 245, "p2").status_code == 200
    assert add(client, sid, 240, "p3").json()["code"] == "rate_limited"


def test_removing_a_photo_rescores_the_rest(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    add(client, sid, 247, "red")
    add(client, sid, 12, "blue")
    res = client.delete(f"/api/v1/sessions/{sid}/photos/red", headers=APP)
    assert res.status_code == 200 and res.json()["photos_scored"] == 1
    assert res.json()["timing"]["photos_embedded"] == 0
    same_answer(res.json()["combined"], identify(client, APP, reds=(12,)).json())
    assert client.delete(f"/api/v1/sessions/{sid}/photos/red", headers=APP).status_code == 404
    last = client.delete(f"/api/v1/sessions/{sid}/photos/blue", headers=APP).json()
    assert last["combined"] is None and last["photos_scored"] == 0


def test_setting_the_place_later_rescores_without_embedding(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    add(client, sid, 247, "p1")
    res = client.patch(f"/api/v1/sessions/{sid}", headers=APP,
                       json={"latitude": 44.98, "longitude": -93.27, "observed_on": "2026-10-11"})
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["context_used"]["latitude"] == 45.0
    assert body["combined"]["context_used"]["observed_on"] == "2026-10-11"
    assert body["timing"]["photos_embedded"] == 0 and body["timing"]["photos_reused"] == 1
    cleared = client.patch(f"/api/v1/sessions/{sid}", headers=APP,
                           json={"latitude": None, "longitude": None}).json()
    assert cleared["context_used"]["latitude"] is None


def test_get_returns_the_current_answer(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    assert client.get(f"/api/v1/sessions/{sid}", headers=APP).json()["combined"] is None
    added = add(client, sid, 247, "p1").json()
    got = client.get(f"/api/v1/sessions/{sid}", headers=APP).json()
    same_answer(got["combined"], added["combined"])


def test_a_closed_expired_or_someone_elses_session_is_gone(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    for res in (client.get(f"/api/v1/sessions/{sid}", headers=OTHER),
                add(client, sid, 247, "p", headers=OTHER),
                client.delete(f"/api/v1/sessions/{sid}", headers=OTHER)):
        assert res.status_code == 404 and res.json()["code"] == "session_gone"
    assert client.delete(f"/api/v1/sessions/{sid}", headers=APP).status_code == 204
    res = client.get(f"/api/v1/sessions/{sid}", headers=APP)
    assert res.status_code == 404 and res.json()["code"] == "session_gone"
    # Idle past 30 minutes, or 2 hours old: gone.
    store = client.app.state.sessions
    now = [0.0]
    store.clock = lambda: now[0]
    sid = open_session(client)["session_id"]
    now[0] = 1801
    assert client.get(f"/api/v1/sessions/{sid}", headers=APP).json()["code"] == "session_gone"
    sid = open_session(client)["session_id"]
    for t in range(1500, 9000, 1500):                 # kept busy, but 2 h is the most
        now[0] = 1801 + t
        client.get(f"/api/v1/sessions/{sid}", headers=APP)
    assert client.get(f"/api/v1/sessions/{sid}", headers=APP).status_code == 404


def test_no_photo_is_kept_only_its_vector(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    add(client, sid, 247, "p1")
    [s] = client.app.state.sessions._by_id.values()
    [photo] = s.photos
    assert set(vars(photo)) == {"ref", "vector", "base", "verdict", "gate_fingerprint", "found"}
    assert isinstance(photo.vector, np.ndarray) and photo.vector.ndim == 1


def test_a_session_holds_ten_photos(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_PHOTOS_PER_MINUTE", "100")
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    for i in range(10):
        assert add(client, sid, 240 + i, f"p{i}").status_code == 200
    res = add(client, sid, 200, "p10")
    assert res.status_code == 400 and res.json()["code"] == "too_many_photos"


@pytest.mark.parametrize("body", [
    {"latitude": 40},
    {"latitude": "40", "longitude": -82},
    {"latitude": 91, "longitude": 0},
    {"observed_on": "11/10/2026"},
    {"client_ref": "x" * 65},
    {"top_k": 0},
    {"model": "nope/nearest"},
])
def test_bad_session_fields_are_refused(conn, tmp_path, body):
    res = app(conn, tmp_path).post("/api/v1/sessions", json=body, headers=APP)
    assert res.status_code == 400 and res.json()["code"] == "bad_request", body


def test_bad_photos_and_refs_are_refused(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    res = client.post(f"/api/v1/sessions/{sid}/photos", headers=APP, data={"photo_ref": "p"},
                      files={"photo": ("x.png", b"not an image", "image/png")})
    assert res.status_code == 415 and res.json()["code"] == "unsupported_image"
    res = add(client, sid, 247, "x" * 65)
    assert res.status_code == 400 and res.json()["code"] == "bad_request"
    res = client.post(f"/api/v1/sessions/{sid}/photos", headers=APP,
                      files={"photo": ("p.png", png(1), "image/png")})
    assert res.status_code == 400                          # no photo_ref
    # A failed photo isn't kept.
    assert client.get(f"/api/v1/sessions/{sid}", headers=APP).json()["photos_scored"] == 0


def test_a_full_store_refuses_new_sessions_while_all_are_active(conn, tmp_path):
    client = app(conn, tmp_path)
    open_session(client)
    client.app.state.sessions.max_bytes = 1
    res = client.post("/api/v1/sessions", json={}, headers=APP)
    assert res.status_code == 503 and res.json()["code"] == "busy"
    assert int(res.headers["retry-after"]) >= 1


def test_sessions_are_off_unless_turned_on(conn, tmp_path):
    client = app(conn, tmp_path, sessions=False)
    res = client.post("/api/v1/sessions", json={}, headers=APP)
    assert res.status_code == 404 and res.json()["code"] == "not_found"


def test_sessions_work_without_the_identify_route(conn, tmp_path):
    client = app(conn, tmp_path, api_v1=False)
    sid = open_session(client)["session_id"]
    assert add(client, sid, 247, "p1").status_code == 200
    assert identify(client, APP).status_code == 404


def test_the_gate_judges_each_photo_once_and_again_only_when_it_changes(conn, tmp_path):
    gate = FakeGate()
    client = app(conn, tmp_path, gate=gate)
    sid = open_session(client)["session_id"]
    body = add(client, sid, 247, "p1").json()
    assert body["combined"]["gate"]["verdict"] == "fungus" and gate.judged == 1
    add(client, sid, 245, "p2")
    assert gate.judged == 2                                 # only the new photo
    client.patch(f"/api/v1/sessions/{sid}", headers=APP, json={"observed_on": "2026-10-11"})
    assert gate.judged == 2                                 # nothing new to judge
    gate.fp = "gate-2"                                      # thresholds changed
    client.patch(f"/api/v1/sessions/{sid}", headers=APP, json={"observed_on": "2026-10-12"})
    assert gate.judged == 4                                 # both stored photos, again
    [s] = client.app.state.sessions._by_id.values()
    assert {p.gate_fingerprint for p in s.photos} == {"gate-2"}


def test_the_store_drops_the_longest_idle_session_when_full():
    now = [0.0]
    store = session_store.Sessions(max_bytes=5000, busy_window=120, clock=lambda: now[0])
    first = store.create("u1", backbone="m", method="x", top_k=5, entered={}, client_ref=None,
                         keep_photos=False)
    now[0] = 500
    second = store.create("u2", backbone="m", method="x", top_k=5, entered={},
                          client_ref=None, keep_photos=False)
    with pytest.raises(session_store.Gone):
        store.get(first.id, "u1")
    assert store.get(second.id, "u2") is second


def test_health_reports_open_sessions_and_their_memory(conn, tmp_path):
    client = app(conn, tmp_path)
    sid = open_session(client)["session_id"]
    add(client, sid, 247, "p1")
    h = client.get("/api/health").json()["sessions"]
    assert h["open"] == 1 and h["photos"] == 1 and 0 < h["bytes"] <= h["max_bytes"]
    assert app(conn, tmp_path / "off", sessions=False).get("/api/health").json()["sessions"] is None


def test_an_unknown_session_is_session_gone_never_not_found(conn, tmp_path):
    # The app reads 404 not_found on these routes as "sessions are switched off here" and
    # session_gone as "open a new one and resend" (Mobile Coordinator, 2026-10-12).
    client = app(conn, tmp_path)
    sid = "s_never-opened"
    for res in (add(client, sid, 247, "p1"),
                client.patch(f"/api/v1/sessions/{sid}", headers=APP, json={}),
                client.get(f"/api/v1/sessions/{sid}", headers=APP),
                client.delete(f"/api/v1/sessions/{sid}/photos/p1", headers=APP),
                client.delete(f"/api/v1/sessions/{sid}", headers=APP)):
        assert res.status_code == 404 and res.json()["code"] == "session_gone", res.request.url


class BatchSensitive(Const):
    """Like a real model on real hardware: a photo's vector moves a little with the size of
    the batch it was embedded in (~1e-4 on the box)."""

    def encode(self, images):
        return super().encode(images) + 1e-3 * len(images)


def test_a_photo_gets_the_same_vector_alone_or_with_others(conn, tmp_path):
    # Found by integration testing (2026-10-12): /identify embedded a request's photos as
    # one batch, a session one at a time, and near-ties swapped. Both now embed singly.
    seed = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", seed.location, "local")
    embed_photos(conn, seed, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    client = TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: BatchSensitive(n),
        limits=open_limits(), background=False, server_key=SERVER_KEY, api_v1=True,
        sessions=True, app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
        fungus_gate=None, note=lambda s: None))
    sid = open_session(client)["session_id"]
    for i, red in enumerate((247, 120, 12)):
        last = add(client, sid, red, f"p{i}").json()
    together = identify(client, APP, reds=(247, 120, 12)).json()
    assert last["combined"]["ranks"] == together["ranks"]           # scores to the last digit
    assert last["combined"]["specimens"] == together["specimens"]


def test_the_shared_headers_are_minted_when_read():
    # However late in a slow run a test starts, its app token is issued then, not at import.
    import base64
    import json
    import time
    body = APP["Authorization"].split(" ", 1)[1][len("mvt."):].split(".")[0]
    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    assert abs(claims["iat"] - time.time()) < 5
