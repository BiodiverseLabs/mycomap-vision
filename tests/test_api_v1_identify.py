"""POST /api/v1/identify and the v1 credentials (docs/api.md §1, §3.1, §4, §5, §6)."""

import time

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from test_api import open_limits, png, post_photos
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY, good_claims, mint

from mycomap_vision import apiv1
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninError, verify_token

SERVER_KEY = "s" * 40
DEV_KEY = "vision_" + "d" * 43


def app_token(**over) -> str:
    claims = {"typ": "vision-api", "sub": "u_member1", "aud": HERE, "plan": "member",
              "limit": {"monthly": 300}, **over}
    return "mvt." + mint(claims)


def bearer(token):
    return {"Authorization": f"Bearer {token}"}


class FakeOrg:
    """mycomap.org's introspection, played in the test: answers per token, or fails."""

    def __init__(self):
        self.answers = {DEV_KEY: {"active": True, "service": "vision", "keyId": "42",
                                  "tier": "standard", "plan": "member",
                                  "limit": {"monthly": 2000}, "limit_source": "plan",
                                  "subject": "u_member1"}}
        self.calls = 0
        self.down = False

    def __call__(self, token):
        self.calls += 1
        if self.down:
            raise ConnectionError("mycomap.org unreachable")
        return self.answers.get(token, {"active": False, "reason": "unknown"})


def built_site(tmp_path):
    """A web/dist as the box has one: the site's catch-all route is then mounted."""
    dist = tmp_path / "dist"
    dist.mkdir()
    (dist / "index.html").write_text("<!doctype html><title>site</title>")
    return dist


def v1_app(conn, tmp_path, monkeypatch=None, api_v1=True, org=None, photos_per_minute=None,
           web_dist=None):
    if monkeypatch and photos_per_minute:
        monkeypatch.setenv("MV_PHOTOS_PER_MINUTE", str(photos_per_minute))
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), background=False, server_key=SERVER_KEY,
                     api_v1=api_v1, app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE),
                     introspect=org or FakeOrg(), note=lambda s: None, web_dist=web_dist)
    return TestClient(app)


def identify(client, headers, reds=(247,), **fields):
    files = [("photos", (f"p{i}.png", png(r), "image/png")) for i, r in enumerate(reds)]
    return client.post("/api/v1/identify", files=files, data=fields, headers=headers)


# --- the answer -----------------------------------------------------------------------

def test_an_app_token_gets_one_models_answer_with_timing_and_every_rank(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    res = identify(client, bearer(app_token()), reds=(247, 245))
    assert res.status_code == 200, res.text
    body = res.json()
    assert body["request_id"].startswith("r_")
    assert body["model"] == {"name": "MycoMap Vision", "version": None, "calibrated_on": None}
    assert body["prior"] == {"applied": False}
    assert "calibration" not in body                    # internal; not part of v1
    assert body["photos"] == 2 and len(body["per_photo"]) == 2
    assert body["ranks"]["species"][0]["name"] == "A x"
    for rank in ("species", "genus", "family"):
        assert body["per_photo"][0]["ranks"][rank]
    assert body["usage"] is None and body["gate"] is None
    t = body["timing"]
    for name in ("queue_ms", "decode_ms", "embed_ms", "score_ms", "model_ms", "total_ms"):
        assert isinstance(t[name], int) and t[name] >= 0, name
    assert t["model_ms"] == t["embed_ms"] + t["score_ms"]
    assert t["total_ms"] >= t["model_ms"]
    assert t["fetch_ms"] is None and t["photos_embedded"] == 2 and t["photos_reused"] == 0
    assert "embed;dur=" in res.headers["server-timing"]
    assert res.headers["cache-control"] == "no-store"


def test_confident_and_answer_follow_the_70_percent_rule(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    body = identify(client, bearer(app_token())).json()
    for rank, cands in body["ranks"].items():
        for i, c in enumerate(cands):
            assert c["confident"] == (i == 0 and c["confidence"] >= 0.70), (rank, c)
    finest = [r for r in ("species", "genus", "family") if body["ranks"][r][0]["confident"]]
    assert body["answer"] == (None if not finest else
                              {"rank": finest[0], "name": body["ranks"][finest[0]][0]["name"],
                               "confidence": body["ranks"][finest[0]][0]["confidence"]})


def test_place_and_date_are_echoed_rounded(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    body = identify(client, bearer(app_token()), latitude="40.123", longitude="-82.987",
                    observed_on="2026-10-11").json()
    assert body["context_used"]["latitude"] == 40.1
    assert body["context_used"]["longitude"] == -83.0
    assert body["context_used"]["observed_on"] == "2026-10-11"


def test_the_websites_identify_also_reports_timing(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    res = post_photos(client, [247], "m1/nearest")
    assert res.status_code == 200
    t = res.json()["timing"]
    assert t["model_ms"] == t["embed_ms"] + t["score_ms"] and t["photos_embedded"] == 1


# --- refusals -------------------------------------------------------------------------

@pytest.mark.parametrize("fields, reds, code, status", [
    ({}, (1,) * 11, "too_many_photos", 400),
    ({"top_k": "0"}, (1,), "bad_request", 400),
    ({"top_k": "11"}, (1,), "bad_request", 400),
    ({"latitude": "40"}, (1,), "bad_request", 400),
    ({"latitude": "91", "longitude": "0"}, (1,), "bad_request", 400),
    ({"observed_on": "11/10/2026"}, (1,), "bad_request", 400),
    ({"model": "nope/nearest"}, (1,), "bad_request", 400),
    ({"model": "m1/magic"}, (1,), "bad_request", 400),
])
def test_bad_requests_are_refused_with_a_code(conn, tmp_path, fields, reds, code, status):
    client = v1_app(conn, tmp_path)
    res = identify(client, bearer(app_token()), reds=reds, **fields)
    assert res.status_code == status and res.json()["code"] == code, res.text


def test_a_file_that_isnt_an_image_is_415_and_no_photos_is_400(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    res = client.post("/api/v1/identify", headers=bearer(app_token()),
                      files=[("photos", ("x.png", b"not an image", "image/png"))])
    assert res.status_code == 415 and res.json()["code"] == "unsupported_image"
    res = client.post("/api/v1/identify", headers=bearer(app_token()), data={"top_k": "5"})
    assert res.status_code == 400 and res.json()["code"] == "bad_request"


def test_v1_identify_is_off_unless_turned_on(conn, tmp_path):
    # With the built site mounted, as on the box: its catch-all must not answer v1 paths.
    client = v1_app(conn, tmp_path, api_v1=False, web_dist=built_site(tmp_path))
    res = identify(client, bearer(app_token()))
    assert res.status_code == 404 and res.json()["code"] == "not_found"


def test_an_unknown_v1_path_is_a_json_404_by_any_method(conn, tmp_path):
    client = v1_app(conn, tmp_path, web_dist=built_site(tmp_path))
    for method in ("get", "post", "put", "patch", "delete"):
        res = client.request(method.upper(), "/api/v1/no-such-route", headers=bearer(app_token()))
        assert res.status_code == 404 and res.json()["code"] == "not_found", method
    # Still a credential first: no route list for strangers.
    assert client.post("/api/v1/no-such-route").status_code == 401
    # The site itself is untouched.
    assert client.get("/some/page").text.startswith("<!doctype html>")


# --- app tokens -----------------------------------------------------------------------

def test_app_tokens_refused_when_expired_forged_or_for_another_site(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    now = int(time.time())
    expired = identify(client, bearer(app_token(iat=now - 1000, exp=now - 100)))
    assert expired.status_code == 401 and expired.json()["code"] == "token_expired"
    other_key = "mvt." + mint({"typ": "vision-api", "sub": "u", "aud": HERE},
                              key=Ed25519PrivateKey.generate())
    for token in (other_key, app_token(aud="https://elsewhere.example"),
                  app_token(typ="session"), app_token(sub=""), "mvt.garbage", "mvt.a.b"):
        res = identify(client, bearer(token))
        assert res.status_code == 401 and res.json()["code"] == "invalid_token", token


def test_a_sign_in_bridge_token_is_not_an_app_token_and_back(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    bridge = mint(good_claims())                     # what the sign-in callback takes
    assert identify(client, bearer("mvt." + bridge)).status_code == 401
    assert identify(client, bearer(bridge)).status_code == 401
    # An app token has no nonce, so it can never finish a sign-in.
    body_and_sig = app_token()[len("mvt."):]
    with pytest.raises(SigninError):
        verify_token(body_and_sig, PUBLIC_KEY, aud=HERE, nonce="n" * 32)


def test_mycomap_orgs_server_key_may_identify_too(conn, tmp_path):
    client = v1_app(conn, tmp_path)
    assert identify(client, bearer(SERVER_KEY)).status_code == 200


# --- developer tokens through .org's introspection -----------------------------------

def test_an_active_developer_token_is_accepted_and_cached(conn, tmp_path):
    org = FakeOrg()
    client = v1_app(conn, tmp_path, org=org)
    assert identify(client, bearer(DEV_KEY)).status_code == 200
    assert identify(client, {"X-API-Key": DEV_KEY}).status_code == 200
    assert org.calls == 1                       # the second answer came from the cache


@pytest.mark.parametrize("answer, status, code", [
    ({"active": False, "reason": "revoked"}, 403, "key_inactive"),
    ({"active": False, "reason": "pending"}, 403, "key_inactive"),
    ({"active": False, "reason": "unknown"}, 401, "invalid_token"),
    ({"active": True, "service": "atlas", "keyId": "7"}, 403, "key_inactive"),
])
def test_inactive_or_foreign_developer_tokens_are_refused(conn, tmp_path, answer, status, code):
    org = FakeOrg()
    org.answers[DEV_KEY] = answer
    res = identify(v1_app(conn, tmp_path, org=org), bearer(DEV_KEY))
    assert res.status_code == status and res.json()["code"] == code


def test_when_org_is_down_a_recent_good_answer_holds_and_an_unknown_token_waits():
    clock = [0.0]
    org = FakeOrg()
    intro = apiv1.Introspector(org, clock=lambda: clock[0], note=lambda s: None)
    assert intro.check(DEV_KEY).key_id == "42"
    org.down = True
    clock[0] = 300                                 # past the 60 s cache, within 10 min
    assert intro.check(DEV_KEY).key_id == "42"
    clock[0] = 700                                 # past 10 minutes: no longer trusted
    with pytest.raises(apiv1.V1Error) as e:
        intro.check(DEV_KEY)
    assert e.value.status == 503 and e.value.code == "auth_unavailable"
    with pytest.raises(apiv1.V1Error) as e:
        intro.check("vision_never-seen")
    assert e.value.status == 503 and e.value.retry_after == 30


def test_a_negative_answer_is_asked_again_after_ten_seconds():
    clock = [0.0]
    org = FakeOrg()
    org.answers[DEV_KEY] = {"active": False, "reason": "pending"}
    intro = apiv1.Introspector(org, clock=lambda: clock[0], note=lambda s: None)
    with pytest.raises(apiv1.V1Error):
        intro.check(DEV_KEY)
    org.answers[DEV_KEY] = FakeOrg().answers[DEV_KEY]       # approved on .org
    clock[0] = 11
    assert intro.check(DEV_KEY).subject == "u_member1"
    assert org.calls == 2


# --- the short-window limit, per credential, never per address ------------------------

def test_photos_per_minute_count_per_member_not_per_address(conn, tmp_path, monkeypatch):
    client = v1_app(conn, tmp_path, monkeypatch, photos_per_minute=3)
    assert identify(client, bearer(app_token()), reds=(1, 2)).status_code == 200
    # The same member through a developer token shares the pool (Steve, 2026-10-11).
    res = identify(client, bearer(DEV_KEY), reds=(1, 2))
    assert res.status_code == 429 and res.json()["code"] == "rate_limited"
    assert int(res.headers["retry-after"]) >= 1
    # Another member on the same address is not held back.
    assert identify(client, bearer(app_token(sub="u_member2")), reds=(1, 2)).status_code == 200


def test_a_key_with_its_own_override_has_its_own_pool():
    member = apiv1.Credential("app", subject="u1")
    pooled = apiv1.Credential("key", subject="u1", key_id="42", limit_source="plan")
    own = apiv1.Credential("key", subject="u1", key_id="43", limit_source="override")
    assert member.rate_key == pooled.rate_key == "sub:u1"
    assert own.rate_key == "key:43"
    assert apiv1.Credential("key", key_id="44").rate_key == "key:44"


def test_the_photo_window_waits_for_the_oldest_photos_to_leave():
    clock = [0.0]
    w = apiv1.PhotoWindow(5, 60, clock=lambda: clock[0])
    assert w.take("a", 3) == 0
    clock[0] = 10
    assert w.take("a", 2) == 0
    clock[0] = 20
    assert w.take("a", 1) == pytest.approx(40)       # the first 3 leave at t=60
    assert w.take("a", 6) == 60                      # never fits
    clock[0] = 61
    assert w.take("a", 3) == 0


# --- the confident rule, on its own ---------------------------------------------------

def result_with(conf_species, conf_genus, conf_family, gate=None):
    def rank(name, c):
        return [{"name": name, "confidence": c}, {"name": name + "2", "confidence": 0.01}]
    return {"ranks": {"species": rank("A x", conf_species), "genus": rank("A", conf_genus),
                      "family": rank("F", conf_family)}, "gate": gate}


def test_answer_is_the_finest_rank_at_70_percent():
    r = apiv1.mark_confident(result_with(0.69, 0.9, 0.99))
    assert r["answer"] == {"rank": "genus", "name": "A", "confidence": 0.9}
    assert [c["confident"] for c in r["ranks"]["species"]] == [False, False]
    assert apiv1.mark_confident(result_with(0.70, 0.9, 0.99))["answer"]["rank"] == "species"
    assert apiv1.mark_confident(result_with(0.1, 0.2, 0.3))["answer"] is None


def test_nothing_is_confident_unless_the_gate_says_fungus():
    for verdict in ("unsure", "not_fungus"):
        r = apiv1.mark_confident(result_with(0.99, 0.99, 0.99, gate={"verdict": verdict}))
        assert r["answer"] is None
    assert apiv1.mark_confident(result_with(0.99, 0.9, 0.9, gate={"verdict": "fungus"})
                                )["answer"]["rank"] == "species"
    # A withheld confidence is never confident.
    assert apiv1.mark_confident(result_with(None, None, None))["answer"] is None
