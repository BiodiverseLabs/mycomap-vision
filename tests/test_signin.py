import base64
import json
import time
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi.testclient import TestClient

from test_api import open_limits, post_photos
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import (NONCE_COOKIE, SESSION_COOKIE, CookieSigner, SigninConfig,
                                   SigninError, load_public_key, safe_return_to, verify_token)

HERE = "http://testserver"
ISSUER = "https://mycomap.org"
SECRET = b"s" * 48
ISSUER_KEY = Ed25519PrivateKey.generate()
PUBLIC_PEM = ISSUER_KEY.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)


def b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def mint(claims: dict, key=ISSUER_KEY, **over) -> str:
    """A token laid out exactly as mycomap.org's signBridgeToken makes it."""
    now = int(time.time())
    full = {"v": 1, "iat": now, "exp": now + 60, **claims, **over}
    body = b64(json.dumps(full, separators=(",", ":")).encode())
    return f"{body}.{b64(key.sign(body.encode()))}"


def good_claims(nonce="n" * 32, aud=HERE):
    return {"sub": "user-42", "aud": aud, "nonce": nonce, "name": "Mycena Fan"}


PUBLIC_KEY = load_public_key(PUBLIC_PEM)


# --- the token ------------------------------------------------------------------

def test_a_token_signed_by_mycomap_org_for_this_site_and_nonce_is_accepted():
    claims = verify_token(mint(good_claims()), PUBLIC_KEY, aud=HERE, nonce="n" * 32)
    assert claims["sub"] == "user-42" and claims["name"] == "Mycena Fan"


@pytest.mark.parametrize("token, reason", [
    (lambda: mint(good_claims(aud="https://dev.mycomap.org")), "audience"),
    (lambda: mint(good_claims(aud="http://testserver.evil.com")), "audience"),
    (lambda: mint(good_claims(nonce="m" * 32)), "nonce"),
    (lambda: mint(good_claims(), exp=int(time.time()) - 1), "expired"),
    (lambda: mint(good_claims(), iat=int(time.time()) + 120), "expired"),
    (lambda: mint(good_claims(), key=Ed25519PrivateKey.generate()), "signature"),
    (lambda: mint(good_claims(), v=2), "malformed"),
    (lambda: mint({**good_claims(), "sub": ""}), "malformed"),
    (lambda: "only-one-part", "malformed"),
    (lambda: "a.b.c", "malformed"),
])
def test_tokens_for_another_site_nonce_time_or_key_are_refused(token, reason):
    with pytest.raises(SigninError) as e:
        verify_token(token(), PUBLIC_KEY, aud=HERE, nonce="n" * 32)
    assert e.value.reason == reason


def test_a_token_whose_claims_were_edited_after_signing_is_refused():
    body, sig = mint(good_claims()).split(".")
    claims = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
    claims["sub"] = "admin"
    forged = b64(json.dumps(claims).encode()) + "." + sig
    with pytest.raises(SigninError) as e:
        verify_token(forged, PUBLIC_KEY, aud=HERE, nonce="n" * 32)
    assert e.value.reason == "signature"


def test_only_an_ed25519_public_key_is_accepted_as_the_issuer_key():
    from cryptography.hazmat.primitives.asymmetric.rsa import generate_private_key
    rsa = generate_private_key(public_exponent=65537, key_size=2048).public_key()
    with pytest.raises(SigninError):
        load_public_key(rsa.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo))
    assert load_public_key(PUBLIC_PEM.decode().replace("\n", "\\n"))  # env-file style


# --- cookies and redirects --------------------------------------------------------

@pytest.mark.parametrize("value", ["//evil.com/x", "https://evil.com", "/\\evil.com",
                                   "evil.com", "", None, "/a\r\nSet-Cookie: x=1"])
def test_return_to_never_leaves_this_site(value):
    assert safe_return_to(value) == "/"


def test_return_to_keeps_a_path_and_query_on_this_site():
    assert safe_return_to("/identify?x=1") == "/identify?x=1"


def test_a_signed_cookie_that_was_edited_or_expired_is_ignored():
    s = CookieSigner(SECRET, "session")
    good = s.sign({"sub": "u", "exp": time.time() + 60})
    assert s.read(good)["sub"] == "u"
    body, mac = good.split(".")
    assert s.read(b64(b'{"sub":"admin","exp":9999999999}') + "." + mac) is None
    assert s.read(s.sign({"sub": "u", "exp": time.time() - 1})) is None
    assert s.read("garbage") is None and s.read(None) is None


def test_a_nonce_cookie_can_never_pass_as_a_session():
    nonce_cookie = CookieSigner(SECRET, "signin-nonce").sign({"sub": "u", "exp": time.time() + 60})
    assert CookieSigner(SECRET, "session").read(nonce_cookie) is None


def test_requiring_sign_in_without_its_key_or_secret_refuses_to_start():
    with pytest.raises(RuntimeError, match="MV_SIGNIN_PUBLIC_KEY_FILE"):
        SigninConfig(mode="all", origin=HERE, secret=SECRET)
    with pytest.raises(RuntimeError, match="MV_SESSION_SECRET_FILE"):
        SigninConfig(mode="all", origin=HERE, public_key=PUBLIC_KEY)
    with pytest.raises(ValueError):
        SigninConfig(mode="maybe")


# --- the site ---------------------------------------------------------------------

def site(conn, tmp_path, mode, origin=HERE):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    signin = SigninConfig(mode=mode, origin=origin, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    app = create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                     limits=open_limits(), signin=signin)
    return TestClient(app, follow_redirects=False)


def sign_in(client, return_to="/"):
    """The whole round trip, with mycomap.org played by `mint`."""
    start = client.get("/auth/signin", params={"returnTo": return_to})
    assert start.status_code == 302
    q = parse_qs(urlsplit(start.headers["location"]).query)
    return client.get("/auth/dev-bridge/callback",
                      params={"token": mint(good_claims(nonce=q["nonce"][0]))})


def test_with_the_whole_site_gated_nothing_but_health_answers_before_sign_in(conn, tmp_path):
    c = site(conn, tmp_path, "all")
    assert c.get("/api/health").status_code == 200
    assert c.get("/api/me").json() == {"signin": "all", "user": None}
    r = c.get("/api/stats")
    assert r.status_code == 401 and r.json()["signin"] == "/auth/signin"
    assert c.get("/api/scoreboard").status_code == 401
    assert post_photos(c, [247], "m1/nearest").status_code == 401
    page = c.get("/models?x=1")
    assert page.status_code == 302
    assert page.headers["location"] == "/auth/signin?returnTo=%2Fmodels%3Fx%3D1"


def test_sign_in_goes_to_mycomap_org_for_this_site_and_comes_back_signed_in(conn, tmp_path):
    c = site(conn, tmp_path, "all")
    start = c.get("/auth/signin", params={"returnTo": "/models"})
    to = urlsplit(start.headers["location"])
    assert f"{to.scheme}://{to.netloc}{to.path}" == f"{ISSUER}/auth/dev-bridge/authorize"
    q = parse_qs(to.query)
    assert q["aud"] == [HERE] and len(q["nonce"][0]) >= 22
    back = c.get("/auth/dev-bridge/callback",
                 params={"token": mint(good_claims(nonce=q["nonce"][0]))})
    assert back.status_code == 302 and back.headers["location"] == "/models"
    assert c.get("/api/stats").status_code == 200
    assert c.get("/api/me").json()["user"] == {"id": "user-42", "name": "Mycena Fan"}
    assert post_photos(c, [247], "m1/nearest").status_code == 200


def test_sign_in_never_sends_the_browser_off_site_afterwards(conn, tmp_path):
    c = site(conn, tmp_path, "all")
    back = sign_in(c, return_to="//evil.com/x")
    assert back.headers["location"] == "/"


def test_a_token_is_useless_in_a_browser_that_did_not_start_the_sign_in(conn, tmp_path):
    c = site(conn, tmp_path, "all")
    start = c.get("/auth/signin")
    nonce = parse_qs(urlsplit(start.headers["location"]).query)["nonce"][0]
    other = TestClient(c.app, follow_redirects=False)
    r = other.get("/auth/dev-bridge/callback", params={"token": mint(good_claims(nonce=nonce))})
    assert r.status_code == 401 and SESSION_COOKIE not in r.cookies
    assert other.get("/api/stats").status_code == 401


def test_a_token_minted_for_another_site_does_not_sign_anyone_in(conn, tmp_path):
    c = site(conn, tmp_path, "all")
    start = c.get("/auth/signin")
    nonce = parse_qs(urlsplit(start.headers["location"]).query)["nonce"][0]
    r = c.get("/auth/dev-bridge/callback",
              params={"token": mint(good_claims(nonce=nonce, aud="https://dev.mycomap.org"))})
    assert r.status_code == 401
    assert c.get("/api/stats").status_code == 401


def test_a_forged_session_cookie_is_not_a_session(conn, tmp_path):
    c = site(conn, tmp_path, "all")
    forged = CookieSigner(b"x" * 48, "session").sign({"sub": "u", "exp": time.time() + 60})
    c.cookies.set(SESSION_COOKIE, forged)
    assert c.get("/api/stats").status_code == 401


def test_signing_out_ends_the_session(conn, tmp_path):
    c = site(conn, tmp_path, "all")
    sign_in(c)
    assert c.get("/api/stats").status_code == 200
    assert c.post("/auth/signout").status_code == 200
    assert c.get("/api/stats").status_code == 401


def test_identify_mode_leaves_the_site_open_and_gates_only_identification(conn, tmp_path):
    c = site(conn, tmp_path, "identify")
    assert c.get("/api/stats").status_code == 200
    assert c.get("/api/scoreboard").status_code == 200
    assert post_photos(c, [247], "m1/nearest").status_code == 401
    sign_in(c)
    assert post_photos(c, [247], "m1/nearest").status_code == 200


def test_with_sign_in_off_everything_is_open_and_there_is_no_sign_in_route(conn, tmp_path):
    c = site(conn, tmp_path, "off")
    assert c.get("/api/stats").status_code == 200
    assert post_photos(c, [247], "m1/nearest").status_code == 200
    assert c.get("/auth/signin").status_code == 404
    assert c.get("/api/me").json() == {"signin": "off", "user": None}


def test_on_an_https_site_the_cookies_are_secure_and_http_only(conn, tmp_path):
    c = site(conn, tmp_path, "all", origin="https://vision.mycomap.org")
    start = c.get("/auth/signin")
    cookie = start.headers["set-cookie"].lower()
    assert cookie.startswith(NONCE_COOKIE) and "secure" in cookie and "httponly" in cookie
    assert "samesite=lax" in cookie
