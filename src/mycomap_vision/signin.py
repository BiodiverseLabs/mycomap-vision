"""Sign in with a mycomap.org account.

mycomap.org vouches for the person and this site trusts it. Nobody creates an
account or a password here. It is mycomap.org's sign-in bridge, which it also
uses for its dev site:

    here   GET /auth/signin?returnTo=/x       remembers a nonce in a signed cookie,
                                              sends the browser to mycomap.org
    .org   GET /auth/dev-bridge/authorize     (after sign-in there if needed) signs
           ?nonce=...&aud=<this origin>       {v, sub, aud, nonce, iat, exp[, name]}
                                              with its PRIVATE Ed25519 key, 60 s
    here   GET /auth/dev-bridge/callback      checks it with the PUBLIC key, the
           ?token=<body>.<sig>                audience and the nonce, starts a session

This site holds only the public key: it can check a token but never mint one.
The nonce lives in the requesting browser's cookie, so a captured token is useless
in another browser. Sessions are our own HMAC-signed cookie; no server state.

Settings (see .env.example):
    MV_SIGNIN                  off | identify | all   (default off)
                               identify: only POST /api/identify and the members'
                               routes (MEMBERS_ONLY: the paper draft) need a session
                               all: every page and API route does, except
                               /api/health, /api/me, /auth/*, the API page, the
                               pages people share, the Research section
                               (research_path) and Human Feedback's reads
                               (feedback_read_path)
    MV_PUBLIC_ORIGIN           this site, e.g. https://vision.mycomap.org (the token's aud)
    MV_SIGNIN_ISSUER           default https://mycomap.org
    MV_SIGNIN_PUBLIC_KEY_FILE  the issuer's Ed25519 public key (PEM)
    MV_SESSION_SECRET_FILE     random bytes that sign our cookies (made on the box)
    MV_SESSION_DAYS            default 7
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

from . import config

MODES = ("off", "identify", "all")
# Never for signed-out visitors, even when the site is open (Steve, 2026-10-09: the paper
# draft and the experiment registry are for mycomap.org members only). A path and everything
# under it.
MEMBERS_ONLY = ("/api/paper", "/api/experiments")


def members_only(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in MEMBERS_ONLY)


# Open to anyone even when every page needs sign-in (Steve, 2026-10-12): the developer API
# page (and its old address) and the files the page shell loads. The shell holds code and
# the site's own text only; every page's data comes from /api/, which stays signed-in, and
# the API page reads only the public outline (/api/docs/v1.json). The site sends a
# signed-out visitor to sign in when they move on to any other page (web/src/App.tsx).
PUBLIC_PAGES = ("/research/api", "/api-docs")
# /favicon.ico: browsers ask for it on their own; sent to sign in, it started a second sign-in
# beside the page's and overwrote its nonce ("sign-in token refused: nonce", 2026-10-21).
PUBLIC_FILES = ("/favicon.png", "/favicon.ico", "/mycomap-logo.png", "/og-card.jpg")
PUBLIC_FILE_DIRS = ("/assets/",)
# The pages people share (Steve, 2026-10-20: link previews, linkcards.py). They answer
# with the same shell, carrying the page's card, so Facebook, X, iMessage and Slack can read
# it; the app sends a signed-out person on to sign in, as a full page load would
# (web/src/lib/research.ts mustSignIn), and every answer stays behind /api/.
SHARE_PAGES = ("/", "/about", "/get-involved")


# The Research section, open to anyone (Steve, 2026-10-22): its pages (and the old addresses
# of two of them, which lead there) and the data they load, read-only. What those answers
# may say to someone signed out is cut in research_public.py. MEMBERS_ONLY stays signed-in
# even under /research: the paper draft and the experiment registry (the Experiments page
# opens, and asks a signed-out visitor to sign in to read them).
RESEARCH_PAGES = ("/models", "/data")
RESEARCH_API = ("/api/stats", "/api/models", "/api/scoreboard", "/api/prospective",
                "/api/benchmarks", "/api/early-answers",
                # How it works (a share page, open to anyone): its accuracy summary, never the
                # sealed test (howgood.py), and the daily seconds per photo, averages only.
                "/api/how-good", "/api/speed/daily")
_RESEARCH_RUN = re.compile(r"^/api/scoreboard/\d{1,9}$")


def research_path(method: str, path: str) -> bool:
    """A Research page or the data it loads (GET or HEAD only); never MEMBERS_ONLY."""
    if method not in ("GET", "HEAD") or members_only(path):
        return False
    return (path == "/research" or path.startswith("/research/") or path in RESEARCH_PAGES
            or path in RESEARCH_API or bool(_RESEARCH_RUN.match(path)))


# Human Feedback, read-only to anyone (Steve, 2026-10-22): its page and every read of its API
# pass here; feedback_api.py decides what each viewer sees (a reviewer everything they had,
# anyone else every photo item and the other items of no project, cut by feedback_public.py)
# and refuses every write to
# anyone but a reviewer (401 signed out, 403 signed in).
FEEDBACK_PAGE = "/feedback"


def feedback_read_path(method: str, path: str) -> bool:
    return method in ("GET", "HEAD") and (
        path == FEEDBACK_PAGE or path.startswith(FEEDBACK_PAGE + "/")
        or path.startswith("/api/feedback/"))


# Routes that check their caller themselves: the identification log's read for the paper
# (identifylog.py), for mycomap.org's server key or a signed-in admin.
SELF_CHECKED = ("/api/identify-log/first",)


def public_page(method: str, path: str) -> bool:
    """A page or shell file anyone may read (GET or HEAD only)."""
    return method in ("GET", "HEAD") and (
        path in PUBLIC_PAGES or path in PUBLIC_FILES or path in SHARE_PAGES
        or any(path.startswith(d) for d in PUBLIC_FILE_DIRS) or run_path(path))


# A run's own link (runs.py; Steve, 2026-10-20): its page, its answer, its photo copies, its
# link-preview image (linkcards.py) and its "Where Vision looked" maps once made (looked.py)
# open for anyone who has the link, signed in or not, to read. The id is the only key.
# Nothing else about runs is open: deleting one or asking for its maps needs sign-in, and no
# route lists them.
_RUN_PATH = re.compile(r"^/(?:r/[A-Za-z0-9_-]{16}|api/runs/[A-Za-z0-9_-]{16}"
                       r"(?:/photos/\d{1,2}\.jpg|/card\.jpg|/looked)?)$")


def run_path(path: str) -> bool:
    return bool(_RUN_PATH.match(path))


SESSION_COOKIE = "mv_session"
NONCE_COOKIE = "mv_signin"
NONCE_TTL_SECONDS = 600
CLOCK_SKEW_SECONDS = 30


class SigninError(ValueError):
    """A token or cookie that must not be trusted; `reason` says why."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def origin_of(url: str | None) -> str | None:
    """scheme://host[:port] of an absolute http(s) URL, lower-cased; None otherwise."""
    if not url or not url.strip():
        return None
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        return None
    port = f":{parts.port}" if parts.port else ""
    return f"{parts.scheme}://{parts.hostname}{port}"


def safe_return_to(value: str | None, default: str = "/") -> str:
    """Only a path on this site: never another host, scheme, or protocol-relative URL."""
    if not value or not value.startswith("/") or value.startswith(("//", "/\\")):
        return default
    if any(c in value for c in "\r\n\\") or urlsplit(value).netloc:
        return default
    return value


def load_public_key(pem: str | bytes):
    """An Ed25519 public key from PEM, or SigninError."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
    from cryptography.hazmat.primitives.serialization import load_pem_public_key
    if isinstance(pem, str):
        pem = pem.strip().replace("\\n", "\n").encode("ascii")
    try:
        key = load_pem_public_key(pem)
    except ValueError as e:
        raise SigninError("key") from e
    if not isinstance(key, Ed25519PublicKey):
        raise SigninError("key")
    return key


def verify_token(token: str, public_key, *, aud: str, nonce: str,
                 now: float | None = None) -> dict:
    """The claims of a mycomap.org bridge token, or SigninError. Same rules as .org's
    verifyBridgeToken: signature over the base64url body, v 1, 60 s life with 30 s of
    clock skew, exact audience, constant-time nonce."""
    from cryptography.exceptions import InvalidSignature
    now = time.time() if now is None else now
    parts = token.split(".")
    if len(parts) != 2 or not all(parts):
        raise SigninError("malformed")
    body, sig = parts
    try:
        public_key.verify(_unb64(sig), body.encode("ascii"))
    except (InvalidSignature, ValueError, UnicodeEncodeError):
        raise SigninError("signature")
    try:
        claims = json.loads(_unb64(body))
    except ValueError:
        raise SigninError("malformed")
    if (not isinstance(claims, dict) or claims.get("v") != 1
            or not isinstance(claims.get("sub"), str) or not claims["sub"]
            or not isinstance(claims.get("exp"), (int, float))
            or not isinstance(claims.get("iat", 0), (int, float))
            or not isinstance(claims.get("nonce"), str)):
        raise SigninError("malformed")
    if claims["exp"] < now or claims.get("iat", 0) > now + CLOCK_SKEW_SECONDS:
        raise SigninError("expired")
    if claims.get("aud") != aud:
        raise SigninError("audience")
    if not hmac.compare_digest(claims["nonce"].encode(), nonce.encode()):
        raise SigninError("nonce")
    return claims


class CookieSigner:
    """base64url(JSON).base64url(HMAC-SHA256): tamper-evident, with its own expiry."""

    def __init__(self, secret: bytes, purpose: str):
        if len(secret) < 32:
            raise ValueError("the session secret must be at least 32 bytes")
        # One key per purpose, so a nonce cookie can never pass as a session.
        self.key = hmac.new(secret, purpose.encode(), hashlib.sha256).digest()

    def sign(self, payload: dict) -> str:
        body = _b64(json.dumps(payload, separators=(",", ":")).encode())
        mac = hmac.new(self.key, body.encode(), hashlib.sha256).digest()
        return f"{body}.{_b64(mac)}"

    def read(self, value: str | None, now: float | None = None) -> dict | None:
        """The payload if the signature holds and it has not expired, else None."""
        if not value or value.count(".") != 1:
            return None
        body, mac = value.split(".")
        want = hmac.new(self.key, body.encode(), hashlib.sha256).digest()
        try:
            if not hmac.compare_digest(_unb64(mac), want):
                return None
            payload = json.loads(_unb64(body))
        except ValueError:
            return None
        now = time.time() if now is None else now
        if not isinstance(payload, dict) or not isinstance(payload.get("exp"), (int, float)) \
                or payload["exp"] < now:
            return None
        return payload


@dataclass
class SigninConfig:
    mode: str = "off"
    origin: str | None = None           # this site; the audience tokens must carry
    issuer: str = "https://mycomap.org"
    public_key: object | None = None
    secret: bytes | None = None
    session_days: float = 7.0
    dataset_public: bool = False        # MV_DATASET_PUBLIC: CC dataset images need no sign-in

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"MV_SIGNIN must be one of {', '.join(MODES)}, not {self.mode!r}")
        if self.mode == "off":
            return
        # Fail closed: a site that should require sign-in never starts without it.
        missing = [n for n, v in (("MV_PUBLIC_ORIGIN", self.origin),
                                  ("MV_SIGNIN_PUBLIC_KEY_FILE", self.public_key),
                                  ("MV_SESSION_SECRET_FILE", self.secret)) if not v]
        if missing:
            raise RuntimeError(f"MV_SIGNIN={self.mode} needs {', '.join(missing)}")
        self.sessions = CookieSigner(self.secret, "session")
        self.nonces = CookieSigner(self.secret, "signin-nonce")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def secure_cookies(self) -> bool:
        return bool(self.origin and self.origin.startswith("https://"))

    @classmethod
    def from_settings(cls) -> "SigninConfig":
        mode = (config.setting("MV_SIGNIN") or "off").lower()
        if mode == "off":
            return cls()
        key_file = config.setting("MV_SIGNIN_PUBLIC_KEY_FILE")
        secret_file = config.setting("MV_SESSION_SECRET_FILE")
        return cls(
            mode=mode,
            origin=origin_of(config.setting("MV_PUBLIC_ORIGIN")),
            issuer=origin_of(config.setting("MV_SIGNIN_ISSUER")) or "https://mycomap.org",
            public_key=load_public_key(Path(key_file).read_bytes()) if key_file else None,
            secret=Path(secret_file).read_bytes().strip() if secret_file else None,
            session_days=float(config.setting("MV_SESSION_DAYS") or 7),
            dataset_public=(config.setting("MV_DATASET_PUBLIC") or "").lower()
            in ("1", "true", "yes"),
        )

    def needs_session(self, method: str, path: str) -> bool:
        """Whether this request may only be answered for a signed-in person."""
        if self.mode == "off" or path in ("/api/health", "/api/me") or path.startswith("/auth/"):
            return False
        # The public API answers its own credentials, never the site's cookie (apiv1.py),
        # and its outline is public (apidocs.py; mycomap.org/api-docs shows it too).
        if path.startswith("/api/v1/") or path == "/api/v1" or path.startswith("/api/docs/"):
            return False
        if public_page(method, path) or research_path(method, path):
            return False
        if feedback_read_path(method, path):
            return False
        # Checks its own caller (mycomap.org's server key, or a signed-in admin), so a
        # server-key call without a cookie reaches it.
        if path in SELF_CHECKED:
            return False
        # The public dataset release's CC images (dataset_api.py), when Steve opens them.
        if self.dataset_public and method in ("GET", "HEAD") and path.startswith("/api/dataset/"):
            return False
        if self.mode == "identify":
            return (method == "POST" and path in ("/api/identify", "/api/identify/observation")) or members_only(path)
        return True

    def authorize_url(self, nonce: str) -> str:
        return (f"{self.issuer}/auth/dev-bridge/authorize?nonce={quote(nonce)}"
                f"&aud={quote(self.origin or '', safe='')}")
