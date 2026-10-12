"""The public API, v1 (docs/api.md): credentials, refusals, and the answers built for it.

Every /api/v1/ path is answered only with a credential (`Authenticator`); the website's
cookie sign-in doesn't apply there (signin.needs_session). A refusal is
`{"code", "detail"}` with its status, plus `Retry-After` when trying again later can work.

Credentials (docs/api.md §1, §6):
- mycomap.org's own server key (VISION_API_KEY on .org, MV_ORG_VISION_KEY here), for
  server-to-server reads such as the taxon counts;
- app tokens `mvt.<body>.<sig>` that mycomap.org mints for a signed-in member, signed with
  the sign-in bridge's Ed25519 key and checked here offline (`AppTokens`);
- developer tokens `vision_…` from the .org key manager, checked through .org's
  service-key introspection and cached (`Introspector`).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable

from fastapi.responses import JSONResponse

PREFIX = "/api/v1/"
RANK_KEYS = {"species": "species", "genus": "genera", "family": "families"}
APP_TOKEN_PREFIX = "mvt."
KEY_PREFIX = "vision_"
CLOCK_SKEW_SECONDS = 30


def is_v1(path: str) -> bool:
    return path.startswith(PREFIX) or path == PREFIX.rstrip("/")


class V1Error(Exception):
    """A refusal, as docs/api.md §5 lists them."""

    def __init__(self, status: int, code: str, detail: str,
                 retry_after: int | None = None, **extra):
        super().__init__(detail)
        self.status, self.code, self.detail = status, code, detail
        self.retry_after, self.extra = retry_after, extra

    def response(self) -> JSONResponse:
        body = {"code": self.code, "detail": self.detail, **self.extra}
        headers = {"Cache-Control": "no-store"}
        if self.retry_after is not None:
            body["retry_after_seconds"] = self.retry_after
            headers["Retry-After"] = str(self.retry_after)
        return JSONResponse(body, status_code=self.status, headers=headers)


@dataclass(frozen=True)
class Credential:
    kind: str                       # "server" (mycomap.org) | "app" (mvt.) | "key" (vision_)
    subject: str | None = None      # whose quota: .org's opaque member id; None for the server
    key_id: str | None = None       # a developer token's id on .org
    plan: str | None = None
    limit: dict | None = field(default=None, hash=False, compare=False)
    limit_source: str | None = None

    @property
    def interactive(self) -> bool:
        """Ahead in the queue (Steve, 2026-10-11): the app's calls, not developer ones."""
        return self.kind in ("app", "server")

    @property
    def rate_key(self) -> str:
        """Whose short-window limit: a key with its own override counts alone; otherwise
        the member's pool (Steve, 2026-10-11)."""
        if self.kind == "key" and (self.limit_source == "override" or not self.subject):
            return f"key:{self.key_id}"
        return f"sub:{self.subject}" if self.subject else self.kind


def bearer(headers) -> str | None:
    """The token from `Authorization: Bearer …` (or `X-API-Key`), else None."""
    auth = headers.get("authorization") or ""
    scheme, _, token = auth.partition(" ")
    if scheme.lower() == "bearer" and token.strip():
        return token.strip()
    key = (headers.get("x-api-key") or "").strip()
    return key or None


def same_secret(given: str, expected: str) -> bool:
    """Constant-time, whatever the lengths (both are hashed first)."""
    return hmac.compare_digest(hashlib.sha256(given.encode()).digest(),
                               hashlib.sha256(expected.encode()).digest())


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def limit_of(source: dict) -> dict | None:
    """The monthly limit a token or an introspection answer gives: {"monthly": n}, or
    {"monthly": None} for unlimited (an explicit null, e.g. staff), or None when it gives
    none at all (quotas.py then applies its default)."""
    if "limit" not in source:
        return None
    limit = source["limit"]
    if limit is None:
        return {"monthly": None}
    if isinstance(limit, bool):
        return None
    if isinstance(limit, int):
        return {"monthly": limit}
    if isinstance(limit, dict) and "monthly" in limit:
        m = limit["monthly"]
        return {**limit, "monthly": m if m is None or (isinstance(m, int)
                                                        and not isinstance(m, bool)) else 0}
    return None


class AppTokens:
    """App tokens: `mvt.<base64url body>.<base64url Ed25519 signature over the body>`, made
    by mycomap.org with the sign-in bridge's key (docs/api.md §6.3). Only a body with
    typ "vision-api" and this site as its exact audience is accepted, so a bridge sign-in
    token (no typ) never works here; these carry no nonce, so they never finish a
    sign-in (signin.verify_token requires one)."""

    def __init__(self, public_key, audience: str | None,
                 clock: Callable[[], float] = time.time):
        self.public_key, self.audience, self.clock = public_key, audience, clock

    def check(self, token: str) -> Credential:
        from cryptography.exceptions import InvalidSignature
        refused = V1Error(401, "invalid_token", "This app token isn't one Vision accepts.")
        if not (self.public_key and self.audience):
            raise V1Error(401, "invalid_token", "App tokens aren't accepted on this server.")
        parts = token[len(APP_TOKEN_PREFIX):].split(".")
        if len(parts) != 2 or not all(parts):
            raise refused
        body, sig = parts
        try:
            self.public_key.verify(_unb64(sig), body.encode("ascii"))
            claims = json.loads(_unb64(body))
        except (InvalidSignature, ValueError, UnicodeError):
            raise refused
        if (not isinstance(claims, dict) or claims.get("v") != 1
                or claims.get("typ") != "vision-api" or claims.get("aud") != self.audience
                or not isinstance(claims.get("sub"), str) or not claims["sub"]
                or not isinstance(claims.get("exp"), (int, float))
                or not isinstance(claims.get("iat", 0), (int, float))):
            raise refused
        now = self.clock()
        if (claims["exp"] < now - CLOCK_SKEW_SECONDS
                or claims.get("iat", 0) > now + CLOCK_SKEW_SECONDS):
            raise V1Error(401, "token_expired", "The app token has expired; get a new one.")
        return Credential("app", subject=claims["sub"], plan=claims.get("plan"),
                          limit=limit_of(claims), limit_source="plan")


class Introspector:
    """Developer tokens (`vision_…`), checked through mycomap.org's
    POST /api/service-keys/introspect and cached: an answer that the token is active for
    `positive_ttl` seconds, that it isn't for `negative_ttl`. When .org can't be reached,
    a token answered active within `stale_ttl` keeps that answer; any other gets
    503 auth_unavailable, never 401 (a good token isn't reported as bad).

    `ask(token) -> dict` makes the call (tests pass a fake). The cache is keyed by the
    token's SHA-256; the token itself is never kept."""

    def __init__(self, ask: Callable[[str], dict] | None, positive_ttl: float = 60,
                 negative_ttl: float = 10, stale_ttl: float = 600,
                 clock: Callable[[], float] = time.monotonic, note=print):
        self.ask, self.clock, self.note = ask, clock, note
        self.positive_ttl, self.negative_ttl = positive_ttl, negative_ttl
        self.stale_ttl = stale_ttl
        self._cache: dict[bytes, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def check(self, token: str) -> Credential:
        if self.ask is None:
            raise V1Error(401, "invalid_token", "Developer tokens aren't accepted on this server.")
        h = hashlib.sha256(token.encode()).digest()
        now = self.clock()
        with self._lock:
            cached = self._cache.get(h)
        if cached:
            at, answer = cached
            if now - at < (self.positive_ttl if answer.get("active") else self.negative_ttl):
                return self._credential(answer)
        try:
            answer = self.ask(token)
            if not isinstance(answer, dict) or not isinstance(answer.get("active"), bool):
                raise ValueError("malformed introspection answer")
        except Exception as e:  # noqa: BLE001 - .org down, slow or odd: fall back below
            self.note(f"[v1] introspection failed: {type(e).__name__}: {e}")
            if cached and cached[1].get("active") and now - cached[0] < self.stale_ttl:
                return self._credential(cached[1])
            raise V1Error(503, "auth_unavailable", "Vision couldn't check this token with "
                          "mycomap.org; try again shortly.", retry_after=30)
        with self._lock:
            self._cache[h] = (now, answer)
            if len(self._cache) > 10_000:
                for k in [k for k, (t, _) in self._cache.items() if now - t > self.stale_ttl]:
                    del self._cache[k]
        return self._credential(answer)

    @staticmethod
    def _credential(answer: dict) -> Credential:
        if not answer.get("active"):
            reason = str(answer.get("reason") or "unknown")
            if reason == "unknown":
                raise V1Error(401, "invalid_token", "This token isn't one Vision accepts.")
            raise V1Error(403, "key_inactive", f"This token is {reason.replace('_', ' ')}.",
                          reason=reason)
        if answer.get("service") != "vision":
            raise V1Error(403, "key_inactive", "This token is for another service.",
                          reason="wrong_service")
        return Credential("key", subject=answer.get("subject") or None,
                          key_id=str(answer.get("keyId")), plan=answer.get("plan"),
                          limit=limit_of(answer), limit_source=answer.get("limit_source") or "plan")


def org_introspection(base_url: str, secret: str, timeout: float = 5.0) -> Callable[[str], dict]:
    """The real `Introspector.ask`: mycomap.org's service-key introspection."""
    import requests
    url = base_url.rstrip("/") + "/api/service-keys/introspect"
    session = requests.Session()

    def ask(token: str) -> dict:
        r = session.post(url, json={"key": token}, timeout=timeout,
                         headers={"Authorization": f"Bearer {secret}"})
        r.raise_for_status()
        return r.json()
    return ask


class Authenticator:
    # Shorter keys are treated as not configured (as .org does for VISION_API_KEY).
    MIN_KEY_LENGTH = 32

    def __init__(self, server_key: str | None, app_tokens: AppTokens | None = None,
                 introspector: Introspector | None = None):
        self.server_key = (server_key if server_key and len(server_key) >= self.MIN_KEY_LENGTH
                           else None)
        self.app_tokens, self.introspector = app_tokens, introspector

    def needs_network(self, headers) -> bool:
        """Whether checking this request's token may call mycomap.org (so run it off the
        event loop)."""
        return (bearer(headers) or "").startswith(KEY_PREFIX) and self.introspector is not None

    def check(self, headers) -> Credential:
        token = bearer(headers)
        if not token:
            raise V1Error(401, "invalid_token", "Send Authorization: Bearer <your Vision token>.")
        if token.startswith(APP_TOKEN_PREFIX) and self.app_tokens:
            return self.app_tokens.check(token)
        if token.startswith(KEY_PREFIX) and self.introspector:
            return self.introspector.check(token)
        if self.server_key and same_secret(token, self.server_key):
            return Credential("server")
        raise V1Error(401, "invalid_token", "This token isn't one Vision accepts.")


class PhotoWindow:
    """At most `limit` photos per key in any sliding `window` seconds (docs/api.md §6.4:
    the box's cost is per photo, while quota counts identifications)."""

    def __init__(self, limit: int, window: float = 60.0,
                 clock: Callable[[], float] = time.monotonic):
        self.limit, self.window, self.clock = limit, window, clock
        self.hits: dict[str, deque] = {}
        self.lock = threading.Lock()

    def take(self, key: str, photos: int) -> float:
        """0 if allowed (and counted), else seconds until these photos would fit."""
        if photos > self.limit:
            return self.window
        with self.lock:
            now = self.clock()
            q = self.hits.setdefault(key, deque())
            while q and q[0][0] <= now - self.window:
                q.popleft()
            free = self.limit - sum(n for _, n in q)
            if photos > free:
                for t, n in q:                      # wait for the oldest to leave the window
                    free += n
                    if free >= photos:
                        return max(t + self.window - now, 0.001)
            q.append((now, photos))
            if len(self.hits) > 10_000:
                for k in [k for k, v in self.hits.items() if not v]:
                    del self.hits[k]
            return 0.0


CONFIDENT_AT = 0.70        # Steve's answer-headline rule, 2026-10-11 (docs/api.md §4)
CONFIDENT_RANKS = ("family", "genus", "species")     # coarse to fine


def mark_confident(result: dict, threshold: float = CONFIDENT_AT) -> dict:
    """Adds `confident` to every candidate and the top-level `answer` (docs/api.md §4).
    A rank's top name is confident when the find is a fungus by the gate (or the gate
    didn't run) and its calibrated confidence reaches `threshold`; `answer` is the finest
    confident rank. A withheld confidence (None) is never confident, nor is any rank
    outside CONFIDENT_RANKS."""
    gate = result.get("gate")
    fungus = gate is None or gate.get("verdict") == "fungus"
    answer = None
    for rank, cands in (result.get("ranks") or {}).items():
        for i, c in enumerate(cands):
            conf = c.get("confidence")
            c["confident"] = bool(i == 0 and fungus and rank in CONFIDENT_RANKS
                                  and conf is not None and conf >= threshold)
    for rank in CONFIDENT_RANKS:
        cands = (result.get("ranks") or {}).get(rank) or []
        if cands and cands[0]["confident"]:
            answer = {"rank": rank, "name": cands[0]["name"],
                      "confidence": cands[0]["confidence"]}
    result["answer"] = answer
    return result


class Timer:
    """The timing block (docs/api.md §4.2), in whole milliseconds."""

    FIELDS = ("queue_ms", "fetch_ms", "decode_ms", "embed_ms", "score_ms")

    def __init__(self, clock: Callable[[], float] = time.monotonic):
        self.clock = clock
        self.start = clock()
        self.ms: dict[str, int | None] = {f: 0 for f in self.FIELDS}
        self.ms["fetch_ms"] = None
        self.photos_embedded = self.photos_reused = 0

    def add(self, name: str, seconds: float) -> None:
        self.ms[name] = (self.ms.get(name) or 0) + round(seconds * 1000)

    def block(self) -> dict:
        out = dict(self.ms)
        out["model_ms"] = out["embed_ms"] + out["score_ms"]
        out["total_ms"] = round((self.clock() - self.start) * 1000)
        out["photos_embedded"], out["photos_reused"] = self.photos_embedded, self.photos_reused
        return out

    @staticmethod
    def header(block: dict) -> str:
        """The same figures as a Server-Timing header."""
        names = {"queue_ms": "queue", "fetch_ms": "fetch", "decode_ms": "decode",
                 "embed_ms": "embed", "score_ms": "score", "total_ms": "total"}
        return ", ".join(f"{n};dur={block[k]}" for k, n in names.items()
                         if block.get(k) is not None)


# What the product calls its model (Steve, 2026-10-11): never the backbone underneath.
# Backbone and method ids stay internal (logs, /api/health, the website's own pages).
PUBLIC_MODEL_NAME = "MycoMap Vision"


def id_date(stamped: str | None) -> str | None:
    """An id that starts with a stamp ('20261008-012435-…') as that date, '2026-10-08'."""
    m = re.match(r"(\d{4})(\d{2})(\d{2})-", stamped or "")
    return f"{m[1]}-{m[2]}-{m[3]}" if m else None


def public_model(card: dict | None, calibration_id: str | None = None) -> dict:
    """The `model` block of every v1 answer (docs/api.md §4): a name; the version of the
    training data behind it and that snapshot's size (modelcard.py); the date its
    confidence was last calibrated. Nothing names a backbone or method."""
    card = card or {}
    data = card.get("data") or {"snapshot": None, "records": 0, "species": 0, "genera": 0}
    return {"name": PUBLIC_MODEL_NAME, "version": card.get("version"),
            "data": {k: data.get(k) for k in ("snapshot", "records", "species", "genera")},
            "calibrated_on": id_date((calibration_id or "")[:16] + "-")}


def taxon_counts(rank_counts: dict, *, records: int, model: dict,
                 nightly_version: int | None, generated_at: datetime | None = None) -> dict:
    """GET /api/v1/taxon-counts (docs/api.md §7): DNA-validated reference records per
    name, from the served index's own counts (Identifier.rank_counts), so every number is
    what the live model knows. A record with no name at a rank isn't counted there."""
    out = {"generated_at": (generated_at or datetime.now(timezone.utc))
           .isoformat(timespec="seconds").replace("+00:00", "Z"),
           "model": model,
           "nightly_version": nightly_version,
           "counts_are": "reference_records"}
    totals = {"records": records}
    for rank, key in RANK_KEYS.items():
        counts = {name: int(n) for name, n in sorted((rank_counts.get(rank) or {}).items(),
                                                      key=lambda kv: str(kv[0]))
                  if name and n > 0}
        out[key] = counts
        totals[key] = len(counts)
    out["totals"] = totals
    return out


def etag_of(body: bytes) -> str:
    return '"' + hashlib.sha256(body).hexdigest()[:32] + '"'


def encode(doc: dict) -> bytes:
    return json.dumps(doc, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
