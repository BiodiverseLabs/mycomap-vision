"""The public API, v1 (docs/api.md): credentials, refusals, and the answers built for it.

Every /api/v1/ path is answered only with a credential (`Authenticator`); the website's
cookie sign-in doesn't apply there (signin.needs_session). A refusal is
`{"code", "detail"}` with its status, plus `Retry-After` when trying again later can work.

Credentials so far: mycomap.org's own server key (VISION_API_KEY on .org,
MV_ORG_VISION_KEY here), for server-to-server reads such as the taxon counts.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import datetime, timezone

from fastapi.responses import JSONResponse

PREFIX = "/api/v1/"
RANK_KEYS = {"species": "species", "genus": "genera", "family": "families"}


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
    kind: str                       # "server": mycomap.org's own key
    subject: str | None = None      # whose quota (docs/api.md §6); None for the server


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


class Authenticator:
    # Shorter keys are treated as not configured (as .org does for VISION_API_KEY).
    MIN_KEY_LENGTH = 32

    def __init__(self, server_key: str | None):
        self.server_key = (server_key if server_key and len(server_key) >= self.MIN_KEY_LENGTH
                           else None)

    def check(self, headers) -> Credential:
        token = bearer(headers)
        if not token:
            raise V1Error(401, "invalid_token", "Send Authorization: Bearer <your Vision token>.")
        if self.server_key and same_secret(token, self.server_key):
            return Credential("server")
        raise V1Error(401, "invalid_token", "This token isn't one Vision accepts.")


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
