"""HTTP API for the frontend (and later mycomap.org).

Run: `.venv/Scripts/mv serve` (defaults to http://127.0.0.1:8010).
Read-only over the manifest; identification runs on the local GPU when there is one.
Public limits (request and image size, rate per IP, GPU queue, allowed backbones)
are in guards.py and set from the environment. Sign-in with a mycomap.org account
(off, identify only, or the whole site) is in signin.py.
"""

from __future__ import annotations

import html
import json
import logging
import secrets
import sqlite3
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Callable
from urllib.parse import quote

import numpy as np
from fastapi import Body, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, RedirectResponse,
                               Response)
from PIL import Image

from . import (answercache, apidocs, apiv1, byobservation, config, confusions, evaluate, fastsearch, feedback, fungusgate, geocode, inat, manifest,
               feedback_org as feedback_org_rules, modelcard, models, name_changes, names, nightly,
               onnxtower, permissions, twins,
               quotas as quota_rules, serving, sessions as session_store, sources, uploads,
               workqueue, yours)
from .dates import real_date
from .embed import SCHEMA as EMBED_SCHEMA
from .embed import normalise, photos_per_second
from .guards import (MAX_FILE_BYTES, MAX_PHOTOS, MAX_REQUEST_BYTES, Gate, Limits, RateLimiter,
                     TooLarge)
from .identify import PHOTO_INFO_SQL, Identifier, NotTrainedHere, photo_info_rows
from .methods import method_ready
from .prior import Context
from .signin import (NONCE_COOKIE, NONCE_TTL_SECONDS, SESSION_COOKIE, SigninConfig, SigninError,
                     load_public_key, origin_of, safe_return_to, verify_token)

log = logging.getLogger("mycomap_vision.api")
NAME_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 30, "6-30"), (31, 10**9, "31+")]
# The working draft of the Vision paper (no page on the site since 2026-10-12), served to signed-in
# members only (signin.MEMBERS_ONLY). It ships with the code, never in the public web bundle.
PAPER_FILE = config.RESEARCH_DIR / "paper" / "draft.md"   # private, never in this repo


def open_manifest(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    permissions.ensure_schema(conn)
    manifest.upgrade(conn)
    return conn


def _setting_number(name: str) -> float | None:
    raw = config.setting(name)
    try:
        return float(raw) if raw else None
    except ValueError:
        return None


def start_background(name: str, every_seconds: float, work: Callable[[], None],
                     note=print) -> threading.Thread:
    """Run `work` now and then every `every_seconds`, on a daemon thread. A failure
    is logged and retried next time; it never stops the server."""
    def loop():
        while True:
            try:
                work()
            except Exception as e:  # noqa: BLE001 - keep serving whatever went wrong
                note(f"[{name}] {type(e).__name__}: {e}")
            threading.Event().wait(every_seconds)
    t = threading.Thread(target=loop, name=name, daemon=True)
    t.start()
    return t


WEB_DIST = config.REPO_ROOT / "web" / "dist"


def reference_stats(conn: sqlite3.Connection, served: set[str]) -> dict:
    """What the Data page shows about the reference set. `served`: the backbones the site
    offers (only their photo counts are shown; the manifest also lists test backbones)."""
    one = lambda sql: conn.execute(sql).fetchone()[0]  # noqa: E731
    labels = names.manifest_labels(conn)
    # Spellings of one name count as one name (names.py).
    name_counts = Counter()
    for name, n in conn.execute("select scientific_name, count(*) from records "
                                "where north_america = 1 and label_conflict = 0 "
                                "and coalesce(scientific_name, '') <> '' group by scientific_name"):
        name_counts[labels.get(name, name).strip()] += n
    buckets = {label: 0 for _, _, label in NAME_BUCKETS}
    for n in name_counts.values():
        for lo, hi, label in NAME_BUCKETS:
            if lo <= n <= hi:
                buckets[label] += 1
    conn.executescript(EMBED_SCHEMA)
    embedded = {b: n for b, n in conn.execute(
        "select backbone, count(*) from embeddings group by 1") if b in served}
    # Who built the reference set: the projects that marked its records green, and the
    # photographers of those records (North America, one name: the records Vision uses).
    used = "r.north_america = 1 and r.label_conflict = 0"
    projects: set[str] = set()
    for (green,) in conn.execute(f"select green_projects from records r where {used}"):
        projects.update(p for p in json.loads(green or "[]") if p)
    # New DNA-verified records in the newest week of validations (the week to the newest
    # validation date, so a late export doesn't read as a quiet week).
    newest = one(f"select max(validated_on) from records r where {used}")
    recent = None
    if newest:
        since = (date.fromisoformat(newest[:10]) - timedelta(days=6)).isoformat()
        recent = {"records": conn.execute(f"select count(*) from records r where {used} "
                                          "and validated_on >= ?", (since,)).fetchone()[0],
                  "from": since, "through": newest[:10]}
    return {
        "records": one("select count(*) from records"),
        "records_north_america": one("select count(*) from records where north_america = 1"),
        "label_conflicts": one("select count(*) from records where label_conflict = 1"),
        "inat_ok": one("select count(*) from inat_observations where status = 'ok'"),
        "inat_missing": one("select count(*) from inat_observations where status = 'missing'"),
        "photos": one("select count(*) from photos"),
        "photos_by_license": dict(conn.execute(
            "select license_class, count(*) from photos group by 1").fetchall()),
        "contributors_arr": one("select count(distinct owner_login) from photos "
                                "where license_class = 'arr'"),
        "embedded": embedded,
        "names": len(name_counts),
        # Temporary codes ("Russula sp. 'IN01'"): species known from DNA, not yet described.
        "names_provisional": sum(1 for n in name_counts if names.parse_name(n).code),
        "names_by_records": buckets,
        "projects": len(projects),
        "photographers": one("select count(distinct p.owner_login) from photos p "
                             "join observation_photos op using (photo_id) "
                             f"join records r using (observation_id) where {used} "
                             "and coalesce(p.owner_login, '') <> ''"),
        "recent_week": recent,
    }


def trained_through(conn: sqlite3.Connection) -> dict[str, str]:
    """Each fine-tuned model's last training date: it learned from nothing validated later."""
    try:
        return dict(conn.execute("select name, trained_through from finetunes"))
    except sqlite3.OperationalError:     # no model fine-tuned on this manifest yet
        return {}


class Cached:
    """A value that is slow to compute and changes slowly (the Data page's statistics: 14 s
    over the full reference set). Answered from memory; once older than `max_age` it is
    recomputed on a background thread while the old value keeps being answered. Only the
    first caller ever waits, and a failed refresh keeps the old value."""

    def __init__(self, compute: Callable[[], dict], max_age: float,
                 clock: Callable[[], float] = time.monotonic, note=print):
        self.compute, self.max_age, self.clock, self.note = compute, max_age, clock, note
        self._lock = threading.Lock()          # guards the fields below
        self._computing = threading.Lock()     # one computation at a time
        self._value: dict | None = None
        self._at = 0.0
        self._refreshing = False

    def get(self) -> dict:
        with self._lock:
            if self._value is not None:
                if self.clock() - self._at >= self.max_age and not self._refreshing:
                    self._refreshing = True
                    threading.Thread(target=self._refresh, name="cached refresh",
                                     daemon=True).start()
                return self._value
        with self._computing:                  # the first caller computes; others wait for it
            with self._lock:
                if self._value is not None:
                    return self._value
            value = self.compute()
            with self._lock:
                self._value, self._at = value, self.clock()
            return value

    def _refresh(self) -> None:
        try:
            with self._computing:
                value = self.compute()
            with self._lock:
                self._value, self._at = value, self.clock()
        except Exception as e:  # noqa: BLE001 - the old value keeps being answered
            self.note(f"[stats] refresh failed, keeping the last figures: {type(e).__name__}: {e}")
        finally:
            with self._lock:
                self._refreshing = False


def preload_specs(value: str | None) -> list[str]:
    """MV_PRELOAD: comma list of backbone/method to load at startup; "default" is the
    model an identification uses when none is named. Unset or "off": none."""
    if not value or value.strip().lower() in ("off", "none", "0", "false"):
        return []
    return [s.strip() for s in value.split(",") if s.strip()]


@dataclass
class Preload:
    """What the server loaded at startup, for /api/health."""
    state: str = "off"                 # off | loading | ready | failed
    models: list[str] = field(default_factory=list)
    seconds: float | None = None
    error: str | None = None


def read_photo(upload: UploadFile, side: int = uploads.kept_side([])) -> tuple[Image.Image, tuple]:
    """One uploaded photo, refusing oversized files and images before decoding them,
    decoded once and kept only at the size the models need (uploads.py: shorter side
    `side`). Also returns its EXIF (latitude, longitude, date), read before shrinking,
    used only for scoring."""
    body = upload.file.read(MAX_FILE_BYTES + 1)
    if len(body) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{upload.filename} is over {MAX_FILE_BYTES // 2**20} MB")
    try:
        return uploads.decode_upload(body, side)
    except TooLarge as e:
        raise HTTPException(413, f"{upload.filename}: {e}")
    except Exception:
        raise HTTPException(400, f"{upload.filename} is not an image we can read")


def fill_context(entered: Context, from_photos: list[tuple]) -> tuple[Context, dict]:
    """What the caller entered wins; gaps are filled from the first photo that has them.
    Returns the context and what was used, rounded to 0.1 degree for the response.
    A 1970-01-01 date, typed or from a photo, counts as none (dates.py)."""
    lat, lng, when = entered.latitude, entered.longitude, real_date(entered.observed_on)
    place_from = "entered" if lat is not None and lng is not None else None
    date_from = "entered" if when else None
    for plat, plng, pdate in from_photos:
        if place_from is None and plat is not None and plng is not None:
            lat, lng, place_from = plat, plng, "photo"
        pdate = real_date(pdate)
        if date_from is None and pdate:
            when, date_from = pdate, "photo"
    used = {"latitude": None if lat is None else round(lat, 1),
            "longitude": None if lng is None else round(lng, 1),
            "observed_on": when, "place_from": place_from, "date_from": date_from}
    return Context(lat, lng, when), used


def create_app(manifest_path: Path | None = None, embeddings_root: Path | None = None,
               backbone_loader: Callable[[str], object] = models.load_backbone,
               web_dist: Path | None = WEB_DIST, limits: Limits | None = None,
               signin: SigninConfig | None = None, background: bool = True,
               note=print, preload: list[str] | None = None,
               layer: "nightly.Layer | None" = None,
               nightly_settings: "nightly.Settings | None" = None,
               nightly_fetch: Callable[[], tuple[list[dict], str]] | None = None,
               nightly_steps: "nightly.Steps | None" = None,
               nightly_fetch_pending: Callable[[], tuple[list[dict], str]] | None = None,
               photo_fetcher: Callable[[], object] | None = None,
               paper_file: Path = PAPER_FILE,
               experiments_dir: Path | None = None,
               server_key: str | None | bool = True,
               taxon_counts: bool | None = None,
               api_v1: bool | None = None,
               sessions: bool | None = None,
               by_observation: bool | None = None,
               observation_fetcher: "byobservation.Fetcher | None" = None,
               quotas: "quota_rules.Quotas | None" = None,
               usage_post: Callable[[dict], None] | None = None,
               app_tokens: "apiv1.AppTokens | None | bool" = True,
               introspect: "Callable[[str], dict] | None | bool" = True,
               dataset_root: Path | None = None, dataset_store: str | None = None,
               dataset_store_reader=None, dataset_allow_draft: bool = False,
               fungus_gate: "fungusgate.FungusGate | None | bool" = True,
               place_search: "geocode.PlaceSearch | None" = None,
               confused_with: "confusions.Served | None | bool" = True,
               feedback_store: "feedback.Store | None" = None,
               feedback_org: "feedback_api.OrgClient | None" = None,
               answer_cache: "answercache.AnswerCache | None" = None,
               linked_accounts: "yours.LinkedCache | None | bool" = True) -> FastAPI:
    """`background` starts, when their settings are present, the photographers'-answers
    sync (MV_ORG_BASE_URL + MV_ORG_VISION_KEY, every MV_PERMISSIONS_SYNC_SECONDS,
    default 300) and the licence refresh (MV_LICENSE_REFRESH_HOURS, off by default).
    `note` prints what they report.

    `layer` (serve() passes it when MV_NIGHTLY is on): the nightly update's layer
    (nightly.py). `manifest_path` is then its manifest, the index adds its shards, and
    with `background` the nightly update runs on its schedule. `nightly_fetch`,
    `nightly_fetch_pending`, `nightly_steps` and `photo_fetcher` replace the reads of
    mycomap.org and iNat (tests).

    `preload` (default: MV_PRELOAD) names models to load and index on a background
    thread at startup, so the first person to identify doesn't wait for it (about
    30-45 s on the server box). The server answers meanwhile; an identification that
    arrives first waits for the load and then uses it.

    The public API v1 (apiv1.py, docs/api.md) lives under /api/v1/ and answers only a
    credential. `server_key` is mycomap.org's key (default: MV_ORG_VISION_KEY(_FILE));
    `taxon_counts` turns on GET /api/v1/taxon-counts (default: MV_TAXON_COUNTS);
    `api_v1` turns on POST /api/v1/identify (default: MV_API_V1); `sessions` the
    app's observation sessions under /api/v1/sessions (default: MV_SESSIONS). `app_tokens` checks the
    app's tokens (default: the sign-in bridge's public key, for MV_PUBLIC_ORIGIN);
    `introspect` asks mycomap.org about a developer token (default: MV_ORG_BASE_URL with
    the server key; tests pass a fake).

    `fungus_gate`: the not-a-fungus gate every identification passes (fungusgate.py).
    True loads the saved one from data/models (none saved, or MV_FUNGUS_GATE=0: no
    gate); a FungusGate uses that one; None or False runs ungated.

    `linked_accounts`: a signed-in person's verified iNat and MO accounts from mycomap.org,
    for "Your collections" on the Identify result (yours.py). True reads them with
    MV_ORG_BASE_URL and the server key (neither set: no panel); tests pass a LinkedCache.

    `confused_with`: mycomap.org's "misidentified as" table (confusions.py) behind the
    `confused_with` field on species candidates. True reads the cache beside the manifest
    (the nightly fills it); a confusions.Served uses that one; None or False: no field."""
    app = FastAPI(title="MycoMap Vision", version="0.1")
    # The image tower as int8 ONNX (onnxtower.py, MV_EMBED_ONNX=1), where the release ships
    # it for the served weights; torch otherwise.
    if config.setting("MV_EMBED_ONNX") == "1":
        from .finetune import models_dir
        backbone_loader = onnxtower.loader(backbone_loader, models_dir(), note=note)
    limits = limits or Limits.from_settings()
    if fungus_gate is True:
        fungus_gate = (fungusgate.FungusGate.load()
                       if config.setting("MV_FUNGUS_GATE", "1") != "0" else None)
    fungus_gate = fungus_gate or None
    signin = signin or SigninConfig.from_settings()
    if linked_accounts is True:
        linked_accounts = yours.LinkedCache.from_settings()
    linked_accounts = linked_accounts or None
    rate = RateLimiter(*limits.rate)
    in_flight = Gate(limits.max_in_flight)

    @app.middleware("http")
    async def cap_request_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        return await call_next(request)

    def person(request: Request) -> dict | None:
        """The signed-in person from our session cookie, or None."""
        if not signin.enabled:
            return None
        return signin.sessions.read(request.cookies.get(SESSION_COOKIE))

    def linked_of(request: Request) -> "yours.Linked | None":
        """The signed-in person's verified accounts (yours.py), from our own session's sub
        only; None when nobody is signed in or mycomap.org can't be read."""
        p = person(request)
        if not p or not p.get("sub") or linked_accounts is None:
            return None
        return linked_accounts.get(str(p["sub"]))

    # Registered after cap_request_size, so it runs first: nothing is read for a
    # request that isn't allowed in.
    @app.middleware("http")
    async def require_signin(request: Request, call_next):
        path = request.url.path
        if signin.needs_session(request.method, path) and not person(request):
            if path.startswith("/api/"):
                return JSONResponse({"detail": "sign in with your mycomap.org account",
                                     "signin": "/auth/signin"}, status_code=401)
            here = path + (f"?{request.url.query}" if request.url.query else "")
            return RedirectResponse(f"/auth/signin?returnTo={quote(here, safe='')}",
                                    status_code=302)
        return await call_next(request)

    # The public API answers credentials, never the site's cookie. Registered after
    # require_signin, so it runs before it; signin.needs_session leaves /api/v1/ alone.
    # Every /api/v1/ path passes here, so no v1 route can be left open by mistake.
    org_secret = permissions.org_key() if server_key is True else server_key or None
    if app_tokens is True:
        key_file = config.setting("MV_SIGNIN_PUBLIC_KEY_FILE")
        try:
            public_key = signin.public_key or (load_public_key(Path(key_file).read_bytes())
                                               if key_file else None)
        except (OSError, SigninError) as e:
            note(f"[v1] app tokens off: {type(e).__name__}: {e}")
            public_key = None
        app_tokens = apiv1.AppTokens(public_key, signin.origin
                                     or origin_of(config.setting("MV_PUBLIC_ORIGIN")))
    if introspect is True:
        base_url = config.setting("MV_ORG_BASE_URL")
        introspect = (apiv1.org_introspection(base_url, org_secret)
                      if base_url and org_secret else None)
    authenticator = apiv1.Authenticator(org_secret, app_tokens or None,
                                        apiv1.Introspector(introspect, note=note)
                                        if introspect else None)

    @app.middleware("http")
    async def require_v1_credential(request: Request, call_next):
        if apiv1.is_v1(request.url.path):
            try:
                if authenticator.needs_network(request.headers):    # may ask mycomap.org
                    request.state.credential = await run_in_threadpool(
                        authenticator.check, request.headers)
                else:
                    request.state.credential = authenticator.check(request.headers)
            except apiv1.V1Error as e:
                return e.response()
        return await call_next(request)

    @app.exception_handler(apiv1.V1Error)
    async def v1_refusal(request: Request, exc: apiv1.V1Error):
        return exc.response()

    @app.exception_handler(RequestValidationError)
    async def bad_fields(request: Request, exc: RequestValidationError):
        if not apiv1.is_v1(request.url.path):
            return await request_validation_exception_handler(request, exc)
        where = "; ".join(f"{'.'.join(str(x) for x in e.get('loc', ())[1:])}: {e.get('msg')}"
                          for e in exc.errors())
        return apiv1.V1Error(400, "bad_request", where or "malformed request").response()

    def signin_failed(reason: str) -> HTMLResponse:
        return HTMLResponse(
            "<!doctype html><meta charset=utf-8><title>Sign-in did not complete</title>"
            "<p>Sign-in with your mycomap.org account did not complete "
            f"({html.escape(reason)}).</p><p><a href=\"/auth/signin\">Try again</a></p>",
            status_code=401)

    @app.get("/api/me")
    def me(request: Request):
        p = person(request)
        return {"signin": signin.mode,
                "user": {"id": p["sub"], "name": p.get("name")} if p else None}

    @app.get("/auth/signin", include_in_schema=False)
    def signin_start(returnTo: str | None = None):  # noqa: N803 (the .org name)
        if not signin.enabled:
            raise HTTPException(404, "sign-in is not configured here")
        nonce = secrets.token_urlsafe(24)
        res = RedirectResponse(signin.authorize_url(nonce), status_code=302)
        res.set_cookie(NONCE_COOKIE, signin.nonces.sign(
            {"nonce": nonce, "returnTo": safe_return_to(returnTo),
             "exp": time.time() + NONCE_TTL_SECONDS}),
            max_age=NONCE_TTL_SECONDS, path="/auth/", httponly=True,
            secure=signin.secure_cookies, samesite="lax")
        return res

    @app.get("/auth/dev-bridge/callback", include_in_schema=False)
    def signin_callback(request: Request, token: str = ""):
        if not signin.enabled:
            raise HTTPException(404, "sign-in is not configured here")
        pending = signin.nonces.read(request.cookies.get(NONCE_COOKIE))
        if not pending or not token:
            return signin_failed("the sign-in link expired")
        try:
            claims = verify_token(token, signin.public_key, aud=signin.origin,
                                  nonce=pending["nonce"])
        except SigninError as e:
            log.warning("sign-in token refused: %s", e.reason)
            return signin_failed("expired" if e.reason == "expired" else "not accepted")
        name = claims.get("name")
        session = {"sub": claims["sub"],
                   "name": name[:120] if isinstance(name, str) else None,
                   "exp": time.time() + signin.session_days * 86400}
        res = RedirectResponse(safe_return_to(pending.get("returnTo")), status_code=302)
        res.set_cookie(SESSION_COOKIE, signin.sessions.sign(session),
                       max_age=int(signin.session_days * 86400), path="/", httponly=True,
                       secure=signin.secure_cookies, samesite="lax")
        res.delete_cookie(NONCE_COOKIE, path="/auth/")
        return res

    # Signing out ends this site's session only. The browser must not go back to a
    # gated page afterwards: that sends it to mycomap.org, which (still signed in)
    # signs it straight back in. So it lands here, an open page that says so.
    @app.post("/auth/signout", include_in_schema=False)
    def signout():
        res = JSONResponse({"ok": True, "next": "/auth/signed-out"})
        res.delete_cookie(SESSION_COOKIE, path="/")
        return res

    @app.get("/auth/signed-out", include_in_schema=False)
    def signed_out():
        if not signin.enabled:
            raise HTTPException(404, "sign-in is not configured here")
        issuer = html.escape(signin.issuer)
        return HTMLResponse(
            "<!doctype html><html lang=en><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width, initial-scale=1'>"
            "<title>Signed out - MycoMap Vision</title>"
            "<style>body{font-family:system-ui,sans-serif;max-width:32rem;margin:15vh auto;"
            "padding:0 1rem;color:#374151}h1{color:#4a3728;font-size:1.4rem}"
            "a.btn{display:inline-block;margin:.25rem .5rem .25rem 0;padding:.55rem 1rem;"
            "border-radius:.375rem;text-decoration:none;font-weight:600}"
            ".go{background:#4a7c3f;color:#fff}.alt{border:1px solid #d1d5db;color:#374151}"
            "</style>"
            "<h1>You've signed out of MycoMap Vision</h1>"
            f"<p>You're still signed in on {issuer.removeprefix('https://')}, so signing "
            "in here again takes one click.</p>"
            "<p><a class='btn go' href='/auth/signin'>Sign in again</a>"
            f"<a class='btn alt' href='{issuer}/api/logout'>Also sign out of "
            f"{issuer.removeprefix('https://')}</a></p>")

    manifest_path = manifest_path or config.MANIFEST_PATH
    conn = open_manifest(manifest_path)
    if confused_with is True:
        confused_with = confusions.Served(confusions.cache_path(manifest_path), note=note)
    confused_with = confused_with or None
    app.state.confused_with = confused_with
    if confused_with is not None and background:    # read at startup, not by the first caller
        threading.Thread(target=confused_with.reload, name="confused-with", daemon=True).start()

    def confused_table() -> "confusions.Table | None":
        return confused_with.table() if confused_with is not None else None
    # Before anything is built: a new release's manifest may hold an older list of who
    # withdrew than the one this box last read (permissions.py).
    permissions.restore_last_good(conn, note=note)
    db_lock = threading.Lock()
    gpu_lock = threading.Lock()
    backbones: dict[str, object] = {}
    # Every identification goes through one queue in front of the model (workqueue.py):
    # interactive calls first, and calls waiting on the same index share one search.
    model_queue = workqueue.ModelQueue(
        gpu_lock, max_wait=_setting_number("MV_QUEUE_MAX_WAIT") or 20.0,
        max_batch_photos=int(_setting_number("MV_QUEUE_BATCH_PHOTOS") or 20), note=note)
    app.state.model_queue = model_queue
    # Quotas (quotas.py, MV_QUOTAS off|count|enforce): monthly identifications per member,
    # counted here, durable on the box, reported to mycomap.org every few minutes.
    if quotas is None:
        quotas = quota_rules.Quotas(
            (config.RELEASE_ROOT / "state" if config.RELEASE_ROOT else config.DATA_DIR)
            / "vision-usage.sqlite",
            mode=(config.setting("MV_QUOTAS") or "off").strip().lower(),
            default_monthly=int(_setting_number("MV_QUOTA_DEFAULT_MONTHLY")
                                or quota_rules.DEFAULT_MONTHLY))
    app.state.quotas = quotas
    # Answers already given for an iNat or MO observation (answercache.py), served again
    # while what Vision answers from is unchanged (answers_version below).
    if answer_cache is None:
        answer_cache = answercache.AnswerCache(
            (config.RELEASE_ROOT / "state" if config.RELEASE_ROOT
             else Path(manifest_path or config.MANIFEST_PATH).parent)
            / "observation-answers.sqlite")
    app.state.answer_cache = answer_cache

    def q(sql: str, params: tuple = ()):
        with db_lock:
            return conn.execute(sql, params).fetchall()

    def embedded_counts() -> dict[str, int]:
        return dict(q("select backbone, count(*) from embeddings group by 1"))

    def permission_view() -> permissions.PermissionView:
        with db_lock:
            return permissions.PermissionView.load(conn)

    def permission_status() -> dict:
        with db_lock:
            return permissions.status_report(conn)

    def photo_lookup(photo_ids: list[int]) -> dict:
        """Licence and owner as they are now (not when the model was built)."""
        if not photo_ids:
            return {}
        marks = ",".join("?" for _ in photo_ids)
        return photo_info_rows(q(f"{PHOTO_INFO_SQL} where photo_id in ({marks})", tuple(photo_ids)))

    def build_index(key: tuple[str, str], version: tuple) -> Identifier:
        backbone, method = key
        with db_lock:
            cal = evaluate.latest_calibration(conn, backbone, method)
        root = embeddings_root / backbone if embeddings_root else None
        # Its own connection, not the shared one: building reads the whole reference set
        # (tens of seconds on the box), and holding db_lock that long stalled every
        # other page. ServedIndexes runs builds one at a time, off the request path.
        own = sqlite3.connect(manifest_path, timeout=30)
        try:
            ident = Identifier(own, backbone, method, root, calibration=cal,
                               train=limits.fit_on_demand, photo_info=False,
                               layer_root=layer.embeddings if layer else None,
                               gate=fungus_gate, specimen_groups=served_groups())
        finally:
            own.close()
        # The fast int8 search (fastsearch.py, MV_SEARCH_INT8=1): packed here, on the build
        # thread, after taking the one packed slot from the index serving now.
        if config.setting("MV_SEARCH_INT8") == "1":
            ref = getattr(getattr(getattr(ident, "nearest", None), "scorer", None), "ref", None)
            if not hasattr(ref, "use_int8"):
                fastsearch.SLOT.last_error = ("the reference vectors aren't memory-mapped here "
                                              "(on a GPU?): the plain search answers")
            else:
                try:
                    start = time.monotonic()
                    ref.use_int8(int(_setting_number("MV_SEARCH_REFINE") or fastsearch.REFINE))
                    note(f"[search] {backbone}/{method}: int8 search packed, "
                         f"{ref.packed.nbytes / 2**20:.0f} MB in {time.monotonic() - start:.0f} s")
                except Exception as e:  # noqa: BLE001 - the mapped search keeps answering
                    fastsearch.SLOT.last_error = f"{type(e).__name__}: {e}"
                    note(f"[search] int8 search off, mapped search kept: {fastsearch.SLOT.last_error}")
        ident.warm()
        return ident

    # Rebuilt (someone withdrew, a new calibration) on its own thread and swapped in
    # whole; the old index is released after, and serves on if a build fails.
    indexes = serving.ServedIndexes(build_index, note=note, release=models.release_memory,
                                    wait=_setting_number("MV_INDEX_WAIT_SECONDS") or 20.0)

    def identifier(backbone: str, method: str, view: permissions.PermissionView) -> Identifier:
        """The index as the data stand now. May wait for a rebuild (serving.Updating if
        it takes too long); never call it holding gpu_lock."""
        return indexes.get((backbone, method), index_version(backbone, method, view))

    def index_version(backbone: str, method: str, view: permissions.PermissionView) -> tuple:
        n = embedded_counts().get(backbone, 0)
        with db_lock:
            cal = evaluate.latest_calibration(conn, backbone, method)
            # A night that only removed or renamed records leaves n alone.
            night = nightly.layer_version(conn) if layer else 0
        # A change in who has said no rebuilds the reference set without their photos.
            # Photos Human Feedback leaves out (synced at startup and before each night).
            excluded = feedback.exclusions_fingerprint(conn)
            # Renames read from mycomap.org's name-change log between nights (name_changes.py).
            renamed = name_changes.version(conn) if layer else 0
        return (n, cal["run_id"] if cal else None, view.fingerprint(), night, excluded, renamed)

    def answers_version(backbone: str, method: str, view: permissions.PermissionView) -> str:
        """What this model answers from now (answercache.answers_version): the release and
        the served index's version. A kept observation answer is served only under it."""
        return answercache.answers_version(model_card().get("version"),
                                           index_version(backbone, method, view))

    warm = Preload()

    # Cross-post twin groups (twins.py): records that are one find on MO and iNat, by the
    # rule the served record-grouped method was adopted and calibrated under (E32). Only
    # that method reads them. MV_TWIN_GROUPS, else the serving release's twin-groups.tsv.
    groups_cache: dict = {}

    def served_groups() -> dict | None:
        if "groups" not in groups_cache:
            path = config.setting("MV_TWIN_GROUPS")
            path = (Path(path) if path else
                    (served_release_dir() or Path("/nonexistent")) / "twin-groups.tsv")
            groups_cache["groups"] = twins.read(path) if path.is_file() else None
        return groups_cache["groups"]

    def served_ready(method: str) -> bool:
        """methods.method_ready, and for a record-grouped method its twin groups here."""
        return method_ready(method) and (not twins.is_grouped(method)
                                         or served_groups() is not None)

    def served_release_dir() -> Path | None:
        """The folder of the release this server answers from, or None (a laptop)."""
        if layer:
            return layer.release_dir
        from .release import current_release
        rid = current_release(config.RELEASE_ROOT) if config.RELEASE_ROOT else None
        return config.RELEASE_ROOT / "releases" / rid if rid else None

    # The public model card (modelcard.py): the training-data version and its size. Read
    # once from the release (a few seconds over the full manifest), in the background at
    # startup, else at its first use.
    card_lock = threading.Lock()
    card_cache: dict = {}

    def model_card() -> dict:
        with card_lock:
            if "card" not in card_cache:
                try:
                    card_cache["card"] = modelcard.card(served_release_dir(), manifest_path)
                except Exception as e:  # noqa: BLE001 - the answer goes on without it
                    note(f"[v1] model card: {type(e).__name__}: {e}")
                    return modelcard.card(None, None)
            return card_cache["card"]
    if background:
        threading.Thread(target=model_card, name="model card", daemon=True).start()

    # /api/health is public: model ids only with mycomap.org's server key (apiv1.public_health).
    # Which backbone answers is read from the database at most once a minute.
    served_cache: dict = {"at": -1e9, "served": None, "ready": []}

    def served_backbones() -> tuple[str | None, list[str]]:
        now = time.monotonic()
        if now - served_cache["at"] > 60:
            ready = ready_backbones()
            served_cache.update(at=now, ready=ready, served=ready[0] if ready else None)
        return served_cache["served"], served_cache["ready"]

    @app.get("/api/health")
    def health(request: Request):
        full = {"ok": True, "code_version": config.code_version(),
                "preload": {"state": warm.state, "models": warm.models,
                            "seconds": warm.seconds, "error": warm.error},
                "nightly": nightly_health(),
                "queue": model_queue.state(),
                "search": fastsearch.SLOT.state(),
                "embed": onnxtower.STATE.state(),
                "quotas": {"mode": quotas.mode},
                "sessions": (app.state.sessions.state()
                             if getattr(app.state, "sessions", None) else None),
                "fungus_gate": fungus_gate.summary() if fungus_gate else None,
                "vectors": vector_storage()}
        given = request.headers.get("authorization", "")
        if org_secret and secrets.compare_digest(given.encode(),
                                                 f"Bearer {org_secret}".encode()):
            return full
        served, ready = served_backbones()
        known = {*limits.allowed_backbones, *ready, *(full["nightly"].get("layer") or {}),
                 *full["embed"], *(m.split("/", 1)[0] for m in warm.models),
                 *(k.split("/", 1)[0] for k in full["vectors"]),
                 *([fungus_gate.base] if fungus_gate else [])}
        return apiv1.public_health(full, served, known)

    def vector_storage() -> dict:
        """Per served index (backbone/method): reference vectors held as int8 and float."""
        out = {}
        for key, idx in indexes.served().items():
            store = getattr(idx, "vector_storage", None)
            if store is not None:
                out["/".join(map(str, key)) if isinstance(key, tuple) else str(key)] = store()
        return out

    def nightly_health() -> dict:
        if not layer:
            return {"enabled": False}
        with db_lock:
            out = nightly.health(conn, nightly_settings, nightly_zone)
        out["confused_with"] = confusions.health(confusions.cache_path(manifest_path))
        return out

    def ready_backbones() -> list[str]:
        return [b for b, n in embedded_counts().items() if n and b in limits.allowed_backbones]

    def default_method() -> str:
        """What an identification uses when no method is named (MV_DEFAULT_METHOD)."""
        return limits.served_default(lambda m: m in evaluate.METHODS and served_ready(m))

    @app.get("/api/models")
    def list_models():
        counts = {b: n for b, n in embedded_counts().items() if b in limits.allowed_backbones}
        with db_lock:
            speed = photos_per_second(conn)
            through = trained_through(conn)
        out = []
        for name, alias in models.ALIASES.items():
            out.append({"backbone": name, "spec": alias.spec, "note": alias.note,
                        "embedded_photos": counts.pop(name, 0),
                        "photos_per_second": speed.get(name),
                        "trained_through": through.get(name)})
        for name, n in counts.items():
            out.append({"backbone": name, "spec": name, "note": "", "embedded_photos": n,
                        "photos_per_second": speed.get(name),
                        "trained_through": through.get(name)})
        return {"backbones": out,
                "methods": [m for m in evaluate.METHODS
                            if limits.method_allowed(m) and served_ready(m)],
                "max_models": limits.max_models,
                "default_method": default_method(),
                "ready": [b["backbone"] for b in out if b["embedded_photos"]]}

    def compute_stats() -> dict:
        """On a connection of its own, never the shared one: over the full set this takes
        ~14 s, and on the shared connection it held db_lock that long, so every
        identification waited behind anyone looking at the Data page (which also asked
        again every 30 s)."""
        own = sqlite3.connect(manifest_path, timeout=30)
        try:
            return reference_stats(own, set(limits.allowed_backbones))
        finally:
            own.close()

    stats_cache = Cached(compute_stats, _setting_number("MV_STATS_SECONDS") or 3600, note=note)
    app.state.stats_cache = stats_cache
    if background:          # counted once at startup, so the first visitor doesn't wait
        threading.Thread(target=stats_cache.get, name="stats", daemon=True).start()

    # Place search for the Identify page's map (geocode.py): Nominatim, cached and paced.
    places = place_search or geocode.PlaceSearch()

    @app.get("/api/geocode/search")
    def geocode_search(q: str = ""):
        try:
            return places.search(q)
        except geocode.GeocodeError as e:
            headers = {"Retry-After": str(e.retry_after)} if e.retry_after else None
            return JSONResponse({"detail": e.detail}, status_code=e.status, headers=headers)

    @app.get("/api/stats")
    def stats():
        # Photographers' answers are cheap and change any minute: always the current ones.
        return {**stats_cache.get(), "permissions": permission_status()}

    @app.get("/api/scoreboard")
    def scoreboard():
        with db_lock:
            return {"runs": evaluate.scoreboard(conn)}

    @app.get("/api/prospective")
    def prospective_report():
        from . import prospective
        with db_lock:
            return {"models": prospective.report(conn), "sealed_after": prospective.seal_time()}

    @app.get("/api/benchmarks")
    def benchmarks():
        """Held-out benchmark results imported with mv benchmark-import (aggregates only)."""
        from . import benchmark_io
        with db_lock:
            return {"benchmarks": benchmark_io.published(conn)}

    from . import experiments

    def good_experiments() -> tuple[list, int]:
        """The registry's good entries, and how many were left out. A bad entry (one an author
        is still writing, say) is logged and skipped, never a 500 for the whole page."""
        found, problems = experiments.load_checked(experiments_dir or experiments.EXPERIMENTS_DIR)
        for name, why in problems:
            log.warning("[experiments] bad entry %s: %s", name, why)
        return found, len(problems)

    @app.get("/api/experiments")
    def experiments_list():
        """The experiment registry (config.RESEARCH_DIR/experiments, private), front matter only. Members only."""
        found = good_experiments()
        return JSONResponse({"experiments": [experiments.summary(e) for e in found[0]],
                             "skipped": found[1]},
                            headers={"Cache-Control": "private, no-store"})

    @app.get("/api/experiments/{slug}")
    def experiment(slug: str):
        if not experiments.SLUG.match(slug):
            raise HTTPException(404, "no such experiment")
        found = {e.slug: e for e in good_experiments()[0]}
        if slug not in found:
            raise HTTPException(404, "no such experiment")
        e = found[slug]
        return JSONResponse({**experiments.summary(e), "markdown": e.body},
                            headers={"Cache-Control": "private, no-store"})

    # On-demand CC images of a public dataset release (dataset_api.py).
    from . import dataset_api
    dataset_api.add_routes(app, root=dataset_root, store_location=dataset_store,
                           store_reader=dataset_store_reader, allow_draft=dataset_allow_draft)

    @app.get("/api/paper")
    def paper():
        """The paper draft as Markdown. Never cached on the way (an unreviewed draft)."""
        try:
            text = paper_file.read_text(encoding="utf-8")
        except OSError:
            raise HTTPException(404, "no paper draft on this server")
        return JSONResponse({"markdown": text}, headers={"Cache-Control": "private, no-store"})

    @app.get("/api/scoreboard/{run_id}")
    def scoreboard_run(run_id: int):
        with db_lock:
            report = evaluate.run_report(conn, run_id)
        if report is None:
            raise HTTPException(404, "no such run")
        return report

    # A plain (not async) route: FastAPI runs it in a worker thread, so GPU work
    # never blocks the event loop and other requests keep being answered.
    @app.post("/api/identify")
    def identify(request: Request, photos: list[UploadFile] = File(...),
                 models_: str = Form("", alias="models"), lat: float | None = Form(None),
                 lng: float | None = Form(None), observed_on: str | None = Form(None)):
        client = request.client.host if request.client else "unknown"
        wait = rate.check(client)
        if wait:
            raise HTTPException(429, "too many identifications from this address; "
                                     f"try again in {int(wait) + 1} s",
                                headers={"Retry-After": str(int(wait) + 1)})
        if not photos:
            raise HTTPException(400, "add at least one photo")
        if len(photos) > MAX_PHOTOS:
            raise HTTPException(400, f"at most {MAX_PHOTOS} photos")
        # The content-length check (cap_request_size) misses a chunked upload.
        if sum(f.size or 0 for f in photos) > MAX_REQUEST_BYTES:
            raise HTTPException(413, f"the photos are over {MAX_REQUEST_BYTES // 2**20} MB together")
        if not in_flight.enter():
            raise HTTPException(503, "busy identifying other photos; try again shortly",
                                headers={"Retry-After": "10"})
        if lat is not None and not -90 <= lat <= 90 or lng is not None and not -180 <= lng <= 180:
            in_flight.leave()
            raise HTTPException(400, "latitude or longitude out of range")
        context = Context(lat, lng, observed_on)
        timer = apiv1.Timer()
        try:
            try:
                model_queue.admit(interactive=True)
            except workqueue.Busy as e:
                raise HTTPException(503, "busy identifying other photos; try again shortly",
                                    headers={"Retry-After": str(int(e.wait_seconds) + 1)}) from e
            out = run_identify(photos, models_, context, timer, linked_of(request))
        finally:
            in_flight.leave()
        out["timing"] = timer.block()
        return out

    def decode_photos(photos: list[UploadFile], timer: "apiv1.Timer") -> list[tuple]:
        start = time.monotonic()
        side = uploads.kept_side(list(backbones.values()))
        read = [read_photo(f, side) for f in photos]         # one at a time, kept small
        timer.add("decode_ms", time.monotonic() - start)
        return read

    def embed_one(ident: Identifier, model, image) -> tuple:
        """One photo -> (its unit vector, float32; the gate's base vector or None when it is
        the vector itself or the gate can't run; the gate's verdict or None). Every photo
        is embedded on its own, by /identify and by sessions alike, so the same photo
        always gets the same vector: a forward pass over a batch differs from one over a
        single photo by ~1e-4, enough to swap near-tied names (integration testing,
        2026-10-12). Call with gpu_lock held."""
        prepare = getattr(model, "prepare", None) or (lambda im: im)
        vecs, base = ident.encode(model, [prepare(image)])
        vector = normalise(vecs)[0].astype(np.float32)
        gate = getattr(ident, "gate", None)
        verdict = gate.judge(base)[0] if gate is not None and base is not None else None
        return vector, (None if base is None or base is vecs else np.asarray(base[0])), verdict

    def embed_and_score(ident: Identifier, backbone: str, images: list, context: "Context",
                        may_show, timer: "apiv1.Timer", top_k: int = 5,
                        interactive: bool = True, higher: bool = False,
                        hidden: np.ndarray | None = None, n_specimens: int = 6,
                        vectors: list | None = None) -> dict:
        """One model's answer through the queue, timed: waiting, embedding, scoring.
        `hidden`: reference columns left out (Identifier.identify_vectors). `vectors`, when
        given, gets the photos' vectors appended (for yours.for_person)."""
        def embed():                                 # the worker holds gpu_lock
            model = backbone_model(backbone)
            parts = [embed_one(ident, model, im) for im in images]
            judged = [v for _, _, v in parts]
            verdicts.append(judged if all(v is not None for v in judged) else None)
            return np.stack([vec for vec, _, _ in parts])

        def finish(vecs, sims):
            if vectors is not None:
                vectors.append(np.asarray(vecs, dtype=np.float32))
            return ident.identify_vectors(vecs, top_k, n_specimens=n_specimens, context=context,
                                          sims=sims, photo_lookup=photo_lookup,
                                          may_show=may_show, gate=verdicts[0],
                                          higher_ranks=higher, hidden=hidden)
        verdicts: list = []
        return model_queue.run(workqueue.Job(
            key=id(ident), photos=len(images), embed=embed, search=ident.nearest.photo_sims,
            finish=finish, interactive=interactive, timer=timer))

    def served_index(backbone: str, method: str, view, timer: "apiv1.Timer") -> Identifier:
        """identifier(), its wait for a rebuild counted as queueing."""
        start = time.monotonic()
        try:
            return identifier(backbone, method, view)
        finally:
            timer.add("queue_ms", time.monotonic() - start)

    def site_models(models_: str) -> list[tuple[str, str]]:
        """The site's model choice ("backbone/method,..."; empty: the default) as
        (backbone, method) pairs, each one offered here, or an HTTPException."""
        ready = ready_backbones()
        wanted = [m.strip() for m in models_.split(",") if m.strip()]
        if not wanted:
            if not ready:
                raise HTTPException(503, "no model has embeddings yet")
            wanted = [f"{ready[0]}/{default_method()}"]
        if limits.max_models and len(wanted) > limits.max_models:
            raise HTTPException(400, f"compare at most {limits.max_models} models at a time here")
        pairs = []
        for spec in wanted:
            backbone, _, method = spec.partition("/")
            method = method or default_method()
            if backbone not in limits.allowed_backbones:
                raise HTTPException(400, f"{backbone!r} is not offered here")
            if backbone not in ready:
                raise HTTPException(400, f"{backbone!r} has no embeddings")
            if method not in evaluate.METHODS:
                raise HTTPException(400, f"unknown method {method!r}")
            if not limits.method_allowed(method) or not served_ready(method):
                raise HTTPException(400, f"{method!r} is not offered here")
            pairs.append((backbone, method))
        return pairs

    def site_index(backbone: str, method: str, view, timer: "apiv1.Timer") -> Identifier:
        """served_index() with its refusals as the site's HTTP errors."""
        try:
            return served_index(backbone, method, view, timer)
        except NotTrainedHere as e:
            raise HTTPException(503, f"{backbone}/{method} isn't ready on this "
                                     f"server ({e})") from e
        except serving.Updating as e:
            raise HTTPException(503, str(e), headers={"Retry-After": "30"}) from e

    def run_identify(photos: list[UploadFile], models_: str, context: "Context",
                     timer: "apiv1.Timer", linked: "yours.Linked | None" = None) -> dict:
        read = decode_photos(photos, timer)
        images = [img for img, _ in read]
        context, used = fill_context(context, [found for _, found in read])
        wanted = site_models(models_)
        view = permission_view()

        def may_show(info) -> bool:
            return view.may_show(info.license_class, info.owner_user_id)

        results = []
        for backbone, method in wanted:
            ident = site_index(backbone, method, view, timer)
            vectors: list = []
            pooled = embed_and_score(ident, backbone, images, context, may_show, timer,
                                     n_specimens=yours.POOL, vectors=vectors)
            result, spare = yours.split_pool(pooled)
            results.append(yours.for_person(ident, result, spare, vectors[0] if vectors else None,
                                            None, linked, photo_lookup, may_show))
        table = confused_table()
        for result in results:
            confusions.annotate(result, table)
        return {"results": results, "context_used": used}

    def backbone_model(backbone: str):
        """The model for one backbone, loaded once. Call with gpu_lock held."""
        if backbone not in backbones:
            backbones[backbone] = backbone_loader(backbone)
            models.release_memory()
        return backbones[backbone]

    def run_preload(specs: list[str]) -> None:
        start = time.monotonic()
        try:
            ready = ready_backbones()
            names = []
            for spec in specs:
                if spec == "default":
                    if not ready:
                        raise ValueError("no model has embeddings yet")
                    spec = f"{ready[0]}/{default_method()}"
                backbone, _, method = spec.partition("/")
                method = method or default_method()
                if backbone not in limits.allowed_backbones or backbone not in ready:
                    raise ValueError(f"{backbone!r} is not offered here or has no embeddings")
                if (method not in evaluate.METHODS or not limits.method_allowed(method)
                        or not served_ready(method)):
                    raise ValueError(f"{method!r} is unknown or not offered here")
                identifier(backbone, method, permission_view())
                with gpu_lock:
                    backbone_model(backbone)
                names.append(f"{backbone}/{method}")
            warm.models, warm.state = names, "ready"
            warm.seconds = round(time.monotonic() - start, 1)
            note(f"[preload] {', '.join(names)} ready in {warm.seconds} s")
        except Exception as e:  # noqa: BLE001 - the server keeps answering; loads on demand
            warm.state, warm.error = "failed", f"{type(e).__name__}: {e}"
            note(f"[preload] {warm.error}")

    specs = preload_specs(config.setting("MV_PRELOAD")) if preload is None else preload
    if specs:
        warm.state = "loading"
        threading.Thread(target=run_preload, args=(specs,), name="preload", daemon=True).start()

    # Background: keep photographers' answers and photo licences current.
    if background:
        base_url, key = config.setting("MV_ORG_BASE_URL"), permissions.org_key()
        if base_url and key:
            every = _setting_number("MV_PERMISSIONS_SYNC_SECONDS") or 300

            def sync_permissions():
                own = sqlite3.connect(manifest_path, timeout=30)
                try:
                    permissions.sync(own, base_url, key)
                finally:
                    own.close()
            start_background("permissions", every, sync_permissions, note)
        else:
            note("[permissions] MV_ORG_BASE_URL / MV_ORG_VISION_KEY(_FILE) not set: all-rights-reserved "
                "photos are not shown, and refusals on mycomap.org are not picked up.")
        hours = _setting_number("MV_LICENSE_REFRESH_HOURS")
        if hours and hours > 0:
            def refresh_licences():
                own = sqlite3.connect(manifest_path, timeout=30)
                own.row_factory = sqlite3.Row
                try:
                    stats = inat.refresh_licenses(own, older_than_hours=hours, log=note)
                    if stats["requested"]:
                        note(f"[licences] re-read {stats['requested']:,} records, "
                            f"{stats['license_changes']:,} licence changes")
                finally:
                    own.close()
            start_background("licences", 3600, refresh_licences, note)

    # Use reported to mycomap.org (quotas.py): running totals every MV_USAGE_REPORT_SECONDS
    # (300), so a failed report is simply sent again next time.
    app.state.report_usage = None
    if quotas.on:
        post = usage_post
        if post is None and config.setting("MV_ORG_BASE_URL") and permissions.org_key():
            post = quota_rules.org_poster(config.setting("MV_ORG_BASE_URL"),
                                          permissions.org_key())
        if post is not None:
            app.state.report_usage = lambda: quota_rules.report(quotas, post, note=note)
            if background:
                start_background("usage report",
                                 _setting_number("MV_USAGE_REPORT_SECONDS") or 300,
                                 app.state.report_usage, note)

    # The nightly update (nightly.py): runs here because this process holds the model.
    nightly_settings = nightly_settings or (nightly.Settings.from_env() if layer else None)
    nightly_zone = None
    if layer:
        try:
            nightly_zone = nightly_settings.zone()
        except Exception as e:  # noqa: BLE001 - reported; the schedule below would fail too
            note(f"[nightly] time zone {nightly_settings.tz!r}: {type(e).__name__}: {e}")

    # mycomap.org's name-change log (name_changes.py), read into the nightly layer only: when
    # a Name Update Vision asked for is done, every few minutes, and before each night. Never
    # without a layer, so a release's own manifest is never written.
    # Read only with mycomap.org's address set (or a client given): a test never reads .org.
    from . import feedback_api
    org_named = feedback_org is not None or bool(config.setting("MV_ORG_BASE_URL"))
    feedback_org = feedback_org or feedback_api.OrgClient()
    name_change_reader = None
    if layer and org_named:
        def layer_db():
            own = manifest.connect(manifest_path)
            own.execute("pragma busy_timeout = 30000")
            return own
        # The cursor in <release root>/state/, across releases (name_changes.py).
        name_change_reader = name_changes.Reader(layer.root.parent.parent
                                                 / "org-name-changes.sqlite", feedback_org,
                                                 layer_db, note=note)
    app.state.name_change_reader = name_change_reader

    def read_name_changes() -> dict | None:
        return name_change_reader.run() if name_change_reader else None

    def run_nightly(accept_removals: bool = False) -> dict:
        """One nightly update, then the preloaded indexes rebuilt so the first person
        after it doesn't wait for the new reference set."""
        def org() -> tuple[str, str]:
            base_url, key = config.setting("MV_ORG_BASE_URL"), permissions.org_key()
            if not (base_url and key):
                raise RuntimeError("MV_ORG_BASE_URL / MV_ORG_VISION_KEY(_FILE) not set")
            return base_url, key
        with db_lock:                      # Human Feedback's photo decisions, once a night
            note(f"[feedback] photo exclusions: {feedback.sync_exclusions(feedback_store, conn)}")
        read_name_changes()                # .org's renames first: tonight counts only the rest
        fetch = nightly_fetch or (lambda: nightly.fetch_green(*org(), log=note))
        fetch_pending = nightly_fetch_pending or (lambda: nightly.fetch_pending(*org(), log=note))

        def loaded(backbone: str):
            def get():
                with gpu_lock:
                    return backbone_model(backbone)
            return get
        served = {b: loaded(b) for b in ready_backbones()}
        own = manifest.connect(manifest_path)
        own.execute("pragma busy_timeout = 30000")      # the permissions sync writes too
        try:
            report = nightly.run_once(own, layer, fetch, served, nightly_settings,
                                      encode_lock=gpu_lock, accept_removals=accept_removals,
                                      steps=nightly_steps, log=note)
        finally:
            own.close()
        if confused_with is not None:          # tonight's table, if the read took
            confused_with.reload()
        if report["changed"]:
            for spec in warm.models:
                backbone, _, method = spec.partition("/")
                try:
                    indexes.get((backbone, method),
                                index_version(backbone, method, permission_view()), wait=None)
                except Exception as e:  # noqa: BLE001 - the next identification retries
                    note(f"[nightly] rebuilding {spec} failed: {type(e).__name__}: {e}")
        # Advance predictions with the model people get by default, on the index just
        # rebuilt: what it says before the DNA answer, scored once the record turns green.
        if nightly_settings.predict > 0 and warm.models:
            report["predictions"] = predict_pending_tonight(report["id"], fetch_pending)
        return report

    def predict_pending_tonight(run_id: int, fetch_pending) -> dict:
        from .prospective import PhotoFetcher
        backbone, _, method = warm.models[0].partition("/")

        def served():
            ident = indexes.get((backbone, method),
                                index_version(backbone, method, permission_view()), wait=None)
            with gpu_lock:
                return ident, nightly.Locked(backbone_model(backbone), gpu_lock)
        own = manifest.connect(manifest_path)
        own.execute("pragma busy_timeout = 30000")
        try:
            return nightly.advance_predictions(own, run_id, fetch_pending, served,
                                               (photo_fetcher or PhotoFetcher)(),
                                               nightly_settings.predict, log=note)
        finally:
            own.close()

    app.state.run_nightly = run_nightly if layer else None
    if layer and background and nightly_zone is None:
        note("[nightly] NOT scheduled: fix MV_NIGHTLY_TZ (an IANA name, e.g. America/New_York)")
    elif layer and background:
        schedule = nightly.Schedule(layer, nightly_settings, nightly_zone)
        nightly.start(schedule, run_nightly, note)

    # Human Feedback (feedback.py, feedback_api.py; Steve, 2026-10-18): the review queue of
    # where Vision and the reference data disagree, for mycomap.org's admins, project leaders
    # and contributors. Its decisions are kept with the usage counts (state/), across
    # releases. "Refresh" re-asks the served model about a reference record from its stored
    # vectors, the record itself left out, behind the same queue as everything else.
    if feedback_store is None:
        feedback_store = feedback.Store(
            (config.RELEASE_ROOT / "state" if config.RELEASE_ROOT else config.DATA_DIR)
            / "human-feedback.sqlite")
    app.state.feedback = feedback_store
    with db_lock:
        feedback.sync_exclusions(feedback_store, conn)

    def feedback_rescore(record_key: str) -> dict:
        source, number = (("mo", record_key[3:]) if record_key.startswith("mo:")
                          else ("inat", record_key))
        ready = ready_backbones()
        if not ready:
            return {"available": False, "reason": "no model is ready"}
        backbone, method = ready[0], default_method()
        timer = apiv1.Timer()
        try:
            ident = served_index(backbone, method, permission_view(), timer)
        except (serving.Updating, NotTrainedHere) as e:
            return {"available": False, "reason": f"the model is being updated ({e})"}
        hidden, own = ident.left_out(source, number)
        if not len(own) or not ident.can_leave_out():
            return {"available": False, "reason": "not a reference record of the served "
                                                  "model: open it in Identify"}
        vecs = ident.reference_vectors(own[:MAX_PHOTOS])

        def finish(v, sims):
            return ident.identify_vectors(v, 5, context=None, sims=sims,
                                          photo_lookup=photo_lookup, gate=None, hidden=hidden)
        result = model_queue.run(workqueue.Job(
            key=id(ident), photos=0, embed=lambda: vecs, search=ident.nearest.photo_sims,
            finish=finish, interactive=False, timer=timer))
        return {"available": True, "model": f"{backbone}/{method}",
                "top5": [[c["name"], c["score"]] for c in result["ranks"]["species"][:5]]}

    from . import photo_review
    taxonomy_now = None

    def genus_known(genus: str) -> bool | None:
        nonlocal taxonomy_now
        if taxonomy_now is None:
            from . import taxonomy
            path = (layer.taxonomy_cache if layer else Path(manifest_path).parent / taxonomy.CACHE)
            taxonomy_now = taxonomy.load(path) or False
        return bool(taxonomy_now) and taxonomy_now.is_genus(genus)

    def name_info(wanted: list[str]) -> dict[str, dict]:
        with db_lock:
            return feedback_org_rules.manifest_name_info(conn, genus_known)(wanted)

    feedback_requests = feedback_api.add_routes(
        app, person=person, store=feedback_store, org=feedback_org, rescore=feedback_rescore,
        photos=photo_review.ReviewPhotos(q, permission_view), name_info=name_info,
        on_name_update_done=read_name_changes, note=note)
    app.state.feedback_requests = feedback_requests
    if background and org_named:           # unfinished Name Updates, and the log beside them
        def follow_name_updates():
            feedback_requests.poll_once()
            read_name_changes()
        start_background("feedback name updates", feedback_org_rules.POLL_SECONDS,
                         follow_name_updates, note)

    v1_on = bool(api_v1 if api_v1 is not None else config.setting("MV_API_V1") == "1")
    sessions_on = bool(sessions if sessions is not None
                       else config.setting("MV_SESSIONS") == "1")
    by_observation_on = v1_on and bool(by_observation if by_observation is not None
                                       else config.setting("MV_BY_OBSERVATION") == "1")

    # The API's public outline (apidocs.py): the Vision site's API page (Research > API) and the tab
    # on mycomap.org/api-docs both render it. Open to anyone, from any site.
    v1_parts = {"identify": v1_on, "sessions": sessions_on,
                "by_observation": by_observation_on, "models": v1_on, "usage": v1_on,
                "taxon_counts": bool(taxon_counts if taxon_counts is not None
                                     else config.setting("MV_TAXON_COUNTS") == "1")}

    @app.get("/api/docs/v1.json")
    def api_outline():
        return JSONResponse(apidocs.outline(v1_parts),
                            headers={"Access-Control-Allow-Origin": "*",
                                     "Cache-Control": "public, max-age=300"})

    # POST /api/v1/identify (docs/api.md §3.1, §4): one model's answer for 1-10 photos,
    # with timing, confident/answer, and the caller's own short-window limit. The app's
    # observation sessions (§3.3) share its helpers.
    if v1_on or sessions_on:
        photos_per_minute = int(_setting_number("MV_PHOTOS_PER_MINUTE") or 20)
        photo_window = apiv1.PhotoWindow(photos_per_minute)

        def check_context(latitude, longitude, observed_on) -> None:
            """Place and date as the caller sent them, or 400 bad_request."""
            def number(v):
                return v is None or (isinstance(v, (int, float)) and not isinstance(v, bool))
            if not (number(latitude) and number(longitude)) or (latitude is None) != (
                    longitude is None) or (latitude is not None and not (
                    -90 <= latitude <= 90 and -180 <= longitude <= 180)):
                raise apiv1.V1Error(400, "bad_request", "send latitude and longitude together, "
                                    "in range")
            if observed_on is not None:
                try:
                    date.fromisoformat(observed_on)
                except (TypeError, ValueError):
                    raise apiv1.V1Error(400, "bad_request", "observed_on is YYYY-MM-DD") from None

        def take_photos(cred: "apiv1.Credential", n: int,
                        interactive: bool | None = None) -> None:
            """The caller's photos-per-minute window, then room in the queue, or refuse.
            `interactive` overrides the credential's (by-observation calls never are)."""
            if cred.kind != "server" and n:
                wait = photo_window.take(cred.rate_key, n)
                if wait:
                    quotas.refused(cred, "rate_limited")
                    raise apiv1.V1Error(429, "rate_limited", "too many photos in the last "
                                        "minute", retry_after=int(wait) + 1)
            try:
                model_queue.admit(interactive=cred.interactive if interactive is None
                                  else interactive)
            except workqueue.Busy as e:
                quotas.refused(cred, "busy")
                wait = int(e.wait_seconds) + 1
                raise apiv1.V1Error(503, "busy", "busy identifying other photos",
                                    retry_after=wait, estimated_wait_seconds=wait) from e

        def v1_index(backbone: str, method: str, view, timer) -> Identifier:
            try:
                return served_index(backbone, method, view, timer)
            except (serving.Updating, NotTrainedHere) as e:
                note(f"[v1] {backbone}/{method} not ready: {e}")
                raise apiv1.V1Error(503, "updating", "the model is being updated; try "
                                    "again shortly", retry_after=30) from e

        def v1_model(spec: str | None) -> tuple[str, str]:
            ready = ready_backbones()
            if not spec:
                if not ready:
                    raise apiv1.V1Error(503, "updating", "no model is ready yet", retry_after=30)
                return ready[0], default_method()
            backbone, _, method = spec.strip().partition("/")
            method = method or default_method()
            if (backbone not in limits.allowed_backbones or backbone not in ready
                    or method not in evaluate.METHODS or not limits.method_allowed(method)
                    or not served_ready(method)):
                raise apiv1.V1Error(400, "bad_request", f"model {spec!r} is not offered here; "
                                    "see GET /api/v1/models")
            return backbone, method

        def v1_photos(photos: list[UploadFile], timer: "apiv1.Timer") -> list[tuple]:
            try:
                return decode_photos(photos, timer)
            except HTTPException as e:
                if e.status_code == 413:
                    raise apiv1.V1Error(413, "too_large", str(e.detail)) from e
                raise apiv1.V1Error(415, "unsupported_image", str(e.detail)) from e

        def v1_identify(request: Request, photos: list[UploadFile] = File(...),
                        latitude: float | None = Form(None), longitude: float | None = Form(None),
                        observed_on: str | None = Form(None), model: str | None = Form(None),
                        top_k: int = Form(5), client_ref: str | None = Form(None)):
            timer = apiv1.Timer()
            cred: apiv1.Credential = request.state.credential
            if not 1 <= len(photos) <= MAX_PHOTOS:
                raise apiv1.V1Error(400, "too_many_photos", f"send 1 to {MAX_PHOTOS} photos")
            if sum(f.size or 0 for f in photos) > MAX_REQUEST_BYTES:
                raise apiv1.V1Error(413, "too_large", "the photos are over "
                                    f"{MAX_REQUEST_BYTES // 2**20} MB together")
            if not 1 <= top_k <= 10:
                raise apiv1.V1Error(400, "bad_request", "top_k is 1 to 10")
            check_context(latitude, longitude, observed_on or None)
            backbone, method = v1_model(model)
            if client_ref is not None and not (
                    1 <= len(client_ref) <= session_store.MAX_REF_LENGTH):
                raise apiv1.V1Error(400, "bad_request", "client_ref is 1 to "
                                    f"{session_store.MAX_REF_LENGTH} characters")
            # The same find counted in the last day (its session, before signal dropped)
            # is free here too: the app finishes a find this way after saving it.
            same_find = quotas.counted_before(cred, client_ref)
            if not same_find:
                quotas.check(cred)                  # before the photo window is spent
            take_photos(cred, len(photos))
            if not in_flight.enter():
                raise apiv1.V1Error(503, "busy", "busy identifying other photos",
                                    retry_after=10, estimated_wait_seconds=10)
            try:
                read = v1_photos(photos, timer)
                context, used = fill_context(Context(latitude, longitude, observed_on),
                                             [found for _, found in read])
                view = permission_view()
                ident = v1_index(backbone, method, view, timer)
                result = embed_and_score(
                    ident, backbone, [img for img, _ in read], context,
                    lambda info: view.may_show(info.license_class, info.owner_user_id),
                    timer, top_k, interactive=cred.interactive, higher=True)
            finally:
                in_flight.leave()
            quotas.count(cred, 0 if same_find else 1, len(photos))
            if not same_find:
                quotas.remember_counted(cred, client_ref)
            body = v1_result(result, backbone, method, used, timer)
            body["usage"] = quotas.usage(cred)
            return JSONResponse(body, headers={"Cache-Control": "no-store",
                                               "Server-Timing": apiv1.Timer.header(body["timing"])})

        def v1_result(result: dict, backbone: str, method: str, used: dict,
                      timer: "apiv1.Timer") -> dict:
            """An identify result in the v1 shape (docs/api.md §4)."""
            cal = result.pop("calibration", None) or {}
            # Backbone and method ids never leave in a v1 answer (Steve, 2026-10-11).
            prior = result.pop("prior", None) or {}
            out = {"request_id": "r_" + secrets.token_urlsafe(12),
                   **result,
                   "model": apiv1.public_model(model_card(),
                                               str(cal.get("comparison_id") or "")),
                   "prior": {"applied": bool(prior.get("applied"))},
                   "context_used": used,
                   "gate": result.get("gate"),
                   "usage": None}
            apiv1.mark_confident(out)
            confusions.annotate(out, confused_table())
            out["timing"] = timer.block()
            return out

        if v1_on:
            app.post("/api/v1/identify")(v1_identify)

            # GET /api/v1/models: the model answering now, by its public name only.
            @app.get("/api/v1/models")
            def v1_models(request: Request):
                backbone, method = v1_model(None)
                ident = v1_index(backbone, method, permission_view(), apiv1.Timer())
                cal = (ident.calibration or {}).get("comparison_id")
                doc = apiv1.models_doc(
                    {**apiv1.public_model(model_card(), str(cal or "")),
                     "answers_version": answers_version(backbone, method, permission_view())},
                    {"records": ident.records, "species": len(ident.index.species),
                     "photos": ident.embedded},
                    {"max_photos": MAX_PHOTOS, "max_file_mb": MAX_FILE_BYTES // 2**20,
                     "max_request_mb": MAX_REQUEST_BYTES // 2**20, "top_k_max": 10,
                     "photos_per_minute": photos_per_minute},
                    {k: bool(v) for k, v in v1_parts.items()}, ident.calibration)
                return JSONResponse(doc, headers={"Cache-Control": "private, max-age=300"})

            # GET /api/v1/usage: this credential's use this month. Reading it costs nothing.
            @app.get("/api/v1/usage")
            def v1_usage(request: Request):
                cred: apiv1.Credential = request.state.credential
                return JSONResponse(apiv1.usage_doc(cred, quotas.usage(cred), quotas.mode),
                                    headers={"Cache-Control": "no-store"})

        # POST /api/v1/identify/{inat|mo}/{number} (docs/api.md §3.2): an observation's own
        # photos, place and date, read politely from iNat or MO (byobservation.py) and kept
        # in memory only. An observation that is itself a DNA reference record is answered
        # from its stored vectors with that record left out of the reference, and without
        # its place and date (the record is part of the place prior); so is one whose
        # photo is filed under a reference record. Behind interactive calls in the queue.
        if by_observation_on:
            fetcher = observation_fetcher or byobservation.Fetcher()

            def flag(payload: dict, name: str) -> bool:
                v = payload.get(name, True)
                if not isinstance(v, bool):
                    raise apiv1.V1Error(400, "bad_request", f"{name} is true or false")
                return v

            def v1_identify_observation(request: Request, source: str, observation_id: str,
                                        payload: dict | None = Body(None)):
                timer = apiv1.Timer()
                timer.ms["fetch_ms"] = 0
                cred: apiv1.Credential = request.state.credential
                if source not in byobservation.SOURCES:
                    raise apiv1.V1Error(404, "not_found", "identify by observation takes "
                                        "inat or mo")
                number = byobservation.valid_id(observation_id)
                if number is None:
                    raise apiv1.V1Error(400, "bad_request", "the observation number is digits")
                payload = {} if payload is None else payload
                if not isinstance(payload, dict):
                    raise apiv1.V1Error(400, "bad_request", "send a JSON object")
                top_k = payload.get("top_k", 5)
                if isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= 10:
                    raise apiv1.V1Error(400, "bad_request", "top_k is 1 to 10")
                use_place, use_date = flag(payload, "use_place"), flag(payload, "use_date")
                model = payload.get("model")
                if model is not None and not isinstance(model, str):
                    raise apiv1.V1Error(400, "bad_request", "model is a string")
                # The same observation for the same pool within a day is the same find.
                client_ref = payload.get("client_ref") or f"{source}:{number}"
                if not isinstance(client_ref, str) or not (
                        1 <= len(client_ref) <= session_store.MAX_REF_LENGTH):
                    raise apiv1.V1Error(400, "bad_request", "client_ref is 1 to "
                                        f"{session_store.MAX_REF_LENGTH} characters")
                backbone, method = v1_model(model)
                same_find = quotas.counted_before(cred, client_ref)
                if not same_find:
                    quotas.check(cred)
                view = permission_view()
                # Already answered under what Vision answers from now: the same answer,
                # without reading the observation or scoring it again.
                version = answers_version(backbone, method, view)
                kept_key = answercache.answer_key(
                    "v1", source, number, backbone, method,
                    {"top_k": top_k, "place": use_place, "date": use_date})
                kept = answer_cache.get(kept_key, version)
                if kept is not None:
                    quotas.count(cred, 0 if same_find else 1, 0)
                    if not same_find:
                        quotas.remember_counted(cred, client_ref)
                    body = {**kept, "request_id": "r_" + secrets.token_urlsafe(12),
                            "cached": True}
                    confusions.annotate(body, confused_table())
                    body["timing"] = timer.block()
                    body["usage"] = quotas.usage(cred)
                    return JSONResponse(body, headers={
                        "Cache-Control": "no-store",
                        "Server-Timing": apiv1.Timer.header(body["timing"])})
                ident = v1_index(backbone, method, view, timer)
                with_own, own = ident.left_out(source, number)
                if len(own) and not ident.can_leave_out():
                    raise apiv1.V1Error(400, "bad_request", "this model can't leave the "
                                        "observation's own reference record out")
                none_used = {"latitude": None, "longitude": None, "observed_on": None,
                             "place_from": None, "date_from": None}
                about = {"source": source, "id": number,
                         "url": sources.record_url(source, number),
                         "photos": None, "photos_used": 0, "place": None,
                         "name_at_source": None, "rank_at_source": None,
                         "photos_shown": None}
                if len(own):
                    # A DNA reference record: its stored vectors, nothing fetched.
                    cols = own[:MAX_PHOTOS]
                    take_photos(cred, len(cols), interactive=False)
                    vecs = ident.reference_vectors(cols)
                    hidden, context, used = with_own, None, none_used
                    about["photos_used"] = n_photos = len(cols)
                else:
                    def admit(obs) -> int:
                        n = min(MAX_PHOTOS, len(obs.photo_urls))
                        if n == 0:
                            raise apiv1.V1Error(422, "no_usable_photos", "the observation has "
                                                "no photos Vision can read")
                        take_photos(cred, n, interactive=False)
                        return n
                    start = time.monotonic()
                    try:
                        obs, pairs = fetcher.fetch(source, number, MAX_PHOTOS, admit=admit,
                                                   indexed=True)
                    except byobservation.NotFound as e:
                        raise apiv1.V1Error(404, "observation_not_found", str(e)) from None
                    except byobservation.Unavailable as e:
                        raise apiv1.V1Error(502, "source_unavailable", str(e),
                                            retry_after=e.retry_after) from None
                    except byobservation.Busy as e:
                        raise apiv1.V1Error(503, "busy", "busy fetching other observations",
                                            retry_after=e.retry_after,
                                            estimated_wait_seconds=e.retry_after) from None
                    finally:
                        timer.add("fetch_ms", time.monotonic() - start)
                    start = time.monotonic()
                    side = uploads.kept_side(list(backbones.values()))
                    images, kept = [], []
                    for index, photo_body in pairs:
                        try:
                            images.append(uploads.decode_upload(photo_body, side)[0])
                        except Exception:               # not an image we can read: skip
                            continue
                        kept.append(index)
                    timer.add("decode_ms", time.monotonic() - start)
                    if not images:
                        raise apiv1.V1Error(422, "no_usable_photos", "none of the "
                                            "observation's photos could be read")
                    # Each scored photo's URL where it may be shown (Creative Commons or
                    # public domain), in per_photo's order; null otherwise.
                    about.update(photos=len(obs.photo_urls), photos_used=len(images),
                                 name_at_source=obs.name_at_source,
                                 rank_at_source=obs.rank_at_source,
                                 photos_shown=[byobservation.shown_photo_url(obs, i)
                                               for i in kept])
                    hidden, _ = ident.left_out(source, number, obs.photo_ids)
                    if len(hidden) and not ident.can_leave_out():
                        raise apiv1.V1Error(400, "bad_request", "this model can't leave "
                                            "the observation's photos out of the reference")
                    if len(hidden):
                        context, used = None, none_used
                    else:
                        lat, lng = ((obs.latitude, obs.longitude) if use_place
                                    else (None, None))
                        context, used = fill_context(
                            Context(lat, lng, obs.observed_on if use_date else None), [])
                        used = {**used,
                                "place_from": "observation" if used["place_from"] else None,
                                "date_from": "observation" if used["date_from"] else None}
                        if used["place_from"]:
                            about["place"] = obs.place
                    n_photos = len(images)
                if not in_flight.enter():
                    raise apiv1.V1Error(503, "busy", "busy identifying other photos",
                                        retry_after=10, estimated_wait_seconds=10)
                try:
                    def may_show(info) -> bool:
                        return view.may_show(info.license_class, info.owner_user_id)
                    if len(own):
                        def finish(v, sims):
                            return ident.identify_vectors(
                                v, top_k, context=None, sims=sims, photo_lookup=photo_lookup,
                                may_show=may_show, gate=None, higher_ranks=True,
                                hidden=hidden)
                        result = model_queue.run(workqueue.Job(
                            key=id(ident), photos=0, embed=lambda: vecs,
                            search=ident.nearest.photo_sims, finish=finish,
                            interactive=False, timer=timer))
                    else:
                        result = embed_and_score(ident, backbone, images, context or Context(),
                                                 may_show, timer, top_k, interactive=False,
                                                 higher=True,
                                                 hidden=hidden if len(hidden) else None)
                finally:
                    in_flight.leave()
                quotas.count(cred, 0 if same_find else 1, n_photos)
                if not same_find:
                    quotas.remember_counted(cred, client_ref)
                body = v1_result(result, backbone, method, used, timer)
                body["observation"] = about
                body["excluded_reference"] = ({"source": source, "source_id": number}
                                              if len(own) else None)
                body["left_out_photos"] = int(len(hidden))
                body["model"] = {**body["model"], "answers_version": version}
                answer_cache.put(kept_key, version, answercache.without_confusions(
                    {k: v for k, v in body.items() if k not in ("request_id", "timing", "usage")}))
                body["cached"] = False
                body["usage"] = quotas.usage(cred)
                return JSONResponse(body, headers={
                    "Cache-Control": "no-store",
                    "Server-Timing": apiv1.Timer.header(body["timing"])})

            app.post("/api/v1/identify/{source}/{observation_id}")(v1_identify_observation)

            # GET /api/v1/answers/{inat|mo}/{number} (docs/api.md §3.2): the answer Vision
            # already holds for an observation under what it answers from now, from an
            # earlier identify-by-observation or from someone's run on the site's Identify
            # page, in the v1 shape. Nothing is read or scored and nothing counted, so .org
            # can show a record's answer the moment it exists. 404 `not_answered` otherwise.
            def v1_kept_answer(request: Request, source: str, observation_id: str,
                               model: str | None = None, top_k: int = 5,
                               use_place: bool = True, use_date: bool = True):
                timer = apiv1.Timer()
                cred: apiv1.Credential = request.state.credential
                if source not in byobservation.SOURCES:
                    raise apiv1.V1Error(404, "not_found", "answers are kept for inat or mo")
                number = byobservation.valid_id(observation_id)
                if number is None:
                    raise apiv1.V1Error(400, "bad_request", "the observation number is digits")
                if not 1 <= top_k <= 10:
                    raise apiv1.V1Error(400, "bad_request", "top_k is 1 to 10")
                backbone, method = v1_model(model)
                view = permission_view()
                version = answers_version(backbone, method, view)
                kept = answer_cache.get(answercache.answer_key(
                    "v1", source, number, backbone, method,
                    {"top_k": top_k, "place": use_place, "date": use_date}), version)
                if kept is not None:
                    body = {**kept, "request_id": "r_" + secrets.token_urlsafe(12)}
                    confusions.annotate(body, confused_table())
                    kept_from = "api"
                else:
                    # The site's Identify page keeps its own answers ("from a link"); they
                    # answer the same photos, place and date with the same model.
                    held = answer_cache.get(answercache.answer_key(
                        "site", source, number, backbone, method,
                        {"place": use_place, "date": use_date}), version)
                    if held is None:
                        raise apiv1.V1Error(404, "not_answered", "Vision holds no answer for "
                                            "this observation under what it answers from now")
                    site_about = dict(held["observation"])
                    own = site_about.pop("excluded_reference", False)
                    left = site_about.pop("left_out_photos", 0)
                    body = v1_result(held["result"], backbone, method, held["used"], timer)
                    body["observation"] = {"place": None, **site_about}
                    body["excluded_reference"] = ({"source": source, "source_id": number}
                                                  if own else None)
                    body["left_out_photos"] = int(left or 0)
                    body["model"] = {**body["model"], "answers_version": version}
                    kept_from = "site"
                body.update(cached=True, kept_from=kept_from, timing=timer.block(),
                            usage=quotas.usage(cred))
                return JSONResponse(body, headers={"Cache-Control": "no-store"})

            app.get("/api/v1/answers/{source}/{observation_id}")(v1_kept_answer)

            # POST /api/identify/observation (the Identify page's "from a link"): the same
            # polite read of an iNat or MO observation as above, for a signed-in person on
            # the site, answered in /api/identify's shape (one result per chosen model).
            # The observation, if it is a DNA reference record, and any reference column
            # holding one of its photos are left out of the answer, and then its place and
            # date are not used (the record is part of the place prior). The page is told
            # which photos it may show: Creative Commons or public domain only
            # (byobservation.shown_photo_url).
            @app.post("/api/identify/observation")
            def identify_observation(request: Request, payload: dict = Body(...)):
                client = request.client.host if request.client else "unknown"
                wait = rate.check(client)
                if wait:
                    raise HTTPException(429, "too many identifications from this address; "
                                             f"try again in {int(wait) + 1} s",
                                        headers={"Retry-After": str(int(wait) + 1)})
                source = payload.get("source")
                number = byobservation.valid_id(str(payload.get("id") or ""))
                if source not in byobservation.SOURCES or number is None:
                    raise HTTPException(400, "give an iNaturalist or Mushroom Observer "
                                             "observation number")
                models_ = payload.get("models") or ""
                if not isinstance(models_, str):
                    raise HTTPException(400, "models is a comma-separated string")
                use_place, use_date = (payload.get(k, True) is not False
                                       for k in ("use_place", "use_date"))
                wanted = site_models(models_)
                timer = apiv1.Timer()
                timer.ms["fetch_ms"] = 0
                view = permission_view()
                versions = [answers_version(b, m, view) for b, m in wanted]
                keys = [answercache.answer_key("site", source, number, b, m,
                                               {"place": use_place, "date": use_date})
                        for b, m in wanted]
                held = [answer_cache.get(k, v) for k, v in zip(keys, versions)]
                linked = linked_of(request)

                def may_show(info) -> bool:
                    return view.may_show(info.license_class, info.owner_user_id)

                # A kept answer is everyone's: it holds no one's records, only the photos'
                # vectors, so the person's own part is made here. One kept before it held
                # them can't make it: answer afresh for a person with accounts.
                needs_vectors = linked is not None and linked.linked
                if held and all(h is not None and (not needs_vectors or yours.decode_vectors(
                        h.get("vectors")) is not None) for h in held):
                    table = confused_table()
                    results = []
                    for (backbone, method), h in zip(wanted, held):
                        vecs = yours.decode_vectors(h.get("vectors")) if needs_vectors else None
                        ident = site_index(backbone, method, view, timer) if needs_vectors else None
                        mine = yours.for_person(
                            ident, h["result"], h.get("spare") or [], vecs,
                            np.asarray(h.get("hidden") or [], dtype=np.int64), linked,
                            photo_lookup, may_show)
                        confusions.annotate(mine, table)
                        results.append(mine)
                    return {"results": results,
                            "context_used": held[0]["used"],
                            "observation": held[0]["observation"],
                            "cached": True,
                            "timing": timer.block()}
                start = time.monotonic()
                try:
                    obs, pairs = fetcher.fetch(source, number, MAX_PHOTOS, indexed=True)
                except byobservation.NotFound as e:
                    raise HTTPException(404, str(e)) from None
                except byobservation.Unavailable as e:
                    raise HTTPException(502, f"{e}; try again shortly",
                                        headers={"Retry-After": str(e.retry_after)}) from None
                except byobservation.Busy as e:
                    raise HTTPException(503, "busy reading other observations; try again "
                                             "shortly",
                                        headers={"Retry-After": str(e.retry_after)}) from None
                finally:
                    timer.add("fetch_ms", time.monotonic() - start)
                start = time.monotonic()
                side = uploads.kept_side(list(backbones.values()))
                images, kept = [], []
                for index, photo_body in pairs:
                    try:
                        images.append(uploads.decode_upload(photo_body, side)[0])
                    except Exception:               # not an image we can read: skip
                        continue
                    kept.append(index)
                timer.add("decode_ms", time.monotonic() - start)
                if not images:
                    raise HTTPException(422, "the observation has no photos Vision can read")

                if not in_flight.enter():
                    raise HTTPException(503, "busy identifying other photos; try again shortly",
                                        headers={"Retry-After": "10"})
                try:
                    try:
                        model_queue.admit(interactive=True)
                    except workqueue.Busy as e:
                        raise HTTPException(503, "busy identifying other photos; try again "
                                                 "shortly",
                                            headers={"Retry-After": str(int(e.wait_seconds) + 1)}
                                            ) from e
                    results, used, own_any, left = [], None, False, 0
                    scored = []                 # per model: (index, vectors, hidden columns)
                    for backbone, method in wanted:
                        ident = site_index(backbone, method, view, timer)
                        hidden, own = ident.left_out(source, number, obs.photo_ids)
                        if len(hidden) and not ident.can_leave_out():
                            raise HTTPException(400, f"{backbone}/{method} can't leave this "
                                                     "observation out of its own answer")
                        if len(hidden):
                            context, this_used = Context(None, None, None), {
                                "latitude": None, "longitude": None, "observed_on": None,
                                "place_from": None, "date_from": None}
                        else:
                            lat, lng = ((obs.latitude, obs.longitude) if use_place
                                        else (None, None))
                            context, this_used = fill_context(
                                Context(lat, lng, obs.observed_on if use_date else None), [])
                            this_used = {
                                **this_used,
                                "place_from": "observation" if this_used["place_from"] else None,
                                "date_from": "observation" if this_used["date_from"] else None}
                        used = used or this_used
                        own_any, left = own_any or bool(len(own)), max(left, int(len(hidden)))
                        vectors: list = []
                        results.append(embed_and_score(
                            ident, backbone, images, context, may_show, timer,
                            hidden=hidden if len(hidden) else None, n_specimens=yours.POOL,
                            vectors=vectors))
                        scored.append((ident, vectors[0] if vectors else None, hidden))
                finally:
                    in_flight.leave()
                about = {
                        "source": source, "id": number,
                        "url": sources.record_url(source, number),
                        "photos": len(obs.photo_urls), "photos_used": len(images),
                        "name_at_source": obs.name_at_source,
                        "rank_at_source": obs.rank_at_source,
                        # One entry per photo scored, in per_photo's order: its URL when
                        # the page may show it, else null.
                        "photos_shown": [byobservation.shown_photo_url(obs, i) for i in kept],
                        "excluded_reference": own_any,
                        "left_out_photos": left,
                }
                table = confused_table()
                answered = []
                for key, version, pooled, (ident, vecs, hidden) in zip(keys, versions, results,
                                                                       scored):
                    result, spare = yours.split_pool(pooled)
                    # Kept for everyone: the plain answer, its spare specimens, the photos'
                    # vectors and what was left out. Nothing about the person asking.
                    answer_cache.put(key, version, {
                        "result": answercache.without_confusions(result),
                        "spare": spare,
                        "vectors": yours.encode_vectors(vecs) if vecs is not None else None,
                        "hidden": [int(c) for c in np.asarray(hidden).tolist()],
                        "used": used, "observation": about})
                    mine = yours.for_person(ident, result, spare, vecs, hidden, linked,
                                            photo_lookup, may_show)
                    confusions.annotate(mine, table)
                    answered.append(mine)
                return {"results": answered, "context_used": used, "observation": about,
                        "cached": False, "timing": timer.block()}

    # The app's observation sessions (docs/api.md §3.3, sessions.py): photos sent one at a
    # time as they're taken; each answer is the combined answer over all of them,
    # re-scored from the stored vectors (never re-embedded, never a photo kept).
    if sessions_on:
        store = session_store.Sessions(
            max_bytes=int((_setting_number("MV_SESSIONS_MAX_MB") or 64) * 2**20))
        app.state.sessions = store

        def session_of(request: Request, sid: str) -> "session_store.Session":
            try:
                return store.get(sid, request.state.credential.rate_key)
            except session_store.Gone:
                raise apiv1.V1Error(404, "session_gone", "this session has ended; open a new "
                                    "one") from None

        def expires_at(s) -> str:
            return (datetime.now(timezone.utc) + timedelta(seconds=store.expires_in(s))
                    ).isoformat(timespec="seconds").replace("+00:00", "Z")

        def entered_context(s, photos) -> tuple["Context", dict]:
            e = s.entered
            return fill_context(Context(e.get("latitude"), e.get("longitude"),
                                        e.get("observed_on")), [p.found for p in photos])

        def rescore(s, cred, timer, image=None, ref=None, found=(None, None, None)) -> dict | None:
            """The session's combined answer, from its stored vectors plus the new photo
            (embedded here, once), through the queue. Keeps the new photo only once it has
            scored. Re-judges stored photos when the gate changed. None with no photos."""
            if image is None and not s.photos:
                return None
            view = permission_view()
            ident = v1_index(s.backbone, s.method, view, timer)
            gate = getattr(ident, "gate", None)
            fingerprint = gate.fingerprint() if gate is not None else None
            added: list = []

            def embed():                                # the worker holds gpu_lock
                if image is not None:
                    vector, base, verdict = embed_one(ident, backbone_model(s.backbone), image)
                    added.append(session_store.Photo(
                        ref=ref, vector=vector, base=base, verdict=verdict,
                        gate_fingerprint=fingerprint if verdict else None, found=found))
                for p in s.photos:                      # judged by an older gate: again
                    if gate is not None and p.gate_fingerprint != fingerprint:
                        base = p.base if p.base is not None else (
                            p.vector if ident.backbone == gate.base else None)
                        if base is not None:
                            p.verdict = gate.judge(np.asarray(base)[None])[0]
                            p.gate_fingerprint = fingerprint
                return np.stack([p.vector for p in [*s.photos, *added]])

            def finish(vecs, sims):
                photos = [*s.photos, *added]
                verdicts = [p.verdict for p in photos]
                context, _ = entered_context(s, photos)
                return ident.identify_vectors(
                    vecs, s.top_k, context=context, sims=sims,
                    gate=verdicts if gate is not None and all(verdicts) else None,
                    photo_lookup=photo_lookup, higher_ranks=True,
                    may_show=lambda info: view.may_show(info.license_class, info.owner_user_id))
            result = model_queue.run(workqueue.Job(
                key=id(ident), photos=1 if image is not None else 0, embed=embed,
                search=ident.nearest.photo_sims, finish=finish,
                interactive=cred.interactive, timer=timer))
            s.photos.extend(added)
            _, used = entered_context(s, s.photos)
            combined = v1_result(result, s.backbone, s.method, used, timer)
            for i, p in enumerate(combined.get("per_photo") or []):
                p["photo_ref"] = s.photos[p.get("photo", i)].ref
            combined.pop("timing", None)
            combined.pop("usage", None)
            s.remember(combined)
            return combined

        def session_answer(s, timer, cred, **extra) -> JSONResponse:
            block, usage = timer.block(), quotas.usage(cred)
            # The combined answer carries this call's timing and usage too, like any v1
            # answer (docs/api.md §4): the app reads combined.timing.
            combined = ({**s.combined, "timing": block, "usage": usage}
                        if s.combined is not None else None)
            body = {**extra, "combined": combined, "photos_scored": len(s.photos),
                    "context_used": entered_context(s, s.photos)[1], "timing": block,
                    "usage": usage,
                    "session": {"session_id": s.id, "expires_at": expires_at(s)}}
            return JSONResponse(body, headers={"Cache-Control": "no-store",
                                               "Server-Timing": apiv1.Timer.header(block)})

        def context_fields(payload: dict) -> dict:
            out = {k: payload[k] for k in ("latitude", "longitude", "observed_on")
                   if k in payload}
            check_context(out.get("latitude"), out.get("longitude"), out.get("observed_on"))
            return out

        @app.post("/api/v1/sessions", status_code=201)
        def v1_open_session(request: Request, payload: dict | None = Body(None)):
            payload = payload or {}
            if not isinstance(payload, dict):
                raise apiv1.V1Error(400, "bad_request", "send a JSON object")
            cred: apiv1.Credential = request.state.credential
            entered = context_fields(payload)
            client_ref = payload.get("client_ref")
            if client_ref is not None and (not isinstance(client_ref, str)
                                           or len(client_ref) > session_store.MAX_REF_LENGTH):
                raise apiv1.V1Error(400, "bad_request", "client_ref is a string of at most "
                                    f"{session_store.MAX_REF_LENGTH} characters")
            top_k = payload.get("top_k", 5)
            if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
                raise apiv1.V1Error(400, "bad_request", "top_k is 1 to 10")
            backbone, method = v1_model(payload.get("model"))
            try:
                s = store.create(cred.rate_key, backbone=backbone, method=method, top_k=top_k,
                                 entered=entered, client_ref=client_ref,
                                 keep_photos=bool(payload.get("keep_photos")))
            except session_store.Full:
                raise apiv1.V1Error(503, "busy", "too many open sessions; try again shortly",
                                    retry_after=60, estimated_wait_seconds=60) from None
            body = {"session_id": s.id, "expires_at": expires_at(s),
                    "ttl_seconds": int(store.idle_ttl), "max_photos": session_store.MAX_PHOTOS,
                    "model": apiv1.public_model(model_card()),
                    "context_used": entered_context(s, [])[1]}
            return JSONResponse(body, status_code=201, headers={"Cache-Control": "no-store"})

        @app.post("/api/v1/sessions/{sid}/photos")
        def v1_session_photo(request: Request, sid: str, photo: UploadFile = File(...),
                             photo_ref: str = Form(...)):
            timer = apiv1.Timer()
            cred: apiv1.Credential = request.state.credential
            s = session_of(request, sid)
            if not 1 <= len(photo_ref) <= session_store.MAX_REF_LENGTH:
                raise apiv1.V1Error(400, "bad_request", "photo_ref is 1 to "
                                    f"{session_store.MAX_REF_LENGTH} characters")
            with s.lock:
                index = s.index_of(photo_ref)
                if index is not None:                   # a resend: the answer as it stands
                    return session_answer(s, timer, cred, photo=photo_entry(s, index))
                if len(s.photos) >= session_store.MAX_PHOTOS:
                    raise apiv1.V1Error(400, "too_many_photos", "a session holds at most "
                                        f"{session_store.MAX_PHOTOS} photos")
                # An identification at the first photo, unless the same find was counted
                # in the last day (a session re-opened after a restart or expiry).
                reopened = not s.counted and quotas.counted_before(cred, s.client_ref)
                if not s.counted and not reopened:
                    quotas.check(cred)
                take_photos(cred, 1)
                try:
                    store.make_room(64 * 1024)
                except session_store.Full:
                    raise apiv1.V1Error(503, "busy", "too many open sessions; try again "
                                        "shortly", retry_after=60,
                                        estimated_wait_seconds=60) from None
                if not in_flight.enter():
                    raise apiv1.V1Error(503, "busy", "busy identifying other photos",
                                        retry_after=10, estimated_wait_seconds=10)
                try:
                    [(image, found)] = v1_photos([photo], timer)
                    rescore(s, cred, timer, image=image, ref=photo_ref, found=found)
                finally:
                    in_flight.leave()
                first = not s.counted and not reopened
                quotas.count(cred, 1 if first else 0, 1)       # once, at its first photo
                if first:
                    quotas.remember_counted(cred, s.client_ref)
                s.counted = True
                return session_answer(s, timer, cred, photo=photo_entry(s, len(s.photos) - 1))

        def photo_entry(s, index: int) -> dict | None:
            per_photo = (s.combined or {}).get("per_photo") or []
            entry = next((p for p in per_photo if p.get("photo") == index), None)
            return {**entry, "index": index, "photo_ref": s.photos[index].ref} if entry else None

        def rescore_now(request: Request, s) -> JSONResponse:
            timer = apiv1.Timer()
            cred: apiv1.Credential = request.state.credential
            if s.photos:
                take_photos(cred, 0)
                rescore(s, cred, timer)
            else:
                s.remember(None)
            return session_answer(s, timer, cred)

        @app.patch("/api/v1/sessions/{sid}")
        def v1_session_context(request: Request, sid: str, payload: dict | None = Body(None)):
            if not isinstance(payload, dict):
                raise apiv1.V1Error(400, "bad_request", "send a JSON object")
            s = session_of(request, sid)
            with s.lock:
                s.entered = {**s.entered, **context_fields(payload)}
                s.entered = {k: v for k, v in s.entered.items() if v is not None}
                check_context(s.entered.get("latitude"), s.entered.get("longitude"),
                              s.entered.get("observed_on"))
                return rescore_now(request, s)

        @app.get("/api/v1/sessions/{sid}")
        def v1_session_state(request: Request, sid: str):
            s = session_of(request, sid)
            return session_answer(s, apiv1.Timer(), request.state.credential)

        @app.delete("/api/v1/sessions/{sid}/photos/{ref}")
        def v1_session_drop_photo(request: Request, sid: str, ref: str):
            s = session_of(request, sid)
            with s.lock:
                index = s.index_of(ref)
                if index is None:
                    raise apiv1.V1Error(404, "not_found", "no photo with that photo_ref in "
                                        "this session")
                del s.photos[index]
                return rescore_now(request, s)

        @app.delete("/api/v1/sessions/{sid}")
        def v1_close_session(request: Request, sid: str):
            try:
                store.delete(sid, request.state.credential.rate_key)
            except session_store.Gone:
                raise apiv1.V1Error(404, "session_gone", "this session has ended") from None
            return Response(status_code=204, headers={"Cache-Control": "no-store"})


    # GET /api/v1/taxon-counts (docs/api.md §7): DNA-validated reference records per name,
    # read from the index the default model serves. Built once per index version.
    if taxon_counts if taxon_counts is not None else config.setting("MV_TAXON_COUNTS") == "1":
        counts_cache: dict = {}

        def served_default() -> tuple[str, str]:
            if warm.models:
                backbone, _, method = warm.models[0].partition("/")
                return backbone, method
            ready = ready_backbones()
            if not ready:
                raise apiv1.V1Error(503, "updating", "no model is ready yet", retry_after=30)
            return ready[0], default_method()

        @app.get("/api/v1/taxon-counts")
        def v1_taxon_counts(request: Request):
            backbone, method = served_default()
            view = permission_view()
            version = index_version(backbone, method, view)
            try:
                ident = identifier(backbone, method, view)
            except (serving.Updating, NotTrainedHere) as e:
                note(f"[v1] {backbone}/{method} not ready: {e}")
                raise apiv1.V1Error(503, "updating", "the model is being updated; try again "
                                    "shortly", retry_after=30) from e
            key = (id(ident), version)   # a failed rebuild may still serve the old index
            if key not in counts_cache:
                doc = apiv1.taxon_counts(
                    ident.rank_counts, records=ident.records,
                    model=apiv1.public_model(model_card()),
                    nightly_version=version[3] if layer else None)
                body = apiv1.encode(doc)
                counts_cache.clear()
                counts_cache[key] = (body, apiv1.etag_of(body))
            body, etag = counts_cache[key]
            headers = {"ETag": etag, "Cache-Control": "private, max-age=3600"}
            if etag in (request.headers.get("if-none-match") or ""):
                return Response(status_code=304, headers=headers)
            return Response(body, media_type="application/json", headers=headers)

    # Any other /api/v1/ path, by any method (a part turned off, a typo): a v1 404, never
    # the site's catch-all below, which would answer a POST with 405.
    @app.api_route("/api/v1/{rest:path}", include_in_schema=False,
                   methods=["GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"])
    def v1_not_found(rest: str):
        raise apiv1.V1Error(404, "not_found", "no such API route (or it isn't turned on here)")

    # The built frontend (web/dist), when present: files as-is, any other
    # non-API path gets index.html so client-side routes work on reload.
    if web_dist and (web_dist / "index.html").exists():
        root = web_dist.resolve()

        @app.get("/{path:path}", include_in_schema=False)
        def site(path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "no such API route")
            target = (root / path).resolve()
            if path and target.is_file() and target.is_relative_to(root):
                return FileResponse(target)
            return FileResponse(root / "index.html")

    return app


def line_buffered_output(streams=None) -> None:
    """Write each line of what the server reports as it happens. Under systemd, stdout is
    a pipe to the journal and Python fills a block before writing it, so a nightly run's
    lines arrived minutes late, or only with the next one."""
    for stream in streams if streams is not None else (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)


def serve(host: str = "127.0.0.1", port: int = 8010) -> None:
    import uvicorn
    line_buffered_output()
    # Behind nginx the client is in X-Forwarded-For; trust it only from nginx itself,
    # so the per-address rate limit sees people, not 127.0.0.1.
    # The nightly update serves the release's layer copy (nightly.py); a new release
    # gets a fresh layer here, and the old release's layer is deleted.
    on = nightly.enabled() and config.RELEASE_ROOT
    layer = nightly.prepare(config.RELEASE_ROOT, prune=True) if on else None
    app = create_app(layer.manifest, layer=layer) if layer else create_app()
    uvicorn.run(app, host=host, port=port, proxy_headers=True,
                forwarded_allow_ips=config.setting("MV_FORWARDED_ALLOW_IPS", "127.0.0.1"))
