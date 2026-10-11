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
from datetime import date, timedelta
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.exception_handlers import request_validation_exception_handler
from fastapi.exceptions import RequestValidationError
from starlette.concurrency import run_in_threadpool
from fastapi.responses import (FileResponse, HTMLResponse, JSONResponse, RedirectResponse,
                               Response)
from PIL import Image

from . import (apiv1, config, evaluate, inat, manifest, models, names, nightly, permissions,
               serving, uploads)
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
# The working draft of the Vision paper (Paper in Progress page), served to signed-in
# members only (signin.MEMBERS_ONLY). It ships with the code, never in the public web bundle.
PAPER_FILE = config.REPO_ROOT / "docs" / "paper" / "draft.md"


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
               app_tokens: "apiv1.AppTokens | None | bool" = True,
               introspect: "Callable[[str], dict] | None | bool" = True,
               dataset_root: Path | None = None, dataset_store: str | None = None,
               dataset_store_reader=None, dataset_allow_draft: bool = False) -> FastAPI:
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
    `api_v1` turns on POST /api/v1/identify (default: MV_API_V1). `app_tokens` checks the
    app's tokens (default: the sign-in bridge's public key, for MV_PUBLIC_ORIGIN);
    `introspect` asks mycomap.org about a developer token (default: MV_ORG_BASE_URL with
    the server key; tests pass a fake)."""
    app = FastAPI(title="MycoMap Vision", version="0.1")
    limits = limits or Limits.from_settings()
    signin = signin or SigninConfig.from_settings()
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
    # Before anything is built: a new release's manifest may hold an older list of who
    # withdrew than the one this box last read (permissions.py).
    permissions.restore_last_good(conn, note=note)
    db_lock = threading.Lock()
    gpu_lock = threading.Lock()
    backbones: dict[str, object] = {}

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
                               layer_root=layer.embeddings if layer else None)
        finally:
            own.close()
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
        return (n, cal["run_id"] if cal else None, view.fingerprint(), night)

    warm = Preload()

    @app.get("/api/health")
    def health():
        return {"ok": True, "code_version": config.code_version(),
                "preload": {"state": warm.state, "models": warm.models,
                            "seconds": warm.seconds, "error": warm.error},
                "nightly": nightly_health()}

    def nightly_health() -> dict:
        if not layer:
            return {"enabled": False}
        with db_lock:
            return nightly.health(conn, nightly_settings, nightly_zone)

    def ready_backbones() -> list[str]:
        return [b for b, n in embedded_counts().items() if n and b in limits.allowed_backbones]

    def default_method() -> str:
        """What an identification uses when no method is named (MV_DEFAULT_METHOD)."""
        return limits.served_default(lambda m: m in evaluate.METHODS and method_ready(m))

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
                            if limits.method_allowed(m) and method_ready(m)],
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
            return {"models": prospective.report(conn)}

    @app.get("/api/benchmarks")
    def benchmarks():
        """Held-out benchmark results imported with mv benchmark-import (aggregates only)."""
        from . import benchmark_io
        with db_lock:
            return {"benchmarks": benchmark_io.published(conn)}

    @app.get("/api/experiments")
    def experiments_list():
        """The experiment registry (docs/experiments), front matter only. Members only."""
        from . import experiments
        found = experiments.load_all(experiments_dir or experiments.EXPERIMENTS_DIR)
        return JSONResponse({"experiments": [experiments.summary(e) for e in found]},
                            headers={"Cache-Control": "private, no-store"})

    @app.get("/api/experiments/{slug}")
    def experiment(slug: str):
        from . import experiments
        if not experiments.SLUG.match(slug):
            raise HTTPException(404, "no such experiment")
        found = {e.slug: e for e in experiments.load_all(experiments_dir
                                                         or experiments.EXPERIMENTS_DIR)}
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
            out = run_identify(photos, models_, context, timer)
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

    def embed_and_score(ident: Identifier, backbone: str, images: list, context: "Context",
                        may_show, timer: "apiv1.Timer", top_k: int = 5) -> dict:
        """One model's answer, timed: waiting for the model, embedding, scoring."""
        waited = time.monotonic()
        with gpu_lock:
            timer.add("queue_ms", time.monotonic() - waited)
            model = backbone_model(backbone)
            start = time.monotonic()
            prepare = getattr(model, "prepare", None) or (lambda im: im)
            vecs = normalise(model.encode([prepare(im) for im in images]))
            timer.add("embed_ms", time.monotonic() - start)
            start = time.monotonic()
            result = ident.identify_vectors(vecs, top_k, context=context,
                                            photo_lookup=photo_lookup, may_show=may_show)
            timer.add("score_ms", time.monotonic() - start)
        timer.photos_embedded += len(images)
        return result

    def served_index(backbone: str, method: str, view, timer: "apiv1.Timer") -> Identifier:
        """identifier(), its wait for a rebuild counted as queueing."""
        start = time.monotonic()
        try:
            return identifier(backbone, method, view)
        finally:
            timer.add("queue_ms", time.monotonic() - start)

    def run_identify(photos: list[UploadFile], models_: str, context: "Context",
                     timer: "apiv1.Timer") -> dict:
        read = decode_photos(photos, timer)
        images = [img for img, _ in read]
        context, used = fill_context(context, [found for _, found in read])
        ready = ready_backbones()
        wanted = [m.strip() for m in models_.split(",") if m.strip()]
        if not wanted:
            if not ready:
                raise HTTPException(503, "no model has embeddings yet")
            wanted = [f"{ready[0]}/{default_method()}"]
        if limits.max_models and len(wanted) > limits.max_models:
            raise HTTPException(400, f"compare at most {limits.max_models} models at a time here")
        view = permission_view()

        def may_show(info) -> bool:
            return view.may_show(info.license_class, info.owner_user_id)

        results = []
        for spec in wanted:
            backbone, _, method = spec.partition("/")
            method = method or default_method()
            if backbone not in limits.allowed_backbones:
                raise HTTPException(400, f"{backbone!r} is not offered here")
            if backbone not in ready:
                raise HTTPException(400, f"{backbone!r} has no embeddings")
            if method not in evaluate.METHODS:
                raise HTTPException(400, f"unknown method {method!r}")
            if not limits.method_allowed(method) or not method_ready(method):
                raise HTTPException(400, f"{method!r} is not offered here")
            try:
                ident = served_index(backbone, method, view, timer)
            except NotTrainedHere as e:
                raise HTTPException(503, f"{backbone}/{method} isn't ready on this "
                                         f"server ({e})") from e
            except serving.Updating as e:
                raise HTTPException(503, str(e), headers={"Retry-After": "30"}) from e
            results.append(embed_and_score(ident, backbone, images, context, may_show, timer))
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
                        or not method_ready(method)):
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

    # The nightly update (nightly.py): runs here because this process holds the model.
    nightly_settings = nightly_settings or (nightly.Settings.from_env() if layer else None)
    nightly_zone = None
    if layer:
        try:
            nightly_zone = nightly_settings.zone()
        except Exception as e:  # noqa: BLE001 - reported; the schedule below would fail too
            note(f"[nightly] time zone {nightly_settings.tz!r}: {type(e).__name__}: {e}")

    def run_nightly(accept_removals: bool = False) -> dict:
        """One nightly update, then the preloaded indexes rebuilt so the first person
        after it doesn't wait for the new reference set."""
        def org() -> tuple[str, str]:
            base_url, key = config.setting("MV_ORG_BASE_URL"), permissions.org_key()
            if not (base_url and key):
                raise RuntimeError("MV_ORG_BASE_URL / MV_ORG_VISION_KEY(_FILE) not set")
            return base_url, key
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

    # POST /api/v1/identify (docs/api.md §3.1, §4): one model's answer for 1-10 photos,
    # with timing, confident/answer, and the caller's own short-window limit.
    if api_v1 if api_v1 is not None else config.setting("MV_API_V1") == "1":
        photo_window = apiv1.PhotoWindow(int(_setting_number("MV_PHOTOS_PER_MINUTE") or 20))

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
                    or not method_ready(method)):
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

        @app.post("/api/v1/identify")
        def v1_identify(request: Request, photos: list[UploadFile] = File(...),
                        latitude: float | None = Form(None), longitude: float | None = Form(None),
                        observed_on: str | None = Form(None), model: str | None = Form(None),
                        top_k: int = Form(5)):
            timer = apiv1.Timer()
            cred: apiv1.Credential = request.state.credential
            if not 1 <= len(photos) <= MAX_PHOTOS:
                raise apiv1.V1Error(400, "too_many_photos", f"send 1 to {MAX_PHOTOS} photos")
            if sum(f.size or 0 for f in photos) > MAX_REQUEST_BYTES:
                raise apiv1.V1Error(413, "too_large", "the photos are over "
                                    f"{MAX_REQUEST_BYTES // 2**20} MB together")
            if not 1 <= top_k <= 10:
                raise apiv1.V1Error(400, "bad_request", "top_k is 1 to 10")
            if (latitude is None) != (longitude is None) or (
                    latitude is not None and not (-90 <= latitude <= 90
                                                  and -180 <= longitude <= 180)):
                raise apiv1.V1Error(400, "bad_request", "send latitude and longitude together, "
                                    "in range")
            if observed_on:
                try:
                    date.fromisoformat(observed_on)
                except ValueError:
                    raise apiv1.V1Error(400, "bad_request", "observed_on is YYYY-MM-DD") from None
            backbone, method = v1_model(model)
            if cred.kind != "server":
                wait = photo_window.take(cred.rate_key, len(photos))
                if wait:
                    raise apiv1.V1Error(429, "rate_limited", "too many photos in the last "
                                        "minute", retry_after=int(wait) + 1)
            if not in_flight.enter():
                raise apiv1.V1Error(503, "busy", "busy identifying other photos",
                                    retry_after=10, estimated_wait_seconds=10)
            try:
                read = v1_photos(photos, timer)
                context, used = fill_context(Context(latitude, longitude, observed_on),
                                             [found for _, found in read])
                view = permission_view()
                try:
                    ident = served_index(backbone, method, view, timer)
                except (serving.Updating, NotTrainedHere) as e:
                    raise apiv1.V1Error(503, "updating", str(e), retry_after=30) from e
                result = embed_and_score(
                    ident, backbone, [img for img, _ in read], context,
                    lambda info: view.may_show(info.license_class, info.owner_user_id),
                    timer, top_k)
            finally:
                in_flight.leave()
            body = v1_result(result, backbone, method, used, timer)
            return JSONResponse(body, headers={"Cache-Control": "no-store",
                                               "Server-Timing": apiv1.Timer.header(body["timing"])})

        def v1_result(result: dict, backbone: str, method: str, used: dict,
                      timer: "apiv1.Timer") -> dict:
            """An identify result in the v1 shape (docs/api.md §4)."""
            cal = result.pop("calibration", None) or {}
            comparison = str(cal.get("comparison_id") or "")
            out = {"request_id": "r_" + secrets.token_urlsafe(12),
                   **result,
                   "model": {"backbone": backbone, "method": method,
                             "version": layer.release_id if layer else None,
                             "calibrated_on": (f"{comparison[:4]}-{comparison[4:6]}-"
                                               f"{comparison[6:8]}"
                                               if comparison[:8].isdigit() else None)},
                   "context_used": used,
                   "gate": result.get("gate"),
                   "usage": None}
            apiv1.mark_confident(out)
            out["timing"] = timer.block()
            return out

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
                raise apiv1.V1Error(503, "updating", str(e), retry_after=30) from e
            key = (id(ident), version)   # a failed rebuild may still serve the old index
            if key not in counts_cache:
                doc = apiv1.taxon_counts(
                    ident.rank_counts, records=ident.records,
                    model={"backbone": backbone, "method": method,
                           "version": layer.release_id if layer else None},
                    nightly_version=version[3] if layer else None)
                body = apiv1.encode(doc)
                counts_cache.clear()
                counts_cache[key] = (body, apiv1.etag_of(body))
            body, etag = counts_cache[key]
            headers = {"ETag": etag, "Cache-Control": "private, max-age=3600"}
            if etag in (request.headers.get("if-none-match") or ""):
                return Response(status_code=304, headers=headers)
            return Response(body, media_type="application/json", headers=headers)

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
