"""HTTP API for the frontend (and later mycomap.org).

Run: `.venv/Scripts/mv serve` (defaults to http://127.0.0.1:8010).
Read-only over the manifest; identification runs on the local GPU when there is one.
Public limits (request and image size, rate per IP, GPU queue, allowed backbones)
are in guards.py and set from the environment. Sign-in with a mycomap.org account
(off, identify only, or the whole site) is in signin.py.
"""

from __future__ import annotations

import html
import logging
import secrets
import sqlite3
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import quote

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from PIL import Image

from . import (config, evaluate, inat, manifest, models, names, nightly, permissions, serving,
               uploads)
from .dates import real_date
from .embed import SCHEMA as EMBED_SCHEMA
from .embed import photos_per_second
from .guards import (MAX_FILE_BYTES, MAX_PHOTOS, MAX_REQUEST_BYTES, Gate, Limits, RateLimiter,
                     TooLarge)
from .identify import PHOTO_INFO_SQL, Identifier, NotTrainedHere, photo_info_rows
from .prior import Context
from .signin import (NONCE_COOKIE, NONCE_TTL_SECONDS, SESSION_COOKIE, SigninConfig, SigninError,
                     safe_return_to, verify_token)

log = logging.getLogger("mycomap_vision.api")
NAME_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 30, "6-30"), (31, 10**9, "31+")]


def open_manifest(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    permissions.ensure_schema(conn)
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
               nightly_steps: "nightly.Steps | None" = None) -> FastAPI:
    """`background` starts, when their settings are present, the photographers'-answers
    sync (MV_ORG_BASE_URL + MV_ORG_VISION_KEY, every MV_PERMISSIONS_SYNC_SECONDS,
    default 300) and the licence refresh (MV_LICENSE_REFRESH_HOURS, off by default).
    `note` prints what they report.

    `layer` (serve() passes it when MV_NIGHTLY is on): the nightly update's layer
    (nightly.py). `manifest_path` is then its manifest, the index adds its shards, and
    with `background` the nightly update runs on its schedule. `nightly_fetch` and
    `nightly_steps` replace the reads of mycomap.org and iNat (tests).

    `preload` (default: MV_PRELOAD) names models to load and index on a background
    thread at startup, so the first person to identify doesn't wait for it (about
    30-45 s on the server box). The server answers meanwhile; an identification that
    arrives first waits for the load and then uses it."""
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

    @app.get("/api/models")
    def list_models():
        counts = {b: n for b, n in embedded_counts().items() if b in limits.allowed_backbones}
        with db_lock:
            speed = photos_per_second(conn)
        out = []
        for name, alias in models.ALIASES.items():
            out.append({"backbone": name, "spec": alias.spec, "note": alias.note,
                        "embedded_photos": counts.pop(name, 0),
                        "photos_per_second": speed.get(name)})
        for name, n in counts.items():
            out.append({"backbone": name, "spec": name, "note": "", "embedded_photos": n,
                        "photos_per_second": speed.get(name)})
        return {"backbones": out,
                "methods": [m for m in evaluate.METHODS if limits.method_allowed(m)],
                "max_models": limits.max_models,
                "ready": [b["backbone"] for b in out if b["embedded_photos"]]}

    @app.get("/api/stats")
    def stats():
        one = lambda sql: q(sql)[0][0]  # noqa: E731
        with db_lock:
            labels = names.manifest_labels(conn)
        # Spellings of one name count as one name (names.py).
        name_counts = Counter()
        for name, n in q("select scientific_name, count(*) from records where north_america = 1 "
                         "and label_conflict = 0 and coalesce(scientific_name, '') <> '' "
                         "group by scientific_name"):
            name_counts[labels.get(name, name).strip()] += n
        buckets = {label: 0 for _, _, label in NAME_BUCKETS}
        for n in name_counts.values():
            for lo, hi, label in NAME_BUCKETS:
                if lo <= n <= hi:
                    buckets[label] += 1
        return {
            "records": one("select count(*) from records"),
            "records_north_america": one("select count(*) from records where north_america = 1"),
            "label_conflicts": one("select count(*) from records where label_conflict = 1"),
            "inat_ok": one("select count(*) from inat_observations where status = 'ok'"),
            "inat_missing": one("select count(*) from inat_observations where status = 'missing'"),
            "photos": one("select count(*) from photos"),
            "photos_by_status": dict(q("select status, count(*) from photos group by 1")),
            "photos_by_license": dict(q("select license_class, count(*) from photos group by 1")),
            "photos_by_size": dict(q("select size || case when store like 's3://%' "
                                     "then ' (S3)' else '' end, count(*) from photo_copies "
                                     "group by 1")),
            "contributors_arr": one("select count(distinct owner_login) from photos "
                                    "where license_class = 'arr'"),
            "embedded": embedded_counts(),
            "permissions": permission_status(),
            "names": len(name_counts),
            "names_by_records": buckets,
        }

    @app.get("/api/scoreboard")
    def scoreboard():
        with db_lock:
            return {"runs": evaluate.scoreboard(conn)}

    @app.get("/api/prospective")
    def prospective_report():
        from . import prospective
        with db_lock:
            return {"models": prospective.report(conn)}

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
        try:
            return run_identify(photos, models_, context)
        finally:
            in_flight.leave()

    def run_identify(photos: list[UploadFile], models_: str, context: "Context") -> dict:
        side = uploads.kept_side(list(backbones.values()))
        read = [read_photo(f, side) for f in photos]         # one at a time, kept small
        images = [img for img, _ in read]
        context, used = fill_context(context, [found for _, found in read])
        ready = ready_backbones()
        wanted = [m.strip() for m in models_.split(",") if m.strip()]
        if not wanted:
            if not ready:
                raise HTTPException(503, "no model has embeddings yet")
            wanted = [f"{ready[0]}/nearest"]
        if limits.max_models and len(wanted) > limits.max_models:
            raise HTTPException(400, f"compare at most {limits.max_models} models at a time here")
        view = permission_view()

        def may_show(info) -> bool:
            return view.may_show(info.license_class, info.owner_user_id)

        results = []
        for spec in wanted:
            backbone, _, method = spec.partition("/")
            method = method or "nearest"
            if backbone not in limits.allowed_backbones:
                raise HTTPException(400, f"{backbone!r} is not offered here")
            if backbone not in ready:
                raise HTTPException(400, f"{backbone!r} has no embeddings")
            if method not in evaluate.METHODS:
                raise HTTPException(400, f"unknown method {method!r}")
            if not limits.method_allowed(method):
                raise HTTPException(400, f"{method!r} is not offered here")
            try:
                ident = identifier(backbone, method, view)
            except NotTrainedHere as e:
                raise HTTPException(503, f"{backbone}/{method} isn't ready on this "
                                         f"server ({e})") from e
            except serving.Updating as e:
                raise HTTPException(503, str(e), headers={"Retry-After": "30"}) from e
            with gpu_lock:
                results.append(ident.identify(backbone_model(backbone), images, context=context,
                                              photo_lookup=photo_lookup, may_show=may_show))
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
                    spec = f"{ready[0]}/nearest"
                backbone, _, method = spec.partition("/")
                method = method or "nearest"
                if backbone not in limits.allowed_backbones or backbone not in ready:
                    raise ValueError(f"{backbone!r} is not offered here or has no embeddings")
                if method not in evaluate.METHODS or not limits.method_allowed(method):
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
        fetch = nightly_fetch
        if fetch is None:
            base_url, key = config.setting("MV_ORG_BASE_URL"), permissions.org_key()
            if not (base_url and key):
                raise RuntimeError("MV_ORG_BASE_URL / MV_ORG_VISION_KEY(_FILE) not set")
            fetch = lambda: nightly.fetch_green(base_url, key, log=note)  # noqa: E731

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
        return report

    app.state.run_nightly = run_nightly if layer else None
    if layer and background and nightly_zone is None:
        note("[nightly] NOT scheduled: fix MV_NIGHTLY_TZ (an IANA name, e.g. America/New_York)")
    elif layer and background:
        schedule = nightly.Schedule(layer, nightly_settings, nightly_zone)
        nightly.start(schedule, run_nightly, note)

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


def serve(host: str = "127.0.0.1", port: int = 8010) -> None:
    import uvicorn
    # Behind nginx the client is in X-Forwarded-For; trust it only from nginx itself,
    # so the per-address rate limit sees people, not 127.0.0.1.
    # The nightly update serves the release's layer copy (nightly.py); a new release
    # gets a fresh layer here, and the old release's layer is deleted.
    layer = nightly.prepare(config.RELEASE_ROOT, prune=True)         if nightly.enabled() and config.RELEASE_ROOT else None
    app = create_app(layer.manifest, layer=layer) if layer else create_app()
    uvicorn.run(app, host=host, port=port, proxy_headers=True,
                forwarded_allow_ips=config.setting("MV_FORWARDED_ALLOW_IPS", "127.0.0.1"))
