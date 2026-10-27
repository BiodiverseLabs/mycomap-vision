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
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, OrderedDict
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

from . import (answercache, apidocs, early_answers, apiv1, byobservation, config, confusions, evaluate, fastsearch, feedback, fungusgate, geocode, identifylog, inat, inat_suggest, linkcards, manifest, phototypes,
               feedback_org as feedback_org_rules, modelcard, models, name_changes, names, nightly,
               excluded_looked as xl, looked, name_context, onnxtower, pair_places, permissions, dhash, runs as run_links, twins, uploadmeter,
               quotas as quota_rules, research_public, serving, sessions as session_store, sources,
               speedlog, uploads,
               workqueue, yours)
from .dates import real_date
from .embed import SCHEMA as EMBED_SCHEMA
from .embed import normalise, photos_per_second
from .guards import (MAX_FILE_BYTES, MAX_PHOTOS, MAX_REQUEST_BYTES, Gate, Limits, RateLimiter,
                     TooLarge)
from .identify import PHOTO_INFO_SQL, Identifier, NotTrainedHere, photo_info_rows
from .identify import left_out_photos as left_out_of
from .licenses import photo_url
from .methods import method_ready
from .prior import Context
from .signin import (NONCE_COOKIE, NONCE_TTL_SECONDS, SESSION_COOKIE, SigninConfig, SigninError,
                     load_public_key, origin_of, run_path, safe_return_to, verify_token)

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


def read_photo(upload: UploadFile, side: int = uploads.kept_side([]),
               hashes: list | None = None,
               bodies: list | None = None) -> tuple[Image.Image, tuple]:
    """One uploaded photo, refusing oversized files and images before decoding them,
    decoded once and kept only at the size the models need (uploads.py: shorter side
    `side`). Also returns its EXIF (latitude, longitude, date), read before shrinking,
    used only for scoring. With `hashes`, appends the file's dHash (None when it can't be
    hashed) for the upload meter (uploadmeter.py); the hash is never kept. With `bodies`,
    appends the file, for the run's link (runs.py keeps a small copy without its EXIF)."""
    body = upload.file.read(MAX_FILE_BYTES + 1)
    if len(body) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{upload.filename} is over {MAX_FILE_BYTES // 2**20} MB")
    if bodies is not None:
        bodies.append(body)
    try:
        decoded = uploads.decode_upload(body, side)
    except TooLarge as e:
        raise HTTPException(413, f"{upload.filename}: {e}")
    except Exception as e:
        # What it was and why, so a refusal can be traced (the bytes and name stay out).
        print(f"[upload] unreadable: {upload.content_type or '?'}, {len(body)} bytes, "
              f"starts {body[:12].hex()}: {type(e).__name__}: {str(e)[:160]}", flush=True)
        raise HTTPException(400, f"{upload.filename} is not an image we can read")
    if hashes is not None:                  # after the size check: never hash a bomb
        try:
            hashes.append(dhash.dhash64(body))
        except Exception:
            hashes.append(None)
    return decoded


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


def first_confused_with(served: "confusions.Served", layer: "nightly.Layer | None",
                        steps: "nightly.Steps | None" = None, note=print) -> None:
    """The "often confused with" table at startup. On the box (a nightly layer) with no
    table yet (nightly.prepare carries the last release's over; none on a first layer, or
    when that one had none), mycomap.org's is read now instead of at the next night;
    a failed read leaves the notes off until then, as before."""
    if layer is not None and not served.cache.is_file():
        note("[confused-with] no table in this release's layer: reading mycomap.org's now")
        try:
            (steps or nightly.Steps()).misidentifications(served.cache, log=note)
        except Exception as e:  # noqa: BLE001 - the night tries again
            note(f"[confused-with] startup read failed: {type(e).__name__}: {e}")
    served.reload()


# Sent with the Identify page's shell when it has an example result: the browser fetches
# /api/example while the page's scripts load (the page's own fetch then takes that answer),
# and opens its connection to iNat's open photo bucket, where the example's photos are.
EXAMPLE_PRELOAD = ("</api/example>; rel=preload; as=fetch; crossorigin=anonymous, "
                   "<https://inaturalist-open-data.s3.amazonaws.com>; rel=preconnect")


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
               card_writer: Callable[[Callable[[], None]], object] | None = None,
               quotas: "quota_rules.Quotas | None" = None,
               upload_meter: "uploadmeter.UploadMeter | None" = None,
               usage_post: Callable[[dict], None] | None = None,
               app_tokens: "apiv1.AppTokens | None | bool" = True,
               introspect: "Callable[[str], dict] | None | bool" = True,
               dataset_root: Path | None = None, dataset_store: str | None = None,
               dataset_store_reader=None, dataset_allow_draft: bool = False,
               fungus_gate: "fungusgate.FungusGate | None | bool" = True,
               photo_tagger: "phototypes.PhotoTagger | None | bool" = True,
               place_search: "geocode.PlaceSearch | None" = None,
               place_suggest: "geocode.PlaceSuggest | None" = None,
               confused_with: "confusions.Served | None | bool" = True,
               feedback_store: "feedback.Store | None" = None,
               feedback_org: "feedback_api.OrgClient | None" = None,
               feedback_places: "pair_places.PairPlaces | None" = None,
               feedback_inat_names: "name_context.InatNames | None" = None,
               answer_cache: "answercache.AnswerCache | None" = None,
               linked_accounts: "yours.LinkedCache | None | bool" = True,
               run_store: "run_links.RunStore | None" = None,
               run_secret: bytes | None = None,
               identify_log: "identifylog.IdentifyLog | None" = None,
               early_store: "early_answers.Store | None" = None,
               inat_suggestions: "inat_suggest.Suggestions | None | bool" = True,
               where_looked: "Callable | None | bool" = True,
               excluded_looked: "Callable | None | bool" = True,
               excluded_unsealed: bool | None = None) -> FastAPI:
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

    `photo_tagger`: the photo-type tagger (phototypes.py) on the gate's frozen vectors:
    micrograph and label photos tagged in the high band are left out of the answer
    (habitat is tagged only). True loads the file MV_PHOTO_TAGGER names (unset: no tagger, nothing tagged; a
    file that fails its checks is refused and reported); None or False: no tagger.

    `linked_accounts`: a signed-in person's verified iNat and MO accounts from mycomap.org,
    for "Your collections" on the Identify result (yours.py). True reads them with
    MV_ORG_BASE_URL and the server key (neither set: no panel); tests pass a LinkedCache.

    `run_store`: where the site's runs are kept, each under its own link /r/<id> (runs.py).
    None reads MV_RUN_LINKS: `s3` (the Vision bucket's site-runs/), a folder, or off (the
    default: no links). `run_secret` keys an iNat/MO run's id (default: the session secret).

    `inat_suggestions`: iNaturalist's suggestions under Vision's answers, asked through
    mycomap.org with the signed-in member's own iNat account (inat_suggest.py). True turns
    them on when MV_INAT_SUGGESTIONS is set (off by default); None or False: off; tests pass
    a Suggestions.

    `identify_log`: the append-only log of every answer given or served (identifylog.py),
    for the paper's truth-independence check. None opens state/identify-log.sqlite on the
    box, the file beside the manifest elsewhere.

    `where_looked`: "Where Vision looked" maps on a run's answer (looked.py), made when
    someone asks and kept beside the run; needs run links. True: on unless
    MV_WHERE_LOOKED=0; None or False: off; tests pass the photo maker (called as
    maker(loaded backbone, backbone name, identifier, name, photo bytes, specimen bytes or
    None) -> the photo's entry).

    `excluded_looked`: Research > Excluded photos (excluded_looked.py), the photos people left
    out in Human Feedback with "where Vision looked" on any of them, on the run maps' worker.
    True: on unless MV_EXCLUDED_LOOKED=0; None or False: off; tests pass the photo maker
    (called as maker(loaded backbone, backbone name, identifier, name, photo bytes, the
    record's own columns to leave out) -> {crop, grid, map, rank}); v1's maps only (Steve,
    2026-10-26: no base model). `excluded_unsealed`: the paper's seal lifted
    (default MV_EXCLUDED_LOOKED_UNSEALED=1; off until the v1-test scoring is committed).

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
    tagger_error: list[str] = []
    if photo_tagger is True:
        try:
            photo_tagger = phototypes.PhotoTagger.from_setting()
        except (phototypes.Refused, OSError, ValueError, KeyError) as e:
            tagger_error.append(f"{type(e).__name__}: {e}")
            note(f"[photo-tagger] not served: {tagger_error[0]}")
            photo_tagger = None
    photo_tagger = photo_tagger or None
    if photo_tagger is not None and fungus_gate is not None and photo_tagger.base != fungus_gate.base:
        # It must read the vector the gate already computes, never a second tower's.
        tagger_error.append(phototypes.base_mismatch(photo_tagger.base, fungus_gate.base))
        note(f"[photo-tagger] not served: {tagger_error[-1]}")
        photo_tagger = None
    if photo_tagger is not None:
        note(f"[photo-tagger] {photo_tagger.version} ({photo_tagger.fingerprint()}) tags every "
             "identified photo")

    def type_options(include=()) -> dict:
        """What a kept observation answer is keyed on besides model, place and date: the
        photos counted anyway, so an answer kept without a photo is never served for a call
        that includes it. The tagger itself is not in the key: it is part of the answers
        version (with_tagger), so an answer kept before it, or under an older one, is still
        served, marked not current, until someone runs it again (Steve, 2026-10-22). Nothing
        without a tagger or an include list (keys as before)."""
        if photo_tagger is None or not include:
            return {}
        return {"include": ".".join(str(i) for i in sorted(include))}

    def with_tagger(index_version: tuple) -> tuple:
        """An index version with the photo-type tagger in it, when one is served: a new
        tagger, like a new release, makes every kept answer not current."""
        if photo_tagger is None:
            return index_version
        return (*index_version, f"photo-tagger:{photo_tagger.fingerprint()}")

    def tagger_mark(payload: dict) -> dict:
        """A kept answer with the version of the tagger that made it (absent without one)."""
        if photo_tagger is not None:
            payload["photo_tagger"] = photo_tagger.version
        return payload

    def asked_include(value, photos: int | None = None) -> list[int]:
        """The photos to count anyway (phototypes.parse_include) when a tagger is served; none
        otherwise, so without one the request is answered exactly as before."""
        if photo_tagger is None:
            return []
        return phototypes.parse_include(value, photos)
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
                "user": {"id": p["sub"], "name": p.get("name")} if p else None,
                "features": {"where_looked": looked_on}}

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
        threading.Thread(target=first_confused_with, args=(confused_with, layer, nightly_steps,
                                                            note),
                         name="confused-with", daemon=True).start()

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
    # Seconds per photo, one row per day (speedlog.py), beside the other state files.
    speed_log = speedlog.SpeedLog(
        (config.RELEASE_ROOT / "state" if config.RELEASE_ROOT
         else Path(manifest_path or config.MANIFEST_PATH).parent) / speedlog.FILE, note=note)
    app.state.speed_log = speed_log
    model_queue = workqueue.ModelQueue(
        gpu_lock, max_wait=_setting_number("MV_QUEUE_MAX_WAIT") or 20.0,
        max_batch_photos=int(_setting_number("MV_QUEUE_BATCH_PHOTOS") or 20), note=note,
        on_batch=speed_log.record)
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
    # How often an uploaded photo is a reference photo (uploadmeter.py; Steve, 2026-10-19):
    # counters only, in the usage store, shown to mycomap.org's server key in /api/health.
    if upload_meter is None:
        # The box's usage store; on a laptop (and in tests) beside the manifest.
        upload_meter = uploadmeter.UploadMeter(
            quotas.path if config.RELEASE_ROOT
            else Path(manifest_path or config.MANIFEST_PATH).parent / quotas.path.name,
            lambda: twins_path(), note=note)
    app.state.upload_meter = upload_meter
    # Answers already given for an iNat or MO observation (answercache.py), served again
    # while what Vision answers from is unchanged (answers_version below).
    if answer_cache is None:
        answer_cache = answercache.AnswerCache(
            (config.RELEASE_ROOT / "state" if config.RELEASE_ROOT
             else Path(manifest_path or config.MANIFEST_PATH).parent)
            / "observation-answers.sqlite")
    app.state.answer_cache = answer_cache
    # Every site run under its own link (runs.py): the bucket's site-runs/ on the box, a
    # folder on a laptop, none unless MV_RUN_LINKS says so. Its owner index sits beside the
    # observation answers.
    if run_store is None:
        run_store = run_links.from_setting(
            config.setting("MV_RUN_LINKS"),
            (config.RELEASE_ROOT / "state" if config.RELEASE_ROOT
             else Path(manifest_path or config.MANIFEST_PATH).parent), note=note)
    app.state.run_store = run_store
    # Where Vision looked (looked.py): needs run links; MV_WHERE_LOOKED=0 turns it off.
    looked_on = bool(where_looked) and run_store is not None and (
        callable(where_looked) or config.setting("MV_WHERE_LOOKED") != "0")
    # Research > Excluded photos (excluded_looked.py): MV_EXCLUDED_LOOKED=0 turns it off.
    excluded_on = bool(excluded_looked) and (
        callable(excluded_looked) or config.setting("MV_EXCLUDED_LOOKED") != "0")
    run_secret = run_secret or signin.secret or secrets.token_bytes(32)
    # iNat's suggestions (inat_suggest.py; Steve, 2026-10-25): off until iNat agrees in
    # writing. Never in the identify log or early answers: mycomap.org logs iNat's side.
    if inat_suggestions is True:
        inat_suggestions = inat_suggest.Suggestions.from_settings(note=note)
    inat_suggestions = inat_suggestions or None
    app.state.inat_suggestions = inat_suggestions
    # Every answer given or served, one row each, insert-only (identifylog.py; Steve,
    # 2026-10-20), beside the observation answers. Written off the request path.
    if identify_log is None:
        identify_log = identifylog.IdentifyLog(identifylog.default_path(manifest_path),
                                               note=note)
    app.state.identify_log = identify_log
    # Answers given an observation before its DNA verdict (early_answers.py; Steve,
    # 2026-10-23): every fresh by-observation answer, frozen ten deep, insert-only, scored
    # nightly once the record turns green. Beside the identification log.
    if early_store is None:
        early_store = early_answers.Store(early_answers.default_path(manifest_path), note=note)
    app.state.early_store = early_store

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

    def twins_path() -> Path | None:
        """The cross-post twin table (twins.py): MV_TWINS, else the served release's, else
        the one beside the manifest; None when there is none (links then leave out only
        their own record, the records they link to and copies of their photos)."""
        named = config.setting("MV_TWINS")
        if named:
            return Path(named)
        for folder in (served_release_dir(), Path(manifest_path).parent):
            if folder is not None and (Path(folder) / twins.FILE).is_file():
                return Path(folder) / twins.FILE
        return None

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
                               gate=fungus_gate, twins_file=twins_path(),
                               specimen_groups=served_groups(), tagger=photo_tagger)
        finally:
            own.close()
        # The fast int8 search (fastsearch.py, MV_SEARCH_INT8=1): packed here, on the build
        # thread, after taking the one packed slot from the index serving now. Only for a
        # backbone offered for identification.
        if config.setting("MV_SEARCH_INT8") == "1" and backbone in ready_backbones():
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

    def same_permissions(old: tuple, new: tuple) -> bool:
        """Two index versions that differ in nothing about who may be shown (the third
        part of index_version) or which cross-posts are one find (the last): the old index
        may answer while the new one builds."""
        return (isinstance(old, tuple) and isinstance(new, tuple) and len(old) == len(new)
                and old[2] == new[2] and old[-1] == new[-1])

    # Rebuilt (someone withdrew, a new calibration) on its own thread and swapped in
    # whole; the old index is released after, and serves on if a build fails.
    indexes = serving.ServedIndexes(build_index, note=note, release=models.release_memory,
                                    wait=_setting_number("MV_INDEX_WAIT_SECONDS") or 20.0,
                                    may_serve_old=same_permissions)

    def identifier(backbone: str, method: str, view: permissions.PermissionView) -> Identifier:
        """The index as the data stand now. While a rebuild that changes no one's
        permission runs (a rename, the night's records, a calibration), the old index
        answers; after a withdrawal it waits for the rebuild (serving.Updating if that
        takes too long). Never call it holding gpu_lock."""
        return indexes.get((backbone, method), index_version(backbone, method, view),
                           stale_ok=True)

    def rebuild_behind(why: str) -> None:
        """Start rebuilding every served index for the data as they stand now, the old one
        answering meanwhile, so the next person doesn't wait for it."""
        for backbone, method in indexes.served():
            try:
                indexes.get((backbone, method),
                            index_version(backbone, method, permission_view()), wait=0,
                            stale_ok=True)
            except serving.Updating:
                pass                                  # a withdrawal: building, callers wait
            except Exception as e:  # noqa: BLE001 - the next identification retries
                note(f"[index] rebuilding {backbone}/{method} after {why} failed: "
                     f"{type(e).__name__}: {e}")

    def index_version(backbone: str, method: str, view: permissions.PermissionView) -> tuple:
        n = embedded_counts().get(backbone, 0)
        with db_lock:
            cal = evaluate.latest_calibration(conn, backbone, method)
            # A night that only removed or renamed records leaves n alone.
            night = nightly.layer_version(conn) if layer else 0
        # A change in who has said no rebuilds the reference set without their photos.
            # Photos Human Feedback leaves out, and the specimen crops it keeps of them
            # (synced at startup and before each night).
            excluded = (feedback.exclusions_fingerprint(conn) + "+"
                        + feedback.crops_fingerprint(conn))
            # Renames read from mycomap.org's name-change log between nights (name_changes.py).
            renamed = name_changes.version(conn) if layer else 0
        # view.fingerprint() stays third and the twins last: same_permissions reads them.
        return (n, cal["run_id"] if cal else None, view.fingerprint(), night, excluded, renamed,
                twins.version(twins_path()))

    def answers_version(backbone: str, method: str, view: permissions.PermissionView) -> str:
        """What this model answers from now (answercache.answers_version): the release and
        the served index's version (the old one's while a rebuild runs behind it). A kept
        observation answer is served only under it."""
        return answercache.answers_version(model_card().get("version"), with_tagger(
            indexes.answering((backbone, method), index_version(backbone, method, view))))

    # --- the identification log (identifylog.py) -----------------------------------------
    def release_now() -> str | None:
        return model_card().get("version")

    def served_version(backbone: str, method: str, ident) -> "Callable[[], str] | None":
        """The answers version of the index that just answered: its version taken now (the
        index may be swapped a moment later), hashed with the release on the log's writer
        thread. None when that index is no longer the one served."""
        try:
            cur = indexes.current((backbone, method))
        except Exception:  # noqa: BLE001 - logged without it
            return None
        if cur is None or cur[1] is not ident:
            return None
        return lambda: answercache.answers_version(release_now(), with_tagger(cur[0]))

    def freeze_answer(**fields) -> None:
        """Keeps a fresh by-observation answer for the benchmark of answers given before the
        DNA verdict (early_answers.Store.record). Never raises, never waits."""
        try:
            early_store.record(code_version=config.code_version(), **fields)
        except Exception as e:  # noqa: BLE001 - the answer goes out regardless
            note(f"[early-answers] {type(e).__name__}: {e}")

    def log_answer(path: str, **fields) -> None:
        """One row of the identification log. Never raises and never waits: the row is
        queued and written on the log's own thread."""
        try:
            identify_log.record(path, **fields)
        except Exception as e:  # noqa: BLE001 - the answer goes out regardless
            note(f"[identify-log] {path}: {type(e).__name__}: {e}")

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
                "photo_tagger": (photo_tagger.summary() if photo_tagger else
                                 {"error": tagger_error[0]} if tagger_error else None),
                "vectors": vector_storage()}
        given = request.headers.get("authorization", "")
        if org_secret and secrets.compare_digest(given.encode(),
                                                 f"Bearer {org_secret}".encode()):
            # Never in the public answer: how often uploads are reference photos, the
            # identification log's counters (rows written, waiting, dropped, failed), and how
            # many runs are kept under their own links and the room their photos take.
            return {**full, "upload_matches": upload_meter.summary(),
                    "identify_log": identify_log.state(),
                    "run_links": run_store.summary() if run_store is not None else None}
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

    # Suggestions as the person types (geocode.py): Photon, as Nominatim's policy forbids
    # autocomplete. Always a list; X-Suggest-Status says why one is empty.
    suggestions = place_suggest or geocode.PlaceSuggest()

    @app.get("/api/geocode/suggest")
    def geocode_suggest(request: Request, q: str = ""):
        places, status = suggestions.suggest(q, request.client.host if request.client else "unknown")
        return JSONResponse(places, headers={"X-Suggest-Status": status})

    @app.get("/api/speed/daily")
    def speed_daily(days: int = 365):
        """Average seconds per photo, day by day (UTC): the How it works chart. Averages
        only, never how many photos."""
        return {"days": speed_log.daily(min(max(days, 1), 3650))}

    # The Research section's data, open to anyone (signin.research_path): what a signed-out
    # caller may read is cut in research_public.py.
    @app.get("/api/stats")
    def stats():
        # Photographers' answers are cheap and change any minute: always the current ones.
        # Never their counts (research_public.permission_status).
        return {**research_public.stats(stats_cache.get()),
                "permissions": research_public.permission_status(permission_status())}

    @app.get("/api/scoreboard")
    def scoreboard(request: Request):
        with db_lock:
            runs = evaluate.scoreboard(conn)
        return {"runs": research_public.scoreboard(runs, person(request) is not None)}

    @app.get("/api/prospective")
    def prospective_report(request: Request):
        from . import prospective
        with db_lock:
            models_ = prospective.report(conn)
        return {"models": research_public.prospective(models_, person(request) is not None),
                "sealed_after": prospective.seal_time()}

    @app.get("/api/early-answers")
    def early_answers_report(request: Request):
        """Answered before the DNA verdict (early_answers.py): aggregates per model and
        context. Always cut at the paper's fixed seal (research_public.SEAL), whatever
        MV_PROSPECTIVE_SEAL says: no sealed-test score on any page before submission."""
        with db_lock:
            groups = early_answers.report(early_store.conn, conn,
                                          sealed_after=research_public.SEAL,
                                          lock=early_store.lock)
        return {"groups": research_public.early_answers(groups, person(request) is not None),
                "sealed_after": research_public.SEAL}

    @app.get("/api/how-good")
    def how_good_report():
        """How it works' "How good is it" (howgood.py): the served model's newest open run,
        top 1/3/5/10, species by reference depth, first photo only. Never the sealed test."""
        from . import howgood
        served, method = ready_backbones(), default_method()
        with db_lock:
            return howgood.how_good(conn, served, method)

    def example_set() -> bool:
        return run_store is not None and run_links.valid_id(config.setting("MV_EXAMPLE_RUN"))

    @app.get("/api/example")
    def example_run():
        """The run the Identify page shows as its "Example result" (Steve, 2026-10-26: "a
        real result with real images"): the id in MV_EXAMPLE_RUN, read on each request, so
        changing it needs only a restart. Null when it is unset or not a run id, or the site
        keeps no run links.
        `view` is the run as its link shows anyone signed out (run_body for nobody), its
        photos checked against today's permissions, so the page needs no second request
        (Steve, 2026-10-26: load it with the page). Null when the run can't be read now; the
        page then opens /api/runs/<id> itself."""
        rid = config.setting("MV_EXAMPLE_RUN")
        if not example_set():
            return JSONResponse({"run": None, "view": None}, headers=NOT_INDEXED)
        try:
            stored = run_store.get(rid)
            if stored is None:                  # deleted since: no example
                return JSONResponse({"run": None, "view": None}, headers=NOT_INDEXED)
            view = run_body(stored, None, None)
        except Exception as e:  # noqa: BLE001 - reported; the page falls back to the run route
            note(f"[example] {rid} not read now: {type(e).__name__}: {e}")
            view = None
        return JSONResponse({"run": rid, "view": view}, headers=NOT_INDEXED)

    @app.get("/api/benchmarks")
    def benchmarks(request: Request):
        """Held-out benchmark results imported with mv benchmark-import (aggregates only)."""
        from . import benchmark_io
        with db_lock:
            found = benchmark_io.published(conn)
        return {"benchmarks": research_public.benchmarks(found, person(request) is not None)}

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
    def scoreboard_run(run_id: int, request: Request):
        with db_lock:
            report = evaluate.run_report(conn, run_id)
            run = evaluate.run_row(conn, run_id)
        signed_in = person(request) is not None
        if report is None or (not signed_in and research_public.sealed_run(run or {})):
            raise HTTPException(404, "no such run")
        # The reference's species per depth band, for the heatmap's companion (aggregates only).
        species = evaluate.reference_species_of(report, (run or {}).get("record_set"))
        return research_public.run_report({**report, "reference_species": species}
                                          if species else report, signed_in)

    # A plain (not async) route: FastAPI runs it in a worker thread, so GPU work
    # never blocks the event loop and other requests keep being answered.
    @app.post("/api/identify")
    def identify(request: Request, photos: list[UploadFile] = File(...),
                 models_: str = Form("", alias="models"), lat: float | None = Form(None),
                 lng: float | None = Form(None), observed_on: str | None = Form(None),
                 include: str = Form(""), inat_ref: str | None = Form(None)):
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
        try:
            counted_anyway = asked_include(include, len(photos))
        except ValueError as e:
            in_flight.leave()
            raise HTTPException(400, str(e)) from None
        context = Context(lat, lng, observed_on)
        timer = apiv1.Timer()
        try:
            try:
                model_queue.admit(interactive=True)
            except workqueue.Busy as e:
                raise HTTPException(503, "busy identifying other photos; try again shortly",
                                    headers={"Retry-After": str(int(e.wait_seconds) + 1)}) from e
            out = run_identify(photos, models_, context, timer, linked_of(request),
                               person(request), counted_anyway, inat_ref)
        finally:
            in_flight.leave()
        out["timing"] = timer.block()
        return out

    def decode_photos(photos: list[UploadFile], timer: "apiv1.Timer",
                      hashes: list | None = None, bodies: list | None = None) -> list[tuple]:
        start = time.monotonic()
        side = uploads.kept_side(list(backbones.values()))
        read = [read_photo(f, side, hashes, bodies) for f in photos]  # one at a time
        timer.add("decode_ms", time.monotonic() - start)
        return read

    def embed_one(ident: Identifier, model, image) -> tuple:
        """One photo -> (its unit vector, float32; the gate's base vector or None when it is
        the vector itself or the gate can't run; the gate's verdict or None; its photo-type
        tag (phototypes.py) or None). The tagger reads the same base vector as the gate, so
        the tower runs once for both. Every photo
        is embedded on its own, by /identify and by sessions alike, so the same photo
        always gets the same vector: a forward pass over a batch differs from one over a
        single photo by ~1e-4, enough to swap near-tied names (integration testing,
        2026-10-12). Call with gpu_lock held."""
        prepare = getattr(model, "prepare", None) or (lambda im: im)
        vecs, base = ident.encode(model, [prepare(image)])
        vector = normalise(vecs)[0].astype(np.float32)
        gate = getattr(ident, "gate", None)
        verdict = gate.judge(base)[0] if gate is not None and base is not None else None
        tagger = getattr(ident, "tagger", None)
        tag = tagger.tag(base)[0] if tagger is not None and base is not None else None
        return (vector, (None if base is None or base is vecs else np.asarray(base[0])), verdict,
                tag)

    def embed_and_score(ident: Identifier, backbone: str, images: list, context: "Context",
                        may_show, timer: "apiv1.Timer", top_k: int = 5,
                        interactive: bool = True, higher: bool = False,
                        hidden: np.ndarray | None = None, n_specimens: int = 6,
                        vectors: list | None = None, pooled: list | None = None,
                        include=(), deep: list | None = None) -> dict:
        """One model's answer through the queue, timed: waiting, embedding, scoring.
        `hidden`: reference columns left out (Identifier.identify_vectors). `vectors`, when
        given, gets the photos' vectors appended (for yours.for_person). `pooled`, when
        given, gets the site's answer from the same photos appended (site_answer), so one
        run answers the API and the Identify page alike. `include`: photos to count even
        though their photo-type tag would leave them out (phototypes.decide). `deep`, when
        given, gets the answer's names ten deep (Identifier.identify_vectors)."""
        def embed():                                 # the worker holds gpu_lock
            model = backbone_model(backbone)
            parts = [embed_one(ident, model, im) for im in images]
            judged = [v for _, _, v, _ in parts]
            verdicts.append(judged if all(v is not None for v in judged) else None)
            tagged = [t for _, _, _, t in parts]
            tags.append(tagged if all(t is not None for t in tagged) else None)
            return np.stack([vec for vec, _, _, _ in parts])

        def finish(vecs, sims):
            if vectors is not None:
                vectors.append(np.asarray(vecs, dtype=np.float32))
            if pooled is not None:
                pooled.append(site_answer(ident, vecs, sims, context, may_show, verdicts[0],
                                          hidden, tags[0], include))
            return ident.identify_vectors(vecs, top_k, n_specimens=n_specimens, context=context,
                                          sims=sims, photo_lookup=photo_lookup,
                                          may_show=may_show, gate=verdicts[0],
                                          higher_ranks=higher, hidden=hidden, types=tags[0],
                                          include=include, deep=deep)
        verdicts: list = []
        tags: list = []
        return model_queue.run(workqueue.Job(
            key=id(ident), photos=len(images), embed=embed, search=ident.nearest.photo_sims,
            finish=finish, interactive=interactive, timer=timer))

    def site_answer(ident: Identifier, vecs, sims, context, may_show, gate, hidden,
                    types=None, include=()) -> dict:
        """The Identify page's answer from photos already embedded: the top 5, the pool of
        specimens the person's own part draws on (yours.POOL), no ranks above family."""
        return ident.identify_vectors(vecs, 5, n_specimens=yours.POOL, context=context,
                                      sims=sims, photo_lookup=photo_lookup, may_show=may_show,
                                      gate=gate, higher_ranks=False, hidden=hidden,
                                      types=types, include=include)

    def site_kept(pooled: dict, vecs, hidden, used: dict, about: dict) -> tuple[dict, dict, list]:
        """(what is kept for the Identify page, its plain answer, its spare specimens).
        Kept for everyone: the plain answer, its spare specimens, the photos' vectors and
        what was left out. Nothing about the person asking."""
        result, spare = yours.split_pool(pooled)
        return ({"result": answercache.without_confusions(result), "spare": spare,
                 "vectors": yours.encode_vectors(vecs) if vecs is not None else None,
                 "hidden": [int(c) for c in np.asarray(
                     hidden if hidden is not None else [], dtype=np.int64).tolist()],
                 "used": used, "observation": about}, result, spare)

    def asked_by(cred: "apiv1.Credential") -> str:
        """Who asked, as a kept answer records it: mycomap.org's server key, the app, or
        a developer key."""
        return {"server": "mycomap.org", "app": "app"}.get(cred.kind, "api")

    def kept_matches_itself(ident: Identifier, source: str, number: str, answer) -> bool:
        """Whether a kept answer shows, among its specimens, a record Vision now leaves out
        of this observation's answer: the record it is, or one of its cross-post twins
        (Identifier.left_out). Such an answer was given before the twin table or link was
        known and matched the observation to itself: it is wrong, not just older, so it is
        answered afresh. This is the one exception to keeping answers until someone runs
        them again."""
        hidden, _ = ident.left_out(source, number)
        if not len(hidden):
            return False
        recs = ident.col_record
        gone = {str(recs.obs[r]) for r in np.unique(recs.rec[hidden]).tolist()}
        return any(str(sp.get("observation_id")) in gone for sp in answercache.specimens_in(answer))

    def reshow_kept(ident: Identifier, answer, view) -> None:
        """An answer kept under an older version, with its specimens' photos checked
        against who may be shown now (answercache.reshow_photos)."""
        answercache.reshow_photos(
            answer, ident.photos_of, photo_lookup,
            lambda info: view.may_show(info.license_class, info.owner_user_id),
            lambda info: photo_url(info.source, info.source_url, "medium"))

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
                     timer: "apiv1.Timer", linked: "yours.Linked | None" = None,
                     who: dict | None = None, include=(), inat_ref: str | None = None) -> dict:
        """The Identify page's answer to uploaded photos; with run links on, kept under
        its own link (`run`), small copies of the photos with it. `include`: photos to count
        though their photo-type tag would leave them out ("Include anyway")."""
        hashes = [] if upload_meter.hashing() else None
        bodies = [] if run_store is not None else None
        read = decode_photos(photos, timer, hashes, bodies)
        images = [img for img, _ in read]
        # The run's id is minted now so iNat's answer, asked for while Vision's runs, is
        # joined to it on mycomap.org and kept beside it here.
        rid = run_links.new_id() if run_store is not None else None
        context, used = fill_context(context, [found for _, found in read])
        # The place entered, else the photos' EXIF one; sent rounded to 0.1 degree.
        inat_key = start_inat(who, images, rid, inat_ref,
                              inat_suggest.rounded_place(context.latitude, context.longitude))
        wanted = site_models(models_)
        view = permission_view()

        def may_show(info) -> bool:
            return view.may_show(info.license_class, info.owner_user_id)

        results, parts, answered = [], [], []
        for backbone, method in wanted:
            ident = site_index(backbone, method, view, timer)
            vectors: list = []
            pooled = embed_and_score(ident, backbone, images, context, may_show, timer,
                                     n_specimens=yours.POOL, vectors=vectors, include=include)
            answered.append((backbone, method, served_version(backbone, method, ident)))
            result, spare = yours.split_pool(pooled)
            vecs = vectors[0] if vectors else None
            if run_store is not None:
                parts.append((ident, result, spare, vecs))
            results.append(yours.for_person(ident, result, spare, vecs,
                                            None, linked, photo_lookup, may_show))
        table = confused_table()
        for result in results:
            confusions.annotate(result, table)
        upload_meter.record("site", len(images), hashes)
        out = {"results": results, "context_used": used}
        if run_store is not None:
            run = without_failing(lambda: keep_run(
                "photos", who, [run_part(i, r, s, v, None) for i, r, s, v in parts], used,
                bodies, rid=rid))
            if run:
                out["run"] = run
            inat_kept(inat_key, rid, who)
        asker = identifylog.site_asker(who)
        for backbone, method, version in answered:
            log_answer(identifylog.SITE_UPLOAD, fresh=True, model=f"{backbone}/{method}",
                       answers_version=version, release=release_now, asked_by=asker,
                       run_id=(out.get("run") or {}).get("id"))
        inat = inat_view(inat_key, who, results)
        if inat is not None:
            out["inat"] = inat
        return out

    # --- iNat's suggestions (inat_suggest.py) ------------------------------------------
    # Vision's answer never waits for them: the photos go to mycomap.org on threads of
    # their own, and the page asks again while they are out (GET /api/inat-suggestions).
    # Keyed by (find, member): a find's key is the page's ref, which another member could
    # send too once this process has forgotten the find; it never reaches someone else's run.
    inat_held: dict[tuple, dict] = {}   # answers back before their run was kept
    inat_runs: dict[tuple, str] = {}    # -> the run, once kept (and the member's own)
    inat_lock = threading.Lock()

    def inat_done_for(sub: str):
        def done(key: str, merged: dict | None) -> None:
            """Every photo of a find has answered: iNat's answer goes beside its run, or
            waits for it."""
            if merged is None or run_store is None:
                return
            with inat_lock:
                rid = inat_runs.pop((key, sub), None)
                if rid is None or run_store.get(rid) is None:  # back before the run is kept
                    inat_held[(key, sub)] = merged
                    return
            run_store.keep_inat(rid, merged)
        return done

    def start_inat(who: dict | None, images: list, rid: str | None,
                   ref: str | None = None, place: tuple[float, float] | None = None) -> str | None:
        """Forwards a site find's photos for the signed-in member; its key, or None."""
        if inat_suggestions is None or not who or not who.get("sub"):
            return None
        try:
            return inat_suggestions.start(str(who["sub"]), images, run_id=rid,
                                          done=inat_done_for(str(who["sub"])), ref=ref,
                                          place=place)
        except Exception as e:  # noqa: BLE001 - no lines; the answer is unaffected
            note(f"[inat-suggest] not sent: {type(e).__name__}")
            return None

    def start_inat_reading(who: dict | None, source: str, number: str, rid: str | None,
                           ref: str | None, use_place: bool) -> str | None:
        """A link whose answer was kept, so Vision read no photos: they are read again off
        the request and forwarded (inat_suggest.start_later). iNat links only."""
        if (inat_suggestions is None or source != "inat" or not who or not who.get("sub")
                or source_fetcher is None):
            return None

        def read():
            obs, pairs = source_fetcher.fetch(source, number, MAX_PHOTOS, indexed=True)
            side = uploads.kept_side(list(backbones.values()))
            images = []
            for _, photo_body in pairs:
                try:
                    images.append(uploads.decode_upload(photo_body, side)[0])
                except Exception:               # not an image we can read: skip
                    continue
            return images, (inat_suggest.public_place(obs) if use_place else None)
        try:
            return inat_suggestions.start_later(str(who["sub"]), read, run_id=rid,
                                                done=inat_done_for(str(who["sub"])), ref=ref)
        except Exception as e:  # noqa: BLE001 - no lines; the answer is unaffected
            note(f"[inat-suggest] not sent: {type(e).__name__}")
            return None

    def inat_kept(key: str | None, rid: str | None, who: dict | None) -> None:
        """The find's run is kept (or couldn't be): iNat's answer goes beside it, if the
        run is this member's (an observation's run belongs to whoever ran it first)."""
        if inat_suggestions is None or key is None:
            return
        sub = str((who or {}).get("sub"))
        run = run_store.get(rid) if run_store is not None and rid else None
        if run is None or ((run.get("owner") or {}).get("id")) != sub:
            with inat_lock:
                inat_held.pop((key, sub), None)
            return
        with inat_lock:
            merged = inat_held.pop((key, sub), None)
            if merged is None:                      # not back yet: done() keeps it
                inat_runs[(key, sub)] = rid
        inat_suggestions.link_run(key, rid)
        if merged is None:
            return
        try:
            run_store.keep_inat(rid, merged)        # False when the run is gone
        except Exception as e:  # noqa: BLE001 - only the run page's copy is lost
            note(f"[inat-suggest] not kept with run {rid}: {type(e).__name__}")

    def inat_view(key: str | None, who: dict | None, results: list[dict]) -> dict | None:
        """What to show under Vision's answer, now that it is out."""
        if inat_suggestions is None or key is None or not who:
            return None
        inat_suggestions.answered(key, inat_suggest.vision_names(results))
        return inat_suggestions.view(key, str(who.get("sub")))

    @app.get("/api/inat-suggestions/{ref}")
    def inat_suggestions_now(request: Request, ref: str, vision: str | None = None):
        """How a find's iNat suggestions stand, for the person who sent its photos only.
        `ref` is the key an answer gave, or site-<inat_ref> the page sent; vision=failed:
        Vision's answer failed or timed out, so the lines show without it."""
        who = person(request)
        view = (inat_suggestions.view(ref, str(who.get("sub")), vision_failed=vision == "failed")
                if inat_suggestions is not None and who else None)
        return JSONResponse({"inat": view}, headers={"Cache-Control": "no-store"})

    # --- run links (runs.py) ----------------------------------------------------------

    def run_owner(who: dict | None) -> dict | None:
        """Who ran it, as the signed session says: mycomap.org id and name."""
        if not who or not who.get("sub"):
            return None
        return {"id": str(who["sub"]), "name": who.get("name")}

    def run_part(ident: Identifier, result: dict, spare: list, vecs, hidden) -> dict:
        """One model's answer as a run keeps it: everyone's answer (no `yours`, no "often
        confused with", which is added afresh when it is shown), the spare specimens and
        the photos' vectors for the owner's own part, and each specimen's reference
        photos, so a photo whose photographer withdrew since is never shown."""
        plain = json.loads(json.dumps({"result": result, "spare": spare},
                                      default=answercache._plain))
        if isinstance(vecs, np.ndarray):
            vecs = yours.encode_vectors(vecs)
        shown = answercache.specimens_in([plain["result"], plain["spare"]])
        return {"result": answercache.without_confusions(plain["result"]),
                "spare": plain["spare"] or [], "vectors": vecs,
                "hidden": [int(c) for c in np.asarray(
                    hidden if hidden is not None else [], dtype=np.int64).tolist()],
                "photo_ids": {str(s["observation_id"]): ident.photos_of(str(s["observation_id"]))
                              for s in shown}}

    def keep_run(kind: str, who: dict | None, parts: list, used: dict, bodies=None,
                 rid: str | None = None, observation: dict | None = None,
                 answered_at: str | None = None) -> dict:
        record = run_links.run_record(
            kind, run_owner(who), parts, used,
            answered_at or answercache.iso_utc(time.time()), model_card().get("version"),
            observation)
        rid = run_store.keep(record, bodies or [], rid=rid)
        return {"id": rid, "url": f"/r/{rid}"}

    def observation_run(who: dict | None, keys: list, held: list, wanted: list, view,
                        timer: "apiv1.Timer", about: dict) -> dict | None:
        """The link of an observation's kept answer: one per kept answer, the same for
        everyone who opens it, its id keyed with the server's secret. It keeps no photos:
        the observation's own (Creative Commons) photo addresses at its source."""
        if run_store is None or not held or any(h is None for h in held):
            return None

        def link() -> dict:
            at = answercache.iso_utc(held[0].answered_at)
            # Keyed on the exact time it was answered: Run again within the same second is
            # still a new answer, with its own link.
            rid = run_links.observation_run_id(run_secret, "|".join(keys),
                                               repr(float(held[0].answered_at)))
            if run_store.get(rid) is not None:
                run_store.opened_by(run_owner(who), rid, "observation")
                card_later(rid)                 # a no-op once its card is kept
                return {"id": rid, "url": f"/r/{rid}"}
            parts = [run_part(site_index(backbone, method, view, timer), h.payload["result"],
                              h.payload.get("spare") or [], h.payload.get("vectors"),
                              h.payload.get("hidden"))
                     for (backbone, method), h in zip(wanted, held)]
            kept = keep_run("observation", who, parts, held[0].payload["used"], rid=rid,
                            observation=run_links.public_observation(about), answered_at=at)
            card_later(rid)                     # its link card, off this request
            return kept
        return without_failing(link)

    def without_failing(make: Callable[[], dict]) -> dict | None:
        """A run's link, or None when it can't be made now (the bucket unreachable): the
        answer is never lost for want of its link."""
        try:
            return make()
        except Exception as e:  # noqa: BLE001 - reported; the answer goes out without a link
            note(f"[runs] no link for this run: {type(e).__name__}: {e}")
            return None

    # Opening a link needs no sign-in (signin.run_path), so reading is limited per address.
    run_reads = RateLimiter(120, 60.0)
    admin_cache: dict[str, tuple[float, bool]] = {}
    NOT_INDEXED = {"Cache-Control": "private, no-store", "X-Robots-Tag": "noindex, nofollow"}

    def is_admin(sub: str) -> bool:
        """A mycomap.org admin (Human Feedback's roles), asked at most every 5 minutes."""
        allow = {x.strip() for x in (config.setting("MV_FEEDBACK_ADMINS") or "").split(",")}
        if sub in allow:
            return True
        hit = admin_cache.get(sub)
        if hit and time.monotonic() - hit[0] < 300:
            return hit[1]
        try:
            roles = feedback_org.roles(sub) if feedback_org is not None else None
        except Exception:  # noqa: BLE001 - not known to be an admin
            roles = None
        admin = bool(roles and roles.get("admin"))
        admin_cache[sub] = (time.monotonic(), admin)
        return admin

    def read_limit(request: Request) -> None:
        wait = run_reads.check(request.client.host if request.client else "unknown")
        if wait:
            raise HTTPException(429, "too many requests from this address; try again shortly",
                                headers={"Retry-After": str(int(wait) + 1)})

    def stored_run(rid: str) -> dict:
        run = run_store.get(rid) if run_store is not None else None
        if run is None:
            raise HTTPException(404, "no such result: the link is wrong, or it was deleted",
                                headers=NOT_INDEXED)
        return run

    @app.get("/api/runs/{rid}")
    def open_run(request: Request, rid: str):
        """A run as its link shows it, to anyone with the link. The person who ran it, signed
        in, also gets their own part ("Your collections"), made now. Never who ran it."""
        read_limit(request)
        return JSONResponse(run_body(stored_run(rid), person(request), request), headers=NOT_INDEXED)

    def run_body(run: dict, who: dict | None, request: Request | None) -> dict:
        """What a run's link shows `who` (None: anyone signed out): its specimen photos
        checked against today's permissions, and, for the person who ran it, their own part
        and iNat's suggestions."""
        rid = run["id"]
        owner = (run.get("owner") or {}).get("id")
        mine = bool(who and owner and str(who.get("sub")) == owner)
        view = permission_view()

        def may_show(info) -> bool:
            return view.may_show(info.license_class, info.owner_user_id)
        linked = linked_of(request) if mine and request is not None else None
        table = confused_table()
        results = []
        for part in run.get("models") or []:
            result = json.loads(json.dumps(part["result"]))
            spare = json.loads(json.dumps(part.get("spare") or []))
            ids = part.get("photo_ids") or {}
            answercache.reshow_photos(
                [result, spare], lambda obs: ids.get(str(obs), []), photo_lookup, may_show,
                lambda info: photo_url(info.source, info.source_url, "medium"))
            ident = vecs = None
            if linked is not None and linked.linked and part.get("vectors"):
                model = result.get("model") or {}
                try:
                    ident = identifier(model.get("backbone"), model.get("method"), view)
                    vecs = yours.decode_vectors(part["vectors"])
                except Exception:  # noqa: BLE001 - a model no longer served: no own part
                    ident = vecs = None
            shown = yours.for_person(
                ident if vecs is not None else None, result, spare, vecs,
                np.asarray(part.get("hidden") or [], dtype=np.int64),
                linked if vecs is not None else None, photo_lookup, may_show)
            confusions.annotate(shown, table)
            results.append(shown)
        inat = run_inat(rid, who, results) if mine else None
        body = {"id": run["id"], "kind": run["kind"], "results": results,
                "context_used": run.get("context_used"),
                "observation": run_links.public_observation(run.get("observation")),
                "answered_at": run.get("answered_at"), "release": run.get("release"),
                "photos": int(run.get("photos") or 0), "mine": mine,
                "can_delete": mine or bool(who and is_admin(str(who.get("sub"))))}
        if inat is not None:
            body["inat"] = inat
        return body

    def run_inat(rid: str, who: dict, results: list[dict]) -> dict | None:
        """iNat's suggestions on a run's page: its owner's only, asked with their iNat
        account. Kept beside the run once back; still on their way, the page asks again."""
        if inat_suggestions is None:
            return None
        kept = run_store.inat(rid)
        if kept is not None:
            return inat_suggest.shown(kept, inat_suggest.vision_names(results))
        return inat_suggestions.view(inat_suggestions.key_of_run(rid), str(who.get("sub")))

    @app.get("/api/runs/{rid}/photos/{name}")
    def run_photo(request: Request, rid: str, name: str):
        """A run's photo copy (1024 px, no EXIF), through this server: never a bucket URL."""
        read_limit(request)
        n = name.removesuffix(".jpg")
        body = (run_store.photo(rid, int(n)) if run_store is not None and n.isdigit()
                and name.endswith(".jpg") else None)
        if body is None:
            raise HTTPException(404, "no such photo", headers=NOT_INDEXED)
        return Response(body, media_type="image/jpeg", headers=NOT_INDEXED)

    # A run's link-preview image (linkcards.py), made once and kept beside the run
    # (site-runs/<id>/card.jpg), and in memory: a run never changes, and one that is
    # deleted is gone from stored_run first. An upload run's is drawn from its own photo
    # copies when first asked for. An iNat/MO run's is drawn from the observation's CC0 /
    # CC BY photos, read from the source once through the same polite fetcher as every
    # observation read, and made when the run is kept, on its own worker (`card_writer`;
    # inline without `background`): a crawler gives up after a few seconds, and MO's pace
    # alone is 5 s a photo. Until it is kept, the run's tags name the site image.
    source_fetcher = observation_fetcher or byobservation.Fetcher()
    if card_writer is None:
        card_writer = (ThreadPoolExecutor(max_workers=1, thread_name_prefix="run-cards").submit
                       if background else (lambda job: job()))
    card_images: "OrderedDict[str, bytes]" = OrderedDict()
    card_lock = threading.Lock()
    card_making = threading.Lock()          # one upload card made at a time
    card_queued: set[str] = set()           # iNat/MO cards waiting for, or on, the worker
    card_failed: dict[str, float] = {}      # run -> when its source last failed (retry later)
    CARD_RETRY_SECONDS = 900
    card_logo = next((p for p in ((web_dist / "mycomap-logo.png") if web_dist else None,
                                  config.REPO_ROOT / "web" / "public" / "mycomap-logo.png")
                      if p is not None and p.is_file()), None)
    site_card_image = (web_dist / "og-card.jpg") if web_dist else None

    def card_from_source(rid: str, run: dict) -> bytes | None:
        """An iNat/MO run's card from the observation's own CC0 / CC BY photos, with their
        credits; None when none qualifies or the source gave none (not asked again for
        CARD_RETRY_SECONDS)."""
        chosen = linkcards.card_photos(run)
        if not chosen:
            return None
        failed = card_failed.get(rid)
        if failed is not None and time.monotonic() - failed < CARD_RETRY_SECONDS:
            return None
        source = (run.get("observation") or {}).get("source")
        try:
            pairs = source_fetcher.photos_at(source, [url for url, _ in chosen])
        except (byobservation.Unavailable, byobservation.Busy) as e:
            note(f"[runs] card for {rid}: the source's photos can't be read now: {e}")
            pairs = []
        bodies, credits = [], []
        for i, body in pairs:
            try:
                bodies.append(run_links.small_copy(body))
            except Exception:  # noqa: BLE001 - not an image we can read: left off the card
                continue
            credits.append(chosen[i][1])
        if not bodies:
            card_failed[rid] = time.monotonic()
            return None
        card_failed.pop(rid, None)
        used = (run.get("observation") or {}).get("photos_used")
        logo = Image.open(card_logo) if card_logo else None
        return linkcards.run_card_jpeg(bodies, linkcards.headline(linkcards.run_answer(run)),
                                       logo, credits=credits,
                                       answered_from=int(used) if used else None)

    def remembered_card(rid: str) -> bytes | None:
        with card_lock:
            if rid in card_images:
                card_images.move_to_end(rid)
                return card_images[rid]
        return None

    def remember_card(rid: str, body: bytes) -> None:
        with card_lock:
            card_images[rid] = body
            while len(card_images) > 64:
                card_images.popitem(last=False)

    def card_later(rid: str) -> None:
        """Makes an iNat/MO run's card off the request, once: when the run is kept, and
        again (after CARD_RETRY_SECONDS) for one whose source failed or whose job a restart
        lost. Nothing for a run with no CC0 / CC BY photo, or whose card is kept."""
        with card_lock:
            if rid in card_queued or rid in card_images:
                return
            card_queued.add(rid)

        def job() -> None:
            try:
                run = run_store.get(rid)
                if run is None or not linkcards.card_photos(run) or run_store.has_card(rid):
                    return
                body = card_from_source(rid, run)
                if body is not None and run_store.keep_card(rid, body):
                    remember_card(rid, body)
            except Exception as e:  # noqa: BLE001 - the site image stands in; tried again
                note(f"[runs] making the card of {rid} failed: {type(e).__name__}: {e}")
            finally:
                with card_lock:
                    card_queued.discard(rid)
        try:
            card_writer(job)
        except Exception as e:  # noqa: BLE001 - the worker is gone (shutting down)
            note(f"[runs] no card for {rid} now: {type(e).__name__}: {e}")
            with card_lock:
                card_queued.discard(rid)

    def observation_card(rid: str) -> bytes | None:
        """An iNat/MO run's kept card, never made here."""
        body = remembered_card(rid)
        if body is None:
            body = run_store.card(rid)
            if body is not None:
                remember_card(rid, body)
        return body

    def upload_card(rid: str, run: dict) -> bytes | None:
        """An upload run's card, made from its own photo copies when first asked for."""
        body = remembered_card(rid)
        if body is not None:
            return body
        with card_making:
            body = remembered_card(rid) or run_store.card(rid)
            if body is None:
                n = int(run.get("photos") or 0)
                bodies = [b for b in (run_store.photo(rid, i) for i in range(min(n, 5))) if b]
                if not bodies:
                    return None
                logo = Image.open(card_logo) if card_logo else None
                body = linkcards.run_card_jpeg(
                    bodies, linkcards.headline(linkcards.run_answer(run)), logo)
                try:
                    run_store.keep_card(rid, body)
                except Exception as e:  # noqa: BLE001 - served now, made again next time
                    note(f"[runs] keeping the card of {rid} failed: {type(e).__name__}: {e}")
            remember_card(rid, body)
        return body

    @app.api_route("/api/runs/{rid}/card.jpg", methods=["GET", "HEAD"])
    def run_card(request: Request, rid: str):
        """The image a shared link shows (Facebook, X, iMessage...): the find's photos and
        the answer. Like the run itself: anyone with the link, never cached or indexed. An
        iNat/MO run's is never made while a crawler waits: until it is kept, the site's
        image stands in (and the card is made, if it never was)."""
        read_limit(request)
        run = stored_run(rid)
        if run.get("kind") == "observation":
            body = observation_card(rid)
            if body is None and linkcards.card_photos(run):
                card_later(rid)
                if site_card_image is not None and site_card_image.is_file():
                    body = site_card_image.read_bytes()
        else:
            body = upload_card(rid, run) if run.get("kind") == "photos" else None
        if body is None:
            raise HTTPException(404, "this result has no photos", headers=NOT_INDEXED)
        return Response(body, media_type="image/jpeg", headers=NOT_INDEXED)

    # The site image's version, read once at startup: its address changes with the picture.
    card_version = linkcards.image_version(web_dist / linkcards.SITE_IMAGE.lstrip("/"))         if web_dist else None

    def public_origin(request: Request) -> str:
        """vision.mycomap.org (MV_PUBLIC_ORIGIN) for the cards' absolute links."""
        return (signin.origin or origin_of(config.setting("MV_PUBLIC_ORIGIN"))
                or str(request.base_url).rstrip("/"))

    def page_card(request: Request, path: str) -> tuple["linkcards.Card | None", int]:
        """The link-preview card for a page shell, and its status: 404 for a run that is
        gone or was never given out."""
        origin = public_origin(request)
        if not run_path(path):
            return linkcards.site_card(origin, path, card_version), 200
        rid = path.removeprefix("/r/")
        read_limit(request)
        try:
            run = run_store.get(rid) if run_store is not None else None
        except Exception as e:  # noqa: BLE001 - the page still loads; the bucket is down
            note(f"[runs] no card for {rid}: {type(e).__name__}: {e}")
            return None, 200
        if run is None:
            return linkcards.gone_card(origin, rid, card_version), 404
        ready = False
        if run.get("kind") == "observation" and linkcards.card_photos(run):
            try:
                ready = remembered_card(rid) is not None or run_store.has_card(rid)
            except Exception as e:  # noqa: BLE001 - the site image until it can be read
                note(f"[runs] no card state for {rid}: {type(e).__name__}: {e}")
            if not ready:
                card_later(rid)
        return linkcards.run_card(origin, rid, run, card_version, card_ready=ready), 200

    # "Where Vision looked" (looked.py; Steve, 2026-10-26): each photo's map of where its
    # evidence for the answer's species was, made when someone asks and kept beside the run
    # (site-runs/<id>/looked.json). Only the photos the run's link shows are read, the
    # specimen matched to each is the one the link shows for it today, and the maps are made
    # one photo at a time behind every identification.
    # The Excluded photos page's maps (excluded_looked.py) take turns on the same worker.
    looked_worker = (looked.Worker(model_queue.idle, gpu_lock, background=background, note=note)
                     if looked_on or excluded_on else None)
    looked_asks = RateLimiter(20, 600.0)
    looked_targets: "OrderedDict[tuple, looked.Target]" = OrderedDict()
    looked_towers: dict[str, looked.Tower] = {}

    def looked_tower(loaded, backbone: str) -> looked.Tower:
        """The backbone's tower, run block by block (call holding gpu_lock)."""
        tower = looked_towers.get(backbone)
        if tower is None:
            # The served torch tower, or (an ONNX tower is served) a torch copy of it.
            found = looked.vit_of(loaded) or looked.vit_of(models.load_backbone(backbone))
            if found is None:
                raise looked.NoMap("this model's maps can't be made")
            tower = looked_towers[backbone] = looked.Tower(*found)
        return tower

    def real_looked_maker(loaded, backbone: str, ident: Identifier, name: str, body: bytes,
                          specimen: bytes | None) -> dict:
        """The served model's map and match for one photo (call holding gpu_lock)."""
        tower = looked_tower(loaded, backbone)
        # An index rebuilt is a new identifier; its size guards against a reused id().
        key = (id(ident), getattr(ident, "embedded", 0), len(ident.index.cols), name)
        target = looked_targets.get(key)
        if target is None:
            target = looked_targets[key] = looked.target_for(ident, name)
            while len(looked_targets) > 32:
                looked_targets.popitem(last=False)
        return looked.photo_map(tower, body, target, specimen)

    looked_maker = where_looked if callable(where_looked) else real_looked_maker

    def looked_off() -> None:
        if not looked_on:
            raise HTTPException(404, "not available here", headers=NOT_INDEXED)

    def first_view(run: dict) -> dict:
        """The run's first answer as its link shows it to anyone today: {photo: per-photo}."""
        view = run_body(run, None, None)
        if not view["results"]:
            return {}
        return {p.get("photo"): p for p in (view["results"][0].get("per_photo") or [])}

    def looked_shown(doc: dict, run: dict) -> dict:
        """The kept maps as the link shows them today: a photo's match only while the link
        still shows that specimen photo (permissions are checked again on every read)."""
        per_photo = first_view(run)
        photos = []
        for entry in doc.get("photos") or []:
            entry = dict(entry)
            match = entry.pop("match", None)
            shown = ((per_photo.get(entry["photo"]) or {}).get("specimens") or [None])[0]
            if (match and shown and shown.get("photo_url") == match.get("photo_url")
                    and str(shown.get("observation_id")) == str(match.get("observation_id"))):
                entry["match"] = {"pairs": match["pairs"], "crop": match.get("crop"), "specimen": {
                    k: shown.get(k) for k in ("observation_id", "name", "similarity", "photo_url",
                                              "photo_owner", "record_url", "source")}}
            else:
                entry["match"] = None
            photos.append(entry)
        return {"status": "ready", "name": doc.get("name"), "model": doc.get("model"),
                "made_at": doc.get("made_at"), "photos": photos}

    def looked_job(rid: str, run: dict) -> looked.Job:
        """The run's maps as a job: one task per photo the answer used and the link shows."""
        parts = run.get("models") or []
        result = (parts[0] or {}).get("result") if parts else None
        species = ((result or {}).get("ranks") or {}).get("species") or []
        if not result or not species:
            raise looked.NoMap("this answer has no species to show the evidence for")
        name = species[0]["name"]
        model = result.get("model") or {}
        backbone, method = model.get("backbone"), model.get("method")
        try:
            ident = identifier(backbone, method, permission_view())
        except Exception as e:  # noqa: BLE001 - a model this server no longer answers with
            raise looked.NoMap("the model that gave this answer is no longer served") from e
        if name not in ident.index.species:
            raise looked.NoMap(f"{name} is not in the reference set any more")
        specimen_of = {i: ((p.get("specimens") or [None])[0]) for i, p in first_view(run).items()}
        left = left_out_of(result)
        if run.get("kind") == "photos":
            urls: list = [None] * int(run.get("photos") or 0)
            source = None
        else:
            obs = run_links.public_observation(run.get("observation")) or {}
            urls = list(obs.get("photos_shown") or [])
            source = obs.get("source")
        tasks = []
        for p in result.get("per_photo") or []:
            i = int(p.get("photo", -1))
            if i in left or not 0 <= i < len(urls) or (source and not urls[i]):
                continue
            spec = specimen_of.get(i)
            if not (spec and spec.get("photo_url") and not spec.get("photo_withheld")):
                spec = None

            def fetch(i=i, spec=spec):
                if source is None:
                    body = run_store.photo(rid, i)
                else:
                    got = source_fetcher.photos_at(source, [urls[i]])
                    body = got[0][1] if got else None
                if body is None:
                    raise looked.NoMap(f"photo {i + 1} could not be read")
                specimen = None
                if spec is not None:
                    try:
                        got = source_fetcher.photos_at(spec.get("source") or "inat",
                                                       [spec["photo_url"]])
                        specimen = got[0][1] if got else None
                    except Exception:  # noqa: BLE001 - the map goes on without its match
                        specimen = None
                return body, specimen

            def make(got, spec=spec):
                body, specimen = got
                loaded = backbone_model(backbone)
                entry = dict(looked_maker(loaded, backbone, ident, name, body, specimen))
                pairs = entry.pop("pairs", None)
                spec_crop = entry.pop("specimen_crop", None)
                entry["match"] = ({"observation_id": spec.get("observation_id"),
                                   "photo_url": spec.get("photo_url"), "pairs": pairs,
                                   "crop": spec_crop}
                                  if spec is not None and specimen is not None and pairs else None)
                return entry

            tasks.append(looked.Task(i, fetch, make))
        if not tasks:
            raise looked.NoMap("none of this answer's photos can be shown here")

        def finish(entries: list[dict]) -> None:
            run_store.keep_looked(rid, {
                "version": looked.VERSION, "name": name,
                "model": {"backbone": backbone, "method": method},
                "made_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "photos": entries})
        return looked.Job(rid, tasks, finish)

    def looked_state(rid: str, run: dict, state: dict | None):
        doc = run_store.looked(rid)
        if doc is not None:
            return JSONResponse(looked_shown(doc, run), headers=NOT_INDEXED)
        if state is None:
            return JSONResponse({"status": "none"}, status_code=404, headers=NOT_INDEXED)
        return JSONResponse(state, status_code=202 if state["status"] == "working" else 503,
                            headers=NOT_INDEXED)

    @app.get("/api/runs/{rid}/looked")
    def read_looked(request: Request, rid: str):
        """A run's maps, to anyone with its link: 200 once made, 202 while being made,
        404 when nobody has asked yet (or not here), 503 when making them failed."""
        read_limit(request)
        looked_off()
        run = stored_run(rid)
        return looked_state(rid, run, looked_worker.status(rid))

    @app.post("/api/runs/{rid}/looked")
    def ask_looked(request: Request, rid: str):
        """Asks for a run's maps: anyone signed in who has the link. 200 when they are kept
        already (or made now), 202 while they are made; 409 when the answer has nothing to
        map, 429 when too many are waiting."""
        looked_off()
        who = person(request)
        if not who:
            raise HTTPException(401, "sign in with your mycomap.org account to ask for this",
                                headers=NOT_INDEXED)
        run = stored_run(rid)
        state = looked_worker.status(rid)
        if run_store.looked(rid) is not None or (state and state["status"] == "working"):
            return looked_state(rid, run, state)
        wait = looked_asks.check(str(who.get("sub")))
        if wait:
            raise HTTPException(429, "you have asked for a lot of these; try again shortly",
                                headers={"Retry-After": str(int(wait) + 1), **NOT_INDEXED})
        try:
            job = looked_job(rid, run)
        except looked.NoMap as e:
            raise HTTPException(409, str(e), headers=NOT_INDEXED) from None
        try:
            looked_worker.submit(job)
        except looked.Busy as e:
            raise HTTPException(429, "Vision is busy making other maps; try again in a minute",
                                headers={"Retry-After": str(int(e.wait_seconds) + 1),
                                         **NOT_INDEXED}) from None
        return looked_state(rid, run, looked_worker.status(rid))

    @app.delete("/api/runs/{rid}", status_code=204)
    def delete_run(request: Request, rid: str):
        """Deletes a run, its photo copies and its link: the person who ran it, or a
        mycomap.org admin."""
        who = person(request)
        if not who:
            raise HTTPException(401, "sign in with your mycomap.org account")
        run = stored_run(rid)
        owner = (run.get("owner") or {}).get("id")
        if not (owner and str(who.get("sub")) == owner) and not is_admin(str(who.get("sub"))):
            raise HTTPException(403, "only the person who ran it, or an admin, can delete it")
        run_store.delete(rid)
        with card_lock:
            card_images.pop(rid, None)
        return Response(status_code=204)

    # --- the identification log's read for the paper (identifylog.first_answers) ----------
    @app.get("/api/identify-log/first", include_in_schema=False)
    def identify_log_first(request: Request, source: str | None = None,
                           since: str | None = None, format: str = "json"):
        """Per observation, the first time Vision answered it, with the path: for the
        paper's truth-independence check. mycomap.org's server key or a signed-in
        mycomap.org admin only; private, never cached. Who asked is never in it."""
        given = request.headers.get("authorization", "")
        by_key = bool(org_secret) and secrets.compare_digest(
            given.encode(), f"Bearer {org_secret}".encode())
        if not by_key:
            who = person(request)
            if not who:
                raise HTTPException(401, "mycomap.org's server key or an admin's sign-in",
                                    headers=NOT_INDEXED)
            if not is_admin(str(who.get("sub"))):
                raise HTTPException(403, "mycomap.org admins only", headers=NOT_INDEXED)
        if source is not None and source not in byobservation.SOURCES:
            raise HTTPException(400, "source is inat or mo", headers=NOT_INDEXED)
        after = None
        if since:
            try:
                after = datetime.fromisoformat(since.replace("Z", "+00:00"))
                after = (after if after.tzinfo else after.replace(tzinfo=timezone.utc)
                         ).timestamp()
            except ValueError:
                raise HTTPException(400, "since is an ISO 8601 time", headers=NOT_INDEXED) from None
        rows = identify_log.first_answers(source=source, since=after)
        if format == "tsv":
            return Response(identifylog.as_tsv(rows), headers=NOT_INDEXED,
                            media_type="text/tab-separated-values; charset=utf-8")
        if format != "json":
            raise HTTPException(400, "format is json or tsv", headers=NOT_INDEXED)
        return JSONResponse({"observations": rows, "count": len(rows),
                             "about": identifylog.FIRST_ABOUT}, headers=NOT_INDEXED)

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

    def read_name_changes(rebuild: bool = True) -> dict | None:
        """Read .org's renames; when one reached the layer, the indexes are rebuilt behind
        the ones answering (not in the night, which rebuilds after its own changes)."""
        out = name_change_reader.run() if name_change_reader else None
        if rebuild and out and (out.get("renamed") or out.get("name_moved")):
            rebuild_behind("a rename")
        return out
    app.state.read_name_changes = read_name_changes

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
            note(f"[feedback] specimen crops: {feedback.sync_crops(feedback_store, conn)}")
        read_name_changes(rebuild=False)   # .org's renames first: tonight counts only the rest
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
            # Every index answering now, waited for: the night's people get tonight's.
            served_now = {"/".join(k) for k in indexes.served()}
            for spec in [*warm.models, *sorted(served_now - set(warm.models))]:
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
        report["early_answers"] = score_early_answers_tonight()
        return report

    def score_early_answers_tonight() -> dict:
        """Answers given before the DNA verdict, decided now that tonight's green records are
        in (early_answers.score_pending). Never fails the night."""
        own = manifest.connect(manifest_path)
        own.execute("pragma busy_timeout = 30000")
        try:
            out = early_answers.score_pending(early_store.conn, own, lock=early_store.lock)
            note(f"[nightly] answered before the verdict: {out}")
            return out
        except Exception as e:  # noqa: BLE001 - the reference set is already updated
            note(f"[nightly] answered before the verdict failed: {type(e).__name__}: {e}")
            return {"error": f"{type(e).__name__}: {e}"[:500]}
        finally:
            own.close()

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
        # A fresh release's layer holds the laptop's export, often days old: one ordinary
        # night soon, after the preload's index build, not at the next 3 a.m.
        with db_lock:
            nightly.arm_catch_up(schedule, conn, ready=lambda: warm.state != "loading",
                                 note=note)
        nightly.start(schedule, run_nightly, note)

    # Human Feedback (feedback.py, feedback_api.py; Steve, 2026-10-18): the review queue of
    # where Vision and the reference data disagree, for mycomap.org's admins, project leaders
    # and contributors. Its decisions are kept with the usage counts (state/), across
    # releases. "Refresh" re-asks the served model about a reference record from its stored
    # vectors, the record itself left out, behind the same queue as everything else.
    if feedback_store is None:
        feedback_store = feedback.Store(feedback.default_path(manifest_path))
    if feedback_store.gate is None:
        # The paper's seal: nothing this server adds to a queue (a reviewer's photo item, a
        # blind refresh) may name a v1-test record (feedback.SealGate).
        feedback_store.gate = feedback.SealGate.from_manifest(conn, lock=db_lock)
    app.state.feedback = feedback_store
    with db_lock:
        feedback.sync_exclusions(feedback_store, conn)
        feedback.sync_crops(feedback_store, conn)

    def feedback_rescore(record_key: str, who: dict | None = None) -> dict:
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
        reviewer = who or {}
        log_answer(identifylog.FEEDBACK_RESCORE, fresh=True, source=source,
                   observation_id=number, model=f"{backbone}/{method}",
                   answers_version=served_version(backbone, method, ident),
                   release=release_now,
                   asked_by=identifylog.Asker("reviewer", str(reviewer.get("sub") or "") or None,
                                              reviewer.get("name")))
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

    def cached_mo_place(key: str) -> str | None:
        """MO's location name from the cache of MO's public points (mopoints.py), if read."""
        try:
            with db_lock:
                r = conn.execute("select location_name from mo_public_points "
                                 "where observation_id = ? and status = 'ok'", (key,)).fetchone()
        except sqlite3.OperationalError:          # a manifest without that table
            return None
        return r[0] if r and r[0] else None

    def name_info(wanted: list[str]) -> dict[str, dict]:
        with db_lock:
            return feedback_org_rules.manifest_name_info(conn, genus_known)(wanted)

    review_photos = photo_review.ReviewPhotos(q, permission_view)
    feedback_requests = feedback_api.add_routes(
        app, person=person, store=feedback_store, org=feedback_org, rescore=feedback_rescore,
        photos=review_photos, name_info=name_info,
        on_name_update_done=read_name_changes, places=feedback_places or pair_places.PairPlaces(
            cached_mo_place), inat_names=feedback_inat_names or name_context.InatNames(),
        note=note)
    app.state.feedback_requests = feedback_requests

    # Research > Excluded photos (excluded_looked.py; Steve, 2026-10-26): the photos people left
    # out in Human Feedback, as a read-only viewer sees them there, and "where Vision looked" on
    # any of them, from the served model (v1 only), on the run maps' worker. The paper's seal
    # holds maps back for records past it until MV_EXCLUDED_LOOKED_UNSEALED=1.
    excluded_list = xl.Listing(feedback_store)
    excluded_maps = (xl.Maps((config.RELEASE_ROOT / "state" if config.RELEASE_ROOT
                              else Path(manifest_path or config.MANIFEST_PATH).parent) / xl.FILE)
                     if excluded_on else None)
    excluded_daily = RateLimiter(200, 86400.0)
    app.state.excluded_limits = (looked_asks, excluded_daily)
    excluded_maker = excluded_looked if callable(excluded_looked) else None
    unsealed = (excluded_unsealed if excluded_unsealed is not None
                else config.setting(xl.UNSEAL_SETTING) == "1")

    def real_excluded_maker(loaded, backbone: str, ident: Identifier, name: str, body: bytes,
                            hidden) -> dict:
        """One excluded photo's map and rank, its own record left out (call holding gpu_lock)."""
        tower = looked_tower(loaded, backbone)
        entry = looked.photo_map(tower, body, looked.target_for(ident, name, hidden), None)
        entry.pop("pairs", None)
        entry["rank"] = xl.rank_of(ident, looked.photo_vector(tower, body), name, hidden)
        return entry

    def excluded_backbone() -> str | None:
        """The served backbone, whose maps (v1's) are the only ones made here."""
        ready = ready_backbones()
        return ready[0] if ready else None

    def excluded_off() -> None:
        if not excluded_on:
            raise HTTPException(404, "not available here", headers=NOT_INDEXED)

    def is_reviewer(request: Request) -> bool:
        return not app.state.feedback_viewer(request)[3]

    def excluded_row(request: Request, photo_key: str) -> "xl.Excluded":
        """The excluded photo as this viewer may see it, or 404: listed to them, and a photo
        anyone may see now (rechecked on every read)."""
        rows = xl.visible(excluded_list.rows(), is_reviewer(request))
        hit = next((r for r in rows if r.photo_key == photo_key), None)
        row = review_photos.one(photo_key) if hit else None
        view = permission_view()
        if (hit is None or row is None or review_photos.declined(row, view)
                or not view.may_show(row.license_class, row.owner_user_id)):
            raise HTTPException(404, "no such photo shown here", headers=NOT_INDEXED)
        return hit

    def excluded_model(model: str) -> str:
        """v1's backbone, or 404: no other model's maps are made or served here (a base map
        kept before is never read)."""
        backbone = excluded_backbone() if model == xl.MODEL else None
        if backbone is None:
            raise HTTPException(404, f"no {model} maps here; only Vision v1's",
                                headers=NOT_INDEXED)
        return backbone

    def excluded_held(hit: "xl.Excluded") -> None:
        if xl.held_back(feedback_store.gate, hit.record_key, unsealed):
            raise HTTPException(403, "this photo's map is held back until the paper's sealed "
                                     "test is scored", headers=NOT_INDEXED)

    def excluded_rid(photo_key: str, model: str) -> str:
        return f"excluded:{model}:{photo_key}"

    def excluded_state(hit: "xl.Excluded", model: str):
        doc = excluded_maps.get(hit.photo_key, model)
        if doc is not None:
            return JSONResponse({"status": "ready", **doc}, headers=NOT_INDEXED)
        state = looked_worker.status(excluded_rid(hit.photo_key, model))
        if state is None:
            return JSONResponse({"status": "none"}, status_code=404, headers=NOT_INDEXED)
        return JSONResponse(state, status_code=202 if state["status"] == "working" else 503,
                            headers=NOT_INDEXED)

    def manifest_names(keys: list[str]) -> dict[str, str]:
        if not keys:
            return {}
        marks = ",".join("?" * len(keys))
        return {str(k): n for k, n in q(f"select observation_id, coalesce(species, "
                                        f"scientific_name) from records where observation_id "
                                        f"in ({marks})", tuple(keys)) if n}

    def excluded_job(hit: "xl.Excluded", model: str, backbone: str) -> looked.Job:
        """One photo's map as a job: the index, the name and the photo are read outside the
        model lock (fetch); the map is made holding it (make)."""
        row = review_photos.one(hit.photo_key)
        source, number = (("mo", hit.record_key[3:]) if hit.record_key.startswith("mo:")
                          else ("inat", hit.record_key))

        def fetch():
            method = default_method()
            try:
                ident = identifier(backbone, method, permission_view())
            except Exception as e:  # noqa: BLE001 - a model this server can't answer with now
                raise looked.NoMap("the model is not ready here") from e
            hidden, own = ident.left_out(source, number)
            name = (ident.col_record[int(own[0])].species if len(own)
                    else manifest_names([hit.record_key]).get(hit.record_key))
            if not name or name not in ident.index.species:
                raise looked.NoMap("the record's name is not in the reference set")
            got = review_photos.image(row, "full")
            if got is None:
                raise looked.NoMap("the photo could not be read just now")
            return ident, method, name, hidden, got[0]

        def make(got):
            ident, method, name, hidden, body = got
            maker = excluded_maker or real_excluded_maker
            entry = dict(maker(backbone_model(backbone), backbone, ident, name, body, hidden))
            return {"name": name, "method": method, "entry": entry}

        def finish(entries: list[dict]) -> None:
            e = entries[0]
            m = e["entry"]
            excluded_maps.keep(hit.photo_key, model, {
                "version": xl.VERSION, "name": e["name"], "model": model,
                "backbone": backbone, "method": e["method"],
                "made_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "map": {"crop": m["crop"], "grid": m["grid"], "map": m["map"]},
                "rank": m.get("rank"), "top": xl.RANK_DEPTH})
        return looked.Job(excluded_rid(hit.photo_key, model),
                          [looked.Task(0, fetch, make)], finish)

    @app.get("/api/research/excluded-photos")
    def excluded_photos(request: Request, category: str = "all", offset: int = 0,
                        limit: int = 48, maps: int = 0):
        """The excluded photos of a category, a page at a time, as a read-only Human Feedback
        viewer sees them; People only to reviewers. Open to anyone."""
        read_limit(request)
        excluded_off()
        rows = xl.visible(excluded_list.rows(), is_reviewer(request))
        mapped = excluded_maps.mapped()
        try:
            chosen = xl.select(rows, category, mapped if maps else None)
        except ValueError as e:
            raise HTTPException(400, str(e), headers=NOT_INDEXED) from None
        start = max(0, offset)
        page = chosen[start:start + min(max(1, limit), xl.PAGE_MAX)]
        info = review_photos.many([(r.photo_key, r.record_key) for r in page])
        shown = {d["photo_key"]: d for d in review_photos.for_viewer(
            [i["row"] for i in info.values()], reviewer=False, thumb="grid")}
        names = manifest_names(sorted({r.record_key for r in page}))
        tiles = []
        for r in page:
            ph, i = shown.get(r.photo_key) or {}, info.get(r.photo_key) or {}
            public = ph.get("shown") == "public"
            held = xl.held_back(feedback_store.gate, r.record_key, unsealed)
            tiles.append({"photo_key": r.photo_key, "record_key": r.record_key,
                          "name": names.get(r.record_key), "category": r.category,
                          "shown": "public" if public else "hidden",
                          "thumb": ph.get("thumb") if public else None,
                          "photo_url": ph.get("full") if public else None,
                          "owner_login": i.get("owner_login") if public else None,
                          "license_code": i.get("license_code") if public else None,
                          "record_url": xl.record_url(r.record_key),
                          "held": held,
                          "maps": [xl.MODEL] if public and not held and r.photo_key in mapped
                          else []})
        return JSONResponse({"categories": xl.categories(rows), "total": len(chosen),
                             "offset": start, "tiles": tiles,
                             "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")},
                            headers=NOT_INDEXED)

    @app.get("/api/research/excluded-photos/looked")
    def read_excluded_looked(request: Request, photo: str, model: str = "v1"):
        """An excluded photo's map, to anyone: 200 once made, 202 while being made, 404 when
        nobody has asked yet (or for any model but v1), 403 while the paper's seal holds it
        back, 503 when it failed."""
        read_limit(request)
        excluded_off()
        hit = excluded_row(request, photo)
        excluded_model(model)
        excluded_held(hit)
        return excluded_state(hit, model)

    @app.post("/api/research/excluded-photos/looked")
    def ask_excluded_looked(request: Request, photo: str, model: str = "v1"):
        """Asks for an excluded photo's map: anyone signed in. 200 when kept already, 202 while
        it is made; 409 when it can't be made, 429 past 20 asks in 10 minutes or 200 in a day
        (with the run maps' asks), or while too many maps are waiting. Who asked is not kept."""
        excluded_off()
        who = person(request)
        if not who:
            raise HTTPException(401, "sign in with your mycomap.org account to ask for this",
                                headers=NOT_INDEXED)
        hit = excluded_row(request, photo)
        backbone = excluded_model(model)
        excluded_held(hit)
        rid = excluded_rid(hit.photo_key, model)
        state = looked_worker.status(rid)
        if excluded_maps.get(hit.photo_key, model) is not None or (
                state and state["status"] == "working"):
            return excluded_state(hit, model)
        sub = str(who.get("sub"))
        wait = looked_asks.check(sub) or excluded_daily.check(sub)
        if wait:
            raise HTTPException(429, "you have asked for a lot of these; try again later",
                                headers={"Retry-After": str(int(wait) + 1), **NOT_INDEXED})
        try:
            looked_worker.submit(excluded_job(hit, model, backbone))
        except looked.Busy as e:
            raise HTTPException(429, "Vision is busy making other maps; try again in a minute",
                                headers={"Retry-After": str(int(e.wait_seconds) + 1),
                                         **NOT_INDEXED}) from None
        state = looked_worker.status(rid)
        if state and state["status"] == "failed":
            raise HTTPException(409, state["detail"], headers=NOT_INDEXED)
        return excluded_state(hit, model)
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

        def v1_photos(photos: list[UploadFile], timer: "apiv1.Timer",
                      hashes: list | None = None) -> list[tuple]:
            try:
                return decode_photos(photos, timer, hashes)
            except HTTPException as e:
                if e.status_code == 413:
                    raise apiv1.V1Error(413, "too_large", str(e.detail)) from e
                raise apiv1.V1Error(415, "unsupported_image", str(e.detail)) from e

        def forward_app_photo(cred: "apiv1.Credential", image, client_ref: str | None,
                              photo_ref: str, lat, lng, open_place: bool) -> None:
            """An app photo to mycomap.org for iNat's suggestions, without waiting: the app's
            own members only (a token naming them, "uid"), never a developer key or
            mycomap.org's server key. The app reads the lines from mycomap.org, so nothing
            comes back here. The place goes only for a find whose geoprivacy is open
            (`inat_location`)."""
            if inat_suggestions is None or cred.kind != "app" or not cred.uid:
                return
            place = (lat, lng) if open_place and lat is not None and lng is not None else None
            try:
                inat_suggestions.forward(cred.uid, image, client_ref=client_ref,
                                         photo_ref=photo_ref, location=place)
            except Exception as e:  # noqa: BLE001 - no lines; the answer is unaffected
                note(f"[inat-suggest] app photo not sent: {type(e).__name__}")

        def v1_identify(request: Request, photos: list[UploadFile] = File(...),
                        latitude: float | None = Form(None), longitude: float | None = Form(None),
                        observed_on: str | None = Form(None), model: str | None = Form(None),
                        top_k: int = Form(5), client_ref: str | None = Form(None),
                        include: str | None = Form(None), inat_location: bool = Form(False)):
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
            try:
                counted_anyway = asked_include(include, len(photos))
            except ValueError as e:
                raise apiv1.V1Error(400, "bad_request", str(e)) from None
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
                hashes = [] if upload_meter.hashing() else None
                read = v1_photos(photos, timer, hashes)
                for n, (img, _) in enumerate(read):
                    forward_app_photo(cred, img, client_ref, str(n + 1), latitude, longitude,
                                      inat_location)
                context, used = fill_context(Context(latitude, longitude, observed_on),
                                             [found for _, found in read])
                view = permission_view()
                ident = v1_index(backbone, method, view, timer)
                result = embed_and_score(
                    ident, backbone, [img for img, _ in read], context,
                    lambda info: view.may_show(info.license_class, info.owner_user_id),
                    timer, top_k, interactive=cred.interactive, higher=True,
                    include=counted_anyway)
            finally:
                in_flight.leave()
            upload_meter.record("api", len(read), hashes)
            quotas.count(cred, 0 if same_find else 1, len(photos))
            if not same_find:
                quotas.remember_counted(cred, client_ref)
            body = v1_result(result, backbone, method, used, timer)
            body["usage"] = quotas.usage(cred)
            log_answer(identifylog.V1_IDENTIFY, fresh=True, model=f"{backbone}/{method}",
                       answers_version=served_version(backbone, method, ident),
                       release=release_now, asked_by=identifylog.credential_asker(cred),
                       client_ref=client_ref)
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
            fetcher = source_fetcher

            def fetched_hashes(ident: Identifier, pairs) -> list[int]:
                """The dHash of each fetched photo, when the index has a twin table to match
                them against (a cross-post's copy of the same picture is left out)."""
                if getattr(ident, "twins", None) is None:
                    return []
                out = []
                for _, photo_body in pairs:
                    try:
                        out.append(dhash.dhash64(photo_body))
                    except Exception:               # not an image we can read: skip
                        continue
                return out

            def flag(payload: dict, name: str) -> bool:
                v = payload.get(name, True)
                if not isinstance(v, bool):
                    raise apiv1.V1Error(400, "bad_request", f"{name} is true or false")
                return v

            def held_v1(source: str, number: str, backbone: str, method: str, top_k: int,
                        use_place: bool, use_date: bool, view, timer,
                        include=()) -> dict | None:
                """The answer Vision holds for an observation, in the v1 shape: the newer of
                identify-by-observation's (`kept_from` api) and the site's Identify page's
                ("from a link", `kept_from` site; same model, place and date), under
                whichever answers version it was given (Steve, 2026-10-19: kept until someone
                runs it again). `answered` says when, who asked, and whether Vision still
                answers from the same reference set; an older one has its specimens' photos
                checked against who may be shown now. None when nothing is held."""
                version = answers_version(backbone, method, view)
                api = answer_cache.held(answercache.answer_key(
                    "v1", source, number, backbone, method,
                    {"top_k": top_k, "place": use_place, "date": use_date,
                     **type_options(include)}))
                site = answer_cache.held(answercache.answer_key(
                    "site", source, number, backbone, method,
                    {"place": use_place, "date": use_date, **type_options(include)}))
                # An older answer that matched the observation to itself is not served.
                ident = None
                if any(k is not None and k.version != version for k in (api, site)):
                    ident = v1_index(backbone, method, view, timer)
                if api is not None and api.version != version and kept_matches_itself(
                        ident, source, number, api.payload):
                    api = None
                if site is not None and site.version != version and kept_matches_itself(
                        ident, source, number, [site.payload["result"], site.payload.get("spare")]):
                    site = None
                if api is None and site is None:
                    return None
                # A run here keeps the site's copy first, so ties go to this one.
                if site is None or (api is not None and api.answered_at >= site.answered_at):
                    kept, kept_from = api, "api"
                    body = {**kept.payload, "request_id": "r_" + secrets.token_urlsafe(12)}
                    body.pop("photo_tagger", None)          # said in `answered`
                    if kept.version != version:
                        reshow_kept(ident, body, view)
                    confusions.annotate(body, confused_table())
                else:
                    kept, kept_from = site, "site"
                    held = kept.payload
                    if kept.version != version:
                        reshow_kept(ident, [held["result"], held.get("spare")], view)
                    site_about = dict(held["observation"])
                    site_about.pop("photos_credit", None)        # the site's, for its cards
                    own = site_about.pop("excluded_reference", False)
                    left = site_about.pop("left_out_photos", 0)
                    body = v1_result(held["result"], backbone, method, held["used"], timer)
                    body["observation"] = {"place": None, **site_about}
                    body["excluded_reference"] = ({"source": source, "source_id": number}
                                                  if own else None)
                    body["left_out_photos"] = int(left or 0)
                    if held.get("release") is not None:
                        body["model"] = {**body["model"], "version": held["release"]}
                body["model"] = {**body["model"], "answers_version": kept.version}
                body.update(cached=True, kept_from=kept_from, answered=kept.about(version))
                return body

            def log_kept_v1(path: str, body: dict, source: str, number: str, backbone: str,
                            method: str, kept_key: str, site_key: str, cred,
                            client_ref: str | None = None) -> None:
                """A kept answer served in the v1 shape (held_v1), as a log row."""
                log_answer(path, fresh=False, source=source, observation_id=number,
                           answer_key=site_key if body.get("kept_from") == "site" else kept_key,
                           answers_version=(body.get("model") or {}).get("answers_version"),
                           model=f"{backbone}/{method}",
                           release=(body.get("model") or {}).get("version"),
                           asked_by=identifylog.credential_asker(cred), client_ref=client_ref)

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
                rerun = payload.get("rerun", False)
                if not isinstance(rerun, bool):
                    raise apiv1.V1Error(400, "bad_request", "rerun is true or false")
                try:
                    counted_anyway = asked_include(payload.get("include"))
                except ValueError as e:
                    raise apiv1.V1Error(400, "bad_request", str(e)) from None
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
                # Already answered (here or on the site): the same answer, without reading
                # the observation or scoring it again, until someone asks to run it again.
                version = answers_version(backbone, method, view)
                kept_key = answercache.answer_key(
                    "v1", source, number, backbone, method,
                    {"top_k": top_k, "place": use_place, "date": use_date,
                     **type_options(counted_anyway)})
                site_key = answercache.answer_key("site", source, number, backbone, method,
                                                  {"place": use_place, "date": use_date,
                                                   **type_options(counted_anyway)})
                body = None if rerun else held_v1(source, number, backbone, method, top_k,
                                                  use_place, use_date, view, timer,
                                                  counted_anyway)
                if body is not None:
                    quotas.count(cred, 0 if same_find else 1, 0)
                    if not same_find:
                        quotas.remember_counted(cred, client_ref)
                    log_kept_v1(identifylog.V1_OBSERVATION, body, source, number, backbone,
                                method, kept_key, site_key, cred, client_ref)
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
                credits = None                  # the site's copy only (its run cards)
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
                    credits = [byobservation.shown_photo_credit(obs, i) for i in kept]
                    hidden, _ = ident.left_out(source, number, obs.photo_ids, obs.linked,
                                               fetched_hashes(ident, pairs))
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
                    # The site's answer comes from the same photos (pooled), kept for the
                    # Identify page so opening the observation there runs nothing again.
                    pooled: list = []
                    deep: list = []             # the names ten deep, for early_answers
                    if len(own):
                        def finish(v, sims):
                            pooled.append(site_answer(ident, v, sims, None, may_show, None,
                                                      hidden))
                            return ident.identify_vectors(
                                v, top_k, context=None, sims=sims, photo_lookup=photo_lookup,
                                may_show=may_show, gate=None, higher_ranks=True,
                                hidden=hidden, deep=deep)
                        result = model_queue.run(workqueue.Job(
                            key=id(ident), photos=0, embed=lambda: vecs,
                            search=ident.nearest.photo_sims, finish=finish,
                            interactive=False, timer=timer))
                        scored_vecs = vecs
                    else:
                        got: list = []
                        result = embed_and_score(ident, backbone, images, context or Context(),
                                                 may_show, timer, top_k, interactive=False,
                                                 higher=True,
                                                 hidden=hidden if len(hidden) else None,
                                                 vectors=got, pooled=pooled,
                                                 include=counted_anyway, deep=deep)
                        scored_vecs = got[0] if got else None
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
                who = asked_by(cred)
                if pooled:
                    site_about = {**{k: v for k, v in about.items() if k != "place"},
                                  "photos_credit": credits,
                                  "excluded_reference": bool(len(own)),
                                  "left_out_photos": int(len(hidden))}
                    if site_about["photos_shown"] is None:   # a reference record: none shown
                        site_about.update(photos=n_photos, photos_shown=[None] * n_photos,
                                          photos_credit=[None] * n_photos)
                    kept_site, _, _ = site_kept(pooled[0], scored_vecs, hidden, used, site_about)
                    kept_site["release"] = model_card().get("version")
                    answer_cache.put(site_key, version, tagger_mark(kept_site), who)
                answer_cache.put(kept_key, version, tagger_mark(answercache.without_confusions(
                    {k: v for k, v in body.items() if k not in ("request_id", "timing", "usage")})),
                    who)
                body["answered"] = answer_cache.held(kept_key).about(version)
                body["cached"] = False
                body["usage"] = quotas.usage(cred)
                lists, depth = early_answers.ranked_lists(deep, body.get("ranks"), top_k)
                freeze_answer(source=source, observation_id=number,
                              model=f"{backbone}/{method}",
                              path=early_answers.V1_OBSERVATION, lists=lists, depth=depth,
                              asked_by=who, release=release_now(), answers_version=version,
                              place_used=bool(used.get("place_from")),
                              date_used=bool(used.get("date_from")),
                              excluded_reference=bool(len(own)),
                              left_out_photos=int(len(hidden)))
                log_answer(identifylog.V1_OBSERVATION, fresh=True, source=source,
                           observation_id=number, answer_key=kept_key, answers_version=version,
                           model=f"{backbone}/{method}", release=release_now,
                           asked_by=identifylog.credential_asker(cred), client_ref=client_ref)
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
                               use_place: bool = True, use_date: bool = True,
                               include: str = ""):
                timer = apiv1.Timer()
                cred: apiv1.Credential = request.state.credential
                if source not in byobservation.SOURCES:
                    raise apiv1.V1Error(404, "not_found", "answers are kept for inat or mo")
                number = byobservation.valid_id(observation_id)
                if number is None:
                    raise apiv1.V1Error(400, "bad_request", "the observation number is digits")
                if not 1 <= top_k <= 10:
                    raise apiv1.V1Error(400, "bad_request", "top_k is 1 to 10")
                try:
                    counted_anyway = asked_include(include)
                except ValueError as e:
                    raise apiv1.V1Error(400, "bad_request", str(e)) from None
                backbone, method = v1_model(model)
                body = held_v1(source, number, backbone, method, top_k, use_place, use_date,
                               permission_view(), timer, counted_anyway)
                if body is None:
                    raise apiv1.V1Error(404, "not_answered", "Vision holds no answer for "
                                        "this observation")
                log_kept_v1(identifylog.V1_ANSWERS, body, source, number, backbone, method,
                            answercache.answer_key("v1", source, number, backbone, method,
                                                   {"top_k": top_k, "place": use_place,
                                                    "date": use_date,
                                                    **type_options(counted_anyway)}),
                            answercache.answer_key("site", source, number, backbone, method,
                                                   {"place": use_place, "date": use_date,
                                                    **type_options(counted_anyway)}),
                            cred)
                body.update(timing=timer.block(), usage=quotas.usage(cred))
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
                # Run it again (the page's "Run again"): never served from what is kept.
                rerun = payload.get("rerun", False) is True
                # The page's name for the find (inat_suggest.py), so iNat's lines can be
                # asked for even when Vision's answer fails.
                inat_ref = payload.get("inat_ref") if isinstance(payload.get("inat_ref"), str) else None
                # "Include anyway": photos counted though their photo-type tag would leave
                # them out; an answer kept for another set of photos is never served for it.
                try:
                    counted_anyway = asked_include(payload.get("include"))
                except ValueError as e:
                    raise HTTPException(400, str(e)) from None
                wanted = site_models(models_)
                timer = apiv1.Timer()
                timer.ms["fetch_ms"] = 0
                view = permission_view()
                versions = [answers_version(b, m, view) for b, m in wanted]
                keys = [answercache.answer_key("site", source, number, b, m,
                                               {"place": use_place, "date": use_date,
                                                **type_options(counted_anyway)})
                        for b, m in wanted]
                # Kept here, or by identify-by-observation (mycomap.org's Validate Records
                # keeps the site's answer too): served under whichever version it was given
                # until someone runs it again (Steve, 2026-10-19).
                held = [] if rerun else [answer_cache.held(k) for k in keys]
                linked = linked_of(request)
                request_person = person(request)

                def may_show(info) -> bool:
                    return view.may_show(info.license_class, info.owner_user_id)

                # A kept answer is everyone's: it holds no one's records, only the photos'
                # vectors, so the person's own part is made here. One kept before it held
                # them is served without that part rather than run again.
                needs_vectors = linked is not None and linked.linked
                # An older answer that matched the observation to itself is answered afresh.
                if held and all(h is not None for h in held) and any(
                        h.version != v and kept_matches_itself(
                            site_index(b, m, view, timer), source, number,
                            [h.payload["result"], h.payload.get("spare")])
                        for (b, m), h, v in zip(wanted, held, versions)):
                    held = []
                if held and all(h is not None for h in held):
                    table = confused_table()
                    results = []
                    for (backbone, method), h, version in zip(wanted, held, versions):
                        kept = h.payload
                        stale = h.version != version
                        vecs = yours.decode_vectors(kept.get("vectors")) if needs_vectors else None
                        ident = (site_index(backbone, method, view, timer)
                                 if needs_vectors or stale else None)
                        if stale:
                            reshow_kept(ident, [kept["result"], kept.get("spare")], view)
                        mine = yours.for_person(
                            ident if vecs is not None else None, kept["result"],
                            kept.get("spare") or [], vecs,
                            np.asarray(kept.get("hidden") or [], dtype=np.int64), linked,
                            photo_lookup, may_show)
                        confusions.annotate(mine, table)
                        mine["answered"] = h.about(version)
                        results.append(mine)
                    run = observation_run(request_person, keys, held, wanted, view, timer,
                                          held[0].payload["observation"])
                    inat_key = start_inat_reading(request_person, source, number,
                                                  (run or {}).get("id"), inat_ref, use_place)
                    inat_kept(inat_key, (run or {}).get("id"), request_person)
                    for (backbone, method), key, h in zip(wanted, keys, held):
                        log_answer(identifylog.SITE_OBSERVATION, fresh=False, source=source,
                                   observation_id=number, answer_key=key,
                                   answers_version=h.version, model=f"{backbone}/{method}",
                                   release=h.payload.get("release"),
                                   asked_by=identifylog.site_asker(request_person),
                                   run_id=(run or {}).get("id"))
                    inat = inat_view(inat_key, request_person, results)
                    return {"results": results,
                            "context_used": held[0].payload["used"],
                            "observation": held[0].payload["observation"],
                            "cached": True,
                            "answered": held[0].about(versions[0]),
                            **({"run": run} if run else {}),
                            **({"inat": inat} if inat is not None else {}),
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
                # iNat's suggestions (iNat links only), while Vision's answer runs: the
                # photos as read, the public place rounded (none when iNat obscures it).
                inat_key = (start_inat(request_person, images, None, inat_ref,
                                       inat_suggest.public_place(obs) if use_place else None)
                            if source == "inat" else None)

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
                    frozen = []                 # per model: what early_answers keeps
                    for backbone, method in wanted:
                        ident = site_index(backbone, method, view, timer)
                        hidden, own = ident.left_out(source, number, obs.photo_ids,
                                                     obs.linked, fetched_hashes(ident, pairs))
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
                        deep: list = []
                        results.append(embed_and_score(
                            ident, backbone, images, context, may_show, timer,
                            hidden=hidden if len(hidden) else None, n_specimens=yours.POOL,
                            vectors=vectors, include=counted_anyway, deep=deep))
                        scored.append((ident, vectors[0] if vectors else None, hidden))
                        frozen.append((deep, bool(len(own)), int(len(hidden)), this_used))
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
                        # Beside it: each shown photo's licence code and who took it, so the
                        # run's link card can credit it (linkcards.card_photos).
                        "photos_credit": [byobservation.shown_photo_credit(obs, i)
                                          for i in kept],
                        "excluded_reference": own_any,
                        "left_out_photos": left,
                }
                table = confused_table()
                answered = []
                for key, version, pooled, (ident, vecs, hidden) in zip(keys, versions, results,
                                                                       scored):
                    kept, result, spare = site_kept(pooled, vecs, hidden, used, about)
                    kept["release"] = model_card().get("version")
                    answer_cache.put(key, version, tagger_mark(kept), "site")
                    mine = yours.for_person(ident, result, spare, vecs, hidden, linked,
                                            photo_lookup, may_show)
                    confusions.annotate(mine, table)
                    mine["answered"] = answer_cache.held(key).about(version)
                    answered.append(mine)
                run = observation_run(request_person, keys,
                                      [answer_cache.held(k) for k in keys], wanted, view,
                                      timer, about)
                inat_kept(inat_key, (run or {}).get("id"), request_person)
                release = model_card().get("version")
                for (backbone, method), version, pooled, (deep, own_m, left_m, used_m) in zip(
                        wanted, versions, results, frozen):
                    lists, depth = early_answers.ranked_lists(deep, pooled.get("ranks"), 5)
                    freeze_answer(source=source, observation_id=number,
                                  model=f"{backbone}/{method}",
                                  path=early_answers.SITE_OBSERVATION, lists=lists, depth=depth,
                                  asked_by="site", release=release, answers_version=version,
                                  place_used=bool(used_m.get("place_from")),
                                  date_used=bool(used_m.get("date_from")),
                                  excluded_reference=own_m, left_out_photos=left_m)
                for (backbone, method), key, version in zip(wanted, keys, versions):
                    log_answer(identifylog.SITE_OBSERVATION, fresh=True, source=source,
                               observation_id=number, answer_key=key, answers_version=version,
                               model=f"{backbone}/{method}", release=release,
                               asked_by=identifylog.site_asker(request_person),
                               run_id=(run or {}).get("id"))
                inat = inat_view(inat_key, request_person, answered)
                return {"results": answered, "context_used": used, "observation": about,
                        "cached": False, "answered": answered[0]["answered"],
                        **({"run": run} if run else {}),
                        **({"inat": inat} if inat is not None else {}),
                        "timing": timer.block()}

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
            tagger = getattr(ident, "tagger", None)
            type_print = tagger.fingerprint() if tagger is not None else None
            added: list = []

            def embed():                                # the worker holds gpu_lock
                if image is not None:
                    vector, base, verdict, tag = embed_one(ident, backbone_model(s.backbone),
                                                           image)
                    added.append(session_store.Photo(
                        ref=ref, vector=vector, base=base, verdict=verdict,
                        gate_fingerprint=fingerprint if verdict else None, found=found,
                        photo_type=tag, type_fingerprint=type_print if tag else None))
                for p in s.photos:                      # judged by an older gate: again
                    base = p.base if p.base is not None else (
                        p.vector if ident.backbone == ident.base_name() else None)
                    if gate is not None and p.gate_fingerprint != fingerprint:
                        if base is not None:
                            p.verdict = gate.judge(np.asarray(base)[None])[0]
                            p.gate_fingerprint = fingerprint
                    if tagger is not None and p.type_fingerprint != type_print:
                        if base is not None:            # tagged by an older tagger: again
                            p.photo_type = tagger.tag(np.asarray(base)[None])[0]
                            p.type_fingerprint = type_print
                return np.stack([p.vector for p in [*s.photos, *added]])

            def finish(vecs, sims):
                photos = [*s.photos, *added]
                verdicts = [p.verdict for p in photos]
                tags = [p.photo_type for p in photos]
                context, _ = entered_context(s, photos)
                return ident.identify_vectors(
                    vecs, s.top_k, context=context, sims=sims,
                    gate=verdicts if gate is not None and all(verdicts) else None,
                    photo_lookup=photo_lookup, higher_ranks=True,
                    may_show=lambda info: view.may_show(info.license_class, info.owner_user_id),
                    types=tags if tagger is not None and all(tags) else None,
                    include=[i for i, p in enumerate(photos) if p.ref in s.include])
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
            log_answer(identifylog.APP_SESSION, fresh=True, model=f"{s.backbone}/{s.method}",
                       answers_version=served_version(s.backbone, s.method, ident),
                       release=release_now, asked_by=identifylog.credential_asker(cred),
                       client_ref=s.client_ref)
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
            if "inat_location" in payload:
                if not isinstance(payload["inat_location"], bool):
                    raise apiv1.V1Error(400, "bad_request", "inat_location is true or false")
                out["inat_location"] = payload["inat_location"]
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
                    hashes = [] if upload_meter.hashing() else None
                    [(image, found)] = v1_photos([photo], timer, hashes)
                    forward_app_photo(cred, image, s.client_ref, photo_ref,
                                      s.entered.get("latitude"), s.entered.get("longitude"),
                                      s.entered.get("inat_location") is True)
                    rescore(s, cred, timer, image=image, ref=photo_ref, found=found)
                finally:
                    in_flight.leave()
                upload_meter.record("session", 1, hashes)
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
            include = payload.get("include_photos")
            if include is not None and (not isinstance(include, list) or not all(
                    isinstance(r, str) for r in include)):
                raise apiv1.V1Error(400, "bad_request", "include_photos is a list of photo_refs")
            s = session_of(request, sid)
            with s.lock:
                if include is not None and photo_tagger is not None:
                    s.include = set(include)
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
        def site(request: Request, path: str):
            if path.startswith("api/"):
                raise HTTPException(404, "no such API route")
            target = (root / path).resolve()
            if path and target.is_file() and target.is_relative_to(root):
                return FileResponse(target)
            # The site ships favicon.png; a browser's own /favicon.ico gets it too.
            if path == "favicon.ico" and (root / "favicon.png").is_file():
                return FileResponse(root / "favicon.png", media_type="image/png")
            # The shell, with the link-preview card of the page (linkcards.py).
            card, status = page_card(request, "/" + path)
            shell = linkcards.with_tags((root / "index.html").read_text(encoding="utf-8"), card)
            # A run's link is found only by whoever was given it: never by a search engine.
            headers = dict(NOT_INDEXED) if run_path("/" + path) else {}
            # The Identify page asks for its example result while its scripts still load, and
            # opens the way to iNat's photos early (Steve, 2026-10-26: load it with the page).
            if path == "" and person(request) is not None and example_set():
                headers["Link"] = EXAMPLE_PRELOAD
            return HTMLResponse(shell, status_code=status, headers=headers or None)

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
    geocode.redact_access_log()       # place names typed into the map stay out of the log
    uvicorn.run(app, host=host, port=port, proxy_headers=True,
                forwarded_allow_ips=config.setting("MV_FORWARDED_ALLOW_IPS", "127.0.0.1"))
