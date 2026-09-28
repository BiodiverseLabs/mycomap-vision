"""HTTP API for the frontend (and later mycomap.org).

Run: `.venv/Scripts/mv serve` (defaults to http://127.0.0.1:8010).
Read-only over the manifest; identification runs on the local GPU when there is one.
Public limits (request and image size, rate per IP, GPU queue, allowed backbones)
are in guards.py and set from the environment.
"""

from __future__ import annotations

import io
import sqlite3
import threading
import warnings
from pathlib import Path
from typing import Callable

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, ImageOps

from . import config, evaluate, models
from .embed import SCHEMA as EMBED_SCHEMA
from .embed import photos_per_second
from .guards import (MAX_FILE_BYTES, MAX_PHOTOS, MAX_REQUEST_BYTES, Gate, Limits, RateLimiter,
                     TooLarge, check_image_size)
from .identify import Identifier
from .prior import Context
NAME_BUCKETS = [(1, 1, "1"), (2, 2, "2"), (3, 5, "3-5"), (6, 30, "6-30"), (31, 10**9, "31+")]


def open_manifest(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.executescript(EMBED_SCHEMA)
    conn.executescript(evaluate.SCOREBOARD_SCHEMA)
    return conn


WEB_DIST = config.REPO_ROOT / "web" / "dist"


def read_photo(upload: UploadFile) -> Image.Image:
    """One uploaded photo, refusing oversized files and images before decoding them."""
    body = upload.file.read(MAX_FILE_BYTES + 1)
    if len(body) > MAX_FILE_BYTES:
        raise HTTPException(413, f"{upload.filename} is over {MAX_FILE_BYTES // 2**20} MB")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            img = Image.open(io.BytesIO(body))
        check_image_size(img)
        img = ImageOps.exif_transpose(img)
        return img.convert("RGB")
    except TooLarge as e:
        raise HTTPException(413, f"{upload.filename}: {e}")
    except Exception:
        raise HTTPException(400, f"{upload.filename} is not an image we can read")


def create_app(manifest_path: Path | None = None, embeddings_root: Path | None = None,
               backbone_loader: Callable[[str], object] = models.load_backbone,
               web_dist: Path | None = WEB_DIST, limits: Limits | None = None) -> FastAPI:
    app = FastAPI(title="MycoMap Vision", version="0.1")
    limits = limits or Limits.from_settings()
    rate = RateLimiter(*limits.rate)
    in_flight = Gate(limits.max_in_flight)

    @app.middleware("http")
    async def cap_request_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > MAX_REQUEST_BYTES:
            return JSONResponse({"detail": "request too large"}, status_code=413)
        return await call_next(request)

    conn = open_manifest(manifest_path or config.MANIFEST_PATH)
    db_lock = threading.Lock()
    gpu_lock = threading.Lock()
    identifiers: dict[tuple[str, str], tuple[tuple, Identifier]] = {}
    backbones: dict[str, object] = {}

    def q(sql: str, params: tuple = ()):
        with db_lock:
            return conn.execute(sql, params).fetchall()

    def embedded_counts() -> dict[str, int]:
        return dict(q("select backbone, count(*) from embeddings group by 1"))

    def identifier(backbone: str, method: str) -> Identifier:
        n = embedded_counts().get(backbone, 0)
        with db_lock:
            cal = evaluate.latest_calibration(conn, backbone, method)
        version = (n, cal["run_id"] if cal else None)
        cached = identifiers.get((backbone, method))
        if cached and cached[0] == version:
            return cached[1]
        root = embeddings_root / backbone if embeddings_root else None
        with db_lock:
            ident = Identifier(conn, backbone, method, root, calibration=cal)
        identifiers[(backbone, method)] = (version, ident)
        return ident

    @app.get("/api/health")
    def health():
        return {"ok": True, "code_version": config.code_version()}

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
        return {"backbones": out, "methods": list(evaluate.METHODS),
                "ready": [b["backbone"] for b in out if b["embedded_photos"]]}

    @app.get("/api/stats")
    def stats():
        one = lambda sql: q(sql)[0][0]  # noqa: E731
        name_counts = q("select count(*) from records where north_america = 1 "
                        "and label_conflict = 0 and coalesce(scientific_name, '') <> '' "
                        "group by scientific_name")
        buckets = {label: 0 for _, _, label in NAME_BUCKETS}
        for (n,) in name_counts:
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
            "names": len(name_counts),
            "names_by_records": buckets,
        }

    @app.get("/api/scoreboard")
    def scoreboard():
        with db_lock:
            return {"runs": evaluate.scoreboard(conn)}

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
        images = [read_photo(f) for f in photos]
        ready = ready_backbones()
        wanted = [m.strip() for m in models_.split(",") if m.strip()]
        if not wanted:
            if not ready:
                raise HTTPException(503, "no model has embeddings yet")
            wanted = [f"{ready[0]}/nearest"]
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
            with gpu_lock:
                ident = identifier(backbone, method)
                if backbone not in backbones:
                    backbones[backbone] = backbone_loader(backbone)
                results.append(ident.identify(backbones[backbone], images, context=context))
        return {"results": results}

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
    uvicorn.run(create_app(), host=host, port=port)
