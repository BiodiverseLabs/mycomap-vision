"""On-demand images of a public dataset release (Steve, 2026-10-09: "let's make sure this
infrastructure exists on the site"). docs/dataset-release.md, section 8.

    GET /api/dataset/<release>                      what the release is: hashes, recipes, counts
    GET /api/dataset/<release>/recipes/<name>       the recipe document (JSON)
    GET /api/dataset/<release>/photos/<photo_key>?recipe=long500-q90
                                                    the derived image, made from our kept
                                                    original and checked against its hash

Only a public variant (`<id>-cc`) is served, and only its CC photos: an all-rights-reserved
photo is never in a public variant, and the photo's licence class is checked again here.
Each image carries its derived sha256, the original's sha256, the recipe, its licence and
its attribution in headers; it never changes, so it is cached for a year. A per-address
rate limit (MV_DATASET_RATE per minute, default 60) keeps the S3 reads in hand.

Where releases are: MV_DATASET_ROOT (default data/research-releases). Where the originals
are: MV_DATASET_PHOTO_STORE (a public variant names no store; the box's AWS user then needs
read access to the photos' `large/` prefix). Whether anyone may fetch without signing in:
MV_DATASET_PUBLIC (signin.py); otherwise it follows the site's sign-in rule.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from . import config
from . import dataset_release as dr
from .guards import RateLimiter

CACHE_ITEMS = 256
DEFAULT_RECIPE = "long500-q90"
YEAR = 365 * 24 * 3600


def _header_text(v: str | None) -> str:
    """Header-safe text: an attribution may hold any character."""
    return quote(v or "", safe=" ()@,.;:'/-_&+")


def add_routes(app: FastAPI, *, root: Path | None = None, store_location: str | None = None,
               store_reader=None, allow_draft: bool = False,
               rate: RateLimiter | None = None) -> None:
    root = Path(root or config.setting("MV_DATASET_ROOT") or dr.ROOT)
    store_location = store_location or config.setting("MV_DATASET_PHOTO_STORE")
    rate = rate or RateLimiter(int(config.setting("MV_DATASET_RATE") or 60), 60.0)
    lock = threading.Lock()
    opened: dict[str, dr.Release] = {}
    images: OrderedDict = OrderedDict()
    reader_box = {}

    def reader():
        if store_reader is not None:
            return store_reader
        if "r" not in reader_box:
            reader_box["r"] = dr.default_store_reader()
        return reader_box["r"]

    def public_release(rid: str) -> dr.Release:
        """A public variant, opened once (its files checked then), or 404."""
        if not rid.endswith("-cc"):
            raise HTTPException(404, "only a public (-cc) dataset release is served")
        with lock:
            rel = opened.get(rid)
            if rel is None:
                try:
                    rel = dr.load_release(rid, root, allow_draft=allow_draft)
                except dr.ReleaseError:
                    raise HTTPException(404, "no such public dataset release")
                if not rel.manifest.get("public_variant_of"):
                    rel.close()
                    raise HTTPException(404, "no such public dataset release")
                opened[rid] = rel
        return rel

    @app.get("/api/dataset/{rid}")
    def dataset(rid: str):
        rel = public_release(rid)
        m = rel.manifest
        return JSONResponse({
            "dataset_release": rid, "release_hash": rel.release_hash,
            "reference_hash": rel.reference_hash, "labels_hash": rel.labels_hash,
            "public_variant_of": m.get("public_variant_of"), "created": m.get("created"),
            "recipes": [r[0] for r in rel.db.execute("select name from recipes order by name")],
            "records": m["counts"]["records_included"], "photos": m["counts"]["photos_included"],
            "photo_url": f"/api/dataset/{rid}/photos/{{photo_key}}?recipe={DEFAULT_RECIPE}"})

    @app.get("/api/dataset/{rid}/recipes/{name}")
    def recipe(rid: str, name: str):
        rel = public_release(rid)
        row = rel.db.execute("select recipe_json, sha256 from recipes where name = ?",
                             (name,)).fetchone()
        if row is None:
            raise HTTPException(404, "no such recipe in this release")
        return JSONResponse({"recipe": name, "sha256": row[1], "document": row[0]},
                            headers={"Cache-Control": f"public, max-age={YEAR}, immutable"})

    @app.get("/api/dataset/{rid}/photos/{photo_key}")
    def photo(rid: str, photo_key: str, request: Request, recipe: str = DEFAULT_RECIPE):
        client = request.client.host if request.client else "unknown"
        wait = rate.check(client)
        if wait:
            raise HTTPException(429, f"too many images from this address; try again in "
                                     f"{int(wait) + 1} s", headers={"Retry-After": str(int(wait) + 1)})
        rel = public_release(rid)
        row = rel.db.execute("select * from photos where photo_key = ?", (photo_key,)).fetchone()
        # A public variant holds CC photos only; never trust that alone.
        if row is None or row["license_class"] not in dr.CC_CLASSES:
            raise HTTPException(404, "no CC photo with that key in this release")
        want = rel.db.execute("select derived_sha256 from derived where photo_key = ? and "
                              "recipe = ?", (photo_key, recipe)).fetchone()
        if want is None:
            raise HTTPException(404, "this release has no hash for that photo and recipe")
        etag = f'"{want[0]}"'
        headers = {
            "Cache-Control": f"public, max-age={YEAR}, immutable", "ETag": etag,
            "X-Derived-SHA256": want[0], "X-Original-SHA256": row["original_sha256"],
            "X-Recipe": recipe, "X-Dataset-Release": rid,
            "X-License": row["license_code"] or "", "X-Attribution": _header_text(row["attribution"]),
        }
        if row["source_url"]:
            headers["Link"] = f'<{row["source_url"]}>; rel="via"'
        ext = (row["original_path"] or "").rsplit(".", 1)[-1].lower()
        media = "image/jpeg" if recipe != "original" else \
            {"png": "image/png", "gif": "image/gif", "webp": "image/webp"}.get(ext, "image/jpeg")
        if request.headers.get("if-none-match") == etag:
            return Response(status_code=304, headers=headers)
        key = (rid, photo_key, recipe)
        with lock:
            body = images.get(key)
            if body is not None:
                images.move_to_end(key)
        if body is None:
            try:
                body = rel.derive(photo_key, recipe, reader(), store_location=store_location).body
            except dr.ReleaseError as e:
                raise HTTPException(502, f"could not make that image: {e}")
            with lock:
                images[key] = body
                while len(images) > CACHE_ITEMS:
                    images.popitem(last=False)
        return Response(body, media_type=media, headers=headers)
