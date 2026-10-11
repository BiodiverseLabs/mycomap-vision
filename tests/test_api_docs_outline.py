"""The API's public outline (apidocs.py): one source for the Vision site's /api-docs page and
the tab on mycomap.org/api-docs. It must match what the server really answers, be open to
anyone, and name no backbone or method."""

import re

from fastapi.testclient import TestClient
from test_api import open_limits
from test_api_v1_identify import FakeOrg, SERVER_KEY
from test_api_v1_public_name import values
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, ISSUER, PUBLIC_KEY, SECRET

from mycomap_vision import apidocs, apiv1
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.signin import SigninConfig

BLOCKS = {"p", "list", "table", "code", "endpoint"}
# Backbone names and method ids. ("Nearest" alone is plain English here: "the nearest
# verified record", the place cue's nearest_record_km.)
IDS = ("bioclip", "dinov", "siglip", "eva02", "convnext", "open_clip", "timm", "-ft-",
       "nearest+", "species-mean", "nearest-mix", "linear+", "hybrid")


def app(conn, tmp_path, api_v1=True, taxon_counts=True, signin=None):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                      limits=open_limits(), background=False, server_key=SERVER_KEY,
                      api_v1=api_v1, taxon_counts=taxon_counts, signin=signin,
                      app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
                      note=lambda s: None)


def v1_routes(fastapi_app) -> set[tuple[str, str]]:
    """(method, path under /api/v1) of every v1 route the server registered."""
    out = set()
    for r in fastapi_app.routes:
        path = getattr(r, "path", "")
        if path.startswith("/api/v1/") and "{rest:path}" not in path:
            for m in getattr(r, "methods", ()) or ():
                if m != "HEAD":
                    out.add((m, path[len("/api/v1"):]))
    return out


def same_shape(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "{}", path)


def test_every_route_the_server_answers_is_in_the_outline_and_marked_available(conn, tmp_path):
    a = app(conn, tmp_path)
    doc = TestClient(a).get("/api/docs/v1.json").json()
    listed = {(e["method"], same_shape(e["path"])): e["status"] for e in apidocs.endpoints(doc)}
    served = {(m, same_shape(p)) for m, p in v1_routes(a)}
    assert served, "no v1 routes registered?"
    for route in served:
        assert listed.get(route) == "available", route
    # And nothing is called available that the server doesn't answer.
    assert {k for k, s in listed.items() if s == "available"} <= served


def test_a_part_turned_off_is_listed_as_planned(conn, tmp_path):
    a = app(conn, tmp_path, api_v1=False, taxon_counts=False)
    doc = TestClient(a).get("/api/docs/v1.json").json()
    assert {e["status"] for e in apidocs.endpoints(doc)} == {"planned"}


def test_the_outline_is_open_to_anyone_from_any_site(conn, tmp_path):
    signin = SigninConfig(mode="all", origin=HERE, issuer=ISSUER, public_key=PUBLIC_KEY,
                          secret=SECRET)
    client = TestClient(app(conn, tmp_path, signin=signin), follow_redirects=False)
    res = client.get("/api/docs/v1.json")
    assert res.status_code == 200
    assert res.headers["access-control-allow-origin"] == "*"
    assert "max-age" in res.headers["cache-control"]
    # The rest of the site still needs the sign-in.
    assert client.get("/api/stats").status_code == 401


def test_the_outline_names_no_backbone_or_method():
    for parts in ({}, {"identify": True, "taxon_counts": True}):
        doc = apidocs.outline(parts)
        assert not [w for w in IDS if any(w in v for v in values(doc))]


def test_the_outline_uses_only_the_blocks_both_pages_render():
    doc = apidocs.outline({"identify": True})
    assert doc["base_url"].startswith("https://") and doc["sections"]
    ids = [s["id"] for s in doc["sections"]]
    assert len(ids) == len(set(ids))
    for s in doc["sections"]:
        for b in s["blocks"]:
            assert b["type"] in BLOCKS, b
            if b["type"] == "endpoint":
                assert b["method"] in ("GET", "POST", "PATCH", "DELETE")
                assert b["path"].startswith("/") and b["status"] in apidocs.STATUSES
                for f in b.get("fields", []):
                    assert len(f) == 4
            if b["type"] == "table":
                assert all(len(r) == len(b["columns"]) for r in b["rows"])
