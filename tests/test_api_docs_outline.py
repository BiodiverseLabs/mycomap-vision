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


def app(conn, tmp_path, api_v1=True, taxon_counts=True, signin=None, sessions=True):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return create_app(tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
                      limits=open_limits(), background=False, server_key=SERVER_KEY,
                      api_v1=api_v1, taxon_counts=taxon_counts, signin=signin,
                      sessions=sessions,
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
    a = app(conn, tmp_path, api_v1=False, taxon_counts=False, sessions=False)
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


def answer_table(doc: dict) -> set[str]:
    """The fields "The answer" section describes (its first column, without backticks)."""
    sec = next(s for s in doc["sections"] if s["id"] == "answer")
    table = next(b for b in sec["blocks"] if b["type"] == "table")
    return {row[0].strip("`") for row in table["rows"]}


# Top-level answer fields the outline leaves to the example rather than the table.
SELF_EVIDENT = {"request_id", "photos", "reference", "confidence_note"}


# Fields of a candidate name that need no row of their own.
CANDIDATE_SELF_EVIDENT = {"name", "score"}


def test_every_field_of_a_real_answer_is_in_the_example_and_described(conn, tmp_path):
    from test_api_v1_identify import app_token, bearer, identify, v1_app
    from test_confusions import A_X, B_Y, write_table
    client = v1_app(conn, tmp_path)
    write_table(tmp_path, A_X, B_Y)          # "A x" prompts, so it carries confused_with
    body = identify(client, bearer(app_token()), reds=(247, 245), latitude="40.1",
                    longitude="-83.0", observed_on="2026-10-11").json()
    doc = apidocs.outline({"identify": True})
    described = answer_table(doc) | SELF_EVIDENT
    assert set(body) - set(apidocs.ANSWER_EXAMPLE) - SELF_EVIDENT == set(), "missing from the example"
    assert set(body) - described == set(), "missing from The answer table"
    # And every field a species or genus name can carry, confused_with included.
    fields = {k for rank in ("species", "genus") for c in body["ranks"][rank] for k in c}
    assert {"place", "confused_with"} <= fields, "the fixture shows both"
    assert fields - CANDIDATE_SELF_EVIDENT - described == set(), "a name's field is undescribed"
    for name in ("prior", "photo_only_top", "place", "confused_with"):
        assert name in described


def test_the_outline_sends_members_to_the_vision_keys_page():
    doc = apidocs.outline({})
    access = next(s for s in doc["sections"] if s["id"] == "access")
    text = " ".join(b.get("text", "") for b in access["blocks"])
    assert "(https://mycomap.org/vision-keys)" in text


def test_every_example_shows_the_live_model_card():
    """The examples show the model callers meet (GET /api/v1/models, v1 live 2026-10-14), in
    the shape the server answers, and no older model."""
    card = apiv1.public_model({"version": "x", "data": {"snapshot": "x", "records": 1,
                                                        "species": 1, "genera": 1}}, "20261013-1")
    assert set(apidocs.LIVE_MODEL) == set(card)
    assert set(apidocs.LIVE_MODEL["data"]) == set(card["data"])
    assert apidocs.LIVE_MODEL["version"] == "2026-10-13"
    assert apidocs.LIVE_MODEL["calibrated_on"] == "2026-10-13"
    assert apidocs.LIVE_REFERENCE == {"records": 169129, "species": 17854, "photos": 684135}
    doc = apidocs.outline({k: True for k in ("identify", "sessions", "by_observation",
                                             "taxon_counts", "usage", "models")})
    text = " ".join(values(doc))
    assert "2026-10-06" not in text and "2026-10-08" not in text, "an older model's card"
    models = next(b for b in apidocs.endpoints(doc) if b["path"] == "/models")
    assert '"confident_at": 0.7' in models["response"]
    assert '"runner_up_below": 0.2' in models["response"], "the species rule (2026-10-18)"
    assert apidocs.ANSWER_EXAMPLE["model"] == apidocs.LIVE_MODEL


def test_the_outline_makes_no_pre_v1_claim_about_calibration_or_the_gate():
    rows = {row[0].strip("`"): row[1] for row in
            next(b for s in apidocs.outline({})["sections"] if s["id"] == "answer"
                 for b in s["blocks"] if b["type"] == "table")["rows"]}
    assert "weekly" not in rows["confidence"]
    assert "dataset release v1" in rows["confidence"]
    assert "92" not in rows["confident"] and "%" not in rows["confident"]
    assert "calibrated on v1's validation split" in rows["confident"]
    assert "0.70" in rows["confident"]                       # genus, family, species outright
    # Species also at 0.50 with the next below 0.20 (Steve, 2026-10-18).
    assert "0.50" in rows["confident"] and "below 0.20" in rows["confident"]
    assert "2026-10-18" in rows["confident"] and "confident_rule" in rows["confident"]
    assert "below 0.50" in rows["gate"] and "0.70" not in rows["gate"]
