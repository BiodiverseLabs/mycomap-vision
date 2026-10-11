"""Steve, 2026-10-11: the product says "MycoMap Vision", never the backbone underneath
(BioCLIP 2). No v1 answer or refusal a caller sees carries a backbone or method id."""

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, png
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

BACKBONE = "bioclip-2-ft-20261007-165400"           # named like the one the box serves
FORBIDDEN = ("bioclip", "dinov", "siglip", "eva02", "convnext", "open_clip", "timm",
             BACKBONE, "nearest+mean", "nearest")


def app(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, BACKBONE, "large", store.location, "local")
    embed_photos(conn, store, Const(BACKBONE), todo, root / BACKBONE, log=lambda s: None)
    conn.commit()
    return TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(allowed_backbones={BACKBONE}), background=False,
        server_key=SERVER_KEY, api_v1=True, taxon_counts=True,
        app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE), introspect=FakeOrg(),
        note=lambda s: None))


def values(x):
    """Every string value in a JSON body (field names like nearest_record_km aren't ids)."""
    if isinstance(x, dict):
        for v in x.values():
            yield from values(v)
    elif isinstance(x, list):
        for v in x:
            yield from values(v)
    elif isinstance(x, str):
        yield x.lower()


def leaks(body) -> list[str]:
    return sorted({w for w in FORBIDDEN for v in values(body) if w.lower() in v})


def test_no_v1_answer_names_the_backbone_or_method(conn, tmp_path):
    client = app(conn, tmp_path)
    res = client.post("/api/v1/identify", headers=bearer(app_token()),
                      files=[("photos", (f"p{i}.png", png(r), "image/png"))
                             for i, r in enumerate((247, 12))],
                      data={"latitude": "40.1", "longitude": "-83", "observed_on": "2026-10-11"})
    assert res.status_code == 200, res.text
    assert leaks(res.json()) == []
    assert res.json()["model"]["name"] == "MycoMap Vision"
    counts = client.get("/api/v1/taxon-counts", headers=bearer(SERVER_KEY))
    assert counts.status_code == 200 and leaks(counts.json()) == []


def test_no_v1_refusal_names_the_backbone(conn, tmp_path):
    client = app(conn, tmp_path)
    for res in (client.post("/api/v1/identify", headers=bearer(app_token()), data={}),
                client.post("/api/v1/identify", headers=bearer(app_token()),
                            files=[("photos", ("x.png", b"nope", "image/png"))]),
                client.get("/api/v1/nothing", headers=bearer(app_token()))):
        assert res.status_code >= 400 and leaks(res.json()) == [], res.text


@pytest.mark.parametrize("release, version", [
    ("20261008-015746-full-bioclip2-ft-20261007", "2026-10-08"),
    ("20261101-000000-anything", "2026-11-01"),
    ("bioclip2-latest", None),
    (None, None),
])
def test_the_public_version_is_the_release_date_only(release, version):
    model = apiv1.public_model(release, "20261008-012435-4ef7b0")
    assert model == {"name": "MycoMap Vision", "version": version,
                     "calibrated_on": "2026-10-08"}
    assert leaks(model) == []
