"""On-demand images of a public dataset release (dataset_api.py): only a public variant,
only CC photos, each image with its hashes, licence and attribution, cached and
rate-limited; behind the site's sign-in unless MV_DATASET_PUBLIC opens it."""

import sys

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_dataset_release import build, world  # noqa: F401  (world is a fixture)
from test_models_and_scoreboard import Const
from test_signin import ISSUER, PUBLIC_KEY, SECRET

from mycomap_vision import dataset_release as dr
from mycomap_vision.api import create_app
from mycomap_vision.signin import SigninConfig


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)


def site(world, mode="off", dataset_public=False, allow_draft=True):
    signin = SigninConfig(mode=mode, origin="http://testserver", issuer=ISSUER,
                          public_key=PUBLIC_KEY, secret=SECRET,
                          dataset_public=dataset_public) if mode != "off" else SigninConfig()
    app = create_app(world["tmp"] / "api-manifest.sqlite", world["tmp"] / "emb",
                     backbone_loader=lambda n: Const(n), limits=open_limits(), signin=signin,
                     background=False, dataset_root=world["root"],
                     dataset_store=str(world["store"].root),
                     dataset_store_reader=dr.default_store_reader(),
                     dataset_allow_draft=allow_draft)
    return TestClient(app, follow_redirects=False)


@pytest.fixture
def released(world):  # noqa: F811
    m = build(world)
    pub = dr.public_variant(m["id"], world["root"], log=lambda *a: None)
    return world, m["id"], pub["id"]


def test_a_cc_photo_is_served_with_its_hashes_licence_and_attribution(released):
    world, base, pub = released
    client = site(world)
    r = client.get(f"/api/dataset/{pub}/photos/inat:11?recipe=long500-q90")
    assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
    assert dr.sha256_bytes(r.content) == r.headers["x-derived-sha256"]
    rel = dr.load_release(pub, world["root"], allow_draft=True)
    stored = rel.db.execute("select derived_sha256 from derived where photo_key = 'inat:11' "
                            "and recipe = 'long500-q90'").fetchone()[0]
    rel.close()
    assert r.headers["x-derived-sha256"] == stored
    assert r.headers["x-license"] == "cc-by" and "Alice" in r.headers["x-attribution"]
    assert "immutable" in r.headers["cache-control"] and r.headers["etag"] == f'"{stored}"'
    again = client.get(f"/api/dataset/{pub}/photos/inat:11",
                       headers={"If-None-Match": r.headers["etag"]})
    assert again.status_code == 304
    info = client.get(f"/api/dataset/{pub}").json()
    assert info["release_hash"] and "long500-q90" in info["recipes"]
    doc = client.get(f"/api/dataset/{pub}/recipes/long500-q90").json()
    assert dr.sha256_bytes(doc["document"].encode()) == doc["sha256"]


def test_an_arr_photo_is_never_served(released):
    world, base, pub = released
    client = site(world)
    for key in ("inat:12", "inat:31", "inat:111"):
        assert client.get(f"/api/dataset/{pub}/photos/{key}").status_code == 404


def test_an_arr_photo_slipped_into_a_public_variant_is_still_refused(released):
    import sqlite3
    world, base, pub = released
    folder = world["root"] / pub
    dr._writable(folder)
    c = sqlite3.connect(folder / dr.RELEASE_DB)
    c.execute("update photos set license_class = 'arr' where photo_key = 'inat:11'")
    c.commit()
    c.close()
    rel = dr.load_release(pub, world["root"], allow_draft=True, check_files=False)
    rel.close()
    import mycomap_vision.dataset_release as mod
    real = mod.load_release
    client = site(world)
    mod.load_release = lambda rid, root, allow_draft=False: real(rid, root, allow_draft=True,
                                                                 check_files=False)
    try:
        assert client.get(f"/api/dataset/{pub}/photos/inat:11").status_code == 404
    finally:
        mod.load_release = real


def test_only_a_public_variant_is_served(released, monkeypatch):
    world, base, pub = released
    opened = []
    real = dr.load_release
    monkeypatch.setattr(dr, "load_release", lambda rid, *a, **k: opened.append(rid) or real(
        rid, *a, **k))
    client = site(world)
    assert client.get(f"/api/dataset/{base}/photos/inat:11").status_code == 404
    assert client.get(f"/api/dataset/{base}").status_code == 404
    assert base not in opened            # a full release (coordinates) is never even opened
    assert client.get("/api/dataset/v9-cc/photos/inat:11").status_code == 404


def test_a_release_merely_named_cc_is_not_served(world):  # noqa: F811
    fake = build(world, name="dry-20261009-120000-cc")      # a full release, -cc in name only
    assert site(world).get(f"/api/dataset/{fake['id']}/photos/inat:11").status_code == 404


def test_a_draft_public_variant_is_not_served_unless_allowed(released):
    world, base, pub = released
    assert site(world, allow_draft=False).get(f"/api/dataset/{pub}").status_code == 404


def test_images_are_rate_limited_per_address(released, monkeypatch):
    world, base, pub = released
    monkeypatch.setenv("MV_DATASET_RATE", "2")
    client = site(world)
    codes = [client.get(f"/api/dataset/{pub}/photos/inat:11").status_code for _ in range(3)]
    assert codes == [200, 200, 429]


@pytest.mark.parametrize("dataset_public, expected", [(False, 401), (True, 200)])
def test_images_follow_the_sites_signin_unless_opened(released, dataset_public, expected):
    world, base, pub = released
    client = site(world, mode="all", dataset_public=dataset_public)
    assert client.get(f"/api/dataset/{pub}/photos/inat:11").status_code == expected
    # Opening the dataset images opens nothing else.
    assert client.get("/api/scoreboard").status_code == 401
