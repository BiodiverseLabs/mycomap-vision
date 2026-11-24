"""The nightly advance predictions answer with the default model (MV_DEFAULT_BACKBONE), not
with whichever model was preloaded first or sorts first, and are skipped, never answered by
another model, while a named default isn't served (Steve, 2026-11-24)."""

import time

from fastapi.testclient import TestClient
from test_api import open_limits
from test_models_and_scoreboard import Const, seed_two_species
from test_nightly import QUIET, RID, FakeOutside, seed_rows
from test_nightly_predictions import pending
from test_prospective import FakeFetcher

from conftest import inat_obs
from mycomap_vision import manifest, nightly
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed

V1, V2 = "m1", "a-v2"                  # v2's stand-in sorts first


def two_model_box(tmp_path):
    """A server box whose release holds both models' vectors."""
    root = tmp_path / "srv"
    rel = root / "releases" / RID
    rel.mkdir(parents=True)
    conn = manifest.connect(rel / "manifest.sqlite")
    store = seed_two_species(conn, tmp_path)
    for name in (V1, V2):
        todo = photos_to_embed(conn, name, "large", store.location, "local")
        embed_photos(conn, store, Const(name), todo, rel / "embeddings" / name, **QUIET)
    conn.commit()
    conn.close()
    (root / "current.txt").write_text(RID + "\n", encoding="utf-8")
    return root


def server(box, default, allowed=(V1, V2), preload=(f"{V2}/nearest", f"{V1}/nearest")):
    layer = nightly.prepare(box, **QUIET)
    fetcher = FakeFetcher({"400": inat_obs(400, photos=[(0, 4000, "cc0", "x")])})
    client = TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name),
        limits=open_limits(allowed_backbones=set(allowed), default_backbone=default),
        background=False, layer=layer, nightly_fetch=lambda: (seed_rows(), "g"),
        nightly_fetch_pending=lambda: ([pending(400)], "g"),
        nightly_steps=FakeOutside({300: 130}).steps(), photo_fetcher=lambda: fetcher,
        nightly_settings=nightly.Settings(predict=300), preload=list(preload),
        note=lambda s: None))
    for _ in range(400):
        if client.get("/api/health").json()["preload"]["state"] != "loading":
            break
        time.sleep(0.05)
    return client


def test_advance_predictions_use_the_default_model_not_the_first_preloaded(tmp_path):
    client = server(two_model_box(tmp_path), default=V1)
    assert client.get("/api/health").json()["preload"]["state"] == "ready"
    report = client.app.state.run_nightly()
    assert report["predictions"]["model"] == f"{V1}/nearest"


def test_unset_advance_predictions_use_the_first_preloaded_as_before(tmp_path):
    client = server(two_model_box(tmp_path), default=None)
    report = client.app.state.run_nightly()
    assert report["predictions"]["model"] == f"{V2}/nearest"


def test_advance_predictions_are_skipped_while_the_named_default_isnt_served(tmp_path):
    client = server(two_model_box(tmp_path), default="m0-missing",
                    allowed=(V1, V2, "m0-missing"))
    report = client.app.state.run_nightly()
    assert "skipped" in report["predictions"] and "model" not in report["predictions"]
    assert "no embeddings" in report["predictions"]["skipped"]
