"""Loading the model at startup, so the first identification doesn't wait for it."""

import time

from fastapi.testclient import TestClient
from test_api import open_limits, post_photos
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision.api import create_app, preload_specs
from mycomap_vision.embed import embed_photos, photos_to_embed


class CountingLoader:
    def __init__(self):
        self.loaded = []

    def __call__(self, name):
        self.loaded.append(name)
        return Const(name)


def app(conn, tmp_path, preload):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    loader = CountingLoader()
    client = TestClient(create_app(tmp_path / "manifest.sqlite", root, backbone_loader=loader,
                                   limits=open_limits(), background=False, preload=preload,
                                   note=lambda s: None))
    return client, loader


def settled(client, timeout=20.0) -> dict:
    end = time.monotonic() + timeout
    while True:
        state = client.get("/api/health").json()["preload"]
        if state["state"] not in ("loading",) or time.monotonic() > end:
            return state
        time.sleep(0.05)


def test_the_default_model_is_loaded_at_startup_and_reused_by_identifications(conn, tmp_path):
    client, loader = app(conn, tmp_path, ["default"])
    state = settled(client)
    assert state["state"] == "ready" and state["models"] == ["m1/nearest"]
    assert loader.loaded == ["m1"]
    assert post_photos(client, [247], "").status_code == 200
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    assert loader.loaded == ["m1"]                       # never loaded a second time


def test_nothing_is_loaded_at_startup_unless_asked(conn, tmp_path):
    client, loader = app(conn, tmp_path, [])
    assert client.get("/api/health").json()["preload"]["state"] == "off"
    assert loader.loaded == []
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    assert loader.loaded == ["m1"]                       # on demand, as before


def test_a_preload_that_fails_leaves_the_server_answering(conn, tmp_path):
    client, loader = app(conn, tmp_path, ["not-offered/nearest"])
    state = settled(client)
    assert state["state"] == "failed" and "not-offered" in state["error"]
    assert loader.loaded == []
    assert post_photos(client, [247], "m1/nearest").status_code == 200


def test_the_preload_setting_reads_a_list_or_off():
    assert preload_specs(None) == [] and preload_specs("") == [] and preload_specs("off") == []
    assert preload_specs("default") == ["default"]
    assert preload_specs(" bioclip-2/nearest, m2 ") == ["bioclip-2/nearest", "m2"]
