"""The server box answers in seconds: it offers only what it has ready, never trains a
classifier itself, and building an index doesn't stall the rest of the site."""

import threading
import time

from fastapi.testclient import TestClient
from test_api import open_limits, post_photos
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import api, config
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed


def seeded(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return root


def client(tmp_path, root, **limits):
    return TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                 backbone_loader=lambda n: Const(n), limits=open_limits(**limits),
                                 background=False, preload=[], note=lambda s: None))


def test_only_the_offered_methods_are_listed_or_run(conn, tmp_path):
    c = client(tmp_path, seeded(conn, tmp_path), allowed_methods={"nearest"})
    assert c.get("/api/models").json()["methods"] == ["nearest"]
    assert post_photos(c, [247], "m1/nearest").status_code == 200
    r = post_photos(c, [247], "m1/species-mean")
    assert r.status_code == 400 and "not offered" in r.json()["detail"]


def test_a_server_that_must_not_train_refuses_an_untrained_method(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    c = client(tmp_path, seeded(conn, tmp_path), fit_on_demand=False)
    r = post_photos(c, [247], "m1/linear")
    assert r.status_code == 503 and "isn't ready on this server" in r.json()["detail"]
    assert not list((tmp_path / "data").rglob("*.npz"))          # nothing was trained
    assert post_photos(c, [247], "m1/nearest").status_code == 200  # untrained methods run


def test_saved_training_is_used_where_training_is_not_allowed(conn, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    root = seeded(conn, tmp_path)
    trainer = client(tmp_path, root)                      # e.g. the laptop or GPU instance
    assert post_photos(trainer, [247], "m1/linear").status_code == 200
    assert list((tmp_path / "data" / "models").glob("*.npz"))
    box = client(tmp_path, root, fit_on_demand=False)
    assert post_photos(box, [247], "m1/linear").status_code == 200


def test_other_pages_answer_while_an_index_is_being_built(conn, tmp_path, monkeypatch):
    real = api.Identifier
    building = threading.Event()

    def slow_identifier(*a, **kw):
        building.set()
        time.sleep(2.0)                                   # the box takes tens of seconds
        return real(*a, **kw)

    monkeypatch.setattr(api, "Identifier", slow_identifier)
    c = client(tmp_path, seeded(conn, tmp_path))
    done = {}
    t = threading.Thread(target=lambda: done.update(r=post_photos(c, [247], "m1/nearest")))
    t.start()
    assert building.wait(5)
    start = time.monotonic()
    assert c.get("/api/stats").status_code == 200
    assert c.get("/api/models").status_code == 200
    assert time.monotonic() - start < 1.0                 # not stuck behind the build
    t.join(10)
    assert done["r"].status_code == 200


def test_a_comparison_is_capped_so_it_cannot_outlast_the_proxy(conn, tmp_path):
    c = client(tmp_path, seeded(conn, tmp_path), max_models=2)
    assert c.get("/api/models").json()["max_models"] == 2
    assert post_photos(c, [247], "m1/nearest,m1/species-mean").status_code == 200
    r = post_photos(c, [247], "m1/nearest,m1/species-mean,m1/nearest+prior")
    assert r.status_code == 400 and "at most 2" in r.json()["detail"]


def test_the_model_limit_comes_from_settings(monkeypatch):
    from mycomap_vision.guards import Limits
    monkeypatch.setenv("MV_MAX_MODELS", "2")
    assert Limits.from_settings().max_models == 2
    monkeypatch.delenv("MV_MAX_MODELS")
    assert Limits.from_settings().max_models is None
