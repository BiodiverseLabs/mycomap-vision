"""Nearest + species average is the identifier's default (Steve, 2026-10-09), and the
identification uses it without computing the photo similarities twice."""

import numpy as np
from test_api import app_with_model, open_limits, post_photos
from test_methods import clusters
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.evaluate import build_index
from mycomap_vision.guards import Limits
from mycomap_vision.methods import NearestAndMean


def test_the_blends_per_photo_scores_average_to_its_record_score():
    vecs, recs, _ = clusters([6] * 4, seed=5)
    index = build_index(recs)
    blend = NearestAndMean()
    blend.fit(vecs, index)
    query = vecs[[0, 7, 13]].astype(np.float32)
    per_photo = blend.per_photo_scores(blend.scorer.sims(query), query)
    assert per_photo.shape == (3, len(index.species))
    assert np.allclose(per_photo.mean(axis=0), blend.species_scores(query), atol=1e-6)
    for i in range(3):
        assert np.allclose(per_photo[i], blend.species_scores(query[i:i + 1]), atol=1e-6)


def test_identifying_with_the_blend_answers_as_if_scored_photo_by_photo(conn, tmp_path):
    from mycomap_vision.identify import Identifier
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    conn.commit()
    from mycomap_vision.identify import map_embeddings
    root = tmp_path / "emb" / "m1"
    ident = Identifier(conn, "m1", "nearest+mean", embeddings_root=root, photo_info=False)
    _, vecs = map_embeddings(conn, "m1", root)
    query = np.asarray(vecs[[0, 3, 5]], dtype=np.float32)
    shared = ident.identify_vectors(query)

    class Plain:                     # the same method without the shared-similarity path
        def __init__(self, m):
            self.m = m

        def species_scores(self, q):
            return self.m.species_scores(q)
    ident.model = Plain(ident.model)
    assert ident.identify_vectors(query) == shared


def test_the_default_is_nearest_plus_mean_where_offered_else_nearest():
    base = dict(rate=(1000, 60.0), max_in_flight=4, allowed_backbones={"m1"})
    assert Limits(**base).served_default() == "nearest+mean"
    assert Limits(**base, allowed_methods={"nearest"}).served_default() == "nearest"
    assert Limits(**base, allowed_methods={"nearest", "nearest+mean"},
                  default_method="species-mean").served_default() == "nearest"
    assert Limits(**base).served_default(lambda m: m != "nearest+mean") == "nearest"


def test_an_identification_naming_no_method_uses_the_default(conn, tmp_path):
    c = app_with_model(conn, tmp_path)
    assert c.get("/api/models").json()["default_method"] == "nearest+mean"
    res = post_photos(c, [247, 248])
    assert res.status_code == 200
    [r] = res.json()["results"]
    assert r["model"]["method"] == "nearest+mean"
    assert len(r["per_photo"]) == 2 and r["ranks"]["species"][0]["name"] == "A x"


def test_a_server_not_offering_the_blend_keeps_nearest(conn, tmp_path):
    c = app_with_model(conn, tmp_path, limits=open_limits(allowed_methods={"nearest"}))
    assert c.get("/api/models").json()["default_method"] == "nearest"
    [r] = post_photos(c, [247]).json()["results"]
    assert r["model"]["method"] == "nearest"
