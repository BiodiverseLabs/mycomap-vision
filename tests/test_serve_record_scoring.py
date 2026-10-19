"""Serving arm (c), nearest+mean-records-balanced(+place) (Steve, 2026-10-18; E32): the served
index groups cross-post twins as one specimen by the groups file E32 passed under, the
default falls back when (c) can't answer, and the public v1 answer keeps its shape. (Leaving
a linked observation's twins out of its own answer is the link path's job.)"""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_api_v1_identify import FakeOrg, SERVER_KEY, app_token, bearer, identify
from test_models_and_scoreboard import Const, seed_two_species
from test_signin import HERE, PUBLIC_KEY

from mycomap_vision import apiv1, twins
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.guards import Limits
from mycomap_vision.identify import Identifier, map_embeddings
from mycomap_vision.methods import method_ready

C = "nearest+mean-records-balanced"
TWINS = {"100": "twin-100", "101": "twin-100"}        # two records of species A: one find


def embedded(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return root


def test_c_is_offered_and_b_and_d_stay_research_only():
    assert method_ready(C)
    assert not method_ready("nearest+mean-records") and not method_ready("nearest+mean-views")


def test_the_served_c_refuses_to_build_without_twin_groups(conn, tmp_path):
    root = embedded(conn, tmp_path)
    with pytest.raises(ValueError, match="cross-post twin"):
        Identifier(conn, "m1", C, embeddings_root=root / "m1", photo_info=False)


def test_the_served_c_scores_a_twin_pair_as_one_specimen_and_other_methods_do_not(
        conn, tmp_path):
    root = embedded(conn, tmp_path)
    c = Identifier(conn, "m1", C, embeddings_root=root / "m1", photo_info=False,
                   specimen_groups=TWINS)
    plain = Identifier(conn, "m1", "nearest+mean", embeddings_root=root / "m1",
                       photo_info=False, specimen_groups=TWINS)
    assert c.records == plain.records == 8
    assert len(c.index.rec_starts) == c.records - 1          # 100 and 101: one run
    assert len(plain.index.rec_starts) == plain.records      # photo scoring: untouched
    assert c.specimens_grouped == 2 and plain.specimens_grouped == 0
    _, vecs = map_embeddings(conn, "m1", root / "m1")
    q = np.asarray(vecs[[0]], dtype=np.float32)
    assert np.allclose(c.model.species_scores(q),
                       c.model.per_photo_scores(c.model.photo_sims(q), q).mean(axis=0))


def test_c_with_a_whole_twin_pair_left_out_answers_as_if_neither_record_were_there(
        conn, tmp_path):
    root = embedded(conn, tmp_path)
    ident = Identifier(conn, "m1", C, embeddings_root=root / "m1", photo_info=False,
                       specimen_groups=TWINS)
    _, vecs = map_embeddings(conn, "m1", root / "m1")
    q = np.asarray(vecs[[1]], dtype=np.float32)
    recs = ident.col_record
    hidden = np.array([c for c in range(len(recs)) if str(recs.obs[recs.rec[c]])
                       in ("100", "101")], dtype=np.int64)
    sims = ident.nearest.photo_sims(q)
    got = ident.model.leave_out(ident.model.per_photo_scores(sims, q), sims, q, hidden)
    conn.execute("update records set label_conflict = 1 where observation_id in ('100','101')")
    conn.commit()
    refit = Identifier(conn, "m1", C, embeddings_root=root / "m1", photo_info=False,
                       specimen_groups=TWINS)
    want = refit.model.per_photo_scores(refit.nearest.photo_sims(q), q)
    for s, name in enumerate(ident.index.species):
        assert np.allclose(got[:, s], want[:, refit.index.species.index(name)], atol=1e-3)


def test_the_default_falls_back_down_the_chain_when_c_cannot_answer():
    base = dict(rate=(1000, 60.0), max_in_flight=4, allowed_backbones={"m1"})
    served = Limits(**base, default_method=C + "+place")
    assert served.served_default() == C + "+place"
    no_store = lambda m: not m.endswith("+place")              # noqa: E731
    assert served.served_default(no_store) == C
    no_twins = lambda m: not twins.is_grouped(m)               # noqa: E731
    assert served.served_default(no_twins) == "nearest+mean+place"
    neither = lambda m: no_store(m) and no_twins(m)            # noqa: E731
    assert served.served_default(neither) == "nearest+mean"
    allowed = Limits(**base, default_method=C + "+place",
                     allowed_methods={"nearest", "nearest+mean"})
    assert allowed.served_default() == "nearest+mean"
    from mycomap_vision.evaluate import METHODS
    known = lambda m: m in METHODS and method_ready(m)          # noqa: E731  (the server's)
    assert Limits(**base, default_method="nearest+mean-typo").served_default(known) ==         "nearest"


def app(tmp_path, root, monkeypatch, default_method, groups_file=None):
    """The server on the seeded manifest (tmp_path/manifest.sqlite) and its embeddings."""
    if groups_file:
        monkeypatch.setenv("MV_TWIN_GROUPS", str(groups_file))
    else:
        monkeypatch.delenv("MV_TWIN_GROUPS", raising=False)
    return TestClient(create_app(
        tmp_path / "manifest.sqlite", root, backbone_loader=lambda n: Const(n),
        limits=open_limits(default_method=default_method), background=False,
        server_key=SERVER_KEY, api_v1=True, app_tokens=apiv1.AppTokens(PUBLIC_KEY, HERE),
        introspect=FakeOrg(), note=lambda s: None))


def test_the_server_serves_c_when_its_twin_groups_are_there_else_falls_back(
        conn, tmp_path, monkeypatch):
    root = embedded(conn, tmp_path)
    without = app(tmp_path, root, monkeypatch, C)
    assert without.get("/api/models").json()["default_method"] == "nearest+mean"
    assert C not in without.get("/api/models").json()["methods"]
    groups = tmp_path / "twins.tsv"
    twins.write(TWINS, groups)
    body = app(tmp_path, root, monkeypatch, C, groups).get("/api/models").json()
    assert body["default_method"] == C and C in body["methods"]


def shape(x):
    """The answer's shape: every key and the type of every leaf; lists by their items."""
    if isinstance(x, dict):
        return {k: shape(v) for k, v in sorted(x.items())}
    if isinstance(x, list):
        return [shape(v) for v in x[:1]]
    return type(x).__name__ if x is not None else "null"


def test_the_v1_answer_under_c_has_the_same_shape_and_names_no_method(
        conn, tmp_path, monkeypatch):
    root = embedded(conn, tmp_path)
    groups = tmp_path / "twins.tsv"
    twins.write(TWINS, groups)
    answers = []
    for default, file in (("nearest+mean", None), (C, groups)):
        res = identify(app(tmp_path, root, monkeypatch, default, file),
                       bearer(app_token()), reds=(247, 245))
        assert res.status_code == 200, res.text
        body = res.json()
        for k in ("request_id", "timing", "usage"):
            body.pop(k, None)
        answers.append(body)
    old, new = answers
    assert "model" in new and new["model"] == old["model"]
    assert shape(new) == shape(old)
    text = json.dumps(new)
    for internal in ("records-balanced", "nearest", "twin-"):
        assert internal not in text, internal
