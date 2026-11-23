"""The fast int8 search (fastsearch.py, MV_SEARCH_INT8): the answers that matter score as
before, and the process never holds two packed matrices (Steve: keep memory low)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_api import open_limits, post_photos
from test_models_and_scoreboard import Const, seed_two_species

from test_public_health import HEALTH_KEY, full_health
from mycomap_vision import fastsearch
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.quantize import quantize_int8
from mycomap_vision.serving import MappedVectors

pytest.importorskip("torch")
try:
    fastsearch.engine()
except RuntimeError:
    pytest.skip("no fbgemm/x86 quantized engine here", allow_module_level=True)


def vectors(n=3000, dim=64, seed=0):
    """Clustered unit vectors, as photos of a few hundred species are."""
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((300, dim))
    v = centres[rng.integers(0, 300, n)] + 0.35 * rng.standard_normal((n, dim))
    return (v / np.linalg.norm(v, axis=1, keepdims=True)).astype(np.float32)


def mixed(v):
    """An int8 release shard and a nightly float16 shard, as the box can hold."""
    half = len(v) // 2
    codes, scale = quantize_int8(v[:half])
    return MappedVectors([codes, v[half:].astype(np.float16)], [scale, None])


def test_the_best_columns_score_exactly_and_the_rest_closely():
    v = vectors()
    mv = mixed(v)
    cols = np.random.default_rng(1).permutation(len(v))           # index order
    sel = mv.select(cols)
    q = vectors(3, seed=9)
    exact = sel.sims(q)
    sel.use_int8(refine=200, slot=fastsearch.Slot())
    fast = sel.sims(q)
    assert sel.packed.engine in ("fbgemm", "x86", "onednn")
    for i in range(len(q)):
        top = np.argsort(-exact[i])[:50]
        assert np.allclose(fast[i, top], exact[i, top], atol=1e-5)  # refined: exact
        assert set(np.argsort(-fast[i])[:50]) == set(top)           # same best columns
    assert np.abs(fast - exact).max() < 0.02                         # the rest: close


def test_never_two_packed_matrices_at_once(monkeypatch):
    slot = fastsearch.Slot()
    v = vectors(800, 32)
    old, new = mixed(v).select(np.arange(800)), mixed(v).select(np.arange(800)[::-1])
    old.use_int8(refine=10, slot=slot)
    assert old.packed is not None and slot.owner is old
    seen = []
    real = fastsearch.PackedInt8.build

    def build(*a, **kw):
        seen.append(old.packed)                  # what the serving index holds meanwhile
        return real(*a, **kw)
    monkeypatch.setattr(fastsearch.PackedInt8, "build", build)
    new.use_int8(refine=10, slot=slot)
    assert seen == [None]                        # dropped BEFORE the new one was packed
    assert old.packed is None and new.packed is not None and slot.owner is new
    assert slot.peak_holders == 1
    # The dropped index still answers, through the mapped search.
    assert old.sims(v[:2]).shape == (2, 800)
    state = slot.state()
    assert state["mode"] in ("fbgemm", "x86", "onednn") and state["packed_rows"] == 800
    assert state["packed_bytes"] == 800 * (32 + 4)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    """The box has no GPU: score on the CPU here too (a GPU keeps vectors on the card)."""
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def seeded(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return root


def server(tmp_path, root):
    return TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                 backbone_loader=lambda n: Const(n), limits=open_limits(),
                                 background=False, fungus_gate=None, note=lambda s: None,
                                 server_key=HEALTH_KEY))


def app(conn, tmp_path):
    return server(tmp_path, seeded(conn, tmp_path))


def test_identify_answers_the_same_with_the_int8_search(conn, tmp_path, monkeypatch):
    root = seeded(conn, tmp_path)
    plain = post_photos(server(tmp_path, root), [247, 12], "m1/nearest").json()
    monkeypatch.setenv("MV_SEARCH_INT8", "1")
    client = server(tmp_path, root)
    fast = post_photos(client, [247, 12], "m1/nearest").json()
    for key in ("ranks", "specimens", "per_photo"):
        assert fast["results"][0][key] == plain["results"][0][key], key
    assert full_health(client)["search"]["mode"] in ("fbgemm", "x86", "onednn")


def test_without_the_engine_the_mapped_search_keeps_answering(conn, tmp_path, monkeypatch):
    monkeypatch.setenv("MV_SEARCH_INT8", "1")
    monkeypatch.setattr(fastsearch, "engine", lambda: (_ for _ in ()).throw(
        RuntimeError("no engine")))
    monkeypatch.setattr(fastsearch, "SLOT", fastsearch.Slot())
    client = app(conn, tmp_path)
    assert post_photos(client, [247], "m1/nearest").status_code == 200
    search = full_health(client)["search"]
    assert search["mode"] == "float" and "no engine" in search["last_error"]


def test_with_int8_on_the_search_stays_on_the_cpu_even_beside_a_gpu(monkeypatch):
    # The laptop embeds on its GPU; the int8 search must still run over the mapped vectors.
    import torch
    from mycomap_vision.methods import Scorer
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    sel = mixed(vectors(200, 16)).select(np.arange(200))
    monkeypatch.setenv("MV_SEARCH_INT8", "1")
    assert Scorer(sel).ref is sel


# --- one packed matrix per served backbone (v1 and v2 side by side, 2026-11-23) ----------

def test_two_served_backbones_each_keep_their_own_packed_matrix():
    slot = fastsearch.Slot(limit=2)
    v = vectors(600, 32)
    one, two = mixed(v).select(np.arange(600)), mixed(v).select(np.arange(500))
    one.use_int8(refine=10, slot=slot, key="v1")
    two.use_int8(refine=10, slot=slot, key="v2")
    assert one.packed is not None and two.packed is not None      # both search fast
    assert slot.peak_holders == 2
    state = slot.state()
    assert state["backbones"]["v1"] == {"engine": one.packed.engine, "rows": 600,
                                        "bytes": 600 * 36, "last_error": None}
    assert state["backbones"]["v2"]["rows"] == 500
    assert state["packed_bytes"] == 600 * 36 + 500 * 36 and state["packed_rows"] == 1100


def test_a_rebuild_never_holds_two_matrices_for_one_backbone(monkeypatch):
    slot = fastsearch.Slot(limit=2)
    v = vectors(600, 32)
    old, other = mixed(v).select(np.arange(600)), mixed(v).select(np.arange(400))
    old.use_int8(refine=10, slot=slot, key="v1")
    other.use_int8(refine=10, slot=slot, key="v2")
    kept = other.packed
    new = mixed(v).select(np.arange(600)[::-1])
    seen = []
    real = fastsearch.PackedInt8.build

    def build(*a, **kw):
        seen.append((old.packed, other.packed))   # what is held while v1's new matrix packs
        return real(*a, **kw)
    monkeypatch.setattr(fastsearch.PackedInt8, "build", build)
    new.use_int8(refine=10, slot=slot, key="v1")
    assert seen == [(None, kept)]                  # v1's old one dropped first; v2's untouched
    assert old.packed is None and new.packed is not None and other.packed is kept
    assert slot.peak_by_key == {"v1": 1, "v2": 1} and slot.peak_holders == 2
    assert old.sims(v[:2]).shape == (2, 600)       # the old index answers, mapped


def test_a_backbone_past_the_limit_searches_mapped_and_says_why():
    slot = fastsearch.Slot(limit=2)
    v = vectors(300, 16)
    held = [mixed(v).select(np.arange(300)) for _ in range(3)]
    held[0].use_int8(refine=10, slot=slot, key="a")
    held[1].use_int8(refine=10, slot=slot, key="b")
    with pytest.raises(fastsearch.Full):
        held[2].use_int8(refine=10, slot=slot, key="c")
    assert held[2].packed is None and held[0].packed is not None and held[1].packed is not None
    assert held[2].sims(v[:1]).shape == (1, 300)


def test_with_v1_and_v2_served_both_search_int8_and_health_says_so_per_backbone(
        conn, tmp_path, monkeypatch):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    for name in ("m1", "m2"):
        todo = photos_to_embed(conn, name, "large", store.location, "local")
        embed_photos(conn, store, Const(name), todo, root / name, log=lambda s: None)
    conn.commit()
    monkeypatch.setenv("MV_SEARCH_INT8", "1")
    monkeypatch.setattr(fastsearch, "SLOT", fastsearch.Slot())
    client = TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                   backbone_loader=lambda n: Const(n),
                                   limits=open_limits(allowed_backbones={"m1", "m2"},
                                                      max_models=2),
                                   background=False, fungus_gate=None, note=lambda s: None,
                                   server_key=HEALTH_KEY))
    for name in ("m1", "m2"):
        assert post_photos(client, [247], f"{name}/nearest").status_code == 200
    search = full_health(client)["search"]
    assert sorted(search["backbones"]) == ["m1", "m2"]
    for state in search["backbones"].values():
        assert state["engine"] in ("fbgemm", "x86", "onednn") and state["rows"] == 8
    assert search["packed_bytes"] == sum(s["bytes"] for s in search["backbones"].values())
    assert fastsearch.SLOT.limit == 2 and fastsearch.SLOT.peak_holders == 2
