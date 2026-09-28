import numpy as np
import pytest

from conftest import inat_obs

from mycomap_vision import evaluate, models
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.inat import save_batch
from mycomap_vision.photos import Result, save_result
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

OPEN = "inaturalist-open-data.s3.amazonaws.com"


def test_any_timm_or_open_clip_model_is_usable_by_spec_without_code():
    assert models.resolve_spec("timm:eva02_large_patch14_clip_336") == \
        "timm:eva02_large_patch14_clip_336"
    assert models.resolve_spec("bioclip-2") == "open_clip:hf-hub:imageomics/bioclip-2"
    with pytest.raises(ValueError, match="unknown backbone"):
        models.resolve_spec("resnet50")


def test_backbones_get_a_stable_safe_storage_name():
    assert models.storage_name("dinov2-l14") == "dinov2-l14"
    assert models.storage_name("timm:vit_large_patch14_dinov2.lvd142m") == "dinov2-l14"
    assert models.storage_name("open_clip:hf-hub:org/model@x") == "open_clip_hf-hub_org_model_x"


def unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


def test_species_mean_scores_against_each_species_average():
    vecs = np.stack([unit(1, 0), unit(0.6, 0.8), unit(0, 1), unit(0.8, 0.6)]).astype(np.float16)
    ref = [evaluate.Record("a", "A x", "A", "F", "2026-01-01", "u", [0]),
           evaluate.Record("b", "A x", "A", "F", "2026-01-01", "u", [1]),
           evaluate.Record("c", "B y", "B", "F", "2026-01-01", "u", [2])]
    index = evaluate.build_index(ref)
    m = evaluate.SpeciesMean()
    m.fit(vecs, index)
    scores = m.species_scores(vecs[[3]])
    assert evaluate.top_labels(scores, index.species, 1) == ["A x"]


class Const:
    """A stand-in backbone whose vector depends only on the photo's colour."""

    def __init__(self, name, twist=0.0):
        self.name, self.dim, self.twist = name, 2, twist

    def encode(self, images):
        out = []
        for im in images:
            r = np.asarray(im, dtype=np.float32)[..., 0].mean() / 255
            out.append([np.cos(r * 3 + self.twist), np.sin(r * 3 + self.twist)])
        return np.asarray(out)


REDS = [250, 240, 245, 10, 20, 15, 248, 12]


def seed_two_species(conn, tmp_path):
    import io
    from PIL import Image
    rows = []
    obs = []
    # Species A is red-ish, B blue-ish; 3 old records each, 1 new record each.
    for i, (sp, _red, when) in enumerate([
        ("A x", 250, "1/1/2026"), ("A x", 240, "1/2/2026"), ("A x", 245, "1/3/2026"),
        ("B y", 10, "1/1/2026"), ("B y", 20, "1/2/2026"), ("B y", 15, "1/3/2026"),
        ("A x", 248, "9/20/2026"), ("B y", 12, "9/21/2026")]):
        oid = str(100 + i)
        rows.append({"observation_id": oid, "scientific_name": sp, "genus": sp.split()[0],
                     "family": "F", "continent": "North America",
                     "validation_status_1": "yes", "validation_date_1": when})
        obs.append(inat_obs(100 + i, photos=[(0, 1000 + i, "cc0", OPEN)]))
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, [o["observation_id"] for o in rows], obs, "t")
    store = LocalStore(tmp_path / "store")
    for i in range(8):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (REDS[i], 0, 0)).save(buf, "PNG")
        rel = f"p/{1000 + i}.png"
        store.put(rel, buf.getvalue())
        save_result(conn, Result(1000 + i, "done", rel, 1, "h"), "large", "now", store.location)
    return store


def test_a_comparison_scores_every_model_on_the_same_records_and_saves_the_scoreboard(conn,
                                                                                       tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    for bb in (Const("m1"), Const("m2", twist=0.4)):
        todo = photos_to_embed(conn, bb.name, "large", store.location, "local")
        if bb.name == "m2":
            todo = todo[:-1]             # m2 is missing one photo: it must drop out for both
        embed_photos(conn, store, bb, todo, root / bb.name, log=lambda s: None)
    result = evaluate.compare(conn, ["m1", "m2"], ["nearest", "species-mean"], test_days=28,
                              embeddings_root=root, log=lambda s: None)
    assert result["shared_photos"] == 7
    assert result["test_records"] == 1            # the new B record lost its only photo
    board = evaluate.scoreboard(conn, result["comparison_id"])
    assert {(r["backbone"], r["method"]) for r in board} == {
        ("m1", "nearest"), ("m1", "species-mean"), ("m2", "nearest"), ("m2", "species-mean")}
    assert len({(r["record_set"], r["n_test"], r["n_reference"]) for r in board}) == 1
    assert all(r["species_top1"] == 1.0 for r in board)
    assert evaluate.run_report(conn, board[0]["id"])["backbone"] in ("m1", "m2")


def test_comparing_a_model_with_no_embeddings_says_how_to_make_them(conn, tmp_path):
    seed_two_species(conn, tmp_path)
    with pytest.raises(ValueError, match="mv embed --backbone nope"):
        evaluate.compare(conn, ["nope"], embeddings_root=tmp_path, log=lambda s: None)


def test_unknown_methods_are_refused(conn):
    with pytest.raises(ValueError, match="unknown method"):
        evaluate.compare(conn, ["m1"], ["magic"], log=lambda s: None)
