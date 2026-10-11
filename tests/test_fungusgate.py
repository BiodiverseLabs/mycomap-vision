"""The not-a-fungus gate (fungusgate.py): a photo that isn't a fungus gets no species
answer and no stated chance, and doesn't sink the photos of a find that are."""

import io
import json
import types

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from test_api import open_limits
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import fungusgate, release
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.fungusgate import FUNGUS, NOT_FUNGUS, UNSURE, FungusGate, Thresholds

# Base-model vectors: (1, 0) is a fungus, (0, 1) a bird.
TEST_GATE = dict(classes=np.array(["mushroom", "bird"]), sides=np.array(["fungus", "other"]),
                 vectors=np.array([[1.0, 0.0], [0.0, 1.0]]), scale=10.0,
                 thresholds=Thresholds(not_fungus=0.2, unsure=0.5))
DUCK_GREEN, UNSURE_GREEN = 255, 135


def gate():
    return FungusGate(**TEST_GATE)


def base_vector(green):
    g = green / 255
    return [1 - g, g]


class Tuned(Const):
    """A fine-tuned stand-in: its own vector follows the photo's red, and the base
    model's vector (what the gate reads) follows its green."""
    base_name = "bioclip-2"

    def encode_with_base(self, images):
        base = [base_vector(float(np.asarray(im, dtype=np.float32)[..., 1].mean()))
                for im in images]
        return self.encode(images), np.asarray(base)


def photo(red, green=0):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (red, green, 0)).save(buf, "PNG")
    return buf.getvalue()


def seeded(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return root


def app(tmp_path, root, loader=Tuned, with_gate=True):
    return TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                 backbone_loader=lambda name: loader(name),
                                 limits=open_limits(), background=False,
                                 fungus_gate=gate() if with_gate else None))


def client_for(conn, tmp_path, loader=Tuned, with_gate=True):
    return app(tmp_path, seeded(conn, tmp_path), loader, with_gate)


def identify(client, photos):
    files = [("photos", (f"p{i}.png", body, "image/png")) for i, body in enumerate(photos)]
    res = client.post("/api/identify", files=files, data={"models": "m1/nearest"})
    assert res.status_code == 200, res.text
    return res.json()["results"][0]


def stated_chances(obj):
    """Every confidence value anywhere in a result that isn't None."""
    if isinstance(obj, dict):
        found = [v for k, v in obj.items() if k == "confidence" and v is not None]
        return found + [c for k, v in obj.items() if k != "gate" for c in stated_chances(v)]
    if isinstance(obj, list):
        return [c for v in obj for c in stated_chances(v)]
    return []


# --- the API ---------------------------------------------------------------------------

def test_a_non_fungus_photo_never_gets_a_calibrated_species_chance(conn, tmp_path):
    r = identify(client_for(conn, tmp_path), [photo(247, DUCK_GREEN)])
    assert r["gate"]["verdict"] == NOT_FUNGUS
    assert r["gate"]["message"] == "This doesn't look like a fungus."
    assert all(r["ranks"][rank] == [] for rank in ("species", "genus", "family"))
    assert r["likely"] == {} and r["specimens"] == [] and r["species_url"] is None
    assert stated_chances(r) == []
    assert r["hints"][0].startswith("This doesn't look like a fungus")


def test_a_duck_photo_among_mushroom_photos_does_not_sink_the_find(conn, tmp_path):
    client = client_for(conn, tmp_path)
    alone = identify(client, [photo(247)])
    # The duck photo is blue-ish to the identifier: counted, it would pull toward B y.
    mixed = identify(client, [photo(247), photo(12, DUCK_GREEN)])
    assert mixed["gate"]["verdict"] == FUNGUS
    assert mixed["gate"]["left_out"] == [1]
    assert mixed["ranks"] == alone["ranks"]
    assert mixed["ranks"]["species"][0]["confidence"] is not None
    assert [p["photo"] for p in mixed["per_photo"]] == [0, 1]
    assert mixed["per_photo"][1]["gate"]["verdict"] == NOT_FUNGUS
    assert mixed["per_photo"][1]["ranks"]["species"] == []
    assert mixed["per_photo"][0]["ranks"]["species"][0]["name"] == "A x"
    assert {s["matched_query_photo"] for s in mixed["specimens"]} == {0}
    assert "Photo 2 doesn't look like a fungus, so it was left out of the answer." in mixed["hints"]


def test_photos_left_out_keep_their_place_in_a_longer_find(conn, tmp_path):
    client = client_for(conn, tmp_path)
    r = identify(client, [photo(12, DUCK_GREEN), photo(247), photo(12, DUCK_GREEN), photo(240)])
    assert r["gate"]["left_out"] == [0, 2]
    assert [p["photo"] for p in r["per_photo"]] == [0, 1, 2, 3]
    assert [p["gate"]["verdict"] for p in r["per_photo"]] == [NOT_FUNGUS, FUNGUS, NOT_FUNGUS, FUNGUS]
    assert all(s["matched_query_photo"] in (1, 3) for s in r["specimens"])
    assert all(s["matched_query_photo"] == p["photo"] for p in r["per_photo"]
               for s in p["specimens"])
    assert "Photos 1 and 3 don't look like a fungus, so they were left out of the answer." in r["hints"]


def test_a_find_the_gate_is_unsure_of_shows_look_alikes_but_no_chance(conn, tmp_path):
    p = float(gate().p_fungus(np.array([base_vector(UNSURE_GREEN)]))[0])
    assert 0.2 <= p < 0.5
    r = identify(client_for(conn, tmp_path), [photo(247, UNSURE_GREEN)])
    assert r["gate"]["verdict"] == UNSURE
    assert r["gate"]["message"] == "Not sure this is a fungus."
    assert r["ranks"]["species"][0]["name"] == "A x"          # the names stay
    assert stated_chances(r) == [] and r["likely"] == {}
    assert r["confidence_withheld"] is True
    assert r["hints"][0].startswith("Not sure this is a fungus")


def test_fungus_photos_pass_the_gate_with_their_answer_unchanged(conn, tmp_path):
    root = seeded(conn, tmp_path)
    gated = identify(app(tmp_path, root), [photo(247), photo(240)])
    ungated = identify(app(tmp_path, root, with_gate=False), [photo(247), photo(240)])
    assert gated["gate"]["verdict"] == FUNGUS and gated["gate"]["left_out"] == []
    assert ungated["gate"] is None
    for key in ("ranks", "likely", "specimens", "hints"):
        assert gated[key] == ungated[key]


def test_the_gate_only_runs_on_a_backbone_whose_base_vectors_it_can_read(conn, tmp_path):
    r = identify(client_for(conn, tmp_path, loader=Const), [photo(247, DUCK_GREEN)])
    assert r["gate"] is None                   # m1 isn't BioCLIP 2: answered as before
    assert r["ranks"]["species"][0]["confidence"] is not None


def test_mv_fungus_gate_0_turns_the_saved_gate_off(conn, tmp_path, monkeypatch):
    models_dir = tmp_path / "data" / "models"
    gate().save(models_dir)
    monkeypatch.setattr(fungusgate.config, "DATA_DIR", tmp_path / "data")
    root = seeded(conn, tmp_path)

    def default_app():                         # create_app's default: the saved gate
        return TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                     backbone_loader=lambda name: Tuned(name),
                                     limits=open_limits(), background=False))
    assert identify(default_app(), [photo(247, DUCK_GREEN)])["gate"]["verdict"] == NOT_FUNGUS
    monkeypatch.setenv("MV_FUNGUS_GATE", "0")
    assert identify(default_app(), [photo(247, DUCK_GREEN)])["gate"] is None


# --- the gate itself -------------------------------------------------------------------

def test_p_fungus_is_the_softmax_mass_on_the_fungus_classes():
    g = gate()
    p = g.p_fungus(np.array([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))
    assert p[0] == pytest.approx(1 / (1 + np.exp(-10)))
    assert p[1] == pytest.approx(1 / (1 + np.exp(10)))
    assert p[2] == pytest.approx(0.5)          # the vector's length doesn't matter


def test_a_class_with_several_prompts_counts_once():
    g = FungusGate(np.array(["mushroom", "bird", "bird", "bird"]),
                   np.array(["fungus", "other", "other", "other"]),
                   np.array([[1.0, 0], [0, 1.0], [0, 1.0], [0, 1.0]]), 10.0,
                   Thresholds(0.2, 0.5))
    assert g.p_fungus(np.array([[1.0, 1.0]]))[0] == pytest.approx(0.5)


def test_verdicts_follow_the_thresholds():
    t = Thresholds(not_fungus=0.2, unsure=0.5)
    assert [t.verdict(p) for p in (0.0, 0.19, 0.2, 0.49, 0.5, 1.0)] == [
        NOT_FUNGUS, NOT_FUNGUS, UNSURE, UNSURE, FUNGUS, FUNGUS]


def test_a_find_is_a_fungus_when_any_photo_is():
    v = fungusgate.observation_verdict
    assert v([NOT_FUNGUS, FUNGUS]) == FUNGUS
    assert v([NOT_FUNGUS, UNSURE]) == UNSURE
    assert v([NOT_FUNGUS, NOT_FUNGUS]) == NOT_FUNGUS


def test_a_saved_gate_loads_back_the_same(tmp_path):
    gate().save(tmp_path)
    back = FungusGate.load(tmp_path)
    assert back.thresholds == TEST_GATE["thresholds"] and back.scale == 10.0
    v = np.array([[0.3, 0.7], [0.9, 0.1]])
    assert np.allclose(back.p_fungus(v), gate().p_fungus(v))
    assert FungusGate.load(tmp_path / "none") is None


def test_every_prompt_class_is_on_one_side_and_fungi_include_lichens():
    classes, sides, texts = fungusgate.prompts()
    side_of = {}
    for c, s in zip(classes, sides):
        assert side_of.setdefault(c, s) == s
    assert side_of["lichen"] == "fungus" and side_of["slime"] == "other"
    assert side_of["bird"] == "other" and side_of["agaric"] == "fungus"
    assert "a photo of a Fungi" not in " ".join(texts)


def test_auroc_and_rates_at_a_threshold():
    f, o = np.array([0.9, 0.8, 0.7, 0.2]), np.array([0.1, 0.3, 0.75])
    assert fungusgate.auroc(f, o) == pytest.approx(9 / 12)   # 9 of 12 pairs in order
    assert fungusgate.auroc(np.array([0.5]), np.array([0.5])) == 0.5
    t = fungusgate.threshold_for_pass_rate(f, 0.75)
    assert t == pytest.approx(0.7)
    r = fungusgate.rates_at(t, f, o)
    assert r["tpr"] == 0.75 and r["fungus_rejected"] == 1
    assert r["fpr"] == pytest.approx(1 / 3) and r["other_passed"] == 1


def test_a_release_ships_the_saved_fungus_gate(conn, tmp_path):
    from test_release import MemoryS3, publish, with_embeddings
    data = tmp_path / "data"
    with_embeddings(conn, data, "m1")
    gate().save(data / "models")
    s3 = MemoryS3()
    rid = publish(conn, data, s3)["id"]
    paths = {f["path"] for f in json.loads(s3.objects[f"releases/{rid}/release.json"])["files"]}
    assert {"models/fungus-gate.json", "models/fungus-gate.npz"} <= paths
    assert release is not None


# --- the frozen vector from the fine-tuned model's pass ---------------------------------

def test_the_gate_reads_the_base_models_vector_not_the_fine_tuned_one(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("open_clip.transformer")
    from mycomap_vision import models

    class TinyClip:
        def __init__(self):
            torch.manual_seed(0)
            self.torch, self.device, self.name, self.dim = torch, "cpu", "tiny", 8
            self.model = tr.VisionTransformer(image_size=16, patch_size=8, width=16, layers=3,
                                              heads=2, mlp_ratio=2.0, output_dim=8).eval()
            self.transform = lambda im: torch.from_numpy(
                np.asarray(im.resize((16, 16)), dtype=np.float32).transpose(2, 0, 1) / 255)

        def prepare(self, image):
            return self.transform(image)

        def encode(self, items):
            x = torch.stack([i if isinstance(i, torch.Tensor) else self.transform(i) for i in items])
            with torch.no_grad():
                return self.model(x).numpy()

        def image_module(self):
            return self.model

    base = TinyClip()
    trained = {n: p.detach() + 0.3 for n, p in base.model.named_parameters()
               if n.startswith(("transformer.resblocks.2.", "ln_post.", "proj"))}
    root = tmp_path / "models"
    root.mkdir()
    torch.save(trained, root / "t-ft.pt")
    (root / "t-ft.json").write_text(json.dumps({"base": "bioclip-2", "base_spec": "tiny",
                                                "weights": "t-ft.pt", "blocks_trained": 1}))
    monkeypatch.setattr(models, "load_backbone", lambda spec: TinyClip())
    ft = models.FinetunedBackbone("t-ft", "t-ft", root=root)
    images = [Image.new("RGB", (16, 16), c) for c in ((200, 30, 10), (10, 120, 220))]
    tuned, frozen = ft.encode_with_base([ft.prepare(i) for i in images])
    assert np.allclose(tuned, ft.encode(images), atol=1e-5)          # the served vector
    assert np.allclose(frozen, TinyClip().encode(images), atol=1e-5)  # the base model's
    assert not np.allclose(tuned, frozen, atol=1e-3)
    assert ft.base_name == "bioclip-2"


# --- the probe and the combined score ---------------------------------------------------

def test_the_logistic_probe_learns_to_separate_fungus_from_not():
    rng = np.random.default_rng(0)
    fungus = rng.normal([2.0, 0.0], 0.5, size=(200, 2))
    other = rng.normal([0.0, 2.0], 0.5, size=(100, 2))
    X = np.concatenate([fungus, other])
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    y = np.r_[np.ones(200), np.zeros(100)]
    w, b = fungusgate.fit_logistic(X, y, l2=1e-4)
    z = X @ w + b
    assert (z[:200] > 0).mean() > 0.97 and (z[200:] < 0).mean() > 0.97


def probe_gate(mode):
    # The probe says "fungus" along the first axis, like the text vectors do.
    return FungusGate(**{**TEST_GATE, "thresholds": Thresholds(0.2, 0.5)},
                      probe_w=np.array([6.0, -6.0]), probe_b=0.0, mode=mode)


def test_the_combined_score_adds_the_zero_shot_and_probe_log_odds():
    v = np.array([[0.8, 0.6]])
    zs, pr, both = (probe_gate(m).p_fungus(v)[0] for m in ("zero-shot", "probe", "both"))
    lo = lambda p: np.log(p / (1 - p))                                   # noqa: E731
    assert lo(both) == pytest.approx(lo(zs) + lo(pr))
    assert pr == pytest.approx(1 / (1 + np.exp(-(0.8 * 6 - 0.6 * 6))))


def test_a_gate_with_a_probe_saves_and_loads_its_mode_and_weights(tmp_path):
    probe_gate("both").save(tmp_path)
    back = FungusGate.load(tmp_path)
    assert back.mode == "both" and np.allclose(back.probe_w, [6.0, -6.0])
    v = np.array([[0.3, 0.7], [0.9, 0.1]])
    assert np.allclose(back.p_fungus(v), probe_gate("both").p_fungus(v))


def test_a_probe_mode_without_a_probe_is_refused():
    with pytest.raises(ValueError, match="needs a probe"):
        FungusGate(**TEST_GATE, mode="probe")
    with pytest.raises(ValueError, match="unknown gate mode"):
        FungusGate(**TEST_GATE, mode="vibes")


def test_the_base_weights_can_come_memory_mapped_from_the_checkpoint_file(tmp_path):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("open_clip.transformer")
    from safetensors.torch import save_file

    from mycomap_vision import models
    torch.manual_seed(0)
    tower = tr.VisionTransformer(image_size=16, patch_size=8, width=16, layers=3, heads=2,
                                 mlp_ratio=2.0, output_dim=8).eval()
    x = torch.rand(2, 3, 16, 16)
    with torch.no_grad():
        before = tower(x)
    path = tmp_path / "open_clip_model.safetensors"
    save_file({f"visual.{n}": p.detach().contiguous() for n, p in tower.named_parameters()}, str(path))
    trained = {n: p.detach() + 0.3 for n, p in tower.named_parameters()
               if n.startswith(("transformer.resblocks.2.", "ln_post.", "proj"))}
    tail = models.BaseTail(tower, 2, trained, weights_file=path)
    assert tail.file_backed
    with torch.no_grad():
        for n, v in trained.items():
            dict(tower.named_parameters())[n].copy_(v)
        h = tower._embeds(x)
        if not tower.transformer.batch_first:
            h = h.transpose(0, 1).contiguous()
        for block in tower.transformer.resblocks[:2]:
            h = block(h)
        assert torch.allclose(tail(tower, h), before, atol=1e-5)          # the base vector
        assert torch.allclose(tail.module(h), tower(x), atol=1e-5)        # the trained one
    with pytest.raises(ValueError, match="before the fork point"):
        models.BaseTail(tower, 2, {"transformer.resblocks.0.ln_1.weight": torch.zeros(16)})


def test_a_server_without_the_gate_never_reads_the_base_tail(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    tr = pytest.importorskip("open_clip.transformer")
    from safetensors.torch import save_file

    from mycomap_vision import models

    def tiny():
        torch.manual_seed(0)
        t = types.SimpleNamespace(torch=torch, device="cpu", name="tiny", dim=8)
        t.model = tr.VisionTransformer(image_size=16, patch_size=8, width=16, layers=3, heads=2,
                                       mlp_ratio=2.0, output_dim=8).eval()
        t.transform = lambda im: torch.from_numpy(
            np.asarray(im.resize((16, 16)), dtype=np.float32).transpose(2, 0, 1) / 255)
        t.prepare = t.transform
        t.image_module = lambda: t.model
        t.encode = lambda items: t.model(torch.stack(list(items))).detach().numpy()
        return t
    base = tiny()
    ckpt = tmp_path / "open_clip_model.safetensors"
    save_file({f"visual.{n}": p.detach().contiguous() for n, p in base.model.named_parameters()},
              str(ckpt))
    root = tmp_path / "models"
    root.mkdir()
    torch.save({n: p.detach() + 0.3 for n, p in base.model.named_parameters()
                if n.startswith("transformer.resblocks.2.")}, root / "t-ft.pt")
    (root / "t-ft.json").write_text(json.dumps({"base": "bioclip-2", "base_spec": "open_clip:hf-hub:x",
                                                "weights": "t-ft.pt", "blocks_trained": 1}))
    monkeypatch.setattr(models, "load_backbone", lambda spec: tiny())
    monkeypatch.setattr(models, "open_clip_weights_file", lambda name: ckpt)
    ft = models.FinetunedBackbone("t-ft", "t-ft", root=root)
    assert ft._base_tail is None                     # nothing read while the gate is unused
    img = [ft.prepare(Image.new("RGB", (16, 16), (200, 30, 10)))]
    _, frozen = ft.encode_with_base(img)
    assert ft._base_tail is not None and ft._base_tail.file_backed
    assert np.allclose(frozen, base.encode(img), atol=1e-5)


def test_a_gate_on_the_served_vectors_needs_no_second_model(conn, tmp_path):
    # A probe on m1's own vectors: m1 maps a photo's red to an angle, and the probe calls
    # a high angle (red, cos < 0) a fungus. Const has no fork (no encode_with_base).
    served = FungusGate(**{**TEST_GATE, "thresholds": Thresholds(0.2, 0.5)}, base="m1",
                        probe_w=np.array([-8.0, 0.0]), mode="probe")
    root = seeded(conn, tmp_path)
    client = TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                   backbone_loader=lambda name: Const(name), limits=open_limits(),
                                   background=False, fungus_gate=served))
    assert not hasattr(Const("m1"), "encode_with_base")
    r = identify(client, [photo(247), photo(12)])
    assert [p["verdict"] for p in r["gate"]["photos"]] == [FUNGUS, NOT_FUNGUS]
    assert r["gate"]["left_out"] == [1]


def test_health_says_which_gate_is_serving_so_a_deploy_can_be_checked(conn, tmp_path):
    root = seeded(conn, tmp_path)
    on = app(tmp_path, root).get("/api/health").json()["fungus_gate"]
    assert on["mode"] == "zero-shot" and on["thresholds"] == {"not_fungus": 0.2, "unsure": 0.5}
    assert on["fingerprint"] == gate().fingerprint()
    assert app(tmp_path, root, with_gate=False).get("/api/health").json()["fungus_gate"] is None
    stricter = FungusGate(**{**TEST_GATE, "thresholds": Thresholds(0.3, 0.5)})
    assert stricter.fingerprint() != gate().fingerprint()       # a re-judge is due


def test_a_non_fungus_answer_has_the_same_fields_as_any_other(conn, tmp_path):
    client = client_for(conn, tmp_path)
    duck, mushroom = identify(client, [photo(247, DUCK_GREEN)]), identify(client, [photo(247)])
    assert set(duck) == set(mushroom)
    assert set(duck["per_photo"][0]) == set(mushroom["per_photo"][0])
