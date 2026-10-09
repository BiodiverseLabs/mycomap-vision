"""The Picek replication's training loop, loss and backbone. Needs torch, which CI doesn't
install, so it runs on machines with the embed extras (the laptop, the AWS trainer)."""

import csv
import json

import numpy as np
import pytest
from test_models_and_scoreboard import seed_two_species
from test_picek import VAL_DAYS

from mycomap_vision import config, evaluate, models, picek
from mycomap_vision.embed import embed_photos, photos_to_embed

torch = pytest.importorskip("torch")
T = pytest.importorskip("torchvision.transforms")


# --- Seesaw -------------------------------------------------------------------------------

def seesaw_by_the_book(z, y, counts, p, q, eps=1e-2):
    """mmdet's SeesawLoss written out with the full C x C matrices."""
    z = z.double()
    n = torch.as_tensor(counts, dtype=torch.float64)
    ratio = n[None, :] / n[:, None]                          # N_j / N_i
    mitigation = torch.where(ratio < 1, ratio ** p, torch.ones_like(ratio))[y]
    sigma = torch.softmax(z, dim=1)
    own = sigma.gather(1, y[:, None]).clamp(min=eps)
    score = sigma / own
    compensation = torch.where(score > 1, score ** q, torch.ones_like(score))
    s = mitigation * compensation
    onehot = torch.nn.functional.one_hot(y, z.shape[1]).double()
    return torch.nn.functional.cross_entropy(z + s.log() * (1 - onehot), y)


def test_seesaw_matches_the_published_loss():
    g = torch.Generator().manual_seed(0)
    z = torch.randn(6, 5, generator=g) * 3
    y = torch.tensor([0, 1, 4, 2, 0, 3])
    counts = [100, 3, 40, 1, 7]
    got = picek.SeesawLoss(counts, 0.8, 2.0)(z, y)
    assert float(got) == pytest.approx(float(seesaw_by_the_book(z, y, counts, 0.8, 2.0)), rel=1e-5)


def test_seesaw_with_no_mitigation_or_compensation_is_cross_entropy():
    z = torch.randn(4, 3)
    y = torch.tensor([0, 1, 2, 1])
    assert float(picek.SeesawLoss([5, 50, 500], 0.0, 0.0)(z, y)) == pytest.approx(
        float(torch.nn.functional.cross_entropy(z, y)), rel=1e-6)


def test_seesaw_spares_a_rare_class_from_a_common_class_photo():
    # A photo of the common class 0 pushes the rare class 1 down less than plain CE does.
    z = torch.zeros(1, 2, requires_grad=True)
    picek.SeesawLoss([1000, 10], 0.8, 0.0)(z, torch.tensor([0])).backward()
    seesaw_push = float(z.grad[0, 1])
    z2 = torch.zeros(1, 2, requires_grad=True)
    torch.nn.functional.cross_entropy(z2, torch.tensor([0])).backward()
    assert 0 < seesaw_push < float(z2.grad[0, 1])


# --- augmentation -------------------------------------------------------------------------

def test_their_heavy_augmentation_and_plain_resize_for_evaluation():
    heavy = picek.train_transform("vit_heavy", 384).transforms
    crop = next(t for t in heavy if isinstance(t, T.RandomResizedCrop))
    assert crop.size == (384, 384) and crop.scale == (0.8, 1.0)
    ra = next(t for t in heavy if isinstance(t, T.RandAugment))
    assert (ra.num_ops, ra.magnitude) == (2, 20)
    assert not any(isinstance(t, T.ColorJitter) for t in heavy)
    ev = picek.eval_transform(384).transforms
    assert isinstance(ev[0], T.Resize) and tuple(ev[0].size) == (384, 384)
    assert not any(isinstance(t, (T.CenterCrop, T.RandomResizedCrop)) for t in ev)
    norm = next(t for t in ev if isinstance(t, T.Normalize))
    assert tuple(norm.mean) == tuple(norm.std) == (0.5, 0.5, 0.5)


def test_the_presets_are_their_published_settings():
    beit = picek.PRESETS["fungitastic-beit-b384"]
    assert (beit.timm_name, beit.image_size, beit.loss, beit.lr, beit.epochs, beit.augment) == (
        "beit_base_patch16_384.in22k_ft_in22k_in1k", 384, "seesaw", 0.01, 50, "vit_heavy")
    cfg = picek.PicekConfig().resolved()
    assert (cfg.momentum, cfg.weight_decay, cfg.effective_batch) == (0.9, 0.0, 256)
    assert (cfg.plateau_factor, cfg.plateau_patience, cfg.plateau_eps) == (0.9, 1, 1e-6)
    assert (cfg.seesaw_p, cfg.seesaw_q) == (0.8, 2.0)
    assert picek.PRESETS["df20-vit-l384"].loss == "ce"


# --- training, registering, scoring -------------------------------------------------------

class TinyTimm(torch.nn.Module):
    """The parts of a timm classifier the training and backbone use."""

    def __init__(self, n_classes):
        super().__init__()
        torch.manual_seed(0)
        self.body = torch.nn.Linear(3, 6)
        self.fc_norm = torch.nn.LayerNorm(6)
        self.head = torch.nn.Linear(6, n_classes)

    def forward_features(self, x):
        return self.body(x.mean(dim=(2, 3)))

    def forward_head(self, x, pre_logits=False):
        x = self.fc_norm(x)
        return x if pre_logits else self.head(x)

    def get_classifier(self):
        return self.head

    def forward(self, x):
        return self.forward_head(self.forward_features(x))


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    monkeypatch.setitem(picek.PRESETS, "tiny", picek.Preset("tiny", 16, "seesaw", 0.05, 3,
                                                            "vit_heavy", "test"))
    monkeypatch.setattr(picek, "create_model", lambda preset, n, pretrained=True: TinyTimm(n))
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    return picek.PicekConfig(preset="tiny", effective_batch=2, micro_batch=2, workers=0,
                             val_days=int(VAL_DAYS), amp=False)


def test_training_saves_registers_and_scores_like_any_model(conn, tmp_path, tiny):
    store = seed_two_species(conn, tmp_path)
    meta = picek.train(conn, store, "large", "picek-tiny", tiny, log=lambda s: None)
    root = config.DATA_DIR / "models"
    for f in ("picek-tiny.json", "picek-tiny.pt", "picek-tiny.classifier.npz",
              "picek-tiny.records.csv"):
        assert (root / f).is_file(), f
    assert meta["kind"] == "classifier" and meta["trained_through"] == "2026-08-24"
    assert meta["epochs_run"] == 3 and meta["species"] == 2
    f1 = [h["val_macro_f1"] for h in meta["history"]]
    assert meta["best_epoch"] == 1 + int(np.argmax(f1))       # kept: best macro-F1
    assert conn.execute("select trained_through from finetunes where name = 'picek-tiny'"
                        ).fetchone()[0] == "2026-08-24"
    with open(root / "picek-tiny.records.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert {r["observation_id"] for r in rows if r["split"] == "validation"} == {"102", "105"}
    assert {r["observation_id"] for r in rows}.isdisjoint({"106", "107"})

    # It loads as a backbone, embeds, and the classifier methods score it in a comparison.
    assert models.resolve_spec("picek-tiny") == "classifier:picek-tiny"
    bb = models.load_backbone("picek-tiny")
    emb = tmp_path / "emb"
    embed_photos(conn, store, bb, photos_to_embed(conn, "picek-tiny", "large", store.location,
                                                  "local"), emb / "picek-tiny", log=lambda s: None)
    out = evaluate.compare(conn, ["picek-tiny"], ["classifier", "classifier+month", "nearest"],
                           embeddings_root=emb, per_image=True, log=lambda s: None)
    for run in out["runs"]:
        assert run["all_photos"]["species"]["all"]["n"] == 2
        assert "macro_f1" in run["all_photos"] and "per_image" in run["all_photos"]

    # Never scored on records it may have seen (here: a 250-day test window).
    with pytest.raises(ValueError, match="trained on records validated up to"):
        evaluate.compare(conn, ["picek-tiny"], ["classifier"], test_days=250,
                         embeddings_root=emb, log=lambda s: None)


def test_the_head_applied_to_the_stored_embedding_gives_the_models_own_answer(conn, tmp_path,
                                                                                tiny):
    store = seed_two_species(conn, tmp_path)
    picek.train(conn, store, "large", "picek-tiny", tiny, log=lambda s: None)
    bb = models.load_backbone("picek-tiny")
    from PIL import Image
    x = torch.stack([bb.prepare(Image.new("RGB", (8, 8), (c, 0, 0))) for c in (250, 10)])
    with torch.no_grad():
        want = torch.log_softmax(bb.model(x.to(bb.device)).cpu() / json.loads(
            (config.DATA_DIR / "models" / "picek-tiny.json").read_text())["temperature"], 1)
    from mycomap_vision.embed import normalise
    m = evaluate.make_method("classifier", "picek-tiny")
    index = evaluate.build_index([evaluate.Record("1", "A x", "A", "F", None, None, [0]),
                                  evaluate.Record("2", "B y", "B", "F", None, None, [1])])
    m.fit(None, index)
    for i in range(2):
        got = m.species_scores(normalise(bb.encode([x[i]])))
        assert np.allclose(got, want[i].numpy(), atol=0.02)


def test_a_stopped_run_saves_nothing(conn, tmp_path, tiny):
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    with pytest.raises(Stopped):
        picek.train(conn, store, "large", "picek-tiny", tiny, log=lambda s: None,
                    should_stop=lambda: True)
    assert not (config.DATA_DIR / "models" / "picek-tiny.pt").exists()


def test_max_steps_caps_a_smoke_run(conn, tmp_path, tiny):
    store = seed_two_species(conn, tmp_path)
    from dataclasses import replace
    meta = picek.train(conn, store, "large", "picek-tiny", replace(tiny, max_steps=1),
                       log=lambda s: None)
    assert meta["steps"] == 1 and meta["stopped_at_max_steps"] and meta["epochs_run"] == 1


def test_with_a_photo_cache_only_where_training_photos_are_read_changes(conn, tmp_path, tiny):
    store = seed_two_species(conn, tmp_path)
    data = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS))
    preset = picek.PRESETS["fungitastic-beit-b384"]
    plain = picek.make_datasets(store.location, store.location, data, preset, data.val)
    cached = picek.make_datasets(str(tmp_path / "cache"), store.location, data, preset,
                                 data.val)
    for a, b in zip(plain, cached):
        assert repr(a.transform) == repr(b.transform)
        assert (a.items, a.draft) == (b.items, b.draft)
    assert cached[0].location == str(tmp_path / "cache")
    assert cached[1].location == plain[1].location == store.location   # validation: originals


def test_training_from_a_photo_cache_records_it(conn, tmp_path, tiny):
    from dataclasses import replace
    store = seed_two_species(conn, tmp_path)
    meta = picek.train(conn, store, "large", "picek-tiny",
                       replace(tiny, cache_px=16, cache_dir=str(tmp_path / "cache")),
                       log=lambda s: None)
    assert meta["photo_cache"]["short_side"] == 16
    assert meta["photo_cache"]["made"] == 4                           # the training photos
    assert meta["config"]["cache_px"] == 16
    assert len(list((tmp_path / "cache").rglob("*.png"))) == 4
