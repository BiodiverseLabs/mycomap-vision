"""BioCLIP 2 at a larger input (paper plan E24, Steve's OK 2026-10-13): `bioclip-2-336` and
`bioclip-2-448` load the same weights with the patch-grid position embeddings resized
(bicubic) and the preprocessing resized; the fine-tune splits a batch the GPU can't hold,
with the same loss, so the recipe stays the one the 224 px model was trained with; and the
trainer's estimate counts the larger input's cost."""

import pytest

from mycomap_vision import models, trainer

torch = pytest.importorskip("torch")
T = pytest.importorskip("torchvision.transforms")


class TinyTower(torch.nn.Module):
    """The parts of an open_clip VisionTransformer that resizing touches."""

    def __init__(self, patch=14, grid=16, width=8):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, width, kernel_size=patch, stride=patch)
        torch.manual_seed(0)
        self.positional_embedding = torch.nn.Parameter(torch.randn(1 + grid * grid, width))
        self.grid_size, self.image_size = (grid, grid), (grid * patch, grid * patch)


def preprocess(size=224):
    return T.Compose([T.Resize(size, interpolation=T.InterpolationMode.BICUBIC),
                      T.CenterCrop(size), T.ToTensor()])


def test_a_larger_input_resizes_the_position_grid_and_keeps_the_class_token():
    tower = TinyTower()
    cls = tower.positional_embedding.data[0].clone()
    tower, tf = models.resize_vit_input(tower, preprocess(), 448)
    assert tuple(tower.positional_embedding.shape) == (1 + 32 * 32, 8)
    assert torch.equal(tower.positional_embedding.data[0], cls)
    assert tower.grid_size == (32, 32) and tower.image_size == (448, 448)
    from PIL import Image
    assert tuple(tf(Image.new("RGB", (900, 600))).shape) == (3, 448, 448)


def test_a_smooth_position_field_stays_smooth_when_resized():
    tower = TinyTower(grid=4, width=1)
    ramp = torch.arange(16, dtype=torch.float32).reshape(4, 4)        # a smooth field
    tower.positional_embedding.data[1:, 0] = ramp.flatten()
    tower, _ = models.resize_vit_input(tower, preprocess(56), 112)
    grid = tower.positional_embedding.data[1:, 0].reshape(8, 8)
    assert grid.min() >= -1 and grid.max() <= 16                       # no ringing to speak of
    assert bool((grid[:, 1:] >= grid[:, :-1] - 1e-4).all())            # still increasing
    with pytest.raises(ValueError, match="multiple of the patch size"):
        models.resize_vit_input(TinyTower(), preprocess(), 300)


def test_the_aliases_name_the_same_weights_at_their_size(monkeypatch):
    assert models.resolve_spec("bioclip-2-336").endswith("bioclip-2?img_size=336")
    assert models.resolve_spec("bioclip-2-448").endswith("bioclip-2?img_size=448")
    assert models.storage_name("bioclip-2-448") == "bioclip-2-448"
    asked = []
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo, f: asked.append((repo, f)) or "w.safetensors")
    assert models.open_clip_weights_file("hf-hub:imageomics/bioclip-2?img_size=448") == \
        "w.safetensors"
    assert asked == [("imageomics/bioclip-2", "open_clip_model.safetensors")]


def test_the_backbone_takes_the_size_option_and_refuses_others(monkeypatch):
    seen = {}

    def fake_load(model_name, pretrained):
        seen["name"] = model_name
        tower = TinyTower()
        tower.forward = lambda x: torch.zeros(len(x), 4)
        return tower, preprocess()
    monkeypatch.setattr(models, "load_open_clip_image_tower", fake_load)
    b = models.OpenClipBackbone("bioclip-2-336", "hf-hub:imageomics/bioclip-2?img_size=336")
    assert seen["name"] == "hf-hub:imageomics/bioclip-2"              # the option is not a name
    assert tuple(b.model.positional_embedding.shape)[0] == 1 + 24 * 24
    with pytest.raises(ValueError, match="img_size only"):
        models.OpenClipBackbone("x", "hf-hub:imageomics/bioclip-2?global_pool=token")


def test_the_estimate_counts_the_larger_inputs_cost():
    stages = trainer.plan_stages(["bioclip-2-448"], ["bioclip-2-448"], "r")
    est = trainer.estimate(stages, 100_000)
    small = trainer.estimate(trainer.plan_stages(["bioclip-2"], ["bioclip-2"], "r"), 100_000)
    assert est["total_hours"] > 3 * small["total_hours"]
    rates = {r["stage"].split(" ")[0]: r["photos_per_second"] for r in est["stages"]}
    assert rates["finetune"] == round(trainer.FINETUNE_RATES["bioclip-2"] / 4.20, 1)
    assert rates["embed"] == round(trainer.EMBED_RATES["bioclip-2"] / 5.03, 1)


# --- the fine-tune at a size the GPU can't hold in one pass ---------------------------------

def tiny_data(conn, tmp_path):
    from test_finetune import embedded
    from test_finetune_torch import Tiny
    return embedded(conn, tmp_path, "tiny", Tiny())


def tiny_finetune(conn, data, out, cfg, loader):
    from mycomap_vision import finetune
    store, root = data
    meta = finetune.finetune(conn, "tiny", store, "large", "tiny-ft", cfg, 28, root, out,
                             loader=loader, log=lambda s: None)
    return meta, torch.load(out / "tiny-ft.pt", weights_only=True)


def cfg_of(**kw):
    from mycomap_vision import finetune
    return finetune.FinetuneConfig(blocks=1, batch_size=4, workers=0, max_steps=3,
                                   lr_backbone=0.05, lr_heads=0.05, **kw)


def test_splitting_a_batch_trains_the_same_model(conn, tmp_path):
    pytest.importorskip("torchvision")
    from test_finetune_torch import Tiny
    data = tiny_data(conn, tmp_path)
    whole, w1 = tiny_finetune(conn, data, tmp_path / "a", cfg_of(), lambda spec: Tiny())
    split, w2 = tiny_finetune(conn, data, tmp_path / "b", cfg_of(micro_batch=1),
                              lambda spec: Tiny())
    assert whole["micro_batch"] == 4 and split["micro_batch"] == 1
    for n in w1:
        assert torch.allclose(w1[n], w2[n], atol=1e-5), n


def test_out_of_gpu_memory_halves_the_pass_and_goes_on(conn, tmp_path):
    pytest.importorskip("torchvision")
    from test_finetune_torch import Tiny

    class Hungry(Tiny):
        def forward_features(self, x):
            if len(x) > 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory (test)")
            return super().forward_features(x)
    meta, _ = tiny_finetune(conn, tiny_data(conn, tmp_path), tmp_path / "m", cfg_of(),
                            lambda spec: Hungry())
    assert meta["micro_batch"] == 2 and meta["steps"] == 3
