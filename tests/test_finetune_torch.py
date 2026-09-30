"""The fine-tuning loop itself. Needs torch, which CI doesn't install, so it runs on
machines with the embed extras (the laptop, the AWS trainer) and is skipped elsewhere."""

import numpy as np
import pytest
from test_finetune import embedded

from mycomap_vision import finetune, models

torch = pytest.importorskip("torch")
T = pytest.importorskip("torchvision.transforms")


torch = pytest.importorskip("torch")
T = pytest.importorskip("torchvision.transforms")


class TinyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.patch_embed = torch.nn.Linear(3, 4)
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(3)])
        self.norm = torch.nn.LayerNorm(4)

    def forward(self, x):
        h = self.patch_embed(x.mean(dim=(2, 3)))
        for b in self.blocks:
            h = h + torch.tanh(b(h))
        return self.norm(h)


class Tiny:
    """A real (tiny) torch backbone with the hooks fine-tuning uses."""
    spec = "tiny-spec"

    def __init__(self, name="tiny"):
        self.name, self.dim, self.device = name, 4, "cpu"
        self.model = TinyNet()
        self.transform = T.Compose([T.Resize(8), T.CenterCrop(8), T.ToTensor(),
                                    T.Normalize([0.5] * 3, [0.25] * 3)])

    def prepare(self, image):
        return self.transform(image)

    def encode(self, items):
        x = torch.stack([i if isinstance(i, torch.Tensor) else self.transform(i) for i in items])
        with torch.no_grad():
            return self.model(x).numpy()

    def image_module(self):
        return self.model

    def forward_features(self, x):
        return self.model(x)


def test_training_changes_only_the_tail_and_the_saved_model_reloads(conn, tmp_path, monkeypatch):
    store, root = embedded(conn, tmp_path, "tiny", Tiny())
    before = {n: p.detach().clone() for n, p in Tiny().model.named_parameters()}
    trained_model = {}

    def loader(spec):
        b = Tiny()
        trained_model["m"] = b.model
        return b
    cfg = finetune.FinetuneConfig(blocks=1, batch_size=4, workers=0, max_steps=5,
                                  lr_backbone=0.05, lr_heads=0.05)
    meta = finetune.finetune(conn, "tiny", store, "large", "tiny-ft", cfg, 28, root,
                             tmp_path / "models", loader=loader, log=lambda s: None)
    after = dict(trained_model["m"].named_parameters())
    changed = {n for n in before if not torch.equal(before[n], after[n])}
    assert changed and changed <= {"blocks.2.weight", "blocks.2.bias", "norm.weight", "norm.bias"}
    assert "patch_embed.weight" not in changed and "blocks.0.weight" not in changed
    saved = torch.load(tmp_path / "models" / "tiny-ft.pt", weights_only=True)
    assert set(saved) == {"blocks.2.weight", "blocks.2.bias", "norm.weight", "norm.bias"}
    assert meta["trained_through"] == "2026-08-24" and meta["photos"] == 6
    assert conn.execute("select base from finetunes where name = 'tiny-ft'").fetchone()[0] == "tiny"
    # Reloading = the base model with the trained weights put back.
    monkeypatch.setattr(models, "load_backbone", lambda spec: Tiny())
    ft = models.FinetunedBackbone("tiny-ft", "tiny-ft", root=tmp_path / "models")
    for n, p in ft.image_module().named_parameters():
        assert torch.equal(p, after[n].detach())
    img = __import__("PIL.Image", fromlist=["x"]).new("RGB", (8, 8), (200, 10, 10))
    assert not np.allclose(ft.encode([img]), Tiny().encode([img]))


def test_a_fine_tune_stopped_by_the_time_limit_saves_and_registers_nothing(conn, tmp_path):
    from mycomap_vision.screening import Stopped
    store, root = embedded(conn, tmp_path, "tiny", Tiny())
    steps = []
    cfg = finetune.FinetuneConfig(blocks=1, batch_size=4, workers=0, max_steps=5)
    with pytest.raises(Stopped, match="time limit"):
        finetune.finetune(conn, "tiny", store, "large", "tiny-ft", cfg, 28, root,
                          tmp_path / "models", loader=lambda s: Tiny(),
                          log=lambda s: steps.append(s),
                          should_stop=lambda: len(steps) >= 1)
    assert not (tmp_path / "models" / "tiny-ft.pt").exists()
    conn.executescript(__import__("mycomap_vision.evaluate", fromlist=["x"]).FINETUNE_SCHEMA)
    assert conn.execute("select count(*) from finetunes where name = 'tiny-ft'").fetchone()[0] == 0
