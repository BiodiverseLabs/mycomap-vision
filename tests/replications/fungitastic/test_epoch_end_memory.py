"""The end of a Picek epoch fits in the trainer's RAM (E4, 2026-10-21: run 20261021-141123 was
OOM-killed on a 16 GB g6.xlarge in epoch 2's validation pass). With 16,257 classes one
validation pass's logits are 47,136 x 16,257 float32 = 3.1 GB, so the pass must hold them
once (no list of parts concatenated, no copy of the known rows for the loss), the best
epoch's pass must not still be held while the next one runs, and the temperature fit must
not make float64 copies of them. And a killed run loses at most the epoch it was in: the
checkpoint is written after every epoch, not only at the epoch rule's block boundaries."""

import tracemalloc
import weakref
from dataclasses import replace

import numpy as np
import pytest
from test_models_and_scoreboard import seed_two_species
from test_retrain_torch import TinyTimm, tiny  # noqa: F401  (the fixture)

from mycomap_vision import config
from mycomap_vision.replications.fungitastic import retrain as picek

torch = pytest.importorskip("torch")


class FakeVal(torch.utils.data.Dataset):
    """n photos (tiny tensors) with labels; every 7th is a class the model doesn't have."""

    def __init__(self, n, n_classes):
        g = torch.Generator().manual_seed(0)
        self.x = torch.rand(n, 3, 2, 2, generator=g)
        self.y = [-1 if i % 7 == 0 else i % n_classes for i in range(n)]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x[i], self.y[i], i


def peak_numpy_bytes(fn):
    tracemalloc.start()
    tracemalloc.reset_peak()
    try:
        out = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return out, peak


def test_a_validation_pass_holds_its_logits_once():
    n, c = 3000, 4000                                  # logits: 48 MB of float32
    model = TinyTimm(c)
    loader = torch.utils.data.DataLoader(FakeVal(n, c), batch_size=64, collate_fn=picek.collate)
    loss_fn = torch.nn.CrossEntropyLoss()
    v, peak = peak_numpy_bytes(lambda: picek.run_validation(model, loader, "cpu", c, loss_fn))
    matrix = n * c * 4
    assert v["logits"].shape == (n, c) and v["n"] == sum(y >= 0 for y in FakeVal(n, c).y)
    block = 1024 * c * 4                               # the loss's fixed working block
    assert peak < matrix + 1.5 * block, f"peak {peak / matrix:.2f}x the logits matrix"
    # the same numbers as before: loss over the known photos, top-1, rows in order
    z, y = v["logits"], v["labels"]
    known = y >= 0
    want = float(loss_fn(torch.from_numpy(z[known]), torch.from_numpy(y[known])))
    assert v["loss"] == pytest.approx(want, rel=1e-5)
    assert v["top1"] == pytest.approx(float((z[known].argmax(1) == y[known]).mean()))
    assert v["rows"].tolist() == list(range(n))


def reference_temperature(logits, labels, groups=None):
    """fit_temperature as it was before 2026-10-22 (float64 copies of every logit)."""
    keep = labels >= 0
    z, y = logits[keep].astype(np.float64), labels[keep]
    if groups is not None:
        g = np.asarray(groups)[keep]
        order = np.argsort(g, kind="stable")
        z, y, g = z[order], y[order], g[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        z = np.add.reduceat(z, starts, axis=0) / np.diff(np.r_[starts, len(g)])[:, None]
        y = y[starts]

    def nll(t):
        s = z / t
        s = s - s.max(axis=1, keepdims=True)
        return float(np.mean(np.log(np.exp(s).sum(axis=1)) - s[np.arange(len(y)), y]))
    grid = np.geomspace(0.05, 20.0, 60)
    best = grid[int(np.argmin([nll(t) for t in grid]))]
    lo, hi = best / 1.15, best * 1.15
    for _ in range(40):
        a, b = lo + 0.382 * (hi - lo), lo + 0.618 * (hi - lo)
        if nll(a) < nll(b):
            hi = b
        else:
            lo = a
    return round(float((lo + hi) / 2), 5)


@pytest.mark.parametrize("grouped", [False, True])
def test_the_temperature_fit_makes_no_float64_copy_of_the_logits_and_finds_the_same_t(grouped):
    rng = np.random.default_rng(1)
    n, c = 8000, 1500                                  # 48 MB of float32
    y = rng.integers(0, c, n)
    y[::9] = -1
    logits = rng.normal(0, 1, (n, c)).astype(np.float32)
    logits[np.arange(n), np.maximum(y, 0)] += 4.0      # a confident, mostly right model
    groups = (np.arange(n) // 3)[rng.permutation(n)] if grouped else None
    t, peak = peak_numpy_bytes(lambda: picek.fit_temperature(logits, y, groups))
    block = 1024 * c * 8                               # one float64 block of rows
    assert peak < 0.4 * logits.nbytes + 3 * block, f"peak {peak / logits.nbytes:.2f}x the logits"
    assert t == pytest.approx(reference_temperature(logits, y, groups), rel=1e-4)


# --- inside the training loop ---------------------------------------------------------------

RULE = picek.EpochRule(2, 8, 2, 0.005)


@pytest.fixture
def scripted(monkeypatch):
    """Validation reports macro-F1 from `f1`; records, as each pass starts, how many earlier
    passes' logits are still alive."""
    state = {"f1": [], "calls": 0, "alive_at_start": [], "refs": []}
    real = picek.run_validation

    def run_validation(*a, **k):
        import gc
        gc.collect()
        state["alive_at_start"].append(sum(r() is not None for r in state["refs"]))
        v = real(*a, **k)
        state["refs"].append(weakref.ref(v["logits"]))
        v["macro_f1"] = state["f1"][min(state["calls"], len(state["f1"]) - 1)]
        state["calls"] += 1
        return v
    monkeypatch.setattr(picek, "run_validation", run_validation)
    return state


def test_the_best_epochs_validation_is_not_held_while_the_next_pass_runs(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.50, 0.40, 0.30, 0.20]          # epoch 1 stays the best
    meta = picek.train(conn, store, "large", "picek-tiny", replace(tiny, epochs=4),
                       log=lambda s: None)
    assert meta["best_epoch"] == 1
    assert scripted["alive_at_start"][:4] == [0, 0, 0, 0]
    # the best epoch's pass is scored again for the temperature (a fifth pass)
    assert scripted["calls"] == 5 and meta["temperature"] > 0


def test_a_checkpoint_is_written_after_every_epoch_not_only_at_block_boundaries(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.10, 0.30, 0.20, 0.31, 0.25, 0.312, 0.9, 0.9]
    saved = []
    meta = picek.train(conn, store, "large", "picek-tiny", replace(tiny, epoch_rule=RULE),
                       log=lambda s: None,
                       on_block=lambda path, st: saved.append(st["epochs_done"]))
    assert saved == [1, 2, 3, 4, 5, 6]
    assert [b["epoch"] for b in meta["blocks"]] == [2, 4, 6]      # decisions stay per block


def test_a_run_killed_mid_block_resumes_from_its_last_finished_epoch(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    f1 = [0.10, 0.30, 0.32, 0.34, 0.342, 0.343, 0.5, 0.5]
    cfg = replace(tiny, epoch_rule=RULE)
    scripted["f1"] = f1
    whole = picek.train(conn, store, "large", "picek-whole", cfg, log=lambda s: None)
    scripted["calls"] = 0
    with pytest.raises(Stopped, match="checkpoint after epoch 3 is kept"):
        picek.train(conn, store, "large", "picek-tiny", cfg, log=lambda s: None,
                    should_stop=lambda: scripted["calls"] >= 3)     # dies in epoch 4
    ckpt = config.DATA_DIR / "models" / f"picek-tiny{picek.CHECKPOINT_SUFFIX}"
    assert torch.load(ckpt, map_location="cpu", weights_only=True)["epoch"] == 3
    scripted["calls"] = 3
    trained = []
    meta = picek.train(conn, store, "large", "picek-tiny", cfg,
                       log=lambda s: trained.append(s) if "epoch " in s and "/" in s else None,
                       resume_from=ckpt)
    assert meta["resumed_from_epoch"] == 3
    assert [h["epoch"] for h in meta["history"]] == list(range(1, 7))
    assert [b["decision"] for b in meta["blocks"]] == [b["decision"] for b in whole["blocks"]]
    assert meta["best_epoch"] == whole["best_epoch"]


def test_a_fixed_length_run_resumes_from_its_last_epoch_too(conn, tmp_path, tiny,
                                                           scripted):  # noqa: F811
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.1, 0.2, 0.3, 0.4]
    cfg = replace(tiny, epochs=4)
    with pytest.raises(Stopped, match="checkpoint after epoch 2 is kept"):
        picek.train(conn, store, "large", "picek-tiny", cfg, log=lambda s: None,
                    should_stop=lambda: scripted["calls"] >= 2)
    ckpt = config.DATA_DIR / "models" / f"picek-tiny{picek.CHECKPOINT_SUFFIX}"
    scripted["calls"] = 2
    meta = picek.train(conn, store, "large", "picek-tiny", cfg, log=lambda s: None,
                       resume_from=ckpt)
    assert meta["resumed_from_epoch"] == 2 and meta["epochs_run"] == 4
    assert meta["best_epoch"] == 4


def test_the_step_log_says_how_much_of_training_waits_for_photos(conn, tmp_path, tiny, scripted,
                                                                monkeypatch):  # noqa: F811
    """Whether the 4-vCPU loader or the GPU sets a run's pace (E4 ran 34.4 photos/s against
    59 assumed) is read off the run's own log."""
    monkeypatch.setattr(picek, "STEP_LOG_EVERY", 1)
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.1]
    lines = []
    picek.train(conn, store, "large", "picek-tiny", replace(tiny, epochs=1, effective_batch=1,
                                                            micro_batch=1), log=lines.append)
    steps = [s for s in lines if "photos/s overall" in s]
    assert steps and all("of training waiting for photos" in s for s in steps)
