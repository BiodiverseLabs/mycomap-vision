"""The epoch rule inside the real training loop (retrain.train), on a tiny model whose
validation macro-F1 is scripted per epoch. Needs torch (the laptop, the AWS trainer)."""

from dataclasses import replace

import pytest
from test_models_and_scoreboard import seed_two_species
from test_retrain_torch import tiny  # noqa: F401  (the fixture)

from mycomap_vision import config
from mycomap_vision.replications.fungitastic import retrain as picek

torch = pytest.importorskip("torch")

# Boundaries at 2, 4, 6, 8 (min 2, cap 8, blocks of 2), gain 0.005.
RULE = picek.EpochRule(2, 8, 2, 0.005)


@pytest.fixture
def scripted(monkeypatch):
    """Validation passes report macro-F1 from `f1[epoch - 1]`, counted from `start`; the
    rest of each pass (logits, loss) is the real one."""
    state = {"f1": [], "calls": 0}
    real = picek.run_validation

    def run_validation(*a, **k):
        v = real(*a, **k)
        v["macro_f1"] = state["f1"][min(state["calls"], len(state["f1"]) - 1)]
        state["calls"] += 1
        return v
    monkeypatch.setattr(picek, "run_validation", run_validation)
    return state


def rule_cfg(tiny):  # noqa: F811
    return replace(tiny, epoch_rule=RULE)


def test_training_stops_when_a_block_adds_too_little_and_keeps_the_best_epoch_overall(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.10, 0.30, 0.20, 0.31, 0.25, 0.312, 0.9, 0.9]
    blocks = []
    meta = picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                       on_block=lambda path, st: blocks.append(st["epochs_done"]))
    assert meta["epochs_run"] == 6                       # 4 -> 6 gained 0.002: stop
    assert [b["decision"] for b in meta["blocks"]] == ["continue", "continue", "stop"]
    assert meta["blocks"][-1]["gain"] == pytest.approx(0.002)
    assert "rose only" in meta["stop_reason"]
    assert meta["best_epoch"] == 6 and meta["best_val_macro_f1"] == pytest.approx(0.312)
    assert blocks == [2, 4, 6]                           # a checkpoint at every boundary
    assert meta["epoch_rule"] == {"min_epochs": 2, "max_epochs": 8, "block": 2,
                                  "min_gain": 0.005}
    assert [h["val_macro_f1"] for h in meta["history"]] == scripted["f1"][:6]


def test_the_kept_model_is_the_global_best_even_when_later_blocks_continue(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    store = seed_two_species(conn, tmp_path)
    # Epoch 3 is the best; the block 3-4 continues on it, 5-6 adds nothing and stops.
    scripted["f1"] = [0.10, 0.20, 0.50, 0.40, 0.40, 0.40, 0.40, 0.40]
    saved = {}

    def on_block(path, st):
        saved[st["epochs_done"]] = torch.load(path, map_location="cpu", weights_only=True)
    meta = picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                       on_block=on_block)
    assert meta["best_epoch"] == 3 and meta["epochs_run"] == 6
    weights = torch.load(config.DATA_DIR / "models" / "picek-tiny.pt", map_location="cpu",
                         weights_only=True)
    best_in_ckpt = saved[6]["best_state"]
    assert saved[6]["best_epoch"] == 3
    assert all(torch.equal(weights[k], best_in_ckpt[k]) for k in weights)
    assert not all(torch.equal(weights[k], saved[6]["model"][k].cpu()) for k in weights)
    # The head saved for scoring is the best epoch's too, not the last epoch's.
    import numpy as np
    with np.load(config.DATA_DIR / "models" / "picek-tiny.classifier.npz") as z:
        assert np.allclose(z["weight"], best_in_ckpt["head.weight"].numpy())


def test_training_stops_at_the_cap_however_much_is_still_gained(conn, tmp_path, tiny,
                                                               scripted):  # noqa: F811
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.1 * e for e in range(1, 9)]
    meta = picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None)
    assert meta["epochs_run"] == 8 and meta["best_epoch"] == 8
    assert meta["blocks"][-1]["decision"] == "stop" and "cap of 8" in meta["stop_reason"]


def test_an_interruption_mid_block_keeps_the_last_block_and_resumes_from_it(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    f1 = [0.10, 0.30, 0.32, 0.34, 0.342, 0.343, 0.5, 0.5]
    # The uninterrupted run, for comparison.
    scripted["f1"], scripted["calls"] = f1, 0
    whole = picek.train(conn, store, "large", "picek-whole", rule_cfg(tiny), log=lambda s: None)

    # Stop as soon as epoch 4 begins (after 3 validation passes): mid-block 3-4.
    scripted["calls"] = 0
    with pytest.raises(Stopped, match="block checkpoint after epoch 2 is kept"):
        picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                    should_stop=lambda: scripted["calls"] >= 3)
    ckpt = config.DATA_DIR / "models" / f"picek-tiny{picek.CHECKPOINT_SUFFIX}"
    assert torch.load(ckpt, map_location="cpu", weights_only=True)["epoch"] == 2
    assert not (config.DATA_DIR / "models" / "picek-tiny.pt").exists()

    # Resumed: epochs 3.. are trained again, the decisions are the uninterrupted run's.
    scripted["calls"] = 2
    meta = picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                       resume_from=ckpt)
    assert meta["resumed_from_epoch"] == 2
    assert meta["epochs_run"] == whole["epochs_run"] == 6
    assert [b["decision"] for b in meta["blocks"]] == [b["decision"] for b in whole["blocks"]]
    assert [h["epoch"] for h in meta["history"]] == list(range(1, 7))
    assert meta["best_epoch"] == whole["best_epoch"] == 6


def test_a_resumed_run_starts_from_the_checkpoints_weights_and_optimizer(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.1 * e for e in range(1, 9)]
    with pytest.raises(Stopped):
        picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                    time_left=lambda: 0.0)
    ckpt = config.DATA_DIR / "models" / f"picek-tiny{picek.CHECKPOINT_SUFFIX}"
    ck = torch.load(ckpt, map_location="cpu", weights_only=True)
    made = []

    def factory():
        from test_retrain_torch import TinyTimm
        made.append(TinyTimm(2))
        return made[-1]
    with pytest.raises(Stopped):                      # stop before any step of epoch 3
        picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                    model_factory=factory, resume_from=ckpt, should_stop=lambda: True)
    state = made[0].state_dict()
    assert all(torch.equal(state[k].cpu(), ck["model"][k]) for k in state)
    assert not all(torch.equal(state[k].cpu(), fresh_tiny_state()[k]) for k in state)


def fresh_tiny_state():
    from test_retrain_torch import TinyTimm
    return TinyTimm(2).state_dict()


def test_a_resumed_run_whose_best_came_before_the_resume_scores_it_again(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.50, 0.60, 0.10, 0.10, 0.10, 0.10, 0.10, 0.10]
    with pytest.raises(Stopped):
        picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                    should_stop=lambda: scripted["calls"] >= 3)
    scripted["calls"] = 2
    meta = picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                       resume_from=config.DATA_DIR / "models"
                       / f"picek-tiny{picek.CHECKPOINT_SUFFIX}")
    assert meta["best_epoch"] == 2 and meta["epochs_run"] == 4
    assert meta["temperature"] > 0                       # fitted on a fresh validation pass


def test_with_too_little_time_for_another_block_the_run_stops_at_the_boundary(
        conn, tmp_path, tiny, scripted):  # noqa: F811
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.1 * e for e in range(1, 9)]
    with pytest.raises(Stopped, match="at the block boundary after epoch 2"):
        picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                    time_left=lambda: 0.0)
    ck = torch.load(config.DATA_DIR / "models" / f"picek-tiny{picek.CHECKPOINT_SUFFIX}",
                    map_location="cpu", weights_only=True)
    assert ck["epoch"] == 2 and scripted["calls"] == 2       # no epoch of the next block


def test_a_checkpoint_of_another_run_is_refused(conn, tmp_path, tiny, scripted):  # noqa: F811
    from mycomap_vision.screening import Stopped
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.1 * e for e in range(1, 9)]
    with pytest.raises(Stopped):
        picek.train(conn, store, "large", "picek-tiny", rule_cfg(tiny), log=lambda s: None,
                    time_left=lambda: 0.0)
    ckpt = config.DATA_DIR / "models" / f"picek-tiny{picek.CHECKPOINT_SUFFIX}"
    with pytest.raises(ValueError, match="not a checkpoint of this run"):
        picek.train(conn, store, "large", "picek-tiny", replace(rule_cfg(tiny), seed=1),
                    log=lambda s: None, resume_from=ckpt)
    with pytest.raises(ValueError, match="not a checkpoint of this run"):
        picek.train(conn, store, "large", "picek-other", rule_cfg(tiny), log=lambda s: None,
                    resume_from=ckpt)


def test_a_fixed_length_run_is_unchanged(conn, tmp_path, tiny, scripted):  # noqa: F811
    store = seed_two_species(conn, tmp_path)
    scripted["f1"] = [0.3, 0.2, 0.1]
    blocks = []
    meta = picek.train(conn, store, "large", "picek-tiny", tiny, log=lambda s: None,
                       on_block=lambda p, s: blocks.append(s))
    assert meta["epochs_run"] == 3 and meta["epoch_rule"] is None and meta["blocks"] == []
    assert meta["stop_reason"] == "fixed length: 3 epochs" and blocks == []
    assert not (config.DATA_DIR / "models" / f"picek-tiny{picek.CHECKPOINT_SUFFIX}").exists()
