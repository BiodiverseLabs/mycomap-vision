"""The Danish recipe's epoch rule (declared 2026-10-13 so that their classifier is not
compared undertrained): at least 15 epochs, then 5-epoch blocks while the best validation
macro-F1 rises by 0.005 or more a block, never past 50. These tests need no torch: the rule,
the estimate, the launcher's resume checks and the trainer's per-block uploads."""

import json

import pytest

from mycomap_vision import aws, trainer
from mycomap_vision.replications.fungitastic import retrain as picek

RULE = picek.EpochRule(15, 50, 5, 0.005)


def curve(at_boundaries: dict[int, float], last: int = 50) -> list[float]:
    """A macro-F1 curve whose best so far is `at_boundaries[e]` after epoch e (flat between)."""
    out, best = [], 0.0
    for e in range(1, last + 1):
        best = at_boundaries.get(e, best)
        out.append(best)
    return out


# --- the rule ------------------------------------------------------------------------------

def test_the_spec_names_the_minimum_and_the_cap_and_the_block_and_gain_are_fixed():
    assert picek.parse_spec("fungitastic-beit-b384@15-50") == ("fungitastic-beit-b384", 50)
    assert picek.epoch_rule_of("fungitastic-beit-b384@15-50") == RULE
    assert picek.epoch_rule_of("fungitastic-beit-b384@15-50@440") == RULE
    assert picek.epoch_rule_of("fungitastic-beit-b384@15") is None       # @15 is a fixed 15
    assert picek.parse_spec("fungitastic-beit-b384@15") == ("fungitastic-beit-b384", 15)
    for bad in ("fungitastic-beit-b384@50-15", "fungitastic-beit-b384@15-",
                "fungitastic-beit-b384@-50", "fungitastic-beit-b384@0-50",
                "fungitastic-beit-b384@15-x"):
        with pytest.raises(ValueError):
            picek.parse_spec(bad)
    assert picek.PicekConfig(epoch_rule=RULE).resolved().epochs == 50


def test_the_rule_decides_at_the_minimum_then_every_five_epochs_and_at_the_cap():
    assert RULE.boundaries() == [15, 20, 25, 30, 35, 40, 45, 50]
    assert picek.EpochRule(15, 48, 5, 0.005).boundaries() == [15, 20, 25, 30, 35, 40, 45, 48]
    f1 = curve({1: 0.1})
    assert all(picek.block_decision(RULE, f1, e) is None
               for e in range(1, 51) if e not in RULE.boundaries())
    first = picek.block_decision(RULE, f1, 15)
    assert first["decision"] == "continue" and first["gain"] is None
    assert first["previous_best_val_macro_f1"] is None


def test_a_block_that_raises_the_best_by_half_a_point_or_more_continues():
    f1 = curve({15: 0.300, 20: 0.305, 25: 0.320})
    at20 = picek.block_decision(RULE, f1, 20)
    assert at20["decision"] == "continue"                 # exactly 0.005 counts as enough
    assert at20["gain"] == pytest.approx(0.005)
    assert at20["previous_best_val_macro_f1"] == 0.300 and at20["best_val_macro_f1"] == 0.305
    assert picek.block_decision(RULE, f1, 25)["decision"] == "continue"
    # 0.105 - 0.100 is 0.00499999... in floating point: still exactly half a point.
    edge = picek.block_decision(RULE, curve({15: 0.100, 20: 0.105}), 20)
    assert edge["decision"] == "continue" and edge["gain"] == pytest.approx(0.005)


def test_a_block_that_adds_less_than_half_a_point_stops_training():
    f1 = curve({15: 0.300, 20: 0.310, 25: 0.3149})
    d = picek.block_decision(RULE, f1, 25)
    assert d["decision"] == "stop" and d["gain"] == pytest.approx(0.0049)
    assert "under 0.005" in d["reason"] and d["decided_on"].startswith("validation macro-F1")


def test_the_gain_is_measured_on_the_best_so_far_not_the_last_epoch():
    # Epoch 20 is worse than epoch 17, but the best so far rose by 0.01 over the block.
    f1 = [0.2] * 14 + [0.30, 0.30, 0.31, 0.29, 0.29, 0.28]
    d = picek.block_decision(RULE, f1, 20)
    assert d["decision"] == "continue" and d["best_epoch"] == 17
    # A block whose epochs all score below the old best adds nothing.
    f1 += [0.305] * 5
    assert picek.block_decision(RULE, f1, 25)["decision"] == "stop"


def test_the_rule_stops_at_fifty_epochs_however_much_is_still_gained():
    f1 = [0.01 * e for e in range(1, 51)]
    for e in RULE.boundaries()[:-1]:
        assert picek.block_decision(RULE, f1, e)["decision"] == "continue"
    last = picek.block_decision(RULE, f1, 50)
    assert last["decision"] == "stop" and "cap of 50 epochs" in last["reason"]
    assert last["gain"] == pytest.approx(0.05)


def test_an_epoch_with_no_validation_score_counts_as_none():
    f1 = [None] * 20
    assert picek.block_decision(RULE, f1, 20)["decision"] == "stop"


# --- the estimate ------------------------------------------------------------------------------

def test_the_estimate_gives_min_expected_and_max_hours_and_checks_time_on_the_max():
    stages = trainer.plan_stages([], None, "r", ["fungitastic-beit-b384@15-50"])[:1]
    est = trainer.estimate(stages, 680_000)
    row = est["stages"][0]
    per_epoch = 680_000 / trainer.picek_rate("fungitastic-beit-b384", False) / 3600
    assert row["epoch_hours"] == pytest.approx(per_epoch, abs=0.01)
    assert row["hours"] == pytest.approx(50 * per_epoch, abs=0.1)          # ~160 h
    assert row["hours_min"] == pytest.approx(15 * per_epoch, abs=0.1)
    assert row["hours_expected"] == pytest.approx(25 * per_epoch, abs=0.1)
    assert row["epochs"] == {"min": 15, "expected": 25, "max": 50}
    assert est["total_hours"] > est["total_hours_expected"] > est["total_hours_min"]
    text = trainer.format_estimate(est)
    assert f"max 50 epochs {row['hours']} h" in text and "15-50 epochs, epoch rule" in text
    with pytest.raises(ValueError, match="Raise --max-hours"):
        aws.check_run_time(est, 100)              # 100 h covers the expected, not the most
    aws.check_run_time(est, est["total_hours"] + 1)


def test_a_resumed_rule_run_is_estimated_for_the_epochs_it_still_has_to_train():
    stages = trainer.plan_stages([], None, "r", ["fungitastic-beit-b384@15-50"])[:1]
    doc = {"stages": [{"kind": "picek", "name": stages[0].name, "checkpoint_epoch": 30},
                      {"kind": "embed", "name": "x", "checkpoint_epoch": 9}]}
    assert trainer.checkpoint_epochs(doc) == {stages[0].name: 30}
    full = trainer.estimate(stages, 680_000)["stages"][0]
    left = trainer.estimate(stages, 680_000, epochs_done={stages[0].name: 30})["stages"][0]
    assert left["hours"] == pytest.approx(20 * full["epoch_hours"], abs=0.1)
    assert left["hours_min"] == left["hours_expected"] == pytest.approx(0, abs=0.05)
    assert "30 done in its checkpoint" in left["stage"]


def test_a_rule_run_gets_disk_for_its_block_checkpoint():
    assert aws.picek_checkpoint_gb(["fungitastic-beit-b384@15"]) == 0
    assert aws.picek_checkpoint_gb(["fungitastic-beit-b384@15-50"]) == 4


# --- the time limit -------------------------------------------------------------------------

def test_the_deadline_says_how_long_is_left():
    now = [0.0]
    d = trainer.deadline(2, clock=lambda: now[0])
    assert not d() and d.seconds_left() == 7200
    now[0] = 7300
    assert d() and d.seconds_left() == 0
    assert trainer.deadline(None) is None


# --- resuming a release run --------------------------------------------------------------------

class Rel:
    id, release_hash = "v1", "f" * 64


def test_a_release_run_resumes_only_on_its_own_release():
    prev = {"dataset_release": {"dataset_release": "v1", "release_hash": "f" * 64}}
    aws.check_resume_release("r", prev, Rel())
    with pytest.raises(ValueError, match="resume it with --release v1"):
        aws.check_resume_release("r", prev, None)
    other = {"dataset_release": {"dataset_release": "v0", "release_hash": "0" * 64}}
    with pytest.raises(ValueError, match="resumes only on its own release"):
        aws.check_resume_release("r", other, Rel())
    with pytest.raises(ValueError, match="resumes only on its own release"):
        aws.check_resume_release("r", {}, Rel())
    aws.check_resume_release("r", {}, None)                 # a live-manifest run


# --- on the instance: every block goes to S3 before the next starts ----------------------------

class Files:
    """RunFiles over a dict of S3 keys."""

    def __init__(self, objects):
        self.objects = objects

    def fetch(self, key, dest):
        if key not in self.objects:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(self.objects[key])
        return True

    def keys(self, prefix):
        return [k for k in self.objects if k.startswith(prefix)]


def run_rule_job(conn, tmp_path, monkeypatch, resume=None, blocks=2):
    """run_job with the default Picek trainer replaced by one that reports `blocks` block
    boundaries through on_block and then writes a model (fake_picek_trainer)."""
    from test_models_and_scoreboard import seed_two_species
    from test_record_sources import record_sources_as
    from test_finetune import ft_loader
    from test_retrain_trainer import fake_picek_trainer

    from mycomap_vision import config
    store = seed_two_species(conn, tmp_path)
    record_sources_as(conn, monkeypatch)
    data_dir = tmp_path / "inst"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)
    events, seen = [], {}

    def default(*args, on_block=None, resume_from=None, time_left=None, **kw):
        seen.update(resume_from=resume_from, time_left=time_left)

        def run(conn, spec, name, models_dir):
            for b in range(blocks):
                path = models_dir / f"{name}{picek.CHECKPOINT_SUFFIX}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"block {b}".encode())
                decisions = [{"epoch": 15 + 5 * k, "decision": "continue"} for k in range(b + 1)]
                on_block(path, {"epochs_done": 15 + 5 * b, "best_epoch": 15 + 5 * b,
                                "best_val_macro_f1": 0.3 + 0.01 * b, "blocks": decisions,
                                "history": [{"epoch": e} for e in range(1, 16 + 5 * b)]})
                events.append(("block", b))
            return fake_picek_trainer(conn, spec, name, models_dir)
        return run
    monkeypatch.setattr(trainer, "_default_picek_trainer", default)

    def upload(path, key):
        body = path.read_bytes()
        events.append(("upload", key, json.loads(body) if key.endswith(".json") else body))
    out = trainer.run_job(conn, store, [], ["classifier"], upload, "r1",
                          loader=ft_loader, data_dir=data_dir, picek=["fungitastic-beit-b384@15-50"],
                          should_stop=trainer.deadline(100), resume=resume, log=lambda s: None)
    return out, events, seen


def test_each_block_checkpoint_and_progress_go_to_s3_before_the_next_block(conn, tmp_path,
                                                                          monkeypatch):
    out, events, seen = run_rule_job(conn, tmp_path, monkeypatch)
    name = "picek-fungitastic-beit-b384-r1"
    key = f"runs/r1/checkpoints/{name}{picek.CHECKPOINT_SUFFIX}"
    for b in range(2):
        at = events.index(("block", b))
        before = events[:at]
        ck = [i for i, e in enumerate(before) if e[0] == "upload" and e[1] == key]
        assert ck and before[ck[-1]][2] == f"block {b}".encode()
        prog = [e for e in before[ck[-1]:] if e[0] == "upload" and e[1] == "runs/r1/progress.json"]
        stage = next(s for s in prog[-1][2]["stages"] if s["kind"] == "picek")
        assert stage["checkpoint"] == key and stage["checkpoint_epoch"] == 15 + 5 * b
        assert len(stage["blocks"]) == b + 1 and stage["rule_decision"] == "continue"
    assert seen["time_left"] is not None and seen["resume_from"] is None
    assert out["state"] == "finished"


def test_a_resumed_run_starts_from_the_latest_block_checkpoint_in_s3(conn, tmp_path,
                                                                    monkeypatch):
    name = "picek-fungitastic-beit-b384-r1"
    key = f"runs/r1/checkpoints/{name}{picek.CHECKPOINT_SUFFIX}"
    progress = {"run_id": "r1", "state": "stopped",
                "stages": [{"kind": "picek", "name": name, "spec": "fungitastic-beit-b384@15-50",
                            "status": "stopped", "checkpoint": key, "checkpoint_epoch": 25}]}
    files = Files({"runs/r1/progress.json": json.dumps(progress).encode(),
                   key: b"the block after epoch 25"})
    out, events, seen = run_rule_job(conn, tmp_path, monkeypatch, resume=files, blocks=0)
    assert seen["resume_from"] is not None
    assert seen["resume_from"].read_bytes() == b"the block after epoch 25"
    assert out["state"] == "finished"
