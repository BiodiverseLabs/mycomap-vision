"""The not-a-fungus gate in `mv paper-score` (paper E10): a find the gate refuses loses its
answer and counts as wrong in the served numbers (the plain row stays gate-off), the claim
is checked on the share of finds refused, and the OOD set's rows are reported on the
headline kinds, with lichens and pathogens on their own, split by training labels."""

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
from test_fungusgate import TEST_GATE  # noqa: E402
from test_paper_score import config, run, world  # noqa: E402,F401

from mycomap_vision import fungusgate, paper_gate  # noqa: E402
from mycomap_vision import paper_outcomes as po  # noqa: E402
from mycomap_vision.fungusgate import FungusGate  # noqa: E402
from mycomap_vision.fungusgate_tune import sha256_file  # noqa: E402
from mycomap_vision.paper_outcomes import Find  # noqa: E402

FUNGUS_V, BIRD_V = [1.0, 0.0], [0.0, 1.0]
REPS = 200


def find(key, observer="o1"):
    return Find(key=key, observer=observer, species="A x", genus="A", family="F",
                name_kind="formal", depth=5, photos=1, attrs={}, near_duplicate={})


def saved_gate(tmp_path) -> tuple[Path, str]:
    g = FungusGate(**TEST_GATE)
    d = tmp_path / "gate"
    g.save(d)
    return d, g.fingerprint()


def vectors_file(path: Path, rows: dict[str, list[float]]) -> Path:
    np.savez(path, keys=np.array(list(rows)), frozen=np.array(list(rows.values()), np.float16),
             ft=np.array(list(rows.values()), np.float16))
    return path


def test_a_refused_find_is_wrong_in_every_cell_of_the_served_twin_only():
    right = {**{c: True for c in po.CELLS}, "species_strict@1": True, "answered": True}
    right["genus_sl@10"] = None                     # a cell it can't be scored on stays so
    outcome = {"1": dict(right), "2": dict(right)}
    verdicts = {"1": {"verdict": fungusgate.NOT_FUNGUS}, "2": {"verdict": fungusgate.UNSURE}}
    twin = paper_gate.served(outcome, verdicts)
    assert twin["1"]["species_strict@1"] is False and twin["1"]["answered"] is False
    assert twin["1"]["genus_sl@10"] is None
    assert twin["2"] == right                       # not sure keeps its names
    assert outcome["1"]["species_strict@1"] is True  # the gate-off row is untouched


def test_a_find_is_a_fungus_when_any_photo_is_and_a_photo_without_a_vector_is_refused():
    g = FungusGate(**TEST_GATE)
    photos = {"1": [{"manifest_photo_id": 10}, {"manifest_photo_id": 11}],
              "2": [{"manifest_photo_id": 12}]}
    vec = {"10": np.array(BIRD_V), "11": np.array(FUNGUS_V), "12": np.array(BIRD_V)}
    v = paper_gate.judge_finds(g, photos, vec)
    assert v["1"]["verdict"] == fungusgate.FUNGUS
    assert v["2"]["verdict"] == fungusgate.NOT_FUNGUS
    del vec["12"]
    with pytest.raises(ValueError, match="no gate vector"):
        paper_gate.judge_finds(g, photos, vec)


@pytest.mark.parametrize("lost, stands", [(1, True), (2, False)])
def test_the_claim_stands_only_when_at_most_half_a_percent_lose_their_answer(lost, stands):
    finds = [find(str(i), f"o{i % 7}") for i in range(200)]
    verdicts = {f.key: {"verdict": fungusgate.NOT_FUNGUS if i < lost else fungusgate.FUNGUS}
                for i, f in enumerate(finds)}
    rep = paper_gate.test_report(finds, verdicts, REPS, 1, 0.005)
    assert rep["lost"]["count"] == lost and rep["lost"]["n"] == 200
    assert rep["claim"]["stands"] is stands


def test_ood_rows_headline_everyday_and_in_scope_split_by_training_labels():
    g = FungusGate(**TEST_GATE)
    rows, vec = [], {}

    def add(n, kind, category, trains, v):
        for _ in range(n):
            name = f"p{len(rows)}.jpg"
            rows.append({"file": name, "kind": kind, "category": category,
                         "in_v1_labels": trains, "owner": f"u{len(rows) % 5}"})
            vec[name] = np.array(v)
    add(98, "everyday", "object", "no", BIRD_V)
    add(2, "everyday", "object", "no", FUNGUS_V)        # 2% of everyday let through
    add(50, "near-ood", "moss", "no", BIRD_V)
    add(10, "near-ood-fungus", "lichen", "yes", FUNGUS_V)
    add(4, "near-ood-fungus", "lichen", "no", BIRD_V)
    rep = paper_gate.ood_report(g, rows, vec, np.array([0.9, 0.95, 0.99]), REPS, 1, 0.02)
    assert rep["headline"]["photos"] == 150             # everyday + non-fungal near-OOD only
    assert rep["headline"]["named"]["count"] == 2
    assert rep["everyday"]["named"]["share"] == pytest.approx(0.02)
    assert rep["everyday"]["check"]["holds"] is True
    assert rep["in_scope"]["near-ood-fungus (trained on: yes)"]["named"]["count"] == 10
    assert rep["in_scope"]["near-ood-fungus (trained on: no)"]["refused"]["count"] == 4
    assert set(rep["by_category"]) == {"object", "moss", "lichen"}
    assert rep["headline"]["tnr_at_95_tpr"] == pytest.approx(148 / 150)
    with pytest.raises(ValueError, match="no gate vector"):
        paper_gate.ood_report(g, rows + [{"file": "x.jpg", "kind": "everyday"}], vec,
                              np.array([0.9]), REPS, 1, 0.02)


def test_a_gate_or_vectors_other_than_the_pinned_ones_are_refused(tmp_path):
    d, fp = saved_gate(tmp_path)
    assert paper_gate.load_gate({"dir": str(d), "fingerprint": fp}, Path).fingerprint() == fp
    with pytest.raises(ValueError, match="fingerprint"):
        paper_gate.load_gate({"dir": str(d), "fingerprint": "0" * 16}, Path)
    f = vectors_file(tmp_path / "v.npz", {"1": FUNGUS_V})
    with pytest.raises(ValueError, match="sha256"):
        paper_gate.pinned({"file": str(f), "sha256": "0" * 64}, Path)


def test_paper_score_serves_through_the_gate_beside_the_gate_off_numbers(world, tmp_path):
    d, fp = saved_gate(tmp_path)
    rows = {str(p): (BIRD_V if str(k) == "5" else FUNGUS_V) for p, k in world["base"].db.execute(
        "select manifest_photo_id, record_key from photos where included = 1")}
    tv = vectors_file(tmp_path / "test-vectors.npz", rows)
    man = tmp_path / "ood.csv"
    with open(man, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, ["file", "kind", "category", "in_v1_labels", "owner"])
        w.writeheader()
        w.writerows([{"file": "e.jpg", "kind": "everyday", "category": "object",
                      "in_v1_labels": "no", "owner": "u"}])
    ov = vectors_file(tmp_path / "ood.npz", {"e.jpg": BIRD_V})
    cfg = config(tmp_path, ["5", "6"], gate={
        "dir": str(d), "fingerprint": fp,
        "test_vectors": {"file": str(tv), "sha256": sha256_file(tv)},
        "ood": {"vectors": {"file": str(ov), "sha256": sha256_file(ov)},
                "manifest": {"file": str(man), "sha256": sha256_file(man)}}})
    cfg["hypotheses"]["B"] = [{"id": "gate", "kind": "mcnemar", "a": "vision-photos",
                               "b": "vision-photos served", "cell": "species_strict@1"}]
    rep = run(world, tmp_path, "val", cfg)
    comps = rep["standard"]["comparators"]
    off = comps["vision-photos"]["primary"]["all"]["table"]["species strict"]["top1"]
    on = comps["vision-photos served"]["primary"]["all"]["table"]["species strict"]["top1"]
    assert (off["right"], on["right"], on["n"]) == (2, 1, 2)       # find 5 refused
    assert rep["gate"]["test"]["lost"]["count"] == 1
    assert rep["gate"]["test"]["claim"]["stands"] is False
    assert rep["gate"]["ood"]["everyday"]["refused"]["count"] == 1
    assert rep["hypotheses"]["B"]["gate"]["only_a"] == 1
    assert set(rep["gate"]["served_comparators"]) == {"vision-photos served", "vision-mean served"}
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Not-a-fungus gate" in md and "### vision-photos served" in md
    assert json.loads((tmp_path / "out" / "report.json").read_text())["gate"]["gate"]["fingerprint"] == fp


def test_a_gate_block_names_exactly_one_source_of_test_vectors(tmp_path):
    from mycomap_vision import paper_score
    for extra in ({}, {"test_vectors": {"file": "v", "sha256": "x"}, "embed_with": "m1"}):
        cfg = config(tmp_path, ["5"], gate={"dir": "g", "fingerprint": "x", **extra})
        with pytest.raises(ValueError, match="test_vectors"):
            paper_score.check_config(cfg)


def test_a_sealed_run_embeds_the_gate_vectors_itself_and_records_them(world, tmp_path):
    """embed_with: the run lists every scored photo's local copy, embeds them through the
    served model's fork, keeps the vectors beside the report and records their sha256."""
    d, fp = saved_gate(tmp_path)
    keys = {str(p): str(k) for p, k in world["base"].db.execute(
        "select manifest_photo_id, record_key from photos where included = 1")}
    listed = {}

    def fake_embed(backbone, lst, out):
        rows = [l.split("\t") for l in Path(lst).read_text(encoding="utf-8").splitlines()]
        listed.update(backbone=backbone, keys={k for k, _ in rows},
                      paths=all(Path(p).is_file() for _, p in rows))
        vectors_file(out, {k: (BIRD_V if keys[k] == "5" else FUNGUS_V) for k, _ in rows})
        return len(rows)
    cfg = config(tmp_path, ["5", "6"], gate={"dir": str(d), "fingerprint": fp,
                                             "embed_with": "served-model"})
    rep = run(world, tmp_path, "val", cfg, gate_embed=fake_embed)
    val_photos = {p for p, k in keys.items() if k in ("5", "6")}
    assert listed["backbone"] == "served-model" and listed["paths"]
    assert listed["keys"] == val_photos                       # every scored photo, no other
    tv = rep["gate"]["test_vectors"]
    assert tv["sha256"] == sha256_file(tmp_path / "out" / "gate" / "gate-vectors.npz")
    assert tv["made_by"].startswith("embedded by this run")
    assert rep["gate"]["test"]["lost"]["count"] == 1
