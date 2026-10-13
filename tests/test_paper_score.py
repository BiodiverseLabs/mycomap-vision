"""`mv paper-score` end to end on releases made by the builder (release_fixture): scored on
val and on a draft test release, with Vision answers from stored vectors, the comparisons
lane's iNat and expert files and a published model's answers; and the freeze guard that
keeps the sealed test closed until the tag, the tagged config and the frozen code agree."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent / "replications" / "fungitastic"))
from release_fixture import BASE, TEST, release_world  # noqa: E402
from test_models_and_scoreboard import Const  # noqa: E402

from mycomap_vision import backbone_screen, paper_score  # noqa: E402
from mycomap_vision import paper_outcomes as po  # noqa: E402
from mycomap_vision.replications.fungitastic import dataset  # noqa: E402

REPS = 200


@pytest.fixture
def world(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    w = release_world(tmp_path, monkeypatch)
    base, test = dataset.load(BASE, True), dataset.load(TEST, True)
    photos = [int(r[0]) for r in base.db.execute("select manifest_photo_id from photos")]
    photos += [int(r[0]) for r in test.db.execute("select manifest_photo_id from photos")]
    backbone_screen.embed(w["conn"], {"photos": photos}, ["m1"], loader=lambda s: Const(s),
                          log=lambda s: None)
    return {**w, "base": base, "test": test}


def e14(path, keys, right):
    lines = []
    for k in keys:
        hit = {f"top{n}": right for n in po.K}
        lines.append({"record_key": k, "photos": 1, "photo_ids": [], "px": 1024,
                      "truth": {"species": 1, "genus": 2, "formal_species": True, "has_genus": True},
                      "vision": {"species": [[1, "A x", 90]], "genus": [[2, "A", 90]]},
                      "combined": {"species": [], "genus": []},
                      "answered": {"vision": True, "combined": False},
                      "vision_hits": {"species": hit, "genus": hit},
                      "combined_hits": {"species": {f"top{n}": False for n in po.K},
                                        "genus": {f"top{n}": False for n in po.K}}})
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return str(path)


def e31(path, keys):
    lines = [{"record_key": k, "status": "ok", "community": {"present": False},
              "ids": [{"role": "identifier", "tier": "1", "before_cutoff": True,
                       "species_right": True, "genus_right": True}]} for k in keys]
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return str(path)


def config(tmp_path, keys, **extra):
    return {"format": 1, "bootstrap": {"reps": REPS, "seed": 1},
            "comparators": [
                {"id": "vision-photos", "kind": "vision", "backbone": "m1", "method": "nearest"},
                {"id": "vision-mean", "kind": "vision", "backbone": "m1",
                 "method": "nearest+mean"},
                {"id": "inat-vision", "kind": "inat", "score": "vision",
                 "file": e14(tmp_path / "e14.jsonl", keys, False)},
                {"id": "experts", "kind": "expert", "tier": "1", "view": "best",
                 "file": e31(tmp_path / "e31.jsonl", keys)}],
            "hypotheses": {"A": [
                {"id": "H1a", "kind": "mcnemar", "a": "vision-photos", "b": "inat-vision",
                 "cell": "species_strict@1", "name_kind": "formal"},
                {"id": "H4", "kind": "mcnemar", "a": "vision-photos", "b": "experts",
                 "cell": "species_strict@1", "has": ["experts"], "alternative": "two-sided"}]},
            "slices": {"attrs": ["season", "source"], "min_finds": 1}, **extra}


def run(world, tmp_path, test, cfg, **kw):
    body = json.dumps(cfg).encode()
    return paper_score.run(world["conn"], world["base"], test, json.loads(body), body,
                           tmp_path / "out", log=lambda s: None, **kw)


def test_scoring_val_against_train_end_to_end(world, tmp_path):
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))
    assert rep["scored"]["split"] == "val" and rep["scored"]["reference_splits"] == ["train"]
    assert rep["finds"]["scored"] == 2 and rep["post_hoc"] is False
    t = rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]
    assert t["table"]["species strict"]["top1"]["n"] == 2
    assert t["table"]["species strict"]["top1"]["rate"] == 1.0     # red A x, dark B y
    inat = rep["standard"]["comparators"]["inat-vision"]["primary"]["all"]
    assert inat["table"]["species strict"]["top1"]["rate"] == 0.0
    h1a = rep["hypotheses"]["A"]["H1a"]
    assert (h1a["n"], h1a["only_a"], h1a["only_b"]) == (2, 2, 0)
    assert h1a["p_test"] == pytest.approx(0.25)
    assert rep["hypotheses"]["A"]["H4"]["n"] == 2
    pred = (tmp_path / "out" / "predictions" / "vision-photos.jsonl").read_text().splitlines()
    assert [json.loads(x)["species"][0] for x in pred] == ["A x", "B y"]
    assert rep["inputs"]["vision-photos"]["predictions_sha256"]
    assert rep["inputs"]["inat-vision"]["sha256"] and rep["inputs"]["inat-vision"]["lines"] == 2
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| species strict |" in md and "## photos only" in md and "Family A" in md
    assert set(rep["slices"]) == {"season", "source"}


def test_a_draft_test_release_is_scored_against_train_and_val(world, tmp_path):
    rep = run(world, tmp_path, world["test"], config(tmp_path, ["7", "8"]))
    assert rep["scored"]["split"] == "test"
    assert rep["scored"]["reference_splits"] == ["train", "val"]
    assert rep["scored"]["test_release"]["dataset_release"] == TEST
    bands = rep["standard"]["comparators"]["vision-photos"]["primary"]["all"][
        "species_by_reference_band"]
    # A x: records 1, 2, mo:9 (train) + 5 (val) = 4; B y: 3, 4 + 6 = 3: both in 1-4.
    assert bands["1-4"]["top1"]["n"] == 2 and rep["guard"]["sealed"] is False


def test_errata_records_leave_the_reference(world, tmp_path):
    errata = tmp_path / "errata.tsv"
    errata.write_text("kind\tkey\treason\tnote\nrecord\t1\tmo_twin_as_inat\t\n"
                      "record\t2\tmo_twin_as_inat\t\nrecord\tmo:9\tx\t\n", encoding="utf-8")
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]), errata_path=errata)
    assert rep["errata"]["reference_records_left_out"] == 3
    bands = rep["standard"]["comparators"]["vision-photos"]["primary"]["all"][
        "species_by_reference_band"]
    assert bands["0"]["top1"]["n"] == 1                 # A x has no reference left
    assert bands["0"]["top1"]["rate"] == 0.0            # and counts as wrong


def test_a_near_duplicate_find_leaves_the_primary_numbers(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"], near_duplicates={"backbone": "m1", "cosine": 0.99})
    rep = run(world, tmp_path, "val", cfg)
    # Const vectors depend only on colour: each val photo has a near-identical train photo.
    assert rep["standard"]["near_duplicates"]["near_duplicate_finds"] == 2
    assert rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]["finds"] == 0
    assert rep["standard"]["comparators"]["vision-photos"]["with_near_duplicates"]["all"][
        "finds"] == 2


def test_near_duplicate_flags_by_bytes_dhash_and_cosine():
    photos = {"a": [{"manifest_photo_id": 1, "original_sha256": "s1", "photo_key": "p1"}],
              "b": [{"manifest_photo_id": 2, "original_sha256": "s2", "photo_key": "p2"}],
              "c": [{"manifest_photo_id": 3, "original_sha256": "s3", "photo_key": "p3"}]}
    nd = paper_score.near_duplicates(photos, {"s1"}, {"p2"}, {1: 0.2, 2: 0.3, 3: 0.995}, 0.99)
    assert nd == {"a": {"bytes": True, "dhash": False, "cosine": False},
                  "b": {"bytes": False, "dhash": True, "cosine": False},
                  "c": {"bytes": False, "dhash": False, "cosine": True}}


def test_max_cosine_matches_the_full_matrix_whatever_the_chunks():
    rng = np.random.default_rng(0)
    q, r = rng.normal(size=(37, 8)), rng.normal(size=(101, 8))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    r /= np.linalg.norm(r, axis=1, keepdims=True)
    want = (q @ r.T).max(axis=1)
    assert np.allclose(paper_score.max_cosine(q, r, q_chunk=5, r_chunk=7), want, atol=1e-6)


def test_place_methods_see_the_public_point_never_the_true_one():
    from mycomap_vision.evaluate import Record
    recs = [Record("1", "A x", "A", "F", None, "o", [0], latitude=47.123, longitude=-122.456),
            Record("2", "A x", "A", "F", None, "o", [1], latitude=40.0, longitude=-100.0)]
    out = paper_score.with_public_points(recs, {"1": (47.1, -122.5)})
    assert (out[0].latitude, out[0].longitude) == (47.1, -122.5)
    assert (out[1].latitude, out[1].longitude) == (None, None)     # no public point: none


def test_a_bad_config_is_refused():
    with pytest.raises(ValueError, match="format"):
        paper_score.check_config({"format": 2})
    with pytest.raises(ValueError, match="unique"):
        paper_score.check_config({"format": 1, "comparators": [
            {"id": "a", "kind": "inat", "file": "x"}, {"id": "a", "kind": "inat", "file": "x"}]})
    with pytest.raises(ValueError, match="unknown comparator"):
        paper_score.check_config({"format": 1, "comparators": [], "hypotheses": {
            "A": [{"id": "H1a", "a": "nope"}]}})


# --- the freeze guard --------------------------------------------------------------------------

def git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def research(tmp_path):
    repo = tmp_path / "research"
    (repo / "paper").mkdir(parents=True)
    git(repo, "init", "-q")
    git(repo, "config", "user.email", "t@example.org")
    git(repo, "config", "user.name", "t")
    body = json.dumps({"format": 1, "code_commit": "abcdef1234567"}).encode()
    (repo / "paper" / "analysis-v1.json").write_bytes(body)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "plan")
    return repo, body


def test_val_and_drafts_need_no_tag_but_never_take_allow_code(research):
    repo, body = research
    assert paper_score.guard(False, body, json.loads(body), repo, "1234567")["sealed"] is False
    with pytest.raises(ValueError, match="only for the sealed test"):
        paper_score.guard(False, body, json.loads(body), repo, "1234567", allow_code="1234567")


def test_the_sealed_test_stays_closed_without_the_freeze_tag(research):
    repo, body = research
    with pytest.raises(paper_score.FrozenError, match="no tag"):
        paper_score.guard(True, body, json.loads(body), repo, "abcdef1")


def test_the_sealed_test_opens_only_with_the_tagged_config_and_the_frozen_code(research):
    repo, body = research
    git(repo, "tag", paper_score.TAG)
    ok = paper_score.guard(True, body, json.loads(body), repo, "abcdef1")
    assert ok["sealed"] and not ok["post_hoc"]
    other = body.replace(b"}", b', "x": 1}')
    with pytest.raises(paper_score.FrozenError, match="not the one tagged"):
        paper_score.guard(True, other, json.loads(other), repo, "abcdef1")
    with pytest.raises(paper_score.FrozenError, match="not the frozen paper-score commit"):
        paper_score.guard(True, body, json.loads(body), repo, "9999999")
    with pytest.raises(paper_score.FrozenError, match="committed code"):
        paper_score.guard(True, body, json.loads(body), repo, "abcdef1-dirty")
    late = paper_score.guard(True, body, json.loads(body), repo, "9999999", allow_code="9999999")
    assert late["post_hoc"] is True


def test_a_sealed_test_release_is_refused_before_anything_is_read(world, tmp_path, monkeypatch,
                                                                  research):
    repo, _ = research
    world["test"].manifest["draft"] = False                 # as the real v1-test will be
    cfg = config(tmp_path, ["7", "8"])
    called = []
    monkeypatch.setattr(paper_score, "build_finds", lambda *a, **k: called.append(1))
    with pytest.raises(paper_score.FrozenError):
        run(world, tmp_path, world["test"], cfg, research_dir=repo)
    assert not called and not (tmp_path / "out").exists()


class SeesPlace:
    """A place method stand-in that records the point it is given for each find."""
    needs_context = True
    seen: list = []

    def fit(self, vectors, index, records=None):
        self.n = len(index.species)

    def species_scores(self, query, context=None):
        SeesPlace.seen.append((context.latitude, context.longitude))
        return np.zeros(self.n, dtype=np.float32)


def test_a_place_method_is_given_the_public_point_never_the_true_one(world, tmp_path,
                                                                     monkeypatch):
    from mycomap_vision import evaluate
    monkeypatch.setitem(evaluate.METHODS, "sees-place", SeesPlace)
    monkeypatch.setattr(paper_score, "public_points",
                        lambda rel: {"5": (47.5, -122.5), "6": (None, None)})
    SeesPlace.seen = []
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"].append({"id": "vision-place", "kind": "vision", "backbone": "m1",
                               "method": "sees-place", "place": True})
    run(world, tmp_path, "val", cfg)
    # The release's true point is (47.0, -122.0) for every record of the fixture.
    assert sorted(SeesPlace.seen, key=str) == sorted([(47.5, -122.5), (None, None)], key=str)
