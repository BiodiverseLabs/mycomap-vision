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
KEY = __import__("mycomap_vision.name_equiv", fromlist=["x"]).ANSWER_KEY


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


def e14(path, keys, right, answer=None):
    lines = []
    for k in keys:
        hit = {f"top{n}": right for n in po.K}
        lines.append({"record_key": k, "photos": 1, "photo_ids": [], "px": 1024,
                      "truth": {"species": 1, "genus": 2, "formal_species": True, "has_genus": True},
                      # iNat's own answer: the truth's name when right, another when not
                      "vision": {"species": [[1, answer or (TRUTH[k] if right else "Z z"), 90]],
                                 "genus": [[2, (answer or TRUTH[k] if right else "Z")[0], 90]]},
                      "combined": {"species": [], "genus": []},
                      "answered": {"vision": True, "combined": False},
                      "vision_hits": {"species": hit, "genus": hit},
                      "combined_hits": {"species": {f"top{n}": False for n in po.K},
                                        "genus": {f"top{n}": False for n in po.K}}})
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return str(path)


TRUTH = {"5": "A x", "6": "B y", "7": "A x", "8": "B y"}     # release_fixture's names


def e31(path, keys):
    lines = [{"record_key": k, "status": "ok", "format": 2, "community": {"present": False},
              "ids": [{"role": "identifier", "tier": "1", "before_cutoff": True,
                       "species_taxon": {"id": 1, "name": TRUTH[k]},
                       "genus_taxon": {"id": 2, "name": TRUTH[k].split()[0]},
                       "species_right": True, "genus_right": True}]} for k in keys]
    path.write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    return str(path)


def config(tmp_path, keys, **extra):
    return {"format": 1, "answer_key": KEY, "bootstrap": {"reps": REPS, "seed": 1},
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
    err = rep["errors"]["comparators"]
    assert set(err) == {"vision-photos", "vision-mean", "inat-vision"}   # experts carry no names
    assert err["vision-photos"]["confused_pairs"]["wrong_answers"] == 0


def test_a_draft_test_release_is_scored_against_train_and_val(world, tmp_path):
    rep = run(world, tmp_path, world["test"], config(tmp_path, ["7", "8"]))
    assert rep["scored"]["split"] == "test"
    assert rep["scored"]["reference_splits"] == ["train", "val"]
    assert rep["scored"]["test_release"]["dataset_release"] == TEST
    bands = rep["standard"]["comparators"]["vision-photos"]["primary"]["all"][
        "species_by_reference_band"]
    # A x: records 1, 2, mo:9 (train) + 5 (val) = 4; B y: 3, 4 + 6 = 3: both in 1-4.
    assert bands["1-4"]["top1"]["n"] == 2 and rep["guard"]["sealed"] is False


def test_an_erratum_given_to_the_release_leaves_the_reference(world, tmp_path):
    from mycomap_vision import errata
    path = tmp_path / "errata.tsv"
    path.write_text("kind\tkey\treason\tnote\nrecord\t1\tmo_twin_as_inat\t\n"
                    "record\t2\tmo_twin_as_inat\t\nrecord\tmo:9\tmo_twin_as_inat\t\n",
                    encoding="utf-8")
    world["base"].use_errata(errata.read(path))
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))
    assert rep["errata"]["reference_records_left_out"] == 3
    assert rep["errata"]["sha256"] == errata.read(path).sha256
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


def test_a_vision_comparator_with_an_unknown_method_is_refused_before_any_scoring():
    with pytest.raises(ValueError, match=r"unknown method 'nearest\+maen'"):
        paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": [
            {"id": "v", "kind": "vision", "backbone": "m1", "method": "nearest+maen"}]})
    paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": [
        {"id": "v", "kind": "vision", "backbone": "m1", "method": "nearest+mean"}]})


def test_a_bad_config_is_refused():
    with pytest.raises(ValueError, match="format"):
        paper_score.check_config({"format": 2})
    with pytest.raises(ValueError, match="unique"):
        paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": [
            {"id": "a", "kind": "inat", "file": "x"}, {"id": "a", "kind": "inat", "file": "x"}]})
    with pytest.raises(ValueError, match="unknown comparator"):
        paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": [], "hypotheses": {
            "A": [{"id": "H1a", "kind": "mcnemar", "a": "nope"}]}})


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


def test_a_config_differing_only_in_line_endings_is_refused_and_says_so(research):
    repo, _ = research
    body = json.dumps({"format": 1, "code_commit": "abcdef1234567"}, indent=1).encode()
    (repo / "paper" / "analysis-v1.json").write_bytes(body)
    git(repo, "commit", "-q", "-am", "multi-line plan")
    git(repo, "tag", paper_score.TAG)
    crlf = body.replace(b"\n", b"\r\n")                       # the same text, CRLF endings
    assert crlf != body
    with pytest.raises(paper_score.FrozenError, match="line endings differ"):
        paper_score.guard(True, crlf, json.loads(crlf), repo, "abcdef1")



def test_a_crlf_copy_of_the_config_has_the_sha_git_stores_everywhere_it_is_used(world, tmp_path):
    """Research pins *.json to LF. A copy rewritten with CRLF (an editor, a Windows builder)
    gives the same bytes, sha256, report config_sha256 and checkpoint stamps as the LF file a
    clean checkout gives, so a run on it resumes the LF run's checkpoints."""
    lf = tmp_path / "analysis.json"
    lf.write_bytes(json.dumps(config(tmp_path, ["5", "6"]), indent=1).encode())
    crlf = tmp_path / "analysis-crlf.json"
    crlf.write_bytes(lf.read_bytes().replace(b"\n", b"\r\n"))
    assert crlf.read_bytes() != lf.read_bytes()
    doc_lf, body_lf = paper_score.load_config(lf)
    doc_crlf, body_crlf = paper_score.load_config(crlf)
    assert body_crlf == body_lf == lf.read_bytes() and doc_crlf == doc_lf
    git_sha = paper_score.sha256_file(lf)
    assert git_sha in paper_score.config_sha_line(crlf, body_crlf)
    assert "CRLF" in paper_score.config_sha_line(crlf, body_crlf)
    assert "CRLF" not in paper_score.config_sha_line(lf, body_lf)
    first = paper_score.run(world["conn"], world["base"], "val", doc_lf, body_lf,
                            tmp_path / "out", log=lambda s: None)
    logged = []
    again = paper_score.run(world["conn"], world["base"], "val", doc_crlf, body_crlf,
                            tmp_path / "out", log=logged.append, resume=True)
    assert first["config_sha256"] == again["config_sha256"] == git_sha
    assert sum("resumed from its checkpoint" in m for m in logged) == 2


def test_a_carriage_return_outside_a_crlf_ending_is_refused(tmp_path):
    path = tmp_path / "analysis.json"
    path.write_bytes(b'{"format": 1,\r "answer_key": "x"}')
    with pytest.raises(ValueError, match="carriage return"):
        paper_score.load_config(path)


def test_a_crlf_copy_of_the_tagged_config_opens_the_sealed_guard(research, tmp_path,
                                                                    monkeypatch):
    repo, _ = research
    body = json.dumps({"format": 1, "code_commit": "abcdef1234567"}, indent=1).encode()
    (repo / "paper" / "analysis-v1.json").write_bytes(body)
    git(repo, "commit", "-q", "-am", "multi-line plan")
    git(repo, "tag", paper_score.TAG)
    copy = tmp_path / "tagged-crlf.json"
    copy.write_bytes(body.replace(b"\n", b"\r\n"))
    monkeypatch.setattr(paper_score, "check_config", lambda doc: None)
    doc, read = paper_score.load_config(copy)
    g = paper_score.guard(True, read, doc, repo, "abcdef1")
    assert g["sealed"] is True and g["config_sha256"] == paper_score.sha256_bytes(body)



# --- fields filled after the tag (post_freeze; PLAN §8) -------------------------------------------

def post_freeze_plan():
    """A tagged plan with five fields left for after the tag: a test-time output and four
    declared-after-freeze parts (a model, its calibration, a list and the list's sha256)."""
    return {"format": 1, "code_commit": "abcdef1234567", "top_k": 10,
            "comparators": [
                {"id": "inat-vision", "kind": "inat", "file": "TBD: e14/v1-test/finds.jsonl"},
                {"id": "public-v1.1", "kind": "vision", "backbone": "TBD: E34 retrain",
                 "method": "nearest", "calibration": "TBD: refit on val"}],
            "curation_v1_1": {"exclusions": {"file": "TBD at the cut-off", "sha256": "TBD"}},
            "freeze": {"models": {"m1": {"weights": "aa"}}, "code_commit": "abcdef1234567"},
            "post_freeze": [
                {"path": "comparators/inat-vision/file", "rule": "E14 at scoring",
                 "kind": "test_output"},
                {"path": "comparators/public-v1.1/backbone", "rule": "the E34 retrain",
                 "kind": "declared_after_freeze"},
                {"path": "comparators/public-v1.1/calibration", "rule": "refit on val",
                 "kind": "declared_after_freeze"},
                {"path": "curation_v1_1/exclusions/file", "rule": "E34's list at the cut-off",
                 "kind": "declared_after_freeze"},
                {"path": "curation_v1_1/exclusions/sha256", "rule": "that list's sha256",
                 "kind": "declared_after_freeze"}]}


def tag_plan(repo, doc):
    body = json.dumps(doc, indent=1).encode()
    (repo / "paper" / "analysis-v1.json").write_bytes(body)
    git(repo, "commit", "-q", "-am", "the plan to freeze")
    git(repo, "tag", paper_score.TAG)
    return body


def filled_plan():
    doc = post_freeze_plan()
    doc["comparators"][0]["file"] = "e14/v1-test/finds.jsonl"
    doc["comparators"][1]["backbone"] = "bioclip-2-336-ft-e34"
    doc["comparators"][1]["calibration"] = {"species": {"temperature": 0.05}}
    doc["freeze"]["models"]["bioclip-2-336-ft-e34"] = {"weights": "bb"}   # a part gained
    return doc


def sealed_guard(repo, doc):
    body = json.dumps(doc, indent=1).encode()
    return paper_score.guard(True, body, doc, repo, "abcdef1")


def test_fields_named_post_freeze_may_be_filled_after_the_tag_and_each_is_recorded(research):
    from mycomap_vision.paper_freeze import canonical_sha
    repo, _ = research
    tagged = tag_plan(repo, post_freeze_plan())
    doc = filled_plan()
    g = sealed_guard(repo, doc)
    assert g["sealed"] is True and g["post_hoc"] is False
    assert g["tagged_config_sha256"] == paper_score.sha256_bytes(tagged)
    assert g["config_sha256"] != g["tagged_config_sha256"]
    rec = {r["path"]: r for r in g["post_freeze"]}
    assert [r["path"] for r in g["post_freeze"]] == [e["path"] for e in doc["post_freeze"]]
    assert rec["comparators/inat-vision/file"]["kind"] == "test_output"
    assert rec["comparators/public-v1.1/calibration"]["value_sha256"] == \
        canonical_sha({"species": {"temperature": 0.05}})
    assert [r["filled"] for r in g["post_freeze"]] == [True, True, True, False, False]
    assert "value_sha256" not in rec["curation_v1_1/exclusions/file"]
    # the tagged config itself, nothing filled yet, opens too and is recorded the same way
    g0 = paper_score.guard(True, tagged, json.loads(tagged), repo, "abcdef1")
    assert not any(r["filled"] for r in g0["post_freeze"])


@pytest.mark.parametrize("change, named", [
    (lambda d: d.update(top_k=5), "top_k: changed since the tag"),
    (lambda d: d["comparators"][1].update(method="nearest+mean"),
     "comparators/public-v1.1/method"),
    (lambda d: d["comparators"][0].update(sha256="ab" * 32),          # a key added
     "comparators/inat-vision/sha256: not in the tagged config"),
    (lambda d: d["comparators"].pop(0), "comparators: 2 entries tagged, 1 now"),
    (lambda d: d["comparators"].reverse(), "entry 'inat-vision' tagged, 'public-v1.1' now"),
    (lambda d: d["post_freeze"][0].update(rule="anything goes"),
     "post_freeze/0/rule: changed since the tag"),
    (lambda d: d.pop("curation_v1_1"), "curation_v1_1: in the tagged config, missing now"),
])
def test_any_change_outside_the_post_freeze_fields_after_the_tag_is_refused(research, change,
                                                                           named):
    repo, _ = research
    tag_plan(repo, post_freeze_plan())
    doc = filled_plan()
    change(doc)
    with pytest.raises(paper_score.FrozenError, match="outside its post_freeze fields") as e:
        sealed_guard(repo, doc)
    assert named in str(e.value)


@pytest.mark.parametrize("change, named", [
    (lambda f: f["models"]["m1"].update(weights="cc"), "freeze/models/m1/weights: changed"),
    (lambda f: f.pop("code_commit"), "freeze/code_commit: frozen at the tag, missing now"),
])
def test_the_freeze_block_may_gain_parts_after_the_tag_but_never_change_or_lose_one(
        research, change, named):
    repo, _ = research
    tag_plan(repo, post_freeze_plan())
    doc = filled_plan()
    change(doc["freeze"])
    with pytest.raises(paper_score.FrozenError, match="outside its post_freeze fields") as e:
        sealed_guard(repo, doc)
    assert named in str(e.value)


def test_a_post_freeze_field_must_still_be_a_placeholder_in_the_tagged_config(research):
    repo, _ = research
    plan = post_freeze_plan()
    plan["comparators"][1]["calibration"] = {"species": {"temperature": 0.07}}   # decided
    body = tag_plan(repo, plan)
    with pytest.raises(paper_score.FrozenError, match="not a \"TBD...\" placeholder"):
        paper_score.guard(True, body, json.loads(body), repo, "abcdef1")


@pytest.mark.parametrize("entry, refused", [
    ({"path": "comparators/inat-vision/file", "kind": "test_output"}, "names the rule"),
    ({"path": "comparators/inat-vision/file", "rule": "r", "kind": "later"}, "kind is one of"),
    ({"path": "comparators/inat-vision/fiel", "rule": "r", "kind": "test_output"},
     "no such field"),
    ({"path": "comparators/inat-vision", "rule": "r", "kind": "test_output"}, "no such field"),
    ({"path": "code_commit", "rule": "r", "kind": "declared_after_freeze"},
     "never filled after the tag"),
    ({"path": "freeze/models", "rule": "r", "kind": "declared_after_freeze"},
     "never filled after the tag"),
    ({"path": "comparators/inat-vision/file", "rule": "r", "kind": "test_output"}, "named twice"),
    ({"path": "curation_v1_1/exclusions", "rule": "r", "kind": "declared_after_freeze"},
     "inside curation_v1_1/exclusions"),
])
def test_a_malformed_post_freeze_list_is_refused_on_every_run(entry, refused):
    doc = post_freeze_plan()
    doc["post_freeze"].append(entry)
    with pytest.raises(ValueError, match=refused):
        paper_score.check_post_freeze(doc)


def test_check_config_checks_a_post_freeze_placeholder_once_it_is_filled(tmp_path):
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"].append({"id": "public-v1.1", "kind": "vision", "backbone": "m1",
                               "method": "nearest",
                               "reference": "TBD: public minus human-excluded photos"})
    with pytest.raises(ValueError, match="reference is one of"):          # not declared later
        paper_score.check_config(cfg)
    cfg["post_freeze"] = [{"path": "comparators/public-v1.1/reference", "rule": "E34",
                           "kind": "declared_after_freeze"}]
    paper_score.check_config(cfg)                                          # pending: later
    cfg["comparators"][-1]["reference"] = "every photo, really"            # filled: checked
    with pytest.raises(ValueError, match="reference is one of"):
        paper_score.check_config(cfg)
    cfg["comparators"][-1]["reference"] = "public_trainable_only"
    paper_score.check_config(cfg)


def test_a_filled_post_freeze_file_is_recorded_by_its_own_sha256_and_must_match_its_pin(
        tmp_path):
    doc = filled_plan()
    lst = tmp_path / "exclusions.tsv"
    lst.write_bytes(b"photo\treason\n")
    doc["curation_v1_1"]["exclusions"]["file"] = str(lst)
    want = paper_score.sha256_file(lst)

    def records(d):
        return paper_score.post_freeze_records(post_freeze_plan(), d)
    doc["curation_v1_1"]["exclusions"]["sha256"] = want
    found = tmp_path / "finds.jsonl"
    found.write_bytes(b"{}\n")
    doc["comparators"][0]["file"] = str(found)
    rec = records(doc)
    paper_score.post_freeze_files(doc, rec, Path)
    by = {r["path"]: r for r in rec}
    assert by["curation_v1_1/exclusions/file"]["file_sha256"] == want
    assert by["comparators/inat-vision/file"]["file_sha256"] == paper_score.sha256_file(found)
    assert "file_sha256" not in by["comparators/public-v1.1/backbone"]
    doc["curation_v1_1"]["exclusions"]["sha256"] = "0" * 64              # pinned, not this file
    with pytest.raises(paper_score.FrozenError, match="is not the 000000000000 pinned"):
        paper_score.post_freeze_files(doc, records(doc), Path)
    doc["curation_v1_1"]["exclusions"]["file"] = str(tmp_path / "missing.tsv")
    with pytest.raises(paper_score.FrozenError, match="missing.tsv not found"):
        paper_score.post_freeze_files(doc, records(doc), Path)


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


def test_calibration_uses_the_val_temperature_given_never_a_refit(world, tmp_path):
    reports = []
    for t in (0.01, 10.0):
        cfg = config(tmp_path, ["5", "6"])
        cfg["comparators"][0]["calibration"] = {"species": {"temperature": t, "floor": 0.2},
                                                "genus": {"temperature": t}}
        reports.append(run(world, tmp_path, "val", cfg))
    sharp, flat = (r["calibration"]["vision-photos"]["species"] for r in reports)
    assert sharp["n"] == flat["n"] == 2
    conf = [r["rules"][0]["confident"] for r in (sharp, flat)]
    assert conf[0] > conf[1]              # T=0.01 is near certain, T=10 near uniform
    assert sharp["scores"]["nll"] < flat["scores"]["nll"]
    assert "likely_sets" in sharp and "likely_sets" not in reports[0]["calibration"][
        "vision-photos"]["genus"]                                     # no genus floor given
    assert "unknown_species" in sharp
    assert "vision-mean" not in reports[0]["calibration"]             # no calibration given
    assert reports[0]["calibration"]["inat-vision"]["species"]["n"] == 2   # iNat's own score


def test_deadly_taxa_come_from_the_config_list(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"], errors={"comparators": ["inat-vision"],
                                                 "deadly": {"genera": ["B"], "species": ["A x"]}})
    cfg["comparators"][2]["file"] = e14(tmp_path / "e14-ax.jsonl", ["5", "6"], False, "A x")
    rep = run(world, tmp_path, "val", cfg)
    d = rep["errors"]["comparators"]["inat-vision"]["deadly"]
    # iNat answered "A x" for both: A x right; B y (deadly genus) called a deadly A x.
    assert d["predicted_from"]["A x"]["same_taxon"] == 1
    assert d["predicted_from"]["B"]["other_deadly"] == 1


def test_photos_sent_at_640_px_are_embedded_apart_and_scored_against_the_stored_reference(
        world, tmp_path):
    seen = []

    def embed(b, ids, px):
        seen.append(px)
        paper_score.ensure_embedded(world["conn"], b, ids, px, loader=lambda s: Const(s),
                                    log=lambda m: None)
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"].append({"id": "vision-640", "kind": "vision", "backbone": "m1",
                               "method": "nearest", "photo_px": 640})
    rep = run(world, tmp_path, "val", cfg, embed_missing=embed)
    assert 640 in seen and None in seen
    n = world["conn"].execute("select count(*) from embeddings where backbone = 'm1-px640'"
                              ).fetchone()[0]
    assert n == 2                                      # only the two scored val photos
    t = rep["standard"]["comparators"]["vision-640"]["primary"]["all"]
    assert t["table"]["species strict"]["top1"]["n"] == 2
    # Nothing to embed the second time.
    assert paper_score.ensure_embedded(world["conn"], "m1", [1005, 1006], 640,
                                       loader=lambda s: Const(s)) == 0


def test_the_resized_backbone_never_enlarges_and_keeps_the_aspect():
    from PIL import Image

    class Seen:
        name, dim = "b", 2

        def prepare(self, im):
            return im.size
    r = paper_score.Resized(Seen(), 640)
    assert r.name == "b-px640"
    assert r.prepare(Image.new("RGB", (1024, 768))) == (640, 480)
    assert r.prepare(Image.new("RGB", (300, 200))) == (300, 200)


def test_a_reference_as_of_a_date_leaves_later_records_out(world, tmp_path):
    cfg = config(tmp_path, ["7", "8"])
    cfg["comparators"].append({"id": "as-of", "kind": "vision", "backbone": "m1",
                               "method": "nearest", "reference_as_of": "2026-01-01"})
    rep = run(world, tmp_path, world["test"], cfg)
    # By 2026-01-01 only records 1 (A x) and 3 (B y) were validated: both still answered.
    assert rep["standard"]["comparators"]["as-of"]["primary"]["all"]["table"][
        "species strict"]["top1"]["rate"] == 1.0
    cfg["comparators"][-1]["reference_as_of"] = "2025-12-31"          # before any record
    rep = run(world, tmp_path, world["test"], cfg)
    assert rep["standard"]["comparators"]["as-of"]["primary"]["all"]["table"][
        "species strict"]["top1"]["rate"] == 0.0


def test_leaving_out_the_finds_own_observer_refits_without_their_records(world, tmp_path):
    from mycomap_vision.evaluate import Record
    vec = np.array([[1, 0], [1, 0], [0, 1], [0.9, 0.1]], np.float32)
    ref = [Record("r1", "A x", "A", "F", "2026-01-01", "ann", [0]),
           Record("r2", "B y", "B", "F", "2026-01-01", "bo", [2])]
    find_ann = Record("q1", "A x", "A", "F", None, "ann", [3], observed_on="2026-05-01")
    find_cy = Record("q2", "A x", "A", "F", None, "cy", [1], observed_on="2026-05-01")
    plain = paper_score.vision_answers(vec, ref, [find_ann, find_cy], "nearest", "m1")
    left = paper_score.vision_answers_leave_out(vec, ref, [find_ann, find_cy], "nearest", "m1",
                                                "observer")
    assert plain["q1"]["species"][0] == "A x" and left["q1"]["species"][0] == "B y"
    assert left["q2"]["species"][0] == "A x"           # cy's own records were never there
    trip = paper_score.vision_answers_leave_out(vec, ref, [find_ann], "nearest", "m1", "trip")
    assert trip["q1"]["species"][0] == "A x"           # ann's record is from another day


def test_first_photo_and_photo_count_options(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"], slices={"attrs": ["photos"], "min_finds": 1})
    cfg["comparators"].append({"id": "first", "kind": "vision", "backbone": "m1",
                               "method": "nearest", "photos": "first"})
    rep = run(world, tmp_path, "val", cfg)
    assert rep["standard"]["comparators"]["first"]["primary"]["all"]["finds"] == 2
    assert set(rep["slices"]["photos"]) == {"1"}


def test_dhash_flags_come_from_the_test_releases_near_base_table(world, tmp_path, monkeypatch):
    test = world["test"]
    key7 = test.db.execute("select photo_key from photos where record_key = '7'").fetchone()[0]
    rep = run(world, tmp_path, test, config(tmp_path, ["7", "8"]))
    assert rep["dhash_flags"]["photos_flagged"] == 0          # a build without flags: empty
    monkeypatch.setattr(type(test), "near_base", lambda self: {
        key7: {"dhash": "ab", "base_photo_key": "base-1", "distance": 2}})
    rep = run(world, tmp_path, test, config(tmp_path, ["7", "8"]))
    assert rep["dhash_flags"]["photos_flagged"] == 1
    assert rep["standard"]["near_duplicates"]["by_kind"]["dhash"] == 1
    assert rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]["finds"] == 1


def test_named_subsets_get_their_own_tables(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"], subsets={"excl-former-heldout": {"former_heldout": "no"},
                                                 "only-heldout": {"former_heldout": "yes"}})
    rep = run(world, tmp_path, "val", cfg)
    assert rep["subsets"]["excl-former-heldout"]["comparators"]["vision-photos"]["primary"][
        "all"]["finds"] == 2                                   # no benchmark records here
    assert rep["subsets"]["only-heldout"]["near_duplicates"]["finds"] == 0


def test_the_reviewed_crosswalk_credits_only_pairs_marked_credit(world, tmp_path, monkeypatch):
    from mycomap_vision.replications.fungitastic import crosswalk as gbif
    keys = {"Hydnum repandum": {"k1"}, "Hydnum washingtonianum": {"k1"},
            "Pycnoporus cinnabarinus": {"k2"}, "Trametes cinnabarina": {"k2"},
            "Fomes fomentarius": {"k3"}, "Fomes excavatus": {"k3"}}

    class NoMatcher:
        def __init__(self, path):
            pass

        def close(self):
            pass
    monkeypatch.setattr(gbif, "Matcher", NoMatcher)
    monkeypatch.setattr(gbif.Crosswalk, "from_cache", classmethod(
        lambda cls, m, names: cls({n: frozenset(v) for n, v in keys.items()})))
    cache = tmp_path / "gbif.sqlite"
    cache.write_bytes(b"cache")
    review = tmp_path / "review.tsv"
    review.write_text("their_answer\tour_label\treview\n"
                      "Pycnoporus cinnabarinus\tTrametes cinnabarina\tcredit\n"
                      "Hydnum repandum\tHydnum washingtonianum\texclude\n", encoding="utf-8")
    spec = {"cache": str(cache), "cache_sha256": paper_score.sha256_file(cache),
            "review": str(review), "review_sha256": paper_score.sha256_file(review)}
    same, info, uncredited = paper_score.crosswalk_rule(spec, set(keys), world["conn"])
    assert same("Pycnoporus cinnabarinus", "Trametes cinnabarina")        # credited
    assert not same("Hydnum repandum", "Hydnum washingtonianum")          # excluded
    assert not same("Fomes fomentarius", "Fomes excavatus")               # never reviewed
    assert dict(uncredited) == {("Hydnum repandum", "Hydnum washingtonianum"): 1,
                                ("Fomes fomentarius", "Fomes excavatus"): 1}
    full, _, _ = paper_score.crosswalk_rule({**spec, "mode": "full"}, set(keys), world["conn"])
    assert full("Fomes fomentarius", "Fomes excavatus")                   # the row beside
    with pytest.raises(ValueError, match="the config pins"):
        paper_score.crosswalk_rule({**spec, "review_sha256": "0" * 64}, set(keys), world["conn"])


def test_each_photo_alone_is_a_per_photo_table_beside_the_finds(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"].append({"id": "each", "kind": "vision", "backbone": "m1",
                               "method": "nearest", "photos": "each"})
    rep = run(world, tmp_path, "val", cfg)
    t = rep["per_photo"]["each"]["species_strict@1"]
    assert t["photos"] == 2 and t["right"] == 2                # one photo per val record here
    assert rep["inputs"]["each"]["photo_answers"] == 2
    assert "each" not in rep["errors"]["comparators"]           # not a per-find answer


def test_the_configs_erratum_is_pinned_and_a_different_cli_one_is_refused(world, tmp_path):
    from mycomap_vision import errata
    path = tmp_path / "errata.tsv"
    path.write_text("kind\tkey\treason\tnote\nrecord\t1\tmo_twin_as_inat\t\n", encoding="utf-8")
    block = {"file": str(path), "sha256": paper_score.sha256_file(path)}
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"], errata=block))
    assert rep["errata"]["sha256"] == block["sha256"]
    assert rep["errata"]["reference_records_left_out"] == 1
    with pytest.raises(ValueError, match="the config pins"):
        run(world, tmp_path, "val", config(tmp_path, ["5", "6"],
                                           errata={**block, "sha256": "0" * 64}))
    other = tmp_path / "other.tsv"
    other.write_text("kind\tkey\treason\tnote\nrecord\t2\tmo_twin_as_inat\t\n", encoding="utf-8")
    world["base"].use_errata(errata.read(other))              # as a differing --errata would
    with pytest.raises(ValueError, match="carries erratum"):
        run(world, tmp_path, "val", config(tmp_path, ["5", "6"], errata=block))


def test_config_paths_resolve_under_the_data_folder_and_the_research_repo(tmp_path):
    from mycomap_vision import config as cfg
    assert paper_score.resolve("data/x/y.tsv") == cfg.DATA_DIR / "x" / "y.tsv"
    assert paper_score.resolve("paper/e21.csv") == cfg.RESEARCH_DIR / "paper" / "e21.csv"
    # An absolute path is kept as given. tmp_path is absolute on every platform; "C:/..."
    # is absolute only on Windows, so this failed on CI's Linux runner.
    absolute = tmp_path / "abs" / "z.tsv"
    assert paper_score.resolve(str(absolute)) == absolute


def test_the_cc_reference_row_uses_only_photos_a_public_model_may_learn_from(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"].append({"id": "cc-reference", "kind": "vision", "backbone": "m1",
                               "method": "nearest", "reference": "public_trainable_only"})
    rep = run(world, tmp_path, "val", cfg)
    full = rep["inputs"]["vision-photos"]["reference_records"]
    cc = rep["inputs"]["cc-reference"]["reference_records"]
    want = sum(1 for r in world["base"].as_records("train", None, public_trainable_only=True))
    assert cc == want and cc < full                 # the fixture's ARR photos leave



def test_a_public_reference_counts_its_own_depth_and_new_observers(world):
    """O1: the headline public model scores against the openly licensed reference, so its
    finds' depth counts only the records there (fixture: 1 and 3 have all-rights-reserved
    photos only, so A x keeps 2 and mo:9, B y keeps 4)."""
    base = world["base"]
    rows = paper_score.reference_rows(base, ("train",), "public_trainable_only")
    assert sorted(str(r["record_key"]) for r in rows) == ["2", "4", "mo:9"]
    assert len(paper_score.reference_rows(base, ("train",), None)) == 5
    finds = [po.Find("5", "x", "A x", "A", "F", "formal", 3, 1, {"observer_new": "known"}),
             po.Find("6", "y", "B y", "B", "F", "formal", 2, 1, {"observer_new": "known"})]
    scored = [{"record_key": "5", "observer_login": "alice"},
              {"record_key": "6", "observer_login": "zed"}]
    own = paper_score.finds_in_reference(finds, scored, rows)
    assert [f.depth for f in own] == [2, 1] and [f.depth for f in finds] == [3, 2]
    assert own[1].attrs["observer_new"] == "new" and finds[1].attrs["observer_new"] == "known"


def test_a_comparators_own_reference_reaches_its_tables_and_is_reported(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"], recoverability={"role": "C4"})
    cfg["comparators"].append({"id": "public-photos", "kind": "vision", "backbone": "m1",
                               "method": "nearest", "reference": "public_trainable_only"})
    cfg["hypotheses"]["side"] = [{"id": "internal-public", "kind": "mcnemar",
                                  "a": "vision-photos", "b": "public-photos",
                                  "cell": "species_strict@1", "alternative": "two-sided"}]
    rep = run(world, tmp_path, "val", cfg)
    assert rep["references"] == {"public_trainable_only": {"records": 3,
                                                           "comparators": ["public-photos"]}}
    from mycomap_vision import paper_tables as pt
    seen = rep["recoverability"]["public-photos"]["denominators"][pt.SEEN]
    assert seen["species_by_band"]["1-4"] == 2
    assert rep["recoverability_units"]["public_trainable_only"][pt.SEEN]["species"] == 2
    side = rep["hypotheses"]["side"]["internal-public"]
    assert side["outside_holm"] and side["p_holm"] is None and side["reject"] is None
    assert rep["hypotheses"]["A"]["H1a"]["p_holm"] is not None          # others unchanged
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "### Family side (outside Holm: disclosed, no decision)" in md


def test_a_reference_other_than_the_two_known_or_on_another_kind_is_refused():
    base = {"format": 1, "answer_key": KEY}
    with pytest.raises(ValueError, match="reference is one of"):
        paper_score.check_config({**base, "comparators": [
            {"id": "v", "kind": "vision", "backbone": "m1", "method": "nearest",
             "reference": "cc_only"}]})
    with pytest.raises(ValueError, match="for a Vision comparator"):
        paper_score.check_config({**base, "comparators": [
            {"id": "i", "kind": "inat", "file": "x", "reference": "public_trainable_only"}]})



def test_a_resumed_run_reuses_only_checkpoints_of_the_same_config_code_and_comparator(
        world, tmp_path, monkeypatch):
    """A run that dies keeps each finished Vision comparator's predictions and stamp; with
    resume they are loaded instead of scored again, and the report is the same. A changed
    config, or no resume, scores again."""
    cfg = config(tmp_path, ["5", "6"])
    first = run(world, tmp_path, "val", cfg)
    pred = tmp_path / "out" / "predictions"
    meta = json.loads((pred / "vision-photos.meta.json").read_text(encoding="utf-8"))
    assert meta["answers"] == 2 and meta["finds"] == 2 and len(meta["config_sha256"]) == 64
    assert not list(pred.glob("*.tmp"))
    scored = []
    real = paper_score.vision_answers

    def counting(*a, **k):
        scored.append(a[3])
        return real(*a, **k)
    monkeypatch.setattr(paper_score, "vision_answers", counting)
    again = run(world, tmp_path, "val", cfg, resume=True)
    assert scored == []                                   # every Vision row from its checkpoint
    assert again["standard"] == first["standard"]
    assert again["hypotheses"] == first["hypotheses"]
    run(world, tmp_path, "val", cfg)                      # no resume: scored again
    assert scored == ["nearest", "nearest+mean"]
    scored.clear()
    cfg2 = config(tmp_path, ["5", "6"], top_k=5)          # another config: its stamp differs
    run(world, tmp_path, "val", cfg2, resume=True)
    assert scored == ["nearest", "nearest+mean"]


def test_a_checkpoint_is_reused_only_when_every_part_of_its_stamp_matches(tmp_path):
    path = tmp_path / "v.jsonl"
    answers = {"2": {"species": [{"name": "B y"}]}, "1": {"species": [{"name": "A x"}]}}
    stamp = paper_score.checkpoint_stamp(b"cfg", "abc1234", {"id": "v", "method": "m"}, 2)
    paper_score.write_checkpoint(path, answers, stamp, {"place_setting": "s"})
    info: dict = {}
    assert paper_score.load_checkpoint(path, stamp, info) == answers
    assert info == {"place_setting": "s"}
    for changed in (paper_score.checkpoint_stamp(b"cfg2", "abc1234", {"id": "v", "method": "m"}, 2),
                    paper_score.checkpoint_stamp(b"cfg", "def5678", {"id": "v", "method": "m"}, 2),
                    paper_score.checkpoint_stamp(b"cfg", "abc1234", {"id": "v", "method": "x"}, 2),
                    paper_score.checkpoint_stamp(b"cfg", "abc1234", {"id": "v", "method": "m"}, 3)):
        assert paper_score.load_checkpoint(path, changed, {}) is None
    path.write_text(path.read_text(encoding="utf-8").splitlines()[0] + "\n", encoding="utf-8")
    assert paper_score.load_checkpoint(path, stamp, {}) is None      # a truncated file


def test_the_attrs_file_is_pinned_by_the_config(world, tmp_path):
    attrs = tmp_path / "attrs.tsv"
    attrs.write_text("record_key\tdistance_band\n5\t<100\n6\t>1500\n", encoding="utf-8")
    block = {"file": str(attrs), "sha256": paper_score.sha256_file(attrs)}
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"], attrs=block,
                                             slices={"attrs": ["distance_band"], "min_finds": 1}))
    assert set(rep["slices"]["distance_band"]) == {"<100", ">1500"}       # read without --attrs
    assert rep["attrs"]["sha256"] == block["sha256"]
    other = tmp_path / "other.tsv"
    other.write_text("record_key\tdistance_band\n5\tno point\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not the attrs file the config pins"):
        run(world, tmp_path, "val", config(tmp_path, ["5", "6"], attrs=block), attrs_path=other)
    with pytest.raises(ValueError, match="the config pins"):
        run(world, tmp_path, "val", config(tmp_path, ["5", "6"],
                                           attrs={**block, "sha256": "0" * 64}))


def test_the_report_names_the_code_that_started_the_run_not_a_later_commit(world, tmp_path,
                                                                          monkeypatch):
    from mycomap_vision import config as cfgmod
    seen = iter(["a" * 40, "b" * 40, "b" * 40, "b" * 40])
    monkeypatch.setattr(cfgmod, "code_version", lambda: next(seen))
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))
    assert rep["code_commit"] == "a" * 40


def place_pins(*names):
    """The config's "place_settings" block pinning `names` as this code has them."""
    from mycomap_vision import placeprior
    from mycomap_vision.paper_freeze import canonical_sha
    return {n: {"sha256": canonical_sha(placeprior.place_setting(n)[1])} for n in names}


def test_a_place_setting_is_built_into_the_place_method_recorded_and_an_unknown_one_refused(
        world, tmp_path, monkeypatch):
    """place_setting: the served place scoring, exactly. The method is built with it (the
    fitted prior's own setting is checked), the report records it, the environment is
    left as it was, and a name that is not a place setting is refused before any read."""
    from mycomap_vision import evaluate, placeprior
    monkeypatch.delenv(placeprior.SETTING_ENV, raising=False)
    built = []
    real_fit = evaluate.fit_method

    def fit(vectors, ref, method, backbone=None):
        index, model = real_fit(vectors, ref, "nearest+mean", backbone)
        model.prior = placeprior.PlaceBoost(store=None)       # reads the setting in force
        built.append(model.prior.setting_name)
        return index, model
    monkeypatch.setattr(evaluate, "fit_method", fit)
    name = "penalty-500-2026-10-13"
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"] = [{"id": "vision-place", "kind": "vision", "backbone": "m1",
                           "method": "nearest+mean+place", "place": True,
                           "place_setting": name}]
    cfg["hypotheses"] = {}
    cfg["place_settings"] = place_pins(name)
    rep = run(world, tmp_path, "val", cfg)
    assert built == [name]
    assert rep["inputs"]["vision-place"]["place_setting"] == name
    assert placeprior.SETTING_ENV not in __import__("os").environ
    for bad in ({"method": "nearest+mean+place", "place_setting": "penalty-501"},
                {"method": "nearest+mean", "place_setting": name}):
        cfg["comparators"][0].update(bad)
        with pytest.raises(ValueError, match="place setting|place_setting|not a place"):
            paper_score.check_config(cfg)


class SeesUuid(SeesPlace):
    """A place method stand-in that records the iNat uuid it is given for each find."""
    uuids: list = []

    def species_scores(self, query, context=None):
        SeesUuid.uuids.append(context.uuid)
        return np.zeros(self.n, dtype=np.float32)


def test_a_place_method_is_given_each_finds_own_inat_uuid_so_it_can_leave_it_out(
        world, tmp_path, monkeypatch):
    """The occurrence prior leaves a find's own iNaturalist observation out of its counts
    (OccurrenceStore.own_contribution) only if it is told the find's uuid: paper-score
    passes it, through the public-point substitution, for every scored find."""
    from mycomap_vision import evaluate
    monkeypatch.setitem(evaluate.METHODS, "sees-uuid", SeesUuid)
    monkeypatch.setattr(paper_score, "public_points",
                        lambda rel: {"5": (47.5, -122.5), "6": (None, None)})
    expected = sorted(u for (u,) in world["base"].db.execute(
        "select inat_uuid from records where split = 'val' and included = 1"))
    assert expected == ["uuid-5", "uuid-6"]                    # the release carries them
    SeesUuid.uuids = []
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"] = [{"id": "vision-place", "kind": "vision", "backbone": "m1",
                           "method": "sees-uuid", "place": True}]
    cfg["hypotheses"] = {}
    run(world, tmp_path, "val", cfg)
    assert sorted(SeesUuid.uuids) == expected


def test_paper_scores_place_rows_leave_each_finds_own_inat_observation_out(tmp_path,
                                                                         monkeypatch):
    """Regression (leak found 2026-10-13): the real place method, as paper-score fits it,
    gives a find whose own observation is the only eastern record of its species exactly
    what it would get had that observation never been counted; scored without its uuid
    (the leak), its species would look in range."""
    from dataclasses import replace
    from occurrence_fixtures import EAST, build_store, obs, standard_observations
    from test_placeprior import OCC, records_east_west
    from test_prior import rec

    from mycomap_vision import occurrence
    recs = records_east_west()
    find = replace(rec("q", OCC, *EAST, "2025-10-01", rows=[12]), uuid="self-1")
    vecs = np.random.default_rng(0).normal(size=(13, 8)).astype(np.float32)
    cal = {"species": {"temperature": 1.0, "floor": None}}
    truths = {"q": {"species": OCC}}

    def p_true(store, f):
        monkeypatch.setattr(occurrence, "default_store_path", lambda: store.path)
        a = paper_score.vision_answers(vecs, recs, [f], "nearest+mean+place", "m",
                                       calibration=cal, truths=truths,
                                       place_setting="boost-2026-10-11")
        return a["q"]["cal"]["species"]["p_true"]
    with_own = build_store(tmp_path / "with", standard_observations()
                           + [obs("self-1", 105, *EAST, date="2025-10-01")])[0]
    never = build_store(tmp_path / "never")[0]
    assert p_true(with_own, find) == pytest.approx(p_true(never, find))
    assert p_true(with_own, replace(find, uuid="")) > p_true(never, find) + 1e-6


def test_a_place_setting_that_reads_inat_distances_is_refused():
    """PlaceBoost's served store drops the pair counts nearest_inat_km would need to take a
    find's own observation out, so a dna+inat setting cannot be scored leak-free here."""
    from mycomap_vision import placeprior
    inat = [n for n, s in placeprior.PLACE_SETTINGS.items()
            if "inat" in str({**placeprior.NO_EXTRAS, **s}["pen_known"])]
    cfg = {"format": 1, "answer_key": KEY, "comparators": [{"id": "p", "kind": "vision",
                                                            "backbone": "m",
                                         "method": "nearest+mean+place", "place": True,
                                         "place_setting": "penalty-500-2026-10-13"}],
           "place_settings": place_pins("penalty-500-2026-10-13")}
    paper_score.check_config(cfg)                                    # dna only: fine
    monkeypatch_name = "test-dna-inat"
    placeprior.PLACE_SETTINGS[monkeypatch_name] = {"pen_known": "dna+inat", "pen_slope": 2.0,
                                                   "pen_cap": 8.0}
    try:
        cfg["comparators"][0]["place_setting"] = monkeypatch_name
        with pytest.raises(ValueError, match="own observation"):
            paper_score.check_config(cfg)
    finally:
        del placeprior.PLACE_SETTINGS[monkeypatch_name]
    assert inat == []                         # none of the named settings reads them today


# --- the decided place setting, pinned by sha (PLAN O9) ----------------------------------------

def _place_config(**extra):
    return {"format": 1, "answer_key": KEY, "comparators": [
        {"id": "vision-place", "kind": "vision", "backbone": "m1",
         "method": "nearest+mean+place", "place": True,
         "place_setting": "penalty-500-s30-2026-10-19"}], **extra}


def test_a_place_setting_is_scored_only_under_the_configs_pinned_sha():
    # research's pin for O9 (paper/analysis-v1.draft.json, 7946bdc): the served preset's sha
    pinned = {"penalty-500-s30-2026-10-19": {
        "sha256": "20eb90670a20c9564b2262e6a3cf7b867922065ac474d21afd6340fcc9f82bc8",
        "grid_id": 748}}
    paper_score.check_config(_place_config(place_settings=pinned))
    with pytest.raises(ValueError, match="no entry"):
        paper_score.check_config(_place_config())
    with pytest.raises(ValueError, match="no entry"):
        paper_score.check_config(_place_config(place_settings=place_pins(
            "penalty-500-2026-10-13")))                         # another setting's pin only
    with pytest.raises(ValueError, match="the config pins"):
        paper_score.check_config(_place_config(place_settings={
            "penalty-500-s30-2026-10-19": {"sha256": "0" * 64}}))


def test_a_place_setting_changed_in_code_after_the_pin_is_refused(monkeypatch):
    from mycomap_vision import placeprior
    cfg = _place_config(place_settings=place_pins("penalty-500-s30-2026-10-19"))
    paper_score.check_config(cfg)
    monkeypatch.setitem(placeprior.PLACE_SETTINGS, "penalty-500-s30-2026-10-19",
                        {**placeprior.PLACE_SETTINGS["penalty-500-s30-2026-10-19"],
                         "shrink": 10.0})
    with pytest.raises(ValueError, match="the config pins"):
        paper_score.check_config(cfg)


def test_a_run_with_an_unpinned_place_setting_is_refused_before_anything_is_read(
        world, tmp_path, monkeypatch):
    monkeypatch.setattr(paper_score, "guard",
                        lambda *a, **k: pytest.fail("read before the pin check"))
    with pytest.raises(ValueError, match="no entry"):
        run(world, tmp_path, "val", _place_config())


# --- per-species rows (PLAN rule 5, 19 Oct) ----------------------------------------------------

def test_per_species_rows_weight_each_species_equally_and_pool_by_the_answer_key():
    from mycomap_vision import name_equiv
    from mycomap_vision import paper_tables as pt
    sp = "Pseudohydnum gelatinosum"
    finds = [po.Find("1", "o1", sp, "Pseudohydnum", "F", "formal", 3, 1),
             po.Find("2", "o2", f"{sp} ssp. pusillum", "Pseudohydnum", "F", "formal", 3, 1),
             po.Find("3", "o1", "B y", "B", "F", "formal", 3, 1),
             po.Find("4", "o3", "C z", "C", "F", "formal", 3, 1),
             po.Find("5", "o3", "C z", "C", "F", "formal", 3, 1),
             po.Find("6", "o4", "", "D", "F", "one-word", 0, 1)]          # no species: left out
    outcome = {"1": {"species_strict@1": True}, "2": {"species_strict@1": False},
               "3": {"species_strict@1": True}, "4": {"species_strict@1": False}}
    # "5" has no outcome line: unanswered, so wrong
    rows = pt.per_species(finds, outcome, "species_strict@1", REPS, 0,
                          lambda n: name_equiv.answer_key(n, KEY))
    assert rows["n_species"] == 3 and rows["finds"] == 5    # the subspecies pools with its species
    assert rows["macro_species_top1"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert rows["species_ge1_right"] == 2
    assert rows["share_species_ge1_right"] == pytest.approx(2 / 3)
    lo, hi = rows["macro_ci95"]
    assert lo <= rows["macro_species_top1"] <= hi
    # per find the same finds give 2 of 5: macro and per-find rates differ by design
    assert pt.rate(finds, outcome, "species_strict@1", REPS, 0)["rate"] == pytest.approx(2 / 5)


def test_the_headline_tables_carry_per_species_rows_only_when_the_config_asks(world, tmp_path):
    block = {"metrics": ["macro_species_top1", "share_species_ge1_right", "n_species"],
             "cell": "species_strict@1", "key": "answer_key",
             "role": "secondary, outside hypothesis tests"}
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"], per_species_rows=block))
    for den in ("all", "has_reference"):
        rows = rep["standard"]["comparators"]["vision-photos"]["primary"][den]["per_species"]
        assert rows["n_species"] == 2 and rows["macro_species_top1"] == 1.0   # A x, B y
        assert rows["share_species_ge1_right"] == 1.0
    inat = rep["standard"]["comparators"]["inat-vision"]["primary"]["all"]["per_species"]
    assert inat["macro_species_top1"] == 0.0 and inat["species_ge1_right"] == 0
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "Per species (species_strict@1; secondary, no test): macro 100.0%" in md
    plain = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))
    assert "per_species" not in plain["standard"]["comparators"]["vision-photos"]["primary"]["all"]
    assert rep["hypotheses"] == plain["hypotheses"]                  # no test reads them


def test_a_per_species_block_with_an_unknown_metric_or_a_genus_cell_is_refused():
    base = {"format": 1, "answer_key": KEY, "comparators": []}
    paper_score.check_config({**base, "per_species_rows": {"metrics": ["n_species"]}})
    with pytest.raises(ValueError, match="unknown metrics"):
        paper_score.check_config({**base, "per_species_rows": {"metrics": ["micro_f1"]}})
    with pytest.raises(ValueError, match="species cell"):
        paper_score.check_config({**base, "per_species_rows": {"cell": "genus_strict@1"}})


# --- species recoverability (PLAN C4, 19 Oct) --------------------------------------------------

def test_recoverability_is_each_species_band_accuracy_averaged_over_species_both_ways():
    from mycomap_vision import paper_tables as pt
    finds = [po.Find("1", "o1", "A a", "A", "F", "formal", 2, 1),       # band 1-4: 1 of 2 right
             po.Find("2", "o2", "A b", "A", "F", "formal", 3, 1),
             po.Find("3", "o3", "B c", "B", "F", "formal", 50, 1),      # band 20-99: 1 of 1
             po.Find("4", "o4", "Z z", "Z", "F", "formal", 0, 1),       # band 0: 0 by construction
             po.Find("5", "o5", "", "Y", "F", "one-word", 9, 1)]        # no species: not used
    outcome = {"1": {f"species_strict@{k}": True for k in (1, 3, 10)},
               "2": {"species_strict@1": False, "species_strict@3": True,
                     "species_strict@10": True},
               "3": {f"species_strict@{k}": True for k in (1, 3, 10)},
               "4": {f"species_strict@{k}": True for k in (1, 3, 10)}}
    # (a) the reference: two sparse species (one provisional), one at 50, one at 200 (100+ has
    # no scored find, so it is left out and counted); (b) the same plus two names with no
    # reference (one formal)
    seen = [(2, True), (3, False), (50, True), (200, True)]
    known = seen + [(0, True), (0, False)]
    r = pt.recoverability(finds, outcome, {pt.KNOWN: known, pt.SEEN: seen}, REPS, 0)
    assert list(r["denominators"]) == [pt.KNOWN, pt.SEEN]               # (b) leads
    assert r["finds"] == 4
    assert r["band_accuracy"]["0"]["top1"] == 0.0                       # never named
    assert r["band_accuracy"]["1-4"]["top1"] == 0.5 and r["band_accuracy"]["1-4"]["top3"] == 1.0
    assert r["band_accuracy"]["100+"]["top1"] is None
    a = r["denominators"][pt.SEEN]["rows"]
    assert a["all names"]["species"] == 3 and a["all names"]["left_out"] == 1
    assert a["all names"]["top1"]["estimate"] == pytest.approx((0.5 + 0.5 + 1.0) / 3)
    assert a["formal names"]["top1"]["estimate"] == pytest.approx((0.5 + 1.0) / 2)
    assert a["every species at 20+ references"]["top1"]["estimate"] == pytest.approx(1.0)
    b = r["denominators"][pt.KNOWN]
    assert b["species_by_band"]["0"] == 2
    assert b["rows"]["all names"]["species"] == 5
    assert b["rows"]["all names"]["top1"]["estimate"] == pytest.approx((0.5 + 0.5 + 1.0) / 5)
    assert b["rows"]["formal names"]["top1"]["estimate"] == pytest.approx((0.5 + 1.0) / 3)
    # the counterfactual lifts the no-reference names too
    assert b["rows"]["every species at 20+ references"]["top1"]["estimate"] == pytest.approx(1.0)
    lo, hi = b["rows"]["all names"]["top1"]["ci95"]
    assert lo <= b["rows"]["all names"]["top1"]["estimate"] <= hi


def test_the_report_gives_recoverability_only_when_the_config_asks(world, tmp_path):
    from mycomap_vision import paper_tables as pt
    block = {"role": "descriptive, C4", "denominators": {
        "b_leads": "every species name in the v1 release; no-reference names at band 0",
        "a": "species in the frozen reference"}}
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"], recoverability=block))
    r = rep["recoverability"]["vision-photos"]
    # train: A x has 3 records, B y 2 -> both in 1-4; both val finds right at top 1. The
    # release names no species outside A x and B y, so both denominators hold the same two.
    for den in (pt.KNOWN, pt.SEEN):
        d = r["denominators"][den]
        assert d["species_by_band"]["1-4"] == 2 and d["species_by_band"]["0"] == 0
        assert d["rows"]["all names"]["top1"]["estimate"] == 1.0
    assert rep["recoverability_units"] == {pt.KNOWN: {"species": 2, "with_reference": 2},
                                           pt.SEEN: {"species": 2, "with_reference": 2}}
    inat = rep["recoverability"]["inat-vision"]["denominators"][pt.KNOWN]
    assert inat["rows"]["all names"]["top1"]["estimate"] == 0.0
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Species recoverability (C4; descriptive, model-based)" in md
    assert f"| {pt.KNOWN} | all names | 2 |" in md
    plain = run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))
    assert plain["recoverability"] is None
    assert rep["hypotheses"] == plain["hypotheses"]                  # no test reads it


def test_the_release_denominator_counts_names_without_a_reference_at_band_zero():
    from mycomap_vision import paper_tables as pt

    class Db:
        def execute(self, q):
            return [("A x", "formal"), ("A x var. y", "formal"), ("C q", "formal"),
                    ("D sp. 'IN01'", "provisional")]
    base = type("Base", (), {"db": Db()})()
    ref = [{"species": "A x", "name_kind": "formal"}] * 3 + [{"species": "B y", "name_kind": "formal"}]
    units, cite = paper_score.recoverability_units(base, ref, KEY)
    known = units[pt.KNOWN]
    assert sorted(d for d, _ in known) == [0, 0, 1, 3]                 # C q and D: no reference
    assert sorted(d for d, _ in units[pt.SEEN]) == [1, 3]
    assert cite == {pt.KNOWN: {"species": 4, "with_reference": 2},
                    pt.SEEN: {"species": 2, "with_reference": 2}}
    assert sum(f for _, f in known) == 3                               # D is provisional


# --- outside-review rows: unseen observers, recent cohort, created after pretraining ----------

def test_recent_cohort_and_after_pretraining_come_from_observed_and_created_dates():
    finds = [po.Find(k, "o", "A x", "A", "F", "formal", 1, 1) for k in ("1", "2", "3", "4")]
    rows = [{"record_key": "1", "observed_on": "2026-09-01"},
            {"record_key": "2", "observed_on": "2026-08-17"},
            {"record_key": "3", "observed_on": None}, {"record_key": "4"}]
    created = {"1": "2024-05-02T00:00:00Z", "2": "2024-04-30T23:59:59Z", "3": ""}
    paper_score.add_generalisation_attrs(finds, rows, "2026-08-17", created, "2024-05-01")
    assert [f.attrs["recent_cohort"] for f in finds] == ["yes", "no", "(no date)", "(no date)"]
    assert [f.attrs["after_pretraining"] for f in finds] == [
        "yes", "no", "(no date)", "(not in the snapshot)"]     # 4: an MO find, say


def test_the_report_carries_the_generalisation_rows_and_the_pinned_created_file(world, tmp_path):
    created = tmp_path / "created_at.tsv"
    created.write_text("record_key\tobservation_id\tcreated_at\n5\t55\t2025-01-01T00:00:00Z\n",
                       encoding="utf-8")
    pre = {"file": str(created), "sha256": paper_score.sha256_file(created),
           "after": "2024-05-01"}
    rep = run(world, tmp_path, "val",
              config(tmp_path, ["5", "6"], generalisation={"pretraining": pre}))
    gen = rep["generalisation"]
    assert gen["recent_after"] == world["base"].val_cutoff
    assert set(gen["rows"]) == {"unseen observers", "recent cohort", "created after 2024-05-01"}
    row = gen["rows"]["created after 2024-05-01"]
    assert (row["finds"], row["others"]) == (1, 1)
    assert row["vision-photos"]["species_strict@1"]["in"]["n"] == 1
    assert gen["near_duplicates"]["vision-photos"]["species_strict@1"]["primary"]["n"] == 2
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Generalisation" in md and "created after 2024-05-01" in md
    assert "## Freeze" in md
    pre["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="created_at file"):
        run(world, tmp_path, "val", config(tmp_path, ["5", "6"],
                                           generalisation={"pretraining": pre}))


def test_an_expert_tier_other_than_1_2_or_pooled_is_refused():
    ok = {"id": "e", "kind": "expert", "tier": "1+2", "view": "first", "file": "x"}
    paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": [ok]})
    with pytest.raises(ValueError, match="expert tier"):
        paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": [{**ok, "tier": "1+3"}]})


def test_a_test_report_counts_what_the_build_left_out_beside_the_scorings_near_copies(
        world, tmp_path):
    counts = world["test"].manifest["counts"]
    # A cross-post twin of a v1 record, as the v1-test build will leave out (its reason name
    # is the builder's; whatever it is, it is counted).
    counts["records_by_source_reason"] = {**counts["records_by_source_reason"],
                                          "inat/twin_of_base": 3}
    rep = run(world, tmp_path, world["test"], config(tmp_path, ["7", "8"]))
    bx = rep["build_exclusions"]
    want = {k: v for k, v in counts["records_by_source_reason"].items() if not k.endswith("/ok")}
    assert bx["records"]["by_source_reason"] == want
    assert bx["records"]["left_out"] == sum(want.values())
    assert bx["records"]["total"] == counts["records"]
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert bx["records"]["by_source_reason"]["inat/twin_of_base"] == 3
    assert "Left out by the test build (records)" in md and "inat/twin_of_base 3" in md
    assert "Near-copies left out of the primary numbers at scoring" in md
    assert run(world, tmp_path, "val", config(tmp_path, ["5", "6"]))["build_exclusions"] is None


def test_on_val_the_pinned_dhash_index_flags_near_copies_of_reference_photos(world, tmp_path):
    from mycomap_vision import dhash as dh
    base = world["base"]
    rows = [dict(r) for r in base.db.execute(
        "select p.photo_key, p.original_sha256, r.record_key, r.split from photos p join "
        "records r on r.record_key = p.record_key where p.included = 1")]
    train_hash = 0x0F0F0F0F0F0F0F0F
    def h(r):
        if r["record_key"] == "5":
            return train_hash ^ 0b11                        # 2 bits from a train photo
        if r["record_key"] == "6":
            return 0xFFFF000000000000                       # far from everything
        return train_hash if r["record_key"] == "1" else 0x00FF00FF00FF00FF   # 32 bits away
    index = {"release": base.id, "release_hash": base.release_hash, "algorithm": dh.ALGORITHM,
             "pillow": dh.pillow_version(), "photos": len(rows),
             "photo_key": [r["photo_key"] for r in rows],
             "sha256": [r["original_sha256"] for r in rows],
             "dhash": [h(r) for r in rows]}
    path = dh.save_index(index, tmp_path / "dhash-index.npz")
    nd = {"dhash_index": {"file": str(path), "sha256": paper_score.sha256_file(path)}}
    rep = run(world, tmp_path, "val", config(tmp_path, ["5", "6"], near_duplicates=nd))
    assert rep["standard"]["near_duplicates"]["by_kind"]["dhash"] == 1
    assert rep["dhash_flags"]["photos_flagged"] == 1
    assert rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]["table"][
        "species strict"]["top1"]["n"] == 1                    # the near-copy left out
    nd["dhash_index"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="dHash index"):
        run(world, tmp_path, "val", config(tmp_path, ["5", "6"], near_duplicates=nd))


# --- the answer key: one function judges every comparator ---------------------------------------

def _four_comparators(tmp_path):
    """A Vision answer set, a published model's file, an iNat file and an experts' file for
    finds 5 (A x) and 6 (B y), each answering the truth's own name."""
    pub = tmp_path / "pub.jsonl"
    pub.write_text("\n".join(json.dumps({"observation_id": k, "result": {
        "species": [{"name": TRUTH[k]}], "genus": [{"name": TRUTH[k][0]}]}}) for k in ("5", "6")),
        encoding="utf-8")
    comps = [{"id": "v", "kind": "vision", "backbone": "m1", "method": "nearest"},
             {"id": "p", "kind": "published", "file": str(pub)},
             {"id": "i", "kind": "inat", "score": "vision",
              "file": e14(tmp_path / "e14.jsonl", ["5", "6"], True)},
             {"id": "e", "kind": "expert", "tier": "1", "view": "first",
              "file": e31(tmp_path / "e31.jsonl", ["5", "6"])}]
    answers = {k: {"species": [TRUTH[k]], "genus": [TRUTH[k][0]]} for k in ("5", "6")}
    return comps, answers


def test_every_comparator_kind_is_judged_by_the_one_answer_key_function(tmp_path, monkeypatch):
    from mycomap_vision import name_equiv
    finds = [po.Find(k, f"o{k}", TRUTH[k], TRUTH[k][0], "F", "formal", 3, 1) for k in ("5", "6")]
    comps, answers = _four_comparators(tmp_path)

    def species_cells(verdict):
        monkeypatch.setattr(name_equiv, "species_credit",
                            lambda a, t, rule=None, taxonomy=None: verdict)
        return {c["id"]: {k: o["species_strict@1"] for k, o in paper_score.outcomes_for(
                    c, finds, answers if c["kind"] == "vision" else None)[0].items()}
                for c in comps}
    # Every answer names the truth exactly, so only the function decides: a comparator
    # judged by any rule of its own (exact names, the lane's booleans) would not flip.
    assert species_cells(True) == {c["id"]: {"5": True, "6": True} for c in comps}
    assert species_cells(False) == {c["id"]: {"5": False, "6": False} for c in comps}


def test_the_config_names_its_answer_key_and_a_comparator_may_only_go_exact():
    from mycomap_vision import name_equiv
    v = {"id": "v", "kind": "vision", "backbone": "m1", "method": "nearest"}
    with pytest.raises(ValueError, match="answer key"):
        paper_score.check_config({"format": 1, "comparators": [v]})
    paper_score.check_config({"format": 1, "answer_key": KEY,
                              "comparators": [v, {**v, "id": "x", "answer_key": "exact"}]})
    with pytest.raises(ValueError, match="only"):
        paper_score.check_config({"format": 1, "answer_key": "exact",
                                  "comparators": [{**v, "answer_key": KEY}]})
    assert KEY == name_equiv.ANSWER_KEY


def test_the_report_names_the_answer_key_and_counts_the_lanes_own_verdicts_apart(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"])
    # iNat's file: names say right, the lane's taxon-id hits say wrong.
    lines = [json.loads(x) for x in Path(e14(tmp_path / "e14b.jsonl", ["5", "6"], True))
             .read_text(encoding="utf-8").splitlines()]
    for x in lines:
        x["vision_hits"] = {r: {f"top{n}": False for n in po.K} for r in ("species", "genus")}
    (tmp_path / "e14b.jsonl").write_text("\n".join(json.dumps(x) for x in lines), encoding="utf-8")
    cfg["comparators"][2]["file"] = str(tmp_path / "e14b.jsonl")
    rep = run(world, tmp_path, "val", cfg)
    assert rep["answer_key"] == KEY and rep["inputs"]["inat-vision"]["answer_key"] == KEY
    inat = rep["standard"]["comparators"]["inat-vision"]["primary"]["all"]["table"]
    assert inat["species strict"]["top1"]["rate"] == 1.0          # the key, not the hits
    assert rep["inputs"]["inat-vision"]["lane_verdicts_disagree_top1"] == {"species": 2,
                                                                           "genus": 2}


def test_an_experts_file_without_names_is_refused():
    f = po.Find("5", "o", "A x", "A", "F", "formal", 3, 1)
    old = {"record_key": "5", "status": "ok", "format": 1, "community": {"present": False},
           "ids": [{"role": "identifier", "tier": "1", "before_cutoff": True,
                    "species_right": True, "genus_right": True}]}
    with pytest.raises(ValueError, match="not 2 or later"):
        po.check_format([old], "e31")
    po.check_format([{**old, "format": 3}], "e31")        # format 3: a superset of 2
    with pytest.raises(ValueError, match="carries no taxon name"):
        po.expert_outcome({**old, "format": 2}, find=f)


def _small_taxonomy(name_equiv, extras=None):
    """The frozen snapshot's shape: Psathyrella candolleana retired into Candolleomyces
    candolleanus (one successor); Coprinellus micaceus beside them."""
    taxa = {1: {"name": "Psathyrella candolleana", "rank_level": 10, "is_active": False,
                "ancestor_ids": [10], "current_synonymous_taxon_ids": [2]},
            2: {"name": "Candolleomyces candolleanus", "rank_level": 10, "is_active": True,
                "ancestor_ids": [20]},
            3: {"name": "Coprinellus micaceus", "rank_level": 10, "is_active": True,
                "ancestor_ids": [30]},
            10: {"name": "Psathyrella", "rank_level": 20, "is_active": True, "ancestor_ids": []},
            20: {"name": "Candolleomyces", "rank_level": 20, "is_active": True,
                 "ancestor_ids": []},
            30: {"name": "Coprinellus", "rank_level": 20, "is_active": True, "ancestor_ids": []}}
    return name_equiv.Taxonomy(taxa, ["Psathyrella candolleana", "Coprinellus micaceus"],
                               extras=extras)


def test_a_genus_moved_on_inat_is_right_both_ways_and_a_neighbour_genus_is_not():
    # PLAN rule 4: Psathyrella candolleana is Candolleomyces candolleanus on iNat. A Vision
    # answer in the label's genus and an identifier's answer in iNat's genus are both right;
    # Coprinellus is not.
    from mycomap_vision import name_equiv
    key = po.AnswerKey(KEY, _small_taxonomy(name_equiv))
    truth = po.Find("1", "o", "Psathyrella candolleana", "Psathyrella", "Psathyrellaceae",
                    "formal", 9, 1)
    vision = po.from_ranked(["Psathyrella candolleana"], ["Psathyrella"], truth, key=key)
    expert = po.expert_outcome(
        {"record_key": "1", "status": "ok", "format": 2, "community": {"present": False},
         "ids": [{"role": "identifier", "tier": "1", "before_cutoff": True,
                  "species_taxon": {"id": 2, "name": "Candolleomyces candolleanus"},
                  "genus_taxon": {"id": 20, "name": "Candolleomyces"}}]}, find=truth, key=key)
    wrong = po.from_ranked(["Coprinellus micaceus"], ["Coprinellus"], truth, key=key)
    assert vision["species_strict@1"] and vision["genus_strict@1"]
    assert expert["species_strict@1"] and expert["genus_strict@1"]
    assert not wrong["species_strict@1"] and not wrong["genus_strict@1"]


def test_the_taxonomy_files_are_pinned_and_extras_never_move_a_label(tmp_path, monkeypatch):
    from mycomap_vision import name_equiv
    snap, succ, vocab, extra = (tmp_path / n for n in ("taxa.csv.gz", "succ.tsv", "vocab.txt",
                                                      "extras.json"))
    for f, body in ((snap, b"snap"), (succ, b"taxon_id\tcurrent_ids\n"),
                    (vocab, b"Psathyrella candolleana\nCoprinellus micaceus\n")):
        f.write_bytes(body)
    base = _small_taxonomy(name_equiv)
    monkeypatch.setattr(name_equiv.Taxonomy, "load", classmethod(
        lambda cls, s, labels=(), successors=None: cls(base.base, labels, {"taxa": "t"})))
    pin = lambda f: {"file": str(f), "sha256": paper_score.sha256_file(f)}  # noqa: E731
    block = {"snapshot": pin(snap), "successors": pin(succ),
             "label_vocabulary": {**pin(vocab), "labels": 2}}
    tx, cite = paper_score.answer_key_taxonomy(block, ["Psathyrella candolleana", "X sp. 'Y1'"])
    assert cite["labels"] == 2 and cite["truths_outside_vocabulary"] == 1
    assert tx.resolves("Psathyrella candolleana") and not tx.resolves("X sp. 'Y1'")
    with pytest.raises(ValueError, match="label vocabulary"):
        paper_score.answer_key_taxonomy({**block, "label_vocabulary": {
            "file": str(vocab), "sha256": "0" * 64}})
    # An answer-side taxon the snapshot lacks: fine. One that re-points a label: refused.
    extra.write_text(json.dumps({"99": {"name": "Candolleomyces novus", "rank_level": 10,
                                        "is_active": True, "ancestor_ids": [20]}}))
    tx, cite = paper_score.answer_key_taxonomy({**block, "extras": {
        "rule": "answer-side taxa", "e31b": pin(extra), "v1_test": "fetched at scoring"}})
    assert cite["extra_taxa"] == 1                     # the config's documented shape
    extra.write_text(json.dumps({"98": {"name": "Coprinellus micaceus", "rank_level": 10,
                                        "is_active": True, "ancestor_ids": [30]}}))
    monkeypatch.setattr(name_equiv.Taxonomy, "__init__", _moving_init(name_equiv))
    with pytest.raises(ValueError, match="change how"):
        paper_score.answer_key_taxonomy({**block, "extras": [pin(extra)]})


def _moving_init(name_equiv):
    """A Taxonomy whose extras (wrongly) join label resolution, to prove the guard fires."""
    orig = name_equiv.Taxonomy.__init__

    def init(self, taxa, labels=(), cite=None, extras=None):
        orig(self, {**taxa, **(extras or {})}, labels, cite)
    return init


def test_truths_that_do_not_resolve_leave_only_the_hypotheses_that_ask(world, tmp_path,
                                                                       monkeypatch):
    from mycomap_vision import name_equiv

    class Tx:                                    # "A x" resolves, "B y" does not
        def resolves(self, label):
            return label == "A x"

        def species_id(self, x, one_to_one=True):
            return None

        current = species_id
    monkeypatch.setattr(paper_score, "answer_key_taxonomy", lambda block, truths=(): (
        Tx(), {"taxa": "t"}))
    monkeypatch.setattr(paper_score, "genus_known", lambda tx, g, s: True)
    monkeypatch.setattr(name_equiv, "species_credit", lambda a, t, rule=None, taxonomy=None:
                        a == t)
    monkeypatch.setattr(name_equiv, "collapse", lambda a, rule=None, taxonomy=None: list(a))
    monkeypatch.setattr(name_equiv, "species_match", lambda a, t, taxonomy=None: {
        "strict": a == t, "sl": a == t, "complex": a == t})
    monkeypatch.setattr(name_equiv, "genus_match", lambda a, t, taxonomy=None, truth_species=None:
                        {"strict": a == t, "sl": a == t})
    cfg = config(tmp_path, ["5", "6"], answer_key_taxonomy={"snapshot": {}})
    cfg["hypotheses"]["A"][0]["truth_resolves"] = True                  # H1a asks
    rep = run(world, tmp_path, "val", cfg)
    assert rep["truth_resolution"] == {"yes": 1, "no": 1, "(no species)": 0}
    assert rep["hypotheses"]["A"]["H1a"]["n"] == 1                       # B y left out
    assert rep["hypotheses"]["A"]["H4"]["n"] == 2                        # didn't ask
    assert rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]["table"][
        "species strict"]["top1"]["n"] == 2                              # Vision keeps both


def test_genus_known_takes_an_inat_genus_or_a_resolved_species():
    from mycomap_vision import name_equiv
    tx = _small_taxonomy(name_equiv)
    assert paper_score.genus_known(tx, "Psathyrella", None)
    assert paper_score.genus_known(tx, "Candolleomyces", None)
    assert paper_score.genus_known(tx, "Cryptogenus", "Psathyrella candolleana")   # by species
    assert not paper_score.genus_known(tx, "Cryptogenus", "Cryptogenus sp. 'X1'")
    assert paper_score.genus_known(tx, "", "Coprinellus sp. 'X1'")    # the label's first word

    class Shared:                                # paper-score asks the shared rule only
        def genus_known(self, label, genus=None):
            seen.append((label, genus))
            return True
    seen = []
    paper_score.genus_known(Shared(), "Amanita", "Amanita muscaria")
    paper_score.genus_known(Shared(), "Amanita", "")
    assert seen == [("Amanita muscaria", "Amanita"), ("Amanita", "Amanita")]


def test_truths_whose_genus_inat_cannot_name_leave_only_the_genus_tests_that_ask(
        world, tmp_path, monkeypatch):
    class Tx:                                    # genus A is an iNat genus, B is not
        def resolves(self, label):
            return True
    monkeypatch.setattr(paper_score, "answer_key_taxonomy", lambda block, truths=(): (
        Tx(), {"taxa": "t"}))
    monkeypatch.setattr(paper_score, "genus_known", lambda tx, g, s: g == "A")
    monkeypatch.setattr(po.AnswerKey, "__post_init__", lambda self: object.__setattr__(
        self, "taxonomy", None))                 # judge plainly; only the attrs matter here
    cfg = config(tmp_path, ["5", "6"], answer_key_taxonomy={"snapshot": {}})
    cfg["hypotheses"]["A"].append({"id": "H1b", "kind": "mcnemar", "a": "vision-photos",
                                   "b": "inat-vision", "cell": "genus_strict@1",
                                   "truth_genus_known": True})
    rep = run(world, tmp_path, "val", cfg)
    assert rep["truth_genus_known"] == {"yes": 1, "no": 1, "(no genus)": 0}
    h1b = rep["hypotheses"]["A"]["H1b"]
    assert h1b["n"] == 1 and h1b["kept_in"]["n"] == 2              # sensitivity: kept in
    assert "kept_in" not in rep["hypotheses"]["A"]["H4"]
    assert rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]["table"][
        "genus strict"]["top1"]["n"] == 2                          # Vision keeps every truth


def test_extras_are_read_from_a_list_or_from_the_configs_documented_mapping():
    one = {"file": "data/x.json", "sha256": "a" * 64, "taxa": 5}
    assert paper_score.extras_files({"extras": [one]}) == [one]
    assert paper_score.extras_files({"extras": {"rule": "answer-side taxa ...", "e31b": one,
                                                "v1_test": "fetched at scoring"}}) == [one]
    assert paper_score.extras_files({}) == [] and paper_score.extras_files(None) == []


def test_a_finds_depth_counts_references_sharing_its_answer_key():
    rows = [{"record_key": "1", "species": "Pseudohydnum gelatinosum", "genus": "Pseudohydnum",
             "observer_login": "a"}]
    refs = [{"species": "Pseudohydnum gelatinosum ssp. pusillum", "observer_login": "b"}] * 3
    f, = paper_score.build_finds(rows, {}, refs)
    assert f.depth == 3                                    # PLAN rule 7: band 1-4, not 0
    hyb = [{"record_key": "2", "species": "Physisporinus sp. 'IN01 x VT01'",
            "genus": "Physisporinus", "observer_login": "a"}]
    g, = paper_score.build_finds(hyb, {}, [{"species": "Physisporinus sp. 'IN01'"}] * 4)
    assert g.depth == 0                                    # a hybrid keeps its own key


def _leave_out_world(seed=3):
    """Many species, observers and multi-photo records; species "S9 only" has every
    reference record from observer o1 (it must vanish for o1's finds)."""
    from mycomap_vision.evaluate import Record
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(10, 16)).astype(np.float32)
    vecs, ref, scored = [], [], []

    def photo(sp):
        v = centres[sp] + 0.6 * rng.normal(size=16).astype(np.float32)
        vecs.append(v / np.linalg.norm(v))
        return len(vecs) - 1
    for i in range(80):
        sp = i % 9
        obs = f"o{i % 5}"
        ref.append(Record(f"r{i}", f"G{sp % 3} s{sp}", f"G{sp % 3}", "F", "2026-01-01", obs,
                          [photo(sp) for _ in range(1 + i % 3)],
                          observed_on=f"2026-0{1 + i % 4}-01"))
    for i in range(4):                                 # S9: only o1's records
        ref.append(Record(f"s9-{i}", "G0 only", "G0", "F", "2026-01-01", "o1",
                          [photo(9), photo(9)], observed_on="2026-02-01"))
    for i in range(30):
        sp = (i * 7) % 10
        name = "G0 only" if sp == 9 else f"G{sp % 3} s{sp}"
        scored.append(Record(f"q{i}", name, name.split()[0], "F", None, f"o{i % 6}",
                             [photo(sp) for _ in range(1 + i % 2)],
                             observed_on=f"2026-0{1 + i % 4}-01"))
    return np.stack(vecs), ref, scored


def _world_twins(ref):
    """Cross-post twin groups in _leave_out_world: pairs of same-species records from two
    observers (so an observer's leave-out hides one side of a pair and keeps the other),
    and one triple."""
    by_species: dict = {}
    for r in ref:
        by_species.setdefault(r.species, []).append(r)
    groups = {}
    for name, recs in sorted(by_species.items()):
        a, b = recs[0], next((r for r in recs[1:] if r.observer != recs[0].observer), None)
        if b is not None:
            groups[a.observation_id] = groups[b.observation_id] = f"twin-{a.observation_id}"
    groups["r1"] = groups["r10"] = groups["r19"] = "twin-r1"       # a triple
    return groups


@pytest.mark.parametrize("method", ["nearest", "nearest+mean", "nearest+mean-records",
                                    "nearest+mean-records-balanced"])
@pytest.mark.parametrize("mode", ["observer", "trip"])
def test_a_masked_leave_out_equals_a_refit_without_the_groups_records(method, mode):
    from mycomap_vision import twins
    vec, ref, scored = _leave_out_world()
    truths = {r.observation_id: {"species": r.species, "genus": r.genus} for r in scored}
    cal = {"species": {"temperature": 0.05, "floor": 0.05}, "genus": {"temperature": 0.05}}
    # a record arm counts cross-post twins as one specimen; the refit regroups whatever
    # of each pair is left in its reference
    kw = dict(top_k=10, calibration=cal, truths=truths,
              twins=_world_twins(ref) if twins.is_grouped(method) else None)
    masked = paper_score.masked_leave_out_answers(vec, ref, scored, method, "m", mode, **kw)
    refit = paper_score.refit_leave_out_answers(vec, ref, scored, method, "m", mode, **kw)
    assert masked.keys() == refit.keys()
    for k in refit:
        assert masked[k]["species"] == refit[k]["species"], k
        assert masked[k]["genus"] == refit[k]["genus"], k
        for rank in ("species", "genus"):
            for f in ("conf", "p_true", "sumsq"):
                # to float rounding: a touched species' average is summed in another order
                assert masked[k]["cal"][rank][f] == pytest.approx(refit[k]["cal"][rank][f],
                                                                  rel=1e-3, abs=1e-4), (k, rank, f)
        assert masked[k]["cal"]["species"]["set"] == refit[k]["cal"]["species"]["set"]
    # A find of observer o1: "G0 only" has no reference left, so it is no answer at all.
    o1 = [r.observation_id for r in scored if r.observer == "o1"]
    if mode == "observer":
        assert all("G0 only" not in masked[k]["species"] for k in o1)



# The leave-out as it was before the plan (methods.py until 26 Oct), kept verbatim as the
# reference the plan-once path must reproduce.
def _legacy_leave_out(model, per_photo, photo_sims, query, hidden):
    from mycomap_vision.methods import NearestAndMeanRecords, rows_at
    starts = model.index.starts
    n_cols = len(model.index.cols)
    hidden = np.unique(np.asarray(hidden, dtype=np.int64))
    q = np.asarray(query, dtype=np.float32)
    out = np.array(per_photo, dtype=np.float32, copy=True)
    records = isinstance(model, NearestAndMeanRecords)
    rec_starts = model.index.rec_starts if records else None
    for u in np.unique(np.searchsorted(starts, hidden, side="right") - 1).tolist():
        lo = int(starts[u])
        hi = int(starts[u + 1]) if u + 1 < len(starts) else n_cols
        keep = np.setdiff1d(np.arange(lo, hi), hidden)
        if not len(keep):
            out[:, u] = -np.inf
            continue
        if records:
            r_hi = int(model.rec_species[u + 1]) if u + 1 < len(starts) else len(rec_starts)
            bounds = np.append(rec_starts[int(model.rec_species[u]):r_hi], hi)
            rec = np.searchsorted(bounds, keep, side="right") - 1
            firsts = np.flatnonzero(np.diff(rec, prepend=-1) != 0)
            best = np.maximum.reduceat(photo_sims[:, keep], firsts, axis=1)
            best = -np.sort(-best, axis=1)[:, :model.k]
            mean = model._species_mean_without(keep, bounds).astype(np.float16)
        else:
            best = -np.sort(-photo_sims[:, keep], axis=1)[:, :model.k]
            mean = rows_at(model.scorer.ref, keep).sum(axis=0)
            mean = (mean / max(float(np.linalg.norm(mean)), 1e-12)).astype(np.float16)
        nearest = best.sum(axis=1, dtype=np.float32) / np.float32(best.shape[1])
        out[:, u] = (model.weight * nearest
                     + (1 - model.weight) * (q @ mean.astype(np.float32)))
    return out


@pytest.mark.parametrize("method", ["nearest+mean", "nearest+mean-records",
                                    "nearest+mean-records-balanced"])
@pytest.mark.parametrize("mode", ["observer", "trip"])
def test_the_plan_once_leave_out_gives_the_former_leave_outs_results(method, mode,
                                                                    monkeypatch):
    """Speed (26 Oct): the query-free part of the leave-out is built once per group of
    finds (leave_out_plan) and applied per find. On a seeded sample it gives the per-photo
    scores of the former species-by-species code (to float rounding of one product), the
    same absent species, the same ranking of every photo, and the same answers."""
    from mycomap_vision import evaluate, twins
    vec, ref, scored = _leave_out_world(seed=11)
    groups = _world_twins(ref) if twins.is_grouped(method) else None
    index, model = evaluate.fit_method(
        vec, ref, method, "m",
        **({"specimen_of": twins.specimen_of(groups, ref)} if groups else {}))
    col_of_row = {int(r): i for i, r in enumerate(np.asarray(index.cols).tolist())}
    hidden_of = {}
    for r in ref:
        hidden_of.setdefault(paper_score.leave_out_key(r, mode), []).extend(
            col_of_row[int(p)] for p in r.photo_rows)
    compared = 0
    for rec in scored:
        hidden = np.asarray(hidden_of.get(paper_score.leave_out_key(rec, mode), []), np.int64)
        if not len(hidden):
            continue
        q = np.asarray(vec[rec.photo_rows], dtype=np.float32)
        sims = model.photo_sims(q)
        per = model.per_photo_scores(sims, q)
        old = _legacy_leave_out(model, per, sims, q, hidden)
        new = model.leave_out_apply(per, sims, q, model.leave_out_plan(hidden))
        assert np.array_equal(np.isinf(old), np.isinf(new))
        fin = np.isfinite(old)
        assert np.allclose(new[fin], old[fin], rtol=0, atol=1e-6)
        assert np.array_equal(np.argsort(-old, axis=1, kind="stable"),
                              np.argsort(-new, axis=1, kind="stable"))
        assert np.array_equal(model.leave_out(per, sims, q, hidden), new)
        compared += 1
    assert compared >= 20
    # end to end: the answers paper-score gives are those of the former leave-out
    truths = {r.observation_id: {"species": r.species, "genus": r.genus} for r in scored}
    cal = {"species": {"temperature": 0.05, "floor": 0.05}, "genus": {"temperature": 0.05}}
    kw = dict(top_k=10, calibration=cal, truths=truths, twins=groups)
    got = paper_score.masked_leave_out_answers(vec, ref, scored, method, "m", mode, **kw)
    with monkeypatch.context() as m:
        orig_plan = type(model).leave_out_plan

        def plan_with_hidden(self, hidden):
            plan = orig_plan(self, hidden)
            plan.hidden = hidden
            return plan
        m.setattr(type(model), "leave_out_plan", plan_with_hidden)
        m.setattr(type(model), "leave_out_apply",
                  lambda self, per, sims, q, plan: _legacy_leave_out(self, per, sims, q,
                                                                     plan.hidden))
        want = paper_score.masked_leave_out_answers(vec, ref, scored, method, "m", mode, **kw)
    assert {k: (a["species"], a["genus"]) for k, a in got.items()} == \
        {k: (a["species"], a["genus"]) for k, a in want.items()}
    for k in got:
        for rank in ("species", "genus"):
            for f in ("conf", "p_true", "sumsq"):
                assert got[k]["cal"][rank][f] == pytest.approx(want[k]["cal"][rank][f],
                                                               rel=1e-5, abs=1e-6)


def test_the_leave_out_plan_is_built_once_per_group_not_once_per_find(monkeypatch):
    from mycomap_vision import methods
    vec, ref, scored = _leave_out_world(seed=3)
    built = []
    orig = methods.NearestAndMean.leave_out_plan
    monkeypatch.setattr(methods.NearestAndMean, "leave_out_plan",
                        lambda self, hidden: built.append(len(hidden)) or orig(self, hidden))
    paper_score.masked_leave_out_answers(vec, ref, scored, "nearest+mean", "m", "observer")
    groups_with_finds = {r.observer for r in scored if r.observer
                         and any(x.observer == r.observer for x in ref)}
    finds_with_hidden = sum(1 for r in scored if r.observer in groups_with_finds)
    assert len(built) == len(groups_with_finds) < finds_with_hidden


def test_a_place_methods_leave_out_is_refitted_never_masked(monkeypatch):
    seen = []
    monkeypatch.setattr(paper_score, "refit_leave_out_answers",
                        lambda *a, **k: seen.append(a[3]) or {})
    monkeypatch.setattr(paper_score, "masked_leave_out_answers",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("masked")))
    paper_score.vision_answers_leave_out(None, [], [], "nearest+mean+place", "m", "observer")
    assert seen == ["nearest+mean+place"]
def test_a_comparators_own_likely_set_target_is_printed_beside_its_achieved_coverage():
    finds = [po.Find(str(i), f"o{i}", "A x", "A", "F", "formal", 3, 1) for i in range(4)]
    cal = {f.key: {"species": {"conf": 0.6, "p_true": 0.6, "sumsq": 0.4, "set": ["A x"]}}
           for f in finds}
    outcomes = {"v": {f.key: {"species_strict@1": True} for f in finds},
                "p": {f.key: {"species_strict@1": True} for f in finds}}
    comps = [{"id": "v", "calibration": {"species": {"temperature": 0.03, "floor": 0.05,
                                                     "target": 0.75}}},
             {"id": "p", "calibration": {"species": {"temperature": 0.03, "floor": 0.03}}}]
    rep = paper_score.calibration_report(finds, {"v": cal, "p": cal}, outcomes,
                                         {"list_target": {"species": 0.8, "genus": 0.9}},
                                         50, 0, comps)
    assert rep["v"]["species"]["likely_sets"]["target"] == 0.75     # its own fit
    assert rep["p"]["species"]["likely_sets"]["target"] == 0.8      # the config-wide one


# --- cross-post twins for the record-grouped methods (E32 arm c, O8) ----------------------------

def test_a_record_grouped_methods_scores_are_the_e32_call_with_twins_grouped():
    """paper-score's answers for arm (c) are those of evaluate.fit_method with specimen_of
    = the groups present in the reference (the record-scoring lane's E32 call), and differ
    from the same arm with the twins left apart."""
    from mycomap_vision import evaluate, twins
    vec, ref, scored = _leave_out_world()
    groups = _world_twins(ref)
    m = "nearest+mean-records-balanced"
    index, model = evaluate.fit_method(vec, ref, m, "m", specimen_of=twins.specimen_of(groups, ref))
    want = {r.observation_id: evaluate.top_labels(
        evaluate.rank_scores(model.species_scores(vec[r.photo_rows]), index, "species"),
        index.labels["species"], 10) for r in scored}
    info: dict = {}
    got = paper_score.vision_answers(vec, ref, scored, m, "m", twins=groups, info=info)
    assert {k: a["species"] for k, a in got.items()} == want
    assert info["twin_records"] == len(twins.specimen_of(groups, ref)) > 0
    index0, model0 = evaluate.fit_method(vec, ref, m, "m")
    apart = {r.observation_id: model0.species_scores(vec[r.photo_rows]) for r in scored}
    grouped = {r.observation_id: model.species_scores(vec[r.photo_rows]) for r in scored}
    assert any(not np.allclose(apart[k], grouped[k]) for k in apart)   # grouping matters


def test_a_record_grouped_method_without_twin_groups_is_refused():
    vec, ref, scored = _leave_out_world()
    with pytest.raises(ValueError, match="twin"):
        paper_score.vision_answers(vec, ref, scored, "nearest+mean-records-balanced", "m")
    paper_score.vision_answers(vec, ref, scored, "nearest+mean", "m")     # not grouped: fine


def test_a_config_with_a_record_grouped_method_and_no_twins_block_is_refused():
    comps = [{"id": "c", "kind": "vision", "backbone": "m1",
              "method": "nearest+mean-records-balanced"}]
    with pytest.raises(ValueError, match="twins"):
        paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": comps})
    paper_score.check_config({"format": 1, "answer_key": KEY, "comparators": comps,
                              "twins": {"validation": {"file": "x", "sha256": "y"}}})


def _twins_block(tmp_path):
    """Val's groups (train: 1 + mo:9, both A x) and the sealed test's (train + val: 1, mo:9
    and 5), each pinned by its sha256."""
    import hashlib
    block = {}
    for which, keys in (("validation", ["1", "mo:9"]), ("sealed_test", ["1", "5", "mo:9"])):
        f = tmp_path / f"twins-{which}.tsv"
        f.write_text("observation_id\tspecimen\n" + "".join(f"{k}\ttwin-1\n" for k in keys),
                     encoding="utf-8")
        block[which] = {"file": str(f), "sha256": hashlib.sha256(f.read_bytes()).hexdigest()}
    return block


def _grouped_config(tmp_path, keys, twins_block):
    cfg = config(tmp_path, keys, twins=twins_block)
    cfg["comparators"].append({"id": "vision-c", "kind": "vision", "backbone": "m1",
                               "method": "nearest+mean-records-balanced"})
    return cfg


def test_val_groups_twins_by_the_validation_file_and_reports_it(world, tmp_path):
    block = _twins_block(tmp_path)
    rep = run(world, tmp_path, "val", _grouped_config(tmp_path, ["5", "6"], block))
    t = rep["inputs"]["vision-c"]["twins"]
    assert t["for"] == "validation" and t["sha256"] == block["validation"]["sha256"]
    assert t["records"] == 2 and t["grouped_records"] == 2
    assert "twins" not in rep["inputs"]["vision-mean"]         # an ungrouped method
    c = rep["standard"]["comparators"]["vision-c"]["primary"]["all"]
    assert c["table"]["species strict"]["top1"]["n"] == 2


def test_a_test_release_groups_twins_by_the_sealed_test_file(world, tmp_path):
    block = _twins_block(tmp_path)
    rep = run(world, tmp_path, world["test"], _grouped_config(tmp_path, ["7", "8"], block))
    t = rep["inputs"]["vision-c"]["twins"]
    assert t["for"] == "sealed_test" and t["sha256"] == block["sealed_test"]["sha256"]
    assert t["grouped_records"] == 3                           # 1, mo:9 and val's 5


def test_twin_groups_are_refused_unless_the_file_is_the_pinned_one(world, tmp_path):
    block = _twins_block(tmp_path)
    bad = {**block, "validation": {**block["validation"], "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="sha256"):
        run(world, tmp_path, "val", _grouped_config(tmp_path, ["5", "6"], bad))
    gone = {**block, "validation": {**block["validation"],
                                    "file": str(tmp_path / "no-such.tsv")}}
    with pytest.raises(ValueError, match="missing"):
        run(world, tmp_path, "val", _grouped_config(tmp_path, ["5", "6"], gone))
    tbd = {**block, "sealed_test": {"file": "TBD (mv twin-groups)", "sha256": "TBD"}}
    with pytest.raises(ValueError, match="no sealed_test file"):
        run(world, tmp_path, world["test"], _grouped_config(tmp_path, ["7", "8"], tbd))
    with pytest.raises(ValueError, match="twins"):
        run(world, tmp_path, "val", _grouped_config(tmp_path, ["5", "6"], None))
    assert not (tmp_path / "out" / "report.json").exists()     # refused before scoring


def test_the_freeze_spec_pins_the_twin_groups_files_for_a_grouped_method(tmp_path, monkeypatch):
    from mycomap_vision import paper_freeze
    from mycomap_vision.replications.fungitastic import dataset as ds
    monkeypatch.setattr(ds, "cite", lambda b: {})
    block = _twins_block(tmp_path)
    base = type("Base", (), {"db": type("Db", (), {"execute": lambda self, q: []})()})()

    def spec(method):
        cfg = {"comparators": [{"id": "c", "kind": "vision", "backbone": "TBD",
                                "method": method}], "twins": block}
        return paper_freeze.compute(cfg, base, None, Path, models_root=tmp_path,
                                    load_vectors=lambda b: None, taxonomy_path=tmp_path / "x")
    assert spec("nearest+mean-records-balanced")["twins"] == {
        k: v["sha256"] for k, v in block.items()}
    assert "twins" not in spec("nearest+mean")



def test_a_provisional_truth_that_resolves_enters_h1a_and_inat_is_matched_by_taxon_id():
    """Rule 4 (27 Oct): H1a keeps truths that resolve one-to-one, formal or provisional.
    A provisional truth iNat holds as a taxon is scored at species for iNat (its answer
    matched by taxon id), enters H1a's paired set, and is counted as provisional in its n."""
    from mycomap_vision import name_equiv as ne
    from mycomap_vision import paper_tables as pt

    def tx_(name, level, anc=()):
        return {"name": name, "rank_level": level, "is_active": True, "ancestor_ids": list(anc)}
    taxa = {1: tx_("Fungi", 70), 50: tx_("Russula", 20, [1]),
            700: tx_("Russula sp. 'NC09'", 10, [1, 50]), 800: tx_("Russula emetica", 10, [1, 50])}
    labels = ["Russula sp. 'NC09'", "Russula emetica", "Russula sp. 'NC10'"]
    tax = ne.Taxonomy(taxa, labels)
    key = po.AnswerKey(ne.ANSWER_KEY, tax)
    finds = [po.Find("p1", "o1", "Russula sp. 'NC09'", "Russula", "", "provisional", 5, 1),
             po.Find("f1", "o2", "Russula emetica", "Russula", "", "formal", 5, 1),
             po.Find("p2", "o3", "Russula sp. 'NC10'", "Russula", "", "provisional", 5, 1)]
    for f in finds:
        f.attrs["truth_resolves"] = paper_score.truth_resolution(tax, f.species)
        f.attrs["truth_genus_known"] = "yes"
    assert [f.attrs["truth_resolves"] for f in finds] == ["yes", "yes", "no"]

    def line(k, tid, name, formal):
        return {"record_key": k, "truth": {"formal_species": formal},
                # iNat's own spelling differs: the taxon id is what matches
                "vision": {"species": [[tid, name, 90]], "genus": [[50, "Russula", 95]]},
                "answered": {"vision": True}}
    inat = {"p1": po.inat_outcome(line("p1", 700, "Russula sp NC09", False), finds[0], "vision", key),
            "f1": po.inat_outcome(line("f1", 800, "Russula emetica", True), finds[1], "vision", key),
            "p2": po.inat_outcome(line("p2", 700, "Russula sp NC09", False), finds[2], "vision", key)}
    assert inat["p1"]["species_strict@1"] is True                     # scored, by taxon id
    assert inat["p2"]["species_strict@1"] is None                     # no iNat taxon: unscored
    vision = {f.key: po.from_ranked([f.species], ["Russula"], f, key=key) for f in finds}
    spec = {"id": "H1a", "kind": "mcnemar", "a": "vision", "b": "inat",
            "cell": "species_strict@1", "truth_resolves": True}
    outcomes = {"vision": vision, "inat": inat}
    assert [f.key for f in pt.select(finds, spec, outcomes)] == ["p1", "f1"]
    r = pt.test_one(spec, finds, outcomes, 50, 1)
    assert r["n"] == 2 and r["n_by_name_kind"] == {"formal": 1, "provisional": 1}



# --- E14 secondary rows (Steve 27 Oct): descriptive, outside Holm ------------------------------

def _f(key, kind="formal", **attrs):
    f = po.Find(key, "o" + key, "A x", "A", "", kind, 5, 1)
    f.attrs.update(attrs)
    return f


@pytest.mark.parametrize("where, want", [
    ({}, [True, True, True, True]),
    ({"name_kind": "formal"}, [True, False, True, True]),
    ({"created_after": "2024-05-01"}, [True, False, None, False]),     # same day is not after
    ({"rg_at_inat_export": "no"}, [False, True, None, True]),
    ({"inat_observation": "none"}, [False, False, False, True]),
    ({"any_of": [{"inat_observation": "none"}, {"created_after": "2024-05-01"}]},
     [True, False, None, True]),
    ({"name_kind": "formal", "created_after": "2024-05-01"}, [True, False, None, False]),
])
def test_each_secondary_row_condition_keeps_the_finds_it_names_and_marks_unknowns(where, want):
    from mycomap_vision import paper_tables as pt
    finds = [_f("1", created_at="2024-06-02T10:00:00Z", rg_at_inat_export="yes",
                inat_observation="yes"),
             _f("2", "provisional", created_at="2023-01-01T00:00:00Z", rg_at_inat_export="no",
                inat_observation="yes"),
             _f("3", created_at="", rg_at_inat_export="unknown", inat_observation="yes"),
             _f("4", created_at="2024-05-01T23:59:00Z", rg_at_inat_export="no",
                inat_observation="none")]
    assert [pt.where_holds(f, where) for f in finds] == want


def test_secondary_rows_run_the_named_hypotheses_on_each_row_outside_holm(world, tmp_path):
    attrs = tmp_path / "attrs.tsv"
    attrs.write_text("record_key\tcreated_at\trg_at_inat_export\tinat_observation\n"
                     "5\t2025-01-02T00:00:00Z\tno\tyes\n6\t\tunknown\tyes\n", encoding="utf-8")
    cfg = config(tmp_path, ["5", "6"])
    cfg["inat_cv"] = {"model_version": "TBD", "training_export_date": "TBD: at the cut"}
    cfg["secondary_rows"] = {"for": ["H1a"], "rows": [
        {"id": "all-finds", "where": {}},
        {"id": "unseen-by-bioclip2", "where": {"created_after": "2024-05-01"}},
        {"id": "unseen-by-inat", "where": {"any_of": [
            {"inat_observation": "none"},
            {"created_after": {"config": "inat_cv/training_export_date"}}]}},
        {"id": "not-rg", "where": {"rg_at_inat_export": "no"}}]}
    paper_score.check_config(cfg)
    rep = run(world, tmp_path, "val", cfg, attrs_path=attrs)
    rows = rep["secondary_rows"]
    assert rows["all-finds"]["finds"] == 2 and rows["all-finds"]["share_of_scored"] == 1.0
    assert rows["all-finds"]["hypotheses"]["H1a"]["n"] == rep["hypotheses"]["A"]["H1a"]["n"]
    assert rows["unseen-by-bioclip2"]["finds"] == 1 and rows["unseen-by-bioclip2"]["unknown"] == 1
    assert rows["unseen-by-bioclip2"]["hypotheses"]["H1a"]["n"] == 1
    h = rows["not-rg"]["hypotheses"]["H1a"]
    assert h["outside_holm"] and h["p_holm"] is None and h["reject"] is None
    assert rows["unseen-by-inat"] == {
        "not_run": "not run: TBD (inat_cv/training_export_date is still a placeholder)"}
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Secondary rows (descriptive, outside Holm)" in md


@pytest.mark.parametrize("change, refused", [
    (lambda b: b["rows"].append({"id": "x", "where": {"colour": "red"}}), "unknown condition"),
    (lambda b: b["for"].append("H9"), "unknown hypothesis"),
    (lambda b: b["rows"].append({"id": "x", "where": {"created_after": "May 2024"}}),
     "is a date"),
    (lambda b: b["rows"].append({"id": "x", "where": {"created_after": {"config": "inat_cv/nope"}}}),
     "no such field"),
    (lambda b: b["rows"].append({"id": "all-finds", "where": {}}), "unique id"),
    (lambda b: b["rows"].append({"id": "x", "where": {"any_of": []}}), "any_of is a list"),
])
def test_a_malformed_secondary_rows_block_is_refused(tmp_path, change, refused):
    cfg = config(tmp_path, ["5", "6"])
    cfg["inat_cv"] = {"training_export_date": "TBD"}
    cfg["secondary_rows"] = {"for": ["H1a"], "rows": [{"id": "all-finds", "where": {}}]}
    change(cfg["secondary_rows"])
    with pytest.raises(ValueError, match=refused):
        paper_score.check_config(cfg)



def test_paper_score_opens_the_manifest_read_only_and_leaves_its_bytes_unchanged(tmp_path,
                                                                               monkeypatch):
    """The paper's manifest is a read-only file pinned by sha256: paper-score (freeze spec
    and scoring) opens it read-only, so it neither fails nor changes its bytes. Only
    --embed-missing, which writes vectors, opens it for writing."""
    import os
    import sqlite3
    from mycomap_vision import cli, config
    m = tmp_path / "manifest-v1-paper.sqlite"
    c = sqlite3.connect(m)
    c.execute("create table t (x)")
    c.commit()
    c.close()
    os.chmod(m, 0o444)
    before = m.read_bytes()
    monkeypatch.setattr(config, "MANIFEST_PATH", m)
    seen = []

    def stub(conn, args):
        try:
            conn.execute("create table u (y)")
            seen.append("wrote")
        except sqlite3.OperationalError as e:
            seen.append(str(e))
    monkeypatch.setattr(cli, "cmd_paper_score", stub)
    try:
        assert cli.main(["paper-score", "--release", "v1", "--test", "val",
                         "--config", "c.json", "--freeze-spec"]) == 0
        assert len(seen) == 1 and "readonly" in seen[0]
        assert m.read_bytes() == before
        assert not (tmp_path / (m.name + "-wal")).exists()
    finally:
        os.chmod(m, 0o644)



# --- amendments: a chain of freeze tags (paper coordinator, 27 Oct) -------------------------------

NEW_CODE = "bbbbbbb1234567"


def amend_base():
    """A frozen plan with an expert comparator, its post_freeze file, a hypothesis and a slice
    naming it: what an amendment removing E31 takes out."""
    d = post_freeze_plan()
    d["comparators"].append({"id": "experts-t1", "kind": "expert", "file": "TBD: e31"})
    d["post_freeze"].append({"path": "comparators/experts-t1/file", "rule": "E31a at scoring",
                             "kind": "test_output"})
    d["hypotheses"] = {"A": [
        {"id": "H1a", "kind": "mcnemar", "a": "public-v1.1", "b": "inat-vision",
         "cell": "species_strict@1"},
        {"id": "H4s", "kind": "mcnemar", "a": "public-v1.1", "b": "experts-t1",
         "cell": "species_strict@1"}]}
    d["slices"] = {"comparators": ["inat-vision", "experts-t1"]}
    return d


def amended(parent_tag=None):
    import copy
    d = copy.deepcopy(amend_base())
    d["comparators"] = [c for c in d["comparators"] if c["id"] != "experts-t1"]
    d["post_freeze"] = [e for e in d["post_freeze"] if not e["path"].startswith("comparators/experts-t1")]
    d["hypotheses"]["A"] = [h for h in d["hypotheses"]["A"] if h["id"] != "H4s"]
    d["slices"]["comparators"] = ["inat-vision"]
    d["code_commit"] = NEW_CODE
    d["freeze"]["code_commit"] = NEW_CODE
    d["amends"] = {"tag": parent_tag or paper_score.TAG, "reason": "E31 removed (Steve, pre-results)",
                   "paths": ["comparators/experts-t1", "hypotheses/A/H4s", "slices/comparators"],
                   "code_files": ["src/mycomap_vision/paper_score.py"]}
    return d


def tag_as(repo, doc, tag):
    body = json.dumps(doc, indent=1).encode()
    (repo / "paper" / "analysis-v1.json").write_bytes(body)
    git(repo, "commit", "-q", "-am", f"plan for {tag}")
    git(repo, "tag", tag)
    return body


def test_an_amendment_tag_becomes_the_frozen_config_and_its_link_is_recorded(research,
                                                                            monkeypatch):
    repo, _ = research
    tag_as(repo, amend_base(), paper_score.TAG)
    a1 = amended()
    body = tag_as(repo, a1, paper_score.TAG + "-a1")
    monkeypatch.setattr(paper_score, "code_changes",
                        lambda a, b: ["src/mycomap_vision/paper_score.py"])
    g = paper_score.guard(True, body, a1, repo, NEW_CODE)
    assert g["sealed"] and g["tag"] == paper_score.TAG + "-a1" and not g["post_hoc"]
    first, link = g["chain"]
    assert first["tag"] == paper_score.TAG and first["code_commit"] == "abcdef1234567"
    assert link["reason"] == "E31 removed (Steve, pre-results)"
    assert link["paths"] == a1["amends"]["paths"]
    assert link["code_changed"] == ["src/mycomap_vision/paper_score.py"]
    base = json.dumps(amend_base(), indent=1).encode()               # the first tag's config
    with pytest.raises(paper_score.FrozenError, match="not the one tagged .*-a1"):
        paper_score.guard(True, base, amend_base(), repo, "abcdef1")
    with pytest.raises(paper_score.FrozenError, match="not the frozen paper-score commit"):
        paper_score.guard(True, body, a1, repo, "abcdef1")             # the first tag's code


@pytest.mark.parametrize("change, refused", [
    (lambda d: d.update(top_k=5), "top_k: changed"),
    (lambda d: next(c for c in d["comparators"] if c["id"] == "inat-vision").update(score="combined"),
     "comparators/inat-vision/score: not in the parent"),
    (lambda d: next(e for e in d["post_freeze"] if e["path"] == "comparators/inat-vision/file")
     .update(rule="anything"), "post_freeze comparators/inat-vision/file: changed"),
    (lambda d: d["freeze"]["models"]["m1"].update(weights="zz"), "freeze/models/m1/weights: changed"),
    (lambda d: d["amends"].update(tag=paper_score.TAG + "-a7"), "its parent in the chain is"),
    (lambda d: d["amends"].update(reason=""), "gives the reason"),
])
def test_an_amendment_that_changes_more_than_it_declares_is_refused(research, monkeypatch,
                                                                   change, refused):
    repo, _ = research
    tag_as(repo, amend_base(), paper_score.TAG)
    a1 = amended()
    change(a1)
    body = tag_as(repo, a1, paper_score.TAG + "-a1")
    monkeypatch.setattr(paper_score, "code_changes",
                        lambda a, b: ["src/mycomap_vision/paper_score.py"])
    with pytest.raises(paper_score.FrozenError, match=refused):
        paper_score.guard(True, body, a1, repo, NEW_CODE)


def test_an_amendment_whose_code_changes_undeclared_files_is_refused(research, monkeypatch):
    repo, _ = research
    tag_as(repo, amend_base(), paper_score.TAG)
    a1 = amended()
    body = tag_as(repo, a1, paper_score.TAG + "-a1")
    monkeypatch.setattr(paper_score, "code_changes", lambda a, b: [
        "src/mycomap_vision/paper_score.py", "src/mycomap_vision/methods.py"])
    with pytest.raises(paper_score.FrozenError, match="does not declare.*methods.py"):
        paper_score.guard(True, body, a1, repo, NEW_CODE)


def test_amendment_tags_out_of_sequence_are_refused(research, monkeypatch):
    repo, _ = research
    tag_as(repo, amend_base(), paper_score.TAG)
    a2 = amended()
    body = tag_as(repo, a2, paper_score.TAG + "-a2")                   # no -a1
    with pytest.raises(paper_score.FrozenError, match="out of sequence"):
        paper_score.guard(True, body, a2, repo, NEW_CODE)


def test_every_link_of_a_longer_chain_is_checked(research, monkeypatch):
    import copy
    repo, _ = research
    tag_as(repo, amend_base(), paper_score.TAG)
    a1 = amended()
    tag_as(repo, a1, paper_score.TAG + "-a1")
    a2 = copy.deepcopy(a1)
    a2["code_commit"] = a2["freeze"]["code_commit"] = "ccccccc1234567"
    a2["freeze"]["models"]["m2"] = {"weights": "cc"}                  # a part gained (E34's model)
    a2["amends"] = {"tag": paper_score.TAG + "-a1", "reason": "E34 scoring code",
                    "paths": ["top_k"], "code_files": ["src/mycomap_vision/paper_score.py"]}
    a2["top_k"] = 5
    body = tag_as(repo, a2, paper_score.TAG + "-a2")
    monkeypatch.setattr(paper_score, "code_changes",
                        lambda a, b: ["src/mycomap_vision/paper_score.py"])
    g = paper_score.guard(True, body, a2, repo, "ccccccc")
    assert [r["tag"] for r in g["chain"]] == [paper_score.TAG, paper_score.TAG + "-a1",
                                             paper_score.TAG + "-a2"]
    a2bad = copy.deepcopy(a2)
    a2bad["amends"]["tag"] = paper_score.TAG                           # skips its parent -a1
    git(repo, "tag", "-d", paper_score.TAG + "-a2")
    body = tag_as(repo, a2bad, paper_score.TAG + "-a2")
    with pytest.raises(paper_score.FrozenError, match="its parent in the chain is .*-a1"):
        paper_score.guard(True, body, a2bad, repo, "ccccccc")


def test_code_changes_lists_the_files_between_two_commits_of_the_code(tmp_path, monkeypatch):
    from mycomap_vision import config as cfgmod
    code = tmp_path / "code"
    (code / "src").mkdir(parents=True)
    git(code, "init", "-q")
    git(code, "config", "user.email", "t@example.org")
    git(code, "config", "user.name", "t")
    (code / "src" / "a.py").write_text("1\n", encoding="utf-8")
    (code / "src" / "b.py").write_text("1\n", encoding="utf-8")
    git(code, "add", "-A")
    git(code, "commit", "-q", "-m", "one")
    one = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    (code / "src" / "b.py").write_text("2\n", encoding="utf-8")
    git(code, "commit", "-q", "-am", "two")
    two = subprocess.run(["git", "-C", str(code), "rev-parse", "HEAD"], capture_output=True,
                         text=True, check=True).stdout.strip()
    monkeypatch.setattr(cfgmod, "REPO_ROOT", code)
    assert paper_score.code_changes(one, two) == ["src/b.py"]
    with pytest.raises(paper_score.FrozenError, match="cannot compare code commits"):
        paper_score.code_changes(one, "0" * 40)


@pytest.mark.parametrize("block, refused", [
    ({"tag": "other-tag", "reason": "r", "paths": [], "code_files": []}, "parent tag"),
    ({"tag": "paper-analysis-freeze-v1", "paths": [], "code_files": []}, "reason"),
    ({"tag": "paper-analysis-freeze-v1", "reason": "r", "paths": "x", "code_files": []}, "paths"),
])
def test_a_malformed_amends_block_is_refused(tmp_path, block, refused):
    cfg = config(tmp_path, ["5", "6"])
    cfg["amends"] = block
    with pytest.raises(ValueError, match=refused):
        paper_score.check_config(cfg)


# --- -a2 (Steve 28 Oct): Vision's own accuracy against a floor, family C, new where keys ---------

def floor_family(cfg):
    cfg["hypotheses"] = {
        "A": [{"id": "A1s", "kind": "floor", "comparator": "vision-photos",
               "cell": "species_strict@1", "floor": 0.5, "alternative": "greater"},
              {"id": "A1g", "kind": "floor", "comparator": "vision-photos",
               "cell": "genus_strict@1", "floor": 0.8, "alternative": "greater"}],
        "C": [{"id": "H1a", "kind": "mcnemar", "a": "vision-photos", "b": "inat-vision",
               "cell": "species_strict@1", "name_kind": "formal"}]}
    return cfg


def test_a_floor_family_scores_end_to_end_and_its_rows_render(world, tmp_path):
    cfg = floor_family(config(tmp_path, ["5", "6"]))
    cfg["secondary_rows"] = {"for": ["A1s", "H1a"], "rows": [
        {"id": "all-finds", "where": {}},
        {"id": "provisional", "where": {"name_kind": "provisional"}}]}
    paper_score.check_config(cfg)
    rep = run(world, tmp_path, "val", cfg)
    a1s = rep["hypotheses"]["A"]["A1s"]
    assert (a1s["n"], a1s["rate"], a1s["floor"]) == (2, 1.0, 0.5)
    assert a1s["p_test"] == a1s["p"] and a1s["p_holm"] is not None
    assert set(rep["hypotheses"]) == {"A", "C"}
    rows = rep["secondary_rows"]
    assert rows["all-finds"]["hypotheses"]["A1s"]["n"] == 2
    assert rows["provisional"]["finds"] == 0
    assert rows["provisional"]["hypotheses"]["A1s"]["n"] == 0
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "| A1s | 2 | 100.0% vs floor 50.0% |" in md and "### Family C" in md


@pytest.mark.parametrize("change, refused", [
    (lambda h: h["A"][0].update(kind="floors"), "kind is one of"),
    (lambda h: h["A"][0].update(floor=0), "strictly between 0 and 1"),
    (lambda h: h["A"][0].update(floor=1.0), "strictly between 0 and 1"),
    (lambda h: h["A"][0].update(floor="0.5"), "strictly between 0 and 1"),
    (lambda h: h["A"][0].update(floor=True), "strictly between 0 and 1"),
    (lambda h: h["A"][0].pop("floor"), "strictly between 0 and 1"),
    (lambda h: h["A"][0].pop("comparator"), "names its comparator"),
    (lambda h: h["A"][0].update(comparator="nope"), "unknown comparator"),
    (lambda h: h["A"][0].update(alternative="two-sided"), "greater or less"),
    (lambda h: h["C"][0].update(id="A1g"), "unique across families"),
    (lambda h: h["C"][0].update(kind="wilcoxon"), "kind is one of"),
    (lambda h: h["C"][0].pop("id"), "has an id"),
])
def test_a_hypothesis_of_unknown_kind_or_a_bad_floor_or_a_repeated_id_is_refused(
        tmp_path, change, refused):
    cfg = floor_family(config(tmp_path, ["5", "6"]))
    paper_score.check_config(cfg)
    change(cfg["hypotheses"])
    with pytest.raises(ValueError, match=refused):
        paper_score.check_config(cfg)


@pytest.mark.parametrize("where, want", [
    ({"truth_resolves": "yes"}, [True, False, False, None]),
    ({"truth_resolves": "no"}, [False, True, False, None]),
    ({"truth_genus_known": "no"}, [False, True, False, None]),
    ({"name_kind": "provisional"}, [False, True, False, False]),
])
def test_the_truth_where_keys_keep_the_finds_iNat_cannot_name_and_mark_no_taxonomy_unknown(
        where, want):
    from mycomap_vision import paper_tables as pt
    finds = [_f("1", truth_resolves="yes", truth_genus_known="yes"),
             _f("2", "provisional", truth_resolves="no", truth_genus_known="no"),
             _f("3", "one-word", truth_resolves="(no species)", truth_genus_known="yes"),
             _f("4")]                                            # no taxonomy in the run
    assert [pt.where_holds(f, where) for f in finds] == want


def test_a_truth_where_key_takes_yes_or_no(tmp_path):
    cfg = config(tmp_path, ["5", "6"])
    cfg["secondary_rows"] = {"for": ["H1a"], "rows": [
        {"id": "inat-cannot-name-species", "where": {"truth_resolves": "no"}},
        {"id": "inat-cannot-name-genus", "where": {"truth_genus_known": "no"}}]}
    paper_score.check_config(cfg)
    cfg["secondary_rows"]["rows"][0]["where"]["truth_resolves"] = False
    with pytest.raises(ValueError, match="truth_resolves is \"yes\" or \"no\""):
        paper_score.check_config(cfg)


# --- -a2 (Steve 28 Oct): v1-test restricted to the US, Canada and Puerto Rico ------------------

SCOPE = {"countries": ["US", "CA", "PR"], "unknown": "out", "rule": "Steve 28 Oct",
         "outside_comparators": ["vision-photos"]}


def _scoped(f_key, country):
    f = po.Find(f_key, "o" + f_key, "A x", "A", "", "formal", 5, 1)
    f.attrs["country"] = country
    return f


def test_the_test_scope_keeps_its_countries_and_puts_other_and_unknown_countries_outside():
    finds = [_scoped("1", "US"), _scoped("2", "MX"), _scoped("3", "unknown"),
             _scoped("4", "CA"), _scoped("5", "PR")]
    inside, outside = paper_score.test_scope_split(finds, SCOPE)
    assert [f.key for f in inside] == ["1", "4", "5"]
    assert [f.key for f in outside] == ["2", "3"]
    assert paper_score.test_scope_split(finds, None) == (finds, [])
    assert [paper_score.country_code(c) for c in (" us", "CA", "", None)] == [
        "US", "CA", "unknown", "unknown"]


def test_a_find_s_country_is_the_release_record_s_and_an_attrs_file_does_not_replace_it():
    rows = [{"record_key": "1", "country": "us", "observer_login": "a", "species": "A x",
             "genus": "A", "name_kind": "formal"},
            {"record_key": "2", "country": None, "observer_login": "b", "species": "A x",
             "genus": "A", "name_kind": "formal"}]
    finds = paper_score.build_finds(rows, {}, [], None, {"1": {"country": "FR"}})
    assert [f.attrs["country"] for f in finds] == ["US", "unknown"]


def _with_country(monkeypatch, of_key):
    real = paper_score.build_finds

    def build(*a, **kw):
        finds = real(*a, **kw)
        for f in finds:
            if f.key in of_key:
                f.attrs["country"] = of_key[f.key]
        return finds
    monkeypatch.setattr(paper_score, "build_finds", build)


def test_the_test_scope_takes_finds_outside_it_out_of_every_family_and_describes_them(
        world, tmp_path, monkeypatch):
    cfg = floor_family(config(tmp_path, ["5", "6"]))
    cfg["secondary_rows"] = {"for": ["A1s", "H1a"], "rows": [{"id": "all-finds", "where": {}}]}
    whole = run(world, tmp_path, "val", cfg)
    assert whole["hypotheses"]["A"]["A1s"]["n"] == 2 and "test_scope" not in whole
    _with_country(monkeypatch, {"6": "MX"})
    cfg["test_scope"] = SCOPE
    paper_score.check_config(cfg)
    rep = run(world, tmp_path, "val", cfg)
    assert rep["finds"]["scored"] == 1
    assert rep["hypotheses"]["A"]["A1s"]["n"] == 1 and rep["hypotheses"]["C"]["H1a"]["n"] == 1
    assert rep["secondary_rows"]["all-finds"]["finds"] == 1
    std = rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]
    assert std["table"]["species strict"]["top1"]["n"] == 1
    ts = rep["test_scope"]
    assert ts["in_scope_finds"] == 1 and ts["scope"]["countries"] == ["US", "CA", "PR"]
    o = ts["outside_test_scope"]
    assert (o["finds"], o["by_country"], o["unknown"]) == (1, {"MX": 1}, 0)
    out_rate = o["comparators"]["vision-photos"]["species_strict@1"]
    assert (out_rate["n"], out_rate["rate"]) == (1, 1.0)                   # dark B y, right
    assert "p" not in out_rate and "p_test" not in out_rate                # never tested
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Outside the test scope (descriptive, never tested)" in md and "| MX | 1 |" in md


def test_a_find_with_no_country_is_outside_and_every_find_outside_leaves_n_0(
        world, tmp_path, monkeypatch):
    _with_country(monkeypatch, {"5": "unknown", "6": "GB"})
    cfg = floor_family(config(tmp_path, ["5", "6"]))
    cfg["test_scope"] = SCOPE
    rep = run(world, tmp_path, "val", cfg)
    # no find left in the test: the comparators have no outcomes, so every hypothesis is
    # reported not tested (p 1 in Holm), as for a missing input
    assert rep["finds"]["scored"] == 0
    for fam in ("A", "C"):
        assert all("not_tested" in h and h["p_holm"] == 1.0
                   for h in rep["hypotheses"][fam].values())
    o = rep["test_scope"]["outside_test_scope"]
    assert (o["finds"], o["by_country"], o["unknown"]) == (2, {"GB": 1}, 1)
    assert o["comparators"]["vision-photos"]["genus_strict@1"]["n"] == 2


def test_a_scope_holding_every_find_scores_exactly_as_no_scope(world, tmp_path):
    cfg = floor_family(config(tmp_path, ["5", "6"]))
    cfg["secondary_rows"] = {"for": ["A1s", "H1a"], "rows": [{"id": "all-finds", "where": {}}]}
    plain = run(world, tmp_path, "val", cfg)
    cfg["test_scope"] = SCOPE                     # the world's finds are all US
    scoped = run(world, tmp_path, "val", cfg)
    for k in ("finds", "standard", "hypotheses", "secondary_rows", "slices", "calibration",
              "errors", "generalisation", "inputs"):
        assert json.dumps(scoped[k], sort_keys=True, default=str) == \
            json.dumps(plain[k], sort_keys=True, default=str), k
    o = scoped["test_scope"]["outside_test_scope"]
    assert o["finds"] == 0 and o["comparators"]["vision-photos"]["species_strict@1"]["n"] == 0


@pytest.mark.parametrize("change, refused", [
    (lambda t: t.update(countries=[]), "non-empty list"),
    (lambda t: t.update(countries="US"), "non-empty list"),
    (lambda t: t.update(countries=["us"]), "upper-case"),
    (lambda t: t.update(countries=["USA"]), "two-letter"),
    (lambda t: t.update(countries=["US", "US"]), "distinct"),
    (lambda t: t.update(unknown="in"), "unknown is one of"),
    (lambda t: t.pop("unknown"), "unknown is one of"),
    (lambda t: t.update(outside_comparators=["nope"]), "outside_comparators"),
])
def test_a_bad_test_scope_is_refused(tmp_path, change, refused):
    cfg = config(tmp_path, ["5", "6"])
    cfg["test_scope"] = dict(SCOPE)
    paper_score.check_config(cfg)
    change(cfg["test_scope"])
    with pytest.raises(ValueError, match=refused):
        paper_score.check_config(cfg)


# --- -a3 (Steve 30 Oct): Vision-assisted finds leave the test; Vision holds stay in -------------

def _snapshot(tmp_path, rows, name="truth-snapshot.tsv"):
    p = tmp_path / name
    p.write_text("\t".join(paper_score.SNAPSHOT_COLUMNS) + "\n"
                 + "".join("\t".join(r) + "\n" for r in rows), encoding="utf-8", newline="\n")
    return p, paper_score.sha256_file(p)


def _truth(path="TBD: at the cut", sha="TBD: at the cut", **kw):
    return {"exclude": "vision_assisted", "rule": "Steve 30 Oct",
            "snapshot": {"file": str(path), "sha256": sha}, **kw}


def _find(key, genus="A"):
    return po.Find(key, "o" + key, f"{genus} x", genus, "", "formal", 5, 1)


def test_a_find_is_vision_assisted_by_its_release_field_or_the_snapshot_and_holds_stay():
    finds = [_find("1"), _find("2"), _find("3"), _find("4"), _find("5")]
    rows = {"1": {"vision_assisted": 1}, "2": {"vision_assisted": 0},
            "3": {"vision_assisted": 0, "held_by_vision": 1}, "4": {}, "5": {}}
    snap = {"assisted": {"1", "4"}, "holds": {"2": "Russula"}}
    kept, out, info = paper_score.truth_split(finds, rows, snap)
    assert [f.key for f in kept] == ["2", "3", "5"] and [f.key for f in out] == ["1", "4"]
    assert info["by_source"] == {"release": 0, "snapshot": 1, "both": 1}
    assert info["holds"] == {"2": "Russula", "3": None}
    kept, out, info = paper_score.truth_split(finds, rows, None)       # the field alone
    assert [f.key for f in out] == ["1"] and info["by_source"]["release"] == 1
    assert info["release_carries_fields"]
    assert not paper_score.truth_split(finds, {}, None)[2]["release_carries_fields"]


def test_the_snapshot_is_read_by_record_key_and_refused_when_not_the_pinned_file(tmp_path):
    p, sha = _snapshot(tmp_path, [("dna_vision", "inat", "4", "", "2026-10-29T03:18:10Z"),
                                  ("hold", "mo", "7", "Amanita", "2026-10-25T01:00:00Z"),
                                  ("hold", "inat", "8", "", "2026-10-25T01:00:00Z")])
    got = paper_score.read_truth_snapshot(p, sha)
    assert got["assisted"] == {"4"} and got["holds"] == {"mo:7": "Amanita", "8": None}
    assert got["rows"] == 3 and got["sha256"] == sha
    with pytest.raises(paper_score.FrozenError, match="not the pinned"):
        paper_score.read_truth_snapshot(p, "0" * 64)
    bad, bsha = _snapshot(tmp_path, [("waiver", "inat", "4", "", "x")], "bad.tsv")
    with pytest.raises(ValueError, match="kind"):
        paper_score.read_truth_snapshot(bad, bsha)
    hdr = tmp_path / "hdr.tsv"
    hdr.write_text("platform\tobservation_id\n", encoding="utf-8", newline="\n")
    with pytest.raises(ValueError, match="header"):
        paper_score.read_truth_snapshot(hdr, paper_score.sha256_file(hdr))


def test_an_unfilled_snapshot_is_refused_for_the_sealed_test_only():
    assert paper_score.truth_snapshot(_truth(), sealed=False, log=lambda *a: None) is None
    with pytest.raises(paper_score.FrozenError, match="not filled"):
        paper_score.truth_snapshot(_truth(), sealed=True)


def test_vision_assisted_finds_leave_every_family_and_are_described_and_holds_are_counted(
        world, tmp_path):
    cfg = floor_family(config(tmp_path, ["5", "6"]))
    cfg["secondary_rows"] = {"for": ["A1s", "H1a"], "rows": [{"id": "all-finds", "where": {}}]}
    p, sha = _snapshot(tmp_path, [("dna_vision", "inat", "6", "", "2026-10-29T03:18:10Z"),
                                  ("hold", "inat", "5", "A", "2026-10-25T01:00:00Z")])
    cfg["truth_independence"] = _truth(p, sha, describe_comparators=["vision-photos"])
    paper_score.check_config(cfg)
    rep = run(world, tmp_path, "val", cfg)
    assert rep["finds"]["scored"] == 1
    assert rep["hypotheses"]["A"]["A1s"]["n"] == 1 and rep["hypotheses"]["C"]["H1a"]["n"] == 1
    assert rep["secondary_rows"]["all-finds"]["finds"] == 1
    std = rep["standard"]["comparators"]["vision-photos"]["primary"]["all"]
    assert std["table"]["species strict"]["top1"]["n"] == 1
    ti = rep["truth_independence"]
    assert ti["snapshot"] == {"file": "truth-snapshot.tsv", "sha256": sha, "rows": 2}
    e = ti["excluded"]
    assert (e["finds"], e["by_source"]) == (1, {"release": 0, "snapshot": 1, "both": 0})
    r = e["comparators"]["vision-photos"]["species_strict@1"]
    assert (r["n"], r["rate"]) == (1, 1.0) and "p" not in r                # never tested
    h = ti["holds_kept"]
    assert (h["holds"], h["took_vision_genus"], h["kept_dna_genus"]) == (1, 1, 0)
    md = (tmp_path / "out" / "report.md").read_text(encoding="utf-8")
    assert "## Vision-assisted (excluded; descriptive, never tested)" in md
    assert "## Vision holds (kept)" in md


def test_with_no_truth_block_or_nobody_assisted_the_scores_are_unchanged(world, tmp_path):
    cfg = floor_family(config(tmp_path, ["5", "6"]))
    plain = run(world, tmp_path, "val", cfg)
    assert "truth_independence" not in plain
    cfg["truth_independence"] = _truth()                 # no snapshot yet, no field set
    same = run(world, tmp_path, "val", cfg)
    for k in ("finds", "standard", "hypotheses", "slices", "calibration", "errors", "inputs"):
        assert json.dumps(same[k], sort_keys=True, default=str) == \
            json.dumps(plain[k], sort_keys=True, default=str), k
    assert same["truth_independence"]["excluded"]["finds"] == 0
    assert same["truth_independence"]["snapshot"] is None


@pytest.mark.parametrize("change, refused", [
    (lambda t: t.update(exclude="held_by_vision"), "exclude is one of"),
    (lambda t: t.pop("snapshot"), "snapshot is"),
    (lambda t: t.update(snapshot={"file": "x.tsv"}), "snapshot is"),
    (lambda t: t.update(snapshot={"file": "x.tsv", "sha256": "abc"}), "64 hex"),
    (lambda t: t.update(describe_comparators=["nope"]), "describe_comparators"),
])
def test_a_bad_truth_independence_block_is_refused(tmp_path, change, refused):
    cfg = config(tmp_path, ["5", "6"])
    cfg["truth_independence"] = _truth(describe_comparators=["vision-photos"])
    paper_score.check_config(cfg)
    change(cfg["truth_independence"])
    with pytest.raises(ValueError, match=refused):
        paper_score.check_config(cfg)
