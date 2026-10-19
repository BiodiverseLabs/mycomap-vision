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


def test_config_paths_resolve_under_the_data_folder_and_the_research_repo():
    from mycomap_vision import config as cfg
    assert paper_score.resolve("data/x/y.tsv") == cfg.DATA_DIR / "x" / "y.tsv"
    assert paper_score.resolve("paper/e21.csv") == cfg.RESEARCH_DIR / "paper" / "e21.csv"
    assert paper_score.resolve("C:/abs/z.tsv").is_absolute()


def test_the_cc_reference_row_uses_only_photos_a_public_model_may_learn_from(world, tmp_path):
    cfg = config(tmp_path, ["5", "6"])
    cfg["comparators"].append({"id": "cc-reference", "kind": "vision", "backbone": "m1",
                               "method": "nearest", "reference": "public_trainable_only"})
    rep = run(world, tmp_path, "val", cfg)
    full = rep["inputs"]["vision-photos"]["reference_records"]
    cc = rep["inputs"]["cc-reference"]["reference_records"]
    want = sum(1 for r in world["base"].as_records("train", None, public_trainable_only=True))
    assert cc == want and cc < full                 # the fixture's ARR photos leave


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
                           "method": "nearest+mean+place", "place_setting": name}]
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
                                         "method": "nearest+mean+place",
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
         "method": "nearest+mean+place", "place_setting": "penalty-500-s30-2026-10-19"}], **extra}


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
