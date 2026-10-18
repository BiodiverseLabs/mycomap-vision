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
        paper_score.check_config({"format": 1, "comparators": [
            {"id": "v", "kind": "vision", "backbone": "m1", "method": "nearest+maen"}]})
    paper_score.check_config({"format": 1, "comparators": [
        {"id": "v", "kind": "vision", "backbone": "m1", "method": "nearest+mean"}]})


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
    cfg = {"format": 1, "comparators": [{"id": "p", "kind": "vision", "backbone": "m",
                                         "method": "nearest+mean+place",
                                         "place_setting": "penalty-500-2026-10-13"}]}
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
    paper_score.check_config({"format": 1, "comparators": [ok]})
    with pytest.raises(ValueError, match="expert tier"):
        paper_score.check_config({"format": 1, "comparators": [{**ok, "tier": "1+3"}]})
