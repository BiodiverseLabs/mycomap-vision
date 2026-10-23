"""Photo-type tags on Identify (phototypes.py, Steve 2026-10-22): which tags leave a photo
out of the answer, the all-left-out fallback, "Include anyway", and the tagger file's
checks (sha256, pinned model)."""

import hashlib
import json

import numpy as np
import pytest

from mycomap_vision import phototypes
from mycomap_vision.phototypes import HIGH, LOW, MEDIUM, PhotoTagger, Refused

CLASSES = ("specimen", "micrograph", "slip", "habitat", "uv", "chemical", "spore_print",
           "dried", "drawing", "tree", "people", "not_fungus")
# The shape of the seeded bands (t13): high at or above the first value, then medium.
BANDS = {"micrograph": [[0.35, HIGH]], "slip": [[0.35, HIGH]],
         "habitat": [[0.5, HIGH], [0.35, MEDIUM]], "uv": [[0.5, MEDIUM], [0.35, LOW]],
         "people": [[0.7, HIGH], [0.35, LOW]], "not_fungus": [[0.35, LOW]]}


def tag(kind, p, band):
    return {"type": kind, "p": p, "band": band, "label": None}


def tagger(W=None, b=None, bands=BANDS):
    """A head on 12-d vectors whose top class is the vector's largest coordinate."""
    return PhotoTagger(np.eye(12, dtype=np.float32) * 20 if W is None else W,
                       np.zeros(12, np.float32) if b is None else b, CLASSES, bands, 0.35,
                       "bioclip-2", "v2")


# --- bands --------------------------------------------------------------------------

def test_a_tag_takes_the_band_of_the_first_range_its_confidence_reaches():
    assert phototypes.band_of(BANDS, "habitat", 0.83, 0.35) == HIGH
    assert phototypes.band_of(BANDS, "habitat", 0.5, 0.35) == HIGH          # inclusive
    assert phototypes.band_of(BANDS, "habitat", 0.49, 0.35) == MEDIUM
    assert phototypes.band_of(BANDS, "micrograph", 0.36, 0.35) == HIGH


def test_under_the_floor_or_any_measured_range_a_tag_is_low_and_a_specimen_has_no_band():
    assert phototypes.band_of(BANDS, "micrograph", 0.34, 0.35) == LOW      # under the floor
    assert phototypes.band_of(BANDS, "drawing", 0.9, 0.35) == LOW          # never measured
    assert phototypes.band_of(BANDS, "specimen", 0.99, 0.35) is None


def test_the_tagger_names_the_top_class_and_shows_a_chip_only_for_high_and_medium_bands():
    t = tagger()
    v = np.zeros((4, 12), np.float32)
    v[0, CLASSES.index("habitat")] = 1                       # habitat, p ~ 1: high
    v[1, CLASSES.index("specimen")] = 1                      # an ordinary photo
    v[2, CLASSES.index("not_fungus")] = 1                    # banded low only
    v[3, CLASSES.index("uv")] = 1
    out = t.tag(v)
    assert [x["type"] for x in out] == ["habitat", "specimen", "not_fungus", "uv"]
    assert [x["band"] for x in out] == [HIGH, None, LOW, MEDIUM]
    assert [x["label"] for x in out] == ["Habitat", None, None, "UV"]
    assert phototypes.LABELS["tree"] == "Phylo tree"
    assert 0.99 < out[0]["p"] <= 1


def test_the_tagger_reads_vectors_at_any_scale():
    t = tagger(W=np.random.default_rng(1).normal(size=(12, 12)).astype(np.float32))
    v = np.random.default_rng(2).normal(size=(3, 12)).astype(np.float32)
    assert np.allclose(t.probs(v), t.probs(v * 7.5), atol=1e-6)


# --- the leave-out decision ---------------------------------------------------------

def test_only_micrograph_and_slip_tags_in_the_high_band_leave_a_photo_out():
    tags = [tag("slip", 0.9, HIGH), tag("micrograph", 0.6, HIGH), tag("slip", 0.4, HIGH),
            tag("micrograph", 0.45, MEDIUM), tag("uv", 0.99, HIGH), tag("people", 0.9, HIGH),
            tag("specimen", 0.9, None), None, tag("habitat", 0.95, HIGH)]
    d = phototypes.decide(tags, list(range(len(tags))))
    assert d == {"left_out": [0, 1, 2], "included": [], "kept_all": False}
    assert phototypes.LEAVE_OUT == ("micrograph", "slip")


def test_a_high_band_habitat_photo_is_tagged_with_its_chip_and_still_counted():
    """Steve, 2026-10-23: habitat is tag-only, whatever its band."""
    t = tagger()
    v = np.zeros((1, 12), np.float32)
    v[0, CLASSES.index("habitat")] = 1
    [h] = t.tag(v)
    assert (h["type"], h["band"], h["label"]) == ("habitat", HIGH, "Habitat")
    assert phototypes.would_leave_out(h) is False
    tags = [tag("specimen", 0.9, None), h]
    d = phototypes.decide(tags, [0, 1])
    assert d == {"left_out": [], "included": [], "kept_all": False}
    assert phototypes.annotate(tags, d)[1]["left_out"] is False
    assert phototypes.hints(tags, d) == []
    # Only habitat photos: none would be left out, so the answer is not marked less reliable.
    assert phototypes.decide([h, h], [0, 1])["kept_all"] is False


def test_include_anyway_counts_a_photo_its_tag_would_leave_out():
    tags = [tag("specimen", 0.9, None), tag("micrograph", 0.9, HIGH), tag("slip", 0.9, HIGH)]
    d = phototypes.decide(tags, [0, 1, 2], include=[1])
    assert d == {"left_out": [2], "included": [1], "kept_all": False}
    # Asking to include a photo nothing would leave out changes nothing.
    assert phototypes.decide(tags, [0, 1, 2], include=[0])["included"] == []


def test_when_every_counted_photo_would_be_left_out_none_is():
    tags = [tag("slip", 0.9, HIGH), tag("micrograph", 0.9, HIGH)]
    assert phototypes.decide(tags, [0, 1]) == {"left_out": [], "included": [], "kept_all": True}
    # One counted anyway: the other is left out as usual, the answer is from the one.
    assert phototypes.decide(tags, [0, 1], include=[0]) == {
        "left_out": [1], "included": [0], "kept_all": False}


def test_only_the_photos_the_gate_kept_are_weighed():
    # Photo 0 is gated (not a fungus); photo 1, the only one counted, is a label photo:
    # leaving it out would leave nothing, so it stays.
    tags = [tag("specimen", 0.9, None), tag("slip", 0.9, HIGH), tag("specimen", 0.9, None)]
    assert phototypes.decide(tags, [1])["kept_all"] is True
    assert phototypes.decide(tags, [1, 2])["left_out"] == [1]


def test_the_hints_name_the_left_out_photos_and_say_when_all_were_used():
    tags = [tag("slip", 0.9, HIGH), tag("specimen", 0.9, None), tag("micrograph", 0.9, HIGH),
            tag("micrograph", 0.9, HIGH)]
    one = phototypes.hints(tags, phototypes.decide(tags[:2], [0, 1]))
    assert one == ["Photo 1 looks like a label or field slip, so it was left out of the answer."]
    many = phototypes.hints(tags, phototypes.decide(tags, [0, 1, 2, 3]))
    assert many == ["Photos 1, 3 and 4 look like micrographs or labels, so "
                    "they were left out of the answer."]
    same = phototypes.hints(tags, phototypes.decide(tags, [1, 2, 3]))
    assert same == ["Photos 3 and 4 look like micrographs, so they were left out of the answer."]
    every = phototypes.hints(tags, phototypes.decide(tags, [0, 2]))
    assert every == ["Only non-specimen photos: this answer is less reliable."]
    assert phototypes.hints(tags, phototypes.decide(tags, [1])) == []


def test_include_is_read_as_photo_indices_and_anything_else_is_refused():
    assert phototypes.parse_include("") == [] and phototypes.parse_include(None) == []
    assert phototypes.parse_include("2, 0,2") == [0, 2]
    assert phototypes.parse_include([3, 1]) == [1, 3]
    for bad in ("x", [True], [-1], {"a": 1}, [1.5]):
        with pytest.raises(ValueError):
            phototypes.parse_include(bad)
    with pytest.raises(ValueError):
        phototypes.parse_include("3", photos=3)                  # photos 0..2 only
    assert phototypes.parse_include("9") == [9]                   # count not known yet
    with pytest.raises(ValueError):
        phototypes.parse_include("10")


# --- the file -----------------------------------------------------------------------

def research_output(tmp_path, W=None):
    """A research model.npz and the seed summary that records its sha256 and bands."""
    model = tmp_path / "model.npz"
    np.savez(model, W=np.eye(12, dtype=np.float32) * 20 if W is None else W,
             b=np.zeros(12, np.float32), classes=np.array(CLASSES))
    sha = hashlib.sha256(model.read_bytes()).hexdigest()
    seed = tmp_path / "t13_v2_seed.json"
    seed.write_text(json.dumps({"model_sha256": sha, "floor": 0.35, "bands": BANDS}))
    return model, seed, sha


def test_build_refuses_a_research_model_that_is_not_pinned(tmp_path):
    model, seed, _ = research_output(tmp_path)
    with pytest.raises(Refused, match="not a pinned"):
        phototypes.build(model, seed)


def test_built_saved_and_loaded_the_tagger_tags_as_before(tmp_path, monkeypatch):
    model, seed, sha = research_output(tmp_path)
    monkeypatch.setitem(phototypes.PINNED, sha, {"version": "test", "base": "bioclip-2"})
    built = phototypes.build(model, seed)
    path = built.save(tmp_path / "out" / "photo-tagger.npz")
    loaded = PhotoTagger.load(path)
    assert loaded.fingerprint() == built.fingerprint() and loaded.version == "test"
    assert loaded.bands == BANDS and loaded.provenance["source_sha256"] == sha
    v = np.random.default_rng(3).normal(size=(5, 12)).astype(np.float32)
    assert loaded.tag(v) == built.tag(v)


def test_a_seed_file_made_for_another_model_is_refused(tmp_path, monkeypatch):
    model, seed, sha = research_output(tmp_path)
    monkeypatch.setitem(phototypes.PINNED, sha, {"version": "test", "base": "bioclip-2"})
    seed.write_text(json.dumps({"model_sha256": "0" * 64, "floor": 0.35, "bands": BANDS}))
    with pytest.raises(Refused, match="bands were made for model"):
        phototypes.build(model, seed)


def test_a_tagger_file_changed_after_it_was_built_is_refused(tmp_path, monkeypatch):
    model, seed, sha = research_output(tmp_path)
    monkeypatch.setitem(phototypes.PINNED, sha, {"version": "test", "base": "bioclip-2"})
    path = phototypes.build(model, seed).save(tmp_path / "photo-tagger.npz")
    np.savez(path, W=np.eye(12, dtype=np.float32), b=np.ones(12, np.float32),
             classes=np.array(CLASSES))
    with pytest.raises(Refused, match="sha256"):
        PhotoTagger.load(path)


def test_a_tagger_file_from_an_unpinned_model_is_refused_at_load(tmp_path, monkeypatch):
    model, seed, sha = research_output(tmp_path)
    monkeypatch.setitem(phototypes.PINNED, sha, {"version": "test", "base": "bioclip-2"})
    path = phototypes.build(model, seed).save(tmp_path / "photo-tagger.npz")
    monkeypatch.delitem(phototypes.PINNED, sha)
    with pytest.raises(Refused, match="not a pinned"):
        PhotoTagger.load(path)


def test_without_the_setting_there_is_no_tagger(monkeypatch):
    monkeypatch.delenv(phototypes.SETTING, raising=False)
    assert PhotoTagger.from_setting() is None


V2 = "526ca22f62e66643193f3d97ba9ab0684d9ab602d8b841895ea1b41d310230c1"
V21_336 = "128f4f874dad41253e57433ddd196a54e2f8929600df82bf706986db86ada1e6"


def test_v2_and_v2_1_336_are_the_only_pinned_models_each_with_its_base():
    assert {sha: (pin["version"], pin["base"]) for sha, pin in phototypes.PINNED.items()} == {
        V2: ("v2", "bioclip-2"), V21_336: ("v2.1-336", "bioclip-2-336")}


def pinned_file(tmp_path, source_sha, base):
    """A tagger file as `mv photo-tagger build` saves it from the research model with that
    sha256 (the weights are a stand-in: the pin is what the checks read)."""
    t = PhotoTagger(np.eye(12, dtype=np.float32) * 20, np.zeros(12, np.float32), CLASSES,
                    BANDS, 0.35, base, None, {"source_sha256": source_sha})
    return t.save(tmp_path / "photo-tagger.npz")


def test_the_336_head_loads_as_v2_1_336_reading_bioclip_2_336_vectors(tmp_path):
    loaded = PhotoTagger.load(pinned_file(tmp_path, V21_336, "bioclip-2-336"))
    assert (loaded.version, loaded.base) == ("v2.1-336", "bioclip-2-336")
    assert loaded.summary()["leave_out"] == ["micrograph", "slip"]


def test_the_336_head_is_refused_when_its_file_says_it_reads_bioclip_2_vectors(tmp_path):
    with pytest.raises(Refused, match="tagger v2.1-336 was trained on bioclip-2-336 vectors"):
        PhotoTagger.load(pinned_file(tmp_path, V21_336, "bioclip-2"))


def test_v2_still_loads_on_bioclip_2_and_is_refused_as_a_336_head(tmp_path):
    assert PhotoTagger.load(pinned_file(tmp_path, V2, "bioclip-2")).version == "v2"
    with pytest.raises(Refused, match="tagger v2 was trained on bioclip-2 vectors"):
        PhotoTagger.load(pinned_file(tmp_path, V2, "bioclip-2-336"))


def test_a_build_records_where_its_pinned_model_and_bands_came_from(tmp_path, monkeypatch):
    model, seed, sha = research_output(tmp_path)
    monkeypatch.setitem(phototypes.PINNED, sha, {**phototypes.PINNED[V21_336]})
    built = phototypes.build(model, seed)
    assert (built.version, built.base) == ("v2.1-336", "bioclip-2-336")
    assert "t20" in built.provenance["source"]
    assert built.provenance["bands_from"].startswith("t21_336_bands.json")


def test_a_head_trained_on_336_px_vectors_is_pinned_with_its_base_and_reads_that_base(
        tmp_path, monkeypatch):
    model, seed, sha = research_output(tmp_path)
    monkeypatch.setitem(phototypes.PINNED, sha, {"version": "v2-336", "base": "bioclip-2-336"})
    path = phototypes.build(model, seed).save(tmp_path / "photo-tagger.npz")
    loaded = PhotoTagger.load(path)
    assert (loaded.base, loaded.version) == ("bioclip-2-336", "v2-336")


def test_a_file_claiming_another_base_than_its_pinned_head_was_trained_on_is_refused(
        tmp_path, monkeypatch):
    model, seed, sha = research_output(tmp_path)
    monkeypatch.setitem(phototypes.PINNED, sha, {"version": "v2", "base": "bioclip-2"})
    path = phototypes.build(model, seed).save(tmp_path / "photo-tagger.npz")
    meta = json.loads(phototypes.sidecar(path).read_text())
    phototypes.sidecar(path).write_text(json.dumps({**meta, "base": "bioclip-2-336"}))
    with pytest.raises(Refused, match="trained on bioclip-2 vectors"):
        PhotoTagger.load(path)


def test_the_refusal_beside_another_gate_names_the_base_the_head_needs():
    why = phototypes.base_mismatch("bioclip-2", "bioclip-2-336")
    assert "reads bioclip-2 frozen vectors" in why
    assert "serve a head trained on bioclip-2-336 frozen vectors" in why
