"""The Picek replication (picek.py) without torch: what it trains on, the metadata prior,
the classifier as a Vision method, and the macro-F1 / per-image numbers it reports."""

import json
import math

import numpy as np
import pytest
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import config, evaluate, heldout, heldout_report, metrics, models, picek
from mycomap_vision.embed import embed_photos, normalise, photos_to_embed
from mycomap_vision.methods import log_softmax, method_ready
from mycomap_vision.prior import Context

# seed_two_species: records 100-102 "A x" and 103-105 "B y" validated Jan 1-3 2026, and
# 106 / 107 on Sept 20 / 21 (the test weeks; cutoff 2026-08-24 with 28 test days).
# 234 days before the cutoff is Jan 2, so a 234-day validation slice holds the Jan 3 pair.
VAL_DAYS = (np.datetime64("2026-08-24") - np.datetime64("2026-01-02")).astype(int)


def photo_ids(items):
    return {int(i[0]) for i in items}


# --- what it trains on ------------------------------------------------------------------

def test_training_is_the_comparisons_reference_minus_a_validation_slice_never_the_test_weeks(
        conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    data = picek.build_data(conn, store.location, "large", test_days=28, val_days=int(VAL_DAYS))
    assert data.trained_through == "2026-08-24"
    assert data.train_through == "2026-01-02"
    assert photo_ids(data.train) == {1000, 1001, 1003, 1004}
    assert photo_ids(data.val) == {1002, 1005}
    assert photo_ids(data.train + data.val).isdisjoint({1006, 1007})
    assert data.classes == ["A x", "B y"]
    assert data.class_photos.tolist() == [2, 2]
    assert {i[2] for i in data.val} == {0, 1}


def benchmark_record(conn, oid="100"):
    heldout.ensure_schema(conn)
    with conn:
        conn.execute("insert into heldout_records (benchmark, observation_id, truth_status, "
                     "name_source, split) values ('b', ?, 'x', 'y', 'dev')", (oid,))


def test_by_default_it_trains_on_exactly_what_a_vision_model_trains_on(conn, tmp_path):
    # A released (development) benchmark's records train Vision too: "match" keeps them.
    store = seed_two_species(conn, tmp_path)
    benchmark_record(conn)
    ids = picek.exclusion_ids(conn, picek.PicekConfig().exclude_benchmarks)
    data = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS), exclude_ids=ids)
    assert 1000 in photo_ids(data.train) and data.excluded_benchmark_records == 0


def test_all_leaves_out_every_benchmark_record_and_a_trainer_needs_the_ids_shipped(conn,
                                                                                  tmp_path):
    store = seed_two_species(conn, tmp_path)
    benchmark_record(conn)
    data = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS),
                            exclude_ids=picek.exclusion_ids(conn, "all"))
    assert 1000 not in photo_ids(data.train + data.val)
    assert data.excluded_benchmark_records == 1
    ids = tmp_path / "ids.txt"
    ids.write_text("100\n101\n", encoding="utf-8")
    assert picek.exclusion_ids(conn, "all", str(ids)) == {"100", "101"}
    with conn:
        conn.execute("drop table heldout_records")         # as in a shipped manifest
    with pytest.raises(ValueError, match="no benchmark tables"):
        picek.exclusion_ids(conn, "all")
    with pytest.raises(ValueError, match="never trained on"):
        picek.exclusion_ids(conn, "none")


def test_sealed_benchmark_records_are_never_trained_on_in_any_mode(conn, tmp_path):
    from mycomap_vision import holdouts
    store = seed_two_species(conn, tmp_path)
    holdouts.ensure_schema(conn)
    with conn:
        conn.execute("insert into benchmark_holdouts values ('100', 'paper', 't')")
    for mode in ("match", "all"):
        data = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS),
                                exclude_ids=picek.exclusion_ids(conn, mode)
                                if mode == "match" else set())
        assert 1000 not in photo_ids(data.train + data.val), mode


def test_the_label_snapshot_proves_which_labelling_a_run_used(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    data = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS))
    snap = picek.label_snapshot(conn, data)
    assert (snap["records_train"], snap["records_validation"], snap["species"]) == (4, 2, 2)
    assert snap["trained_through"] == "2026-08-24" and snap["exclude_benchmarks"] == "match"
    assert snap["known_label_problems"]["label_conflict_records"] == 0
    with conn:
        conn.execute("update records set scientific_name = 'B y' where observation_id = '100'")
    other = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS))
    assert picek.labels_hash(other) != snap["labels_hash"]       # a relabel shows
    assert picek.labels_hash(picek.build_data(conn, store.location, "large", 28,
                                              int(VAL_DAYS))) == picek.labels_hash(other)
    assert "hash" in picek.format_snapshot(snap)


def test_a_one_word_name_trains_no_species_class(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    with conn:
        conn.execute("update records set scientific_name = 'A' where observation_id = '101'")
    data = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS))
    assert 1001 not in photo_ids(data.train)
    assert data.one_word_records == 1
    assert "A" not in data.classes


def test_month_and_place_counts_come_from_training_photos_only(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    data = picek.build_data(conn, store.location, "large", 28, int(VAL_DAYS))
    # Every seeded observation is from 2025-09-01 (iNat's date): month index 8.
    assert data.month_counts[:, 8].tolist() == [2, 2]
    assert data.month_counts.sum() == 4


# --- the embedding carries the classifier's input exactly --------------------------------

def test_features_survive_the_normalised_float16_embedding():
    rng = np.random.default_rng(0)
    f = rng.normal(size=(50, 768)).astype(np.float32) * rng.uniform(5, 30, size=(50, 1))
    scale = 20 * float(np.median(np.linalg.norm(f, axis=1)))
    stored = normalise(picek.encode_features(f, scale))         # what embed_photos keeps
    back = picek.decode_features(stored, scale)
    w = rng.normal(size=(100, 768)).astype(np.float32) / 30
    assert np.allclose(back @ w.T, f @ w.T, atol=0.05 * np.abs(f @ w.T).max())
    rel = np.linalg.norm(back - f, axis=1) / np.linalg.norm(f, axis=1)
    assert rel.max() < 3e-3
    # Cosine between stored vectors stays the features' cosine (nearest still works).
    fn = f / np.linalg.norm(f, axis=1, keepdims=True)
    s = stored.astype(np.float32)
    assert np.abs(s @ s.T - fn @ fn.T).max() < 0.01


# --- the classifier as a method ---------------------------------------------------------

def write_head(root, name="clf", classes=("A x", "B y", "C z"), genus=("A", "B", "C"),
               weight=None, temperature=2.0, month_counts=None, cells=None, scale=100.0):
    root.mkdir(parents=True, exist_ok=True)
    k = len(classes)
    data = picek.PicekData(list(classes), list(genus), [], [], np.ones(k, np.int64),
                           month_counts if month_counts is not None else np.zeros((k, 12), np.int32),
                           cells or {}, "2026-08-24", "2026-07-27")
    w = weight if weight is not None else np.eye(k, 4, dtype=np.float32) * 5
    picek.save_head(root / f"{name}.classifier.npz", data, w, np.zeros(k, np.float32),
                    temperature, scale, 4.0)
    (root / f"{name}.json").write_text(json.dumps({"name": name, "kind": picek.KIND}),
                                       encoding="utf-8")
    return scale


def index_with(species, one_word=()):
    recs = [evaluate.Record(str(i), s, s.split()[0], "F", "2026-01-01", "o", [i])
            for i, s in enumerate(species)]
    recs += [evaluate.Record(f"w{i}", "", w, "F", "2026-01-01", "o", [100 + i], taxon=w)
             for i, w in enumerate(one_word)]
    return evaluate.build_index(recs)


def method(name, root, backbone="clf"):
    m = evaluate.METHODS[name]()
    m.for_backbone(backbone, root)
    return m


def test_an_observation_is_the_mean_of_its_photos_logits_over_t_then_softmax(tmp_path):
    scale = write_head(tmp_path, temperature=2.0)
    f = np.array([[3.0, 0, 0, 0], [0, 1.0, 0, 0]], np.float32)
    query = normalise(picek.encode_features(f, scale))
    m = method("classifier", tmp_path)
    index = index_with(["A x", "B y", "C z"])
    m.fit(None, index)
    got = m.species_scores(query)
    logits = f @ (np.eye(3, 4) * 5).T
    want = log_softmax((logits / 2.0).mean(axis=0))
    assert np.allclose(got, want, atol=0.02)


def test_species_the_classifier_never_learned_get_no_probability_and_one_word_groups_none(
        tmp_path):
    scale = write_head(tmp_path)
    m = method("classifier", tmp_path)
    index = index_with(["A x", "B y", "D new"], one_word=["Russula"])
    m.fit(None, index)
    s = m.species_scores(normalise(picek.encode_features(np.ones((1, 4), np.float32), scale)))
    pos = {sp: i for i, sp in enumerate(index.species)}
    assert s[pos["D new"]] == pytest.approx(picek.LOG_FLOOR)
    assert s[pos["Russula"]] == pytest.approx(picek.LOG_FLOOR)
    assert np.isfinite(s).all()


def test_the_month_prior_multiplies_the_image_probability_once_not_twice(tmp_path):
    # Every class seen equally in every month: the prior is flat, so the answer must be
    # the image answer itself. Squaring p(c|x) (their code's quirk) would sharpen it.
    scale = write_head(tmp_path, month_counts=np.full((3, 12), 5, np.int32))
    q = normalise(picek.encode_features(np.array([[1.0, 0.5, 0, 0]], np.float32), scale))
    index = index_with(["A x", "B y", "C z"])
    plain, month = method("classifier", tmp_path), method("classifier+month", tmp_path)
    plain.fit(None, index)
    month.fit(None, index)
    assert np.allclose(month.species_scores(q, Context(observed_on="2026-09-15")),
                       plain.species_scores(q), atol=1e-6)


def head_with_months(tmp_path):
    counts = np.zeros((3, 12), np.int32)
    counts[0, 8] = 10          # A x: September only
    counts[1, 3] = 10          # B y: April only
    # C z: no dated photos
    write_head(tmp_path, month_counts=counts)
    return picek.load_head("clf", tmp_path)


def test_unsmoothed_month_prior_is_dfs_raw_estimate_and_all_but_removes_unseen_months(tmp_path):
    head = head_with_months(tmp_path)
    raw = picek.MetadataPrior(beta=0.0, beta_genus=0.0)
    r = raw.log_ratio(head, Context(observed_on="2026-09-10"))
    p_m = (10 + 1) / (20 + 12)                       # p(September), all fungi
    assert r[0] == pytest.approx(math.log(1.0 / p_m))    # p(Sep | A x) = 1
    assert r[1] < -20                                    # never in September: ~0
    assert r[2] == pytest.approx(0.0)                    # no dates: neutral


def test_smoothed_month_prior_keeps_an_unseen_month_possible(tmp_path):
    head = head_with_months(tmp_path)
    r = picek.MetadataPrior().log_ratio(head, Context(observed_on="2026-09-10"))
    assert r[0] > 0 > r[1] > -5
    assert picek.MetadataPrior().log_ratio(head, Context()).tolist() == [0, 0, 0]
    assert picek.MetadataPrior().log_ratio(head, None).tolist() == [0, 0, 0]


def test_the_place_prior_is_ours_and_only_in_its_own_method(tmp_path):
    cell = picek.cell_of(45.0, -122.0, 4.0)
    write_head(tmp_path, cells={cell: {0: 10}, picek.cell_of(35.0, -80.0, 4.0): {1: 10}})
    head = picek.load_head("clf", tmp_path)
    ctx = Context(45.5, -121.5, None)
    assert picek.MetadataPrior(month=False, place=True).log_ratio(head, ctx)[0] > 0
    assert picek.ClassifierMonth.prior.place is False
    assert picek.ClassifierMonthPlace.prior.place is True


def test_classifier_methods_need_their_own_model_and_are_not_offered_on_the_site(tmp_path):
    with pytest.raises(ValueError, match="needs the model"):
        evaluate.make_method("classifier")
    with pytest.raises(ValueError, match="not a classifier model"):
        evaluate.make_method("classifier", "bioclip-2")
    assert not method_ready("classifier") and not method_ready("classifier+month")
    assert method_ready("nearest")


def test_a_classifier_model_resolves_to_its_own_loader(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    write_head(tmp_path / "models", name="picek-x")
    assert models.resolve_spec("picek-x") == "classifier:picek-x"
    assert models.storage_name("classifier:picek-x") == "picek-x"


# --- temperature, macro-F1 ---------------------------------------------------------------

def test_the_fitted_temperature_undoes_overconfidence():
    rng = np.random.default_rng(1)
    true = rng.normal(size=(4000, 5)) * 2
    p = np.exp(log_softmax(true, axis=1))
    y = np.array([rng.choice(5, p=row) for row in p])
    assert picek.fit_temperature(true * 3, y) == pytest.approx(3, rel=0.1)
    assert picek.fit_temperature(true * 3, np.r_[y[:-1], -1]) == pytest.approx(3, rel=0.1)


def test_temperature_by_observation_averages_its_photos_first():
    z = np.array([[4.0, 0], [0, 1.0], [3.0, 0]])
    groups = np.array(["a", "a", "b"])
    t = picek.fit_temperature(z, np.array([0, 0, 0]), groups)
    assert t == pytest.approx(0.05, rel=0.2)           # both observations right: sharpen


def test_macro_f1_counts_every_label_in_truth_or_predictions():
    # A: tp 1 fn 1 -> F1 2/3; B: fp 1 -> 0; C: tp 1 -> 1. Mean over A, B, C.
    assert metrics.macro_f1(["A", "A", "C"], ["A", "B", "C"]) == pytest.approx((2 / 3 + 0 + 1) / 3)
    assert metrics.macro_f1([], []) is None


def test_every_comparison_reports_species_macro_f1_and_per_image_when_asked(conn, tmp_path):
    store = seed_two_species(conn, tmp_path)
    root = tmp_path / "emb"
    embed_photos(conn, store, Const("m1"), photos_to_embed(conn, "m1", "large", store.location,
                                                           "local"), root / "m1",
                 log=lambda s: None)
    out = evaluate.compare(conn, ["m1"], ["nearest"], embeddings_root=root, per_image=True,
                           log=lambda s: None)
    report = out["runs"][0]["all_photos"]
    assert report["macro_f1"] == {"species": 1.0, "n": 2}
    assert report["per_image"]["species"] == {"n": 2, "top1": 1.0}
    assert report["per_image"]["species_macro_f1"] == 1.0
    plain = evaluate.compare(conn, ["m1"], ["nearest"], embeddings_root=root, log=lambda s: None)
    assert "per_image" not in plain["runs"][0]["all_photos"]


# --- the held-out report ----------------------------------------------------------------

class SameLabeller:
    def label(self, name):
        return (name or "").strip()

    def same(self, a, b):
        return self.label(a) == self.label(b)


def truth(sp):
    return heldout.Truth(sp, sp.split()[0], "F", sp, False, None)


def test_the_held_out_report_gives_per_photo_and_per_record_numbers_and_macro_f1():
    truths = {"1": truth("A x"), "2": truth("B y")}
    answers = {"1": {"per_photo": [{"ranks": {"species": [{"name": "A x"}],
                                              "genus": [{"name": "A"}]}},
                                   {"ranks": {"species": [{"name": "B y"}],
                                              "genus": [{"name": "B"}]}}]},
               "2": {"per_photo": [{"ranks": {"species": [{"name": "B y"}],
                                              "genus": [{"name": "B"}]}}]}}
    out = heldout_report.per_image(answers, truths, SameLabeller())
    assert out["species"]["n"] == 3 and out["species"]["right"] == 2
    assert out["species_macro_f1"] == pytest.approx(2 / 3, abs=1e-4)   # A and B both 2/3
    judged = {"1": {"species": (True, True, "A x", 0.9)},
              "2": {"species": (False, True, "A x", 0.5)}}
    f1 = heldout_report.species_macro_f1(judged, truths, SameLabeller())
    assert f1 == {"n": 2, "macro_f1": round((2 / 3 + 0) / 2, 4)}


def test_the_report_prints_macro_f1_and_per_photo_beside_per_record():
    from mycomap_vision.cli import format_f1_and_per_image
    text = format_f1_and_per_image({"m/classifier": {
        "species": {"top1": {"rate": 0.5}}, "species_macro_f1": {"macro_f1": 0.25},
        "per_image": {"species": {"rate": 0.4}, "species_macro_f1": 0.2}}})
    assert "m/classifier" in text and "25.0%" in text and "40.0%" in text


# --- the photo cache (an option for 4-vCPU instances) -----------------------------------

def test_the_photo_cache_shrinks_each_training_photo_once_and_keeps_its_path(tmp_path):
    from PIL import Image

    from mycomap_vision.storage import LocalStore
    src = LocalStore(tmp_path / "src")
    for rel, size in (("a/1.jpg", (1024, 768)), ("b/2.jpg", (300, 500))):
        buf = __import__("io").BytesIO()
        Image.new("RGB", size, (200, 10, 10)).save(buf, "JPEG")
        src.put(rel, buf.getvalue())
    src.put("c/3.jpg", b"not a photo")
    items = [(1, "a/1.jpg", 0), (2, "b/2.jpg", 1), (3, "c/3.jpg", 1), (1, "a/1.jpg", 0)]
    out = picek.cache_photos(src, items, tmp_path / "cache", 440, threads=2,
                             log=lambda s: None)
    assert (out["photos"], out["made"], out["failed"]) == (3, 2, 1)
    assert Image.open(tmp_path / "cache" / "a/1.jpg").size == (587, 440)     # aspect kept
    assert Image.open(tmp_path / "cache" / "b/2.jpg").size == (300, 500)     # never enlarged
    again = picek.cache_photos(src, items, tmp_path / "cache", 440, threads=2,
                               log=lambda s: None)
    assert again["kept"] == 2 and again["made"] == 0


def test_a_cache_is_asked_for_in_the_spec_and_never_below_the_input_size():
    assert picek.cache_of("fungitastic-beit-b384@15@440") == 440
    assert picek.cache_of("fungitastic-beit-b384@15") is None
    assert picek.parse_spec("fungitastic-beit-b384@15@440") == ("fungitastic-beit-b384", 15)
    with pytest.raises(ValueError, match="cache size"):
        picek.parse_spec("fungitastic-beit-b384@15@300")
    assert picek.PicekConfig().cache_px is None                  # off unless asked for
