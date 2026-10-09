"""Observation sets (obsets.py, obsets_head.py): set-level scoring and the trained set head."""

import sqlite3

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mycomap_vision import methods, obsets, obsets_head  # noqa: E402
from mycomap_vision.evaluate import Record  # noqa: E402


def unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def toy_reference(specs, dim=8, seed=0):
    """specs: [(observation id, species, validated_on, [photo vectors])]."""
    rng = np.random.default_rng(seed)
    vecs, recs = [], []
    for oid, sp, day, photos in specs:
        rows = []
        for p in photos:
            rows.append(len(vecs))
            vecs.append(unit(p if p is not None else rng.normal(size=dim)))
        recs.append(Record(oid, sp, sp.split()[0], "Fam", day, "obs", rows))
    vecs = np.stack(vecs).astype(np.float16)
    ids = np.arange(100, 100 + len(vecs))
    return obsets.build_reference("toy", ids, vecs, recs), vecs, recs


def e(i, dim=8):
    v = np.zeros(dim, dtype=np.float32)
    v[i] = 1
    return v


# --- the segment operations -------------------------------------------------------------

def test_segment_maxima_and_top_k_means_match_a_plain_loop():
    rng = np.random.default_rng(3)
    seg = np.array([0, 0, 0, 1, 2, 2, 3, 3, 3, 3])
    x = rng.uniform(-1, 1, size=(4, len(seg))).astype(np.float32)
    starts, counts = obsets.starts_of(seg, 4)
    t = lambda a: torch.from_numpy(a)  # noqa: E731
    got_max = obsets.seg_max(t(x), t(seg), 4).numpy()
    got_top2 = obsets.seg_topk_mean(t(x), t(seg), t(starts), t(counts), 2).numpy()
    got_mean = obsets.seg_mean(t(x[0]), t(seg), t(counts)).numpy()
    for s in range(4):
        cols = x[:, seg == s]
        assert np.allclose(got_max[:, s], cols.max(axis=1))
        best = -np.sort(-cols, axis=1)[:, :2]
        assert np.allclose(got_top2[:, s], best.mean(axis=1), atol=1e-6)
        assert np.isclose(got_mean[s], cols[0].mean())


# --- the reference layout -------------------------------------------------------------------

def test_each_reference_record_keeps_its_photos_in_one_run_even_when_a_photo_is_shared():
    ref, vecs, recs = toy_reference([("1", "A a", "2026-01-01", [e(0), e(1)]),
                                     ("2", "B b", "2026-01-01", [e(2)]),
                                     ("3", "A a", "2026-01-01", [e(3)])])
    recs[2].photo_rows.append(recs[0].photo_rows[0])        # one photo under two records
    ref = obsets.build_reference("toy", np.arange(len(vecs)), vecs, recs)
    runs = np.flatnonzero(np.diff(ref.col_rec, prepend=-1) != 0)
    assert len(runs) == ref.records == 3
    assert list(np.bincount(ref.col_rec)) == [len(r.photo_rows) for r in
                                              [recs[0], recs[2], recs[1]]]
    assert [ref.index.species[u] for u in ref.rec_unit] == ["A a", "A a", "B b"]


# --- Step 1 scores ----------------------------------------------------------------------------

def _random_reference():
    rng = np.random.default_rng(5)
    specs = []
    for i in range(30):
        sp = f"G{i % 3} s{i % 7}"
        specs.append((str(i), sp, "2026-01-01", [rng.normal(size=8) for _ in range(rng.integers(1, 5))]))
    return toy_reference(specs)


def test_query_coverage_is_nearest_and_nearest_and_mean_matches_the_served_method():
    ref, vecs, recs = _random_reference()
    engine = obsets.SetEngine(ref, device="cpu")
    q = unit(np.random.default_rng(9).normal(size=(3, 8))).astype(np.float16)
    got = engine.scores(q)
    for name, cls in (("nearest", methods.NearestSpecimen), ("nearest+mean", methods.NearestAndMean)):
        m = cls()
        m.fit(vecs, ref.index)
        m.scorer.torch = None                         # the CPU path, float32 like the engine
        m.scorer.ref = np.asarray(vecs[ref.index.cols], dtype=np.float16)
        if hasattr(m, "means"):
            m.means.torch = None
            m.means.ref = np.asarray(methods.species_means(vecs, ref.index), dtype=np.float16)
        assert np.allclose(got[name], m.species_scores(q), atol=2e-3), name


def test_obs_forward_needs_every_query_photo_matched_within_one_reference_record():
    # Species A: two records, each holding one of the query's two views exactly.
    # Species B: one record holding both views, each a little off.
    near0, near1 = unit(e(0) + 0.3 * e(2)), unit(e(1) + 0.3 * e(2))
    ref, *_ = toy_reference([("1", "A a", "2026-01-01", [e(0)]),
                             ("2", "A a", "2026-01-01", [e(1)]),
                             ("3", "B b", "2026-01-01", [near0, near1])])
    s = obsets.SetEngine(ref, device="cpu").scores(np.stack([e(0), e(1)]))
    a, b = ref.index.species.index("A a"), ref.index.species.index("B b")
    assert s["nearest"][a] > s["nearest"][b]           # a different record per photo is fine
    assert s["obs-forward"][b] > s["obs-forward"][a]   # but not for a set match


def test_chamfer_counts_reference_photos_the_query_does_not_cover():
    ref, *_ = toy_reference([("1", "A a", "2026-01-01", [e(0)]),
                             ("2", "B b", "2026-01-01", [e(0), e(5), e(6)])])
    s = obsets.SetEngine(ref, device="cpu").scores(np.stack([e(0)]))
    a, b = ref.index.species.index("A a"), ref.index.species.index("B b")
    assert np.isclose(s["obs-forward"][a], s["obs-forward"][b])
    assert np.isclose(s["obs-chamfer"][a], 1.0)
    assert np.isclose(s["obs-chamfer"][b], (1.0 + 1 / 3) / 2, atol=1e-3)


def test_top2_over_observations_keeps_a_single_record_species():
    ref, *_ = toy_reference([("1", "A a", "2026-01-01", [e(0)]),
                             ("2", "B b", "2026-01-01", [e(0)]),
                             ("3", "B b", "2026-01-01", [e(4)])])
    s = obsets.SetEngine(ref, device="cpu").scores(np.stack([e(0)]))
    a, b = ref.index.species.index("A a"), ref.index.species.index("B b")
    assert np.isclose(s["obs-chamfer-top2"][a], 1.0)
    assert s["obs-chamfer-top2"][a] > s["obs-chamfer-top2"][b]


def test_answers_name_species_only_from_species_groups():
    ref, *_ = toy_reference([("1", "A a", "2026-01-01", [e(0)]),
                             ("2", "B b", "2026-01-01", [e(1)])])
    out = obsets.ranked(np.array([0.9, 0.1], dtype=np.float32), ref.index, top=10)
    assert [c["name"] for c in out["species"]] == ["A a", "B b"]
    assert [c["name"] for c in out["genus"]] == ["A", "B"]


def test_the_manifest_is_opened_read_only(tmp_path):
    path = tmp_path / "m.sqlite"
    sqlite3.connect(path).execute("create table t (x)").connection.commit()
    conn = obsets.read_only(path)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("insert into t values (1)")


# --- Step 2: the set head ---------------------------------------------------------------------

def test_the_set_head_answers_a_1_photo_observation():
    head = obsets_head.make_head(8).eval()
    z, alpha = obsets_head.embed_query(head, unit(e(3))[None], "cpu")
    assert z.shape == (8,)
    assert np.isclose(float(z.norm()), 1.0, atol=1e-5)
    assert np.allclose(alpha, [1.0])


def test_the_untrained_set_head_is_the_plain_mean_of_the_photos():
    head = obsets_head.make_head(8).eval()
    q = unit(np.random.default_rng(1).normal(size=(4, 8)))
    z, alpha = obsets_head.embed_query(head, q, "cpu")
    assert np.allclose(alpha, 0.25)
    assert np.allclose(z.numpy(), unit(q.mean(axis=0)), atol=1e-5)


def _dated_reference():
    specs = []
    rng = np.random.default_rng(2)
    for i in range(40):
        day = "2026-08-01" if i < 30 else "2026-09-20"
        specs.append((str(1000 + i), f"G{i % 4} s{i % 4}", day,
                      [unit(e(i % 4) + 0.2 * rng.normal(size=8)) for _ in range(1 + i % 3)]))
    return toy_reference(specs)[0]


def test_held_out_ids_never_enter_training():
    ref = _dated_reference()
    held = {"1003", "1035"}
    split = obsets_head.training_split(ref, held)
    used = set(ref.rec_obs[split.train].tolist()) | set(ref.rec_obs[split.val].tolist())
    assert not used & held
    assert len(used) == ref.records - 2
    leaky = obsets_head.Split(np.arange(ref.records), np.array([], dtype=np.int64))
    with pytest.raises(obsets_head.HeldOutInTraining):
        obsets_head.check_no_held_out(ref, leaky, held)


def test_the_head_trains_on_older_records_and_stops_on_newer_ones():
    ref = _dated_reference()
    split = obsets_head.training_split(ref, set(), until="2026-09-07")
    assert all(d <= "2026-09-07" for d in ref.rec_validated[split.train])
    assert all(d > "2026-09-07" for d in ref.rec_validated[split.val])
    cfg = obsets_head.TrainConfig(epochs=3, batch=8, patience=2)
    head, notes = obsets_head.train_head(ref, split, cfg, device="cpu", log=lambda *_: None)
    assert notes["train_records"] == 30 and notes["val_records"] == 10
    assert notes["best_val_top1"] >= notes["history"][0]["val_top1"]


def test_an_attention_only_head_keeps_the_photos_own_space():
    ref = _dated_reference()
    split = obsets_head.training_split(ref, set(), until="2026-09-07")
    cfg = obsets_head.TrainConfig(epochs=2, batch=8, patience=2, project=False)
    head, _ = obsets_head.train_head(ref, split, cfg, device="cpu", log=lambda *_: None)
    assert float(head.W.weight.abs().sum()) == 0.0
    q = unit(np.random.default_rng(4).normal(size=(3, 8)))
    z, alpha = obsets_head.embed_query(head, q, "cpu")
    assert np.allclose(z.numpy(), unit((alpha[:, None] * q).sum(axis=0)), atol=1e-5)


def test_excluded_records_leave_the_reference_and_its_depth_counts():
    ref, vecs, recs = toy_reference([("1", "A a", "2026-01-01", [e(0)]),
                                     ("2", "A a", "2026-01-01", [e(1)]),
                                     ("3", "B b", "2026-01-01", [e(2)])])
    clean = obsets.build_reference("toy", np.arange(len(vecs)), vecs,
                                   [r for r in recs if r.observation_id not in {"2"}])
    assert "2" not in set(clean.rec_obs.tolist())
    assert obsets.rank_counts(ref)["species"] == {"A a": 2, "B b": 1}
    assert obsets.rank_counts(clean)["species"] == {"A a": 1, "B b": 1}
    s = obsets.SetEngine(clean, device="cpu").scores(np.stack([e(1)]))
    assert s["nearest"].max() < 0.5                 # the left-out photo can't be matched


def test_an_id_list_reads_one_id_per_line(tmp_path):
    path = tmp_path / "ids.txt"
    path.write_text("# audit\n101\n\n202\n", encoding="utf-8")
    assert obsets.read_id_list(path) == {"101", "202"}


def test_the_head_blend_check_scores_newer_records_against_older_ones_only():
    ref = _dated_reference()
    split = obsets_head.training_split(ref, set(), until="2026-09-07")
    cfg = obsets_head.TrainConfig(epochs=1, batch=8)
    head = obsets_head.make_head(8).eval()
    out = obsets_head.head_time_slice_check(ref, split, head, cfg, weights=(0.5,))
    assert out["records"] == len(split.val)
    assert set(out["methods"]) == {"nearest+mean", "set-head-top2", "blend:set-head-top2@0.5"}


# --- reproducibility -----------------------------------------------------------------------

def test_the_reference_hash_names_the_records_and_their_labels_not_their_order():
    specs = [("1", "A a", "2026-01-01", [e(0)]), ("2", "B b", "2026-01-01", [e(1)])]
    one, *_ = toy_reference(specs)
    two, *_ = toy_reference(list(reversed(specs)))
    relabelled, *_ = toy_reference([("1", "A c", "2026-01-01", [e(0)]), specs[1]])
    assert obsets.reference_hash(one) == obsets.reference_hash(two)
    assert obsets.reference_hash(one) != obsets.reference_hash(relabelled)


def test_a_manifest_snapshot_is_a_full_copy_and_never_overwrites(tmp_path):
    src = tmp_path / "manifest.sqlite"
    c = sqlite3.connect(src)
    c.execute("create table t (x)")
    c.executemany("insert into t values (?)", [(i,) for i in range(5)])
    c.commit()
    c.close()
    out = obsets.snapshot_manifest(src, tmp_path / "snap.sqlite")
    assert sqlite3.connect(tmp_path / "snap.sqlite").execute("select count(*) from t").fetchone()[0] == 5
    assert out["sha256"] == obsets.file_sha256(tmp_path / "snap.sqlite")
    with pytest.raises(FileExistsError):
        obsets.snapshot_manifest(src, tmp_path / "snap.sqlite")


def test_nearest_plus_prior_here_is_the_served_nearest_plus_prior():
    from functools import partial

    from mycomap_vision.methods import AsLogProb, NearestSpecimen
    from mycomap_vision.prior import Context, WithPrior
    rng = np.random.default_rng(6)
    ref, vecs, recs = _random_reference()
    for r in recs:
        r.latitude, r.longitude = float(rng.uniform(30, 50)), float(rng.uniform(-120, -70))
        r.observed_on = f"2025-{rng.integers(1, 13):02d}-10"
    ref = obsets.build_reference("toy", np.arange(len(vecs)), vecs, recs)
    q = unit(rng.normal(size=(2, 8))).astype(np.float16)
    ctx = Context(40.0, -100.0, "2025-06-10")
    nearest = obsets.SetEngine(ref, device="cpu").scores(q)["nearest"]
    got = obsets.with_prior(nearest, obsets.fit_prior(ref), ctx)
    served = WithPrior(partial(AsLogProb, NearestSpecimen))
    served.fit(vecs, ref.index, recs)
    served.base.base.scorer.torch = None
    served.base.base.scorer.ref = np.asarray(vecs[ref.index.cols], dtype=np.float16)
    assert np.allclose(got, served.species_scores(q, ctx), atol=0.1)
    assert np.argmax(got) == np.argmax(served.species_scores(q, ctx))
