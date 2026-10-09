import sys

import numpy as np
import pytest

from mycomap_vision import loo
from mycomap_vision.evaluate import Record, build_index
from mycomap_vision.methods import NearestAndMean


@pytest.fixture(autouse=True)
def no_torch(monkeypatch):
    # The reference comparison must run on numpy (as the scan does), not a shared GPU.
    monkeypatch.setitem(sys.modules, "torch", None)


def reference(seed=0, dim=12):
    """Five species; records of 1-3 photos; some observers record twice on one day."""
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(5, dim))
    vecs, recs = [], []
    plan = [(0, "ann", "2026-01-01", 2), (0, "ann", "2026-01-01", 1), (0, "bob", "2026-01-02", 3),
            (0, "cat", "2026-01-03", 1), (1, "bob", "2026-01-02", 2), (1, "dan", "2026-01-04", 1),
            (1, "eve", "2026-01-05", 2), (2, "ann", "2026-01-01", 1), (2, "fay", None, 2),
            (3, "gus", "2026-01-06", 1), (3, "gus", "2026-01-06", 2), (4, None, None, 1),
            (4, "hal", "2026-01-07", 3), (4, "ida", "2026-01-08", 2)]
    for i, (s, who, day, n) in enumerate(plan):
        rows = []
        for _ in range(n):
            v = centres[s] + 0.6 * rng.normal(size=dim)
            vecs.append(v / np.linalg.norm(v))
            rows.append(len(vecs) - 1)
        recs.append(Record(f"r{i}", f"G{s % 2} sp{s}", f"G{s % 2}", "F", "2026-02-01", who,
                           rows, observed_on=day))
    return np.asarray(vecs, dtype=np.float16), recs


def scan(vecs, recs):
    index = build_index(recs)
    layout, ordered = loo.build_layout(recs, index)
    ref = vecs[layout.col_rows].astype(np.float32)
    sums = loo.unit_sums(ref, layout)
    return layout, ordered, ref, sums, loo.as_served(sums)


def test_a_left_out_record_scores_as_if_its_observer_day_was_never_in_the_reference():
    vecs, recs = reference()
    layout, ordered, ref, sums, means = scan(vecs, recs)
    block = loo.score_records(ref, layout, list(range(layout.n_records)), sums, means)
    for j, rec in enumerate(ordered):
        gone = {ordered[g].observation_id
                for g in layout.group_recs[layout.rec_group[j]].tolist()}
        rest = [r for r in recs if r.observation_id not in gone]
        index = build_index(rest)
        model = NearestAndMean()
        model.fit(vecs, index)
        want = dict(zip(index.species, model.species_scores(vecs[rec.photo_rows])))
        for u, name in enumerate(layout.units):
            got = block.unit[j, u]
            if name in want:
                assert got == pytest.approx(want[name], abs=2e-3), (rec.observation_id, name)
            else:
                assert got == -np.inf, (rec.observation_id, name)


def test_a_species_known_only_from_the_left_out_day_is_no_candidate():
    vecs, recs = reference()
    layout, ordered, ref, sums, means = scan(vecs, recs)
    j = next(i for i, r in enumerate(ordered) if r.observation_id == "r9")   # gus, sp3 only
    block = loo.score_records(ref, layout, [j], sums, means)
    sp3 = layout.units.index("G1 sp3")
    assert block.unit[0, sp3] == -np.inf
    assert block.available[0, sp3] == 0


def test_a_record_without_observer_or_day_leaves_out_only_itself():
    vecs, recs = reference()
    layout, ordered, *_ = scan(vecs, recs)
    j = next(i for i, r in enumerate(ordered) if r.observation_id == "r11")
    assert [ordered[g].observation_id for g in layout.group_recs[layout.rec_group[j]]] == ["r11"]


def test_a_photo_filed_under_two_records_is_hidden_from_both():
    vecs, recs = reference()
    shared = recs[5].photo_rows[0]          # dan's sp1 photo, also filed under eve's record
    recs[6].photo_rows.append(shared)
    layout, ordered, ref, sums, means = scan(vecs, recs)
    j = next(i for i, r in enumerate(ordered) if r.observation_id == "r6")
    hidden = loo.left_out_cols(layout, j)
    assert set(np.flatnonzero(layout.col_rows == shared).tolist()) <= set(hidden.tolist())
    block = loo.score_records(ref, layout, [j], sums, means)
    eve_photo = len(ordered[j].photo_rows) - 1
    rows = block.photo_owner == 0
    assert block.record_max[rows][eve_photo].max() < 0.999


def test_two_equal_best_photos_both_count_in_the_top_two():
    vecs, recs = reference()
    twin = recs[3].photo_rows[0]
    recs.append(Record("twin", recs[3].species, recs[3].genus, "F", "2026-02-01", "zed",
                       [len(vecs)], observed_on="2026-03-01"))
    vecs = np.vstack([vecs, vecs[twin:twin + 1]])
    recs.append(Record("q", "G0 sp4", "G0", "F", "2026-02-01", "yan", [len(vecs)],
                       observed_on="2026-03-02"))
    vecs = np.vstack([vecs, vecs[twin:twin + 1]])     # a query identical to the twins
    layout, ordered, ref, sums, means = scan(vecs, recs)
    j = next(i for i, r in enumerate(ordered) if r.observation_id == "q")
    block = loo.score_records(ref, layout, [j], sums, means)
    u = layout.units.index(recs[3].species)
    nearest = (block.unit[0, u] - 0.4 * float(ref[layout.rec_starts[j]] @ means[u])) / 0.6
    assert nearest == pytest.approx(1.0, abs=2e-3)


def rec(**kw):
    base = {"label": 0, "pred": 1, "conf": 0.9, "margin": 0.05, "nb_pred": 8,
            "best_match": 0.8, "dup_other_genus": 0.0}
    return {**base, **kw}


RULES = loo.Rules()
SUPPORT = {0: 30, 1: 30}


def test_a_confident_well_neighboured_answer_against_a_recognised_label_is_a_wrong_label():
    assert loo.classify(rec(), RULES, {}, {}, {0: 0.8, 1: 0.8}, SUPPORT, "") == "a"


def test_an_unsure_or_poorly_neighboured_answer_is_not_flagged():
    self_rate = {0: 0.8, 1: 0.8}
    assert loo.classify(rec(conf=0.3, margin=0.0), RULES, {}, {}, self_rate, SUPPORT, "") == ""
    assert loo.classify(rec(nb_pred=3), RULES, {}, {}, self_rate, SUPPORT, "") == ""


def test_a_label_its_own_records_do_not_support_is_not_called_wrong():
    assert loo.classify(rec(), RULES, {}, {}, {0: 0.2, 1: 0.8}, SUPPORT, "") == ""
    assert loo.classify(rec(), RULES, {}, {}, {0: 0.9, 1: 0.8}, {0: 2, 1: 30}, "") == ""


def test_two_writings_of_one_name_are_the_same_species_under_two_names():
    assert loo.classify(rec(), RULES, {}, {}, {0: 0.8, 1: 0.8}, SUPPORT,
                        "same name, other writing") == "b"


def test_a_loose_name_relation_needs_a_systematic_pair():
    self_rate = {0: 0.8, 1: 0.8}
    rel = "provisional beside formal, one genus"
    assert loo.classify(rec(), RULES, {}, {(0, 1): 1}, self_rate, SUPPORT, rel) == "a"
    assert loo.classify(rec(), RULES, {(0, 1): 0.1}, {(0, 1): 3}, self_rate, SUPPORT,
                        rel) == "b"


def test_labels_mostly_predicted_as_each_other_are_one_species():
    rates = {(0, 1): 0.6, (1, 0): 0.55}
    assert loo.classify(rec(), RULES, rates, {}, {0: 0.3, 1: 0.4}, SUPPORT, "") == "b"


def test_mutual_confusion_between_two_kept_species_is_a_look_alike_not_a_mislabel():
    rates = {(0, 1): 0.25, (1, 0): 0.3}
    assert loo.classify(rec(), RULES, rates, {}, {0: 0.6, 1: 0.6}, SUPPORT, "") == "d"


def test_a_record_that_matches_nothing_has_wrong_photos():
    assert loo.classify(rec(best_match=0.4), RULES, {}, {}, {0: 0.8}, SUPPORT, "") == "c"


def test_related_names_reads_writing_variants_and_provisional_codes():
    from mycomap_vision import name_equiv
    from mycomap_vision.heldout import name_key
    assert loo.related_names("Russula sp. 'IN07'", "Russula sp-IN07", name_key,
                             name_equiv) == "same name, other writing"
    assert loo.related_names("Russula sp. 'IN07'", "Russula emetica", name_key,
                             name_equiv) == "provisional beside formal, one genus"
    assert loo.related_names("Russula emetica", "Lactarius deliciosus", name_key,
                             name_equiv) == ""


def test_an_outside_query_scores_as_against_a_reference_without_the_hidden_records():
    vecs, recs = reference(seed=3)
    layout, ordered, ref, sums, means = scan(vecs, recs)
    rng = np.random.default_rng(9)
    q = rng.normal(size=(3, vecs.shape[1]))
    q = (q / np.linalg.norm(q, axis=1, keepdims=True)).astype(np.float16).astype(np.float32)
    drop = {"r2", "r12"}
    hidden = np.concatenate([layout.record_cols(i) for i, r in enumerate(ordered)
                             if r.observation_id in drop])
    block = loo.score_queries(q, np.zeros(3, dtype=int), [hidden], ref, layout, sums, means)
    rest = [r for r in recs if r.observation_id not in drop]
    index = build_index(rest)
    model = NearestAndMean()
    model.fit(vecs, index)
    want = dict(zip(index.species, model.species_scores(q.astype(np.float16))))
    for u, name in enumerate(layout.units):
        assert block.unit[0, u] == pytest.approx(want[name], abs=2e-3)
    full = loo.score_queries(q, np.zeros(3, dtype=int), [np.zeros(0, dtype=int)], ref, layout,
                             sums, means)
    model.fit(vecs, build_index(recs))
    assert full.unit[0] == pytest.approx(model.species_scores(q.astype(np.float16)), abs=2e-3)


def test_a_label_known_only_from_the_same_collecting_day_is_not_well_supported():
    self_rate = {0: 0.9, 1: 0.8}
    assert loo.classify(rec(label_left=1), RULES, {}, {}, self_rate, SUPPORT, "") == ""
    assert loo.classify(rec(label_left=5), RULES, {}, {}, self_rate, SUPPORT, "") == "a"


def test_a_removal_set_prepared_once_scores_like_one_hidden_per_query():
    vecs, recs = reference(seed=4)
    layout, ordered, ref, sums, means = scan(vecs, recs)
    q = ref[:4].copy()
    hidden = np.concatenate([layout.record_cols(i) for i, r in enumerate(ordered)
                             if r.observation_id in {"r5", "r13"}])
    once = loo.score_queries(q, np.zeros(4, dtype=int), [hidden], ref, layout, sums, means)
    s2, m2 = loo.without(ref, layout, sums, hidden)
    pre = loo.score_queries(q, np.zeros(4, dtype=int), [hidden], ref, layout, s2, m2,
                            adjust_means=False)
    assert np.allclose(once.unit, pre.unit, atol=1e-4)


def test_a_photo_that_is_another_genus_records_picture_is_a_wrong_photo():
    self_rate = {0: 0.8, 1: 0.8}
    assert loo.classify(rec(dup_other_genus=0.97), RULES, {}, {}, self_rate, SUPPORT, "") == "c"
    # A close look-alike in another genus is not a duplicate.
    assert loo.classify(rec(pred=0, dup_other_genus=0.9), RULES, {}, {}, self_rate, SUPPORT,
                        "") == ""


def test_a_record_the_model_agrees_with_has_no_mislabel_strength():
    assert loo.strength(rec(pred=0), {}, {0: 0.9}) == 0.0
    assert loo.strength(rec(), {}, {0: 0.9}) > 0.5


def test_a_review_key_maps_to_the_manifests_own_key():
    old = {"123", "456"}
    new = {"123", "mo:456"}
    assert loo.release_key("inat:123", old) == "123"
    assert loo.release_key("mo:456", old) == "456"         # before record sources
    assert loo.release_key("mo:456", new) == "mo:456"
    assert loo.release_key("mo:123", new) is None          # never the iNat record by mistake
    assert loo.release_key("inat:999", old) is None


def test_only_person_checked_exclusions_reach_the_release_list_and_missing_keys_fail():
    reviewed = [
        {"record_key": "inat:1", "decision": "exclude-record", "checked_by": "Steve",
         "category": "c"},
        {"record_key": "inat:2", "decision": "exclude-record", "checked_by": "", "category": "c"},
        {"record_key": "inat:3", "decision": "relabel-on-.com", "checked_by": "Steve"},
        {"record_key": "inat:9", "decision": "exclude-record", "checked_by": "Steve"},
        {"record_key": "inat:1", "decision": "exclude-photo", "photo_id": "77",
         "checked_by": "Steve", "category": "c"},
    ]
    rows, problems = loo.exclusion_rows(reviewed, {"1", "2", "3"}, {77})
    assert [(r["kind"], r["key"]) for r in rows] == [("record", "1"), ("photo", "inat:77")]
    assert any("inat:2" in p for p in problems) and any("inat:9" in p for p in problems)
