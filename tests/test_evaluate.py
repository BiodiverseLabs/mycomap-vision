import numpy as np

from mycomap_vision.evaluate import (Record, bucket_of, build_index, evaluate, rank_scores,
                                     species_scores, split_by_time, top_labels)


def unit(*v):
    a = np.asarray(v, dtype=np.float32)
    return a / np.linalg.norm(a)


# Photo vectors: rows 0-5. Two Russula species and one Amanita.
VEC = np.stack([
    unit(1, 0, 0, 0),      # 0 Russula emetica, ref
    unit(1, .1, 0, 0),     # 1 Russula emetica, ref
    unit(0, 1, 0, 0),      # 2 Russula rara (a single specimen), ref
    unit(0, 0, 1, 0),      # 3 Amanita muscaria, ref
    unit(0, .95, .1, 0),   # 4 query cap photo, near R. rara
    unit(0, 0, .2, 1),     # 5 query photo like nothing much; faintly Amanita
]).astype(np.float16)


def rec(oid, species, rows, vdate="2026-01-01", family="Russulaceae"):
    return Record(oid, species, species.split()[0], family, vdate, "u", list(rows))


REF = [rec("a", "Russula emetica", [0]), rec("b", "Russula emetica", [1]),
       rec("c", "Russula rara", [2]), rec("d", "Amanita muscaria", [3], family="Amanitaceae")]


def test_the_newest_weeks_are_the_test_set():
    recs = [rec("1", "X y", [], "2026-09-27"), rec("2", "X y", [], "2026-08-01"),
            rec("3", "X y", [], None)]
    ref, test, cutoff = split_by_time(recs, 28)
    assert cutoff == "2026-08-30"
    assert [r.observation_id for r in test] == ["1"]
    assert sorted(r.observation_id for r in ref) == ["2", "3"]   # undated stays reference


def test_a_single_specimen_species_is_not_outvoted_by_a_well_sampled_one():
    # 50 specimens of a common look-alike, each a fair match (~0.89), against one
    # specimen of the right species that matches better (~0.995). Counting or summing
    # matches would pick the common species; the best single match must not.
    vecs = np.stack([unit(0, 1, 0, 0), unit(.1, 1, 0, 0)] + [unit(.5, 1, 0, 0)] * 50)
    ref = [rec("r", "Russula rara", [1])] + [rec(f"c{i}", "Russula emetica", [2 + i])
                                             for i in range(50)]
    index = build_index(ref)
    sims = vecs[[0]] @ vecs[index.cols].T
    assert top_labels(species_scores(sims, index), index.species, 1) == ["Russula rara"]


def test_genus_and_family_scores_are_the_best_species_inside_them():
    index = build_index(REF)
    scores = np.asarray([0.2, 0.9, 0.5], dtype=np.float32)   # order: A. muscaria, R. emetica, R. rara
    assert index.species == ["Amanita muscaria", "Russula emetica", "Russula rara"]
    genus = rank_scores(scores, index, "genus")
    assert index.labels["genus"] == ["Amanita", "Russula"]
    assert np.allclose(genus, [0.2, 0.9])
    fam = rank_scores(scores, index, "family")
    assert top_labels(fam, index.labels["family"], 2) == ["Russulaceae", "Amanitaceae"]


def test_accuracy_is_reported_per_rank_and_per_reference_count():
    test = [rec("q1", "Russula rara", [4], "2026-09-20"),
            rec("q2", "Russula nova", [5], "2026-09-20")]     # species absent from reference
    out = evaluate(VEC, REF, test)
    assert out["species"]["1 ref"] == {"n": 1, "top1": 1.0, "top5": 1.0}
    assert out["species"]["novel (0 refs)"]["top1"] == 0.0
    assert out["species"]["all"]["n"] == 2
    assert out["genus"]["all"]["n"] == 2


def test_extra_photos_change_the_answer_only_through_their_own_similarity():
    # The record's first photo looks like nothing; its second is the telling one.
    test = [rec("q", "Russula rara", [5, 4], "2026-09-20")]
    assert evaluate(VEC, REF, test)["species"]["all"]["top1"] == 1.0
    assert evaluate(VEC, REF, test, first_photo_only=True)["species"]["all"]["top1"] == 0.0


def test_reference_count_buckets():
    assert [bucket_of(n) for n in (0, 1, 2, 4, 30, 31)] == [
        "novel (0 refs)", "1 ref", "2 refs", "3-5 refs", "6-30 refs", "31+ refs"]


def test_calibration_is_sharper_for_a_reliable_model_than_for_a_guessing_one():
    from mycomap_vision.evaluate import T_GRID
    rng = np.random.default_rng(0)
    dim, n_species = 16, 20
    centres = rng.normal(size=(n_species, dim))
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)

    def world(noise):
        vecs, ref, test = [], [], []
        for s in range(n_species):
            for j in range(3):
                v = centres[s] + noise * rng.normal(size=dim)
                vecs.append(v / np.linalg.norm(v))
                recs = ref if j < 2 else test
                recs.append(rec(f"{s}-{j}", f"G{s} sp{s}", [len(vecs) - 1],
                                "2026-01-01" if j < 2 else "2026-09-20", family=f"F{s}"))
        return np.asarray(vecs, dtype=np.float16), ref, test

    sharp = evaluate(*world(0.05))["calibration"]["species"]
    flat = evaluate(*world(3.0))["calibration"]["species"]
    assert sharp["n"] == flat["n"] == n_species
    assert sharp["temperature"] < flat["temperature"]
    assert T_GRID[0] <= sharp["temperature"] <= T_GRID[-1]


def test_results_are_also_broken_down_by_project_and_observer():
    test = [Record("q1", "Russula rara", "Russula", "Russulaceae", "2026-09-20", "ann", [4],
                   projects=("Indiana", "Fungal Diversity")),
            Record("q2", "Russula nova", "Russula", "Russulaceae", "2026-09-20", "bob", [5],
                   projects=("Indiana",))]
    out = evaluate(VEC, REF, test)
    projects = {g["name"]: g for g in out["groups"]["project"]}
    assert projects["Indiana"]["n"] == 2 and projects["Indiana"]["species_top1"] == 0.5
    assert projects["Fungal Diversity"] == {"name": "Fungal Diversity", "n": 1,
                                            "species_top1": 1.0, "genus_top1": 1.0}
    observers = [g["name"] for g in out["groups"]["observer"]]
    assert sorted(observers) == ["ann", "bob"]
    assert out["groups"]["project"][0]["name"] == "Indiana"          # largest first
