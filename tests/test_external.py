"""Published fungi classifiers as outside baselines (external.py, external_report.py): the
class map rebuilt from public metadata, their names read as Vision's, the authors' record
rule, answers kept out of the manifest until imported, the scoreboard row, and the
protocol tables (coverage, formal species, same vocabulary)."""

import csv
import json
import sqlite3

import numpy as np
import pytest

from test_heldout import (NAME, fetched, frozen, green, predicted, reference)
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision import (cli, config, evaluate, external, external_report, heldout,
                            heldout_report)
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.evaluate import Record
from mycomap_vision.external import ClassLabel, ExternalModel
from mycomap_vision.records import build_records, save_records


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    import sys
    monkeypatch.setitem(sys.modules, "torch", None)


FAKE = ExternalModel("fake-b384", "BVRA/fake_b384", "vit_fake", 3, "Fake Fungi",
                     "fake-train.csv", "class_id")
CLASSES = [("Russula emetica (Schaeff.) Pers.", "Russula emetica", "Russulaceae"),
           ("Amanita muscaria (L.) Lam.", "Amanita muscaria", "Amanitaceae"),
           ("Agaricus campestris L.", "Agaricus campestris", "Agaricaceae")]


def write_metadata(path, classes=CLASSES, id_column="class_id", rows_per_class=2, extra=()):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([id_column, "scientificName", "species", "family"])
        for cid, (sci, sp, fam) in enumerate(classes):
            for _ in range(rows_per_class):
                w.writerow([cid, sci, sp, fam])
        for row in extra:
            w.writerow(row)
    return path


@pytest.fixture
def fake_model(tmp_path, monkeypatch):
    """FAKE installed under <tmp>/data/external with its config, weights and class map."""
    root = tmp_path / "data" / "external"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(external.MODELS, FAKE.short, FAKE)
    folder = FAKE.folder(root)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(
        {"architecture": "vit_fake", "num_classes": 3, "input_size": [3, 384, 384]}))
    (folder / "pytorch_model.bin").write_bytes(b"weights")
    write_metadata(root / "metadata" / FAKE.metadata)
    external.write_labels(FAKE, root)
    return root


def red_first(images):
    """Logits that see colour: a red photo is a Russula emetica, anything else Amanita."""
    out = []
    for im in images:
        r, g, _b = np.asarray(im, dtype=np.float32).reshape(-1, 3).mean(0)
        out.append([6.0, 1.0, 0.0] if r > g else [0.0, 6.0, 1.0])
    return np.asarray(out, dtype=np.float32)


# --- the class map ------------------------------------------------------------------------------

@pytest.mark.parametrize("sci, fallback, want", [
    ("Gliophorus perplexus (A.H.Sm. & Hesler) Kovalenko", "", "Gliophorus perplexus"),
    ("Amanita muscaria var. formosa Pers.", "", "Amanita muscaria var. formosa"),
    ("Hygrocybe glutinipes ssp. rubra Bon", "", "Hygrocybe glutinipes subsp. rubra"),
    ("Achroomyces disciformis (Fr.) Donk", "Platygloea disciformis", "Achroomyces disciformis"),
    ("Fungi", "Fallback name", "Fallback name"),
])
def test_a_class_is_the_name_its_records_were_made_under_without_the_author(sci, fallback,
                                                                            want):
    assert external.canonical_name(sci, fallback) == want


def test_the_class_map_is_one_name_per_id_from_the_training_metadata(tmp_path):
    path = write_metadata(tmp_path / "m.csv", extra=[(-1, "Unknown sp.", "", "")])
    labels = external.build_labels(path, "class_id", 3)
    assert [(c.class_id, c.name, c.images) for c in labels] == [
        (0, "Russula emetica", 2), (1, "Amanita muscaria", 2), (2, "Agaricus campestris", 2)]


def test_a_map_missing_a_class_or_with_one_more_than_the_weights_is_refused(tmp_path):
    path = write_metadata(tmp_path / "m.csv")
    with pytest.raises(ValueError, match="the checkpoint has 4"):
        external.build_labels(path, "class_id", 4)
    with pytest.raises(ValueError, match="the checkpoint has 2"):
        external.build_labels(path, "class_id", 2)


def test_an_id_whose_rows_carry_two_names_is_refused(tmp_path):
    path = write_metadata(tmp_path / "m.csv", extra=[(1, "Amanita pantherina", "", "")])
    with pytest.raises(ValueError, match="class 1 has 2 names"):
        external.build_labels(path, "class_id", 3)


def test_the_saved_map_names_its_source_and_a_config_that_disagrees_is_refused(fake_model):
    body = json.loads(external.labels_path(FAKE, fake_model).read_text())
    assert body["source"] == FAKE.metadata and len(body["source_sha256"]) == 64
    assert [c.name for c in external.read_labels(FAKE, fake_model)][0] == "Russula emetica"
    cfg = FAKE.folder(fake_model) / "config.json"
    cfg.write_text(json.dumps({"architecture": "vit_fake", "num_classes": 4}))
    with pytest.raises(ValueError, match="4 classes"):
        external.check_config(FAKE, fake_model)


def test_a_new_class_map_or_new_weights_is_a_new_checkpoint_id(fake_model):
    first = external.checkpoint_id(FAKE, fake_model)
    write_metadata(fake_model / "metadata" / FAKE.metadata, rows_per_class=3)
    external.write_labels(FAKE, fake_model)
    second = external.checkpoint_id(FAKE, fake_model)
    assert first != second and first.startswith("fake-b384:")


# --- their names in Vision's -----------------------------------------------------------------

def test_a_class_takes_vision_s_name_then_its_accepted_name_then_its_own(conn):
    save_records(conn, build_records([green(1, "Fomitopsis betulina", family="Fomitopsidaceae"),
                                      green(2, "Russula emetica")], "t"))
    labels = [ClassLabel(0, "Piptoporus betulinus", "Fomitopsis betulina", "Fomitopsidaceae", 9),
              ClassLabel(1, "Russula emetica", "Russula emetica", "Russulaceae", 9),
              ClassLabel(2, "Lepista nuda", "Lepista nuda", "Tricholomataceae", 9)]
    cn = external.vision_names(labels, heldout.Labeller(conn))
    assert cn.species == ["Fomitopsis betulina", "Russula emetica", "Lepista nuda"]
    assert cn.genus == ["Fomitopsis", "Russula", "Lepista"]
    assert cn.family == ["Fomitopsidaceae", "Russulaceae", "Tricholomataceae"]
    assert cn.how == {"accepted name, known to Vision": 1, "own name, known to Vision": 1,
                      "not a Vision name": 1}


# --- the authors' record rule ---------------------------------------------------------------------

def names_of(species, genus=None, family=None):
    return external.ClassNames(species, genus or [s.split()[0] for s in species],
                               family or ["F"] * len(species), list(species))


def test_a_record_averages_its_photos_logits_not_their_probabilities():
    cn = names_of(["A a", "B b", "C c"])
    # One photo is very sure of B; three lean A. Mean logits pick B, mean probabilities A.
    logits = np.array([[0, 20, 0], [5, 0, 0], [5, 0, 0], [5, 0, 0]], dtype=np.float32)
    got = external.record_answer(logits, cn)
    assert got["species"][0]["name"] == "B b"
    assert got["per_image_top1"] == ["B b", "A a", "A a", "A a"]
    assert got["photo_only"] is True and got["temperature"] == 1.0


def test_classes_that_land_on_one_name_are_one_candidate_and_ranks_above_sum_classes():
    cn = names_of(["Gliophorus psittacinus", "Gliophorus psittacinus", "Hygrocybe miniata"],
                  family=["Hygrophoraceae"] * 3)
    got = external.record_answer(np.array([[1.0, 1.0, 1.5]]), cn)
    assert [c["name"] for c in got["species"]] == ["Gliophorus psittacinus", "Hygrocybe miniata"]
    p = external.softmax(np.array([1.0, 1.0, 1.5]))
    assert got["species"][0]["confidence"] == pytest.approx(p[0] + p[1], abs=1e-4)
    assert got["family"] == [{"name": "Hygrophoraceae", "confidence": 1.0}]
    assert got["top_class"] == "Hygrocybe miniata"      # the best single class; summed, it loses


# --- coverage and the same vocabulary ---------------------------------------------------------

def test_coverage_sorts_true_names_by_what_the_model_can_name(conn):
    save_records(conn, build_records([green(1, "Russula emetica")], "t"))
    labeller = heldout.Labeller(conn)
    vocab = external_report.Vocabulary(names_of(["Russula emetica", "Cortinarius pacificus"]))
    truths = [labeller.truth(n) for n in ("Russula emetica", "Russula sp. 'IN07'",
                                          "Thaxterogaster pacificus", "Amanita lavendula",
                                          "Russula")]
    cov = external_report.coverage(((t, 1) for t in truths), vocab)
    assert {k: v["records"] for k, v in cov["species"].items()} == {
        "provisional (temporary code)": 1, "formal, in vocabulary": 1,
        "formal, in vocabulary s.l. only": 1, "formal, not in vocabulary": 1,
        "one word, no species": 1}
    assert {k: v["records"] for k, v in cov["genus"].items()} == {
        "in vocabulary": 3, "in vocabulary s.l. only": 1, "not in vocabulary": 1,
        "no genus": 0}


def test_restricting_vision_keeps_only_the_models_names_in_visions_order():
    vocab = external_report.Vocabulary(names_of(["Russula emetica", "Amanita muscaria"]))
    answer = {"species": [{"name": "Russula sp. 'IN07'"}, {"name": "Amanita muscaria"},
                          {"name": "Lactarius x"}, {"name": "Russula emetica"}],
              "genus": [{"name": "Russula"}]}
    got, left = external_report.restrict(answer, vocab)
    assert [c["name"] for c in got["species"]] == ["Amanita muscaria", "Russula emetica"]
    assert left == 2 and got["genus"] == answer["genus"]


def test_visions_full_answer_counts_ten_deep_even_when_the_method_had_fewer_names():
    stored = {"species": [{"name": f"S s{i}"} for i in range(5)], "genus": [], "family": []}
    full = {"ranks": {"species": [{"name": f"S s{i}"} for i in range(7)],
                      "genus": [{"name": "S"}], "family": []}}
    got = external_report.deepest(stored, full)
    assert got["top_k"] == 10 and len(got["species"]) == 7
    assert external_report.deepest(stored, None) is stored


# --- a held-out split: JSONL first, then the import -----------------------------------------------

def bench_world(conn, tmp_path, monkeypatch, **freeze):
    root = reference(conn, tmp_path / "emb")
    frozen(conn, tmp_path, **freeze)
    fetched(conn, tmp_path, monkeypatch)
    return root


def test_predict_writes_jsonl_and_never_the_manifest(conn, tmp_path, monkeypatch, fake_model):
    bench_world(conn, tmp_path, monkeypatch)
    ids = heldout.benchmark_ids(conn, NAME, split="dev")
    before = conn.total_changes
    out = external.predict_heldout(conn, NAME, FAKE, ids, tmp_path / "fake.jsonl",
                                   classify=red_first, root=fake_model, log=lambda s: None)
    assert conn.total_changes == before
    lines = [json.loads(x) for x in (tmp_path / "fake.jsonl").read_text().splitlines()]
    assert {x["observation_id"] for x in lines} == {"101", "103", "105"}
    assert all(x["backbone"] == "external:fake-b384" and x["method"] == "mean-logits"
               for x in lines)
    assert lines[0]["result"]["species"][0]["name"] == "Russula emetica"
    assert out["reference_hash"] == external.checkpoint_id(FAKE, fake_model)
    # Logits are cached in the benchmark's own folder: a rerun classifies nothing.
    again = external.predict_heldout(conn, NAME, FAKE, ids, tmp_path / "again.jsonl",
                                     classify=lambda ims: pytest.fail("classified again"),
                                     root=fake_model, log=lambda s: None)
    assert again["already"] == out["classified"]


def predicted_external(conn, tmp_path, fake_model):
    ids = heldout.benchmark_ids(conn, NAME, split="dev")
    path = tmp_path / "fake.jsonl"
    external.predict_heldout(conn, NAME, FAKE, ids, path, classify=red_first, root=fake_model,
                             log=lambda s: None)
    return path


def test_import_stores_answers_and_a_run_keyed_by_the_checkpoint(conn, tmp_path, monkeypatch,
                                                                 fake_model):
    bench_world(conn, tmp_path, monkeypatch)
    path = predicted_external(conn, tmp_path, fake_model)
    out = heldout.import_external(conn, NAME, path, "external:fake-b384")
    ref = external.checkpoint_id(FAKE, fake_model)
    assert (out["stored"], out["reference_hash"]) == (3, ref)
    run = conn.execute("select method, place, size, reference_hash from heldout_runs where "
                       "backbone = 'external:fake-b384'").fetchall()
    assert [tuple(r) for r in run] == [("mean-logits", "-", "large", ref)]
    stored = conn.execute("select count(*) from heldout_predictions where backbone = ? and "
                          "reference_hash = ?", ("external:fake-b384", ref)).fetchone()[0]
    assert stored == 3
    assert heldout.import_external(conn, NAME, path, "external:fake-b384")["already"] == 3


@pytest.mark.parametrize("change, message", [
    (lambda r: {**r, "backbone": "external:other"}, "not 'external:fake-b384'"),
    (lambda r: {**r, "observation_id": "999"}, "not a record of"),
    (lambda r: {**r, "result": {**r["result"], "species": [{"name": "X"}]}}, "name and confidence"),
    (lambda r: {**r, "reference_hash": "another"}, "one method, size and checkpoint"),
])
def test_import_refuses_answers_that_are_not_this_models_on_this_set(conn, tmp_path, monkeypatch,
                                                                     fake_model, change, message):
    bench_world(conn, tmp_path, monkeypatch)
    path = predicted_external(conn, tmp_path, fake_model)
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    lines[-1] = change(lines[-1])
    path.write_text("\n".join(json.dumps(x) for x in lines))
    with pytest.raises(ValueError, match=message):
        heldout.import_external(conn, NAME, path, "external:fake-b384")
    assert conn.execute("select count(*) from heldout_predictions where backbone like "
                        "'external:fake%'").fetchone()[0] == 0


def test_import_takes_only_external_backbones(conn, tmp_path, monkeypatch, fake_model):
    bench_world(conn, tmp_path, monkeypatch)
    path = predicted_external(conn, tmp_path, fake_model)
    with pytest.raises(ValueError, match="not an external model"):
        heldout.import_external(conn, NAME, path, "bioclip-2")


def test_the_heldout_report_scores_an_imported_model_but_never_takes_its_run_as_the_reference(
        conn, tmp_path, monkeypatch, fake_model):
    root = bench_world(conn, tmp_path, monkeypatch)
    predicted(conn, tmp_path, root)
    heldout.import_external(conn, NAME, predicted_external(conn, tmp_path, fake_model),
                            "external:fake-b384")        # newer than Vision's run
    out = heldout_report.report(conn, NAME, split="dev", out_dir=tmp_path / "reports",
                                log=lambda s: None)
    assert out["reference"]["backbone"] == "toy"
    # dev: Russula emetica named right; Tubaria hiemalis is outside its vocabulary.
    assert out["models"]["external:fake-b384/mean-logits"]["species"]["top1"]["rate"] == 0.5
    assert "external:fake-b384/mean-logits" in out["summary"]["models"]
    depth = out["summary"]["models"]["external:fake-b384/mean-logits"][
        "species_by_true_species_reference_records"]
    assert depth["1-4"]["n"] == 1            # Russula emetica: 2 reference records in toy


def test_the_protocol_report_restricts_vision_to_the_models_vocabulary(conn, tmp_path,
                                                                       monkeypatch, fake_model):
    root = bench_world(conn, tmp_path, monkeypatch)
    predicted(conn, tmp_path, root)
    path = predicted_external(conn, tmp_path, fake_model)
    before = conn.total_changes
    out = external_report.report(conn, NAME, split="dev", results_files=[path],
                                 vision_backbone="toy")
    assert conn.total_changes == before
    m = out["models"]["external:fake-b384"]
    # dev: 101 Russula emetica (in vocabulary), 103 'Russula' (one word), 105 Tubaria
    # hiemalis (formal, not in vocabulary).
    split = m["coverage"]["split"]["species"]
    assert split["formal, in vocabulary"]["records"] == 1
    assert split["formal, not in vocabulary"]["records"] == 1
    assert split["one word, no species"]["records"] == 1
    assert m["coverage"]["north_american_records"]["species"]["formal, in vocabulary"][
        "records"] == 3
    assert m["formal_species"]["rows"]["species strict"]["n"] == 2
    same = m["same_vocabulary"]
    assert same["records"] == 1
    assert set(same["models"]) == {"external:fake-b384", "toy/nearest",
                                   "toy/nearest restricted to fake-b384"}
    restricted = same["models"]["toy/nearest restricted to fake-b384"]
    assert restricted["rows"]["species strict"]["top1"]["rate"] == 1.0
    assert "genus strict" not in restricted["rows"]
    assert m["formal_species"]["per_image_top1"]["n"] == 2
    assert "toy/nearest" in out["vision_models"]
    assert "(iii) same vocabulary" in external_report.format_report(out)


def test_the_protocol_report_never_scores_a_sealed_test_split(conn, tmp_path, monkeypatch,
                                                              fake_model):
    bench_world(conn, tmp_path, monkeypatch, holdout=True)
    with pytest.raises(ValueError, match="sealed benchmark's test split"):
        external_report.report(conn, NAME, split="test", vision_backbone="toy")


# --- a saved scoreboard comparison ---------------------------------------------------------------

def test_the_standard_block_counts_top_1_3_5_10_by_the_true_species_band():
    recs = [Record("1", "A x", "A", "F", None, None), Record("2", "B y", "B", "F", None, None)]
    answers = {"1": {"all": {"species": [{"name": "A x"}], "genus": [{"name": "A"}],
                             "family": [{"name": "F"}]}},
               "2": {"all": {"species": [{"name": f"Z z{i}"} for i in range(4)]
                             + [{"name": "B y"}], "genus": [], "family": []}}}
    out = external.score_records(recs, answers, {"A x": 3, "B y": 0})
    std = out["standard"]
    assert std["k"] == [1, 3, 5, 10] and std["bands"] == ["0", "1-4", "5-19", "20-99", "100+"]
    assert std["species"]["all"] == {"n": 2, "top1": 0.5, "top3": 0.5, "top5": 1.0, "top10": 1.0}
    assert std["species"]["1-4"]["top1"] == 1.0 and std["species"]["0"]["top1"] == 0.0
    assert out["species"]["all"]["top1"] == 0.5


def comparison(conn, tmp_path, monkeypatch):
    store = seed_two_species(conn, tmp_path)
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, tmp_path / "emb" / "m1", log=lambda s: None)
    monkeypatch.setattr(external, "reference_photo_inputs", lambda conn, pids, stores=None: [
        (p, lambda p=p: (store.root / f"p/{p}.png").read_bytes()) for p in pids])
    return evaluate.compare(conn, ["m1"], embeddings_root=tmp_path / "emb", log=lambda s: None)


TWO = ExternalModel("two-b384", "BVRA/two_b384", "vit_two", 2, "Two", "two.csv", "class_id")


def install_two(tmp_path, monkeypatch):
    root = tmp_path / "data" / "external"
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    monkeypatch.setitem(external.MODELS, TWO.short, TWO)
    TWO.folder(root).mkdir(parents=True)
    (TWO.folder(root) / "pytorch_model.bin").write_bytes(b"w")
    write_metadata(root / "metadata" / TWO.metadata,
                   classes=[("A x L.", "A x", "F"), ("B y L.", "B y", "F")])
    external.write_labels(TWO, root)
    return root


def bright_is_a(images):
    return np.asarray([[4.0, 0.0] if np.asarray(im)[..., 0].mean() > 128 else [0.0, 4.0]
                       for im in images], dtype=np.float32)


def test_the_baseline_joins_a_comparison_with_the_standard_block_and_replaces_its_row(
        conn, tmp_path, monkeypatch):
    cmp = comparison(conn, tmp_path, monkeypatch)
    root = install_two(tmp_path, monkeypatch)
    for _ in range(2):
        out = external.run_comparison(conn, cmp["comparison_id"], TWO, classify=bright_is_a,
                                      embeddings_root=tmp_path / "emb", root=root,
                                      log=lambda s: None)
    rows = [r for r in evaluate.scoreboard(conn, cmp["comparison_id"])
            if r["backbone"] == "external:two-b384"]
    assert len(rows) == 1 and rows[0]["method"] == "mean-logits"
    assert rows[0]["species_top1"] == 1.0 and out["answered"] == cmp["test_records"]
    report = evaluate.run_report(conn, rows[0]["id"])
    assert report["all_photos"]["standard"]["k"] == [1, 3, 5, 10]
    assert report["external"]["licence"] == "CC-BY-NC-4.0"
    assert report["external"]["photo_only"] is True


def test_the_baseline_refuses_a_comparison_whose_records_have_changed(conn, tmp_path,
                                                                     monkeypatch):
    cmp = comparison(conn, tmp_path, monkeypatch)
    root = install_two(tmp_path, monkeypatch)
    conn.execute("update eval_runs set record_set = 'stale'")
    conn.commit()
    with pytest.raises(RuntimeError, match="changed since it ran"):
        external.run_comparison(conn, cmp["comparison_id"], TWO, classify=bright_is_a,
                                embeddings_root=tmp_path / "emb", root=root,
                                log=lambda s: None)


# --- the command line ------------------------------------------------------------------------------

@pytest.mark.parametrize("action", ["coverage", "predict", "report"])
def test_reading_external_commands_open_the_manifest_read_only(tmp_path, monkeypatch, action):
    from mycomap_vision import manifest
    path = tmp_path / "manifest.sqlite"
    manifest.connect(path).close()
    monkeypatch.setattr(config, "MANIFEST_PATH", path)
    seen = {}

    def spy(conn, args):
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            conn.execute("create table x (y)")
        seen["ok"] = True
    monkeypatch.setattr(cli, "cmd_external", spy)
    argv = {"coverage": ["external", "coverage", "--name", "b"],
            "predict": ["external", "predict", "--name", "b", "--split", "dev"],
            "report": ["external", "report", "--name", "b"]}[action]
    cli.main(argv)
    assert seen == {"ok": True}


def test_model_names_resolve_by_short_name_backbone_or_hugging_face_id():
    m = external.MODELS["df20-vit-l384"]
    for name in ("df20-vit-l384", "external:df20-vit-l384", "BVRA/vit_large_patch16_384.ft_df20_384",
                 "hf-hub:BVRA/vit_large_patch16_384.ft_df20_384"):
        assert external.model_for(name) is m
    with pytest.raises(ValueError, match="unknown external model"):
        external.model_for("BVRA/nothing")
