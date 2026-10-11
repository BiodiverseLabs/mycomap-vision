"""A one-word name ("Russula", "Agaricales") is never a species (Steve, 2026-09-30): no
species label, no species class, never a species candidate, not scored at species; it
still counts at genus when the word is a genus, and at family."""

import io
import json

import numpy as np
from PIL import Image

from conftest import inat_obs
from test_models_and_scoreboard import OPEN, Const
from test_taxonomy import FakeInat, client

from mycomap_vision import evaluate, finetune, prospective, taxonomy
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.identify import Identifier
from mycomap_vision.inat import save_batch
from mycomap_vision.inat_cv import Truth, score_records
from mycomap_vision.photos import Result, save_result
from mycomap_vision.prior import RangeSeasonPrior
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

# (name on .org, genus column, family column, red of its photo, validated)
RECORDS = [
    ("Russula emetica", "Russula", "Russulaceae", 250, "1/1/2026"),
    ("Russula emetica", "Russula", "Russulaceae", 245, "1/2/2026"),
    ("Russula", "Russula", "Russulaceae", 170, "1/3/2026"),          # one word: a genus
    ("Amanita muscaria", "Amanita", "Amanitaceae", 20, "1/1/2026"),
    ("Amanita muscaria", "Amanita", "Amanitaceae", 25, "1/2/2026"),
    ("Cortinariaceae", "", "Cortinariaceae", 100, "1/2/2026"),       # one word: a family
    ("Agaricales", "", "", 130, "1/2/2026"),                         # one word: an order
    ("Russula", "Russula", "Russulaceae", 172, "9/20/2026"),        # test records: the
    ("Amanita muscaria", "Amanita", "Amanitaceae", 22, "9/21/2026"),  # newest weeks
]
PHOTOS = {1000 + i: i for i in range(len(RECORDS))}


def seed(conn, tmp_path, records=RECORDS):
    rows, obs = [], []
    for i, (name, genus, family, _red, when) in enumerate(records):
        rows.append({"source": "iNaturalist", "observation_id": str(100 + i), "scientific_name": name, "genus": genus,
                     "family": family, "continent": "North America",
                     "validation_status_1": "yes", "validation_date_1": when})
        obs.append(inat_obs(100 + i, photos=[(0, 1000 + i, "cc0", OPEN)]))
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, [o["observation_id"] for o in rows], obs, "t")
    store = LocalStore(tmp_path / "store")
    for i, (*_x, red, _when) in enumerate(records):
        buf = io.BytesIO()
        Image.new("RGB", (8, 8), (red, 0, 0)).save(buf, "PNG")
        store.put(f"p/{1000 + i}.png", buf.getvalue())
        save_result(conn, Result(1000 + i, "done", f"p/{1000 + i}.png", 1, "h"), "large", "now",
                    store.location)
    root = tmp_path / "emb"
    todo = photos_to_embed(conn, "m1", "large", store.location, "local")
    embed_photos(conn, store, Const("m1"), todo, root / "m1", log=lambda s: None)
    conn.commit()
    return store, root


def loaded(conn):
    return {r.observation_id: r for r in evaluate.load_records(conn, {p: p for p in PHOTOS})}


def test_a_one_word_genus_has_no_species_label_and_keeps_its_genus_and_family(conn, tmp_path):
    seed(conn, tmp_path)
    recs = loaded(conn)
    russula = recs["102"]
    assert (russula.species, russula.genus, russula.family) == ("", "Russula", "Russulaceae")
    assert russula.unit == "Russula"
    assert (recs["100"].species, recs["100"].unit) == ("Russula emetica", "Russula emetica")


def test_a_one_word_name_above_genus_has_no_genus_label(conn, tmp_path):
    seed(conn, tmp_path)
    recs = loaded(conn)
    cort = recs["105"]
    assert (cort.species, cort.genus, cort.family) == ("", "", "Cortinariaceae")
    assert "106" not in recs            # Agaricales: no label at any rank Vision scores


def test_inat_says_which_one_word_names_are_genera(conn, tmp_path):
    # .org's genus column says "Agaricales"; iNat has it as an order, so it is no genus,
    # and its family comes from its own place in iNat's tree (an order has none).
    records = [("Agaricales", "Agaricales", "Agaricaceae", 130, "1/2/2026"),
               ("Cortinariaceae", "Cortinariaceae", "", 100, "1/2/2026"),
               ("Russula", "", "", 170, "1/3/2026")]
    seed(conn, tmp_path, records)
    taxonomy.fetch(conn, tmp_path / taxonomy.CACHE, client(FakeInat()), log=lambda s: None)
    recs = {r.observation_id: r for r in evaluate.load_records(conn, {1000: 0, 1001: 1, 1002: 2})}
    assert "100" not in recs
    assert (recs["101"].genus, recs["101"].family) == ("", "Cortinariaceae")
    # iNat has Russula as a genus, though .org's genus column is blank.
    assert (recs["102"].species, recs["102"].genus, recs["102"].family) == (
        "", "Russula", "Russulaceae")
    tax = taxonomy.load(tmp_path / taxonomy.CACHE)
    assert taxonomy.labels_for("Agaricales", "Agaricales", "", False, tax).genus == ""
    assert taxonomy.labels_for("Fungi", "", "", False, None) == taxonomy.Labels("", "", "", "Fungi")


def test_a_one_word_name_adds_no_species_class(conn, tmp_path):
    store, root = seed(conn, tmp_path)
    recs = evaluate.load_records(conn, {p: p for p in PHOTOS})
    index = evaluate.build_index(evaluate.with_rows(recs, PHOTOS))
    assert "Russula" not in index.labels["species"]
    assert "Cortinariaceae" not in index.labels["species"]
    assert index.labels["species"] == ["Amanita muscaria", "Russula emetica"]
    assert "Russula" in index.labels["genus"]
    ts = finetune.build_trainset(conn, "m1", store.location, "large", embeddings_root=root)
    assert ts.names["species"] == ["Amanita muscaria", "Russula emetica"]
    one = [i for i, (pid, _) in enumerate(ts.items) if pid == 1002][0]
    species, genus, family = ts.labels[one]
    assert species == -1 and ts.names["genus"][genus] == "Russula"
    assert ts.names["family"][family] == "Russulaceae"
    # Still drawn for training (at genus and family), not left out for want of a species.
    assert finetune.sampling_weights(ts.units, 0.5)[one] > 0


def test_a_one_word_name_is_never_a_species_candidate(conn, tmp_path):
    _store, root = seed(conn, tmp_path)
    ident = Identifier(conn, "m1", "nearest", embeddings_root=root / "m1",
                       model_cache=tmp_path / "models")
    # A photo just like the genus-only Russula record's.
    query = Const("m1").encode([Image.new("RGB", (8, 8), (170, 0, 0))])
    query = query / np.linalg.norm(query, axis=1, keepdims=True)
    out = ident.identify_vectors(query, top_k=10)
    species = [c["name"] for c in out["ranks"]["species"]]
    assert species and "Russula" not in species and "Cortinariaceae" not in species
    assert out["ranks"]["genus"][0]["name"] == "Russula"
    top = out["specimens"][0]
    assert (top["name"], top["species"], top["genus"], top["species_url"]) == (
        "Russula", "", "Russula", None)
    for p in out["per_photo"]:
        assert "Russula" not in [c["name"] for c in p["ranks"]["species"]]


def test_a_one_word_test_record_is_not_scored_at_species_but_counts_at_genus(conn, tmp_path):
    _store, root = seed(conn, tmp_path)
    result = evaluate.compare(conn, ["m1"], embeddings_root=root, log=lambda s: None)
    assert result["test_records"] == 2
    [run] = result["runs"]
    assert run["all_photos"]["species"]["all"]["n"] == 1       # Amanita muscaria only
    assert run["all_photos"]["genus"]["all"]["n"] == 2         # ...and Russula
    assert run["all_photos"]["genus"]["all"]["top1"] == 1.0
    assert run["all_photos"]["calibration"]["species"]["n"] == 1


def test_the_range_score_puts_a_one_word_record_in_its_own_group(conn, tmp_path):
    seed(conn, tmp_path)
    recs = evaluate.load_records(conn, {p: p for p in PHOTOS})
    index = evaluate.build_index(evaluate.with_rows(recs, PHOTOS))
    prior = RangeSeasonPrior()
    prior.fit(recs, index.species)
    g = prior.genus_of_species
    unit = index.species.index
    assert g[unit("Russula")] == g[unit("Russula emetica")] != g[unit("Amanita muscaria")]


def test_an_advance_prediction_of_a_one_word_record_is_checked_at_genus_only(conn, tmp_path):
    seed(conn, tmp_path)
    conn.executescript(prospective.SCHEMA)
    result = {"species": [{"name": "Russula emetica", "confidence": 0.9}],
              "genus": [{"name": "Russula", "confidence": 0.95}],
              "family": [{"name": "Russulaceae", "confidence": 0.99}]}
    conn.execute("insert into predictions values ('102', 'm1', 'nearest', '2025-12-01', 'x', "
                 "5, 1, ?)", (json.dumps(result),))
    conn.commit()
    [row] = prospective.report(conn)
    assert row["resolved"] == 1
    assert (row["species_top1"], row["genus_top1"], row["family_top1"]) == (None, 1.0, 1.0)
    assert row["mean_species_confidence"] is None


def test_inats_baseline_skips_species_for_a_one_word_record():
    rec = evaluate.Record("1", "", "Russula", "Russulaceae", None, None, [], taxon="Russula")
    photos = {"1": [{"species": {7: {"name": "Russula emetica", "vision": 0.9, "combined": 0.9}},
                     "genus": {5: {"name": "Russula", "vision": 0.9, "combined": 0.9}},
                     "family": {3: {"name": "Russulaceae", "vision": 0.9, "combined": 0.9}}}]}
    out = score_records([rec], photos, {"1": Truth(None, 5, 3, False)}, {}, "vision")
    assert out["species"] == {}
    assert out["genus"]["all"] == {"n": 1, "top1": 1.0, "top5": 1.0}
