"""Spellings of one name are one species wherever Vision turns a name into a label;
names a person has to decide stay apart; the manifest keeps the name as .org has it."""

import io
import json
import sqlite3

from fastapi.testclient import TestClient
from PIL import Image

from conftest import inat_obs
from test_api import open_limits
from test_models_and_scoreboard import OPEN, Const

from mycomap_vision import cli, config, evaluate, finetune
from mycomap_vision.api import create_app
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.identify import Identifier
from mycomap_vision.inat import save_batch
from mycomap_vision.photos import Result, save_result
from mycomap_vision.records import build_records, save_records
from mycomap_vision.storage import LocalStore

HOUSE = "Inocybe sp. 'PNW18'"
OLD = "Inocybe PNW18"
OLDER = 'Inocybe "sp-PNW18"'

# (name on .org, genus column, red of its photo, validated)
RECORDS = [
    (HOUSE, "Inocybe", 250, "1/1/2026"),
    (HOUSE, "Inocybe", 240, "1/2/2026"),
    (OLDER, "", 245, "1/3/2026"),
    ("Mycena sp. 'IN7'", "Mycena", 10, "1/1/2026"),          # IN7 / IN07: a person decides
    ("Mycena sp. 'IN07'", "Mycena", 120, "1/2/2026"),
    ("Craterellus sp. 'neotubaeformis'", "Craterellus", 60, "1/1/2026"),   # a code that is
    ("Craterellus neotubaeformis", "Craterellus", 180, "1/2/2026"),        # also a plain name
    (OLD, "An old genus", 248, "9/20/2026"),                 # test records: the newest weeks
    ("Mycena sp. 'IN7'", "Mycena", 12, "9/21/2026"),
]
PHOTOS = {1000 + i: i for i in range(len(RECORDS))}


def seed(conn, tmp_path):
    """A manifest with its photos embedded by the stand-in backbone m1."""
    rows, obs = [], []
    for i, (name, genus, _red, when) in enumerate(RECORDS):
        oid = str(100 + i)
        rows.append({"source": "iNaturalist", "observation_id": oid, "scientific_name": name, "genus": genus,
                     "family": "F", "continent": "North America",
                     "validation_status_1": "yes", "validation_date_1": when})
        obs.append(inat_obs(100 + i, photos=[(0, 1000 + i, "cc0", OPEN)]))
    save_records(conn, build_records(rows, "t"))
    save_batch(conn, [o["observation_id"] for o in rows], obs, "t")
    store = LocalStore(tmp_path / "store")
    for i, (_name, _genus, red, _when) in enumerate(RECORDS):
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


def stored_names(conn):
    return conn.execute("select observation_id, scientific_name, genus from records "
                        "order by 1").fetchall()


def test_two_spellings_of_one_code_are_one_species_in_the_records_and_the_index(conn, tmp_path):
    seed(conn, tmp_path)
    records = evaluate.load_records(conn, {p: p for p in PHOTOS})
    inocybe = [r for r in records if r.genus == "Inocybe"]
    assert len(inocybe) == 4
    assert {r.species for r in inocybe} == {HOUSE}
    index = evaluate.build_index(evaluate.with_rows(records, PHOTOS))
    assert [s for s in index.species if s.startswith("Inocybe")] == [HOUSE]
    assert index.ref_count[HOUSE] == 4


def test_a_merged_label_names_its_own_genus_and_remembers_the_stored_spelling(conn, tmp_path):
    seed(conn, tmp_path)
    by_id = {r.observation_id: r for r in evaluate.load_records(conn, {p: p for p in PHOTOS})}
    assert (by_id["107"].genus, by_id["107"].stored_name) == ("Inocybe", OLD)     # not the column
    assert (by_id["102"].genus, by_id["102"].stored_name) == ("Inocybe", OLDER)
    assert (by_id["100"].genus, by_id["100"].stored_name) == ("Inocybe", "")


def test_names_a_person_has_to_decide_stay_separate_species(conn, tmp_path):
    seed(conn, tmp_path)
    records = evaluate.load_records(conn, {p: p for p in PHOTOS})
    species = {r.species for r in records}
    assert {"Mycena sp. 'IN7'", "Mycena sp. 'IN07'", "Craterellus sp. 'neotubaeformis'",
            "Craterellus neotubaeformis"} <= species
    assert len(species) == 5
    index = evaluate.build_index(evaluate.with_rows(records, PHOTOS))
    assert index.ref_count["Mycena sp. 'IN7'"] == 2 and index.ref_count["Mycena sp. 'IN07'"] == 1


def test_the_manifest_keeps_every_name_as_it_is_stored(conn, tmp_path):
    _store, root = seed(conn, tmp_path)
    before = stored_names(conn)
    evaluate.load_records(conn, {p: p for p in PHOTOS})
    evaluate.compare(conn, ["m1"], embeddings_root=root, log=lambda s: None)
    Identifier(conn, "m1", "nearest", embeddings_root=root / "m1",
               model_cache=tmp_path / "models")
    assert stored_names(conn) == before
    assert [tuple(r)[1] for r in before][7] == OLD


def test_a_test_record_spelled_the_old_way_is_right_against_the_house_spelling(conn, tmp_path):
    _store, root = seed(conn, tmp_path)
    result = evaluate.compare(conn, ["m1"], embeddings_root=root, log=lambda s: None)
    assert (result["test_records"], result["reference_records"]) == (2, 7)
    [run] = result["runs"]
    species = run["all_photos"]["species"]
    assert species["all"] == {"n": 2, "top1": 1.0, "top5": 1.0}
    assert "novel (0 refs)" not in species            # the old spelling has reference records
    assert species["3-5 refs"]["n"] == 1              # ...all three of them, of both spellings


def test_training_and_the_identifier_see_one_species_per_name(conn, tmp_path):
    store, root = seed(conn, tmp_path)
    ts = finetune.build_trainset(conn, "m1", store.location, "large", embeddings_root=root)
    assert [s for s in ts.names["species"] if s.startswith("Inocybe")] == [HOUSE]
    ident = Identifier(conn, "m1", "nearest", embeddings_root=root / "m1",
                       model_cache=tmp_path / "models")
    assert [s for s in ident.index.species if s.startswith("Inocybe")] == [HOUSE]
    assert ident.rank_counts["species"][HOUSE] == 4
    assert ident.rank_counts["species"]["Mycena sp. 'IN07'"] == 1


def test_the_data_page_counts_spellings_of_one_name_once(conn, tmp_path):
    _store, root = seed(conn, tmp_path)
    client = TestClient(create_app(tmp_path / "manifest.sqlite", root,
                                   backbone_loader=lambda name: Const(name),
                                   limits=open_limits(), background=False))
    stats = client.get("/api/stats").json()
    assert stats["names"] == 5
    assert stats["names_by_records"] == {"1": 3, "2": 1, "3-5": 1, "6-30": 0, "31+": 0}


# --- mv name-spellings ------------------------------------------------------------------

def run_cli(monkeypatch, capsys, path, *args):
    monkeypatch.setattr(config, "MANIFEST_PATH", path)
    assert cli.main(["name-spellings", *args]) == 0
    return capsys.readouterr().out


def test_name_spellings_says_what_is_merged_and_lists_what_a_person_decides(
        conn, tmp_path, monkeypatch, capsys):
    seed(conn, tmp_path)
    out = run_cli(monkeypatch, capsys, tmp_path / "manifest.sqlite")
    assert "7 names in the manifest are 5 labels in Vision" in out
    assert "3 names need a fix on mycomap.org (6 records)" in out
    assert "merged in Vision (only the writing differs): 1 names, 3 spellings, 4 records " \
           "(2 re-labelled)" in out
    assert "left for a person, kept as separate labels: 2 names, 4 spellings, 5 records" in out
    listed = out.split("left for a person")[1]
    assert "mycena sp. 'in7'  [zero_padding]  proposed: Mycena sp. 'IN07'" in listed
    assert "      2  Mycena sp. 'IN7'" in listed and "      1  Mycena sp. 'IN07'" in listed
    assert "craterellus sp. 'neotubaeformis'  [name_or_code]  a person chooses" in listed
    assert "      1  Craterellus neotubaeformis" in listed
    assert "Inocybe" not in listed                    # merged names are counted, not listed


def test_name_spellings_json_is_the_full_list_and_nothing_else(conn, tmp_path, monkeypatch,
                                                                capsys):
    seed(conn, tmp_path)
    out = json.loads(run_cli(monkeypatch, capsys, tmp_path / "manifest.sqlite", "--json"))
    assert out["summary"]["merged_in_vision"]["groups"] == 1
    assert len(out["groups"]) == 3
    merged = next(g for g in out["groups"] if g["confidence"] == "same")
    assert merged["proposed_name"] == HOUSE and merged["records_to_fix"] == 2
    assert {s["name"]: s["by_source"] for s in merged["spellings"]} == {
        HOUSE: {"inat": 2}, OLD: {"inat": 1}, OLDER: {"inat": 1}}


def test_name_spellings_shows_a_space_that_cannot_be_seen(tmp_path, monkeypatch, capsys):
    path = tmp_path / "bare.sqlite"
    bare = sqlite3.connect(path)
    bare.execute("create table records (observation_id text, source text, scientific_name text)")
    bare.executemany("insert into records values (?, 'inat', ?)",
                     [("1", "Mycena sp. 'IN7'"), ("2", "Mycena\xa0sp. 'IN07'")])
    bare.commit()
    bare.close()
    out = run_cli(monkeypatch, capsys, path)
    assert "Mycena\\xa0sp. 'IN07'" in out


def test_name_spellings_leaves_the_manifest_exactly_as_it_found_it(tmp_path, monkeypatch,
                                                                    capsys):
    # A manifest holding nothing but records: any other command would add its tables.
    path = tmp_path / "bare.sqlite"
    bare = sqlite3.connect(path)
    bare.execute("create table records (observation_id text, source text, scientific_name text)")
    bare.executemany("insert into records values (?, 'inat', ?)", [("1", HOUSE), ("2", OLD)])
    bare.commit()
    bare.close()
    before = path.read_bytes()
    out = run_cli(monkeypatch, capsys, path)
    assert "2 names in the manifest are 1 labels in Vision" in out
    assert path.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bare.sqlite"]
