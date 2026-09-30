"""Family comes from iNaturalist per genus: one answer per genus, within Fungi only;
a genus in doubt is listed for a person and keeps .org's family. No real network:
iNat is a stand-in session."""

import csv

import pytest

from mycomap_vision import config, evaluate, taxonomy
from mycomap_vision.records import build_records, save_records

LIFE, FUNGI, PLANTAE = 48460, 47170, 47126

# id: (name, rank, rank_level, ancestor ids above it, active, current ids)
TAXA = {
    LIFE: ("Life", "stateofmatter", 100, [], True, []),
    FUNGI: ("Fungi", "kingdom", 70, [LIFE], True, []),
    PLANTAE: ("Plantae", "kingdom", 70, [LIFE], True, []),
    47169: ("Basidiomycota", "phylum", 60, [LIFE, FUNGI], True, []),
    50814: ("Agaricomycetes", "class", 50, [LIFE, FUNGI, 47169], True, []),
    48341: ("Russulales", "order", 40, [LIFE, FUNGI, 47169, 50814], True, []),
    48340: ("Russulaceae", "family", 30, [LIFE, FUNGI, 47169, 50814, 48341], True, []),
    48339: ("Russula", "genus", 20, [LIFE, FUNGI, 47169, 50814, 48341, 48340], True, []),
    47167: ("Agaricales", "order", 40, [LIFE, FUNGI, 47169, 50814], True, []),
    47166: ("Entolomataceae", "family", 30, [LIFE, FUNGI, 47169, 50814, 47167], True, []),
    47165: ("Entoloma", "genus", 20, [LIFE, FUNGI, 47169, 50814, 47167, 47166], True, []),
    205894: ("Nolanea", "genus", 20, [LIFE, FUNGI, 47169, 50814, 47167, 47166], False, [47165]),
    50001: ("Hygrophoraceae", "family", 30, [LIFE, FUNGI, 47169, 50814, 47167], True, []),
    50002: ("Hygrocybe", "genus", 20, [LIFE, FUNGI, 47169, 50814, 47167, 50001], True, []),
    50003: ("Cortinariaceae", "family", 30, [LIFE, FUNGI, 47169, 50814, 47167], True, []),
    # A fungal genus and a plant genus with one name; a plant genus with no fungal twin.
    60001: ("Homonyma", "genus", 20, [LIFE, FUNGI, 47169, 50814, 47167, 47166], True, []),
    60002: ("Homonyma", "genus", 20, [LIFE, PLANTAE], True, []),
    60003: ("Plantonly", "genus", 20, [LIFE, PLANTAE], True, []),
    # Two fungal genera of one name.
    60004: ("Duplex", "genus", 20, [LIFE, FUNGI, 47169, 50814, 47167, 47166], True, []),
    60005: ("Duplex", "genus", 20, [LIFE, FUNGI, 47169, 50814, 48341, 48340], True, []),
}


def taxon(tid):
    name, rank, level, above, active, current = TAXA[tid]
    return {"id": tid, "name": name, "rank": rank, "rank_level": level, "is_active": active,
            "ancestor_ids": above + [tid], "current_synonymous_taxon_ids": current,
            "provisional": False}


class Response:
    def __init__(self, body, status=200, headers=None):
        self.body, self.status_code, self.headers = body, status, headers or {}
        self.text = str(body)

    def json(self):
        return self.body


class FakeInat:
    """Answers /v1/taxa searches and /v1/taxa/<ids> from TAXA; records every call."""

    def __init__(self, fail_first=0):
        self.headers = {}
        self.calls = []
        self.fail_first = fail_first

    def get(self, url, params=None, timeout=None):
        assert url.startswith(taxonomy.API), url            # nothing else is ever asked
        self.calls.append((url, dict(params or {})))
        if self.fail_first:
            self.fail_first -= 1
            return Response({}, 429, {"Retry-After": "5"})
        path = url[len(taxonomy.API):]
        if path == "/taxa":
            p = params or {}
            active = p.get("is_active", "true") != "false"
            q = p["q"].lower()
            hits = [taxon(t) for t, v in TAXA.items()
                    if v[0].lower() == q and v[4] == active
                    and (p.get("rank") is None or v[1] == p["rank"])]
            return Response({"results": hits})
        ids = [int(x) for x in path.split("/")[2].split(",")]
        return Response({"results": [taxon(i) for i in ids if i in TAXA]})

    def searched(self):
        return [p["q"] for u, p in self.calls if u.endswith("/taxa")]


def client(fake):
    return taxonomy.InatTaxa(session=fake, sleep=lambda s: None)


def seed(conn, rows):
    """rows: (name, genus column, family column)."""
    save_records(conn, build_records(
        [{"observation_id": str(100 + i), "scientific_name": name, "genus": genus,
          "family": family, "continent": "North America", "validation_status_1": "yes",
          "validation_date_1": "1/1/2026"} for i, (name, genus, family) in enumerate(rows)],
        "t"))


def fetch(conn, tmp_path, fake=None, **kw):
    fake = fake or FakeInat()
    out = taxonomy.fetch(conn, tmp_path / taxonomy.CACHE, client(fake), log=lambda s: None, **kw)
    return fake, out


def answers(tmp_path):
    return taxonomy.load(tmp_path / taxonomy.CACHE).answers


def test_a_genus_of_the_same_name_outside_fungi_is_ignored(conn, tmp_path):
    seed(conn, [("Homonyma alba", "Homonyma", ""), ("Plantonly nigra", "Plantonly", "")])
    fetch(conn, tmp_path)
    a = answers(tmp_path)
    assert a[("Homonyma", "genus")][:3] == ("ok", None, 60001)      # the fungus, not the plant
    assert a[("Homonyma", "genus")][4] == 1                         # one homonym, ignored
    assert a[("Plantonly", "genus")][:2] == ("doubt", taxonomy.NOT_FOUND)
    tax = taxonomy.load(tmp_path / taxonomy.CACHE)
    assert tax.genera["Homonyma"]["family"] == "Entolomataceae"
    assert "Plantonly" not in tax.genera


def test_two_fungal_genera_of_one_name_are_in_doubt_and_not_applied(conn, tmp_path):
    seed(conn, [("Duplex ambigua", "Duplex", "Entolomataceae")])
    fetch(conn, tmp_path)
    status, reason, *_ = answers(tmp_path)[("Duplex", "genus")]
    assert (status, reason) == ("doubt", taxonomy.SEVERAL)
    [rec] = evaluate_records(conn)
    assert rec["family"] == "Entolomataceae"                         # .org's, not a guess


def test_an_inactive_genus_is_in_doubt_and_names_its_current_replacement(conn, tmp_path):
    seed(conn, [("Nolanea verna", "Nolanea", "")])
    fetch(conn, tmp_path)
    status, reason, tid, reps, _h = answers(tmp_path)[("Nolanea", "genus")]
    assert (status, reason, tid) == ("doubt", taxonomy.INACTIVE, 205894)
    out = taxonomy.report(conn, tmp_path / taxonomy.CACHE, tmp_path / "reports")
    [row] = read_csv(out["report"])
    assert (row["genus"], row["why"], row["inat_replacement"]) == (
        "Nolanea", taxonomy.INACTIVE, "Entoloma")
    assert "Nolanea" not in taxonomy.load(tmp_path / taxonomy.CACHE).genera


def test_a_name_that_is_not_a_latin_genus_is_listed_without_asking_inat(conn, tmp_path):
    seed(conn, [("Sistotrema brinkmannii", "Sistotrema5", ""), ("Unknown", "", "")])
    fake, _ = fetch(conn, tmp_path)
    assert fake.searched() == []
    a = answers(tmp_path)
    assert a[("Sistotrema5", "genus")][1] == a[("Unknown", "genus")][1] == taxonomy.NOT_LATIN


def test_a_rerun_asks_nothing_already_answered_and_refresh_asks_again(conn, tmp_path):
    seed(conn, [("Russula emetica", "Russula", ""), ("Nolanea verna", "Nolanea", "")])
    first, out = fetch(conn, tmp_path)
    assert out["asked"] == 2 and sorted(first.searched()) == ["Nolanea", "Nolanea", "Russula"]
    again, out = fetch(conn, tmp_path)
    assert out["asked"] == 0 and again.calls == []                  # every answer was kept
    recent, out = fetch(conn, tmp_path, older_than_days=30)
    assert out["asked"] == 0
    import sqlite3
    with sqlite3.connect(tmp_path / taxonomy.CACHE) as db:          # answered long ago
        db.execute("update answers set asked_at = '2026-01-01T00:00:00+00:00' "
                   "where name = 'Russula'")
    old, out = fetch(conn, tmp_path, older_than_days=30)
    assert out["asked"] == 1 and old.searched() == ["Russula"]
    refreshed, out = fetch(conn, tmp_path, refresh=True)
    assert out["asked"] == 2 and "Russula" in refreshed.searched()


def test_an_interrupted_run_resumes_where_it_stopped(conn, tmp_path):
    seed(conn, [("Russula emetica", "Russula", ""), ("Entoloma sericeum", "Entoloma", ""),
                ("Entoloma sinuatum", "Entoloma", "")])
    fetch(conn, tmp_path, limit=1)                                   # Entoloma: most records
    rest, out = fetch(conn, tmp_path)
    assert out["asked"] == 1 and rest.searched() == ["Russula"]


def test_inat_is_asked_politely_and_retried_after_429(conn, tmp_path):
    seed(conn, [("Russula emetica", "Russula", "")])
    slept = []
    fake = FakeInat(fail_first=2)
    c = taxonomy.InatTaxa(session=fake, sleep=slept.append)
    taxonomy.fetch(conn, tmp_path / taxonomy.CACHE, c, log=lambda s: None)
    assert fake.headers["User-Agent"] == config.USER_AGENT
    assert sum(slept) >= 30 + 60                                     # backed off, twice
    assert answers(tmp_path)[("Russula", "genus")][0] == "ok"
    # At most one request a second: every call waits its turn.
    paced = taxonomy.InatTaxa(session=FakeInat(), sleep=slept.append)
    slept.clear()
    for _ in range(3):
        paced.get("/taxa", {"q": "Russula", "rank": "genus"})
    assert paced.pace.interval == 1.0
    assert len(slept) == 2 and all(s >= 0.9 for s in slept)


def evaluate_records(conn):
    """Every record's labels as load_records gives them (all with one photo)."""
    from conftest import inat_obs
    from mycomap_vision.inat import save_batch
    ids = [r[0] for r in conn.execute("select observation_id from records order by 1")]
    save_batch(conn, ids, [inat_obs(int(i), photos=[(0, 5000 + int(i), "cc0", "h")])
                           for i in ids], "t")
    recs = evaluate.load_records(conn, {5000 + int(i): 0 for i in ids})
    return [{"id": r.observation_id, "species": r.species, "genus": r.genus,
             "family": r.family} for r in sorted(recs, key=lambda r: r.observation_id)]


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_inats_family_goes_to_every_record_of_a_genus_and_conflicts_are_listed(conn, tmp_path):
    seed(conn, [
        ("Russula emetica", "Russula", "Russulaceae"),
        ("Russula emetica", "Russula", "Russulaceae"),
        ("Russula nigricans", "Russula", "Amanitaceae"),       # a stray family on .org
        ("Russula rosea", "Russula", ""),                     # blank on .org
        ("Hygrocybe miniata", "Hygrocybe", "Tricholomataceae"),   # most records say
        ("Hygrocybe conica", "Hygrocybe", "Tricholomataceae"),    # another family
        ("Hygrocybe coccinea", "Hygrocybe", "Hygrophoraceae"),
    ])
    # Written beside the manifest, where load_records looks for it.
    fetch(conn, tmp_path)
    recs = evaluate_records(conn)
    assert {r["family"] for r in recs if r["genus"] == "Russula"} == {"Russulaceae"}
    assert {r["family"] for r in recs if r["genus"] == "Hygrocybe"} == {"Hygrophoraceae"}
    tax = taxonomy.load(tmp_path / taxonomy.CACHE)
    assert tax.genera["Russula"] == {"phylum": "Basidiomycota", "class": "Agaricomycetes",
                                     "order": "Russulales", "family": "Russulaceae"}
    out = taxonomy.report(conn, tmp_path / taxonomy.CACHE, tmp_path / "reports")
    rows = {r["genus"]: r for r in read_csv(out["report"])}
    assert set(rows) == {"Hygrocybe"}                        # Russula agrees with most records
    assert rows["Hygrocybe"]["why"] == taxonomy.DIFFERS
    assert rows["Hygrocybe"]["org_families"] == "Tricholomataceae 2; Hygrophoraceae 1"
    assert out["top_disagreements"][0]["genus"] == "Hygrocybe"
    assert out["in_doubt"] == {taxonomy.DIFFERS: 1}


def test_a_blank_family_is_filled_and_the_counts_say_so(conn, tmp_path):
    seed(conn, [("Russula rosea", "Russula", ""), ("Russula emetica", "Russula", "Russulaceae"),
                ("Russula nigricans", "Russula", "Amanitaceae"),
                ("Duplex ambigua", "Duplex", "")])
    fetch(conn, tmp_path)
    by_id = {r["id"]: r for r in evaluate_records(conn)}
    assert by_id["100"]["family"] == "Russulaceae"            # filled
    assert by_id["102"]["family"] == "Russulaceae"            # changed
    assert by_id["103"]["family"] == ""                       # in doubt: left as .org has it
    out = taxonomy.report(conn, tmp_path / taxonomy.CACHE, tmp_path / "reports")
    assert (out["records_family_filled"], out["records_family_changed"]) == (1, 1)
    assert (out["genus_names"], out["asked"], out["applied"]) == (2, 2, 1)


def test_without_a_cache_orgs_family_is_kept(conn, tmp_path):
    seed(conn, [("Russula nigricans", "Russula", "Amanitaceae")])
    assert evaluate_records(conn)[0]["family"] == "Amanitaceae"


def test_the_cache_is_its_own_file_and_the_manifest_is_untouched(conn, tmp_path):
    seed(conn, [("Russula emetica", "Russula", "")])
    conn.commit()
    tables = {r[0] for r in conn.execute("select name from sqlite_master")}
    fetch(conn, tmp_path)
    assert {r[0] for r in conn.execute("select name from sqlite_master")} == tables
    assert (tmp_path / "taxonomy" / "inat_genera.sqlite").is_file()


def test_a_release_ships_the_taxonomy_cache(conn, tmp_path):
    from mycomap_vision import release
    seed(conn, [("Russula emetica", "Russula", "")])
    fetch(conn, tmp_path)
    (tmp_path / "embeddings" / "m1").mkdir(parents=True)
    (tmp_path / "embeddings" / "m1" / "part.npy").write_bytes(b"x")
    from mycomap_vision.embed import SCHEMA
    conn.executescript(SCHEMA)
    conn.execute("insert into embeddings values ('m1', 1, 0, 0, 't')")
    files = [rel for _p, rel in release.release_files(conn, ["m1"], tmp_path)]
    assert "taxonomy/inat_genera.sqlite" in files


def test_mv_fetch_taxonomy_reads_the_manifest_only_and_reports(conn, tmp_path, monkeypatch,
                                                                capsys):
    from mycomap_vision import cli
    seed(conn, [("Russula emetica", "Russula", ""), ("Nolanea verna", "Nolanea", "")])
    conn.commit()
    conn.close()
    manifest = tmp_path / "manifest.sqlite"
    before = manifest.read_bytes()
    monkeypatch.setattr(config, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(config, "REPORTS_DIR", tmp_path / "reports")
    fake = FakeInat()
    stand_in = client(fake)
    monkeypatch.setattr(taxonomy, "InatTaxa", lambda: stand_in)
    assert cli.main(["fetch-taxonomy"]) == 0
    out = capsys.readouterr().out
    assert '"applied": 1' in out and taxonomy.INACTIVE in out
    assert manifest.read_bytes() == before
    assert (tmp_path / "reports" / "taxonomy-doubts.csv").is_file()
    assert cli.main(["taxonomy"]) == 0 and '"asked": 2' in capsys.readouterr().out
    assert len(fake.calls) == 4                    # the second command asked nothing


@pytest.mark.parametrize("word,expected", [("Russula", True), ("Sistotrema5", False),
                                           ("'Mycena'", False), ("galerina", False),
                                           ("Unknown", False), ("-", False)])
def test_only_a_latin_word_is_asked_about(word, expected):
    assert taxonomy.is_latin_word(word) is expected
