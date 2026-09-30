from functools import partial

from conftest import inat_obs
from test_models_and_scoreboard import Const
from test_taxonomy import FakeInat, Response, client

from mycomap_vision import taxonomy
from mycomap_vision.inat import save_batch
from mycomap_vision.photos import pending_photos
from mycomap_vision.records import build_records, save_records
from mycomap_vision.refresh import Steps, active_backbones, new_genera_taxonomy, refresh
from mycomap_vision.storage import LocalStore


def row(oid, name="A x", when="9/1/2026"):
    return {"observation_id": oid, "scientific_name": name, "genus": name.split()[0],
            "family": "F", "continent": "North America", "validation_status_1": "yes",
            "validation_date_1": when}


def test_the_export_reports_new_removed_and_renamed_records(conn):
    save_records(conn, build_records([row("1"), row("2"), row("3")], "t1"))
    changes = save_records(conn, build_records([row("1"), row("2", "B y"), row("4")], "t2"))
    assert changes == {"new": 1, "removed": 1, "renamed": 1}
    seen = dict(conn.execute("select observation_id, first_seen_at from records"))
    assert seen == {"1": "t1", "2": "t1", "4": "t2"}      # first sighting is never overwritten


def test_a_new_scope_download_takes_only_records_first_seen_in_this_refresh(conn):
    save_records(conn, build_records([row("5")], "t1"))
    save_records(conn, build_records([row("5"), row("6")], "t2"))
    save_batch(conn, ["5", "6"], [
        inat_obs(5, photos=[(0, 50, "cc0", "inaturalist-open-data.s3.amazonaws.com")]),
        inat_obs(6, photos=[(0, 60, "cc0", "inaturalist-open-data.s3.amazonaws.com")])], "t2")
    assert [r["photo_id"] for r in pending_photos(conn, True, None, first_seen_since="t2")] == [60]
    assert sorted(r["photo_id"] for r in pending_photos(conn, True, None)) == [50, 60]


class Recorder:
    def __init__(self):
        self.calls = []

    def export(self, conn):
        self.calls.append("export")
        return {"exported_at": "T", "new": 3, "removed": 0, "renamed": 1}

    def fetch(self, conn, log=print):
        self.calls.append("fetch")
        return {"ok": 3}

    def taxonomy(self, conn, max_minutes=None, log=print):
        self.calls.append(("taxonomy", max_minutes))
        return {"asked": 0}

    def download(self, conn, store, size, first_seen_since, log=print):
        self.calls.append(("download", size, first_seen_since))
        return {"done": 0}

    def compare(self, conn, backbones, methods, embeddings_root=None, log=print):
        self.calls.append(("compare", tuple(backbones), tuple(methods)))
        return {"comparison_id": "c1"}


def steps_for(rec):
    return Steps(export=rec.export, fetch=rec.fetch, taxonomy=rec.taxonomy,
                 download=rec.download, load_backbone=lambda spec: Const(spec),
                 compare=rec.compare)


def test_refresh_runs_the_stages_in_order_and_compares_the_active_backbones(conn, tmp_path):
    rec = Recorder()
    conn.executescript("create table if not exists embeddings (backbone text, photo_id integer, "
                       "shard integer, row integer, created_at text, primary key (backbone, photo_id))")
    conn.execute("insert into embeddings values ('m-old', 1, 0, 0, '2026-01-01')")
    conn.execute("insert into embeddings values ('m-new', 1, 0, 0, '2026-02-01')")
    assert active_backbones(conn) == ["m-old", "m-new"]
    report = refresh(conn, LocalStore(tmp_path), scope="new", steps=steps_for(rec),
                     embeddings_root=tmp_path / "emb", log=lambda s: None)
    assert rec.calls == ["export", "fetch", ("taxonomy", 20.0), ("download", "medium", "T"),
                         ("compare", ("m-old", "m-new"), ("nearest", "species-mean"))]
    assert report.embedded == {"m-old": 0, "m-new": 0}
    assert report.comparison == {"comparison_id": "c1"}


def test_an_all_scope_refresh_downloads_every_missing_photo(conn, tmp_path):
    rec = Recorder()
    refresh(conn, LocalStore(tmp_path), scope="all", steps=steps_for(rec), compare=False,
            log=lambda s: None)
    assert ("download", "medium", None) in rec.calls
    assert not any(isinstance(c, tuple) and c[0] == "compare" for c in rec.calls)


# --- iNat's taxonomy for new genera, in the weekly loop --------------------------------

def taxonomy_steps(rec, tmp_path, fake):
    """The real lookup the refresh runs, against a stand-in iNat that never sleeps."""
    steps = steps_for(rec)
    steps.taxonomy = partial(new_genera_taxonomy, cache=tmp_path / taxonomy.CACHE,
                             client=client(fake))
    return steps


def test_the_weekly_refresh_asks_inat_only_about_genera_it_has_never_answered(conn, tmp_path):
    save_records(conn, build_records([row("1", "Russula emetica")], "t1"))
    first = FakeInat()
    refresh(conn, LocalStore(tmp_path), steps=taxonomy_steps(Recorder(), tmp_path, first),
            compare=False, log=lambda s: None)
    assert first.searched() == ["Russula"]
    save_records(conn, build_records([row("1", "Russula emetica"),
                                      row("2", "Entoloma sericeum")], "t2"))
    later = FakeInat()
    report = refresh(conn, LocalStore(tmp_path), steps=taxonomy_steps(Recorder(), tmp_path, later),
                     compare=False, log=lambda s: None)
    assert later.searched() == ["Entoloma"]                     # Russula is not asked again
    assert report.taxonomy["asked"] == 1
    genera = taxonomy.load(tmp_path / taxonomy.CACHE).genera
    assert genera["Entoloma"]["family"] == "Entolomataceae"


def test_the_taxonomy_step_can_be_skipped(conn, tmp_path):
    rec = Recorder()
    report = refresh(conn, LocalStore(tmp_path), steps=steps_for(rec), compare=False,
                     lookup_taxonomy=False, log=lambda s: None)
    assert not any(isinstance(c, tuple) and c[0] == "taxonomy" for c in rec.calls)
    assert report.taxonomy == {"skipped": True}
    assert ("download", "medium", "T") in rec.calls


class InatDown(FakeInat):
    def get(self, url, params=None, timeout=None):
        self.calls.append((url, dict(params or {})))
        return Response({}, 503)


def test_the_refresh_carries_on_when_inat_is_down_and_keeps_the_answers_it_had(conn, tmp_path):
    save_records(conn, build_records([row("1", "Russula emetica")], "t1"))
    refresh(conn, LocalStore(tmp_path), steps=taxonomy_steps(Recorder(), tmp_path, FakeInat()),
            compare=False, log=lambda s: None)
    save_records(conn, build_records([row("1", "Russula emetica"),
                                      row("2", "Entoloma sericeum")], "t2"))
    rec, said = Recorder(), []
    conn.executescript("create table if not exists embeddings (backbone text, photo_id integer, "
                       "shard integer, row integer, created_at text, "
                       "primary key (backbone, photo_id))")
    conn.execute("insert into embeddings values ('m1', 1, 0, 0, '2026-01-01')")
    report = refresh(conn, LocalStore(tmp_path), steps=taxonomy_steps(rec, tmp_path, InatDown()),
                     embeddings_root=tmp_path / "emb", log=said.append)
    assert "503" in report.taxonomy["error"] or "kept failing" in report.taxonomy["error"]
    assert any("taxonomy lookup failed, carrying on" in s for s in said)
    assert ("download", "medium", "T") in rec.calls and report.comparison   # the rest ran
    assert "Russula" in taxonomy.load(tmp_path / taxonomy.CACHE).genera      # nothing lost


def test_the_taxonomy_lookup_stops_asking_at_its_time_and_the_next_refresh_carries_on(conn,
                                                                                      tmp_path):
    save_records(conn, build_records([row("1", "Russula emetica"), row("2", "Entoloma sericeum"),
                                      row("3", "Entoloma sinuatum")], "t1"))
    now = [0.0]

    class Slow(FakeInat):                  # every request takes a minute
        def get(self, url, params=None, timeout=None):
            now[0] += 60
            return super().get(url, params, timeout)
    fake = Slow()
    out = taxonomy.fetch(conn, tmp_path / taxonomy.CACHE, client(fake), max_minutes=0.5,
                         clock=lambda: now[0], log=lambda s: None)
    assert fake.searched() == ["Entoloma"] and out["left"] == 1 and out["out_of_time"]
    rest = FakeInat()
    report = refresh(conn, LocalStore(tmp_path), steps=taxonomy_steps(Recorder(), tmp_path, rest),
                     compare=False, log=lambda s: None)
    assert rest.searched() == ["Russula"] and report.taxonomy["left"] == 0
