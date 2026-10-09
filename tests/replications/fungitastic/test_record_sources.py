"""A Picek run trains only on iNaturalist observations. Records from Mushroom Observer,
MyCoPortal or .com sequences fetched AS iNat ids carry other observations' photos, so the
launch (and the instance, before its first stage) refuses any training or validation
record whose source isn't 'inat', and refuses a real launch outright while the manifest's
sources are not migrated (the record-sources-v1 marker row; before it, records.source is a
guess from the id). A dry run sends nothing, so it only warns."""

import pytest
from test_models_and_scoreboard import seed_two_species
from test_retrain_trainer import fake_picek_trainer, head_sha

from mycomap_vision import aws, config, trainer
from mycomap_vision.replications.fungitastic import retrain


def migration_table(conn):
    """The record-sources fix's marker table (feat/record-sources-mo), without its row."""
    conn.execute("create table manifest_migrations (name text primary key, applied_at text "
                 "not null, code_version text, detail text)")


def record_sources_as(conn, monkeypatch=None, default="inat", **by_id):
    """A manifest migrated by the record-sources fix: its marker row, the raw .org source
    column, and records.source normalised (`default`, or by_id[observation_id])."""
    with conn:
        migration_table(conn)
        conn.execute("insert into manifest_migrations values (?, 't', 'sha', null)",
                     (retrain.MIGRATION,))
        conn.execute("alter table records add column org_source text")
        conn.execute("update records set source = ?", (default,))
        for oid, source in by_id.items():
            conn.execute("update records set source = ? where observation_id = ?",
                         (source, oid))


@pytest.fixture
def launchable(conn, tmp_path, monkeypatch):
    """A manifest whose photos are in the bucket, with AWS forbidden: anything that would
    be sent fails the test."""
    seed_two_species(conn, tmp_path)
    with conn:
        conn.executemany("insert into photo_copies values (?, 's3://bkt/', 'large', ?, 1, 'h', "
                         "'t')", [(1000 + i, f"p/{1000 + i}.png") for i in range(8)])
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setattr(aws, "release_commit", lambda **k: head_sha())

    def no_aws(*a, **k):
        raise AssertionError("a refused launch must not touch AWS")
    monkeypatch.setattr(aws, "s3_client", no_aws)
    monkeypatch.setattr(aws, "session", no_aws)
    monkeypatch.setattr("shutil.which", lambda name: None)
    return conn


def launch(conn, tmp_path, dry_run, log=lambda s: None):
    return aws.launch_trainer(conn, [], ["classifier"], max_hours=48, instance_type="g6.xlarge",
                              picek=["fungitastic-beit-b384@15"], dry_run=dry_run,
                              dry_run_dir=tmp_path / "dry", log=log)


def first_record(conn):
    return conn.execute("select min(observation_id) from records").fetchone()[0]


def test_a_real_launch_is_refused_while_the_manifest_records_no_source(launchable, tmp_path):
    with pytest.raises(retrain.RecordSourceError, match="not migrated"):
        launch(launchable, tmp_path, dry_run=False)


def test_without_a_source_the_dry_run_still_runs_and_warns_loudly(launchable, tmp_path):
    said = []
    out = launch(launchable, tmp_path, dry_run=True, log=said.append)
    assert out["sent"].startswith("nothing")
    assert any(s.startswith("WARNING (a real launch is refused)") and "not migrated" in s
               for s in said)
    assert out["picek_labels"]["record_sources"]["recorded"] is False
    assert any("record sources: NOT MIGRATED" in s for s in said)


def test_a_manifest_without_the_migration_table_is_refused(launchable, tmp_path):
    with launchable:
        launchable.execute("update records set source = 'inat'")
    with pytest.raises(retrain.RecordSourceError, match="not migrated"):
        launch(launchable, tmp_path, dry_run=False)


def test_the_old_guess_inat_everywhere_is_refused_until_the_marker_row_is_there(
        launchable, tmp_path):
    # The trap: before the migration records.source says 'inat' for any numeric id (MO ids
    # too), and opening a manifest with the new code already adds org_source. Neither
    # proves anything; only the marker row does.
    with launchable:
        migration_table(launchable)
        launchable.execute("alter table records add column org_source text")
        launchable.execute("update records set source = 'inat'")
    assert not retrain.sources_migrated(launchable)
    with pytest.raises(retrain.RecordSourceError, match="not migrated"):
        launch(launchable, tmp_path, dry_run=False)


def test_a_launch_is_refused_while_any_record_is_not_from_inaturalist(launchable, tmp_path,
                                                                      monkeypatch):
    record_sources_as(launchable, monkeypatch, **{first_record(launchable): "mo"})
    with pytest.raises(retrain.RecordSourceError, match=r"1 of the 6 .*inat: 5, mo: 1"):
        launch(launchable, tmp_path, dry_run=False)
    said = []
    out = launch(launchable, tmp_path, dry_run=True, log=said.append)
    assert out["picek_labels"]["record_sources"] == {
        "recorded": True, "migration": "record-sources-v1", "records": 6,
        "by_source": {"inat": 5, "mo": 1}, "not_inat": 1}
    assert any(s.startswith("WARNING (a real launch is refused)") for s in said)


def test_a_record_of_unknown_source_counts_as_not_inaturalist(launchable, monkeypatch):
    record_sources_as(launchable, monkeypatch, **{first_record(launchable): "unknown"})
    labels, _ = aws.picek_launch_labels(launchable, "s3://bkt/", "large", 28, "match",
                                        log=lambda s: None, dry_run=True)
    assert labels["record_sources"]["by_source"] == {"inat": 5, "unknown": 1}
    assert labels["record_sources"]["not_inat"] == 1


def test_an_inat_only_snapshot_passes_the_launch_check(launchable, monkeypatch):
    record_sources_as(launchable, monkeypatch)
    said = []
    labels, _ = aws.picek_launch_labels(launchable, "s3://bkt/", "large", 28, "match",
                                        log=said.append, dry_run=False)
    assert labels["record_sources"]["not_inat"] == 0
    assert any("are iNaturalist observations" in s for s in said)


def run_on_instance(conn, store, tmp_path, monkeypatch, trained):
    data_dir = tmp_path / "inst"
    monkeypatch.setattr(config, "DATA_DIR", data_dir)

    def picek_trainer(*a, **k):
        trained.append(a)
        return fake_picek_trainer(*a, **k)
    return trainer.run_job(conn, store, [], ["classifier"], lambda p, k: None, "r1",
                           data_dir=data_dir, picek=["fungitastic-beit-b384@2"],
                           picek_trainer=picek_trainer, log=lambda s: None)


def test_the_instance_refuses_before_its_first_stage_without_a_source(conn, tmp_path,
                                                                      monkeypatch):
    store = seed_two_species(conn, tmp_path)
    trained = []
    with pytest.raises(retrain.RecordSourceError, match="not migrated"):
        run_on_instance(conn, store, tmp_path, monkeypatch, trained)
    assert trained == []
    assert not (tmp_path / "inst" / trainer.PROGRESS_FILE).exists()


def test_the_instance_refuses_a_manifest_with_a_record_not_from_inaturalist(conn, tmp_path,
                                                                            monkeypatch):
    store = seed_two_species(conn, tmp_path)
    record_sources_as(conn, monkeypatch, **{first_record(conn): "mycoportal"})
    trained = []
    with pytest.raises(retrain.RecordSourceError, match="mycoportal: 1"):
        run_on_instance(conn, store, tmp_path, monkeypatch, trained)
    assert trained == []


def test_the_dry_run_reports_the_audit_files_non_inat_records(launchable, tmp_path,
                                                             monkeypatch):
    data = tmp_path / "laptop-data"
    audit = data / retrain.AUDIT_TSV
    audit.parent.mkdir(parents=True)
    ids = [r[0] for r in launchable.execute("select observation_id from records "
                                            "order by observation_id")]
    rows = ["observation_id\tsource\tcoalesce"] + [
        f"{oid}\t{'MO Observations' if n == 0 else 'iNaturalist'}\tNA"
        for n, oid in enumerate(ids)]
    audit.write_text("\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setattr(config, "DATA_DIR", data)
    said = []
    out = launch(launchable, tmp_path, dry_run=True, log=said.append)
    assert any("Audit file" in s and "1 of the 6" in s and "MO Observations 1" in s
               for s in said)
    assert "audit" not in str(out["picek_labels"]).lower()      # never shipped
