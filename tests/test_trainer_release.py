"""Training from a dataset release (docs/dataset-release.md §8): the fine-tune learns from
the release's train split only, the run sees only the release's records, the instance gets
the trainer copy (shipped with the run, without private.sqlite, its MANIFEST.json pinned at
launch), and every model and run file names the release.

The release here is a draft cut from the two-species manifest (seed_two_species): six
January records (train) and two September records (val, cutoff 2026-08-24)."""

import json
import shutil
from pathlib import Path

import pytest
from test_finetune import fake_finetuner, ft_loader
from test_models_and_scoreboard import Const, seed_two_species
from test_trainer_stages import FakeEC2, FakeLaunchS3, FakeSSM

from mycomap_vision import aws, config, finetune, trainer
from mycomap_vision import dataset_release as dr
from mycomap_vision.embed import embed_photos, photos_to_embed
from mycomap_vision.manifest import connect
from mycomap_vision.replications.fungitastic import dataset as release_files

VAL_PHOTOS = {1006, 1007}


def s3_copies(conn, ids=range(1000, 1008), store="s3://bkt/"):
    conn.executemany("insert into photo_copies values (?, ?, 'large', ?, 1, 'h', 't')",
                     [(i, store, f"large/{i}.jpg") for i in ids])
    conn.commit()


@pytest.fixture
def cut(tmp_path):
    """(manifest conn, its path, photo store, the draft release, releases root)."""
    path = tmp_path / "data" / "manifest.sqlite"
    conn = connect(path)
    store = seed_two_species(conn, tmp_path)
    s3_copies(conn)
    root = tmp_path / "research-releases"
    rid = dr.draft_id()
    dr.build(name=rid, manifest_path=path, root=root, derive_photos="none",
             val_cutoff="2026-08-24", log=lambda s: None)
    rel = dr.load_release(rid, root, allow_draft=True)
    yield conn, path, store, rel, root
    rel.close()


def embed(conn, store, root, name="base"):
    todo = photos_to_embed(conn, name, "large", store.location, "local")
    embed_photos(conn, store, Const(name), todo, root / name, log=lambda s: None)


# --- what a fine-tune on a release learns from ---------------------------------------

def test_a_fine_tune_on_a_release_learns_from_its_train_split_only(cut, tmp_path):
    conn, _, store, rel, _ = cut
    embed(conn, store, tmp_path / "emb")
    ts = finetune.build_trainset(conn, "base", store.location, "large", 28, tmp_path / "emb",
                                 release=rel)
    assert {pid for pid, _ in ts.items}.isdisjoint(VAL_PHOTOS)
    assert len(ts.items) == 6 and ts.records == 6
    assert ts.trained_through == rel.val_cutoff == "2026-08-24"


def test_the_releases_val_cutoff_decides_what_is_trained_on_not_test_days(cut, tmp_path):
    conn, path, store, _, root = cut
    # A release whose val split starts 2026-01-02: the four January 1/2 records train;
    # the 1/3 records and September are val. test_days alone would train on six.
    dr.build(name="dry-20260101-000001", manifest_path=path, root=root, derive_photos="none",
             val_cutoff="2026-01-02", log=lambda s: None)
    early = dr.load_release("dry-20260101-000001", root, allow_draft=True)
    embed(conn, store, tmp_path / "emb")
    ts = finetune.build_trainset(conn, "base", store.location, "large", 28, tmp_path / "emb",
                                 release=early)
    assert ts.records == 4 and ts.trained_through == "2026-01-02"
    assert finetune.build_trainset(conn, "base", store.location, "large", 28,
                                   tmp_path / "emb").records == 6
    early.close()


def test_a_test_release_is_never_trained_on(cut, tmp_path):
    conn, _, store, rel, _ = cut
    embed(conn, store, tmp_path / "emb")
    rel.manifest["test_for"] = "v1"
    with pytest.raises(dr.SealedLeak):
        finetune.build_trainset(conn, "base", store.location, "large", 28, tmp_path / "emb",
                                release=rel)


def test_a_model_trained_on_a_release_cites_it_and_takes_its_snapshot_date_as_version(cut):
    rel = cut[3]
    f = finetune.release_fields(rel)
    assert f["dataset_release"] == rel.cite()
    assert f["dataset_release"]["release_hash"] == rel.release_hash
    assert f["version"] == rel.snapshot_date
    assert len(f["version"]) == 10 and f["version"][4] == "-"
    assert f["split_trained"] == "train" and f["val_cutoff"] == "2026-08-24"
    assert finetune.release_fields(None) == {}
    assert finetune.model_version({"name": "old"}) is None
    assert finetune.model_version({"version": "2026-10-11"}) == "2026-10-11"


# --- the run on the instance -----------------------------------------------------------

def test_a_release_run_sees_only_the_release_compares_on_val_and_cites_it(cut, tmp_path):
    conn, _, store, rel, _ = cut
    conn.execute("insert into records (observation_id, source, scientific_name, genus, family, "
                 "north_america, validated_on, label_conflict, exported_at) values "
                 "('999', 'inat', 'A x', 'A', 'F', 1, '2026-01-04', 0, 't')")
    conn.commit()
    uploads = {}
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=fake_finetuner, release=rel, log=lambda s: None)
    assert out["state"] == "finished"
    assert out["dataset_release"] == release_files.cite(rel)
    progress = json.loads(uploads["runs/r1/progress.json"])
    assert progress["dataset_release"]["release_hash"] == rel.release_hash
    assert conn.execute("select count(*) from records where observation_id = '999'"
                        ).fetchone()[0] == 0           # outside the release: left out
    report = json.loads(uploads[f"runs/r1/reports/compare-{out['comparison_id']}.json"])
    assert report["dataset_release"]["dataset_release"] == rel.id
    assert report["test_records"] == 2 and report["reference_records"] == 6


def test_a_release_run_refuses_a_manifest_missing_the_releases_photos(cut, tmp_path):
    conn, _, store, rel, _ = cut
    conn.execute("delete from photo_copies where photo_id = 1003")
    conn.commit()
    with pytest.raises(ValueError, match="have no large copy"):
        trainer.run_job(conn, store, ["timm:a"], ["nearest"], lambda p, k: None, "r1",
                        loader=ft_loader, data_dir=tmp_path / "inst", release=rel,
                        log=lambda s: None)


def test_a_release_run_refuses_a_manifest_that_lacks_release_records(cut, tmp_path):
    conn, _, store, rel, _ = cut
    conn.execute("delete from records where observation_id = '101'")
    conn.commit()
    with pytest.raises(ValueError, match="not the manifest the release was cut from"):
        trainer.restrict_to_release(conn, rel, store.location, "large")


def test_a_release_run_is_never_a_rehearsal(cut, tmp_path):
    conn, _, store, rel, _ = cut
    with pytest.raises(ValueError, match="rehearsal"):
        trainer.run_job(conn, store, ["timm:a"], ["nearest"], lambda p, k: None, "r1",
                        loader=ft_loader, data_dir=tmp_path / "inst", sample_records=3,
                        release=rel, log=lambda s: None)


# --- the launcher on the laptop and the instance --------------------------------------

def script_release(rel):
    return {**release_files.cite(rel), "shipped": list(release_files.shipped_files(rel))}


def test_the_instance_gets_the_release_files_and_the_manifest_hash_the_launch_pinned(cut):
    rel = cut[3]
    ud = aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                      finetune=["bioclip-2"], release=script_release(rel))
    sha = release_files.manifest_sha256(rel)
    assert f"--release {rel.id} --release-manifest {sha} --allow-draft" in ud
    assert f"$RUN/release/{rel.id}/release.sqlite" in ud and "private.sqlite" not in ud
    assert "--release" not in aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10,
                                                           "bkt", finetune=["bioclip-2"])
    bad = {**script_release(rel), "manifest_sha256": "x; rm -rf /"}
    with pytest.raises(ValueError, match="manifest hash"):
        aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                     finetune=["bioclip-2"], release=bad)


def test_the_launcher_refuses_a_release_whose_photos_are_not_in_s3(cut):
    conn, _, _, rel, _ = cut
    assert aws.check_release_photos(conn, rel, "s3://bkt/", "large") == 8
    conn.execute("delete from photo_copies where photo_id = 1002 and store = 's3://bkt/'")
    conn.commit()
    with pytest.raises(ValueError, match="1 photos of release"):
        aws.check_release_photos(conn, rel, "s3://bkt/", "large")


def test_the_launcher_refuses_a_test_release(cut):
    conn, _, _, rel, _ = cut
    rel.manifest["test_for"] = "v1"
    with pytest.raises(ValueError, match="test release"):
        aws.check_release_photos(conn, rel, "s3://bkt/", "large")


def launch_env(monkeypatch, tmp_path, root, s3, ec2):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "data")
    (tmp_path / "data").mkdir(exist_ok=True)
    shutil.copytree(root, tmp_path / "data" / release_files.RELEASES_DIR)
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setenv("MV_AWS_REGION", "us-east-2")
    monkeypatch.setattr(aws, "s3_client", lambda: s3)
    monkeypatch.setattr(aws, "session", lambda: type("S", (), {
        "client": lambda self, n: {"ec2": ec2, "ssm": FakeSSM()}[n]})())
    monkeypatch.setattr(aws, "release_commit", lambda **k: "0123456789abcdef")
    monkeypatch.setattr(aws, "code_tarball", lambda ref: b"tgz")


def test_a_launch_from_a_release_ships_it_with_the_run_and_names_it_to_the_fine_tune(
        cut, tmp_path, monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13, release=rel.id,
                             allow_draft=True, log=lambda s: None)
    run = out["run_id"]
    assert out["dataset_release"]["release_hash"] == rel.release_hash
    assert f"runs/{run}/release/{rel.id}/release.sqlite" in s3.objects
    assert not any(k.endswith("private.sqlite") for k in s3.objects)
    assert f"--release {rel.id} --release-manifest" in ec2.launched["UserData"]


def test_a_release_launch_without_a_fine_tune_or_a_picek_stage_is_refused(cut, tmp_path,
                                                                          monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    with pytest.raises(ValueError, match="name --finetune"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=[], release=rel.id,
                           allow_draft=True, log=lambda s: None)
    with pytest.raises(ValueError, match="rehearsal"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           sample_records=3, release=rel.id, allow_draft=True,
                           log=lambda s: None)
    assert ec2.launched is None and not s3.objects


def test_the_release_on_the_instance_must_be_the_one_the_launch_pinned(cut, tmp_path,
                                                                       monkeypatch):
    from types import SimpleNamespace

    from mycomap_vision import cli
    rel, root = cut[3], cut[4]
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "inst")
    dest = tmp_path / "inst" / release_files.RELEASES_DIR / rel.id
    dest.mkdir(parents=True)
    for f in release_files.shipped_files(rel):             # what the instance script copies
        shutil.copyfile(root / rel.id / f, dest / f)
    args = SimpleNamespace(release=rel.id, release_manifest="c" * 64, allow_draft=True)
    with pytest.raises(Exception, match="(?i)manifest"):
        cli.trainer_release(args)
    args.release_manifest = release_files.manifest_sha256(rel)
    got = cli.trainer_release(args)
    assert got.release_hash == rel.release_hash and not got.has_private
    got.close()
