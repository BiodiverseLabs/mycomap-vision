"""A training launch from a dataset release needs a manifest that still holds every record
the release includes (the instance keeps only those, trainer.restrict_to_release). A later
export removes records .org deleted (v1, 2026-10-14: 43 MO-number rows), so the live
manifest can stop covering an older release: the launcher says so before anything is paid
for, and `--manifest FILE` launches from a private copy of the manifest the release was
cut from, never writing that file."""

import shutil

import pytest
from test_trainer_release import cut, launch_env  # noqa: F401
from test_trainer_stages import FakeEC2, FakeLaunchS3

from mycomap_vision import aws, cli, config
from mycomap_vision import dataset_release as dr


def drop_record(conn, key="100"):
    conn.execute("delete from records where observation_id = ?", (key,))
    conn.commit()


def test_a_manifest_that_lost_a_release_record_is_refused_before_any_spend(cut, tmp_path,
                                                                           monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    drop_record(conn)
    with pytest.raises(ValueError, match=r"1 records of release .* are not in this manifest"):
        aws.check_release_photos(conn, rel, "s3://bkt/", "large")
    with pytest.raises(ValueError, match="--manifest"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           release=rel.id, allow_draft=True, log=lambda s: None)
    assert ec2.launched is None and not s3.objects


def test_the_launch_runs_from_a_copy_of_the_manifest_the_release_was_cut_from(cut, tmp_path,
                                                                              monkeypatch):
    conn, path, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    conn.execute("pragma wal_checkpoint(truncate)")
    frozen = tmp_path / "manifest-v1-freeze.sqlite"
    shutil.copyfile(path, frozen)                        # as `mv manifest-freeze` keeps it
    before = dr.sha256_file(frozen)
    drop_record(conn)                                    # the live manifest moves on
    copy = cli.launch_manifest_copy(str(frozen))
    opened = next(r[2] for r in copy.execute("pragma database_list") if r[1] == "main")
    assert opened.endswith("manifest-launch.sqlite")     # the copy, never the frozen file
    out = aws.launch_trainer(copy, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             release=rel.id, allow_draft=True, log=lambda s: None)
    assert out["run_id"] and ec2.launched is not None
    assert dr.sha256_file(frozen) == before              # never written
    assert (config.DATA_DIR / "manifest-launch.sqlite").is_file()
    held = {r[0] for r in copy.execute("select observation_id from records")}
    assert "100" in held


def test_manifest_and_resume_do_not_go_together(tmp_path):
    with pytest.raises(SystemExit, match="--resume keeps the run's own"):
        cli.main(["aws-launch-trainer", "--resume", "r1", "--manifest", str(tmp_path / "m")])
    with pytest.raises(SystemExit, match="no manifest at"):
        cli.launch_manifest_copy(str(tmp_path / "missing.sqlite"))
