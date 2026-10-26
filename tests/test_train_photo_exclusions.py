"""Photos a person excluded in Human Feedback leave what a release fine-tune learns from and
is compared against (paper E34/E35, Steve 2026-10-26): never trained on, never in a reference
(train, or train + val), while the val queries a model is judged on stay exactly the same.
The list is pinned by its sha256 from the launch to the instance, an arm takes only the rows
with its reasons, and the list is named with its counts wherever the release is cited."""
import hashlib
import json
import shutil
from types import SimpleNamespace

import pytest
from test_finetune import fake_finetuner, ft_loader
from test_trainer_release import FakeEC2, FakeLaunchS3, cut, embed, launch_env  # noqa: F401

from mycomap_vision import aws, cli, config, evaluate, finetune, trainer
from mycomap_vision import dataset_release as dr
from mycomap_vision.replications.fungitastic import dataset as release_files

COLS = ["photo_key", "manifest_photo_id", "record_key", "v1_split", "decision", "reason"]


def key_of(rel, pid):
    return rel.db.execute("select photo_key from photos where manifest_photo_id = ?",
                          (pid,)).fetchone()[0]


def export(rel, path, rows):
    """A photo-exclusion export (mv feedback export-photo-exclusions) of (photo id, decision
    [, reason]); the reason is slip unless given."""
    lines = ["\t".join(COLS)]
    for pid, decision, *reason in rows:
        key = key_of(rel, pid) if isinstance(pid, int) else pid
        lines.append("\t".join([key, str(pid), "r", "train", decision, *(reason or ["slip"])]))
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


# --- the release reader ------------------------------------------------------------------

def test_an_excluded_train_photo_is_never_trained_on_and_its_record_drops_out(cut, tmp_path):
    conn, _, store, rel, _ = cut
    embed(conn, store, tmp_path / "emb")
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo")])
    rel.use_photo_exclusions(path, sha)
    ts = finetune.build_trainset(conn, "base", store.location, "large", 28, tmp_path / "emb",
                                 release=rel)
    assert 1000 not in {pid for pid, _ in ts.items}
    assert len(ts.items) == 5 and ts.records == 5        # record 100 had only that photo
    assert ts.photo_policy["photo_exclusions"]["sha256"] == sha


def test_val_queries_are_the_same_with_or_without_an_exclusion_list(cut, tmp_path):
    conn, _, store, rel, _ = cut
    embed(conn, store, tmp_path / "emb")
    before = evaluate.shared_records(conn, ["base"], embeddings_root=tmp_path / "emb",
                                     release=rel)
    # A listed val photo is a query when val is scored: it stays. The train photo leaves
    # the reference it is compared against.
    path, sha = export(rel, tmp_path / "x.tsv", [(1006, "exclude_photo"), (1001, "exclude_photo")])
    rel.use_photo_exclusions(path, sha)
    after = evaluate.shared_records(conn, ["base"], embeddings_root=tmp_path / "emb",
                                    release=rel)
    assert [(r.observation_id, r.photo_rows) for r in after.test] == \
        [(r.observation_id, r.photo_rows) for r in before.test]
    assert [r.observation_id for r in rel.as_records("val")] == ["106", "107"]
    assert 1006 in rel.photo_ids(("val",))
    assert len(after.ref) == len(before.ref) - 1
    assert 1001 not in {p for r in after.ref for p in r.photo_rows}
    assert 1001 not in rel.photo_ids(("train",))


def test_a_train_plus_val_reference_leaves_out_the_listed_val_photos_too(cut, tmp_path):
    rel = cut[3]
    path, sha = export(rel, tmp_path / "x.tsv", [(1006, "exclude_photo"), (1001, "exclude_photo")])
    rel.use_photo_exclusions(path, sha)
    # The reference a test find is compared against, and the served index: train + val.
    ref = rel.as_records(("train", "val"))
    assert {r.observation_id for r in ref} == {"100", "102", "103", "104", "105", "107"}
    assert not {1001, 1006} & set(rel.photo_ids())


def test_keep_photo_rows_take_nothing_out_and_the_counts_are_cited(cut, tmp_path):
    rel = cut[3]
    path, sha = export(rel, tmp_path / "x.tsv", [(1002, "exclude_photo"), (1003, "keep_photo"),
                                                 (1007, "exclude_photo"),
                                                 ("inat:999999", "exclude_photo")])
    got = rel.use_photo_exclusions(path, sha)
    assert got == {"file": "x.tsv", "sha256": sha, "reasons": "all", "photos_listed": 3,
                   "photos_left_out": {"train": 1, "val": 1},
                   "records_left_with_no_photo": {"train": 1, "val": 1},
                   "other_split_kept": 0, "not_in_release": 1}
    assert 1003 in rel.photo_ids(("train",)) and 1002 not in rel.photo_ids(("train",))
    assert rel.cite()["photo_exclusions"] == got
    assert release_files.cite(rel)["photo_exclusions"]["sha256"] == sha
    fresh = dr.load_release(rel.id, rel.folder.parent, allow_draft=True)
    assert "photo_exclusions" not in fresh.cite()
    fresh.close()


def test_an_arm_takes_only_the_rows_with_its_reasons(cut, tmp_path):
    rel = cut[3]
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo", "slip"),
                                                 (1001, "exclude_photo", "micrograph"),
                                                 (1002, "exclude_photo", "habitat")])
    got = rel.use_photo_exclusions(path, sha, ["slip"])
    assert got["reasons"] == ["slip"] and got["photos_left_out"]["train"] == 1
    assert 1000 not in rel.photo_ids() and {1001, 1002} <= set(rel.photo_ids())
    assert rel.use_photo_exclusions(path, sha, ["micrograph", "habitat"])["photos_listed"] == 2
    assert 1000 in rel.photo_ids() and not {1001, 1002} & set(rel.photo_ids())
    with pytest.raises(dr.ReleaseError, match="no excluded photo has reason"):
        rel.use_photo_exclusions(path, sha, ["uv"])


def test_a_list_that_is_not_the_one_pinned_is_refused(cut, tmp_path):
    rel = cut[3]
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo")])
    with pytest.raises(dr.ReleaseError, match="pinned"):
        rel.use_photo_exclusions(path, "0" * 64)
    with pytest.raises(dr.ReleaseError, match="pinned"):
        rel.use_photo_exclusions(path, None)
    assert rel.photo_exclusions is None and 1000 in rel.photo_ids()
    bad = tmp_path / "bad.tsv"
    bad.write_bytes(b"photo_id\n1000\n")
    with pytest.raises(dr.ReleaseError, match="photo_key"):
        rel.use_photo_exclusions(bad, hashlib.sha256(bad.read_bytes()).hexdigest())


# --- the run on the instance -------------------------------------------------------------

def test_a_run_with_exclusions_compares_on_the_same_val_and_records_the_list(cut, tmp_path):
    conn, _, store, rel, _ = cut
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo")])
    rel.use_photo_exclusions(path, sha)
    uploads = {}
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=fake_finetuner, release=rel, log=lambda s: None)
    assert out["state"] == "finished"
    progress = json.loads(uploads["runs/r1/progress.json"])
    assert progress["dataset_release"]["photo_exclusions"]["sha256"] == sha
    report = json.loads(uploads[f"runs/r1/reports/compare-{out['comparison_id']}.json"])
    assert report["test_records"] == 2 and report["reference_records"] == 5


def test_the_instance_applies_the_list_only_when_its_hash_is_the_launchs(cut, tmp_path,
                                                                         monkeypatch):
    rel, root = cut[3], cut[4]
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "inst")
    dest = tmp_path / "inst" / release_files.RELEASES_DIR / rel.id
    dest.mkdir(parents=True)
    for f in release_files.shipped_files(rel):
        shutil.copyfile(root / rel.id / f, dest / f)
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo")])
    args = SimpleNamespace(release=rel.id, allow_draft=True,
                           release_manifest=release_files.manifest_sha256(rel),
                           exclude_photos=str(path), exclude_photos_sha256="f" * 64,
                           exclude_photo_reasons="slip")
    with pytest.raises(dr.ReleaseError, match="pinned"):
        cli.trainer_release(args)
    args.exclude_photos_sha256 = sha
    got = cli.trainer_release(args)
    assert 1000 not in got.photo_ids() and got.cite()["photo_exclusions"]["sha256"] == sha
    assert got.cite()["photo_exclusions"]["reasons"] == ["slip"]
    got.close()


# --- the launcher ------------------------------------------------------------------------

def test_a_launch_ships_the_list_pinned_and_names_it_to_the_fine_tune(cut, tmp_path,
                                                                      monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo")])
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13, release=rel.id,
                             allow_draft=True, exclude_photos=path, log=lambda s: None)
    run = out["run_id"]
    assert s3.objects[f"runs/{run}/photo-exclusions.tsv"] == path.read_bytes()
    ud = ec2.launched["UserData"]
    assert '$RUN/photo-exclusions.tsv" data/photo-exclusions.tsv || exit 1' in ud
    assert f"--exclude-photos data/photo-exclusions.tsv --exclude-photos-sha256 {sha}" in ud
    assert "--exclude-photo-reasons" not in ud
    assert out["dataset_release"]["photo_exclusions"]["photos_left_out"]["train"] == 1


def test_a_launch_of_one_arm_names_its_reasons_to_the_instance(cut, tmp_path, monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo", "slip"),
                                                 (1001, "exclude_photo", "micrograph")])
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13, release=rel.id,
                             allow_draft=True, exclude_photos=path,
                             exclude_photo_reasons=["slip"], log=lambda s: None)
    assert (f"--exclude-photos-sha256 {sha} --exclude-photo-reasons slip"
            in ec2.launched["UserData"])
    assert out["dataset_release"]["photo_exclusions"]["reasons"] == ["slip"]
    assert out["dataset_release"]["photo_exclusions"]["photos_listed"] == 1


def test_a_launch_without_a_list_sends_none(cut, tmp_path, monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13, release=rel.id,
                             allow_draft=True, log=lambda s: None)
    assert not any(k.endswith("photo-exclusions.tsv") for k in s3.objects)
    assert "--exclude-photos" not in ec2.launched["UserData"]
    assert "photo_exclusions" not in out["dataset_release"]


def test_a_list_is_refused_outside_a_release_fine_tune(cut, tmp_path, monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo")])
    for kw in ({"finetune": ["bioclip-2"]},                         # no release
               {"finetune": [], "release": rel.id, "picek": ["fungitastic-beit-b384"]},
               {"finetune": ["bioclip-2"], "release": rel.id, "resume": "20261001-000000"}):
        with pytest.raises(ValueError, match="--exclude-photos"):
            aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], allow_draft=True,
                               exclude_photos=path, log=lambda s: None, **kw)
    with pytest.raises(ValueError, match="give both"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           release=rel.id, allow_draft=True, exclude_photo_reasons=["slip"],
                           log=lambda s: None)
    assert ec2.launched is None and not s3.objects
    script = {**release_files.cite(rel), "shipped": list(release_files.shipped_files(rel))}
    with pytest.raises(ValueError, match="release fine-tune only"):
        aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                     photo_exclusions_sha256=sha)
    with pytest.raises(ValueError, match="hash"):
        aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                     finetune=["bioclip-2"], release=script,
                                     photo_exclusions_sha256="x; rm -rf /")
    with pytest.raises(ValueError, match="reasons"):
        aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                     finetune=["bioclip-2"], release=script,
                                     photo_exclusions_sha256=sha,
                                     photo_exclusion_reasons=["slip; rm -rf /"])


def test_a_dry_run_writes_the_list_beside_the_instance_script(cut, tmp_path, monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    monkeypatch.setattr(aws, "check_code_archive", lambda body, picek=None: {})  # b"tgz"
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo")])
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13, release=rel.id,
                             allow_draft=True, exclude_photos=path, dry_run=True,
                             dry_run_dir=tmp_path / "dry", log=lambda s: None)
    assert (tmp_path / "dry" / "photo-exclusions.tsv").read_bytes() == path.read_bytes()
    assert f"--exclude-photos-sha256 {sha}" in (tmp_path / "dry" / "user-data.sh").read_text()
    assert out["dataset_release"]["photo_exclusions"]["sha256"] == sha
    assert not s3.objects and ec2.launched is None
