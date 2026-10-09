"""The Picek run on a dataset release (docs/dataset-release.md; releases made by the
builder, release_fixture): verified before anything else; the release is the source of
truth for records, labels, splits, photos, sources and benchmark exclusion; its
labels_hash is the run's labels identity; only release.sqlite, MANIFEST.json and
SHA256SUMS ship (never private.sqlite); the instance verifies them again before its first
stage; a test release is never trained on."""

import json

import pytest
from release_fixture import BASE, TEST, release_world
from test_retrain_trainer import fake_picek_trainer, head_sha

from mycomap_vision import aws, trainer
from mycomap_vision import dataset_release as dr
from mycomap_vision.replications.fungitastic import dataset, retrain
from mycomap_vision.storage import LocalStore


@pytest.fixture
def world(tmp_path, monkeypatch):
    return release_world(tmp_path, monkeypatch)


def rel_of(rid=BASE):
    return dataset.load(rid, allow_draft=True)


def data_of(world, rel):
    return retrain.build_data_from_release(world["conn"], rel, world["store"].location, "large")


# --- what trains ----------------------------------------------------------------------------

def test_training_takes_the_train_split_and_selects_on_the_val_split(world):
    data = data_of(world, rel_of())
    assert sorted(data.train_records) == ["1", "2", "3", "4", "mo:9"]
    assert sorted(data.val_records) == ["5", "6"]
    # The month and place prior counts come from the train split only.
    assert int(data.month_counts.sum()) == len(data.train) == 5
    assert sum(sum(c.values()) for c in data.cells.values()) == 5
    assert data.train_through == "2026-01-03" and data.trained_through == "2026-03-02"


def test_a_test_release_is_never_trained_on(world):
    with pytest.raises(dr.SealedLeak):
        data_of(world, rel_of(TEST))


def test_a_leaked_non_trainable_record_raises(world, monkeypatch):
    rel = rel_of()
    real = rel.training_records

    def leaky(include_val=False):
        rows = [dict(r) for r in real(include_val)]
        rows[0]["split"] = "test"
        return rows
    monkeypatch.setattr(rel, "training_records", leaky)
    with pytest.raises(dr.SealedLeak, match="not trainable"):
        data_of(world, rel)


def test_a_store_copy_that_is_not_the_releases_original_is_refused(world):
    with world["conn"]:
        world["conn"].execute("update photo_copies set sha256 = 'other' where photo_id = 1001")
    with pytest.raises(dr.ReleaseError, match="another original"):
        data_of(world, rel_of())


def test_benchmark_exclusion_follows_the_release_and_a_mismatch_is_refused(world,
                                                                          monkeypatch):
    rel = rel_of()
    retrain.check_release_exclusion(rel, "all")          # no benchmark record trains here
    monkeypatch.setattr(dataset, "benchmark_records", lambda rel, splits=None: 3)
    retrain.check_release_exclusion(rel, "match")
    with pytest.raises(ValueError, match="trains on 3 held-out benchmark"):
        retrain.check_release_exclusion(rel, "all")


# --- sources and identity -------------------------------------------------------------------

def test_allowed_sources_are_the_releases_included_sources(world):
    rel = rel_of()
    snap = retrain.release_snapshot(rel, data_of(world, rel))
    assert snap["allowed_sources"] == ["inat", "mo"]
    assert snap["record_sources"]["by_source"] == {"inat": 6, "mo": 1}
    retrain.require_allowed_sources(snap["record_sources"])


def test_a_release_including_a_source_beyond_inat_and_mo_is_refused(world, monkeypatch):
    rel = rel_of()
    monkeypatch.setattr(dataset, "allowed_sources", lambda rel: ("inat", "mycoportal"))
    with pytest.raises(retrain.RecordSourceError, match="non-empty subset"):
        retrain.release_snapshot(rel, data_of(world, rel))


def test_the_snapshot_carries_the_releases_identity_and_the_photo_licence_policy(world):
    rel = rel_of()
    snap = retrain.release_snapshot(rel, data_of(world, rel))
    assert snap["labels_hash"] == rel.labels_hash != snap["own_labels_hash"]
    assert snap["dataset_release"] == {**rel.cite(),
                                       "manifest_sha256": dataset.manifest_sha256(rel)}
    assert snap["val_cutoff"] == rel.val_cutoff
    assert "arr" in snap["photo_licence_policy"]
    assert sum(snap["photo_licences"].values()) == 7 and snap["photo_licences"]["arr"] >= 1
    assert f"dataset release {BASE}" in retrain.format_snapshot(snap)


# --- the launch ----------------------------------------------------------------------------

@pytest.fixture
def launchable(world, monkeypatch):
    """The release's photos are in the bucket; AWS is forbidden."""
    conn = world["conn"]
    with conn:
        conn.execute("insert into photo_copies select photo_id, 's3://bkt/', size, path, bytes, "
                     "sha256, downloaded_at from photo_copies where size = 'large'")
    monkeypatch.setattr(aws, "bucket", lambda: "bkt")
    monkeypatch.setattr(aws, "release_commit", lambda **k: head_sha())

    def no_aws(*a, **k):
        raise AssertionError("must not touch AWS")
    monkeypatch.setattr(aws, "s3_client", no_aws)
    monkeypatch.setattr(aws, "session", no_aws)
    monkeypatch.setattr("shutil.which", lambda name: None)
    return conn


def launch(conn, tmp_path, release=BASE, allow_draft=True, dry_run=True, exclude="match"):
    return aws.launch_trainer(conn, [], ["classifier"], max_hours=48, instance_type="g6.xlarge",
                              picek=["fungitastic-beit-b384@15"], dry_run=dry_run,
                              dry_run_dir=tmp_path / "dry", release=release,
                              allow_draft=allow_draft, picek_exclude=exclude,
                              log=lambda s: None)


def test_a_dry_run_on_a_release_ships_what_training_needs_and_never_private(
        launchable, tmp_path):
    out = launch(launchable, tmp_path)
    rel = rel_of()
    sha = dataset.manifest_sha256(rel)
    assert out["dataset_release"]["manifest_sha256"] == sha
    assert out["picek_labels"]["labels_hash"] == rel.labels_hash
    shipped = tmp_path / "dry" / "release" / BASE
    assert sorted(p.name for p in shipped.iterdir()) == sorted(dataset.SHIPPED)
    script = (tmp_path / "dry" / "user-data.sh").read_text(encoding="utf-8")
    assert f"--picek-labels-hash {rel.labels_hash}" in script
    assert f"--picek-release {BASE} --picek-release-manifest {sha} --picek-allow-draft" in script
    for f in dataset.SHIPPED:
        assert f"release/{BASE}/{f}" in script
    assert "private.sqlite" not in script


@pytest.mark.parametrize("break_it,match", [
    ("draft", "draft"), ("alter", "does not match"), ("test release", "test release"),
    ("benchmark-all", "held-out benchmark")])
def test_a_launch_refuses_a_bad_release_before_anything_is_sent(launchable, tmp_path,
                                                                monkeypatch, break_it, match):
    rid = TEST if break_it == "test release" else BASE
    if break_it == "alter":
        db = tmp_path / "data" / "research-releases" / BASE / "release.sqlite"
        db.chmod(0o644)
        with open(db, "ab") as f:
            f.write(b"x")
    if break_it == "benchmark-all":
        monkeypatch.setattr(dataset, "benchmark_records", lambda rel, splits=None: 2)
    with pytest.raises((ValueError, RuntimeError), match=match):
        launch(launchable, tmp_path, release=rid, allow_draft=break_it != "draft",
               dry_run=False, exclude="all" if break_it == "benchmark-all" else "match")


# --- the instance -----------------------------------------------------------------------------

def shipped_copy(world, tmp_path, monkeypatch, alter=False):
    """The instance's data folder: the shipped release files only."""
    import shutil

    from mycomap_vision import config
    sha = dataset.manifest_sha256(rel_of())
    inst = tmp_path / "inst"
    folder = inst / "research-releases" / BASE
    folder.mkdir(parents=True)
    for f in dataset.SHIPPED:
        shutil.copyfile(world["root"] / BASE / f, folder / f)
    monkeypatch.setattr(config, "DATA_DIR", inst)
    if alter:
        with open(folder / "release.sqlite", "ab") as fh:
            fh.write(b"x")
    return inst, sha


def run_job(world, inst, sha, trained):
    def picek_trainer(*a, **k):
        trained.append(a)
        return fake_picek_trainer(*a, **k)
    return trainer.run_job(world["conn"], LocalStore(world["store"].root), [], ["classifier"],
                           lambda p, k: None, "r1", data_dir=inst,
                           picek=["fungitastic-beit-b384@2"], picek_trainer=picek_trainer,
                           picek_release=BASE, picek_release_manifest=sha,
                           picek_allow_draft=True, log=lambda s: None)


def test_the_instance_verifies_the_shipped_release_and_records_it(world, tmp_path, monkeypatch):
    inst, sha = shipped_copy(world, tmp_path, monkeypatch)
    assert not (inst / "research-releases" / BASE / "private.sqlite").exists()
    trained = []
    try:
        run_job(world, inst, sha, trained)
    except Exception:                  # the toy stand-in's later stages are not the point
        pass
    assert trained, "the Picek stage ran on the verified release"
    progress = json.loads((inst / trainer.PROGRESS_FILE).read_text(encoding="utf-8"))
    rel = dataset.load(BASE, True, files=dataset.SHIPPED)      # the instance's copy
    assert progress["dataset_release"] == {**rel.cite(), "manifest_sha256": sha}


@pytest.mark.parametrize("how", ["altered", "other manifest"])
def test_the_instance_refuses_an_altered_or_other_release_before_its_first_stage(
        world, tmp_path, monkeypatch, how):
    inst, sha = shipped_copy(world, tmp_path, monkeypatch, alter=how == "altered")
    trained = []
    with pytest.raises(dr.ReleaseError):
        run_job(world, inst, sha if how == "altered" else "0" * 64, trained)
    assert trained == []
