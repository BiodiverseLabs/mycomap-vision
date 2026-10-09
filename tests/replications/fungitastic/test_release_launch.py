"""The Picek run on a dataset release (docs/dataset-release.md): the release is verified
before anything else, it is the source of truth for records, labels, splits, photos,
allowed sources and benchmark exclusion, its labels_hash is the run's labels identity,
only what training needs ships (never private.sqlite), and the instance verifies the
shipped files again before its first stage."""

import json

import pytest
from release_fixture import make_release
from test_models_and_scoreboard import seed_two_species
from test_record_sources import launch, launchable  # noqa: F401  (fixture)
from test_retrain_trainer import fake_picek_trainer

from mycomap_vision import config, trainer
from mycomap_vision.replications.fungitastic import dataset, retrain

RID = "dry-20261009-1830"


@pytest.fixture
def data_dir(tmp_path, monkeypatch):
    d = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", d)
    return d


def open_rel(data_dir, **kw):
    make_release(data_dir, RID, **kw)
    return dataset.load(RID, allow_draft=True)


# --- verification ---------------------------------------------------------------------------

def test_an_altered_release_file_is_refused(data_dir):
    folder = make_release(data_dir, RID)
    with open(folder / "release.sqlite", "ab") as f:
        f.write(b"x")
    with pytest.raises(dataset.ReleaseError, match="altered"):
        dataset.load(RID, allow_draft=True)


def test_a_draft_needs_allow_draft_and_an_unfinished_release_is_refused(data_dir):
    folder = make_release(data_dir, RID)
    with pytest.raises(dataset.ReleaseError, match="allow-draft"):
        dataset.load(RID)
    assert dataset.load(RID, allow_draft=True).draft
    (folder / "MANIFEST.json").unlink()
    with pytest.raises(dataset.ReleaseError, match="not a finished release"):
        dataset.load(RID, allow_draft=True)


def test_another_manifest_than_the_launchs_is_refused(data_dir):
    rel = open_rel(data_dir)
    with pytest.raises(dataset.ReleaseError, match="not the one the launch recorded"):
        dataset.load(RID, True, expect_manifest_sha256="0" * 64)
    assert dataset.load(RID, True, expect_manifest_sha256=rel.manifest_sha256).id == RID


# --- what trains ----------------------------------------------------------------------------

def test_training_takes_the_releases_train_split_and_validates_on_its_val_split(
        conn, tmp_path, data_dir):
    store = seed_two_species(conn, tmp_path)
    rel = open_rel(data_dir)
    data = retrain.build_data_from_release(conn, rel, store.location, "large")
    assert sorted(data.train_records) == ["100", "101", "103", "104"]
    assert sorted(data.val_records) == ["102", "105"]
    assert not {"106", "107"} & set(data.train_records + data.val_records)    # test never
    # The month and place prior counts come from the train split only.
    assert int(data.month_counts.sum()) == len(data.train) == 4
    assert data.trained_through == "2026-01-03" and data.train_through == "2026-01-02"
    assert data.photo_licences == {"arr": 3, "open": 3}


def test_a_test_or_sealed_record_in_the_training_data_raises(conn, tmp_path, data_dir,
                                                             monkeypatch):
    store = seed_two_species(conn, tmp_path)
    rel = open_rel(data_dir, splits={"107": "sealed"})
    real = rel.records

    def leaky(split):          # a reader bug that hands a test record out as training data
        return real(split) + (real("test") if split == "train" else [])
    monkeypatch.setattr(rel, "records", leaky)
    with pytest.raises(dataset.SealedLeak):
        retrain.build_data_from_release(conn, rel, store.location, "large")


def test_a_photo_that_differs_from_the_releases_original_is_refused(conn, tmp_path, data_dir):
    store = seed_two_species(conn, tmp_path)
    with conn:
        conn.execute("update photo_copies set sha256 = 'other' where photo_id = 1000")
    rel = open_rel(data_dir)
    with pytest.raises(ValueError, match="differ from release"):
        retrain.build_data_from_release(conn, rel, store.location, "large")


def test_benchmark_exclusion_follows_the_release_and_a_mismatch_is_refused(data_dir):
    rel = open_rel(data_dir, benchmark={"100": "heldout-2026-10-08", "106": "heldout-x"})
    retrain.check_release_exclusion(rel, "match")
    with pytest.raises(ValueError, match="trains on 1 held-out benchmark"):
        retrain.check_release_exclusion(rel, "all")


def test_all_is_accepted_when_the_release_trains_on_no_benchmark_record(tmp_path, monkeypatch):
    d = tmp_path / "d"
    monkeypatch.setattr(config, "DATA_DIR", d)
    make_release(d, RID, benchmark={"106": "heldout-x"})          # only a test record
    retrain.check_release_exclusion(dataset.load(RID, True), "all")


# --- allowed sources come from the release ---------------------------------------------------

def test_the_releases_allowed_sources_admit_its_mo_records(conn, tmp_path, data_dir):
    store = seed_two_species(conn, tmp_path)
    rel = open_rel(data_dir, sources={"100": "mo"})
    snap = retrain.release_snapshot(rel, retrain.build_data_from_release(
        conn, rel, store.location, "large"))
    assert snap["allowed_sources"] == ["inat", "mo"]
    assert snap["record_sources"]["by_source"] == {"inat": 5, "mo": 1}
    retrain.require_allowed_sources(snap["record_sources"])


def test_a_release_allowing_inat_only_refuses_its_mo_record(conn, tmp_path, data_dir):
    store = seed_two_species(conn, tmp_path)
    rel = open_rel(data_dir, sources={"100": "mo"}, allowed_sources=("inat",))
    snap = retrain.release_snapshot(rel, retrain.build_data_from_release(
        conn, rel, store.location, "large"))
    with pytest.raises(retrain.RecordSourceError, match="mo: 1"):
        retrain.require_allowed_sources(snap["record_sources"])


def test_a_release_without_allowed_sources_or_with_a_forbidden_one_is_refused(data_dir,
                                                                               tmp_path,
                                                                               monkeypatch):
    rel = open_rel(data_dir, omit=("allowed_sources",))
    with pytest.raises(dataset.ReleaseError, match="names no allowed_sources"):
        retrain.allowed_sources(release=rel)
    d = tmp_path / "d2"
    monkeypatch.setattr(config, "DATA_DIR", d)
    make_release(d, RID, allowed_sources=("inat", "mycoportal"))
    with pytest.raises(retrain.RecordSourceError, match="non-empty subset"):
        retrain.allowed_sources(release=dataset.load(RID, True))


def test_the_snapshot_carries_the_releases_identity_and_the_photo_licence_policy(
        conn, tmp_path, data_dir):
    store = seed_two_species(conn, tmp_path)
    rel = open_rel(data_dir)
    snap = retrain.release_snapshot(rel, retrain.build_data_from_release(
        conn, rel, store.location, "large"))
    assert snap["labels_hash"] == rel.labels_hash != snap["own_labels_hash"]
    assert snap["dataset_release"]["manifest_sha256"] == rel.manifest_sha256
    assert snap["dataset_release"]["dataset_release"] == RID
    assert "arr" in snap["photo_licence_policy"] and snap["photo_licences"]["arr"] == 3
    assert "val split" in snap["validation"]
    assert "dataset release " + RID in retrain.format_snapshot(snap)


# --- the launch ----------------------------------------------------------------------------

def release_launch(conn, tmp_path, release=RID, allow_draft=True, dry_run=True, said=None):
    from mycomap_vision import aws
    return aws.launch_trainer(conn, [], ["classifier"], max_hours=48, instance_type="g6.xlarge",
                              picek=["fungitastic-beit-b384@15"], dry_run=dry_run,
                              dry_run_dir=tmp_path / "dry", release=release,
                              allow_draft=allow_draft,
                              log=(said.append if said is not None else lambda s: None))


def test_a_dry_run_on_a_release_ships_what_training_needs_and_never_private(
        launchable, tmp_path, monkeypatch):   # noqa: F811
    d = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", d)
    make_release(d, RID)
    out = release_launch(launchable, tmp_path)
    rel = dataset.load(RID, True)
    assert out["dataset_release"]["manifest_sha256"] == rel.manifest_sha256
    assert out["picek_labels"]["labels_hash"] == rel.labels_hash
    shipped = tmp_path / "dry" / "release" / RID
    assert sorted(p.name for p in shipped.iterdir()) == sorted(dataset.SHIPPED)
    script = (tmp_path / "dry" / "user-data.sh").read_text(encoding="utf-8")
    assert f"--picek-labels-hash {rel.labels_hash}" in script
    assert (f"--picek-release {RID} --picek-release-manifest {rel.manifest_sha256} "
            "--picek-allow-draft") in script
    for f in dataset.SHIPPED:
        assert f"release/{RID}/{f}" in script
    assert "private.sqlite" not in script


@pytest.mark.parametrize("break_it,match", [
    ("draft", "allow-draft"), ("alter", "altered"), ("benchmark-all", "held-out benchmark")])
def test_a_launch_refuses_a_bad_release_before_anything_is_sent(launchable, tmp_path,
                                                                monkeypatch, break_it, match):  # noqa: F811
    d = tmp_path / "data"
    monkeypatch.setattr(config, "DATA_DIR", d)
    folder = make_release(d, RID, benchmark={"100": "heldout-2026-10-08"})
    if break_it == "alter":
        with open(folder / "release.sqlite", "ab") as f:
            f.write(b"x")
    from mycomap_vision import aws
    with pytest.raises(ValueError, match=match):
        aws.launch_trainer(launchable, [], ["classifier"], max_hours=48,
                           picek=["fungitastic-beit-b384@15"], dry_run=False,
                           picek_exclude="all" if break_it == "benchmark-all" else "match",
                           release=RID, allow_draft=break_it != "draft", log=lambda s: None)


# --- the instance -----------------------------------------------------------------------------

def instance_release(tmp_path, monkeypatch, alter=False):
    """The shipped copy: only release.sqlite, MANIFEST.json, SHA256SUMS."""
    d = tmp_path / "inst"
    monkeypatch.setattr(config, "DATA_DIR", d)
    folder = make_release(d, RID)
    (folder / "private.sqlite").unlink()          # never shipped
    sha = dataset.load(RID, True, files=dataset.SHIPPED).manifest_sha256
    if alter:
        with open(folder / "release.sqlite", "ab") as f:
            f.write(b"x")
    return d, sha


def run_release_job(conn, store, data_dir, sha, trained):
    def picek_trainer(*a, **k):
        trained.append(a)
        return fake_picek_trainer(*a, **k)
    return trainer.run_job(conn, store, [], ["classifier"], lambda p, k: None, "r1",
                           data_dir=data_dir, picek=["fungitastic-beit-b384@2"],
                           picek_trainer=picek_trainer, picek_release=RID,
                           picek_release_manifest=sha, picek_allow_draft=True,
                           log=lambda s: None)


def test_the_instance_verifies_the_shipped_release_and_trains_on_it(conn, tmp_path, monkeypatch):
    store = seed_two_species(conn, tmp_path)
    d, sha = instance_release(tmp_path, monkeypatch)
    trained = []
    run_release_job(conn, store, d, sha, trained)
    assert trained
    progress = json.loads((d / trainer.PROGRESS_FILE).read_text(encoding="utf-8"))
    assert progress["dataset_release"] == RID and progress["dataset_release_manifest"] == sha


@pytest.mark.parametrize("how", ["altered", "other manifest"])
def test_the_instance_refuses_an_altered_or_other_release_before_its_first_stage(
        conn, tmp_path, monkeypatch, how):
    store = seed_two_species(conn, tmp_path)
    d, sha = instance_release(tmp_path, monkeypatch, alter=how == "altered")
    trained = []
    with pytest.raises(dataset.ReleaseError):
        run_release_job(conn, store, d, sha if how == "altered" else "0" * 64, trained)
    assert trained == []
