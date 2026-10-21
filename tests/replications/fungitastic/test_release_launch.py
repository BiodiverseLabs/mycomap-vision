"""The Picek run on a dataset release (docs/dataset-release.md; releases made by the
builder, release_fixture): verified before anything else; the release is the source of
truth for records, labels, splits, photos, sources and benchmark exclusion; its
labels_hash is the run's labels identity; every release file but private.sqlite ships
(release.sqlite, MANIFEST.json, SHA256SUMS, DATASET_CARD.md), and the instance opens it
with the builder's trainer mode (load_release(without_private=True)) before its first
stage; the place prior's counts come from the release's own places, made on the laptop
and shipped; a test release is never trained on."""

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
    assert out["picek_labels"]["public_trainable_only"] is False
    shipped = tmp_path / "dry" / "release" / BASE
    files = dataset.shipped_files(rel)
    assert sorted(files) == sorted([dr.CARD, dr.MANIFEST, dr.RELEASE_DB, dr.SUMS])
    assert sorted(p.name for p in shipped.iterdir()) == sorted(files)
    script = (tmp_path / "dry" / "user-data.sh").read_text(encoding="utf-8")
    assert f"--picek-labels-hash {rel.labels_hash}" in script
    assert f"--picek-release {BASE} --picek-release-manifest {sha} --picek-allow-draft" in script
    for f in files:
        assert f"release/{BASE}/{f}" in script
    assert "private.sqlite" not in script
    # The place counts the instance trains its place prior on, made here.
    place = out["place_cells"]
    assert place["sha256"] == out["picek_labels"]["place_prior"]["cells_sha256"]
    assert f"--picek-place-cells data/{aws.PICEK_PLACE_KEY} " \
           f"--picek-place-cells-sha256 {place['sha256']}" in script
    doc = retrain.read_place_cells(tmp_path / "dry" / aws.PICEK_PLACE_KEY, place["sha256"])
    assert doc["dataset_release"] == BASE and "47" not in str(doc["cells"].values())


def test_the_trainer_copy_is_every_file_but_private_and_the_builder_checks_each(world,
                                                                               tmp_path,
                                                                               monkeypatch):
    rel = rel_of()
    assert dr.PRIVATE_DB not in dataset.shipped_files(rel)
    inst, _sha = shipped_copy(world, tmp_path, monkeypatch)
    dataset.load(BASE, True, without_private=True).close()
    with pytest.raises(dr.ReleaseError, match="private.sqlite"):   # the full check
        dataset.load(BASE, True)
    (inst / "research-releases" / BASE / dr.CARD).unlink()
    with pytest.raises(dr.ReleaseError, match="DATASET_CARD.md"):
        dataset.load(BASE, True, without_private=True)


def test_a_file_name_an_instance_script_cant_carry_is_refused(world, monkeypatch):
    rel = rel_of()
    monkeypatch.setitem(rel.manifest, "files",
                        [*rel.manifest["files"], {"path": "inputs/a b.csv"}])
    with pytest.raises(dr.ReleaseError, match="can't carry"):
        dataset.shipped_files(rel)


# --- the place prior ------------------------------------------------------------------------

def test_place_counts_come_from_the_releases_places_not_the_manifests(world):
    with world["conn"]:                          # the live manifest has moved on
        world["conn"].execute("update records set latitude = -33.0, longitude = 151.0")
    data = data_of(world, rel_of())
    assert data.place_source == retrain.PLACE_RELEASE
    assert list(data.cells) == [retrain.cell_of(47.0, -122.0, 4.0)]


def test_the_instance_trains_its_place_prior_on_the_counts_made_at_launch(world, tmp_path,
                                                                          monkeypatch):
    laptop = data_of(world, rel_of())
    doc = retrain.place_cells_doc(rel_of(), laptop)
    shipped_copy(world, tmp_path, monkeypatch)
    rel = dataset.load(BASE, True, without_private=True)
    assert not rel.has_private
    inst = retrain.build_data_from_release(world["conn"], rel, world["store"].location,
                                           "large", place_cells=doc)
    assert inst.place_source == retrain.PLACE_SHIPPED
    assert inst.cells == laptop.cells and inst.cells
    assert (retrain.release_snapshot(rel, inst)["place_prior"]["cells_sha256"]
            == retrain.place_cells_sha256(doc))


def test_without_private_or_shipped_counts_the_run_refuses_to_train(world, tmp_path,
                                                                     monkeypatch):
    shipped_copy(world, tmp_path, monkeypatch)
    rel = dataset.load(BASE, True, without_private=True)
    data = retrain.build_data_from_release(world["conn"], rel, world["store"].location, "large")
    assert data.place_source == retrain.PLACE_NONE and data.cells == {}
    assert retrain.place_cells_doc(rel, data) is None
    cfg = retrain.PicekConfig(release=BASE, allow_draft=True, shipped_release=True)

    def never(*a):
        raise AssertionError("must refuse before a model is made")
    with pytest.raises(ValueError, match="no place counts"):
        retrain.train(world["conn"], world["store"], "large", "picek-x", cfg,
                      out_dir=tmp_path / "models", log=lambda s: None, model_factory=never)


@pytest.mark.parametrize("break_it,match", [
    ("hash", "not the ones the launch made"), ("release", "not dry-"),
    ("grid", "degree grid"), ("species", "doesn't train")])
def test_shipped_place_counts_that_are_not_the_launchs_are_refused(world, tmp_path, break_it,
                                                                  match):
    rel = rel_of()
    doc = retrain.place_cells_doc(rel, data_of(world, rel))
    sha = retrain.place_cells_sha256(doc)
    if break_it == "hash":
        doc["cells"][next(iter(doc["cells"]))]["A x"] += 1
    elif break_it == "release":
        doc["dataset_release"] = "v0"
    elif break_it == "grid":
        doc["cell_degrees"] = 2.0
    else:
        doc["cells"][next(iter(doc["cells"]))]["Z z"] = 1
    path = tmp_path / "place.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        got = retrain.read_place_cells(path, sha if break_it == "hash" else None)
        pos = {"A x": 0, "B y": 1}
        retrain.cells_from_doc(got, rel, pos, 4.0)


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


def test_the_command_passes_allow_draft_to_the_launch(world, monkeypatch):
    """mv aws-launch-trainer --release <draft> --allow-draft reaches launch_trainer with
    allow_draft (it re-opens the release by id), so a draft dry run can run at all."""
    from mycomap_vision import cli, config
    monkeypatch.setattr(config, "MANIFEST_PATH", world["path"])
    monkeypatch.setattr(dr, "ROOT", world["root"])
    got = {}

    def fake_launch(conn, backbones, methods, **kw):
        got.update(kw)
        return {"dry_run": True}
    monkeypatch.setattr(aws, "launch_trainer", fake_launch)
    cli.main(["aws-launch-trainer", "--release", BASE, "--allow-draft", "--backbones", "none",
              "--finetune", "none", "--picek", "fungitastic-beit-b384@15", "--dry-run"])
    assert got["allow_draft"] is True and got["release"].id == BASE


# --- the instance -----------------------------------------------------------------------------

def shipped_copy(world, tmp_path, monkeypatch, alter=False):
    """The instance's data folder: the shipped release files only."""
    import shutil

    from mycomap_vision import config
    sha = dataset.manifest_sha256(rel_of())
    inst = tmp_path / "inst"
    folder = inst / "research-releases" / BASE
    folder.mkdir(parents=True)
    for f in dataset.shipped_files(rel_of()):
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
    rel = dataset.load(BASE, True, without_private=True)      # the instance's copy
    assert progress["dataset_release"] == {**rel.cite(), "manifest_sha256": sha}


@pytest.mark.parametrize("how", ["altered", "other manifest"])
def test_the_instance_refuses_an_altered_or_other_release_before_its_first_stage(
        world, tmp_path, monkeypatch, how):
    inst, sha = shipped_copy(world, tmp_path, monkeypatch, alter=how == "altered")
    trained = []
    with pytest.raises(dr.ReleaseError):
        run_job(world, inst, sha if how == "altered" else "0" * 64, trained)
    assert trained == []


# --- the recipe on the public model's photos only (paper E4-recipe; Steve, 2026-10-21) -------

def public_data(world, rel):
    return retrain.build_data_from_release(world["conn"], rel, world["store"].location, "large",
                                           public_trainable_only=True)


def test_a_public_run_trains_and_validates_on_public_trainable_photos_only(world):
    rel = rel_of()
    every, public = data_of(world, rel), public_data(world, rel)
    not_public = {int(r[0]) for r in rel.db.execute(
        "select manifest_photo_id from photos where included = 1 and public_trainable = 0")}
    assert not_public                                   # the release has an ARR photo
    ids = lambda d: {t[0] for t in d.train} | {v[0] for v in d.val}   # noqa: E731
    assert ids(every) & not_public                      # internal benchmarking keeps it
    assert not ids(public) & not_public                 # the public run never sees it
    # a record whose photos are all non-public drops out, in train and in val alike
    only_arr = {k for k in every.train_records + every.val_records
                if all(int(p["public_trainable"]) == 0 for p in rel.photos(k))}
    assert set(public.train_records + public.val_records) == \
        set(every.train_records + every.val_records) - only_arr
    # Seesaw's class counts and the month prior come from the photos kept
    assert int(public.class_photos.sum()) == len(public.train)
    assert int(public.month_counts.sum()) <= len(public.train) < len(every.train) + 1
    snap = retrain.release_snapshot(rel, public)
    assert snap["public_trainable_only"] is True
    assert snap["photo_licence_policy"] == retrain.PHOTO_LICENCE_POLICY_PUBLIC
    assert "arr" not in snap["photo_licences"]
    assert retrain.release_snapshot(rel, every)["public_trainable_only"] is False


def test_a_public_recipe_launch_ships_the_flag_and_the_public_counts(launchable, tmp_path):
    out = aws.launch_trainer(launchable, [], ["classifier"], max_hours=48,
                             instance_type="g6.xlarge", picek=["fungitastic-beit-b384@15"],
                             dry_run=True, dry_run_dir=tmp_path / "dry", release=BASE,
                             allow_draft=True, public_trainable_only=True, log=lambda s: None)
    assert out["picek_labels"]["public_trainable_only"] is True
    assert out["picek_labels"]["photo_licence_policy"] == retrain.PHOTO_LICENCE_POLICY_PUBLIC
    script = (tmp_path / "dry" / "user-data.sh").read_text(encoding="utf-8")
    assert "--public-trainable-only" in script


def test_public_trainable_only_without_a_fine_tune_or_the_recipe_is_refused(launchable,
                                                                            tmp_path):
    with pytest.raises(ValueError, match="--finetune or --picek"):
        aws.launch_trainer(launchable, ["bioclip-2"], ["nearest"], max_hours=48,
                           instance_type="g6.xlarge", dry_run=True,
                           dry_run_dir=tmp_path / "dry", release=BASE, allow_draft=True,
                           public_trainable_only=True, log=lambda s: None)


def test_the_recipes_estimate_counts_the_photos_it_trains_on_not_every_stored_photo():
    stages = trainer.plan_stages([], None, "r", ["fungitastic-beit-b384@15"])
    every = trainer.estimate(stages, 700_000)
    public = trainer.estimate(stages, 700_000, picek_photos=350_000)
    assert public["stages"][0]["hours"] == pytest.approx(every["stages"][0]["hours"] / 2, abs=0.1)
    assert public["stages"][1]["hours"] == every["stages"][1]["hours"]      # its embed: every photo
    assert "350,000 photos an epoch" in trainer.format_estimate(public)
    assert "photos an epoch" not in trainer.format_estimate(every)


@pytest.mark.parametrize("public", [False, True])
def test_a_recipe_launch_on_a_release_is_priced_on_the_photos_it_trains_on(launchable, tmp_path,
                                                                          monkeypatch, public):
    seen = {}
    real = trainer.estimate

    def spy(*a, **k):
        seen.update(k, photos=a[1])
        return real(*a, **k)
    monkeypatch.setattr(trainer, "estimate", spy)
    aws.launch_trainer(launchable, [], ["classifier"], max_hours=48, instance_type="g6.xlarge",
                       picek=["fungitastic-beit-b384@15"], dry_run=True,
                       dry_run_dir=tmp_path / "dry", release=BASE, allow_draft=True,
                       public_trainable_only=public, log=lambda s: None)
    rel = rel_of()
    assert seen["picek_photos"] == len(rel.photo_ids(("train", "val"),
                                                     public_trainable_only=public))
    if public:
        assert seen["picek_photos"] < len(rel.photo_ids(("train", "val")))
