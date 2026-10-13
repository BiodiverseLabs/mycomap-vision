"""Two trainer options for the paper (plan E22, E23; 2026-10-13):

- `--seed`: the fine-tune's random seed, so a second run measures the spread between runs;
- `--public-trainable-only`: the public model learns only from photos its released weights
  may learn from (CC including NC, or all-rights-reserved with the owner's permission); a
  record left with none drops out, and so does a class.

Both travel from the launch to the fine-tune on the instance, are recorded in the run
(progress.json, result.json) and the model, and are kept by --resume. Every fine-tune on a
release records the photos it learnt from by licence class.

The release is the two-species draft (test_trainer_release.cut) with photo 1000 made
all-rights-reserved with no answer, and 1001 all-rights-reserved with permission granted."""

import json

import pytest
from test_finetune import ft_loader
from test_trainer_release import cut, embed, launch_env, script_release  # noqa: F401
from test_trainer_stages import FakeEC2, FakeLaunchS3

from mycomap_vision import aws, cli, finetune, trainer
from mycomap_vision import dataset_release as dr


def relicensed(cut, name, arr_no_answer=(1000,), arr_granted=(1001,)):
    conn, path, _, _, root = cut
    for pid in arr_no_answer:
        conn.execute("update photos set license_class = 'arr', license_code = '', "
                     "owner_user_id = 66 where photo_id = ?", (pid,))
    for pid in arr_granted:
        conn.execute("update photos set license_class = 'arr', license_code = '', "
                     "owner_user_id = 55 where photo_id = ?", (pid,))
    conn.execute("insert into photo_permissions (inat_user_id, status) values (55, 'granted')")
    conn.commit()
    conn.execute("pragma wal_checkpoint(truncate)")
    dr.build(name=name, manifest_path=path, root=root, derive_photos="none",
             val_cutoff="2026-08-24", log=lambda s: None)
    return dr.load_release(name, root, allow_draft=True)


def trainset(cut, tmp_path, rel, **kw):
    conn, _, store, _, _ = cut
    embed(conn, store, tmp_path / "emb")
    return finetune.build_trainset(conn, "base", store.location, "large", 28,
                                   tmp_path / "emb", release=rel, **kw)


def test_the_public_model_learns_only_from_cc_or_granted_photos(cut, tmp_path):
    rel = relicensed(cut, "dry-20261013-200001")
    try:
        ts = trainset(cut, tmp_path, rel, public_trainable_only=True)
        assert {pid for pid, _ in ts.items} == {1001, 1002, 1003, 1004, 1005}
        assert ts.records == 5                      # record 100 had only its ARR photo
        assert ts.photo_policy == {"policy": "public_trainable", "photos": 5,
                                   "photos_by_licence_class": {"arr_permission_granted": 1,
                                                               "open": 4},
                                   "photos_left_out_by_policy": 1}
    finally:
        rel.close()


def test_every_fine_tune_on_a_release_records_the_licences_it_learnt_from(cut, tmp_path):
    rel = relicensed(cut, "dry-20261013-200002")
    try:
        ts = trainset(cut, tmp_path, rel)
        assert len(ts.items) == 6
        assert ts.photo_policy == {"policy": "all", "photos": 6,
                                   "photos_by_licence_class": {"arr": 1,
                                                               "arr_permission_granted": 1,
                                                               "open": 4},
                                   "photos_left_out_by_policy": 0}
    finally:
        rel.close()


def test_a_class_with_no_public_photo_is_not_in_the_public_model(cut, tmp_path):
    rel = relicensed(cut, "dry-20261013-200003", arr_no_answer=(1000, 1001, 1002),
                     arr_granted=())
    try:
        assert "A x" in trainset(cut, tmp_path, rel).names["species"]
        ts = trainset(cut, tmp_path, rel, public_trainable_only=True)
        assert ts.names["species"] == ["B y"] and ts.records == 3
    finally:
        rel.close()


def test_public_trainable_only_is_refused_without_a_release(cut, tmp_path):
    conn, _, store, _, _ = cut
    embed(conn, store, tmp_path / "emb")
    with pytest.raises(ValueError, match="give --release"):
        finetune.build_trainset(conn, "base", store.location, "large", 28, tmp_path / "emb",
                                public_trainable_only=True)
    with pytest.raises(ValueError, match="needs the shipped release"):
        trainer.run_job(conn, store, ["timm:a"], ["nearest"], lambda p, k: None, "r0",
                        data_dir=tmp_path / "inst", finetune=["timm:a"],
                        public_trainable_only=True, log=lambda s: None)


def test_the_seed_and_photo_policy_reach_the_fine_tune(monkeypatch):
    seen = {}

    def fake(conn, base, store, size, name, cfg=None, **kw):
        seen.update(seed=cfg.seed, public=kw["public_trainable_only"], release=kw["release"])
        return {"name": name}
    monkeypatch.setattr(finetune, "finetune", fake)
    run = trainer._default_finetuner("store", "large", 28, lambda s: None, release="REL",
                                     seed=3, public_trainable_only=True)
    run(None, "base", "m", "emb", "models")
    assert seen == {"seed": 3, "public": True, "release": "REL"}


def test_the_run_records_its_seed_and_photo_policy(cut, tmp_path):
    conn, _, store, rel, _ = cut
    uploads = {}

    def finetuner(conn, base, name, embeddings_root, models_dir):
        from test_finetune import fake_finetuner
        meta = fake_finetuner(conn, base, name, embeddings_root, models_dir)
        meta.update(seed=7, train_photos={"policy": "public_trainable", "photos": 6})
        return meta
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r7",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=finetuner, release=rel, seed=7,
                          public_trainable_only=True, log=lambda s: None)
    assert out["seed"] == 7 and out["public_trainable_only"] is True
    progress = json.loads(uploads["runs/r7/progress.json"])
    assert progress["seed"] == 7 and progress["public_trainable_only"] is True
    stage = next(s for s in progress["stages"] if s["kind"] == "finetune")
    assert stage["seed"] == 7 and stage["train_photos"]["policy"] == "public_trainable"
    again = trainer.resume_request(progress)
    assert again["seed"] == 7 and again["public_trainable_only"] is True
    old = trainer.resume_request({**progress, "seed": None, "public_trainable_only": None})
    assert old["seed"] == 0 and old["public_trainable_only"] is False   # a run from before


def test_the_instance_script_carries_the_seed_and_policy(cut, monkeypatch):
    rel = cut[3]
    kw = dict(finetune=["bioclip-2"], release=script_release(rel))
    ud = aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt", seed=2,
                                      public_trainable_only=True, **kw)
    job = ud[ud.index("aws-train-job"):]
    assert " --seed 2" in job and " --public-trainable-only" in job
    plain = aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt", **kw)
    assert "--seed" not in plain and "--public-trainable-only" not in plain
    # The instance's job command hands both to the run.
    from types import SimpleNamespace

    from mycomap_vision import storage
    seen = {}

    def run_job(conn, store, backbones, methods, upload, run_id, **kw):
        seen.update(seed=kw["seed"], public=kw["public_trainable_only"])
        return {"comparison_id": None}
    monkeypatch.setattr(trainer, "run_job", run_job)
    monkeypatch.setattr(storage, "S3Store", lambda source: SimpleNamespace(client=None,
                                                                           bucket="bkt"))
    args = cli.main.__globals__["argparse"].Namespace(
        source="s3://bkt", stop_after_hours=None, spot=False, backbones="b", methods="m",
        run_id="r", size="large", test_days=28, batch_size=8, readers=1, finetune="b",
        sample_records=None, picek="", picek_exclude_benchmarks="match",
        picek_exclude_ids=None, picek_labels_hash=None, picek_release=None,
        picek_release_manifest=None, picek_allow_draft=False, picek_place_cells=None,
        picek_place_cells_sha256=None, resume=False, release=None, seed=2,
        public_trainable_only=True)
    cli.cmd_aws_train_job(None, args)
    assert seen == {"seed": 2, "public": True}


def test_a_launch_ships_the_seed_and_policy_and_refuses_them_out_of_place(cut, tmp_path,
                                                                          monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    with pytest.raises(ValueError, match="give --release and --finetune"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           public_trainable_only=True, log=lambda s: None)
    with pytest.raises(ValueError, match="whole number"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"], seed=-1,
                           log=lambda s: None)
    assert ec2.launched is None
    lines = []
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13, release=rel.id,
                             allow_draft=True, seed=1, public_trainable_only=True,
                             log=lines.append)
    assert out["seed"] == 1 and out["public_trainable_only"] is True
    job = ec2.launched["UserData"][ec2.launched["UserData"].index("aws-train-job"):]
    assert " --seed 1" in job and " --public-trainable-only" in job
    assert any("public-trainable photos only: 6 of the train split's 6" in s for s in lines)
    assert any("fine-tune seed 1" in s for s in lines)


def test_a_trained_model_keeps_its_seed(conn, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("torchvision")
    from test_finetune import embedded
    from test_finetune_torch import Tiny
    store, root = embedded(conn, tmp_path, "tiny", Tiny())
    cfg = finetune.FinetuneConfig(blocks=1, batch_size=4, workers=0, max_steps=2, seed=5)
    meta = finetune.finetune(conn, "tiny", store, "large", "tiny-ft", cfg, 28, root,
                             tmp_path / "models", loader=lambda spec: Tiny(),
                             log=lambda s: None)
    saved = json.loads((tmp_path / "models" / "tiny-ft.json").read_text(encoding="utf-8"))
    assert meta["seed"] == saved["seed"] == saved["config"]["seed"] == 5
    assert "train_photos" not in saved                     # no release, no photo policy
