"""Specimen crops as EXTRA views of their records in a release fine-tune (the v2 beta, Steve
2026-11-01; decision O12: a crop never replaces a photo). A crop the Recovered labels queue
kept on a photo a person left out as a field slip becomes one more vector of its record in
everything that learns from or is compared against the train split, embedded by every embed
stage of the run, while the queries a model is judged on stay the same. It counts only while
the run's own exclusion list leaves its photo out as a slip. The crop list is pinned by its
sha256 from the launch to the instance, like the exclusion list.

And a run that trains on every record (no val split held out) is launched with --no-compare:
it makes no comparison and reports no accuracy (paper PLAN §8, seal guard rail a).

The release is the draft of test_trainer_release: records 100-105 train (photos 1000-1005),
106-107 val (photos 1006-1007), one 8x8 photo each."""
import hashlib
import io
import json
import shutil
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image
from test_finetune import fake_finetuner, ft_loader
from test_train_photo_exclusions import export
from test_trainer_release import FakeEC2, FakeLaunchS3, cut, launch_env  # noqa: F401

from mycomap_vision import aws, cli, config, finetune, trainer
from mycomap_vision import embed as embed_mod
from mycomap_vision import dataset_release as dr
from mycomap_vision.replications.fungitastic import dataset as release_files

CROP_COLS = ["photo_key", "manifest_photo_id", "record_key", "v1_split", "n", "l", "t", "r",
             "b", "decision"]


def crops_file(rel, path, rows):
    """A crop export (mv feedback export-photo-crops) of (photo id, n[, record, box])."""
    lines = ["\t".join(CROP_COLS)]
    for pid, n, *rest in rows:
        rk = rest[0] if rest else rel.db.execute(
            "select record_key from photos where manifest_photo_id = ?", (pid,)).fetchone()[0]
        box = rest[1] if len(rest) > 1 else (0.0, 0.0, 0.5, 0.5)
        key = rel.db.execute("select photo_key from photos where manifest_photo_id = ?",
                             (pid,)).fetchone()
        key = key[0] if key else f"inat:{pid}"
        lines.append("\t".join([key, str(pid), rk, "train", str(n), *map(str, box), "default"]))
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def with_slips(rel, tmp_path, slips=(1000,), others=()):
    """The exclusion list applied: `slips` left out as field slips, `others` as habitat."""
    path, sha = export(rel, tmp_path / "x.tsv",
                       [(p, "exclude_photo", "slip") for p in slips]
                       + [(p, "exclude_photo", "habitat") for p in others])
    rel.use_photo_exclusions(path, sha)
    return path, sha


def vid(pid, n):
    return embed_mod.crop_vector_id(pid, n)


def test_the_release_numbers_crops_as_embed_does():
    assert dr.CROP_SLOTS == embed_mod.CROP_SLOTS
    assert dr.SLIP_REASON == "slip"


# --- the release reader ------------------------------------------------------------------

def test_a_crop_of_a_slip_photo_is_an_extra_view_and_keeps_its_record_in(cut, tmp_path):
    rel = cut[3]
    with_slips(rel, tmp_path, slips=(1000, 1001))
    # Record 100's only photo is a slip: without its crop the record drops out of training.
    assert "100" not in {r.observation_id for r in rel.as_records("train", {})}
    path, sha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1), (1000, 2), (1001, 1)])
    got = rel.use_photo_crops(path, sha)
    assert got == {"file": "c.tsv", "sha256": sha, "crops_listed": 3, "crops_used": 3,
                   "photos": 2, "records": 2, "records_kept_by_crops_alone": 2, "not_used": {}}
    rows = {p: p for p in [1002, 1003, 1004, 1005, vid(1000, 1), vid(1000, 2), vid(1001, 1)]}
    train = {r.observation_id: r.photo_rows for r in rel.as_records("train", rows)}
    assert train["100"] == [vid(1000, 1), vid(1000, 2)]
    assert train["101"] == [vid(1001, 1)]
    assert train["102"] == [1002]
    assert rel.cite()["photo_crops"] == got
    assert release_files.cite(rel)["photo_crops"]["sha256"] == sha


def test_crops_never_replace_or_drop_a_photo_and_add_nothing_to_val_queries(cut, tmp_path):
    rel = cut[3]
    with_slips(rel, tmp_path, slips=(1000, 1006))
    path, sha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1), (1006, 1)])
    before_photos = rel.photo_ids()
    val_before = [(r.observation_id, r.photo_rows) for r in rel.as_records("val", None)]
    rel.use_photo_crops(path, sha)
    assert rel.photo_ids() == before_photos                  # photos only: no crop in here
    rows = {p: p for p in range(1000, 1008)} | {vid(1000, 1): vid(1000, 1),
                                                vid(1006, 1): vid(1006, 1)}
    # The val queries keep their own photos, slip or not, and gain no crop.
    assert [(r.observation_id, r.photo_rows) for r in rel.as_records("val", rows)] == \
        [("106", [1006]), ("107", [1007])]
    assert [(r.observation_id, r.photo_rows) for r in rel.as_records("val", None)] == val_before
    # The train + val reference holds the val record's crop, not its slip photo.
    ref = {r.observation_id: r.photo_rows for r in rel.as_records(("train", "val"), rows)}
    assert ref["106"] == [vid(1006, 1)] and ref["100"] == [vid(1000, 1)]


def test_a_crop_counts_only_while_this_runs_list_leaves_its_photo_out_as_a_slip(cut, tmp_path):
    rel = cut[3]
    with_slips(rel, tmp_path, slips=(1000,), others=(1001,))
    path, sha = crops_file(rel, tmp_path / "c.tsv", [
        (1000, 1),                       # used
        (1001, 1),                       # left out, but as habitat
        (1002, 1),                       # not left out at all
        (1003, 1, "104"),                # names another record than its photo's
        (999999, 1, "100")])             # not a photo of this release
    got = rel.use_photo_crops(path, sha)
    assert got["crops_used"] == 1 and got["not_used"] == {
        "other_record": 1, "photo_left_out_not_as_slip": 1, "photo_not_in_release": 1,
        "photo_not_left_out": 1}
    assert rel.crop_rows() == [(vid(1000, 1), 1000, (0.0, 0.0, 0.5, 0.5))]


def test_an_arm_that_keeps_the_slips_takes_no_crop(cut, tmp_path):
    rel = cut[3]
    path, sha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo", "slip"),
                                                 (1001, "exclude_photo", "micrograph")])
    rel.use_photo_exclusions(path, sha, ["micrograph"])
    cpath, csha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1)])
    got = rel.use_photo_crops(cpath, csha)
    assert got["crops_used"] == 0 and got["not_used"] == {"photo_not_left_out": 1}
    # Applying another exclusion list later drops the crops chosen against the old one.
    rel.use_photo_exclusions(path, sha)
    assert rel.photo_crops is None and rel.crop_rows() == []


def test_a_crop_list_is_refused_unpinned_badly_formed_or_before_the_exclusions(cut, tmp_path):
    rel = cut[3]
    path, sha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1)])
    with pytest.raises(dr.ReleaseError, match="exclusion list first"):
        rel.use_photo_crops(path, sha)
    with_slips(rel, tmp_path)
    with pytest.raises(dr.ReleaseError, match="pinned"):
        rel.use_photo_crops(path, "0" * 64)
    for rows, why in (([(1000, 1), (1000, 1)], "twice"),
                      ([(1000, 0)], "usable box"),
                      ([(1000, 16)], "usable box"),
                      ([(1000, 1, "100", (0.6, 0.0, 0.5, 0.5))], "usable box"),
                      ([(1000, 1, "100", (0.0, 0.0, 1.2, 0.5))], "usable box")):
        bad, bsha = crops_file(rel, tmp_path / "bad.tsv", rows)
        with pytest.raises(dr.ReleaseError, match=why):
            rel.use_photo_crops(bad, bsha)
    (tmp_path / "cols.tsv").write_bytes(b"photo_key\tn\ninat:1000\t1\n")
    with pytest.raises(dr.ReleaseError, match="columns"):
        rel.use_photo_crops(tmp_path / "cols.tsv",
                            hashlib.sha256((tmp_path / "cols.tsv").read_bytes()).hexdigest())
    assert rel.photo_crops is None


# --- what the fine-tune learns from ---------------------------------------------------------

def embed_with_crops(conn, store, rel, root, name="base"):
    trainer.write_release_crops(conn, rel)
    todo = embed_mod.photos_to_embed(conn, name, "large", store.location, "local")
    todo += embed_mod.crops_to_embed(conn, name, "large", store.location)
    embed_mod.embed_photos(conn, store, ft_loader(name), todo, root / name, log=lambda s: None)


def test_the_fine_tune_trains_on_a_crop_as_its_photo_cut_to_the_box(cut, tmp_path):
    conn, _, store, rel, _ = cut
    with_slips(rel, tmp_path, slips=(1000,))
    path, sha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1, "100", (0.25, 0.0, 0.75, 0.5))])
    rel.use_photo_crops(path, sha)
    embed_with_crops(conn, store, rel, tmp_path / "emb")
    ts = finetune.build_trainset(conn, "base", store.location, "large", 28, tmp_path / "emb",
                                 release=rel)
    by_id = {it[0]: it for it in ts.items}
    assert 1000 not in by_id                                 # the slip photo itself: never
    photo_path = conn.execute("select path from photo_copies where photo_id = 1000 and "
                              "store = ?", (store.location,)).fetchone()[0]
    assert by_id[vid(1000, 1)] == (vid(1000, 1), photo_path, (0.25, 0.0, 0.75, 0.5))
    assert len(ts.items) == 6 and ts.records == 6            # record 100 kept by its crop
    ids, _ = embed_mod.load_embeddings(conn, "base", tmp_path / "emb" / "base")
    i = list(ids.tolist()).index(vid(1000, 1))
    assert ts.base_rows[[it[0] for it in ts.items].index(vid(1000, 1))] == i
    assert ts.photo_policy["photos"] == 5
    assert ts.photo_policy["photo_crops"]["crops_trained_on"] == 1
    assert ts.photo_policy["photo_crops"]["sha256"] == sha


def test_a_training_crop_is_read_whole_and_cut_to_its_box(tmp_path):
    from mycomap_vision.storage import LocalStore
    store = LocalStore(tmp_path / "s")
    img = Image.new("RGB", (8, 4), (0, 0, 255))
    img.paste((255, 0, 0), (0, 0, 4, 4))                      # left half red
    buf = io.BytesIO()
    img.save(buf, "PNG")
    store.put("p.png", buf.getvalue())
    ds = finetune.PhotoDataset(store.location, [(1, "p.png"), (-17, "p.png", (0, 0, 0.5, 1))],
                               np.zeros((2, 3), dtype=np.int64), lambda im: im, draft=4)
    whole, crop = ds[0][0], ds[1][0]
    assert whole.size == (8, 4) and crop.size == (4, 4)
    assert crop.getpixel((1, 1)) == (255, 0, 0) and crop.getpixel((3, 3)) == (255, 0, 0)


# --- the run on the instance -------------------------------------------------------------

def test_a_run_embeds_the_crops_in_every_stage_and_with_no_compare_scores_nothing(cut,
                                                                                  tmp_path):
    conn, _, store, rel, _ = cut
    with_slips(rel, tmp_path, slips=(1000, 1001))
    path, sha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1), (1000, 2), (1001, 1)])
    rel.use_photo_crops(path, sha)
    uploads = {}
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"],
                          lambda p, k: uploads.__setitem__(k, p.read_bytes()), "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=fake_finetuner, release=rel, no_compare=True,
                          log=lambda s: None)
    assert out["state"] == "finished" and out["comparison_id"] is None
    assert not any("/reports/compare-" in k for k in uploads)
    progress = json.loads(uploads["runs/r1/progress.json"])
    assert "no accuracy" in progress["comparison"]
    assert progress["dataset_release"]["photo_crops"]["sha256"] == sha
    crops = {vid(1000, 1), vid(1000, 2), vid(1001, 1)}
    for name in ("timm_a", "timm_a-ft-r1"):
        have = {r[0] for r in conn.execute("select photo_id from embeddings where backbone = ?",
                                           (name,))}
        assert crops <= have and set(range(1000, 1008)) <= have
        assert out["embedded"][name]["photos"] == 11          # 8 photos + 3 crops


def test_a_run_without_crops_embeds_none_even_if_the_copy_holds_a_crop_table(cut, tmp_path):
    conn, _, store, rel, _ = cut
    from mycomap_vision.feedback import ensure_crops_schema
    ensure_crops_schema(conn)
    conn.execute("insert into feedback_photo_crops values (1000, 1, 'inat:1000', 0, 0, .5, .5, "
                 "'v')")
    conn.commit()
    out = trainer.run_job(conn, store, ["timm:a"], ["nearest"], lambda p, k: None, "r1",
                          loader=ft_loader, data_dir=tmp_path / "inst", finetune=["timm:a"],
                          finetuner=fake_finetuner, release=rel, log=lambda s: None)
    assert out["comparison_id"] is not None                   # compared, as before
    assert conn.execute("select count(*) from embeddings where photo_id < 0").fetchone()[0] == 0


def test_the_instance_applies_the_crops_only_with_the_launchs_hash(cut, tmp_path, monkeypatch):
    rel, root = cut[3], cut[4]
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "inst")
    dest = tmp_path / "inst" / release_files.RELEASES_DIR / rel.id
    dest.mkdir(parents=True)
    for f in release_files.shipped_files(rel):
        shutil.copyfile(root / rel.id / f, dest / f)
    xpath, xsha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo", "slip")])
    cpath, csha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1)])
    args = SimpleNamespace(release=rel.id, allow_draft=True,
                           release_manifest=release_files.manifest_sha256(rel),
                           exclude_photos=str(xpath), exclude_photos_sha256=xsha,
                           exclude_photo_reasons="", photo_crops=str(cpath),
                           photo_crops_sha256="f" * 64)
    with pytest.raises(dr.ReleaseError, match="pinned"):
        cli.trainer_release(args)
    args.photo_crops_sha256 = csha
    got = cli.trainer_release(args)
    assert got.crop_rows() == [(vid(1000, 1), 1000, (0.0, 0.0, 0.5, 0.5))]
    assert got.cite()["photo_crops"]["sha256"] == csha
    got.close()


# --- the launcher ------------------------------------------------------------------------

def test_a_launch_ships_the_crops_pinned_and_tells_the_instance(cut, tmp_path, monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    xpath, xsha = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo", "slip")])
    cpath, csha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1), (1000, 2)])
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13, release=rel.id,
                             allow_draft=True, exclude_photos=xpath, photo_crops=cpath,
                             log=lambda s: None)
    run = out["run_id"]
    assert s3.objects[f"runs/{run}/photo-crops.tsv"] == cpath.read_bytes()
    ud = ec2.launched["UserData"]
    assert '$RUN/photo-crops.tsv" data/photo-crops.tsv || exit 1' in ud
    assert f"--photo-crops data/photo-crops.tsv --photo-crops-sha256 {csha}" in ud
    assert "--no-compare" not in ud
    assert out["dataset_release"]["photo_crops"]["crops_used"] == 2


def test_a_crop_list_is_refused_without_the_exclusions_or_when_nothing_applies(cut, tmp_path,
                                                                               monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    xpath, _ = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo", "slip"),
                                                (1001, "exclude_photo", "habitat")])
    cpath, _ = crops_file(rel, tmp_path / "c.tsv", [(1000, 1)])
    with pytest.raises(ValueError, match="--exclude-photos"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           release=rel.id, allow_draft=True, photo_crops=cpath,
                           log=lambda s: None)
    with pytest.raises(ValueError, match="nothing to add"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           release=rel.id, allow_draft=True, exclude_photos=xpath,
                           exclude_photo_reasons=["habitat"], photo_crops=cpath,
                           log=lambda s: None)
    assert ec2.launched is None and not s3.objects
    script = {**release_files.cite(rel), "shipped": list(release_files.shipped_files(rel))}
    with pytest.raises(ValueError, match="exclusion list"):
        aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                     finetune=["bioclip-2"], release=script,
                                     photo_crops_sha256="a" * 64)
    with pytest.raises(ValueError, match="hash"):
        aws.render_trainer_user_data("r", ["bioclip-2"], ["nearest"], 10, "bkt",
                                     finetune=["bioclip-2"], release=script,
                                     photo_exclusions_sha256="a" * 64,
                                     photo_crops_sha256="x; rm -rf /")


def test_a_release_with_no_val_split_launches_only_with_no_compare(cut, tmp_path, monkeypatch):
    conn, path, _, _, root = cut
    # Every record train: a cutoff after the newest validation (the v2 beta's shape).
    dr.build(name="dry-20261101-211500", manifest_path=path, root=root, derive_photos="none",
             val_cutoff="2026-11-01", log=lambda s: None)
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    with pytest.raises(ValueError, match="no val record"):
        aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                           instance_type="g6.xlarge", max_hours=13,
                           release="dry-20261101-211500", allow_draft=True, log=lambda s: None)
    assert ec2.launched is None
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], finetune=["bioclip-2"],
                             instance_type="g6.xlarge", max_hours=13,
                             release="dry-20261101-211500", allow_draft=True, no_compare=True,
                             log=lambda s: None)
    assert out["no_compare"] is True
    assert ec2.launched["UserData"].rstrip().endswith("--no-compare")


def test_a_dry_run_writes_the_crops_and_counts_them_in_the_estimate(cut, tmp_path, monkeypatch):
    conn, _, _, rel, root = cut
    ec2, s3 = FakeEC2(), FakeLaunchS3()
    launch_env(monkeypatch, tmp_path, root, s3, ec2)
    monkeypatch.setattr(aws, "check_code_archive", lambda body, picek=None: {})
    xpath, _ = export(rel, tmp_path / "x.tsv", [(1000, "exclude_photo", "slip")])
    cpath, csha = crops_file(rel, tmp_path / "c.tsv", [(1000, 1), (1000, 2), (1000, 3)])
    kw = dict(finetune=["bioclip-2"], instance_type="g6.xlarge", max_hours=13,
              release=rel.id, allow_draft=True, exclude_photos=xpath, dry_run=True,
              log=lambda s: None)
    out = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], photo_crops=cpath,
                             no_compare=True, dry_run_dir=tmp_path / "dry", **kw)
    assert (tmp_path / "dry" / "photo-crops.tsv").read_bytes() == cpath.read_bytes()
    assert f"--photo-crops-sha256 {csha}" in (tmp_path / "dry" / "user-data.sh").read_text()
    plain = aws.launch_trainer(conn, ["bioclip-2"], ["nearest"], dry_run_dir=tmp_path / "dry2",
                               **kw)
    def trained(est):
        return next(r["photos"] for r in est["stages"] if r["stage"].startswith("finetune"))
    # The embed stages embed all 8 photos (the slip one too) and the crops; the fine-tune
    # learns from the 5 train photos left in and the crops.
    assert out["estimate"]["photos"] == 11 and trained(out["estimate"]) == 8
    assert plain["estimate"]["photos"] == 8 and trained(plain["estimate"]) == 5
    assert out["estimate"]["compare_hours"] == 0 and plain["estimate"]["compare_hours"] > 0
    assert not s3.objects and ec2.launched is None


def test_the_estimate_times_the_fine_tune_on_what_it_learns_from():
    stages = trainer.plan_stages(["bioclip-2.5-336"], ["bioclip-2.5-336"], "r")
    est = trainer.estimate(stages, 700_000, train_photos=600_000, compare=False)
    rows = dict(zip(["embed", "finetune", "embed ft"], est["stages"]))
    same = trainer.estimate(stages, 700_000, compare=False)
    assert rows["finetune"]["photos"] == 600_000
    assert rows["finetune"]["hours"] == pytest.approx(
        next(r["hours"] for r in same["stages"] if r["stage"].startswith("finetune"))
        * 6 / 7, abs=0.1)
    assert rows["embed"]["hours"] == rows["embed ft"]["hours"] == next(
        r["hours"] for r in same["stages"] if r["stage"].startswith("embed"))
    assert est["compare_hours"] == 0 and same["total_hours"] > est["total_hours"]
