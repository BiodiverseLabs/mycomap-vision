"""A sealed test release and its base (paper plan §4, 2026-10-13): a shared record refuses
the build; a test photo byte-identical to a base photo is dropped and counted (a record left
with none drops out); a model-free dHash FLAGS near-copies of base photos without dropping
anything, against an index of the base made once (in-region for a large base) and checked
before use. A release without flags keeps the hashes it was made with."""

import io
import json

import numpy as np
import pytest
from PIL import Image
from test_aws_dataset import BucketS3, launch_env
from test_dataset_release import add_photo, add_record, build, world  # noqa: F401
from test_trainer_stages import FakeEC2

from mycomap_vision import aws_dataset, dhash
from mycomap_vision import dataset_release as dr
from mycomap_vision import dataset_s3

FREEZE = "Steve 2026-10-26 (test)"


def blocks(seed: int, size=(640, 480)) -> Image.Image:
    """A photo with a strong, seed-specific dHash: 9 x 8 random blocks, blown up."""
    rng = np.random.default_rng(seed)
    small = Image.fromarray(rng.integers(0, 256, (8, 9, 3), dtype=np.uint8))
    return small.resize(size, Image.Resampling.NEAREST)


def encode(img: Image.Image, quality=95) -> bytes:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    return buf.getvalue()


def near_copy(img: Image.Image) -> bytes:
    """What a re-post looks like: smaller and re-compressed (other bytes, same picture)."""
    return encode(img.resize((320, 240), Image.Resampling.LANCZOS), quality=70)


def add_body(world, key, pid, body):
    add_photo(world["conn"], world["store"], key, pid, copy=False)
    rel = world["store"].put(pid, body)
    world["conn"].execute("insert into photo_copies values (?, ?, 'large', ?, ?, ?, 'now')",
                          (pid, str(world["store"].root), rel, len(body),
                           dr.sha256_bytes(body)))


def commit(world):
    world["conn"].commit()
    world["conn"].execute("pragma wal_checkpoint(truncate)")


def ids_file(tmp, *keys):
    p = tmp / f"ids-{'-'.join(k.replace(':', '_') for k in keys)}.csv"
    p.write_text("observation_id\n" + "\n".join(keys) + "\n", encoding="utf-8")
    return p


@pytest.fixture
def base(world):
    """A base release holding record 30, whose one photo has a strong dHash."""
    add_record(world["conn"], "30", "Russula emetica", vdate="2026-02-01")
    add_body(world, "30", 301, encode(blocks(1)))
    commit(world)
    m = build(world, name="dry-20261013-100000")
    return m["id"]


def index_of(world, rid):
    rel = dr.load_release(rid, world["root"], allow_draft=True)
    try:
        index = dhash.build_index(rel, dr.default_store_reader(), log=lambda s: None)
    finally:
        rel.close()
    return dhash.save_index(index, world["tmp"] / "extras" / rid / dhash.INDEX_FILE)


def test_a_byte_identical_copy_of_a_base_photo_is_dropped_and_counted(world, base):
    add_record(world["conn"], "40", "Russula emetica", vdate="2026-11-01")
    add_photo(world["conn"], world["store"], "40", 401, seed=11)     # photo 11's bytes
    add_photo(world["conn"], world["store"], "40", 402)              # its own photo
    commit(world)
    t = build(world, name="dry-20261013-110000-test", test_for=base,
              test_ids=ids_file(world["tmp"], "40"))
    assert t["photos_identical_to_base_dropped"] == 1
    assert t["counts"]["photos_by_source_reason"]["inat/copy_in_base"] == 1
    rel = dr.load_release(t["id"], world["root"], allow_draft=True)
    try:
        assert [r["record_key"] for r in rel.records("test")] == ["40"]
        assert [p["photo_key"] for p in rel.photos("40")] == ["inat:402"]
    finally:
        rel.close()


def test_a_test_record_whose_only_photos_are_copies_drops_out(world, base):
    add_record(world["conn"], "41", "Russula emetica", vdate="2026-11-01")
    add_photo(world["conn"], world["store"], "41", 411, seed=21)     # photo 21's bytes only
    add_record(world["conn"], "42", "Russula emetica", vdate="2026-11-01")
    add_photo(world["conn"], world["store"], "42", 421)
    commit(world)
    t = build(world, name="dry-20261013-120000-test", test_for=base,
              test_ids=ids_file(world["tmp"], "41", "42"))
    rel = dr.load_release(t["id"], world["root"], allow_draft=True)
    try:
        assert [r["record_key"] for r in rel.records("test")] == ["42"]
        row = rel.db.execute("select included, reason, split from records "
                             "where record_key = '41'").fetchone()
        assert tuple(row) == (0, "no_photos", None)
    finally:
        rel.close()


def test_a_record_already_in_the_base_still_refuses_the_whole_build(world, base):
    with pytest.raises(dr.SealedLeak, match="already"):
        build(world, name="dry-20261013-130000-test", test_for=base,
              test_ids=ids_file(world["tmp"], "1"))


def test_a_near_copy_is_flagged_and_kept_and_an_unrelated_photo_is_not(world, base):
    add_record(world["conn"], "50", "Russula emetica", vdate="2026-11-01")
    add_body(world, "50", 501, near_copy(blocks(1)))                 # re-post of photo 301
    add_body(world, "50", 502, encode(blocks(2)))                    # another picture
    commit(world)
    index = index_of(world, base)
    t = build(world, name="dry-20261013-140000-test", test_for=base, base_dhash=index,
              test_ids=ids_file(world["tmp"], "50"))
    nb = t["near_base"]
    assert nb["photos"] == 2 and nb["photos_flagged"] == 1 and nb["records_flagged"] == 1
    assert nb["index_sha256"] == dr.sha256_file(index) and nb["algorithm"] == dhash.ALGORITHM
    rel = dr.load_release(t["id"], world["root"], allow_draft=True)
    try:
        flags = rel.near_base()
        assert flags["inat:501"]["base_photo_key"] == "inat:301"
        assert 0 <= flags["inat:501"]["distance"] <= dhash.NEAR
        assert flags["inat:502"]["base_photo_key"] is None
        assert len(rel.photos("50")) == 2                            # nothing dropped
        assert [r["record_key"] for r in rel.records("test")] == ["50"]
    finally:
        rel.close()
    assert "near_base" in t["hashes"]
    assert dr.verify(t["id"], world["root"])["ok"]


def test_a_frozen_test_release_needs_its_bases_dhash_index(world, base):
    add_record(world["conn"], "51", "Russula emetica", vdate="2026-11-01")
    add_photo(world["conn"], world["store"], "51", 511)
    commit(world)
    with pytest.raises(dr.ReleaseError, match="needs --base-dhash"):
        build(world, name="v1-test", test_for=base, freeze_approved=FREEZE,
              test_ids=ids_file(world["tmp"], "51"))
    with pytest.raises(dr.ReleaseError, match="for a test release"):
        build(world, base_dhash=index_of(world, base))
    m = build(world, name="v1-test", test_for=base, freeze_approved=FREEZE,
              base_dhash=index_of(world, base), test_ids=ids_file(world["tmp"], "51"))
    assert m["near_base"]["photos"] == 1 and not m["draft"]


def test_an_index_of_another_release_or_pillow_or_missing_a_photo_is_refused(world, base):
    path = index_of(world, base)
    rel = dr.load_release(base, world["root"], allow_draft=True)
    try:
        good = dhash.load_index(path)
        dhash.check_index(good, rel)
        for change, match in (({"release_hash": "0" * 64}, "not dry-"),
                              ({"pillow": "1.0"}, "Pillow 1.0"),
                              ({"algorithm": "dhash32"}, "made with dhash32")):
            with pytest.raises(dhash.DhashError, match=match):
                dhash.check_index({**good, **change}, rel)
        short = {**good, "sha256": good["sha256"][1:]}
        with pytest.raises(dhash.DhashError, match="misses 1 included"):
            dhash.check_index(short, rel)
    finally:
        rel.close()


def test_a_release_without_flags_keeps_the_hash_it_was_made_with(world, base):
    """v1 was cut before near_base existed: its release_hash is over the seven original
    components, and must still verify."""
    rel = dr.load_release(base, world["root"], allow_draft=True)
    try:
        hashes = rel.manifest["hashes"]
        assert "near_base" not in hashes
        seven = dr.sha256_bytes("".join(f"{n}={hashes[n]}\n" for n in dr.COMPONENTS).encode())
        assert rel.release_hash == seven
    finally:
        rel.close()
    assert dr.verify(base, world["root"])["ok"]
    flagged = {**hashes, "near_base": "a" * 64}
    assert dr.release_hash(flagged) != dr.release_hash({**flagged, "near_base": "b" * 64})


# --- dHash itself -----------------------------------------------------------------------------

def test_dhash_survives_a_resize_and_recompression_and_tells_pictures_apart():
    a = dhash.dhash64(encode(blocks(1)))
    assert bin(a ^ dhash.dhash64(near_copy(blocks(1)))).count("1") <= dhash.NEAR
    others = [bin(a ^ dhash.dhash64(encode(blocks(s)))).count("1") for s in range(2, 30)]
    assert min(others) > dhash.NEAR
    assert dhash.dhash64(encode(blocks(1))) == a                    # the same every time


def test_nearest_flags_within_the_threshold_only_across_chunks():
    base = np.array([0, 0xFFFF_FFFF_FFFF_FFFF, 0x0F0F], dtype=np.uint64)
    q = np.array([0b1111, 0b11111, 0xFFFF_FFFF_FFFF_FFF0, 0x0F0F] * 100, dtype=np.uint64)
    idx, dist = dhash.nearest(q, base, chunk=7)
    assert idx[:4].tolist() == [0, -1, 1, 2] and dist[:4].tolist() == [4, -1, 4, 0]
    assert idx.tolist() == idx[:4].tolist() * 100
    empty = dhash.nearest(q[:2], np.array([], dtype=np.uint64))
    assert empty[0].tolist() == [-1, -1]


# --- the base's index, made in the bucket's region ------------------------------------------

def test_the_instance_writes_the_bases_index_beside_its_kept_copy(world, base, tmp_path):
    s3 = BucketS3()
    dataset_s3.upload_release(base, world["root"], s3, "bkt", log=lambda s: None)
    rel = dr.load_release(base, world["root"], allow_draft=True)
    request = {"job": aws_dataset.DHASH_JOB, "name": base, "release_hash": rel.release_hash}
    out = aws_dataset.run_build_job("20261013-000001", request, s3, "bkt",
                                    root=tmp_path / "inst" / "research-releases",
                                    log=lambda s: None)
    assert out["state"] == "finished" and out["photos"] == rel.db.execute(
        "select count(*) from photos where included = 1").fetchone()[0]
    keys = list(s3.objects)
    assert keys[-1] == aws_dataset.build_prefix("20261013-000001") + aws_dataset.RESULT_FILE
    key = aws_dataset.dhash_index_key(base)
    assert key.startswith("research-release-extras/") and key in s3.objects
    assert not any(k.startswith(dataset_s3.release_prefix(base)) and k.endswith(".npz")
                   for k in keys)                                   # never inside the release
    home = aws_dataset.pull_dhash_index(base, s3, "bkt", dest=tmp_path / "home" / "i.npz",
                                        log=lambda s: None)
    dhash.check_index(dhash.load_index(home), rel)
    rel.close()
    s3.objects[key] += b"x"
    with pytest.raises(dr.ReleaseError, match="does not match its stored sha256"):
        aws_dataset.pull_dhash_index(base, s3, "bkt", dest=tmp_path / "home2" / "i.npz",
                                     log=lambda s: None)


def test_the_instance_refuses_a_release_other_than_the_one_the_launch_saw(world, base,
                                                                         tmp_path):
    s3 = BucketS3()
    dataset_s3.upload_release(base, world["root"], s3, "bkt", log=lambda s: None)
    request = {"job": aws_dataset.DHASH_JOB, "name": base, "release_hash": "0" * 64}
    with pytest.raises(dr.ReleaseError, match="not the release the launch saw"):
        aws_dataset.run_build_job("20261013-000002", request, s3, "bkt",
                                  root=tmp_path / "inst", log=lambda s: None)
    doc = json.loads(s3.objects[aws_dataset.build_prefix("20261013-000002")
                                + aws_dataset.RESULT_FILE])
    assert doc["state"] == "failed" and aws_dataset.dhash_index_key(base) not in s3.objects


def test_the_index_launch_is_refused_before_any_spend_without_a_kept_copy_or_twice(
        world, base, tmp_path, monkeypatch):
    s3, ec2 = BucketS3(), FakeEC2()
    launch_env(monkeypatch, tmp_path, s3, ec2)
    with pytest.raises(dr.ReleaseError, match="no kept copy"):
        aws_dataset.launch_dhash_index(base, log=lambda s: None)
    dataset_s3.upload_release(base, world["root"], s3, "bkt", log=lambda s: None)
    out = aws_dataset.launch_dhash_index(base, log=lambda s: None)
    prefix = aws_dataset.build_prefix(out["build_id"])
    req = json.loads(s3.objects[prefix + aws_dataset.REQUEST_FILE])
    assert req["job"] == "dhash-index" and req["name"] == base
    assert ec2.launched["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert out["then"] == f"mv dataset pull-dhash-index {base}"
    s3.objects[aws_dataset.dhash_index_key(base)] = b"index"
    ec2.launched = None
    with pytest.raises(dr.ReleaseError, match="already holds"):
        aws_dataset.launch_dhash_index(base, log=lambda s: None)
    assert ec2.launched is None


def test_the_index_estimate_counts_every_included_photo():
    h = aws_dataset.estimate_dhash_hours(680_000)
    assert 1.0 <= h <= 2.0
