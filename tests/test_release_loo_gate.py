"""The leave-one-out gate on a frozen release (Steve, 2026-10-11: "review first, then
build"): v1 is not cut without the person-checked LOO exclusion list, or Steve's explicit
word that there is none, and the release records which."""

import pytest
from test_dataset_release import build, reasons, world  # noqa: F401  (the fixture)

from mycomap_vision import dataset_release as dr

FREEZE = "Steve 2026-10-20 (test)"


def loo_list(tmp_path, rows="record\t2\tLOO mislabel\tchecked by Steve\n"):
    p = tmp_path / "loo-mislabel-2026-10-09.tsv"
    p.write_text("kind\tkey\treason\tnote\n" + rows, encoding="utf-8")
    return p


def test_a_frozen_release_waits_for_the_loo_list(world):
    with pytest.raises(dr.ReleaseError, match="waits for the LOO exclusion list"):
        build(world, name="v1", freeze_approved=FREEZE)
    assert not (world["root"] / "v1").exists()


def test_the_loo_list_excludes_its_records_and_the_release_records_its_hash(world, tmp_path):
    lst = loo_list(tmp_path)
    m = build(world, name="v1", freeze_approved=FREEZE, loo_exclusions=lst)
    assert m["loo_exclusions"] == {"file": lst.name, "rows": 1, "sha256": dr.sha256_file(lst)}
    assert reasons(world, "v1")["2"] == ("review:loo-mislabel-2026-10-09", None)
    assert (world["root"] / "v1" / "inputs" / lst.name).is_file()


def test_an_empty_reviewed_list_is_a_real_answer(world, tmp_path):
    m = build(world, name="v1", freeze_approved=FREEZE, loo_exclusions=loo_list(tmp_path, ""))
    assert m["loo_exclusions"]["rows"] == 0


def test_no_loo_exclusions_only_with_who_said_so(world):
    with pytest.raises(dr.ReleaseError, match="who said so"):
        build(world, name="v1", freeze_approved=FREEZE, no_loo_exclusions="  ")
    m = build(world, name="v1", freeze_approved=FREEZE, no_loo_exclusions="Steve, 2026-10-20")
    assert m["loo_exclusions"] == {"none": "Steve, 2026-10-20"}


def test_a_list_and_no_list_together_or_a_malformed_list_are_refused(world, tmp_path):
    with pytest.raises(dr.ReleaseError, match="not both"):
        build(world, name="v1", freeze_approved=FREEZE, loo_exclusions=loo_list(tmp_path),
              no_loo_exclusions="Steve")
    bad = tmp_path / "bad.tsv"
    bad.write_text("observation_id,decision\n2,exclude\n", encoding="utf-8")
    with pytest.raises(dr.ReleaseError, match="not a review list"):
        build(world, name="v1", freeze_approved=FREEZE, loo_exclusions=bad)
    with pytest.raises(dr.ReleaseError, match="not found"):
        build(world, name="v1", freeze_approved=FREEZE, loo_exclusions=tmp_path / "nope.tsv")


def test_a_key_the_frozen_manifest_lacks_stops_the_build(world, tmp_path):
    with pytest.raises(dr.ReleaseError):
        build(world, name="v1", freeze_approved=FREEZE,
              loo_exclusions=loo_list(tmp_path, "record\t999999\tLOO mislabel\tx\n"))
    assert not (world["root"] / "v1").exists()


def test_drafts_and_test_releases_need_no_loo_list():
    assert dr.loo_gate("dry-20990101-000001", None, None, None) == {"not_required": "draft"}
    assert dr.loo_gate("v1-test", "v1", None, None) == {"not_required": "test release"}
