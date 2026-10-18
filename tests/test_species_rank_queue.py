"""Human Feedback "A species and its own varieties" (species_rank_split): labels the answer key
reads as one species (Steve, 2026-10-18: species rank counts), queued so the records say which
name they keep. Groups whose labels all sit in a "One name, several writings" item are left
out (queued there already). "Use this name for all" opens a Name Update on mycomap.org."""

import csv

import pytest

from mycomap_vision import feedback as fb
from mycomap_vision import feedback_org as fo

COLS = ["answer_key", "labels_in_group", "label", "records_train", "records_val", "kind"]


def groups_file(tmp_path):
    rows = [
        ("agaricus porphyrocephalus", 3, "Agaricus porphyrocephalus", 1, 0, "species"),
        ("agaricus porphyrocephalus", 3, "Agaricus porphyrocephalus ssp. pallidus", 5, 0, "infraspecific"),
        ("agaricus porphyrocephalus", 3, "Agaricus porphyrocephalus var. pallidus", 46, 1, "infraspecific"),
        ("lactarius quietus", 2, "Lactarius quietus", 0, 1, "species"),
        ("lactarius quietus", 2, "Lactarius quietus var. incanus", 30, 3, "infraspecific"),
        ("russula x", 2, "Russula x ssp. y", 4, 0, "infraspecific"),
        ("russula x", 2, "Russula x subsp. y", 2, 0, "infraspecific"),
    ]
    path = tmp_path / "groups.tsv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(COLS)
        w.writerows(rows)
    return path


def store_with_writing_split(tmp_path):
    store = fb.Store(tmp_path / "f.sqlite")
    store.seed([{"item_type": "name_writing_split", "subject": "Russula x ssp. y; Russula x subsp. y",
                 "rule": "R1", "current": "Russula x ssp. y (4) | Russula x subsp. y (2)",
                 "proposed": "Russula x ssp. y"},
                {"item_type": "name_writing_split",
                 "subject": "Agaricus porphyrocephalus ssp. pallidus; Agaricus porphyrocephalus var. pallidus",
                 "rule": "R1", "current": "Agaricus porphyrocephalus var. pallidus (47) | "
                                          "Agaricus porphyrocephalus ssp. pallidus (5)",
                 "proposed": "Agaricus porphyrocephalus var. pallidus"}], "writings")
    return store


def test_each_answer_key_with_several_labels_is_one_item_unless_already_queued(tmp_path):
    store = store_with_writing_split(tmp_path)
    rows = {r["evidence"]["answer_key"]: r for r in
            fb.rows_from_answer_key_groups(groups_file(tmp_path), store)}
    assert set(rows) == {"agaricus porphyrocephalus", "lactarius quietus"}   # russula x: queued
    aga = rows["agaricus porphyrocephalus"]
    assert aga["proposed"] == "Agaricus porphyrocephalus var. pallidus"      # most used
    assert fo.writings(aga["current"])[0] == "Agaricus porphyrocephalus var. pallidus"
    assert aga["records"] == 53
    assert "ssp. pallidus" in aga["evidence"]["also_in_writing_split"]
    lq = rows["lactarius quietus"]
    assert lq["evidence"]["labels_without_train_records"] == "Lactarius quietus"
    assert fb.SECTION_OF["species_rank_split"] == "cleanup"
    # Without a store nothing is known to be queued: every group is an item.
    assert len(fb.rows_from_answer_key_groups(groups_file(tmp_path))) == 3


def test_seeding_twice_adds_nothing_new(tmp_path, conn):
    store = store_with_writing_split(tmp_path)
    first = fb.seed_files(store, conn, answer_key_groups=groups_file(tmp_path))
    again = fb.seed_files(store, conn, answer_key_groups=groups_file(tmp_path))
    assert first[0]["added"] == 2 and again[0]["added"] == 0


def test_use_this_name_for_all_opens_one_name_update_and_keep_apart_none(tmp_path):
    store = store_with_writing_split(tmp_path)
    item = next(r for r in fb.rows_from_answer_key_groups(groups_file(tmp_path), store)
                if r["evidence"]["answer_key"] == "lactarius quietus")
    item = {**item, "id": 1}
    p = fo.plan(item, "use_name", "Lactarius quietus")
    assert p.kind == "same_taxon" and p.new_name == "Lactarius quietus"
    assert p.old_names == ["Lactarius quietus var. incanus"]
    assert fo.plan(item, "keep_apart") is None
    with pytest.raises(fb.FeedbackError, match="the name to keep is one of"):
        fo.plan(item, "use_name", "Lactarius quietis")
