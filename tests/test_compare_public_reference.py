"""The public model's reference (paper E23, E23b): `mv compare --release <id> --reference
public-trainable` scores the release's val records against a reference holding only the
photos the public model may learn from (CC including NC, or all-rights-reserved with the
owner's permission). A train record left with no such photo drops out of the reference;
the scored records are the same as with the full reference, and every row says which
reference it used.

The release is test_trainer_seed_public.relicensed: photo 1000 (record 100's only photo)
all-rights-reserved with no answer, 1001 all-rights-reserved with permission granted."""

import json

import pytest
from test_trainer_release import cut, embed  # noqa: F401
from test_trainer_seed_public import relicensed

from mycomap_vision import evaluate


def run(cut, tmp_path, rel, **kw):
    conn, _, store, _, _ = cut
    embed(conn, store, tmp_path / "emb")
    return evaluate.compare(conn, ["base"], ["nearest"], embeddings_root=tmp_path / "emb",
                            release=rel, sets=False, log=lambda s: None, **kw)


def test_the_public_reference_leaves_out_records_with_no_public_photo(cut, tmp_path):
    rel = relicensed(cut, "dry-20261018-100001")
    try:
        everyone = run(cut, tmp_path, rel)
        public = run(cut, tmp_path, rel, public_reference=True)
    finally:
        rel.close()
    assert everyone["reference_records"] == 6 and public["reference_records"] == 5
    assert everyone["test_records"] == public["test_records"] == 2    # same scored records
    assert public["record_set"] != everyone["record_set"]
    assert everyone["reference"] == "every included photo"
    assert public["reference"] == "public_trainable"
    conn = cut[0]
    saved = json.loads(conn.execute("select report_json from eval_runs where comparison_id = ?",
                                    (public["comparison_id"],)).fetchone()[0])
    assert saved["reference"] == "public_trainable"
    assert saved["dataset_release"]["dataset_release"] == "dry-20261018-100001"


def test_a_public_reference_needs_a_release(cut, tmp_path):
    with pytest.raises(ValueError, match="needs a dataset release"):
        run(cut, tmp_path, None, public_reference=True)
