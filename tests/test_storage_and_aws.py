import sqlite3

import pytest

from mycomap_vision import aws
from mycomap_vision.manifest import SCHEMA, connect, snapshot
from mycomap_vision.storage import LocalStore, S3Store, open_store, parse_s3_url


class FakeS3:
    def __init__(self):
        self.puts = []

    def put_object(self, **kw):
        self.puts.append(kw)


def test_s3_urls_split_into_bucket_and_prefix():
    assert parse_s3_url("s3://bucket") == ("bucket", "")
    assert parse_s3_url("s3://bucket/a/b/") == ("bucket", "a/b/")
    with pytest.raises(ValueError):
        parse_s3_url("https://bucket/a")
    with pytest.raises(ValueError):
        parse_s3_url("s3:///a")


def test_s3_store_keeps_the_same_relative_paths_as_local_with_an_image_type():
    fake = FakeS3()
    store = S3Store("s3://mv-data/v1", client=fake)
    store.put("photos/large/234/1234.jpeg", b"img")
    assert fake.puts == [{"Bucket": "mv-data", "Key": "v1/photos/large/234/1234.jpeg",
                          "Body": b"img", "ContentType": "image/jpeg"}]
    assert store.location == "s3://mv-data/v1/"


def test_local_store_writes_atomically(tmp_path):
    LocalStore(tmp_path).put("photos/x/1.jpg", b"abc")
    assert (tmp_path / "photos/x/1.jpg").read_bytes() == b"abc"
    assert not list(tmp_path.rglob("*.part"))


def test_open_store_defaults_to_the_data_folder(tmp_path):
    assert isinstance(open_store(None, tmp_path), LocalStore)
    assert open_store(None, tmp_path).location == str(tmp_path)


def test_snapshot_is_a_complete_copy_of_a_manifest_in_use(tmp_path):
    conn = connect(tmp_path / "m.sqlite")
    conn.execute("insert into license_history values (1, 'cc0', 't')")
    conn.commit()
    snap = snapshot(conn, tmp_path / "snap.sqlite")
    copy = sqlite3.connect(snap)
    assert copy.execute("select count(*) from license_history").fetchone()[0] == 1


def test_old_manifests_gain_the_store_column(tmp_path):
    path = tmp_path / "old.sqlite"
    old = sqlite3.connect(path)
    # The schema as first shipped: the same, minus the store column.
    old.executescript("\n".join(line for line in SCHEMA.splitlines()
                                if not line.strip().startswith("store ")))
    assert "store" not in {r[1] for r in old.execute("pragma table_info(photos)")}
    old.close()
    conn = connect(path)
    cols = {r[1] for r in conn.execute("pragma table_info(photos)")}
    assert "store" in cols


def test_the_instance_always_shuts_itself_down():
    ud = aws.render_user_data("20260928-120000", "large", 120)
    assert "shutdown -h +7260" in ud                 # backstop: max hours + 1 hour
    assert "trap finish EXIT" in ud and "shutdown -h now" in ud
    assert "--size large" in ud and "--max-hours 120" in ud
    assert '--dest "s3://$BUCKET"' in ud
    assert "RUN=runs/20260928-120000" in ud
    assert 'aws s3 cp "s3://$BUCKET/$RUN/code.tar.gz"' in ud


def test_instances_terminate_on_shutdown_and_carry_the_project_tag():
    args = aws.run_instance_args("r1", "ami-1", "#!/bin/bash")
    assert args["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert args["IamInstanceProfile"] == {"Name": "mycomap-vision-instance"}
    assert args["MetadataOptions"] == {"HttpTokens": "required"}
    for spec in args["TagSpecifications"]:
        assert {"Key": "Project", "Value": "mycomap-vision"} in spec["Tags"]
    assert {s["ResourceType"] for s in args["TagSpecifications"]} == {"instance", "volume"}


def test_pulling_the_manifest_refuses_while_it_is_open_here(tmp_path):
    dest = tmp_path / "manifest.sqlite"
    dest.write_bytes(b"")
    (tmp_path / "manifest.sqlite-wal").write_bytes(b"")
    with pytest.raises(RuntimeError, match="still open"):
        aws.pull_manifest(dest)
