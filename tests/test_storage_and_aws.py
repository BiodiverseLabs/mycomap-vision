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
    # The schema as first shipped: no photos.store column and no photo_copies table.
    before, rest = SCHEMA.split("-- Every held copy of a photo")
    first = before + rest.split("primary key (photo_id, store, size)\n);")[1]
    old.executescript("\n".join(line for line in first.splitlines()
                                if not line.strip().startswith("store            text,")))
    assert "store" not in {r[1] for r in old.execute("pragma table_info(photos)")}
    old.close()
    conn = connect(path)
    cols = {r[1] for r in conn.execute("pragma table_info(photos)")}
    assert "store" in cols


def test_the_instance_always_shuts_itself_down():
    ud = aws.render_user_data("20260928-120000", "large", 120, "my-bucket")
    assert "shutdown -h +7260" in ud                 # backstop: max hours + 1 hour
    assert "trap finish EXIT" in ud and "shutdown -h now" in ud
    assert "BUCKET=my-bucket" in ud
    assert "--size large" in ud and "--max-hours 120" in ud
    assert '--dest "s3://$BUCKET"' in ud
    assert "RUN=runs/20260928-120000" in ud
    assert 'aws s3 cp "s3://$BUCKET/$RUN/code.tar.gz"' in ud


def test_instances_terminate_on_shutdown_and_carry_the_project_tag(monkeypatch):
    monkeypatch.delenv("MV_INSTANCE_ROLE", raising=False)
    args = aws.run_instance_args("r1", "ami-1", "#!/bin/bash")
    assert args["InstanceInitiatedShutdownBehavior"] == "terminate"
    assert args["IamInstanceProfile"] == {"Name": "mycomap-vision-instance"}
    assert args["MetadataOptions"] == {"HttpTokens": "required"}
    for spec in args["TagSpecifications"]:
        assert {"Key": "Project", "Value": "mycomap-vision"} in spec["Tags"]
    assert {s["ResourceType"] for s in args["TagSpecifications"]} == {"instance", "volume"}


def test_pulling_the_instance_manifest_merges_s3_copies_and_keeps_local_work(tmp_path):
    local = connect(tmp_path / "local" / "manifest.sqlite")
    remote_path = tmp_path / "remote.sqlite"
    remote = connect(remote_path)
    for c in (local, remote):
        c.executemany("insert into photos (photo_id, license_class, source_url, first_seen_at, "
                      "license_checked_at, status) values (?, 'open', 'u', 't', 't', 'pending')",
                      [(1,), (2,), (3,)])
    local.execute("insert into photo_copies values (1, 'C:/data', 'medium', 'p/1.jpg', 1, 'h', 't')")
    local.execute("create table embeddings (backbone text, photo_id int)")
    local.execute("insert into embeddings values ('m', 1)")
    remote.execute("insert into photo_copies values (1, 's3://b/', 'large', 'p/1.jpg', 9, 'h', 't')")
    remote.execute("insert into photo_copies values (2, 's3://b/', 'large', 'p/2.jpg', 9, 'h', 't')")
    remote.execute("update photos set status = 'missing', error = 'http 404' where photo_id = 3")
    local.commit()
    remote.commit()
    remote.close()
    out = aws.merge_manifest(local, remote_path)
    assert out == {"s3_copies_added": 2, "marked_missing": 1}
    copies = {tuple(r) for r in local.execute("select photo_id, store, size from photo_copies")}
    assert copies == {(1, "C:/data", "medium"), (1, "s3://b/", "large"), (2, "s3://b/", "large")}
    assert local.execute("select count(*) from embeddings").fetchone()[0] == 1
    row = local.execute("select status, error from photos where photo_id = 3").fetchone()
    assert tuple(row) == ("missing", "http 404")


def test_single_copy_manifests_are_backfilled_with_their_own_folder_as_the_store(tmp_path):
    path = tmp_path / "data" / "manifest.sqlite"
    c = connect(path)
    c.execute("insert into photos (photo_id, license_class, source_url, first_seen_at, "
              "license_checked_at, status, local_path, size, store) values "
              "(1, 'open', 'u', 't', 't', 'done', 'p/1.jpg', 'medium', null)")
    c.execute("delete from photo_copies")
    c.commit()
    c.close()
    again = connect(path)
    rows = [tuple(r) for r in again.execute("select photo_id, store, size, path from photo_copies")]
    assert rows == [(1, str(path.parent), "medium", "p/1.jpg")]



def test_iam_policies_are_filled_from_settings_with_no_placeholders_left(monkeypatch):
    import json
    monkeypatch.setenv("MV_S3_BUCKET", "bkt")
    monkeypatch.setenv("MV_AWS_REGION", "eu-west-1")
    monkeypatch.setenv("MV_INSTANCE_ROLE", "role-x")
    for name in ("instance-policy.template.json", "ops-policy.template.json"):
        text = aws.render_policy(name)
        assert "{{" not in text
        json.loads(text)
    ops = aws.render_policy("ops-policy.template.json")
    assert "arn:aws:s3:::bkt/*" in ops and "eu-west-1" in ops and "role/role-x" in ops


def test_missing_settings_say_which_variable_to_set(monkeypatch):
    monkeypatch.delenv("MV_S3_BUCKET", raising=False)
    with pytest.raises(RuntimeError, match="MV_S3_BUCKET"):
        aws.bucket()


def test_dotenv_fills_only_unset_variables(tmp_path, monkeypatch):
    from mycomap_vision import config
    env = tmp_path / ".env"
    env.write_text("\n".join(["# comment", "MV_T1=from-file", "MV_T2='quoted'", ""]))
    monkeypatch.setenv("MV_T1", "from-env")
    monkeypatch.delenv("MV_T2", raising=False)
    config.load_dotenv(env)
    import os
    assert os.environ["MV_T1"] == "from-env" and os.environ["MV_T2"] == "quoted"
    monkeypatch.delenv("MV_T2")
