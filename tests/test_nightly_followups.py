import io
import sqlite3

from test_nightly import QUIET, RID, box  # noqa: F401

from mycomap_vision import nightly
from mycomap_vision.api import line_buffered_output


def layer_backbones(layer):
    c = sqlite3.connect(layer.manifest)
    try:
        return [r[0] for r in c.execute("select backbone from nightly_layer order by backbone")]
    finally:
        c.close()


def embed_rows_without_vectors(box, backbone):
    """The laptop's manifest lists backbones the release ships no vectors for."""
    c = sqlite3.connect(box / "releases" / RID / "manifest.sqlite")
    with c:
        c.executemany("insert into embeddings values (?, ?, 0, ?, 'now')",
                      [(backbone, 1000 + i, i) for i in range(8)])
    c.close()


def test_a_new_layer_tracks_only_the_backbones_the_release_ships_vectors_for(box):
    embed_rows_without_vectors(box, "dinov2-b14")
    layer = nightly.prepare(box, **QUIET)
    assert layer_backbones(layer) == ["m1"]
    c = sqlite3.connect(layer.manifest)
    assert list(nightly.health(c, nightly.Settings())["layer"]) == ["m1"]
    c.close()


def test_an_older_layer_forgets_the_backbones_the_release_ships_no_vectors_for(box):
    embed_rows_without_vectors(box, "dinov2-b14")
    layer = nightly.prepare(box, **QUIET)
    c = sqlite3.connect(layer.manifest)                 # as start_layer used to note it
    with c:
        c.execute("insert into nightly_layer values ('dinov2-b14', 1, 8, 'then')")
    c.close()
    said = []
    nightly.prepare(box, log=said.append)
    assert layer_backbones(layer) == ["m1"]
    assert "dinov2-b14" in said[0]


def test_a_backbone_the_nights_added_vectors_for_is_never_forgotten(box):
    layer = nightly.prepare(box, **QUIET)
    c = sqlite3.connect(layer.manifest)
    with c:
        c.execute("insert into nightly_layer values ('gone', 1, 8, 'then')")
        c.execute("insert into embeddings values ('gone', 5, 1, 0, 'now')")    # a nightly shard
    assert nightly.drop_unshipped(c, layer.release_dir) == []
    c.close()
    assert layer_backbones(layer) == ["gone", "m1"]


def test_the_servers_lines_reach_the_log_as_they_are_written():
    raw = io.BytesIO()
    out = io.TextIOWrapper(raw, encoding="utf-8", newline="\n")   # a pipe, as under systemd
    out.write("[nightly] waiting\n")
    assert raw.getvalue() == b""                           # block-buffered: nothing yet
    line_buffered_output([out, object()])                  # a stream without reconfigure is left
    out.write("[nightly] records: 1 new\n")
    assert raw.getvalue() == b"[nightly] waiting\n[nightly] records: 1 new\n"
