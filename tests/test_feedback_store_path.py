"""The Human Feedback queue is kept on the box under <release root>/state/ and elsewhere
beside the manifest, never in the data folder by default: a test run (whose manifest is
temporary) must not reach a laptop's real queue (Coordinator, 2026-10-19)."""

from pathlib import Path

import conftest
from mycomap_vision import config, feedback as fb


def test_on_the_box_the_queue_is_in_the_release_roots_state(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RELEASE_ROOT", Path("/srv/mycomap-vision"))
    assert fb.default_path(tmp_path / "manifest.sqlite") == Path(
        "/srv/mycomap-vision/state/human-feedback.sqlite")


def test_off_the_box_the_queue_is_beside_the_manifest(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RELEASE_ROOT", None)
    assert fb.default_path(tmp_path / "m" / "manifest.sqlite") == (
        tmp_path / "m" / "human-feedback.sqlite")
    assert fb.default_path() == config.MANIFEST_PATH.parent / "human-feedback.sqlite"


def test_the_server_keeps_a_tests_queue_beside_its_temporary_manifest(conn, tmp_path):
    from test_api_v1_identify import v1_app
    client = v1_app(conn, tmp_path)
    assert Path(client.app.state.feedback.path) == tmp_path / "human-feedback.sqlite"


def test_tests_never_default_to_the_real_data_folder():
    assert conftest.REAL_FEEDBACK_STORE.parent != config.DATA_DIR
    assert not str(config.MANIFEST_PATH).startswith(str(conftest.REAL_FEEDBACK_STORE.parent))


def test_the_guard_sees_a_change_to_the_real_queue(tmp_path):
    store = tmp_path / "human-feedback.sqlite"
    before = conftest._state(store)
    fb.Store(store)                                   # creating it is a change
    assert conftest._state(store) != before
