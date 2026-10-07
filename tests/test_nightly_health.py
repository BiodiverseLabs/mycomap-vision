import ast
import io
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from test_api import open_limits
from test_models_and_scoreboard import Const
from test_nightly import EST, RID, FakeOutside, QUIET, box, night, row, seed_rows  # noqa: F401

from mycomap_vision import config, manifest, nightly, nightly_notes
from mycomap_vision.api import create_app

NOW = datetime(2026, 10, 8, 12, 0, tzinfo=timezone.utc)


def health_of(box, layer, **kw):
    client = TestClient(create_app(
        layer.manifest if layer else box / "releases" / RID / "manifest.sqlite",
        box / "releases" / RID / "embeddings", backbone_loader=lambda name: Const(name),
        limits=open_limits(), background=False, layer=layer, note=lambda s: None,
        nightly_settings=nightly.Settings(), **kw))
    return client.get("/api/health").json()["nightly"]


def test_health_says_the_nightly_update_is_off_without_a_layer(box):
    assert health_of(box, None) == {"enabled": False}


def test_health_reports_no_run_before_the_first_night(box):
    layer = nightly.prepare(box, **QUIET)
    block = health_of(box, layer)
    assert block["enabled"] and block["state"] == "never" and block["last_ok_at"] is None
    assert block["layer"]["m1"]["release_photos"] == 8


def test_health_reports_what_the_last_night_changed(box):
    layer = nightly.prepare(box, **QUIET)
    night(layer, seed_rows()[1:] + [row(300, "C z")], FakeOutside({300: 130}))
    block = health_of(box, layer)
    assert block["state"] == "ok" and block["last_ok_at"] and not block["refused"]
    assert block["changed"] == {"new": 1, "removed": 1, "renamed": 0, "embedded": 1,
                                "predicted": 0}
    assert block["layer"]["m1"]["nightly_photos"] == 1


def test_health_reports_a_refusal_without_its_text(box):
    layer = nightly.prepare(box, **QUIET)
    conn = manifest.connect(layer.manifest)

    def refused():
        raise nightly.NightlyRefused("could not reach https://internal.example: x; "
                                     "nothing was changed")
    nightly.run_once(conn, layer, refused, {}, **QUIET)
    conn.close()
    block = health_of(box, layer)
    assert block["state"] == "failed" and block["refused"] is True
    assert "internal.example" not in json.dumps(block)


def test_health_gives_the_next_run_when_the_zone_is_known(box):
    layer = nightly.prepare(box, **QUIET)
    conn = manifest.connect(layer.manifest)
    block = nightly.health(conn, nightly.Settings(), EST, now=NOW)
    conn.close()
    assert block["next_run"] == "2026-10-09T08:00:00+00:00"


# --- what verify.sh prints -----------------------------------------------------------

def block(**kw):
    base = {"enabled": True, "state": "ok", "last_run_at": "2026-10-08T07:00:01+00:00",
            "last_ok_at": "2026-10-08T07:20:00+00:00", "next_run": "2026-10-09T07:00:00+00:00",
            "changed": {"new": 3, "removed": 1, "renamed": 2, "embedded": 9},
            "layer": {"m1": {"release_photos": 100, "nightly_photos": 5, "share": 0.05,
                             "new_release_due": False}}}
    return {**base, **kw}


def levels(b, now=NOW):
    return [lvl for lvl, _ in nightly_notes.notes(b, now)]


def test_a_good_night_is_one_ok_line_with_what_it_changed():
    [(lvl, text)] = nightly_notes.notes(block(), NOW)
    assert lvl == "ok" and "3 new, 1 removed, 2 renamed, 9 photos embedded" in text


def test_off_and_not_yet_run_are_fine():
    assert levels({"enabled": False}) == ["ok"]
    assert levels(block(state="never", last_ok_at=None, last_run_at=None)) == ["ok"]


def test_a_failed_or_refused_night_warns():
    [(lvl, text)] = nightly_notes.notes(block(state="failed", refused=True), NOW)
    assert lvl == "warn" and "refused" in text and "mv nightly" in text


def test_no_successful_night_for_a_day_warns():
    stale = block(last_ok_at=(NOW - timedelta(hours=27)).isoformat())
    assert levels(stale) == ["ok", "warn"]
    assert levels(block(state="failed", last_ok_at=None)) == ["warn", "warn"]


def test_a_night_running_for_hours_warns():
    started = (NOW - timedelta(hours=4)).isoformat()
    assert levels(block(state="running", last_run_at=started))[0] == "warn"
    started = (NOW - timedelta(minutes=20)).isoformat()
    assert levels(block(state="running", last_run_at=started))[0] == "ok"


def test_a_layer_past_its_share_calls_for_a_new_release():
    due = block(layer={"m1": {"release_photos": 100, "nightly_photos": 21, "share": 0.21,
                              "new_release_due": True}})
    lines = nightly_notes.notes(due, NOW)
    assert lines[-1][0] == "warn" and "21%" in lines[-1][1] and "new release" in lines[-1][1]


def test_a_server_without_the_block_is_a_warning_not_a_crash():
    assert levels(None) == ["warn"]


def run_notes(stdin: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "mycomap_vision.nightly_notes"], input=stdin,
                          capture_output=True, text=True, cwd=config.REPO_ROOT,
                          env={"PYTHONPATH": str(config.REPO_ROOT / "src"),
                               "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")})


def test_verify_prints_the_notes_and_never_fails_a_deploy():
    good = run_notes(json.dumps({"ok": True, "nightly": block()}))
    assert good.returncode == 0 and good.stdout.startswith("ok    last nightly run")
    bad = run_notes("<html>502</html>")
    assert bad.returncode == 0 and bad.stdout.startswith("WARN  could not read")


def test_the_notes_need_only_the_standard_library_wherever_verify_runs():
    tree = ast.parse((config.REPO_ROOT / "src" / "mycomap_vision" / "nightly_notes.py")
                     .read_text(encoding="utf-8"))
    imported = {a.name.split(".")[0] for n in ast.walk(tree) if isinstance(n, ast.Import)
                for a in n.names}
    imported |= {n.module.split(".")[0] for n in ast.walk(tree)
                 if isinstance(n, ast.ImportFrom) and n.module and n.level == 0}
    assert imported <= set(sys.stdlib_module_names)


def test_verify_runs_the_notes_and_the_box_starts_with_the_update_off():
    verify = (config.REPO_ROOT / "deploy" / "lightsail" / "verify.sh").read_text(encoding="utf-8")
    assert "-m mycomap_vision.nightly_notes" in verify
    template = (config.REPO_ROOT / "deploy" / "lightsail" / "vision.env.template").read_text(
        encoding="utf-8")
    assert "\n# MV_NIGHTLY=1\n" in template
    assert not any(line.startswith("MV_NIGHTLY") for line in template.splitlines())
