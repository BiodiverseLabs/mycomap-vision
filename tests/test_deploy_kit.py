"""The Lightsail kit's files agree with each other and with the app's limits."""

import json
import re

from test_nightly import box  # noqa: F401  (a server box with a release, for the preload wait)

from mycomap_vision import aws, config, guards

KIT = config.REPO_ROOT / "deploy" / "lightsail"


def read(name: str) -> str:
    return (KIT / name).read_text(encoding="utf-8")


def env_template() -> dict[str, str]:
    out = {}
    for line in read("vision.env.template").splitlines():
        if line.strip() and not line.startswith("#"):
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip()
    return out


def test_nginx_the_service_and_deploy_agree_on_the_app_port():
    service_port = re.search(r"mv serve --host 127\.0\.0\.1 --port (\d+)",
                             read("mycomap-vision.service")).group(1)
    assert "proxy_pass http://127.0.0.1:<PORT>;" in read("nginx-vision.conf.template")
    assert f'PORT="${{PORT:-{service_port}}}"' in read("deploy.sh")


def test_every_request_passes_the_sign_in_gate_in_the_app():
    conf = read("nginx-vision.conf.template")
    # nginx must not serve the built site itself, or pages would skip the gate.
    assert "try_files" not in conf and "root /var/www/mycomap-vision" not in conf
    assert conf.count("proxy_pass ") == 1


def test_nginx_accepts_the_largest_request_the_app_allows():
    mb = int(re.search(r"client_max_body_size (\d+)m;", read("nginx-vision.conf.template")).group(1))
    assert mb * 2**20 >= guards.MAX_REQUEST_BYTES


def test_the_app_sees_the_visitor_address_not_a_forged_header():
    conf = read("nginx-vision.conf.template")
    assert "proxy_set_header X-Forwarded-For $remote_addr;" in conf
    assert "$proxy_add_x_forwarded_for" not in conf       # appending would pass a forged value
    assert "include /etc/nginx/snippets/cloudflare-realip.conf;" in conf
    assert "real_ip_header CF-Connecting-IP;" in read("cloudflare-realip.sh")


def test_api_errors_are_never_turned_into_pages():
    assert "proxy_intercept_errors off;" in read("nginx-vision.conf.template")


def test_pages_allow_the_inat_photo_hosts_and_stay_out_of_search_engines():
    headers = read("nginx-vision-headers.conf")
    for host in ("https://inaturalist-open-data.s3.amazonaws.com", "https://static.inaturalist.org"):
        assert host in headers
    assert 'X-Robots-Tag "noindex, nofollow"' in headers


def test_pages_allow_every_host_a_shown_photo_comes_from():
    # Specimen cards show MO photos too (record sources, 2026-10-09): licenses.photo_url
    # gives https://mushroomobserver.org/images/..., which may redirect to a subdomain.
    from mycomap_vision import licenses
    img_src = re.search(r"img-src ([^;]+);", read("nginx-vision-headers.conf")).group(1).split()
    mo = licenses.photo_url("mo", "https://mushroomobserver.org/images/thumb/12.jpg?1", "medium")
    assert mo.startswith("https://mushroomobserver.org/")
    assert "https://mushroomobserver.org" in img_src and "https://*.mushroomobserver.org" in img_src
    inat = licenses.photo_url("inat", "https://inaturalist-open-data.s3.amazonaws.com/photos/1/large.jpg",
                              "medium")
    assert any(inat.startswith(h + "/") for h in img_src), inat


def test_pages_allow_the_map_tiles_and_nothing_else_new():
    # The Identify page's map (web/src/components/LocationPicker.tsx): its tiles are images.
    picker = (config.REPO_ROOT / "web" / "src" / "components" / "LocationPicker.tsx").read_text(
        encoding="utf-8")
    headers = read("nginx-vision-headers.conf")
    img_src = re.search(r"img-src ([^;]+);", headers).group(1).split()
    for url in re.findall(r'url: "(https://[^"]+)"', picker):
        host = re.match(r"https://([^/]+)/", url).group(1).replace("{s}.", "*.")
        assert f"https://{host}" in img_src, host
    # Place search goes through Vision's own /api/geocode, so the page talks to no one else.
    assert "connect-src 'self';" in headers


def test_the_box_starts_with_the_whole_site_behind_mycomap_org_sign_in():
    env = env_template()
    assert env["MV_SIGNIN"] == "all"
    assert env["MV_PUBLIC_ORIGIN"] == "https://vision.mycomap.org"
    assert env["MV_SIGNIN_ISSUER"] == "https://mycomap.org"
    assert env["MV_RELEASE_ROOT"] == "/srv/mycomap-vision"


def test_the_service_may_write_only_where_releases_and_weights_live():
    service = read("mycomap-vision.service")
    env = env_template()
    assert "ProtectSystem=strict" in service
    writable = re.search(r"ReadWritePaths=(.+)", service).group(1).split()
    for key in ("MV_RELEASE_ROOT", "HF_HOME"):
        assert any(env[key] == w or env[key].startswith(w + "/") for w in writable), key


def test_no_secret_address_or_bucket_is_committed_in_the_kit():
    for path in list(KIT.iterdir()) + [config.REPO_ROOT / "docs" / "deploy.md"]:
        text = path.read_text(encoding="utf-8")
        assert not re.search(r"\b\d{1,3}(\.\d{1,3}){3}\b", text.replace("127.0.0.1", "")), path
        assert "PRIVATE KEY" not in text, path
        assert "mycomap-vision-data" not in text, path
    env = env_template()
    assert env["MV_S3_BUCKET"] == "<BUCKET>" and env["MV_AWS_REGION"] == "<REGION>"


def test_the_box_may_only_read_releases(monkeypatch):
    monkeypatch.setenv("MV_S3_BUCKET", "bucket-x")
    monkeypatch.setenv("MV_AWS_REGION", "us-east-2")
    policy = json.loads(aws.render_policy("box-policy.template.json"))
    [statement] = policy["Statement"]
    assert statement["Effect"] == "Allow"
    assert statement["Action"] == "s3:GetObject"
    assert statement["Resource"] == "arn:aws:s3:::bucket-x/releases/*"


def test_scripts_stop_at_the_first_failure():
    for name in ("provision.sh", "deploy.sh", "cloudflare-realip.sh"):
        assert "set -euo pipefail" in read(name), name


def test_the_web_build_runs_where_its_pnpm_version_is_pinned():
    deploy = read("deploy.sh")
    assert "pnpm --dir" not in deploy
    assert "(cd web && pnpm install --frozen-lockfile" in deploy


def _preload_wait(tmp_path, answers: list[str | None],
                  verify: str = 'echo "VERIFY RAN"') -> "subprocess.CompletedProcess[str]":
    """Run deploy.sh's own preload wait (up to verify.sh), in bash, as the box would: under
    `set -euo pipefail`, with `curl` answering each /api/health call from `answers` in turn
    (None: the call times out, as while the model loads) and `sleep` adding 100 s at once.
    verify.sh is replaced by `verify` (a marker line by default)."""
    import subprocess

    import pytest
    bash = _git_bash()
    if not bash:
        pytest.skip("needs bash")
    script = read("deploy.sh").replace("\r\n", "\n")
    wait = script[script.index("preload_state() {"):script.index("bash deploy/lightsail/verify.sh")]
    tmp_path.mkdir(parents=True, exist_ok=True)
    for i, a in enumerate(answers):
        (tmp_path / f"answer{i}").write_text("" if a is None else a, encoding="utf-8")
        if a is None:
            (tmp_path / f"answer{i}.timeout").write_text("", encoding="utf-8")
    d = str(tmp_path).replace("\\", "/")
    fake = f"""set -euo pipefail
PORT=8010
curl() {{
  n=$(cat "{d}/calls" 2>/dev/null || echo 0); echo $((n + 1)) > "{d}/calls"
  [ -f "{d}/answer$n" ] || n={len(answers) - 1}   # the last answer repeats
  if [ -f "{d}/answer$n.timeout" ]; then echo "curl: (28) timed out" >&2; return 28; fi
  cat "{d}/answer$n"
}}
sleep() {{ SECONDS=$((SECONDS + 100)); }}
"""
    return subprocess.run([bash, "-c", fake + wait + verify + "\n"],
                          capture_output=True, text=True, timeout=60)


def _preload_state_seen_by_deploy(health_body: str, tmp_path) -> str:
    """deploy.sh's own reading of the preload state in `health_body`."""
    out = _preload_wait(tmp_path, [health_body])
    assert out.returncode == 0, out.stdout + out.stderr
    first = out.stdout.splitlines()[1]
    return ("ready" if "ready after" in first else "failed" if "preload failed" in first
            else "off" if "no preload" in first else first)


def test_deploy_waits_for_the_model_and_not_for_the_nightly_updates_state(box, tmp_path):
    import time

    from fastapi.testclient import TestClient
    from test_api import open_limits
    from test_models_and_scoreboard import Const
    from test_nightly import QUIET, RID

    from mycomap_vision import nightly
    from mycomap_vision.api import create_app
    layer = nightly.prepare(box, **QUIET)
    client = TestClient(create_app(
        layer.manifest, box / "releases" / RID / "embeddings",
        backbone_loader=lambda name: Const(name), limits=open_limits(), background=False,
        layer=layer, note=lambda s: None, preload=["m1/nearest"],
        nightly_settings=nightly.Settings()))
    for _ in range(200):
        body = client.get("/api/health").text
        if '"preload":{"state":"loading"' not in body:
            break
        time.sleep(0.05)
    assert '"nightly":' in body and body.count('"state":') == 2     # both blocks have a state
    assert _preload_state_seen_by_deploy(body, tmp_path / "a") == "ready"
    assert _preload_state_seen_by_deploy('{"ok":true,"preload":{"state":"failed","error":"x"},'
                                         '"nightly":{"state":"ok"}}', tmp_path / "b") == "failed"


LOADING = '{"ok":true,"preload":{"state":"loading"},"nightly":{"state":"ok"}}'
READY = '{"ok":true,"preload":{"state":"ready"},"nightly":{"state":"ok"}}'


def test_a_slow_health_call_while_the_model_loads_does_not_end_the_deploy(tmp_path):
    # 2026-10-11: one health call timed out during the 70 s preload and set -e ended the
    # deploy before verify.sh ran.
    out = _preload_wait(tmp_path, [LOADING, None, None, READY])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ready after" in out.stdout
    assert out.stdout.rstrip().endswith("VERIFY RAN")


def test_health_without_a_preload_block_keeps_waiting_instead_of_failing(tmp_path):
    out = _preload_wait(tmp_path, ['{"ok":true}', READY])
    assert out.returncode == 0, out.stdout + out.stderr
    assert out.stdout.rstrip().endswith("VERIFY RAN")


def test_a_model_still_loading_after_five_minutes_is_reported_and_verify_still_runs(tmp_path):
    out = _preload_wait(tmp_path, [LOADING])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "still loading after 300 s" in out.stdout
    assert out.stdout.rstrip().endswith("VERIFY RAN")


def test_verify_runs_last_so_a_failed_check_still_fails_the_deploy(tmp_path):
    script = read("deploy.sh").replace("\r\n", "\n")
    assert script.rstrip().endswith("bash deploy/lightsail/verify.sh")
    out = _preload_wait(tmp_path, [LOADING, None, READY], verify='echo "VERIFY RAN"; exit 1')
    assert "VERIFY RAN" in out.stdout
    assert out.returncode != 0


def test_a_failed_preload_whose_health_then_times_out_still_reaches_verify(tmp_path):
    out = _preload_wait(tmp_path, ['{"ok":true,"preload":{"state":"failed","error":"x"}}', None])
    assert out.returncode == 0, out.stdout + out.stderr
    assert "preload failed" in out.stdout
    assert out.stdout.rstrip().endswith("VERIFY RAN")


def _git_bash():
    import os
    import shutil
    if os.name == "nt":
        path = r"C:\Program Files\Git\bin\bash.exe"     # not WSL's bash.exe
        return path if os.path.exists(path) else None
    return shutil.which("bash")


def test_a_pull_that_changes_deploy_sh_runs_the_new_script(tmp_path):
    import subprocess

    import pytest
    bash = _git_bash()
    if not bash:
        pytest.skip("needs bash and git")

    def git(cwd, *args):
        subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c",
                        "core.autocrlf=false", *args], cwd=cwd, check=True,
                       capture_output=True)

    script = read("deploy.sh").replace("\r\n", "\n")
    origin = tmp_path / "origin"
    (origin / "deploy" / "lightsail").mkdir(parents=True)
    target = origin / "deploy" / "lightsail" / "deploy.sh"
    target.write_bytes(script.encode())
    git(origin, "init", "-q", "-b", "main")
    git(origin, "add", "-A")
    git(origin, "commit", "-q", "-m", "old")
    app = tmp_path / "app"
    git(tmp_path, "clone", "-q", str(origin), str(app))
    # The next commit changes the script: its new line must run in this deploy.
    marker = 'echo "    now at $(git log --oneline -1)"\n'
    assert marker in script
    target.write_bytes(script.replace(marker, marker + 'echo "NEW SCRIPT RAN"\n').encode())
    git(origin, "commit", "-q", "-am", "new")
    env_file = tmp_path / "vision.env"
    env_file.write_text("MV_S3_BUCKET=b\n")
    import os
    env = {**os.environ, "APP_DIR": str(app).replace("\\", "/"),
           "ENV_FILE": str(env_file).replace("\\", "/"), "DEPLOY_STOP_AFTER_PULL": "1",
           "HOME": str(tmp_path).replace("\\", "/")}
    out = subprocess.run([bash, str(app / "deploy" / "lightsail" / "deploy.sh")], env=env,
                         capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "running the new one" in out.stdout
    assert out.stdout.count("NEW SCRIPT RAN") == 1
