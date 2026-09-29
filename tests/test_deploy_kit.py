"""The Lightsail kit's files agree with each other and with the app's limits."""

import json
import re

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
