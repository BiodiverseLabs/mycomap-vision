"""The Paper in Progress page: the draft is for signed-in mycomap.org members only, even
when the rest of the site is open to everyone (Steve, 2026-10-09)."""

import pytest
from fastapi.testclient import TestClient
from test_api import open_limits
from test_signin import ISSUER, PUBLIC_KEY, SECRET, sign_in
from test_models_and_scoreboard import Const, seed_two_species

from mycomap_vision.api import PAPER_FILE, create_app
from mycomap_vision.signin import SigninConfig

DRAFT = "# A DNA-verified fungal identifier\n\nDraft text.\n"


def site(conn, tmp_path, mode, paper_file=None):
    seed_two_species(conn, tmp_path)
    conn.commit()
    if paper_file is None:
        paper_file = tmp_path / "draft.md"
        paper_file.write_text(DRAFT, encoding="utf-8")
    signin = SigninConfig(mode=mode, origin="http://testserver", issuer=ISSUER,
                          public_key=PUBLIC_KEY, secret=SECRET)
    app = create_app(tmp_path / "manifest.sqlite", tmp_path / "emb",
                     backbone_loader=lambda n: Const(n), limits=open_limits(), signin=signin,
                     background=False, paper_file=paper_file)
    return TestClient(app, follow_redirects=False)


@pytest.mark.parametrize("mode", ["identify", "all"])
def test_a_signed_out_visitor_never_gets_the_draft_even_when_the_site_is_open(conn, tmp_path,
                                                                            mode):
    c = site(conn, tmp_path, mode)
    r = c.get("/api/paper")
    assert r.status_code == 401 and r.json()["signin"] == "/auth/signin"
    assert "Draft text" not in r.text


def test_the_open_site_still_answers_its_public_pages_signed_out(conn, tmp_path):
    c = site(conn, tmp_path, "identify")
    assert c.get("/api/scoreboard").status_code == 200
    assert c.get("/api/paper").status_code == 401


@pytest.mark.parametrize("mode", ["identify", "all"])
def test_a_signed_in_member_reads_the_draft_and_it_is_never_cached(conn, tmp_path, mode):
    c = site(conn, tmp_path, mode)
    sign_in(c)
    r = c.get("/api/paper")
    assert r.status_code == 200
    assert r.json() == {"markdown": DRAFT}
    assert r.headers["cache-control"] == "private, no-store"


def test_a_server_without_a_draft_says_so(conn, tmp_path):
    c = site(conn, tmp_path, "all", paper_file=tmp_path / "missing.md")
    sign_in(c)
    assert c.get("/api/paper").status_code == 404


def test_the_draft_ships_with_the_code():
    assert PAPER_FILE.is_file()
    assert PAPER_FILE.read_text(encoding="utf-8").startswith("# ")
