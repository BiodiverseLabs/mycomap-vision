"""The Research pages are open to signed-out visitors (Steve, 2026-10-22), and the redesign
(Steve, 2026-10-25) reads more feeds on them: the landing's live tiles, the benchmark cards,
the API page's endpoint count. Every feed the pages read (web/src/components/researchQueries.ts
RESEARCH_FEEDS, read from the file so the two can't drift apart) must answer someone signed out
even when the site needs sign-in, or a visitor sees pages with no numbers. The experiment
registry stays members only."""
import re

import pytest

from mycomap_vision import config
from test_signin import gated_site_with_pages


def research_feeds() -> list[str]:
    src = (config.REPO_ROOT / "web" / "src" / "components" / "researchQueries.ts").read_text(encoding="utf-8")
    block = re.search(r"RESEARCH_FEEDS = \[(.*?)\] as const", src, re.S)
    assert block, "RESEARCH_FEEDS not found in researchQueries.ts"
    return re.findall(r'"(/api/[^"]+)"', block.group(1))


def test_the_pages_list_the_feeds_they_read():
    assert set(research_feeds()) >= {"/api/how-good", "/api/stats", "/api/scoreboard", "/api/prospective",
                                     "/api/early-answers", "/api/models", "/api/speed/daily", "/api/docs/v1.json"}


@pytest.mark.parametrize("feed", research_feeds())
def test_every_feed_the_research_pages_read_answers_a_signed_out_visitor(conn, tmp_path, feed):
    c = gated_site_with_pages(conn, tmp_path)
    assert c.get("/api/me").json()["user"] is None
    assert c.get(feed).status_code == 200, feed


def test_a_run_report_is_read_signed_out_too(conn, tmp_path):
    # The charts read each run's standard summary from /api/scoreboard/{id}: never sent to sign in.
    c = gated_site_with_pages(conn, tmp_path)
    assert c.get("/api/scoreboard/1").status_code not in (302, 303, 307, 401)


def test_the_experiment_registry_stays_members_only(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    assert c.get("/research/experiments").status_code == 200, "the page opens and asks to sign in"
    assert c.get("/api/experiments").status_code in (302, 303, 307, 401)
