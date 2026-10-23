"""Get involved is open to signed-out visitors (Steve, 2026-10-22): the page and every feed it
reads (web/src/lib/getInvolved.ts GET_INVOLVED_FEEDS) must answer someone signed out even
when the site needs sign-in, or they see a page with no numbers."""
import pytest

from test_signin import gated_site_with_pages

FEEDS = ("/api/stats", "/api/how-good", "/api/feedback/summary", "/api/docs/v1.json")


def test_the_page_answers_a_signed_out_visitor(conn, tmp_path):
    c = gated_site_with_pages(conn, tmp_path)
    res = c.get("/get-involved")
    assert res.status_code == 200
    assert "<title>" in res.text


@pytest.mark.parametrize("feed", FEEDS)
def test_every_feed_the_page_reads_answers_a_signed_out_visitor(conn, tmp_path, feed):
    c = gated_site_with_pages(conn, tmp_path)
    assert c.get("/api/me").json()["user"] is None
    assert c.get(feed).status_code == 200, feed

