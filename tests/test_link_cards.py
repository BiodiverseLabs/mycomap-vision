"""Link previews (linkcards.py): the card Facebook, X, iMessage and Slack show for a Vision
link. Steve, 2026-10-20. The shared pages carry the site's card while every answer stays
behind sign-in; a run's link carries its own answer and photos, never the place, the date or
who ran it; a deleted run's card is gone."""

import html
import io
import re

import pytest
from PIL import Image
from test_run_links import identify, jpeg_with_gps, person, site, visitor
from test_signin import HERE

from mycomap_vision import linkcards, runs
from mycomap_vision.signin import SigninConfig, public_page, run_path

SHELL = ('<!doctype html><html><head><meta charset="UTF-8" /><title>MycoMap Vision</title>'
         '</head><body><div id="root"></div><script src="/assets/app.js"></script></body></html>')


def dist(tmp_path):
    d = tmp_path / "dist"
    (d / "assets").mkdir(parents=True)
    (d / "index.html").write_text(SHELL, encoding="utf-8")
    (d / "assets" / "app.js").write_text("console.log('app')", encoding="utf-8")
    Image.new("RGB", (1200, 630), (143, 199, 62)).save(d / "og-card.jpg")
    return d


def card_of(page_html: str) -> dict:
    """The og:/twitter: tags of a page, unescaped."""
    return {k: html.unescape(v) for k, v in re.findall(
        r'<meta (?:property|name)="((?:og|twitter):[a-z:_]+)" content="([^"]*)" />', page_html)}


CRAWLERS = ("facebookexternalhit/1.1 (+http://www.facebook.com/externalhit_uatext.php)",
            "Twitterbot/1.0", "Slackbot-LinkExpanding 1.0 (+https://api.slack.com/robots)",
            "Mozilla/5.0 (Macintosh) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17 Safari",
            "WhatsApp/2.23", "")


# --- the shared pages ------------------------------------------------------------------------

@pytest.mark.parametrize("path,title", [("/", linkcards.SITE_TITLE),
                                        ("/about", "How MycoMap Vision works"),
                                        ("/get-involved", "Get involved with MycoMap Vision")])
def test_a_crawler_reading_a_shared_page_gets_visions_own_card_not_the_sign_in_page(
        conn, tmp_path, path, title):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    for agent in CRAWLERS:
        res = visitor(app).get(path, headers={"user-agent": agent})
        assert res.status_code == 200, (agent, res.status_code)
        card = card_of(res.text)
        assert card["og:title"] == title and card["twitter:title"] == title
        assert card["og:description"] == linkcards.SITE_DESCRIPTION
        assert card["og:url"] == HERE + path
        assert card["og:image"] == HERE + "/og-card.jpg" == card["twitter:image"]
        assert card["og:site_name"] == "MycoMap Vision"
        assert card["twitter:card"] == "summary_large_image"
        assert (card["og:image:width"], card["og:image:height"]) == ("1200", "630")


def test_the_site_card_says_it_is_free_needs_a_mycomap_account_and_is_no_edibility_advice():
    text = linkcards.SITE_DESCRIPTION
    assert "Free" in text and "mycomap.org account" in text and "Not edibility advice" in text
    assert "DNA" in linkcards.SITE_TITLE
    assert len(text) <= 160


def test_the_cards_image_opens_without_sign_in(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    res = visitor(app).get("/og-card.jpg")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"


def test_a_shared_page_gives_a_signed_out_visitor_only_the_shell_and_its_card(conn, tmp_path):
    """Nothing behind sign-in comes with it: the page is index.html, byte for byte, plus the
    card's tags, and every answer still needs sign-in."""
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    anyone = visitor(app)
    page = anyone.get("/").text
    block = re.search(r"<!-- link preview -->.*?(?=</head>)", page, re.S).group(0)
    assert page.replace(block, "") == SHELL
    assert identify(anyone).status_code == 401
    for path in ("/api/models", "/api/runs", "/api/feedback/queue", "/api/paper"):
        assert anyone.get(path).status_code == 401, path


def test_every_other_page_still_sends_a_signed_out_visitor_to_sign_in(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    anyone = visitor(app)
    for path in ("/research", "/research/results", "/research/experiments", "/feedback",
                 "/about/x", "/get-involved/", "/r/short"):
        res = anyone.get(path)
        assert res.status_code == 302 and res.headers["location"].startswith("/auth/signin"), path
    # Only reading: a shared page refuses anything else without sign-in.
    assert anyone.post("/").status_code in (302, 401, 405)
    assert not public_page("POST", "/") and public_page("GET", "/") and public_page("HEAD", "/")


def test_signed_in_people_get_the_same_shell_with_its_card(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    page = person(app, "user-1").get("/about")
    assert page.status_code == 200 and card_of(page.text)["og:title"] == "How MycoMap Vision works"
    assert "x-robots-tag" not in page.headers


def test_the_shared_pages_are_open_whatever_sign_in_mode_the_site_runs():
    cfg = SigninConfig(mode="off")
    assert not cfg.needs_session("GET", "/")
    for page in linkcards.SHARE_PAGES:
        assert public_page("GET", page)


# --- a run's link ----------------------------------------------------------------------------

def test_a_runs_link_carries_its_answer_and_an_image_of_its_photos(conn, tmp_path):
    app, store, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    res = visitor(app).get(f"/r/{rid}", headers={"user-agent": CRAWLERS[0]})
    assert res.status_code == 200
    card = card_of(res.text)
    want = linkcards.run_title(linkcards.headline(linkcards.run_answer(store.get(rid))))
    assert card["og:title"] == want and want.startswith("MycoMap Vision")
    assert card["og:description"] == linkcards.RUN_DESCRIPTION
    assert card["og:url"] == f"{HERE}/r/{rid}"
    assert card["og:image"] == f"{HERE}/api/runs/{rid}/card.jpg"
    # Still only for whoever was given the link: never indexed or cached.
    assert res.headers["x-robots-tag"] == "noindex, nofollow"
    assert "no-store" in res.headers["cache-control"]


def test_a_runs_card_never_carries_the_place_the_date_or_who_ran_it(conn, tmp_path):
    app, store, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    rid = identify(person(app, "user-42", name="Agaricus Fan"), lat="40.12345",
                   lng="-82.98765", observed_on="2026-10-11").json()["run"]["id"]
    kept = store.get(rid)
    assert kept["owner"]["id"] == "user-42" and kept["context_used"]["latitude"] == 40.1
    block = re.search(r"<!-- link preview -->.*?(?=</head>)",
                      visitor(app).get(f"/r/{rid}").text, re.S).group(0)
    for secret in ("user-42", "Agaricus Fan", "40.1", "-83.0", "82.9", "2026-10-11",
                   kept["answered_at"][:10]):
        assert secret not in block, secret


def test_a_runs_card_image_is_its_photos_at_1200_by_630_with_no_exif(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    rid = identify(person(app, "user-1"),
                   photos=(("a.jpg", jpeg_with_gps(), "image/jpeg"),
                           ("b.jpg", jpeg_with_gps(colour=(0, 0, 247)), "image/jpeg"))
                   ).json()["run"]["id"]
    anyone = visitor(app)
    res = anyone.get(f"/api/runs/{rid}/card.jpg", headers={"user-agent": CRAWLERS[0]})
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    assert res.headers["x-robots-tag"] == "noindex, nofollow"
    assert "no-store" in res.headers["cache-control"]
    img = Image.open(io.BytesIO(res.content))
    assert img.size == (1200, 630) and not img.getexif()
    # The first photo (red) leads on the left; the second (blue) sits beside it.
    r, g, b = img.convert("RGB").getpixel((300, 300))
    assert r > 200 and b < 60
    r, g, b = img.convert("RGB").getpixel((900, 300))
    assert b > 200 and r < 60
    # Some link checkers ask with HEAD first.
    assert anyone.head(f"/api/runs/{rid}/card.jpg").status_code == 200
    assert run_path(f"/api/runs/{rid}/card.jpg") and not run_path(f"/api/runs/{rid}/card.png")


def test_a_deleted_runs_link_and_card_image_are_gone(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    owner = person(app, "user-1")
    rid = identify(owner).json()["run"]["id"]
    anyone = visitor(app)
    assert anyone.get(f"/api/runs/{rid}/card.jpg").status_code == 200   # made, and kept
    assert owner.delete(f"/api/runs/{rid}").status_code == 204
    page = anyone.get(f"/r/{rid}")
    assert page.status_code == 404
    card = card_of(page.text)
    assert card["og:title"] == linkcards.GONE_TITLE
    assert card["og:image"] == HERE + "/og-card.jpg"
    assert page.headers["x-robots-tag"] == "noindex, nofollow"
    assert anyone.get(f"/api/runs/{rid}/card.jpg").status_code == 404


def test_a_link_never_given_out_has_no_answer_in_its_card(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    rid = runs.new_id()
    page = visitor(app).get(f"/r/{rid}")
    assert page.status_code == 404 and card_of(page.text)["og:title"] == linkcards.GONE_TITLE
    assert visitor(app).get(f"/api/runs/{rid}/card.jpg").status_code == 404


def test_an_observation_runs_card_uses_the_site_image_and_names_no_observation(conn, tmp_path):
    app, store, _ = site(conn, tmp_path, by_observation=True, web_dist=dist(tmp_path))
    res = person(app, "user-1").post("/api/identify/observation",
                                     json={"source": "inat", "id": 555})
    rid = res.json()["run"]["id"]
    page = visitor(app).get(f"/r/{rid}")
    card = card_of(page.text)
    assert card["og:image"] == HERE + "/og-card.jpg"
    assert card["og:title"].startswith("MycoMap Vision")
    block = re.search(r"<!-- link preview -->.*?(?=</head>)", page.text, re.S).group(0)
    assert "555" not in block and "inaturalist" not in block.lower()
    assert visitor(app).get(f"/api/runs/{rid}/card.jpg").status_code == 404


def test_the_page_still_loads_when_the_bucket_cant_be_read(conn, tmp_path, monkeypatch):
    app, store, _ = site(conn, tmp_path, web_dist=dist(tmp_path))

    def down(*a, **k):
        raise OSError("the bucket is unreachable")
    monkeypatch.setattr(store, "get", down)
    page = visitor(app).get(f"/r/{runs.new_id()}")
    assert page.status_code == 200 and '<div id="root">' in page.text


# --- the answer, as the card states it --------------------------------------------------------

def names(**ranks):
    return {"ranks": {rank: [{"name": n, "confidence": c} for n, c in v]
                      for rank, v in ranks.items()}}


def test_the_card_states_the_finest_settled_rank_as_the_page_does():
    h = linkcards.headline(names(species=[("Pluteus thomsonii", 0.99)],
                                 genus=[("Pluteus", 0.99)]))
    assert linkcards.run_title(h) == "MycoMap Vision says: Pluteus thomsonii (99%)"
    h = linkcards.headline(names(species=[("Pluteus sp. 'thomsonii-IN01'", 0.35),
                                          ("Pluteus hongoi", 0.11)],
                                 genus=[("Pluteus", 0.94)], family=[("Pluteaceae", 0.95)]))
    assert linkcards.run_title(h) == "MycoMap Vision says: genus Pluteus (94%)"
    # A species at 62% with the next at 7% is the answer (settled.py, Steve 2026-10-18).
    h = linkcards.headline(names(species=[("Russula a", 0.62), ("Russula b", 0.07)],
                                 genus=[("Russula", 0.9)]))
    assert (h.rank, h.name) == ("species", "Russula a")


def test_with_nothing_settled_the_card_says_so_and_gives_the_top_family():
    h = linkcards.headline(names(species=[("A x", 0.2)], genus=[("A", 0.4)],
                                 family=[("Agaricaceae", 0.45)]))
    assert linkcards.run_title(h) == ("MycoMap Vision: no confident answer yet "
                                      "(family Agaricaceae, 45%)")


def test_when_the_photos_may_not_show_a_fungus_the_card_states_no_chance():
    r = names(species=[("A x", 0.9)], genus=[("A", 0.95)])
    r["gate"] = {"verdict": "unsure"}
    title = linkcards.run_title(linkcards.headline(r))
    assert title == "MycoMap Vision: the closest look-alike is A x" and "%" not in title
    r["gate"] = {"verdict": "not_fungus"}
    assert "don't look like a fungus" in linkcards.run_title(linkcards.headline(r))
    r = names(species=[("A x", 0.9)])
    r["confidence_withheld"] = True
    assert "%" not in linkcards.run_title(linkcards.headline(r))


def test_the_likely_list_leads_when_there_is_one_as_on_the_page():
    r = names(species=[("A x", 0.3), ("A y", 0.25)], genus=[("A", 0.5)])
    r["likely"] = {"genus": {"names": [{"name": "B", "confidence": 0.8}]}}
    assert linkcards.headline(r).name == "B"


def test_every_tag_value_is_escaped():
    card = linkcards.Card('Say "hi" <script>alert(1)</script> & go', "d", "https://x/r/a",
                          "https://x/i.jpg")
    out = linkcards.tags(card)
    assert "<script>" not in out and "&quot;hi&quot;" in out and "&amp; go" in out
    assert card_of(out)["og:title"] == card.title


def test_the_tags_go_at_the_end_of_the_head_and_a_page_without_a_card_is_untouched():
    card = linkcards.site_card("https://v", "/")
    page = linkcards.with_tags(SHELL, card)
    assert page.index("og:title") < page.index("</head>") < page.index('<div id="root">')
    assert linkcards.with_tags(SHELL, None) == SHELL
    assert linkcards.site_card("https://v", "/research") is None


# --- the card image ---------------------------------------------------------------------------

def photo(colour, size=(800, 600)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "JPEG")
    return buf.getvalue()


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8])
def test_a_card_shows_up_to_five_photos_without_overlapping(n):
    boxes = linkcards.tiles(min(n, 5))
    assert len(boxes) == min(n, 5)
    for i, (x, y, w, h) in enumerate(boxes):
        assert x >= 0 and y >= 0 and x + w <= 1200 and y + h <= 630
        for (x2, y2, w2, h2) in boxes[i + 1:]:
            assert x + w <= x2 or x2 + w2 <= x or y + h <= y2 or y2 + h2 <= y
    body = linkcards.run_card_jpeg([photo((i * 30, 120, 90)) for i in range(n)],
                                   linkcards.Headline("settled", "species", "A x", 0.9))
    assert Image.open(io.BytesIO(body)).size == (1200, 630)


def test_a_card_is_made_even_where_no_font_is_installed(monkeypatch):
    monkeypatch.setattr(linkcards, "FONT_DIRS", ())
    body = linkcards.run_card_jpeg([photo((200, 10, 10))],
                                   linkcards.Headline("settled", "genus", "Pluteus", 0.94))
    assert Image.open(io.BytesIO(body)).size == (1200, 630)


def test_every_kind_of_answer_can_be_drawn():
    for h in (linkcards.Headline("settled", "family", "Pluteaceae", 0.8),
              linkcards.Headline("look_alike", "species", "A x"),
              linkcards.Headline("unsettled", "family", "Agaricaceae", 0.45),
              linkcards.Headline("not_fungus"), linkcards.Headline("none"),
              linkcards.Headline("settled", "species", "Cortinarius sp. 'glandicolor-CA01'", 0.71)):
        assert linkcards.run_card_jpeg([photo((10, 120, 10))], h)[:2] == b"\xff\xd8"


def test_a_card_image_made_before_a_run_was_deleted_is_never_served_after(conn, tmp_path):
    """Whatever deletes it (the page, an admin, a later clean-up of the bucket), a gone run
    has no card: the store is asked every time, not only the image kept in memory."""
    app, store, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    rid = identify(person(app, "user-1")).json()["run"]["id"]
    anyone = visitor(app)
    assert anyone.get(f"/api/runs/{rid}/card.jpg").status_code == 200
    store.delete(rid)
    assert anyone.get(f"/api/runs/{rid}/card.jpg").status_code == 404
