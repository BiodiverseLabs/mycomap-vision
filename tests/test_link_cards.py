"""Link previews (linkcards.py): the card Facebook, X, iMessage and Slack show for a Vision
link. Steve, 2026-10-20. The shared pages carry the site's card while every answer stays
behind sign-in; a run's link carries its own answer and photos, never the place, the date or
who ran it; a deleted run's card is gone."""

import html
import io
import re
from types import SimpleNamespace

import pytest
from PIL import Image, ImageDraw
from test_api_v1_by_observation import INAT_PHOTO, MO_PHOTO, SERVER, Response, fetcher
from test_run_links import identify, jpeg_with_gps, person, site, visitor
from test_signin import HERE

from mycomap_vision import byobservation as bo, linkcards, runs
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


def site_image(tmp_path) -> str:
    """The site image's address as the tags give it: versioned by the file's content."""
    v = linkcards.image_version(tmp_path / "dist" / "og-card.jpg")
    assert v and len(v) == 10
    return f"{HERE}/og-card.jpg?v={v}"


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
        assert card["og:image"] == site_image(tmp_path) == card["twitter:image"]
        assert card["og:site_name"] == "MycoMap Vision"
        assert card["twitter:card"] == "summary_large_image"
        assert (card["og:image:width"], card["og:image:height"]) == ("1200", "630")


def test_the_site_card_says_it_is_free_needs_a_mycomap_account_and_is_no_edibility_advice():
    text = linkcards.SITE_DESCRIPTION
    assert "Free" in text and "mycomap.org account" in text and "Not edibility advice" in text
    assert "DNA" in linkcards.SITE_TITLE
    assert len(text) <= 160


def test_the_cards_image_opens_without_sign_in_at_its_versioned_address(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    for address in ("/og-card.jpg", site_image(tmp_path).removeprefix(HERE)):
        res = visitor(app).get(address)
        assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg", address


def test_a_new_site_image_gets_a_new_address_so_facebook_fetches_it_again(conn, tmp_path):
    """Facebook keeps an image by its URL: after tagline A went live (2026-10-20) a re-scrape
    still showed the old picture at the same address."""
    d = dist(tmp_path)
    app, _, _ = site(conn, tmp_path, web_dist=d)
    before = card_of(visitor(app).get("/").text)["og:image"]
    Image.new("RGB", (1200, 630), (20, 60, 20)).save(d / "og-card.jpg")
    (tmp_path / "again").mkdir()
    app, _, _ = site(conn, tmp_path / "again", web_dist=d)
    after = card_of(visitor(app).get("/").text)
    assert before != after["og:image"] and after["og:image"] == site_image(tmp_path)
    assert after["twitter:image"] == after["og:image"]
    # Every card that uses the site image carries the same versioned address.
    gone = card_of(visitor(app).get(f"/r/{runs.new_id()}").text)
    assert gone["og:image"] == after["og:image"]


def test_without_a_site_image_file_the_address_has_no_version(tmp_path):
    assert linkcards.image_version(tmp_path / "missing.jpg") is None
    assert linkcards.site_image("https://v") == "https://v/og-card.jpg"


def test_a_shared_page_gives_a_signed_out_visitor_only_the_shell_and_its_card(conn, tmp_path):
    """Nothing behind sign-in comes with it: the page is index.html, byte for byte, plus the
    card's tags, and every answer still needs sign-in."""
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    anyone = visitor(app)
    page = anyone.get("/").text
    block = re.search(r"<!-- link preview -->.*?(?=</head>)", page, re.S).group(0)
    assert page.replace(block, "") == SHELL
    assert identify(anyone).status_code == 401
    for path in ("/api/experiments", "/api/runs", "/api/feedback/queue", "/api/paper"):
        assert anyone.get(path).status_code == 401, path


def test_every_other_page_still_sends_a_signed_out_visitor_to_sign_in(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    anyone = visitor(app)
    # (The Research section is open to anyone: test_research_public.py.)
    for path in ("/feedback", "/about/x", "/get-involved/", "/r/short", "/researchx"):
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
    assert card["og:image"] == site_image(tmp_path)
    assert page.headers["x-robots-tag"] == "noindex, nofollow"
    assert anyone.get(f"/api/runs/{rid}/card.jpg").status_code == 404


def test_a_link_never_given_out_has_no_answer_in_its_card(conn, tmp_path):
    app, _, _ = site(conn, tmp_path, web_dist=dist(tmp_path))
    rid = runs.new_id()
    page = visitor(app).get(f"/r/{rid}")
    assert page.status_code == 404 and card_of(page.text)["og:title"] == linkcards.GONE_TITLE
    assert visitor(app).get(f"/api/runs/{rid}/card.jpg").status_code == 404


def test_an_observation_run_with_no_cc0_or_cc_by_photo_uses_the_site_image(conn, tmp_path):
    """The fake iNat's 555 states no licence: all rights reserved, never on a card."""
    app, store, _ = site(conn, tmp_path, by_observation=True, web_dist=dist(tmp_path))
    res = person(app, "user-1").post("/api/identify/observation",
                                     json={"source": "inat", "id": 555})
    rid = res.json()["run"]["id"]
    page = visitor(app).get(f"/r/{rid}")
    card = card_of(page.text)
    assert card["og:image"] == site_image(tmp_path)
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


# --- an iNat or MO run's card: the observation's own CC0 / CC BY photos, credited ------------

PHOTOGRAPHER = "Jane Doe"
RED, GREEN, GREY = (230, 20, 20), (20, 200, 20), (128, 128, 128)
BLUE, YELLOW = (20, 20, 230), (230, 230, 20)
# One find's photos, each under another licence: only 9101 (CC BY) and 9104 (CC0) may go on
# its card; 9102 is non-commercial, 9103 all rights reserved, 9105 share-alike.
LICENSED = [(9101, "cc-by", RED), (9102, "cc-by-nc", GREEN), (9103, None, GREY),
            (9104, "cc0", BLUE), (9105, "cc-by-sa", YELLOW)]


def licensed_source():
    """The fake iNat (test_api_v1_by_observation) with observation 600: Jane Doe's photos."""
    f = fetcher()
    f.session.inat[600] = {
        "id": 600, "observed_on": "2026-10-11", "location": "40.12,-82.98", "obscured": False,
        "taxon": {"name": "A x", "rank": "species"},
        "user": {"login": "jdoe", "name": PHOTOGRAPHER},
        "observation_photos": [{"position": i, "photo": {
            "id": pid, "url": INAT_PHOTO.format(id=pid, size="square"), "license_code": lic}}
            for i, (pid, lic, _) in enumerate(LICENSED)]}
    for pid, _, colour in LICENSED:
        f.session.photos[pid] = photo(colour, (400, 300))
    return f


class Jobs(list):
    """The card worker (create_app's card_writer) as a test runs it: jobs wait for run()."""

    def __call__(self, job):
        self.append(job)

    def run(self):
        while self:
            self.pop(0)()


def observation_run(conn, tmp_path, source=None, make_card=True):
    """An iNat run of observation 600 on the site; its card job run unless make_card=False.
    `identified`: how many photo reads the identification itself made."""
    source = source or licensed_source()
    jobs = Jobs()
    app, store, bucket = site(conn, tmp_path, by_observation=True, web_dist=dist(tmp_path),
                              observation_fetcher=source, card_writer=jobs)
    owner = person(app, "user-1")
    res = owner.post("/api/identify/observation", json={"source": "inat", "id": 600})
    assert res.status_code == 200, res.text
    run = SimpleNamespace(app=app, store=store, bucket=bucket, source=source.session,
                          owner=owner, rid=res.json()["run"]["id"], jobs=jobs,
                          identified=len(photo_reads(source.session)))
    if make_card:
        jobs.run()
    return run


def photo_reads(source) -> list[str]:
    return [u for u in source.urls() if "/photos/" in u or "/images/" in u]


def large(pid: int) -> str:
    return INAT_PHOTO.format(id=pid, size="large")


def card_file(r):
    return r.bucket / runs.PREFIX / r.rid / runs.CARD


def test_an_observation_runs_card_is_made_from_its_cc0_and_cc_by_photos_only(conn, tmp_path):
    r = observation_run(conn, tmp_path)
    kept = r.store.get(r.rid)["observation"]
    # The run keeps each shown photo's licence and photographer (none for the reserved one).
    assert [c and c["licence"] for c in kept["photos_credit"]] == [
        "cc-by", "cc-by-nc", None, "cc0", "cc-by-sa"]
    assert {c["by"] for c in kept["photos_credit"] if c} == {PHOTOGRAPHER}
    # Only the CC BY and the CC0 photo were read for its card, in the page's order.
    assert photo_reads(r.source)[r.identified:] == [large(9101), large(9104)]
    assert card_of(visitor(r.app).get(f"/r/{r.rid}").text)["og:image"] == \
        f"{HERE}/api/runs/{r.rid}/card.jpg"
    res = visitor(r.app).get(f"/api/runs/{r.rid}/card.jpg")
    assert res.status_code == 200 and res.headers["content-type"] == "image/jpeg"
    assert res.headers["x-robots-tag"] == "noindex, nofollow"
    img = Image.open(io.BytesIO(res.content))
    assert img.size == (1200, 630) and not img.getexif()
    red, green, blue = img.convert("RGB").getpixel((300, 300))          # 9101 leads
    assert red > 200 and green < 60 and blue < 60
    red, green, blue = img.convert("RGB").getpixel((900, 300))          # 9104 beside it
    assert blue > 200 and red < 60 and green < 60


def test_the_photographers_credit_is_printed_on_the_card(conn, tmp_path, monkeypatch):
    drawn = []
    real = ImageDraw.ImageDraw.text

    def spy(self, xy, text, *a, **k):
        drawn.append(text)
        return real(self, xy, text, *a, **k)
    monkeypatch.setattr(ImageDraw.ImageDraw, "text", spy)
    r = observation_run(conn, tmp_path)
    assert visitor(r.app).get(f"/api/runs/{r.rid}/card.jpg").status_code == 200
    assert "Jane Doe, CC BY via iNaturalist" in drawn
    assert "Jane Doe, CC0 via iNaturalist" in drawn
    assert not any("NC" in str(t) or "SA" in str(t) for t in drawn)


def test_a_credit_sits_at_the_top_right_of_the_card():
    h = linkcards.Headline("settled", "species", "A x", 0.9)
    pale = [photo((235, 235, 235), (1200, 630))]
    plain = Image.open(io.BytesIO(linkcards.run_card_jpeg(pale, h))).convert("RGB")
    credited = Image.open(io.BytesIO(linkcards.run_card_jpeg(
        pale, h, credits=["Jane Doe, CC BY via iNaturalist"] * 2))).convert("RGB")
    assert min(plain.getpixel((1100, 31))) > 200
    assert max(credited.getpixel((1100, 31))) < 120
    # The same credit twice is printed once: nothing below the first pill.
    assert min(credited.getpixel((1100, 80))) > 200
    # A credit too long for the room is cut short, never drawn over the badge.
    long_one = Image.open(io.BytesIO(linkcards.run_card_jpeg(
        pale, h, credits=["A" * 300 + ", CC BY via iNaturalist"]))).convert("RGB")
    assert long_one.getpixel((300, 47)) == plain.getpixel((300, 47))


def test_only_cc0_cc_by_and_public_domain_get_a_credit_and_cc_by_needs_a_name():
    assert linkcards.credit("inat", "cc-by", "Jane  Doe") == "Jane Doe, CC BY via iNaturalist"
    assert linkcards.credit("inat", "cc0", None) == "CC0 via iNaturalist"
    assert linkcards.credit("mo", "cc-by", "Ann") == "Ann, CC BY via Mushroom Observer"
    assert linkcards.credit("mo", "pd", "Ann") == "Ann, public domain via Mushroom Observer"
    for licence in ("cc-by-nc", "cc-by-sa", "cc-by-nd", "cc-by-nc-sa", "cc-by-nc-nd", "", None):
        assert linkcards.credit("inat", licence, "Jane Doe") is None, licence
    assert linkcards.credit("inat", "cc-by", "  ") is None
    assert linkcards.credit("elsewhere", "cc0", "Jane Doe") is None
    # A run kept before licences were, or an upload run, has none for its card.
    assert linkcards.card_photos({"kind": "observation", "observation": {
        "source": "inat", "photos_shown": ["https://x/1.jpg"]}}) == []
    assert linkcards.card_photos({"kind": "photos", "photos": 2}) == []


def test_an_observation_runs_card_is_made_when_the_run_is_kept_never_while_a_crawler_waits(
        conn, tmp_path):
    """Facebook's scraper gives up after a few seconds, and MO's pace alone is 5 s a photo:
    the card is made on the card worker when the run is kept. Until it is kept, the tags
    name the site image (so no crawler keeps a stand-in under the card's address) and the
    card's address answers with the site image, reading nothing."""
    r = observation_run(conn, tmp_path, make_card=False)
    assert len(r.jobs) == 1                                  # queued when the run was kept
    anyone = visitor(r.app)
    assert card_of(anyone.get(f"/r/{r.rid}").text)["og:image"] == site_image(tmp_path)
    early = anyone.get(f"/api/runs/{r.rid}/card.jpg")
    assert early.status_code == 200
    assert early.content == (tmp_path / "dist" / "og-card.jpg").read_bytes()
    assert photo_reads(r.source)[r.identified:] == []        # nothing read on a request
    assert len(r.jobs) == 1                                  # and no second job for it
    assert not card_file(r).exists()
    r.jobs.run()
    assert photo_reads(r.source)[r.identified:] == [large(9101), large(9104)]
    assert card_of(anyone.get(f"/r/{r.rid}").text)["og:image"] == \
        f"{HERE}/api/runs/{r.rid}/card.jpg"
    made = anyone.get(f"/api/runs/{r.rid}/card.jpg").content
    assert made == card_file(r).read_bytes() != early.content


def test_an_observation_runs_photos_are_read_from_the_source_once_and_kept_beside_it(
        conn, tmp_path):
    r = observation_run(conn, tmp_path)
    anyone = visitor(r.app)
    first = anyone.get(f"/api/runs/{r.rid}/card.jpg").content
    for _ in range(3):
        assert anyone.get(f"/api/runs/{r.rid}/card.jpg").content == first
    assert anyone.head(f"/api/runs/{r.rid}/card.jpg").status_code == 200
    anyone.get(f"/r/{r.rid}")
    # Opening the observation again on the site: the same run, nothing made again.
    again = person(r.app, "user-2").post("/api/identify/observation",
                                         json={"source": "inat", "id": 600}).json()
    assert again["run"]["id"] == r.rid
    r.jobs.run()
    assert len(photo_reads(r.source)) - r.identified == 2
    assert card_file(r).read_bytes() == first
    # At the photo host's pace, like every observation read.
    t = [at for at, url in r.source.log if url in (large(9101), large(9104))][-2:]
    assert t[1] - t[0] >= bo.INTERVALS[bo.STATIC_HOST]


def test_a_card_kept_beside_the_run_is_served_without_reading_the_source(conn, tmp_path):
    """After a restart, or made on another server: the bucket's card, not the source."""
    r = observation_run(conn, tmp_path, make_card=False)
    kept = photo((1, 2, 3), (1200, 630))
    assert r.store.keep_card(r.rid, kept)
    assert card_of(visitor(r.app).get(f"/r/{r.rid}").text)["og:image"] == \
        f"{HERE}/api/runs/{r.rid}/card.jpg"
    assert visitor(r.app).get(f"/api/runs/{r.rid}/card.jpg").content == kept
    r.jobs.run()                                     # the job queued at keep finds it made
    assert photo_reads(r.source)[r.identified:] == []


def test_a_deleted_observation_runs_card_is_gone_with_it(conn, tmp_path):
    r = observation_run(conn, tmp_path)
    anyone = visitor(r.app)
    assert anyone.get(f"/api/runs/{r.rid}/card.jpg").status_code == 200
    assert card_file(r).is_file()
    assert r.owner.delete(f"/api/runs/{r.rid}").status_code == 204
    assert anyone.get(f"/api/runs/{r.rid}/card.jpg").status_code == 404
    assert not card_file(r).exists()
    page = anyone.get(f"/r/{r.rid}")
    assert page.status_code == 404 and card_of(page.text)["og:image"] == site_image(tmp_path)


def test_a_run_deleted_before_its_card_is_made_never_gets_one(conn, tmp_path):
    r = observation_run(conn, tmp_path, make_card=False)
    assert r.owner.delete(f"/api/runs/{r.rid}").status_code == 204
    r.jobs.run()
    assert photo_reads(r.source)[r.identified:] == [] and not card_file(r).exists()
    assert visitor(r.app).get(f"/api/runs/{r.rid}/card.jpg").status_code == 404


def test_an_observation_runs_tags_carry_no_place_observation_owner_or_photographer(
        conn, tmp_path):
    r = observation_run(conn, tmp_path)
    page = visitor(r.app).get(f"/r/{r.rid}", headers={"user-agent": CRAWLERS[0]})
    block = re.search(r"<!-- link preview -->.*?(?=</head>)", page.text, re.S).group(0)
    assert card_of(page.text)["og:image"] == f"{HERE}/api/runs/{r.rid}/card.jpg"
    for secret in ("600", "inaturalist", "static.", "jane doe", "jdoe", "user-1", "mycena fan",
                   "40.1", "-83.0", "82.9", "2026-10-11",
                   r.store.get(r.rid)["answered_at"][:10]):
        assert secret not in block.lower(), secret


def test_when_the_source_gives_no_photo_the_card_is_the_site_image_and_nothing_is_kept(
        conn, tmp_path):
    r = observation_run(conn, tmp_path, make_card=False)
    for pid in (9101, 9104):
        r.source.fail[large(pid)] = Response(503)
    r.jobs.run()
    assert len(photo_reads(r.source)) - r.identified == 2
    assert not card_file(r).exists()
    anyone = visitor(r.app)
    assert card_of(anyone.get(f"/r/{r.rid}").text)["og:image"] == site_image(tmp_path)
    res = anyone.get(f"/api/runs/{r.rid}/card.jpg")
    assert res.status_code == 200
    assert res.content == (tmp_path / "dist" / "og-card.jpg").read_bytes()
    # Not asked again at once: a crawler's retries cost the source nothing.
    r.jobs.run()
    anyone.get(f"/api/runs/{r.rid}/card.jpg")
    r.jobs.run()
    assert len(photo_reads(r.source)) - r.identified == 2


def test_the_api_answer_has_no_credits_but_the_sites_kept_copy_does(conn, tmp_path):
    """identify-by-observation (docs/api.md section 3.2) keeps its shape; the site's copy of
    the answer, which a run is made from, carries the credits."""
    source = licensed_source()
    source.session.inat[601] = {**source.session.inat[600], "id": 601}
    app, store, _ = site(conn, tmp_path, by_observation=True, web_dist=dist(tmp_path),
                         observation_fetcher=source)
    want = ["cc-by", "cc-by-nc", None, "cc0", "cc-by-sa"]
    # Asked through the API first: its answer has none, the site's copy has them.
    api = visitor(app).post("/api/v1/identify/inat/600", headers=SERVER)
    assert api.status_code == 200, api.text
    assert "photos_credit" not in api.json()["observation"]
    res = person(app, "user-1").post("/api/identify/observation",
                                     json={"source": "inat", "id": 600})
    assert res.json()["cached"] is True
    credits = store.get(res.json()["run"]["id"])["observation"]["photos_credit"]
    assert [c and c["licence"] for c in credits] == want
    # Run on the site first, then served through the API from the site's copy: none.
    res = person(app, "user-1").post("/api/identify/observation",
                                     json={"source": "inat", "id": 601})
    assert [c and c["licence"] for c in res.json()["observation"]["photos_credit"]] == want
    again = visitor(app).post("/api/v1/identify/inat/601", headers=SERVER).json()
    assert again["kept_from"] == "site" and "photos_credit" not in again["observation"]


# --- reading the photographer and licence from iNat and MO ------------------------------------

def test_an_inat_photo_is_credited_to_its_observer_by_name_else_login():
    f = licensed_source()
    obs = bo.parse_inat(f.session.inat[600])
    assert obs.photo_credits == [PHOTOGRAPHER] * 5
    assert bo.shown_photo_credit(obs, 0) == {"licence": "cc-by", "by": PHOTOGRAPHER}
    assert bo.shown_photo_credit(obs, 2) is None                 # all rights reserved
    raw = {**f.session.inat[600], "user": {"login": "jdoe", "name": ""}}
    assert bo.parse_inat(raw).photo_credits[0] == "jdoe"
    assert "user:(login:!t,name:!t)" in bo.INAT_FIELDS


def test_an_mo_photo_is_credited_to_its_copyright_holder_and_its_licence_read_by_name():
    raw = {"id": 777, "date": "2026-10-11", "consensus": {"name": "B y", "rank": "species"},
           "owner": {"login_name": "obsowner", "legal_name": "Obs Owner"},
           "primary_image": {"id": 5001, "original_url": MO_PHOTO.format(id=5001, size="orig"),
                             "license": "Creative Commons Attribution v4.0 (Wikipedia compatible)",
                             "copyright_holder": "Ann Smith"},
           "images": [{"id": 5002, "original_url": MO_PHOTO.format(id=5002, size="orig"),
                       "license": "Public Domain"}]}
    obs = bo.parse_mo(raw)
    assert obs.photo_credits == ["Ann Smith", "Obs Owner"]
    assert bo.shown_photo_credit(obs, 0) == {"licence": "cc-by", "by": "Ann Smith"}
    assert bo.shown_photo_credit(obs, 1) == {"licence": "pd", "by": "Obs Owner"}
    assert bo.licence_code("mo", "Creative Commons Non-commercial v3.0") == "cc-by-nc-sa"


def test_a_kept_runs_photos_are_read_only_from_the_sources_own_photo_hosts():
    f = fetcher()
    pairs = f.photos_at("inat", ["https://elsewhere.example/photos/9001/large.jpg",
                                 "http://static.inaturalist.org/photos/9001/large.jpg",
                                 MO_PHOTO.format(id=5001, size="960"), large(9001)])
    assert [i for i, _ in pairs] == [3]
    assert f.session.urls() == [large(9001)]
