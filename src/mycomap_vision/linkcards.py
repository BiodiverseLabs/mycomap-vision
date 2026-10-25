"""Link previews: the card Facebook, X, iMessage, Slack and the like show for a Vision link
(Open Graph and X card tags; Steve, 2026-10-20: "lets create a much better card and text for
vision.mycomap.org links on Facebook").

Every page needs sign-in, so a crawler used to be sent to /auth/signin and came back with
mycomap.org's login card. Now the pages people share answer with the site's page shell
(index.html, the same file /research/api and every /r/<id> already serve to anyone), with
the card's tags written into its head:

- `/`, `/about` and `/get-involved` (SHARE_PAGES): the site card, its text as Steve picked
  it, and web/public/og-card.jpg. A signed-out person still goes to sign in: the app sends
  them (web/src/lib/research.ts mustSignIn), and every answer and photo stays behind /api/.
- `/r/<id>`, a run's own link: the answer as its page states it (the same settled rule,
  settled.py) and a card image made from the find's photos (`run_card_jpeg`). Never the
  place, the date or who ran it. A deleted run, or an id never given out, answers 404 with
  the site's image and "this result isn't available".
- An iNat or MO run's card is drawn from the observation's own photos, but only those
  free to reuse with a credit (CARD_LICENCES: CC0, CC BY, public domain; never NC, SA, ND
  or all rights reserved), with the credit printed on it ("Jane Doe, CC BY via
  iNaturalist"; Steve, 2026-10-20: shared results "include images from that search"). The
  card is made when the run is kept, off the request (a crawler gives up after a few
  seconds; MO's pace alone is 5 s a photo): the photos are read from the source once, by
  the server, and the card is kept beside the run (runs.RunStore.keep_card). Until it is
  kept, and when no photo qualifies, the tags name the site's image.

Pure functions only; api.py serves them.
"""

from __future__ import annotations

import hashlib
import html
import io
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from .guards import photo_formats
from .settled import runner_up, settled

SITE_NAME = "MycoMap Vision"
# Steve's pick, 2026-10-20 ("ID fungi not fungus ID"; then tagline A, not "community votes").
SITE_TITLE = "MycoMap Vision: ID fungi, built on DNA"
SITE_DESCRIPTION = ("Matches your photos with 150,000+ DNA-verified records and can name "
                    "species known only from DNA. Free with a mycomap.org account. "
                    "Not edibility advice.")
SITE_IMAGE = "/og-card.jpg"
SITE_IMAGE_ALT = ("MycoMap Vision: identify fungi from your photos. A new foundation for "
                  "fungal ID, built on DNA.")
CARD_SIZE = (1200, 630)

# The pages people share. Every other page still sends a signed-out visitor to sign in.
SHARE_PAGES = {
    "/": SITE_TITLE,
    "/about": "How MycoMap Vision works",
    "/get-involved": "Get involved with MycoMap Vision",
}

RUN_DESCRIPTION = ("From photos, compared with DNA-sequenced specimens. Not DNA-confirmed, "
                   "and not edibility advice. Open the link for every rank and the matching "
                   "specimens.")
GONE_TITLE = "MycoMap Vision: this result isn't available"
GONE_DESCRIPTION = "The link is wrong, or the person who ran it deleted it."

RANKS_FINEST_FIRST = ("species", "genus", "family")

# The licences an observation's photo may carry to go on a card, and how its credit names
# them. CC BY needs the photographer's name; CC0 and public domain need none.
CARD_LICENCES = {"cc0": "CC0", "cc-by": "CC BY", "pd": "public domain"}
SOURCE_NAMES = {"inat": "iNaturalist", "mo": "Mushroom Observer"}
MAX_CARD_PHOTOS = 5


@dataclass(frozen=True)
class Card:
    title: str
    description: str
    url: str
    image: str
    image_alt: str = ""


def image_version(path: Path) -> str | None:
    """The first 10 hex of the site image's sha256, for its address (`?v=`): Facebook keeps
    an image by its URL, so a new picture needs a new address before a re-scrape shows it
    (2026-10-20: the old card stayed after tagline A went live). None when there is no file."""
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    except OSError:
        return None


def site_image(origin: str, version: str | None = None) -> str:
    return origin + SITE_IMAGE + (f"?v={version}" if version else "")


def site_card(origin: str, path: str, version: str | None = None) -> Card | None:
    """The card of a page people share, or None for any other page."""
    title = SHARE_PAGES.get(path)
    if title is None:
        return None
    return Card(title, SITE_DESCRIPTION, origin + path, site_image(origin, version),
                SITE_IMAGE_ALT)


def gone_card(origin: str, rid: str, version: str | None = None) -> Card:
    return Card(GONE_TITLE, GONE_DESCRIPTION, f"{origin}/r/{rid}", site_image(origin, version),
                SITE_IMAGE_ALT)


# --- what a run's answer says ----------------------------------------------------------------

@dataclass(frozen=True)
class Headline:
    """What a run's card says. `kind`: settled | look_alike | unsettled | not_fungus | none."""
    kind: str
    rank: str | None = None
    name: str | None = None
    confidence: float | None = None


def _top(result: dict, rank: str) -> dict | None:
    likely = ((result.get("likely") or {}).get(rank) or {}).get("names") or []
    ranked = (result.get("ranks") or {}).get(rank) or []
    c = likely[0] if likely else (ranked[0] if ranked else None)
    return c if c and c.get("name") else None


def headline(result: dict | None) -> Headline:
    """The answer box's headline (web/src/lib/answer.ts answerFor, gate.ts): the finest
    settled rank's top name; the closest look-alike when the photos may not show a fungus
    (no chance stated); the top family when nothing is settled."""
    if not result:
        return Headline("none")
    gate = result.get("gate") or {}
    verdict = gate.get("verdict")
    if verdict == "not_fungus":
        return Headline("not_fungus")
    withheld = bool(result.get("confidence_withheld")) or (bool(gate) and verdict != "fungus")
    tops = [(rank, _top(result, rank)) for rank in RANKS_FINEST_FIRST]
    if withheld:
        for rank, c in tops:
            if c:
                return Headline("look_alike", rank, c["name"])
        return Headline("none")
    if not any(c for _, c in tops):
        return Headline("none")
    for rank, c in tops:
        if not c:
            continue
        other = runner_up(c["name"], (result.get("ranks") or {}).get(rank) or [],
                          ((result.get("likely") or {}).get(rank) or {}).get("names") or [])
        if settled(rank, c.get("confidence"), other):
            return Headline("settled", rank, c["name"], c.get("confidence"))
    family = _top(result, "family")
    if family:
        return Headline("unsettled", "family", family["name"], family.get("confidence"))
    return Headline("none")


def pct(value: float | None) -> str:
    return "" if value is None else f"{round(100 * value)}%"


def _ranked(h: Headline) -> str:
    """'Pluteus thomsonii', 'genus Pluteus', 'family Pluteaceae'."""
    return h.name if h.rank == "species" else f"{h.rank} {h.name}"


def run_title(h: Headline) -> str:
    if h.kind == "settled":
        return f"MycoMap Vision says: {_ranked(h)} ({pct(h.confidence)})"
    if h.kind == "look_alike":
        return f"MycoMap Vision: the closest look-alike is {_ranked(h)}"
    if h.kind == "unsettled":
        chance = f", {pct(h.confidence)}" if h.confidence is not None else ""
        return f"MycoMap Vision: no confident answer yet ({_ranked(h)}{chance})"
    if h.kind == "not_fungus":
        return "MycoMap Vision: these photos don't look like a fungus"
    return "An identification from MycoMap Vision"


def run_answer(run: dict) -> dict | None:
    """The answer a run's page leads with: its first model's, all photos together."""
    models = run.get("models") or []
    return (models[0] or {}).get("result") if models else None


def credit(source: str | None, licence: str | None, by: str | None) -> str | None:
    """The credit a card prints for an observation's photo ("Jane Doe, CC BY via
    iNaturalist"), or None when the photo may not go on a card: a licence other than
    CARD_LICENCES, or CC BY with no one to credit."""
    code = (licence or "").strip().lower()
    words = CARD_LICENCES.get(code)
    where = SOURCE_NAMES.get(source or "")
    by = " ".join(str(by or "").split())
    if words is None or where is None or (code == "cc-by" and not by):
        return None
    return f"{by + ', ' if by else ''}{words} via {where}"


def card_photos(run: dict) -> list[tuple[str, str]]:
    """An iNat/MO run's photos that may go on its card, as (address at the source, credit),
    in the page's order, at most MAX_CARD_PHOTOS. Only shown photos (photos_shown) whose
    kept licence is CC0, CC BY or public domain (photos_credit); none for an upload run, or
    for a run kept before licences were (its card is the site's image)."""
    if run.get("kind") != "observation":
        return []
    obs = run.get("observation") or {}
    shown = obs.get("photos_shown") or []
    credits = obs.get("photos_credit") or []
    out = []
    for url, about in zip(shown, credits):
        line = (credit(obs.get("source"), about.get("licence"), about.get("by"))
                if url and isinstance(about, dict) else None)
        if line:
            out.append((url, line))
    return out[:MAX_CARD_PHOTOS]


def run_card(origin: str, rid: str, run: dict, version: str | None = None,
             card_ready: bool = False) -> Card:
    """A run's card. Only the answer and the find's own photos go in it: never the place,
    the date, the observation, or who ran it (nor who took an observation's photos: that
    credit is printed on the image). `card_ready`: an iNat/MO run's card.jpg is kept (made
    off the request, when the run was kept); until then the tags name the site image, so a
    crawler never keeps a stand-in under the card's address."""
    h = headline(run_answer(run))
    if run.get("kind") == "photos":
        photos = int(run.get("photos") or 0)
    else:                   # an iNat/MO run: its card once made (api.py card_later)
        photos = len(card_photos(run)) if card_ready else 0
    image = f"{origin}/api/runs/{rid}/card.jpg" if photos else site_image(origin, version)
    alt = (f"{photos} photo{'s' if photos != 1 else ''} of one find, identified by MycoMap Vision"
           if photos else SITE_IMAGE_ALT)
    return Card(run_title(h), RUN_DESCRIPTION, f"{origin}/r/{rid}", image, alt)


# --- the tags --------------------------------------------------------------------------------

def tags(card: Card) -> str:
    """Open Graph and X card tags, every value escaped."""
    w, h = CARD_SIZE

    def meta(attr: str, key: str, value: str) -> str:
        return f'<meta {attr}="{key}" content="{html.escape(value, quote=True)}" />'
    rows = [
        meta("property", "og:type", "website"),
        meta("property", "og:site_name", SITE_NAME),
        meta("property", "og:title", card.title),
        meta("property", "og:description", card.description),
        meta("property", "og:url", card.url),
        meta("property", "og:image", card.image),
        meta("property", "og:image:width", str(w)),
        meta("property", "og:image:height", str(h)),
        meta("property", "og:image:alt", card.image_alt),
        meta("name", "twitter:card", "summary_large_image"),
        meta("name", "twitter:title", card.title),
        meta("name", "twitter:description", card.description),
        meta("name", "twitter:image", card.image),
        meta("name", "twitter:image:alt", card.image_alt),
    ]
    return "\n    ".join(rows)


def with_tags(shell: str, card: Card | None) -> str:
    """The page shell with the card's tags at the end of its head."""
    if card is None:
        return shell
    block = "<!-- link preview -->\n    " + tags(card) + "\n  "
    at = shell.find("</head>")
    return shell[:at] + block + shell[at:] if at >= 0 else block + shell


# --- a run's card image ----------------------------------------------------------------------

GAP = 4
INK = (28, 25, 23)
GREEN_TEXT = (98, 145, 34)
GREEN_ON_DARK = (170, 222, 95)
FONT_DIRS = (Path("/usr/share/fonts/truetype/dejavu"), Path("C:/Windows/Fonts"))
FONT_FILES = {False: ("DejaVuSans.ttf", "segoeui.ttf", "arial.ttf"),
              True: ("DejaVuSans-Bold.ttf", "segoeuib.ttf", "arialbd.ttf")}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """DejaVu Sans on the box (fonts-dejavu-core), Segoe or Arial on Windows, else Pillow's
    own font: a card is never refused for want of a font."""
    for d in FONT_DIRS:
        for name in FONT_FILES[bold]:
            if (d / name).is_file():
                return ImageFont.truetype(str(d / name), size)
    return ImageFont.load_default(size)


def tiles(n: int, size: tuple[int, int] = CARD_SIZE) -> list[tuple[int, int, int, int]]:
    """Where each photo goes, (x, y, w, h): one fills the card; otherwise the first is large
    on the left and the next ones (up to four) sit beside it; a sixth and later are left
    out."""
    W, H = size
    if n <= 1:
        return [(0, 0, W, H)]
    big = 600 if n >= 4 else 700
    x, rw = big + GAP, W - big - GAP
    if n == 2:
        return [(0, 0, big, H), (x, 0, rw, H)]
    hh = (H - GAP) // 2
    if n == 3:
        return [(0, 0, big, H), (x, 0, rw, hh), (x, H - hh, rw, hh)]
    if n == 4:
        th = (H - 2 * GAP) // 3
        return [(0, 0, big, H)] + [(x, i * (th + GAP), rw, th if i < 2 else H - 2 * (th + GAP))
                                   for i in range(3)]
    cw = (rw - GAP) // 2
    return [(0, 0, big, H)] + [(x + (i % 2) * (W - x - cw), 0 if i < 2 else H - hh, cw, hh)
                               for i in range(4)]


def _slanted(text: str, size: int, fill) -> Image.Image:
    """Text in a bold face slanted like an italic (the box has no italic sans), for a
    scientific name."""
    f = font(size, bold=True)
    left, top, right, bottom = f.getbbox(text)
    w, h = right + 4, bottom + 4
    lean = 0.21
    extra = int(lean * h) + 2
    layer = Image.new("RGBA", (w + extra, h), (0, 0, 0, 0))
    # Drawn at the left edge: the slant moves the top to the right, into `extra`.
    ImageDraw.Draw(layer).text((0, 0), text, font=f, fill=fill)
    return layer.transform(layer.size, Image.AFFINE, (1, lean, -lean * h, 0, 1, 0),
                           resample=Image.BICUBIC)


def badge(card: Image.Image, logo: Image.Image | None, at: tuple[int, int] = (28, 28)) -> int:
    """The MycoMap Vision badge: the logo and the name on a white pill (large, Steve
    2026-10-20: "lets also make the MycoMap Vision logo larger"); its right edge."""
    d = ImageDraw.Draw(card, "RGBA")
    f = font(44, bold=True)
    x, y = at
    pad, height = 14, 96
    mark = None
    if logo is not None:
        mark = logo.convert("RGBA")
        mark.thumbnail((height - 12, height - 12))
    words_x = x + pad + (mark.width + 12 if mark else 8)
    width = words_x - x + d.textlength("MycoMap Vision", font=f) + 26
    d.rounded_rectangle([x, y, x + width, y + height], radius=height // 2,
                        fill=(255, 255, 255, 238))
    if mark:
        card.paste(mark, (x + pad, y + (height - mark.height) // 2), mark)
    ty = y + (height - (f.getbbox("MycoMap")[3])) // 2 - 2
    d.text((words_x, ty), "MycoMap", font=f, fill=INK)
    d.text((words_x + d.textlength("MycoMap ", font=f), ty), "Vision", font=f, fill=GREEN_TEXT)
    return int(x + width)


def fitted(d: ImageDraw.ImageDraw, text: str, f, room: float) -> str:
    """`text`, cut short with an ellipsis to fit `room` px."""
    if d.textlength(text, font=f) <= room:
        return text
    while text and d.textlength(text + "…", font=f) > room:
        text = text[:-1]
    return text.rstrip(" ,") + "…"


def credits_on(card: Image.Image, lines: list[str], left: int) -> None:
    """Each photo credit on a dark pill at the top right, clear of the badge (`left`)."""
    d = ImageDraw.Draw(card, "RGBA")
    f = font(22)
    y, pad, height = 28, 14, 38
    room = card.width - 28 - left - 24 - 2 * pad
    for line in lines:
        text = fitted(d, line, f, room)
        w = d.textlength(text, font=f) + 2 * pad
        x = card.width - 28 - w
        d.rounded_rectangle([x, y, x + w, y + height], radius=height // 2, fill=(0, 0, 0, 170))
        d.text((x + pad, y + (height - f.getbbox("Ag")[3]) // 2 - 1), text, font=f,
               fill=(255, 255, 255, 240))
        y += height + 6


def card_lines(h: Headline, photos: int) -> tuple[str, str | None, str | None, str]:
    """(the line above, the rank word, the name, the chance) printed on a run's card."""
    from_photos = f"from {photos} photo{'s' if photos != 1 else ''}"
    if h.kind == "settled":
        return (f"MycoMap Vision says, {from_photos}",
                None if h.rank == "species" else h.rank, h.name, pct(h.confidence))
    if h.kind == "look_alike":
        return (f"Closest look-alike, {from_photos}",
                None if h.rank == "species" else h.rank, h.name, "")
    if h.kind == "unsettled":
        return (f"No confident answer yet, {from_photos}", h.rank, h.name, pct(h.confidence))
    if h.kind == "not_fungus":
        return ("These photos don't look like a fungus", None, None, "")
    return (f"Identified by MycoMap Vision, {from_photos}", None, None, "")


def run_card_jpeg(photos: list[bytes], h: Headline, logo: Image.Image | None = None,
                  quality: int = 85, credits: list[str] = (),
                  answered_from: int | None = None) -> bytes:
    """A run's 1200x630 card: its photos (first large, up to four beside it), the badge, and
    the answer printed at the bottom. Made from the run's own kept copies (no EXIF), or an
    observation's CC0 / CC BY photos with their `credits` printed at the top right (each
    distinct one once). The card is re-encoded from pixels, so no metadata comes back.
    `answered_from`: how many photos the answer used, when not the ones shown."""
    W, H = CARD_SIZE
    card = Image.new("RGBA", CARD_SIZE, (24, 24, 22, 255))
    shown = photos[:5]
    for body, (x, y, w, hh) in zip(shown, tiles(len(shown))):
        with Image.open(io.BytesIO(body), formats=photo_formats()) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            card.paste(ImageOps.fit(img, (w, hh), Image.LANCZOS), (x, y))
    band = 230
    fade = Image.linear_gradient("L").resize((W, band)).point(
        lambda v: int(220 * (v / 255) ** 1.3))
    shade = Image.new("RGBA", (W, band), (0, 0, 0, 255))
    shade.putalpha(fade)
    card.alpha_composite(shade, (0, H - band))
    right = badge(card, logo)
    credits_on(card, list(dict.fromkeys(credits)), right)
    above, rank, name, chance = card_lines(h, answered_from or len(photos))
    d = ImageDraw.Draw(card, "RGBA")
    d.text((36, H - 160), above, font=font(30), fill=(255, 255, 255, 235))
    x, base = 36, H - 108
    if rank:
        rf = font(32, bold=True)
        d.text((x, base + 14), rank, font=rf, fill=(255, 255, 255, 235))
        x += int(d.textlength(rank + " ", font=rf))
    if name:
        size = 62 if len(name) <= 26 else 48 if len(name) <= 36 else 38
        layer = _slanted(name, size, (255, 255, 255, 255)) if rank in (None, "species", "genus") \
            else None
        if layer is None:       # family names are not italic
            nf = font(size, bold=True)
            d.text((x, base), name, font=nf, fill="white")
            x += int(d.textlength(name, font=nf))
        else:
            card.alpha_composite(layer, (x, base))
            x += layer.width
    if chance:
        d.text((x + 18, base + 6), chance, font=font(46, bold=True), fill=GREEN_ON_DARK)
    out = io.BytesIO()
    card.convert("RGB").save(out, "JPEG", quality=quality, optimize=True)
    return out.getvalue()
