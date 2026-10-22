"""Draw web/public/og-card.jpg, the site's link-preview image (linkcards.py): 1200x630, the
MycoMap Vision brand and line over a full-bleed photo.

The photo must be one MycoMap may publish. Steve's pick (2026-10-20: "Lets use the main
background hero from the main site"): mycomap.org's homepage hero, Jesse Bauer's photo on
Unsplash (pzwH-a4aF3s; Unsplash License, no credit required), as mycota.com serves it:
https://mycota.com/wp-content/uploads/2024/05/jesse-bauer-pzwH-a4aF3s-unsplash-scaled.jpg

    python scripts/make_og_card.py <photo.jpg>
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from mycomap_vision.linkcards import CARD_SIZE, GREEN_TEXT, INK, font

REPO = Path(__file__).resolve().parent.parent
GREEN = (143, 199, 62)


def draw(photo: Path, credit: str, logo: Path, focus=(0.5, 0.5)) -> Image.Image:
    """The photo fills the card, darkened towards the left, with the brand and the line in
    white over it: the look of mycomap.org's own homepage hero."""
    W, H = CARD_SIZE
    with Image.open(photo) as img:
        card = ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), CARD_SIZE,
                            Image.LANCZOS, centering=focus).convert("RGBA")
    # Dark only behind the words on the left, clear from just past the middle, so the right
    # half shows the photo as it is (Steve, 2026-10-20: "have it not go as dark black to the
    # right").
    reach = int(W * 0.66)
    column = Image.new("L", (W, 1))
    column.putdata([int(205 * max(0.0, 1 - x / reach) ** 1.25) for x in range(W)])
    fade = column.resize((W, H))
    shade = Image.new("RGBA", CARD_SIZE, (8, 10, 6, 255))
    shade.putalpha(fade)
    card.alpha_composite(shade)
    d = ImageDraw.Draw(card, "RGBA")
    if credit:
        cf = font(17)
        tw = d.textlength(credit, font=cf)
        d.text((W - tw - 14, H - 30), credit, font=cf, fill=(255, 255, 255, 200))

    # The brand, large (Steve, 2026-10-20: "lets also make the MycoMap Vision logo larger").
    mark = Image.open(logo).convert("RGBA")
    mark.thumbnail((130, 130))
    card.alpha_composite(mark, (56, 44))
    bf = font(60, bold=True)
    bx = 56 + mark.width + 20
    by = 44 + (mark.height - bf.getbbox("MycoMap")[3]) // 2
    d.text((bx, by), "MycoMap", font=bf, fill="white")
    d.text((bx + d.textlength("MycoMap ", font=bf), by), "Vision", font=bf, fill=GREEN)

    hf = font(66, bold=True)
    y = 232
    for line in ("Identify fungi", "from your photos"):
        d.text((54, y), line, font=hf, fill="white")
        y += 80
    sf = font(30)
    y += 14
    # Steve's pick, 2026-10-22: Vision as a new foundation model for fungi, still built on DNA.
    d.text((57, y), "A new foundation for fungal ID, built on DNA.", font=sf,
           fill=(255, 255, 255, 225))
    pf = font(25, bold=True)
    pill = "Free · from the nonprofit MycoMap"
    d.rounded_rectangle([57, 540, 57 + d.textlength(pill, font=pf) + 40, 588], radius=24,
                        fill=GREEN + (255,))
    d.text((77, 548), pill, font=pf, fill="white")
    return card.convert("RGB")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--credit", default="", help="printed bottom right, when the licence asks")
    ap.add_argument("--logo", type=Path, default=REPO / "web" / "public" / "mycomap-logo.png")
    ap.add_argument("--out", type=Path, default=REPO / "web" / "public" / "og-card.jpg")
    a = ap.parse_args()
    draw(a.photo, a.credit, a.logo).save(a.out, quality=88, optimize=True, progressive=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
