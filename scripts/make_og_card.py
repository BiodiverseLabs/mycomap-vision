"""Draw web/public/og-card.jpg, the site's link-preview image (linkcards.py): 1200x630, the
MycoMap Vision brand and line on the left, a DNA-validated reference photo on the right with
its credit printed on it (Facebook shows no credit of its own).

The photo must be one MycoMap may publish: CC0 or CC BY, credited here. Steve's pick
(2026-10-20): iNat photo 328268128, Amanita muscaria var. guessowii, by Jessica Williams,
CC BY; scientific names only on the cards.

    python scripts/make_og_card.py <photo.jpg> --credit "Amanita muscaria var. guessowii · Jessica Williams, CC BY"
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from mycomap_vision.linkcards import CARD_SIZE, GREEN_TEXT, INK, font

REPO = Path(__file__).resolve().parent.parent
MUTED = (108, 101, 94)
GREEN = (143, 199, 62)


def draw(photo: Path, credit: str, logo: Path, focus=(0.55, 0.5)) -> Image.Image:
    W, H = CARD_SIZE
    card = Image.new("RGB", CARD_SIZE, "white")
    pw = 600
    with Image.open(photo) as img:
        card.paste(ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (pw, H),
                                Image.LANCZOS, centering=focus), (W - pw, 0))
    d = ImageDraw.Draw(card, "RGBA")
    cf = font(17)
    tw = d.textlength(credit, font=cf)
    d.rectangle([W - tw - 24, H - 34, W, H], fill=(0, 0, 0, 140))
    d.text((W - tw - 12, H - 30), credit, font=cf, fill=(255, 255, 255, 235))

    # The brand, large (Steve, 2026-10-20: "lets also make the MycoMap Vision logo larger").
    mark = Image.open(logo).convert("RGBA")
    mark.thumbnail((140, 140))
    card.paste(mark, (52, 40), mark)
    bx = 52 + mark.width + 18
    size = 56
    while size > 30 and bx + d.textlength("MycoMap Vision", font=font(size, bold=True)) > W - pw - 28:
        size -= 2
    bf = font(size, bold=True)
    by = 40 + (mark.height - bf.getbbox("MycoMap")[3]) // 2
    d.text((bx, by), "MycoMap", font=bf, fill=INK)
    d.text((bx + d.textlength("MycoMap ", font=bf), by), "Vision", font=bf, fill=GREEN_TEXT)

    hf = font(58, bold=True)
    y = 228
    for line in ("Identify fungi", "from your photos"):
        d.text((50, y), line, font=hf, fill=INK)
        y += 70
    sf = font(28)
    y += 18
    for line in ("Built on DNA-verified records,", "not community votes."):
        d.text((52, y), line, font=sf, fill=MUTED)
        y += 38
    pf = font(24, bold=True)
    pill = "Free · from the nonprofit MycoMap"
    d.rounded_rectangle([52, 548, 52 + d.textlength(pill, font=pf) + 36, 592], radius=22,
                        fill=GREEN + (255,))
    d.text((70, 554), pill, font=pf, fill="white")
    return card


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", type=Path)
    ap.add_argument("--credit", required=True)
    ap.add_argument("--logo", type=Path, default=REPO / "web" / "public" / "mycomap-logo.png")
    ap.add_argument("--out", type=Path, default=REPO / "web" / "public" / "og-card.jpg")
    a = ap.parse_args()
    draw(a.photo, a.credit, a.logo).save(a.out, quality=88, optimize=True, progressive=True)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
