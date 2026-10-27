// Recovered labels (Steve, 2026-10-26): field-slip photos a person excluded, where the specimen
// detector found a specimen and proposes a crop box. A crop goes back into Vision unless a person
// rejects it. The crop is coordinates on the ORIGINAL photo (slipcrop.py), never a new image file:
// the page draws the box over the original and, for reviewers, shows the box's part of the same
// image by CSS. The public read-only view shows only the outline on the original.

import type { RecoveredCrop, RecoveredTile } from "./api";

export const RECOVERED_TYPE = "recovered_label";

/** The slip tranches (the photo-type tagger's slip p when the photo was seeded), best first. */
export const TRANCHES: { code: string; label: string }[] = [
  { code: "all", label: "Every tranche" },
  { code: "high", label: "High (p ≥ 0.60)" },
  { code: "medium", label: "Medium (0.40–0.60)" },
  { code: "low", label: "Low (0.20–0.40)" },
  { code: "rule", label: "Earlier rules (no tagger p)" },
];

export const RECOVERED_SORTS: { code: string; label: string }[] = [
  { code: "riskiest", label: "Smallest specimen first" },
  { code: "largest", label: "Largest specimen first" },
  { code: "newest", label: "Newest exclusion first" },
];

/** A box as CSS percentages of the photo it is drawn on. */
export function boxStyle(box: RecoveredCrop["box"], width: number, height: number) {
  const [l, t, r, b] = box;
  const pct = (v: number, of: number) => `${Math.max(0, Math.min(100, (v / of) * 100)).toFixed(3)}%`;
  return { left: pct(l, width), top: pct(t, height), width: pct(r - l, width), height: pct(b - t, height) };
}

/** The CSS that shows only the box's part of the original image in a square frame: the image as a
 *  background, scaled so the box fills the frame, and placed at the box. No new image is made. */
export function cropBackground(src: string, box: RecoveredCrop["box"], width: number, height: number) {
  const [l, t, r, b] = box;
  const bw = Math.max(1, r - l), bh = Math.max(1, b - t);
  const side = Math.max(bw, bh);                  // a non-square box is centred in the square
  const ox = l - (side - bw) / 2, oy = t - (side - bh) / 2;
  const sizeX = (width / side) * 100, sizeY = (height / side) * 100;
  // background-position % maps the image point p to the frame point p: solve for the box's corner
  const pos = (o: number, img: number) => (img === side ? 0 : (o / (img - side)) * 100);
  return {
    backgroundImage: `url("${src.replace(/"/g, "%22")}")`,
    backgroundSize: `${sizeX.toFixed(3)}% ${sizeY.toFixed(3)}%`,
    backgroundPosition: `${pos(ox, width).toFixed(3)}% ${pos(oy, height).toFixed(3)}%`,
    backgroundRepeat: "no-repeat",
  };
}

/** The share of the photo a crop keeps, as a percent for the tile ("crop 18% of the photo"). */
export const cropShare = (box: RecoveredCrop["box"], width: number, height: number): number =>
  Math.round(((box[2] - box[0]) * (box[3] - box[1]) * 100) / Math.max(1, width * height));

export type CropChoice = "include" | "reject";
/** The reviewer's choice per crop, by `${item_id}:${n}`. */
export type CropMarks = Record<string, CropChoice>;

export const cropKey = (itemId: number, n: number) => `${itemId}:${n}`;

/** Where each crop starts: its saved decision, else Include (the queue's default). */
export function startChoice(tile: Pick<RecoveredTile, "status">, crop: Pick<RecoveredCrop, "decision">): CropChoice {
  return crop.decision ?? "include";
}

export function flipCrop(marks: CropMarks, tile: RecoveredTile, n: number): CropMarks {
  const crop = tile.crops.find((c) => c.n === n);
  if (!crop) return marks;
  const k = cropKey(tile.item_id, n);
  const now = marks[k] ?? startChoice(tile, crop);
  return { ...marks, [k]: now === "include" ? "reject" : "include" };
}

// --- moving and resizing a crop (Steve, 2026-10-27: "click the original photo and move/modify
// the cropped area") ------------------------------------------------------------------------------

export type Box = RecoveredCrop["box"];
/** A reviewer's own box per crop (original photo pixels), by cropKey. */
export type BoxEdits = Record<string, Box>;
export type Handle = "move" | "nw" | "ne" | "sw" | "se";

/** The smallest crop side, in photo pixels: 4% of the short side, at least 16. */
export const minSide = (w: number, h: number) => Math.max(16, Math.round(0.04 * Math.min(w, h)));

/** `box` dragged by (dx, dy) photo pixels at `handle`: moved whole, or one corner moved, never
 *  past the photo's edges and never smaller than minSide. Whole pixels. */
export function dragBox(box: Box, handle: Handle, dx: number, dy: number, w: number, h: number): Box {
  let [l, t, r, b] = box;
  const m = minSide(w, h);
  if (handle === "move") {
    const bw = r - l, bh = b - t;
    l = Math.min(Math.max(0, l + dx), w - bw);
    t = Math.min(Math.max(0, t + dy), h - bh);
    return [Math.round(l), Math.round(t), Math.round(l + bw), Math.round(t + bh)];
  }
  if (handle.includes("w")) l = Math.min(Math.max(0, l + dx), r - m);
  if (handle.includes("e")) r = Math.max(Math.min(w, r + dx), l + m);
  if (handle.includes("n")) t = Math.min(Math.max(0, t + dy), b - m);
  if (handle.includes("s")) b = Math.max(Math.min(h, b + dy), t + m);
  return [Math.round(l), Math.round(t), Math.round(r), Math.round(b)];
}

/** The box a crop has now: the reviewer's edit, else the saved or proposed one. */
export const boxOf = (tile: Pick<RecoveredTile, "item_id">, crop: Pick<RecoveredCrop, "n" | "box">,
                      edits: BoxEdits): Box => edits[cropKey(tile.item_id, crop.n)] ?? crop.box;

const sameBox = (a: Box, b: Box) => a.every((v, i) => v === b[i]);

/** One photo's decision as sent: the crops kept (an empty list rejects them all), any box a
 *  reviewer moved or resized (by crop number), and the exclusion reason when it changed. */
export interface RecoveredDecision {
  item_id: number;
  include: number[];
  boxes?: Record<string, Box>;
  reason?: string;
}

/** The exclusion reasons a reviewer changed, by item id. */
export type ReasonEdits = Record<number, string>;

/** The decisions to save for these tiles: every open tile (Include is a decision too, once the
 *  reviewer has seen it), and a decided tile only where a crop, a box or the reason changed. */
export function pendingDecisions(tiles: RecoveredTile[], marks: CropMarks, seen: Set<number>,
                                 edits: BoxEdits = {}, reasons: ReasonEdits = {}): RecoveredDecision[] {
  const out: RecoveredDecision[] = [];
  for (const t of tiles) {
    const choice = (c: RecoveredCrop) => marks[cropKey(t.item_id, c.n)] ?? startChoice(t, c);
    const moved = t.crops.filter((c) => !sameBox(boxOf(t, c, edits), c.box));
    // A box is sent only for a kept crop (the server refuses one for a rejected crop).
    const keptMoved = moved.filter((c) => choice(c) === "include");
    const reason = reasons[t.item_id] && reasons[t.item_id] !== t.reason ? reasons[t.item_id] : undefined;
    const changed = t.crops.some((c) => choice(c) !== startChoice(t, c)) || moved.length > 0 || !!reason;
    if ((t.status === "open" && seen.has(t.item_id)) || changed) {
      const d: RecoveredDecision = { item_id: t.item_id, include: t.crops.filter((c) => choice(c) === "include").map((c) => c.n) };
      if (keptMoved.length) d.boxes = Object.fromEntries(keptMoved.map((c) => [String(c.n), boxOf(t, c, edits)]));
      if (reason) d.reason = reason;
      out.push(d);
    }
  }
  return out;
}

/** Only a photo left out as a field slip gives crops (the server's rule): another reason, or
 *  keeping the photo, leaves its crops unused. */
export const CROP_REASON = "slip";
export const KEEP_PHOTO = "keep";
export const cropsUsed = (reason: string | null | undefined) => (reason ?? CROP_REASON) === CROP_REASON;

/** The tile's one-line description: why it is here and how big the find is. */
export function tileFacts(t: Pick<RecoveredTile, "slip_p" | "specimen_area" | "crops">): string {
  const parts = [];
  if (t.slip_p != null) parts.push(`slip p ${t.slip_p.toFixed(2)}`);
  parts.push(`specimen ${(t.specimen_area * 100).toFixed(t.specimen_area < 0.1 ? 1 : 0)}% of the photo`);
  if (t.crops.length > 1) parts.push(`${t.crops.length} crops`);
  return parts.join(" · ");
}
