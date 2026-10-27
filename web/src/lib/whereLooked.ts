// "Where Vision looked" (looked.py; Steve, 2026-10-26): how a photo's map is drawn, where the
// square the model sees sits on the photo, the numbered match spots, and what the panel says
// while the maps are made or when they can't be.
import type { LookedPhoto, LookedView, PhotoResult } from "./api";

/** Spotlight keeps the photo's strongest 30% of patches clear and dims the rest. */
export const KEEP = 0.3;
/** How often the page asks again while the maps are made. */
export const POLL_MS = 1500;

/** Spotlight: per cell, how much to darken it (0 to 0.8). By rank, not by value, so every
 *  photo shows its own strongest evidence, whatever the scale of its map. */
export function spotlightAlpha(map: number[], keep = KEEP): number[] {
  const n = map.length;
  if (n < 2) return map.map(() => 0);
  const order = map.map((v, i) => [v, i] as const).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const rank = new Array<number>(n);
  order.forEach(([, i], pos) => { rank[i] = pos / (n - 1); });
  return rank.map((r) => Math.pow(Math.min(1, Math.max(0, (1 - keep - r) / (1 - keep))), 0.6) * 0.8);
}

/** Each cell's colour: [r, g, b, alpha], the spotlight's dimming. Spotlight is the one look
 *  (Steve, 2026-10-26: Glow is gone). */
export function cellColours(map: number[]): [number, number, number, number][] {
  return spotlightAlpha(map).map((a) => [18, 18, 18, a]);
}

const pc = (v: number) => `${+(v * 100).toFixed(2)}%`;

/** Where the square the model sees sits on the photo (CSS box, percentages). */
export function cropBox(crop: LookedPhoto["crop"]): { left: string; top: string; width: string; height: string } {
  return { left: pc(crop.x), top: pc(crop.y), width: pc(crop.w), height: pc(crop.h) };
}

/** The strips of the photo the model never sees (a tall or wide photo), as CSS boxes. */
export function unseenStrips(crop: LookedPhoto["crop"]): { side: "top" | "bottom" | "left" | "right"; style: Record<string, string> }[] {
  const out: { side: "top" | "bottom" | "left" | "right"; style: Record<string, string> }[] = [];
  if (crop.y > 0.01) {
    out.push({ side: "top", style: { left: "0", right: "0", top: "0", height: pc(crop.y) } });
    out.push({ side: "bottom", style: { left: "0", right: "0", bottom: "0", height: pc(1 - crop.y - crop.h) } });
  }
  if (crop.x > 0.01) {
    out.push({ side: "left", style: { top: "0", bottom: "0", left: "0", width: pc(crop.x) } });
    out.push({ side: "right", style: { top: "0", bottom: "0", right: "0", width: pc(1 - crop.x - crop.w) } });
  }
  return out;
}

/** A whole photo placed so that only its seen square fills a square box (CSS, percentages). */
export function squareOf(crop: LookedPhoto["crop"]): { left: string; top: string; width: string; height: string } {
  return { width: pc(1 / crop.w), height: pc(1 / crop.h), left: pc(-crop.x / crop.w), top: pc(-crop.y / crop.h) };
}

/** The numbered match spots on each square, in percent. */
export function markers(pairs: number[][]): { n: number; photo: { x: number; y: number }; specimen: { x: number; y: number } }[] {
  return pairs.map(([qx, qy, rx, ry], i) => ({
    n: i + 1, photo: { x: qx * 100, y: qy * 100 }, specimen: { x: rx * 100, y: ry * 100 },
  }));
}

/** The photo the panel opens on: the one that, on its own, puts the answer highest. */
export function firstPhoto(photos: LookedPhoto[], perPhoto: PhotoResult[]): number | null {
  if (!photos.length) return null;
  const own = (i: number) => perPhoto.find((p) => p.photo === i)?.overall_top.species;
  const score = (i: number) => {
    const o = own(i);
    return o ? (o.position === 1 ? 1 : 0) + (o.confidence ?? 0) / 2 : -1;
  };
  return photos.map((p) => p.photo).reduce((best, i) => (score(i) > score(best) ? i : best));
}

/** "On its own, photo 5 puts Laetiporus sulphureus first, at 93%." */
export function ownLine(photo: number, perPhoto: PhotoResult[]): { lead: string; name: string; tail: string } | null {
  const o = perPhoto.find((p) => p.photo === photo)?.overall_top.species;
  if (!o) return null;
  const where = o.position === 1 ? "first" : `at number ${o.position}`;
  const chance = o.confidence == null ? "" : `, at ${Math.round(o.confidence * 100)}%`;
  return { lead: `On its own, photo ${photo + 1} puts `, name: o.name, tail: ` ${where}${chance}.` };
}

/** What the button says while the maps are made. */
export function workingLine(view: Extract<LookedView, { status: "working" }>): string {
  if (view.done === 0 && view.ahead > 0) return "Waiting for a quiet moment on the server…";
  return view.photos > 1 ? `Looking at photo ${Math.min(view.done + 1, view.photos)} of ${view.photos}…`
    : "Looking at your photo…";
}

/** The panel's line when the maps can't be shown, from the server's refusal. */
export function refusalLine(status: number, detail: string): string {
  if (status === 401) return "Sign in with your mycomap.org account to see where Vision looked.";
  if (status === 409) return `There is nothing to show here: ${detail.replace(/\.$/, "")}.`;
  if (status === 429) return "Vision is busy right now. Try again in a minute.";
  if (status === 503) return "The maps could not be made just now. Try again later.";
  return "The maps could not be loaded. Try again later.";
}
