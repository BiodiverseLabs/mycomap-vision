/** iNaturalist's suggestions under MycoMap Vision's answer (inat_suggest.py; Steve, 2026-10-25).
 *  The server sends them with Vision's answer, or `pending` while the photos are still with
 *  iNat; the page then asks again until they are back or the wait is over. Never shown
 *  before Vision's answer, never to anyone but the member whose iNat account asked. */

export interface InatLine {
  taxon_id: number;
  name: string;
  rank: string;
  score: number;
  /** "iNaturalist suggests · 82%", as the app says it. */
  label: string;
}

export interface InatNotFungus {
  share: number;
  top?: { taxon_id: number; name: string; rank: string; score: number } | null;
  /** "iNaturalist suggests this may not be a fungus · 92%". */
  label: string;
}

/** One photo's full answer from iNat, as mycomap.org parsed it (the website's dropdown). */
export interface InatFull {
  common_ancestor?: { taxon_id: number; name: string; rank: string } | null;
  results: { taxon_id?: number; name: string; rank?: string; score?: number | null;
             vision_score?: number | null; seen_nearby?: boolean; in_scope?: boolean }[];
}

export type InatView =
  | { pending: true; ref: string }
  | { pending?: undefined; lines: InatLine[]; not_fungus: InatNotFungus | null; full: InatFull[] | null };

/** How long the page keeps asking, from Vision's answer (the server gives up at 12 s from
 *  the photos' arrival, so this is only a backstop), and how often. */
export const POLL_MS = 1500;
export const GIVE_UP_MS = 15_000;

/** The page's own name for a find, sent with its photos (`inat_ref`), so iNat's lines can
 *  still be asked for when Vision's answer fails or never arrives. */
export function newInatRef(): string {
  const abc = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
  return Array.from(crypto.getRandomValues(new Uint8Array(16)), (b) => abc[b & 63]).join("");
}

/** What the page asks for, by how Vision's answer stands: nothing while Vision is still
 *  checking; what came with its answer; after a failure or timeout, the find by its own
 *  ref (Steve, 2026-10-25: the lines then show with a Retry). */
export function inatForPage(vision: "checking" | "answered" | "failed", answered: InatView | null | undefined,
                            ref: string | null | undefined):
    { initial: InatView; visionFailed: boolean } | null {
  if (vision === "checking") return null;
  if (vision === "answered") return answered ? { initial: answered, visionFailed: false } : null;
  return ref ? { initial: { pending: true, ref: `site-${ref}` }, visionFailed: true } : null;
}

export type InatState = "none" | "checking" | "shown";

export function inatState(view: InatView | null | undefined): InatState {
  if (!view) return "none";
  if (view.pending) return "checking";
  return view.lines.length || view.not_fungus || view.full?.length ? "shown" : "none";
}

/** When to ask again: while pending and within the wait; otherwise never. */
export function pollInterval(view: InatView | null | undefined, startedAt: number, now: number): number | false {
  return view?.pending && now - startedAt < GIVE_UP_MS ? POLL_MS : false;
}

/** What the page shows: past the wait a still-pending answer is nothing (dropped quietly). */
export function settledView(view: InatView | null | undefined, startedAt: number, now: number): InatView | null {
  if (!view) return null;
  if (view.pending && now - startedAt >= GIVE_UP_MS) return null;
  return view;
}

export interface InatRow { photo: number; name: string; rank: string; score: string; value: number | null; nearby: string }

/** The dropdown at first: the best 10 rows above 1% (Steve, 2026-10-25: iNat's full answer can
 *  run to 200 rows, mostly orders and families under 1%), the rest behind "Show all". */
export const TOP_ROWS = 10;
export const TOP_ROWS_ABOVE = 1;

/** The dropdown's rows: every photo's suggestions, best first, scores rounded. */
export function fullRows(full: InatFull[] | null | undefined): InatRow[] {
  return (full ?? []).flatMap((f, i) =>
    [...f.results]
      .sort((a, b) => (b.score ?? -1) - (a.score ?? -1))
      .map((r) => ({
        photo: i + 1,
        name: r.name,
        rank: r.rank ?? "",
        score: r.score == null ? "" : `${Math.round(r.score)}%`,
        value: r.score ?? null,
        nearby: r.seen_nearby ? "yes" : "",
      })));
}

/** The rows shown before "Show all": above 1%, best first over every photo, at most 10. */
export function topRows(rows: InatRow[]): InatRow[] {
  return rows
    .filter((r) => r.value != null && r.value > TOP_ROWS_ABOVE)
    .sort((a, b) => (b.value ?? 0) - (a.value ?? 0))
    .slice(0, TOP_ROWS);
}
