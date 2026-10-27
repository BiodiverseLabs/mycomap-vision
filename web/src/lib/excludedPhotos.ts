// Research > Excluded photos (Steve, 2026-10-26): the photos people left out of Vision in Human
// Feedback, by why they were left out, and "where Vision looked" on any of them (looked.py,
// excluded_looked.py). Only photos anyone may see are shown; the rest are placeholders.
import { ApiError, type LookedPhoto } from "./api";

/** Why a photo was left out: photo_review.REASONS' codes, in the order the page offers them
 *  (most photos first, as people excluded them). */
export const CATEGORIES: { code: string; label: string }[] = [
  { code: "micrograph", label: "Micrographs" },
  { code: "slip", label: "Field slips and labels" },
  { code: "habitat", label: "Habitat and scenes" },
  { code: "uv", label: "UV reactions" },
  { code: "dried", label: "Dried specimens" },
  { code: "spore_print", label: "Spore prints" },
  { code: "chemical", label: "Chemical spot tests" },
  { code: "drawing_micro", label: "Drawings – microscopy" },
  { code: "phylo_tree", label: "Phylogenetic trees / BLAST" },
  { code: "people", label: "People" },
  { code: "other", label: "Other" },
  { code: "blurry", label: "Blurry or badly exposed" },
  { code: "not_fungus", label: "Not a fungus" },
  { code: "drawing_specimen", label: "Drawings – specimen" },
  { code: "species_description", label: "Species descriptions" },
];

export const categoryLabel = (code: string) => CATEGORIES.find((c) => c.code === code)?.label ?? code;

/** The one model a map comes from: the site's (Steve, 2026-10-26: no base model, no side by
 *  side). */
export const MAP_MODEL = "v1";
export const MAP_MODEL_LABEL = "MycoMap Vision v1";

export interface ExcludedTile {
  photo_key: string;
  record_key: string;
  /** The record's DNA-verified name: the name a map explains. */
  name: string;
  category: string;
  /** "public": a photo anyone may see (thumb and photo_url set); "hidden": a placeholder. */
  shown: "public" | "hidden";
  thumb?: string | null;
  photo_url?: string | null;
  owner_login?: string | null;
  license_code?: string | null;
  record_url?: string | null;
  /** ["v1"] when its map is kept already, else []. */
  maps: (typeof MAP_MODEL)[];
  /** Its map is held back until the paper's sealed test is scored (the photo still shows). */
  held?: boolean;
}

export interface ExcludedCategory { code: string; label: string; count: number }

export interface ExcludedPage {
  categories: ExcludedCategory[];
  total: number;
  offset: number;
  tiles: ExcludedTile[];
  /** When the list was read from Human Feedback. */
  as_of: string;
}

/** One photo's map: its state as the run maps' (200 ready, 202 working, 404 none, 503 failed). */
export type ExcludedLooked =
  | { status: "ready"; name: string; model: typeof MAP_MODEL; backbone: string; method: string; made_at: string;
      map: Omit<LookedPhoto, "photo" | "match">;
      /** Where the photo alone puts the record's name among every species, its own record left
       *  out: 1..top, or null below the top. */
      rank: number | null; top: number }
  | { status: "working"; photos: number; done: number; ahead: number }
  | { status: "none" }
  | { status: "failed"; detail?: string };

export const PAGE_SIZE = 48;

export const HELD_LINE = "This photo's record was first DNA-verified after the paper's test set was sealed, so its map is held back until that test is scored.";

/** The page's address for a category and page. */
export function listUrl(category: string, offset: number, withMaps: boolean, limit = PAGE_SIZE): string {
  const q = new URLSearchParams({ category, offset: String(offset), limit: String(limit) });
  if (withMaps) q.set("maps", "1");
  return `/api/research/excluded-photos?${q}`;
}

export const lookedUrl = (photoKey: string) =>
  `/api/research/excluded-photos/looked?${new URLSearchParams({ photo: photoKey, model: MAP_MODEL })}`;

async function lookedReply(res: Response): Promise<ExcludedLooked> {
  if (res.status === 200 || res.status === 202 || res.status === 503) return res.json();
  if (res.status === 404) {
    const body = await res.json().catch(() => null);
    if (body && body.status === "none") return { status: "none" };
  }
  const body = await res.json().catch(() => null);
  throw new ApiError(res.status, (body && typeof body.detail === "string" && body.detail) || res.statusText);
}

export const excludedApi = {
  async list(category: string, offset: number, withMaps: boolean): Promise<ExcludedPage> {
    const res = await fetch(listUrl(category, offset, withMaps));
    if (!res.ok) throw new ApiError(res.status, res.statusText);
    return res.json();
  },
  read: async (photoKey: string) => lookedReply(await fetch(lookedUrl(photoKey))),
  ask: async (photoKey: string) => lookedReply(await fetch(lookedUrl(photoKey), {
    method: "POST", credentials: "same-origin" })),
};

/** "1–48 of 16,245". */
export function rangeLine(offset: number, shown: number, total: number): string {
  if (!total) return "No photos";
  const f = (n: number) => n.toLocaleString("en-US");
  return `${f(offset + 1)}–${f(offset + shown)} of ${f(total)}`;
}

/** Where the photo alone puts the name: "first", "at number 4", "not in its top 10". */
export function rankLine(rank: number | null | undefined, top = 10): string {
  if (rank == null) return `not in its top ${top}`;
  return rank === 1 ? "first" : `at number ${rank}`;
}

/** The line under the button for a refusal, in the run maps' words. */
export function refusal(status: number, detail: string): string {
  if (status === 401) return "Sign in with your mycomap.org account to ask for a map. Maps someone already asked for are open to everyone.";
  if (status === 403) return HELD_LINE;
  if (status === 409) return `There is no map for this photo: ${detail.replace(/\.$/, "")}.`;
  if (status === 429) return "Vision is busy, or you have asked for a lot of maps; try again in a minute.";
  if (status === 503) return "The map could not be made just now. Try again later.";
  return "The map could not be loaded. Try again later.";
}
