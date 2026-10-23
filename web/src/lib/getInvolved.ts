// The Get involved page's numbers, shaped for its sections (Steve, 2026-10-22: the approved
// redesign). Every figure comes from the API as it is now (/api/stats, /api/how-good,
// /api/feedback/summary, /api/docs/v1.json), each open to a signed-out visitor; a part the
// server doesn't send is left out, never made up.

import type { FeedbackSummary, HowGood, Stats } from "./api";
import type { Outline } from "./apiOutline";
import { groupBySection } from "./feedback";
import { namesByRecords } from "./howItWorks";

/** The feeds the page reads, every one open signed out (tests/test_get_involved.py). */
export const GET_INVOLVED_FEEDS = ["/api/stats", "/api/how-good", "/api/feedback/summary",
                                   "/api/docs/v1.json"] as const;

/** mycomap.org pages the page links to (its routes, as the page linked them before). */
export const PROJECTS_URL = "https://mycomap.org/projects";
export const START_PROJECT_URL = "https://mycomap.org/start-project";
export const VISION_KEYS_URL = "https://mycomap.org/vision-keys";

/** Names resting on a single DNA-verified record: the banner's share. */
export function singleRecordNames(s: Stats | undefined): { n: number; names: number; share: number } | null {
  const one = namesByRecords(s)?.find((r) => r.key === "1");
  return one && s ? { n: one.n, names: s.names, share: one.share } : null;
}

/** The depth chart's rows: species right first time by the true species' reference records
 *  (/api/how-good species_by_depth), only the bands the run measured. */
export function depthBars(h: HowGood | undefined): { band: string; label: string; none: boolean;
                                                     top1: number; n: number }[] | null {
  const rows = (h?.species_by_depth ?? [])
    .filter((b) => typeof b.top1 === "number" && Number.isFinite(b.top1))
    .map((b) => ({ band: b.band, label: b.band.replace("-", "–"), none: b.band === "0",
                   top1: b.top1 as number, n: b.n }));
  return rows.length ? rows : null;
}

/** The project grid: one dot per project, or one per few when there are too many to draw. */
export function projectDots(n: number | null | undefined, max = 1200): { dots: number; per: number } | null {
  if (!n || !Number.isFinite(n) || n <= 0) return null;
  const per = Math.ceil(n / max);
  return { dots: Math.ceil(n / per), per };
}

/** The review groups in the page's order, each with a short name. */
export const REVIEW_ORDER = ["validation", "cleanup", "views"];
const REVIEW_TITLE: Record<string, string> = { validation: "Validation", cleanup: "Cleanup",
                                               views: "Photo views" };
/** The queues each group names first, the plainest to a newcomer; the rest by open items. */
const FEATURED = ["label_vs_own_dna", "photos_unlike_label", "reference_label_doubt",
                  "non_specimen_photo", "crosspost_twin", "name_inactive_on_inat", "photo_view"];

export interface ReviewGroup {
  code: string;
  title: string;
  queues: number;
  /** Open items this visitor can browse, across the group's queues. */
  open: number;
  /** Up to `shown` queues, by name; a group of one queue says what it holds instead. */
  lines: string[];
  more: number;
}

/** A queue's description cut to its first clause: "What each photo shows: cap top, gills,
 *  stem and base…". */
export function firstClause(text: string, max = 60): string {
  const end = text.search(/(\.\s|…|\.$)/);
  let t = (end >= 0 ? text.slice(0, end) : text).trim();
  if (t.length <= max && text[end] !== "…") return t;
  if (t.length > max) {
    const comma = t.lastIndexOf(",", max);
    t = (comma > 0 ? t.slice(0, comma) : t.slice(0, max)).trim();
  }
  return `${t}…`;
}

/** Human Feedback's queues, grouped as the page shows them, with their open counts (never
 *  how many were decided, nor by whom). Null when the summary didn't load. */
export function reviewGroups(summary: FeedbackSummary[] | null | undefined, shown = 3): ReviewGroup[] | null {
  if (!summary?.length) return null;
  const rank = (code: string) => (REVIEW_ORDER.includes(code) ? REVIEW_ORDER.indexOf(code) : REVIEW_ORDER.length);
  const feat = (t: string) => (FEATURED.includes(t) ? FEATURED.indexOf(t) : FEATURED.length);
  return groupBySection(summary)
    .map((g, i) => ({ g, i }))
    .sort((a, b) => rank(a.g.code) - rank(b.g.code) || a.i - b.i)
    .map(({ g }) => {
      const ordered = g.types.map((t, i) => ({ t, i }))
        .sort((a, b) => feat(a.t.type) - feat(b.t.type) || (b.t.open || 0) - (a.t.open || 0) || a.i - b.i)
        .map(({ t }) => t);
      const lines = g.types.length === 1 && ordered[0].about
        ? [firstClause(ordered[0].about)]
        : ordered.slice(0, shown).map((t) => t.title);
      return { code: g.code, title: REVIEW_TITLE[g.code] ?? g.title, queues: g.types.length, open: g.open,
               lines, more: Math.max(0, g.types.length - lines.length) };
    });
}

/** The developer card: how many endpoints answer now, the ones it names (each only while
 *  the outline lists it as available) and the identify address. Null without the outline. */
export function apiCard(o: Outline | null | undefined): {
  available: number; identifyUrl: string | null; highlights: { method: string; text: string }[];
} | null {
  if (!o?.sections) return null;
  const eps = o.sections.flatMap((s) => s.blocks)
    .flatMap((b) => (b.type === "endpoint" && b.status === "available" ? [b] : []));
  const has = (method: string, path: string) => eps.find((e) => e.method === method && e.path === path);
  const identify = has("POST", "/identify");
  const highlights: { method: string; text: string }[] = [];
  if (identify) highlights.push({ method: "POST", text: identify.summary.replace(/\.$/, "") });
  if (has("POST", "/identify/inat/{observation_id}") && has("POST", "/identify/mo/{observation_id}")) {
    highlights.push({ method: "POST", text: "Identify an iNaturalist or Mushroom Observer observation" });
  }
  if (has("POST", "/sessions") && has("POST", "/sessions/{session_id}/photos")) {
    highlights.push({ method: "POST", text: "Observation sessions, for apps that add photos as they're taken" });
  }
  if (has("GET", "/taxon-counts")) {
    highlights.push({ method: "GET", text: "Reference records per species, genus and family" });
  }
  return { available: eps.length, highlights,
           identifyUrl: identify && o.base_url ? `${o.base_url.replace(/\/$/, "")}/identify` : null };
}
