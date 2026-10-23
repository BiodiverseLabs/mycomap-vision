// Answered before the DNA verdict (GET /api/early-answers): what the Results card shows.
import type { EarlyAnswersGroup } from "./api";

export const LEVELS = ["species strict", "species s.l.", "species complex (beta)", "genus strict",
                       "genus s.l."] as const;
export const KS = [1, 3, 5, 10] as const;
export const DEPTH_BANDS = ["0", "1-4", "5-19", "20-99", "100+"] as const;

/** Why a first answer was left out, in the card's words (early_answers.OUTCOMES). */
export const LEFT_OUT_LABELS: Record<string, string> = {
  vision_assisted: "validated with Vision's help",
  answered_after_verdict: "answered on or after the verdict",
  in_reference: "itself a reference record when answered",
  twin_in_reference: "a twin was in the reference",
  not_a_fungus_label: "not a fungus label",
};

/** The left-out counts as "label: n", Vision's help first and always shown (Steve,
 *  2026-10-23); other reasons only when they happened. */
export function leftOutLines(g: EarlyAnswersGroup): string[] {
  const out: string[] = [];
  for (const [key, label] of Object.entries(LEFT_OUT_LABELS)) {
    const n = g.left_out?.[key] ?? 0;
    if (key === "vision_assisted" || n > 0) out.push(`${label}: ${n.toLocaleString("en-US")}`);
  }
  return out;
}

/** "1 Oct – 3 Oct 2026", one day, or null when nothing is scored yet. */
export function dayRange(r: { from: string | null; through: string | null }): string | null {
  if (!r.from || !r.through) return null;
  const fmt = (d: string, year: boolean) => new Date(`${d}T00:00:00Z`).toLocaleDateString(
    "en-GB", { day: "numeric", month: "short", ...(year ? { year: "numeric" } : {}), timeZone: "UTC" });
  return r.from === r.through ? fmt(r.from, true) : `${fmt(r.from, false)} – ${fmt(r.through, true)}`;
}

/** Answers held back for the paper's sealed test, over every group (signed in only). */
export function sealedCount(groups: EarlyAnswersGroup[]): number {
  return groups.reduce((n, g) => n + (g.sealed ?? 0), 0);
}
