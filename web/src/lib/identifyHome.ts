/**
 * The Identify page's three stages and the live numbers its empty state shows (Steve,
 * 2026-10-26: the approved redesign, .mockups/vision-identify-home). Every number comes from
 * /api/stats or /api/how-good; one the server doesn't send is left out, never made up.
 */
import { num, pct, RANKS, type HowGood, type Rank, type Stats, type TopK } from "@/lib/api";
import { longDate, recentWeekLine } from "@/lib/publicView";

/**
 * "empty": nothing chosen yet; the hero sells what Vision does. "photos": photos are in,
 * ready to identify. "answer": identifying or answered; the answer is the page's hero and
 * the photo box moves to the side, folded.
 * An error leaves the page where it was asked from, with the photos and the message.
 */
export type HomeStage = "empty" | "photos" | "answer";

export function homeStage(s: { photos: number; pending: boolean; answered: boolean }): HomeStage {
  if (s.pending || s.answered) return "answer";
  return s.photos > 0 ? "photos" : "empty";
}

/** The three cards under "Why MycoMap Vision", each null when its numbers aren't there. */
export interface WhyNumbers {
  references: { records: string; photos: string | null; week: string | null } | null;
  provisional: { names: string; total: string; share: number; sharePct: string } | null;
  tested: { records: string; after: string | null; rows: TopKRow[] } | null;
}

export interface TopKRow { rank: Rank; cells: { k: number; value: string }[] }

const RANK_ORDER: Rank[] = [...RANKS].reverse(); // species, genus, family

/** Top-1/3/5/10 per rank, finest first ("59.2%"). Null without a top-k report. */
export function topKTable(h: HowGood | undefined | null): TopKRow[] | null {
  const t = h?.top_k;
  if (!t?.k?.length) return null;
  const rows = RANK_ORDER.flatMap((rank) => {
    const s = t[rank] as TopK | null | undefined;
    if (!s) return [];
    const cells = t.k.map((k) => ({ k, v: s[`top${k}` as keyof TopK] as number | null | undefined }))
      .filter((c): c is { k: number; v: number } => c.v != null)
      .map((c) => ({ k: c.k, value: pct(c.v, 1) }));
    return cells.length ? [{ rank, cells }] : [];
  });
  return rows.length ? rows : null;
}

export function whyNumbers(stats: Stats | undefined | null, howGood: HowGood | undefined | null): WhyNumbers {
  const served = stats ? Object.values(stats.embedded ?? {}).filter((n) => n > 0) : [];
  const references = stats && stats.records_north_america > 0 ? {
    records: num(stats.records_north_america),
    photos: served.length ? num(Math.max(...served)) : null,
    week: recentWeekLine(stats.recent_week),
  } : null;
  const prov = stats?.names_provisional;
  const provisional = stats && prov != null && prov > 0 && stats.names > 0 ? {
    names: num(prov), total: num(stats.names), share: prov / stats.names,
    sharePct: `${Math.round((100 * prov) / stats.names)}%`,
  } : null;
  const rows = topKTable(howGood);
  const run = howGood?.run;
  const tested = rows && run && run.n_test > 0 ? {
    records: num(run.n_test), after: run.cutoff ? longDate(run.cutoff) : null, rows,
  } : null;
  return { references, provisional, tested };
}
