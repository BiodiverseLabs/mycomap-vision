/**
 * The Identify page's three stages and the live numbers its empty state shows (Steve,
 * 2026-10-26: the approved redesign, .mockups/vision-identify-home). Every number comes from
 * /api/stats or /api/how-good; one the server doesn't send is left out, never made up.
 */
import { num, pct, RANKS, type HowGood, type Rank, type RunView, type Specimen, type Stats, type TopK } from "@/lib/api";
import { answerFor } from "@/lib/answer";
import { chanceWithheld, gateBanner } from "@/lib/gate";
import { sourceName } from "@/lib/observationLink";
import { longDate, recentWeekLine } from "@/lib/publicView";
import { runPath, runPhotos } from "@/lib/runLink";

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

/**
 * The empty page's "Example result" (Steve, 2026-10-26: "a real result with real images"):
 * one real run, named by the server (MV_EXAMPLE_RUN), as anyone with its link sees it. Its
 * photos are the observation's own Creative Commons ones and the specimens' photos the run
 * route re-checks against today's permissions, so nothing here is shown that the run's own
 * page wouldn't show. Null (no example) when the run has no firm answer to show, or its
 * photos may not be shown.
 */
export interface ExampleView {
  href: string;
  /** "iNaturalist 91443365", linking to the observation; null for an upload. */
  source: { label: string; url: string | null } | null;
  photos: string[];
  /** "Tom Norton (CC BY)": who took the find's photos, and their licences. */
  credit: string | null;
  headline: { rank: Rank; name: string; chance: string | null };
  above: { rank: Rank; name: string; chance: string | null }[];
  ranks: { rank: Rank; name: string; value: number; chance: string }[];
  specimens: Specimen[];
  /** Each photo on its own: how it rates the answer, and whether it puts it first. */
  perPhoto: { url: string; chance: string | null; agrees: boolean }[];
  /** How many photos the answer is from. */
  count: number;
}

const licenceWords = (code: string | null | undefined) =>
  code ? code.toLowerCase() === "cc0" ? "CC0" : code.toUpperCase().replace(/-/g, " ") : null;

export function exampleView(run: RunView | null | undefined, specimens = 3): ExampleView | null {
  const r = run?.results[0];
  if (!run || !r) return null;
  const banner = gateBanner(r);
  const a = answerFor(r, { notFungus: banner?.verdict === "not_fungus", withheld: chanceWithheld(r) });
  const h = a?.headline;
  if (!a || !h || a.lookAlike || h.confidence == null) return null;
  const photos = runPhotos(run).filter(Boolean);
  if (!photos.length) return null;
  const o = run.observation;
  const by = new Map<string, string | null>();
  for (const c of o?.photos_credit ?? []) if (c?.by && !by.has(c.by)) by.set(c.by, licenceWords(c.licence));
  const credit = by.size ? [...by].map(([who, lic]) => (lic ? `${who} (${lic})` : who)).join(", ") : null;
  const named = (n: { rank: Rank; name: string; confidence: number | null }) =>
    ({ rank: n.rank, name: n.name, chance: n.confidence == null ? null : pct(n.confidence) });
  const ranks = RANKS.flatMap((rank) => {
    const top = r.ranks[rank]?.[0];
    return top?.confidence == null ? [] : [{ rank, name: top.name, value: top.confidence, chance: pct(top.confidence) }];
  });
  const shown = runPhotos(run);
  return {
    href: runPath(run.id),
    source: o ? { label: `${sourceName(o.source)} ${o.id}`, url: o.url } : null,
    photos, credit,
    headline: named(h), above: a.above.map(named), ranks,
    specimens: r.specimens.filter((s) => s.photo_url).slice(0, specimens),
    perPhoto: r.per_photo.flatMap((p) => {
      const url = shown[p.photo];
      const sp = p.overall_top.species;
      return url ? [{ url, chance: sp?.confidence == null ? null : pct(sp.confidence), agrees: sp?.position === 1 }] : [];
    }),
    count: o?.photos_used ?? run.photos ?? r.photos,
  };
}
