/**
 * The answer box at the top of each Identify result (Steve, 2026-10-11): the finest settled
 * rank's top name, the coarser ranks under it, and the names still in play one rank finer.
 * Also the small text rules around it, so the page itself holds no logic.
 */
import { RANKS, type Candidate, type ConfusedWith, type IdentifyResult, type PlaceCue, type Rank } from "@/lib/api";
import { coverageWords } from "@/lib/publicView";

/** A rank counts as settled when its top name reaches this calibrated chance (Steve,
 *  2026-10-11). */
export const SETTLED = 0.7;
/** A species also counts as settled at this chance... (Steve, 2026-10-18: one rule with
 *  the app and .org collection importance) */
export const SPECIES_SETTLED_MIN = 0.5;
/** ...when the runner-up species is below this. */
export const SPECIES_RUNNER_UP_MAX = 0.2;

/**
 * Whether a rank's top name is settled: the one rule behind the headline. The server holds
 * the same rule (settled.py) for the API's `confident` flags and its "the genus looks
 * settled" hint; tests/data/settled_rule_cases.json is the case table both are tested on.
 * `runnerUp`: the best other name's chance at that rank, null when there is none.
 */
export function isSettled(rank: Rank, top: number | null, runnerUp: number | null): boolean {
  if (top == null) return false;
  if (top >= SETTLED) return true;
  return rank === "species" && top >= SPECIES_SETTLED_MIN && (runnerUp == null || runnerUp < SPECIES_RUNNER_UP_MAX);
}

/** At most this many names are listed as still in play. */
export const MAX_IN_PLAY = 3;
/** Without a likely set, a name below this chance isn't worth listing as in play. */
const IN_PLAY_FLOOR = 0.05;

const FINEST_FIRST: Rank[] = [...RANKS].reverse(); // species, genus, family

export interface Named {
  rank: Rank;
  name: string;
  /** null when the chance is withheld (photos may not show a fungus). */
  confidence: number | null;
  place?: PlaceCue;
  confused_with?: ConfusedWith;
}

export interface Answer {
  /** The headline: the finest settled rank's top name. A look-alike when the chance is
   *  withheld; null when no rank is settled. */
  headline: Named | null;
  /** True when the headline is only the closest look-alike (no chance stated). */
  lookAlike: boolean;
  /** The top names of the ranks above the headline (or of every rank when none is settled). */
  above: Named[];
  /** The names still in play one rank finer than the headline (at family when nothing is
   *  settled); null when the headline is at species or nothing is worth listing. */
  inPlay: { rank: Rank; names: Named[]; more: number } | null;
}

type Result = Pick<IdentifyResult, "ranks" | "likely">;

/** The top name at a rank: from its likely list when there is one, else the ranking. */
function top(r: Result, rank: Rank): Named | null {
  const fromLikely = r.likely?.[rank]?.names[0];
  const c = fromLikely ?? r.ranks[rank]?.[0];
  return c ? { rank, name: c.name, confidence: c.confidence, place: c.place, confused_with: c.confused_with } : null;
}

/** The best chance of any name other than `name` at a rank, in its ranking or likely list. */
function runnerUp(r: Result, rank: Rank, name: string): number | null {
  const others = [...(r.ranks[rank] ?? []), ...(r.likely?.[rank]?.names ?? [])]
    .filter((c) => c.name !== name && c.confidence != null)
    .map((c) => c.confidence as number);
  return others.length ? Math.max(...others) : null;
}

/** Names worth considering at a rank: its likely list, else the ranking's top few. */
function contenders(r: Result, rank: Rank): Named[] {
  const likely = r.likely?.[rank]?.names;
  const list = likely?.length
    ? likely
    : (r.ranks[rank] ?? []).filter((c, i) => i === 0 || (c.confidence ?? 0) >= IN_PLAY_FLOOR);
  return list.map((c) => ({ rank, name: c.name, confidence: c.confidence, place: c.place, confused_with: c.confused_with }));
}

/**
 * What the answer box says, or null when it should not show: the photos don't look like a
 * fungus (the gate's notice says so instead) or there are no names at all.
 */
export function answerFor(r: Result, gate: { notFungus: boolean; withheld: boolean }): Answer | null {
  if (gate.notFungus) return null;
  if (gate.withheld) {
    // No chance can be stated, so nothing is "settled": name the closest look-alike at the
    // finest rank that has one, and nothing else.
    const finest = FINEST_FIRST.map((rank) => top(r, rank)).find(Boolean) ?? null;
    return finest
      ? { headline: { ...finest, confidence: null, place: undefined }, lookAlike: true, above: [], inPlay: null }
      : null;
  }
  const tops = FINEST_FIRST.map((rank) => top(r, rank));
  if (!tops.some(Boolean)) return null;
  const at = tops.findIndex((t) => t != null && isSettled(t.rank, t.confidence, runnerUp(r, t.rank, t.name)));
  const headline = at >= 0 ? tops[at] : null;
  const headlineRank = headline?.rank;
  const above = RANKS.filter((rank) => !headlineRank || RANKS.indexOf(rank) < RANKS.indexOf(headlineRank))
    .map((rank) => top(r, rank))
    .filter((t): t is Named => t != null)
    .reverse(); // nearest rank first: genus before family
  // One rank finer than the headline; family when nothing is settled.
  const finer: Rank | null = headlineRank
    ? (RANKS[RANKS.indexOf(headlineRank) + 1] ?? null)
    : "family";
  let inPlay: Answer["inPlay"] = null;
  if (finer) {
    const names = contenders(r, finer);
    if (names.length) inPlay = { rank: finer, names: names.slice(0, MAX_IN_PLAY), more: Math.max(0, names.length - MAX_IN_PLAY) };
  }
  return {
    headline,
    lookAlike: false,
    above: headline ? above : [],
    inPlay,
  };
}

/**
 * The place lane's far-find cue for one name, in words; empty when the response carries
 * none (no place or date given, or no prior served). Distances are whole km and never
 * reveal where a record is.
 */
export function placeCue(place: PlaceCue | undefined | null): string[] {
  if (!place) return [];
  const out: string[] = [];
  // Distances come rounded to 10 km and measured to iNat's obscuring cell: "about".
  const km = typeof place.nearest_record_km === "number" ? Math.max(10, Math.round(place.nearest_record_km)) : null;
  if (place.far_from_known === true) {
    // A far find should stand out as a possible range extension, not read as an error.
    out.push(km != null
      ? `Far from known records, nearest about ${km.toLocaleString()} km (possible range extension)`
      : "Far from known records (possible range extension)");
  } else if (km != null) {
    out.push(`Known within about ${km.toLocaleString()} km`);
  }
  if (place.in_season === false) out.push("Outside its known season");
  return out;
}

/**
 * "Often confused with G. subspectabilis · 9% of DNA-checked finds" under a species name that
 * people often get wrong (mycomap.org/misid), as the server words it; null otherwise.
 */
export function confusedNote(n: { confused_with?: ConfusedWith | null } | null | undefined): string | null {
  const note = n?.confused_with?.note;
  return typeof note === "string" && note.trim() ? note : null;
}

/** The same note inside a sentence, in brackets after a name: "often confused with …". */
export function confusedAside(n: { confused_with?: ConfusedWith | null } | null | undefined): string | null {
  const note = confusedNote(n);
  return note ? note[0].toLowerCase() + note.slice(1) : null;
}

/** "From the photos alone: X" when the place and date changed the top name at a rank. */
export function photoOnlyNote(r: Pick<IdentifyResult, "photo_only_top">, headline: Named | null): string | null {
  if (!headline) return null;
  const alone = r.photo_only_top?.[headline.rank];
  return alone && alone !== headline.name ? `From the photos alone: ${alone}` : null;
}

/** The confidence note without internal run ids such as "(comparison 20261008-012435-4ef7b0)". */
export function publicNote(note: string): string {
  return note.replace(/\s*\(comparison [^)]*\)/gi, "").replace(/\s{2,}/g, " ").trim();
}

/**
 * One sentence for how often the likely lists held the right name, instead of one per rank.
 * Ranks with the same coverage share their words.
 */
export function coverageSummary(likely: IdentifyResult["likely"]): string | null {
  const parts = RANKS.filter((rank) => likely?.[rank]?.names.length)
    .map((rank) => ({ rank, words: coverageWords(likely![rank]!.coverage) }));
  if (!parts.length) return null;
  const groups: { ranks: Rank[]; words: string }[] = [];
  for (const p of parts) {
    const g = groups.find((x) => x.words === p.words);
    if (g) g.ranks.push(p.rank);
    else groups.push({ ranks: [p.rank], words: p.words });
  }
  const list = (xs: string[]) => (xs.length < 2 ? xs[0] : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);
  const body = groups.length === 1
    ? `the right name was in these lists ${groups[0].words} at ${list(groups[0].ranks)}`
    : `the right name was in these lists ${groups.map((g) => `${g.words} at ${list(g.ranks)}`)
        .reduce((acc, x, i, all) => (i === 0 ? x : `${acc}${i === all.length - 1 ? ", and " : ", "}${x}`), "")}`;
  return `In tests on the newest DNA-verified records, ${body}.`;
}

/** True when a model that was run uses place and date (range and season). */
export function usesPlace(methods: string[]): boolean {
  return methods.some((m) => m.includes("+prior") || m.includes("+place"));
}

/**
 * The server's hints for the answer box. Its "the genus looks settled, but A and B are
 * close" opening repeats the box's "species not settled" line, so only the advice after it
 * stays.
 */
export function boxHints(hints: string[], speciesInPlay: boolean): string[] {
  if (!speciesInPlay) return hints;
  return hints.map((h) => {
    const m = /^The genus looks settled, but .+? are close\.\s+(.+)$/s.exec(h);
    return m ? m[1].charAt(0).toUpperCase() + m[1].slice(1) : h;
  });
}

// --- the full breakdown (Steve, 2026-10-13: the answer first, every rank one click away) --

/** Names under this chance are left out of the breakdown: they would read "0%". */
export const BREAKDOWN_FLOOR = 0.01;
/** Without a fitted likely set (one photo on its own), names at this chance or more get a bar. */
export const LISTED_FLOOR = 0.05;
/** At most this many names with a bar, and this many in the "also possible" line. */
export const BREAKDOWN_MAX = 3;

/** The names with a bar at one rank when there is no likely set: the top name, then any at
 *  LISTED_FLOOR or more. Empty when the chance is withheld. */
export function listedNames(candidates: Candidate[]): (Candidate & { confidence: number })[] {
  const [top, ...rest] = candidates;
  if (!top || top.confidence == null) return [];
  return [top, ...rest.filter((c) => (c.confidence ?? 0) >= LISTED_FLOOR)]
    .slice(0, BREAKDOWN_MAX)
    .map((c) => ({ ...c, confidence: c.confidence ?? 0 }));
}

/** "Also possible, less likely": the names not listed, at BREAKDOWN_FLOOR or more. */
export function alsoPossible(candidates: Candidate[], listed: { name: string }[]): Candidate[] {
  const shown = new Set(listed.map((n) => n.name));
  return candidates.filter((c) => !shown.has(c.name) && (c.confidence ?? 0) >= BREAKDOWN_FLOOR)
    .slice(0, BREAKDOWN_MAX);
}

/** Whether the breakdown shows: when asked for, and always when there is no answer to lead with. */
export function breakdownOpen(asked: boolean, hasAnswer: boolean, noRanks: boolean): boolean {
  return !noRanks && (asked || !hasAnswer);
}

/** The breakdown's button, saying which view it breaks down. `photo`: 0-based, or null for all. */
export function breakdownLabel(photos: number, photo: number | null): string {
  const which = photos > 1 ? (photo == null ? `: all ${photos} photos` : `: photo ${photo + 1} on its own`) : "";
  return `Full breakdown${which} · every rank, with its chances`;
}

/** Whether a place cue warns (far from known records, or out of season). */
export function placeWarns(place: PlaceCue | undefined | null): boolean {
  return place?.far_from_known === true || place?.in_season === false;
}

/** The place lines under a name: under the names still in play only a warning ("known
 *  nearby" there is noise); under the headline everything placeCue says. */
export function placeLines(place: PlaceCue | undefined | null, inPlay: boolean): string[] {
  return inPlay && !placeWarns(place) ? [] : placeCue(place);
}
