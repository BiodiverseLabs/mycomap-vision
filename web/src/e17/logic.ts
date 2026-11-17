// The E17 review page's logic, as pure functions (tested in logic.test.ts). Everything here
// works from the item's own JSON: no lookups, nothing from another host.

export type Hit = {
  hit_no: number;
  name: string;
  identity: number;
  coverage: number;
  /** GenBank only: an accession, or null for a community sequence (schema v1.3). */
  accession?: string | null;
};
export type Search = "stored" | "none";
export type Sequence = {
  n: number;
  withdrawn: boolean;
  length: number | null;
  local_search: Search;
  genbank_search: Search;
  local: Hit[];
  genbank: Hit[];
};
export type Photo = { n: number; width: number; height: number };
export type Item = {
  schema: string;
  code: string;
  name_under_review: string;
  place: string;
  when: string;
  photos: Photo[];
  sequences: Sequence[];
};
export type Verdict = "supported" | "wrong" | "cannot_tell";
export type ShownBy = "sequences" | "photographs" | "both";
export type Answer = {
  verdict: Verdict | null;
  better_name: string;
  shown_by: ShownBy | null;
  recognised: boolean;
  looked_up: boolean;
  notes: string;
};
export type SavedResponse = {
  verdict: Verdict | null;
  better_name: string | null;
  shown_by: ShownBy | null;
  recognised: boolean;
  looked_up: boolean;
  notes: string | null;
  client_saved_at: string | null;
  saved_at: string;
};
export type ListEntry = { code: string; position: number; answered: boolean };
export type Table = "local" | "genbank";

export const EMPTY_ANSWER: Answer = {
  verdict: null, better_name: "", shown_by: null, recognised: false, looked_up: false, notes: "",
};

export function answerFrom(saved: SavedResponse | null | undefined): Answer {
  if (!saved) return { ...EMPTY_ANSWER };
  return {
    verdict: saved.verdict ?? null,
    better_name: saved.better_name ?? "",
    shown_by: saved.shown_by ?? null,
    recognised: !!saved.recognised,
    looked_up: !!saved.looked_up,
    notes: saved.notes ?? "",
  };
}

// --- names -------------------------------------------------------------------------------

// Hit names that are no name at all: never a suggestion, never agreeing with anything.
export const PLACEHOLDER_NAMES = ["unnamed local sequence", "(no name)", "(no current name)"];
// A second word that is not a species epithet: the name agrees only with itself.
const NOT_EPITHET = new Set(["sp.", "sp", "spp.", "spp", "cf.", "cf", "aff.", "aff", "nr.", "nr", "?"]);
const BARE_GENUS_SUFFIX = new Set(["sp.", "sp", "spp.", "spp"]);

/** Names compared as text: trimmed, single spaces, lower case, straight quotes. */
export function normName(name: string): string {
  return name
    .replace(/[‘’‛′]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\s+/g, " ")
    .trim()
    .toLowerCase();
}

export function isPlaceholder(name: string): boolean {
  return PLACEHOLDER_NAMES.includes(normName(name));
}

/** The genus as written (the first word), or null for a placeholder or no name. */
export function genusOf(name: string): string | null {
  if (isPlaceholder(name)) return null;
  const first = name.trim().split(/\s+/)[0] ?? "";
  return /^[A-Za-z][A-Za-z-]*$/.test(first) ? first : null;
}

/** What two names must share to be the same species: genus + epithet, or, for a name with
 * no epithet (sp. 'IN07', cf. x, sp.), the whole name. Null for a genus alone or no name. */
export function speciesKey(name: string): string | null {
  if (genusOf(name) === null) return null;
  const words = normName(name).split(" ");
  if (words.length < 2) return null;
  if (NOT_EPITHET.has(words[1])) return words.join(" ");
  return `${words[0]} ${words[1]}`;
}

/** A genus by itself, or with only sp./sp/spp. after it ("Russula", "Russula sp."). A
 * provisional name ("Russula sp. 'IN01'") is not bare. */
export function isBareGenus(name: string): boolean {
  if (genusOf(name) === null) return false;
  const words = normName(name).split(" ");
  return words.length === 1 || (words.length === 2 && BARE_GENUS_SUFFIX.has(words[1]));
}

export type Agreement = "sp" | "gen" | "other";

/** A hit's agreement with the name under review: same species, same genus, or other. */
export function agreement(hitName: string, reviewed: string): Agreement {
  const a = speciesKey(hitName);
  const b = speciesKey(reviewed);
  if (a !== null && a === b) return "sp";
  const g = genusOf(hitName);
  const h = genusOf(reviewed);
  if (g !== null && h !== null && g.toLowerCase() === h.toLowerCase()) return "gen";
  return "other";
}

export const MARKER: Record<Agreement, string> = { sp: "=sp", gen: "=gen", other: "" };

// --- the per-sequence summary --------------------------------------------------------------

export type BestHit = { name: string; identity: number; coverage: number; table: Table };
export type Match = { rank: number; identity: number } | null;
export type Summary = {
  best: BestHit | null;
  local: Match;
  genbank: Match;
  localState: TableState;
  genbankState: TableState;
};

/** What a table holds: hits, a stored search that found none, or no stored search. */
export type TableState = "hits" | "no-hits" | "no-search";

export function tableState(seq: Sequence, table: Table): TableState {
  const search = table === "local" ? seq.local_search : seq.genbank_search;
  if (search === "none") return "no-search";
  return (table === "local" ? seq.local : seq.genbank).length ? "hits" : "no-hits";
}

export const TABLE_STATE_TEXT: Record<Exclude<TableState, "hits">, string> = {
  "no-hits": "No hits",
  "no-search": "No stored search",
};

function firstMatch(hits: Hit[], reviewed: string): Match {
  const i = hits.findIndex((h) => agreement(h.name, reviewed) === "sp");
  return i < 0 ? null : { rank: i + 1, identity: hits[i].identity };
}

/** Best hit: each table's rank 1 (array order is rank), the higher identity, then the higher
 * coverage, then local. And the first hit agreeing with the reviewed name in each table. */
export function summarise(seq: Sequence, reviewed: string): Summary {
  const tops: BestHit[] = [];
  if (seq.local.length) tops.push({ ...pick(seq.local[0]), table: "local" });
  if (seq.genbank.length) tops.push({ ...pick(seq.genbank[0]), table: "genbank" });
  let best: BestHit | null = null;
  for (const t of tops) {
    if (!best || t.identity > best.identity || (t.identity === best.identity && t.coverage > best.coverage)) {
      best = t;
    }
  }
  return {
    best,
    local: firstMatch(seq.local, reviewed),
    genbank: firstMatch(seq.genbank, reviewed),
    localState: tableState(seq, "local"),
    genbankState: tableState(seq, "genbank"),
  };
}

function pick(h: Hit) {
  return { name: h.name, identity: h.identity, coverage: h.coverage };
}

export function pct(x: number, decimals = 1): string {
  return x.toFixed(decimals);
}

export function coverageText(x: number): string {
  return Number.isInteger(x) ? String(x) : x.toFixed(1);
}

const TABLE_LABEL: Record<Table, string> = { local: "Local", genbank: "GenBank" };

/** The summary's two lines, as text. */
export function summaryLines(s: Summary): { best: string; reviewed: string } {
  const best = s.best
    ? `${s.best.name} · ${pct(s.best.identity)}% identity · ${coverageText(s.best.coverage)}% coverage (${s.best.table === "local" ? "local" : "GenBank"})`
    : s.localState === "no-search" && s.genbankState === "no-search"
      ? TABLE_STATE_TEXT["no-search"]
      : TABLE_STATE_TEXT["no-hits"];
  const part = (t: Table, m: Match, state: TableState) =>
    state !== "hits"
      ? `${TABLE_LABEL[t]}: ${TABLE_STATE_TEXT[state]}`
      : m ? `${TABLE_LABEL[t]} rank ${m.rank} · ${pct(m.identity)}%` : `${TABLE_LABEL[t]}: no hit with this name`;
  return {
    best,
    reviewed: `${part("local", s.local, s.localState)}  |  ${part("genbank", s.genbank, s.genbankState)}`,
  };
}

// --- sorting ------------------------------------------------------------------------------

export type SortKey = "rank" | "name" | "identity" | "coverage";
export type SortDir = "asc" | "desc";
export type Sort = { key: SortKey; dir: SortDir };
export const DEFAULT_SORT: Sort = { key: "rank", dir: "asc" };

/** A heading clicked: the same column flips; a new one starts highest first for numbers,
 * A to Z for names, rank 1 first. */
export function nextSort(current: Sort, key: SortKey): Sort {
  if (current.key === key) return { key, dir: current.dir === "asc" ? "desc" : "asc" };
  return { key, dir: key === "identity" || key === "coverage" ? "desc" : "asc" };
}

export type RankedHit = Hit & { rank: number };

/** Hits sorted (a new array); ties keep rank order. Rank is the stored array order. */
export function sortHits(hits: Hit[], sort: Sort): RankedHit[] {
  const ranked = hits.map((h, i) => ({ ...h, rank: i + 1 }));
  const sign = sort.dir === "asc" ? 1 : -1;
  const cmp = (a: RankedHit, b: RankedHit): number => {
    switch (sort.key) {
      case "name":
        return normName(a.name).localeCompare(normName(b.name));
      case "identity":
        return a.identity - b.identity;
      case "coverage":
        return a.coverage - b.coverage;
      default:
        return a.rank - b.rank;
    }
  };
  return ranked.sort((a, b) => sign * cmp(a, b) || a.rank - b.rank);
}

// --- better-name suggestions ----------------------------------------------------------------

export type NamedIdentity = { name: string; identity: number };

/** Every hit of the item, in order: sequence by sequence, local before GenBank, by rank. */
function allHits(item: Item): Hit[] {
  return item.sequences.flatMap((s) => [...s.local, ...s.genbank]);
}

/** The distinct real names in the item's hits (compared as text), each with its best
 * identity, highest first; ties keep the order the names first appear. */
export function distinctNames(item: Item): NamedIdentity[] {
  const seen = new Map<string, NamedIdentity & { order: number }>();
  allHits(item).forEach((h, order) => {
    if (isPlaceholder(h.name) || !h.name.trim()) return;
    const key = normName(h.name);
    const had = seen.get(key);
    if (!had) seen.set(key, { name: h.name.trim().replace(/\s+/g, " "), identity: h.identity, order });
    else if (h.identity > had.identity) had.identity = h.identity;
  });
  return [...seen.values()]
    .sort((a, b) => b.identity - a.identity || a.order - b.order)
    .map(({ name, identity }) => ({ name, identity }));
}

/** The better-name field's autocomplete list: every distinct real name in the item's hits. */
export function autocompleteCandidates(item: Item): string[] {
  return distinctNames(item).map((n) => n.name);
}

export type Chips = { names: NamedIdentity[]; genus: string | null };
export const MAX_NAME_CHIPS = 4;

/** One-click suggestions: up to four distinct hit names by best identity, never the name
 * under review (compared as the colouring compares), a placeholder or a bare genus; and the
 * genus of the item's top hit (the highest identity; a placeholder top hit gives way to the
 * best hit with a genus). An item with no hits has no chips. */
export function chips(item: Item): Chips {
  const reviewed = item.name_under_review;
  const names = distinctNames(item)
    .filter((n) => agreement(n.name, reviewed) !== "sp" && !isBareGenus(n.name))
    .slice(0, MAX_NAME_CHIPS);
  let top: Hit | null = null;
  for (const h of allHits(item)) {
    if (genusOf(h.name) === null) continue;
    if (!top || h.identity > top.identity) top = h;
  }
  return { names, genus: top ? genusOf(top.name) : null };
}

// --- keys -----------------------------------------------------------------------------------

export type KeyAction =
  | { type: "verdict"; value: Verdict }
  | { type: "shownBy"; value: ShownBy }
  | { type: "toggle"; field: "recognised" | "looked_up" }
  | { type: "next" }
  | { type: "prev" }
  | { type: "focusBetterName" }
  | { type: "escape" }
  | { type: "photoNext" }
  | { type: "photoPrev" };

export type KeyInput = { key: string; shiftKey?: boolean; ctrlKey?: boolean; metaKey?: boolean; altKey?: boolean };

const VERDICT_KEYS: Record<string, Verdict> = { "1": "supported", "2": "wrong", "3": "cannot_tell" };
const SHOWN_KEYS: Record<string, ShownBy> = { s: "sequences", p: "photographs", b: "both" };

/** What a key does. Nothing while typing in a text field, except Esc; nothing with Ctrl,
 * Cmd or Alt (the browser's own shortcuts); in the photo viewer only the arrows and Esc. */
export function keyAction(e: KeyInput, ctx: { typing: boolean; lightbox: boolean }): KeyAction | null {
  if (e.key === "Escape") return { type: "escape" };
  if (ctx.typing || e.ctrlKey || e.metaKey || e.altKey) return null;
  if (ctx.lightbox) {
    if (e.key === "ArrowRight") return { type: "photoNext" };
    if (e.key === "ArrowLeft") return { type: "photoPrev" };
    return null;
  }
  if (e.key in VERDICT_KEYS) return { type: "verdict", value: VERDICT_KEYS[e.key] };
  const k = e.key.toLowerCase();
  if (k in SHOWN_KEYS && e.key.length === 1) return { type: "shownBy", value: SHOWN_KEYS[k] };
  if (k === "r" && e.key.length === 1) return { type: "toggle", field: "recognised" };
  if (k === "l" && e.key.length === 1) return { type: "toggle", field: "looked_up" };
  if (e.key === "N" || (e.key === "n" && e.shiftKey) || e.key === "ArrowLeft") return { type: "prev" };
  if (e.key === "n" || e.key === "ArrowRight") return { type: "next" };
  if (e.key === "/") return { type: "focusBetterName" };
  return null;
}

export function applyKey(answer: Answer, action: KeyAction): Answer | null {
  switch (action.type) {
    case "verdict":
      return { ...answer, verdict: action.value };
    case "shownBy":
      return { ...answer, shown_by: action.value };
    case "toggle":
      return { ...answer, [action.field]: !answer[action.field] };
    default:
      return null;
  }
}

// --- moving between items -------------------------------------------------------------------

/** The next (dir 1) or previous (-1) item's index, skipping answered ones when asked; null
 * at either end. */
export function stepIndex(items: ListEntry[], current: number, dir: 1 | -1, unansweredOnly: boolean): number | null {
  for (let i = current + dir; i >= 0 && i < items.length; i += dir) {
    if (!unansweredOnly || !items[i].answered) return i;
  }
  return null;
}

/** "Go to": an item number (its position) or a code. */
export function goToIndex(query: string, items: ListEntry[]): number | null {
  const q = query.trim();
  if (!q) return null;
  if (/^\d+$/.test(q) && !/^[0-9]{5}$/.test(q)) {
    const i = items.findIndex((it) => it.position === Number(q));
    if (i >= 0) return i;
  }
  const code = q.toUpperCase();
  const byCode = items.findIndex((it) => it.code === code);
  if (byCode >= 0) return byCode;
  if (/^\d+$/.test(q)) {
    const i = items.findIndex((it) => it.position === Number(q));
    if (i >= 0) return i;
  }
  return null;
}

export function whenText(when: string): string {
  return when === "not recorded" ? "month not recorded" : when;
}

export function lengthText(length: number | null): string {
  return length === null ? "length not recorded" : `${length} bases`;
}

/** "Saved hh:mm" in the reviewer's own clock. */
export function savedText(iso: string | null | undefined): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "";
  return `Saved ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
}

// --- the instructions -----------------------------------------------------------------------

/** Open the instructions by themselves on a first visit: once the list has loaded, nobody
 * has answered any item yet, and they have not been opened by themselves already in this
 * page session (closed once, they stay closed across item changes). */
export function autoOpenInstructions(items: ListEntry[] | null, alreadyDecided: boolean): boolean {
  if (alreadyDecided || items === null) return false;
  return items.every((it) => !it.answered);
}

// --- the accession column ---------------------------------------------------------------------

export type AccessionCell = { kind: "link"; text: string; href: string } | { kind: "text"; text: string };
export const COMMUNITY_SEQUENCE = "community sequence";

/** A GenBank row's accession: a link to NCBI, or "community sequence" (null, schema v1.3)
 * with no link. */
export function accessionCell(hit: Hit): AccessionCell {
  if (!hit.accession) return { kind: "text", text: COMMUNITY_SEQUENCE };
  return { kind: "link", text: hit.accession, href: `https://www.ncbi.nlm.nih.gov/nuccore/${encodeURIComponent(hit.accession)}` };
}
