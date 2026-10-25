// The API outline served at /api/docs/v1.json (src/mycomap_vision/apidocs.py): the one
// source for this site's API page (Research > API) and the tab on mycomap.org/api-docs.

export type Block =
  | { type: "p"; text: string }
  | { type: "list"; items: string[] }
  | { type: "table"; columns: string[]; rows: string[][] }
  | { type: "code"; lang?: string; text: string }
  | {
      type: "endpoint";
      method: string;
      path: string;
      status: "available" | "planned";
      summary: string;
      auth?: string;
      /** [name, type, required, notes] */
      fields?: string[][];
      request?: string;
      response?: string;
      notes?: string[];
    };

export interface Section {
  id: string;
  title: string;
  blocks: Block[];
}

export interface Outline {
  title: string;
  version: string;
  base_url: string;
  page_url: string;
  summary: string;
  sections: Section[];
}

export type Inline =
  | { kind: "text"; text: string }
  | { kind: "code"; text: string }
  | { kind: "bold"; text: string }
  | { kind: "link"; text: string; href: string };

const TOKEN = /`([^`]+)`|\*\*([^*]+)\*\*|\[([^\]]+)\]\((https:\/\/[^)\s]+)\)/g;

/** The outline's inline marks: `code`, **bold** and [label](https://…) links. Anything
 *  else, including a link that isn't https, stays plain text. */
export function inlineParts(text: string): Inline[] {
  const out: Inline[] = [];
  let last = 0;
  for (const m of text.matchAll(TOKEN)) {
    if (m.index! > last) out.push({ kind: "text", text: text.slice(last, m.index) });
    if (m[1] !== undefined) out.push({ kind: "code", text: m[1] });
    else if (m[2] !== undefined) out.push({ kind: "bold", text: m[2] });
    else out.push({ kind: "link", text: m[3], href: m[4] });
    last = m.index! + m[0].length;
  }
  if (last < text.length) out.push({ kind: "text", text: text.slice(last) });
  return out;
}

export async function fetchOutline(url = "/api/docs/v1.json"): Promise<Outline> {
  const res = await fetch(url, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`The API outline didn't load (${res.status}).`);
  return res.json();
}

// --- the docs layout (Research > API, the approved redesign) --------------------------------

export type EndpointBlock = Extract<Block, { type: "endpoint" }>;

/** Every endpoint in the outline, in its order, with the section it is in. */
export function endpointIndex(o: Outline): (EndpointBlock & { section: string })[] {
  return o.sections.flatMap((s) => s.blocks.filter((b): b is EndpointBlock => b.type === "endpoint")
    .map((b) => ({ ...b, section: s.id })));
}

/** How many endpoints answer now, of all listed. */
export function endpointCounts(o: Outline): { available: number; total: number } {
  const all = endpointIndex(o);
  return { available: all.filter((b) => b.status === "available").length, total: all.length };
}

/** A table of field names and what they mean ("Field", ...): shown as a field list. */
export const isFieldTable = (b: Block) => b.type === "table" && b.columns[0] === "Field" && b.columns.length === 2;
/** The errors table ("Status", "code", "When"): each status as a coloured badge. */
export const isStatusTable = (b: Block) => b.type === "table" && b.columns[0] === "Status";
/** A status badge's class: 5xx red, everything else amber. */
export const statusClass = (status: string) => (/^5/.test(status.trim()) ? "e5" : "e4");

export type DocPart =
  | { kind: "section"; id: string; title: string; blocks: Block[] }
  | { kind: "cards"; id: string; title: string; cards: { title: string; items: string[] }[] };

/** The page's parts in order: each section as it is, except a run of sections that hold only
 *  a list (limits, photos, rules), which become one part of side-by-side cards. */
export function docParts(o: Outline): DocPart[] {
  const out: DocPart[] = [];
  for (const s of o.sections) {
    const onlyList = s.blocks.length > 0 && s.blocks.every((b) => b.type === "list");
    const last = out[out.length - 1];
    if (onlyList) {
      const card = { title: s.title, items: s.blocks.flatMap((b) => (b.type === "list" ? b.items : [])) };
      if (last?.kind === "cards") { last.cards.push(card); last.title = `${last.title}, ${s.title.toLowerCase()}`; continue; }
      out.push({ kind: "cards", id: s.id, title: s.title, cards: [card] });
      continue;
    }
    out.push({ kind: "section", id: s.id, title: s.title, blocks: s.blocks });
  }
  return out;
}
