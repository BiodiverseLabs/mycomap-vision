// The API outline served at /api/docs/v1.json (src/mycomap_vision/apidocs.py): the one
// source for this site's /api-docs page and the tab on mycomap.org/api-docs.

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
