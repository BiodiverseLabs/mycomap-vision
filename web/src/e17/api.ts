// The review page's calls: only its own /api/e17/<token>/ paths, same origin.

import type { Item, ListEntry, SavedResponse } from "./logic";
import type { Body } from "./saver";

/** The path token, from /review/<token>. */
export function tokenFromPath(pathname: string): string {
  const m = /^\/review\/([A-Za-z0-9_-]+)$/.exec(pathname);
  return m ? m[1] : "";
}

export type ItemWithResponse = Item & { response: SavedResponse | null };

export function client(token: string) {
  const base = `/api/e17/${token}`;
  const get = async <T,>(path: string): Promise<T> => {
    const r = await fetch(base + path, { credentials: "same-origin", cache: "no-store" });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return (await r.json()) as T;
  };
  return {
    items: () => get<{ count: number; items: ListEntry[] }>("/items"),
    item: (code: string) => get<ItemWithResponse>(`/item/${encodeURIComponent(code)}`),
    photoUrl: (code: string, n: number) => `${base}/photo/${encodeURIComponent(code)}/${n}`,
    respond: async (code: string, body: Body): Promise<SavedResponse> => {
      const r = await fetch(`${base}/response/${encodeURIComponent(code)}`, {
        method: "POST",
        credentials: "same-origin",
        cache: "no-store",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      return ((await r.json()) as { response: SavedResponse }).response;
    },
  };
}
