/**
 * Place suggestions as you type in the map's search box (Steve, 2026-10-20: "autocomplete
 * without the need to search"). Vision's /api/geocode/suggest asks Photon (geocode.py),
 * never Nominatim, whose policy forbids autocomplete; the Search button still asks
 * Nominatim. These are the rules; the box and its list are components/LocationPicker.tsx.
 */

export interface Place {
  display_name: string;
  lat: number;
  lon: number;
}

/** Why a list of suggestions is what it is (the server's X-Suggest-Status). */
export type SuggestStatus = "ok" | "short" | "long" | "limited" | "unavailable";

export interface Suggestions {
  query: string;
  places: Place[];
  status: SuggestStatus;
}

/** Ask once typing has paused this long. */
export const SUGGEST_DELAY_MS = 350;
/** Fewer letters than this (spaces don't count) aren't sent. */
export const MIN_LETTERS = 3;

export const letters = (q: string) => q.replace(/\s+/g, "").length;

/** The server's suggestions for `q`; anything that goes wrong is an empty "unavailable". */
export async function fetchSuggestions(q: string, get: typeof fetch = fetch): Promise<Suggestions> {
  const none = (status: SuggestStatus): Suggestions => ({ query: q, places: [], status });
  try {
    const res = await get(`/api/geocode/suggest?q=${encodeURIComponent(q)}`);
    if (!res.ok) return none("unavailable");
    const body = await res.json();
    const status = (res.headers.get("X-Suggest-Status") ?? "ok") as SuggestStatus;
    return { query: q, places: Array.isArray(body) ? (body as Place[]) : [], status };
  } catch {
    return none("unavailable");
  }
}

interface Timers {
  setTimeout: (fn: () => void, ms: number) => unknown;
  clearTimeout: (id: unknown) => void;
}

/**
 * Asks only once typing pauses for `delay` ms, and keeps only the answer for the words now
 * in the box (an answer for words since changed is dropped). Too few letters clear the list
 * at once: `onAnswer(null)`.
 */
export function createSuggester({ ask, onAnswer, delay = SUGGEST_DELAY_MS, timers }: {
  ask: (q: string) => Promise<Suggestions>;
  onAnswer: (s: Suggestions | null) => void;
  delay?: number;
  timers?: Timers;
}) {
  const t: Timers = timers ?? {
    setTimeout: (fn, ms) => globalThis.setTimeout(fn, ms),
    clearTimeout: (id) => globalThis.clearTimeout(id as ReturnType<typeof setTimeout>),
  };
  let timer: unknown = null;
  let turn = 0;
  const stop = () => {
    if (timer !== null) t.clearTimeout(timer);
    timer = null;
    turn++;
  };
  return {
    type(raw: string) {
      stop();
      const q = raw.trim();
      if (letters(q) < MIN_LETTERS) return onAnswer(null);
      const mine = turn;
      timer = t.setTimeout(async () => {
        timer = null;
        const answer = await ask(q);
        if (mine === turn) onAnswer(answer);
      }, delay);
    },
    /** Nothing pending is asked or shown (a pick, a Search, the map closing). */
    cancel: stop,
  };
}

/**
 * A key pressed in the search box while the list shows `count` places and `active` is the
 * highlighted one (-1 for none). `handled` false leaves the key alone: Enter with nothing
 * highlighted submits the form (the full Search), Escape with no list closes the map.
 */
export function listKey(key: string, count: number, active: number):
  { handled: boolean; active: number; pick?: number; close?: boolean } {
  if (count === 0) return { handled: false, active: -1 };
  switch (key) {
    case "ArrowDown":
      return { handled: true, active: active + 1 >= count ? 0 : active + 1 };
    case "ArrowUp":
      return { handled: true, active: active <= 0 ? count - 1 : active - 1 };
    case "Enter":
      return active >= 0 && active < count ? { handled: true, active: -1, pick: active } : { handled: false, active };
    case "Escape":
      return { handled: true, active: -1, close: true };
    default:
      return { handled: false, active };
  }
}

/** What to say under the box about the suggestions; null when the list says it all. */
export function suggestNote(s: Suggestions | null): string | null {
  if (!s) return null;
  if (s.status === "ok" || s.status === "short") {
    return s.status === "ok" && s.places.length === 0
      ? "No suggestions for that. Press Search to look further."
      : null;
  }
  return "Suggestions are paused. Press Search to find the place.";
}
