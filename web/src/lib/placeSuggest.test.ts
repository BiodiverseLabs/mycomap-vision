import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  createSuggester, fetchSuggestions, listKey, MIN_LETTERS, SUGGEST_DELAY_MS, suggestNote, type Suggestions,
} from "./placeSuggest";

/** Timers the test moves by hand. */
function fakeTimers() {
  let now = 0, next = 1;
  const due = new Map<number, { at: number; fn: () => void }>();
  return {
    setTimeout: (fn: () => void, ms: number) => { const id = next++; due.set(id, { at: now + ms, fn }); return id; },
    clearTimeout: (id: unknown) => { due.delete(id as number); },
    advance(ms: number) {
      now += ms;
      for (const [id, t] of [...due].sort((a, b) => a[1].at - b[1].at)) {
        if (t.at <= now) { due.delete(id); t.fn(); }
      }
    },
    pending: () => due.size,
  };
}

const PERRY = { display_name: "Perry County, Ohio, United States", lat: 39.74, lon: -82.24 };
const flush = () => new Promise((r) => setImmediate(r));

function rig(answerFor: (q: string) => Promise<Suggestions> = async (q) => ({ query: q, places: [PERRY], status: "ok" })) {
  const timers = fakeTimers();
  const asked: string[] = [];
  const answers: (Suggestions | null)[] = [];
  const s = createSuggester({
    ask: (q) => { asked.push(q); return answerFor(q); },
    onAnswer: (a) => answers.push(a),
    timers,
  });
  return { s, timers, asked, answers };
}

test("suggestions wait for a pause in typing, and ask once for the words then in the box", async () => {
  assert.ok(SUGGEST_DELAY_MS >= 300 && SUGGEST_DELAY_MS <= 400);
  const { s, timers, asked, answers } = rig();
  for (const q of ["Per", "Perr", "Perry", "Perry C", "Perry Co"]) {
    s.type(q);
    timers.advance(100);                       // typing on: nothing asked yet
  }
  assert.deepEqual(asked, []);
  timers.advance(SUGGEST_DELAY_MS);
  await flush();
  assert.deepEqual(asked, ["Perry Co"]);
  assert.equal(answers.length, 1);
  assert.deepEqual(answers[0]?.places, [PERRY]);
});

test("too few letters are never sent and clear the list at once", () => {
  assert.equal(MIN_LETTERS, 3);
  const { s, timers, asked, answers } = rig();
  s.type("Pe");
  s.type(" P e ");
  timers.advance(SUGGEST_DELAY_MS * 2);
  assert.deepEqual(asked, []);
  assert.deepEqual(answers, [null, null]);
  assert.equal(timers.pending(), 0);
});

test("an answer for words since changed is dropped", async () => {
  let release!: () => void;
  const slow = new Promise<void>((r) => { release = r; });
  const { s, timers, answers } = rig(async (q) => {
    if (q === "Perry") await slow;
    return { query: q, places: [{ ...PERRY, display_name: q }], status: "ok" };
  });
  s.type("Perry");
  timers.advance(SUGGEST_DELAY_MS);            // asked for "Perry", still answering…
  s.type("Hocking");
  timers.advance(SUGGEST_DELAY_MS);
  await flush();
  release();
  await flush();
  assert.deepEqual(answers.map((a) => a?.query), ["Hocking"]);
});

test("a pick or a Search cancels what is pending", async () => {
  const { s, timers, asked, answers } = rig();
  s.type("Perry County");
  s.cancel();
  timers.advance(SUGGEST_DELAY_MS * 2);
  await flush();
  assert.deepEqual(asked, []);
  assert.deepEqual(answers, []);
});

test("arrow keys move through the list, Enter picks, Escape closes it", () => {
  assert.deepEqual(listKey("ArrowDown", 3, -1), { handled: true, active: 0 });
  assert.deepEqual(listKey("ArrowDown", 3, 2), { handled: true, active: 0 }, "wraps to the top");
  assert.deepEqual(listKey("ArrowUp", 3, -1), { handled: true, active: 2 });
  assert.deepEqual(listKey("ArrowUp", 3, 1), { handled: true, active: 0 });
  assert.deepEqual(listKey("Enter", 3, 1), { handled: true, active: -1, pick: 1 });
  assert.deepEqual(listKey("Escape", 3, 1), { handled: true, active: -1, close: true });
  assert.deepEqual(listKey("a", 3, 1), { handled: false, active: 1 });
});

test("Enter with nothing highlighted, or no list, is left to the full Search", () => {
  assert.equal(listKey("Enter", 3, -1).handled, false);
  assert.equal(listKey("Enter", 0, -1).handled, false);
  assert.equal(listKey("ArrowDown", 0, -1).handled, false);
  assert.equal(listKey("Escape", 0, -1).handled, false, "Escape with no list closes the map");
});

test("when suggestions can't be had, the page says to press Search", () => {
  const none = (status: Suggestions["status"]): Suggestions => ({ query: "Perry", places: [], status });
  for (const st of ["limited", "unavailable", "long"] as const) {
    assert.match(suggestNote(none(st)) ?? "", /Press Search/);
  }
  assert.match(suggestNote(none("ok")) ?? "", /Press Search/, "nothing suggested");
  assert.equal(suggestNote({ query: "Perry", places: [PERRY], status: "ok" }), null);
  assert.equal(suggestNote(none("short")), null);
  assert.equal(suggestNote(null), null);
});

test("the server's suggestions are read with their status; any failure is an empty 'unavailable'", async () => {
  const urls: string[] = [];
  const ok = (async (u: string) => {
    urls.push(u);
    return new Response(JSON.stringify([PERRY]), { headers: { "X-Suggest-Status": "ok" } });
  }) as typeof fetch;
  assert.deepEqual(await fetchSuggestions("Perry County, OH", ok), { query: "Perry County, OH", places: [PERRY], status: "ok" });
  assert.equal(urls[0], "/api/geocode/suggest?q=Perry%20County%2C%20OH");
  const limited = (async () => new Response("[]", { headers: { "X-Suggest-Status": "limited" } })) as typeof fetch;
  assert.equal((await fetchSuggestions("Perry", limited)).status, "limited");
  const signedOut = (async () => new Response('{"detail":"sign in"}', { status: 401 })) as typeof fetch;
  assert.deepEqual(await fetchSuggestions("Perry", signedOut), { query: "Perry", places: [], status: "unavailable" });
  const offline = (async () => { throw new TypeError("offline"); }) as typeof fetch;
  assert.deepEqual(await fetchSuggestions("Perry", offline), { query: "Perry", places: [], status: "unavailable" });
});

test("the map's box suggests as you type and keeps Search for the full search", () => {
  const src = readFileSync(new URL("../components/LocationPicker.tsx", import.meta.url), "utf-8");
  assert.match(src, /onChange=\{\(e\) => typeQuery\(e\.target\.value\)\}/, "typing asks for suggestions");
  assert.match(src, /onKeyDown=\{onSearchKey\}/, "keys move through the list");
  assert.match(src, /fetch\(`\/api\/geocode\/search\?q=/, "Search still asks the full place search");
  assert.match(src, /<Button type="submit"/, "the Search button stays");
  assert.doesNotMatch(src, /nominatim|photon/i, "the page talks only to Vision");
  // A suggestion is picked exactly as a search result: one pick() for both lists.
  assert.match(src, /onClick=\{\(\) => pick\(r\)\}/);
  assert.match(src, /placePin\(\{ lat: r\.lat, lng: r\.lon \}, true\)/);
});
