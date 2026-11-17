import { test } from "node:test";
import assert from "node:assert/strict";

import {
  agreement, applyKey, autocompleteCandidates, chips, DEFAULT_SORT, EMPTY_ANSWER, goToIndex, isBareGenus,
  keyAction, MARKER, nextSort, savedText, sortHits, speciesKey, stepIndex, summarise, summaryLines,
  type Hit, type Item, type Sequence,
} from "./logic";

const h = (hit_no: number, name: string, identity: number, coverage = 100, accession?: string): Hit =>
  accession ? { hit_no, name, identity, coverage, accession } : { hit_no, name, identity, coverage };
const seq = (local: Hit[], genbank: Hit[], n = 1, withdrawn = false): Sequence => ({ n, withdrawn, length: 612, local, genbank });
const item = (name: string, sequences: Sequence[]): Item => ({
  schema: "e17-item/1", code: "K7Q2M", name_under_review: name, place: "Ontario", when: "September 2025",
  photos: [], sequences,
});

// --- agreement colouring -------------------------------------------------------------------

test("a hit of the same species is =sp, of the same genus =gen, anything else unmarked", () => {
  assert.equal(agreement("Fakeus exemplaris", "Fakeus exemplaris"), "sp");
  assert.equal(agreement("Fakeus amoenus", "Fakeus exemplaris"), "gen");
  assert.equal(agreement("Otherella quieta", "Fakeus exemplaris"), "other");
  assert.equal(MARKER.sp, "=sp");
  assert.equal(MARKER.gen, "=gen");
  assert.equal(MARKER.other, "");
});

test("names compare as text: case, spacing and curly quotes don't matter; extra words after the epithet do not split a species", () => {
  assert.equal(agreement("  fakeus   EXEMPLARIS ", "Fakeus exemplaris"), "sp");
  assert.equal(agreement("Fakeus exemplaris var. minor", "Fakeus exemplaris"), "sp");
  assert.equal(agreement("Fakeus sp. \u2018IN07\u2019", "Fakeus sp. 'IN07'"), "sp");
});

test("a provisional or cf. name agrees as a species only with the very same name", () => {
  assert.equal(agreement("Fakeus sp. 'IN07'", "Fakeus sp. 'IN07'"), "sp");
  assert.equal(agreement("Fakeus sp. 'NY11'", "Fakeus sp. 'IN07'"), "gen");
  assert.equal(agreement("Fakeus sp.", "Fakeus sp. 'IN07'"), "gen");
  assert.equal(agreement("Fakeus cf. exemplaris", "Fakeus exemplaris"), "gen");
  assert.equal(speciesKey("Fakeus"), null);
  assert.equal(agreement("Fakeus", "Fakeus exemplaris"), "gen");
});

test("an unnamed local sequence agrees with nothing", () => {
  assert.equal(agreement("unnamed local sequence", "Fakeus exemplaris"), "other");
  assert.equal(agreement("unnamed local sequence", "unnamed local sequence"), "other");
  assert.equal(agreement("(no name)", "Fakeus exemplaris"), "other");
});

// --- the per-sequence summary --------------------------------------------------------------

const s1 = seq(
  [h(1, "Fakeus sp. 'IN07'", 99.7), h(2, "Fakeus sp. 'IN07'", 99.5), h(3, "Fakeus exemplaris", 97.9, 99), h(4, "Fakeus exemplaris", 97.6, 99)],
  [h(1, "Fakeus exemplaris", 98.0, 99, "ZZ000001.1"), h(2, "Fakeus amoenus", 96.9, 98, "ZZ000002.1")],
);

test("the summary gives the best hit and the reviewed name's rank and identity in each table", () => {
  const s = summarise(s1, "Fakeus exemplaris");
  assert.deepEqual(s.best, { name: "Fakeus sp. 'IN07'", identity: 99.7, coverage: 100, table: "local" });
  assert.deepEqual(s.local, { rank: 3, identity: 97.9 });
  assert.deepEqual(s.genbank, { rank: 1, identity: 98.0 });
  const lines = summaryLines(s);
  assert.equal(lines.best, "Fakeus sp. 'IN07' · 99.7% identity · 100% coverage (local)");
  assert.equal(lines.reviewed, "Local rank 3 · 97.9%  |  GenBank rank 1 · 98.0%");
});

test("a table without the reviewed name says so, and an empty table says it has no hits", () => {
  const s = summarise(seq([h(1, "Otherella quieta", 90)], []), "Fakeus exemplaris");
  assert.equal(summaryLines(s).reviewed, "Local: no hit with this name  |  GenBank: no hits");
  assert.equal(summaryLines(summarise(seq([], []), "Fakeus exemplaris")).best, "no hits");
});

test("the best hit is the higher identity of the two tables' first rows; ties go to coverage, then local", () => {
  const gb = summarise(seq([h(1, "A b", 97)], [h(1, "C d", 98, 90, "ZZ000001.1")]), "X y");
  assert.equal(gb.best?.table, "genbank");
  const cov = summarise(seq([h(1, "A b", 98, 90)], [h(1, "C d", 98, 95, "ZZ000001.1")]), "X y");
  assert.equal(cov.best?.name, "C d");
  const tie = summarise(seq([h(1, "A b", 98, 95)], [h(1, "C d", 98, 95, "ZZ000001.1")]), "X y");
  assert.equal(tie.best?.table, "local");
  // Rank is the stored order: a later row with a higher identity is not "best".
  const order = summarise(seq([h(1, "A b", 95), h(2, "E f", 99)], []), "X y");
  assert.equal(order.best?.name, "A b");
});

// --- sorting --------------------------------------------------------------------------------

const hits = [h(1, "Fakeus b", 97, 99), h(2, "fakeus a", 99, 98), h(3, "Fakeus c", 97, 100)];

test("tables sort by identity, coverage or name, ties kept in rank order, and back to rank", () => {
  assert.deepEqual(sortHits(hits, DEFAULT_SORT).map((x) => x.rank), [1, 2, 3]);
  assert.deepEqual(sortHits(hits, { key: "identity", dir: "desc" }).map((x) => x.rank), [2, 1, 3]);
  assert.deepEqual(sortHits(hits, { key: "identity", dir: "asc" }).map((x) => x.rank), [1, 3, 2]);
  assert.deepEqual(sortHits(hits, { key: "coverage", dir: "desc" }).map((x) => x.rank), [3, 1, 2]);
  assert.deepEqual(sortHits(hits, { key: "name", dir: "asc" }).map((x) => x.rank), [2, 1, 3]);
  assert.deepEqual(sortHits(hits, { key: "name", dir: "desc" }).map((x) => x.rank), [3, 1, 2]);
  assert.deepEqual(hits.map((x) => x.hit_no), [1, 2, 3], "the item's own order is untouched");
});

test("a heading clicked twice flips; a new numeric column starts highest first, names A to Z", () => {
  assert.deepEqual(nextSort(DEFAULT_SORT, "identity"), { key: "identity", dir: "desc" });
  assert.deepEqual(nextSort({ key: "identity", dir: "desc" }, "identity"), { key: "identity", dir: "asc" });
  assert.deepEqual(nextSort({ key: "identity", dir: "desc" }, "name"), { key: "name", dir: "asc" });
  assert.deepEqual(nextSort({ key: "name", dir: "asc" }, "coverage"), { key: "coverage", dir: "desc" });
});

// --- autocomplete and chips ------------------------------------------------------------------

test("autocomplete offers each distinct hit name once, best identity first, never a placeholder", () => {
  const it = item("Fakeus exemplaris", [
    seq([h(1, "Fakeus sp. 'IN07'", 99.7), h(2, "unnamed local sequence", 99.9), h(3, "fakeus  sp. 'IN07'", 99.1)],
      [h(1, "Fakeus exemplaris", 98.0, 99, "ZZ000001.1"), h(2, "(no name)", 99, 99, "ZZ000002.1")]),
  ]);
  assert.deepEqual(autocompleteCandidates(it), ["Fakeus sp. 'IN07'", "Fakeus exemplaris"]);
});

const full = item("Fakeus exemplaris", [
  seq(
    [h(1, "Fakeus sp. 'IN07'", 99.7), h(2, "Fakeus exemplaris", 97.9), h(3, "Fakeus sp. 'NY11'", 96.8), h(4, "unnamed local sequence", 95.2)],
    [h(1, "Fakeus exemplaris", 98.0, 99, "ZZ000001.1"), h(2, "Fakeus amoenus", 96.9, 98, "ZZ000002.1"), h(3, "Fakeus sp.", 96.4, 97, "ZZ000003.1"), h(4, "Otherella quieta", 86.1, 91, "ZZ000004.1")],
  ),
  seq([h(1, "Fakeus sp. 'IN07'", 98.8)], [h(1, "Mockia fictiva", 90.0, 90, "ZZ000005.1"), h(2, "Fakeus minor", 85, 80, "ZZ000006.1")], 2, true),
]);

test("chips are the distinct hit names by best identity, highest first, at most four, each with that identity", () => {
  const c = chips(full);
  assert.deepEqual(c.names, [
    { name: "Fakeus sp. 'IN07'", identity: 99.7 },
    { name: "Fakeus amoenus", identity: 96.9 },
    { name: "Fakeus sp. 'NY11'", identity: 96.8 },
    { name: "Mockia fictiva", identity: 90.0 },
  ]);
  assert.ok(!c.names.some((n) => n.name === "Otherella quieta"), "the fifth is cut by the cap");
});

test("chip ties keep the order the names first appear in the item", () => {
  const it = item("X y", [seq([h(1, "Beta one", 97), h(2, "Alpha two", 97)], [h(1, "Gamma three", 97, 99, "ZZ000001.1")])]);
  assert.deepEqual(chips(it).names.map((n) => n.name), ["Beta one", "Alpha two", "Gamma three"]);
});

test("the name under review is never a chip, compared the way the colouring compares", () => {
  const it = item("Fakeus exemplaris", [seq([h(1, "fakeus  EXEMPLARIS", 99.9), h(2, "Fakeus exemplaris var. minor", 99.5), h(3, "Fakeus amoenus", 97)], [])]);
  assert.deepEqual(chips(it).names.map((n) => n.name), ["Fakeus amoenus"]);
});

test("the genus chip is the genus of the item's single highest-identity hit", () => {
  assert.equal(chips(full).genus, "Fakeus");
  const it = item("Fakeus exemplaris", [seq([h(1, "Fakeus amoenus", 97)], []), seq([], [h(1, "Mockia fictiva", 99.2, 90, "ZZ000001.1")], 2)]);
  assert.equal(chips(it).genus, "Mockia");
});

test("an unnamed top hit gives the genus chip to the best hit that has a genus", () => {
  const it = item("X y", [seq([h(1, "unnamed local sequence", 99.9), h(2, "Fakeus amoenus", 97)], [])]);
  assert.equal(chips(it).genus, "Fakeus");
});

test("an item with no hits has no chips", () => {
  assert.deepEqual(chips(item("Fakeus exemplaris", [])), { names: [], genus: null });
  assert.deepEqual(chips(item("Fakeus exemplaris", [seq([], [])])), { names: [], genus: null });
});

test("placeholders are never chips: unnamed local sequence, (no name), (no current name), in any case", () => {
  const it = item("Fakeus exemplaris", [seq(
    [h(1, "unnamed local sequence", 99.9), h(2, "Unnamed Local Sequence", 99.8), h(3, "(no name)", 99.7), h(4, "(No Current Name)", 99.6), h(5, "Fakeus amoenus", 90)],
    [],
  )]);
  const c = chips(it);
  assert.deepEqual(c.names.map((n) => n.name), ["Fakeus amoenus"]);
  assert.equal(c.genus, "Fakeus");
  const only = item("Fakeus exemplaris", [seq([h(1, "unnamed local sequence", 99.9), h(2, "(no name)", 98)], [])]);
  assert.deepEqual(chips(only), { names: [], genus: null });
});

test("a bare genus (alone, or with sp., sp or spp.) is never a name chip; a provisional sp. 'IN01' is", () => {
  for (const bare of ["Russula", "Russula sp.", "Russula sp", "Russula spp.", "russula  SP."]) {
    assert.equal(isBareGenus(bare), true, bare);
  }
  assert.equal(isBareGenus("Russula sp. 'IN01'"), false);
  assert.equal(isBareGenus("unnamed local sequence"), false);
  const it = item("Fakeus exemplaris", [seq([h(1, "Russula sp.", 99.9), h(2, "Russula sp", 99.8), h(3, "Russula spp.", 99.7), h(4, "Russula", 99.6), h(5, "Russula sp. 'IN01'", 99)], [])]);
  const c = chips(it);
  assert.deepEqual(c.names, [{ name: "Russula sp. 'IN01'", identity: 99 }]);
  assert.equal(c.genus, "Russula", "the genus chip covers the bare names");
});

// --- keys ------------------------------------------------------------------------------------

const free = { typing: false, lightbox: false };

test("the keys map to their actions", () => {
  assert.deepEqual(keyAction({ key: "1" }, free), { type: "verdict", value: "supported" });
  assert.deepEqual(keyAction({ key: "2" }, free), { type: "verdict", value: "wrong" });
  assert.deepEqual(keyAction({ key: "3" }, free), { type: "verdict", value: "cannot_tell" });
  assert.deepEqual(keyAction({ key: "s" }, free), { type: "shownBy", value: "sequences" });
  assert.deepEqual(keyAction({ key: "P", shiftKey: true }, free), { type: "shownBy", value: "photographs" });
  assert.deepEqual(keyAction({ key: "b" }, free), { type: "shownBy", value: "both" });
  assert.deepEqual(keyAction({ key: "r" }, free), { type: "toggle", field: "recognised" });
  assert.deepEqual(keyAction({ key: "l" }, free), { type: "toggle", field: "looked_up" });
  assert.deepEqual(keyAction({ key: "n" }, free), { type: "next" });
  assert.deepEqual(keyAction({ key: "ArrowRight" }, free), { type: "next" });
  assert.deepEqual(keyAction({ key: "ArrowLeft" }, free), { type: "prev" });
  assert.deepEqual(keyAction({ key: "N", shiftKey: true }, free), { type: "prev" });
  assert.deepEqual(keyAction({ key: "/" }, free), { type: "focusBetterName" });
  assert.deepEqual(keyAction({ key: "Escape" }, free), { type: "escape" });
  assert.equal(keyAction({ key: "x" }, free), null);
  assert.equal(keyAction({ key: "4" }, free), null);
});

test("while typing in a text field only Esc acts", () => {
  const typing = { typing: true, lightbox: false };
  for (const key of ["1", "2", "3", "s", "p", "b", "r", "l", "n", "N", "/", "ArrowRight", "ArrowLeft"]) {
    assert.equal(keyAction({ key }, typing), null, key);
  }
  assert.deepEqual(keyAction({ key: "Escape" }, typing), { type: "escape" });
});

test("the browser's own shortcuts are left alone", () => {
  assert.equal(keyAction({ key: "r", ctrlKey: true }, free), null);
  assert.equal(keyAction({ key: "l", metaKey: true }, free), null);
  assert.equal(keyAction({ key: "ArrowLeft", altKey: true }, free), null);
});

test("in the photo viewer the arrows move between photos, and nothing else answers", () => {
  const box = { typing: false, lightbox: true };
  assert.deepEqual(keyAction({ key: "ArrowRight" }, box), { type: "photoNext" });
  assert.deepEqual(keyAction({ key: "ArrowLeft" }, box), { type: "photoPrev" });
  assert.deepEqual(keyAction({ key: "Escape" }, box), { type: "escape" });
  assert.equal(keyAction({ key: "1" }, box), null);
  assert.equal(keyAction({ key: "n" }, box), null);
});

test("answer keys set the verdict and shown-by, and toggle the two checks", () => {
  let a = applyKey(EMPTY_ANSWER, { type: "verdict", value: "wrong" })!;
  a = applyKey(a, { type: "shownBy", value: "both" })!;
  a = applyKey(a, { type: "toggle", field: "recognised" })!;
  assert.deepEqual(a, { ...EMPTY_ANSWER, verdict: "wrong", shown_by: "both", recognised: true });
  assert.equal(applyKey(a, { type: "toggle", field: "recognised" })!.recognised, false);
  assert.equal(applyKey(a, { type: "next" }), null);
});

// --- moving between items ----------------------------------------------------------------------

const list = [
  { code: "00042", position: 1, answered: true },
  { code: "PLM4T", position: 2, answered: false },
  { code: "K7Q2M", position: 3, answered: true },
  { code: "W2N6D", position: 4, answered: false },
];

test("next and previous step one item, or to the next unanswered one when asked, and stop at the ends", () => {
  assert.equal(stepIndex(list, 0, 1, false), 1);
  assert.equal(stepIndex(list, 1, 1, true), 3);
  assert.equal(stepIndex(list, 3, -1, true), 1);
  assert.equal(stepIndex(list, 1, -1, true), null);
  assert.equal(stepIndex(list, 3, 1, false), null);
});

test("go to takes an item number or a code", () => {
  assert.equal(goToIndex("3", list), 2);
  assert.equal(goToIndex("k7q2m", list), 2);
  assert.equal(goToIndex("00042", list), 0);
  assert.equal(goToIndex("9", list), null);
  assert.equal(goToIndex("", list), null);
});

test("saved time is hh:mm on the reviewer's clock, empty when unknown", () => {
  assert.match(savedText("2026-11-16T21:14:00.000Z"), /^Saved \d\d:\d\d$/);
  assert.equal(savedText(null), "");
  assert.equal(savedText("nonsense"), "");
});
