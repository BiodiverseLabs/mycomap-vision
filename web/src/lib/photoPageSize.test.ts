import { test } from "node:test";
import assert from "node:assert/strict";
import { PAGE_SIZE_STORAGE_KEY, PAGE_SIZES, readPageSize, SUMMARY_PAGE_SIZE, writePageSize } from "./photoSummary";

// Steve, 2026-10-21: the Summary's photos per page is a choice of 150, 300 or 500, and it stays
// for the viewer across page refreshes (kept in the browser).

const mem = () => {
  const m = new Map<string, string>();
  return { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v), m };
};

test("the choices are 150, 300 and 500, starting at 150", () => {
  assert.deepEqual([...PAGE_SIZES], [150, 300, 500]);
  assert.equal(SUMMARY_PAGE_SIZE, 150);
  assert.equal(readPageSize(() => mem()), 150);
});

test("a chosen page size is remembered across refreshes", () => {
  const s = mem();
  writePageSize(() => s, 500);
  assert.equal(s.m.get(PAGE_SIZE_STORAGE_KEY), "500");
  assert.equal(readPageSize(() => s), 500);                  // a fresh read, as after a refresh
  writePageSize(() => s, 300);
  assert.equal(readPageSize(() => s), 300);
});

test("anything else stored, or storage that throws, falls back to 150", () => {
  const s = mem();
  s.m.set(PAGE_SIZE_STORAGE_KEY, "10000");
  assert.equal(readPageSize(() => s), 150);
  const broken = { getItem: () => { throw new Error("blocked"); }, setItem: () => { throw new Error("blocked"); } };
  assert.equal(readPageSize(() => broken), 150);
  assert.doesNotThrow(() => writePageSize(() => broken, 500));
});
