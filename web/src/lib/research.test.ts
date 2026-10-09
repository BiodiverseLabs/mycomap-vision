import { test } from "node:test";
import assert from "node:assert/strict";
import { BENCHMARKS, isResearchPath, MOVED, PUBLIC_NAV, researchNav, STATUS_LABEL } from "./research";
import type { Me } from "./api";

const me = (signin: Me["signin"], user: Me["user"]): Me => ({ signin, user });
const someone = { id: "7", name: "Ann" };

test("the public menu leads with the tool and how to help; the numbers sit under Research", () => {
  assert.deepEqual(PUBLIC_NAV.map((n) => n.label), ["Identify", "How it works", "Get involved"]);
  assert.ok(researchNav(me("identify", null)).some((n) => n.href === "/research/results"));
});

test("the paper shows in the Research menu only to signed-in members", () => {
  const paper = (m: Me | null | undefined) => researchNav(m).some((n) => n.href === "/research/paper");
  assert.equal(paper(me("identify", null)), false);
  assert.equal(paper(undefined), false);
  assert.equal(paper(me("all", someone)), true);
  assert.equal(paper(me("off", null)), true, "a local site without sign-in");
});

test("pages that moved under Research keep their old addresses", () => {
  assert.deepEqual(MOVED, { "/models": "/research/results", "/data": "/research/data",
                            "/paper": "/research/paper" });
  for (const to of Object.values(MOVED)) assert.ok(isResearchPath(to));
  assert.equal(isResearchPath("/researchers"), false);
});

test("every benchmark page is complete and addressable", () => {
  const slugs = BENCHMARKS.map((b) => b.slug);
  assert.equal(new Set(slugs).size, slugs.length, "unique slugs");
  for (const b of BENCHMARKS) {
    assert.match(b.slug, /^[a-z0-9-]+$/);
    for (const field of [b.name, b.summary, b.whyItMatters, b.records, b.scoring]) assert.ok(field.length > 10);
    assert.ok(STATUS_LABEL[b.status]);
    if (b.status !== "planned") assert.ok(b.results, `${b.slug} has results to point to`);
  }
});

test("the held-out benchmark says it is not the paper's test set", () => {
  const h = BENCHMARKS.find((b) => b.slug === "held-out")!;
  assert.match(h.records, /development benchmark/);
});
