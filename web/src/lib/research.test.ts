import { test } from "node:test";
import assert from "node:assert/strict";
import { API_PATH, BENCHMARKS, isResearchPath, MOVED, mustSignIn, PUBLIC_NAV, researchNav,
         STATUS_LABEL } from "./research";
import type { Me } from "./api";

const me = (signin: Me["signin"], user: Me["user"]): Me => ({ signin, user });
const someone = { id: "7", name: "Ann" };

test("the public menu leads with the tool and how to help; the numbers sit under Research", () => {
  assert.deepEqual(PUBLIC_NAV.map((n) => n.label), ["Identify", "How it works", "Get involved"]);
  assert.ok(researchNav(me("identify", null)).some((n) => n.href === "/research/results"));
});

test("the paper draft is off the Research menu for everyone, members included", () => {
  for (const m of [undefined, me("identify", null), me("all", someone), me("off", null)]) {
    assert.ok(!researchNav(m).some((n) => /paper/i.test(n.href + n.label)));
  }
});

test("experiments show only to signed-in members", () => {
  const has = (m: Me | null | undefined) => researchNav(m).some((n) => n.href === "/research/experiments");
  assert.equal(has(me("identify", null)), false);
  assert.equal(has(undefined), false);
  assert.equal(has(me("all", someone)), true);
  assert.equal(has(me("off", null)), true, "a local site without sign-in");
});

test("pages that moved under Research keep their old addresses", () => {
  assert.deepEqual(MOVED, { "/models": "/research/results", "/data": "/research/data",
                            "/paper": "/research", "/research/paper": "/research",
                            "/api-docs": "/research/api" });
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

test("API is the last item under Research, for visitors and members alike", () => {
  for (const m of [undefined, me("identify", null), me("all", null), me("all", someone)]) {
    const last = researchNav(m).at(-1)!;
    assert.deepEqual([last.href, last.label], ["/research/api", "API"]);
  }
  assert.equal(API_PATH, "/research/api");
});

test("a signed-out visitor reads the API page but signs in for any other page", () => {
  const visitor = me("all", null);
  assert.equal(mustSignIn(visitor, "/research/api"), false);
  assert.equal(mustSignIn(visitor, "/api-docs"), false);
  for (const page of ["/", "/research", "/research/results", "/research/paper", "/about"]) {
    assert.equal(mustSignIn(visitor, page), true, page);
  }
});

test("nobody else is sent to sign in by the site itself", () => {
  assert.equal(mustSignIn(me("all", someone), "/research/results"), false);
  assert.equal(mustSignIn(me("identify", null), "/about"), false);
  assert.equal(mustSignIn(me("off", null), "/"), false);
  assert.equal(mustSignIn(undefined, "/"), false, "not known yet: wait for /api/me");
  // The sign-in pages themselves never send people to sign in: that would loop.
  assert.equal(mustSignIn(me("all", null), "/auth/signin"), false);
  assert.equal(mustSignIn(me("all", null), "/auth/signed-out"), false);
});
