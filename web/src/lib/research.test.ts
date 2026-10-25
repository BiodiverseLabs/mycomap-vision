import { test } from "node:test";
import assert from "node:assert/strict";
import { API_PATH, BASE_MODEL_SCREEN, BENCHMARKS, CITATIONS, DATA_LABEL, E33, fungiCards, FUNGI_CARDS,
         isBarTabActive, isResearchPath, MEASURE_LABEL, MOVED, mustSignIn, PUBLIC_NAV, researchBar, researchNav,
         screenLead, SHOW_PENDING_FUNGI_CARDS, STATUS_LABEL, STATUS_ORDER } from "./research";
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
    for (const field of [b.name, b.summary, b.whyItMatters, b.records, b.scoring, b.question]) assert.ok(field.length > 10, b.slug);
    assert.ok(STATUS_LABEL[b.status]);
    if (b.status !== "planned" && b.status !== "sealed") assert.ok(b.results, `${b.slug} has results to point to`);
    assert.ok(b.protocol.length >= 3 && b.protocol.length <= 6, `${b.slug} protocol steps`);
    assert.ok(b.links.length > 0, `${b.slug} links`);
  }
});

test("each benchmark carries the facets the filter reads: measures, data, and a size or a headline", () => {
  for (const b of BENCHMARKS) {
    assert.ok(b.measures.length > 0 && b.measures.every((m) => MEASURE_LABEL[m]), b.slug);
    assert.ok(b.data.length > 0 && b.data.every((d) => DATA_LABEL[d]), b.slug);
    assert.ok(b.size || b.headline, `${b.slug}: a size, or a headline the size is read from`);
    if (!b.headline) assert.ok(b.pending, `${b.slug}: says something while it has no number`);
    if (b.lastRun === "run") assert.equal(b.headline?.run, "served", `${b.slug}: its date comes from its run`);
  }
});

test("no benchmark text hard-codes a number the API serves", () => {
  // The served run's figures (species top 1 and the like) are read live; percentages in the
  // copy would go stale with the next run (95%, the interval level, is a setting, not a result).
  for (const b of BENCHMARKS) {
    for (const t of [b.summary, b.whyItMatters, b.records, b.scoring, b.question]) {
      assert.doesNotMatch(t.replace(/\b95%/g, ""), /\d+(\.\d+)?%/, `${b.slug}: ${t}`);
    }
  }
});

test("the measures filter has the mockup's values, Dangerous mistakes among them", () => {
  assert.deepEqual(Object.values(MEASURE_LABEL), ["Accuracy", "Rare species", "Unknown species", "Generalisation",
    "Leakage", "Outside models", "Live / prospective", "Dangerous mistakes"]);
  assert.deepEqual(Object.values(DATA_LABEL), ["v1 validation", "Sealed test", "Live", "Pre-v1 held-out"]);
  assert.ok(BENCHMARKS.find((b) => b.slug === "poisonous-mistakes")!.measures.includes("danger"));
});

test("the held-out benchmark says it is not the paper's test set", () => {
  const h = BENCHMARKS.find((b) => b.slug === "held-out")!;
  assert.match(h.records, /development benchmark/);
  assert.equal(h.status, "history");
  assert.deepEqual(h.data, ["prev1"]);
});

test("v1-test is Sealed (Steve, 2026-10-25), last but planned and history in the status order", () => {
  assert.equal(STATUS_LABEL.sealed, "Sealed");
  assert.deepEqual(STATUS_ORDER, ["running", "partial", "sealed", "planned", "history"]);
  const t = BENCHMARKS.find((b) => b.slug === "sealed-test")!;
  assert.equal(t.status, "sealed");
  assert.deepEqual(t.data, ["sealed"]);
  assert.equal(t.headline, undefined, "nothing on the sealed test is read for a card");
  assert.equal(STATUS_LABEL.history, "Development history");
  const v1 = BENCHMARKS[0];
  assert.equal(v1.slug, "newest-weeks");
  assert.match(v1.longName!, /dataset release v1/);
  assert.match(BENCHMARKS.find((b) => b.slug === "held-out")!.records, /part of dataset release v1 \(5,679 in its training split, 7,381 in its validation split/);
});

test("the Research bar leads with the overview; Experiments is there for everyone, with its lock", () => {
  const visitor = researchBar(me("identify", null));
  assert.deepEqual(visitor.slice(0, 2).map((n) => n.label), ["Overview", "Results"]);
  assert.equal(visitor.find((n) => n.href === "/research/experiments")!.members, true, "locked for a visitor");
  assert.deepEqual(researchBar(me("all", someone)), visitor);
  assert.equal(visitor.filter((n) => n.members).length, 1, "only Experiments is members-only");
  assert.ok(!researchNav(me("identify", null)).some((n) => n.href === "/research/experiments"), "the header menu still hides it");
  assert.equal(isBarTabActive("/research", "/research"), true);
  assert.equal(isBarTabActive("/research", "/research/results"), false, "Overview only on the overview");
  assert.equal(isBarTabActive("/research/benchmarks", "/research/benchmarks/rare-species"), true);
  assert.equal(isBarTabActive("/research/benchmarks", "/research/benchmarksx"), false);
});

test("the base-model screen quotes its committed entry: one run, FungiTastic BEiT-B (frozen) beside the bases", () => {
  assert.equal(BASE_MODEL_SCREEN.queries, 664);
  assert.match(BASE_MODEL_SCREEN.source, /experiments\/2026-10-13-v1-backbone-screen\.md, addendum of 25 Oct \(origin\/main ef3a8df\)/);
  assert.deepEqual(BASE_MODEL_SCREEN.rows.map((r) => [r.name, r.top1, r.top3, r.top5, r.top10]), [
    ["BioCLIP 2", 70.9, 85.2, 88.9, 92.6], ["FungiTastic BEiT-B (frozen)", 68.8, 81.3, 84.5, 88.5],
    ["BioCLIP", 58.3, 72.6, 78.0, 83.4], ["DINOv3 large", 54.7, 70.2, 76.0, 80.1]]);
  assert.deepEqual(BASE_MODEL_SCREEN.rows.filter((r) => r.run), [], "one run only: no row from another run");
  assert.ok(!BASE_MODEL_SCREEN.rows.some((r) => /MycoMap Vision|fine-tun/i.test(r.name)), "base models only");
  for (const r of BASE_MODEL_SCREEN.rows) assert.ok(r.top1 <= r.top3 && r.top3 <= r.top5 && r.top5 <= r.top10, r.name);
  assert.deepEqual(screenLead(BASE_MODEL_SCREEN.rows), { best: BASE_MODEL_SCREEN.rows[0], points: 2.1 });
  assert.equal(screenLead(BASE_MODEL_SCREEN.rows.slice(0, 1)), null);
});

test("the fungi-specific cards cite the committed E33 entry; the pending card stays off until its numbers land", () => {
  assert.equal(SHOW_PENDING_FUNGI_CARDS, false);
  assert.deepEqual(fungiCards().map((c) => c.key), ["d", "c", "fewshot"]);
  assert.deepEqual(fungiCards(true).map((c) => c.key), ["d", "c", "fewshot", "a"]);
  assert.equal(E33.commit, "8147947");
  for (const c of FUNGI_CARDS) assert.match(c.source, /experiments\/2026-10-23-fungitastic-own-test\.md \(origin\/main 8147947\)/);
  const d = FUNGI_CARDS.find((c) => c.key === "d")!;
  assert.ok(d.kind === "replication" && d.ours === 75.28 && d.theirs === 75.3 && !d.interim);
  const v = FUNGI_CARDS.find((c) => c.key === "c")!;
  assert.ok(v.kind === "versus" && v.interim, "their recipe on our photos is interim");
  if (v.kind === "versus") {
    assert.deepEqual(v.rows.map((r) => r.top1), [55.31, 51.23]);
    assert.equal(v.finds, 12874);
  }
  assert.ok(FUNGI_CARDS.find((c) => c.key === "fewshot")!.interim);
});

test("citations come as text and BibTeX, the site first and the base model second", () => {
  assert.deepEqual(CITATIONS.map((c) => c.key), ["site", "gu", "stevens", "picek22", "picek25"]);
  for (const c of CITATIONS) {
    assert.match(c.bibtex, /^@(misc|article|inproceedings)\{\w+,\n[^]*\n\}$/, c.key);
    assert.doesNotMatch(c.text + c.bibtex, /zenodo|hugging ?face/i, c.key);
  }
});

test("API is the last item under Research, for visitors and members alike", () => {
  for (const m of [undefined, me("identify", null), me("all", null), me("all", someone)]) {
    const last = researchNav(m).at(-1)!;
    assert.deepEqual([last.href, last.label], ["/research/api", "API"]);
  }
  assert.equal(API_PATH, "/research/api");
});

test("a signed-out visitor reads the API page and the whole Research section, and signs in for the rest", () => {
  const visitor = me("all", null);
  assert.equal(mustSignIn(visitor, "/research/api"), false);
  assert.equal(mustSignIn(visitor, "/api-docs"), false);
  for (const page of ["/research", "/research/results", "/research/data", "/research/protocols",
                      "/research/benchmarks/sealed-test", "/research/experiments",
                      "/research/for-researchers", "/models", "/data"]) {
    assert.equal(mustSignIn(visitor, page), false, page);
  }
  assert.equal(mustSignIn(visitor, "/feedback"), false, "Human Feedback, read-only");
  assert.equal(mustSignIn(visitor, "/about"), false, "How it works, open to anyone");
  assert.equal(mustSignIn(visitor, "/get-involved"), false, "Get involved, open to anyone");
  for (const page of ["/", "/get-involvedx", "/aboutx", "/feedbackx", "/researchx"]) {
    assert.equal(mustSignIn(visitor, page), true, page);
  }
});

test("of the pages whose link card crawlers may read, only Identify sends a signed-out person to sign in", () => {
  // The server answers /, /about and /get-involved with the shell and its link-preview card
  // (signin.SHARE_PAGES, linkcards.py); the app sends a signed-out person on from / only
  // (How it works and Get involved are open to anyone).
  const visitor = me("all", null);
  assert.equal(mustSignIn(visitor, "/"), true, "/");
  for (const page of ["/about", "/get-involved"]) {
    assert.equal(mustSignIn(visitor, page), false, page);
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

test("the sealed test says when its records start and publishes nothing before it is opened", () => {
  const t = BENCHMARKS.find((b) => b.slug === "sealed-test")!;
  assert.equal(t.status, "sealed");
  assert.equal(t.results, undefined);
  assert.match(t.records, /after dataset release v1's label snapshot \(12 October 2026, 13:17 UTC\)/);
  assert.match(t.records, /at least 3,000 records/);
  assert.match(t.records, /no numbers are published before it is opened/);
  assert.match(t.pending!.text, /No numbers are published before it is opened/);
});

test("the v1 benchmark's split starts at the release's own cutoff", () => {
  assert.match(BENCHMARKS[0].records, /eight weeks after 17 August 2026/);
});
