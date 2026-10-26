import { test } from "node:test";
import assert from "node:assert/strict";
import { API_PATH, BASE_MODEL_SCREEN, BENCHMARKS, CITATIONS, crossHeadline, DATA_LABEL, E33, FUNGI_CARDS,
         isBarTabActive, isResearchPath, isWideCard, MEASURE_LABEL, MOVED, mustSignIn, PUBLIC_NAV, researchBar,
         researchNav, screenLead, STATUS_LABEL, STATUS_ORDER, V1_PUBLISHED, type FungiCard } from "./research";
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

const card = <K extends FungiCard["kind"]>(key: string, kind: K): Extract<FungiCard, { kind: K }> => {
  const c = FUNGI_CARDS.find((x) => x.key === key);
  assert.ok(c && c.kind === kind, `card ${key} is a ${kind} card`);
  return c as Extract<FungiCard, { kind: K }>;
};

test("the fungi-specific cards: every one committed, each citing its entry on research main, none pending", () => {
  assert.deepEqual(FUNGI_CARDS.map((c) => c.key), ["d", "c", "b1", "b2", "a", "fewshot"]);
  assert.deepEqual(FUNGI_CARDS.filter(isWideCard).map((c) => c.key), ["b2", "a", "fewshot"]);
  assert.equal(E33.commit, "6090354");
  assert.equal(V1_PUBLISHED.commit, "5792c17");
  for (const c of FUNGI_CARDS) {
    const entry = c.key === "a" ? /experiments\/2026-10-13-v1-published-models\.md.*\(origin\/main 5792c17\)/
      : /experiments\/2026-10-23-fungitastic-own-test\.md \(origin\/main 6090354\)/;
    assert.match(c.source, entry, c.key);
  }
  // Only (c), E4 still training, is interim in E33; every other card is final there.
  assert.deepEqual(FUNGI_CARDS.filter((c) => c.interim).map((c) => c.key), ["c"]);
  for (const c of FUNGI_CARDS) if (!c.interim) assert.ok(c.tag, `${c.key} has a pill`);
});

test("no sealed-test number, no image, and the agreed model names on the fungi cards", () => {
  const all = JSON.stringify(FUNGI_CARDS);
  assert.doesNotMatch(all, /v1-test|sealed/i, "these are validation finds or their own test sets");
  assert.doesNotMatch(all, /\.(jpe?g|png|webp)\b|https?:|zenodo|huggingface/i, "scores only: no images or download links");
  assert.doesNotMatch(all, /guidance|micrograph|habitat/i, "the val-only guidance look stays off the site");
  const names = FUNGI_CARDS.flatMap((c) => c.kind === "pairs" ? c.tables.flatMap((t) => t.rows.map((r) => r.name))
    : c.kind === "published" ? c.rows.map((r) => r.name) : c.kind === "versus" ? c.rows.map((r) => r.name) : []);
  for (const n of names) {
    assert.match(n, /^(MycoMap Vision v1(-open)?|BioCLIP 2 \(base model, not fine-tuned\)|BioCLIP \(base model, not fine-tuned\)|FungiTastic (BEiT-B|ViT-B|Swin-B|EfficientNetV2-B3|ConvNeXt-B)\/384|FungiTastic recipe|FungiTastic paper: BioCLIP|Danish Fungi 2020 ViT-L\/384)$/, n);
  }
});

test("(d) the replication: their checkpoints give their Table 8 numbers; the three added ones score lower", () => {
  const d = card("d", "replication");
  assert.deepEqual([d.ours, d.theirs, d.ours3, d.theirs3], [75.28, 75.3, 86.68, 86.7]);
  assert.match(d.also, /74\.59 \/ 86\.39 against their 74\.9 \/ 86\.3/);
  assert.match(d.also, /Swin-B\/384 74\.24, EfficientNetV2-B3\/384 72\.03, ConvNeXt-B\/384 70\.71/);
});

test("(c) their recipe on our open photos stays interim at epoch 13", () => {
  const v = card("c", "versus");
  assert.ok(v.interim);
  assert.deepEqual(v.rows.map((r) => r.top1), [55.31, 51.23]);
  assert.deepEqual([v.finds, v.withSpecies], [12874, 12798]);
});

test("(b1) our North American reference on their test sets: E33's rows, 43.77 per observation on the species we have", () => {
  const b1 = card("b1", "cross");
  assert.deepEqual(b1.rows.map((r) => [r.split, r.per, r.of, r.n, ...r.topk]), [
    ["ClosedSet-Test", "image", "ours", 54841, 37.23, 55.61, 62.85, 72.09],
    ["ClosedSet-Test", "observation", "ours", 29558, 43.77, 63.07, 70.45, 79.49],
    ["ClosedSet-Test", "image", "all", 91756, 22.25, 33.24, 37.56, 43.09],
    ["ClosedSet-Test", "observation", "all", 48361, 26.75, 38.55, 43.06, 48.59],
    ["DNA-Test", "image", "ours", 2782, 16.10, 27.93, 34.51, 42.95],
    ["DNA-Test", "observation", "ours", 1100, 23.73, 39.82, 46.64, 55.73],
    ["DNA-Test", "image", "all", 5105, 8.78, 15.22, 18.81, 23.41],
  ]);
  assert.equal(crossHeadline(b1.rows)?.topk[0], 43.77);
  assert.match(b1.note, /37\.18 per image, 43\.86 per observation/);
  assert.match(b1.note, /photos only/);
});

test("(b2) head to head on their test sets: every model's top 1/3/5/10 per image and per observation, and the paired intervals", () => {
  const b2 = card("b2", "pairs");
  const rows = (split: string) => b2.tables.find((t) => t.split === split)!.rows.map((r) => [r.name, ...r.image, ...(r.observation ?? [])]);
  assert.deepEqual(b2.tables.map((t) => [t.split, t.images, t.observations]), [["ClosedSet-Test", 91756, 48361], ["DNA-Test", 5105, 2041]]);
  assert.deepEqual(rows("ClosedSet-Test"), [
    ["MycoMap Vision v1-open", 77.46, 89.09, 91.81, 94.25, 88.31, 96.76, 98.21, 99.13],
    ["MycoMap Vision v1", 77.54, 89.12, 91.81, 94.31, 88.29, 96.78, 98.22, 99.13],
    ["BioCLIP 2 (base model, not fine-tuned)", 76.36, 88.26, 91.04, 93.60, 87.61, 96.38, 97.86, 98.88],
    ["FungiTastic BEiT-B/384", 75.28, 86.68, 89.56, 92.26, 87.42, 96.01, 97.52, 98.66],
    ["FungiTastic ViT-B/384", 74.59, 86.39, 89.39, 92.10, 86.41, 95.47, 97.10, 98.34],
    ["FungiTastic Swin-B/384", 74.24, 86.33, 89.33, 92.02],
    ["FungiTastic EfficientNetV2-B3/384", 72.03, 84.67, 88.02, 91.22],
    ["FungiTastic ConvNeXt-B/384", 70.71, 83.83, 87.29, 90.74],
  ]);
  assert.deepEqual(rows("DNA-Test"), [
    ["MycoMap Vision v1-open", 41.86, 60.63, 67.44, 74.46, 56.59, 78.83, 86.58, 92.60],
    ["MycoMap Vision v1", 41.78, 61.25, 67.44, 74.55, 56.83, 79.27, 86.38, 92.41],
    ["BioCLIP 2 (base model, not fine-tuned)", 37.67, 56.28, 63.70, 71.36, 51.54, 73.35, 81.58, 88.49],
    ["FungiTastic BEiT-B/384", 38.73, 56.87, 63.82, 70.91, 57.28, 78.69, 85.84, 92.45],
    ["FungiTastic ViT-B/384", 36.38, 53.50, 60.24, 67.82],
    ["FungiTastic Swin-B/384", 37.96, 57.51, 63.66, 71.26],
    ["FungiTastic EfficientNetV2-B3/384", 35.55, 54.44, 61.27, 69.21],
    ["FungiTastic ConvNeXt-B/384", 34.01, 53.12, 60.00, 68.15],
  ]);
  assert.deepEqual(b2.diffs!.map((r) => [r.name, ...r.cells.map((x) => [x.d, x.lo, x.hi])]), [
    ["MycoMap Vision v1-open − FungiTastic BEiT-B/384", [2.18, 1.92, 2.47], [0.90, 0.63, 1.17], [3.13, 1.63, 4.55], [-0.69, -2.79, 1.32]],
    ["MycoMap Vision v1-open − FungiTastic ViT-B/384", [2.87, 2.59, 3.16], [1.91, 1.62, 2.18], [5.48, 4.01, 6.87], [3.77, 1.57, 5.88]],
    ["MycoMap Vision v1 − FungiTastic BEiT-B/384", [2.27, 2.00, 2.54], [0.87, 0.60, 1.15], [3.06, 1.59, 4.48], [-0.44, -2.55, 1.57]],
    ["What the fine-tune adds: MycoMap Vision v1-open − BioCLIP 2 (base model, not fine-tuned)", [1.11, 0.93, 1.28], [0.71, 0.51, 0.90], [4.19, 3.22, 5.20], [5.05, 3.58, 6.61]],
  ]);
  for (const r of b2.diffs!) for (const x of r.cells) assert.ok(x.lo <= x.d && x.d <= x.hi, r.name);
  assert.match(b2.lead, /Both sides are photo only/);
  assert.match(b2.lead, /not trained on FungiTastic/);
  assert.equal(b2.versus, "FungiTastic BEiT-B/384");
});

test("(a) their published models on our v1 validation finds: species top-k, genus, and the 2,411 finds of species they have", () => {
  const a = card("a", "published");
  assert.deepEqual([a.finds, a.withSpecies, a.shared], [12874, 12798, 2411]);
  assert.deepEqual(a.rows.map((r) => [r.name, ...r.species, r.genus, r.shared]), [
    ["MycoMap Vision v1", 56.3, 73.6, 78.9, 84.3, 85.8, 67.6],
    ["MycoMap Vision v1-open", 55.3, 72.6, 77.7, 83.3, 85.4, 66.8],
    ["FungiTastic BEiT-B/384", 10.7, 14.1, 15.2, 16.5, 58.6, 56.9],
    ["FungiTastic Swin-B/384", 10.3, 13.7, 14.9, 16.2, 56.6, 54.7],
    ["FungiTastic EfficientNetV2-B3/384", 10.2, 13.5, 14.7, 15.9, 56.2, 54.1],
    ["FungiTastic ViT-B/384", 10.1, 13.4, 14.6, 15.8, 55.6, 53.5],
    ["FungiTastic ConvNeXt-B/384", 9.6, 13.0, 14.4, 15.8, 53.9, 51.0],
    ["Danish Fungi 2020 ViT-L/384", 10.2, 12.9, 13.8, 14.6, 55.6, 52.6],
  ]);
  assert.deepEqual(a.rows.map((r) => [...r.ci, ...r.sharedCi]), [
    [53.4, 59.2, 64.6, 70.4], [52.4, 58.2, 63.9, 69.5], [9.4, 12.4, 53.1, 61.2], [8.9, 12.1, 50.3, 59.7],
    [8.8, 11.9, 50.1, 58.6], [8.7, 11.7, 49.4, 57.9], [8.2, 11.4, 46.7, 56.2], [8.9, 11.6, 49.2, 55.9],
  ]);
  for (const r of a.rows) {
    assert.ok(r.ci[0] <= r.species[0]! && r.species[0]! <= r.ci[1], r.name);
    assert.ok(r.sharedCi[0] <= r.shared && r.shared <= r.sharedCi[1], r.name);
  }
});

test("few-shot: E33's centroid rows beside BioCLIP, BioCLIP 2 base and their published 21.8, with the served method", () => {
  const f = card("fewshot", "pairs");
  assert.deepEqual(f.tables.map((t) => [t.split, t.images, t.observations]), [["FewShot-Test", 1909, 999]]);
  assert.deepEqual(f.tables[0].rows.map((r) => [r.name, r.detail, ...r.image, ...(r.observation ?? [])]), [
    ["MycoMap Vision v1-open", "class centroid", 46.41, 61.92, 67.78, 76.27, 49.75, 64.46, 70.87, 78.48],
    ["MycoMap Vision v1", "class centroid", 45.94, 61.92, 68.31, 77.06, 49.25, 63.96, 71.67, 79.98],
    ["MycoMap Vision v1-open", "served method", 44.84, 59.98, 67.31, 75.22, 47.65, 63.46, 70.27, 77.98],
    ["MycoMap Vision v1", "served method", 45.31, 61.08, 68.05, 76.43, 48.35, 63.76, 70.77, 79.38],
    ["BioCLIP 2 (base model, not fine-tuned)", "class centroid, 336 px", 43.11, 58.88, 66.21, 73.86, 47.25, 63.16, 68.97, 76.68],
    ["BioCLIP (base model, not fine-tuned)", "class centroid, our pipeline", 24.99, 37.40, 43.84, 52.28, 29.83, 42.54, 48.25, 56.86],
    ["FungiTastic paper: BioCLIP", "their Table 4, best centroid", 21.8, 32.6, null, null],
  ]);
  assert.equal(f.tables[0].rows.at(-1)!.digits, 1, "their 21.8 is shown as they publish it");
});

test("every top-k row on the fungi cards rises from top 1 to top 10", () => {
  const rise = (k: readonly (number | null)[], what: string) => {
    const v = k.filter((x): x is number => x != null);
    for (let i = 1; i < v.length; i++) assert.ok(v[i - 1] <= v[i], what);
  };
  for (const c of FUNGI_CARDS) {
    if (c.kind === "pairs") for (const t of c.tables) for (const r of t.rows) { rise(r.image, r.name); if (r.observation) rise(r.observation, r.name); }
    if (c.kind === "published") for (const r of c.rows) rise(r.species, r.name);
    if (c.kind === "cross") for (const r of c.rows) rise(r.topk, `${r.split} ${r.per} ${r.of}`);
  }
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
