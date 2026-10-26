import { test } from "node:test";
import assert from "node:assert/strict";
import { ADVANCE_PAUSED_SINCE, API_PATH, BASE_MODEL_SCREEN, BENCHMARKS, cellDigits, CITATIONS, DATA_LABEL, diffsHeading,
         E33, isBarTabActive, isResearchPath, MEASURE_LABEL, MOVED, mustSignIn, PUBLIC_NAV, researchBar, researchNav,
         resultCards, SEALED_TEST, STATUS_LABEL, STATUS_ORDER, TEST_SET_RESULTS, V1_IMAGE_SIZE, V1_PUBLISHED, valueCaption,
         type TestSetResults } from "./research";
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
  assert.deepEqual(STATUS_ORDER, ["running", "partial", "paused", "sealed", "planned", "history"]);
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
});

const set = (key: string): TestSetResults => {
  const s = TEST_SET_RESULTS.find((x) => x.key === key);
  assert.ok(s, key);
  return s;
};
/** A set's rows as [name, detail, ...values], so a wrong, moved or missing number fails. */
const table = (key: string) => set(key).rows.map((r) => [r.name, r.detail ?? "", ...r.cells]);
const diffs = (key: string) => set(key).diffs!.rows.map((r) => [r.name, ...r.cells.map((x) => [x.d, x.lo, x.hi])]);

test("results by test set: every set committed, each citing its entry on research main; grouped into cards by test set", () => {
  assert.deepEqual(TEST_SET_RESULTS.map((s) => s.key), ["fungitastic-closed-ft-train", "fungitastic-closed-mycomap-ref",
    "fungitastic-dna-ft-train", "fungitastic-dna-mycomap-ref", "fungitastic-fewshot", "v1-val-as-released", "v1-val-open-photos",
    "v1-val-image-size"]);
  assert.deepEqual(resultCards().map((c) => [c.title, c.sets.map((s) => s.condition)]), [
    ["FungiTastic ClosedSet-Test", ["reference: FungiTastic-Train", "reference: MycoMap North American records"]],
    ["FungiTastic DNA-Test", ["reference: FungiTastic-Train", "reference: MycoMap North American records"]],
    ["FungiTastic FewShot-Test", ["reference: FungiTastic FewShot-Train (7,819 images, 2,427 species)"]],
    ["MycoMap v1 validation (12,798 finds)", ["models as released", "training: open photos only (E4)", "image size: 224 px (v0) and 336 px (v1)"]],
  ]);
  assert.deepEqual(resultCards([{ ...set("fungitastic-fewshot"), card: "FungiTastic" }, { ...set("fungitastic-dna-ft-train"), card: "FungiTastic" }])
    .map((c) => [c.title, c.sets.length]), [["FungiTastic", 2]], "a card can aggregate different test sets");
  assert.equal(E33.commit, "2e2576a");
  assert.equal(V1_PUBLISHED.commit, "5792c17");
  assert.equal(V1_IMAGE_SIZE.commit, "7ce997d");
  for (const s of TEST_SET_RESULTS) {
    const entry = s.key === "v1-val-as-released" ? /experiments\/2026-10-13-v1-published-models\.md.*\(origin\/main 5792c17\)/
      : s.key === "v1-val-image-size" ? /experiments\/2026-10-18-v1-image-size\.md \(origin\/main 7ce997d\)/
      : /experiments\/2026-10-23-fungitastic-own-test\.md \(origin\/main 2e2576a\)/;
    assert.match(s.source, entry, s.key);
    assert.ok(s.about && s.size && s.maker, s.key);
    for (const r of s.rows) {
      assert.equal(r.cells.length, s.columns.length, `${s.key}: ${r.name}`);
      if (r.cis) assert.equal(r.cis.length, s.columns.length, `${s.key}: ${r.name}`);
    }
  }
  // E33 marks E4 final (stopped at epoch 20, best epoch 13): nothing is interim.
  assert.deepEqual(TEST_SET_RESULTS.filter((s) => s.interim).map((s) => s.key), []);
});

test("results by test set: no sealed-test number, no image, no competitive framing, and the agreed model names", () => {
  const all = JSON.stringify(TEST_SET_RESULTS);
  assert.doesNotMatch(all, /v1-test|sealed/i, "these are validation finds or FungiTastic's own test sets");
  assert.doesNotMatch(all, /\.(jpe?g|png|webp)\b|https?:|zenodo|huggingface/i, "scores only: no images or download links");
  assert.doesNotMatch(all, /guidance|micrograph|habitat/i, "the val-only guidance look stays off the site");
  assert.doesNotMatch(all, /head to head|\bbeats?\b|\blead|\bwins?\b|\bversus\b|\bvs\.?\b|\btheir\b|\bour\b|\bwe\b/i, "not a competition (Steve, 2026-10-26)");
  for (const r of TEST_SET_RESULTS.flatMap((s) => s.rows)) {
    assert.match(r.name, /^(MycoMap Vision v[01](-open)?|BioCLIP 2 \(base model, not fine-tuned\)|BioCLIP \(base model, not fine-tuned\)|FungiTastic (BEiT-B|ViT-B|Swin-B|EfficientNetV2-B3|ConvNeXt-B)\/384|FungiTastic recipe: BEiT-B\/384|Danish Fungi 2020 ViT-L\/384)$/, r.name);
  }
});

test("FungiTastic ClosedSet-Test, FungiTastic-Train as reference: every model's top 1/3/5/10 per image and per observation, the scoring check and the paired intervals", () => {
  assert.deepEqual(table("fungitastic-closed-ft-train"), [
    ["MycoMap Vision v1-open", "retrieval, FungiTastic-Train as reference", 77.46, 89.09, 91.81, 94.25, 88.31, 96.76, 98.21, 99.13],
    ["MycoMap Vision v1", "retrieval, FungiTastic-Train as reference", 77.54, 89.12, 91.81, 94.31, 88.29, 96.78, 98.22, 99.13],
    ["BioCLIP 2 (base model, not fine-tuned)", "336 px · retrieval, FungiTastic-Train as reference", 76.36, 88.26, 91.04, 93.60, 87.61, 96.38, 97.86, 98.88],
    ["FungiTastic BEiT-B/384", "classifier trained on FungiTastic-Train", 75.28, 86.68, 89.56, 92.26, 87.42, 96.01, 97.52, 98.66],
    ["FungiTastic ViT-B/384", "classifier trained on FungiTastic-Train", 74.59, 86.39, 89.39, 92.10, 86.41, 95.47, 97.10, 98.34],
    ["FungiTastic Swin-B/384", "classifier · per image, all 91,832 images", 74.24, 86.33, 89.33, 92.02, null, null, null, null],
    ["FungiTastic EfficientNetV2-B3/384", "classifier · per image, all 91,832 images", 72.03, 84.67, 88.02, 91.22, null, null, null, null],
    ["FungiTastic ConvNeXt-B/384", "classifier · per image, all 91,832 images", 70.71, 83.83, 87.29, 90.74, null, null, null, null],
    ["Danish Fungi 2020 ViT-L/384", "classifier, 1,604 classes · per image, all 91,832 images", 69.15, 79.41, 81.70, 83.84, null, null, null, null],
  ]);
  assert.deepEqual(diffs("fungitastic-closed-ft-train"), [
    ["MycoMap Vision v1-open − FungiTastic BEiT-B/384", [2.18, 1.92, 2.47], [0.90, 0.63, 1.17]],
    ["MycoMap Vision v1-open − FungiTastic ViT-B/384", [2.87, 2.59, 3.16], [1.91, 1.62, 2.18]],
    ["MycoMap Vision v1 − FungiTastic BEiT-B/384", [2.27, 2.00, 2.54], [0.87, 0.60, 1.15]],
    ["MycoMap Vision v1-open − BioCLIP 2 (base model, not fine-tuned)", [1.11, 0.93, 1.28], [0.71, 0.51, 0.90]],
  ]);
  const check = set("fungitastic-closed-ft-train").checks!.join(" ");
  assert.match(check, /^Scoring check: FungiTastic BEiT-B\/384 as published scores 75\.28 at top 1 and 86\.68 at top 3 in this pipeline, against 75\.3 and 86\.7 in the FungiTastic paper \(Table 8\); FungiTastic ViT-B\/384 74\.59 \/ 86\.39 against 74\.9 \/ 86\.3\.$/);
  assert.match(set("fungitastic-closed-ft-train").notes.join(" "), /photos alone/);
});

test("FungiTastic DNA-Test, FungiTastic-Train as reference: every model, and the paired intervals", () => {
  assert.deepEqual(table("fungitastic-dna-ft-train"), [
    ["MycoMap Vision v1-open", "retrieval, FungiTastic-Train as reference", 41.86, 60.63, 67.44, 74.46, 56.59, 78.83, 86.58, 92.60],
    ["MycoMap Vision v1", "retrieval, FungiTastic-Train as reference", 41.78, 61.25, 67.44, 74.55, 56.83, 79.27, 86.38, 92.41],
    ["BioCLIP 2 (base model, not fine-tuned)", "336 px · retrieval, FungiTastic-Train as reference", 37.67, 56.28, 63.70, 71.36, 51.54, 73.35, 81.58, 88.49],
    ["FungiTastic BEiT-B/384", "classifier trained on FungiTastic-Train", 38.73, 56.87, 63.82, 70.91, 57.28, 78.69, 85.84, 92.45],
    ["FungiTastic ViT-B/384", "classifier · per image", 36.38, 53.50, 60.24, 67.82, null, null, null, null],
    ["FungiTastic Swin-B/384", "classifier · per image", 37.96, 57.51, 63.66, 71.26, null, null, null, null],
    ["FungiTastic EfficientNetV2-B3/384", "classifier · per image", 35.55, 54.44, 61.27, 69.21, null, null, null, null],
    ["FungiTastic ConvNeXt-B/384", "classifier · per image", 34.01, 53.12, 60.00, 68.15, null, null, null, null],
    ["Danish Fungi 2020 ViT-L/384", "classifier, 1,604 classes · per image", 31.75, 41.49, 44.72, 48.33, null, null, null, null],
  ]);
  assert.deepEqual(diffs("fungitastic-dna-ft-train"), [
    ["MycoMap Vision v1-open − FungiTastic BEiT-B/384", [3.13, 1.63, 4.55], [-0.69, -2.79, 1.32]],
    ["MycoMap Vision v1-open − FungiTastic ViT-B/384", [5.48, 4.01, 6.87], [3.77, 1.57, 5.88]],
    ["MycoMap Vision v1 − FungiTastic BEiT-B/384", [3.06, 1.59, 4.48], [-0.44, -2.55, 1.57]],
    ["MycoMap Vision v1-open − BioCLIP 2 (base model, not fine-tuned)", [4.19, 3.22, 5.20], [5.05, 3.58, 6.61]],
  ]);
  for (const k of ["fungitastic-closed-ft-train", "fungitastic-dna-ft-train"]) {
    for (const r of set(k).diffs!.rows) for (const x of r.cells) assert.ok(x.lo <= x.d && x.d <= x.hi, r.name);
  }
});

test("FungiTastic ClosedSet-Test and DNA-Test with MycoMap's North American records as the reference: E33's rows", () => {
  assert.deepEqual(table("fungitastic-closed-mycomap-ref").map((r) => r.slice(2)), [
    [37.23, 55.61, 62.85, 72.09, 43.77, 63.07, 70.45, 79.49],
    [22.25, 33.24, 37.56, 43.09, 26.75, 38.55, 43.06, 48.59],
    [37.18, 55.49, 62.98, 72.59, 43.86, 62.90, 70.46, 80.25],
    [22.54, 33.64, 38.18, 44.01, 27.12, 38.89, 43.57, 49.63],
  ]);
  assert.deepEqual(table("fungitastic-closed-mycomap-ref").map((r) => r[0]),
    ["MycoMap Vision v1-open", "MycoMap Vision v1-open", "MycoMap Vision v1", "MycoMap Vision v1"]);
  assert.match(set("fungitastic-closed-mycomap-ref").rows[0].detail!, /54,841 test images \(29,558 observations\) of the 979 species/);
  assert.match(set("fungitastic-closed-mycomap-ref").rows[2].detail!, /55,631 test images \(29,905 observations\) of the 994 species/);
  assert.deepEqual(table("fungitastic-dna-mycomap-ref").map((r) => r.slice(2)), [
    [16.10, 27.93, 34.51, 42.95, 23.73, 39.82, 46.64, 55.73],
    [8.78, 15.22, 18.81, 23.41, null, null, null, null],
  ]);
  assert.match(set("fungitastic-dna-mycomap-ref").rows[0].detail!, /2,782 test images \(1,100 observations\) of the 383 species/);
});

test("FungiTastic FewShot-Test: the published BioCLIP 21.8, and E33's centroid and served-method rows", () => {
  assert.deepEqual(table("fungitastic-fewshot"), [
    ["BioCLIP (base model, not fine-tuned)", "class centroid, as published in the FungiTastic paper (Table 4)", 21.8, 32.6, null, null, null, null, null, null],
    ["BioCLIP (base model, not fine-tuned)", "class centroid", 24.99, 37.40, 43.84, 52.28, 29.83, 42.54, 48.25, 56.86],
    ["BioCLIP 2 (base model, not fine-tuned)", "336 px · class centroid", 43.11, 58.88, 66.21, 73.86, 47.25, 63.16, 68.97, 76.68],
    ["MycoMap Vision v1-open", "class centroid", 46.41, 61.92, 67.78, 76.27, 49.75, 64.46, 70.87, 78.48],
    ["MycoMap Vision v1", "class centroid", 45.94, 61.92, 68.31, 77.06, 49.25, 63.96, 71.67, 79.98],
    ["BioCLIP (base model, not fine-tuned)", "closest specimens + species average", 24.67, 36.14, 42.85, 52.49, 28.93, 41.24, 47.35, 56.26],
    ["BioCLIP 2 (base model, not fine-tuned)", "336 px · closest specimens + species average", 42.74, 57.52, 64.75, 73.02, 46.25, 62.46, 68.77, 75.18],
    ["MycoMap Vision v1-open", "closest specimens + species average", 44.84, 59.98, 67.31, 75.22, 47.65, 63.46, 70.27, 77.98],
    ["MycoMap Vision v1", "closest specimens + species average", 45.31, 61.08, 68.05, 76.43, 48.35, 63.76, 70.77, 79.38],
  ]);
  const s = set("fungitastic-fewshot");
  assert.equal(cellDigits(s, s.rows[0], 0), 1, "the published 21.8 is shown as published");
  assert.equal(cellDigits(s, s.rows[1], 0), 2);
  assert.equal(s.size, "1,909 images · 999 observations · 567 species");
});

test("MycoMap v1 validation, models as released: species top-k, genus, the 2,411 finds in FungiTastic's classes, and the intervals", () => {
  const s = set("v1-val-as-released");
  assert.deepEqual(s.columns.map((c) => c.group), ["Species: % of 12,798 finds", "Species: % of 12,798 finds",
    "Species: % of 12,798 finds", "Species: % of 12,798 finds", "Genus: % of 12,865 finds",
    "Species in FungiTastic's classes: % of 2,411 finds"]);
  assert.deepEqual(table("v1-val-as-released").map((r) => [r[0], ...r.slice(2)]), [
    ["MycoMap Vision v1", 56.3, 73.6, 78.9, 84.3, 85.8, 67.6],
    ["MycoMap Vision v1-open", 55.3, 72.6, 77.7, 83.3, 85.4, 66.8],
    ["MycoMap Vision v1-open", 55.9, 73.4, 78.5, 84.1, 85.6, null],
    ["FungiTastic BEiT-B/384", 10.7, 14.1, 15.2, 16.5, 58.6, 56.9],
    ["FungiTastic Swin-B/384", 10.3, 13.7, 14.9, 16.2, 56.6, 54.7],
    ["FungiTastic EfficientNetV2-B3/384", 10.2, 13.5, 14.7, 15.9, 56.2, 54.1],
    ["FungiTastic ViT-B/384", 10.1, 13.4, 14.6, 15.8, 55.6, 53.5],
    ["FungiTastic ConvNeXt-B/384", 9.6, 13.0, 14.4, 15.8, 53.9, 51.0],
    ["Danish Fungi 2020 ViT-L/384", 10.2, 12.9, 13.8, 14.6, 55.6, 52.6],
  ]);
  assert.deepEqual(s.rows.map((r) => [r.cis![0], r.cis![4], r.cis![5]]), [
    [[53.4, 59.2], [84.0, 87.5], [64.6, 70.4]], [[52.4, 58.2], [83.6, 87.0], [63.9, 69.5]], [[53.0, 58.9], [83.8, 87.3], null],
    [[9.4, 12.4], [56.0, 61.4], [53.1, 61.2]], [[8.9, 12.1], null, [50.3, 59.7]], [[8.8, 11.9], null, [50.1, 58.6]],
    [[8.7, 11.7], [52.8, 58.5], [49.4, 57.9]], [[8.2, 11.4], null, [46.7, 56.2]], [[8.9, 11.6], [52.9, 58.3], [49.2, 55.9]],
  ]);
  for (const r of s.rows) r.cells.forEach((v, k) => {
    const ci = r.cis?.[k];
    if (ci && v != null) assert.ok(ci[0] <= v && v <= ci[1], `${r.name} column ${k}`);
  });
});

test("MycoMap v1 validation, trained on open photos only: E4 final at its best epoch 13, with the paired interval", () => {
  const s = set("v1-val-open-photos");
  assert.ok(!s.interim);
  assert.deepEqual(table("v1-val-open-photos").map((r) => [r[0], ...r.slice(2)]), [
    ["FungiTastic recipe: BEiT-B/384", 51.23, 67.18, 72.90, 79.03, 80.22],
    ["MycoMap Vision v1-open", 55.31, 72.61, 77.65, 83.29, 84.9],
    ["MycoMap Vision v1", 56.32, 73.55, 78.87, 84.28, 85.4],
  ]);
  assert.match(s.rows[0].detail!, /final: best epoch 13/);
  assert.deepEqual(diffs("v1-val-open-photos"), [
    ["FungiTastic recipe: BEiT-B/384 − MycoMap Vision v1-open", [-4.04, -4.91, -3.24], [-5.41, -6.37, -4.53], [-4.71, -5.60, -3.92], [-4.22, -4.97, -3.59]],
  ]);
  for (const x of s.diffs!.rows[0].cells) assert.ok(x.lo <= x.d && x.d <= x.hi);
  assert.match(s.notes.join(" "), /55\.27 rather than 55\.31/);
  assert.match(s.notes.join(" "), /stopped by its declared epoch rule at epoch 20; the final model is its best epoch, 13/);
  assert.doesNotMatch(JSON.stringify(s), /still training|so far/);
  assert.equal(cellDigits(s, s.rows[1], 4), 1, "84.9 as the entry gives it");
});

test("every top-k run of values rises from top 1 to top 10", () => {
  for (const s of TEST_SET_RESULTS) for (const r of s.rows) {
    const groups = new Map<string, number[]>();
    s.columns.forEach((c, k) => {
      const v = r.cells[k];
      if (c.group && v != null && (c.label === "Top 1" || /^\d+$/.test(c.label))) groups.set(c.group, [...(groups.get(c.group) ?? []), v]);
    });
    for (const [g, v] of groups) for (let i = 1; i < v.length; i++) assert.ok(v[i - 1] <= v[i], `${s.key} ${r.name} ${g}`);
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
  assert.match(t.records, /closes once it holds 3,000 records or on 30 November 2026, whichever comes first, with at least 2,000/);
  assert.match(t.records, /No numbers are published before it is opened/);
  assert.match(t.pending!.text, /No numbers are published before it is opened/);
});

test("the v1 benchmark's split starts at the release's own cutoff", () => {
  assert.match(BENCHMARKS[0].records, /eight weeks after 17 August 2026/);
});

test("every results table says what its values are a percentage of (Steve, 2026-10-26)", () => {
  for (const s of TEST_SET_RESULTS) {
    for (const c of s.columns) assert.match(c.group ?? "", /(^|: )% of (images|observations|(\d[\d,]* )?finds)$/, `${s.key}: ${c.group}`);
  }
  assert.match(valueCaption(set("fungitastic-closed-ft-train")), /^Each value is the % of test images \(or, under “% of observations”, of observations/);
  assert.match(valueCaption(set("v1-val-as-released")), /^Each value is the % of finds whose true species/);
  assert.doesNotMatch(JSON.stringify(TEST_SET_RESULTS.map((s) => s.columns)), /"Per image"|"Per observation"/);
});

test("the MycoMap-reference rows say which test images they cover, and that a species the reference lacks is a miss", () => {
  for (const k of ["fungitastic-closed-mycomap-ref", "fungitastic-dna-mycomap-ref"]) {
    for (const r of set(k).rows) {
      assert.match(r.detail!, /^(only the [\d,]+ test images .* of the [\d,]+ species its reference holds|every test image, all [\d,]+ species.*a species its reference lacks counts as a miss)/, `${k}: ${r.detail}`);
    }
  }
  assert.equal(set("fungitastic-closed-mycomap-ref").rows[0].detail,
    "only the 54,841 test images (29,558 observations) of the 979 species its reference holds · reference: open photos, 16,153 species");
});

test("every BioCLIP 2 base-model row states its image size: 336 px in E33", () => {
  const rows = TEST_SET_RESULTS.flatMap((s) => s.rows).filter((r) => r.name === "BioCLIP 2 (base model, not fine-tuned)");
  assert.equal(rows.length, 4);
  for (const r of rows) assert.match(r.detail ?? "", /^336 px · /, r.detail);
});

test("paired differences say 'at top 1' only when every column is top 1", () => {
  assert.equal(diffsHeading(set("fungitastic-closed-ft-train")), "Paired differences at top 1, in points, with 95% intervals");
  assert.equal(diffsHeading(set("v1-val-image-size")), "Paired differences at top 1, in points, with 95% intervals");
  assert.equal(diffsHeading(set("v1-val-open-photos")), "Paired differences in points, with 95% intervals");
  assert.doesNotMatch(set("v1-val-open-photos").notes.join(" "), /scored the same way/, "E4 and the Vision rows answer differently");
});

test("MycoMap v1 validation, 224 px against 336 px: v1-image-size's paired step from v0 to v1", () => {
  const s = set("v1-val-image-size");
  assert.deepEqual(table("v1-val-image-size"), [
    ["MycoMap Vision v0", "224 px, development · closest specimens + species average, by specimen", 54.19],
    ["MycoMap Vision v1", "336 px · closest specimens + species average, by specimen", 56.32],
  ]);
  assert.deepEqual(diffs("v1-val-image-size"), [["MycoMap Vision v1 − MycoMap Vision v0", [2.13, 1.52, 2.75]]]);
  assert.ok(Math.abs(56.32 - 54.19 - s.diffs!.rows[0].cells[0].d) < 0.005);
  assert.match(s.notes.join(" "), /1\.73, 1\.38 and 1\.04 points/);
  assert.match(s.notes.join(" "), /\+0\.11 \[−0\.24, \+0\.46\]/);
  assert.doesNotMatch(JSON.stringify(s.columns), /Genus/, "its genus is scored by exact string, unlike the card's other genus figures");
});

test("the sealed test's timetable is the plan's (O6): frozen first, closed at 3,000 records or on 30 November", () => {
  const t = BENCHMARKS.find((b) => b.slug === "sealed-test")!;
  assert.deepEqual(SEALED_TEST, { freeze: "around the end of October 2026", target: 3000, floor: 2000, closeBy: "2026-11-30" });
  for (const x of [t.pending!.text, t.records, ...t.protocol.map((p) => p.text)]) {
    assert.doesNotMatch(x, /26 October|~26 Oct|frozen around 30 October/, x);
  }
  assert.match(t.pending!.text, /^The analysis is frozen around the end of October 2026; the set closes once it holds 3,000 records or on 30 November 2026/);
  assert.equal(t.size, "2,000–3,000 records");
});

test("Poisonous mistakes is the plan's E21: mycomap.org's 60 deadly names, both directions, on the sealed test, no cost weight", () => {
  const b = BENCHMARKS.find((x) => x.slug === "poisonous-mistakes")!;
  const all = [b.summary, b.records, b.scoring, b.pending!.text, ...b.protocol.map((p) => p.text)].join(" ");
  assert.deepEqual(b.data, ["sealed"]);
  assert.equal(b.size, "60 deadly names");
  assert.match(all, /60 names on mycomap\.org's list of deadly fungi/);
  assert.doesNotMatch(all, /100 times|cost-weighted|Phalloideae|Orellani|Gyromitra/);
  assert.match(b.scoring, /strictly and sensu lato/);
});

test("Unknown species is partly measured: the v1 validation look named by its model, AUROC on the sealed test", () => {
  const b = BENCHMARKS.find((x) => x.slug === "unknown-species")!;
  assert.equal(b.status, "partial");
  assert.deepEqual(b.data, ["v1val", "sealed"]);
  assert.match(b.pending!.text, /MycoMap Vision v0, 224 px/);
  assert.match(b.pending!.text, /of 453 finds .* 11\.5% got a confident species answer, always wrong, and the genus was right for 58\.3%/);
  assert.match(b.pending!.text, /\(AUROC\) is scored on the sealed test/);
});

test("Advance predictions says it is paused, and why, while no records reach Vision", () => {
  const b = BENCHMARKS.find((x) => x.slug === "advance-predictions")!;
  assert.equal(b.status, "paused");
  assert.equal(STATUS_LABEL.paused, "Paused");
  assert.equal(ADVANCE_PAUSED_SINCE, "2026-10-19");
  assert.equal(b.pending!.lead, "Paused since 19 October 2026.");
  assert.match(b.pending!.text, /No records awaiting validation reach Vision from mycomap\.org/);
  assert.equal(b.lastRun, "paused since 19 October 2026");
});

test("the Results blurb names MycoMap Vision and the published models, not 'ours'", () => {
  const all = JSON.stringify(researchBar());
  assert.doesNotMatch(all, /\bours?\b/i);
  assert.match(all, /Every approach, MycoMap Vision and published models, scored on the same DNA-verified records\./);
});
