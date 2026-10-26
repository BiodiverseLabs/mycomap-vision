import { test } from "node:test";
import assert from "node:assert/strict";
import { OPEN_REFERENCE, openModelHeadline, pointsDiff, REFERENCE_LABEL, twoModelLabel, twoModelTip,
         TWO_MODELS } from "./twoModels";

test("the two models are MycoMap Vision v1 and v1-open, with their training data; the size in the tooltip", () => {
  assert.deepEqual(TWO_MODELS.models.map(twoModelLabel), [
    "MycoMap Vision v1 (public + non-public training data)",
    "MycoMap Vision v1-open (open data only)",
  ]);
  assert.deepEqual(TWO_MODELS.models.map(twoModelTip), [
    "336 px · bioclip-2-336-ft-20261018-171420",
    "336 px · bioclip-2-336-ft-20261020-173106",
  ]);
});

test("the shipped figures hang together: same finds, depth bands that add up, the paired difference", () => {
  const [site, open] = TWO_MODELS.models;
  assert.equal(site.species.n, open.species.n);
  assert.equal(site.species.n, TWO_MODELS.paired.n);
  for (const m of TWO_MODELS.models) {
    assert.equal(Object.values(m.depth).reduce((s, d) => s + d.n, 0), m.species.n, m.backbone);
    for (const t of [m.species, m.genus, ...Object.values(m.depth)]) {
      assert.ok(t.top1 <= t.top3 && t.top3 <= t.top5 && t.top5 <= t.top10, m.backbone);
    }
  }
  assert.equal(Math.round((open.species.top1 - site.species.top1) * 1e4) / 1e4, TWO_MODELS.paired.diff);
  const [lo, hi] = TWO_MODELS.paired.ci95;
  assert.ok(lo <= TWO_MODELS.paired.diff && TWO_MODELS.paired.diff <= hi);
});

test("a paired difference reads in points with its interval", () => {
  assert.equal(pointsDiff(-0.0039, [-0.0077, -0.0001]), "−0.39 [−0.77, −0.01]");
  assert.equal(pointsDiff(0.0213, [0.0152, 0.0275]), "+2.13 [+1.52, +2.75]");
});

test("v1-open has two numbers, each labelled with its reference: 55.3% open-only (the paper's), 55.9% full", () => {
  const h = openModelHeadline();
  assert.equal((100 * h.open.top1).toFixed(1), "55.3");
  assert.equal((100 * h.full.top1).toFixed(1), "55.9");
  assert.equal(h.open.n, h.full.n, "the same finds");
  assert.match(REFERENCE_LABEL.open, /open-data reference only/);
  assert.match(REFERENCE_LABEL.open, /headline/);
  assert.match(REFERENCE_LABEL.full, /full reference/);
  assert.ok(OPEN_REFERENCE.referenceRecords < TWO_MODELS.referenceRecords);
  for (const t of [...Object.values(OPEN_REFERENCE.species), ...Object.values(OPEN_REFERENCE.genus)]) {
    assert.ok(t.top1 <= t.top3 && t.top3 <= t.top5 && t.top5 <= t.top10);
  }
  const [site, open] = TWO_MODELS.models.map((m) => OPEN_REFERENCE.species[m.backbone]);
  assert.equal(Math.round((open.top1 - site.top1) * 1e4) / 1e4, OPEN_REFERENCE.paired.diff);
});
