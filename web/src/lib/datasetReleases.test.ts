import { test } from "node:test";
import assert from "node:assert/strict";
import { DATASET_RELEASES, shortHash, V1 } from "./datasetReleases";

test("each release's counts add up, as its MANIFEST's do", () => {
  for (const r of DATASET_RELEASES) {
    assert.equal(r.records.train + r.records.val, r.records.included, `${r.id} splits`);
    assert.equal(r.records.bySource.inat + r.records.bySource.mo, r.records.included, `${r.id} sources`);
    const l = r.photos.byLicence;
    assert.equal(l.open + l.nc + l.arr, r.photos.included, `${r.id} licences`);
    assert.ok(r.records.included <= r.records.candidates);
    for (const h of [r.releaseHash, r.referenceHash]) assert.match(h, /^[0-9a-f]{64}$/);
    assert.match(r.codeCommit, /^[0-9a-f]{40}$/);
    assert.ok(r.labelSnapshot <= r.created && r.valCutoff < r.labelSnapshot);
  }
});

test("v1 is the release the paper's numbers come from", () => {
  assert.equal(V1.id, "v1");
  assert.equal(shortHash(V1.releaseHash), "f99777a694f2");
  assert.equal(V1.valCutoff, "2026-08-17");
});
