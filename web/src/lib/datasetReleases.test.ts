import { test } from "node:test";
import assert from "node:assert/strict";
import { DATASET_RELEASES, ERRATUM_REASON, erratumReasons, shortHash, utcDay, utcMinute, V1 } from "./datasetReleases";

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

test("an erratum's reasons add up to its records, each reason in plain words", () => {
  for (const r of DATASET_RELEASES) {
    for (const e of [r.erratum, ...r.erratumHistory]) {
      if (!e) continue;
      const parts = Object.entries(e.byReason);
      assert.equal(parts.reduce((n, [, v]) => n + v, 0), e.records, `${r.id} erratum ${e.id}`);
      for (const [why] of parts) assert.ok(ERRATUM_REASON[why as keyof typeof ERRATUM_REASON], why);
      assert.ok(e.records < r.records.included);
    }
  }
});

test("v1 is scored with erratum r4, the one every published v1 run uses; r3 stays as history", () => {
  assert.deepEqual(V1.erratum, { id: "r4", records: 650,
                                 byReason: { crosspost_twin: 475, mo_twin_as_inat: 72, nonfungal_label: 103 } });
  assert.deepEqual(V1.erratumHistory.map((e) => [e.id, e.records]), [["r3", 175]]);
  assert.match(erratumReasons(V1.erratum!), /^475 cross-posted twins .*, 72 Mushroom Observer finds held as iNaturalist records, 103 labels that name no fungus$/);
  assert.equal(erratumReasons(V1.erratumHistory[0]),
    "72 Mushroom Observer finds held as iNaturalist records, 103 labels that name no fungus", "a reason with no records is left out");
});

test("release dates read in UTC, whatever the reader's time zone", () => {
  assert.equal(utcDay(V1.valCutoff), "17 August 2026");
  assert.equal(utcMinute(V1.labelSnapshot), "12 October 2026, 13:17 UTC");
  assert.equal(utcDay("2026-10-12T23:59:00Z"), "12 October 2026");
});
