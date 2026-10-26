import { test } from "node:test";
import assert from "node:assert/strict";
import { homeStage, topKTable, whyNumbers } from "./identifyHome";
import type { HowGood, Stats } from "./api";

const STATS = {
  records: 182926, records_north_america: 177390, label_conflicts: 0, inat_ok: 1, inat_missing: 0,
  photos: 700746, photos_by_license: {}, embedded: { "served-model": 687249 },
  permissions: { last_good_sync: null, last_attempt: null, fresh_for_showing: true },
  names: 19483, names_provisional: 11044, names_by_records: {},
  recent_week: { records: 365, from: "2026-10-19", through: "2026-10-25" },
} as Stats;

const tk = (n: number, top1: number, top3: number, top5: number, top10: number) => ({ n, top1, top3, top5, top10 });
const HOW_GOOD: HowGood = {
  run: { id: 169, backbone: "served-model", method: "m", cutoff: "2026-08-17", test_days: 0, n_test: 12874,
         n_reference: 155648 },
  top_k: { k: [1, 3, 5, 10], species: tk(12798, 0.5917, 0.7528, 0.8015, 0.8513),
           genus: tk(12865, 0.8575, 0.9389, 0.9569, 0.9733), family: tk(12759, 0.9059, 0.9628, 0.9755, 0.9853) },
};

test("the page is empty until photos are chosen", () => {
  const none = { photos: 0, pending: false, answered: false };
  assert.equal(homeStage(none), "empty", "a link typed in the box keeps the page as it is");
  assert.equal(homeStage({ ...none, photos: 2 }), "photos");
});

test("identifying or answered, the answer is the page; an error stays where it was asked", () => {
  const none = { photos: 3, pending: false, answered: false };
  assert.equal(homeStage({ ...none, pending: true }), "answer", "the waiting card takes the answer's place");
  assert.equal(homeStage({ ...none, answered: true }), "answer");
  assert.equal(homeStage({ ...none, photos: 0, answered: true }), "answer", "an answer from a link has no photos");
  assert.equal(homeStage(none), "photos", "a failed run leaves the photos and the message");
});

test("benchmark numbers are top-1, 3, 5 and 10, finest rank first, from the served model's test", () => {
  const rows = topKTable(HOW_GOOD)!;
  assert.deepEqual(rows.map((r) => r.rank), ["species", "genus", "family"]);
  assert.deepEqual(rows[0].cells, [{ k: 1, value: "59.2%" }, { k: 3, value: "75.3%" }, { k: 5, value: "80.2%" },
                                   { k: 10, value: "85.1%" }]);
  assert.equal(rows[2].cells[3].value, "98.5%");
});

test("a rank or a k the report lacks is left out, and no report shows no table", () => {
  assert.equal(topKTable(undefined), null);
  assert.equal(topKTable({ run: null }), null, "an older server without top_k");
  const part = topKTable({ ...HOW_GOOD, top_k: { k: [1, 3], species: { n: 5, top1: 0.5, top3: null, top5: null, top10: null },
                                                 genus: null } })!;
  assert.deepEqual(part, [{ rank: "species", cells: [{ k: 1, value: "50.0%" }] }]);
});

test("the empty page's numbers are the live ones", () => {
  const w = whyNumbers(STATS, HOW_GOOD);
  assert.deepEqual(w.references, {
    records: "177,390", photos: "687,249",
    week: "365 North American records were DNA-verified in the week to 25 October 2026.",
  });
  assert.equal(w.provisional?.names, "11,044");
  assert.equal(w.provisional?.total, "19,483");
  assert.equal(w.provisional?.sharePct, "57%");
  assert.equal(w.tested?.records, "12,874");
  assert.equal(w.tested?.after, "17 August 2026");
});

test("a number the server doesn't send is left out, never made up", () => {
  const w = whyNumbers(undefined, undefined);
  assert.deepEqual(w, { references: null, provisional: null, tested: null }, "still loading, or the API is down");
  const bare = whyNumbers({ ...STATS, names_provisional: undefined, recent_week: null, embedded: {} }, { run: null });
  assert.equal(bare.provisional, null, "an older server without provisional names");
  assert.equal(bare.references?.week, null, "no week counted");
  assert.equal(bare.references?.photos, null, "no model's photos embedded");
  assert.equal(bare.tested, null, "no test run to report");
  assert.equal(whyNumbers({ ...STATS, records_north_america: 0 }, null).references, null, "never a zero as a headline");
});
