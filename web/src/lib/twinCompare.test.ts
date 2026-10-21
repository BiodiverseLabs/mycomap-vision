import { test } from "node:test";
import assert from "node:assert/strict";
import type { FeedbackOption } from "./api";
import { distanceText, optionGroups, pairSides, placeLine, suspectSite } from "./twinCompare";

const photo = (key: string) => ({ photo_key: key, position: 1, shown: "public" as const,
                                  thumb: `https://x/${key}.jpg`, full: `https://x/${key}-l.jpg` });

const pair = (over = {}) => ({
  record_keys: ["172307494", "mo:324331"],
  links: ["https://www.inaturalist.org/observations/172307494", "https://mushroomobserver.org/obs/324331"],
  evidence: { distance_km: "9336", mo_placement: "MO GPS", inat_accuracy_m: "20",
              likely_wrong_copy: "MO copy: named location" } as Record<string, unknown>,
  live: [{ record_key: "mo:324331", name: "Xylodon spathulatus" }, { record_key: "172307494", unavailable: true }],
  photos: { "mo:324331": [photo("mo:1"), photo("mo:2")], "172307494": [] } as Record<string, ReturnType<typeof photo>[]> | undefined,
  ...over,
});

test("the MO copy is always the left column, each with its own link, photos and placing", () => {
  const [mo, inat] = pairSides(pair());
  assert.equal(mo.site, "MO");
  assert.equal(mo.link, "https://mushroomobserver.org/obs/324331");
  assert.deepEqual(mo.photos.map((p) => p.photo_key), ["mo:1", "mo:2"]);
  assert.equal(mo.place, "MO GPS");
  assert.equal(inat.site, "iNat");
  assert.equal(inat.link, "https://www.inaturalist.org/observations/172307494");
  assert.deepEqual(inat.photos, []);
  assert.equal(inat.place, "accuracy 20 m");
});

test("the label now comes from mycomap.org only when it answered for that record", () => {
  const [mo, inat] = pairSides(pair());
  assert.equal(mo.nameNow, "Xylodon spathulatus");
  assert.equal(inat.nameNow, null);
});

test("the copy the rule suspects is marked, and neither when it is unclear", () => {
  assert.deepEqual(pairSides(pair()).map((s) => s.suspect), [true, false]);
  const unclear = pairSides(pair({ evidence: { likely_wrong_copy: "unclear: both look precise" } }));
  assert.deepEqual(unclear.map((s) => s.suspect), [false, false]);
  assert.equal(suspectSite("iNat copy: accuracy 5000 m"), "iNat");
  assert.equal(suspectSite(undefined), null);
});

test("an item without photos has no columns (not a pair, or an older server)", () => {
  assert.deepEqual(pairSides(pair({ photos: undefined })), []);
});

test("an iNat copy with no accuracy says so in words", () => {
  const [, inat] = pairSides(pair({ evidence: { inat_accuracy_m: "not recorded" } }));
  assert.equal(inat.place, "accuracy not recorded");
});

test("each copy's place reads as its site lists it, or says why there is none", () => {
  assert.deepEqual(placeLine({ site: "MO", text: "Seattle, Washington, USA", status: "ok" }, "MO", false),
                   { text: "Seattle, Washington, USA", muted: false });
  assert.equal(placeLine({ site: "iNat", text: null, status: "none" }, "iNat", false).text, "no place listed on iNat");
  assert.equal(placeLine({ site: "MO", text: null, status: "unavailable" }, "MO", false).text, "MO did not answer just now");
  assert.equal(placeLine(undefined, "iNat", true).text, "asking iNat…");
  assert.equal(placeLine(undefined, "iNat", false).text, "–");
});

test("the distance reads as km apart, with thousands separated", () => {
  assert.equal(distanceText("9336"), "9,336 km apart");
  assert.equal(distanceText("7.4"), "7.4 km apart");
  assert.equal(distanceText(""), "");
  assert.equal(distanceText(undefined), "");
});

const opt = (code: string, group: string | null = null): FeedbackOption =>
  ({ code, label: code, applies: "org", needs_value: false, group });

test("Use iNat and Use MO sit under one Duplicate observation heading; Not duplicate under its own", () => {
  const g = optionGroups([opt("fix_mo_location", "Duplicate observation"),
                          opt("fix_inat_location", "Duplicate observation"), opt("both_right", "Not duplicate")]);
  assert.deepEqual(g.map((x) => [x.group, x.options.map((o) => o.code)]),
                   [["Duplicate observation", ["fix_mo_location", "fix_inat_location"]], ["Not duplicate", ["both_right"]]]);
});

test("options without groups stay one per block, in order", () => {
  assert.deepEqual(optionGroups([opt("a"), opt("b")]).map((x) => x.options.map((o) => o.code)), [["a"], ["b"]]);
});
