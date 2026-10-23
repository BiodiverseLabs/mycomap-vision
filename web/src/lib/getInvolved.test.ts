import { test } from "node:test";
import assert from "node:assert/strict";
import {
  apiCard, depthBars, firstClause, GET_INVOLVED_FEEDS, projectDots, reviewGroups, singleRecordNames,
} from "./getInvolved";
import type { FeedbackSummary, HowGood, Stats } from "./api";
import type { Outline } from "./apiOutline";

const q = (type: string, section: string, open: number, title = type, about = ""): FeedbackSummary => ({
  type, title, about, granularity: "record", model_flagged: false, section, blind: false, open, decided: 99,
});

/** Shaped like the box's summary on 22 October 2026, signed out. */
const SUMMARY: FeedbackSummary[] = [
  q("name_writing_split", "cleanup", 0, "One name, several writings"),
  q("group_label", "validation", 0, "Group or complex labels"),
  q("name_retired", "cleanup", 0, "Names retired on MycoMap's synonym list"),
  q("name_inactive_on_inat", "cleanup", 0, "Names iNaturalist no longer uses"),
  q("crosswalk", "validation", 0, "Provisional code and formal name: one lineage?"),
  q("label_vs_own_dna", "validation", 0, "The record's own DNA names another taxon"),
  q("photos_unlike_label", "validation", 0, "Photos unlike the label"),
  q("vision_names_same_taxon", "validation", 0, "Vision may name the same taxon"),
  q("reference_label_doubt", "validation", 22, "Reference records in doubt"),
  q("non_specimen_photo", "cleanup", 19383, "Photos that show no specimen"),
  q("twin_location_conflict", "cleanup", 0, "One find, two places"),
  q("photo_view", "views", 4984, "Photo views",
    "What each photo shows: cap top, gills, stem and base, growing in place, several specimens… The photo-type tagger suggests tags."),
  q("species_rank_split", "cleanup", 0, "A species and its own varieties"),
  q("crosspost_twin", "cleanup", 0, "One find posted to both iNaturalist and Mushroom Observer"),
];

test("the page reads only the four feeds that answer signed out", () => {
  assert.deepEqual([...GET_INVOLVED_FEEDS],
                   ["/api/stats", "/api/how-good", "/api/feedback/summary", "/api/docs/v1.json"]);
});

test("the banner's share is the names resting on one record, or nothing", () => {
  const s = { names: 19467, names_by_records: { "1": 8131, "2": 2603 } } as unknown as Stats;
  const one = singleRecordNames(s)!;
  assert.deepEqual([one.n, one.names, Math.round(one.share * 100)], [8131, 19467, 42]);
  assert.equal(singleRecordNames(undefined), null);
  assert.equal(singleRecordNames({ names: 19467, names_by_records: { "2": 5 } } as unknown as Stats), null);
  assert.equal(singleRecordNames({ names: 0, names_by_records: { "1": 5 } } as unknown as Stats), null);
});

test("the depth chart has a row only for each band the run measured", () => {
  const h = { run: null, species_by_depth: [
    { band: "0", n: 649, top1: 0, top3: 0, top5: 0, top10: 0 },
    { band: "1-4", n: 1391, top1: 0.199, top3: null, top5: null, top10: null },
    { band: "5-19", n: 2864, top1: null, top3: null, top5: null, top10: null },
  ] } as HowGood;
  assert.deepEqual(depthBars(h), [
    { band: "0", label: "0", none: true, top1: 0, n: 649 },
    { band: "1-4", label: "1–4", none: false, top1: 0.199, n: 1391 },
  ]);
  assert.equal(depthBars({ run: null }), null);
  assert.equal(depthBars(undefined), null);
});

test("one dot per project, and one per few only past the grid's size", () => {
  assert.deepEqual(projectDots(836), { dots: 836, per: 1 });
  assert.deepEqual(projectDots(1200), { dots: 1200, per: 1 });
  assert.deepEqual(projectDots(2401), { dots: 801, per: 3 });
  for (const none of [0, -3, null, undefined, NaN]) assert.equal(projectDots(none), null);
});

test("review groups come in the page's order with their open counts and never a decided count", () => {
  const g = reviewGroups(SUMMARY)!;
  assert.deepEqual(g.map((x) => [x.title, x.queues, x.open]),
                   [["Validation", 6, 22], ["Cleanup", 7, 19383], ["Photo views", 1, 4984]]);
  assert.deepEqual(g[0].lines, ["The record's own DNA names another taxon", "Photos unlike the label",
                                "Reference records in doubt"]);
  assert.equal(g[0].more, 3);
  assert.deepEqual(g[1].lines, ["Photos that show no specimen",
                                "One find posted to both iNaturalist and Mushroom Observer",
                                "Names iNaturalist no longer uses"]);
  assert.equal(g[1].more, 4);
  assert.deepEqual(g[2].lines, ["What each photo shows: cap top, gills, stem and base…"], "a lone queue says what it holds");
  assert.equal(g[2].more, 0);
  assert.ok(!JSON.stringify(g).includes("99"), "no decided count");
  assert.equal(reviewGroups([]), null);
  assert.equal(reviewGroups(undefined), null);
});

test("queues of a section the page doesn't know come last, under their own name", () => {
  const g = reviewGroups([q("new_kind", "atlas", 3, "A new queue"), ...SUMMARY.slice(0, 2)])!;
  assert.deepEqual(g.map((x) => x.code), ["validation", "cleanup", "other"]);
  assert.equal(g[2].title, "Other");
  assert.deepEqual(g[2].lines, ["A new queue"], "no description: its title");
});

test("a description is cut to its first clause", () => {
  assert.equal(firstClause("Short and done. More after."), "Short and done");
  assert.equal(firstClause("No full stop at all"), "No full stop at all");
  assert.equal(firstClause("Ends with a trail…"), "Ends with a trail…");
  assert.equal(firstClause("a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w."),
               "a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t…");
});

const ep = (method: string, path: string, status: "available" | "planned" = "available", summary = "x") =>
  ({ type: "endpoint" as const, method, path, status, summary });
const OUTLINE: Outline = {
  title: "MycoMap Vision API", version: "v1", base_url: "https://vision.mycomap.org/api/v1",
  page_url: "", summary: "", sections: [
    { id: "identify", title: "Identify", blocks: [
      { type: "p", text: "Base URL" },
      ep("POST", "/identify", "available", "Identify one find from 1–10 photos."),
      ep("POST", "/identify/inat/{observation_id}"), ep("POST", "/identify/mo/{observation_id}"),
      ep("GET", "/answers/{inat|mo}/{observation_id}")] },
    { id: "sessions", title: "Sessions", blocks: [ep("POST", "/sessions"), ep("POST", "/sessions/{session_id}/photos"),
      ep("PATCH", "/sessions/{session_id}"), ep("GET", "/sessions/{session_id}"),
      ep("DELETE", "/sessions/{session_id}/photos/{photo_ref}"), ep("DELETE", "/sessions/{session_id}")] },
    { id: "reference", title: "Reference", blocks: [ep("GET", "/taxon-counts"), ep("GET", "/models"), ep("GET", "/usage"),
      ep("GET", "/someday", "planned")] },
  ],
};

test("the developer card counts the endpoints that answer now, from the outline", () => {
  const c = apiCard(OUTLINE)!;
  assert.equal(c.available, 13, "planned ones are not counted");
  assert.equal(c.identifyUrl, "https://vision.mycomap.org/api/v1/identify");
  assert.deepEqual(c.highlights.map((h) => `${h.method} ${h.text}`), [
    "POST Identify one find from 1–10 photos",
    "POST Identify an iNaturalist or Mushroom Observer observation",
    "POST Observation sessions, for apps that add photos as they're taken",
    "GET Reference records per species, genus and family",
  ]);
  assert.equal(apiCard(undefined), null);
});

test("an endpoint the outline no longer lists as available is not named", () => {
  const less: Outline = { ...OUTLINE, sections: [
    { id: "identify", title: "Identify", blocks: [ep("POST", "/identify", "planned"),
                                                  ep("POST", "/identify/inat/{observation_id}")] }] };
  const c = apiCard(less)!;
  assert.equal(c.available, 1);
  assert.equal(c.identifyUrl, null);
  assert.deepEqual(c.highlights, [], "iNaturalist alone is not 'iNaturalist or Mushroom Observer'");
});
