import { test } from "node:test";
import assert from "node:assert/strict";
import { ApiError, type FeedbackItem, type FeedbackOption } from "./api";
import { appliesWords, blastAnchor, statCardLines, decisionBody, evidenceRows, feedbackAccess, formatTop5, groupBySection,
         inatLinks, isNameItem, noProposal, roleText, usableOptions,
         historyOrgText, itemBadges, linkedItem, openQueue, orgChipClass, pageText, parsePhotoKeys, prefillValue,
         queueParam, QUEUE_STORAGE_KEY, readLastQueue, writeLastQueue,
         recordLabel, refreshVerdict, REVEAL_BUTTON, REVEAL_CONFIRM, REVEAL_SEEN, revealNote, revealProblem,
         shownRule, showsFeedbackNav, stepLook, stepText, storedVerdict, valueBox, whenText } from "./feedback";

const item = (over: Partial<FeedbackItem> = {}): FeedbackItem => ({
  id: 1, item_type: "photos_unlike_label", subject: "123", granularity: "record",
  rule: "Vision ranks the label 40th", model_flagged: true, record_keys: ["123"],
  links: ["https://www.inaturalist.org/observations/123"], current: "Amanita muscaria",
  proposed: "", evidence: {}, projects: [], records: 1, status: "open", created_at: 0, ...over,
});

const opt = (over: Partial<FeedbackOption> = {}): FeedbackOption =>
  ({ code: "keep", label: "Keep as it is", applies: "note", needs_value: false, ...over });

test("staff get the admin BLAST form, in a new tab", () => {
  const a = blastAnchor({ blast_link: { href: "https://mycomap.org/admin/blast-results/1/2", kind: "admin" } });
  assert.deepEqual(a, { href: "https://mycomap.org/admin/blast-results/1/2", label: "BLAST results (admin)",
                        target: "_blank", rel: "noopener" });
});

test("everyone else gets the public BLAST page, never labelled admin", () => {
  const a = blastAnchor({ blast_link: { href: "https://mycomap.org/mycoblast/77", kind: "public" } });
  assert.equal(a?.href, "https://mycomap.org/mycoblast/77");
  assert.equal(a?.target, "_blank");
  assert.equal(a?.rel, "noopener");
  assert.doesNotMatch(a!.label, /admin/i);
});

test("no BLAST link from the server: nothing is shown (the admin address is never built here)", () => {
  assert.equal(blastAnchor({ blast_link: null }), null);
  assert.equal(blastAnchor({}), null);
  assert.equal(blastAnchor({ blast_link: { href: "", kind: "admin" } }), null);
});

test("the menu offers Human Feedback only to signed-in people", () => {
  assert.equal(showsFeedbackNav({ signin: "all", user: { id: "1", name: "A" } }), true);
  assert.equal(showsFeedbackNav({ signin: "all", user: null }), false);
  assert.equal(showsFeedbackNav({ signin: "off", user: null }), false);
  assert.equal(showsFeedbackNav(undefined), false);
});

test("401 asks to sign in, 403 says who the page is for", () => {
  assert.equal(feedbackAccess(new ApiError(401, "x")), "signin");
  assert.equal(feedbackAccess(new ApiError(403, "x")), "forbidden");
  assert.equal(feedbackAccess(new ApiError(500, "x")), "error");
  assert.equal(feedbackAccess(null), null);
});

test("a check-sample item looks exactly like the other model-flagged items while open", () => {
  const flagged = item();
  const sample = item({ blind: true, hidden_until_decided: true, seed: "7",
                        rule: "blind sample of finds Vision answered right (seed 7)" });
  assert.deepEqual(itemBadges(sample), itemBadges(flagged));
  assert.equal(shownRule(sample), shownRule(flagged));
  assert.deepEqual(evidenceRows(sample), evidenceRows(flagged));
  const shown = JSON.stringify([itemBadges(sample), shownRule(sample), evidenceRows(sample)]);
  assert.doesNotMatch(shown, /blind|seed/i);
});

test("once decided the rule is shown, without the word blind or the seed", () => {
  const r = shownRule(item({ status: "decided", blind: true,
                             rule: "blind sample of finds Vision answered right (seed 7)" }));
  assert.equal(r, "check sample of finds Vision answered right");
  assert.equal(shownRule(item({ model_flagged: false })), "Vision ranks the label 40th");
});

test("evidence lists the label, the proposal and non-empty evidence, never analysis fields", () => {
  const rows = evidenceRows(item({ proposed: "Amanita persicina",
                                   evidence: { in_v1: "yes", v1_val_outcome: "", blind_seed: "3", n: 4 } }));
  assert.deepEqual(rows, [["Current", "Amanita muscaria"], ["Proposed", "Amanita persicina"],
                          ["In v1", "yes"], ["N", "4"]]);
});

test("a decision that needs a value refuses an empty one and sends the trimmed value", () => {
  const rename = opt({ code: "rename", needs_value: true, applies: "org" });
  assert.deepEqual(decisionBody(rename, { value: "  " }), { error: "Enter the value this decision needs." });
  assert.deepEqual(decisionBody(rename, { value: " Amanita persicina ", note: " seen " }),
                   { body: { decision: "rename", value: "Amanita persicina", note: "seen" } });
  assert.deepEqual(decisionBody(opt(), { value: "ignored" }), { body: { decision: "keep" } });
});

test("leaving photos out of Vision needs the photo keys, listed once each", () => {
  const ex = opt({ code: "exclude_photos", applies: "vision" });
  assert.deepEqual(decisionBody(ex, { photoKeys: " " }), { error: "List the photos to leave out." });
  assert.deepEqual(decisionBody(ex, { photoKeys: "123/1, 123/2,123/1" }),
                   { body: { decision: "exclude_photos", photo_keys: ["123/1", "123/2"] } });
  assert.deepEqual(parsePhotoKeys("a,, b c"), ["a", "b", "c"]);
});

test("the value box starts with the item's proposal for options that need a value", () => {
  assert.equal(prefillValue({ needs_value: true }, { proposed: "Laccaria bicolor" }), "Laccaria bicolor");
  assert.equal(prefillValue({ needs_value: true }, { proposed: null }), "");
  assert.equal(prefillValue({ needs_value: false }, { proposed: "x" }), "");
});

test("each option says where it takes effect", () => {
  assert.deepEqual(["org", "vision", "note"].map(appliesWords), ["on mycomap.org", "in Vision", "note only"]);
});

test("a refresh says still disagrees, now agrees, or that Vision can't re-check", () => {
  assert.equal(refreshVerdict({ still_disagrees: true, vision: { available: true } }).text, "Still disagrees");
  assert.equal(refreshVerdict({ still_disagrees: false, vision: { available: true } }).text, "Now agrees");
  assert.equal(refreshVerdict({ still_disagrees: null, vision: { available: false, reason: "no photos" } }).text,
               "Vision can't re-check this record (no photos)");
  assert.equal(refreshVerdict({ still_disagrees: null, vision: { available: true } }).tone, "unknown");
  assert.deepEqual([storedVerdict(1).tone, storedVerdict(0).tone, storedVerdict(null).tone],
                   ["disagrees", "agrees", "unknown"]);
});

test("records, times and pages read plainly", () => {
  assert.equal(recordLabel("123"), "iNat 123");
  assert.equal(recordLabel("mo:45"), "MO 45");
  assert.equal(whenText(0), "1970-01-01 00:00 UTC");
  assert.equal(whenText("2026-10-18T14:05:09Z"), "2026-10-18 14:05 UTC");
  assert.equal(pageText(25, 25, 312), "26–50 of 312");
  assert.equal(pageText(0, 0, 0), "0 of 0");
});

// --- whole-name decisions open a Name Update on mycomap.org ----------------------------------

test("a Name Update option shows the new name, editable, prefilled with the proposal", () => {
  const accept = opt({ code: "accept", label: "Use the accepted name for all", applies: "org",
                       name_update: "accepted_name" });
  const it = item({ proposed: " Amanita accepted " });
  assert.equal(prefillValue(accept, it), "Amanita accepted");
  assert.deepEqual(valueBox(accept), { label: "New name", required: false });
  assert.deepEqual(valueBox(opt({ needs_value: true, name_update: "respelling" })),
                   { label: "New name", required: true });
  assert.deepEqual(valueBox(opt({ needs_value: true })), { label: "Value", required: true });
  assert.equal(valueBox(opt()), null);
  assert.equal(prefillValue(opt(), it), "");
});

test("an edited new name is sent; an empty one leaves the server's proposal", () => {
  const accept = opt({ code: "accept", applies: "org", name_update: "accepted_name" });
  assert.deepEqual(decisionBody(accept, { value: " Amanita fixed " }),
                   { body: { decision: "accept", value: "Amanita fixed" } });
  assert.deepEqual(decisionBody(accept, { value: "  " }), { body: { decision: "accept" } });
  assert.deepEqual(decisionBody(opt({ code: "keep" }), { value: "x" }), { body: { decision: "keep" } });
});

test("the chip's tone picks its colours and the steps their marks", () => {
  assert.match(orgChipClass("good"), /myco-green/);
  assert.match(orgChipClass("bad"), /amber/);
  assert.match(orgChipClass("busy"), /sky/);
  assert.equal(stepLook("done").mark, "●");
  assert.equal(stepLook("todo").mark, "○");
  assert.match(stepLook("stopped").markClass, /amber/);
  assert.match(stepLook("current").textClass, /font-medium/);
});

test("Sent shows the time it was sent; other steps say what the server said", () => {
  const sentAt = Date.UTC(2026, 9, 18, 14, 1) / 1000;
  const t = stepText({ step: "sent", state: "done", label: "Sent" }, { sent_at: sentAt });
  assert.equal(t, "Sent 14:01 UTC");
  assert.equal(stepText({ step: "sent", state: "stopped", label: "Not sent yet" }, { sent_at: sentAt }),
               "Not sent yet");
  assert.equal(stepText({ step: "searched", state: "done", label: "Searched: 1 record" }, { sent_at: null }),
               "Searched: 1 record");
});

test("a link back from mycomap.org opens its item; anything else opens none", () => {
  assert.equal(linkedItem("?item=124"), 124);
  assert.equal(linkedItem("?tab=x&item=7"), 7);
  for (const s of ["", "?item=", "?item=0", "?item=-3", "?item=1e3", "?item=12a", "?item=9999999999"]) {
    assert.equal(linkedItem(s), null, s);
  }
});

test("History names the Name Update request and where it stands", () => {
  assert.equal(historyOrgText({ request_id: 7, chip: { text: "on mycomap.org: waiting for a person", tone: "busy" } }),
               ", Name Update request 7: waiting for a person");
  assert.equal(historyOrgText({ request_id: 7, chip: { text: "applied on mycomap.org · 2 renamed", tone: "good" } }),
               ", Name Update request 7: applied · 2 renamed");
  assert.equal(historyOrgText({ request_id: null, chip: { text: "not sent to mycomap.org yet", tone: "bad" } }),
               ", not sent yet");
  assert.equal(historyOrgText(null), "");
});

// --- the two sections --------------------------------------------------------------------------

const queue = (type: string, section: string | undefined, open: number) => ({ type, section, open, title: type });

test("the queues are shown in two sections, cleanup first, each in the summary's order with its open total", () => {
  const groups = groupBySection([
    queue("photos_unlike_label", "validation", 5), queue("non_specimen_photo", "cleanup", 10),
    queue("group_label", "validation", 0), queue("name_retired", "cleanup", 2),
  ]);
  assert.deepEqual(groups.map((g) => [g.code, g.title, g.open, g.types.map((t) => t.type)]), [
    ["cleanup", "Data cleanup", 12, ["non_specimen_photo", "name_retired"]],
    ["validation", "Data validation", 5, ["photos_unlike_label", "group_label"]],
  ]);
  assert.match(groups[0].intro, /^Fix what a rule found: photos that show no specimen/);
  assert.match(groups[1].intro, /reviewed blind: Vision's answer stays hidden unless you choose to see it\.$/);
});

test("a queue in a section the page doesn't know (or none) is kept, last, under Other", () => {
  const groups = groupBySection([queue("new_kind", "triage", 3), queue("crosswalk", "validation", 1),
                                 queue("unsorted", undefined, 4)]);
  assert.deepEqual(groups.map((g) => [g.code, g.title, g.open, g.types.map((t) => t.type)]), [
    ["validation", "Data validation", 1, ["crosswalk"]],
    ["other", "Other", 7, ["new_kind", "unsorted"]],
  ]);
  assert.deepEqual(groupBySection([]), []);
});

// --- "Show Vision's answer" ----------------------------------------------------------------

test("a revealed answer reads as Vision's top 5 in one line, scores to two places", () => {
  assert.equal(formatTop5({ available: true, reason: null,
                            top5: [["A x", 0.712], ["B y", 0.4], ["C z", 0.05], ["D w", 0.01], ["E v", 0], ["F u", 0]] }),
               "Vision's top 5: A x 0.71 · B y 0.40 · C z 0.05 · D w 0.01 · E v 0.00");
});

test("when Vision has no answer the line says why", () => {
  assert.equal(formatTop5({ available: false, top5: [], reason: "no photos Vision can read" }),
               "Vision has no answer for this record: no photos Vision can read");
  assert.equal(formatTop5({ available: false, top5: [], reason: null }), "Vision has no answer for this record.");
  assert.equal(formatTop5({ available: true, top5: [], reason: null }), "Vision has no answer for this record.");
  assert.equal(formatTop5(null), "Vision has no answer for this record.");
});

test("a seen answer carries the note that the decision is out of the blind check; an unseen one none", () => {
  assert.equal(revealNote({ at: 1, answer: { available: false, top5: [], reason: "x" } }),
               "Seen — this decision won't count in the blind check.");
  assert.equal(revealNote({ at: 1, answer: { available: true, top5: [["A x", 1]], reason: null } }), REVEAL_SEEN);
  assert.equal(revealNote(null), null);
  assert.equal(revealNote(undefined), null);
});

test("the reveal button and its confirm are fixed words, the same for every item", () => {
  assert.equal(REVEAL_BUTTON, "Show Vision's answer");
  assert.equal(REVEAL_CONFIRM, "Your decision on this record will be counted as made with Vision's answer seen. Show it?");
});

test("a refused reveal says the item has nothing hidden any more; other failures ask to try again", () => {
  assert.match(revealProblem(new ApiError(409, "nothing is hidden on this item")), /Nothing is hidden/);
  assert.equal(revealProblem(new ApiError(401, "Sign in")), "Sign in");
  assert.match(revealProblem(new Error("network")), /could not be shown/);
});

test("a photo queue's card counts photos and the observations they are in", () => {
  assert.deepEqual(statCardLines({ open: 14827, decided: 0, records_open: 8417, records_decided: 0 }),
                   { big: "14,827", unit: "photos", sub: "open in 8,417 observations · 0 decided" });
  assert.deepEqual(statCardLines({ open: 1, decided: 3, records_open: 1, records_decided: 2 }),
                   { big: "1", unit: "photo", sub: "open in 1 observation · 3 decided (2 observations done)" });
  assert.deepEqual(statCardLines({ open: 80, decided: 1 }),
                   { big: "80", unit: "", sub: "open · 1 decided" });
});

// --- which queue the page opens ----------------------------------------------------------------

test("the page opens no queue by itself: only a pick, a link, or the one last opened", () => {
  const types = ["non_specimen_photo", "photos_unlike_label", "group_label"];
  const base = { picked: null, param: null, linkedPending: false, linkedType: null, stored: null, types };
  assert.equal(openQueue(base), null, "first visit: none (not the first queue with open items)");
  assert.equal(openQueue({ ...base, stored: "group_label" }), "group_label");
  assert.equal(openQueue({ ...base, stored: "gone_queue" }), null, "a queue no longer listed");
  assert.equal(openQueue({ ...base, param: "photos_unlike_label", stored: "group_label" }), "photos_unlike_label");
  assert.equal(openQueue({ ...base, linkedType: "group_label", stored: "non_specimen_photo" }), "group_label");
  assert.equal(openQueue({ ...base, linkedPending: true, stored: "non_specimen_photo" }), null,
               "a linked item still loading: wait, don't open the remembered queue first");
  assert.equal(openQueue({ ...base, picked: "group_label", param: "non_specimen_photo" }), "group_label");
  assert.equal(openQueue({ ...base, stored: "group_label", types: [] }), null, "summary not loaded");
  assert.equal(openQueue({ ...base, stored: "group_label", types: undefined }), null);
});

test("the last queue is remembered per viewer; blocked storage remembers nothing", () => {
  const m = new Map<string, string>();
  const s = { getItem: (k: string) => m.get(k) ?? null, setItem: (k: string, v: string) => void m.set(k, v) };
  assert.equal(readLastQueue(() => s), null);
  writeLastQueue(() => s, "group_label");
  assert.equal(m.get(QUEUE_STORAGE_KEY), "group_label");
  assert.equal(readLastQueue(() => s), "group_label");
  const broken = () => { throw new Error("blocked"); };
  assert.equal(readLastQueue(broken), null);
  assert.doesNotThrow(() => writeLastQueue(broken, "x"));
  m.set(QUEUE_STORAGE_KEY, "<script>");
  assert.equal(readLastQueue(() => s), null);
  assert.equal(queueParam("?queue=non_specimen_photo&item=3"), "non_specimen_photo");
  assert.equal(queueParam("?queue=Bad Value"), null);
  assert.equal(queueParam(""), null);
});

// --- The names behind a name item (Steve, 2026-10-21) ---

const taxon = (id: number, name: string, active = true) =>
  ({ id, name, rank: "species", active, observations: 0, changes: 0, url: `https://www.inaturalist.org/taxa/${id}` });

test("an inactive iNat name with no current name links its taxon and iNat's other writing", () => {
  const links = inatLinks({ inat: { status: "inactive", asked: "Callistosporium luteoolivaceum",
    taxon: taxon(1287172, "Callistosporium luteoolivaceum", false), current: [],
    near: [taxon(118354, "Callistosporium luteo-olivaceum")], search_url: "s" } });
  assert.deepEqual(links, [
    { text: "iNat taxon 1287172 (inactive)", href: "https://www.inaturalist.org/taxa/1287172" },
    { text: "iNat writes it Callistosporium luteo-olivaceum", href: "https://www.inaturalist.org/taxa/118354", strong: true },
  ]);
});

test("an inactive name links the current name iNat points to", () => {
  const links = inatLinks({ inat: { status: "inactive", asked: "A b", taxon: taxon(5, "A b", false),
    current: [taxon(6, "A c")], near: [], search_url: "s" } });
  assert.equal(links[1].text, "iNat's current name: A c");
  assert.equal(links[1].href, "https://www.inaturalist.org/taxa/6");
  const chain = inatLinks({ inat: { status: "inactive", asked: "Boletus pallidoroseus", taxon: taxon(500028, "Boletus pallidoroseus", false),
    current: [taxon(1068083, "Lanmaoa pallidorosea", false)], near: [], search_url: "s" } });
  assert.equal(chain[1].text, "iNat's current name: Lanmaoa pallidorosea (also inactive)");
});

test("a name iNat has no taxon of, or not asked yet, links iNat's search", () => {
  for (const status of ["not_found", "not_asked", "unavailable"] as const) {
    const links = inatLinks({ inat: { status, asked: "A b", taxon: null, current: [], near: [],
      search_url: "https://www.inaturalist.org/taxa/search?q=A%20b" } });
    assert.deepEqual(links, [{ text: "Search iNaturalist", href: "https://www.inaturalist.org/taxa/search?q=A%20b" }]);
  }
});

test("roles read as where the name is used", () => {
  assert.equal(roleText(["records", "synonym list"]), "on the records · on MycoMap's synonym list");
});

test("a name item with no proposal offers no 'use iNat's name'; other items keep every option", () => {
  const opts = [opt({ code: "accept" }), opt({ code: "keep" }), opt({ code: "rename" })];
  const inactive = item({ item_type: "name_inactive_on_inat", proposed: "" });
  assert.equal(noProposal(inactive), true);
  assert.deepEqual(usableOptions(inactive, opts).map((o) => o.code), ["keep", "rename"]);
  const proposed = item({ item_type: "name_inactive_on_inat", proposed: "A c" });
  assert.deepEqual(usableOptions(proposed, opts).map((o) => o.code), ["accept", "keep", "rename"]);
  assert.equal(noProposal(item({ item_type: "group_label", proposed: "" })), false);
  assert.equal(isNameItem(item({ item_type: "name_retired" })), true);
});
