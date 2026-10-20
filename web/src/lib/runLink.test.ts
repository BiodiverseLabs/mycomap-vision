import assert from "node:assert/strict";
import { test } from "node:test";
import { isRunPath, keptLine, runIdFrom, runPath, runPhotos } from "./runLink";
import { mustSignIn } from "./research";

const ID = "Hql9fOB7-987lWJy";

test("a run's link is /r/ and its 16-character id, and nothing else counts as one", () => {
  assert.equal(runPath(ID), `/r/${ID}`);
  assert.equal(runIdFrom(`/r/${ID}`), ID);
  for (const path of ["/r/short", `/r/${ID}/x`, `/r/${ID}x`, "/r/", "/", `/x/r/${ID}`, `/r/${ID.slice(0, 15)}!`]) {
    assert.equal(isRunPath(path), false, path);
  }
});

test("a signed-out visitor opening a run's link is not sent to sign in; elsewhere they still are", () => {
  const visitor = { signin: "all" as const, user: null };
  assert.equal(mustSignIn(visitor, `/r/${ID}`), false);
  assert.equal(mustSignIn(visitor, "/"), true);
  assert.equal(mustSignIn(visitor, `/r/${ID}/more`), true);
});

test("an upload's run shows its own copies; an observation's shows the source's photos", () => {
  assert.deepEqual(runPhotos({ id: ID, kind: "photos", photos: 2 }),
                   [`/api/runs/${ID}/photos/0.jpg`, `/api/runs/${ID}/photos/1.jpg`]);
  assert.deepEqual(runPhotos({ id: ID, kind: "observation", photos: 0,
                               observation: { photos_shown: ["https://x/1.jpg", null] } }),
                   ["https://x/1.jpg", ""]);
});

test("the page says plainly that an upload's photos are kept, and never an expiry", () => {
  assert.match(keptLine("photos"), /keeps it, with smaller copies of your photos/);
  assert.match(keptLine("photos"), /until you delete it/);
  assert.match(keptLine("observation"), /Only people you give the link to/);
  for (const k of ["photos", "observation"] as const) assert.doesNotMatch(keptLine(k), /months|expire|until \d/);
});
