import { test } from "node:test";
import assert from "node:assert/strict";

import { EMPTY_ANSWER, type SavedResponse } from "./logic";
import { Saver, type Body, type SaveState } from "./saver";

/** Timers run by hand, and a server whose answers the test settles. */
function rig() {
  let clock = 0;
  const timers: { at: number; fn: () => void; id: number }[] = [];
  let nextId = 1;
  const posts: { code: string; body: Body; settle: (ok: boolean) => void }[] = [];
  const states: SaveState[] = [];
  const saver = new Saver({
    post: (code, body) => new Promise<SavedResponse>((resolve, reject) => {
      posts.push({
        code, body,
        settle: (ok) => (ok ? resolve({ ...body, better_name: body.better_name || null, notes: body.notes || null, saved_at: "2026-11-16T21:14:00Z" }) : reject(new Error("offline"))),
      });
    }),
    onState: (s) => states.push(s),
    schedule: (fn, ms) => { const id = nextId++; timers.push({ at: clock + ms, fn, id }); return id; },
    cancel: (id) => { const i = timers.findIndex((t) => t.id === id); if (i >= 0) timers.splice(i, 1); },
    now: () => "client-time",
  });
  const advance = async (ms: number) => {
    clock += ms;
    for (;;) {
      const due = timers.filter((t) => t.at <= clock).sort((a, b) => a.at - b.at)[0];
      if (!due) break;
      timers.splice(timers.indexOf(due), 1);
      due.fn();
    }
    await Promise.resolve();
  };
  const settle = async (ok: boolean) => {
    posts[posts.length - 1].settle(ok);
    await new Promise((r) => setImmediate(r));
  };
  return { saver, posts, states, advance, settle, last: () => states[states.length - 1] };
}

test("changes are saved once, about 600 ms after the last one, with the latest answer", async () => {
  const r = rig();
  r.saver.start("K7Q2M", null);
  r.saver.change("K7Q2M", { ...EMPTY_ANSWER, verdict: "wrong" });
  await r.advance(400);
  r.saver.change("K7Q2M", { ...EMPTY_ANSWER, verdict: "wrong", shown_by: "both" });
  await r.advance(500);
  assert.equal(r.posts.length, 0, "still inside the pause");
  assert.equal(r.last().status, "pending");
  await r.advance(100);
  assert.equal(r.posts.length, 1);
  assert.deepEqual(r.posts[0].body, { ...EMPTY_ANSWER, verdict: "wrong", shown_by: "both", client_saved_at: "client-time" });
  assert.equal(r.posts[0].code, "K7Q2M");
  await r.settle(true);
  assert.equal(r.last().status, "saved");
  assert.equal(r.last().savedAt, "2026-11-16T21:14:00Z");
  assert.ok(r.saver.clean);
});

test("a failed save stays visible and is retried until it lands", async () => {
  const r = rig();
  r.saver.start("K7Q2M", null);
  r.saver.change("K7Q2M", { ...EMPTY_ANSWER, verdict: "supported" });
  await r.advance(600);
  await r.settle(false);
  assert.equal(r.last().status, "failed");
  assert.equal(r.saver.clean, false);
  await r.advance(999);
  assert.equal(r.posts.length, 1);
  await r.advance(1);
  assert.equal(r.posts.length, 2, "retried");
  await r.settle(false);
  assert.equal(r.last().status, "failed");
  assert.equal(r.last().failures, 2);
  await r.advance(2000);
  assert.equal(r.posts.length, 3);
  await r.settle(true);
  assert.equal(r.last().status, "saved");
  assert.ok(r.saver.clean);
});

test("leaving waits for the save to land, and a failed save holds it", async () => {
  const r = rig();
  r.saver.start("K7Q2M", null);
  r.saver.change("K7Q2M", { ...EMPTY_ANSWER, notes: "x" });
  let left = 0;
  r.saver.whenClean(() => left++);
  assert.equal(r.posts.length, 1, "saved at once, not after the pause");
  assert.equal(left, 0);
  await r.settle(false);
  assert.equal(left, 0, "still held after a failure");
  assert.throws(() => r.saver.start("PLM4T", null));
  await r.advance(1000);
  await r.settle(true);
  assert.equal(left, 1);
  r.saver.start("PLM4T", null);
});

test("a change made while a save is on its way is saved after it", async () => {
  const r = rig();
  r.saver.start("K7Q2M", "2026-11-16T20:00:00Z");
  assert.equal(r.last().status, "saved");
  r.saver.change("K7Q2M", { ...EMPTY_ANSWER, verdict: "wrong" });
  await r.advance(600);
  r.saver.change("K7Q2M", { ...EMPTY_ANSWER, verdict: "cannot_tell" });
  await r.settle(true);
  assert.equal(r.saver.clean, false);
  assert.equal(r.posts.length, 2, "the newer answer goes at once");
  assert.equal(r.posts[1].body.verdict, "cannot_tell");
  await r.settle(true);
  assert.ok(r.saver.clean);
});

test("leaving a clean item happens at once", () => {
  const r = rig();
  r.saver.start("K7Q2M", null);
  let left = false;
  r.saver.whenClean(() => { left = true; });
  assert.ok(left);
  assert.equal(r.posts.length, 0);
});

test("an edit is saved only under the item it was made on, never the item opened since", async () => {
  const r = rig();
  r.saver.start("K7Q2M", null);
  assert.equal(r.saver.change("K7Q2M", { ...EMPTY_ANSWER, notes: "about K7Q2M" }), true);
  await r.advance(600);
  await r.settle(true);
  r.saver.start("PLM4T", null);
  // A late edit from the page still showing K7Q2M's answer: refused, nothing posted.
  assert.equal(r.saver.change("K7Q2M", { ...EMPTY_ANSWER, notes: "about K7Q2M", verdict: "wrong" }), false);
  await r.advance(5000);
  assert.equal(r.posts.length, 1);
  assert.ok(r.saver.clean);
  assert.equal(r.saver.change("PLM4T", { ...EMPTY_ANSWER, verdict: "supported" }), true);
  await r.advance(600);
  assert.deepEqual(r.posts.map((p) => [p.code, p.body.notes, p.body.verdict]),
    [["K7Q2M", "about K7Q2M", null], ["PLM4T", "", "supported"]]);
});

test("nothing is saved before an item is open", () => {
  const r = rig();
  assert.equal(r.saver.change("K7Q2M", { ...EMPTY_ANSWER, verdict: "wrong" }), false);
});
