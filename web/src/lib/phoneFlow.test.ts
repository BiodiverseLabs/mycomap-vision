import { test } from "node:test";
import assert from "node:assert/strict";
import { shouldScrollTo, showPinnedBar, waitingSteps } from "./phoneFlow";

const phone = { viewportWidth: 375, viewportHeight: 812, header: 64 };
const desktop = { viewportWidth: 1280, viewportHeight: 900, header: 64 };

test("on a phone the answer is scrolled to when it lands below the upper part of the screen", () => {
  assert.equal(shouldScrollTo({ ...phone, top: 460 }), true,
    "57% down: the case the old 60% rule missed");
  assert.equal(shouldScrollTo({ ...phone, top: 2000 }), true, "far below");
  assert.equal(shouldScrollTo({ ...phone, top: -300 }), true, "above the screen");
  assert.equal(shouldScrollTo({ ...phone, top: 30 }), true, "under the sticky header");
});

test("on a phone nothing moves when the answer already starts near the top", () => {
  assert.equal(shouldScrollTo({ ...phone, top: 120 }), false);
  assert.equal(shouldScrollTo({ ...phone, top: 0.4 * 812 }), false);
});

test("side by side, the page scrolls only when the answer is out of view", () => {
  assert.equal(shouldScrollTo({ ...desktop, top: 160 }), false, "beside the form, in view");
  assert.equal(shouldScrollTo({ ...desktop, top: 600 }), false);
  assert.equal(shouldScrollTo({ ...desktop, top: -900 }), true, "scrolled past it, at the button");
  assert.equal(shouldScrollTo({ ...desktop, top: 800 }), true);
});

test("the pinned Identify bar shows once photos are chosen and until there's an answer", () => {
  const base = { hasPhotos: true, hasAnswer: false, signInNeeded: false, inlineButtonVisible: false };
  assert.equal(showPinnedBar(base), true);
  assert.equal(showPinnedBar({ ...base, hasPhotos: false }), false, "nothing to identify yet");
  assert.equal(showPinnedBar({ ...base, hasAnswer: true }), false, "the answer has its own Start over");
  assert.equal(showPinnedBar({ ...base, signInNeeded: true }), false, "sign in comes first");
  assert.equal(showPinnedBar({ ...base, inlineButtonVisible: true }), false, "never two Identify buttons on screen");
});

const states = (s: ReturnType<typeof waitingSteps>) => s.map((x) => x.state).join(" ");

test("while the photos upload, sending is the active step with its progress", () => {
  const before = waitingSteps({ photos: 3, sent: null, waitSeconds: null, referencePhotos: 592926 });
  assert.equal(states(before), "active todo todo");
  assert.equal(before[0].label, "Sending your 3 photos");
  assert.equal(before[0].detail, undefined, "no % before the first progress event");
  const half = waitingSteps({ photos: 3, sent: 0.426, waitSeconds: null, referencePhotos: 592926 });
  assert.equal(half[0].detail, "42%");
  assert.equal(waitingSteps({ photos: 1, sent: 0.1, waitSeconds: null, referencePhotos: null })[0].label,
    "Sending your photo");
});

test("once sent, comparing is the active step and names the reference photos", () => {
  const s = waitingSteps({ photos: 3, sent: 1, waitSeconds: null, referencePhotos: 592926 });
  assert.equal(states(s), "done active todo");
  assert.equal(s[0].label, "Photos sent");
  assert.equal(s[1].label, `Comparing with ${(592926).toLocaleString()} DNA-verified photos`);
  assert.equal(waitingSteps({ photos: 3, sent: 1, waitSeconds: null, referencePhotos: null })[1].label,
    "Comparing with DNA-verified photos");
});

test("a busy server shows the photos waiting in line, and re-sending them shows again", () => {
  const busy = waitingSteps({ photos: 3, sent: 1, waitSeconds: 12, referencePhotos: 592926 });
  assert.equal(states(busy), "done active todo");
  assert.equal(busy[1].label, "Busy: your photos are next in line");
  assert.equal(busy[1].detail, "trying again in 12 s");
  const resend = waitingSteps({ photos: 3, sent: 0.3, waitSeconds: 12, referencePhotos: 592926 });
  assert.equal(states(resend), "active todo todo");
});
