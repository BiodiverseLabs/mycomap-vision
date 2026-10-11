/**
 * The Identify page on a phone (Steve, 2026-10-11): an Identify bar pinned to the bottom,
 * a waiting card with real steps, and a scroll to the answer when it arrives.
 */
import { num } from "@/lib/api";

/** Below Tailwind's lg the form and the results stack in one column. */
export const SINGLE_COLUMN_BELOW = 1024;

/**
 * Whether to scroll to something that just appeared (the waiting card, the answer box).
 * In one column (phones) it always comes into view unless its top already sits in the
 * upper part of the screen. Side by side, only when it's out of view (above the header or
 * low on the screen), since the results already sit beside the form there.
 */
export function shouldScrollTo(v: {
  viewportWidth: number;
  viewportHeight: number;
  /** The element's top, relative to the viewport. */
  top: number;
  /** Height of the sticky site header. */
  header: number;
}): boolean {
  const inUpperPart = v.top >= v.header && v.top <= v.viewportHeight * 0.4;
  if (v.viewportWidth < SINGLE_COLUMN_BELOW) return !inUpperPart;
  return v.top < v.header || v.top > v.viewportHeight * 0.8;
}

/**
 * The pinned Identify bar shows on phones (CSS hides it from lg up) once there are photos
 * and until an answer is shown, except while the page's own Identify button is on screen
 * or identifying needs a sign-in first.
 */
export function showPinnedBar(s: {
  hasPhotos: boolean;
  hasAnswer: boolean;
  signInNeeded: boolean;
  inlineButtonVisible: boolean;
}): boolean {
  return s.hasPhotos && !s.hasAnswer && !s.signInNeeded && !s.inlineButtonVisible;
}

export type StepState = "done" | "active" | "todo";
export interface Step {
  key: "send" | "compare" | "answer";
  label: string;
  state: StepState;
  detail?: string;
}

/**
 * The waiting card's steps. `sent` is the upload's progress, 0 to 1 (null before the first
 * progress event); `waitSeconds` is set while a busy server keeps the photos in line.
 */
export function waitingSteps(s: {
  photos: number;
  sent: number | null;
  waitSeconds: number | null;
  referencePhotos: number | null;
}): Step[] {
  const uploaded = s.sent != null && s.sent >= 1;
  const send: Step = uploaded
    ? { key: "send", label: s.photos === 1 ? "Photo sent" : "Photos sent", state: "done" }
    : {
        key: "send",
        label: `Sending your ${s.photos === 1 ? "photo" : `${s.photos} photos`}`,
        state: "active",
        detail: s.sent != null ? `${Math.floor(s.sent * 100)}%` : undefined,
      };
  const compare: Step = s.waitSeconds != null
    ? {
        key: "compare",
        label: "Busy: your photos are next in line",
        state: uploaded ? "active" : "todo",
        detail: `trying again in ${s.waitSeconds} s`,
      }
    : {
        key: "compare",
        label: s.referencePhotos
          ? `Comparing with ${num(s.referencePhotos)} DNA-verified photos`
          : "Comparing with DNA-verified photos",
        state: uploaded ? "active" : "todo",
      };
  return [send, compare, { key: "answer", label: "Family, genus and species, each with its chance", state: "todo" }];
}
