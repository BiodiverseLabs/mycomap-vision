// Human Feedback read-only (Steve, 2026-10-22): anyone signed out, and signed-in people who are
// not mycomap.org admins, project leaders or contributors, browse the queues and decide nothing.
// The server decides who that is (/api/feedback/me's read_only) and cuts what they see
// (feedback_public.py); the page shows no decision control, Auto-save or batch action to them.

import type { FeedbackMe, PhotoFlag } from "./api";

/** The line above the queues for a read-only viewer, and whether signing in may help; null
 *  for a reviewer. */
export function readOnlyNote(me: Pick<FeedbackMe, "user" | "read_only"> | undefined):
    { text: string; signIn: boolean } | null {
  if (!me?.read_only) return null;
  return me.user
    ? { text: "Read-only: reviewing is for mycomap.org admins, project leaders and contributors.",
        signIn: false }
    : { text: "Read-only: mycomap.org admins, project leaders and contributors sign in to review.",
        signIn: true };
}

/** A photo's place in its queue, in words: its flags, then the outcome when decided. */
export function photoLines(p: {
  flags?: PhotoFlag[]; suggested?: { view: string; p: number }[]; status: string | null;
  decision: { decision: string; reason?: string | null; views?: string[] | null } | null;
}, reasonLabel: (code: string) => string = (c) => c): string[] {
  const out: string[] = [];
  const flags = (p.flags ?? []).map((f) => (f.p == null ? f.kind : `${f.kind} (p ${f.p.toFixed(2)})`));
  if (flags.length) out.push(`Flagged: ${flags.join(", ")}`);
  const views = (p.suggested ?? []).map((s) => `${s.view} (p ${s.p.toFixed(2)})`);
  if (views.length) out.push(`Suggested: ${views.join(", ")}`);
  if (p.status !== "decided" || !p.decision) {
    out.push("Open");
    return out;
  }
  const d = p.decision;
  if (d.views) out.push(`Decided: ${d.views.length ? d.views.join(", ") : "no view"}`);
  else if (d.decision === "exclude_photo") out.push(`Decided: Exclude${d.reason ? ` (${reasonLabel(d.reason)})` : ""}`);
  else if (d.decision === "keep") out.push("Decided: Include");
  else out.push(`Decided: ${d.decision.replace(/_/g, " ")}`);
  return out;
}
