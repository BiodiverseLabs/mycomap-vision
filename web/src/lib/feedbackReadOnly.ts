// Who may do what on Human Feedback goes by the mycomap.org site role (Steve, 2026-11-20): anyone
// signed out browses the Photos tab and decides nothing; any signed-in account tags its photos
// (those anyone may see); Contributors, Lab Contributors, Project Leaders and admins also see
// all-rights-reserved photos and the Taxonomy tab. The server decides who is which
// (/api/feedback/me's read_only and taxonomy) and cuts what they see (feedback_public.py); the
// page shows no decision control, Auto-save or batch action to a read-only viewer.

import type { FeedbackMe, PhotoFlag } from "./api";

export const TAXONOMY_WHO = "mycomap.org Contributors, Lab Contributors, Project Leaders and admins";

/** The line above the queues for anyone who does not review everything, and whether signing in
 *  may help; null for a Contributor or higher. */
export function readOnlyNote(me: Pick<FeedbackMe, "user" | "read_only" | "taxonomy"> | undefined):
    { text: string; signIn: boolean } | null {
  if (!me) return null;
  if (me.read_only) {
    return me.user
      ? { text: "Read-only: this account can't decide here.", signIn: false }
      : { text: "Read-only: sign in with your mycomap.org account to tag photos.", signIn: true };
  }
  if (me.taxonomy === false) {
    return { text: "You can tag the photos anyone may see. All-rights-reserved photos and the "
              + `Taxonomy queues are for ${TAXONOMY_WHO}.`, signIn: false };
  }
  return null;
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
