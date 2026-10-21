// One find, two places (and the other record-pair queues): the two copies side by side, MO on
// the left and iNat on the right, each with its photos, the label mycomap.org has now and how
// it is placed (Steve, 2026-10-21: with both copies' photos in view, one look tells whether
// the two records are one find). The rules live here so they can be tested (twinCompare.test.ts).

import type { FeedbackItemDetail, FeedbackOption, PairPhoto, PairPlace } from "./api";
import { recordLabel } from "./feedback";

export interface PairSide {
  key: string;
  site: "MO" | "iNat";
  label: string;
  link: string | null;
  /** The label on mycomap.org now, when it answered for this record. */
  nameNow: string | null;
  /** How this copy is placed: "MO GPS", "accuracy 20 m", … ("" when the item does not say). */
  place: string;
  /** The rule's guess at the copy whose place is wrong. */
  suspect: boolean;
  photos: PairPhoto[];
}

const text = (v: unknown) => (v == null ? "" : String(v).trim());

/** Which copy the rule thinks is wrong: "MO copy: named location" → MO; "unclear: …" → none. */
export function suspectSite(likelyWrong: unknown): "MO" | "iNat" | null {
  const s = text(likelyWrong).toLowerCase();
  if (s.startsWith("mo copy")) return "MO";
  if (s.startsWith("inat copy")) return "iNat";
  return null;
}

/** The item's records as columns, MO first; [] when the item carries no photos (not a record
 *  pair, or an older server), so the page keeps its plain layout. */
export function pairSides(item: Pick<FeedbackItemDetail, "record_keys" | "links" | "evidence" | "live" | "photos">): PairSide[] {
  if (!item.photos) return [];
  const ev = item.evidence ?? {};
  const wrong = suspectSite(ev.likely_wrong_copy);
  const acc = text(ev.inat_accuracy_m);
  const sides = item.record_keys.map((key, i): PairSide => {
    const site = key.startsWith("mo:") ? "MO" : "iNat";
    const live = (item.live ?? []).find((r) => r.record_key === key && !r.unavailable);
    const place = site === "MO" ? text(ev.mo_placement)
      : !acc ? "" : /^\d+(\.\d+)?$/.test(acc) ? `accuracy ${Number(acc).toLocaleString("en-US")} m` : `accuracy ${acc}`;
    return { key, site, label: recordLabel(key), link: item.links[i] ?? null, nameNow: live?.name ?? null,
             place, suspect: wrong === site, photos: item.photos?.[key] ?? [] };
  });
  return sides.sort((a, b) => (a.site === b.site ? 0 : a.site === "MO" ? -1 : 1));
}

/** The "Listed at" line of a copy: the place as its site lists it, or why there is none. */
export function placeLine(place: PairPlace | undefined, site: "MO" | "iNat", loading: boolean):
    { text: string; muted: boolean } {
  if (place?.status === "ok" && place.text) return { text: place.text, muted: false };
  if (place?.status === "none") return { text: `no place listed on ${site}`, muted: true };
  if (place?.status === "unavailable") return { text: `${site} did not answer just now`, muted: true };
  return { text: loading ? `asking ${site}…` : "–", muted: true };
}

/** "9336" → "9,336 km apart"; "" when the item gives no distance. */
export function distanceText(km: unknown): string {
  const s = text(km);
  if (!s || Number.isNaN(Number(s))) return "";
  return `${Number(s).toLocaleString("en-US", { maximumFractionDigits: 1 })} km apart`;
}

/** The decision options in order, consecutive options of one group under one heading. */
export function optionGroups(options: FeedbackOption[]): { group: string | null; options: FeedbackOption[] }[] {
  const out: { group: string | null; options: FeedbackOption[] }[] = [];
  for (const o of options) {
    const g = o.group ?? null;
    const last = out[out.length - 1];
    if (last && g !== null && last.group === g) last.options.push(o);
    else out.push({ group: g, options: [o] });
  }
  return out;
}
