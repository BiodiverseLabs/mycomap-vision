// What the Identify page says about each photo's type (phototypes.py on the server): a small
// chip ("Habitat", "Microscopy", "UV", …) on a photo that isn't an ordinary specimen photo,
// and "Include anyway" on a photo its type left out of the answer.

import type { IdentifyResult, PhotoResult } from "./api";

/** The chip's words for a photo, or null when it reads as an ordinary specimen photo (or
 *  its tag is too unsure to show, or the server tags no photos). */
export function photoTypeChip(p: Pick<PhotoResult, "photo_type">): string | null {
  return p.photo_type?.label ?? null;
}

/** True when this photo's type left it out of the answer (a micrograph, label or habitat
 *  photo), as the gate leaves out a photo that isn't a fungus. */
export function leftOutByType(p: Pick<PhotoResult, "photo_type">): boolean {
  return !!p.photo_type?.left_out;
}

/** True when this photo was counted although its type would have left it out. */
export function includedAnyway(p: Pick<PhotoResult, "photo_type">): boolean {
  return !!p.photo_type?.included;
}

/** The photos to offer "Include anyway" for: each left out by its type, with its chip. */
export function includeOffers(r: Pick<IdentifyResult, "per_photo">): { photo: number; label: string }[] {
  return r.per_photo
    .filter(leftOutByType)
    .map((p) => ({ photo: p.photo, label: photoTypeChip(p) ?? "Not a specimen photo" }));
}

/** The photos this answer counted anyway (the request's include list, as answered). */
export function includedNow(r: Pick<IdentifyResult, "photo_types">): number[] {
  return r.photo_types?.included ?? [];
}

/** The include list for running again with `photo` counted too. */
export function withIncluded(include: number[], photo: number): number[] {
  return [...new Set([...include, photo])].sort((a, b) => a - b);
}

/** The include list for running again with `photo` left out again. */
export function withoutIncluded(include: number[], photo: number): number[] {
  return include.filter((i) => i !== photo);
}

/** What a left-out photo's notice says when it is picked on its own. */
export function leftOutNotice(p: Pick<PhotoResult, "photo" | "photo_type">): string {
  const what = LOOKS[p.photo_type?.type ?? ""] ?? "not a photo of the fungus itself";
  return `Photo ${p.photo + 1} looks like ${what}, so it was left out of the answer.`;
}

const LOOKS: Record<string, string> = {
  micrograph: "a micrograph",
  slip: "a label or field slip",
  habitat: "a habitat photo",
};
