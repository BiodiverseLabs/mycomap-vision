// The frozen dataset releases research numbers come from (docs/dataset-release.md). A release
// never changes once cut, so its facts are written here from its MANIFEST.json rather than
// read live; the Reference data page shows them beside the live numbers, which change nightly.

export interface DatasetRelease {
  id: string;
  /** When the release was cut, and the moment its labels were read (UTC). */
  created: string;
  labelSnapshot: string;
  /** Records first validated after this date are the validation split. */
  valCutoff: string;
  records: { included: number; candidates: number; train: number; val: number;
             bySource: Record<"inat" | "mo", number> };
  photos: { included: number; byLicence: Record<"open" | "nc" | "arr", number> };
  labels: number;
  releaseHash: string;
  referenceHash: string;
  codeCommit: string;
  /** The DOI of its public (Creative Commons) variant, once deposited with the paper. */
  doi: string | null;
  /** The erratum in use (docs/dataset-release.md, Errata): records found faulty after the cut,
   *  left out of references, training and scoring. The release itself never changes. */
  erratum: Erratum | null;
}

export type ErratumReason = "mo_twin_as_inat" | "nonfungal_label";

export interface Erratum { id: string; records: number; byReason: Record<ErratumReason, number> }

/** Each reason in plain words, for a count of records. */
export const ERRATUM_REASON: Record<ErratumReason, string> = {
  mo_twin_as_inat: "Mushroom Observer finds held as iNaturalist records",
  nonfungal_label: "labels that name no fungus",
};

/** An erratum's records by reason, in words: "72 Mushroom Observer finds held as …, 103 …". */
export const erratumReasons = (e: Erratum): string =>
  (Object.entries(e.byReason) as [ErratumReason, number][])
    .map(([why, n]) => `${n.toLocaleString("en-US")} ${ERRATUM_REASON[why]}`).join(", ");

export const V1: DatasetRelease = {
  id: "v1",
  created: "2026-10-12T22:46:18Z",
  labelSnapshot: "2026-10-12T13:17:51Z",
  valCutoff: "2026-08-17",
  records: { included: 169_135, candidates: 182_236, train: 155_781, val: 13_354,
             bySource: { inat: 160_857, mo: 8_278 } },
  photos: { included: 680_047,
            byLicence: { open: 102_406 + 31_105, nc: 435_749 + 6_089, arr: 104_698 } },
  labels: 17_972,
  releaseHash: "f99777a694f25614ec5eb85ec4daecaf4245d090290114ee80bf8b02993924f0",
  referenceHash: "5e59b79a40987a3bb62405732062f8d74f016798184cba991d2fb785268d7457",
  codeCommit: "f38b82b344511eb0c882cde11afc43b6816e0449",
  doi: null,
  erratum: { id: "r3", records: 175, byReason: { mo_twin_as_inat: 72, nonfungal_label: 103 } },
};

export const DATASET_RELEASES: DatasetRelease[] = [V1];

const MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
                "September", "October", "November", "December"];

/** A date as the Research pages write it, in UTC: "17 August 2026". */
export function utcDay(iso: string): string {
  const d = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
  return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]} ${d.getUTCFullYear()}`;
}

/** A moment to the minute, in UTC: "12 October 2026, 13:17 UTC". */
export function utcMinute(iso: string): string {
  const d = new Date(iso);
  const two = (n: number) => String(n).padStart(2, "0");
  return `${utcDay(iso)}, ${two(d.getUTCHours())}:${two(d.getUTCMinutes())} UTC`;
}

/** A hash as people quote it: the first 12 characters. */
export const shortHash = (h: string) => h.slice(0, 12);
