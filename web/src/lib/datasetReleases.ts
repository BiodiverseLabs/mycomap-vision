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
  /** The Zenodo DOI of its public (Creative Commons) variant, once deposited. */
  doi: string | null;
}

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
};

export const DATASET_RELEASES: DatasetRelease[] = [V1];

/** A hash as people quote it: the first 12 characters. */
export const shortHash = (h: string) => h.slice(0, 12);
