// How each approach on the Results page names a species, what it learned from, and its
// status (Results > Approaches compared; the approved redesign's shortened table, from the
// longer one the page carried before). The served method is
// nearest+mean-records-balanced+place (methods.NearestAndMeanRecordsBalanced, k 2, weight 0.6;
// placeprior, setting penalty-500-s30-2026-10-19, its own calibration).

export interface Approach { name: string; how: string; status: string; outside?: boolean }

export const APPROACHES: Approach[] = [
  { name: "Closest specimens + species average, by specimen + place and date",
    how: "Each reference record scores by its best-matching photo; a species takes its two best records, blended with its average look; nearby, in-season species move up.",
    status: "On the site (336 px since 20 Oct)" },
  { name: "… by specimen, photos only", how: "The same from the photos alone.",
    status: "On the site without place or date" },
  { name: "Closest specimens + species average",
    how: "Two best photos (possibly one specimen) blended with the species' average look.",
    status: "Tested; served before scoring by specimen" },
  { name: "Closest specimen", how: "Each species scored by its best match to all your photos.",
    status: "Tested; served until October 2026" },
  { name: "Species average", how: "Your photos against each species' average photo.", status: "Tested" },
  { name: "Its own classifier", how: "The fine-tune's species head.", status: "Tested; level with retrieval" },
  { name: "iNaturalist's computer vision", outside: true,
    how: "iNat's own model, best score across your photos, with or without location.",
    status: "To be scored on the sealed test" },
  { name: "Danish Fungi method on MycoMap records", outside: true,
    how: "Whole network retrained as a classifier (BEiT-B, 384 px), on v1's open photos (E4).",
    // research repo, experiments/2026-10-23-fungitastic-own-test.md (origin/main 2e2576a): E4 final
    status: "Final: 51.23% species top 1 on v1 validation finds, at its best epoch, 13 (Benchmarks: MycoMap v1 validation)" },
  { name: "Published Danish Fungi / FungiTastic models", outside: true,
    how: "The authors' models, unchanged.",
    // research repo, experiments/2026-10-13-v1-published-models.md, addendum of 25 Oct (origin/main 5792c17)
    status: "BEiT-B: 10.7% species top 1 on v1 validation finds; 56.9% on the species it has a class for" },
];
