// A kept observation run as GET /api/runs/<id> sends it, shaped like the Identify page's
// real example (iNaturalist 91443365), for the example-result tests.
import type { RunView } from "./api";

const cand = (name: string, confidence: number) => ({ name, score: confidence, confidence, reference_records: 50 });
const spec = (n: number, photo: string | null, owner = `owner${n}`) => ({
  observation_id: `inat:${n}`, name: "Laetiporus sulphureus", species: "Laetiporus sulphureus", genus: "Laetiporus",
  family: "Laetiporaceae", similarity: 0.88, matched_query_photo: 0, photo_url: photo, photo_owner: owner,
  photo_withheld: photo == null, source: "inat", source_id: String(n), record_url: `https://www.inaturalist.org/observations/${n}`,
  inat_url: null, species_url: null,
});
const onItsOwn = (photo: number, position: number, confidence: number) => ({
  photo, ranks: { family: [], genus: [], species: [] }, specimens: [],
  overall_top: { family: null, genus: null, species: { name: "Laetiporus sulphureus", position, confidence } },
});
export const EXAMPLE_RUN = {
  id: "AAAAAAAAAAAAAAAA", kind: "observation", photos: 0, mine: false, can_delete: false,
  answered_at: "2026-10-26T02:49:19Z", release: "2026-10-13.4",
  context_used: { latitude: 43.9, longitude: -72.2, observed_on: "2021-08-16", place_from: "observation", date_from: "observation" },
  observation: {
    source: "inat", id: "91443365", url: "https://www.inaturalist.org/observations/91443365", photos: 3, photos_used: 3,
    name_at_source: "Laetiporus sulphureus", rank_at_source: "species", excluded_reference: false, left_out_photos: 0,
    photos_shown: ["https://img/1.jpg", null, "https://img/3.jpg"],
    photos_credit: [{ licence: "cc-by", by: "Tom Norton" }, null, { licence: "cc-by", by: "Tom Norton" }],
  },
  results: [{
    model: { backbone: "served-model", method: "m" }, reference: { records: 1, species: 1, photos: 1 }, photos: 3,
    ranks: { family: [cand("Laetiporaceae", 0.9999)], genus: [cand("Laetiporus", 0.9998)],
             species: [cand("Laetiporus sulphureus", 0.8037), cand("Laetiporus cincinnatus", 0.186)] },
    specimens: [spec(1, "https://img/s1.jpg"), spec(2, null), spec(3, "https://img/s3.jpg"), spec(4, "https://img/s4.jpg"),
                spec(5, "https://img/s5.jpg")],
    per_photo: [onItsOwn(0, 1, 0.7737), onItsOwn(1, 1, 0.6847), onItsOwn(2, 2, 0.31)],
    hints: [], species_url: null, confidence_note: "", gate: null,
  }],
} as unknown as RunView;
