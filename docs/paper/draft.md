# Photo identification of North American fungi from DNA-verified references

**Draft 0.2, 9 October 2026.** A living draft for *Mycologia* (Original Research). Every number
here is preliminary and will be rerun on the final model and test sets before submission. Where
a result comes from a small or interim test, the text says so.

**Authors:** Stephen Russell¹ (corresponding author), MycoMap contributors [to be confirmed]

¹ MycoMap.org

**Running head:** DNA-verified photo identification of fungi

> **Where things stand.** Solid: the data pipeline, the full-data fine-tuned model, the
> evaluation on the newest four weeks of validations (1,152 records), the reference-depth
> analysis and the method comparisons. Still to do: the 13,145-record held-out benchmark with
> the standard report, iNaturalist's computer vision on the same records, the location and
> season prior, per-species (macro) metrics, open-set analysis, leakage checks, and the
> separately validated ~1,000-record paper test set. See Appendix A (benchmark plan) and
> Appendix B (next tests).

---

## ABSTRACT

Photo identification tools for fungi learn from community-identified records, so their mistakes
are partly inherited from their training labels, and they are rarely tested against DNA. We built
MycoMap Vision, an identifier for North American fungi whose every reference record is backed by
a DNA barcode reviewed in a MycoMap validation project. The reference set holds about 160,000
records with about 590,000 photographs of nearly 18,000 species, many known only by provisional
names, and 43% of names have a single record. Rather than training a classifier, we retrieve the
most similar DNA-verified specimens: photographs are embedded with BioCLIP 2, partially
fine-tuned on the reference set, and each species is scored by how well its best specimen matches
all photographs of a find. Each taxonomic rank receives a calibrated confidence and a list of
likely names with a stated coverage. On the 1,152 records validated in the four weeks after the
training cutoff, the identifier named the species correctly for [34.5–38.4]% of finds, the genus
for [71.6–74.4]% and the family for [79.7–81.8]%. Accuracy depended chiefly on how many
DNA-verified records the true species had: [13]% for species with one to four references against
[66–68]% for species with more than 100. Reference depth, not the method, set most of the
remaining error. [Comparison with iNaturalist's computer vision on the same records, held-out
results and the effect of a location prior to be added.] Because reference depth grows weekly as
records are sequenced, the identifier is updated nightly and its accuracy is reported by depth.

**KEY WORDS:** barcoding; BioCLIP; citizen science; computer vision; deep learning; iNaturalist;
macrofungi; nearest-neighbor retrieval

*(Mycologia: key words in alphabetical order, not repeating title words.)*

---

## INTRODUCTION

*[Draft outline; prose to be written.]*

1. **The identification problem.** Most fungal records in North America now come from community
   science platforms. Macroscopic identification is unreliable in many genera (*Cortinarius*,
   *Inocybe*, *Russula*, *Entoloma*), and many collections belong to species without a formal name.
   DNA barcoding (ITS) has become the reference standard for these groups. Cite: community
   sequencing programs, MycoMap, the share of provisional names.
2. **Automated photo identification today.** iNaturalist's computer vision suggests names to
   millions of users. Its training labels are community identifications, so errors in hard groups
   can be learned and repeated. Published tests of mushroom apps against expert identifications
   report modest accuracy (Hodgson et al. 2023: iNaturalist correct for 35% of 78 specimens). We
   found no peer-reviewed study testing a fungal photo identifier against DNA-verified names at
   scale (Appendix C).
3. **Machine-learning work on fungi.** Danish Fungi 2020 (Picek et al. 2022a) and the FungiCLEF
   challenges (2022–2025) established benchmarks for fungal image recognition, including open-set
   and poisonous-species costs. FungiTastic (Picek et al. 2025) added a DNA-sequenced test subset.
   Foundation models trained on the tree of life (BioCLIP, Stevens et al. 2024; BioCLIP 2, Gu et
   al. 2025) give strong image features without task-specific training.
4. **The long tail.** Verified records are extremely uneven across species: most named and
   provisional species have few. Classifiers trained on such data favor common species. Retrieval
   of the nearest verified specimen lets a species known from one record compete.
5. **Aims.** (i) Build a photo identifier whose references are all DNA-verified; (ii) measure it
   on records validated after its training data, by reference depth; (iii) compare it with
   iNaturalist's computer vision on the same records; (iv) report calibrated confidence and
   likely-name lists usable by community scientists; (v) show how accuracy changes as the
   reference set grows.

---

## MATERIALS AND METHODS

### Reference records and labels

**Source.** Records are iNaturalist observations that were sequenced (usually ITS) and marked
validated ("green") in at least one MycoMap validation project, meaning a project reviewer
accepted the sequence-based name for the observation. Records were exported read-only from
MycoMap.org. Observations, photographs, photographer, license and coordinates came from the
iNaturalist API. Records outside North America were excluded.

**Counts (export of 28 September 2026).** 164,201 validated records, 159,353 in North America;
146 with conflicting validated names were held out. The iNaturalist metadata covered 155,858
observations and 593,224 photographs (about 3.2 photos per record; 65% of records have two or
more). Photographs were stored at 1024 px on the long side (593,214 retrieved; 10 had been removed
from iNaturalist).

**Name of a record.** The label is the observation's own name on MycoMap after a refresh from the
legacy database, not the name of a linked sequence. A record can carry several sequences,
including non-target or discarded ones.

**Name cleaning.**
- *Spellings.* The same provisional name reaches MycoMap written several ways (`Inocybe sp.
  'PNW18'`, `Inocybe PNW18`, `Inocybe "sp-PNW18"`). Spellings that differ only in punctuation,
  spacing, capitals or a "sp." were merged under the form `Genus sp. 'CODE'`; ambiguous cases were
  left for a person. On the 28 September export, 20,609 names became 19,843 labels (1,035 names
  merged, 2,546 records relabeled).
- *Higher ranks.* Family, order, class and phylum were taken from iNaturalist's taxonomy, one
  answer per genus, within kingdom Fungi (27,788 blank families filled; 33,326 records changed
  family, e.g. *Hygrocybe* to Hygrophoraceae).
- *One-word names.* A record named only to genus or higher (e.g. "Russula") counts at genus and
  family only, never as a species; records with no usable rank ("Fungi", "Unknown") were dropped
  (1,239 records).
- *Guest organisms.* Records whose DNA name is an organism living inside the photographed fungus
  (yeasts such as *Teunomyces*, *Candida* and *Rhodotorula*) were excluded from reference and test
  sets, because the photographs show the host. Mycoparasites usually photographed on their host
  (*Spinellus*, *Syzygites*, *Mycogone*, *Sepedonium*, *Cladobotryum*) were also excluded by
  default; visible molds and parasites that are themselves the subject (*Trichoderma*,
  *Penicillium*, *Pilobolus*, *Taphrina* and others) were kept.

**Photographs and permission.** About 16% of photographs (96,719 from 1,251 photographers) are
"all rights reserved". These are used for training and matching under a permission process run
through MycoMap.org; they are not displayed to users unless the photographer has granted
permission. Photographs withdrawn by their owner are excluded from all sets.

**Reference depth over time.** Species-labeled records with photographs grew from 63,796 (12,141
species, 703 with 20+ records) in March 2025 to 151,522 (17,935 species, 1,919 with 20+ records)
by September 2026 (TABLE 1).

### Image features

Photographs were embedded with BioCLIP 2 (Gu et al. 2025), a vision transformer (ViT-L/14)
trained with hierarchical contrastive learning on about 200 million images across the tree of
life. Before choosing it we screened eight frozen image models on the same records (TABLE 2):
BioCLIP 2; DINOv2 base and large (Oquab et al. 2024); DINOv3 base and large at 512 px; SigLIP 2
large; EVA-02 large; and ConvNeXt V2 large.

**Fine-tuning.** We fine-tuned the last 4 of BioCLIP 2's 24 transformer blocks on the reference
records only (validated through 7 September 2026), with cosine classifiers at species, genus and
family initialized from class mean embeddings. Photographs were sampled with weight
1/√(photos of the species) to temper the dominance of common species. Augmentation excluded hue
and saturation changes, because color is diagnostic in fungi. Training ran two epochs (18,328
steps, batch 64) on one NVIDIA L4 GPU (AWS g6.xlarge); the whole run, including embedding all
photographs twice, took about 9 h and cost about US$8. The fine-tuned model is used only as a
feature extractor; its classifier heads are not used for identification (see Results).

### Identification by nearest verified specimen

For a query find with photographs *q*₁…*qₙ* and a species *s* with reference photographs *R*ₛ:

- **Nearest specimen** (`nearest`): score(*s*) = mean over *i* of max over *r* ∈ *R*ₛ of
  cos(*qᵢ*, *r*). A species known from one specimen competes on the same terms as one with a
  thousand.
- **Species average** (`species-mean`): cosine similarity to the mean embedding of the species.
- **Blend** (`nearest+mean`, experimental): 0.6 × (mean of each photo's two best matches) + 0.4 ×
  species average. Chosen on the evaluation set below; to be confirmed on held-out records.

Genus and family scores are the best species score inside them, so every rank gets its own
answer. Trained alternatives (linear and hybrid classifiers on the embeddings, balanced softmax)
were also tested.

### Confidence and likely-name lists

Scores at each rank are turned into probabilities with a softmax whose temperature is fitted by
minimizing negative log-likelihood on the evaluation records (separately per rank and method).
For each rank we also report a list of likely names built by split-conformal prediction (least
ambiguous set-valued classifier): the target coverage starts at 90% and is lowered until the
mean list size on held-back records is five names or fewer; lists are cut at 15 names. A species
absent from the reference set can never be listed and counts as a miss.

### Location and season (in progress)

A prior from DNA-verified records alone was tested first. A prior from iNaturalist occurrence data
is built but not yet tuned: observer-days per 0.5° cell for every fungal taxon in North America
(9.07 million observations from the 27 September 2026 iNaturalist open-data export), a density
score (150 km kernel, shrunk to genus) and a season score (week × 10° latitude band), both capped,
plus a strong penalty only when a species with at least 20 occurrences has none within 1,500 km
and no DNA record nearby. Each scored record's own observation is removed from the counts. MycoMap
Atlas range maps are planned as a second source.

### Evaluation design

**Temporal split.** The main evaluation identifies the records validated in the 28 days after the
reference cutoff, using only records validated before it, as a new find would be met. For the
full-data model (cutoff 7 September 2026): 1,152 test records, 152,915 reference records, 3.6
photographs per record.

**Held-out benchmark.** 13,145 North American records validated on the legacy MycoMap database
but never imported into Vision's data (a synchronization fault), so never seen in training or as
references. Split 3,000 for development and tuning, 10,145 for testing.

**Paper test set (planned).** About 1,000 records validated after the method is frozen, registered
as held out before any model sees them (Appendix A).

**Advance predictions.** Every night the identifier predicts up to 300 records still awaiting
DNA results; predictions are scored when the records are validated. Nothing could have seen the
answer.

**Baseline.** iNaturalist's computer vision (score_image endpoint), given every photograph of a
record, scored photo-only and with location, taking the best score across photographs.

**Scoring.** Top-1, -3, -5 and -10 accuracy at species, genus and family, per find (all photos)
and for the first photograph only; species accuracy by the true species' number of reference
records (0, 1–4, 5–19, 20–99, 100+); names compared strictly, *sensu lato* (genus splits in
*Cortinarius* s.l. and *Inocybe* s.l., gender endings) and as species complexes (beta).
Provisional names count as species. Statistics: Wilson 95% intervals, McNemar / sign tests on
paired records.

### Software and serving

Code is in Python (PYTORCH, OPEN_CLIP, TIMM, FASTAPI) with a React front end. The public site
(vision.mycomap.org) runs on a 4 GB CPU-only server; all training runs on short-lived GPU
instances. The reference index is updated nightly with newly validated records.

---

## RESULTS (preliminary)

### Choice of image model

**TABLE 2.** Frozen image models on the same 68 test records (8,000-record development sample,
28 September 2026). Top-1 accuracy (%), best method for each model. A direction only: 68 records.

| Model | Species | Genus | Family |
|---|---|---|---|
| iNaturalist CV, with location | 29.4 | 67.7 | 70.6 |
| iNaturalist CV, photo only | 27.9 | 64.7 | 67.7 |
| BioCLIP 2 | 25.0 | 61.8 | 71.4 |
| DINOv3-L, 512 px | 16.2 | 42.6 | — |
| EVA-02 L | 13.2 | 33.8 | 44.6 |
| DINOv2-L | 10.3 | 36.8 | 44.6 |
| DINOv2-B | 10.3 | 30.9 | 41.1 |
| ConvNeXt V2 L | 8.8 | 29.4 | 32.1 |
| SigLIP 2 L | 5.9 | 19.1 | 28.6 |

No general-purpose model came near BioCLIP 2, which was trained on the tree of life. Photographs
at 1024 px and 500 px gave the same results within one or two records.

### Full-data model

**TABLE 3.** Records validated 8 September – 5 October 2026 (n = 1,152), against 152,915 earlier
records. Top-1 accuracy (%), all photographs of a record.

| Model and method | Species | Genus | Family |
|---|---|---|---|
| Frozen BioCLIP 2, nearest | 32.6 | 68.6 | 76.3 |
| Frozen BioCLIP 2, species average | 30.7 | 69.7 | 78.2 |
| Fine-tuned, nearest (**served**) | 34.5 | 71.6 | 79.7 |
| Fine-tuned, species average | 34.3 | 72.7 | 81.2 |
| Fine-tuned, nearest+mean (experimental) | **38.4** | **74.4** | **81.8** |
| Fine-tuned, hybrid classifier | 26.0 | 66.5 | 74.6 |
| Fine-tuned, linear classifier | 17.7 | 55.8 | 66.4 |

Top-5: nearest 56.8 / 89.2 / 93.6; nearest+mean 61.2 / 90.8 / 94.6. First photograph only, species:
nearest 28.9, nearest+mean 35.0. Fine-tuning added 2–3 points at every rank. Small classifiers
trained on top of the finished image features lost to retrieval at every rank, as they did on the
development sample. This does not test a classifier trained end to end: retraining the whole
network as a species classifier, as in Danish Fungi 2020 and FungiTastic, is being replicated on
our records (Appendix A, A12). The fine-tune's own classifier heads were not saved and have not
been scored as an identifier.

*[To add: 95% intervals, macro (per-species) accuracy, the standard top-1/3/5/10 ×
strict/s.l./complex table.]*

### Reference depth explains most errors

**TABLE 4.** Species top-1 (%) by the true species' number of reference records, same records as
TABLE 3.

| Reference records (test records) | nearest | nearest+mean |
|---|---|---|
| 0 (141) | — | — |
| 1–4 (215) | 12.6 | 13.0 |
| 5–19 (296) | 31.8 | 39.9 |
| 20–99 (364) | 54.1 | 58.5 |
| 100+ (105) | 65.7 | 67.6 |
| Provisional names (506) | 27.9 | 32.6 |
| Formal names (615) | 40.0 | 43.1 |

Genus accuracy for species with no reference record was 50–51%.

Twelve percent of test records (141) belonged to species with no reference record at all, which
no reference-based method can name. For truths with one to four references, `nearest` ranked the
true species first for 12%, second to fifth for 14%, 6th–20th for 24% and below 20th for 46%. Most
sparse-species misses are therefore not near misses that a better scoring rule could recover: the
photographs held for those species do not resemble the query (other angles, stages or
observers), or the names split what the photographs cannot.

**Depth corrections lose.** Because `nearest` takes a maximum over a species' photographs, a
deep species gets more chances. Every correction we tried traded a few sparse-species records for
more deep-species ones: fixed penalties on the log of photo count, subtracting each species'
expected best match, per-species z-scores, hubness correction, and per-depth offsets fitted with
observer-grouped cross-validation (TABLE S1). Test records arrive at natural frequencies, so a
deep species really is more likely.

**What helps is steadier evidence per species.** The species average is better for species with
5–19 references (39.2% vs 31.8%) and worse for those with 100+ (52.4% vs 65.7%); `nearest` is the
reverse. Blending them (`nearest+mean`) kept the strengths of both: 76 records fixed against 33
broken at species (sign test *P* < 0.0001), 56 against 25 at genus. Provisional names gained most
(+4.7 points). This setting was chosen on the same records it is reported on and must be
confirmed on the held-out set.

**Depth grows every week.** The share of species with 20 or more verified records rose from 5.8%
to 10.7% in 18 months (TABLE 1). Methods are therefore compared within depth bands, and settings
are refitted on each new evaluation.

### Photographs: how many, and which

Using every photograph of a record beat the first photograph alone by 5.6 points at species
(`nearest`). Per-photo voting schemes gave no gain over the mean similarity. Removing or
down-weighting photographs dominated by collection slips, baskets, habitat or microscopy changed
results by one or two records in either direction. Hand-labeling all 369 photographs of a
100-record audit and removing exactly the uninformative ones changed species accuracy by +2/−2
records. Some single photograph had the right species for 40.9% of records against 34.3% for the
combination, so a perfect photo picker could gain about five points, but no signal we tried
(similarity, margin, confidence, agreement, photo type) finds that photograph.

### Location and season

A prior built from DNA-verified records alone did not help (development sample: species 24.2% with
no prior, 22.0–23.1% with weights 0.1–1; held-out pilot: 45 to 47 of 95). In 22 of 95 pilot records
the true species had no DNA-verified record within 300 km: verified records are too sparse to
draw ranges. *[Occurrence-based prior: results on the development split to come.]*

### Comparison with iNaturalist's computer vision

*[Pending: iNaturalist CV on a 2,000-record subsample of the held-out set (303 done) and on the
1,152 records of TABLE 3 (349 done).]* Interim results are not comparable and are not cited. On
the development sample (307 records, against a reference set of only 8,830 records), iNaturalist
led at genus (71.0% vs 62.6%), and the gap lay entirely in genera with few references: for genera
with 100+ references both reached 80.9%.

*Caveat for the final comparison:* iNaturalist's model may have been trained on some of our test
observations, with labels that were themselves informed by our DNA results. Paper test records
will have iNaturalist's identifications snapshotted before analysis, and the model version and
date recorded.

### Confidence and likely-name lists

Species confidence was calibrated with temperature 0.037 (NLL 3.22 per record; ECE 0.064 over 10
bins). Likely-name lists on held-back halves of TABLE 3's records (`nearest`):

| Rank | Coverage | Mean list size |
|---|---|---|
| Family | 90% | 1.8 |
| Genus | 90% | 3.8 |
| Species | 55% (62% when the species has references) | 3.4 |

Species lists cannot reach 90% because 31% of species truths are unlistable (no reference record,
or beyond the 15-name cut). With `nearest+mean` the species list covers 60% (69% when the species
has references) with 4.1 names. *[To add: reliability diagrams, risk–coverage curves, coverage by
depth band.]*

### Held-out records

*[Pending: all 3,000 development and 10,145 test records with the standard report.]* A 100-record
pilot (answer key = observation name): species 47.4%, genus 78.6%, family 86.5% (`nearest`). In a
99-record audit, species accuracy was 12% for true species with fewer than 20 references and 63%
for 20 or more. *Sensu lato* matching changed nothing at species (40.4%); species-complex matching
gave 43.4%; genus *s.l.* gave 80% against 77% strict.

### Speed and cost

One identification with one photograph takes 4–10 s on the 4 GB CPU server; capacity is about
13 identifications per minute. Eight-bit quantization of the image model was 2.35 times faster
but lost about 2 points at genus and family (33.9 / 69.4 / 77.8 vs 34.5 / 71.6 / 79.7), so the
full-precision model is served.

---

## DISCUSSION

*[Draft points; prose to be written once results are final.]*

1. **DNA-verified references change what accuracy means.** Our errors are measured against
   sequence-backed names, including provisional species that no community-trained model has as a
   class. Set our numbers honestly beside app studies scored against expert identifications.
2. **Depth is the main lever, and it grows by itself.** Most sparse-species misses are far misses;
   no scoring rule fixes them. Each sequenced record moves its species toward the bands where
   accuracy is 40–68%. Argue for targeted sequencing of sparse species, guided by the advance
   predictions.
3. **Retrieval over classification for long-tailed fungi.** Small classifiers on fixed features
   lost; whether a fully trained classifier does better on well-sampled species awaits the
   replication (A12). Retrieval keeps one-record species in play and lets the reference set grow
   nightly without retraining; a hybrid (classifier for deep species, retrieval for the tail) is
   the likely outcome to test.
4. **Genus is reliable; species is a short list.** For community use, show a calibrated list
   rather than one name, especially for species complexes and provisional names within look-alike
   groups.
5. **What photographs do not show.** Look-alikes within complexes and microscopic characters; the
   tool asks for underside and stem photographs when species are close.
6. **Limitations.** North America only; iNaturalist photographs only; label noise (stale names on
   the legacy database, found and being corrected); sampling bias toward projects and regions
   that sequence; one backbone family; not for edibility.
7. **Safety.** Not for edibility decisions. Report results for deadly genera (Appendix A).

---

## DATA AVAILABILITY STATEMENT

*[Draft.]* Record identifiers, splits (test, development, held-out) and per-record predictions will
be deposited [Zenodo/Dryad, DOI]. Sequences are in GenBank [accessions via MycoMap]. Code [GitHub +
Zenodo DOI]. Photographs remain with their iNaturalist owners under their licenses; model weights
[availability to decide: trained partly on all-rights-reserved photographs]. Coordinates are not
released beyond iNaturalist's public precision.

## ACKNOWLEDGMENTS

*[To write: the community scientists, sequencing programs and validation-project reviewers; the
photographers who granted permission.]*

## FUNDING

*[To write, or "No specific funding was received".]*

## DISCLOSURE STATEMENT

*[To write: competing interests (e.g. MycoMap roles), or none.]*

## DECLARATION OF GENERATIVE AI USE

*[Draft, required by Taylor & Francis.]* Claude Opus 5.5 (Anthropic) was used under the authors'
direction to write and test the software, run analyses and draft parts of this manuscript. The
authors reviewed and take responsibility for all content. No AI-generated or AI-altered images
appear in the results.

---

## LITERATURE CITED

*(CSE name–year, full journal names. Entries marked [check] need their details confirmed before
submission.)*

Bertinetto L, Mueller R, Tertikas K, Samangooei S, Lord NA. 2020. Making better mistakes:
leveraging class hierarchies with deep networks. In: Proceedings of the IEEE/CVF Conference on
Computer Vision and Pattern Recognition. [check pages]

Garcin C, Joly A, Bonnet P, Lombardo J-C, Affouard A, Chouet M, Servajean M, Lorieul T, Salmon J.
2021. Pl@ntNet-300K: a plant image dataset with high label ambiguity and a long-tailed
distribution. In: Proceedings of the Neural Information Processing Systems Track on Datasets and
Benchmarks.

Gu J, Stevens S, Campolongo EG, et al. 2025. BioCLIP 2: emergent properties from scaling
hierarchical contrastive learning. arXiv:2505.23883. [check authors]

Guo C, Pleiss G, Sun Y, Weinberger KQ. 2017. On calibration of modern neural networks. In:
Proceedings of the 34th International Conference on Machine Learning. PMLR 70:1321–1330. [check]

Hodgson D, McKenzie M, May E, Greene S. 2023. A comparison of the accuracy of mushroom
identification applications using digital photographs. Clinical Toxicology. 61(3):166–172.
doi:10.1080/15563650.2022.2162917.

Kapoor S, Cantrell EM, Peng K, et al. 2024. REFORMS: consensus-based recommendations for
machine-learning-based science. Science Advances. 10(18):eadk3452. doi:10.1126/sciadv.adk3452.

Liu Z, Miao Z, Zhan X, Wang J, Gong B, Yu SX. 2019. Large-scale long-tailed recognition in an open
world. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.

Mac Aodha O, Cole E, Perona P. 2019. Presence-only geographical priors for fine-grained image
classification. In: Proceedings of the IEEE/CVF International Conference on Computer Vision.
p. 9596–9606.

Oquab M, Darcet T, Moutakanni T, et al. 2024. DINOv2: learning robust visual features without
supervision. Transactions on Machine Learning Research. [check]

Picek L, Šulc M, Matas J, Jeppesen TS, Heilmann-Clausen J, Læssøe T, Frøslev T. 2022a. Danish
Fungi 2020 – not just another image recognition dataset. In: Proceedings of the IEEE/CVF Winter
Conference on Applications of Computer Vision. p. 1525–1535.
doi:10.1109/WACV51458.2022.00334.

Picek L, Šulc M, Matas J, Heilmann-Clausen J, Jeppesen TS, Lind E. 2022b. Automatic fungi
recognition: deep learning meets mycology. Sensors. 22(2):633. doi:10.3390/s22020633.

Picek L, et al. 2022c. Overview of FungiCLEF 2022: fungi recognition as an open set
classification problem. CEUR Workshop Proceedings. 3180. [check authors]

Picek L, et al. 2023. Overview of FungiCLEF 2023: fungi recognition beyond 1/0 cost. CEUR Workshop
Proceedings. 3497. [check authors]

Picek L, et al. 2025. FungiTastic: a multi-modal dataset and benchmark for image categorization.
arXiv:2408.13632. [check final venue]

Stevens S, Wu J, Thompson MJ, et al. 2024. BioCLIP: a vision foundation model for the tree of
life. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
doi:10.1109/CVPR52733.2024.01836.

Šulc M, Picek L, Matas J, Jeppesen TS, Heilmann-Clausen J. 2020. Fungi recognition: a practical
use case. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision.
p. 2305–2313. doi:10.1109/WACV45572.2020.9093624.

Van Horn G, Mac Aodha O, Song Y, et al. 2018. The iNaturalist species classification and detection
dataset. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition.
[check]

Vaze S, Han K, Vedaldi A, Zisserman A. 2022. Open-set recognition: a good closed-set classifier is
all you need? In: International Conference on Learning Representations.

---

## TABLE 1. Reference depth over time

Species-labeled records with photographs, by validation date.

| Reference set up to | Records | Species | Species with 20+ records |
|---|---|---|---|
| 7 Mar 2025 | 63,796 | 12,141 | 703 (5.8%) |
| 7 Sep 2025 | 84,858 | 14,056 | 1,022 (7.3%) |
| 7 Mar 2026 | 108,421 | 15,653 | 1,350 (8.6%) |
| 7 Jun 2026 | 139,086 | 17,013 | 1,771 (10.4%) |
| 7 Sep 2026 | 151,522 | 17,935 | 1,919 (10.7%) |

## TABLE S1. Depth corrections (supplementary)

Species top-1 (%) on TABLE 3's records, fine-tuned model, by the true species' references.

| Scoring | All | 1–4 | 5–19 | 20–99 | 100+ | Fixed / broken vs nearest |
|---|---|---|---|---|---|---|
| nearest | 34.5 | 12.6 | 31.8 | 54.1 | 65.7 | |
| best − 0.01 ln N | 33.7 | 14.4 | 34.5 | 51.4 | 55.2 | +24 / −33 |
| best − expected best (×0.25) | 33.8 | 14.9 | 34.5 | 51.4 | 55.2 | +25 / −33 |
| per-species z-score | 12.1 | 18.6 | 16.9 | 11.8 | 2.9 | +35 / −286 |
| hubness (CSLS-style) | 34.3 | 15.3 | 33.8 | 51.6 | 60.0 | +30 / −33 |
| per-depth offsets (fitted) | 33.9 | 13.5 | 34.8 | 49.2 | 65.7 | +18 / −25 |
| mean of top-2 photos | 36.0 | 10.7 | 32.8 | 56.6 | 73.3 | +36 / −20 |
| species average | 34.3 | 13.5 | 39.2 | 50.8 | 52.4 | +87 / −89 |
| 0.6 top-2 + 0.4 species average | 38.4 | 13.0 | 39.9 | 58.5 | 67.6 | +76 / −33 |

---

# APPENDIX A. Benchmark plan

What reviewers in fungal and biodiversity image recognition expect, and where we stand. Sources:
DF20 and FungiCLEF overviews, FungiTastic, open long-tailed recognition (Liu et al. 2019),
open-set recognition (Vaze et al. 2022), calibration (Guo et al. 2017), REFORMS (Kapoor et al.
2024). ✅ done · 🟡 partial · ⬜ to do.

### Must have

| # | Benchmark | Status | Plan |
|---|---|---|---|
| A1 | Top-1/top-5 at species, genus, family, per find, on the temporal and held-out sets | 🟡 | Temporal done; held-out via `mv heldout report` |
| A2 | 95% intervals, bootstrap clustered by observer; paired McNemar tests for every comparison | 🟡 | Wilson intervals and McNemar in the report; add observer-clustered bootstrap |
| A3 | **Macro (per-species) accuracy and macro-F1** beside micro, overall and per depth band, with species and record counts per band. Macro-F1 is the headline metric of DF20 and FungiTastic | ⬜ | Add to the report; tie bands to the long-tail protocol (many / medium / few / zero) and explain why ours count records, not images |
| A4 | **Leakage control**: near-duplicate photos between test and reference, same observer and day, observer-disjoint analysis, prior's occurrence data excludes test records | 🟡 | Report has same-observer/day and identical-file breakdowns; add observer-disjoint scores and duplicate counts |
| A5 | Paired comparison with **iNaturalist CV** on the same records, with the training-contamination caveat | 🟡 | 2,000-record subsample running; snapshot iNat IDs and model version for the paper set |
| A6 | **Open-set / unknown species**: how finds of species with no reference behave: AUROC (known vs unknown) and TNR at 95% TPR from the top score, as in FungiTastic; abstention rate | ⬜ | 12% of test finds have no reference species; compute both and an "unknown species" threshold |
| A7 | **Calibration evidence**: reliability diagrams per rank, ECE and Brier on data not used to fit; likely-list coverage and size **by depth band** | 🟡 | ECE/NLL done on the fitting set; refit on development, score on test |
| A8 | **Ablations**: frozen vs fine-tuned; nearest vs species average vs classifier heads; with/without prior; generic backbones; BioCLIP v1 | 🟡 | All but BioCLIP v1 and the prior done on the temporal set; repeat on held-out |
| A9 | **Error analysis** by taxonomic distance; strict / *s.l.* / complex; top confused pairs with examples | 🟡 | Name-equivalence scoring merged; add confusion pairs and lowest-common-ancestor depth |
| A10 | **Toxic species**: errors for *Amanita* sect. *Phalloideae*, *Galerina*, *Lepiota*, *Cortinarius* sect. *Orellani*, *Gyromitra*; explicit "not for edibility" | ⬜ | List target taxa; report predicted-as and predicted-from rates |
| A11 | Data and code availability; split ids; REFORMS checklist as a supplement | ⬜ | Deposit at submission |
| A12 | **Their method on our data**: the Danish Fungi / FungiTastic recipe (whole network retrained as a classifier, BEiT-B/16 at 384 px, rare-class loss, photo probabilities averaged, month prior) trained on our records and scored on the same records, to separate the method from the data | 🟡 | Being built (replication of Picek et al.); compute estimate before any GPU run |
| A13 | **Fair comparison with models that lack provisional names**: genus and family on all records; species on formally named truths only; a same-vocabulary comparison (Vision limited to the other model's names, and unlimited); each model's share of records it cannot name at species, reported as a result; one taxonomy mapping | 🟡 | Protocol agreed; applies to iNat CV and the published Danish Fungi / FungiTastic models |
| A14 | **Per photograph and per find**: report both, as Picek's code keeps them separate | 🟡 | First-photo scores exist; add every-photo-alone scores |

### Commonly expected

| # | Benchmark | Status |
|---|---|---|
| B1 | Risk–coverage curve (accuracy when confident) and accuracy at the site's thresholds, beside iNat CV | ⬜ |
| B2 | **External test**: FungiTastic DNA-sequenced test subset and/or DF20, species shared with our references | ⬜ |
| B3 | Top-k for k = 1 to 10 (FungiCLEF 2025 reports all, top-5 primary) | 🟡 (1/3/5/10 now recorded) |
| B4 | Location/season prior gain by region and depth; does it suppress rare or out-of-range species? | ⬜ |
| B5 | **Label-noise audit**: share of "errors" that are label errors; results with and without provisional names | 🟡 |
| B6 | Compute: parameters, index size, latency, training cost | 🟡 |
| B7 | Model card and datasheet for the reference set | ⬜ |
| B9 | Cost-weighted errors: poisonous species called edible weighted 100:1 (FungiCLEF organizers' cost) | ⬜ |
| B8 | Accuracy by number of photographs per find | 🟡 |

### Nice to have

| # | Benchmark | Status |
|---|---|---|
| C1 | Expert panel and community identification on a stratified subset, against DNA | ⬜ |
| C2 | Mean taxonomic distance of mistakes | ⬜ |
| C3 | Geographic blocks / ecoregions; east vs west of 100° W | 🟡 |
| C4 | Class-conditional conformal lists tuned for rare species | ⬜ |
| C5 | Season and image-quality strata | ⬜ |
| C6 | Cost-weighted metric (FungiCLEF poisonous-confusion style) | ⬜ |
| C7 | Learning curve: accuracy vs reference-set size (rerun at the TABLE 1 dates) | ⬜ |

---

# APPENDIX B. Next tests (in order)

1. **"Before" number on the 3,000 development records** with the served model: nearest,
   nearest+mean and nearest+prior, in the standard summary (top 1/3/5/10 × strict/*s.l.*/complex;
   species by depth band; iNat on the same records). Confirms or rejects `nearest+mean`.
2. **Finish iNat CV on the 2,000-record subsample**, then on TABLE 3's records.
3. **Tune the occurrence prior** on development, check on test; count how often it penalizes the
   true species.
4. **Classify development misses by cause**: stale label, no reference, look-alike in a complex,
   photograph problem, guest organism.
5. **Measure answer-key noise** after the legacy-name refresh: rescore with corrected names.
6. **Relabel, retrain, rescore** the same development set ("after"); fine-tuning variants (more
   blocks, more epochs, metric-learning loss) on the GPU trainer. Save the fine-tune's classifier
   heads this time and score them as an identifier beside nearest specimen.
6a. **Danish Fungi method on our records** (A12) and the published Danish Fungi / FungiTastic
   models (A13) on the development set, once their compute is approved.
7. **Advance predictions**: score the 7 October batch (1,000) and the nightly 300s as they
   validate.
8. **Paper-only additions** from Appendix A: macro metrics, open-set AUROC, observer-disjoint and
   duplicate checks, toxic-species table, external FungiTastic/DF20 test, learning curve.
9. **Freeze the paper test set** (~1,000 newly validated records, registered as held out, with
   iNaturalist identifications snapshotted first); score once.

**Open questions for the authors:** which provisional-name convention to print (Mycologia's
guidance is silent); whether model weights can be released given all-rights-reserved photographs;
which external benchmark to prioritize; whether to include an expert or community comparison.

---

# APPENDIX C. Mycologia requirements (working notes; remove before submission)

Checked 8 October 2026 against Taylor & Francis's Mycologia instructions (updated 14 September
2026) and the "Mycologia Instructions for Authors 2025" guide.

- **Article type:** Original Research. Single-anonymous review, two referees. Submission via
  Editorial Manager. First submission may be in any reasonable format (bioRxiv format accepted) if
  lines and pages are numbered, double-spaced, figures high resolution, and data available to
  reviewers.
- **Length:** no word or page limit found; no abstract word limit found.
- **Order:** running head (3–5 words; authors + running head ≤ 60 characters), title in sentence
  case (no higher ranks or abbreviations), authors, ABSTRACT (one paragraph, stands alone, no
  authorities), KEY WORDS (alphabetical, not repeating the title), INTRODUCTION, MATERIALS AND
  METHODS, RESULTS, DISCUSSION, DATA AVAILABILITY STATEMENT, ACKNOWLEDGMENTS, LITERATURE CITED,
  legends.
- **Style:** American English, CSE 8th edition name–year, full journal names; software names in
  small caps, versions without "v."
- **Figures:** 8.2 or 17.1 cm wide, ≤ 23.4 cm tall with legend; photos ≥ 300 dpi (600 preferred),
  line art 600–1200 dpi; sans-serif ~12 pt; no gridlines. Tables in separate files, one-sentence
  title, no vertical lines; tables over two printed pages go to the supplement (Figshare).
- **Names:** italicize genus and below; authorities optional except where needed, at first use,
  abbreviated per Index Fungorum. No rule found for provisional names.
- **Data:** DATA AVAILABILITY STATEMENT required; sequences in GenBank/INSDC; datasets in Dryad or
  similar, not only on an author's website. No explicit code policy (GitHub + Zenodo DOI fits).
- **Declarations:** funding, disclosure (competing interests) and a declaration of generative AI
  use (required even if none; name and version of the tool, how and why it was used). AI cannot be
  an author; AI-generated images are not allowed in results.
- **Preprints:** allowed before submission (cite it; reviewers' anonymity not guaranteed).
- **Costs:** no submission or page charges; color free online, print color charged. Optional open
  access US$3,940 (no MSA discount found).
- **Journal statistics:** 302 submissions in 2025, 32% accepted, 54.5% of decisions desk
  rejections, ~41 days to first decision.
- **AI special issue in 2027:** not found on the Taylor & Francis or MSA sites or in the MSA
  council's Mycologia report (searched 8 October 2026). Confirm with the managing editor
  (mycologia@uky.edu).
- **Prior work in Mycologia:** no image-recognition identification paper found (2018–2026); no
  peer-reviewed DNA-verified test of iNaturalist's computer vision on fungi found anywhere.

---

# Changelog

- **0.2 (9 Oct 2026).** Clarified that the classifiers which lost were small heads on fixed
  features, not a fully trained classifier; added A12 (Danish Fungi method on our data), A13 (fair
  comparison with models lacking provisional names), A14 (per photo and per find), B9 (poisonous
  cost); macro-F1 moved to the must-haves; open-set adds TNR at 95% TPR. The website's Models
  page now charts every approach on the same records.
- **0.1.1 (9 Oct 2026).** Corresponding author: Stephen Russell.
- **0.1 (9 Oct 2026).** First full draft from results through the 1,152-record temporal test,
  the depth-bias and photograph experiments, and the 100-record held-out pilot. Benchmark plan
  and Mycologia notes added.
