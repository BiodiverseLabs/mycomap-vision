import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/Layout";
import { api } from "@/lib/api";
import { BENCHMARKS, researchNav, STATUS_LABEL, type Benchmark } from "@/lib/research";
import { erratumReasons, shortHash, V1 } from "@/lib/datasetReleases";

// The Research section (Steve, 2026-10-09): everything detailed, one level below the tool.

const body = "container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl text-[#5c4a3a] leading-relaxed";
const h2 = "font-display font-semibold text-xl text-[#4a3728] mt-8 mb-2";
const link = "text-myco-green underline";

/** The base model's own page: credit to its authors, not a release of ours. */
const BIOCLIP2_URL = "https://huggingface.co/imageomics/bioclip-2";

/** The works the identifier builds on or is compared with, as the paper draft cites them
 *  (research repo, paper/draft.md, References). */
const CITATIONS: { key: string; text: string; href: string }[] = [
  { key: "Gu et al. 2025",
    text: "Gu J, Stevens S, Campolongo EG, et al. 2025. BioCLIP 2: emergent properties from scaling hierarchical contrastive learning. arXiv:2505.23883.",
    href: "https://arxiv.org/abs/2505.23883" },
  { key: "Stevens et al. 2024",
    text: "Stevens S, Wu J, Thompson MJ, et al. 2024. BioCLIP: a vision foundation model for the tree of life. In: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. doi:10.1109/CVPR52733.2024.01836.",
    href: "https://doi.org/10.1109/CVPR52733.2024.01836" },
  { key: "Picek et al. 2022",
    text: "Picek L, Šulc M, Matas J, Jeppesen TS, Heilmann-Clausen J, Læssøe T, Frøslev T. 2022. Danish Fungi 2020 – not just another image recognition dataset. In: Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision. p. 1525–1535. doi:10.1109/WACV51458.2022.00334.",
    href: "https://doi.org/10.1109/WACV51458.2022.00334" },
  { key: "Picek et al. 2025",
    text: "Picek L, et al. 2025. FungiTastic: a multi-modal dataset and benchmark for image categorization. arXiv:2408.13632.",
    href: "https://arxiv.org/abs/2408.13632" },
];

/** What the identifier is built on and how it was fine-tuned (Steve, 2026-10-22). The recipe
 *  is the finetunes row of the served model on the box (bioclip-2-336-ft-20261018-171420:
 *  blocks_trained 4 of 24, 2 epochs, 155,781 records, 624,785 photos, 17,077 species); the
 *  public model's 527,842 photos are from the research repo's paper/costs.csv (E23b). */
function BaseModel() {
  return (
    <>
      <h2 className={h2}>The base model: BioCLIP 2</h2>
      <p>
        MycoMap Vision is built on BioCLIP 2 (Gu et al. 2025), an image model for the tree of life
        from the Imageomics Institute, the successor of BioCLIP (Stevens et al. 2024). Its weights
        are at <a href={BIOCLIP2_URL} className={link} target="_blank" rel="noreferrer">huggingface.co/imageomics/bioclip-2</a>.
      </p>
      <h2 className={h2}>How it was fine-tuned</h2>
      <p>
        The last 4 of BioCLIP 2's 24 transformer blocks were trained for 2 epochs on dataset release{" "}
        {V1.id}'s training split: 155,781 DNA-verified records, 624,785 photos and 17,077 species,
        with classifiers at species, genus and family. The site's model (BioCLIP 2, 336 px,
        fine-tuned, public + non-public training data) learned from all of those photos; the
        paper's headline model (BioCLIP 2, 336 px, fine-tuned, open data only) was trained the same
        way on the 527,842 Creative Commons photos among them. Earlier tests used the same recipe at
        224 px.
      </p>
      <p className="mt-2">
        "Frozen BioCLIP 2" in the results means BioCLIP 2's own image vectors, unchanged, used to
        find the nearest DNA-verified records; it is not BioCLIP 2's zero-shot naming from text
        prompts.
      </p>
    </>
  );
}

function References() {
  return (
    <>
      <h2 className={h2}>References</h2>
      <ul className="space-y-2 text-sm">
        {CITATIONS.map((c) => (
          <li key={c.key}>
            {c.text}{" "}
            <a href={c.href} className={link} target="_blank" rel="noreferrer">Link</a>
          </li>
        ))}
      </ul>
    </>
  );
}

export function ResearchPage() {
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  return (
    <>
      <PageHeader title="Research">
        How MycoMap Vision is tested, on what records, and how it compares with other approaches.
        Every reference record is backed by a DNA barcode reviewed in a MycoMap validation project.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 grid gap-3 md:grid-cols-2 max-w-5xl">
        {researchNav(me.data).map((n) => (
          <Link key={n.href} href={n.href}
                className="block rounded-lg border border-[#A87146]/20 bg-white p-4 hover:border-myco-green/50 hover:bg-myco-green/5">
            <span className="font-semibold text-[#4a3728]">{n.label}</span>
            <p className="text-sm text-muted-foreground mt-1">{n.blurb}</p>
          </Link>
        ))}
      </div>
    </>
  );
}

function StatusBadge({ b }: { b: Benchmark }) {
  const tone = b.status === "running" ? "bg-myco-green/10 text-myco-green"
    : b.status === "partial" ? "bg-[#f3eee6] text-[#7a5a3a]"
    : b.status === "history" ? "border border-dashed border-gray-300 text-gray-500" : "bg-gray-100 text-gray-600";
  return <span className={`rounded px-2 py-0.5 text-xs font-medium ${tone}`}>{STATUS_LABEL[b.status]}</span>;
}

export function BenchmarksPage() {
  return (
    <>
      <PageHeader title="Benchmarks">
        The named test sets. Each measures one thing, on DNA-verified records the identifier did
        not learn from, and is scored the same way every time (see{" "}
        <Link href="/research/protocols" className="text-myco-green underline">Evaluation protocols</Link>).
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl space-y-3">
        {BENCHMARKS.map((b) => (
          <Link key={b.slug} href={`/research/benchmarks/${b.slug}`}
                className="block rounded-lg border border-[#A87146]/20 bg-white p-4 hover:border-myco-green/50 hover:bg-myco-green/5">
            <span className="flex flex-wrap items-center gap-2">
              <span className="font-semibold text-[#4a3728]">{b.name}</span>
              <StatusBadge b={b} />
            </span>
            <p className="text-sm text-muted-foreground mt-1">{b.summary}</p>
          </Link>
        ))}
      </div>
    </>
  );
}

export function BenchmarkPage({ slug }: { slug: string }) {
  const b = BENCHMARKS.find((x) => x.slug === slug);
  if (!b) return <PageHeader title="Benchmark not found" />;
  return (
    <>
      <PageHeader title={b.name}>{b.summary}</PageHeader>
      <div className={body}>
        <p className="flex items-center gap-2"><span className="text-sm">Status:</span> <StatusBadge b={b} /></p>
        <h2 className={h2}>Why it matters</h2>
        <p>{b.whyItMatters}</p>
        <h2 className={h2}>Records</h2>
        <p>{b.records}</p>
        <h2 className={h2}>How it is scored</h2>
        <p>{b.scoring}</p>
        <h2 className={h2}>Results</h2>
        {b.results ? (
          <p><Link href={b.results.href} className="text-myco-green underline">{b.results.label}</Link></p>
        ) : (
          <p>Not measured yet.</p>
        )}
        {b.elsewhere && (
          <>
            <h2 className={h2}>Elsewhere</h2>
            <p>{b.elsewhere}</p>
          </>
        )}
        <p className="mt-8 text-sm"><Link href="/research/benchmarks" className="text-myco-green underline">All benchmarks</Link></p>
      </div>
    </>
  );
}

const METRICS: [string, string][] = [
  ["Top 1, 3, 5 and 10", "The right name is the first answer, or among the first 3, 5 or 10. Reported at species, genus and family, for all photos of a find together, and for the first photo alone."],
  ["Photos only, and with place and date", "Reported apart: the photos alone, and the photos with the find's public place and date, so what place and date add is visible."],
  ["Species strict", "The answer names exactly the DNA-verified species, written either way (a temporary code counts like a published name; spellings of the same code are one name)."],
  ["Species sensu lato", "Also right when the genus was split recently (Cortinarius s.l., Inocybe s.l.) or only the gender ending differs. Beside strict, never instead of it."],
  ["Species complex (beta)", "Also right within the same species complex. Shown for context while the rule is tested."],
  ["Genus strict and sensu lato", "The same at genus."],
  ["By reference depth", "Species results by how many DNA-verified records the true species has: 0, 1-4, 5-19, 20-99, 100+. A species with no record can't be named by any approach that learns from references."],
  ["Outside models on the same records", "iNaturalist's computer vision (photo only, and with location) and other published models are scored on exactly the records ours are, or not shown."],
  ["Intervals and paired tests", "95% intervals (Wilson; observer-clustered bootstrap where shown) and McNemar tests when two approaches answer the same records."],
  ["Calibration", "Whether a stated confidence of 80% is right about 80% of the time, measured on records not used to set it; the coverage of each list of likely names."],
];

export function ProtocolsPage() {
  return (
    <>
      <PageHeader title="Evaluation protocols">
        How every MycoMap Vision result is scored, so results from different tests, models and
        months can be compared.
      </PageHeader>
      <div className={body}>
        <h2 className={h2}>The answer key</h2>
        <p>
          A record's name is its observation name on MycoMap, kept current from the DNA-validation
          review, not the name of a linked sequence. Records whose DNA name is an organism living
          inside the photographed fungus are left out, because the photos show the host.
        </p>
        {V1.erratum && (
          <p className="mt-2" data-testid="text-protocols-erratum">
            A record found faulty after a dataset release was cut is listed in its erratum and left
            out of the references and of scoring; the release itself never changes. Release{" "}
            {V1.id}'s erratum {V1.erratum.id} lists {V1.erratum.records.toLocaleString("en-US")}{" "}
            ({erratumReasons(V1.erratum)}).
          </p>
        )}
        <h2 className={h2}>What is reported, every time</h2>
        <div className="overflow-x-auto rounded-lg border border-[#A87146]/20 my-3">
          <table className="w-full text-sm">
            <tbody>
              {METRICS.map(([what, how]) => (
                <tr key={what} className="border-t first:border-t-0 align-top">
                  <td className="px-3 py-2 font-medium text-[#4a3728] whitespace-nowrap">{what}</td>
                  <td className="px-3 py-2">{how}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <h2 className={h2}>No number without its test</h2>
        <p>
          Every accuracy figure on this site names the records it was measured on and how many,
          whether the right answer had to be first or only near the top, and the model: its base,
          its image size (224 or 336 px), and whether it is frozen or fine-tuned, and on what
          training data.
        </p>
        <BaseModel />
        <h2 className={h2}>Keeping tests clean</h2>
        <p>
          A test record is never a reference or training record. Held-out sets are listed before
          any model sees them, and a sealed test set is scored once. Results are also broken down
          by whether the same observer recorded a reference on the same day, and whether a photo
          duplicates a reference photo.
        </p>
        <References />
      </div>
    </>
  );
}

export function ForResearchersPage() {
  return (
    <>
      <PageHeader title="For researchers">
        Citing MycoMap Vision, and testing your own model against DNA-verified records.
      </PageHeader>
      <div className={body}>
        <h2 className={h2}>How to cite</h2>
        <p>
          A paper describing the method and its tests is in preparation. Until
          it is published, please cite the site: MycoMap Vision, https://vision.mycomap.org,
          accessed (date), and the base model it is built on, BioCLIP 2 (Gu et al. 2025).
        </p>
        <BaseModel />
        <h2 className={h2}>Test your model on our records</h2>
        <p>
          Every result now comes from dataset release {V1.id}, frozen with release hash{" "}
          <span className="font-mono">{shortHash(V1.releaseHash)}…</span> (see{" "}
          <Link href="/research/data" className="text-myco-green underline">Reference data</Link>). Its
          public variant (Creative Commons photos only) is planned with the paper, so other
          models can be scored on the same records with the same protocols. If this would help your work, tell us at{" "}
          <a href="mailto:info@mycomap.org" className="text-myco-green underline">info@mycomap.org</a>.
        </p>
        <h2 className={h2}>Where the references come from</h2>
        <p>
          Community scientists' iNaturalist observations, sequenced through MycoMap and validated
          in MycoMap projects. See{" "}
          <Link href="/research/data" className="text-myco-green underline">Reference data</Link>.
        </p>
        <References />
      </div>
    </>
  );
}
