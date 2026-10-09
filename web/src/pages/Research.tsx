import { useQuery } from "@tanstack/react-query";
import { Link } from "wouter";
import { PageHeader } from "@/components/Layout";
import { api } from "@/lib/api";
import { BENCHMARKS, researchNav, STATUS_LABEL, type Benchmark } from "@/lib/research";

// The Research section (Steve, 2026-10-09): everything detailed, one level below the tool.

const body = "container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl text-[#5c4a3a] leading-relaxed";
const h2 = "font-display font-semibold text-xl text-[#4a3728] mt-8 mb-2";

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
    : b.status === "partial" ? "bg-[#f3eee6] text-[#7a5a3a]" : "bg-gray-100 text-gray-600";
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
          and whether the right answer had to be first or only near the top.
        </p>
        <h2 className={h2}>Keeping tests clean</h2>
        <p>
          A test record is never a reference or training record. Held-out sets are listed before
          any model sees them, and a sealed test set is scored once. Results are also broken down
          by whether the same observer recorded a reference on the same day, and whether a photo
          duplicates a reference photo.
        </p>
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
          A paper describing the method and its tests is in preparation for <em>Mycologia</em>. Until
          it is published, please cite the site: MycoMap Vision, https://vision.mycomap.org,
          accessed (date).
        </p>
        <h2 className={h2}>Test your model on our records</h2>
        <p>
          We are considering releasing the benchmark record lists (iNaturalist observation ids,
          DNA-verified names and splits), so other models can be scored on the same records with
          the same protocols. Photos would come from iNaturalist under their own licences. If this
          would help your work, tell us at{" "}
          <a href="mailto:info@mycomap.org" className="text-myco-green underline">info@mycomap.org</a>.
        </p>
        <h2 className={h2}>Where the references come from</h2>
        <p>
          Community scientists' iNaturalist observations, sequenced through MycoMap and validated
          in MycoMap projects. See{" "}
          <Link href="/research/data" className="text-myco-green underline">Reference data</Link>.
        </p>
      </div>
    </>
  );
}
