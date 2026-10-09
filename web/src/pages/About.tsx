import { useQuery } from "@tanstack/react-query";
import { PageHeader } from "@/components/Layout";
import { api, num, pct } from "@/lib/api";
import { headline, modelName } from "@/lib/publicView";

export function AboutPage() {
  return (
    <>
      <PageHeader title="How it works">
        A photo identifier built on DNA, not on community votes.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl space-y-6 text-[#5c4a3a] leading-relaxed">
        <Section title="DNA-verified references only - Updated daily">
          Most photo identifiers learn from records named by community agreement, so their
          mistakes are built in. MycoMap Vision learns only from records whose name is backed by
          DNA sequencing and marked green in a MycoMap project. Every night it picks up the
          records validated that day, drops any no longer validated, and follows renames on
          mycomap.org.
        </Section>
        <Section title="Every photo, not one">
          A mushroom often can't be named from its cap alone. Each of your photos is compared
          with every photo of every DNA-verified record, and each species is scored by how well
          its best-matching specimen fits your photos as a set. Adding an underside or stem photo
          can change the answer.
        </Section>
        <Section title="Rare species are not drowned out">
          Almost half the names in the reference set have a single DNA-verified record. Because a
          species is scored by its closest specimen, not by how many records it has, a species
          known from one specimen competes on equal terms with one known from a thousand. The
          answer says when a match rests on a single specimen.
        </Section>
        <Section title="An answer at every rank">
          Family, genus and species each get their own confidence, calibrated on the newest test
          records so it can be read as a rough probability. Often the genus is clear while the
          species is not; the page says so, and suggests what would help.
        </Section>
        <HowGood />
        <Section title="Measured on the newest weeks">
          Each model is tested on the records that turned green most recently, identified using
          only the records verified before them, the way it meets a new find. The Results page under Research
          shows every test, broken down by how many reference records each species has.
        </Section>
        <Section title="What's still to come">
          Recognising which part of the mushroom each photo shows, range and season from DNA
          records, and a reference set that keeps growing as more records are sequenced.
        </Section>
      </div>
    </>
  );
}

/** The served model's newest measured results, with iNat on the same records when tested. */
function HowGood() {
  const board = useQuery({ queryKey: ["scoreboard"], queryFn: api.scoreboard });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const ahead = useQuery({ queryKey: ["prospective"], queryFn: api.prospective });
  const h = board.data && models.data ? headline(board.data.runs, models.data.ready) : null;
  const advance = h ? ahead.data?.models.find((m) => m.backbone === h.ours.backbone) : undefined;
  return (
    <section data-testid="section-how-good">
      <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-2">How good is it</h2>
      {!h ? (
        <p>{board.isLoading || models.isLoading ? "Loading the latest results…" : "Results for this model are not published yet."}</p>
      ) : (
        <>
          <p>
            On the newest {num(h.nTest)} DNA-verified records (validated after {h.cutoff}),
            identified from the {num(h.nReference)} records verified before them, the first
            answer was right for:
          </p>
          <div className="grid grid-cols-3 gap-3 my-4 text-center">
            <Score rank="Species" value={h.ours.species_top1} />
            <Score rank="Genus" value={h.ours.genus_top1} />
            <Score rank="Family" value={h.ours.family_top1} />
          </div>
          {h.inat && (
            <p>
              {modelName("external:inat-cv")}, given the same photos
              {h.inat.method === "combined-max" ? " and locations" : ""}, was right for{" "}
              {pct(h.inat.species_top1, 1)} at species, {pct(h.inat.genus_top1, 1)} at genus and{" "}
              {pct(h.inat.family_top1, 1)} at family.
            </p>
          )}
          <p className="mt-3">
            Species is the hard rank: many fungi are told apart only by DNA, and a good share of
            the names here are new, provisional ones that no photo guide shows yet. Genus is
            right far more often, and the page tells you when the species is uncertain.
          </p>
          {advance && advance.predicted > 0 && (
            <p className="mt-3">
              It is also tested ahead of the DNA: every night it names records still waiting for
              their sequence, and checks each answer once the DNA name arrives.{" "}
              {num(advance.predicted)} named so far
              {advance.resolved ? `, ${num(advance.resolved)} with their DNA answer in` : ", none with their DNA answer in yet"}.
            </p>
          )}
          <p className="mt-3 text-sm text-muted-foreground" title={h.ours.backbone}>
            Model: {modelName(h.ours.backbone)}, nearest specimen. Test {h.comparisonId}.
          </p>
        </>
      )}
    </section>
  );
}

function Score({ rank, value }: { rank: string; value: number | null }) {
  return (
    <div className="rounded-lg border border-[#A87146]/20 bg-[#faf9f7] py-3">
      <div className="text-2xl font-semibold text-[#4a3728] tabular-nums">{pct(value, 1)}</div>
      <div className="text-xs uppercase tracking-wider text-muted-foreground mt-1">{rank}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-2">{title}</h2>
      <p>{children}</p>
    </section>
  );
}
