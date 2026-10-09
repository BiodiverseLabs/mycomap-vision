import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Check } from "lucide-react";
import { PageHeader } from "@/components/Layout";
import { SafetyNote } from "@/components/SafetyNote";
import { api, num, pct, type Stats } from "@/lib/api";
import {
  headline, JOIN_URL, modelName, recentWeekLine, releaseNote, SEQUENCING_URL, shareWords,
} from "@/lib/publicView";

const DANISH_STUDY_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8779018/";

export function AboutPage() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const s = stats.data;
  return (
    <>
      <PageHeader title="How it works">
        Identify a fungus from all your photos, compared with DNA-sequenced specimens, not
        community votes.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl space-y-6 text-[#5c4a3a] leading-relaxed">
        <SafetyNote />
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
        <Provisional s={s} />
        <Section title="An answer at every rank">
          Family, genus and species each get their own confidence, calibrated on the newest test
          records so it can be read as a rough probability. Often the genus is clear while the
          species is not; the page says so, and suggests what would help.
        </Section>
        <FirstStep />
        <HowGood />
        <Section title="Measured on the newest weeks">
          Each model is tested on the records that turned green most recently, identified using
          only the records verified before them, the way it meets a new find.{" "}
          <a className="underline" href="/research/results">Results</a> shows every test, broken
          down by how many reference records each species has.
        </Section>
        <Standards />
        <Contributors s={s} />
        <BecomesReference s={s} />
        <Openness />
        <Section title="What's still to come">
          Recognising which part of the mushroom each photo shows, range and season from DNA
          records, and a reference set that keeps growing as more records are sequenced.
        </Section>
        <Block title="Free, from the nonprofit MycoMap" testId="section-nonprofit">
          <p>
            MycoMap Vision is free to use. It is made by MycoMap, a nonprofit, and built on the
            records its members and projects have sequenced. To support the work, become a
            member or donate at <ExternalLink href={JOIN_URL}>mycomap.org/join</ExternalLink>.
            Questions: <a className="underline" href="mailto:info@mycomap.org">info@mycomap.org</a>.
          </p>
        </Block>
      </div>
    </>
  );
}

function Provisional({ s }: { s?: Stats }) {
  const p = s?.names_provisional;
  const share = s && p != null && s.names > 0 ? { p, n: s.names } : null;
  return (
    <Block title={share ? `${shareWords(share.p, share.n)} of the names are provisional species`
                        : "Many names are provisional species"} testId="section-provisional">
      <p>
        {share
          ? <>{num(share.p)} of the {num(share.n)} names in the reference set ({pct(share.p / share.n)}) are </>
          : "Many names in the reference set are "}
        temporary codes, written like <span className="sci">Russula</span> sp. 'IN01'. Each is
        a species told apart by its DNA that has not been formally described yet. An identifier
        that learns from community names can't suggest them: there is no name to vote on.
      </p>
      <p className="mt-3">
        When Vision suggests one, your find may belong to a species still waiting to be
        described. The code is a working label, not a formal name, and it changes when the
        species is published.
      </p>
    </Block>
  );
}

function FirstStep() {
  return (
    <Block title="A first step, not the last word" testId="section-first-step">
      <ol className="list-decimal pl-5 space-y-1">
        <li>Vision suggests names, with how likely each is.</li>
        <li>You compare your find with the DNA-verified specimens it shows, photo by photo.</li>
        <li>
          DNA settles it. MycoMap offers{" "}
          <ExternalLink href={SEQUENCING_URL}>free DNA sequencing</ExternalLink> for finds like
          yours.
        </li>
      </ol>
      <p className="mt-3">
        Vision learned from DNA-verified records, but its answer for your find comes from photos
        alone. People checking the suggestions make the difference: the Danish Fungal Atlas
        found that in its app, the AI's first suggestion alone was right for 69% of
        submissions, and with people choosing among the suggestions and experts checking them,
        93% were right (<ExternalLink href={DANISH_STUDY_URL}>Picek et al., Sensors 2022</ExternalLink>).
      </p>
    </Block>
  );
}

const STANDARDS = [
  "Labels come only from records DNA-validated (marked green) in a MycoMap project.",
  "No community identifications from iNaturalist or Mushroom Observer are used as labels.",
  "Names follow renames on mycomap.org every night, and records no longer validated drop out.",
  "Tested on the newest weeks of records, never on a random split, and ahead of the DNA on records still waiting for their sequence.",
  "Every accuracy figure comes with its test set and rank.",
  "North America only: a find from elsewhere is compared with North American species.",
  "Photos are shown only when Creative Commons licensed or their photographer has given permission.",
  "Locations hidden on iNaturalist stay hidden.",
];

function Standards() {
  return (
    <Block title="Our standards" testId="section-standards">
      <ul className="space-y-2">
        {STANDARDS.map((t) => (
          <li key={t} className="flex gap-2">
            <Check className="h-4 w-4 mt-1 shrink-0 text-myco-green" aria-hidden />
            <span>{t}</span>
          </li>
        ))}
      </ul>
    </Block>
  );
}

function Contributors({ s }: { s?: Stats }) {
  return (
    <Block title="Built by contributors" testId="section-contributors">
      <p>
        Vision exists because people found, photographed and sequenced these fungi.
        {s && (
          <>
            {" "}Its reference set holds {num(s.records_north_america)} DNA-verified North
            American records
            {s.photographers ? <>, photographed by {num(s.photographers)} people</> : null}
            {s.projects ? <> and marked green in {num(s.projects)} MycoMap projects</> : null}.
          </>
        )}
      </p>
      <p className="mt-3">
        The photos come from iNaturalist, where the finds were posted, and stay their
        photographers' own.
        {s && s.permissions.granted > 0 && (
          <> Thank you to the {num(s.permissions.granted)} photographers who gave permission for
          their photos to be shown here.</>
        )}
      </p>
    </Block>
  );
}

function BecomesReference({ s }: { s?: Stats }) {
  const week = recentWeekLine(s?.recent_week);
  return (
    <Block title="Your sequenced find becomes a reference" testId="section-becomes-reference">
      <p>
        When a find is sequenced and its DNA name is validated in a MycoMap project, it joins
        the reference set the next night, with its photos. From then on Vision can match the
        next find like it.
        {week && ` ${week}`}{" "}
        <ExternalLink href={SEQUENCING_URL}>How to get a find sequenced</ExternalLink>.
      </p>
    </Block>
  );
}

function Openness() {
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const served = models.data?.backbones.filter((b) => b.embedded_photos) ?? [];
  return (
    <Block title="Open about how it's made" testId="section-openness">
      <ul className="list-disc pl-5 space-y-2">
        <li>
          A paper describing Vision is in preparation. Until it is out, please cite MycoMap
          Vision (vision.mycomap.org) with the date you used it.
        </li>
        <li>
          Photo licences, and which photographers have given permission, are on the{" "}
          <a className="underline" href="/research/data">Reference data page</a>.
        </li>
        <li>
          Every test of every model is on the{" "}
          <a className="underline" href="/research/results">Results page</a>, and how we test is
          in the <a className="underline" href="/research/protocols">evaluation protocols</a>.
        </li>
        {served.map((b) => (
          <li key={b.backbone} title={b.backbone} data-testid="text-release-note">
            Model in use: {releaseNote(b)}
          </li>
        ))}
      </ul>
    </Block>
  );
}

function ExternalLink({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a className="underline" href={href} target="_blank" rel="noreferrer">
      {children}
      <ArrowUpRight className="inline h-3 w-3 ml-0.5 align-baseline" aria-hidden />
    </a>
  );
}

function Block({ title, testId, children }: { title: string; testId?: string; children: React.ReactNode }) {
  return (
    <section data-testid={testId}>
      <h2 className="font-display font-semibold text-2xl text-[#4a3728] mb-2">{title}</h2>
      {children}
    </section>
  );
}

/** The served model's newest measured results, with iNat on the same records when tested. */
function HowGood() {
  const board = useQuery({ queryKey: ["scoreboard"], queryFn: api.scoreboard });
  const models = useQuery({ queryKey: ["models"], queryFn: api.models });
  const ahead = useQuery({ queryKey: ["prospective"], queryFn: api.prospective });
  const h = board.data && models.data ? headline(board.data.runs, models.data.ready, models.data.default_method ?? "nearest") : null;
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
