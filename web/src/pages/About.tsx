import { PageHeader } from "@/components/Layout";

export function AboutPage() {
  return (
    <>
      <PageHeader title="How it works">
        A photo identifier built on DNA, not on community votes.
      </PageHeader>
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-3xl space-y-6 text-[#5c4a3a] leading-relaxed">
        <Section title="DNA-verified references only">
          Most photo identifiers learn from records named by community agreement, so their
          mistakes are built in. MycoMap Vision learns only from records whose name is backed by
          DNA sequencing and marked green in a MycoMap project. When a name changes on
          mycomap.org, it changes here at the next sync.
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
          Family, genus and species each get their own confidence. Often the genus is clear while
          the species is not; the page says so, and suggests what would help.
        </Section>
        <Section title="Measured on the newest weeks">
          Each model is tested on records that turned green most recently, identified using only
          the records verified before them. The Models page shows the results, broken down by
          how many reference records each species has.
        </Section>
        <Section title="What's still to come">
          Calibrated confidence, recognising which part of the mushroom each photo shows, range
          and season from DNA records, and fine-tuned models. Confidence today is relative among
          the candidates, not yet a calibrated probability.
        </Section>
      </div>
    </>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section>
      <h2 className="font-display text-2xl text-[#4a3728] mb-2">{title}</h2>
      <p>{children}</p>
    </section>
  );
}
