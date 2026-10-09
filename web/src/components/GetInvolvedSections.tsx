import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, Camera, Dna, FlaskConical, HandHeart, ImageUp, Users } from "lucide-react";
import { api, num } from "@/lib/api";
import {
  JOIN_URL, PHOTO_CHECKLIST, recentWeekLine, RESEARCH_PATH, SEQUENCING_URL,
} from "@/lib/publicView";

// The Get involved page's content: ways to help, each with a real mycomap.org page, and
// real counts from /api/stats where they help (left out when the server doesn't send them).

const ORG = "https://mycomap.org";

export function GetInvolvedSections() {
  const stats = useQuery({ queryKey: ["stats"], queryFn: api.stats });
  const s = stats.data;
  const week = recentWeekLine(s?.recent_week);
  return (
    <div className="space-y-8 text-[#5c4a3a] leading-relaxed" data-testid="get-involved-sections">
      <Way icon={Camera} title="Take photos that identify" testId="way-photos">
        <p>
          A good set of photos helps you, Vision and everyone who checks your find later.
          Photograph each find from every side:
        </p>
        <ul className="mt-2 grid gap-1.5 sm:grid-cols-2">
          {PHOTO_CHECKLIST.map((c) => (
            <li key={c.what}>
              <span className="font-medium">{c.what}</span>
              <span className="text-muted-foreground">: {c.why}</span>
            </li>
          ))}
        </ul>
        <p className="mt-2">
          Post them to <Out href="https://www.inaturalist.org">iNaturalist</Out>: the photos of
          MycoMap's records come from there. Then <a className="underline" href="/">identify
          your find</a> here from all of them.
        </p>
      </Way>

      <Way icon={Dna} title="Get your find DNA-sequenced" testId="way-sequencing">
        <p>
          MycoMap offers <Out href={SEQUENCING_URL}>free DNA sequencing</Out>. Once your find's
          DNA name is validated in a MycoMap project, it joins Vision's reference set the next
          night, with your photos, and helps name the next find like it.
          {week && ` ${week}`}
        </p>
      </Way>

      <Way icon={Users} title="Join or start a MycoMap project" testId="way-projects">
        <p>
          Projects collect, sequence and validate the records Vision learns from
          {s?.projects ? <>: {num(s.projects)} projects have records in the reference set so far</> : null}.
          {" "}<Out href={`${ORG}/projects`}>Find a project</Out> near you, or{" "}
          <Out href={`${ORG}/start-project`}>start one</Out> for your club, park or survey.
        </p>
      </Way>

      <Way icon={ImageUp} title="Let your photos be shown" testId="way-permission">
        <p>
          Vision shows a reference photo only when it is Creative Commons licensed or its
          photographer has said yes.
          {s && s.contributors_arr > 0 && (
            <> {num(s.contributors_arr)} photographers have all-rights-reserved photos in the
            reference set.</>
          )}
          {" "}If you are one of them, you can <Out href={`${ORG}/photo-permission`}>give
          permission on mycomap.org</Out>, and change your mind at any time.
        </p>
      </Way>

      <Way icon={HandHeart} title="Support the nonprofit" testId="way-support">
        <p>
          Vision is free, from the nonprofit MycoMap. <Out href={JOIN_URL}>Become a member or
          donate</Out> to keep it that way.
        </p>
      </Way>

      <Way icon={FlaskConical} title="For researchers" testId="way-research">
        <p>
          How each model is tested, the results and the reference data are in the{" "}
          <a className="underline" href={RESEARCH_PATH}>Research section</a>. Questions or
          collaborations: <a className="underline" href="mailto:info@mycomap.org">info@mycomap.org</a>.
        </p>
      </Way>
    </div>
  );
}

function Way({ icon: Icon, title, testId, children }: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  testId: string;
  children: React.ReactNode;
}) {
  return (
    <section className="flex gap-4" data-testid={testId}>
      <span className="mt-1 flex h-9 w-9 shrink-0 items-center justify-center rounded-full bg-myco-green/10">
        <Icon className="h-5 w-5 text-myco-green" />
      </span>
      <div className="min-w-0">
        <h2 className="font-display font-semibold text-xl text-[#4a3728] mb-1">{title}</h2>
        {children}
      </div>
    </section>
  );
}

function Out({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a className="underline" href={href} target="_blank" rel="noreferrer">
      {children}
      <ArrowUpRight className="inline h-3 w-3 ml-0.5 align-baseline" aria-hidden />
    </a>
  );
}
