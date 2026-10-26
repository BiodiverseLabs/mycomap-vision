import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { CalendarClock, CalendarDays, ImageOff, MapPin, Microscope, RotateCcw, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { PageHeader } from "@/components/Layout";
import { api, ApiError, runApi, signInUrl, type ContextUsed, type RunView } from "@/lib/api";
import { localWhen } from "@/lib/keptRun";
import { sourceName } from "@/lib/observationLink";
import { publicModelName } from "@/lib/publicNames";
import { runPhotos } from "@/lib/runLink";
import { OthersRun, ResultCard, WhatNext } from "@/pages/Identify";
import { InatSuggestions } from "@/components/InatSuggestions";

/** A run's own link, /r/<id>: the answer as it was given, read-only, to anyone with the
 *  link. Whoever ran it also gets their own part and Delete; signed-in viewers can identify
 *  the photos again on their own Identify page. */
export function RunPage({ id }: { id: string }) {
  const me = useQuery({ queryKey: ["me"], queryFn: api.me, retry: false });
  const run = useQuery({ queryKey: ["run", id], queryFn: () => runApi.open(id), retry: false });
  const [deleted, setDeleted] = useState(false);

  if (run.isLoading) return null;
  if (deleted || !run.data) {
    const missing = deleted || (run.error instanceof ApiError && run.error.status === 404);
    return (
      <>
        <PageHeader title={deleted ? "Deleted" : missing ? "Result not found" : "Couldn't open this result"}>
          {deleted
            ? "This result and its link are gone, with the copies of the photos."
            : missing
              ? "This link was deleted, or it was never made."
              : (run.error as Error | null)?.message ?? ""}
        </PageHeader>
        <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 max-w-xl">
          <SignInToIdentify signedIn={!!me.data?.user} />
        </div>
      </>
    );
  }
  const r = run.data;
  // Anyone may identify on a site without sign-in; otherwise only a signed-in person.
  const signedIn = !!me.data?.user || me.data?.signin === "off";
  const remove = async () => {
    if (!window.confirm("Delete this result? The link stops working for everyone who has it.")) return;
    await runApi.remove(id);
    setDeleted(true);
  };
  return (
    <>
      <PageHeader title="An identification from MycoMap Vision">
        Vision compares photos with DNA-sequenced specimens. This is one find's answer, as it was given;
        it doesn't change.
      </PageHeader>
      {/* The answer first, its details beside it (below it on a phone), as on Identify
          (Steve's approved redesign, 2026-10-26). */}
      <div className="container mx-auto px-4 sm:px-6 lg:px-8 py-8 grid items-start gap-6 lg:gap-7 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="order-2 space-y-6 lg:sticky lg:top-20">
          <RunFacts r={r} />
          {signedIn ? <AgainCard r={r} /> : <SignInToIdentify signedIn={false} />}
          {r.can_delete && (
            <div className="rounded-lg border border-[#A87146]/20 p-4 text-sm text-[#5c4a3a]" data-testid="card-run-delete">
              <p>
                {r.mine ? "You ran this." : "As an admin you can delete this."} Deleting it removes the answer
                {r.kind === "photos" ? " and the copies of the photos" : ""} from Vision, and the link stops working.
              </p>
              <Button variant="outline" size="sm" className="mt-3 text-destructive" onClick={remove}
                      data-testid="button-run-delete">
                <Trash2 /> Delete this result
              </Button>
            </div>
          )}
        </div>
        <div className="min-w-0">
          <div className="mb-4"><RunContext used={r.context_used} /></div>
          <OthersRun.Provider value={!r.mine}>
            <div className={`grid gap-6 ${r.results.length > 1 ? "xl:grid-cols-2" : ""}`}>
              {r.results.map((res, i) => (
                <ResultCard key={`${res.model.backbone}/${res.model.method}`} r={res} urls={runPhotos(r)}
                            photos={r.kind === "photos" ? r.photos : r.observation?.photos_used ?? res.photos}
                            runId={i === 0 ? r.id : undefined} />
              ))}
            </div>
            {r.mine && r.inat && <InatSuggestions initial={r.inat} />}
            <WhatNext results={r.results} />
          </OthersRun.Provider>
        </div>
      </div>
    </>
  );
}

/** The run's photos, and when and with what it was answered. */
function RunFacts({ r }: { r: RunView }) {
  const o = r.observation;
  const urls = runPhotos(r);
  const model = r.results[0]?.model;
  return (
    <Card>
      <CardContent className="pt-5 space-y-3 text-sm text-[#5c4a3a]">
        {o ? (
          <p>
            {o.url
              ? <a href={o.url} target="_blank" rel="noreferrer" className="font-medium underline">{sourceName(o.source)} {o.id}</a>
              : <span className="font-medium">{sourceName(o.source)} {o.id}</span>}
            {o.name_at_source && <> · called <span className="sci">{o.name_at_source}</span> there</>}
          </p>
        ) : (
          <p className="font-medium text-[#4a3728]">{urls.length} photo{urls.length === 1 ? "" : "s"} of one find</p>
        )}
        <div className="grid grid-cols-3 gap-2" data-testid="run-photos">
          {urls.map((u, i) => <RunPhoto key={i} url={u} n={i + 1} source={o ? sourceName(o.source) : null} />)}
        </div>
        <ul className="space-y-1 text-xs text-muted-foreground" data-testid="run-facts">
          <li className="flex items-center gap-1.5">
            <CalendarClock className="h-3.5 w-3.5" aria-hidden /> Answered {localWhen(r.answered_at)}
          </li>
          <li className="flex items-center gap-1.5">
            <Microscope className="h-3.5 w-3.5" aria-hidden />
            {model ? publicModelName(model.backbone) : "MycoMap Vision"}{r.release ? `, release ${r.release}` : ""}
          </li>
        </ul>
      </CardContent>
    </Card>
  );
}

/** One photo. An observation's photo is shown from its source, as the Identify page shows it;
 *  one that isn't Creative Commons, or was deleted or made private there since, is a
 *  numbered placeholder saying so. */
function RunPhoto({ url, n, source }: { url: string; n: number; source: string | null }) {
  const [gone, setGone] = useState(false);
  if (!url || gone) {
    return (
      <div className="aspect-square w-full rounded-md border bg-muted flex flex-col items-center justify-center gap-1 p-1 text-center text-[11px] leading-tight text-muted-foreground"
           data-testid="run-photo-missing">
        <ImageOff className="h-4 w-4" aria-hidden />
        Photo {n}
        <span>{!url ? "not Creative Commons" : source ? `no longer at ${source}` : "not available"}</span>
      </div>
    );
  }
  return <img src={url} alt={`Photo ${n}`} onError={() => setGone(true)}
              className="aspect-square w-full rounded-md border object-cover" />;
}

/** Identify these photos again (a fresh answer on the viewer's own Identify page; this one
 *  stays as it is), or open the observation on Identify. */
function AgainCard({ r }: { r: RunView }) {
  const o = r.observation;
  const href = o?.url ? `/?observation=${encodeURIComponent(o.url)}` : `/?run=${encodeURIComponent(r.id)}`;
  return (
    <Card>
      <CardContent className="pt-5 space-y-3 text-sm text-[#5c4a3a]">
        <p className="font-semibold text-[#4a3728]">Run it again</p>
        <p>
          {o
            ? "Opens this observation on Identify, where Run again gives today's answer. This one stays as it is."
            : "Puts these photos, place and date on your Identify page to identify again with today's Vision. This one stays as it is."}
        </p>
        <Button asChild className="w-full" data-testid="button-run-again-from-link">
          <a href={href}><RotateCcw /> {o ? "Open on Identify" : "Identify these photos again"}</a>
        </Button>
      </CardContent>
    </Card>
  );
}

function SignInToIdentify({ signedIn }: { signedIn: boolean }) {
  return (
    <Card className="border-myco-green/40 bg-myco-green/5" data-testid="card-run-signin">
      <CardContent className="pt-5 space-y-3 text-sm text-[#5c4a3a]">
        <p className="font-semibold text-[#4a3728] text-base">Identify your own find</p>
        <p>
          MycoMap Vision is free from the nonprofit MycoMap.
          {signedIn ? "" : " Sign in with a free mycomap.org account to identify your own photos."}
        </p>
        <Button asChild className="w-full">
          <a href={signedIn ? "/" : signInUrl()}>{signedIn ? "Identify a fungus" : "Sign in with mycomap.org"}</a>
        </Button>
      </CardContent>
    </Card>
  );
}

/** The place and date the answer used, as the Identify page shows them (place to 0.1°). */
function RunContext({ used }: { used: ContextUsed }) {
  if (!used.place_from && !used.date_from) {
    return <p className="text-sm text-muted-foreground">From the photos alone (no place or date given).</p>;
  }
  return (
    <p className="text-sm text-muted-foreground flex flex-wrap gap-x-4 gap-y-1" data-testid="run-context">
      {used.place_from && used.latitude != null && (
        <span className="inline-flex items-center gap-1"><MapPin className="h-3.5 w-3.5" /> near {used.latitude}, {used.longitude}</span>
      )}
      {used.date_from && used.observed_on && (
        <span className="inline-flex items-center gap-1"><CalendarDays className="h-3.5 w-3.5" /> {used.observed_on}</span>
      )}
    </p>
  );
}
