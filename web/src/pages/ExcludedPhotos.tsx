import { useEffect, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { ArrowUpRight, ImageOff, ScanEye, X } from "lucide-react";
import type { Me } from "@/lib/api";
import { ApiError } from "@/lib/api";
import { cropBox, POLL_MS, unseenStrips } from "@/lib/whereLooked";
import {
  categoryLabel, excludedApi, HELD_LINE, MAP_MODEL_LABEL, PAGE_SIZE, rangeLine, rankLine, refusal,
  type ExcludedLooked, type ExcludedPage, type ExcludedTile,
} from "@/lib/excludedPhotos";
import { nf } from "@/lib/researchView";
import { ResearchHero, ResearchShell, Section } from "@/components/ResearchKit";
import { MapLayer } from "@/components/WhereLooked";
import { useMe } from "@/components/researchQueries";

// Research > Excluded photos (Steve, 2026-10-26): every photo people left out of Vision in Human
// Feedback, by why, and "where Vision looked" on any of them, for looking at attention in its
// own right. Only photos anyone may see are shown (feedback_public.py's rules); who left a photo
// out, and their notes, are never shown.

export function ExcludedPhotosPage() {
  const me = useMe();
  const [category, setCategory] = useState("all");
  const [offset, setOffset] = useState(0);
  const [withMaps, setWithMaps] = useState(false);
  const list = useQuery({
    queryKey: ["excluded-photos", category, offset, withMaps],
    queryFn: () => excludedApi.list(category, offset, withMaps),
    placeholderData: (prev) => prev,
  });
  return (
    <ExcludedPhotos me={me.data} page={list.data} loading={list.isLoading} failed={list.isError}
                    category={category} withMaps={withMaps}
                    onCategory={(c) => { setCategory(c); setOffset(0); }}
                    onWithMaps={(v) => { setWithMaps(v); setOffset(0); }}
                    onPage={setOffset} />
  );
}

export function ExcludedPhotos({ me, page, loading, failed, category, withMaps, onCategory, onWithMaps, onPage }: {
  me?: Me | null; page?: ExcludedPage; loading?: boolean; failed?: boolean; category: string; withMaps: boolean;
  onCategory: (c: string) => void; onWithMaps: (v: boolean) => void; onPage: (offset: number) => void;
}) {
  const [open, setOpen] = useState<ExcludedTile | null>(null);
  const cats = page?.categories ?? [];
  const all = cats.reduce((n, c) => n + c.count, 0);
  return (
    <ResearchShell me={me}>
      <ResearchHero crumbs={[{ href: "/research", label: "Research" }, { label: "Excluded photos" }]}
                    title="Excluded photos" testId="excluded-hero"
                    lede={<>Photos people left out of Vision in Human Feedback: micrographs, field slips, habitat
                      shots and more. Browse them by why they were left out, and see where Vision looked on any of
                      them.</>}>
        {page && (
          <p className="mt-3 text-sm text-[#5c4a3a]" data-testid="text-excluded-counts">
            <b>{nf(all)}</b> photos left out by people, in {cats.length} kinds. Photos not everyone may see
            are placeholders.
          </p>
        )}
      </ResearchHero>
      <Section testId="section-excluded-browse">
        <div className="flex flex-wrap gap-1.5" role="tablist" aria-label="Why the photo was left out">
          <Chip on={category === "all"} onClick={() => onCategory("all")} label="All" count={all} />
          {cats.map((c) => (
            <Chip key={c.code} on={category === c.code} onClick={() => onCategory(c.code)} label={c.label} count={c.count} />
          ))}
        </div>
        <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm text-[#5c4a3a]">
          <label className="flex items-center gap-2">
            <input type="checkbox" checked={withMaps} onChange={(e) => onWithMaps(e.target.checked)}
                   className="accent-[#5e9a2c]" data-testid="toggle-with-maps" />
            Only photos someone already mapped
          </label>
          {page && <span data-testid="text-excluded-range">{rangeLine(page.offset, page.tiles.length, page.total)}</span>}
        </div>
        {failed && <p className="mt-4 text-sm text-amber-800">The excluded photos could not be loaded. Try again later.</p>}
        {loading && !page && <p className="mt-4 text-sm text-muted-foreground">Reading Human Feedback…</p>}
        {page && (
          <div className="mt-4 grid grid-cols-2 gap-3 sm:grid-cols-4 lg:grid-cols-6" data-testid="grid-excluded">
            {page.tiles.map((t) => <Tile key={t.photo_key} t={t} onOpen={() => setOpen(t)} />)}
          </div>
        )}
        {page && page.total > PAGE_SIZE && (
          <div className="mt-4 flex justify-center gap-2">
            <button type="button" disabled={page.offset === 0} onClick={() => onPage(Math.max(0, page.offset - PAGE_SIZE))}
                    className="rounded-md border border-[#A87146]/25 bg-white px-3 py-1 text-sm disabled:opacity-40">Previous</button>
            <button type="button" disabled={page.offset + PAGE_SIZE >= page.total} onClick={() => onPage(page.offset + PAGE_SIZE)}
                    className="rounded-md border border-[#A87146]/25 bg-white px-3 py-1 text-sm disabled:opacity-40">Next</button>
          </div>
        )}
        <p className="mt-6 text-xs leading-relaxed text-[#7a6a5a]">
          A photo is here when a person's latest decision in Human Feedback left it out of Vision; a flag
          alone never puts it here. A photo is shown only if anyone may see it: Creative Commons, or its
          owner said yes on mycomap.org. Who decided, and their notes, are never shown.
        </p>
      </Section>
      {open && <PhotoDialog t={open} me={me} onClose={() => setOpen(null)} />}
    </ResearchShell>
  );
}

function Chip({ on, onClick, label, count }: { on: boolean; onClick: () => void; label: string; count: number }) {
  return (
    <button type="button" role="tab" aria-selected={on} onClick={onClick}
            className={`rounded-full border px-3 py-1 text-[13px] ${on ? "border-[#4a7a22] bg-[#4a7a22] text-white" : "border-[#A87146]/25 bg-white text-[#4a3728] hover:bg-[#f6f9ef]"}`}>
      {label} <span className={on ? "text-white/80" : "text-[#8a7b6c]"}>{nf(count)}</span>
    </button>
  );
}

function Tile({ t, onOpen }: { t: ExcludedTile; onOpen: () => void }) {
  const hidden = t.shown !== "public" || !t.thumb;
  return (
    <button type="button" onClick={onOpen} disabled={hidden} data-testid="tile-excluded"
            className="group text-left disabled:cursor-default">
      <div className="relative aspect-square overflow-hidden rounded-md bg-[#efe9e2]">
        {hidden ? (
          <div className="flex h-full w-full flex-col items-center justify-center gap-1 px-2 text-center text-[11px] text-[#8a7b6c]">
            <ImageOff className="h-5 w-5" aria-hidden />
            Not shown publicly
          </div>
        ) : (
          <img src={t.thumb!} alt="" loading="lazy" className="h-full w-full object-cover transition group-hover:scale-[1.03]" />
        )}
        {t.maps.length > 0 && (
          <span className="absolute right-1 top-1 flex items-center gap-1 rounded-full bg-white/90 px-1.5 py-0.5 text-[10px] font-semibold text-[#3f6b1d]"
                title="Someone already asked where Vision looked">
            <ScanEye className="h-3 w-3" aria-hidden /> map
          </span>
        )}
      </div>
      <div className="mt-1 truncate text-[12px] text-[#4a3728]"><span className="sci">{t.name ?? "—"}</span></div>
      <div className="truncate text-[11px] text-[#8a7b6c]">{categoryLabel(t.category)}</div>
    </button>
  );
}

/** One photo, large, with v1's map on request, as a spotlight. */
export function PhotoDialog({ t, me, onClose }: { t: ExcludedTile; me?: Me | null; onClose: () => void }) {
  const [strength, setStrength] = useState(75);
  useEffect(() => {
    const esc = (e: KeyboardEvent) => { if (e.key === "Escape") onClose(); };
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [onClose]);
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/50 p-3 sm:p-8"
         role="dialog" aria-modal="true" aria-label="Excluded photo" onClick={onClose}>
      <div className="w-full max-w-[1000px] rounded-lg bg-white p-4 shadow-xl sm:p-5" onClick={(e) => e.stopPropagation()}
           data-testid="dialog-excluded">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[.08em] text-[#6b8f2a]">{categoryLabel(t.category)}</p>
            <h2 className="font-display text-xl font-semibold text-[#4a3728]"><span className="sci">{t.name}</span></h2>
            <p className="text-[13px] text-[#6f6257]">
              The record's DNA-verified name
              {t.record_url && <> · <a href={t.record_url} target="_blank" rel="noreferrer" className="underline">
                {t.record_key.startsWith("mo:") ? `MO ${t.record_key.slice(3)}` : `iNat ${t.record_key}`}
                <ArrowUpRight className="ml-0.5 inline h-3 w-3" aria-hidden /></a></>}
              {t.owner_login && <> · © {t.owner_login}{t.license_code && `, ${t.license_code.toUpperCase()}`}</>}
            </p>
          </div>
          <button type="button" onClick={onClose} aria-label="Close" className="rounded-md border border-[#A87146]/25 p-1">
            <X className="h-4 w-4" aria-hidden />
          </button>
        </div>
        <div className="mt-3 flex flex-wrap items-center gap-x-5 gap-y-2">
          <label className="flex min-w-[160px] flex-1 items-center gap-2">
            <span className="shrink-0 text-[11px] uppercase tracking-wide text-[#7a6a5a]">Spotlight strength</span>
            <input type="range" min={0} max={100} value={strength} onChange={(e) => setStrength(+e.target.value)}
                   className="w-full accent-[#5e9a2c]" aria-label="Strength of the map" />
          </label>
        </div>
        <div className="mt-3">
          <MapFigure t={t} me={me} strength={strength} />
        </div>
        <details className="mt-3 rounded-md border border-[#A87146]/20 px-3 py-2 text-[13px] text-[#4a3728]">
          <summary className="cursor-pointer font-semibold">What this map does and doesn't mean</summary>
          <ul className="mt-2 list-disc space-y-1 pl-4">
            <li><b>It shows</b> which parts of the photo Vision's score for the record's name depended on: hiding
              the brightest tenth costs about six times as much score as hiding a random tenth (314 photos checked).</li>
            <li><b>It doesn't show why.</b> Vision compares patterns with photos of DNA-verified specimens; a bright
              patch means "this part looked like the specimens", nothing more.</li>
            <li><b>Scored without its own record.</b> The record's other photos are DNA-verified specimens too, so
              they are left out: the map shows what the photo shares with other people's finds of the name.</li>
            <li>Vision sees only the photo's centre square; the striped strips are never seen.</li>
          </ul>
        </details>
      </div>
    </div>
  );
}

function MapFigure({ t, me, strength }: { t: ExcludedTile; me?: Me | null; strength: number }) {
  const [view, setView] = useState<ExcludedLooked | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [asking, setAsking] = useState(false);
  // A map someone already asked for is open to anyone: read it straight away.
  useEffect(() => {
    if (t.held) return;
    let live = true;
    setView(null);
    setProblem(null);
    excludedApi.read(t.photo_key).then((v) => { if (live) setView(v); }).catch(() => { if (live) setView({ status: "none" }); });
    return () => { live = false; };
  }, [t.photo_key, t.held]);
  useEffect(() => {
    if (view?.status !== "working") return;
    const id = setTimeout(() => {
      excludedApi.read(t.photo_key).then(setView).catch((e) => {
        setProblem(refusal(e instanceof ApiError ? e.status : 0, String(e.message ?? "")));
      });
    }, POLL_MS);
    return () => clearTimeout(id);
  }, [view, t.photo_key]);
  const ask = async () => {
    setAsking(true);
    setProblem(null);
    try {
      setView(await excludedApi.ask(t.photo_key));
    } catch (e) {
      setProblem(refusal(e instanceof ApiError ? e.status : 0, e instanceof Error ? e.message : ""));
    } finally {
      setAsking(false);
    }
  };
  const ready = view?.status === "ready" ? view : null;
  if (t.held) {
    return (
      <figure className="m-0" data-testid="figure-excluded-v1">
        <Photo t={t} />
        <p className="mt-2 text-[13px] text-[#6f6257]" data-testid="text-excluded-held">{HELD_LINE}</p>
      </figure>
    );
  }
  return (
    <figure className="m-0" data-testid="figure-excluded-v1">
      <Photo t={t} ready={ready} strength={strength} />
      {ready ? (
        <p className="mt-2 text-[13px] text-[#5c4a3a]" data-testid="text-excluded-map-line">
          {MAP_MODEL_LABEL}: clear where this photo's evidence for <span className="sci">{ready.name}</span> was.
          {" "}On its own, the photo puts it {rankLine(ready.rank, ready.top)}.
        </p>
      ) : (
        <button type="button" onClick={ask} disabled={asking || view?.status === "working" || view === null}
                className="mt-2 flex w-full items-center gap-2 rounded-md border border-myco-green bg-[#f4f9ee] px-3 py-2 text-sm font-semibold text-[#3f6b1d] hover:bg-[#eaf4df] disabled:opacity-80"
                data-testid="button-excluded-looked-v1">
          <ScanEye className="h-4 w-4 shrink-0" aria-hidden />
          <span>Show where Vision looked</span>
          <span className="ml-auto text-xs font-normal text-[#6b7a5c]">
            {view?.status === "working" ? (view.ahead > 0 ? "waiting for a quiet moment…" : "looking…")
              : asking ? "asking…" : me?.user ? "takes a few seconds" : "sign in to ask"}
          </span>
        </button>
      )}
      {problem && <p className="mt-1 text-xs text-amber-800" role="status">{problem}</p>}
    </figure>
  );
}

/** The photo, with its map over the square the model sees once there is one. */
function Photo({ t, ready, strength = 75 }: {
  t: ExcludedTile; ready?: Extract<ExcludedLooked, { status: "ready" }> | null; strength?: number;
}) {
  return (
    <div className="flex justify-center overflow-hidden rounded-md bg-[#111]">
      <div className="relative">
        <img src={t.photo_url ?? ""} alt={`${categoryLabel(t.category)} photo of ${t.name}`} className="block max-h-[62vh] w-auto max-w-full" />
        {ready && (
          <>
            <div className="absolute" style={cropBox(ready.map.crop)}>
              <MapLayer entry={ready.map} opacity={strength / 100} />
            </div>
            {unseenStrips(ready.map.crop).map((s) => (
              <div key={s.side} className="absolute bg-[repeating-linear-gradient(135deg,rgba(255,255,255,.55)_0_6px,rgba(255,255,255,.35)_6px_12px)]"
                   style={s.style} />
            ))}
          </>
        )}
      </div>
    </div>
  );
}
