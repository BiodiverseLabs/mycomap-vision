// Research > Excluded photos > Modified photos (Steve, 2026-10-27): each field-slip photo whose
// Recovered Labels crops go back into Vision, before (the photo as people left it out) and after
// (each kept crop, exactly the square the model reads), with where Vision looked on both. The
// crop is the original image shown through CSS (recoveredLabels.cropBackground): no new file.
import { useState } from "react";
import { ArrowRight, ArrowUpRight, ImageOff, ScanEye } from "lucide-react";
import { cropBox, unseenStrips } from "@/lib/whereLooked";
import { boxStyle, cropBackground } from "@/lib/recoveredLabels";
import {
  cropLine, HELD_LINE, MODIFIED_PAGE, noMapLine, rangeLine, rankLine, type ModifiedCard, type ModifiedCrop, type ModifiedMap,
  type ModifiedPage,
} from "@/lib/excludedPhotos";
import { nf } from "@/lib/researchView";
import { MapLayer } from "@/components/WhereLooked";

export function ModifiedPhotos({ page, loading, failed, filter, withMaps, onFilter, onWithMaps, onPage }: {
  page?: ModifiedPage; loading?: boolean; failed?: boolean; filter: string; withMaps: boolean;
  onFilter: (f: string) => void; onWithMaps: (v: boolean) => void; onPage: (offset: number) => void;
}) {
  const [strength, setStrength] = useState(75);
  return (
    <div data-testid="section-modified">
      <p className="max-w-3xl text-sm text-[#5c4a3a]">
        Field-slip photos whose specimen crops go back into Vision (Human Feedback › Recovered labels): the photo
        as people left it out, and each crop exactly as Vision reads it, with where Vision looked on both.
      </p>
      {page?.include_open && (
        <p className="mt-2 max-w-3xl text-[13px] text-[#6f6257]" data-testid="text-modified-open">
          A crop no one has reviewed yet is used as the detector proposed it. When a reviewer leaves a crop out or
          moves its box, this page and its maps follow at Vision's next quiet hours.
        </p>
      )}
      <div className="mt-3 flex flex-wrap gap-1.5" role="tablist" aria-label="Whose box">
        {(page?.filters ?? []).map((f) => (
          <button key={f.code} type="button" role="tab" aria-selected={filter === f.code} onClick={() => onFilter(f.code)}
                  className={`rounded-full border px-3 py-1 text-[13px] ${filter === f.code ? "border-[#4a7a22] bg-[#4a7a22] text-white" : "border-[#A87146]/25 bg-white text-[#4a3728] hover:bg-[#f6f9ef]"}`}>
            {f.label} <span className={filter === f.code ? "text-white/80" : "text-[#8a7b6c]"}>{nf(f.count)}</span>
          </button>
        ))}
      </div>
      <div className="mt-3 flex flex-wrap items-center justify-between gap-3 text-sm text-[#5c4a3a]">
        <label className="flex items-center gap-2">
          <input type="checkbox" checked={withMaps} onChange={(e) => onWithMaps(e.target.checked)}
                 className="accent-[#5e9a2c]" data-testid="toggle-modified-with-maps" />
          Only photos already mapped
        </label>
        <label className="flex min-w-[220px] flex-1 items-center gap-2 sm:max-w-sm">
          <span className="shrink-0 text-[11px] uppercase tracking-wide text-[#7a6a5a]">Spotlight strength</span>
          <input type="range" min={0} max={100} value={strength} onChange={(e) => setStrength(+e.target.value)}
                 className="w-full accent-[#5e9a2c]" aria-label="Strength of the maps" />
        </label>
        {page && <span data-testid="text-modified-range">{rangeLine(page.offset, page.cards.length, page.total)}</span>}
      </div>
      {failed && <p className="mt-4 text-sm text-amber-800">The modified photos could not be loaded. Try again later.</p>}
      {loading && !page && <p className="mt-4 text-sm text-muted-foreground">Reading Recovered labels…</p>}
      <div className="mt-4 space-y-4">
        {page?.cards.map((c) => <Card key={c.item_id} c={c} strength={strength} />)}
      </div>
      {page && page.total > MODIFIED_PAGE && (
        <div className="mt-4 flex justify-center gap-2">
          <button type="button" disabled={page.offset === 0} onClick={() => onPage(Math.max(0, page.offset - MODIFIED_PAGE))}
                  className="rounded-md border border-[#A87146]/25 bg-white px-3 py-1 text-sm disabled:opacity-40">Previous</button>
          <button type="button" disabled={page.offset + MODIFIED_PAGE >= page.total} onClick={() => onPage(page.offset + MODIFIED_PAGE)}
                  className="rounded-md border border-[#A87146]/25 bg-white px-3 py-1 text-sm disabled:opacity-40">Next</button>
        </div>
      )}
      <p className="mt-6 text-xs leading-relaxed text-[#7a6a5a]">
        A crop is the original photo shown through a square window: no new image is made. Vision reads only that
        square. Maps are made in Vision's quiet hours; one not made yet reads "map coming". Only photos anyone may
        see are shown, and who reviewed them is never shown.
      </p>
    </div>
  );
}

function Card({ c, strength }: { c: ModifiedCard; strength: number }) {
  const many = c.crops.length > 1;
  if (c.shown !== "public") {
    return (
      <article className="rounded-lg border border-[#A87146]/20 bg-white p-3 sm:p-4" data-testid="card-modified-hidden">
        <h3 className="font-display text-base font-semibold text-[#4a3728]"><span className="sci">{c.name ?? "—"}</span></h3>
        <p className="mt-2 flex items-center gap-2 text-[13px] text-[#6f6257]">
          <ImageOff className="h-4 w-4 shrink-0" aria-hidden />
          This photo is not shown here: only photos anyone may see are.
        </p>
      </article>
    );
  }
  return (
    <article className="rounded-lg border border-[#A87146]/20 bg-white p-3 sm:p-4" data-testid="card-modified">
      <header className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
        <h3 className="font-display text-base font-semibold text-[#4a3728]"><span className="sci">{c.name ?? "—"}</span></h3>
        <span className={`rounded-full px-2 py-0.5 text-[11px] font-semibold ${c.decided ? "bg-[#eaf4df] text-[#3f6b1d]" : "bg-amber-100 text-amber-900"}`}>
          {c.decided ? "Reviewed" : "Open · included by default"}
        </span>
      </header>
      <p className="text-[12px] text-[#6f6257]">
        {c.record_url && <a href={c.record_url} target="_blank" rel="noreferrer" className="underline">
          {c.record_key.startsWith("mo:") ? `MO ${c.record_key.slice(3)}` : `iNat ${c.record_key}`}
          <ArrowUpRight className="ml-0.5 inline h-3 w-3" aria-hidden /></a>}
        {c.owner_login && <> · © {c.owner_login}{c.license_code && `, ${c.license_code.toUpperCase()}`}</>}
      </p>
      <div className="mt-3 flex flex-wrap items-start gap-x-4 gap-y-3">
        <figure className="m-0 max-w-full">
          <figcaption className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-[#7a6a5a]">Before · as left out</figcaption>
          <div className="relative inline-block max-w-full overflow-hidden rounded-md bg-[#111] align-top">
            <img src={c.photo_url ?? ""} alt={`Field slip photo of ${c.name}`} loading="lazy"
                 className="block h-auto max-h-[320px] w-auto max-w-full" />
            {c.map && <Overlay m={c.map} strength={strength} />}
            {c.crops.map((k) => (
              <div key={k.n} className="absolute border-2 border-[#9fd36a] shadow-[0_0_0_1px_rgba(0,0,0,.5)]"
                   style={boxStyle(k.box, c.width, c.height)}>
                {many && <span className="absolute left-0.5 top-0.5 rounded bg-black/60 px-1 text-[10px] text-white">{k.n}</span>}
              </div>
            ))}
          </div>
          <MapStatus m={c.map} noMap={c.no_map} held={c.held} />
        </figure>
        <ArrowRight className="mt-[150px] hidden h-5 w-5 shrink-0 text-[#9a8a7a] sm:block" aria-hidden />
        <div className="min-w-0 flex-1 basis-[200px]">
          <div className="mb-1 text-[11px] font-semibold uppercase tracking-wide text-[#7a6a5a]">After · what Vision reads</div>
          <div className="flex flex-wrap gap-3">
            {c.crops.map((k) => <CropFigure key={k.n} c={c} k={k} many={many} strength={strength} />)}
          </div>
        </div>
      </div>
    </article>
  );
}

function CropFigure({ c, k, many, strength }: { c: ModifiedCard; k: ModifiedCrop; many: boolean; strength: number }) {
  return (
    <figure className={`m-0 ${many ? "w-[150px] sm:w-[200px]" : "w-[240px] sm:w-[320px]"} max-w-full`} data-testid="figure-modified-crop">
      <div className="relative aspect-square overflow-hidden rounded-md bg-[#111]"
           style={c.photo_url ? cropBackground(c.photo_url, k.box, c.width, c.height) : undefined}
           role="img" aria-label={`${cropLine(k, many)}, the part of the photo Vision reads`}>
        {k.map && (
          <div className="absolute" style={cropBox(k.map.map.crop)}>
            <MapLayer entry={k.map.map} opacity={strength / 100} />
          </div>
        )}
      </div>
      <figcaption className="mt-1 text-[12px] text-[#4a3728]">{cropLine(k, many)}</figcaption>
      <MapStatus m={k.map} noMap={k.no_map} held={c.held} />
    </figure>
  );
}

/** The map over the square of the photo the model sees, and the strips it never sees. */
function Overlay({ m, strength }: { m: ModifiedMap; strength: number }) {
  return (
    <>
      <div className="absolute" style={cropBox(m.map.crop)}>
        <MapLayer entry={m.map} opacity={strength / 100} />
      </div>
      {unseenStrips(m.map.crop).map((s) => (
        <div key={s.side} className="absolute bg-[repeating-linear-gradient(135deg,rgba(255,255,255,.55)_0_6px,rgba(255,255,255,.35)_6px_12px)]"
             style={s.style} />
      ))}
    </>
  );
}

function MapStatus({ m, noMap, held }: { m: ModifiedMap | null; noMap: string | null; held: boolean }) {
  if (held) {
    return <p className="mt-1 text-[12px] text-[#6f6257]" data-testid="text-map-held">{HELD_LINE}</p>;
  }
  if (!m && noMap) {
    return <p className="mt-1 text-[12px] text-[#6f6257]" data-testid="text-map-none">{noMapLine(noMap)}</p>;
  }
  if (!m) {
    return (
      <p className="mt-1 flex items-center gap-1 text-[12px] text-[#8a7b6c]" data-testid="text-map-coming">
        <ScanEye className="h-3.5 w-3.5" aria-hidden /> Map coming: Vision makes these in its quiet hours.
      </p>
    );
  }
  return (
    <p className="mt-1 text-[12px] text-[#5c4a3a]" data-testid="text-map-line">
      Clear where its evidence for <span className="sci">{m.name}</span> was; on its own it puts it {rankLine(m.rank, m.top ?? 10)}.
    </p>
  );
}
