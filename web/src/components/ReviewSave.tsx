import { useCallback, useEffect, useRef } from "react";
import { RotateCcw, Save } from "lucide-react";
import { confirmLeave, guardUnload, reviewUnsaved, savePageLabel, unsavedLine } from "@/lib/reviewSave";
import { BTN } from "./ReviewPhotos";

// The review pages' Auto-save toggle, the "Save page" bar shown while it is off, and the guard
// that asks before unsaved changes are left behind (Steve, 2026-10-21; rules in lib/reviewSave.ts).

/** The Auto-save switch, at the top of the review. */
export function AutosaveToggle({ on, onChange, title }: { on: boolean; onChange: (on: boolean) => void; title: string }) {
  return (
    <button type="button" role="switch" aria-checked={on} onClick={() => onChange(!on)} title={title}
            data-testid="review-autosave"
            className={`inline-flex min-h-10 items-center gap-2 rounded-md border px-3 py-1.5 text-sm font-medium ${
              on ? "border-myco-green/50 bg-myco-green/10 text-myco-green"
                : "border-[#A87146]/30 bg-white text-gray-700 hover:bg-myco-green/5 dark:border-border dark:bg-transparent dark:text-foreground/80"}`}>
      <span aria-hidden className={`relative inline-block h-5 w-9 shrink-0 rounded-full transition-colors ${
        on ? "bg-myco-green" : "bg-gray-300 dark:bg-gray-600"}`}>
        <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${on ? "left-[1.125rem]" : "left-0.5"}`} />
      </span>
      Auto-save {on ? "on" : "off"}
    </button>
  );
}

/** Auto-save off: how many changes are unsaved, "Save page (N changes)" and Discard. */
export function SaveBar({ count, busy = false, onSave, onDiscard, sticky = false, note }: {
  count: number; busy?: boolean; onSave: () => void; onDiscard: () => void; sticky?: boolean; note?: string;
}) {
  return (
    <div className={`flex min-w-0 flex-wrap items-center gap-x-3 gap-y-2 rounded-md border px-3 py-2 text-sm ${
           count > 0 ? "border-amber-400 bg-amber-50 dark:border-amber-600 dark:bg-amber-950/60"
             : "border-[#A87146]/20 bg-white dark:border-border dark:bg-card"} ${
           sticky ? "sticky top-16 z-30 shadow-sm" : ""}`}
         role="region" aria-label="Unsaved changes" data-testid="review-save-bar">
      <span className={`font-medium ${count > 0 ? "text-amber-900 dark:text-amber-200" : "text-muted-foreground"}`}
            aria-live="polite">{busy ? "Saving…" : unsavedLine(count)}</span>
      {note && <span className="text-xs text-muted-foreground">{note}</span>}
      <span className="ml-auto flex flex-wrap gap-2">
        <button type="button" className={BTN} onClick={onDiscard} disabled={count === 0 || busy}>
          <RotateCcw className="h-3.5 w-3.5" aria-hidden /> Discard
        </button>
        <button type="button" onClick={onSave} disabled={count === 0 || busy} data-testid="review-save-page"
                className="inline-flex min-h-10 items-center gap-1 rounded-md bg-myco-green px-4 py-1.5 text-sm font-medium text-white hover:bg-myco-green/90 disabled:opacity-50">
          <Save className="h-4 w-4" aria-hidden /> {savePageLabel(count)}
        </button>
      </span>
    </div>
  );
}

const ask = (message: string) => (typeof window === "undefined" ? true : window.confirm(message));

/** Asks before `unsaved` changes are left: the browser's own question when the tab is closed or
 *  reloaded; a confirm for a click on a link inside the site, and for the queue picker above
 *  (through reviewUnsaved). Returns the check for the page's own paging, tabs and filters. */
export function useLeaveGuard(unsaved: number, source: string): () => boolean {
  const ref = useRef(unsaved);
  ref.current = unsaved;
  useEffect(() => {
    reviewUnsaved.set(source, unsaved);
  }, [source, unsaved]);
  useEffect(() => () => reviewUnsaved.set(source, 0), [source]);
  useEffect(() => {
    const onUnload = (e: BeforeUnloadEvent) => { guardUnload(e, ref.current); };
    const onClick = (e: MouseEvent) => {
      if (!(ref.current > 0) || e.defaultPrevented || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
      const a = e.target instanceof Element ? e.target.closest("a[href]") : null;
      if (!(a instanceof HTMLAnchorElement) || a.target === "_blank" || a.hasAttribute("download")) return;
      if (a.origin !== window.location.origin) return;
      if (a.pathname === window.location.pathname && a.search === window.location.search) return;
      if (!confirmLeave(ref.current, ask)) {
        e.preventDefault();
        e.stopImmediatePropagation();
      }
    };
    window.addEventListener("beforeunload", onUnload);
    document.addEventListener("click", onClick, true);
    return () => {
      window.removeEventListener("beforeunload", onUnload);
      document.removeEventListener("click", onClick, true);
    };
  }, []);
  return useCallback(() => confirmLeave(ref.current, ask), []);
}
