import { useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { BTN } from "@/components/ReviewPhotos";
import { feedbackApi, type Revealed } from "@/lib/api";
import { formatTop5, REVEAL_BUTTON, REVEAL_CONFIRM, revealNote, revealProblem } from "@/lib/feedback";
import { cn } from "@/lib/utils";

// "Show Vision's answer" on a blind item (Steve, 2026-10-18), on label-review cards and in the
// generic item detail. Before it is clicked it is the same small button on every open blind item,
// whatever the item: nothing in it depends on the item but its id. Clicking asks first, since the
// decision then counts as made with the answer seen; the server records the look.

// A secondary button, smaller than the outcome buttons (still a 32px tap target).
const SMALL_BTN = cn(BTN, "min-h-8 px-2 py-1 text-xs");

export function RevealAnswer({ itemId, revealed, onRevealed }: {
  itemId: number; revealed: Revealed | null | undefined; onRevealed?: (r: Revealed) => void;
}) {
  const [shown, setShown] = useState<Revealed | null>(revealed ?? null);
  const [asking, setAsking] = useState(false);
  const reveal = useMutation({
    mutationFn: () => feedbackApi.reveal(itemId),
    onSuccess: (r) => { setShown(r); setAsking(false); onRevealed?.(r); },
  });
  const seen = shown ?? revealed ?? null;
  if (seen) {
    return (
      <div className="min-w-0 space-y-0.5 rounded-md bg-[#faf9f7] px-3 py-2 text-sm dark:bg-muted/40"
           data-testid={`reveal-${itemId}`}>
        <p className="min-w-0 break-words text-[#5c4a3a] dark:text-foreground/90">{formatTop5(seen.answer)}</p>
        <p className="text-xs text-muted-foreground">{revealNote(seen)}</p>
      </div>
    );
  }
  if (!asking) {
    return (
      <div>
        <button type="button" className={SMALL_BTN} onClick={() => { reveal.reset(); setAsking(true); }}>
          {REVEAL_BUTTON}
        </button>
      </div>
    );
  }
  return (
    <div className="flex min-w-0 flex-wrap items-center gap-2 text-xs" role="group" aria-label={REVEAL_BUTTON}>
      <span className="min-w-0 break-words text-muted-foreground">{REVEAL_CONFIRM}</span>
      <button type="button" className={SMALL_BTN} disabled={reveal.isPending} onClick={() => reveal.mutate()}>
        {reveal.isPending ? "Showing…" : "Show"}
      </button>
      <button type="button" className={SMALL_BTN} disabled={reveal.isPending} onClick={() => setAsking(false)}>
        Cancel
      </button>
      {reveal.error && (
        <span className="basis-full text-red-700 dark:text-red-300" role="alert">{revealProblem(reveal.error)}</span>
      )}
    </div>
  );
}
