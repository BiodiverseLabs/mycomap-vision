// Autosave for the review page: every change is saved about 600 ms after the last one; a
// failed save stays visible and is retried; leaving an item waits until its save has landed.
// No framework here, and the clock and timers are passed in, so it is tested on its own.

import type { Answer, SavedResponse } from "./logic";

export type SaveStatus = "idle" | "pending" | "saving" | "saved" | "failed";
export type SaveState = { status: SaveStatus; savedAt: string | null; failures: number };
export type Body = Answer & { client_saved_at: string };

export type SaverOptions = {
  post: (code: string, body: Body) => Promise<SavedResponse>;
  onState: (s: SaveState) => void;
  onSaved?: (code: string, saved: SavedResponse) => void;
  schedule?: (fn: () => void, ms: number) => unknown;
  cancel?: (handle: unknown) => void;
  now?: () => string;
  debounceMs?: number;
  retryMs?: number[];
};

export const DEBOUNCE_MS = 600;
export const RETRY_MS = [1000, 2000, 5000, 10000, 15000];

export class Saver {
  private code: string | null = null;
  private draft: Answer | null = null;
  private changes = 0;
  private saved = 0;
  private inflight = false;
  private timer: unknown = null;
  private retry: unknown = null;
  private waiter: (() => void) | null = null;
  private state: SaveState = { status: "idle", savedAt: null, failures: 0 };
  private readonly o: Required<SaverOptions>;

  constructor(options: SaverOptions) {
    this.o = {
      onSaved: () => {},
      schedule: (fn, ms) => setTimeout(fn, ms),
      cancel: (h) => clearTimeout(h as ReturnType<typeof setTimeout>),
      now: () => new Date().toISOString(),
      debounceMs: DEBOUNCE_MS,
      retryMs: RETRY_MS,
      ...options,
    };
  }

  /** Nothing waiting to be saved and nothing on its way. */
  get clean(): boolean {
    return this.changes === this.saved && !this.inflight;
  }

  /** A new item is open (only once the last one is clean). */
  start(code: string, savedAt: string | null): void {
    if (!this.clean) throw new Error("the open item has an unsaved change");
    this.stopTimers();
    this.code = code;
    this.draft = null;
    this.changes = this.saved = 0;
    this.waiter = null;
    this.set({ status: savedAt ? "saved" : "idle", savedAt, failures: 0 });
  }

  /** An edit made on item `code`. Refused (false) unless that is the open item, so an edit
   * can never be saved under another item's code. */
  change(code: string, answer: Answer): boolean {
    if (this.code === null || code !== this.code) return false;
    this.draft = answer;
    this.changes += 1;
    if (this.state.status !== "failed") this.set({ ...this.state, status: "pending" });
    if (this.timer !== null) this.o.cancel(this.timer);
    this.timer = this.o.schedule(() => {
      this.timer = null;
      this.flush();
    }, this.o.debounceMs);
    return true;
  }

  /** Run `then` once everything is saved: at once if it is, else after the save lands
   * (saving now rather than after the pause). A later call replaces an earlier wait. */
  whenClean(then: () => void): void {
    if (this.clean) {
      then();
      return;
    }
    this.waiter = then;
    this.flush();
  }

  flush(): void {
    if (this.timer !== null) {
      this.o.cancel(this.timer);
      this.timer = null;
    }
    if (this.inflight || this.code === null || this.draft === null) return;
    if (this.changes === this.saved) {
      this.release();
      return;
    }
    if (this.retry !== null) {
      this.o.cancel(this.retry);
      this.retry = null;
    }
    const code = this.code;
    const sent = this.changes;
    this.inflight = true;
    this.set({ ...this.state, status: "saving" });
    this.o.post(code, { ...this.draft, client_saved_at: this.o.now() }).then(
      (saved) => {
        this.inflight = false;
        if (code !== this.code) return;
        this.saved = Math.max(this.saved, sent);
        this.o.onSaved(code, saved);
        if (this.changes > this.saved) {
          this.set({ status: "saving", savedAt: saved.saved_at, failures: 0 });
          this.flush();
          return;
        }
        this.set({ status: "saved", savedAt: saved.saved_at, failures: 0 });
        this.release();
      },
      () => {
        this.inflight = false;
        if (code !== this.code) return;
        const failures = this.state.failures + 1;
        this.set({ ...this.state, status: "failed", failures });
        const wait = this.o.retryMs[Math.min(failures - 1, this.o.retryMs.length - 1)];
        this.retry = this.o.schedule(() => {
          this.retry = null;
          this.flush();
        }, wait);
      },
    );
  }

  private release(): void {
    const w = this.waiter;
    this.waiter = null;
    if (w) w();
  }

  private stopTimers(): void {
    if (this.timer !== null) this.o.cancel(this.timer);
    if (this.retry !== null) this.o.cancel(this.retry);
    this.timer = this.retry = null;
  }

  private set(s: SaveState): void {
    this.state = s;
    this.o.onState(s);
  }
}
