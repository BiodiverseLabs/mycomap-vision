// The E17 blind review page. Its own small bundle (vite.e17.config.ts): no site shell, no
// /api/me, no fonts, nothing from another host. It shows only what the item JSON holds.

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { client, tokenFromPath, type ItemWithResponse } from "./api";
import {
  agreement, answerFrom, applyKey, autocompleteCandidates, chips as chipsOf, coverageText, DEFAULT_SORT, EMPTY_ANSWER,
  genusOf, goToIndex, keyAction, lengthText, MARKER, nextSort, pct, savedText, sortHits, stepIndex, summarise,
  summaryLines, accessionCell, tableState, TABLE_STATE_TEXT, autoOpenInstructions, whenText,
  type Answer, type Hit, type Item, type ListEntry, type Sequence, type ShownBy, type Sort, type SortKey, type TableState, type Verdict,
} from "./logic";
import { parseMarkdown, type Inline, type ListItem } from "./markdown";
import { Saver, type SaveState } from "./saver";
import instructionsText from "./instructions.md?raw";

const token = tokenFromPath(window.location.pathname);
const api = client(token);

const VERDICTS: { value: Verdict; key: string; label: string }[] = [
  { value: "supported", key: "1", label: "Supported" },
  { value: "wrong", key: "2", label: "Wrong" },
  { value: "cannot_tell", key: "3", label: "Cannot tell" },
];
const SHOWN: { value: ShownBy; key: string; label: string }[] = [
  { value: "sequences", key: "S", label: "Sequences" },
  { value: "photographs", key: "P", label: "Photographs" },
  { value: "both", key: "B", label: "Both" },
];

function isTextField(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  if (el.isContentEditable || el.tagName === "TEXTAREA") return true;
  return el.tagName === "INPUT" && !["checkbox", "radio", "button", "submit"].includes((el as HTMLInputElement).type);
}

function positionFromHash(): number | null {
  const m = /^#(\d{1,4})$/.exec(window.location.hash);
  return m ? Number(m[1]) : null;
}

/** A name with its genus in italics, the way the mockup writes provisional names. */
function NameText({ name }: { name: string }) {
  const g = genusOf(name);
  if (!g) return <>{name}</>;
  const rest = name.trim().slice(g.length);
  const words = rest.trim().split(/\s+/);
  const epithet = words[0] && /^[a-z][a-z-]+$/.test(words[0]) ? words[0] : null;
  if (epithet) return <i>{name}</i>;
  return <><i>{g}</i>{rest}</>;
}

export function App() {
  const [list, setList] = useState<{ count: number; items: ListEntry[] } | null>(null);
  const [index, setIndex] = useState<number | null>(null);
  const [item, setItem] = useState<ItemWithResponse | null>(null);
  const [failed, setFailed] = useState<string | null>(null);
  // The answer being edited and the item it belongs to, always set together: an edit is
  // saved under the code it was made on, never another item's.
  const [draft, setDraft] = useState<{ code: string | null; answer: Answer }>({ code: null, answer: EMPTY_ANSWER });
  const answer = draft.answer;
  const [save, setSave] = useState<SaveState>({ status: "idle", savedAt: null, failures: 0 });
  const [held, setHeld] = useState(false);
  const [unansweredOnly, setUnansweredOnly] = useState(false);
  const [gotoText, setGotoText] = useState("");
  const [gotoMiss, setGotoMiss] = useState(false);
  const [photo, setPhoto] = useState(0);
  const [lightbox, setLightbox] = useState(false);
  const [help, setHelp] = useState(false);
  const autoOpened = useRef(false);
  const betterRef = useRef<HTMLInputElement>(null);
  const draftRef = useRef(draft);
  draftRef.current = draft;

  const saver = useMemo(() => new Saver({
    post: api.respond,
    onState: setSave,
    onSaved: (code, saved) => setList((l) => l && {
      ...l,
      items: l.items.map((it) => (it.code === code ? { ...it, answered: !!saved.verdict } : it)),
    }),
  }), []);

  // The list, then the first item: the one in the address, else the first unanswered.
  useEffect(() => {
    api.items().then((l) => {
      setList(l);
      // A first visit (nobody has answered anything yet): the instructions open by
      // themselves, once; after that they are one click away.
      if (autoOpenInstructions(l.items, autoOpened.current)) setHelp(true);
      autoOpened.current = true;
      const want = positionFromHash();
      const at = want !== null ? l.items.findIndex((it) => it.position === want) : -1;
      const first = l.items.findIndex((it) => !it.answered);
      setIndex(at >= 0 ? at : first >= 0 ? first : 0);
    }, () => setFailed("The review list could not be loaded. Reload the page to try again."));
  }, []);

  const entry = list && index !== null ? list.items[index] : null;

  useEffect(() => {
    if (!entry) return;
    let live = true;
    setItem(null);
    setPhoto(0);
    setLightbox(false);
    api.item(entry.code).then((it) => {
      if (!live) return;
      setItem(it);
      saver.start(it.code, it.response?.saved_at ?? null);
      setDraft({ code: it.code, answer: answerFrom(it.response) });
    }, () => live && setFailed("This item could not be loaded. Reload the page to try again."));
    if (window.location.hash !== `#${entry.position}`) {
      window.history.replaceState(null, "", `#${entry.position}`);
    }
    return () => { live = false; };
  }, [entry?.code]);   // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    document.title = entry && list ? `Review ${entry.position}/${list.count}` : "Review";
  }, [entry?.position, list?.count]);   // eslint-disable-line react-hooks/exhaustive-deps

  // Never close the tab on an unsaved change without asking.
  useEffect(() => {
    const warn = (e: BeforeUnloadEvent) => {
      if (!saver.clean) {
        e.preventDefault();
        e.returnValue = "";
      }
    };
    window.addEventListener("beforeunload", warn);
    return () => window.removeEventListener("beforeunload", warn);
  }, [saver]);

  /** An edit to the answer of item `code`; the saver refuses it if that item is no longer
   * the open one. */
  const edit = useCallback((code: string | null, next: Answer) => {
    if (code === null || !saver.change(code, next)) return;
    setDraft({ code, answer: next });
  }, [saver]);
  const change = (next: Answer) => edit(draft.code, next);

  /** Leave for another item only once this one's save has landed. */
  const go = useCallback((target: number | null) => {
    if (target === null || !list || target === index) return;
    setHeld(true);
    saver.whenClean(() => {
      setHeld(false);
      setIndex(target);
    });
  }, [list, index, saver]);

  const step = useCallback((dir: 1 | -1) => {
    if (!list || index === null) return;
    go(stepIndex(list.items, index, dir, unansweredOnly));
  }, [list, index, unansweredOnly, go]);

  const photos = item?.photos ?? [];
  const movePhoto = useCallback((d: number) => {
    if (photos.length) setPhoto((p) => (p + d + photos.length) % photos.length);
  }, [photos.length]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const action = keyAction(e, { typing: isTextField(e.target), lightbox });
      if (!action) return;
      if (action.type === "escape") {
        if (lightbox) setLightbox(false);
        else if (help) setHelp(false);
        else if (isTextField(e.target)) (e.target as HTMLElement).blur();
        return;
      }
      e.preventDefault();
      if (action.type === "photoNext") movePhoto(1);
      else if (action.type === "photoPrev") movePhoto(-1);
      else if (action.type === "next") step(1);
      else if (action.type === "prev") step(-1);
      else if (action.type === "focusBetterName") betterRef.current?.focus();
      else if (item) {
        const d = draftRef.current;
        const next = applyKey(d.answer, action);
        if (next) edit(d.code, next);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [lightbox, help, item, step, movePhoto, edit]);

  const answered = list ? list.items.filter((it) => it.answered).length : 0;
  const count = list?.count ?? 0;
  const canPrev = !!list && index !== null && stepIndex(list.items, index, -1, unansweredOnly) !== null;
  const canNext = !!list && index !== null && stepIndex(list.items, index, 1, unansweredOnly) !== null;

  const submitGoto = () => {
    if (!list) return;
    const target = goToIndex(gotoText, list.items);
    setGotoMiss(target === null);
    if (target !== null) {
      setGotoText("");
      go(target);
    }
  };

  const saveText = save.status === "failed"
    ? "Not saved: retrying"
    : save.status === "saving" || save.status === "pending"
      ? "Saving…"
      : savedText(save.savedAt);

  return (
    <div className="e17">
      <header className="top">
        <div className="top-in">
          <span className="brand">Review</span>
          <span>{entry ? <>Item <b>{entry.position}</b> of {count} &middot; code <b>{entry.code}</b></> : "Loading…"}</span>
          <div className="progress">
            <div className="bar" role="progressbar" aria-valuemin={0} aria-valuemax={count} aria-valuenow={answered}>
              <i style={{ width: `${count ? (100 * answered) / count : 0}%` }} />
            </div>
            <span>{answered}/{count} answered</span>
          </div>
          <label>
            Go to
            <input
              type="text" value={gotoText} placeholder="no. or code" aria-label="Go to item number or code"
              onChange={(e) => { setGotoText(e.target.value); setGotoMiss(false); }}
              onKeyDown={(e) => { if (e.key === "Enter") submitGoto(); }}
            />
          </label>
          {gotoMiss && <span className="goto-miss">no such item</span>}
          <label><input type="checkbox" checked={unansweredOnly} onChange={(e) => setUnansweredOnly(e.target.checked)} /> Unanswered only</label>
          <button type="button" className="linkbtn" onClick={() => setHelp(true)}>Instructions</button>
        </div>
      </header>

      {failed && <div className="loading">{failed}</div>}
      {!failed && !item && <div className="loading">Loading…</div>}
      {!failed && item && (
        <main className="wrap">
          <Photos item={item} current={photo} onPick={setPhoto} onOpen={() => setLightbox(true)} />
          <Evidence item={item} />
        </main>
      )}

      <footer className="answer">
        <div className="answer-in">
          <div className="grp" role="group" aria-label="Verdict">
            <span className="lbl">Verdict</span>
            {VERDICTS.map((v) => (
              <button
                key={v.value} type="button" aria-pressed={answer.verdict === v.value} disabled={!item}
                className={`opt ${v.value === "wrong" ? "wrong" : ""} ${answer.verdict === v.value ? "on" : ""}`}
                onClick={() => change({ ...answer, verdict: v.value })}
              ><kbd>{v.key}</kbd>{v.label}</button>
            ))}
          </div>
          <div className="grp" role="group" aria-label="Shown by">
            <span className="lbl">Shown by</span>
            {SHOWN.map((s) => (
              <button
                key={s.value} type="button" aria-pressed={answer.shown_by === s.value} disabled={!item}
                className={`opt ${answer.shown_by === s.value ? "on" : ""}`}
                onClick={() => change({ ...answer, shown_by: s.value })}
              ><kbd>{s.key}</kbd>{s.label}</button>
            ))}
          </div>
          <div className="nav">
            <div className="checks">
              <label><input type="checkbox" checked={answer.recognised} disabled={!item} onChange={(e) => change({ ...answer, recognised: e.target.checked })} /> Recognised (R)</label>
              <label><input type="checkbox" checked={answer.looked_up} disabled={!item} onChange={(e) => change({ ...answer, looked_up: e.target.checked })} /> Looked up (L)</label>
            </div>
            <span className={`saved ${save.status === "failed" ? "failed" : save.status === "saving" || save.status === "pending" ? "busy" : ""}`} role="status">{saveText}</span>
            {held && !saver.clean && <span className="held">Saving before moving on</span>}
            <button type="button" onClick={() => step(-1)} disabled={!canPrev}>&larr; Prev</button>
            <button type="button" className="next" onClick={() => step(1)} disabled={!canNext}>Next &rarr;</button>
          </div>
          <div className="fields">
            <div className="better">
              {item && <Chips item={item} value={answer.better_name} onPick={(name) => change({ ...answer, better_name: name })} />}
              <input
                ref={betterRef} list="e17-names" value={answer.better_name} disabled={!item}
                placeholder="Better name if wrong (/ to focus)" aria-label="Better name"
                onChange={(e) => change({ ...answer, better_name: e.target.value })}
              />
              <datalist id="e17-names">
                {item && autocompleteCandidates(item).map((n) => <option key={n} value={n} />)}
              </datalist>
            </div>
            <textarea
              value={answer.notes} placeholder="Notes" aria-label="Notes" disabled={!item}
              onChange={(e) => change({ ...answer, notes: e.target.value })}
            />
          </div>
        </div>
        <div className="keys">
          Name suggestions are the distinct names in this item's own hits, best identity first, other than the name
          under review, plus the top hit's genus. Keys: 1 / 2 / 3 verdict &middot; S / P / B shown by &middot; / better
          name &middot; R recognised &middot; L looked up &middot; &rarr; or N next &middot; &larr; or Shift+N previous
          &middot; Esc closes the photo. Every change saves automatically; edits are kept.
        </div>
      </footer>

      {lightbox && item && photos.length > 0 && (
        <div className="light" role="dialog" aria-label="Photo" onClick={() => setLightbox(false)}>
          <img src={api.photoUrl(item.code, photos[photo].n)} alt={`Photo ${photo + 1} of ${photos.length}`} onClick={(e) => e.stopPropagation()} />
          {photos.length > 1 && <>
            <button type="button" className="ctl prev" aria-label="Previous photo" onClick={(e) => { e.stopPropagation(); movePhoto(-1); }}>&larr;</button>
            <button type="button" className="ctl nextp" aria-label="Next photo" onClick={(e) => { e.stopPropagation(); movePhoto(1); }}>&rarr;</button>
          </>}
          <button type="button" className="close" onClick={() => setLightbox(false)}>Close (Esc)</button>
          <div className="cap">Photo {photo + 1} of {photos.length} &middot; arrow keys to move, Esc to close</div>
        </div>
      )}
      {help && <Instructions onClose={() => setHelp(false)} />}
    </div>
  );
}

function Photos({ item, current, onPick, onOpen }: { item: Item; current: number; onPick: (i: number) => void; onOpen: () => void }) {
  const photos = item.photos;
  return (
    <section className="card photos" aria-label="Photographs">
      {photos.length === 0 ? (
        <div className="photo-none">No photographs for this item</div>
      ) : (
        <>
          <button type="button" className="photo-main" onClick={onOpen} aria-label="Enlarge photo">
            <img src={api.photoUrl(item.code, photos[current].n)} alt={`Photo ${current + 1} of ${photos.length}`} />
          </button>
          <div className="photo-count">Photo {current + 1} of {photos.length} (click to enlarge)</div>
          {photos.length > 1 && (
            <div className="thumbs">
              {photos.map((p, i) => (
                <button key={p.n} type="button" className={i === current ? "on" : ""} onClick={() => onPick(i)} aria-label={`Photo ${i + 1}`}>
                  <img src={api.photoUrl(item.code, p.n)} alt="" loading="lazy" />
                </button>
              ))}
            </div>
          )}
        </>
      )}
      <div className="meta">{item.place} &middot; {whenText(item.when)}</div>
    </section>
  );
}

function Evidence({ item }: { item: Item }) {
  return (
    <section className="card" aria-label="DNA evidence">
      <p className="name">{item.name_under_review}</p>
      <div className="sub">Name under review</div>
      {item.sequences.length === 0 && <div className="sub">No sequences for this item.</div>}
      {item.sequences.map((s) => <SequenceBlock key={s.n} seq={s} reviewed={item.name_under_review} />)}
      <div className="legend">
        <span><i className="sp" />=sp same species as the name under review</span>
        <span><i className="gen" />=gen same genus</span>
        <span><i className="oth" />other genus</span>
      </div>
      <div className="legend">
        <span>Click a column heading to sort. GenBank accessions open NCBI in a new tab.</span>
      </div>
    </section>
  );
}

function SequenceBlock({ seq, reviewed }: { seq: Sequence; reviewed: string }) {
  const [open, setOpen] = useState(!seq.withdrawn);
  const lines = summaryLines(summarise(seq, reviewed));
  const best = summarise(seq, reviewed).best;
  return (
    <div className={`seq ${seq.withdrawn ? "withdrawn" : ""} ${open ? "" : "collapsed"}`}>
      <div className="seq-h">
        <b>Sequence {seq.n}</b>
        <span>{lengthText(seq.length)}</span>
        {seq.withdrawn && <span className="badge">Withdrawn</span>}
        <span className="grow" />
        {seq.withdrawn && (
          <button type="button" className="linkbtn" aria-expanded={open} onClick={() => setOpen(!open)}>
            {open ? "Hide hits" : "Show hits"}
          </button>
        )}
      </div>
      <div className="summary">
        <div>
          <span className="k">Best hit</span>
          {best ? <><NameText name={best.name} />{lines.best.slice(best.name.length)}</> : lines.best}
        </div>
        <div><span className="k">Name under review</span>{lines.reviewed}</div>
      </div>
      {open && (
        <div className="tables">
          <HitTable title="Local database" hits={seq.local} state={tableState(seq, "local")} reviewed={reviewed} withAccession={false} />
          <HitTable title="GenBank" hits={seq.genbank} state={tableState(seq, "genbank")} reviewed={reviewed} withAccession />
        </div>
      )}
    </div>
  );
}

function HitTable({ title, hits, state, reviewed, withAccession }: { title: string; hits: Hit[]; state: TableState; reviewed: string; withAccession: boolean }) {
  const [sort, setSort] = useState<Sort>(DEFAULT_SORT);
  const rows = sortHits(hits, sort);
  const head = (key: SortKey, label: string, num = false) => (
    <th className={num ? "num" : ""} aria-sort={sort.key === key ? (sort.dir === "asc" ? "ascending" : "descending") : "none"}>
      <button type="button" onClick={() => setSort(nextSort(sort, key))}>
        {label}{sort.key === key && key !== "rank" ? (sort.dir === "asc" ? " ▲" : " ▼") : ""}
      </button>
    </th>
  );
  return (
    <div className="tbl">
      <h4>{title}</h4>
      {state !== "hits" ? <div className="empty">{TABLE_STATE_TEXT[state]}</div> : (
        <table>
          <thead>
            <tr>
              {head("rank", "#")}
              {head("name", "Name")}
              {head("identity", "Identity %", true)}
              {head("coverage", "Cov. %", true)}
              {withAccession && <th>Accession</th>}
              <th aria-label="Agreement" />
            </tr>
          </thead>
          <tbody>
            {rows.map((h) => {
              const a = agreement(h.name, reviewed);
              return (
                <tr key={h.rank} className={a === "other" ? "oth" : a}>
                  <td>{h.rank}</td>
                  <td className="nm">{h.name}</td>
                  <td className="num">{pct(h.identity)}</td>
                  <td className="num">{coverageText(h.coverage)}</td>
                  {withAccession && <AccessionTd hit={h} />}
                  <td className="mk">{MARKER[a]}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

function AccessionTd({ hit }: { hit: Hit }) {
  const cell = accessionCell(hit);
  return (
    <td>
      {cell.kind === "link"
        ? <a href={cell.href} target="_blank" rel="noreferrer noopener">{cell.text}</a>
        : <span className="community">{cell.text}</span>}
    </td>
  );
}

function Chips({ item, value, onPick }: { item: Item; value: string; onPick: (name: string) => void }) {
  const c = chipsOf(item);
  if (!c.names.length && !c.genus) return null;
  const on = (name: string) => value.trim() === name;
  return (
    <div className="chips">
      <span className="lbl">Better name:</span>
      {c.names.map((n) => (
        <button key={n.name} type="button" className={`chip ${on(n.name) ? "on" : ""}`} onClick={() => onPick(n.name)}>
          <NameText name={n.name} /><small>{pct(n.identity)}%</small>
        </button>
      ))}
      {c.genus && (
        <button type="button" className={`chip ${on(c.genus) ? "on" : ""}`} onClick={() => onPick(c.genus!)}>
          genus only: <i>{c.genus}</i>
        </button>
      )}
    </div>
  );
}

const INSTRUCTIONS = parseMarkdown(instructionsText);

/** Inline Markdown as React elements: text stays text, never HTML. */
function InlineText({ content }: { content: Inline[] }) {
  return <>{content.map((c, i) => c.kind === "bold" ? <b key={i}><InlineText content={c.children ?? []} /></b>
    : c.kind === "italic" ? <i key={i}><InlineText content={c.children ?? []} /></i>
      : <span key={i}>{c.text}</span>)}</>;
}

function MdList({ items }: { items: ListItem[] }) {
  return (
    <ul>
      {items.map((it, i) => (
        <li key={i}><InlineText content={it.content} />{it.children.length > 0 && <MdList items={it.children} />}</li>
      ))}
    </ul>
  );
}

/** The pinned instructions (instructions.md), as written. */
function Instructions({ onClose }: { onClose: () => void }) {
  return (
    <div className="panel-back" onClick={onClose}>
      <div className="panel" role="dialog" aria-label="Instructions" onClick={(e) => e.stopPropagation()}>
        {INSTRUCTIONS.map((b, i) => b.kind === "heading"
          ? (b.level === 2 ? <h2 key={i}><InlineText content={b.content} /></h2> : <h3 key={i}><InlineText content={b.content} /></h3>)
          : b.kind === "paragraph" ? <p key={i}><InlineText content={b.content} /></p>
            : <MdList key={i} items={b.items} />)}
        <div className="close-row"><button type="button" onClick={onClose}>Close (Esc)</button></div>
      </div>
    </div>
  );
}
