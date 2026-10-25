# Vision capacity note (public API and app sessions)

2026-10-11. Measured on the live box, off-peak (19:07–19:20 UTC, outside
the 07:00–09:30 nightly), read-only. These were short separate processes that exited when
done; the server was untouched and healthy before and after. CPU steal was 0–2% during the
runs, so the box had **full CPU credits**: these are best-case figures. Steve decides the
box; the recommendation is at the end.

> **Status as of 2026-10-25:** this note is historical. Its figures were measured on the
> original 4 GB, 2 vCPU box. The box has since moved to Lightsail 16 GB / 4 vCPU, with a systemd drop-in that raises the server's `MemoryMax` to 12G (option F
> below). Read the 4 GB limits here as the reasons for that move, not as today's box.

## The box

Lightsail 4 GB, "2 vCPUs" = **one physical core with hyperthreading** (Xeon Platinum
8259CL, Cascade Lake, AVX-512 VNNI), burstable on CPU credits. 605,204 reference photos
(592,926 in the release + 12,278 added by nightly runs), 174,462 records, 19,728 names,
1,545 genera. It was staying 4 GB at the time (Steve); see the status note above.

## What one identification costs

Served model `bioclip-2-ft-20261007-165400 / nearest+mean`.

**Embedding** (the model turns a photo into a vector), photos at the size the server
keeps them (448 px shorter side):

| photos in one call | seconds | per photo |
|---|---|---|
| 1 | 1.64 | 1.64 |
| 3 | 4.4 | 1.46 |
| 5 | 7.6–9.9 | 1.5–2.0 |
| 10 | 16–36 | 1.6–3.6 |

About **1.6 s a photo**, linear. The second hyperthread barely helps (torch sees one
core), and batches of 5–10 get slower and noisier per photo (the model's activations for
10 photos push the 4 GB box toward swap). Embedding one photo at a time is the steady
choice.

**Search** (score against every reference photo, build the answer), `identify_vectors`:

| photos | seconds |
|---|---|
| 1 | 1.94 |
| 3 | 2.34 |
| 5 | 2.67 |
| 10 | 2.84 |

**~1.85 s fixed per call, ~0.1 s per extra photo.** The fixed part is `Scorer.sims`
widening all 605k float16 reference vectors to float32, block by block, on every call
(~930 MB of conversion). That cost is memory bandwidth on one thread and doesn't depend
on photo count. It grows with the reference set: about +3 ms per 1,000 photos the
nightly adds.

**So, with full credits:**

| request | model time | requests/min at most |
|---|---|---|
| 1 photo | ~3.6 s | ~16 |
| 3 photos (typical) | ~6.7 s | ~9 |
| 10 photos | ~19 s | ~3 |

That matches the 3.9–7.8 s per request seen in use. Real traffic today is ~30
identifications a day, the box idle nearly all the time.

**With credits drained** we've measured ~80% steal: everything ~5× slower (a 3-photo
identification ~35 s, past what people wait for). Lightsail bundles earn credits at a
fixed baseline. Above it, the balance drains; the nightly (embedding plus 300 advance
predictions, ~2 h at full speed) spends some every day. Steady load above roughly the
baseline share of these figures runs the box into throttling. Sustained, assume
**about a fifth of the table above** until the console's CPU graph (Lightsail → Metrics
→ CPU utilization, "burstable zone") shows the real baseline for this bundle.

## Memory

At steady state the server holds ~2.2 GB (of the 3.5 GB `MemoryMax`), most of it file
pages of the memory-mapped vectors. The nightly index swap peaks at 3.3 GB + 1.9 GB of
swap. An embed of 10 photos at once costs ~+0.4 GB transient; one photo at a time costs
little.

**Sessions, hard cap `MV_SESSIONS_MAX_MB=64`.** Per photo a session keeps:

| what | size |
|---|---|
| served-model vector + the gate's base vector (float32, re-judging when gate thresholds change) | 6 KB |
| that photo's species score row (19,728 names, float16), so adding a photo never re-searches the others | 40 KB |
| its best 2,000 records with scores (for the combined "closest specimens") | 16 KB |
| its per-photo answer | ~6 KB |
| **total** | **~70 KB** |

64 MB ≈ 900 photos in open sessions. At full speed the box embeds at most ~1,000
photos in the 30-minute idle window, so CPU always binds before the cap. At the cap, the
oldest idle session is dropped. If every session has been active in the last 2 minutes,
a new one gets 503 `busy`. The cap is accounted in bytes, not session count, and
`/api/health` reports it.

With these rows a session's next photo costs **one embed (1.6 s) plus ~0.1 s** to
combine. Without them, each added photo would re-search all the session's photos
(+1.9–2.8 s each), so a 5-photo session would cost ~19 s of CPU instead of ~9 s.
Species, genus and higher ranks combine exactly (nearest+mean's score is the mean of the
per-photo rows). The combined "closest specimens" come from the union of each photo's
top 2,000 records. It's exact unless a specimen sits outside every photo's top 2,000,
which can't win the top 6 in practice. A test checks it against a full re-search.

## Options

| | What it buys | Cost per month | Risk / work |
|---|---|---|---|
| **A. Batch the search** (queued requests share one widening pass) | search ~1.9 s per *batch* not per request: under load, throughput roughly doubles for 1–3-photo calls | $0 | small; part of the v1 queue |
| **B. Session score rows** (above) | a session's added photo costs ~1.7 s not ~4 s | $0, 64 MB | in the sessions branch |
| **C. int8 search** (references quantized, VNNI int8 matmul, ~450 MB instead of 930 MB mapped) | search ~1.9 s → est. 0.2–0.4 s | $0 | must re-check accuracy: different scores, a new release format |
| **D. ONNX int8 image tower** | embed est. 1.6 → ~0.6 s/photo (2.6× on the laptop) | $0 | tried 9/29: same top-1 within 0.4 pt but only 74–85% of answers identical, kept full precision; redo on the fine-tuned model |
| **E. Lightsail 8 GB, 2 vCPU** | memory room (nightly swap, sessions), same CPU | ~$44 (from ~$24) | still burstable, still one core |
| **F. Lightsail 16 GB, 4 vCPU** | ~2× CPU, room for everything | ~$84 | still burstable |
| **G. EC2 c7i.large** (2 vCPU, 4 GB, *not burstable*, AMX) | full speed all day; AMX bf16/int8 should cut embed and search well below a second | ~$65 on-demand, ~$45 with a 1-yr savings plan | moves off Lightsail: new box setup, same deploy scripts |
| **H. EC2 c7i.xlarge** (4 vCPU, 8 GB) | the above ×2, plus memory headroom | ~$130 on-demand, ~$90 with savings | as G |

Prices are us-east-2 list prices as I know them. Check the AWS and Lightsail pricing pages
before deciding.

**Queue and Retry-After.** The v1 queue estimates a request's wait from an exponentially
weighted mean of the last embed seconds per photo and search seconds per batch. Those
estimates reflect throttling as soon as it starts. Past `MV_QUEUE_MAX_WAIT` (20 s) a
call gets 503 `busy` with that estimate as `Retry-After`. Interactive calls first.

## Recommendation

1. **Now, $0:** A + B, built into the v1 queue and sessions branches. They roughly
   double what the current box can serve under load and make sessions cheap.
2. **Plan limits** (for the .org table), sized to the throttled box, not the burst:
   about 1,500–2,500 identifications a day for everyone together. Free member
   **300/month**, supporting member **3,000/month**, staff unlimited. A developer-token
   global ceiling of `MV_API_DAILY_PHOTOS=2000`. Raise them when the box changes.
3. **When use grows** (sustained steal above ~10%, or `queue_ms` p95 over 10 s,
   both visible after v1 ships), move answering to **G (c7i.large, ~$65)**, not a
   bigger Lightsail. Its CPU isn't burstable, and AMX matters more than cores for this
   model.
4. Re-measure with credits drained once, on a weekday evening, to replace the "÷5" with
   a number.

## Options C and D, measured on the box (2026-10-12)

Steve asked for C and D for real-time suggestions (the box purchase, G, waits for full
release). Both were built behind their own settings and measured on the box, 03:10-03:26
UTC, in separate transient processes at the lowest priority (CPU steal 0-0.1%, the
service healthy throughout), on the served model and its 605,204 reference vectors.

**C, fast int8 search (`MV_SEARCH_INT8=1`): taken.** The reference rows are packed once per
index build for torch's int8 matrix product (fbgemm; the box's CPU has AVX-512 but no
VNNI), and each query's best 4,096 columns are re-scored exactly as before.

| | 1 photo | 3 photos |
|---|---|---|
| float search (before) | 1.61 s | 1.88 s |
| int8 search | 0.061 s | 0.139 s |

Top answer the same in 30/30 queries, top-50 overlap 100%. The packed copy holds 475 MB of
process memory; packing takes 8 s and needs nothing beyond it. The process holds at most
one packed copy: an index being rebuilt (the nightly update) takes it from the one serving,
which answers through the mapped search until the swap, so a rebuild never holds two.
The search no longer streams the 930 MB of mapped vectors on every call either.

**D, the image tower as ONNX (`MV_EMBED_ONNX=1`): not for this box.** int8 ONNX failed
accuracy (cosine to torch down to 0.89 on real photos; keeping attention in float32 didn't
help), so the export is float32, which gives torch's vectors exactly (cosine 1.0000). On a
laptop it embedded 2.2x faster than torch, but not on the box:

| one photo per pass, 2 threads | load | anonymous memory | file-backed | s / photo |
|---|---|---|---|---|
| torch | 14.8 s | 539 MB | 1,007 MB | 1.76 |
| ONNX float32 | 17.2 s | 1,989 MB | 71 MB | 1.98 |

Slower, and +1.45 GB of process memory (torch keeps most of its weights file-mapped), so
the setting stays off here. It falls back to torch cleanly, and may be worth measuring
again on another CPU.

**So, per photo on the box:** about 1.8 s of server time (embed 1.76 + search 0.06)
instead of about 3.4 s. The rest is embedding, which only a faster CPU (G) shortens.
