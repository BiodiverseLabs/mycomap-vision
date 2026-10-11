"""The queue in front of the model (docs/api.md §5, docs/capacity.md).

One worker answers identifications one batch at a time, holding the model lock (the
nightly update takes the same lock between batches). Two rules:

- **Interactive first.** The website and app calls go ahead of developer and
  by-observation calls (Steve, 2026-10-11); within each, first come first served.
- **One search per batch.** Scoring against the reference set costs ~1.85 s per call on
  the box whatever the number of photos (the float16 reference vectors are widened
  once per call), so the jobs waiting on the same index when the worker starts are
  embedded one after another and then scored together in one pass. Their answers are
  the same as one at a time (`identify_vectors(..., sims=)` takes each job's rows).

A job is refused up front when the work already ahead of it is estimated to take more
than `max_wait` seconds (`Busy`, with that estimate for Retry-After). The estimate uses
the measured seconds per photo embedded and per search pass, updated after each batch.

Timing per job (apiv1.Timer): `embed_ms` its own photos, `score_ms` the shared search plus
its own scoring, `queue_ms` everything else between arriving here and its answer
(waiting for jobs ahead, and for the other jobs of its batch).
"""

from __future__ import annotations

import itertools
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Callable

import numpy as np


class Busy(RuntimeError):
    def __init__(self, wait_seconds: float):
        super().__init__(f"about {wait_seconds:.0f} s of work is ahead")
        self.wait_seconds = wait_seconds


@dataclass
class Job:
    key: object                                     # jobs with the same key share a search
    photos: int                                     # photos this job embeds
    embed: Callable[[], np.ndarray]                 # -> (n, d) unit vectors to score: the
                                                    # new photos' (and a session's stored ones)
    search: Callable[[np.ndarray], np.ndarray]      # (n, d) -> (n, reference photos)
    finish: Callable[[np.ndarray, np.ndarray], dict]  # (vectors, their sims rows) -> answer
    interactive: bool = True
    timer: object = None                            # apiv1.Timer, or None
    seq: int = 0
    arrived: float = 0.0
    future: Future = field(default_factory=Future)


class ModelQueue:
    def __init__(self, lock: threading.Lock, max_wait: float = 20.0,
                 max_batch_photos: int = 20, embed_seconds: float = 1.6,
                 search_seconds: float = 2.0, clock: Callable[[], float] = time.monotonic,
                 note=print, start: bool = True):
        self.lock, self.max_wait, self.max_batch_photos = lock, max_wait, max_batch_photos
        self.embed_seconds, self.search_seconds = embed_seconds, search_seconds
        self.clock, self.note = clock, note
        self._jobs: list[Job] = []
        self._running_photos = 0
        self._cond = threading.Condition()
        self._seq = itertools.count()
        self.batches = 0                            # searches run (for tests and health)
        if start:
            threading.Thread(target=self._work, name="model queue", daemon=True).start()

    # --- the callers' side --------------------------------------------------------------

    def ahead_seconds(self, interactive: bool = True) -> float:
        """Estimated seconds of work ahead of a new job of this kind."""
        with self._cond:
            return self._ahead(interactive)

    def _ahead(self, interactive: bool) -> float:
        photos = self._running_photos + sum(j.photos for j in self._jobs
                                            if j.interactive or not interactive)
        if not photos:
            return 0.0
        return photos * self.embed_seconds + self.search_seconds

    def admit(self, interactive: bool = True) -> None:
        """Raise Busy when the work ahead would take longer than `max_wait`."""
        wait = self.ahead_seconds(interactive)
        if wait > self.max_wait:
            raise Busy(wait)

    def run(self, job: Job, timeout: float | None = 600.0) -> dict:
        """Queue the job and wait for its answer (its own exception, if it failed)."""
        job.seq, job.arrived = next(self._seq), self.clock()
        with self._cond:
            self._jobs.append(job)
            self._cond.notify()
        return job.future.result(timeout=timeout)

    def state(self) -> dict:
        with self._cond:
            return {"queued_jobs": len(self._jobs),
                    "queued_photos": sum(j.photos for j in self._jobs),
                    "embed_seconds_per_photo": round(self.embed_seconds, 2),
                    "search_seconds": round(self.search_seconds, 2),
                    "batches": self.batches}

    # --- the worker -----------------------------------------------------------------------

    def _take_batch(self) -> list[Job]:
        """The first job by priority, and the jobs waiting with the same key behind it
        (still by priority) up to max_batch_photos. Call holding _cond."""
        ordered = sorted(self._jobs, key=lambda j: (not j.interactive, j.seq))
        first = ordered[0]
        batch, photos = [first], first.photos
        for j in ordered[1:]:
            if j.key == first.key and photos + j.photos <= self.max_batch_photos:
                batch.append(j)
                photos += j.photos
        for j in batch:
            self._jobs.remove(j)
        self._running_photos = photos
        return batch

    def _work(self) -> None:
        while True:
            with self._cond:
                while not self._jobs:
                    self._cond.wait()
                batch = self._take_batch()
            try:
                self.run_batch(batch)
            finally:
                with self._cond:
                    self._running_photos = 0

    def run_batch(self, batch: list[Job]) -> None:
        """Embed each job, score them all in one search, finish each. A job that fails
        fails alone."""
        with self.lock:
            done: list[tuple[Job, np.ndarray, float]] = []
            for job in batch:
                start = self.clock()
                try:
                    vecs = np.asarray(job.embed())
                except Exception as e:  # noqa: BLE001 - this job's photos; the rest go on
                    job.future.set_exception(e)
                    continue
                done.append((job, vecs, self.clock() - start))
            if not done:
                return
            start = self.clock()
            try:
                sims = done[0][0].search(np.concatenate([v for _, v, _ in done]))
            except Exception as e:  # noqa: BLE001 - the whole pass failed: tell them all
                for job, _, _ in done:
                    job.future.set_exception(e)
                return
            searched = self.clock() - start
            self.batches += 1
            row = 0
            for job, vecs, embedded in done:
                rows = sims[row:row + len(vecs)]
                row += len(vecs)
                start = self.clock()
                try:
                    answer = job.finish(vecs, rows)
                except Exception as e:  # noqa: BLE001
                    job.future.set_exception(e)
                    continue
                own = self.clock() - start
                if job.timer is not None:
                    job.timer.add("embed_ms", embedded)
                    job.timer.add("score_ms", searched + own)
                    job.timer.add("queue_ms", max(0.0, self.clock() - job.arrived
                                                  - embedded - searched - own))
                    job.timer.photos_embedded += min(job.photos, len(vecs))
                    job.timer.photos_reused += max(0, len(vecs) - job.photos)
                job.future.set_result(answer)
        self._learn(sum(len(v) for _, v, _ in done), sum(e for _, _, e in done), searched)

    def _learn(self, photos: int, embedded: float, searched: float, alpha: float = 0.3) -> None:
        """Move the estimates toward what this batch took (so throttling shows at once)."""
        if photos:
            self.embed_seconds += alpha * (embedded / photos - self.embed_seconds)
        self.search_seconds += alpha * (searched - self.search_seconds)
