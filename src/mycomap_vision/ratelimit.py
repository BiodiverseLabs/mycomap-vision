"""Politeness limits for iNat: requests per second and bytes per time window."""

from __future__ import annotations

import threading
import time
from collections import deque
from typing import Callable


class MinInterval:
    """At most one call every `interval` seconds, shared across threads."""

    def __init__(self, interval: float, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.interval = interval
        self._clock = clock
        self._sleep = sleep
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = self._clock()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            self._sleep(delay)

    def pause(self, seconds: float) -> None:
        """Push the next slot out, e.g. after a 429."""
        with self._lock:
            self._next = max(self._next, self._clock() + seconds)


class ByteBudget:
    """No more than `limit` bytes inside any sliding `window` seconds."""

    def __init__(self, limit: int, window: float, clock: Callable[[], float] = time.monotonic):
        self.limit = limit
        self.window = window
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._used = 0
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        while self._events and self._events[0][0] <= now - self.window:
            _, n = self._events.popleft()
            self._used -= n

    def used(self) -> int:
        with self._lock:
            self._trim(self._clock())
            return self._used

    def wait_time(self) -> float:
        """Seconds until the window has room again (0 when it has room now)."""
        with self._lock:
            now = self._clock()
            self._trim(now)
            if self._used < self.limit:
                return 0.0
            # Room opens once enough of the oldest events age out.
            excess = self._used - self.limit
            freed = 0
            for t, n in self._events:
                freed += n
                if freed > excess:
                    return max(0.0, t + self.window - now)
            return self.window

    def add(self, n: int) -> None:
        with self._lock:
            now = self._clock()
            self._trim(now)
            self._events.append((now, n))
            self._used += n

    def add_past(self, seconds_ago: float, n: int) -> None:
        """Count bytes fetched `seconds_ago` (e.g. by an earlier run), so a restart
        doesn't start with a fresh budget."""
        if seconds_ago >= self.window:
            return
        with self._lock:
            now = self._clock()
            self._events.append((now - max(seconds_ago, 0.0), n))
            self._events = deque(sorted(self._events))
            self._used += n
            self._trim(now)
