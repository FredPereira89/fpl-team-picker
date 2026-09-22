"""A rate limit shared by threads fetching from the FPL API."""

import threading
import time


class FetchCancelled(Exception):
    """The fetch this caller belongs to was stopped while it waited."""


class TokenBucket:
    """Space acquisitions without banking an idle burst.

    Waiting callers reserve no slot. They recheck both the next available
    start time and any shared pause after waking, so a 429 also holds callers
    that were already waiting. Zero or negative rates disable spacing while
    retaining pauses and cancellation.
    """

    def __init__(self, rate: float, clock=time.monotonic, sleep=time.sleep):
        self.rate = float(rate)
        self._clock, self._sleep = clock, sleep
        self._lock = threading.Lock()
        self._next = 0.0
        self._paused_until = 0.0

    def pause_for(self, seconds: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, self._clock() + float(seconds))

    def acquire(self, cancel: threading.Event | None = None) -> None:
        while True:
            if cancel is not None and cancel.is_set():
                raise FetchCancelled
            with self._lock:
                now = self._clock()
                ready_at = max(self._next, self._paused_until)
                if now >= ready_at:
                    if self.rate > 0:
                        self._next = now + 1.0 / self.rate
                    return
                wait = ready_at - now
            if cancel is not None:
                cancel.wait(wait)
            else:
                self._sleep(wait)
