import threading
import time

import pytest

from fpl.data.throttle import FetchCancelled, TokenBucket


class FakeClock:
    def __init__(self):
        self.t = 0.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.t

    def sleep(self, s: float) -> None:
        self.sleeps.append(round(s, 9))
        self.t += s


def test_spaces_acquisitions_at_the_rate():
    clk = FakeClock()
    bucket = TokenBucket(5, clock=clk, sleep=clk.sleep)
    for _ in range(3):
        bucket.acquire()
    assert clk.sleeps == [0.2, 0.2]


def test_zero_rate_never_waits():
    clk = FakeClock()
    bucket = TokenBucket(0, clock=clk, sleep=clk.sleep)
    for _ in range(50):
        bucket.acquire()
    assert clk.sleeps == []


def test_an_idle_bucket_does_not_bank_a_burst():
    clk = FakeClock()
    bucket = TokenBucket(5, clock=clk, sleep=clk.sleep)
    bucket.acquire()
    clk.t = 10.0
    bucket.acquire()
    bucket.acquire()
    assert clk.sleeps == [0.2]


def test_pause_holds_every_caller_even_uncapped():
    clk = FakeClock()
    bucket = TokenBucket(0, clock=clk, sleep=clk.sleep)
    bucket.pause_for(3.0)
    bucket.acquire()
    assert clk.sleeps == [3.0]


def test_a_pause_holds_back_threads_already_waiting():
    bucket = TokenBucket(10)
    lock = threading.Lock()
    starts: list[float] = []
    paused: dict = {}

    def worker():
        for _ in range(3):
            bucket.acquire()
            started = time.monotonic()
            with lock:
                starts.append(started)
                if len(starts) == 2:
                    bucket.pause_for(0.5)
                    paused["at"] = time.monotonic()

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(starts) == 12
    later = [t for t in starts if t > paused["at"]]
    assert later
    assert min(later) >= paused["at"] + 0.5 - 0.02


def test_cancel_wakes_a_waiting_caller():
    bucket = TokenBucket(0)
    bucket.pause_for(30.0)
    stop = threading.Event()
    outcome: dict = {}

    def waiter():
        try:
            bucket.acquire(stop)
            outcome["result"] = "acquired"
        except FetchCancelled:
            outcome["result"] = "cancelled"

    thread = threading.Thread(target=waiter)
    started = time.monotonic()
    thread.start()
    time.sleep(0.05)
    stop.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert outcome["result"] == "cancelled"
    assert time.monotonic() - started < 1.0


def test_a_cancelled_caller_never_acquires():
    stop = threading.Event()
    stop.set()
    with pytest.raises(FetchCancelled):
        TokenBucket(0).acquire(stop)


def test_is_safe_across_threads():
    bucket = TokenBucket(20)
    started = time.monotonic()
    threads = [threading.Thread(target=lambda: [bucket.acquire() for _ in range(5)])
               for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert time.monotonic() - started >= 0.9
