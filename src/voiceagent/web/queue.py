"""An ordered, bounded wait for the one thing that can run at a time.

`server.py` used to refuse a second request outright, and that was right for a
developer tool: synthesis is slow enough to look hung, the natural response is
to click again, and every extra click made the machine slower rather than the
answer sooner. Refusing turned a death spiral into "still working".

It is the wrong answer for something someone paid for. A 429 reads as broken.

So this queues instead --- **while keeping the property that made refusing
safe**. The danger was never ordering, it was unbounded load, and the two are
separable:

  * **The queue is short and hard-capped.** Past `max_waiting` the honest answer
    is still no. A queue of fifty on a machine that serves one at a time is a
    four-minute lie.
  * **Every waiter is told its position**, which is what stops the clicking. A
    person who can see "2 ahead of you" does not retry; a person staring at a
    spinner does.
  * **Nothing is retried on the caller's behalf.** A dropped connection leaves
    the queue, so an abandoned tab does not hold a slot.

Concurrency stays 1 and that is not a tuning choice: both engines are single
shared mutable objects and the Indic path calls `set_reference()` on the shared
instance, so two overlapping requests could answer one in the other's voice.
Memory is the other half --- two Indic generations on this machine thrash into
swap and neither finishes.
"""

from __future__ import annotations

import asyncio
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass

#: How many may wait behind the one running. Four is chosen against the clock,
#: not for tidiness: at roughly 5 s a generation, the last person in a full
#: queue waits about 25 s, which is a wait a person will sit through. Ten would
#: be a minute, which is a wait they abandon --- having already been charged the
#: attention of watching it.
DEFAULT_MAX_WAITING = 4

#: Seed for the running average, used before anything has been measured. Close
#: to a short Hindi sentence on this hardware. It is replaced by the first real
#: measurement, so it only ever shapes the very first estimate.
INITIAL_JOB_SECONDS = 5.0

#: Weight of the newest measurement in the running average. High enough to track
#: a machine that has just started swapping --- the failure mode here is quoting
#: an optimistic wait from yesterday's idle timings.
EWMA_ALPHA = 0.3


class Full(Exception):
    """Raised when the queue is at capacity. The caller should refuse, honestly."""

    def __init__(self, waiting: int, eta_seconds: float) -> None:
        self.waiting = waiting
        self.eta_seconds = eta_seconds
        super().__init__(
            f"{waiting} requests are already waiting and this machine runs one at "
            f"a time, so the wait would be about {eta_seconds:.0f}s. Try again "
            "shortly --- retrying now makes it slower, not sooner."
        )


@dataclass(frozen=True)
class Ticket:
    """A place in line. `ahead` is how many were in front at the moment of joining."""

    ahead: int
    eta_seconds: float


class SynthesisQueue:
    def __init__(
        self,
        max_waiting: int = DEFAULT_MAX_WAITING,
        initial_job_seconds: float = INITIAL_JOB_SECONDS,
    ) -> None:
        #: `asyncio.Lock` is FIFO in CPython --- waiters are woken in the order
        #: they arrived --- which is the whole ordering guarantee here. Written
        #: down because it is a language-implementation detail the fairness of
        #: this queue rests on, and a fair-looking queue that silently serves
        #: last-in-first-out is worse than an unfair one that admits it.
        self._lock = asyncio.Lock()
        self._waiting = 0
        self._running = False
        self._max_waiting = max_waiting
        self._mean_job_seconds = initial_job_seconds
        self._started_at: float | None = None

    # --- state ------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self._running

    @property
    def waiting(self) -> int:
        return self._waiting

    @property
    def depth(self) -> int:
        """Everyone in the system: the one running plus everyone behind it."""
        return self._waiting + (1 if self._running else 0)

    @property
    def mean_job_seconds(self) -> float:
        return self._mean_job_seconds

    def eta_seconds(self, ahead: int) -> float:
        """How long someone with `ahead` in front should expect to wait.

        Counts the *remaining* part of the running job rather than a whole one.
        Quoting a full job for one already half done is how an estimate ends up
        consistently pessimistic, and an estimate nobody trusts is not used.
        """
        remaining = 0.0
        if self._running and self._started_at is not None:
            elapsed = time.perf_counter() - self._started_at
            remaining = max(0.0, self._mean_job_seconds - elapsed)
        elif self._running:
            remaining = self._mean_job_seconds
        return remaining + max(0, ahead) * self._mean_job_seconds

    def snapshot(self) -> dict:
        """What `/api/queue` serves. No identities, only shape."""
        return {
            "running": self._running,
            "waiting": self._waiting,
            "depth": self.depth,
            "capacity": self._max_waiting,
            "accepting": self._waiting < self._max_waiting,
            "mean_job_seconds": round(self._mean_job_seconds, 2),
            "eta_seconds": round(self.eta_seconds(self._waiting), 1),
            "running_for_seconds": (
                round(time.perf_counter() - self._started_at, 1)
                if self._running and self._started_at is not None
                else None
            ),
        }

    # --- the wait ---------------------------------------------------------

    @asynccontextmanager
    async def slot(self):
        """Wait for the machine, then hold it. Raises `Full` rather than queueing
        past capacity.

        The counter is incremented *before* awaiting and decremented in a
        `finally`, so a caller cancelled while waiting --- a closed tab, a client
        timeout --- releases its place. Without that, abandoned requests would
        fill the queue and the cap would slowly become a deadlock.
        """
        if self._waiting >= self._max_waiting:
            raise Full(self._waiting, self.eta_seconds(self._waiting))

        ahead = self.depth
        ticket = Ticket(ahead=ahead, eta_seconds=self.eta_seconds(ahead))

        self._waiting += 1
        try:
            await self._lock.acquire()
        finally:
            self._waiting -= 1

        self._running = True
        self._started_at = time.perf_counter()
        try:
            yield ticket
        finally:
            elapsed = time.perf_counter() - (self._started_at or time.perf_counter())
            self._running = False
            self._started_at = None
            self._lock.release()
            #: Recorded after release so a slow measurement never widens the
            #: window in which nobody can start. Failures are recorded too: a
            #: generation that died still occupied the machine for that long,
            #: and an average that only counts successes under-quotes every
            #: wait that follows a failure.
            self._observe(elapsed)

    def _observe(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._mean_job_seconds = (
            EWMA_ALPHA * seconds + (1 - EWMA_ALPHA) * self._mean_job_seconds
        )
