"""The queue that replaced an outright refusal, and the safety it had to keep.

Refusing was not a bug. Synthesis looks hung, so people click again, and every
click made the machine slower rather than the answer sooner --- refusing turned
that spiral into "still working". Queueing is only allowed here because the
danger was unbounded load rather than ordering, so most of these tests are about
the cap and the released slot, not about the happy path.
"""

from __future__ import annotations

import asyncio

import pytest

from voiceagent.web.queue import Full, SynthesisQueue


async def _hold(q, seconds, started=None, done=None):
    async with q.slot():
        if started is not None:
            started.set()
        await asyncio.sleep(seconds)
    if done is not None:
        done.set()


# --- the cap, which is what keeps queueing safe ---------------------------


@pytest.mark.asyncio
async def test_past_capacity_it_refuses_instead_of_queueing():
    q = SynthesisQueue(max_waiting=2)
    running = asyncio.Event()
    holder = asyncio.create_task(_hold(q, 0.2, started=running))
    await running.wait()

    waiters = [asyncio.create_task(_hold(q, 0.01)) for _ in range(2)]
    await asyncio.sleep(0.01)

    with pytest.raises(Full):
        async with q.slot():
            pass

    await asyncio.gather(holder, *waiters)


@pytest.mark.asyncio
async def test_the_refusal_says_how_long_the_wait_would_be():
    """"Busy" with no number is indistinguishable from "stuck", which is the
    confusion that caused the retrying in the first place."""
    q = SynthesisQueue(max_waiting=1, initial_job_seconds=4.0)
    running = asyncio.Event()
    holder = asyncio.create_task(_hold(q, 0.15, started=running))
    await running.wait()
    waiter = asyncio.create_task(_hold(q, 0.01))
    await asyncio.sleep(0.01)

    with pytest.raises(Full) as caught:
        async with q.slot():
            pass
    assert caught.value.eta_seconds > 0
    assert "retrying now makes it slower" in str(caught.value)

    await asyncio.gather(holder, waiter)


# --- ordering and exclusion ------------------------------------------------


@pytest.mark.asyncio
async def test_only_one_runs_at_a_time():
    """Not a tuning choice: the engines are shared mutable objects and the Indic
    path calls set_reference() on the shared instance, so two overlapping
    requests could answer one in the other's voice."""
    q = SynthesisQueue()
    concurrent = 0
    peak = 0

    async def job():
        nonlocal concurrent, peak
        async with q.slot():
            concurrent += 1
            peak = max(peak, concurrent)
            await asyncio.sleep(0.02)
            concurrent -= 1

    await asyncio.gather(*(job() for _ in range(5)))
    assert peak == 1


@pytest.mark.asyncio
async def test_waiters_are_served_in_the_order_they_arrived():
    q = SynthesisQueue(max_waiting=5)
    order: list[int] = []

    async def job(n):
        async with q.slot():
            order.append(n)
            await asyncio.sleep(0.01)

    first = asyncio.create_task(job(0))
    await asyncio.sleep(0.005)
    rest = []
    for n in range(1, 4):
        rest.append(asyncio.create_task(job(n)))
        await asyncio.sleep(0.002)

    await asyncio.gather(first, *rest)
    assert order == [0, 1, 2, 3]


@pytest.mark.asyncio
async def test_a_waiter_is_told_how_many_are_ahead():
    """The position is what stops the clicking. A person who can see "2 ahead of
    you" does not retry; a person staring at a spinner does."""
    q = SynthesisQueue(max_waiting=4)
    running = asyncio.Event()
    holder = asyncio.create_task(_hold(q, 0.15, started=running))
    await running.wait()

    seen = {}

    async def waiter():
        async with q.slot() as ticket:
            seen["ahead"] = ticket.ahead
            seen["eta"] = ticket.eta_seconds

    task = asyncio.create_task(waiter())
    await asyncio.gather(holder, task)
    assert seen["ahead"] == 1
    assert seen["eta"] > 0


# --- the failure that would turn the cap into a deadlock -------------------


@pytest.mark.asyncio
async def test_a_cancelled_waiter_releases_its_place():
    """A closed tab must not hold a slot. Without this the cap fills with
    abandoned requests and slowly becomes a deadlock."""
    q = SynthesisQueue(max_waiting=1)
    running = asyncio.Event()
    holder = asyncio.create_task(_hold(q, 0.3, started=running))
    await running.wait()

    abandoned = asyncio.create_task(_hold(q, 0.01))
    await asyncio.sleep(0.01)
    assert q.waiting == 1

    abandoned.cancel()
    with pytest.raises(asyncio.CancelledError):
        await abandoned
    assert q.waiting == 0

    await holder


@pytest.mark.asyncio
async def test_the_slot_is_released_when_the_body_raises():
    q = SynthesisQueue()
    with pytest.raises(ValueError):
        async with q.slot():
            raise ValueError("synthesis blew up")
    assert not q.running
    async with q.slot():
        pass


# --- the estimate ----------------------------------------------------------


@pytest.mark.asyncio
async def test_the_estimate_learns_from_what_actually_happened():
    q = SynthesisQueue(initial_job_seconds=10.0)
    async with q.slot():
        await asyncio.sleep(0.02)
    assert q.mean_job_seconds < 10.0


@pytest.mark.asyncio
async def test_a_failed_job_still_counts_toward_the_estimate():
    """It occupied the machine for that long. An average that counts only
    successes under-quotes every wait that follows a failure."""
    q = SynthesisQueue(initial_job_seconds=10.0)
    with pytest.raises(RuntimeError):
        async with q.slot():
            await asyncio.sleep(0.02)
            raise RuntimeError("boom")
    assert q.mean_job_seconds < 10.0


def test_an_idle_queue_reports_no_wait():
    q = SynthesisQueue()
    snap = q.snapshot()
    assert snap["depth"] == 0
    assert snap["accepting"] is True
    assert snap["running_for_seconds"] is None
