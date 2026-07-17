import asyncio
from types import SimpleNamespace

from prime_rl.orchestrator.dispatcher import RolloutDispatcher


def test_policy_update_waits_for_older_eval_generation_to_drain():
    async def run() -> None:
        state = {"queued_step": 25}
        dispatcher = RolloutDispatcher.__new__(RolloutDispatcher)
        dispatcher.eval_source = SimpleNamespace(
            has_pending_before=lambda version: state["queued_step"] is not None and state["queued_step"] < version
        )
        dispatcher.groups = {}
        dispatcher.inflight = {}
        dispatcher.eval_work_changed = asyncio.Event()

        exact_update = asyncio.create_task(dispatcher.wait_for_eval_before_policy_update(25))
        await asyncio.sleep(0)
        assert exact_update.done()

        newer_update = asyncio.create_task(dispatcher.wait_for_eval_before_policy_update(26))
        await asyncio.sleep(0)
        assert not newer_update.done()

        state["queued_step"] = None
        dispatcher.groups["eval-group"] = SimpleNamespace(kind="eval", eval_step=25)
        dispatcher.eval_work_changed.set()
        await asyncio.sleep(0)
        assert not newer_update.done()

        dispatcher.groups.clear()
        dispatcher.inflight["eval-task"] = SimpleNamespace(kind="eval", eval_step=25)
        dispatcher.eval_work_changed.set()
        await asyncio.sleep(0)
        assert not newer_update.done()

        dispatcher.inflight.clear()
        dispatcher.eval_work_changed.set()
        await newer_update

    asyncio.run(run())
