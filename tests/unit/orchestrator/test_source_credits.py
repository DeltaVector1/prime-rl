import asyncio
from types import SimpleNamespace

import pytest
import verifiers.v1 as vf
from pydantic import ValidationError

from prime_rl.configs.orchestrator import TrainSourceConfig
from prime_rl.orchestrator.dispatcher import Dispatcher
from prime_rl.orchestrator.envs import Envs
from prime_rl.orchestrator.train_source import TrainSource
from prime_rl.orchestrator.types import DispatchFailure, GroupCancellation, Policy, Progress


def test_source_credit_config(monkeypatch) -> None:
    monkeypatch.setattr(vf, "resolve_env_field", lambda data, annotation: data)
    config = {"env": vf.SingleAgentEnvConfig.model_construct(taskset=vf.TasksetConfig(id="test"))}
    assert TrainSourceConfig(**config).max_outstanding is None
    assert TrainSourceConfig(**config, group_size=4, max_outstanding=1).max_outstanding == 1
    with pytest.raises(ValidationError):
        TrainSourceConfig(**config, max_outstanding=0)


@pytest.mark.parametrize("global_limit,fast_limit", [(3, 5), (8, 2)])
def test_source_credits_bound_blocked_work_and_complete_partial_groups(global_limit, fast_limit) -> None:
    async def run():
        queues = {name: asyncio.Queue() for name in ("slow", "fast")}

        async def finish_sessions(ids):
            pass

        clients = SimpleNamespace(train_client=None, finish_sessions=finish_sessions)
        envs = Envs()
        envs._envs = {}
        for name, limit, ratio in [("slow", 1, 1e9), ("fast", fast_limit, 1.0)]:

            async def run_episode(*, task_data, source=name, **kwargs):
                outcome = await queues[source].get()
                if outcome == "fail":
                    raise RuntimeError("local task failure")
                task = vf.Task(vf.TaskData(**task_data))
                return vf.Episode(
                    env=vf.EnvInfo(id=source, name=source),
                    task=vf.TraceTask(type=type(task).__name__, data=task.data, key=task.key, hash=task.hash),
                    traces=[],
                    ok=True,
                )

            envs._envs[name] = SimpleNamespace(
                name=name,
                config=SimpleNamespace(group_size=4, max_outstanding=limit, ratio=ratio, curriculum=None),
                clients=clients,
                uses_live_policy=True,
                tasks=iter([vf.Task(vf.TaskData(idx=i, prompt=f"task {i}")) for i in range(10)]),
                num_tasks=10,
                run=run_episode,
            )
        source = TrainSource(envs)
        dispatcher = Dispatcher(
            train_envs=envs,
            eval_envs=None,
            train_source=source,
            eval_source=None,
            policy_clients=clients,
            policy=Policy(),
            progress=Progress(),
            initial_max_inflight=global_limit,
            max_inflight_ceiling=global_limit,
            dispatch_per_minute=None,
            max_off_policy_steps=1,
            run_id="local",
            run_name=None,
        )
        try:
            await asyncio.wait_for(dispatcher.fill_inflight(), timeout=1)
            assert dispatcher.inflight_by_env == {("train", "slow"): 1, ("train", "fast"): 2}
            assert dispatcher.current_inflight == 3 <= global_limit
            slow_gid = next(gid for gid, g in dispatcher.groups.items() if g.env_name == "slow")
            fast_gid = next(gid for gid, g in dispatcher.groups.items() if g.env_name == "fast")
            assert dispatcher.groups[slow_gid].episodes_to_schedule == 3
            assert dispatcher.groups[fast_gid].episodes_to_schedule == 2
            before = source.state_dict()
            await asyncio.wait_for(dispatcher.fill_inflight(), timeout=1)
            assert source.state_dict() == before

            results = []
            for outcome in ["ok", "fail", "ok", "ok"]:
                await asyncio.sleep(0)
                queues["fast"].put_nowait(outcome)
                done, _ = await asyncio.wait(dispatcher.inflight, timeout=1, return_when=asyncio.FIRST_COMPLETED)
                assert len(done) == 1
                await dispatcher.handle_completed_request(done.pop())
                results.append(dispatcher.out_q.get_nowait())
                if len(results) == 4:
                    assert fast_gid not in dispatcher.groups
                await asyncio.wait_for(dispatcher.fill_inflight(), timeout=1)
                assert dispatcher.current_inflight == len(dispatcher.inflight) <= global_limit
                assert dispatcher.inflight_by_env[("train", "slow")] == 1
                assert dispatcher.inflight_by_env[("train", "fast")] <= fast_limit
            assert sum(isinstance(result, DispatchFailure) for result in results) == 1
            assert all(
                str(result.group.id if isinstance(result, vf.Episode) else result.group_id) == str(fast_gid)
                for result in results
            )

            assert await dispatcher.drop_group(slow_gid, reason="overload") == 4
            cancellation = dispatcher.out_q.get_nowait()
            assert isinstance(cancellation, GroupCancellation)
            assert cancellation.count == 4
            assert ("train", "slow") not in dispatcher.inflight_by_env
            for gid in list(dispatcher.groups):
                await dispatcher.drop_group(gid, reason="overload")
                dispatcher.out_q.get_nowait()
            await dispatcher.fill_inflight()
            assert dispatcher.inflight_by_env[("train", "slow")] == 1

            dispatcher.progress.step = 4
            await dispatcher.on_version_pending(4)
            assert not dispatcher.inflight
            assert not dispatcher.groups
            assert dispatcher.current_inflight == 0
        finally:
            await dispatcher.cancel_inflight_episodes()
        assert dispatcher.current_inflight == 0

    asyncio.run(run())
