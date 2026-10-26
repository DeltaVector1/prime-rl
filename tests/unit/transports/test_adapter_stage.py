"""Exercise the real stage, receive, and watcher code without GPU imports."""

import ast
import asyncio
import json
import logging
import re
import time
import traceback
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from tenacity import retry, retry_if_exception, stop_after_attempt, stop_after_delay, wait_exponential

root = Path(__file__).resolve().parents[3]


def load_symbols(path, names, namespace):
    tree = ast.parse((root / path).read_text())
    selected = [
        node
        for node in tree.body
        if getattr(node, "name", None) in names
        or (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in names for target in node.targets)
        )
    ]
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(root / path), "exec"), namespace)


@pytest.fixture
def code():
    namespace = dict(
        asyncio=asyncio,
        time=time,
        re=re,
        httpx=httpx,
        Path=Path,
        ABC=ABC,
        abstractmethod=abstractmethod,
        get_logger=lambda: logging.getLogger("adapter-stage-test"),
        format_time=str,
        retry=retry,
        retry_if_exception=retry_if_exception,
        stop_after_attempt=stop_after_attempt,
        stop_after_delay=stop_after_delay,
        wait_exponential=wait_exponential,
    )
    load_symbols("src/prime_rl/utils/pathing.py", {"get_step_path", "wait_for_path"}, namespace)
    load_symbols(
        "src/prime_rl/transports/weights/base.py",
        {"WeightReceiver", "SENDER_READY_MARKER", "RECEIVER_READY_MARKER", "FINISHED_MARKER"},
        namespace,
    )
    load_symbols(
        "src/prime_rl/orchestrator/clients.py",
        {
            "stage_lora_adapter",
            "load_lora_adapter",
            "_is_retryable_lora_error",
            "LORA_LOAD_READ_TIMEOUT_S",
            "LORA_LOAD_TOTAL_TIMEOUT_S",
            "AdminPlane",
        },
        namespace,
    )
    load_symbols("src/prime_rl/transports/weights/filesystem.py", {"FileSystemWeightReceiver"}, namespace)
    load_symbols("src/prime_rl/orchestrator/watcher.py", {"WeightWatcher"}, namespace)
    return namespace


def manifest(path):
    return {
        "path": str(path),
        "files": {
            name: {"size": 0 if name == ".finished" else 16, "sha256": "a" * 64}
            for name in ("adapter_config.json", "adapter_model.safetensors", ".finished")
        },
    }


def receiver(code, tmp_path, stage_clients, load_client, timeout=1):
    path = tmp_path / "step_1"
    path.mkdir()
    for name in ("adapter_config.json", ".finished", ".sender_ready"):
        (path / name).touch()
    plane = SimpleNamespace(stage_clients=stage_clients, clients=[load_client], update_times={"old": [999]})
    return code["FileSystemWeightReceiver"](tmp_path, SimpleNamespace(timeout=timeout), plane, "policy")


def test_both_hosts_stage_before_any_load_and_whole_time_is_preserved(code, tmp_path):
    async def run():
        started = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        finished = [asyncio.Event(), asyncio.Event()]
        calls = []

        async def stage(request):
            index = 0 if request.url.host == "host-a" else 1
            assert request.url.path == "/stage_lora_adapter"
            assert request.headers["Authorization"] == "Bearer local-fixture"
            assert request.extensions["timeout"] == {"connect": 10.0, "read": 1, "write": 60.0, "pool": 10.0}
            started[index].set()
            await release[index].wait()
            finished[index].set()
            return httpx.Response(200, json=manifest(json.loads(request.content)["lora_path"]))

        async def load(request):
            assert all(event.is_set() for event in finished)
            assert request.extensions["timeout"]["read"] == 30.0
            calls.append(request.url.path)
            return httpx.Response(200)

        async with (
            httpx.AsyncClient(
                base_url="http://host-a",
                transport=httpx.MockTransport(stage),
                headers={"Authorization": "Bearer local-fixture"},
            ) as first,
            httpx.AsyncClient(
                base_url="http://host-b",
                transport=httpx.MockTransport(stage),
                headers={"Authorization": "Bearer local-fixture"},
            ) as second,
            httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(load)) as engine,
        ):
            received = receiver(code, tmp_path, [first, second], engine)
            watcher = code["WeightWatcher"](received, policy=SimpleNamespace(version=0), observers=[])
            task = asyncio.create_task(watcher.apply_policy_update(1))
            await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
            assert calls == []
            release[0].set()
            await asyncio.wait_for(finished[0].wait(), 1)
            assert calls == []
            assert not task.done()
            release[1].set()
            await asyncio.wait_for(task, 1)
            assert calls == ["/load_lora_adapter"]
            times = received.admin_plane.update_times
            assert set(times) == {"adapter_stage", "adapter_load"}
            assert watcher.last_update_weights_time >= sum(values[0] for values in times.values())
            assert watcher.gauges()["watcher/adapter_stage/max"] == times["adapter_stage"][0]
            assert watcher.policy.version == 1
            assert code["LORA_LOAD_TOTAL_TIMEOUT_S"] == 120.0

    asyncio.run(run())


@pytest.mark.parametrize(
    "failure",
    [
        "404",
        "500",
        "disconnect",
        "wrong_path",
        "missing_file",
        "empty_record",
        "bool_size",
        "negative_size",
        "bad_hash",
        "different_hash",
    ],
)
def test_stage_failure_never_loads(code, tmp_path, failure):
    async def run():
        calls = []

        async def stage(request):
            data = manifest(json.loads(request.content)["lora_path"])
            if request.url.host == "host-a":
                if failure in {"404", "500"}:
                    return httpx.Response(int(failure))
                if failure == "disconnect":
                    raise httpx.ReadError("connection closed")
                if failure == "wrong_path":
                    data["path"] += "-other"
                elif failure == "missing_file":
                    del data["files"][".finished"]
                elif failure == "empty_record":
                    data["files"][".finished"] = {}
                elif failure == "bool_size":
                    data["files"][".finished"]["size"] = True
                elif failure == "negative_size":
                    data["files"][".finished"]["size"] = -1
                elif failure in {"bad_hash", "different_hash"}:
                    data["files"][".finished"]["sha256"] = "bad" if failure == "bad_hash" else "b" * 64
            return httpx.Response(200, json=data)

        async def load(request):
            calls.append(request)
            return httpx.Response(200)

        async with (
            httpx.AsyncClient(base_url="http://host-a", transport=httpx.MockTransport(stage)) as first,
            httpx.AsyncClient(base_url="http://host-b", transport=httpx.MockTransport(stage)) as second,
            httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(load)) as engine,
        ):
            received = receiver(code, tmp_path, [first, second], engine)
            with pytest.raises(RuntimeError, match="Adapter staging"):
                await received.receive(1)
            assert calls == []
            assert set(received.admin_plane.update_times) == {"adapter_stage"}

    asyncio.run(run())


def test_shared_deadline_cancels_all_hosts_and_never_loads(code, tmp_path):
    async def run():
        cancelled = set()
        entered = set()
        calls = []

        async def stage(request):
            entered.add(request.url.host)
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.add(request.url.host)

        async def load(request):
            calls.append(request)
            return httpx.Response(200)

        async with (
            httpx.AsyncClient(base_url="http://host-a", transport=httpx.MockTransport(stage)) as first,
            httpx.AsyncClient(base_url="http://host-b", transport=httpx.MockTransport(stage)) as second,
            httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(load)) as engine,
        ):
            received = receiver(code, tmp_path, [first, second], engine, timeout=0.03)
            with pytest.raises(TimeoutError, match="host-a.*host-b"):
                await received.receive(1)
            assert entered == cancelled == {"host-a", "host-b"}
            assert calls == []
            assert received.admin_plane.update_times["adapter_stage"][0] < 0.2

    asyncio.run(run())


def test_empty_stage_list_keeps_local_load(code, tmp_path):
    async def run():
        calls = []

        async def load(request):
            calls.append(request.url.path)
            return httpx.Response(200)

        async with httpx.AsyncClient(base_url="http://engine", transport=httpx.MockTransport(load)) as engine:
            received = receiver(code, tmp_path, [], engine)
            await received.receive(1)
            assert calls == ["/load_lora_adapter"]
            assert set(received.admin_plane.update_times) == {"adapter_load"}
            watcher = code["WeightWatcher"](received, policy=SimpleNamespace(version=0), observers=[])
            assert "watcher/adapter_stage/max" not in watcher.gauges()

    asyncio.run(run())


def test_error_diagnostics_hide_url_credentials_and_query(code):
    async def run():
        async def stage(request):
            return httpx.Response(500)

        async with httpx.AsyncClient(
            base_url="http://secret-user:secret-pass@host-a/?secret-key=value", transport=httpx.MockTransport(stage)
        ) as client:
            with pytest.raises(RuntimeError) as failure:
                await code["stage_lora_adapter"](SimpleNamespace(stage_clients=[client]), Path("/adapter"), 1)
            rendered = "".join(traceback.format_exception(failure.value))
            assert "host-a" in rendered
            assert "HTTP 500" in rendered
            for secret in ("secret-user", "secret-pass", "secret-key"):
                assert secret not in rendered

    asyncio.run(run())


def test_admin_plane_closes_stage_clients(code):
    async def run():
        plane = object.__new__(code["AdminPlane"])
        plane.clients = [httpx.AsyncClient()]
        plane._router_clients = []
        plane.stage_clients = [httpx.AsyncClient(), httpx.AsyncClient()]
        await plane.aclose()
        assert all(client.is_closed for client in plane.clients + plane.stage_clients)

    asyncio.run(run())
