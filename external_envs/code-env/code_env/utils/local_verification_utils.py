import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

from .deepcoder_utils import (
    BASE_IMPORTS,
    compare_stdout_results,
    generate_cb_wrapper_script,
    process_input_output,
)

_PARALLEL_LIMIT = int(os.environ.get("CODE_ENV_LOCAL_PARALLEL_TESTS", "32"))


async def _run_python(script: Path, timeout: int, stdin: str | None = None) -> subprocess.CompletedProcess[str]:
    def _run() -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(script)],
            input=stdin,
            text=True,
            capture_output=True,
            timeout=timeout,
            cwd=str(script.parent),
        )

    try:
        return await asyncio.to_thread(_run)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess([sys.executable, str(script)], 124, "", "timeout")


async def run_standard_input(
    generated_code: str,
    inputs: List,
    outputs: List,
    timeout_per_test: int,
) -> list[bool | None]:
    with tempfile.TemporaryDirectory(prefix="code-env-") as tmp:
        root = Path(tmp)
        script = root / "script.py"
        script.write_text(generated_code, encoding="utf-8")
        sem = asyncio.Semaphore(_PARALLEL_LIMIT)

        async def run_single_test(test_case_inputs, test_case_outputs) -> bool:
            async with sem:
                if isinstance(test_case_inputs, list):
                    test_case_inputs = "\n".join(str(k) for k in test_case_inputs)
                if isinstance(test_case_outputs, list):
                    test_case_outputs = "\n".join(str(k) for k in test_case_outputs)
                proc = await _run_python(script, timeout=timeout_per_test, stdin=str(test_case_inputs))
                return proc.returncode == 0 and compare_stdout_results(proc.stdout, str(test_case_outputs))

        return await asyncio.gather(
            *[run_single_test(i, o) for i, o in zip(inputs, outputs)]
        )


async def run_func_call(
    generated_code: str,
    fn_name: str,
    inputs: List,
    outputs: List,
    timeout_per_test: int,
) -> list[bool | None]:
    with tempfile.TemporaryDirectory(prefix="code-env-") as tmp:
        root = Path(tmp)
        sem = asyncio.Semaphore(_PARALLEL_LIMIT)

        async def run_single_test(i: int, test_case_inputs, test_case_outputs) -> bool:
            async with sem:
                script = root / f"script_{i}.py"
                if isinstance(test_case_inputs, list):
                    test_case_inputs = "\n".join(repr(x) for x in test_case_inputs)
                script.write_text(
                    generate_cb_wrapper_script(generated_code, fn_name, str(test_case_inputs)),
                    encoding="utf-8",
                )
                proc = await _run_python(script, timeout=timeout_per_test)
                if proc.returncode != 0:
                    return False
                try:
                    result_data = json.loads(proc.stdout.strip())
                    if not result_data.get("success", False):
                        return False
                    exec_outputs = result_data["result"]
                    expected = json.loads(test_case_outputs) if isinstance(test_case_outputs, str) else test_case_outputs
                    if isinstance(exec_outputs, tuple):
                        exec_outputs = list(exec_outputs)
                    if exec_outputs == expected:
                        return True
                    if isinstance(expected, list) and exec_outputs == expected[0]:
                        return True
                    try:
                        if isinstance(exec_outputs[0], tuple):
                            exec_outputs = [list(x) for x in exec_outputs]
                            return isinstance(expected, list) and exec_outputs == expected[0]
                    except Exception:
                        pass
                    return False
                except Exception:
                    return False

        return await asyncio.gather(
            *[run_single_test(i, test_case_inputs, test_case_outputs) for i, (test_case_inputs, test_case_outputs) in enumerate(zip(inputs, outputs))]
        )


async def run_test_cases_local(generated_code: str, verification_info: dict) -> list[bool]:
    generated_code = f"{BASE_IMPORTS}\n{generated_code}"
    inputs = []
    outputs = []
    for test_case_inputs, test_case_outputs in zip(
        verification_info["test_case_inputs"], verification_info["test_case_outputs"]
    ):
        test_case_inputs = json.loads(test_case_inputs)
        test_case_outputs = json.loads(test_case_outputs)
        test_case_inputs, test_case_outputs = process_input_output(test_case_inputs, test_case_outputs)
        inputs.append(test_case_inputs)
        outputs.append(test_case_outputs)

    timeout = verification_info["timeout"]
    if not verification_info["fn_name"]:
        results = await run_standard_input(generated_code, inputs, outputs, timeout)
    else:
        results = await run_func_call(generated_code, verification_info["fn_name"], inputs, outputs, timeout)
    return [result for result in results if result is not None]
