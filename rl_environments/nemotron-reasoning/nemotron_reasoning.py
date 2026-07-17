"""Nemotron reasoning env for the clean Nemotron-RL blend.

Sub-datasets:

* ``reasoning_gym`` - nvidia/Nemotron-RL-ReasoningGym-v1
* ``math`` - nvidia/Nemotron-RL-Math-v2
* ``science`` - nvidia/Nemotron-RL-Science-v1
* ``arc_agi`` - ARC-AGI transductive output-grid tasks
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import json
import os
import re
import sys
from typing import Any

import reasoning_gym
import verifiers as vf
from datasets import Dataset
from nemotron_reasoning_guardrails import (
    coerce,
    extract_completion_text,
    extract_final_answer,
    extract_tool_defs,
    format_prompt_for_judge,
    guard_env,
    judge_equivalence,
    load_jsonl,
    make_disjoint_dataset_builders,
    make_guard_config,
    merge_system_prompt,
    message_role,
    message_tool_calls,
    normalize_answer,
    openai_client,
    parse_bool,
)

PYTHON_TOOL_NAME = "stateful_python_code_exec"
PYTHON_TOOL_OUTPUT_LIMIT = 8000

JSON_GRID_RE = re.compile(r"(\[\s*\[.*?\]\s*\])", re.DOTALL)


def _input_to_prompt(
    payload: Any,
    fallback_question: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    payload = coerce(payload)
    items = payload.get("input") if isinstance(payload, dict) else payload
    if isinstance(items, list):
        messages: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip().lower() or "user"
            if role not in {"system", "user", "assistant", "tool"}:
                role = "user"
            content = item.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(part.get("text") or part.get("content") or "") if isinstance(part, dict) else str(part)
                    for part in content
                )
            if str(content).strip():
                messages.append({"role": role, "content": str(content)})
        if messages:
            return merge_system_prompt(messages, system_prompt)
    if fallback_question:
        return merge_system_prompt([{"role": "user", "content": str(fallback_question)}], system_prompt)
    return []


def _object_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_tool_args(args: Any) -> Any:
    args = coerce(args)
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"raw": args}
    return args


def _tool_call_parts(tool_call: Any, index: int = 0) -> tuple[str, str, Any]:
    tool_call = coerce(tool_call)
    if hasattr(tool_call, "model_dump"):
        try:
            tool_call = tool_call.model_dump()
        except Exception:
            pass
    if not isinstance(tool_call, dict):
        call_id = str(_object_get(tool_call, "id", f"call_{index}") or f"call_{index}")
        name = str(_object_get(tool_call, "name", "") or "")
        args = _normalize_tool_args(_object_get(tool_call, "arguments", {}))
        return call_id, name, args

    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        args = function.get("arguments", {})
    else:
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})
    call_id = str(tool_call.get("id") or f"call_{index}")
    return call_id, str(name or ""), _normalize_tool_args(args)


def _completion_tool_calls(completion: Any) -> list[dict[str, Any]]:
    if not isinstance(completion, list):
        return []
    calls: list[dict[str, Any]] = []
    for message in completion:
        if message_role(message) != "assistant":
            continue
        for idx, tool_call in enumerate(message_tool_calls(message)):
            call_id, name, arguments = _tool_call_parts(tool_call, idx)
            if name:
                calls.append({"id": call_id, "name": name, "arguments": arguments})
    return calls


def _use_boxed_format(row_index: int, seed: int) -> bool:
    return (row_index + seed) % 2 == 0


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _set_message_content(message: Any, content: str) -> Any:
    if isinstance(message, dict):
        updated = dict(message)
        updated["content"] = content
        return updated
    setattr(message, "content", content)
    return message


def _strip_final_answer_format_instruction(content: str) -> str:
    patterns = [
        r"Make sure your answer is inside\s*\\boxed\s*\{\s*\}\s*\.?",
        r"The final answer must be placed at the end of your response and enclosed within\s*\\boxed\s*\{\s*\}\s*\.?\s*It is essential to adhere to this format\.?",
        r"Put your final answer in\s*\\boxed\s*\{\s*\}\s*\.?",
        r"Express your answer using\s*\\boxed\s*\{\s*\}\s*\.?",
        r"Provide just the answer inside\s*\\boxed\s*\{\s*\}\s*\.?",
        r"Give the answer in\s*\\boxed\s*\{\s*\}\s*format\.?",
        r"Place your final answer in\s*\\boxed\s*\{\s*\}\s*\.?",
        r"Make sure to use\s*\\boxed\s*\{\s*\}\s*for your answer\.?",
        r"Conclude with\s*\(Answer:\s*X\),\s*where X is the final answer\.?",
        r"The last line of your response should be in the following format:\s*'Answer:\s*\\boxed\s*\{.*?\}'\s*\(e\.g\.\s*'.*?'\)\.?",
        r"[^.\n]*\\boxed\s*\{\s*\}[^.\n]*(?:\.|$)",
    ]
    stripped = content
    for pattern in patterns:
        stripped = re.sub(pattern, "", stripped, flags=re.IGNORECASE | re.DOTALL)
    stripped = re.sub(r"[ \t]+\n", "\n", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped)
    return stripped.strip()


def _rewrite_final_answer_prompt(prompt: list[dict[str, str]], boxed: bool) -> list[dict[str, str]]:
    messages = [dict(message) for message in prompt]
    user_idx = next((idx for idx, msg in enumerate(messages) if str(msg.get("role", "")).lower() == "user"), None)
    if user_idx is None:
        return messages
    content = _strip_final_answer_format_instruction(_message_content(messages[user_idx]))
    if boxed:
        instruction = "Put your final answer on the last line as `Answer: \\boxed{X}`."
    else:
        instruction = "Put your final answer on the last line as `Answer: X`."
    messages[user_idx] = _set_message_content(messages[user_idx], f"{content}\n\n{instruction}".strip())
    return messages


# Runs model-authored code in a fresh `python -I` subprocess. The builtins
# allowlist below raises the cost of casual mischief but is NOT a security
# boundary: restricted-exec Python is escapable (e.g. via __subclasses__), and
# numpy/sympy can reach the filesystem anyway. It relies on the training host
# being trusted. Put it behind a real sandbox before running untrusted policies.
PYTHON_TOOL_RUNNER = r"""
import ast
import contextlib
import io
import json
import sys
import traceback

ALLOWED_IMPORT_ROOTS = {
    "cmath",
    "collections",
    "decimal",
    "fractions",
    "functools",
    "itertools",
    "math",
    "numpy",
    "random",
    "re",
    "statistics",
    "sympy",
}


def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = str(name).split(".", 1)[0]
    if root not in ALLOWED_IMPORT_ROOTS:
        raise ImportError(f"import of {name!r} is not available in this tool")
    return __import__(name, globals, locals, fromlist, level)


SAFE_BUILTINS = {
    "__build_class__": __build_class__,
    "__import__": safe_import,
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "Exception": Exception,
    "filter": filter,
    "float": float,
    "format": format,
    "int": int,
    "isinstance": isinstance,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "pow": pow,
    "print": print,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "TypeError": TypeError,
    "ValueError": ValueError,
    "zip": zip,
}


def execute_cell(code, namespace):
    tree = ast.parse(str(code), mode="exec")
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        prefix = ast.Module(body=tree.body[:-1], type_ignores=[])
        ast.fix_missing_locations(prefix)
        exec(compile(prefix, "<python_tool>", "exec"), namespace, namespace)
        expr = ast.Expression(tree.body[-1].value)
        ast.fix_missing_locations(expr)
        value = eval(compile(expr, "<python_tool>", "eval"), namespace, namespace)
        if value is not None:
            print(repr(value))
    else:
        exec(compile(tree, "<python_tool>", "exec"), namespace, namespace)


def main():
    payload = json.loads(sys.stdin.read() or "{}")
    namespace = {"__builtins__": SAFE_BUILTINS, "__name__": "__python_tool__"}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            for previous in payload.get("history") or []:
                execute_cell(previous, namespace)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            execute_cell(payload.get("code", ""), namespace)
        print(json.dumps({"stdout": output.getvalue(), "error": None}))
    except BaseException as exc:
        print(
            json.dumps(
                {
                    "stdout": locals().get("output", io.StringIO()).getvalue()
                    if "output" in locals()
                    else "",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=3),
                }
            )
        )


if __name__ == "__main__":
    main()
"""


async def _run_python_tool(code: str, state: vf.State, timeout: float) -> str:
    if not code.strip():
        return "Error: missing required `code` argument."
    history = state.setdefault("python_tool_history", [])
    payload = json.dumps({"history": history, "code": code})
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-I",
            "-c",
            PYTHON_TOOL_RUNNER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(payload.encode()), timeout=timeout)
    except asyncio.TimeoutError:
        with contextlib.suppress(Exception):
            proc.kill()
            await proc.wait()
        return f"Error: Python execution timed out after {timeout:g} seconds."
    except Exception as exc:
        return f"Error: Python execution failed to start: {type(exc).__name__}: {exc}"

    # stderr alone is not a failure: numpy and sympy routinely warn there on a
    # perfectly good run. Only the runner's JSON verdict on stdout decides, and
    # stderr is surfaced solely when that verdict is missing or unreadable.
    try:
        result = json.loads(stdout.decode(errors="replace"))
    except Exception:
        details = stdout.decode(errors="replace").strip() or stderr.decode(errors="replace").strip()
        return f"Error: Python tool runner returned invalid output:\n{details}"[:PYTHON_TOOL_OUTPUT_LIMIT]
    if result.get("error"):
        output = str(result.get("stdout") or "").strip()
        error = str(result.get("error") or "unknown error")
        return (f"{output}\nError: {error}" if output else f"Error: {error}")[:PYTHON_TOOL_OUTPUT_LIMIT]

    history.append(code)
    output = str(result.get("stdout") or "").strip()
    if not output:
        output = "Execution completed with no output."
    return output[:PYTHON_TOOL_OUTPUT_LIMIT]


class PythonToolAnswerEnv(vf.MultiTurnEnv):
    """Final-answer task with optional row-provided native Python tool access."""

    def __init__(self, python_tool_timeout: float = 10.0, max_turns: int = 3, **kwargs: Any):
        self.python_tool_timeout = float(python_tool_timeout)
        super().__init__(max_turns=max_turns, **kwargs)

    async def setup_state(self, state: vf.State) -> vf.State:
        await super().setup_state(state)
        info = state.get("info") if isinstance(state.get("info"), dict) else {}
        try:
            tool_defs = json.loads(info.get("tool_defs_json") or "[]")
        except Exception:
            tool_defs = []
        state["tool_defs"] = self._normalize_tool_defs(tool_defs) or []
        state["python_tool_history"] = []
        state["executed_python_tool_calls"] = []

        if state["tool_defs"]:
            sampling_args = dict(state.get("sampling_args") or {})
            extra_body = dict(sampling_args.get("extra_body") or {})
            extra_body.setdefault("tool_choice", "auto")
            sampling_args["extra_body"] = extra_body
            state["sampling_args"] = sampling_args
        return state

    @vf.stop
    async def no_tools_called(self, state: vf.State) -> bool:
        trajectory = state.get("trajectory") or []
        if not trajectory:
            return False
        completion = trajectory[-1].get("completion") if isinstance(trajectory[-1], dict) else None
        if not completion:
            return False
        last_message = completion[-1]
        return message_role(last_message) == "assistant" and not message_tool_calls(last_message)

    async def env_response(self, messages: vf.Messages, state: vf.State, **_kwargs) -> vf.Messages:
        if not messages:
            return []
        tool_messages: list[vf.ToolMessage] = []
        for idx, tool_call in enumerate(message_tool_calls(messages[-1])):
            tool_call_id, tool_name, tool_args = _tool_call_parts(tool_call, idx)
            if tool_name != PYTHON_TOOL_NAME:
                content = f"Error: unknown tool {tool_name!r}."
            elif not isinstance(tool_args, dict):
                content = f"Error: expected JSON object arguments for {PYTHON_TOOL_NAME}."
            else:
                code = str(tool_args.get("code") or "")
                content = await _run_python_tool(code, state, self.python_tool_timeout)
                state.setdefault("executed_python_tool_calls", []).append({"name": tool_name, "code": code})
            tool_messages.append(vf.ToolMessage(role="tool", content=content, tool_call_id=tool_call_id))
        return tool_messages


# --- ReasoningGym ---------------------------------------------------------


_RG_SCORE_FN_CACHE: dict[str, Any] = {}


def _resolve_rg_score_fn(source_dataset: str):
    if not source_dataset:
        return None
    if source_dataset in _RG_SCORE_FN_CACHE:
        return _RG_SCORE_FN_CACHE[source_dataset]
    try:
        fn = reasoning_gym.get_score_answer_fn(source_dataset)
    except Exception:
        fn = None
    _RG_SCORE_FN_CACHE[source_dataset] = fn
    return fn


async def reasoning_gym_score(completion, answer, info, **_kwargs) -> float:
    model_answer = extract_final_answer(extract_completion_text(completion))
    source_dataset = (info or {}).get("source_dataset", "")
    score_fn = _resolve_rg_score_fn(source_dataset)
    if score_fn is not None:
        try:
            entry_metadata = json.loads((info or {}).get("entry_metadata_json", "{}"))
        except Exception:
            entry_metadata = {}
        try:
            return float(score_fn(answer=model_answer, entry={"answer": answer, "metadata": entry_metadata}))
        except Exception:
            pass
    return 1.0 if normalize_answer(model_answer) == normalize_answer(answer) else 0.0


def _build_reasoning_gym(num_examples: int, seed: int, system_prompt: str | None) -> Dataset:
    raw = load_jsonl("nvidia/Nemotron-RL-ReasoningGym-v1", "data/train.jsonl", num_examples, seed)
    rows: list[dict[str, Any]] = []
    for row in raw:
        prompt = _input_to_prompt(row.get("input"), row.get("question"), system_prompt)
        answer = str(row.get("answer", "")).strip()
        metadata = coerce(row.get("metadata"))
        if not prompt or not answer:
            continue
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "prompt": prompt,
                "answer": answer,
                "info": {
                    "uuid": str(row.get("uuid", "")),
                    "source_dataset": str(metadata.get("source_dataset", "")),
                    "entry_metadata_json": json.dumps(metadata, default=str),
                    "subtask": "reasoning_gym",
                },
            }
        )
    return Dataset.from_list(rows)


def _reasoning_gym_env(
    num_train_examples: int, num_eval_examples: int, dataset_seed: int, system_prompt: str | None
) -> vf.Environment:
    train_dataset, eval_dataset = make_disjoint_dataset_builders(
        lambda num_examples: _build_reasoning_gym(num_examples, dataset_seed, system_prompt),
        num_train_examples,
        num_eval_examples,
    )
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=vf.Rubric(funcs=[reasoning_gym_score]),
        system_prompt=system_prompt,
    )


# --- Answer-equivalence datasets -----------------------------------------


def _build_expected_answer_rows(
    repo: str,
    filename: str,
    subtask: str,
    num_examples: int,
    seed: int,
    system_prompt: str | None,
) -> Dataset:
    raw = load_jsonl(repo, filename, num_examples, seed)
    rows: list[dict[str, Any]] = []
    for row in raw:
        responses_create_params = row.get("responses_create_params")
        prompt = _input_to_prompt(
            responses_create_params,
            row.get("question") or row.get("problem"),
            system_prompt,
        )
        expected = str(row.get("expected_answer") or row.get("answer") or "").strip()
        if not prompt or not expected:
            continue
        boxed = _use_boxed_format(len(rows), seed) if subtask in {"math", "science"} else True
        if subtask in {"math", "science"}:
            prompt = _rewrite_final_answer_prompt(prompt, boxed)
        metadata = coerce(row.get("metadata"))
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "prompt": prompt,
                "answer": expected,
                "info": {
                    "uuid": str(row.get("uuid") or row.get("id") or metadata.get("uuid") or ""),
                    "verifier_type": str(row.get("verifier_type", "")),
                    "metadata_json": json.dumps(metadata, default=str),
                    "subtask": subtask,
                    "answer_format": "boxed" if boxed else "answer_line",
                    "tool_defs_json": json.dumps(extract_tool_defs(responses_create_params), ensure_ascii=False),
                },
            }
        )
    return Dataset.from_list(rows)


def _expected_answer_env(
    repo: str,
    filename: str,
    subtask: str,
    judge_client,
    judge_model: str,
    judge_sampling_args: dict[str, Any] | None,
    num_train_examples: int,
    num_eval_examples: int,
    dataset_seed: int,
    system_prompt: str | None,
    enable_native_python_tools: bool = True,
    python_tool_timeout: float = 10.0,
    python_tool_max_turns: int = 3,
) -> vf.Environment:
    async def answer_score(prompt, completion, answer, state=None, **_kwargs) -> float:
        candidate = extract_final_answer(extract_completion_text(completion))
        return await judge_equivalence(
            judge_client,
            judge_model,
            judge_sampling_args,
            format_prompt_for_judge(prompt),
            str(answer),
            candidate,
            state,
            kind=f"nemotron_reasoning.{subtask}.equivalence",
        )

    async def python_tool_call_emitted(completion, **_kwargs) -> float:
        return 1.0 if any(call.get("name") == PYTHON_TOOL_NAME for call in _completion_tool_calls(completion)) else 0.0

    rubric = vf.Rubric(funcs=[answer_score])
    if enable_native_python_tools:
        rubric.add_metric(python_tool_call_emitted)
        env_cls: type[vf.MultiTurnEnv] = PythonToolAnswerEnv
        env_kwargs = {
            "python_tool_timeout": python_tool_timeout,
            "max_turns": python_tool_max_turns,
        }
    else:
        env_cls = vf.SingleTurnEnv
        env_kwargs = {}

    train_dataset, eval_dataset = make_disjoint_dataset_builders(
        lambda num_examples: _build_expected_answer_rows(
            repo, filename, subtask, num_examples, dataset_seed, system_prompt
        ),
        num_train_examples,
        num_eval_examples,
    )
    return env_cls(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
        **env_kwargs,
    )


# --- ARC-AGI transductive --------------------------------------------------


def _parse_grid(text: str) -> list[list[int]] | None:
    candidate = extract_final_answer(text)
    match = JSON_GRID_RE.search(candidate)
    if match:
        try:
            parsed = json.loads(match.group(1))
            if _is_grid(parsed):
                return parsed
        except Exception:
            pass
    try:
        parsed = ast.literal_eval(candidate)
        if _is_grid(parsed):
            return parsed
    except Exception:
        pass
    rows: list[list[int]] = []
    for raw_line in candidate.splitlines():
        line = raw_line.strip().strip("|")
        if not line:
            continue
        if re.fullmatch(r"[0-9 ]+", line):
            parts = line.split() if " " in line else list(line)
            rows.append([int(part) for part in parts])
    return rows if _is_grid(rows) else None


def _is_grid(value: Any) -> bool:
    return (
        isinstance(value, list)
        and bool(value)
        and all(isinstance(row, list) and row and all(isinstance(cell, int) for cell in row) for row in value)
    )


async def arc_grid_score(completion, answer, **_kwargs) -> float:
    predicted = _parse_grid(extract_completion_text(completion))
    try:
        expected = json.loads(answer)
    except Exception:
        expected = None
    return 1.0 if predicted is not None and predicted == expected else 0.0


def _build_arc_transductive(filename: str, num_examples: int, seed: int, system_prompt: str | None) -> Dataset:
    raw = load_jsonl("nvidia/Nemotron-RL-ARC-AGI-v1", filename, num_examples, seed)
    rows: list[dict[str, Any]] = []
    for row in raw:
        prompt = _input_to_prompt(row.get("responses_create_params"), system_prompt=system_prompt)
        expected = row.get("expected_output")
        if not prompt or not _is_grid(expected):
            continue
        rows.append(
            {
                "prompt": prompt,
                "answer": json.dumps(expected),
                "info": {
                    "problem_id": str(row.get("problem_id", "")),
                    "task_id": str(row.get("task_id", "")),
                    "difficulty_bucket": str(row.get("difficulty_bucket", "")),
                    "subtask": "arc_agi_transductive",
                },
            }
        )
    return Dataset.from_list(rows)


def _arc_agi_env(
    num_train_examples: int, num_eval_examples: int, dataset_seed: int, system_prompt: str | None
) -> vf.Environment:
    # ARC ships its own train/validation files, so no disjoint split is needed.
    return vf.SingleTurnEnv(
        dataset=lambda: _build_arc_transductive(
            "data/transductive/train.jsonl", num_train_examples, dataset_seed, system_prompt
        ),
        eval_dataset=lambda: _build_arc_transductive(
            "data/transductive/validation.jsonl", num_eval_examples, dataset_seed + 1, system_prompt
        ),
        rubric=vf.Rubric(funcs=[arc_grid_score]),
        system_prompt=system_prompt,
    )


# --- Composer --------------------------------------------------------------


_ALIASES = {
    "reasoning-gym": "reasoning_gym",
    "rg": "reasoning_gym",
    "math_v2": "math",
    "math-v2": "math",
    "science_v1": "science",
    "arc": "arc_agi",
    "arc-agi": "arc_agi",
}

_JUDGE_DATASETS = {"math", "science"}
_DEFAULT_DATASETS = ["reasoning_gym", "math", "science", "arc_agi"]


def _resolve_datasets(dataset: str) -> list[str]:
    valid = {"reasoning_gym", "math", "science", "arc_agi"}
    if not dataset or dataset == "all":
        return list(_DEFAULT_DATASETS)
    out: list[str] = []
    for name in dataset.split(","):
        key = _ALIASES.get(name.strip(), name.strip())
        if key in valid:
            out.append(key)
    return out or list(_DEFAULT_DATASETS)


def load_environment(
    dataset: str = "all",
    num_train_examples: int = -1,
    num_eval_examples: int = 256,
    dataset_seed: int = 42,
    system_prompt: str | None = None,
    judge_model: str = "google/gemma-4-26B-A4B-it",
    judge_base_url: str | None = "http://127.0.0.1:8000/v1",
    judge_api_key_var: str = "VLLM_API_KEY",
    judge_sampling_args: dict | None = None,
    enable_anti_hacking: bool = True,
    enable_anti_hacking_judges: bool = True,
    anti_hacking_reasoning_required: bool = False,
    anti_hacking_allow_renderer_stripped_tool_calls: bool = False,
    anti_hacking_format_reward_weight: float = 0.1,
    enable_native_python_tools: bool = True,
    python_tool_timeout: float = 10.0,
    python_tool_max_turns: int = 3,
    **kwargs: Any,
) -> vf.Environment:
    """Load a Nemotron reasoning env or a comma-separated subset."""
    keys = _resolve_datasets(dataset)
    enable_native_python_tools = parse_bool(enable_native_python_tools)
    needs_judge = any(key in _JUDGE_DATASETS for key in keys) or (
        parse_bool(enable_anti_hacking) and parse_bool(enable_anti_hacking_judges)
    )
    judge_client = openai_client(os.environ.get(judge_api_key_var, "dummy-key"), judge_base_url) if needs_judge else None
    guard_config = make_guard_config(
        enable_anti_hacking, enable_anti_hacking_judges, judge_client, judge_model, judge_sampling_args,
        anti_hacking_reasoning_required, anti_hacking_allow_renderer_stripped_tool_calls, anti_hacking_format_reward_weight,
    )

    judge_args = (judge_client, judge_model, judge_sampling_args, num_train_examples, num_eval_examples, dataset_seed, system_prompt)
    python_tool = (enable_native_python_tools, float(python_tool_timeout), int(python_tool_max_turns))
    builders = {
        "reasoning_gym": lambda: _reasoning_gym_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "math": lambda: _expected_answer_env("nvidia/Nemotron-RL-Math-v2", "data/train.jsonl", "math", *judge_args, *python_tool),
        "science": lambda: _expected_answer_env("nvidia/Nemotron-RL-Science-v1", "so_openq.jsonl", "science", *judge_args, *python_tool),
        "arc_agi": lambda: _arc_agi_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt),
    }
    envs = [guard_env(builders[key](), guard_config) for key in keys]
    names = [f"nemotron-reasoning-{key.replace('_', '-')}" for key in keys]
    return envs[0] if len(envs) == 1 else vf.EnvGroup(envs=envs, env_names=names)
