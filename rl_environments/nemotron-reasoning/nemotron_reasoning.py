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
import random
import re
import sys
from typing import Any

import reasoning_gym
import verifiers as vf
from datasets import Dataset
from huggingface_hub import hf_hub_download
from nemotron_reasoning_guardrails import (
    AntiHackingConfig,
    compose_system_prompt,
    guard_env,
    merge_system_prompt,
    parse_bool,
)
from openai import AsyncOpenAI


class _RoundRobinChatCompletions:
    def __init__(self, clients: list[AsyncOpenAI]):
        self._clients = clients
        self._index = 0

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        client = self._clients[self._index % len(self._clients)]
        self._index += 1
        return await client.chat.completions.create(*args, **kwargs)


class _RoundRobinChat:
    def __init__(self, clients: list[AsyncOpenAI]):
        self.completions = _RoundRobinChatCompletions(clients)


class _RoundRobinOpenAI:
    def __init__(self, clients: list[AsyncOpenAI]):
        self.chat = _RoundRobinChat(clients)


def _normalize_base_urls(base_url: Any) -> list[str]:
    if base_url is None:
        return []
    if isinstance(base_url, str):
        value = base_url.strip()
        if value.startswith("["):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                parsed = value
            raw_urls = parsed if isinstance(parsed, (list, tuple)) else [parsed]
        else:
            raw_urls = [value]
    elif isinstance(base_url, (list, tuple)):
        raw_urls = base_url
    else:
        raw_urls = [base_url]

    urls: list[str] = []
    for raw_url in raw_urls:
        if raw_url is None:
            continue
        url = str(raw_url).strip().rstrip("/")
        if not url:
            continue
        if not url.endswith("/v1"):
            url = f"{url}/v1"
        urls.append(url)
    return urls


def _openai_client(api_key: str, base_url: Any, http_client: Any = None) -> Any:
    urls = _normalize_base_urls(base_url)
    if len(urls) <= 1:
        return AsyncOpenAI(api_key=api_key, base_url=urls[0] if urls else base_url, http_client=http_client)
    clients = [AsyncOpenAI(api_key=api_key, base_url=url, http_client=http_client) for url in urls]
    return _RoundRobinOpenAI(clients)


RG_DATASET = "nvidia/Nemotron-RL-ReasoningGym-v1"
MATH_DATASET = "nvidia/Nemotron-RL-Math-v2"
SCIENCE_DATASET = "nvidia/Nemotron-RL-Science-v1"
ARC_AGI_DATASET = "nvidia/Nemotron-RL-ARC-AGI-v1"
PYTHON_TOOL_NAME = "stateful_python_code_exec"
PYTHON_TOOL_OUTPUT_LIMIT = 8000

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
ANSWER_LINE_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)
JSON_GRID_RE = re.compile(r"(\[\s*\[.*?\]\s*\])", re.DOTALL)

SCORE_TAG_RE = re.compile(r"<score>\s*([01])\s*</score>|<score>\s*([01])", re.IGNORECASE)

EQUIVALENCE_PROMPT = """You are grading final-answer equivalence.

Task:
{question}

Reference answer:
{expected}

Candidate answer:
{candidate}

Score 1 if the candidate final answer is mathematically, scientifically, or factually equivalent to the reference answer. Score 0 otherwise. Ignore formatting differences, but do not give credit for unsupported guesses or answers that only partially overlap.

Output exactly:
<score>0</score> or <score>1</score>
"""


def _coerce(value: Any) -> Any:
    if value is None or isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            try:
                return ast.literal_eval(value)
            except Exception:
                return value
    return value


def _input_to_prompt(
    payload: Any,
    fallback_question: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    payload = _coerce(payload)
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


def _normalize_tool_def(raw_tool: Any) -> dict[str, Any] | None:
    raw_tool = _coerce(raw_tool)
    if not isinstance(raw_tool, dict):
        return None
    spec = raw_tool.get("function") if isinstance(raw_tool.get("function"), dict) else raw_tool
    name = str(spec.get("name") or "").strip()
    if not name:
        return None
    parameters = spec.get("parameters")
    if isinstance(parameters, str):
        try:
            parameters = json.loads(parameters)
        except Exception:
            parameters = {}
    if not isinstance(parameters, dict):
        parameters = {"type": "object", "properties": {}}
    tool: dict[str, Any] = {
        "name": name,
        "description": str(spec.get("description") or ""),
        "parameters": parameters,
    }
    strict = spec.get("strict", raw_tool.get("strict"))
    if strict is not None:
        tool["strict"] = bool(strict)
    return tool


def _extract_tool_defs(payload: Any) -> list[dict[str, Any]]:
    payload = _coerce(payload)
    tools = payload.get("tools") if isinstance(payload, dict) else None
    if not isinstance(tools, list):
        return []
    normalized = [_normalize_tool_def(tool) for tool in tools]
    return [tool for tool in normalized if tool is not None]


def _load_jsonl(repo: str, filename: str, num_examples: int, seed: int) -> list[dict[str, Any]]:
    path = hf_hub_download(repo, filename, repo_type="dataset")
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_examples > 0:
        rows = rows[:num_examples]
    return rows


def _completion_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        for message in reversed(completion):
            role = str(message.get("role", "") if isinstance(message, dict) else getattr(message, "role", "")).lower()
            if role == "assistant":
                return str((message.get("content") if isinstance(message, dict) else getattr(message, "content", "")) or "")
        for message in reversed(completion):
            content = str((message.get("content") if isinstance(message, dict) else getattr(message, "content", "")) or "")
            if content:
                return content
    return ""


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def _object_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _message_tool_calls(message: Any) -> list[Any]:
    calls = message.get("tool_calls") if isinstance(message, dict) else getattr(message, "tool_calls", None)
    return calls if isinstance(calls, list) else []


def _normalize_tool_args(args: Any) -> Any:
    args = _coerce(args)
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"raw": args}
    return args


def _tool_call_parts(tool_call: Any, index: int = 0) -> tuple[str, str, Any]:
    tool_call = _coerce(tool_call)
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
        if _message_role(message) != "assistant":
            continue
        for idx, tool_call in enumerate(_message_tool_calls(message)):
            call_id, name, arguments = _tool_call_parts(tool_call, idx)
            if name:
                calls.append({"id": call_id, "name": name, "arguments": arguments})
    return calls


def _last_boxed_content(text: str) -> str | None:
    starts = [m.start() for m in re.finditer(r"\\boxed\s*\{", text or "")]
    for start in reversed(starts):
        open_idx = text.find("{", start)
        if open_idx < 0:
            continue
        depth = 0
        for idx in range(open_idx, len(text)):
            char = text[idx]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[open_idx + 1:idx].strip()
    return None


def _extract_final_answer(text: str) -> str:
    if not text:
        return ""
    answer_matches = list(ANSWER_TAG_RE.finditer(text))
    if answer_matches:
        return answer_matches[-1].group(1).strip()
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return boxed
    for line in reversed(text.splitlines()):
        answer_line = ANSWER_LINE_RE.match(line)
        if answer_line:
            return answer_line.group(1).strip()
    return text.strip()


def _normalize_answer(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\\\[(.*?)\\\]", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"\\\((.*?)\\\)", r"\1", value, flags=re.DOTALL)
    value = value.replace("$", "")
    value = re.sub(r"\\boxed\{(.*?)\}", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"\s+", " ", value).strip().lower()
    value = value.strip(" .,:;`'\"")
    return value


async def _judge_equivalence(
    judge_client: AsyncOpenAI | None,
    judge_model: str,
    judge_sampling_args: dict[str, Any] | None,
    question: str,
    expected: str,
    candidate: str,
    state: vf.State | None = None,
) -> float:
    if _normalize_answer(candidate) == _normalize_answer(expected):
        return 1.0
    if judge_client is None:
        return 0.0
    sampling = {k: v for k, v in (judge_sampling_args or {}).items() if v is not None}
    sampling.setdefault("temperature", 0.0)
    sampling.setdefault("max_tokens", 32)
    prompt = EQUIVALENCE_PROMPT.format(question=question[-4000:], expected=expected, candidate=candidate)
    try:
        response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            **sampling,
        )
    except Exception as exc:
        if state is not None:
            logs = state.setdefault("judge_logs", [])
            if isinstance(logs, list):
                logs.append(
                    {
                        "kind": "nemotron_reasoning.equivalence",
                        "model": judge_model,
                        "prompt": prompt,
                        "response": None,
                        "error": repr(exc),
                    }
                )
        return 0.0
    verdict = str(response.choices[0].message.content or "")
    values = [
        match.group(1) or match.group(2)
        for match in SCORE_TAG_RE.finditer(verdict)
        if (match.group(1) or match.group(2)) in {"0", "1"}
    ]
    parsed_score = 0.0 if not values or len(set(values)) > 1 else (1.0 if values[-1] == "1" else 0.0)
    if state is not None:
        logs = state.setdefault("judge_logs", [])
        if isinstance(logs, list):
            logs.append(
                {
                    "kind": "nemotron_reasoning.equivalence",
                    "model": judge_model,
                    "prompt": prompt,
                    "response": verdict,
                    "error": None,
                    "parsed_score": parsed_score,
                }
            )
    if not values or len(set(values)) > 1:
        return 0.0
    return 1.0 if values[-1] == "1" else 0.0


def _prompt_text(prompt: Any) -> str:
    if not isinstance(prompt, list):
        return str(prompt)
    parts: list[str] = []
    for msg in prompt:
        role = str(msg.get("role", "") if isinstance(msg, dict) else getattr(msg, "role", "")).upper()
        content = str((msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", "")) or "")
        if role:
            parts.append(f"{role}: {content}")
    return "\n\n".join(parts)


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
        instruction = "Put your final answer on the last line as `Answer: \\\\boxed{X}`."
    else:
        instruction = "Put your final answer on the last line as `Answer: X`."
    messages[user_idx] = _set_message_content(messages[user_idx], f"{content}\n\n{instruction}".strip())
    return messages


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

    if stderr:
        return f"Error: Python tool runner wrote to stderr:\n{stderr.decode(errors='replace')[:PYTHON_TOOL_OUTPUT_LIMIT]}"
    try:
        result = json.loads(stdout.decode(errors="replace"))
    except Exception:
        return f"Error: Python tool runner returned invalid output:\n{stdout.decode(errors='replace')[:PYTHON_TOOL_OUTPUT_LIMIT]}"
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
        return _message_role(last_message) == "assistant" and not _message_tool_calls(last_message)

    async def env_response(self, messages: vf.Messages, state: vf.State, **_kwargs) -> vf.Messages:
        if not messages:
            return []
        tool_messages: list[vf.ToolMessage] = []
        for idx, tool_call in enumerate(_message_tool_calls(messages[-1])):
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
    model_answer = _extract_final_answer(_completion_text(completion))
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
    return 1.0 if _normalize_answer(model_answer) == _normalize_answer(answer) else 0.0


def _build_reasoning_gym(num_examples: int, seed: int, system_prompt: str | None) -> Dataset:
    raw = _load_jsonl(RG_DATASET, "data/train.jsonl", num_examples, seed)
    rows: list[dict[str, Any]] = []
    for row in raw:
        prompt = _input_to_prompt(row.get("input"), row.get("question"), system_prompt)
        answer = str(row.get("answer", "")).strip()
        metadata = _coerce(row.get("metadata"))
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


def _reasoning_gym_env(num_train_examples: int, num_eval_examples: int, dataset_seed: int, system_prompt: str | None) -> vf.Environment:
    return vf.SingleTurnEnv(
        dataset=lambda: _build_reasoning_gym(num_train_examples, dataset_seed, system_prompt),
        eval_dataset=lambda: _build_reasoning_gym(num_eval_examples, dataset_seed + 1, system_prompt),
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
    raw = _load_jsonl(repo, filename, -1, seed)
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
        metadata = _coerce(row.get("metadata"))
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
                    "tool_defs_json": json.dumps(_extract_tool_defs(responses_create_params), ensure_ascii=False),
                },
            }
        )
        if num_examples > 0 and len(rows) >= num_examples:
            break
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
        candidate = _extract_final_answer(_completion_text(completion))
        return await _judge_equivalence(
            judge_client,
            judge_model,
            judge_sampling_args,
            _prompt_text(prompt),
            str(answer),
            candidate,
            state,
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

    return env_cls(
        dataset=lambda: _build_expected_answer_rows(repo, filename, subtask, num_train_examples, dataset_seed, system_prompt),
        eval_dataset=lambda: _build_expected_answer_rows(repo, filename, subtask, num_eval_examples, dataset_seed + 1, system_prompt),
        rubric=rubric,
        system_prompt=system_prompt,
        **env_kwargs,
    )


# --- ARC-AGI transductive --------------------------------------------------


def _parse_grid(text: str) -> list[list[int]] | None:
    candidate = _extract_final_answer(text)
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
    predicted = _parse_grid(_completion_text(completion))
    try:
        expected = json.loads(answer)
    except Exception:
        expected = None
    return 1.0 if predicted is not None and predicted == expected else 0.0


def _build_arc_transductive(filename: str, num_examples: int, seed: int, system_prompt: str | None) -> Dataset:
    raw = _load_jsonl(ARC_AGI_DATASET, filename, num_examples, seed)
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


def _arc_agi_env(num_train_examples: int, num_eval_examples: int, dataset_seed: int, system_prompt: str | None) -> vf.Environment:
    return vf.SingleTurnEnv(
        dataset=lambda: _build_arc_transductive("data/transductive/train.jsonl", num_train_examples, dataset_seed, system_prompt),
        eval_dataset=lambda: _build_arc_transductive("data/transductive/validation.jsonl", num_eval_examples, dataset_seed + 1, system_prompt),
        rubric=vf.Rubric(funcs=[arc_grid_score]),
        system_prompt=system_prompt,
    )


# --- Composer --------------------------------------------------------------


_LOADERS = {
    "reasoning_gym": _reasoning_gym_env,
    "arc_agi": _arc_agi_env,
}

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
    enable_task_judges: bool = True,
    enable_anti_hacking: bool = True,
    enable_anti_hacking_judges: bool = True,
    anti_hacking_judge_model: str | None = None,
    anti_hacking_judge_base_url: str | None = None,
    anti_hacking_judge_api_key_var: str | None = None,
    anti_hacking_judge_timeout: float = 120.0,
    anti_hacking_incoherent_multiplier: float = 0.1,
    anti_hacking_meta_multiplier: float = 0.01,
    anti_hacking_reasoning_required: bool = True,
    anti_hacking_output_prompt: str | None = None,
    anti_hacking_format_reward_weight: float = 0.1,
    enable_structured_marker_gate: bool = False,
    enable_native_python_tools: bool = True,
    python_tool_timeout: float = 10.0,
    python_tool_max_turns: int = 3,
    **kwargs: Any,
) -> vf.Environment:
    """Load a Nemotron reasoning env or a comma-separated subset."""
    keys = _resolve_datasets(dataset)
    enable_task_judges = parse_bool(enable_task_judges)
    enable_anti_hacking = parse_bool(enable_anti_hacking)
    enable_anti_hacking_judges = parse_bool(enable_anti_hacking_judges)
    anti_hacking_reasoning_required = parse_bool(anti_hacking_reasoning_required)
    enable_structured_marker_gate = parse_bool(enable_structured_marker_gate)
    enable_native_python_tools = parse_bool(enable_native_python_tools)
    if enable_anti_hacking:
        system_prompt = compose_system_prompt(system_prompt, anti_hacking_output_prompt)

    needs_task_judge = enable_task_judges and any(key in _JUDGE_DATASETS for key in keys)
    needs_guard_judge = enable_anti_hacking and enable_anti_hacking_judges
    judge_client = None
    if needs_task_judge or needs_guard_judge:
        judge_client = _openai_client(api_key=os.environ.get(judge_api_key_var, "dummy-key"), base_url=judge_base_url)

    guard_config = None
    if enable_anti_hacking:
        guard_client = judge_client
        guard_model = anti_hacking_judge_model or judge_model
        guard_base_url = anti_hacking_judge_base_url or judge_base_url
        guard_key_var = anti_hacking_judge_api_key_var or judge_api_key_var
        if enable_anti_hacking_judges and (
            guard_client is None or guard_model != judge_model or guard_base_url != judge_base_url
        ):
            guard_client = _openai_client(api_key=os.environ.get(guard_key_var, "dummy-key"), base_url=guard_base_url)
        guard_config = AntiHackingConfig(
            judge_client=guard_client if enable_anti_hacking_judges else None,
            judge_model=guard_model,
            judge_sampling_args=judge_sampling_args or {"temperature": 0.0, "max_tokens": 96},
            judge_timeout=float(anti_hacking_judge_timeout),
            enable_judges=enable_anti_hacking_judges,
            reasoning_required=anti_hacking_reasoning_required,
            enable_structured_marker_gate=enable_structured_marker_gate,
            format_reward_weight=float(anti_hacking_format_reward_weight),
            incoherent_penalty_multiplier=float(anti_hacking_incoherent_multiplier),
            meta_commentary_multiplier=float(anti_hacking_meta_multiplier),
        )

    judge_args = (
        judge_client if needs_task_judge else None,
        judge_model,
        judge_sampling_args,
        num_train_examples,
        num_eval_examples,
        dataset_seed,
        system_prompt,
    )
    builders = {
        "reasoning_gym": lambda: _reasoning_gym_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "math": lambda: _expected_answer_env(
            MATH_DATASET,
            "data/train.jsonl",
            "math",
            *judge_args,
            enable_native_python_tools,
            float(python_tool_timeout),
            int(python_tool_max_turns),
        ),
        "science": lambda: _expected_answer_env(
            SCIENCE_DATASET,
            "so_openq.jsonl",
            "science",
            *judge_args,
            enable_native_python_tools,
            float(python_tool_timeout),
            int(python_tool_max_turns),
        ),
        "arc_agi": lambda: _arc_agi_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt),
    }

    envs: list[vf.Environment] = []
    names: list[str] = []
    for key in keys:
        envs.append(guard_env(builders[key](), guard_config))
        names.append(f"nemotron-reasoning-{key.replace('_', '-')}")
    if len(envs) == 1:
        return envs[0]
    return vf.EnvGroup(envs=envs, env_names=names)
