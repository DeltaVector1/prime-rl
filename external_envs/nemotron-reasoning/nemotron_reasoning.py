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
import json
import os
import random
import re
from typing import Any

import reasoning_gym
import verifiers as vf
from datasets import Dataset
from huggingface_hub import hf_hub_download
from openai import AsyncOpenAI

from nemotron_reasoning_guardrails import AntiHackingConfig, compose_system_prompt, guard_env, merge_system_prompt, parse_bool

RG_DATASET = "nvidia/Nemotron-RL-ReasoningGym-v1"
MATH_DATASET = "nvidia/Nemotron-RL-Math-v2"
SCIENCE_DATASET = "nvidia/Nemotron-RL-Science-v1"
ARC_AGI_DATASET = "nvidia/Nemotron-RL-ARC-AGI-v1"

ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
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
    except Exception:
        return 0.0
    verdict = str(response.choices[0].message.content or "")
    values = [
        match.group(1) or match.group(2)
        for match in SCORE_TAG_RE.finditer(verdict)
        if (match.group(1) or match.group(2)) in {"0", "1"}
    ]
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
        prompt = _input_to_prompt(
            row.get("responses_create_params"),
            row.get("question") or row.get("problem"),
            system_prompt,
        )
        expected = str(row.get("expected_answer") or row.get("answer") or "").strip()
        if not prompt or not expected:
            continue
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
) -> vf.Environment:
    async def answer_score(prompt, completion, answer, **_kwargs) -> float:
        candidate = _extract_final_answer(_completion_text(completion))
        return await _judge_equivalence(
            judge_client,
            judge_model,
            judge_sampling_args,
            _prompt_text(prompt),
            str(answer),
            candidate,
        )

    return vf.SingleTurnEnv(
        dataset=lambda: _build_expected_answer_rows(repo, filename, subtask, num_train_examples, dataset_seed, system_prompt),
        eval_dataset=lambda: _build_expected_answer_rows(repo, filename, subtask, num_eval_examples, dataset_seed + 1, system_prompt),
        rubric=vf.Rubric(funcs=[answer_score]),
        system_prompt=system_prompt,
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
    **kwargs: Any,
) -> vf.Environment:
    """Load a Nemotron reasoning env or a comma-separated subset."""
    keys = _resolve_datasets(dataset)
    enable_task_judges = parse_bool(enable_task_judges)
    enable_anti_hacking = parse_bool(enable_anti_hacking)
    enable_anti_hacking_judges = parse_bool(enable_anti_hacking_judges)
    anti_hacking_reasoning_required = parse_bool(anti_hacking_reasoning_required)
    enable_structured_marker_gate = parse_bool(enable_structured_marker_gate)
    if enable_anti_hacking:
        system_prompt = compose_system_prompt(system_prompt, anti_hacking_output_prompt)

    needs_task_judge = enable_task_judges and any(key in _JUDGE_DATASETS for key in keys)
    needs_guard_judge = enable_anti_hacking and enable_anti_hacking_judges
    judge_client = None
    if needs_task_judge or needs_guard_judge:
        judge_client = AsyncOpenAI(api_key=os.environ.get(judge_api_key_var, "dummy-key"), base_url=judge_base_url)

    guard_config = None
    if enable_anti_hacking:
        guard_client = judge_client
        guard_model = anti_hacking_judge_model or judge_model
        guard_base_url = anti_hacking_judge_base_url or judge_base_url
        guard_key_var = anti_hacking_judge_api_key_var or judge_api_key_var
        if enable_anti_hacking_judges and (
            guard_client is None or guard_model != judge_model or guard_base_url != judge_base_url
        ):
            guard_client = AsyncOpenAI(api_key=os.environ.get(guard_key_var, "dummy-key"), base_url=guard_base_url)
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
        "math": lambda: _expected_answer_env(MATH_DATASET, "data/train.jsonl", "math", *judge_args),
        "science": lambda: _expected_answer_env(SCIENCE_DATASET, "so_openq.jsonl", "science", *judge_args),
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
