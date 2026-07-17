"""Knowledge category env for the clean Nemotron-RL blend.

Exposes a single ``load_environment`` that returns a ``vf.EnvGroup`` over the
selected sub-datasets:

* ``mcqa`` — nvidia/Nemotron-RL-knowledge-mcqa, scored by exact letter match
* ``qa_abstention`` — nvidia/Nemotron-RL-QA-Abstention-v1, scored by
  abstention-aware answer equivalence
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from nemotron_knowledge_guardrails import (
    coerce,
    extract_completion_text,
    extract_final_answer,
    format_prompt_for_judge,
    guard_env,
    judge_equivalence,
    last_boxed_content,
    load_jsonl,
    make_disjoint_dataset_builders,
    make_guard_config,
    merge_system_prompt,
    normalize_answer,
    openai_client,
    parse_bool,
)

MCQA_ANSWER_RE = re.compile(r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])", re.IGNORECASE)
MCQA_FALLBACK_RE = re.compile(r"\b([A-P])\b\s*$", re.MULTILINE)

ABSTENTION_ANSWERS = {"[idk]", "idk", "i don't know", "i do not know"}


def _input_to_prompt(
    payload: Any,
    fallback_question: str | None = None,
    system_prompt: str | None = None,
) -> list[dict[str, str]]:
    payload = coerce(payload)
    items = payload.get("input") if isinstance(payload, dict) else payload
    if isinstance(items, list):
        msgs: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip().lower() or "user"
            content = item.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p) for p in content
                )
            if str(content).strip():
                msgs.append({"role": role, "content": str(content)})
        if msgs:
            return merge_system_prompt(msgs, system_prompt)
    if fallback_question:
        return merge_system_prompt([{"role": "user", "content": str(fallback_question)}], system_prompt)
    return []


def _use_boxed_format(row_index: int, seed: int) -> bool:
    return (row_index + seed) % 2 == 0


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _rewrite_mcqa_prompt(prompt: list[dict[str, str]], boxed: bool) -> list[dict[str, str]]:
    messages = [dict(message) for message in prompt]
    user_idx = next((idx for idx, msg in enumerate(messages) if str(msg.get("role", "")).lower() == "user"), None)
    if user_idx is None:
        return messages
    content = _message_content(messages[user_idx])
    labels = re.findall(r"(?m)^([A-P])\s*:", content)
    choice_spec = "/".join(dict.fromkeys(labels)) or "A/B/C/D"
    first_choice = labels[0] if labels else "A"
    if boxed:
        instruction = (
            "Answer the following multiple choice question. The last line of your response should be in the "
            f"following format: 'Answer: \\boxed{{{choice_spec}}}' (e.g. 'Answer: \\boxed{{{first_choice}}}')."
        )
    else:
        instruction = (
            "Answer the following multiple choice question. The last line of your response should be in the "
            f"following format: 'Answer: {choice_spec}' (e.g. 'Answer: {first_choice}')."
        )
    # Replace via a function: a plain replacement string would interpret the
    # backslash escapes in `instruction` and turn \boxed into a backspace.
    content = re.sub(
        r"\AAnswer the following multiple choice question\..*?(?:\n\s*\n)",
        lambda _match: instruction + "\n\n",
        content,
        count=1,
        flags=re.DOTALL,
    )
    if content == _message_content(messages[user_idx]):
        content = f"{instruction}\n\n{content}"
    messages[user_idx] = {**messages[user_idx], "content": content}
    return messages


# --- MCQA ---------------------------------------------------------------------


def _mcqa_extract_letter(text: str) -> str:
    if not text:
        return ""
    boxed = last_boxed_content(text)
    if boxed is not None:
        boxed = boxed.strip().upper()
        if re.fullmatch(r"[A-P0-9]", boxed):
            return boxed
    m = MCQA_ANSWER_RE.findall(text)
    if m:
        return m[-1].strip().upper()
    m = MCQA_FALLBACK_RE.findall(text)
    if m:
        return m[-1].strip().upper()
    return text.strip().upper()


async def mcqa_correct(completion, answer, **_kwargs) -> float:
    predicted = _mcqa_extract_letter(extract_completion_text(completion))
    expected = str(answer).strip().upper()
    return 1.0 if predicted and predicted == expected else 0.0


def _build_mcqa(num_examples: int, seed: int, system_prompt: str | None) -> Dataset:
    raw = load_dataset("nvidia/Nemotron-RL-knowledge-mcqa", split="train").shuffle(seed=seed)
    if num_examples > 0:
        raw = raw.select(range(min(num_examples, len(raw))))
    rows: list[dict[str, Any]] = []
    for row in raw:
        prompt = _input_to_prompt(row.get("responses_create_params"), system_prompt=system_prompt)
        expected = str(row.get("expected_answer", "")).strip()
        if not prompt or not expected:
            continue
        boxed = _use_boxed_format(len(rows), seed)
        rows.append(
            {
                "prompt": _rewrite_mcqa_prompt(prompt, boxed),
                "answer": expected,
                "info": {
                    "uuid": str(row.get("uuid", "")),
                    "subtask": "mcqa",
                    "answer_format": "boxed" if boxed else "answer_line",
                },
            }
        )
    return Dataset.from_list(rows)


def _mcqa_env(
    num_train_examples: int, num_eval_examples: int, dataset_seed: int, system_prompt: str | None
) -> vf.Environment:
    train_dataset, eval_dataset = make_disjoint_dataset_builders(
        lambda num_examples: _build_mcqa(num_examples, dataset_seed, system_prompt),
        num_train_examples,
        num_eval_examples,
    )
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=vf.Rubric(funcs=[mcqa_correct]),
        system_prompt=system_prompt,
    )


# --- QA abstention -----------------------------------------------------------


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
        prompt = _input_to_prompt(
            row.get("responses_create_params"),
            row.get("question") or row.get("problem"),
            system_prompt,
        )
        expected = str(row.get("expected_answer") or row.get("answer") or "").strip()
        if not prompt or not expected:
            continue
        metadata = coerce(row.get("metadata"))
        if not isinstance(metadata, dict):
            metadata = {}
        rows.append(
            {
                "prompt": prompt,
                "answer": expected,
                "info": {
                    "uuid": str(row.get("uuid") or row.get("id") or metadata.get("uuid") or ""),
                    "domain": str(row.get("domain", "")),
                    "source": str(row.get("source", "")),
                    "verifier_type": str(row.get("verifier_type", "")),
                    "metadata_json": json.dumps(metadata, default=str),
                    "subtask": subtask,
                },
            }
        )
    return Dataset.from_list(rows)


def _qa_abstention_env(
    judge_client,
    judge_model,
    judge_sampling_args,
    num_train_examples: int,
    num_eval_examples: int,
    dataset_seed: int,
    system_prompt: str | None,
) -> vf.Environment:
    async def qa_score(prompt, completion, answer, state=None, **_kwargs) -> float:
        candidate = extract_final_answer(extract_completion_text(completion))
        if normalize_answer(candidate) in ABSTENTION_ANSWERS:
            return 1.0 if normalize_answer(answer) in ABSTENTION_ANSWERS else 0.0
        return await judge_equivalence(
            judge_client,
            judge_model,
            judge_sampling_args,
            format_prompt_for_judge(prompt),
            str(answer),
            candidate,
            state,
            kind="nemotron_knowledge.equivalence",
        )

    train_dataset, eval_dataset = make_disjoint_dataset_builders(
        lambda num_examples: _build_expected_answer_rows(
            "nvidia/Nemotron-RL-QA-Abstention-v1", "data/train.jsonl", "qa_abstention", num_examples, dataset_seed, system_prompt
        ),
        num_train_examples,
        num_eval_examples,
    )
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=vf.Rubric(funcs=[qa_score]),
        system_prompt=system_prompt,
    )


# --- Composer ----------------------------------------------------------------


_ALIASES = {
    "qa-abstention": "qa_abstention",
    "abstention": "qa_abstention",
}

_JUDGE_DATASETS = {"qa_abstention"}
_DEFAULT_DATASETS = ["mcqa", "qa_abstention"]


def _resolve_datasets(dataset: str) -> list[str]:
    valid = set(_DEFAULT_DATASETS)
    if not dataset or dataset == "all":
        return list(_DEFAULT_DATASETS)
    out: list[str] = []
    for d in dataset.split(","):
        key = _ALIASES.get(d.strip(), d.strip())
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
    **kwargs: Any,
) -> vf.Environment:
    """Knowledge env. ``dataset="all"`` selects MCQA and QA-abstention."""
    keys = _resolve_datasets(dataset)
    needs_judge = any(key in _JUDGE_DATASETS for key in keys) or (
        parse_bool(enable_anti_hacking) and parse_bool(enable_anti_hacking_judges)
    )
    judge_client = openai_client(os.environ.get(judge_api_key_var, "dummy-key"), judge_base_url) if needs_judge else None
    guard_config = make_guard_config(
        enable_anti_hacking, enable_anti_hacking_judges, judge_client, judge_model, judge_sampling_args,
        anti_hacking_reasoning_required, anti_hacking_allow_renderer_stripped_tool_calls, anti_hacking_format_reward_weight,
    )

    judge_args = (judge_client, judge_model, judge_sampling_args, num_train_examples, num_eval_examples, dataset_seed, system_prompt)
    builders = {
        "mcqa": lambda: _mcqa_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "qa_abstention": lambda: _qa_abstention_env(*judge_args),
    }
    envs = [guard_env(builders[key](), guard_config) for key in keys]
    names = [f"nemotron-knowledge-{key.replace('_', '-')}" for key in keys]
    return envs[0] if len(envs) == 1 else vf.EnvGroup(envs=envs, env_names=names)
