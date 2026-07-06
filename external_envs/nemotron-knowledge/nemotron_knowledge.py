"""Knowledge category env for the clean Nemotron-RL blend.

Exposes a single ``load_environment`` that returns a ``vf.EnvGroup`` over the
selected sub-datasets:

* ``mcqa`` — nvidia/Nemotron-RL-knowledge-mcqa, scored by exact letter match
* ``qa_abstention`` — nvidia/Nemotron-RL-QA-Abstention-v1, scored by
  abstention-aware answer equivalence
"""

from __future__ import annotations

import ast
import json
import os
import random
import re
from typing import Any

import verifiers as vf
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from nemotron_knowledge_guardrails import (
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


MCQA_DATASET = "nvidia/Nemotron-RL-knowledge-mcqa"
QA_ABSTENTION_DATASET = "nvidia/Nemotron-RL-QA-Abstention-v1"

MCQA_ANSWER_RE = re.compile(r"Answer\s*:\s*(?!Answer)\s*([A-Za-z0-9])", re.IGNORECASE)
MCQA_FALLBACK_RE = re.compile(r"\b([A-P])\b\s*$", re.MULTILINE)
ANSWER_LINE_RE = re.compile(r"^\s*(?:final\s+)?answer\s*[:=]\s*(.+?)\s*$", re.IGNORECASE)
ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BOXED_RE = re.compile(r"\\boxed\{(.+?)\}")
SCORE_TAG_RE = re.compile(r"<score>\s*([01])\s*</score>|<score>\s*([01])", re.IGNORECASE)

EQUIVALENCE_PROMPT = """You are grading final-answer equivalence.

Task:
{question}

Reference answer:
{expected}

Candidate answer:
{candidate}

Score 1 if the candidate final answer is factually equivalent to the reference answer. Score 0 otherwise. Ignore formatting differences, but do not give credit for unsupported guesses or partial overlap.

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
        msgs: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "user")).strip().lower() or "user"
            content = item.get("content", "")
            if isinstance(content, list):
                content = "\n".join(
                    str(p.get("text") or p.get("content") or "") if isinstance(p, dict) else str(p)
                    for p in content
                )
            if str(content).strip():
                msgs.append({"role": role, "content": str(content)})
        if msgs:
            return merge_system_prompt(msgs, system_prompt)
    if fallback_question:
        return merge_system_prompt([{"role": "user", "content": str(fallback_question)}], system_prompt)
    return []


def _load_jsonl(repo: str, filename: str, num_examples: int, seed: int) -> list[dict[str, Any]]:
    path = hf_hub_download(repo, filename, repo_type="dataset")
    raw: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                raw.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    rng = random.Random(seed)
    rng.shuffle(raw)
    if num_examples > 0:
        raw = raw[:num_examples]
    return raw


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


def _normalize_answer(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\\\[(.*?)\\\]", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"\\\((.*?)\\\)", r"\1", value, flags=re.DOTALL)
    value = value.replace("$", "")
    value = re.sub(r"\\boxed\{(.*?)\}", r"\1", value, flags=re.DOTALL)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value.strip(" .,:;`'\"")


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
            f"following format: 'Answer: \\\\boxed{{{choice_spec}}}' (e.g. 'Answer: \\\\boxed{{{first_choice}}}')."
        )
    else:
        instruction = (
            "Answer the following multiple choice question. The last line of your response should be in the "
            f"following format: 'Answer: {choice_spec}' (e.g. 'Answer: {first_choice}')."
        )
    content = re.sub(
        r"\AAnswer the following multiple choice question\..*?(?:\n\s*\n)",
        instruction + "\n\n",
        content,
        count=1,
        flags=re.DOTALL,
    )
    if content == _message_content(messages[user_idx]):
        content = f"{instruction}\n\n{content}"
    messages[user_idx] = _set_message_content(messages[user_idx], content)
    return messages


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
    judge_input = EQUIVALENCE_PROMPT.format(question=question[-4000:], expected=expected, candidate=candidate)
    try:
        response = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_input}],
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


# --- MCQA ---------------------------------------------------------------------


def _mcqa_extract_letter(text: str) -> str:
    if not text:
        return ""
    boxed = _last_boxed_content(text)
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
    text = _completion_text(completion)
    predicted = _mcqa_extract_letter(text)
    expected = str(answer).strip().upper()
    return 1.0 if predicted and predicted == expected else 0.0


def _build_mcqa(num_examples: int, seed: int, system_prompt: str | None) -> Dataset:
    raw = load_dataset(MCQA_DATASET, split="train")
    if seed is not None:
        raw = raw.shuffle(seed=seed)
    if num_examples > 0:
        raw = raw.select(range(min(num_examples, len(raw))))
    rows: list[dict[str, Any]] = []
    for row in raw:
        prompt = _input_to_prompt(row.get("responses_create_params"), system_prompt=system_prompt)
        expected = str(row.get("expected_answer", "")).strip()
        if not prompt or not expected:
            continue
        boxed = _use_boxed_format(len(rows), seed)
        prompt = _rewrite_mcqa_prompt(prompt, boxed)
        rows.append(
            {
                "prompt": prompt,
                "answer": expected,
                "info": {
                    "uuid": str(row.get("uuid", "")),
                    "subtask": "mcqa",
                    "answer_format": "boxed" if boxed else "answer_line",
                },
            }
        )
    return Dataset.from_list(rows)


def _mcqa_env(num_train_examples: int, num_eval_examples: int, dataset_seed: int, system_prompt: str | None) -> vf.Environment:
    return vf.SingleTurnEnv(
        dataset=lambda: _build_mcqa(num_train_examples, dataset_seed, system_prompt),
        eval_dataset=lambda: _build_mcqa(num_eval_examples, dataset_seed + 1, system_prompt),
        rubric=vf.Rubric(funcs=[mcqa_correct]),
        system_prompt=system_prompt,
    )


# --- Answer extraction -------------------------------------------------------


def _extract_response_answer(text: str) -> str:
    if not text:
        return ""
    m = list(ANSWER_TAG_RE.finditer(text))
    if m:
        return m[-1].group(1).strip()
    boxed = _last_boxed_content(text)
    if boxed is not None:
        return boxed
    for line in reversed(text.splitlines()):
        answer_line = ANSWER_LINE_RE.match(line)
        if answer_line:
            return answer_line.group(1).strip()
    return text.strip()


# --- QA abstention -----------------------------------------------------------


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
                    "domain": str(row.get("domain", "")),
                    "source": str(row.get("source", "")),
                    "verifier_type": str(row.get("verifier_type", "")),
                    "metadata_json": json.dumps(metadata, default=str),
                    "subtask": subtask,
                },
            }
        )
        if num_examples > 0 and len(rows) >= num_examples:
            break
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
    async def qa_score(prompt, completion, answer, **_kwargs) -> float:
        candidate = _extract_response_answer(_completion_text(completion))
        if _normalize_answer(candidate) in {"[idk]", "idk", "i don't know", "i do not know"}:
            return 1.0 if _normalize_answer(answer) in {"[idk]", "idk", ""} else 0.0
        return await _judge_equivalence(
            judge_client,
            judge_model,
            judge_sampling_args,
            _prompt_text(prompt),
            str(answer),
            candidate,
        )

    return vf.SingleTurnEnv(
        dataset=lambda: _build_expected_answer_rows(
            QA_ABSTENTION_DATASET, "data/train.jsonl", "qa_abstention", num_train_examples, dataset_seed, system_prompt
        ),
        eval_dataset=lambda: _build_expected_answer_rows(
            QA_ABSTENTION_DATASET, "data/train.jsonl", "qa_abstention", num_eval_examples, dataset_seed + 1, system_prompt
        ),
        rubric=vf.Rubric(funcs=[qa_score]),
        system_prompt=system_prompt,
    )


# --- Composer ----------------------------------------------------------------


_LOADERS = {
    "mcqa": _mcqa_env,
}

_ALIASES = {
    "qa-abstention": "qa_abstention",
    "abstention": "qa_abstention",
}

_JUDGE_DATASETS = {"qa_abstention"}
_DEFAULT_DATASETS = ["mcqa", "qa_abstention"]


def _resolve_datasets(dataset: str) -> list[str]:
    valid = {"mcqa", "qa_abstention"}
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
    """Knowledge env. ``dataset="all"`` selects MCQA and QA-abstention."""
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
        "mcqa": lambda: _mcqa_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "qa_abstention": lambda: _qa_abstention_env(*judge_args),
    }
    envs: list[vf.Environment] = []
    names: list[str] = []
    for key in keys:
        env = builders[key]()
        envs.append(guard_env(env, guard_config))
        names.append(f"nemotron-knowledge-{key.replace('_', '-')}")
    if len(envs) == 1:
        return envs[0]
    return vf.EnvGroup(envs=envs, env_names=names)
