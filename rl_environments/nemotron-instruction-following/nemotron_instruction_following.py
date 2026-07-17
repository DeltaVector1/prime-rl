"""Instruction-following category env for the clean Nemotron-RL blend.

Sub-datasets (selectable via the ``dataset`` kwarg):

* ``ifeval`` — nvidia/Nemotron-RL-instruction_following (48 IFEval/IFBench
  constraint checkers; deterministic, no judge)
* ``adversarial`` — nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1
  (judge-graded against per-row criteria + judge prompt template)
* ``structured_v2`` — nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2
* ``calendar`` — nvidia/Nemotron-RL-Instruction-Following-Calendar-v2
  (judge-graded against ``exp_cal_state``)
* ``multichallenge`` — nvidia/Nemotron-RL-Multichallenge-v1
  (judge-graded multi-turn-context final responses)
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import os
import random
import re
import xml.etree.ElementTree as ET
from typing import Any

import jsonschema
import pyarrow.parquet as pq
import verifiers as vf
import yaml
from datasets import Dataset, load_dataset
from huggingface_hub import hf_hub_download
from nemotron_ifeval_checkers import CHECKERS, _evaluate_constraints
from nemotron_instruction_following_guardrails import (
    append_judge_log,
    coerce,
    extract_completion_text,
    extract_tool_defs,
    format_prompt_for_judge,
    guard_env,
    load_jsonl,
    make_disjoint_dataset_builders,
    make_guard_config,
    merge_system_prompt,
    message_content,
    message_role,
    message_tool_calls,
    openai_client,
    parse_bool,
    strip_think_tags,
)

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - py310 fallback
    import tomli as tomllib



def _input_to_prompt(
    payload: Any,
    fallback_prompt: str | None = None,
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
    if fallback_prompt:
        return merge_system_prompt([{"role": "user", "content": str(fallback_prompt)}], system_prompt)
    return []


def _object_get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _normalize_args(args: Any) -> Any:
    args = coerce(args)
    if isinstance(args, str):
        try:
            return json.loads(args)
        except Exception:
            return {"raw": args}
    return args


def _tool_call_signature(tool_call: Any) -> tuple[str, Any]:
    tool_call = coerce(tool_call)
    if hasattr(tool_call, "model_dump"):
        try:
            tool_call = tool_call.model_dump()
        except Exception:
            pass
    if not isinstance(tool_call, dict):
        name = _object_get(tool_call, "name", "")
        args = _object_get(tool_call, "arguments", {})
        return str(name or ""), _normalize_args(args)
    function = tool_call.get("function")
    if isinstance(function, dict):
        name = function.get("name")
        args = function.get("arguments", {})
    else:
        name = tool_call.get("name")
        args = tool_call.get("arguments", {})
    return str(name or ""), _normalize_args(args)


def _completion_tool_calls(completion: Any) -> list[dict[str, Any]]:
    if not isinstance(completion, list):
        return []
    calls: list[dict[str, Any]] = []
    for message in completion:
        if message_role(message) != "assistant":
            continue
        for tool_call in message_tool_calls(message):
            name, arguments = _tool_call_signature(tool_call)
            if name:
                calls.append({"name": name, "arguments": arguments})
    return calls


# === IFEval sub-env =========================================================


def _ifeval_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt) -> vf.Environment:
    async def ifeval_score(completion, info, **_kwargs) -> float:
        text = extract_completion_text(completion)
        ids = (info or {}).get("instruction_ids") or []
        kwargs_list = (info or {}).get("kwargs_list") or []
        return float(_evaluate_constraints(text, ids, kwargs_list)["pass_fraction"])

    async def ifeval_strict(completion, info, **_kwargs) -> float:
        text = extract_completion_text(completion)
        ids = (info or {}).get("instruction_ids") or []
        kwargs_list = (info or {}).get("kwargs_list") or []
        return float(_evaluate_constraints(text, ids, kwargs_list)["fully_passed"])

    def _build(num_examples: int) -> Dataset:
        raw = load_jsonl("nvidia/Nemotron-RL-instruction_following", "instruction_following.jsonl", num_examples, dataset_seed)
        rows: list[dict[str, Any]] = []
        for row in raw:
            prompt = _input_to_prompt(row.get("responses_create_params"), row.get("prompt"), system_prompt)
            ids = coerce(row.get("instruction_id_list")) or []
            kwargs_list = coerce(row.get("kwargs")) or []
            if not prompt or not isinstance(ids, list) or not isinstance(kwargs_list, list):
                continue
            # A constraint with no checker always counts as failed, which makes the
            # row unwinnable: constant-zero reward, no gradient, and a dragged-down
            # mean. Drop rows we cannot actually grade rather than train on them.
            if not ids or any(str(cid) not in CHECKERS for cid in ids):
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "info": {
                        "id": str(row.get("id", "")),
                        "instruction_ids": list(ids),
                        "kwargs_list": list(kwargs_list),
                        "subtask": "ifeval",
                    },
                }
            )
        return Dataset.from_list(rows)

    train_dataset, eval_dataset = make_disjoint_dataset_builders(_build, num_train_examples, num_eval_examples)
    rubric = vf.Rubric(funcs=[ifeval_score])
    rubric.add_metric(ifeval_strict)
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
    )


# === Structured-outputs sub-env =============================================


JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", re.DOTALL)
JSON_OBJECT_RE = re.compile(r"(\{.*\}|\[.*\])", re.DOTALL)


def _extract_json(text: str) -> Any | None:
    if not text:
        return None
    m = JSON_FENCE_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = JSON_OBJECT_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    try:
        return json.loads(text.strip())
    except Exception:
        return None


# === Nemotron Ultra instruction-following additions ==========================


def _schema_from_str(schema_str: str) -> dict[str, Any]:
    try:
        schema = json.loads(schema_str)
    except Exception:
        return {}
    if isinstance(schema, dict) and isinstance(schema.get("schema"), dict):
        return schema["schema"]
    return schema if isinstance(schema, dict) else {}


def _extract_structured_text(text: str) -> str:
    text = (text or "").strip()
    fence = re.search(r"```(?:json|yaml|yml|xml|toml|csv)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return fence.group(1).strip() if fence else text


def _parse_structured_payload(text: str, schema_type: str) -> Any | None:
    content = _extract_structured_text(text)
    kind = (schema_type or "json").strip().lower()
    try:
        if kind == "json":
            return json.loads(content)
        if kind in {"yaml", "yml"}:
            return yaml.safe_load(content)
        if kind == "toml":
            return tomllib.loads(content)
        if kind == "csv":
            return list(csv.DictReader(io.StringIO(content)))
        if kind == "xml":
            ET.fromstring(content)
            return {"_xml_well_formed": True}
    except Exception:
        return None
    return None


def _structured_payload_valid(payload: Any, schema_str: str, schema_type: str) -> float:
    if payload is None:
        return 0.0
    # XML rows carry no XSD, so well-formedness (checked by the parser above) is
    # the only thing we can assert. Every other type is graded against its schema;
    # rows whose schema will not parse are dropped at build time, so an empty
    # schema here cannot hand out a free pass.
    if schema_type == "xml":
        return 1.0
    schema = _schema_from_str(schema_str)
    if not schema:
        return 0.0
    try:
        jsonschema.validate(payload, schema)
        return 1.0
    except Exception:
        return 0.0


def _tool_payload_candidates(completion: Any, info: dict[str, Any]) -> list[Any]:
    calls = _completion_tool_calls(completion)
    expected_tool_name = str(info.get("tool_name") or "").strip()
    payload_key = str(info.get("tool_payload_key") or "").strip()
    if payload_key.lower() in {"", "none", "null"}:
        payload_key = ""

    candidates: list[Any] = []
    for call in calls:
        if expected_tool_name and call.get("name") != expected_tool_name:
            continue
        arguments = _normalize_args(call.get("arguments"))
        if payload_key and isinstance(arguments, dict) and payload_key in arguments:
            candidates.append(arguments[payload_key])
        else:
            candidates.append(arguments)
    return candidates


class NativeToolCallSingleTurnEnv(vf.SingleTurnEnv):
    """Single assistant response that must use row-specific native tools."""

    async def setup_state(self, state: vf.State) -> vf.State:
        await super().setup_state(state)
        info = state.get("info") if isinstance(state.get("info"), dict) else {}
        raw_tool_defs = info.get("tool_defs_json")
        try:
            tool_defs = json.loads(raw_tool_defs or "[]")
        except Exception:
            tool_defs = []
        state["tool_defs"] = self._normalize_tool_defs(tool_defs) or []

        sampling_args = dict(state.get("sampling_args") or {})
        extra_body = dict(sampling_args.get("extra_body") or {})
        tool_choice = str(info.get("tool_choice") or "auto").strip()
        if tool_choice and tool_choice.lower() not in {"none", "null"}:
            extra_body["tool_choice"] = tool_choice
        if "parallel_tool_calls" in info:
            extra_body["parallel_tool_calls"] = parse_bool(info.get("parallel_tool_calls"))
        sampling_args["extra_body"] = extra_body
        state["sampling_args"] = sampling_args
        return state


def _build_structured_v2(config_name: str, num_examples: int, seed: int, system_prompt: str | None) -> Dataset:
    file_by_config = {
        "direct_generation": "direct_generation/train-00000-of-00001.parquet",
        "diversified_tasks": "diversified_tasks/train-00000-of-00001.parquet",
        "tool_calling_extraction": "tool_calling_extraction/train-00000-of-00001.parquet",
    }
    filename = file_by_config[config_name]
    path = hf_hub_download("nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2", filename, repo_type="dataset")
    table = pq.read_table(path)
    rows = table.to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    if num_examples > 0:
        rows = rows[:num_examples]
    out: list[dict[str, Any]] = []
    for row in rows:
        responses_create_params = row.get("responses_create_params")
        prompt = _input_to_prompt(responses_create_params, system_prompt=system_prompt)
        schema_str = str(row.get("schema_str", "")).strip()
        schema_type = str(row.get("schema_type", "json")).strip().lower()
        if not prompt:
            continue
        # Without a parseable schema there is nothing to grade against, and the
        # row would score 1.0 for any syntactically valid payload. XML is exempt:
        # those rows are graded on well-formedness alone by design.
        if schema_type != "xml" and not _schema_from_str(schema_str):
            continue
        tool_defs = extract_tool_defs(responses_create_params)
        out.append({
            "prompt": prompt,
            "info": {
                "schema_str": schema_str,
                "schema_type": schema_type,
                "problem_type": str(row.get("problem_type", "")),
                "response_mode": str(row.get("response_mode", "")),
                "source_record_id": str(row.get("source_record_id", "")),
                "tool_choice": str(row.get("tool_choice", "")),
                "tool_name": str(row.get("tool_name", "")),
                "tool_payload_key": str(row.get("tool_payload_key", "")),
                "parallel_tool_calls": bool(row.get("parallel_tool_calls", False)),
                "tool_defs_json": json.dumps(tool_defs, ensure_ascii=False, default=str),
                "subtask": f"structured_v2_{config_name}",
            },
        })
    return Dataset.from_list(out)


def _structured_v2_env(config_name, num_train_examples, num_eval_examples, dataset_seed, system_prompt) -> vf.Environment:
    async def structured_v2_valid(completion, info, **_kwargs) -> float:
        info = info or {}
        schema_type = (info or {}).get("schema_type", "json")
        if str(info.get("response_mode") or "").lower() == "tool_call":
            candidates = _tool_payload_candidates(completion, info)
            return max(
                [_structured_payload_valid(candidate, info.get("schema_str", ""), schema_type) for candidate in candidates]
                or [0.0]
            )
        text = extract_completion_text(completion)
        payload = _parse_structured_payload(text, schema_type)
        return _structured_payload_valid(payload, info.get("schema_str", ""), schema_type)

    async def structured_v2_tool_call_emitted(completion, **_kwargs) -> float:
        return 1.0 if _completion_tool_calls(completion) else 0.0

    async def structured_v2_expected_tool_called(completion, info, **_kwargs) -> float:
        expected_tool_name = str((info or {}).get("tool_name") or "").strip()
        if not expected_tool_name:
            return 0.0
        return 1.0 if any(call.get("name") == expected_tool_name for call in _completion_tool_calls(completion)) else 0.0

    rubric = vf.Rubric(funcs=[structured_v2_valid])
    if config_name == "tool_calling_extraction":
        rubric.add_metric(structured_v2_tool_call_emitted)
        rubric.add_metric(structured_v2_expected_tool_called)
        env_cls: type[vf.SingleTurnEnv] = NativeToolCallSingleTurnEnv
    else:
        env_cls = vf.SingleTurnEnv

    train_dataset, eval_dataset = make_disjoint_dataset_builders(
        lambda num_examples: _build_structured_v2(config_name, num_examples, dataset_seed, system_prompt),
        num_train_examples,
        num_eval_examples,
    )
    return env_cls(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=rubric,
        system_prompt=system_prompt,
    )


def _format_verification_score(text: str, verifier: dict[str, Any]) -> float:
    verifier_type = str(verifier.get("type", ""))
    if verifier_type in {"regex", "inline_prose"}:
        patterns = verifier.get("verify_regex") or []
        min_matches = int(verifier.get("verify_min_matches", 1) or 1)
        compiled = [re.compile(str(pattern)) for pattern in patterns]
        matching_lines = 0
        for line in text.splitlines():
            if any(rx.search(line) for rx in compiled):
                matching_lines += 1
        return 1.0 if matching_lines >= min_matches else 0.0
    if verifier_type == "string_match":
        expected = [str(x) for x in (verifier.get("expected_markers") or [])]
        patterns = [str(x) for x in (verifier.get("patterns") or [])]
        missing = [marker for marker in expected if marker not in text]
        spurious: list[str] = []
        expected_set = set(expected)
        for pattern in patterns:
            for found in re.finditer(pattern, text):
                if found.group(0) not in expected_set:
                    spurious.append(found.group(0))
        return 1.0 if not missing and not spurious else 0.0
    return 0.0


def _build_format_verification(
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
        prompt = _input_to_prompt(row.get("responses_create_params"), system_prompt=system_prompt)
        verifier = coerce(row.get("verifier"))
        if not prompt or not isinstance(verifier, dict):
            continue
        rows.append({
            "prompt": prompt,
            "info": {
                "verifier_json": json.dumps(verifier, default=str),
                "subtask": subtask,
            },
        })
    return Dataset.from_list(rows)


def _format_verification_env(
    repo, filename, subtask, num_train_examples, num_eval_examples, dataset_seed, system_prompt
) -> vf.Environment:
    async def format_score(completion, info, **_kwargs) -> float:
        text = extract_completion_text(completion)
        try:
            verifier = json.loads((info or {}).get("verifier_json", "{}"))
        except Exception:
            verifier = {}
        return _format_verification_score(text, verifier)

    train_dataset, eval_dataset = make_disjoint_dataset_builders(
        lambda num_examples: _build_format_verification(repo, filename, subtask, num_examples, dataset_seed, system_prompt),
        num_train_examples,
        num_eval_examples,
    )
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=vf.Rubric(funcs=[format_score]),
        system_prompt=system_prompt,
    )


# === Judge-based sub-envs ==================================================


YES_NO_RE = re.compile(r"\b(YES|NO)\b", re.IGNORECASE)
TAGGED_SCORE_RE = re.compile(r"<score>\s*([0-3])\s*</score>|<score>\s*([0-3])", re.IGNORECASE)
SCORE_RE = re.compile(r"^\s*([0-3])\s*[\.\!]*\s*$")

CRITERION_PROMPT = """You evaluate an assistant response against a single yes/no criterion.

Question to ask about the response: {question}

Conversation context (recent turns):
{context}

Assistant response (final turn under evaluation):
{response}

Answer with just YES or NO.
"""

CALENDAR_PROMPT = """You evaluate whether an assistant correctly scheduled a set of calendar events.

Expected calendar state:
{exp_cal_state}

Assistant response:
{response}

Answer with a single integer score 0-3 only.
"""

ADVERSARIAL_DEFAULT_TEMPLATE = """## Question
{prompt}

---
## Student Response
{model_response}

---
## Standard Response
{standard_response}

---
## Evaluation Criteria
{criteria}

---

Answer with just YES or NO.
"""

ULTRA_CRITERION_PROMPT = """You evaluate whether an assistant response satisfies a criterion.

Conversation / task prompt:
{prompt}

Reference response, if any:
{reference}

Assistant response:
{response}

Criterion:
{criterion}

Answer with just YES or NO.
"""


def _last_user_content(prompt) -> str:
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if message_role(msg) == "user":
                return message_content(msg)
    return str(prompt)


def _parse_yes_no(verdict: str) -> str | None:
    """Take the last standalone YES/NO, ignoring any hidden reasoning.

    Judges are not always terse: a reasoning model answers after a <think> block,
    and requiring the whole reply to *be* the word YES silently scored every such
    verdict 0.
    """
    text = strip_think_tags(verdict or "").strip()
    matches = YES_NO_RE.findall(text)
    return matches[-1].upper() if matches else None


def _parse_0_3_score(verdict: str) -> int | None:
    text = strip_think_tags(verdict or "").strip()
    tagged_values = [
        value for match in TAGGED_SCORE_RE.finditer(text) if (value := match.group(1) or match.group(2)) is not None
    ]
    if tagged_values:
        return int(tagged_values[-1]) if len(set(tagged_values)) == 1 else None
    for line in reversed(text.splitlines()):
        match = SCORE_RE.fullmatch(line)
        if match:
            return int(match.group(1))
    return None


async def _judge_yes_no(
    judge_client,
    judge_model,
    sampling: dict[str, Any],
    judge_input: str,
    state: vf.State | None = None,
    kind: str = "nemotron_instruction_following.yes_no",
) -> float:
    try:
        result = await judge_client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": judge_input}],
            **sampling,
        )
        verdict = (result.choices[0].message.content or "").strip()
    except Exception as exc:
        append_judge_log(state, kind=kind, model=judge_model, prompt=judge_input, response=None, error=exc)
        return 0.0
    parsed = _parse_yes_no(verdict)
    parsed_score = 1.0 if parsed == "YES" else 0.0
    append_judge_log(
        state,
        kind=kind,
        model=judge_model,
        prompt=judge_input,
        response=verdict,
        parsed_score=parsed_score,
    )
    if parsed == "YES":
        return 1.0
    if parsed == "NO":
        return 0.0
    return 0.0


def _deterministic_instruction_score(text: str, instructions: list[dict[str, Any]]) -> float | None:
    ids: list[str] = []
    kwargs_list: list[dict[str, Any]] = []
    for item in instructions:
        if not isinstance(item, dict):
            continue
        instruction_id = str(item.get("instruction_id", "")).strip()
        if instruction_id not in CHECKERS:
            continue
        ids.append(instruction_id)
        kwargs_list.append({k: v for k, v in item.items() if k not in {"instruction_id", "source", "uid"}})
    if not ids:
        return None
    return float(_evaluate_constraints(text, ids, kwargs_list)["pass_fraction"])


def _build_ultra_judge_rows(
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
        prompt = _input_to_prompt(row.get("responses_create_params"), system_prompt=system_prompt)
        if not prompt:
            prompt = _input_to_prompt(row.get("messages"), system_prompt=system_prompt)
        if not prompt:
            continue
        judge_items: list[dict[str, Any]] = []
        instructions = coerce(row.get("instructions")) or []
        llm_judge = coerce(row.get("llm_judge")) or []
        rubric = coerce(row.get("rubric")) or []
        if isinstance(llm_judge, list):
            judge_items.extend([x for x in llm_judge if isinstance(x, dict)])
        if isinstance(rubric, list):
            judge_items.extend([x for x in rubric if isinstance(x, dict)])
        principle = str(row.get("principle", "")).strip()
        if principle:
            judge_items.append({"content": principle})
        rows.append({
            "prompt": prompt,
            "info": {
                "id": str(row.get("id") or row.get("uuid") or ""),
                "instructions_json": json.dumps(instructions, default=str),
                "judge_items_json": json.dumps(judge_items, default=str),
                "reference_response": str(row.get("reference_response", "")),
                "judge_prompt_template": str(row.get("judge_prompt_template", "")),
                "subtask": subtask,
            },
        })
    return Dataset.from_list(rows)


def _ultra_judge_env(
    repo: str,
    filename: str,
    subtask: str,
    judge_client,
    judge_model,
    judge_sampling_args,
    num_train_examples,
    num_eval_examples,
    dataset_seed,
    system_prompt,
) -> vf.Environment:
    async def ultra_score(prompt, completion, info, state=None, **_kwargs) -> float:
        response = extract_completion_text(completion)
        try:
            instructions = json.loads((info or {}).get("instructions_json", "[]"))
        except Exception:
            instructions = []
        try:
            judge_items = json.loads((info or {}).get("judge_items_json", "[]"))
        except Exception:
            judge_items = []
        scores: list[float] = []
        det_score = _deterministic_instruction_score(response, instructions if isinstance(instructions, list) else [])
        if det_score is not None:
            scores.append(det_score)

        sampling = {k: v for k, v in (judge_sampling_args or {}).items() if v is not None}
        sampling.setdefault("temperature", 0.0)
        # Enough room for a reasoning judge to think before answering. An 8-token
        # cap truncates the reply before the YES/NO ever lands, and every verdict
        # then parses as 0.
        sampling.setdefault("max_tokens", JUDGE_MAX_TOKENS)
        prompt_text = format_prompt_for_judge(prompt)
        reference = (info or {}).get("reference_response", "")
        template = (info or {}).get("judge_prompt_template", "")

        async def _check(item: dict[str, Any]) -> float:
            criterion = str(item.get("content") or item.get("criteria") or item.get("question") or "").strip()
            if not criterion:
                return 0.0
            if template:
                try:
                    judge_input = template.format(
                        prompt=prompt_text,
                        model_response=response,
                        standard_response=reference,
                        criteria=criterion,
                    )
                except Exception:
                    judge_input = ULTRA_CRITERION_PROMPT.format(
                        prompt=prompt_text, reference=reference, response=response, criterion=criterion
                    )
            else:
                judge_input = ULTRA_CRITERION_PROMPT.format(
                    prompt=prompt_text, reference=reference, response=response, criterion=criterion
                )
            return await _judge_yes_no(
                judge_client,
                judge_model,
                sampling,
                judge_input,
                state=state,
                kind=f"nemotron_instruction_following.{subtask}.criterion",
            )

        if judge_items:
            scores.extend(await asyncio.gather(*[_check(item) for item in judge_items if isinstance(item, dict)]))
        return float(sum(scores) / len(scores)) if scores else 0.0

    train_dataset, eval_dataset = make_disjoint_dataset_builders(
        lambda num_examples: _build_ultra_judge_rows(repo, filename, subtask, num_examples, dataset_seed, system_prompt),
        num_train_examples,
        num_eval_examples,
    )
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=vf.Rubric(funcs=[ultra_score]),
        system_prompt=system_prompt,
    )


def _calendar_env(
    judge_client, judge_model, judge_sampling_args,
    num_train_examples, num_eval_examples, dataset_seed, system_prompt,
) -> vf.Environment:
    async def calendar_judge(completion, info, state=None, **_kwargs) -> float:
        response = extract_completion_text(completion)
        exp_state = (info or {}).get("exp_cal_state", "")
        judge_input = CALENDAR_PROMPT.format(exp_cal_state=exp_state, response=response)
        sampling = {k: v for k, v in (judge_sampling_args or {}).items() if v is not None}
        sampling.setdefault("temperature", 0.0)
        try:
            result = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_input}],
                **sampling,
            )
            verdict = (result.choices[0].message.content or "").strip()
        except Exception as exc:
            append_judge_log(
                state,
                kind="nemotron_instruction_following.calendar",
                model=judge_model,
                prompt=judge_input,
                response=None,
                error=exc,
            )
            return 0.0
        parsed = _parse_0_3_score(verdict)
        score = float(parsed / 3.0) if parsed is not None else 0.0
        append_judge_log(
            state,
            kind="nemotron_instruction_following.calendar",
            model=judge_model,
            prompt=judge_input,
            response=verdict,
            parsed_score=score,
        )
        return score

    async def calendar_is_json(completion, **_kwargs) -> float:
        text = extract_completion_text(completion)
        return 1.0 if _extract_json(text) is not None else 0.0

    def _build_split(filename: str, num: int, seed: int) -> Dataset:
        raw = load_jsonl("nvidia/Nemotron-RL-Instruction-Following-Calendar-v2", filename, num, seed)
        rows: list[dict[str, Any]] = []
        for row in raw:
            prompt = _input_to_prompt(row.get("responses_create_params"), system_prompt=system_prompt)
            exp_state = row.get("exp_cal_state")
            if not prompt or exp_state is None:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "info": {
                        "exp_cal_state": json.dumps(exp_state, default=str),
                        "subtask": "calendar",
                    },
                }
            )
        return Dataset.from_list(rows)

    # A plain Rubric: calendar_judge calls the judge itself, so JudgeRubric's own
    # judge machinery was only ever dead configuration.
    rubric = vf.Rubric(funcs=[calendar_judge])
    rubric.add_metric(calendar_is_json)
    # Calendar ships train/validation files, so no disjoint split is needed.
    return vf.SingleTurnEnv(
        dataset=lambda: _build_split("train.jsonl", num_train_examples, dataset_seed),
        eval_dataset=lambda: _build_split("validation.jsonl", num_eval_examples, dataset_seed + 1),
        rubric=rubric,
        system_prompt=system_prompt,
    )


def _adversarial_env(
    judge_client, judge_model, judge_sampling_args,
    num_train_examples, num_eval_examples, dataset_seed, system_prompt,
) -> vf.Environment:
    def _format_with(template: str, fallback: str, **fields: Any) -> str:
        base = template.strip() or fallback
        try:
            return base.format(**fields)
        except (KeyError, IndexError):
            return fallback.format(**fields)

    async def adversarial_score(prompt, completion, info, state=None, **_kwargs) -> float:
        response = extract_completion_text(completion)
        user_prompt = (info or {}).get("user_prompt", "")
        reference = (info or {}).get("reference_response", "")
        template = (info or {}).get("judge_prompt_template", "")
        try:
            criteria = json.loads((info or {}).get("criteria_json", "[]"))
        except Exception:
            criteria = []
        if not criteria:
            return 0.0
        sampling = {k: v for k, v in (judge_sampling_args or {}).items() if v is not None}
        sampling.setdefault("temperature", 0.0)

        async def _check(item: dict) -> float:
            criterion = str(item.get("criteria") or item.get("question") or "").strip()
            if not criterion:
                return 0.0
            judge_input = _format_with(
                template, ADVERSARIAL_DEFAULT_TEMPLATE,
                prompt=user_prompt, model_response=response,
                standard_response=reference, criteria=criterion,
            )
            if "YES" not in judge_input.upper()[-200:]:
                judge_input = judge_input + "\n\nAnswer with just YES or NO."
            try:
                result = await judge_client.chat.completions.create(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_input}],
                    **sampling,
                )
                verdict = (result.choices[0].message.content or "").strip()
            except Exception as exc:
                append_judge_log(
                    state,
                    kind="nemotron_instruction_following.adversarial",
                    model=judge_model,
                    prompt=judge_input,
                    response=None,
                    error=exc,
                )
                return 0.0
            score = 1.0 if _parse_yes_no(verdict) == "YES" else 0.0
            append_judge_log(
                state,
                kind="nemotron_instruction_following.adversarial",
                model=judge_model,
                prompt=judge_input,
                response=verdict,
                parsed_score=score,
            )
            return score

        scores = await asyncio.gather(*[_check(c) for c in criteria if isinstance(c, dict)])
        return float(sum(scores) / len(scores)) if scores else 0.0

    def _build(num_examples: int) -> Dataset:
        raw = load_dataset("nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1", split="train").shuffle(seed=dataset_seed)
        if num_examples > 0:
            raw = raw.select(range(min(num_examples, len(raw))))
        rows: list[dict[str, Any]] = []
        for row in raw:
            prompt = _input_to_prompt(row.get("responses_create_params"), row.get("prompt"), system_prompt)
            rubric_items = coerce(row.get("rubric")) or []
            if not prompt or not isinstance(rubric_items, list) or not rubric_items:
                continue
            rows.append(
                {
                    "prompt": prompt,
                    "info": {
                        "uuid": str(row.get("uuid", "")),
                        "user_prompt": str(row.get("prompt", "")),
                        "reference_response": str(row.get("reference_response", "")),
                        "judge_prompt_template": str(row.get("judge_prompt_template", "")),
                        "criteria_json": json.dumps(rubric_items, default=str),
                        "subtask": "adversarial",
                    },
                }
            )
        return Dataset.from_list(rows)

    train_dataset, eval_dataset = make_disjoint_dataset_builders(_build, num_train_examples, num_eval_examples)
    # adversarial_score calls the judge itself, so a plain Rubric is enough.
    return vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        rubric=vf.Rubric(funcs=[adversarial_score]),
        system_prompt=system_prompt,
    )


# === Composer ===============================================================


_ALIASES = {
    "adversarial-if": "adversarial",
    "structured": "structured_v2",
    "structured-outputs": "structured_v2",
    "structured-outputs-v2": "structured_v2",
    "structured-v2-direct": "structured_v2_direct",
    "structured-v2-diversified": "structured_v2_diversified",
    "structured-v2-tool-calling": "structured_v2_tool_calling",
    "citation-formatting": "citation_format",
    "freeform-formatting": "freeform_formatting",
    "identity-following": "identity",
    "multiturn": "multichallenge",
    "multiturn-chat": "multichallenge",
    "multi-challenge": "multichallenge",
    "inverse-if": "inverse_ifeval",
    "inverse-ifeval": "inverse_ifeval",
    "sys-bench": "sysbench",
    "cf-bench": "cfbench",
    "instruction-following": "ifeval",
}

_NEEDS_JUDGE = {"adversarial", "calendar", "identity", "sysbench", "cfbench", "multichallenge", "inverse_ifeval"}


def _resolve_datasets(dataset: str) -> list[str]:
    keys = {
        "ifeval", "adversarial", "structured_v2", "structured_v2_direct",
        "structured_v2_diversified", "structured_v2_tool_calling",
        "citation_format", "freeform_formatting",
        "calendar", "identity", "sysbench", "cfbench",
        "multichallenge", "inverse_ifeval",
    }
    if not dataset or dataset == "all":
        return [
            "ifeval", "structured_v2_direct", "structured_v2_diversified",
            "citation_format", "freeform_formatting", "calendar",
            "multichallenge", "adversarial",
            "identity", "sysbench", "cfbench", "inverse_ifeval",
        ]
    out: list[str] = []
    for d in dataset.split(","):
        key = _ALIASES.get(d.strip(), d.strip())
        if key in keys:
            out.append(key)
    return out or ["ifeval", "structured_v2_direct", "structured_v2_diversified", "calendar", "multichallenge", "adversarial"]


def load_environment(
    dataset: str = "all",
    num_train_examples: int = -1,
    num_eval_examples: int = 256,
    dataset_seed: int = 42,
    judge_model: str = "google/gemma-4-26B-A4B-it",
    judge_base_url: str | None = "http://127.0.0.1:8000/v1",
    judge_api_key_var: str = "VLLM_API_KEY",
    judge_sampling_args: dict | None = None,
    system_prompt: str | None = None,
    enable_anti_hacking: bool = True,
    enable_anti_hacking_judges: bool = True,
    anti_hacking_reasoning_required: bool = False,
    anti_hacking_allow_renderer_stripped_tool_calls: bool = False,
    anti_hacking_format_reward_weight: float = 0.1,
    **kwargs: Any,
) -> vf.Environment:
    """Instruction-following env composing the NVIDIA Nemotron-RL sources.

    ``dataset`` is ``all`` (default), one of the supported clean-blend
    selectors, or a comma-separated subset. Judge env vars are required only when
    the selection includes a judge-graded sub-env such as adversarial, calendar,
    or multichallenge.
    """
    keys = _resolve_datasets(dataset)
    needs_judge = any(k in _NEEDS_JUDGE for k in keys) or (
        parse_bool(enable_anti_hacking) and parse_bool(enable_anti_hacking_judges)
    )
    judge_client = openai_client(os.environ.get(judge_api_key_var, "dummy-key"), judge_base_url) if needs_judge else None
    guard_config = make_guard_config(
        enable_anti_hacking, enable_anti_hacking_judges, judge_client, judge_model, judge_sampling_args,
        anti_hacking_reasoning_required, anti_hacking_allow_renderer_stripped_tool_calls, anti_hacking_format_reward_weight,
    )

    judge_args = (judge_client, judge_model, judge_sampling_args, num_train_examples, num_eval_examples, dataset_seed, system_prompt)
    builders = {
        "ifeval": lambda: _ifeval_env(num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "structured_v2": lambda: _structured_v2_env("diversified_tasks", num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "structured_v2_direct": lambda: _structured_v2_env("direct_generation", num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "structured_v2_diversified": lambda: _structured_v2_env("diversified_tasks", num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "structured_v2_tool_calling": lambda: _structured_v2_env("tool_calling_extraction", num_train_examples, num_eval_examples, dataset_seed, system_prompt),
        "citation_format": lambda: _format_verification_env(
            "nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1", "ds3_citation_train.jsonl", "citation_format",
            num_train_examples, num_eval_examples, dataset_seed, system_prompt,
        ),
        "freeform_formatting": lambda: _format_verification_env(
            "nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1", "ds2_freeform_train.jsonl", "freeform_formatting",
            num_train_examples, num_eval_examples, dataset_seed, system_prompt,
        ),
        "calendar": lambda: _calendar_env(*judge_args),
        "adversarial": lambda: _adversarial_env(*judge_args),
        "identity": lambda: _ultra_judge_env(
            "nvidia/Nemotron-RL-Identity-Following-v1", "train.jsonl", "identity", *judge_args,
        ),
        "sysbench": lambda: _ultra_judge_env(
            "nvidia/Nemotron-RL-SysBench-v1", "data/train.jsonl", "sysbench", *judge_args,
        ),
        "cfbench": lambda: _ultra_judge_env(
            "nvidia/Nemotron-RL-CFBench-v1", "data/train.jsonl", "cfbench", *judge_args,
        ),
        "multichallenge": lambda: _ultra_judge_env(
            "nvidia/Nemotron-RL-Multichallenge-v1", "data/advanced.jsonl", "multichallenge", *judge_args,
        ),
        "inverse_ifeval": lambda: _ultra_judge_env(
            "nvidia/Nemotron-RL-InverseIFEval-v1", "data/train.jsonl", "inverse_ifeval", *judge_args,
        ),
    }
    envs = [guard_env(builders[key](), guard_config) for key in keys]
    names = [f"nemotron-instruction-following-{key}" for key in keys]
    return envs[0] if len(envs) == 1 else vf.EnvGroup(envs=envs, env_names=names)
