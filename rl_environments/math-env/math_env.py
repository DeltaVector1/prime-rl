import ast
import asyncio
import os
import re
import time
from typing import Any, Callable

import httpx
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers.rubrics.experimental.hybrid_math_rubric import HybridMathRubric
from verifiers.utils.data_utils import extract_boxed_answer

DEFAULT_HTTPX_TIMEOUT = 1200
DEFAULT_HTTPX_CONNECTIONS = 8192
DEFAULT_HTTPX_MAX_ALIVE_CONNECTIONS = 8192

DEFAULT_INSTRUCTION_PROMPT = (
    "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n"
)

DEFAULT_INSTRUCTION_PROMPT_POST = ""


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


class StrictMaybeThinkParser(vf.MaybeThinkParser):
    """Parser that returns empty string for unfinished think section. Else, it behaves like MaybeThinkParser."""

    def __init__(self, extract_fn: Callable[[str], str] = lambda x: x):
        super().__init__(extract_fn=extract_fn)

    def parse(self, text: str) -> str:
        if "<think>" in text and "</think>" not in text:
            return ""
        return super().parse(text)


def _is_empty_model_response_error(error: Any) -> bool:
    return isinstance(error, vf.EmptyModelResponseError)


EMPTY_MODEL_RESPONSE_SENTINEL = "[empty model response: no visible answer or tool call]"


def _plain_messages(messages: Any) -> list[dict[str, Any]]:
    if isinstance(messages, str):
        return [{"role": "user", "content": messages}]
    if not isinstance(messages, list):
        return []

    plain: list[dict[str, Any]] = []
    for message in messages:
        if isinstance(message, dict):
            plain.append(dict(message))
            continue
        dump = getattr(message, "model_dump", None)
        if callable(dump):
            plain.append(dump(exclude_none=True))
            continue
        plain.append(
            {
                "role": str(getattr(message, "role", "user")),
                "content": getattr(message, "content", str(message)),
            }
        )
    return plain


def _ensure_empty_model_response_trajectory(state: vf.State, reason: str) -> None:
    if state.get("trajectory"):
        return

    prompt = _plain_messages(state.get("prompt") or [])
    completion = [{"role": "assistant", "content": EMPTY_MODEL_RESPONSE_SENTINEL}]
    state["completion"] = completion
    state.setdefault("raw_completion", EMPTY_MODEL_RESPONSE_SENTINEL)
    state["trajectory"] = [
        {
            "prompt": prompt,
            "completion": completion,
            "response": vf.Response(
                id=f"{state.get('trajectory_id', 'empty')}-empty-response-zero-guard",
                created=int(time.time()),
                model=str(state.get("model") or ""),
                usage=None,
                message=vf.ResponseMessage(
                    content=EMPTY_MODEL_RESPONSE_SENTINEL,
                    finish_reason="stop",
                    is_truncated=False,
                    tokens=None,
                ),
            ),
            "tokens": None,
            "reward": 0.0,
            "advantage": None,
            "is_truncated": False,
            "trajectory_id": str(state.get("trajectory_id", "")),
            "extras": {
                "empty_model_response_zero_guard": True,
                "empty_model_response_reason": reason,
            },
        }
    ]


def _mark_empty_model_response_zero(state: vf.State, error: vf.EmptyModelResponseError) -> None:
    reason = str(error)
    _ensure_empty_model_response_trajectory(state, reason)
    state["error"] = None
    state["reward"] = 0.0
    state["is_completed"] = True
    state["stop_condition"] = "empty_model_response_zero_guard"
    breakdown = {
        "empty_model_response_zero_guard": 1.0,
        "empty_model_response_synthetic_step": 1.0,
        "empty_model_response_reasoning_only": float("reasoning but no content" in reason),
        "empty_model_response_reason": reason,
        "final_reward_formula": "0 because the model returned no visible answer/tool call",
    }
    state.setdefault("reward_breakdown", {})["empty_model_response_zero_guard"] = breakdown
    metrics = dict(state.get("metrics", {}) or {})
    metrics.update({key: value for key, value in breakdown.items() if isinstance(value, int | float)})
    state["metrics"] = metrics


class ZeroOnEmptyModelResponseMixin:
    async def _run_rollout_state(self, input, client, model: str, sampling_args):
        state = await self.rollout(input, client, model, sampling_args)
        state["timing"].scoring.start = time.time()
        if _is_empty_model_response_error(state.get("error")):
            _mark_empty_model_response_zero(state, state["error"])
            state["timing"].scoring.end = time.time()
            await self.rubric.cleanup(state)
            return state
        if self.score_rollouts:
            await self.rubric.score_rollout(state)
        else:
            await self.rubric.dummy_score_rollout(state)
        state["timing"].scoring.end = time.time()
        await self.rubric.cleanup(state)
        return state


class ZeroOnEmptyModelResponseSingleTurnEnv(ZeroOnEmptyModelResponseMixin, vf.SingleTurnEnv):
    pass


class PythonEnvWithLabels(ZeroOnEmptyModelResponseMixin, vf.PythonEnv):
    """PythonEnv that adds sandbox labels."""

    def __init__(self, sandbox_labels: list[str] = ["math-env"], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.sandbox_request = self.sandbox_request.model_copy(update={"labels": sandbox_labels}, deep=True)


def _value_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_value_to_text(item) for item in value]
        return "\n".join(part for part in parts if part).strip()
    if isinstance(value, dict):
        for key in ("text", "content", "reasoning", "reasoning_content", "thinking", "summary"):
            text = _value_to_text(value.get(key))
            if text:
                return text
        return ""
    return str(value).strip()


def _message_get(message: Any, key: str) -> Any:
    if isinstance(message, dict):
        if key in message:
            return message.get(key)
        for container_key in ("additional_kwargs", "metadata", "extra"):
            container = message.get(container_key)
            if isinstance(container, dict) and key in container:
                return container.get(key)
        return None
    value = getattr(message, key, None)
    if value is not None:
        return value
    for container_key in ("additional_kwargs", "metadata", "extra"):
        container = getattr(message, container_key, None)
        if isinstance(container, dict) and key in container:
            return container.get(key)
    return None


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "")).lower()
    return str(getattr(message, "role", "")).lower()


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content") or "")
    return str(getattr(message, "content", "") or "")


def _strip_think_tags(text: str) -> str:
    if not text:
        return text
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if re.search(r"</think>", cleaned, flags=re.IGNORECASE):
        cleaned = re.sub(r"^.*?</think>\s*", "", cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()


def _extract_think_trace(text: str) -> str:
    if not text:
        return ""
    matches = re.findall(r"<think>(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        return "\n".join(matches).strip()
    auto = re.search(r"^(.*?)</think>", text, flags=re.DOTALL | re.IGNORECASE)
    if auto:
        return auto.group(1).strip()
    unclosed = re.search(r"<think>(.*?)$", text, flags=re.DOTALL | re.IGNORECASE)
    if unclosed:
        return unclosed.group(1).strip()
    return ""


def _extract_provider_reasoning_trace(message: Any) -> str:
    for key in ("reasoning", "reasoning_content", "thinking", "thinking_content", "thinking_blocks"):
        text = _value_to_text(_message_get(message, key))
        if text:
            return text
    details = _message_get(message, "reasoning_details")
    text = _value_to_text(details)
    if text and not text.startswith("{"):
        return text
    return ""


def _extract_message_reasoning_trace(message: Any) -> str:
    content_trace = _extract_think_trace(_message_content(message))
    if content_trace:
        return content_trace
    return _extract_provider_reasoning_trace(message)


def _has_unclosed_think(text: str) -> bool:
    if not text:
        return False
    opens = len(re.findall(r"<think>", text, flags=re.IGNORECASE))
    closes = len(re.findall(r"</think>", text, flags=re.IGNORECASE))
    return opens > closes


def _assistant_messages(completion: Any) -> list[Any]:
    if isinstance(completion, str):
        return [{"role": "assistant", "content": completion}]
    if not isinstance(completion, list):
        return []
    return [message for message in completion if _message_role(message) == "assistant"]


def _count_words(text: str) -> int:
    return len([part for part in re.split(r"\s+", (text or "").strip()) if part])


def _rollout_is_truncated(state: vf.State) -> bool:
    if state.get("is_truncated"):
        return True
    for step in state.get("trajectory") or []:
        if isinstance(step, dict):
            if step.get("is_truncated"):
                return True
        elif getattr(step, "is_truncated", False):
            return True
    return False


class MathZeroGuardRubric(vf.Rubric):
    """Hard-zero malformed reasoning/output shapes before math scoring."""

    def __init__(self, base_rubric: vf.Rubric, reasoning_required: bool = True):
        super().__init__(parser=base_rubric.parser)
        self.base_rubric = base_rubric
        self.reasoning_required = reasoning_required

    def _get_reward_func_names(self) -> list[str]:
        return self.base_rubric._get_reward_func_names() + ["math_zero_guard_multiplier"]

    def _get_reward_funcs(self) -> list:
        return self.base_rubric._get_reward_funcs()

    def _get_reward_weights(self) -> list[float]:
        return self.base_rubric._get_reward_weights()

    @property
    def has_group_rewards(self) -> bool:
        return self.base_rubric.has_group_rewards

    @property
    def has_advantages(self) -> bool:
        return self.base_rubric.has_advantages

    def _guard_breakdown(self, completion: Any, *, is_truncated: bool = False) -> dict[str, Any]:
        assistant_messages = _assistant_messages(completion)
        visible_texts = [_strip_think_tags(_message_content(message)) for message in assistant_messages]
        reasoning_traces = [
            trace for message in assistant_messages if (trace := _extract_message_reasoning_trace(message))
        ]
        visible_words = sum(_count_words(text) for text in visible_texts)
        missing_reasoning = self.reasoning_required and any(
            text.strip() and not _extract_message_reasoning_trace(message)
            for message, text in zip(assistant_messages, visible_texts)
        )
        zero_visible_output = visible_words == 0
        unclosed_think = any(_has_unclosed_think(_message_content(message)) for message in assistant_messages)
        multiplier = 0.0 if (is_truncated or unclosed_think or zero_visible_output or missing_reasoning) else 1.0
        return {
            "math_zero_guard_multiplier": multiplier,
            "math_zero_guard_truncated": float(is_truncated),
            "math_zero_guard_unclosed_think": float(unclosed_think),
            "math_zero_guard_zero_visible_output": float(zero_visible_output),
            "math_zero_guard_missing_reasoning": float(missing_reasoning),
            "math_zero_guard_reasoning_traces": float(len(reasoning_traces)),
            "math_zero_guard_visible_words": float(visible_words),
            "final_reward_formula": (
                "0 if truncated or unclosed_think or zero_visible_output or missing_reasoning else base_math_reward"
            ),
        }

    async def score_rollout(self, state: vf.State):
        breakdown = self._guard_breakdown(
            state.get("completion"),
            is_truncated=_rollout_is_truncated(state),
        )
        state.setdefault("reward_breakdown", {})["math_zero_guard"] = breakdown
        numeric_metrics = {key: value for key, value in breakdown.items() if isinstance(value, int | float)}
        if breakdown["math_zero_guard_multiplier"] == 0.0:
            state["reward"] = 0.0
            state["metrics"] = numeric_metrics
            return
        await self.base_rubric.score_rollout(state)
        metrics = dict(state.get("metrics", {}) or {})
        metrics.update(numeric_metrics)
        state["metrics"] = metrics

    async def score_group(self, states: list[vf.State]):
        await asyncio.gather(*(self.score_rollout(state) for state in states))

    async def cleanup(self, state: vf.State):
        await self.base_rubric.cleanup(state)

    async def teardown(self):
        await self.base_rubric.teardown()


def load_environment(
    dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
    dataset_subset: str = "math",
    dataset_split: str = "train",
    dataset_shuffle: bool = False,
    dataset_seed: int = 42,
    question_key: str = "question",
    answer_key: str = "answer",
    info_key: str = "info",
    difficulty_key: str | None = None,
    min_avg_reward: float = 0.0,
    max_avg_reward: float = 1.0,
    judge_model: str | None = None,
    judge_base_url: str | list[str] | None = "https://api.pinference.ai/api/v1",
    judge_sampling_args: dict = {},
    judge_api_key_var: str | None = "PRIME_API_KEY",
    judge_prompt: str = HybridMathRubric.DEFAULT_JUDGE_PROMPT,
    judge_timeout: int = DEFAULT_HTTPX_TIMEOUT,
    judge_connections: int = DEFAULT_HTTPX_CONNECTIONS,
    judge_max_alive_connections: int = DEFAULT_HTTPX_CONNECTIONS,
    system_prompt: str | None = None,
    instruction_prompt: str = DEFAULT_INSTRUCTION_PROMPT,
    instruction_prompt_post: str = DEFAULT_INSTRUCTION_PROMPT_POST,
    math_verify_timeout: int = 5,
    python_tool: bool = False,
    max_turns: int = 100,
    max_startup_wait_seconds: int = 60,
    pip_install_packages: str = "numpy sympy scipy",
    sandbox_cpu_cores: int = 1,
    sandbox_memory_gb: int = 1,
    sandbox_disk_size_gb: int = 1,
    sandbox_gpu_count: int = 0,
    sandbox_timeout_minutes: int = 120,
    sandbox_timeout_per_command_seconds: int = 60,
    sandbox_client_max_workers: int = 10,
    sandbox_labels: list[str] = ["math-env"],
    enable_zero_guardrails: bool = True,
    reasoning_required: bool = True,
    map_kwargs: dict = {},
    filter_kwargs: dict = {},
    **kwargs,
) -> vf.Environment:
    def build_dataset():
        ds = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        if difficulty_key is not None:
            ds = ds.filter(lambda x: min_avg_reward <= x[difficulty_key] <= max_avg_reward, **filter_kwargs)
        ds = ds.map(
            lambda x: {
                "question": instruction_prompt + x[question_key] + instruction_prompt_post,
                "answer": x[answer_key],
                "info": x.get(info_key, {}),
            },
            **map_kwargs,
        ).select_columns(["question", "answer", "info"])
        if dataset_shuffle:
            ds = ds.shuffle(seed=dataset_seed)
        return ds

    judge_client = None
    if judge_model is not None:
        api_key = (os.getenv(judge_api_key_var) if judge_api_key_var else None) or "EMPTY"
        http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(judge_timeout),
            limits=httpx.Limits(
                max_connections=judge_connections, max_keepalive_connections=judge_max_alive_connections
            ),
        )
        judge_client = _openai_client(api_key=api_key, base_url=judge_base_url, http_client=http_client)

    rubric = HybridMathRubric(
        parser=StrictMaybeThinkParser(extract_boxed_answer),
        use_judge_fallback=judge_model is not None,
        judge_model=judge_model or HybridMathRubric.DEFAULT_JUDGE_MODEL,
        judge_client=judge_client,
        judge_sampling_args=judge_sampling_args,
        judge_prompt=judge_prompt,
        timeout_seconds=math_verify_timeout,
    )
    if enable_zero_guardrails:
        rubric = MathZeroGuardRubric(rubric, reasoning_required=reasoning_required)

    if python_tool:
        if system_prompt is None:
            pip_install_prompt = (
                f"In addition to the Python standard library, you have access to: {pip_install_packages}."
                if pip_install_packages.strip()
                else "You may only use the Python standard library."
            )
            system_prompt = "Use Python for all calculations. Give your answer inside \\boxed{}."
            system_prompt += "\n\n" + pip_install_prompt
        env = PythonEnvWithLabels(
            dataset=build_dataset,
            rubric=rubric,
            max_turns=max_turns,
            system_prompt=system_prompt,
            parser=rubric.parser,
            # python env args
            max_startup_wait_seconds=max_startup_wait_seconds,
            pip_install_packages=pip_install_packages,
            # sandbox env args
            cpu_cores=sandbox_cpu_cores,
            memory_gb=sandbox_memory_gb,
            disk_size_gb=sandbox_disk_size_gb,
            gpu_count=sandbox_gpu_count,
            timeout_minutes=sandbox_timeout_minutes,
            timeout_per_command_seconds=sandbox_timeout_per_command_seconds,
            sandbox_client_max_workers=sandbox_client_max_workers,
            sandbox_labels=sandbox_labels,
            **kwargs,
        )
    else:
        env = ZeroOnEmptyModelResponseSingleTurnEnv(
            dataset=build_dataset,
            parser=rubric.parser,
            rubric=rubric,
            system_prompt=system_prompt,
        )
    return env
