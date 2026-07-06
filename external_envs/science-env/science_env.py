import ast
import os
from typing import Any, Callable

import httpx
import verifiers as vf
from datasets import load_dataset
from openai import AsyncOpenAI
from verifiers.rubrics.experimental.hybrid_math_rubric import HybridMathRubric
from verifiers.utils.data_utils import extract_boxed_answer

# We set higher timeouts than default to avoid judge timeout during eval
DEFAULT_HTTPX_TIMEOUT = 1200
DEFAULT_HTTPX_CONNECTIONS = 8192
DEFAULT_HTTPX_MAX_ALIVE_CONNECTIONS = 8192

DEFAULT_INSTRUCTION_PROMPT = (
    "Solve the following problem. Make sure to put the answer (and only answer) inside \\boxed{}.\n\n"
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


def load_environment(
    dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
    dataset_subset: str = "science",
    dataset_split: str = "train",
    dataset_shuffle: bool = False,
    dataset_seed: int = 42,
    difficulty_key: str | None = "avg@16_qwen3_4b_instruct_2507",
    min_avg_reward: float = 0.0,
    max_avg_reward: float = 1.0,
    judge_model: str | None = None,
    judge_base_url: str | list[str] | None = "https://api.pinference.ai/api/v1",
    judge_sampling_args: dict = {},
    judge_api_key_var: str | None = "PRIME_API_KEY",
    judge_prompt: str = HybridMathRubric.DEFAULT_JUDGE_PROMPT,
    judge_timeout: float = DEFAULT_HTTPX_TIMEOUT,
    judge_connections: int = DEFAULT_HTTPX_CONNECTIONS,
    judge_max_alive_connections: int = DEFAULT_HTTPX_MAX_ALIVE_CONNECTIONS,
    instruction_prompt: str = DEFAULT_INSTRUCTION_PROMPT,
    instruction_prompt_post: str = DEFAULT_INSTRUCTION_PROMPT_POST,
    math_verify_timeout: int = 5,
    map_kwargs: dict = {},
    filter_kwargs: dict = {},
    **kwargs,
) -> vf.Environment:
    def build_dataset():
        ds = load_dataset(dataset_name, dataset_subset, split=dataset_split)
        if difficulty_key is not None:
            ds = ds.filter(lambda x: min_avg_reward <= x[difficulty_key] <= max_avg_reward, **filter_kwargs)
        ds = ds.map(
            lambda x: {"question": instruction_prompt + x["question"] + instruction_prompt_post}, **map_kwargs
        ).select_columns(["question", "answer"])
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
    return vf.SingleTurnEnv(dataset=build_dataset, rubric=rubric)
