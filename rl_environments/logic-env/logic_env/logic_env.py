import json
import time
from typing import Any, Callable

import verifiers as vf
from datasets import load_dataset

from .base.data import Data
from .reasoning_guard import ReasoningGuardRubric
from .task2verifier import verifier_classes


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


def load_environment(
    dataset_name: str = "PrimeIntellect/INTELLECT-3-RL",
    dataset_subset: str = "logic",
    dataset_split: str = "train",
    dataset_shuffle: bool = False,
    dataset_seed: int = 42,
    difficulty_key: str = "avg@16_qwen3_4b_instruct_2507",
    min_avg_reward: float = 0.0,
    max_avg_reward: float = 1.0,
    tasks_to_skip: list[str] = ["arc_agi", "arc_agi_2", "buggy_tables"],
    enable_zero_guardrails: bool = True,
    reasoning_required: bool = True,
    **kwargs,
) -> vf.Environment:
    def build_dataset():
        ds = (
            load_dataset(dataset_name, dataset_subset, split=dataset_split)
            .map(lambda x: {"info": json.loads(x["info"]), "answer": ""})
            .filter(lambda x: x["info"]["task_name"] not in tasks_to_skip)
            .filter(lambda x: min_avg_reward <= x.get(difficulty_key, 0) <= max_avg_reward)
            .select_columns(["question", "answer", "info"])
        )
        if dataset_shuffle:
            ds = ds.shuffle(seed=dataset_seed)
        return ds

    def correct_answer(completion: vf.Messages, info: vf.Info, **kwargs) -> float:
        game_data = info["game_data_str"] or info["game_data"]
        task = info["task_name"]
        verifier_cls = verifier_classes.get(task)
        if verifier_cls is None:
            raise ValueError(f"Verifier class not found for task: {task}")
        verifier = verifier_cls()
        data_obj = Data.from_json_str(game_data)
        parsed_answer = parser.parse_answer(completion)
        return float(verifier.verify(data_obj, parsed_answer))

    parser = StrictMaybeThinkParser()
    rubric = vf.Rubric(parser=parser, funcs=[correct_answer], weights=[1.0])
    if enable_zero_guardrails:
        rubric = ReasoningGuardRubric(rubric, reasoning_required=reasoning_required)
    return ZeroOnEmptyModelResponseSingleTurnEnv(dataset=build_dataset, parser=parser, rubric=rubric)
