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


def _mark_empty_model_response_zero(state: vf.State, error: vf.EmptyModelResponseError) -> None:
    reason = str(error)
    state["error"] = None
    state["reward"] = 0.0
    state["is_completed"] = True
    state["stop_condition"] = "empty_model_response_zero_guard"
    breakdown = {
        "empty_model_response_zero_guard": 1.0,
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
            ds = ds.shuffle(seed=42)
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
