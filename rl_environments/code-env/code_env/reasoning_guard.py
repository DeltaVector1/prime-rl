import re
from typing import Any

import verifiers as vf


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


def reasoning_guard_breakdown(
    completion: Any,
    *,
    reasoning_required: bool = True,
    metric_prefix: str = "reasoning_zero_guard",
    is_truncated: bool = False,
) -> dict[str, Any]:
    assistant_messages = _assistant_messages(completion)
    visible_texts = [_strip_think_tags(_message_content(message)) for message in assistant_messages]
    reasoning_traces = [
        trace for message in assistant_messages if (trace := _extract_message_reasoning_trace(message))
    ]
    visible_chars = sum(len(text.strip()) for text in visible_texts)
    visible_words = sum(_count_words(text) for text in visible_texts)
    zero_visible_output = visible_chars == 0
    missing_reasoning = reasoning_required and visible_chars > 0 and not reasoning_traces
    unclosed_think = any(_has_unclosed_think(_message_content(message)) for message in assistant_messages)
    multiplier = 0.0 if (is_truncated or unclosed_think or zero_visible_output or missing_reasoning) else 1.0
    return {
        f"{metric_prefix}_multiplier": multiplier,
        f"{metric_prefix}_truncated": float(is_truncated),
        f"{metric_prefix}_unclosed_think": float(unclosed_think),
        f"{metric_prefix}_zero_visible_output": float(zero_visible_output),
        f"{metric_prefix}_missing_reasoning": float(missing_reasoning),
        f"{metric_prefix}_reasoning_traces": float(len(reasoning_traces)),
        f"{metric_prefix}_visible_chars": float(visible_chars),
        f"{metric_prefix}_visible_words": float(visible_words),
        "final_reward_formula": (
            "0 if truncated or unclosed_think or zero_visible_output or missing_reasoning else base_reward"
        ),
    }


class ReasoningGuardRubric(vf.Rubric):
    """Hard-zero malformed reasoning/output shapes before task scoring."""

    def __init__(
        self,
        base_rubric: vf.Rubric,
        *,
        reasoning_required: bool = True,
        guard_name: str = "reasoning_zero_guard",
    ):
        super().__init__(parser=base_rubric.parser)
        self.base_rubric = base_rubric
        self.reasoning_required = reasoning_required
        self.guard_name = guard_name
        self.multiplier_key = f"{guard_name}_multiplier"

    def _get_reward_func_names(self) -> list[str]:
        return self.base_rubric._get_reward_func_names() + [self.multiplier_key]

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

    def guard_breakdown(self, completion: Any) -> dict[str, Any]:
        return reasoning_guard_breakdown(
            completion,
            reasoning_required=self.reasoning_required,
            metric_prefix=self.guard_name,
        )

    async def score_rollout(self, state: vf.State):
        breakdown = reasoning_guard_breakdown(
            state.get("completion"),
            reasoning_required=self.reasoning_required,
            metric_prefix=self.guard_name,
            is_truncated=_rollout_is_truncated(state),
        )
        state.setdefault("reward_breakdown", {})[self.guard_name] = breakdown
        numeric_metrics = {key: value for key, value in breakdown.items() if isinstance(value, int | float)}
        if breakdown[self.multiplier_key] == 0.0:
            state["reward"] = 0.0
            state["metrics"] = numeric_metrics
            return

        await self.base_rubric.score_rollout(state)
        metrics = dict(state.get("metrics", {}) or {})
        metrics.update(numeric_metrics)
        state["metrics"] = metrics

    async def score_group(self, states: list[vf.State]):
        for state in states:
            await self.score_rollout(state)

    async def cleanup(self, state: vf.State):
        await self.base_rubric.cleanup(state)

    async def teardown(self):
        await self.base_rubric.teardown()
