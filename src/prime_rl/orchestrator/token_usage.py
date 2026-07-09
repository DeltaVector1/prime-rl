from __future__ import annotations

from typing import Any

import verifiers as vf

from prime_rl.transport import TrainingSample

TOKEN_USAGE_KEYS = ("input_tokens", "output_tokens", "final_input_tokens", "final_output_tokens")


def _as_number(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _lengths_from_tokens(tokens: dict[str, Any]) -> tuple[float, float]:
    prompt_len = len(tokens.get("prompt_ids") or [])
    completion_mask = tokens.get("completion_mask")
    completion_ids = tokens.get("completion_ids") or []
    if completion_mask is None:
        output_len = len(completion_ids)
        input_len = prompt_len
    else:
        output_len = sum(bool(x) for x in completion_mask)
        input_len = prompt_len + len(completion_mask) - output_len
    return float(input_len), float(output_len)


def _derive_from_samples(samples: list[TrainingSample] | None) -> tuple[float, float] | None:
    if not samples:
        return None
    input_tokens = 0.0
    output_tokens = 0.0
    for sample in samples:
        output_len = sum(sample.completion_mask)
        input_len = len(sample.prompt_ids) + len(sample.completion_mask) - output_len
        input_tokens += input_len
        output_tokens += output_len
    return input_tokens, output_tokens


def _derive_from_trajectory(raw: vf.RolloutOutput) -> tuple[float, float] | None:
    for step in reversed(raw.get("trajectory") or []):
        tokens = step.get("tokens") if isinstance(step, dict) else None
        if isinstance(tokens, dict):
            return _lengths_from_tokens(tokens)
    return None


def normalize_token_usage(
    raw: vf.RolloutOutput,
    *,
    samples: list[TrainingSample] | None = None,
) -> dict[str, float]:
    """Ensure a rollout has stable token accounting keys.

    Some env-side synthetic zero-reward paths can return a valid rollout
    without verifiers' ``token_usage`` block. Prefer lengths derived from the
    trainer samples, then from trajectory tokens, and use zeros only when no
    tokenized content exists.
    """

    existing = raw.get("token_usage")
    usage: dict[str, Any] = dict(existing) if isinstance(existing, dict) else {}
    derived = _derive_from_samples(samples)
    if derived is None:
        derived = _derive_from_trajectory(raw)
    if derived is None:
        derived = (0.0, 0.0)

    derived_input, derived_output = derived
    final_input = _as_number(
        usage.get("final_input_tokens"),
        _as_number(usage.get("input_tokens"), derived_input),
    )
    final_output = _as_number(
        usage.get("final_output_tokens"),
        _as_number(usage.get("output_tokens"), derived_output),
    )
    input_tokens = _as_number(usage.get("input_tokens"), final_input)
    output_tokens = _as_number(usage.get("output_tokens"), final_output)

    normalized = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_input_tokens": final_input,
        "final_output_tokens": final_output,
    }
    raw["token_usage"] = normalized
    return normalized


def rollout_token_count(raw: vf.RolloutOutput) -> float:
    usage = normalize_token_usage(raw)
    return usage["final_input_tokens"] + usage["final_output_tokens"]


def rollout_final_output_tokens(raw: vf.RolloutOutput) -> float:
    return normalize_token_usage(raw)["final_output_tokens"]
