import uuid

import pytest

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.orchestrator.metrics import MetricsBuilder
from prime_rl.orchestrator.types import Progress, TrainBatchMetrics, TrainRollout


def _rollout(
    *,
    env_name: str,
    group_id: uuid.UUID,
    reward: float,
    custom_score: float,
    filter_results: dict[str, bool] | None = None,
    stop_condition: str = "stop",
) -> TrainRollout:
    return TrainRollout(
        raw={
            "reward": reward,
            "metrics": {"custom_score": custom_score},
            "token_usage": {"final_input_tokens": 3, "final_output_tokens": 2},
            "trajectory": [{}, {}],
            "timing": {
                "total": 10.0,
                "setup": {"duration": 1.0},
                "generation": {"duration": 2.0},
                "model": {"duration": 3.0},
                "env": {"duration": 4.0},
                "scoring": {"duration": 5.0},
                "overhead": 6.0,
            },
            "stop_condition": stop_condition,
        },
        env_name=env_name,
        example_id=0,
        group_id=group_id,
        policy_version=0,
        off_policy_steps=0,
        filter_results=filter_results or {"repetition": False},
    )


def test_metrics_builder_logs_concrete_env_metrics_env_first(tmp_path):
    alpha_group = uuid.uuid4()
    beta_group = uuid.uuid4()
    rollouts = [
        _rollout(env_name="alpha", group_id=alpha_group, reward=1.0, custom_score=0.2),
        _rollout(env_name="alpha", group_id=alpha_group, reward=0.0, custom_score=0.6),
        _rollout(env_name="beta", group_id=beta_group, reward=1.0, custom_score=0.8),
        _rollout(env_name="beta", group_id=beta_group, reward=1.0, custom_score=1.0),
    ]
    config = OrchestratorConfig(
        output_dir=tmp_path,
        train={
            "env": [
                {"id": "dummy-alpha", "name": "alpha", "group_size": 2},
                {"id": "dummy-beta", "name": "beta", "group_size": 2},
            ]
        },
    )
    batch_metrics = TrainBatchMetrics(
        n_trainable=4,
        num_prefill_tokens=12,
        num_decode_tokens=8,
        rollout_prefill_lens=[3, 3, 3, 3],
        rollout_decode_lens=[2, 2, 2, 2],
        samples_per_rollout=[1, 1, 1, 1],
        samples_shipped=4,
    )

    to_log = MetricsBuilder(config).build(
        step=0,
        rollouts=rollouts,
        metrics=batch_metrics,
        progress=Progress(),
        step_time=1.0,
        save_ckpt_time=0.0,
        teacher_logprobs_time=0.0,
        pre_filter_seen=0,
        pre_filter_dropped=0,
        pre_filter_dropped_by_name={},
    )

    assert to_log["alpha/reward/mean"] == pytest.approx(0.5)
    assert to_log["alpha/seq_len/mean"] == pytest.approx(5.0)
    assert to_log["alpha/timing/env/mean"] == pytest.approx(4.0)
    assert to_log["alpha/metrics/custom_score/mean"] == pytest.approx(0.4)
    assert to_log["alpha/filters/repetition"] == pytest.approx(0.0)
    assert to_log["alpha/solve_none"] == pytest.approx(0.0)
    assert to_log["beta/reward/mean"] == pytest.approx(1.0)

    assert "reward/alpha/mean" not in to_log
    assert "seq_len/alpha/mean" not in to_log
    assert "timing/alpha/env/mean" not in to_log
    assert "metrics/alpha/custom_score" not in to_log
    assert "filters/alpha/repetition" not in to_log

    # Aggregate metrics intentionally keep the reserved "all" pseudo-env shape.
    assert to_log["reward/all/mean"] == pytest.approx(0.75)
