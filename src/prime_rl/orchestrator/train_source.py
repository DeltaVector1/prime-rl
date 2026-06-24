"""TrainSource: weighted round-robin across train envs, infinite pull.

Weights default to configured ``ratio`` (when every env sets one) or to
per-env dataset size. ``next_example`` reshuffles on cursor exhaustion."""

from __future__ import annotations

import random

from prime_rl.orchestrator.envs import TrainEnvs


class TrainSource:
    """``next_example(available_permits)`` picks a weighted-RR env and
    returns its next example (or ``None`` when the env's per-call permit
    cost doesn't fit — the dispatch loop retries when permits free up).
    Returned dicts carry ``env_name`` + ``example_id``."""

    def __init__(self, train_envs: TrainEnvs, *, seed: int | None) -> None:
        self.rng = random.Random(seed)
        self.envs = list(train_envs)
        if not self.envs:
            raise ValueError("TrainSource needs at least one train env")

        self.examples: dict[str, list[dict]] = {}
        self.cursors: dict[str, int] = {}
        # Group-scoring envs reserve ``group_size`` permits up front;
        # per-rollout envs need 1
        self.env_costs: dict[str, int] = {}
        for env in self.envs:
            rows: list[dict] = []
            for row in env.get_dataset(seed=seed):
                ex = dict(row)
                ex["env_name"] = env.name
                rows.append(ex)
            self.rng.shuffle(rows)
            self.examples[env.name] = rows
            self.cursors[env.name] = 0
            self.env_costs[env.name] = env.config.group_size if env.requires_group_scoring else 1

        self.env_names = [e.name for e in self.envs]
        configured_ratios = [e.config.ratio for e in self.envs]
        if all(r is not None for r in configured_ratios):
            self.weights: list[float] = [float(r) for r in configured_ratios]  # type: ignore[arg-type]
        else:
            self.weights = [float(len(self.examples[name])) for name in self.env_names]
        self.total_weight = sum(max(0.0, weight) for weight in self.weights)
        if self.total_weight <= 0.0:
            raise ValueError("TrainSource needs at least one env with positive ratio or examples")
        self.current_weights = {name: 0.0 for name in self.env_names}

    def _next_env_name(self, available_permits: int) -> str | None:
        next_weights = dict(self.current_weights)
        for name, weight in zip(self.env_names, self.weights, strict=True):
            next_weights[name] += max(0.0, weight)
        env_name = max(self.env_names, key=lambda name: next_weights[name])
        if self.env_costs[env_name] > available_permits:
            return None
        self.current_weights = next_weights
        self.current_weights[env_name] -= self.total_weight
        return env_name

    def next_example(self, available_permits: int) -> dict | None:
        env_name = self._next_env_name(available_permits)
        if env_name is None:
            return None
        rows = self.examples[env_name]
        cursor = self.cursors[env_name]
        if cursor >= len(rows):
            self.rng.shuffle(rows)
            cursor = 0
        example = rows[cursor]
        self.cursors[env_name] = cursor + 1
        return example
