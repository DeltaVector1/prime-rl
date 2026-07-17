#!/usr/bin/env python3
"""Validate and compare fixed-evaluation gates from saved Prime RL rollouts."""

from __future__ import annotations

import argparse
import hashlib
import math
import re
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import orjson

SOURCES = (
    "code",
    "logic",
    "science",
    "nemotron-trajectory",
    "nemotron-function-calling",
    "nemotron-swe",
)
RUN_DIR_RE = re.compile(r"^run-\d{8}_\d{6}-([A-Za-z0-9]+)$")
TRAIN_PROMPT_END = b',"completion":'


@dataclass(frozen=True)
class SourceGate:
    env_name: str
    path: Path
    rows: list[dict[str, Any]]
    by_prompt: dict[str, dict[str, Any]]
    reward_mean: float
    protocol_failures: int
    errors: int
    cancelled: int
    truncated: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True, help="Candidate checkpoint step to audit.")
    parser.add_argument("--baseline-step", type=int, default=0)
    parser.add_argument("--panel", choices=("eval", "confirm"), required=True)
    parser.add_argument(
        "--expected-count",
        type=int,
        help="Expected rows per source (defaults to 16 for eval and 48 for confirm).",
    )
    parser.add_argument("--sources", default=",".join(SOURCES))
    parser.add_argument("--rollout-root", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wandb-run",
        default="auto",
        help="W&B entity/project/run, project/run, run ID, or 'auto' from the local run directory.",
    )
    parser.add_argument("--skip-wandb", action="store_true")
    parser.add_argument("--skip-train-overlap", action="store_true")
    return parser.parse_args()


def prompt_hash(prompt: Any) -> str:
    payload = orjson.dumps(prompt, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(payload).hexdigest()


def panel_digest(prompt_hashes: set[str]) -> str:
    return hashlib.sha256("\n".join(sorted(prompt_hashes)).encode()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [orjson.loads(line) for line in path.read_bytes().splitlines() if line]


def resolve_rollout_root(output_dir: Path, explicit: Path | None, steps: set[int]) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        missing = [step for step in steps if not (root / f"step_{step}").is_dir()]
        if missing:
            raise FileNotFoundError(f"Missing rollout step directories under {root}: {missing}")
        return root

    output_dir = output_dir.resolve()
    candidates = [output_dir / "rollouts", output_dir / "run_default" / "rollouts"]
    candidates.extend(sorted(output_dir.glob("run_*/rollouts")))
    matches = [
        root
        for root in candidates
        if all(
            (root / f"step_{step}").is_dir()
            and any((root / f"step_{step}").glob("eval_rollouts*.jsonl"))
            for step in steps
        )
    ]
    unique = list(dict.fromkeys(root.resolve() for root in matches))
    if len(unique) != 1:
        raise RuntimeError(
            f"Expected exactly one rollout root containing steps {sorted(steps)}, found {unique}; "
            "pass --rollout-root explicitly"
        )
    return unique[0]


def visible_content(message: dict[str, Any]) -> bool:
    content = message.get("content")
    if isinstance(content, str):
        return bool(content.strip())
    return bool(content)


def reasoning_only_message(message: Any) -> bool:
    if not isinstance(message, dict) or message.get("role") != "assistant":
        return False
    reasoning = message.get("reasoning_content")
    has_reasoning = bool(reasoning.strip()) if isinstance(reasoning, str) else bool(reasoning)
    return has_reasoning and not visible_content(message) and not bool(message.get("tool_calls"))


def is_protocol_failure(row: dict[str, Any]) -> bool:
    if row.get("empty_model_response_reasoning_only") or row.get("empty_model_response_zero_guard"):
        return True
    if row.get("stop_condition") == "empty_model_response_zero_guard":
        return True
    if any(reasoning_only_message(message) for message in row.get("completion") or []):
        return True
    for turn in row.get("trajectory") or []:
        if any(reasoning_only_message(message) for message in turn.get("completion") or []):
            return True
    return False


def error_kind(row: dict[str, Any]) -> str | None:
    error = row.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        value = error.get("error") or error.get("type")
        return str(value) if value is not None else "Unknown"
    return str(error)


def including_errors_mean(rows: list[dict[str, Any]]) -> tuple[float, int, int]:
    non_cancelled = [row for row in rows if error_kind(row) != "Cancelled"]
    if not non_cancelled:
        return 0.0, 0, len(rows)
    rewards = [0.0 if row.get("error") is not None else float(row["reward"]) for row in non_cancelled]
    errors = sum(row.get("error") is not None for row in rows)
    cancelled = sum(error_kind(row) == "Cancelled" for row in rows)
    return float(sum(rewards) / len(rewards)), errors, cancelled


def load_source_gate(
    *,
    path: Path,
    env_name: str,
    step: int,
    expected_count: int,
    violations: list[str],
) -> SourceGate:
    rows = read_jsonl(path)
    prefix = f"{env_name} step {step}"
    if len(rows) != expected_count:
        violations.append(f"{prefix}: expected {expected_count} rows, found {len(rows)}")

    by_prompt: dict[str, dict[str, Any]] = {}
    for index, row in enumerate(rows):
        row_prefix = f"{prefix} row {index}"
        if row.get("env_name") != env_name:
            violations.append(f"{row_prefix}: env_name={row.get('env_name')!r}")
        if row.get("eval_step") != step:
            violations.append(f"{row_prefix}: eval_step={row.get('eval_step')!r}, expected {step}")
        if row.get("policy_version") != step:
            violations.append(f"{row_prefix}: policy_version={row.get('policy_version')!r}, expected {step}")
        if row.get("off_policy_steps") != 0:
            violations.append(f"{row_prefix}: off_policy_steps={row.get('off_policy_steps')!r}, expected 0")
        reward = row.get("reward")
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            violations.append(f"{row_prefix}: non-finite or non-numeric reward {reward!r}")
        digest = prompt_hash(row.get("prompt"))
        if digest in by_prompt:
            violations.append(f"{prefix}: duplicate prompt hash {digest}")
        by_prompt[digest] = row

    reward_mean, errors, cancelled = including_errors_mean(rows)
    if errors:
        violations.append(f"{prefix}: {errors} errored rows ({cancelled} cancelled)")
    return SourceGate(
        env_name=env_name,
        path=path,
        rows=rows,
        by_prompt=by_prompt,
        reward_mean=reward_mean,
        protocol_failures=sum(is_protocol_failure(row) for row in rows),
        errors=errors,
        cancelled=cancelled,
        truncated=sum(bool(row.get("is_truncated")) for row in rows),
    )


def load_panel(
    *,
    rollout_root: Path,
    step: int,
    panel: str,
    sources: tuple[str, ...],
    expected_count: int,
    violations: list[str],
) -> dict[str, SourceGate]:
    step_dir = rollout_root / f"step_{step}"
    expected_envs = {source: f"{source}-{panel}" for source in sources}
    found_paths = {
        path.name.removeprefix("eval_rollouts_").removesuffix(".jsonl"): path
        for path in step_dir.glob("eval_rollouts_*.jsonl")
    }
    actual_panel_envs = {env for env in found_paths if env.endswith(f"-{panel}")}
    missing = set(expected_envs.values()) - actual_panel_envs
    unexpected = actual_panel_envs - set(expected_envs.values())
    if missing:
        violations.append(f"step {step} {panel}: missing environments {sorted(missing)}")
    if unexpected:
        violations.append(f"step {step} {panel}: unexpected environments {sorted(unexpected)}")

    result: dict[str, SourceGate] = {}
    for source, env_name in expected_envs.items():
        path = found_paths.get(env_name)
        if path is None:
            continue
        result[source] = load_source_gate(
            path=path,
            env_name=env_name,
            step=step,
            expected_count=expected_count,
            violations=violations,
        )
    return result


def pair_panels(
    baseline: dict[str, SourceGate],
    candidate: dict[str, SourceGate],
    violations: list[str],
) -> dict[str, np.ndarray]:
    paired: dict[str, np.ndarray] = {}
    for source in sorted(set(baseline) & set(candidate)):
        baseline_hashes = set(baseline[source].by_prompt)
        candidate_hashes = set(candidate[source].by_prompt)
        if baseline_hashes != candidate_hashes:
            missing = baseline_hashes - candidate_hashes
            added = candidate_hashes - baseline_hashes
            violations.append(
                f"{source}: candidate prompt set differs from baseline "
                f"(missing={len(missing)}, added={len(added)})"
            )
            continue
        deltas: list[float] = []
        for digest in sorted(baseline_hashes):
            before = baseline[source].by_prompt[digest]
            after = candidate[source].by_prompt[digest]
            if before.get("example_id") != after.get("example_id"):
                violations.append(
                    f"{source}: prompt {digest} changed example_id "
                    f"{before.get('example_id')!r}->{after.get('example_id')!r}"
                )
            before_reward = 0.0 if before.get("error") is not None else float(before["reward"])
            after_reward = 0.0 if after.get("error") is not None else float(after["reward"])
            deltas.append(after_reward - before_reward)
        paired[source] = np.asarray(deltas, dtype=np.float64)
    return paired


def stratified_bootstrap(
    paired: dict[str, np.ndarray], *, samples: int, seed: int
) -> tuple[float, float, float] | None:
    if not paired or samples < 1 or any(len(values) == 0 for values in paired.values()):
        return None
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples, dtype=np.float64)
    for values in paired.values():
        indices = rng.integers(0, len(values), size=(samples, len(values)))
        draws += values[indices].mean(axis=1) / len(paired)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high), float(np.mean(draws > 0.0))


def extract_train_prompt(line: bytes) -> Any:
    end = line.find(TRAIN_PROMPT_END)
    if end == -1:
        return orjson.loads(line)["prompt"]
    prefix = orjson.loads(line[:end] + b"}")
    return prefix["prompt"]


def check_train_overlap(
    *,
    rollout_root: Path,
    through_step: int,
    eval_hashes: set[str],
    violations: list[str],
) -> tuple[int, int, set[str]]:
    files = sorted(
        (
            path
            for path in rollout_root.glob("step_*/train_rollouts.jsonl")
            if int(path.parent.name.removeprefix("step_")) <= through_step
        ),
        key=lambda path: int(path.parent.name.removeprefix("step_")),
    )
    if not files:
        violations.append(f"no train_rollouts.jsonl files found through step {through_step}")
        return 0, 0, set()
    rows = 0
    overlap: set[str] = set()
    for path in files:
        with path.open("rb") as handle:
            for line in handle:
                if not line.strip():
                    continue
                rows += 1
                digest = prompt_hash(extract_train_prompt(line))
                if digest in eval_hashes:
                    overlap.add(digest)
    if overlap:
        violations.append(
            f"{len(overlap)} evaluation prompts overlap trainer-bound rollouts through step {through_step}"
        )
    return len(files), rows, overlap


def auto_wandb_ref(output_dir: Path) -> str:
    latest = output_dir / "wandb" / "latest-run"
    if not latest.exists():
        raise FileNotFoundError(f"Cannot auto-detect W&B run: {latest} does not exist")
    match = RUN_DIR_RE.match(latest.resolve().name)
    if match is None:
        raise RuntimeError(f"Cannot extract W&B run ID from {latest.resolve().name!r}")
    config_path = output_dir / "configs" / "orchestrator.toml"
    with config_path.open("rb") as handle:
        config = tomllib.load(handle)
    project = config["wandb"]["project"]
    return f"{project}/{match.group(1)}"


def normalize_wandb_ref(output_dir: Path, value: str) -> str:
    if value == "auto":
        return auto_wandb_ref(output_dir)
    if "/" not in value:
        config_path = output_dir / "configs" / "orchestrator.toml"
        with config_path.open("rb") as handle:
            project = tomllib.load(handle)["wandb"]["project"]
        return f"{project}/{value}"
    return value


def crosscheck_wandb(
    *,
    output_dir: Path,
    run_ref: str,
    panels: list[dict[str, SourceGate]],
    steps: list[int],
    violations: list[str],
) -> tuple[str, str, int]:
    import wandb

    run = wandb.Api(timeout=60).run(normalize_wandb_ref(output_dir, run_ref))
    expected: dict[str, dict[int, SourceGate]] = {}
    for panel, step in zip(panels, steps):
        for gate in panel.values():
            expected.setdefault(gate.env_name, {})[step] = gate

    def scan_env(item: tuple[str, dict[int, SourceGate]]) -> tuple[str, list[dict[str, Any]]]:
        env_name, _ = item
        prefix = f"eval/{env_name}"
        keys = [
            "step",
            f"{prefix}/avg@1_including_errors",
            f"{prefix}/policy_version",
            f"{prefix}/cancelled_count",
            f"{prefix}/errored_count",
        ]
        return env_name, list(run.scan_history(keys=keys, page_size=1000))

    with ThreadPoolExecutor(max_workers=min(8, len(expected))) as pool:
        history = dict(pool.map(scan_env, expected.items()))

    checked = 0
    for env_name, by_step in expected.items():
        prefix = f"eval/{env_name}"
        for step, gate in by_step.items():
            rows = [row for row in history[env_name] if row.get("step") == step]
            if len(rows) != 1:
                violations.append(f"W&B {env_name} step {step}: expected one exact history row, found {len(rows)}")
                continue
            row = rows[0]
            expected_values = {
                f"{prefix}/avg@1_including_errors": gate.reward_mean,
                f"{prefix}/policy_version": float(step),
                f"{prefix}/cancelled_count": float(gate.cancelled),
                f"{prefix}/errored_count": float(gate.errors - gate.cancelled),
            }
            for key, expected_value in expected_values.items():
                actual = row.get(key)
                if not isinstance(actual, (int, float)) or not math.isclose(
                    float(actual), expected_value, rel_tol=1e-9, abs_tol=1e-9
                ):
                    violations.append(
                        f"W&B {env_name} step {step}: {key}={actual!r}, saved rollout value={expected_value}"
                    )
            checked += 1
    return "/".join(run.path), run.state, checked


def automated_gate_checks(
    *,
    step: int,
    panel: str,
    baseline: dict[str, SourceGate],
    candidate: dict[str, SourceGate],
    macro_delta: float,
    bootstrap: tuple[float, float, float] | None,
) -> list[tuple[str, bool]]:
    source_deltas = {
        source: candidate[source].reward_mean - baseline[source].reward_mean
        for source in set(baseline) & set(candidate)
    }
    guard_before = sum(gate.protocol_failures for gate in baseline.values())
    guard_after = sum(gate.protocol_failures for gate in candidate.values())
    code_guard_before = baseline.get("code").protocol_failures if "code" in baseline else 0
    code_guard_after = candidate.get("code").protocol_failures if "code" in candidate else 0
    ci_low, _ci_high, probability = bootstrap if bootstrap is not None else (float("nan"),) * 3

    if step == 25 and panel == "eval":
        return [
            ("promising: macro delta >= 0.04", macro_delta >= 0.04),
            ("promising: bootstrap P(delta > 0) >= 0.80", probability >= 0.80),
            ("promising: protocol failures non-increasing", guard_after <= guard_before),
            ("promising: improvement spans at least two sources", sum(v > 0 for v in source_deltas.values()) >= 2),
            ("futility absent: macro delta > -0.03", macro_delta > -0.03),
            ("futility absent: protocol failures increased by < 8", guard_after - guard_before < 8),
            ("futility absent: code delta > -0.125", source_deltas.get("code", 0.0) > -0.125),
            (
                "futility absent: fewer than two sources regress by >= 0.125",
                sum(value <= -0.125 for value in source_deltas.values()) < 2,
            ),
        ]
    if step == 50 and panel == "confirm":
        return [
            ("confirmation: macro delta >= 0.05", macro_delta >= 0.05),
            ("confirmation: paired-bootstrap 95% lower bound > 0", ci_low > 0.0),
            ("confirmation: at least four sources non-regressing", sum(v >= 0 for v in source_deltas.values()) >= 4),
            ("confirmation: no source delta < -0.0625", all(v >= -0.0625 for v in source_deltas.values())),
            ("confirmation: protocol failures non-increasing", guard_after <= guard_before),
            ("confirmation: code protocol failures non-increasing", code_guard_after <= code_guard_before),
        ]
    return []


def format_source_line(source: str, before: SourceGate, after: SourceGate) -> str:
    delta = after.reward_mean - before.reward_mean
    return (
        f"  {source:<28} {before.reward_mean:>7.4f} -> {after.reward_mean:>7.4f} "
        f"({delta:+.4f})  guards {before.protocol_failures:>2}->{after.protocol_failures:<2} "
        f"truncated {before.truncated:>2}->{after.truncated:<2}"
    )


def main() -> None:
    args = parse_args()
    if args.step < 0 or args.baseline_step < 0:
        raise ValueError("steps must be non-negative")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap-samples must be positive")
    sources = tuple(source.strip() for source in args.sources.split(",") if source.strip())
    if len(sources) != len(set(sources)):
        raise ValueError("--sources contains duplicates")
    expected_count = args.expected_count or (16 if args.panel == "eval" else 48)
    output_dir = args.output_dir.resolve()
    rollout_root = resolve_rollout_root(output_dir, args.rollout_root, {args.baseline_step, args.step})
    violations: list[str] = []
    baseline = load_panel(
        rollout_root=rollout_root,
        step=args.baseline_step,
        panel=args.panel,
        sources=sources,
        expected_count=expected_count,
        violations=violations,
    )
    candidate = (
        baseline
        if args.step == args.baseline_step
        else load_panel(
            rollout_root=rollout_root,
            step=args.step,
            panel=args.panel,
            sources=sources,
            expected_count=expected_count,
            violations=violations,
        )
    )

    paired: dict[str, np.ndarray] = {}
    bootstrap = None
    if args.step != args.baseline_step:
        paired = pair_panels(baseline, candidate, violations)
        if len(paired) == len(sources):
            bootstrap = stratified_bootstrap(paired, samples=args.bootstrap_samples, seed=args.seed)

    all_eval_hashes = {
        digest for gate in list(baseline.values()) + list(candidate.values()) for digest in gate.by_prompt
    }
    train_files = train_rows = 0
    train_overlap: set[str] = set()
    if not args.skip_train_overlap:
        train_files, train_rows, train_overlap = check_train_overlap(
            rollout_root=rollout_root,
            through_step=args.step,
            eval_hashes=all_eval_hashes,
            violations=violations,
        )

    wandb_path = wandb_state = "skipped"
    wandb_rows = 0
    if not args.skip_wandb:
        panels = [baseline] if args.step == args.baseline_step else [baseline, candidate]
        steps = [args.baseline_step] if args.step == args.baseline_step else [args.baseline_step, args.step]
        wandb_path, wandb_state, wandb_rows = crosscheck_wandb(
            output_dir=output_dir,
            run_ref=args.wandb_run,
            panels=panels,
            steps=steps,
            violations=violations,
        )

    before_macro = float(np.mean([gate.reward_mean for gate in baseline.values()])) if baseline else float("nan")
    after_macro = float(np.mean([gate.reward_mean for gate in candidate.values()])) if candidate else float("nan")
    macro_delta = after_macro - before_macro

    print(f"Gate integrity: {'PASS' if not violations else 'FAIL'}")
    print(f"Rollouts: {rollout_root}")
    print(
        f"Panel: {args.panel} | baseline step {args.baseline_step} | candidate step {args.step} | "
        f"expected {expected_count} rows/source"
    )
    if args.step == args.baseline_step:
        for source in sources:
            if source not in baseline:
                continue
            gate = baseline[source]
            print(
                f"  {source:<28} reward {gate.reward_mean:.4f}  guards {gate.protocol_failures:>2} "
                f"truncated {gate.truncated:>2}  prompts {panel_digest(set(gate.by_prompt))[:12]}"
            )
        print(f"Macro: {before_macro:.6f}")
    else:
        for source in sources:
            if source in baseline and source in candidate:
                print(format_source_line(source, baseline[source], candidate[source]))
        print(f"Macro: {before_macro:.6f} -> {after_macro:.6f} ({macro_delta:+.6f})")
        if bootstrap is not None:
            low, high, probability = bootstrap
            print(
                f"Paired source-stratified bootstrap ({args.bootstrap_samples} draws, seed {args.seed}): "
                f"95% CI [{low:+.6f}, {high:+.6f}], P(delta > 0)={probability:.4f}"
            )
        for label, passed in automated_gate_checks(
            step=args.step,
            panel=args.panel,
            baseline=baseline,
            candidate=candidate,
            macro_delta=macro_delta,
            bootstrap=bootstrap,
        ):
            print(f"  [{'PASS' if passed else 'FAIL'}] {label}")

    if args.skip_train_overlap:
        print("Train overlap: SKIPPED")
    else:
        print(
            f"Train overlap: scanned {train_rows} trainer-bound rows in {train_files} files through step {args.step}; "
            f"{len(train_overlap)} overlapping prompt hashes"
        )
    print(f"W&B cross-check: {wandb_path} ({wandb_state}), {wandb_rows} environment-step rows checked")
    print(
        "Manual confirmation still required for positive tool/SWE trajectories and for the step-25 routine direction "
        "when deciding the step-50 heldout gate."
    )
    if violations:
        print("Violations:")
        for violation in violations:
            print(f"  - {violation}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
