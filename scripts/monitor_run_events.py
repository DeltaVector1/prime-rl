#!/usr/bin/env python3
"""Wake a Codex session when a long-running Prime RL run needs attention."""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

STEP_RE = re.compile(r"Step (\d+) \|")
EVAL_COMPLETE_RE = re.compile(r"Evaluated ([\w-]+) \(Step (\d+)\).*$", re.MULTILINE)
FATAL_RE = re.compile(
    r"CUDA out of memory|OutOfMemoryError|NCCL.*(?:error|failed)|"
    r"ChildFailedError|SignalException|Training failed|Orchestrator failed|"
    r"RL trainer failed|EngineCore failed|APIServer process died|Worker.*died",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--notify-every", type=int, default=5)
    parser.add_argument("--first-steps", default="0,1")
    parser.add_argument(
        "--eval-targets",
        default="",
        help="Comma-separated eval_step:environment_count gates; wake only after the full gate completes.",
    )
    parser.add_argument("--stall-seconds", type=int, default=2700)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def latest_step(trainer_log: Path) -> int:
    if not trainer_log.exists():
        return -1
    matches = STEP_RE.findall(trainer_log.read_text(errors="replace"))
    return int(matches[-1]) if matches else -1


def process_alive(output_dir: Path) -> bool:
    result = subprocess.run(
        ["pgrep", "-af", "configs/trainer.toml"],
        check=False,
        capture_output=True,
        text=True,
    )
    trainer_config = (output_dir / "configs" / "trainer.toml").resolve()
    for line in result.stdout.splitlines():
        _, _, command = line.partition(" ")
        for argument in shlex.split(command):
            if argument.endswith("/configs/trainer.toml") and Path(argument).resolve() == trainer_config:
                return True
    return False


def new_log_text(log_path: Path, offset: int) -> tuple[int, str]:
    if not log_path.exists():
        return 0, ""
    size = log_path.stat().st_size
    if size < offset:
        offset = 0
    with log_path.open(errors="replace") as handle:
        handle.seek(offset)
        text = handle.read()
        new_offset = handle.tell()
    return new_offset, text


def fatal_line(text: str) -> str | None:
    match = FATAL_RE.search(text)
    if match is None:
        return None
    line_start = text.rfind("\n", 0, match.start()) + 1
    line_end = text.find("\n", match.end())
    if line_end == -1:
        line_end = len(text)
    return text[line_start:line_end].strip()


def append_event(path: Path, event: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")


def parse_eval_targets(value: str) -> dict[int, int]:
    targets: dict[int, int] = {}
    for item in value.split(","):
        if not item.strip():
            continue
        step_text, count_text = item.split(":", maxsplit=1)
        step = int(step_text)
        count = int(count_text)
        if step < 0 or count < 1:
            raise ValueError(f"Invalid eval target {item!r}; expected non-negative step and positive count")
        targets[step] = count
    return targets


_wake_process: subprocess.Popen[str] | None = None


def wake_codex(session_id: str, output_dir: Path, reason: str, event_log: Path) -> None:
    global _wake_process
    if _wake_process is not None and _wake_process.poll() is None:
        append_event(
            event_log,
            {
                "created_at": datetime.now(UTC).isoformat(),
                "event": "wake_suppressed",
                "reason": reason,
            },
        )
        return

    prompt = (
        f"Automated Prime RL monitor event for {output_dir}: {reason}. "
        "Use skills/training/monitor-run/SKILL.md, inspect the live logs and W&B metrics, "
        "then continue the active training objective. Another Codex turn may still be active. "
        "Never interrupt, restart, or clean a healthy run, and never act on stale log failures."
    )
    wake_log = (output_dir / "monitor_codex.log").open("a")
    _wake_process = subprocess.Popen(
        [
            "codex",
            "exec",
            "resume",
            session_id,
            prompt,
            "-c",
            'approval_policy="never"',
            "-c",
            'sandbox_mode="danger-full-access"',
        ],
        stdout=wake_log,
        stderr=subprocess.STDOUT,
        text=True,
    )
    wake_log.close()
    append_event(
        event_log,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "event": "wake_started",
            "pid": _wake_process.pid,
            "reason": reason,
        },
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    trainer_log = output_dir / "logs" / "trainer.log"
    watched_logs = [
        trainer_log,
        output_dir / "logs" / "orchestrator.log",
        output_dir / "logs" / "inference.log",
    ]
    event_log = output_dir / "monitor_events.jsonl"
    first_steps = {int(step) for step in args.first_steps.split(",") if step.strip()}
    eval_targets = parse_eval_targets(args.eval_targets)
    eval_completions: dict[int, set[str]] = defaultdict(set)
    notified_eval_steps: set[int] = set()
    offsets = {path: path.stat().st_size if path.exists() else 0 for path in watched_logs}
    orchestrator_log = output_dir / "logs" / "orchestrator.log"
    last_step = latest_step(trainer_log)
    last_progress_at = time.monotonic()
    stall_reported = False
    seen_alive = process_alive(output_dir)

    append_event(
        event_log,
        {
            "created_at": datetime.now(UTC).isoformat(),
            "event": "monitor_started",
            "last_step": last_step,
            "notify_every": args.notify_every,
            "stall_seconds": args.stall_seconds,
        },
    )

    while True:
        time.sleep(args.poll_seconds)
        step = latest_step(trainer_log)
        if step > last_step:
            last_step = step
            last_progress_at = time.monotonic()
            stall_reported = False
            if step in first_steps or (args.notify_every > 0 and (step + 1) % args.notify_every == 0):
                wake_codex(args.session_id, output_dir, f"trainer completed step {step}", event_log)

        orchestrator_active = False
        for path in watched_logs:
            offsets[path], new_text = new_log_text(path, offsets[path])
            fatal = fatal_line(new_text)
            if fatal is not None:
                wake_codex(args.session_id, output_dir, f"fatal log line in {path.name}: {fatal}", event_log)
            if path == orchestrator_log and new_text:
                orchestrator_active = True
                for env_name, eval_step_text in EVAL_COMPLETE_RE.findall(new_text):
                    eval_step = int(eval_step_text)
                    target = eval_targets.get(eval_step)
                    if target is None:
                        wake_codex(
                            args.session_id,
                            output_dir,
                            f"evaluation completed for {env_name} at step {eval_step}",
                            event_log,
                        )
                        continue
                    eval_completions[eval_step].add(env_name)
                    if len(eval_completions[eval_step]) >= target and eval_step not in notified_eval_steps:
                        notified_eval_steps.add(eval_step)
                        wake_codex(
                            args.session_id,
                            output_dir,
                            f"full {target}-environment evaluation gate completed at step {eval_step}",
                            event_log,
                        )

        if orchestrator_active:
            last_progress_at = time.monotonic()
            stall_reported = False

        alive = process_alive(output_dir)
        if seen_alive and not alive:
            wake_codex(args.session_id, output_dir, "trainer process exited", event_log)
            return
        seen_alive = seen_alive or alive

        stalled_for = time.monotonic() - last_progress_at
        if not stall_reported and stalled_for >= args.stall_seconds:
            stall_reported = True
            wake_codex(
                args.session_id,
                output_dir,
                f"no completed trainer step for {int(stalled_for)} seconds",
                event_log,
            )


if __name__ == "__main__":
    main()
