#!/usr/bin/env python3
"""Wake a Codex session when selected GPUs become available."""

from __future__ import annotations

import argparse
import subprocess
import time
from datetime import UTC, datetime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True, help="Comma-separated zero-based GPU indices")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--threshold-mib", type=int, default=1024)
    parser.add_argument("--poll-seconds", type=int, default=30)
    return parser.parse_args()


def gpu_memory_used() -> dict[int, int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    usage: dict[int, int] = {}
    for line in result.stdout.splitlines():
        index_text, memory_text = line.split(",", maxsplit=1)
        usage[int(index_text.strip())] = int(memory_text.strip())
    return usage


def main() -> None:
    args = parse_args()
    gpu_indices = [int(value) for value in args.gpus.split(",")]
    if not gpu_indices:
        raise ValueError("At least one GPU index is required")
    if args.threshold_mib < 0 or args.poll_seconds < 1:
        raise ValueError("threshold-mib must be non-negative and poll-seconds must be positive")

    while True:
        usage = gpu_memory_used()
        missing = [index for index in gpu_indices if index not in usage]
        if missing:
            raise ValueError(f"nvidia-smi did not report GPU indices {missing}")
        busy = {index: usage[index] for index in gpu_indices if usage[index] > args.threshold_mib}
        if not busy:
            break
        print(f"{datetime.now(UTC).isoformat()} waiting for GPU release: {busy}", flush=True)
        time.sleep(args.poll_seconds)

    prompt = (
        f"GPUs {args.gpus} are now below {args.threshold_mib} MiB used. "
        "Continue the active Prime RL goal: recheck all GPU/process ownership, then launch the validated "
        "OPD protocol-recovery trial through the tmux Launcher and monitor its fixed evaluation gates."
    )
    subprocess.run(
        [
            "codex",
            "exec",
            "resume",
            args.session_id,
            prompt,
            "-c",
            'approval_policy="never"',
            "-c",
            'sandbox_mode="danger-full-access"',
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
