# Trinity Mini Prime RL Run

This directory contains the single-node RL config for `arcee-ai/Trinity-Mini`.

## What To Watch

- `reward/all/mean` is only a coarse blended signal. Use `reward/tool-calling/mean`, `reward/instruction-following/mean`, `reward/knowledge/mean`, `reward/reasoning/mean`, and `reward/decensor/mean` to see which env is moving.
- Binary-style envs do not show smooth hill climbing. For those, watch `solve_none`, `solve_some`, `solve_all`, `filter/zero_advantage`, and per-env reward histograms. A flat mean for the first few dozen steps is not automatically a broken run.
- Mixed binary and fractional rewards are valid in Prime RL because advantages are computed per completed rollout group. They are not globally normalized, so the equal `ratio = 1.0` env mix is intentional; change ratios only after checking per-env reward and advantage distributions.
- `num_turns` is logged as an average over rollouts. Values like `1.5` mean the batch had a mix of one-turn and two-turn trajectories, not fractional conversations.
- The decensor env samples integer target turns from 1 to 6 with geometric decay. Saved rollout JSONL contains the raw `trajectory`; inspect it if `num_turns/decensor/mean` is unexpectedly low.
- Guardrail health shows up in env-specific metrics: `anti_hacking_*`, `task_reward`, `anti_hacking_format_reward`, decensor `unclosed_think`, coherency, markdown, and meta-commentary fields. If `reward/*/mean` stays flat, check whether guard multipliers are zeroing the task rewards.
- Baseline preservation is tracked by eval scalars every 25 steps: `eval/primeintellect-aime2025/*`, `eval/primeintellect-aime2024/*`, `eval/primeintellect-math500/*`, `eval/primeintellect-gpqa/*`, `eval/primeintellect-ifeval/*`, and `eval/primeintellect-mmlu-pro/*`.

## Local Notes

- The five training env packages were installed from the `mangymango/*` hub envs, but the config uses local package IDs so the run does not reinstall private envs on startup.
- The judge endpoint at `http://216.243.220.29:40001/v1/models` advertises the model id `Gemma3`, so the env config uses `Gemma3`.
- Trinity Mini's tokenizer template always opens generation with `<think>`, and the renderer parses `</think>` with `reasoning_parser = "think"`.
- Renderers must load Trinity Mini with the vanilla HF tokenizer. The fastokens shim does not support Trinity's `Digits` pre-tokenizer and causes `ModelError() -> ValueError('pre-tokenizer error: unsupported pre-tokenizer type: Digits')` in env workers.
- The renderer pool is capped at 1 per env worker. A 64-slot pool constructs too many Trinity tokenizer copies and can race lazy `transformers` tokenizer imports on worker startup.
- The config uses the cached local Trinity Mini snapshot to avoid HF API repo-list calls during vLLM startup; the HF API was rate-limiting this node.
