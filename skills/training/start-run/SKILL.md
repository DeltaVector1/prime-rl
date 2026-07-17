---
name: start-run
description: How to launch prime-rl training runs — the `rl`, `sft`, and `inference` entrypoints, their config classes, and single-node/SLURM/dry-run modes. Use when starting a run or picking the right entrypoint.
---

# Start a run

All entrypoints run via `uv run <command>` and accept TOML configs via `@ path/to.toml` plus CLI overrides.

## Config system at a glance

[`pydantic-config`](https://github.com/PrimeIntellect-ai/pydantic-config) — Pydantic-based TOML + CLI loader. Highlights (see the `configs` skill for full mechanics):

- Config files via `@ path` (TOML / YAML / JSON); CLI args layer on top, deep-merged with class defaults.
- Nested groups via dotted CLI paths — kebab-case on the CLI, snake_case in TOML.
- Bool toggles: bare `--flag` enables, `--no-flag` disables (nested too).
- Lists: space-separated or JSON literal. Dicts: JSON literal, deep-merged with file values.
- Optional sub-configs (`WandbConfig | None`): bare `--wandb` enables defaults; `--wandb @ wandb.toml` enables from a file; `--no-wandb` disables.
- Discriminated unions are switched by the `type` tag (e.g. `--optimizer.type muon`).
- Validation aliases let renamed fields keep working; legacy keys can be remapped in a `model_validator(mode="before")`.
- Auto-generated `--help` panels from `Field(description=...)` or PEP 224 docstrings.
- Friendly errors: required-field boxes, validator errors point at the offending flag, unknown flags get a "did you mean" hint.

## `rl` — RL training

Launches inference server, orchestrator, and trainer as subprocesses.

```bash
uv run rl @ examples/reverse_text/rl.toml
uv run rl @ examples/reverse_text/rl.toml @ examples/reverse_text/slurm_rl.toml   # SLURM
uv run rl @ examples/reverse_text/rl.toml --dry-run                                # write scripts, don't run
```

`--dry-run` still applies `clean_output_dir`. Point experimental configs at a new output directory, or disable cleanup, before validating a config whose existing outputs must be preserved.

- Config: `RLConfig` (`packages/prime-rl-configs/src/prime_rl/configs/rl.py`)
- Entrypoint: `src/prime_rl/entrypoints/rl.py`
- SLURM: single- and multi-node
- Environment packages: before launching a config with a non-core verifier env id,
  verify the package imports under `uv run` (for example
  `uv run python -c "import importlib.util; print(importlib.util.find_spec('rlm_swe'))"`).
  If a local env exists under `deps/research-environments/environments/` but does not
  import, add it to the root `pyproject.toml` env extra, workspace members, and
  `[tool.uv.sources]`, then run `uv sync --all-extras`.

For a Verifiers `MultiTurnEnv`, an `env_response` that recognizes the final
assistant action must set `state["final_env_response"]` before returning. Stop
conditions are checked before `env_response`, so updating only an action index
inside `env_response` permits one extra model generation after success. Use an
empty list when no final environment text should be appended.

Custom Verifiers state fields cross an env-server boundary only when the env
config lists them in `state_columns`. Environments that mark synthetic or
otherwise non-trainable rollouts with `state["skip_training"] = true` must set
`state_columns = ["skip_training"]`; otherwise Prime receives the fabricated
trajectory without the flag and can mistakenly ship it to the trainer.

Reasoning parsers can return sampled tokens and hidden reasoning while visible
content and tool calls are empty. By default Verifiers raises before converting
that native response, so the token IDs and logprobs are lost and an environment
can only record a synthetic non-trainable zero. Set
`orchestrator.preserve_reasoning_only_responses = true` for an RL/OPD experiment
that needs those real tokens retained. Prime forwards the opt-in on the
serialized Verifiers client config so it reaches the environment workers that
construct the actual rollout clients. Acceptance still requires non-empty
sampled completion token IDs and a matching completion-logprob sidecar; hidden
reasoning without that sidecar remains an empty-response error. The environment
still scores the turn as zero visible output, but mixed-reward groups can now
assign it a negative reward advantage. Keep the option explicit because
multi-turn tool environments will treat the reasoning-only turn as a failed
attempt and may offer correction feedback before termination.

## `sft` — SFT training

Launches torchrun internally — never call torchrun directly.

```bash
uv run sft @ examples/reverse_text/sft.toml
uv run sft @ examples/reverse_text/sft.toml --slurm
uv run sft @ examples/reverse_text/sft.toml --dry-run
```

- Config: `SFTConfig` (`packages/prime-rl-configs/src/prime_rl/configs/sft.py`)
- Entrypoint: `src/prime_rl/entrypoints/sft.py`
- SLURM: single- and multi-node

## `inference` — vLLM server

OpenAI-compatible API plus prime-rl custom endpoints (`/update_weights`, `/load_lora_adapter`, `/init_broadcaster`). Always use this entrypoint — never `vllm serve` directly.

```bash
uv run inference @ configs/debug/infer.toml
uv run inference --model.name Qwen/Qwen3-0.6B --model.enforce-eager
```

Smoke checks:

```bash
curl http://<host>:<port>/health
curl http://<host>:<port>/v1/models
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-0.6B", "messages": [{"role": "user", "content": "Hi"}], "max_tokens": 50}'
```

If model loading stops between checkpoint shards without an exception, verify the cached snapshot before retrying:

```bash
uv run hf cache verify <repo-id> --fail-on-missing-files
```

Resume any missing shard with `uv run hf download <repo-id> <filename>`. An interrupted vLLM load can leave an orphan worker holding the Hugging Face cache lock; terminate that worker before retrying the download.

For a local data-parallel teacher behind one URL, set `--api-server-count 1` and set the teacher client's `dp_rank_count` to the teacher DP size. This keeps one stable frontend while `X-data-parallel-rank` pins requests to both engines.

Parser settings are model-specific, even within one model family. Follow the
exact checkpoint model card and smoke-test both visible content and reasoning
fields. In particular, Trinity-Mini uses `deepseek_r1`, while
Trinity-Large-Preview documents the Hermes tool parser without a reasoning
parser; applying `deepseek_r1` to the latter reclassifies ordinary answer text
as hidden reasoning and leaves visible content empty.

OPD teacher prefill returns prompt logprobs for every token. For large vocabularies, cap the vLLM prefill chunk so the logits tensor does not exhaust memory; for Trinity Mini use `--gpu-memory-utilization 0.90 --vllm-extra '{"max_num_batched_tokens": 1024}'`. A 200k-token vocabulary with fp32 logprobs needs about 0.8 GiB per 1,024-token chunk, versus about 6.6 GiB at vLLM's 8,192-token default.

OPD supports different fast tokenizers through Dual-Pointer Chunk Alignment. Prime decodes each student sample, converts supported chat markers into the teacher family, retokenizes the exact text for teacher prefill, finds the minimal decoded-text chunks, and projects each teacher chunk's log-probability budget over student tokens using the frozen student log-probabilities as the semantic prior. Alignment or text-boundary failures are fatal because silently training on mismatched chunks corrupts the model.

For a cross-tokenizer OPD run, size the teacher context for the teacher's
retokenized length rather than the student's configured sequence length. A
student sample at its context limit can expand under the teacher tokenizer and
otherwise fail the entire teacher batch with an HTTP 400. Keep enough margin
for that expansion; for a 32K Trinity-Mini student with DeepSeek-V4 as teacher,
use a 64K teacher context.

The semantic-prior ratio is singular when every student token in a chunk has
log-probability exactly zero. In that case the prior contains no relative
credit information, so split the teacher chunk budget evenly across the
student tokens. A one-token chunk therefore receives the teacher budget
directly. Positive student log-probability sums remain invalid.

Some teacher backends can produce non-finite prompt logprobs at isolated token
positions. The inference endpoint replaces only those wire values with zero and
returns their indices. OPD substitutes the frozen student's logprob at an
affected same-tokenizer position, or neutralizes the minimal DPCA chunk that
contains an affected teacher token. If any sample has more than 1% non-finite
teacher positions, OPD neutralizes the entire teacher batch to frozen-student
targets so a partially corrupted teacher cannot create an extreme update; the
reward objective, when enabled, still trains that step. The orchestrator logs
the maximum fraction, severe-sample count, and batch-neutralized flag to W&B.
Other teacher errors still propagate. Frequent warnings or repeated full-batch
neutralization indicate teacher numerical instability and should be
investigated.

DeepSeek-V4 on the pinned vLLM stack returns finite prompt logprobs for the
first request wave but can return non-finite values when the engine is reused.
Set `orchestrator.teacher.recycle_after_logprobs = true` and run the teacher
command inside a restart loop. The orchestrator calls `/recycle` after it has
received the whole teacher batch, waits for the server to go down and become
healthy again, and only then releases the batch to the trainer. For example:

```bash
while true; do
  uv run inference @ teacher.toml
  sleep 2
done
```

Do not enable recycling unless the server is supervised by such a loop; the
orchestrator will otherwise wait for a replacement until its recycle timeout.

Only visible assistant semantic content is aligned. Hidden reasoning before `</think>`, generated end-of-turn special tokens, and structural whitespace retain the frozen student's log-probability. Reasoning protocol tokens are excluded from semantic alignment even when the tokenizer deliberately registers them with `special=false` so parsers can see them. By default, a trainable `</think>` receives a zero log-probability target. Set `orchestrator.opd_reasoning_close_advantage` to give that transition a sequence-normalized advantage relative to the frozen student when a stronger answer-emission signal is required. The configured value is multiplied by the sample's trainable-token count so one transition is not diluted by long reasoning traces; `trainer.opd_loss.teacher_tau` still scales the resulting advantage. For responses without a reasoning close token, the full trainable assistant content remains semantic. For Qwen-style students and DeepSeek teachers, Prime maps the role markers while preserving the student's visible response text verbatim; this includes Trinity's `<tool_call>` surface form.

OPD uses `trainer.opd_loss` independently from the RL loss configuration. `teacher_tau` weights the per-token teacher KL, while `reward_tau` mixes in the group-relative rollout advantage across trainable policy tokens. The default `reward_tau = 0` preserves pure distillation. A positive value makes environment reward an explicit part of hybrid OPD. `reward_gate_teacher = true` restricts teacher KL to trajectories with positive environment reward, which is useful when dense teacher updates on failed or all-zero groups regress the task metric. Use fixed evals and empty-response rates to tune these fields rather than inferring improvement from training loss alone.

Reward-gated OPD requires the trainer's packed reward tensor on the same CUDA device as trainer and teacher log-probabilities. A device-mismatch failure on the first forward/backward pass means the reward tensor was not transferred with the other loss inputs; validate the packed `compute_loss` path before restarting.

For a 4-GPU tensor-parallel teacher, use one API frontend and one teacher client rank (`api_server_count = 1`, `parallel.tp = 4`, `parallel.dp = 1`, and `teacher.client.dp_rank_count = 1`). `dp_rank_count` counts data-parallel engines, not tensor-parallel ranks.

DeepSeek-V4's vLLM backend currently requires an FP8 KV cache. Set `vllm_extra.kv_cache_dtype = "fp8"`; leaving the default `auto` fails during engine construction before weights load.

On the pinned Torch/vLLM stack, DeepSeek-V4 graph compilation can fail during the profiling forward pass with `AssertionError: auto_functionalized was not removed`. Set `model.enforce_eager = true` for the teacher to bypass TorchInductor; teacher prefill remains correct, with lower throughput.

If CUDA runtime is present but the container does not include `nvcc`, extract both the NVIDIA `cuda-nvcc-12-9` and `cuda-nvvm-12-9` Debian packages into the same directory under `/tmp`, then launch DeepSeek-V4 with `CUDA_HOME=$(uv run scripts/prepare_cuda_wheel_home.py --nvcc-root /tmp/cuda-nvcc-12-9/usr/local/cuda-12.9) TILELANG_TARGET=cuda TILELANG_EXECUTION_BACKEND=nvrtc`. DeepGEMM needs the full compiler and NVVM tree; the `nvidia-cuda-nvcc-cu12` Python wheel alone only supplies `ptxas`. The helper creates a temporary CUDA tree from the extracted compiler plus the installed runtime, NVCC-header, and CCCL wheels without modifying `.venv`. Without the explicit target, TileLang reports no CUDA; without NVRTC, it selects an unavailable system compiler; without the merged header layout, the mHC and DeepGEMM kernels cannot find `cuda_device_runtime_api.h`, `crt/host_defines.h`, `cuda/std`, or `nv/target`. Prime's inference patch also upgrades TileLang's stale C++17 NVRTC flag to C++20 because TileLang 0.1.9's reduction headers use explicit lambda template parameters.

DeepSeek-V4's sparse-attention CuTeDSL source calls `cute.arch.fmin`, but recent `nvidia-cutlass-dsl` builds export only `fmax`. Prime restores the missing operation through the identity `min(a, b) = -max(-a, -b)` in its vLLM worker patch layer. If the first real prefill request shuts the server down with `AttributeError: module 'cutlass.cute.arch' has no attribute 'fmin'`, verify the general plugin loaded in every worker before changing dependency pins.

Pure OPD does not require a nonzero reward advantage, and hybrid OPD can still
obtain teacher signal from all-pass or all-fail groups. Explicitly disable the
zero-advantage filter in both filter stages; changing only
`pre_batch_filters` leaves the default enforced post-batch filter active and
silently discards valid OPD training samples.

```toml
[[orchestrator.pre_batch_filters]]
type = "zero_advantage"
enforce = false

[[orchestrator.post_batch_filters]]
type = "zero_advantage"
enforce = false
```

When a train environment has no native eval dataset, a separately constructed
eval environment may silently reuse the beginning of the same deterministically
shuffled train dataset. Reserve the eval prefix after shuffling and start the
train dataset after that prefix, or provide a source-disjoint eval split. Before
launching, compare stable prompt or trajectory identifiers and require zero
overlap between the full eval set and the sampled train set.

- Config: `InferenceConfig` (`packages/prime-rl-configs/src/prime_rl/configs/inference.py`)
- Entrypoint: `src/prime_rl/entrypoints/inference.py`
- SLURM: single-node, multi-node, and disaggregated deployments

## Summary

| Command | Purpose | Typical use |
|---------|---------|-------------|
| `rl` | Full RL pipeline | Production RL training |
| `sft` | Supervised fine-tuning | SFT and hard-distill |
| `inference` | vLLM server | Standalone serving / debugging |

## Key paths

- `src/prime_rl/entrypoints/` — `rl`, `sft`, `inference` (+ `trainer`, `orchestrator` for direct launches)
- `packages/prime-rl-configs/src/prime_rl/configs/` — all config classes
- `configs/debug/` — minimal debug configs
- `examples/` — full example configs (e.g. `reverse_text/`)
