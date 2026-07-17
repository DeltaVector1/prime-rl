---
name: monitor-run
description: Monitor an ongoing prime-rl training run — find the output directory, tail logs, check key metrics, inspect SLURM jobs, and restart safely. Use when asked to check on a run, debug training, or investigate performance.
---

# Monitor a run

## Runbook

### On launch

1. Find the output dir and read the resolved configs at `{output_dir}/configs/` (start with `rl.toml`).
2. Confirm all processes are alive and the run is making progress.
3. Write the initial summary into `{output_dir}/STATUS.md`.

### Recurring check-ins

Default cadence: **1 hour** (researcher can override). At each check-in:

1. Confirm processes are alive.
2. Grep logs for errors/warnings; note current step and key metrics.
3. **Append** an entry to `{output_dir}/STATUS.md` (never overwrite):

```markdown
## YYYY-MM-DD HH:MM UTC

**Step**: {current_step} / {max_steps}
**Health**: {Healthy | Degraded | Down}

**Progress**: reward/mean, seq_len, truncation, eval scores, env-specific metrics.
**Stability**: entropy, mismatch_kl, grad_norm — flag spikes.
**Performance**: trainer vs orchestrator step time, env lag, inference pressure.

**Notes**: anything unusual (errors, restarts, hangs). Omit if nothing notable.
```

Do not run pytest while a training or inference process is live. The autouse
session cleanup in `tests/conftest.py` terminates processes matching `torchrun`
and `VLLM`, including an unrelated active run.

For runs with long steps, start the event watcher in a dedicated tmux window.
It wakes the specified Codex session after steps 0 and 1, then every N steps,
after each full per-environment evaluation finishes, and immediately on a fatal
log line, trainer exit, or prolonged stall. Orchestrator activity counts as
progress while a full evaluation is draining, so an uncapped step-0 evaluation
does not produce a false trainer-stall alert. The watcher itself makes no model
requests between events. Wakeups are non-blocking and serialized: if a Codex
continuation is already handling one event, later events are recorded but do
not start a competing continuation. A wakeup must inspect current process
health and must not interrupt or restart a healthy run based on stale logs:

For mixed eval cadences, pass `--eval-targets` so the watcher wakes once after
the complete gate instead of once per environment, for example
`--eval-targets 0:12,25:6,50:12,75:6`. Each count must include every eval
environment scheduled at that step.

After a fixed gate completes, use `scripts/audit_eval_gate.py` to validate the
saved JSONLs against W&B, enforce single-policy and prompt-pair integrity,
check trainer-bound prompt overlap, and compute the equal-source paired
bootstrap. For the 16-row routine / 48-row confirmation schedule:

```bash
uv run scripts/audit_eval_gate.py --output-dir {output_dir} --step 0 --panel eval
uv run scripts/audit_eval_gate.py --output-dir {output_dir} --step 0 --panel confirm
uv run scripts/audit_eval_gate.py --output-dir {output_dir} --step 25 --panel eval
uv run scripts/audit_eval_gate.py --output-dir {output_dir} --step 50 --panel eval
uv run scripts/audit_eval_gate.py --output-dir {output_dir} --step 50 --panel confirm
uv run scripts/audit_eval_gate.py --output-dir {output_dir} --step 75 --panel eval
```

The default baseline is step 0 and W&B run discovery is automatic. An audit
failure invalidates the measurement even when the printed reward delta is
positive. The step-50 confirmation still requires manual inspection of
positive tool/SWE trajectories and the separate step-25 routine direction.

Per-request parser tracebacks are rollout-quality signals, not process-fatal
events. Audit them with the completed rollout file; reserve fatal wakeups for
OOM, NCCL, launcher/trainer/orchestrator failure, or dead server/worker events.

```bash
uv run scripts/monitor_run_events.py \
  --output-dir {output_dir} \
  --session-id {codex_session_id} \
  --first-steps 0,1 \
  --notify-every 5 \
  --eval-targets 0:6,25:6 \
  --stall-seconds 2700
```

Its append-only event log is `{output_dir}/monitor_events.jsonl`. Run only one
watcher per output directory. When `clean_output_dir = true`, start the watcher
after the trainer recreates the output directory; trainer liveness detection
must resolve both relative and absolute `configs/trainer.toml` arguments.

When a validated run is waiting only for GPUs owned by another workload, use
`scripts/wait_for_gpu_release.py` in a dedicated tmux window to wake the owning
Codex session after the selected devices fall below the memory threshold. The
woken turn must recheck process ownership before launching; the waiter never
stops processes or starts training itself.

### Restarting a run

**Never restart unless the researcher explicitly asked.** Confirm the exact restart command and the conditions that warrant one.

**Never** run kill or launch commands from your own shell. Dispatch them to the tmux **Launcher** window so the researcher sees what was executed:

```bash
SESSION=$(tmux display-message -p '#S')
tmux send-keys -t "$SESSION:Launcher" 'your command here' Enter
```

After a restart, verify all processes are back up and progress resumed before the next check-in.

---

## Reference

### Where to find things

- `scripts/tmux.sh` launches the run with a `Launcher` window in the named tmux session. The Claude window receives the output dir and session name in its appended prompt — if either is missing, **ask** rather than guess.
- `{output_dir}/configs/` — resolved TOMLs (`rl.toml` has the full picture).
- `{output_dir}/logs/` — see below.
- `{output_dir}/rollouts/step_N/` — saved rollouts for a single unnamed run.
- `{output_dir}/{run_id}/rollouts/step_N/` — saved rollouts when the orchestrator uses a named run. Resolve the actual path before inspecting files:

```bash
find {output_dir} -type f -path '*/rollouts/step_*/train_rollouts.jsonl' | sort -V | tail -1
```

### Logs

```
{output_dir}/logs/
├── trainer.log                # rank 0 stdout
├── orchestrator.log           # orchestrator stdout
├── inference.log              # vLLM stdout
├── trainer/
│   ├── node_*.log             # per-node (multi-node only)
│   └── torchrun/              # per-rank stdout/stderr
├── inference/
│   ├── node_*.log             # per-node (multi-node only)
│   └── router_0.log           # vllm-router per replica (multi-node only)
└── envs/{train,eval}/{env_name}/
    ├── env_server.log
    └── env_worker_*.log
```

Usually tailing `trainer.log`, `orchestrator.log`, and `inference.log` is enough. Drop into per-node or per-rank logs only when debugging. All logs are loguru with `HH:mm:ss  LEVEL  message`; levels: `DEBUG`, `INFO`, `SUCCESS`, `WARNING`, `ERROR`.

Scan for problems:

```bash
grep -E "WARNING|ERROR" {output_dir}/logs/{trainer,orchestrator,inference}.log
grep -E "WARNING|ERROR" {output_dir}/logs/envs/train/*/env_{server,worker_*}.log
```

### Metrics

All metrics print to the console log (and W&B when configured).

When checking a live W&B run through `wandb.Api()`, use `run.state` for status
and `run._attrs.get("heartbeatAt")` for freshness. The installed public `Run`
object does not expose an `updated_at` attribute. Use `run.scan_history()` when
exact recent step rows are required; `run.summary` only reflects the latest
value written for each key and shared trainer/orchestrator runs interleave rows.

**Progress** — orchestrator log:

| Metric | Description |
|--------|-------------|
| `reward/{all,env}/mean` | mean training reward |
| `seq_len/{all,env}/mean` | avg sequence length (tokens) |
| `num_turns/{all,env}/mean` | avg turns per rollout (multi-turn only) |
| `is_truncated/{all,env}/mean` | fraction truncated |
| `empty_rollouts/{all,env}`, `errored_rollouts/{all,env}` | fraction empty/errored |
| `metrics/{env}/{metric}` | env-specific (e.g. pass rate) |
| `eval/{env}/{avg@k,pass@k}` | eval scores when configured |

For benchmark quality, prefer `eval/{env}/avg@k_including_errors`: it counts
non-cancelled model failures as zero. The valid-only `avg@k` remains useful for
separating model failures from infrastructure errors but can overstate accuracy
when the model returns reasoning without a visible answer.

Every fixed-eval gate must contain exactly one `policy_version`, equal to its
`eval_step`. A mixed set is not a checkpoint measurement: reject that gate even
if its aggregate reward improved. During an overlapping train/eval pipeline,
the dispatcher holds an N+1 weight adoption until all queued, grouped, and
in-flight eval generation for N drains; the expected log pair is `Deferring
policy vN+1 weight update...` followed by `Fixed eval generation drained...`.
Any `mixed policy versions` warning means the barrier failed and must be fixed
before comparing hyperparameters.

For guarded envs, compare `reward/{env}/mean` with `metrics/{env}/task_reward`. If task reward is healthy but final reward is flat or much lower, inspect `metrics/{env}/anti_hacking_multiplier` first, then the suppressors: `anti_hacking_coherency`, `anti_hacking_reasoning_coherency`, `anti_hacking_reasoning_ethics`, `anti_hacking_missing_reasoning`, `anti_hacking_local_meta`, and `anti_hacking_meta_commentary`. Coherency judges should evaluate visible text quality only; they should not penalize missing `<think>` tags when reasoning traces were stripped before judging.

**Stability** — trainer log:

| Metric | Description |
|--------|-------------|
| `mismatch_kl/{all,env}/{mean,std,max}` | KL between trainer and (old) inference policy over trainable tokens |
| `entropy/{all,env}/{mean,std,max}` | policy entropy over trainable tokens |
| `masked_advantage_{positive,negative}/mean` | fraction of DPPO-masked tokens with +/- advantage |
| `optim/grad_norm` | spikes may precede divergence |

For ECHO envs, the trainer success line must show nonzero `ECHO NLL` and `Env. Tokens`. A zero environment-token fraction means observation masks did not reach training, so the run is not a valid ECHO comparison.

**Performance** — trainer and orchestrator step independently, so comparing step times shows who's waiting on whom.

| Source | Metric | Description |
|--------|--------|-------------|
| trainer | `time/step` | total trainer step |
| trainer | `time/wait_for_batch` | **high → orchestrator is bottleneck** |
| trainer | `time/forward_backward`, `time/broadcast_weights`, `time/save_ckpt` | phase timings |
| trainer | `perf/throughput`, `perf/mfu` | tokens/s and MFU % |
| orchestrator | `time/step`, `time/generate_completions`, `time/update_weights` | phase timings |
| orchestrator | `time/wait_for_ckpt` | **high → trainer is bottleneck** |
| orchestrator | `scheduler/async_level`, `scheduler/inflight_rollouts` | scheduler state |
| env server | event loop lag (min/mean/p90/p99/max), active task distribution | periodic |

The rollout window is approximately `batch_size * oversampling_factor`. When comparing batch sizes, hold this product constant to isolate batch effects. If the trainer mostly waits for batches while vLLM has no request queue and low KV-cache use, benchmark a larger rollout window before changing completion length or trainer settings.

For MoE models with LoRA on routed experts, compare vLLM generation throughput before and after loading an adapter. Sparse final eval requests can fall to single-digit tokens/s through the fused MoE LoRA path even when batched training rollouts were fast. Restricting LoRA targets is a hyperparameter change and requires a matched quality trial; do not diagnose this tail as judge latency.

For live vLLM stats, query Prometheus directly:

```bash
curl -s http://localhost:8000/metrics | grep -E "num_requests|gpu_cache_usage"
# vllm:num_requests_running, vllm:num_requests_waiting, vllm:gpu_cache_usage_perc (→1.0 = KV cache saturated)
```

### Rollouts

```
{rollout_root}/step_N/
├── train_rollouts.jsonl   # all train rollouts (vf.RolloutOutput, trajectory excluded)
├── eval_rollouts.jsonl    # single-environment eval, only present when eval ran
├── eval_rollouts_{env}.jsonl  # one file per environment for mixed evals
└── train_rollouts.bin     # binary batch consumed by the trainer
```

Resolve both eval layouts with `find {rollout_root}/step_N -maxdepth 1 -type f
-name 'eval_rollouts*.jsonl'` before auditing a checkpoint.

The complete audit stream, including trajectory and judge details when enabled, is `{rollout_root}/rollouts.jsonl`.

For environments that replace an empty visible response with a synthetic zero-reward trajectory, dashboard `no_response` can remain zero even while the policy is failing. Count `stop_condition == "empty_model_response_zero_guard"` in the saved step JSONL, and split `metrics.empty_model_response_reasoning_only` from genuinely empty generations. A rising reasoning-only fraction together with a falling trainable count is output-protocol collapse, even when loss, KL, and throughput remain numerically stable.

When `preserve_reasoning_only_responses` is enabled, those failures no longer
use the synthetic stop condition. Count assistant turns whose
`reasoning_content` is non-empty while both visible `content` and `tool_calls`
are empty, and cross-check the environment's zero-visible-output guard metric.
Only generations with non-empty completion token IDs and a matching logprob
sidecar are preserved; missing or malformed sidecars must still surface as
empty-response errors. Preserved generations should remain in trainer-bound
samples and receive negative reward advantage in mixed-reward groups; a falling
rate at fixed eval is the protocol success signal.

For hybrid OPD, monitor `reward_advantage`, `teacher_kl`, `teacher_gate`, and `combined_advantage` together. `teacher_gate` is the fraction of trained policy tokens whose trajectory was eligible for teacher KL. `reward/all/mean` is still a changing-prompt training metric, so require a fixed-eval improvement before calling the run better. A close-token advantage can prevent reasoning-only drift, but the decisive protocol checks remain `empty_model_response_zero_guard` and the trainer-bound rollout count.

Also monitor `opd/teacher_logprobs/non_finite_fraction_max`,
`opd/teacher_logprobs/severe_samples`, and
`opd/teacher_logprobs/batch_neutralized`. Isolated invalid teacher positions are
neutralized locally. If any sample exceeds the 1% corruption threshold, the
whole teacher batch is replaced by frozen-student targets, making that optimizer
step reward-only. A nonzero neutralization metric is safe containment, but
repeated events mean the teacher backend remains unhealthy.

Compute training pass@k from the unfiltered `rollout` records in the complete
audit stream, grouped by `labels.group_id`. Do not compute it from
`step_N/train_rollouts.jsonl`: an enforced zero-advantage filter removes
all-pass and all-fail groups before that file is written, which makes pass@k
look substantially better than the policy actually is. For a group with `n`
samples and `c` strict successes, use the standard estimator
`1 - C(n-c, k) / C(n, k)` and report the number of complete groups. Treat an
environment's final binary task reward as success; do not count partial judge
credit as a pass.

For sequential tool trajectories, audit the message structure as well as the
aggregate reward. If the dataset expects one action followed by an environment
observation, every assistant action turn must contain exactly one tool call and
every call ID must receive a tool response. A full-reward rollout containing
multiple bundled calls on one turn is a scorer leak, even if its first call is
correct.

Also verify that no model turn appears after the expected final assistant
answer. In Verifiers multi-turn envs this usually means the env recognized
completion inside `env_response` but did not set `state["final_env_response"]`,
so the framework generated once more before rechecking stop conditions.

Count `env_name` in the step JSONL when validating a mixture. Async completion times and pre-batch filtering can make the actual training rows differ materially from the nominal ratios printed in the step summary.

```bash
wc -l {rollout_root}/step_42/train_rollouts.jsonl
head -1 {rollout_root}/step_42/train_rollouts.jsonl | uv run python -m json.tool
jq '.reward' {rollout_root}/step_42/train_rollouts.jsonl
```

### Common failure modes

A few warnings are normal. Escalate when errors are persistent, growing, or hit a large fraction of rollouts.

- **Env workers**: exceptions in env code, timeouts, sandbox errors, OOM kills (most common source — runs user code).
- **Orchestrator**: empty/errored rollout spikes, weight-broadcast failures, checkpoint errors.
- **Trainer**: NCCL/CUDA errors, OOM, NaN loss or gradients.
- **Inference**: NCCL/CUDA errors, OOM, request timeouts.
- **Zero-step rollouts**: inspect `stop_condition` and `error` in `train_rollouts.jsonl`. A rollout with `trajectory=[]` should carry an explicit timeout or prompt-length error; `EmptyTrajectory` means an env completed without producing a step or an error.
- **Eval contamination**: environments without a native eval split can fall back to their train dataset. Compare stable prompt or source identifiers and require zero overlap between eval rows and the train sample window.

### Process tree

All processes use `setproctitle` so they're visible in `ps`/`htop`/`pstree`:

```
PRIME-RL::Launcher
├── PRIME-RL::Inference          (vLLM server, GPU 0)
├── PRIME-RL::Orchestrator       (CPU-only)
│   └── Verifiers::EnvServer     (ZMQ env server per environment)
│       └── Verifiers::EnvWorker0..N
├── torchrun
│   └── PRIME-RL::Trainer        (GPU 1+)
└── tail trainer.log
```

For multi-node runs, trainer and inference processes are on separate nodes — use `srun` or `ssh` to inspect them.
