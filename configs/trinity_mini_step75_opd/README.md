# Cross-tokenizer OPD

`opd.toml` trains `NewEden/Trinity-Mini-Ichthyo` with equal sampling weights
across code, logic, science, Nemotron multi-turn trajectories, Nemotron
function calling, and Nemotron SWE tool use. Training uses groups of eight and
reserves 64 deterministic, source- and prompt-disjoint examples from every
source. The first 48 reserved examples form the confirmation panel and the
remaining 16 form the routine validation panel.

The OPD objective uses learning rate `1e-6`, reward weight `1.0`, gated teacher
weight `0.025`, and a `0.001` reasoning-close advantage. Student context is
32K. `teacher.toml` serves `deepseek-ai/DeepSeek-V4-Flash` with 64K context as
the four-GPU teacher. Science scoring uses the `gemma` judge at
`http://195.26.233.5:40000/v1`.

`protocol_recovery.toml` is the bounded 75-step recovery overlay. It writes to
`outputs/opd-protocol-recovery`, checkpoints every 25 steps, and logs to W&B as
`OPD-protocol-recovery-g8-t025`.

Start the teacher on physical GPUs 4-7:

```bash
curl -fL -o /tmp/cuda-nvcc-12-9.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-nvcc-12-9_12.9.86-1_amd64.deb
curl -fL -o /tmp/cuda-nvvm-12-9.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-nvvm-12-9_12.9.86-1_amd64.deb
mkdir -p /tmp/cuda-nvcc-12-9
dpkg-deb -x /tmp/cuda-nvcc-12-9.deb /tmp/cuda-nvcc-12-9
dpkg-deb -x /tmp/cuda-nvvm-12-9.deb /tmp/cuda-nvcc-12-9
CUDA_HOME=$(uv run scripts/prepare_cuda_wheel_home.py --nvcc-root /tmp/cuda-nvcc-12-9/usr/local/cuda-12.9) CUDA_VISIBLE_DEVICES=4,5,6,7 TILELANG_TARGET=cuda TILELANG_EXECUTION_BACKEND=nvrtc uv run inference @ configs/trinity_mini_step75_opd/teacher.toml
```

Because the orchestrator recycles the teacher after each log-probability phase,
run that command under the restart loop described in
`skills/training/start-run/SKILL.md`.

After `http://127.0.0.1:8001/v1/models` is healthy, start the bounded recovery
run on physical GPUs 0-3. Prime assigns two GPUs to student inference and two
to student training:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run rl \
  @ configs/trinity_mini_step75_opd/opd.toml \
  @ configs/trinity_mini_step75_opd/protocol_recovery.toml
```

Routine validation evaluates 16 examples per source every 25 steps. The
independent 48-example-per-source confirmation panel runs every 50 steps. On a
fresh run both panels establish their step-0 baselines before training starts.
Only compare a gate after confirming every rollout has
`eval_step == policy_version`, zero off-policy steps, and the expected prompt
set.
