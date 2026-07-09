decensor-env

Unified RL reward environment for decensoring/anti-refusal training.
Expects datasets with a "prompt" column (plain string).
All judges are toggleable via TOML args.

Reward formula:
  final = base_reward * word_mult * coherency_mult * md_presence_mult * md_correctness_mult * reasoning_mult * slop_mult * reasoning_quality_mult

Where base_reward = 1.0 - (refusal_score / 10.0)

Feature toggles (all in [env.args]):
  enable_system_prompts    - prepend style/length system prompts (default: true)
  enable_adherence         - adherence judge gate (default: true)
  enable_word_count        - word count multiplier (default: true)
  enable_structured_markers - XML/bracket/YAML gate (default: true)
  enable_markdown_judges   - markdown presence + correctness judges (default: true)
  enable_reasoning_trace   - penalize ethics-based refusal in <think> (default: true)
  reasoning_enabled        - add reasoning instructions to prompts (default: true)
  reasoning_output_prompt  - override the default prompt requiring a closed visible <think>...</think> block before the answer
  enable_slop_guard        - slop-guard heuristic multiplier (default: false)
  markdown_exempt_probability - % of rollouts exempt from markdown penalty (default: 0.25)

Judges:
  refusal     - rates refusal severity 1-9, always on
  coherency   - coherent + complete + no roleplay narration, always on
  adherence   - style prompt adherence, early exit gate
  markdown    - presence (no unnecessary md) + correctness (right list types)
  reasoning   - checks <think> trace for ethics-based refusal reasoning
  slop-guard  - local heuristic, no LLM call, scores 0-100

By default, the reasoning contract is prepended to the system prompt and a short
format reminder is appended to the final user turn. It tells the policy that the
first characters of every assistant turn must be `<think>`, followed by a closed
visible `<think>...</think>` block, then the answer.
Reasoning traces can also come from provider-side assistant message fields such
as `reasoning`, `reasoning_content`, or `thinking_blocks` when a host exposes
them. For hosted Laguna-style training, prefer visible `<think>` traces and set
model-side hidden thinking options off if the renderer strips hidden reasoning
before scoring.

Example TOML:
  [[env]]
  id = "mangymango/decensor-env"

  [env.args]
  dataset_names = ["NewEden/RL-Seed-Mix-Iter-3"]
  dataset_ratios = [1.0]
  num_train_examples = 19000
  judge_model = "Qwen/Qwen3-VL-32B-Instruct-FP8"
  judge_base_url = "http://72.46.85.157:31974/v1"
  enable_system_prompts = false
  enable_adherence = false
  enable_word_count = false
  enable_slop_guard = true
