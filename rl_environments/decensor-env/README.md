# decensor-env

Multi-turn anti-refusal training environment built from `NewEden/RL-seed-Decensor` with a configurable generalist keep-alive mix.

## Rollouts

Decensor rows supply the initial user prompt and target turn count. After every assistant response, an OpenAI-compatible user-simulator endpoint generates the next user turn until `target_turns` or `max_turns` is reached. Keep-alive rows remain single-turn and preserve their source conversation as context.

The decensor prompt mix uses style, plaintext, and markdown buckets. Optional length requirements are added to the system prompt. Turn counts follow `turn_decay_ratio`, and source rows are sampled across harm categories.

Train and eval examples are built once and split by initial user prompt. Eval prompts are excluded from training. `num_train_examples=-1` uses all locally available examples remaining after the eval holdout.

## Guardrails

Every generated assistant turn must satisfy the configured reasoning contract. A rollout is hard-zeroed before judge calls when it is truncated, has an unclosed `<think>` tag, contains reasoning without visible output, or contains visible output without a reasoning trace when `reasoning_enabled=true`. User-simulator request and parse failures propagate to the orchestrator retry path instead of inserting synthetic user text.

The scoring pipeline records each judge prompt, response, model, and error in `state["judge_logs"]`. Decensor rows use refusal, coherency, reasoning, meta-commentary, word-count, and bucket-specific markdown terms. Keep-alive rows use the lighter refusal, coherency, reasoning, and meta-commentary path.

```text
base_reward = 1 - refusal_score / 10
final_reward = base_reward * word_multiplier * coherent_mult
             * md_presence_mult * md_correctness_mult * reasoning_mult
             * meta_mult * (0.90 + 0.10 * reasoning_quality)
```

Disabled terms contribute a multiplier of `1`. Keep-alive rows omit the word-count and markdown terms.

## Key Arguments

| Name | Default | Description |
|---|---:|---|
| `decensor_dataset_name` | `NewEden/RL-seed-Decensor` | Local JSONL path or Hugging Face dataset name |
| `num_train_examples` | `10000` | Train rows; `-1` uses the full local source after eval holdout |
| `num_eval_examples` | `500` | Source-disjoint eval rows |
| `max_turns` | `6` | Maximum assistant turns |
| `turn_decay_ratio` | `0.5` | Geometric decay for target turn counts |
| `keep_alive_ratio` | `0.15` | Fraction of generalist keep-alive prompts |
| `judge_model` | `google/gemma-4-26B-A4B-it` | Judge model name |
| `judge_base_url` | local OpenAI-compatible URL | Judge endpoint or endpoint list |
| `user_sim_model` | judge model | User-simulator model name |
| `user_sim_base_url` | judge endpoint | User-simulator endpoint or endpoint list |
| `reasoning_enabled` | `true` | Require a reasoning trace on every assistant turn |
| `enable_word_count` | `true` | Apply length-requirement scoring |
| `enable_markdown_judges` | `true` | Apply bucket-aware markdown scoring |
| `enable_reasoning_trace` | `true` | Reject ethics or policy-based refusal reasoning |
| `enable_reasoning_coherency` | `true` | Reject incoherent reasoning traces |
| `enable_meta_commentary` | `true` | Penalize prompt or grader meta-commentary |
