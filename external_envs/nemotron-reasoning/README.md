# nemotron-reasoning

Reasoning environment for the clean Nemotron RL blend: ReasoningGym, Math-v2, Science-v1, and ARC-AGI. QA-abstention is owned by `nemotron-knowledge`, and Ultra blend data is intentionally excluded.

## Sub-datasets

| `dataset` key | Source | Scoring |
|---|---|---|
| `reasoning_gym` | `nvidia/Nemotron-RL-ReasoningGym-v1` | `reasoning_gym` task scorer with normalized fallback |
| `math` | `nvidia/Nemotron-RL-Math-v2` | Exact normalized/boxed-answer match first, then local LLM answer-equivalence judge |
| `science` | `nvidia/Nemotron-RL-Science-v1` | Exact normalized answer match first, then local LLM answer-equivalence judge |
| `arc_agi` | `nvidia/Nemotron-RL-ARC-AGI-v1` | Deterministic transductive grid match |

Use `dataset="all"` for `reasoning_gym,math,science,arc_agi` or pass a comma-separated subset such as `"math,arc_agi"`.

## Scope Notes

The ARC-AGI implementation covers transductive grid-output tasks with deterministic matching. It does not execute arbitrary ARC induction programs. The SWE data is handled in `nemotron-tool-calling` as action/message matching rather than sandboxed issue-resolution execution.

## Anti-hacking Guardrails

All sub-envs are wrapped by default. The guard helper is packaged as `nemotron_reasoning_guardrails.py` so it cannot collide with other Nemotron env guardrail modules in a shared training venv. The guard preserves the raw model output in `raw_completion`, strips closed `<think>...</think>` traces before task scoring, and then applies hard anti-hacking gates plus a small format reward:

```text
final_reward = anti_hacking_multiplier * ((1 - anti_hacking_format_reward_weight) * task_reward + anti_hacking_format_reward_weight * anti_hacking_format_reward)
```

The reward is a scalar, normally in `[0, 1]` when the underlying task scorer is in `[0, 1]`. It is not globally binary: some subtasks produce fractional task rewards, and the small format term gives a nonzero signal for valid reasoning plus visible final output even when hard tasks are initially wrong.

### Required reasoning trace

`anti_hacking_reasoning_required` defaults to `true`, matching `decensor-env`. The guard requires each assistant turn to include a parseable reasoning trace, but it does not inject a default output prompt unless `anti_hacking_output_prompt` is explicitly provided. Accepted trace forms are:

- `<think>...</think>` followed by visible answer text
- auto-open template output where reasoning appears before a closing `</think>`
- provider-side reasoning fields on the assistant message, such as `reasoning`, `reasoning_content`, or `thinking_blocks`, when the model host strips `<think>` tags out of visible `content`

Hard local gates set the multiplier to `0.0` for any unclosed `<think>`, no visible answer after stripping reasoning, or missing reasoning trace. These local gates run even when LLM guard judges are disabled. For hosted Laguna-style training, prefer visible `<think>` traces and set model-side hidden-thinking options off if the renderer strips hidden reasoning before scoring.

### LLM guard judges

When `enable_anti_hacking_judges=true`, the guard calls these judge prompts from `nemotron_reasoning_guardrails.py` after local hard gates pass:

- `COHERENCY_JUDGE_PROMPT`: scores the visible transcript; incoherent, truncated, malformed, repetitive, contradictory, or roleplay-action narration in any assistant turn applies `anti_hacking_incoherent_multiplier`.
- `REASONING_TRACE_JUDGE_PROMPT`: scores extracted `<think>` traces; ethical/policy/evaluator-gaming refusal or rubric manipulation in reasoning hard-fails the rollout.
- `REASONING_COHERENCY_JUDGE_PROMPT`: scores extracted `<think>` traces; incoherent reasoning hard-fails the rollout.
- `META_COMMENTARY_JUDGE_PROMPT`: scores visible output for hidden prompt, rubric, judge, or evaluator meta-commentary and applies `anti_hacking_meta_multiplier`.

Task judges and guard judges default to a local vLLM endpoint:

| Arg | Default |
|---|---|
| `judge_model` | `google/gemma-4-26B-A4B-it` |
| `judge_base_url` | `http://127.0.0.1:8000/v1` |
| `judge_api_key_var` | `VLLM_API_KEY` |
| `enable_task_judges` | `true` |
| `enable_anti_hacking` | `true` |
| `enable_anti_hacking_judges` | `true` |

## Quickstart

Start a local vLLM OpenAI-compatible server for the judge/model endpoint, then run:

```bash
prime env install nemotron-reasoning --path ./environments
VLLM_API_KEY=dummy prime eval run nemotron-reasoning \
  --provider vllm \
  --api-base-url http://127.0.0.1:8000/v1 \
  --api-key-var VLLM_API_KEY \
  --model google/gemma-4-26B-A4B-it \
  --disable-env-server \
  -n 4 -r 1 \
  -a '{"dataset":"arc_agi","num_eval_examples":4,"enable_anti_hacking_judges":false}'
```

## Arguments

| Name | Default | Description |
|---|---|---|
| `dataset` | `all` | `all` (`reasoning_gym,math,science,arc_agi`), `reasoning_gym`, `math`, `science`, `arc_agi`, or comma-separated subset |
| `num_train_examples` | `-1` | Number of shuffled train rows; `-1` uses all available rows |
| `num_eval_examples` | `256` | Number of shuffled eval rows |
| `dataset_seed` | `42` | Shuffle seed; eval uses `dataset_seed + 1` |
| `system_prompt` | `None` | Optional system message merged into dataset system prompts |
| `judge_model` | `google/gemma-4-26B-A4B-it` | Model name sent to the local judge endpoint |
| `judge_base_url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible local judge endpoint |
| `judge_api_key_var` | `VLLM_API_KEY` | Environment variable used for the judge client key |
| `judge_sampling_args` | `None` | Optional judge generation args, for example `{"temperature":0.0,"max_tokens":64}` |
| `enable_task_judges` | `true` | Enables LLM answer-equivalence judges for open-answer datasets |
| `enable_anti_hacking` | `true` | Enables the guardrail wrapper |
| `enable_anti_hacking_judges` | `true` | Enables local LLM guard judges |
| `anti_hacking_judge_model` | `None` | Optional separate guard judge model |
| `anti_hacking_judge_base_url` | `None` | Optional separate guard judge endpoint |
| `anti_hacking_judge_api_key_var` | `None` | Optional separate guard judge API-key env var |
| `anti_hacking_judge_timeout` | `120.0` | Timeout in seconds for each guard judge call |
| `anti_hacking_incoherent_multiplier` | `0.1` | Reward multiplier for incoherent visible answers |
| `anti_hacking_meta_multiplier` | `0.01` | Reward multiplier for meta-commentary about scoring or system prompts |
| `anti_hacking_reasoning_required` | `true` | Require every assistant turn to include a valid `<think>` trace; missing traces hard-fail |
| `anti_hacking_output_prompt` | `None` | Optional prompt prepended to dataset system prompts when explicitly configured |
| `anti_hacking_format_reward_weight` | `0.1` | Weight for the local format reward after hard gates pass |
| `enable_structured_marker_gate` | `false` | Penalize misplaced `<answer>` markers when enabled |

## Metrics

Task metrics include `reasoning_gym_accuracy`, `answer_equivalence`, `arc_agi_accuracy`, and dataset-specific task tags. The guard wrapper adds `anti_hacking_*` metrics for missing reasoning, zero-visible-output, unclosed reasoning tags, coherency, meta-commentary, `anti_hacking_reasoning_words`, `anti_hacking_reasoning_quality`, `anti_hacking_format_reward`, `task_reward`, and the final guard multiplier.
