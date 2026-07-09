# nemotron-instruction-following

Instruction-following env combining latest individual Nemotron-RL instruction-following datasets without the older structured-output or MultiTurnChat releases.

## Sub-datasets

| `dataset` key | Source | Scoring |
|---|---|---|
| `ifeval` | `nvidia/Nemotron-RL-instruction_following` | Deterministic IFEval/IFBench constraint checks |
| `structured_v2`, `structured_v2_direct`, `structured_v2_diversified`, `structured_v2_tool_calling` | `nvidia/Nemotron-RL-Instruction-Following-Structured-Outputs-v2` | JSON/YAML/TOML/CSV/XML parser plus schema checks where applicable; `structured_v2_tool_calling` exposes native row tools and scores native tool-call arguments |
| `citation_format` | `nvidia/Nemotron-RL-Instruction-Following-Citation-Formatting-v1` | NeMo-Gym string/regex format verifier |
| `freeform_formatting` | `nvidia/Nemotron-RL-Instruction-Following-Free-Form-Formatting-v1` | NeMo-Gym string/regex format verifier |
| `calendar` | `nvidia/Nemotron-RL-Instruction-Following-Calendar-v2` | Local LLM judge against expected calendar state |
| `multiturn` | alias for `multichallenge` | Compatibility alias for the latest MultiChallenge selector |
| `adversarial` | `nvidia/Nemotron-RL-Instruction-Following-Adversarial-v1` | Local LLM judge over per-row criteria |
| `identity` | `nvidia/Nemotron-RL-Identity-Following-v1` | Deterministic checks plus local LLM criteria judge |
| `sysbench` | `nvidia/Nemotron-RL-SysBench-v1` | Local LLM criteria judge |
| `cfbench` | `nvidia/Nemotron-RL-CFBench-v1` | Local LLM criteria judge |
| `multichallenge` | `nvidia/Nemotron-RL-Multichallenge-v1` | Local LLM criteria judge |
| `inverse_ifeval` | `nvidia/Nemotron-RL-InverseIFEval-v1` | Deterministic checks plus local LLM criteria judge |

## Anti-hacking Guardrails

All sub-envs are wrapped with decensor-style guardrails by default. The guard helper is packaged as `nemotron_instruction_following_guardrails.py` so it cannot collide with other Nemotron env guardrail modules in a shared training venv. The guard preserves the raw model output in `raw_completion`, strips closed `<think>...</think>` traces before task scoring, and then applies hard anti-hacking gates plus a small format reward:

```text
final_reward = anti_hacking_multiplier * ((1 - anti_hacking_format_reward_weight) * task_reward + anti_hacking_format_reward_weight * anti_hacking_format_reward)
```

The reward is a scalar, normally in `[0, 1]` when the underlying task scorer is in `[0, 1]`. It is not globally binary: IFEval pass fractions, calendar `0..3` normalized scores, averaged criteria judges, and the small local format term can return fractional task rewards.

### Required reasoning trace

`anti_hacking_reasoning_required` defaults to `true`, matching `decensor-env`. The guard requires each assistant turn to include a parseable reasoning trace, but it does not inject a default output prompt unless `anti_hacking_output_prompt` is explicitly provided. Accepted trace forms are:

- `<think>...</think>` followed by visible answer text
- auto-open template output where reasoning appears before a closing `</think>`
- provider-side reasoning fields on the assistant message, such as `reasoning`, `reasoning_content`, or `thinking_blocks`, when the model host strips `<think>` tags out of visible `content`

Hard local gates set the multiplier to `0.0` for any unclosed `<think>`, no visible answer after stripping reasoning, or missing reasoning trace. These local gates run even when LLM guard judges are disabled. For hosted Laguna-style training, prefer visible `<think>` traces and set model-side hidden-thinking options off if the renderer strips hidden reasoning before scoring.

### LLM guard judges

When `enable_anti_hacking_judges=true`, the guard calls these judge prompts from `nemotron_instruction_following_guardrails.py` after local hard gates pass:

- `COHERENCY_JUDGE_PROMPT`: scores the visible transcript; incoherent, truncated, malformed, repetitive, contradictory, or roleplay-action narration in any assistant turn applies `anti_hacking_incoherent_multiplier`.
- `REASONING_TRACE_JUDGE_PROMPT`: scores extracted `<think>` traces; ethical/policy/evaluator-gaming refusal or rubric manipulation in reasoning hard-fails the rollout.
- `REASONING_COHERENCY_JUDGE_PROMPT`: scores extracted `<think>` traces; incoherent reasoning hard-fails the rollout.
- `META_COMMENTARY_JUDGE_PROMPT`: scores visible output for hidden prompt, rubric, judge, or evaluator meta-commentary and applies `anti_hacking_meta_multiplier`.

Guardrails default to enabled and point judges at local vLLM:

| Arg | Default |
|---|---|
| `judge_model` | `google/gemma-4-26B-A4B-it` |
| `judge_base_url` | `http://127.0.0.1:8000/v1` |
| `judge_api_key_var` | `VLLM_API_KEY` |
| `enable_anti_hacking` | `true` |
| `enable_anti_hacking_judges` | `true` |

## Quickstart

```bash
prime env install nemotron-instruction-following --path ./environments
VLLM_API_KEY=dummy prime eval run nemotron-instruction-following \
  --provider vllm \
  --api-base-url http://127.0.0.1:8000/v1 \
  --api-key-var VLLM_API_KEY \
  --model google/gemma-4-26B-A4B-it \
  --disable-env-server \
  -n 4 -r 1 \
  -a '{"dataset":"citation_format","num_eval_examples":4}'
```

Use `dataset="all"` for the clean instruction blend or a comma-separated subset.

## Tool-Calling Structured Outputs

`structured_v2_tool_calling` is intentionally not scored as visible JSON text. Rows in the source split set `response_mode="tool_call"` and include OpenAI-compatible tool schemas under `responses_create_params.tools`; the env injects those schemas as runtime `tool_defs`, passes the dataset `tool_choice` and `parallel_tool_calls` flags through sampling args, and validates the emitted tool-call arguments against the row schema. A plain assistant message containing matching JSON receives `0.0` on these rows.

Other instruction splits with multi-message prompts, such as `calendar`, `sysbench`, `cfbench`, `multichallenge`, and `structured_v2_diversified`, are history-context final-response tasks. They remain one assistant response over the full chat history rather than interactive multi-turn rollouts.

## Arguments

| Name | Default | Description |
|---|---|---|
| `dataset` | `all` | `all`, `ifeval`, `structured_v2`, `structured_v2_direct`, `structured_v2_diversified`, `structured_v2_tool_calling`, `citation_format`, `freeform_formatting`, `calendar`, `multiturn`/`multichallenge`, `adversarial`, `identity`, `sysbench`, `cfbench`, `inverse_ifeval`, or comma-separated subset |
| `num_train_examples` | `-1` | Number of shuffled train rows; `-1` uses all available rows |
| `num_eval_examples` | `256` | Number of shuffled eval rows |
| `dataset_seed` | `42` | Shuffle seed; eval uses `dataset_seed + 1` |
| `system_prompt` | `None` | Optional system message merged into dataset system prompts |
| `judge_model` | `google/gemma-4-26B-A4B-it` | Model name sent to judge-graded task and guard endpoints |
| `judge_base_url` | `http://127.0.0.1:8000/v1` | OpenAI-compatible local judge endpoint |
| `judge_api_key_var` | `VLLM_API_KEY` | Environment variable used for the judge client key |
| `judge_sampling_args` | `None` | Optional judge generation args, for example `{"temperature":0.0,"max_tokens":64}` |
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
| `enable_structured_marker_gate` | `false` | Penalize misplaced structured markers when enabled |

## Metrics

Task rubrics emit subtask metrics such as `ifeval_score`, `ifeval_strict`, `schema_valid`, `structured_v2_tool_call_emitted`, `structured_v2_expected_tool_called`, `calendar_judge`, and criteria pass rates. The guard wrapper adds `anti_hacking_*` metrics for missing reasoning, zero-visible-output, unclosed reasoning tags, coherency, meta-commentary, `anti_hacking_reasoning_words`, `anti_hacking_reasoning_quality`, `anti_hacking_format_reward`, `task_reward`, and the final guard multiplier.
