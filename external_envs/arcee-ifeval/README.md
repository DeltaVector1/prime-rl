# ifeval

### Overview
- **Environment ID**: `ifeval`
- **Short description**: Single-turn instruction following evaluation using RLVR-IFeval with JSON constraint rewards. Adds explicit <think>...</think> reasoning section before the final answer.
- **Tags**: ifeval, single-turn, chat, constraints, none-reasoning, train, eval

### Dataset
- **Source**: `allenai/RLVR-IFeval`
- **Splits**: Uses `validation` or `test` if present; otherwise creates a train/eval holdout from `train` via `train_test_split` (seed=42).
- **Fields used**:
  - `messages`: list of chat messages containing the question
  - `ground_truth`: JSON string with constraint and args

### Prompting & Schema
- **System message**: "Follow the user's instructions exactly. First, think step-by-step inside <think>...</think>. Then, after </think>, provide ONLY your final response that satisfies the user's constraints."
- **User message**: Contains the question from the first user message in the original messages
- **Example schema per example**:
  - `prompt`: list of messages `[{"role":"system",...}, {"role":"user",...}]`
  - `answer`: JSON string with constraint and args (ground truth)

### Parser & Rewards
- **Parser**:
  - `reasoning=false` (default): None — evaluates the whole assistant message as the answer.
  - `reasoning=true`: `SingleThinkXMLParser` — evaluates only the section after `</think>` as the final answer.
- **Rewards**:
  - IFEvalRubric: JSON-constraint check on the “final answer” text (whole message if `reasoning=false`, or after `</think>` if `reasoning=true`).
  - When `reasoning=true`: adds shaping (+0.2) for a well-formed single <think>…</think> block with non-empty final answer; otherwise -0.2.

### Environment Arguments
| Arg | Type | Default | Description |
| --- | ---- | ------- | ----------- |
| `num_train_examples` | int | `-1` | Limit training set size (`-1` for all) |
| `num_eval_examples` | int | `-1` | Limit eval set size (`-1` for all) |
| `reasoning` | bool | `false` | If true, require `<think>…</think>` and score only the post-`</think>` final answer (adds small shaping reward) |

### Quickstart

Evaluate with defaults (uses the env's internal dataset handling):

```bash
uv run vf-eval ifeval \
  -a '{"num_train_examples":-1, "num_eval_examples":-1, "reasoning": true}'
```

Notes:
- Use `-a` / `--env-args` to pass environment-specific configuration as a JSON object.
- Reports (if produced) will be placed under `./environments/ifeval/reports/`.

## Evaluation Reports

<!-- Do not edit below this line. Content is auto-generated. -->
<!-- vf:begin:reports -->
<p>No reports found. Run <code>uv run vf-eval ifeval -a '{"key": "value"}'</code> to generate one.</p>
<!-- vf:end:reports -->
