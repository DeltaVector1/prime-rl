import asyncio
import ctypes
import gc
import logging
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path

import orjson
import verifiers as vf
from verifiers.utils.client_utils import setup_openai_client
from verifiers.utils.save_utils import make_serializable

from prime_rl.configs.orchestrator import OrchestratorConfig
from prime_rl.transport import TrainingSample
from prime_rl.utils.client import setup_inference_pool
from prime_rl.utils.logger import InterceptHandler, get_logger
from prime_rl.utils.utils import (
    get_broadcast_dir,
    get_ckpt_dir,
    get_step_path,
)

MAX_TEACHER_NON_FINITE_FRACTION = 0.01


@dataclass(frozen=True)
class TeacherLogprobsBatch:
    values: list[list[float]]
    max_non_finite_fraction: float
    severe_sample_count: int
    batch_neutralized: bool

    def metrics(self) -> dict[str, float]:
        return {
            "opd/teacher_logprobs/non_finite_fraction_max": self.max_non_finite_fraction,
            "opd/teacher_logprobs/severe_samples": float(self.severe_sample_count),
            "opd/teacher_logprobs/batch_neutralized": float(self.batch_neutralized),
        }


async def setup_student_inference_pool(*, config: OrchestratorConfig, tokenizer):
    """Build the student inference pool + matching renderer. Returns
    ``(renderer | None, inference_pool)``; ``renderer`` is ``None`` on the
    MITO path (``config.renderer is None``)."""
    from renderers.base import create_renderer

    client_config = config.student.client.model_copy(
        update={"preserve_reasoning_only_responses": config.preserve_reasoning_only_responses}
    )
    model_name = config.student.model.name

    if config.renderer is not None:
        renderer = create_renderer(tokenizer, config.renderer)
        get_logger().info(f"Initialized {type(renderer).__name__} for {model_name}")
        inference_pool = await setup_inference_pool(
            client_config,
            model_name=model_name,
            train_client_type="renderer",
            eval_client_type="openai_chat_completions",
            renderer_config=config.renderer,
            pool_size=config.pool_size,
        )
        get_logger().info("Using direct renderer rollout client")
        return renderer, inference_pool

    get_logger().info("Using MITO (openai_chat_completions) for rollouts")
    inference_pool = await setup_inference_pool(
        client_config,
        model_name=model_name,
        train_client_type="openai_chat_completions",
        eval_client_type="openai_chat_completions",
    )
    return None, inference_pool


def get_model_completion_len(output: vf.RolloutOutput) -> int:
    """Sum of model-generated completion tokens across all turns (excludes
    environment-injected tokens between turns)."""
    return sum(len(step["tokens"]["completion_ids"]) for step in output["trajectory"] if step.get("tokens"))


def get_tool_response_len(output: vf.RolloutOutput) -> int:
    """Total tool-response tokens consumed across the whole rollout, read from a
    harness-emitted metric (e.g. RLM's `rlm_total_tool_response_tokens`, deduped
    across turns/branches/sub-RLMs). Returns 0 when no such metric is present."""
    metrics = output.get("metrics") or {}
    for key, value in metrics.items():
        if key.endswith("total_tool_response_tokens") and isinstance(value, (int, float)):
            return int(value)
    return 0


def save_rollouts(rollouts: list[vf.RolloutOutput], path: Path, exclude_keys: set[str] | None = None) -> None:
    """Save rollouts to a JSONL file using verifiers serialization."""
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "wb") as f:
        for rollout in rollouts:
            row = {k: v for k, v in rollout.items() if k not in exclude_keys} if exclude_keys else rollout
            f.write(orjson.dumps(row, default=make_serializable, option=opts))


def append_jsonl_records(records: list[dict], path: Path) -> None:
    """Append JSONL records using the same serializer as rollout saves."""
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    opts = orjson.OPT_APPEND_NEWLINE | orjson.OPT_SERIALIZE_NUMPY
    with open(path, "ab") as f:
        for record in records:
            f.write(orjson.dumps(record, default=make_serializable, option=opts))


def intercept_vf_logging(logger: str = "verifiers", level: str = "DEBUG", prefix: str | None = None):
    """Intercepts verifiers logging and routes through prime-rl logger with optional prefix."""
    vf_logger = logging.getLogger(logger)
    vf_logger.handlers.clear()
    vf_logger.addHandler(InterceptHandler(prefix=prefix))
    vf_logger.setLevel(level.upper())
    vf_logger.propagate = False


def set_default_executor(max_workers: int = 64) -> None:
    """Scale the default asyncio thread pool so asyncio.to_thread has enough capacity."""
    get_logger().info(f"Setting default executor to ThreadPoolExecutor(max_workers={max_workers})")
    asyncio.get_event_loop().set_default_executor(ThreadPoolExecutor(max_workers=max_workers))


def trim_process_memory() -> None:
    """Return freed heap pages to the OS on glibc systems."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception as exc:
        get_logger().debug(f"malloc_trim(0) failed: {exc!r}")


def _decode_tokens(tokenizer, token_ids: list[int]) -> str:
    return tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def dual_pointer_chunk_alignment(
    student_ids: list[int],
    teacher_ids: list[int],
    student_tokenizer,
    teacher_tokenizer,
) -> list[tuple[int, int, int, int]]:
    """Return all minimal decoded-text synchronization chunks."""
    chunks: list[tuple[int, int, int, int]] = []
    student_start = teacher_start = 0

    while student_start < len(student_ids) and teacher_start < len(teacher_ids):
        student_end = student_start + 1
        teacher_end = teacher_start + 1

        while True:
            student_text = _decode_tokens(student_tokenizer, student_ids[student_start:student_end])
            teacher_text = _decode_tokens(teacher_tokenizer, teacher_ids[teacher_start:teacher_end])

            if student_text == teacher_text and not student_text.endswith("\ufffd"):
                chunks.append((student_start, student_end, teacher_start, teacher_end))
                student_start = student_end
                teacher_start = teacher_end
                break

            if len(student_text) < len(teacher_text):
                student_end += 1
            elif len(teacher_text) < len(student_text):
                teacher_end += 1
            else:
                student_incomplete = student_text.endswith("\ufffd")
                teacher_incomplete = teacher_text.endswith("\ufffd")
                if student_incomplete and not teacher_incomplete:
                    student_end += 1
                elif teacher_incomplete and not student_incomplete:
                    teacher_end += 1
                elif student_incomplete and teacher_incomplete:
                    student_end += 1
                    teacher_end += 1
                else:
                    raise ValueError(
                        "DPCA reached unequal decoded prefixes with the same length: "
                        f"student={student_text!r}, teacher={teacher_text!r}"
                    )

            if student_end > len(student_ids) or teacher_end > len(teacher_ids):
                raise ValueError(
                    "DPCA could not synchronize token sequences: "
                    f"student suffix={_decode_tokens(student_tokenizer, student_ids[student_start:])!r}, "
                    f"teacher suffix={_decode_tokens(teacher_tokenizer, teacher_ids[teacher_start:])!r}"
                )

    if student_start != len(student_ids) or teacher_start != len(teacher_ids):
        raise ValueError(
            "DPCA exhausted only one token sequence: "
            f"student={student_start}/{len(student_ids)}, teacher={teacher_start}/{len(teacher_ids)}"
        )
    return chunks


def project_teacher_chunk_logprobs(
    student_logprobs: list[float],
    teacher_logprobs: list[float],
    chunks: list[tuple[int, int, int, int]],
    invalid_teacher_indices: set[int] | None = None,
) -> list[float]:
    """Project each teacher chunk budget using the student's semantic prior."""
    invalid_teacher_indices = invalid_teacher_indices or set()
    projected = [0.0] * len(student_logprobs)
    for student_start, student_end, teacher_start, teacher_end in chunks:
        student_chunk = student_logprobs[student_start:student_end]
        if any(index in invalid_teacher_indices for index in range(teacher_start, teacher_end)):
            projected[student_start:student_end] = student_chunk
            continue
        teacher_budget = sum(teacher_logprobs[teacher_start:teacher_end])
        student_budget = sum(student_chunk)
        if student_budget > 0.0:
            raise ValueError(
                "DPCA semantic-prior projection requires a non-positive student chunk log-likelihood, "
                f"got {student_budget} for chunk {student_start}:{student_end}"
            )
        if student_budget == 0.0:
            target = [teacher_budget / len(student_chunk)] * len(student_chunk)
        else:
            scale = teacher_budget / student_budget
            target = [scale * logprob for logprob in student_chunk]
        if not all(math.isfinite(value) for value in target):
            raise ValueError(f"DPCA produced non-finite projected logprobs for chunk {student_start}:{student_end}")
        projected[student_start:student_end] = target
    return projected


def _student_to_teacher_chat_text(text: str, student_tokenizer, teacher_tokenizer) -> str:
    student_vocab = student_tokenizer.get_vocab()
    teacher_vocab = teacher_tokenizer.get_vocab()
    student_is_qwen = "<|im_start|>" in student_vocab and "<|im_end|>" in student_vocab
    teacher_is_deepseek = all(
        token in teacher_vocab
        for token in ("<｜begin▁of▁sentence｜>", "<｜User｜>", "<｜Assistant｜>", "<｜end▁of▁sentence｜>")
    )
    if not (student_is_qwen and teacher_is_deepseek):
        return text

    replacements = [
        ("<|begin_of_text|><|im_start|>system\n", "<｜begin▁of▁sentence｜>"),
        (
            "<|begin_of_text|><|im_start|>user\n",
            "<｜begin▁of▁sentence｜><｜User｜>",
        ),
        ("<|im_start|>system\n", "<｜begin▁of▁sentence｜>"),
        ("<|im_end|>\n<|im_start|>user\n", "<｜User｜>"),
        ("<|im_start|>user\n", "<｜begin▁of▁sentence｜><｜User｜>"),
        ("<|im_end|>\n<|im_start|>assistant\n", "<｜Assistant｜>"),
        ("<|im_start|>assistant\n", "<｜Assistant｜>"),
        ("<|im_end|>\n", "<｜end▁of▁sentence｜>"),
        ("<|im_end|>", "<｜end▁of▁sentence｜>"),
        ("<|begin_of_text|>", "<｜begin▁of▁sentence｜>"),
    ]
    for student_marker, teacher_marker in replacements:
        text = text.replace(student_marker, teacher_marker)
    return text


def _trainable_content_spans(sample: TrainingSample, student_tokenizer) -> list[tuple[int, int]]:
    token_ids = list(sample.prompt_ids) + list(sample.completion_ids)
    mask = [False] * len(sample.prompt_ids) + list(sample.completion_mask)
    protocol_ids = set(student_tokenizer.all_special_ids)
    student_vocab = student_tokenizer.get_vocab()
    protocol_ids.update(student_vocab[token] for token in ("<think>", "</think>") if token in student_vocab)
    reasoning_close_id = student_vocab.get("</think>")
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(mask):
        if not mask[index]:
            index += 1
            continue

        run_end = index
        while run_end < len(mask) and mask[run_end]:
            run_end += 1
        if reasoning_close_id is not None:
            close_index = next(
                (position for position in range(index, run_end) if token_ids[position] == reasoning_close_id),
                None,
            )
            if close_index is not None:
                index = close_index + 1

        while index < run_end:
            if token_ids[index] in protocol_ids:
                index += 1
                continue
            span_start = index
            while index < run_end and token_ids[index] not in protocol_ids:
                index += 1
            spans.append((span_start, index))
    return spans


def _find_teacher_token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    indices = [index for index, (left, right) in enumerate(offsets) if left >= start and right <= end and right > left]
    if not indices or offsets[indices[0]][0] != start or offsets[indices[-1]][1] != end:
        raise ValueError(f"Teacher tokenization does not preserve trainable text boundary {start}:{end}")
    if indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError(f"Teacher tokenization produced a non-contiguous span for text boundary {start}:{end}")
    return indices[0], indices[-1] + 1


def align_and_project_teacher_logprobs(
    sample: TrainingSample,
    teacher_ids: list[int],
    teacher_offsets: list[tuple[int, int]],
    teacher_logprobs: list[float],
    student_tokenizer,
    teacher_tokenizer,
    invalid_teacher_indices: set[int] | None = None,
    reasoning_close_advantage: float | None = None,
) -> list[float]:
    student_ids = list(sample.prompt_ids) + list(sample.completion_ids)
    student_old_logprobs = [0.0] * len(sample.prompt_ids) + list(sample.completion_logprobs)
    projected = list(student_old_logprobs)
    teacher_full_text = _decode_tokens(teacher_tokenizer, teacher_ids)

    for student_start, student_end in _trainable_content_spans(sample, student_tokenizer):
        student_prefix = _decode_tokens(student_tokenizer, student_ids[:student_start])
        student_through_span = _decode_tokens(student_tokenizer, student_ids[:student_end])
        teacher_start_char = len(_student_to_teacher_chat_text(student_prefix, student_tokenizer, teacher_tokenizer))
        teacher_end_char = len(
            _student_to_teacher_chat_text(student_through_span, student_tokenizer, teacher_tokenizer)
        )
        student_text = student_through_span[len(student_prefix) :]

        if teacher_full_text[teacher_start_char:teacher_end_char] != student_text:
            raise ValueError(
                "Teacher retokenization changed trainable response text: "
                f"student={student_text!r}, "
                f"teacher={teacher_full_text[teacher_start_char:teacher_end_char]!r}"
            )

        teacher_start, teacher_end = _find_teacher_token_span(teacher_offsets, teacher_start_char, teacher_end_char)
        student_span_ids = student_ids[student_start:student_end]
        teacher_span_ids = teacher_ids[teacher_start:teacher_end]
        chunks = dual_pointer_chunk_alignment(
            student_span_ids,
            teacher_span_ids,
            student_tokenizer,
            teacher_tokenizer,
        )
        target = project_teacher_chunk_logprobs(
            student_old_logprobs[student_start:student_end],
            teacher_logprobs[teacher_start:teacher_end],
            chunks,
            {
                index - teacher_start
                for index in (invalid_teacher_indices or set())
                if teacher_start <= index < teacher_end
            },
        )
        projected[student_start:student_end] = target

    closing_reasoning_id = student_tokenizer.get_vocab().get("</think>")
    if closing_reasoning_id is not None:
        trainable_mask = [False] * len(sample.prompt_ids) + list(sample.completion_mask)
        trainable_tokens = sum(trainable_mask)
        for index, (token_id, is_trainable) in enumerate(zip(student_ids, trainable_mask, strict=True)):
            if is_trainable and token_id == closing_reasoning_id:
                projected[index] = (
                    0.0
                    if reasoning_close_advantage is None
                    else student_old_logprobs[index] + reasoning_close_advantage * trainable_tokens
                )
    return projected


async def compute_teacher_logprobs(
    clients: list[vf.ClientConfig],
    model_name: str,
    samples: list[TrainingSample],
    student_tokenizer=None,
    teacher_tokenizer=None,
    recycle_server: bool = False,
    reasoning_close_advantage: float | None = None,
) -> TeacherLogprobsBatch:
    """Compute teacher model logprobs for a batch of training samples via prefill."""
    import httpx

    from prime_rl.inference.vllm.serving_tokens import PrimeRlGenerateResponse

    cross_tokenizer = (
        student_tokenizer is not None
        and teacher_tokenizer is not None
        and student_tokenizer.get_vocab() != teacher_tokenizer.get_vocab()
    )

    async def _compute_single(
        client_config: vf.ClientConfig,
        sample: TrainingSample,
    ) -> tuple[list[float], float]:
        client = setup_openai_client(client_config)
        student_ids = list(sample.prompt_ids) + list(sample.completion_ids)
        teacher_offsets = None
        if cross_tokenizer:
            student_text = _decode_tokens(student_tokenizer, student_ids)
            teacher_text = _student_to_teacher_chat_text(student_text, student_tokenizer, teacher_tokenizer)
            encoding = teacher_tokenizer(
                teacher_text,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            token_ids = list(encoding["input_ids"])
            teacher_offsets = [tuple(offset) for offset in encoding["offset_mapping"]]
        else:
            token_ids = student_ids

        # Two escape hatches from ``AsyncOpenAI.post``:
        #   1. URL — ``/inference/v1/generate`` is mounted at server root, not
        #      under ``/v1``. Pass an absolute URL so the SDK's
        #      ``_prepare_url`` skips the base-url merge (it short-circuits
        #      when the path passes ``httpx.URL.is_relative_url`` as False).
        #   2. Parse — vLLM's ``GenerateResponse`` is a plain
        #      ``pydantic.BaseModel`` and the SDK's parse layer rejects any
        #      ``cast_to`` that doesn't subclass ``openai.BaseModel``. Use
        #      ``cast_to=httpx.Response`` so the SDK still builds the request
        #      (preserving ``auth_headers``, retries, timeouts, idempotency
        #      keys) and just hands us the raw response to validate ourselves.
        base = str(client.base_url).rstrip("/").removesuffix("/v1")
        http_response = await client.post(
            f"{base}/inference/v1/generate",
            cast_to=httpx.Response,
            body={
                "model": model_name,
                "token_ids": token_ids,
                "sampling_params": {
                    "max_tokens": 1,
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "prompt_logprobs": 1,
                },
            },
        )
        response = PrimeRlGenerateResponse.model_validate_json(http_response.content)
        prompt_logprobs = response.prompt_logprobs or []
        if len(prompt_logprobs) != len(token_ids):
            raise ValueError(
                f"Teacher returned {len(prompt_logprobs)} prompt logprobs for {len(token_ids)} input tokens"
            )

        invalid_indices = set(response.non_finite_prompt_logprob_indices)
        student_old_logprobs = [0.0] * len(sample.prompt_ids) + list(sample.completion_logprobs)
        flat: list[float] = []
        for index, (token_id, entry) in enumerate(zip(token_ids, prompt_logprobs)):
            if not entry:
                flat.append(0.0)
                continue
            token_logprob = entry.get(token_id)
            if token_logprob is None:
                raise ValueError(f"Teacher prompt logprobs are missing input token ID {token_id}")
            lp = token_logprob.logprob if hasattr(token_logprob, "logprob") else token_logprob.get("logprob")
            value = float(lp) if lp is not None else 0.0
            if not math.isfinite(value):
                invalid_indices.add(index)
                value = 0.0
            flat.append(value)
        if invalid_indices:
            get_logger().warning(
                f"Teacher returned non-finite prompt logprobs at {len(invalid_indices)}/{len(token_ids)} "
                f"token positions for {sample.env_name}"
            )
        invalid_fraction = len(invalid_indices) / len(token_ids) if token_ids else 0.0
        if not cross_tokenizer:
            for index in invalid_indices:
                flat[index] = student_old_logprobs[index]
            return flat, invalid_fraction
        assert teacher_offsets is not None
        return (
            align_and_project_teacher_logprobs(
                sample,
                token_ids,
                teacher_offsets,
                flat,
                student_tokenizer,
                teacher_tokenizer,
                invalid_indices,
                reasoning_close_advantage,
            ),
            invalid_fraction,
        )

    computed = await asyncio.gather(
        *[_compute_single(client, sample) for client, sample in zip(cycle(clients), samples)]
    )
    results = [values for values, _ in computed]
    invalid_fractions = [fraction for _, fraction in computed]
    severe_sample_count = sum(fraction > MAX_TEACHER_NON_FINITE_FRACTION for fraction in invalid_fractions)
    batch_neutralized = severe_sample_count > 0
    if batch_neutralized:
        get_logger().warning(
            f"Neutralizing the full OPD teacher batch because {severe_sample_count}/{len(samples)} sample(s) "
            f"exceeded {MAX_TEACHER_NON_FINITE_FRACTION:.1%} non-finite teacher positions"
        )
        results = [[0.0] * len(sample.prompt_ids) + list(sample.completion_logprobs) for sample in samples]
    if recycle_server:
        client = setup_openai_client(clients[0])
        base = str(client.base_url).rstrip("/").removesuffix("/v1")
        await client.post(
            f"{base}/recycle",
            cast_to=httpx.Response,
            body={},
        )

        saw_shutdown = False
        deadline = time.monotonic() + 1800
        async with httpx.AsyncClient(timeout=2.0) as health_client:
            while time.monotonic() < deadline:
                try:
                    health = await health_client.get(f"{base}/health")
                    ready = health.status_code == 200
                except httpx.HTTPError:
                    ready = False
                if not ready:
                    saw_shutdown = True
                elif saw_shutdown:
                    get_logger().info("Teacher server recycled and is ready")
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("Teacher server did not recycle within 1800 seconds")
    return TeacherLogprobsBatch(
        values=results,
        max_non_finite_fraction=max(invalid_fractions, default=0.0),
        severe_sample_count=severe_sample_count,
        batch_neutralized=batch_neutralized,
    )


def get_weight_dir(output_dir: Path, step: int, check_exists: bool = True, wait_timeout: int | None = None) -> Path:
    """Get the weight directory for a given checkpoint step.

    Args:
        output_dir: The output directory for the run.
        step: The checkpoint step.
        check_exists: If True, raises FileNotFoundError if no weight directory exists.
            If False, returns the broadcast directory path without checking existence
            (useful for NCCL mode where weights are broadcasted, not stored on disk).
        wait_timeout: Maximum time in seconds to wait for a stable directory to appear.
            If None, no waiting is performed.
    """
    ckpt_weight_dir = get_step_path(get_ckpt_dir(output_dir), step) / "weight"
    broadcast_weight_dir = get_step_path(get_broadcast_dir(output_dir), step)

    def find_stable_dir() -> Path | None:
        # For checkpoint weights, check STABLE file in parent directory (checkpoints/step_{step}/STABLE)
        ckpt_step_dir = get_step_path(get_ckpt_dir(output_dir), step)
        if (ckpt_step_dir / "STABLE").exists() and ckpt_weight_dir.exists():
            return ckpt_weight_dir

        # For broadcast weights, check STABLE file in the broadcast directory itself
        if (broadcast_weight_dir / "STABLE").exists() and broadcast_weight_dir.exists():
            return broadcast_weight_dir

        return None

    # Check immediately, then wait if needed
    result = find_stable_dir()
    if result is None and wait_timeout:
        start_time = time.time()
        while time.time() - start_time < wait_timeout:
            time.sleep(1)
            result = find_stable_dir()
            if result:
                break

    if result:
        return result
    if not check_exists:
        return broadcast_weight_dir

    raise FileNotFoundError(f"No weight directory found for checkpoint step {step}")
