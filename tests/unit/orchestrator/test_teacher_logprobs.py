import asyncio
import json

import httpx
import verifiers as vf

from prime_rl.orchestrator import utils as orchestrator_utils
from prime_rl.transport import TrainingSample


class _FakeOpenAIClient:
    """Stand-in for ``AsyncOpenAI`` that captures the sole ``.post()`` call and
    returns a synthesized ``httpx.Response`` so ``cast_to=httpx.Response`` is
    handed back verbatim, mirroring the real SDK's short-circuit at
    ``AsyncAPIClient._process_response``."""

    def __init__(self, payload: dict):
        # Match what AsyncOpenAI exposes — utils.py reads ``str(client.base_url)``.
        self.base_url = "http://fake-host:8000/v1"
        self._payload = payload
        self.calls: list[dict] = []

    async def post(self, url, *, cast_to, body):
        self.calls.append({"url": url, "cast_to": cast_to, "body": body})
        request = httpx.Request("POST", url, json=body)
        return httpx.Response(
            status_code=200,
            content=json.dumps(self._payload).encode(),
            request=request,
        )


class _PieceTokenizer:
    def __init__(self, pieces: dict[int, str], encoding: list[tuple[str, int]]):
        self.pieces = pieces
        self.encoding = encoding
        self.all_special_ids: list[int] = []

    def get_vocab(self) -> dict[str, int]:
        return {piece: token_id for token_id, piece in self.pieces.items()}

    def decode(self, token_ids, **_kwargs) -> str:
        return "".join(self.pieces[token_id] for token_id in token_ids)

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping):
        assert not add_special_tokens
        assert return_offsets_mapping
        input_ids = []
        offsets = []
        cursor = 0
        for piece, token_id in self.encoding:
            assert text.startswith(piece, cursor)
            input_ids.append(token_id)
            offsets.append((cursor, cursor + len(piece)))
            cursor += len(piece)
        assert cursor == len(text)
        return {"input_ids": input_ids, "offset_mapping": offsets}


class _SequenceTokenizer:
    def __init__(self, decoded: dict[tuple[int, ...], str]):
        self.decoded = decoded

    def decode(self, token_ids, **_kwargs) -> str:
        return self.decoded[tuple(token_ids)]


def test_dual_pointer_chunk_alignment_finds_minimal_chunks():
    student = _PieceTokenizer({1: "a", 2: "bc", 3: "d"}, [])
    teacher = _PieceTokenizer({10: "ab", 11: "c", 12: "d"}, [])

    chunks = orchestrator_utils.dual_pointer_chunk_alignment([1, 2, 3], [10, 11, 12], student, teacher)

    assert chunks == [(0, 2, 0, 2), (2, 3, 2, 3)]


def test_dual_pointer_chunk_alignment_does_not_sync_replacement_fragments():
    student = _SequenceTokenizer({(1,): "\ufffd", (1, 2): "é"})
    teacher = _SequenceTokenizer({(10,): "\ufffd", (10, 11): "é"})

    chunks = orchestrator_utils.dual_pointer_chunk_alignment([1, 2], [10, 11], student, teacher)

    assert chunks == [(0, 2, 0, 2)]


def test_project_teacher_chunk_logprobs_preserves_budget_and_semantic_prior():
    projected = orchestrator_utils.project_teacher_chunk_logprobs(
        student_logprobs=[-1.0, -3.0],
        teacher_logprobs=[-0.5, -1.5],
        chunks=[(0, 2, 0, 2)],
    )

    assert projected == [-0.5, -1.5]
    assert sum(projected) == -2.0


def test_project_teacher_chunk_logprobs_splits_zero_student_budget_evenly():
    projected = orchestrator_utils.project_teacher_chunk_logprobs(
        student_logprobs=[0.0, 0.0],
        teacher_logprobs=[-1.0],
        chunks=[(0, 2, 0, 1)],
    )

    assert projected == [-0.5, -0.5]


def test_compute_teacher_logprobs_uses_inference_generate(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-test",
                "choices": [],
                # Upstream wire shape: list[dict[token_id, Logprob] | None]
                "prompt_logprobs": [
                    None,
                    {"999": {"logprob": -0.01}, "2": {"logprob": -0.7}},
                    {"998": {"logprob": -0.02}, "3": {"logprob": -0.3}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)

        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[True],
            completion_ids=[2, 3],
            completion_mask=[True, True],
            completion_logprobs=[-0.1, -0.2],
            completion_temperatures=[1.0, 1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
        )

        assert result.values == [[0.0, -0.7, -0.3]]
        assert result.batch_neutralized is False
        assert fake_client.calls == [
            {
                "url": "http://fake-host:8000/inference/v1/generate",
                "cast_to": httpx.Response,
                "body": {
                    "model": "teacher-model",
                    "token_ids": [1, 2, 3],
                    "sampling_params": {
                        "max_tokens": 1,
                        "temperature": 1.0,
                        "top_p": 1.0,
                        "prompt_logprobs": 1,
                    },
                },
            }
        ]

    asyncio.run(_run())


def test_compute_teacher_logprobs_projects_cross_tokenizer_chunks(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-cross-tokenizer",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {11: {"logprob": -0.5}},
                    {12: {"logprob": -1.5}},
                ],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)
        student_tokenizer = _PieceTokenizer({1: "P", 2: "a", 3: "bc"}, [])
        teacher_tokenizer = _PieceTokenizer(
            {10: "P", 11: "ab", 12: "c"},
            [("P", 10), ("ab", 11), ("c", 12)],
        )
        sample = TrainingSample(
            prompt_ids=[1],
            prompt_mask=[False],
            completion_ids=[2, 3],
            completion_mask=[True, True],
            completion_logprobs=[-1.0, -3.0],
            completion_temperatures=[1.0, 1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
            student_tokenizer=student_tokenizer,
            teacher_tokenizer=teacher_tokenizer,
        )

        assert result.values == [[0.0, -0.5, -1.5]]
        assert result.batch_neutralized is False
        assert fake_client.calls[0]["body"]["token_ids"] == [10, 11, 12]

    asyncio.run(_run())


def test_align_teacher_logprobs_distills_visible_content_and_anchors_reasoning_close():
    student_tokenizer = _PieceTokenizer(
        {1: "P", 2: "reason", 3: "</think>", 4: "answer", 5: "<eos>"},
        [],
    )
    student_tokenizer.all_special_ids = [5]
    teacher_tokenizer = _PieceTokenizer(
        {10: "P", 11: "reason", 12: "</think>", 13: "answer", 14: "<eos>"},
        [("P", 10), ("reason", 11), ("</think>", 12), ("answer", 13), ("<eos>", 14)],
    )
    sample = TrainingSample(
        prompt_ids=[1],
        prompt_mask=[False],
        completion_ids=[2, 3, 4, 5],
        completion_mask=[True, True, True, True],
        completion_logprobs=[-1.0, -0.01, -2.0, -0.02],
        completion_temperatures=[1.0, 1.0, 1.0, 1.0],
        env_name="test-env",
    )

    projected = orchestrator_utils.align_and_project_teacher_logprobs(
        sample=sample,
        teacher_ids=[10, 11, 12, 13, 14],
        teacher_offsets=[(0, 1), (1, 7), (7, 15), (15, 21), (21, 26)],
        teacher_logprobs=[0.0, -0.5, -7.0, -0.25, -9.0],
        student_tokenizer=student_tokenizer,
        teacher_tokenizer=teacher_tokenizer,
        reasoning_close_advantage=1.0,
    )

    assert projected == [0.0, -1.0, 3.99, -0.25, -0.02]


def test_compute_teacher_logprobs_neutralizes_batch_after_severe_non_finite_response(monkeypatch):
    async def _run():
        fake_client = _FakeOpenAIClient(
            {
                "request_id": "gen-non-finite",
                "choices": [],
                "prompt_logprobs": [
                    None,
                    {"2": {"logprob": -0.2}},
                    {"3": {"logprob": 0.0}},
                    {"4": {"logprob": -0.7}},
                ],
                "non_finite_prompt_logprob_indices": [2],
                "kv_transfer_params": None,
            }
        )
        monkeypatch.setattr(orchestrator_utils, "setup_openai_client", lambda _: fake_client)
        sample = TrainingSample(
            prompt_ids=[1, 2],
            prompt_mask=[False, False],
            completion_ids=[3, 4],
            completion_mask=[True, True],
            completion_logprobs=[-0.4, -1.2],
            completion_temperatures=[1.0, 1.0],
            env_name="test-env",
        )

        result = await orchestrator_utils.compute_teacher_logprobs(
            clients=[vf.ClientConfig()],
            model_name="teacher-model",
            samples=[sample],
        )

        assert result.values == [[0.0, 0.0, -0.4, -1.2]]
        assert result.batch_neutralized is True
        assert result.severe_sample_count == 1
        assert result.max_non_finite_fraction == 0.25

    asyncio.run(_run())


def test_project_teacher_chunk_logprobs_neutralizes_invalid_chunk():
    projected = orchestrator_utils.project_teacher_chunk_logprobs(
        student_logprobs=[-1.0, -3.0, -2.0],
        teacher_logprobs=[-0.5, 0.0, -0.7],
        chunks=[(0, 2, 0, 2), (2, 3, 2, 3)],
        invalid_teacher_indices={1},
    )

    assert projected == [-1.0, -3.0, -0.7]
