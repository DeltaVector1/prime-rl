import pytest
import torch

from prime_rl.configs.trainer import CustomLossConfig, DefaultLossConfig, OPDLossConfig
from prime_rl.trainer.rl.loss import LossInputs, LossOutputs, compute_entropy, compute_loss, opd_loss_fn, setup_loss_fns

pytestmark = [pytest.mark.gpu]


def test_grpo_loss():
    trainer_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(50, dtype=torch.float32).cuda(), torch.randn(30, dtype=torch.float32).cuda()]
    advantages = [torch.randn(50).cuda(), torch.randn(30).cuda()]
    loss_mask = [torch.ones(50, dtype=torch.bool).cuda(), torch.ones(30, dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_gspo_loss():
    trainer_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    inference_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    teacher_logprobs = [torch.randn(40, dtype=torch.float32).cuda(), torch.randn(60, dtype=torch.float32).cuda()]
    advantages = [torch.randn(40).cuda(), torch.randn(60).cuda()]
    loss_mask = [torch.ones(40, dtype=torch.bool).cuda(), torch.ones(60, dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig(dppo_mask_high=10.0))
    loss, _ = compute_loss(
        trainer_logprobs,
        inference_logprobs,
        teacher_logprobs,
        advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=1.0,
    )
    assert loss.shape == ()


def test_entropy_loss():
    shifted_logits = torch.randn(10, 10, 10, dtype=torch.float32).cuda()
    entropy = compute_entropy(shifted_logits)
    assert entropy.shape == (10, 10)


def test_setup_loss_fns_with_custom_config():
    """Test setup_loss_fns with CustomLossConfig importing a custom loss."""
    loss_config = CustomLossConfig(
        import_path="tests.unit.train.rl.test_loss._dummy_custom_loss",
        kwargs={"multiplier": 2.0},
    )
    loss_fns = setup_loss_fns(loss_config)

    inputs = LossInputs(
        trainer_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        inference_logprobs=torch.randn(50, dtype=torch.float32).cuda(),
        teacher_logprobs=None,
        advantages=torch.randn(50).cuda(),
        loss_mask=torch.ones(50, dtype=torch.bool).cuda(),
    )

    result = loss_fns["rl"](inputs)
    assert isinstance(result, LossOutputs)
    assert result.loss.shape == ()
    assert "custom_metric" in result.metrics


def test_sft_loss_matches_masked_nll():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.zeros(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=2,
        training_mode="sft",
    )

    # loss = -sum(masked logprobs) / loss_scale = -(-0.1 - 0.2) / 2 = 0.15
    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics


def test_sft_loss_override_uses_masked_nll_with_default_loss_config():
    trainer_logprobs = [torch.tensor([-0.1, -0.5, -0.2], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.zeros(3, dtype=torch.float32).cuda()]
    advantages = [torch.ones(3, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, False, True], dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig())
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=2,
        training_mode="sft",
    )

    assert torch.isclose(loss, torch.tensor(0.15, device=loss.device), atol=1e-6)
    assert "nll" in metrics
    assert "mismatch_kl" not in metrics


def test_opd_advantage_combines_reward_and_reward_gated_teacher_signal():
    result = opd_loss_fn(
        LossInputs(
            trainer_logprobs=torch.tensor([-0.2], dtype=torch.float32).cuda(),
            inference_logprobs=torch.tensor([-1.0], dtype=torch.float32).cuda(),
            teacher_logprobs=torch.tensor([-0.5], dtype=torch.float32).cuda(),
            advantages=-torch.ones(1, dtype=torch.float32).cuda(),
            loss_mask=torch.ones(1, dtype=torch.bool).cuda(),
            rewards=torch.zeros(1, dtype=torch.float32).cuda(),
        ),
        OPDLossConfig(reward_tau=0.25, reward_gate_teacher=True),
    )

    assert torch.isclose(result.metrics["teacher_kl"], torch.tensor(0.5, device=result.loss.device))
    assert torch.isclose(result.metrics["teacher_gate"], torch.tensor(0.0, device=result.loss.device))
    assert torch.isclose(result.metrics["reward_advantage"], torch.tensor(-1.0, device=result.loss.device))
    assert torch.isclose(result.metrics["combined_advantage"], torch.tensor(-0.25, device=result.loss.device))
    assert torch.isclose(result.metrics["is_masked_low"], torch.tensor(0.0, device=result.loss.device))

    loss_fns = setup_loss_fns(DefaultLossConfig(), OPDLossConfig(reward_gate_teacher=True))
    loss, metrics = compute_loss(
        trainer_logprobs=[torch.tensor([-0.2], dtype=torch.float32).cuda()],
        inference_logprobs=[torch.tensor([-1.0], dtype=torch.float32).cuda()],
        teacher_logprobs=[torch.tensor([-0.5], dtype=torch.float32).cuda()],
        advantages=[-torch.ones(1, dtype=torch.float32).cuda()],
        loss_mask=[torch.ones(1, dtype=torch.bool).cuda()],
        rewards=[torch.zeros(1, dtype=torch.float32).cuda()],
        loss_fns=loss_fns,
        loss_scale=1,
        training_mode="opd",
    )
    assert loss.is_cuda
    assert torch.isclose(metrics["teacher_gate"], torch.tensor([0.0], device=loss.device)).all()


def test_echo_loss_adds_environment_cross_entropy():
    trainer_logprobs = [torch.tensor([-0.1, -0.5], dtype=torch.float32).cuda()]
    inference_logprobs = [torch.tensor([-0.1, 0.0], dtype=torch.float32).cuda()]
    advantages = [torch.zeros(2, dtype=torch.float32).cuda()]
    loss_mask = [torch.tensor([True, True], dtype=torch.bool).cuda()]
    environment_mask = [torch.tensor([False, True], dtype=torch.bool).cuda()]

    loss_fns = setup_loss_fns(DefaultLossConfig(echo_alpha=0.2))
    loss, metrics = compute_loss(
        trainer_logprobs=trainer_logprobs,
        inference_logprobs=inference_logprobs,
        teacher_logprobs=None,
        advantages=advantages,
        loss_mask=loss_mask,
        loss_fns=loss_fns,
        loss_scale=1,
        training_mode="echo",
        environment_mask=environment_mask,
    )

    assert torch.isclose(loss, torch.tensor(0.1, device=loss.device), atol=1e-6)
    assert torch.isclose(metrics["echo_nll"], torch.tensor([0.5], device=loss.device), atol=1e-6).all()
    assert torch.isclose(metrics["echo_token_fraction"], torch.tensor([0.5], device=loss.device)).all()


def _dummy_custom_loss(inputs: LossInputs, multiplier: float = 1.0) -> LossOutputs:
    """A simple custom loss for testing."""
    loss = (inputs.trainer_logprobs[inputs.loss_mask].sum() * multiplier).abs()
    return LossOutputs(
        loss=loss,
        metrics={"custom_metric": torch.tensor(multiplier)},
    )
