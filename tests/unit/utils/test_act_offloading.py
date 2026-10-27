import gc
import weakref
from contextlib import nullcontext

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from prime_rl.utils.act_offloading import OffloadActivations


@pytest.fixture(autouse=True)
def disable_gc():
    was_enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    yield
    if was_enabled:
        gc.enable()
    gc.collect()


@pytest.fixture(
    params=[
        "cpu",
        pytest.param(
            "cuda", marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]
        ),
    ]
)
def offload_device(request, monkeypatch):
    device = torch.device(request.param)
    if device.type == "cpu":
        monkeypatch.setattr(torch.accelerator, "is_available", lambda: True)
        monkeypatch.setattr(torch.accelerator, "current_stream", nullcontext)
        monkeypatch.setattr(torch.accelerator, "current_accelerator", lambda *args: device)
    return device


def make_offloader(device, saved_refs, min_offload_size=1024):
    offloader = OffloadActivations(use_pin_memory=device.type == "cuda", min_offload_size=min_offload_size)
    pack = offloader.pack_hook

    def record_pack(tensor):
        packed = pack(tensor)
        saved, modified = packed
        assert saved.grad_fn is None
        assert not saved.requires_grad
        if modified:
            assert saved.device.type == "cpu"
            if device.type == "cuda":
                assert saved.is_pinned()
        saved_refs.append(weakref.ref(saved))
        return packed

    offloader.pack_hook = record_pack
    return offloader


def live_saved_tensors(saved_refs):
    return sum(ref() is not None for ref in saved_refs)


def test_frozen_checkpoint_inputs_are_released_across_microbatches(offload_device):
    device = offload_device
    torch.manual_seed(0)
    frozen = [torch.nn.Linear(16, 16).to(device).requires_grad_(False) for _ in range(3)]
    trained = torch.nn.Linear(16, 16).to(device)
    saved_refs = []

    for tokens in [64, 96, 128, 64, 96, 128, 64, 96]:
        inputs = torch.randn(tokens, 16, device=device)
        hidden = inputs
        for layer in frozen:
            hidden = checkpoint(layer, hidden, use_reentrant=False)
        expected_output = checkpoint(trained, hidden, use_reentrant=False)
        expected_output.sum().backward()
        expected_gradients = [param.grad.clone() for param in trained.parameters()]
        trained.zero_grad()

        offloader = make_offloader(device, saved_refs)
        with offloader:
            hidden = inputs
            for layer in frozen:
                hidden = checkpoint(layer, hidden, use_reentrant=False)
            output = checkpoint(trained, hidden, use_reentrant=False)
        torch.testing.assert_close(output, expected_output, rtol=0, atol=0)
        output.sum().backward()
        for param, expected in zip(trained.parameters(), expected_gradients):
            torch.testing.assert_close(param.grad, expected, rtol=0, atol=0)
        trained.zero_grad()
        del output, expected_output, hidden, offloader
        assert live_saved_tensors(saved_refs) == 0


@pytest.mark.parametrize("min_offload_size", [1024, 10**9])
def test_separate_backward_calls_keep_their_saved_tensors(offload_device, min_offload_size):
    first = torch.randn(64, 16, device=offload_device, requires_grad=True)
    second = torch.randn(96, 16, device=offload_device, requires_grad=True)
    saved_refs = []
    offloader = make_offloader(offload_device, saved_refs, min_offload_size)
    with offloader:
        first_loss = (first * first).sum()
        second_loss = (second * second).sum()
    assert live_saved_tensors(saved_refs) == 4

    first_loss.backward()
    assert live_saved_tensors(saved_refs) == 2
    second_loss.backward()
    assert live_saved_tensors(saved_refs) == 0
    torch.testing.assert_close(first.grad, 2 * first, rtol=0, atol=0)
    torch.testing.assert_close(second.grad, 2 * second, rtol=0, atol=0)


@pytest.mark.parametrize("min_offload_size", [1024, 10**9])
def test_discarded_forward_releases_saved_tensors(offload_device, min_offload_size):
    inputs = torch.randn(64, 16, device=offload_device, requires_grad=True)
    values = inputs.sin()
    saved_refs = []
    offloader = make_offloader(offload_device, saved_refs, min_offload_size)
    with offloader:
        output = (values * values).sum()
    assert live_saved_tensors(saved_refs) == 2
    del output, offloader
    assert live_saved_tensors(saved_refs) == 0


@pytest.mark.parametrize("min_offload_size", [1024, 10**9])
def test_retained_graph_can_be_unpacked_again(offload_device, min_offload_size):
    inputs = torch.randn(64, 16, device=offload_device, requires_grad=True)
    saved_refs = []
    offloader = make_offloader(offload_device, saved_refs, min_offload_size)
    with offloader:
        loss = (inputs * inputs).sum()
    loss.backward(retain_graph=True)
    expected = inputs.grad.clone()
    assert live_saved_tensors(saved_refs) == 2

    inputs.grad = None
    loss.backward()
    assert live_saved_tensors(saved_refs) == 0
    torch.testing.assert_close(inputs.grad, expected, rtol=0, atol=0)
