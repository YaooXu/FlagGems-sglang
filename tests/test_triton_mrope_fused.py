# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.6 mRoPE: ordinary pytest body with an explicit Torch oracle."""

from copy import deepcopy

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import get_candidate
from flaggems_sglang.testing.mrope_workloads import build_inputs, workloads

from . import conftest as cfg

CASES = workloads(comprehensive=True)
TOLERANCE = dict(atol=2e-3, rtol=0.016)


def _clone(inputs, device=None):
    return {
        name: (
            torch.empty_strided(
                value.shape,
                value.stride(),
                dtype=value.dtype,
                device=value.device if device is None else device,
            ).copy_(value)
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
        )
        for name, value in inputs.items()
    }


@pytest.mark.triton_mrope_fused
@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[f"{c['model']}-{c['tokens']}-{c['variant']}" for c in CASES],
)
def test_triton_mrope_fused(case):
    reference = get_reference("triton_mrope_fused")
    try:
        candidate = get_candidate("triton_mrope_fused", reference=reference)
    except ImportError:
        pytest.skip("triton_mrope_fused: supply a candidate with --override")
    inputs = build_inputs(case, torch.bfloat16, cfg.device)
    before = _clone(inputs)
    # This test owns its copies because the ABI mutates q/k and returns None.
    expected_inputs = _clone(inputs, "cpu" if cfg.TO_CPU else None)
    assert reference(**expected_inputs) is None
    storage = {
        name: value.untyped_storage().data_ptr()
        for name, value in inputs.items()
        if isinstance(value, torch.Tensor)
    }
    assert candidate(**inputs) is None
    for name, original in before.items():
        actual = inputs[name]
        if not isinstance(original, torch.Tensor):
            assert actual == original
            continue
        assert (
            actual.shape == original.shape
            and actual.stride() == original.stride()
        )
        assert (
            actual.dtype == original.dtype and actual.device == original.device
        )
        assert actual.untyped_storage().data_ptr() == storage[name]
        if name in ("q", "k"):
            torch.testing.assert_close(
                actual.cpu(),
                expected_inputs[name].cpu(),
                **TOLERANCE,
                equal_nan=False,
            )
            assert torch.equal(
                actual.view(case["tokens"], -1, 256)[..., 64:],
                original.view(case["tokens"], -1, 256)[..., 64:],
            )
        else:
            assert torch.equal(actual, original)
