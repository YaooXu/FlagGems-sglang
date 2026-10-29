# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Compare an injected candidate with SGLang's original Triton mRoPE."""

from copy import deepcopy

import pytest
import torch

from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference import mrotary_embedding as native
from flaggems_sglang.reference_candidate import get_candidate

from . import conftest as cfg

WORKLOADS = native.WORKLOADS


def _make_case(case):
    if cfg.TO_CPU:
        pytest.fail(
            "The native Triton reference requires a GPU; omit --ref cpu."
        )
    inputs = native.build_inputs(case, torch.bfloat16, cfg.device)
    reference_inputs = {
        name: (
            torch.empty_strided(
                value.shape,
                value.stride(),
                dtype=value.dtype,
                device=value.device,
            ).copy_(value)
            if isinstance(value, torch.Tensor)
            else deepcopy(value)
        )
        for name, value in inputs.items()
    }
    storage = {name: inputs[name].untyped_storage() for name in ("q", "k")}

    def check(actual, expected):
        # Both callables return None. Their in-place q/k results are compared.
        assert actual is expected is None
        for name, value in inputs.items():
            wanted = reference_inputs[name]
            if not isinstance(value, torch.Tensor):
                assert value == wanted
                continue
            assert (
                value.shape == wanted.shape
                and value.stride() == wanted.stride()
            )
            assert (
                value.dtype == wanted.dtype and value.device == wanted.device
            )
            if name in ("q", "k"):
                assert (
                    value.untyped_storage().data_ptr()
                    == storage[name].data_ptr()
                )
                torch.testing.assert_close(
                    value, wanted, atol=2e-3, rtol=0.016, equal_nan=False
                )
                assert torch.equal(
                    value.view(case["tokens"], -1, 256)[..., 64:],
                    wanted.view(case["tokens"], -1, 256)[..., 64:],
                )
            else:
                assert torch.equal(value, wanted)

    return dict(inputs, _reference_inputs=reference_inputs, check=check)


_CASE_FACTORIES = [lambda row=row: _make_case(row) for row in WORKLOADS]


@pytest.mark.mrotary_embedding
@pytest.mark.parametrize(
    "case_idx",
    range(len(WORKLOADS)),
    ids=[
        f"{c['model']}-{c['tokens']}-stride{c['position_stride']}"
        for c in WORKLOADS
    ],
)
def test_mrotary_embedding(case_idx):
    case = _CASE_FACTORIES[case_idx]()
    check = case.pop("check")
    reference_inputs = case.pop("_reference_inputs", case)
    reference = get_reference("mrotary_embedding")
    expected = reference(**reference_inputs)
    try:
        candidate = get_candidate("mrotary_embedding", reference=reference)
    except ImportError:
        pytest.skip("mrotary_embedding: supply a candidate with --override")
    actual = candidate(**case)
    check(actual, expected)
