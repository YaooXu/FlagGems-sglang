# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-free listing, exact replay and real mRoPE candidate timing."""

import pytest
import torch

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import reference_mode_enabled
from flaggems_sglang.testing.mrope_workloads import (
    build_inputs,
    describe,
    parameters,
    workloads,
)

from . import conftest
from .cases import BenchmarkCasePlan, single_input_builder
from .op_benchmark import OpBenchmark, is_case_listing

ALL = workloads(comprehensive=True)
SHAPES = [(i,) for i in range(4)]
MORE_SHAPES = [(4,)]


def _input_fn(shape, cur_dtype, device):
    yield (build_inputs(ALL[shape[0]], cur_dtype, device),)


def _case_fn(shape, cur_dtype):
    del cur_dtype
    case = ALL[shape[0]]
    yield BenchmarkCasePlan(
        shape=describe(case),
        params=dict(parameters(case), configuration=shape),
        builder_args=(shape,),
    )


@pytest.mark.triton_mrope_fused
def test_perf_triton_mrope_fused():
    bench = OpBenchmark(
        op_name="triton_mrope_fused",
        torch_op=get_reference("triton_mrope_fused"),
        case_fn=_case_fn,
        build_inputs_fn=single_input_builder(_input_fn),
        dtypes=[torch.bfloat16],
        shapes=SHAPES,
        more_shapes=MORE_SHAPES,
        shape_desc="Qwen3.6 TP4 mRoPE configuration",
    )
    if not (
        is_case_listing()
        or reference_mode_enabled()
        or conftest.Config.reference_only
    ):
        try:
            bench.set_gems_from_module(flaggems_sglang)
        except AttributeError:
            pytest.skip(
                "triton_mrope_fused: supply a candidate with --override"
            )
    bench.run()
