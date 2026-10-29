# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Check the real in-place test body with honest and incorrect candidates."""

import inspect
import json
from pathlib import Path

import pytest

import flaggems_sglang
from flaggems_sglang.reference import get_reference
from flaggems_sglang.reference_candidate import reference_test_mode
from tests import test_triton_mrope_fused as module


def test_mrope_definition_matches_the_callable():
    path = (
        Path(__file__).resolve().parents[2]
        / "definitions/triton_mrope_fused.json"
    )
    definition = json.loads(path.read_text())
    assert [p["name"] for p in definition["parameters"]] == list(
        inspect.signature(get_reference("triton_mrope_fused")).parameters
    )
    assert definition["outputs"] == []
    assert definition["effects"] == dict(
        mutates=["q", "k"], returns_alias_of={}
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_mrope_explicit_selection_and_independent_mutable_buffers(
    monkeypatch, enabled
):
    reference = get_reference("triton_mrope_fused")
    calls, q_buffers = [], []

    def oracle(**inputs):
        calls.append("reference")
        q_buffers.append(inputs["q"])
        return reference(**inputs)

    def candidate(**inputs):
        calls.append("candidate")
        q_buffers.append(inputs["q"])
        return reference(**inputs)

    monkeypatch.setattr(module.cfg, "device", "cpu")
    monkeypatch.setattr(module.cfg, "TO_CPU", False)
    monkeypatch.setattr(module, "get_reference", lambda name: oracle)
    monkeypatch.setattr(
        flaggems_sglang, "triton_mrope_fused", candidate, raising=False
    )
    with reference_test_mode(enabled) as state:
        module.test_triton_mrope_fused(module.CASES[-1])
    assert calls == ["reference", "reference" if enabled else "candidate"]
    assert q_buffers[0] is not q_buffers[1]
    assert state.candidate_calls == int(enabled)


@pytest.mark.parametrize(
    "mistake",
    [
        "no_op",
        "return_tensor",
        "cache_write",
        "tail_write",
        "replace_storage",
        "wrong_dtype",
    ],
)
def test_mrope_real_assertions_reject_incorrect_candidates(
    monkeypatch, mistake
):
    reference = get_reference("triton_mrope_fused")

    def candidate(**inputs):
        if mistake == "no_op":
            return None
        reference(**inputs)
        if mistake == "return_tensor":
            return inputs["q"]
        if mistake == "cache_write":
            inputs["cos_sin_cache"][0, 0] += 1
        if mistake == "tail_write":
            inputs["q"].view(inputs["q"].shape[0], -1, 256)[..., 100] += 0.5
        if mistake == "replace_storage":
            inputs["q"].data = inputs["q"].clone()
        if mistake == "wrong_dtype":
            inputs["q"].data = inputs["q"].float()
        return None

    monkeypatch.setattr(module.cfg, "device", "cpu")
    monkeypatch.setattr(module.cfg, "TO_CPU", False)
    monkeypatch.setattr(
        flaggems_sglang, "triton_mrope_fused", candidate, raising=False
    )
    with pytest.raises(AssertionError):
        module.test_triton_mrope_fused(module.CASES[-1])
