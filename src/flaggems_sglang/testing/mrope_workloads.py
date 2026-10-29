# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Tensor-free Qwen3.6 mRoPE cases and deterministic lazy inputs."""


def workloads(comprehensive=False):
    cases = [
        dict(model=model, tokens=tokens, q_heads=heads, variant="documented")
        for model, heads in (("27b", 6), ("35b", 4))
        for tokens in (20, 1)
    ]
    if comprehensive:
        cases.append(
            dict(
                model="35b",
                tokens=3,
                q_heads=4,
                variant="distinct_strided_axes",
            )
        )
    return cases


def describe(case):
    return dict(
        q=(case["tokens"], case["q_heads"] * 256),
        k=(case["tokens"], 256),
        cos_sin_cache=(262400, 64),
        positions=(3, case["tokens"]),
    )


def parameters(case):
    return dict(
        case,
        mrope_section=[11, 11, 10],
        head_size=256,
        rotary_dim=64,
        mrope_interleaved=True,
        mrope_interleaved_glm=False,
        is_neox_style=True,
        axis_map=None,
        position_dtype="torch.int64",
        position_stride=[
            512 if case["variant"] != "documented" else case["tokens"],
            1,
        ],
        activation_dtype="torch.bfloat16",
        seed=0,
        decode_position=20,
        prefill_position_recipe="arange(T)",
        distinct_axes_offsets=(
            [0, 7, 14] if case["variant"] != "documented" else None
        ),
    )


def build_inputs(case, dtype, device):
    import torch

    shape = describe(case)
    generator = torch.Generator(device=device).manual_seed(0)

    def normal(size, scale):
        return (
            torch.randn(
                size, device=device, dtype=torch.float32, generator=generator
            )
            * scale
        )

    angles = normal((262400, 32), 1.0)
    cache = torch.cat((angles.cos(), angles.sin()), -1).to(dtype)
    count = case["tokens"]
    position = torch.arange(count, device=device, dtype=torch.int64)
    if count == 1:
        position += 20
    positions = position.repeat(3, 1)
    if case["variant"] != "documented":
        backing = torch.zeros((3, 512), device=device, dtype=torch.int64)
        backing[:, :count].copy_(
            positions + torch.arange(3, device=device)[:, None] * 7
        )
        positions = backing[:, :count]
    return dict(
        q=normal(shape["q"], 0.5).to(dtype),
        k=normal(shape["k"], 0.5).to(dtype),
        cos_sin_cache=cache,
        positions=positions,
        mrope_section=[11, 11, 10],
        head_size=256,
        rotary_dim=64,
        mrope_interleaved=True,
        mrope_interleaved_glm=False,
        is_neox_style=True,
        axis_map=None,
    )
