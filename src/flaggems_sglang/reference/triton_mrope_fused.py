# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0

"""Portable Torch oracle for the SGLang v0.5.11 in-place mRoPE ABI."""

import torch


def reference(
    q,
    k,
    cos_sin_cache,
    positions,
    mrope_section,
    head_size,
    rotary_dim,
    mrope_interleaved,
    mrope_interleaved_glm,
    is_neox_style,
    axis_map,
):
    half = rotary_dim // 2
    index = torch.arange(half, device=q.device)
    if mrope_interleaved:
        if mrope_interleaved_glm:
            axes = axis_map[:half].long()
        else:
            h_mask = (index % 3 == 1) & (index <= 3 * mrope_section[1])
            w_mask = (index % 3 == 2) & (index <= 3 * mrope_section[2])
            axes = torch.where(h_mask, 1, torch.where(w_mask, 2, 0))
    else:
        axes = torch.where(
            index < mrope_section[0],
            0,
            torch.where(index < sum(mrope_section[:2]), 1, 2),
        )
    selected = positions[axes].transpose(0, 1).long()
    cos = cos_sin_cache[selected, index].float().unsqueeze(1)
    sin = cos_sin_cache[selected, index + half].float().unsqueeze(1)
    for value in (q, k):
        heads = value.view(value.shape[0], -1, head_size)
        left = (
            heads[..., :half] if is_neox_style else heads[..., :rotary_dim:2]
        )
        right = (
            heads[..., half:rotary_dim]
            if is_neox_style
            else heads[..., 1:rotary_dim:2]
        )
        first, second = left.float().clone(), right.float().clone()
        left.copy_((first * cos - second * sin).to(value.dtype))
        right.copy_((second * cos + first * sin).to(value.dtype))
